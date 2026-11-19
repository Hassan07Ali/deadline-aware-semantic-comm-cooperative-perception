"""Stage 1 (GPU): run the FROZEN pretrained HEAL encoders once and cache per-car BEV features.

Usage (from ~/HEAL):
  python phase3_scripts/extract_features.py --cond clean|degraded --out /path/cache_clean [--stride 1] [--scenes 0,1] [--max_frames N] [--probe]

Per frame it stores: ego (LiDAR m1) feature, collaborator (camera m2) feature, both single-agent occupancy maps
(sender-side foreground probability p, ego confidence e), pose transforms, GT boxes, camera quality inputs, measured
encoder times. `--cond degraded` applies the C1 sensor corruptions to the RAW inputs:
camera brightness x0.5 + Gaussian blur sigma 1.5 px, and LiDAR keeps 50 % of the points.
Outage bursts are applied later (stage 2) because an outage just means "this modality delivers nothing".
Nothing is trained or tuned here.
"""
import argparse
import importlib
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Subset

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils.common_utils import update_dict
from opencood.utils.transformation_utils import normalize_pairwise_tfm

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
LAP = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)


def fast_voxel_pooling(self, geom_feats, x):
    """Same result as HEAL's LiftSplatShoot.voxel_pooling (sum of the features that land in one BEV cell), but computed with a
    single scatter-add instead of sort + cumsum + boolean indexing (179 ms -> a few ms on the RTX 5050). Verified numerically in --probe."""
    B, N, D, H, W, C = x.shape
    Np = B * N * D * H * W
    x = x.reshape(Np, C)
    g = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long().view(Np, 3)
    bidx = torch.arange(B, device=x.device).repeat_interleave(Np // B)
    kept = ((g[:, 0] >= 0) & (g[:, 0] < self.nx[0]) & (g[:, 1] >= 0) & (g[:, 1] < self.nx[1]) & (g[:, 2] >= 0) & (g[:, 2] < self.nx[2]))
    g, xk, bk = g[kept], x[kept], bidx[kept]
    nx0, ny, nz = int(self.nx[0]), int(self.nx[1]), int(self.nx[2])
    flat = ((bk * nz + g[:, 2]) * ny + g[:, 1]) * nx0 + g[:, 0]
    final = torch.zeros(B * nz * ny * nx0, C, device=x.device, dtype=x.dtype).index_add_(0, flat, xk)
    final = final.view(B, nz, ny, nx0, C).permute(0, 4, 1, 2, 3)      # B, C, Z, Y, X
    return torch.cat(final.unbind(dim=2), 1)


def install_fast_pooling(model):
    import types
    enc = model.encoder_m2
    enc._orig_voxel_pooling = enc.voxel_pooling
    enc.voxel_pooling = types.MethodType(fast_voxel_pooling, enc)
    return enc


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="opencood/logs/m1m2_infer")
    ap.add_argument("--range", default="70.4,40")
    ap.add_argument("--cond", choices=["clean", "degraded"], default="clean")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--scenes", default="")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--lidar_keep", type=float, default=0.5, help="degraded cond: fraction of LiDAR points kept")
    ap.add_argument("--probe", action="store_true", help="print shapes + check equivalence with the stock model, save nothing")
    return ap.parse_args()


def load_hypes(opt):
    hypes = yaml_utils.load_yaml(None, SimpleNamespace(model_dir=opt.model_dir))
    xr, yr = [float(v) for v in opt.range.split(",")]
    z0, z1 = hypes["postprocess"]["anchor_args"]["cav_lidar_range"][2], hypes["postprocess"]["anchor_args"]["cav_lidar_range"][5]
    rng = [-xr, -yr, z0, xr, yr, z1]
    hypes = update_dict(hypes, {"cav_lidar_range": rng, "lidar_range": rng, "gt_range": rng})
    lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
    hypes = getattr(lib, hypes["yaml_parser"])(hypes)
    hypes["validate_dir"] = hypes["test_dir"]
    return hypes


def degrade_images(imgs):
    """C1 camera corruption: brightness x0.5 then Gaussian blur (sigma 1.5 px). imgs normalised [.., 3, H, W]."""
    shp = imgs.shape
    x = imgs.reshape(-1, 3, shp[-2], shp[-1]).cpu()
    x = x * STD + MEAN
    x = (x * 0.5).clamp(0, 1)
    x = TF.gaussian_blur(x, kernel_size=9, sigma=1.5)
    x = (x - MEAN) / STD
    return x.reshape(shp).to(imgs.device)


def camera_quality(imgs):
    """Per-camera proxy inputs from the pixels actually used: mean brightness [0,1] and Laplacian variance."""
    shp = imgs.shape
    x = imgs.reshape(-1, 3, shp[-2], shp[-1]).cpu() * STD + MEAN
    gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    bright = gray.mean(dim=(1, 2, 3))
    lap = torch.nn.functional.conv2d(gray, LAP)
    lapvar = lap.var(dim=(1, 2, 3))
    return bright.numpy().astype(np.float32), lapvar.numpy().astype(np.float32)


def lidar_tile_counts(inp, mods):
    """LiDAR points per 1.6 m tile (4 x 4 pillars of 0.4 m), indexed by AGENT: uint16 [n_agents, 50, 88] (camera agents stay 0)."""
    coords = inp["voxel_coords"].cpu().numpy()            # [N, 4] = slot-in-m1-batch, z, y, x
    npts = inp["voxel_num_points"].cpu().numpy()
    out = np.zeros((len(mods), 50, 88), np.int64)
    slot = 0
    for a, m in enumerate(mods):
        if m != "m1":
            continue
        sel = coords[:, 0] == slot
        ty, tx = coords[sel, 2] // 4, coords[sel, 3] // 4
        ok = (ty >= 0) & (ty < 50) & (tx >= 0) & (tx < 88)
        np.add.at(out[a], (ty[ok], tx[ok]), npts[sel][ok])
        slot += 1
    return np.minimum(out, 65535).astype(np.uint16)


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.no_grad()
def encode_agents(model, data):
    """Replicates HeterPyramidCollab.forward up to the per-agent BEV features (same ops, same order)."""
    mods = data["agent_modality_list"]
    feats, times = {}, {}
    for m in model.modality_name_list:
        if m not in mods:
            continue
        t0 = sync()
        f = getattr(model, f"encoder_{m}")(data, m)
        f = getattr(model, f"backbone_{m}")({"spatial_features": f})["spatial_features_2d"]
        f = getattr(model, f"aligner_{m}")(f)
        if model.sensor_type_dict[m] == "camera":
            _, _, H, W = f.shape
            th = int(H * getattr(model, f"crop_ratio_H_{m}"))
            tw = int(W * getattr(model, f"crop_ratio_W_{m}"))
            f = torch_center_crop(f, th, tw)
        times[m] = (sync() - t0) * 1000.0
        feats[m] = f
    count = {m: 0 for m in model.modality_name_list}
    per_agent = []
    for m in mods:
        per_agent.append(feats[m][count[m]])
        count[m] += 1
    return torch.stack(per_agent), times


def torch_center_crop(f, th, tw):
    import torchvision
    return torchvision.transforms.CenterCrop((th, tw))(f)


@torch.no_grad()
def occupancy(model, feats):
    """Single-agent occupancy probability (HEAL's own per-agent head, level 0) = sender-side p and ego confidence."""
    fl = model.pyramid_backbone.get_multiscale_feature(feats)
    occ = model.pyramid_backbone.single_head_0(fl[0])
    p = torch.sigmoid(occ)
    if p.shape[-2:] != feats.shape[-2:]:
        p = torch.nn.functional.interpolate(p, size=feats.shape[-2:], mode="bilinear", align_corners=False)
    return p[:, 0]


def main():
    opt = get_args()
    hypes = load_hypes(opt)
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.cuda().eval()
    enc_m2 = install_fast_pooling(model)

    if opt.cond == "degraded":
        import opencood.data_utils.datasets.intermediate_heter_fusion_dataset as dsmod
        orig = dsmod.shuffle_points

        def shuffle_drop(pts):
            pts = orig(pts)
            seed = int(abs(float(pts[:, :3].sum())) * 1000) % (2 ** 32 - 1)
            keep = np.random.RandomState(seed).rand(len(pts)) < opt.lidar_keep
            return pts[keep]
        dsmod.shuffle_points = shuffle_drop

    ds = build_dataset(hypes, visualize=True, train=False)
    lens = [0] + list(ds.len_record)
    chosen = [int(x) for x in opt.scenes.split(",")] if opt.scenes else list(range(len(lens) - 1))
    idxs = []
    for sc in chosen:
        idxs += list(range(lens[sc], lens[sc + 1], opt.stride))
    if opt.max_frames:
        idxs = idxs[: opt.max_frames]
    sub = Subset(ds, idxs)
    sub.collate_batch_test = ds.collate_batch_test
    loader = DataLoader(sub, batch_size=1, num_workers=3, collate_fn=ds.collate_batch_test, shuffle=False)
    print("frames:", len(idxs), "cond:", opt.cond, flush=True)
    if not opt.probe:
        os.makedirs(opt.out, exist_ok=True)

    t_start = time.time()
    for n, batch in enumerate(loader):
        if batch is None:
            continue
        idx = idxs[n]
        batch = train_utils.to_device(batch, torch.device("cuda"))
        data = batch["ego"]
        mods = data["agent_modality_list"]
        scene_i = max(i for i in range(len(lens) - 1) if lens[i] <= idx)

        # camera quality inputs are read from the pixels that are actually used
        cam_q = None
        if "inputs_m2" in data and "m2" in mods:
            if opt.cond == "degraded":
                data["inputs_m2"]["imgs"] = degrade_images(data["inputs_m2"]["imgs"])
            br, lv = camera_quality(data["inputs_m2"]["imgs"][0])
            ext = data["inputs_m2"]["extrinsics"][0].cpu().numpy()          # [Ncam, 4, 4] camera->lidar
            yaw = np.arctan2(ext[:, 1, 2], ext[:, 0, 2]).astype(np.float32)  # optical axis (z of camera) in lidar frame
            cam_q = {"brightness": br, "lapvar": lv, "yaw": yaw}

        lidar_cnt = None
        if "inputs_m1" in data and any(x == "m1" for x in mods):
            lidar_cnt = lidar_tile_counts(data["inputs_m1"], mods)
        feats, t_enc = encode_agents(model, data)               # [N_agents, 64, H, W]
        p = occupancy(model, feats)                              # [N_agents, H, W]

        if opt.probe and n < 3 and "inputs_m2" in data:
            fast_f = enc_m2(data, "m2")
            enc_m2.voxel_pooling, keep_fast = enc_m2._orig_voxel_pooling, enc_m2.voxel_pooling
            orig_f = enc_m2(data, "m2")
            enc_m2.voxel_pooling = keep_fast
            print(f"[probe {n}] voxel pooling fast vs original: max abs diff {float((fast_f-orig_f).abs().max()):.2e}, "
                  f"relative {float((fast_f-orig_f).abs().sum()/orig_f.abs().sum()):.2e}", flush=True)
        if opt.probe and n < 3:
            full = model(data)
            affine = normalize_pairwise_tfm(data["pairwise_t_matrix"], model.H, model.W, model.fake_voxel_size)
            fused, _ = model.pyramid_backbone.forward_collab(feats, data["record_len"], affine, mods, model.cam_crop_info)
            if model.shrink_flag:
                fused = model.shrink_conv(fused)
            cls = model.cls_head(fused)
            print(f"[probe {n}] agents={mods} feats={tuple(feats.shape)} p={tuple(p.shape)} "
                  f"max|cls_ours-cls_stock|={float((cls - full['cls_preds']).abs().max()):.2e} t_enc_ms={ {k: round(v,1) for k,v in t_enc.items()} }", flush=True)
            print("   pairwise_t_matrix[0,:2,:2]=\n", data["pairwise_t_matrix"][0, :2, :2].cpu().numpy().round(3))
            print("   lidar_poses=", data["lidar_pose"].cpu().numpy().round(2).tolist())
        if opt.probe and lidar_cnt is not None and n < 3:
            gt = data["object_bbx_center"][0][data["object_bbx_mask"][0].bool()].cpu().numpy()[:, :2]
            tx_, ty_ = ((gt[:, 0] + 70.4) // 1.6).astype(int), ((gt[:, 1] + 40.0) // 1.6).astype(int)
            ok_ = (tx_ >= 0) & (tx_ < 88) & (ty_ >= 0) & (ty_ < 50)
            c0 = lidar_cnt[0].astype(float)
            print(f"[probe {n}] LiDAR points per tile at GT objects {c0[ty_[ok_], tx_[ok_]].mean():.1f} (row=y,col=x) vs "
                  f"{c0[49-ty_[ok_], tx_[ok_]].mean():.1f} (flipped y) vs {c0.mean():.1f} (map mean)", flush=True)
        if opt.probe and n >= 3:
            break
        if opt.probe:
            continue

        keep = {k: v for k, v in data.items()
                if k in ("pairwise_t_matrix", "record_len", "anchor_box", "transformation_matrix", "transformation_matrix_clean",
                         "object_bbx_center", "object_bbx_mask", "object_ids", "cav_id_list", "agent_modality_list", "lidar_pose", "lidar_pose_clean")}
        rec = {
            "idx": idx, "scene": scene_i, "scene_name": os.path.basename(ds.scenario_folders[scene_i]),
            "frame_in_scene": idx - lens[scene_i], "cond": opt.cond,
            "ego_feat": feats[0].half().cpu(),
            "col_feat": feats[1].half().cpu() if feats.shape[0] > 1 else None,
            "p_ego": p[0].half().cpu(), "p_col": p[1].half().cpu() if feats.shape[0] > 1 else None,
            "t_enc_ms": t_enc, "cam_q": cam_q, "lidar_cnt": lidar_cnt,
            "n_lidar_pts_note": opt.cond,
            "data": {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in keep.items()},
        }
        torch.save(rec, os.path.join(opt.out, f"{idx:05d}.pt"))
        if n % 50 == 0:
            el = time.time() - t_start
            print(f"{n}/{len(idxs)}  {el:.0f}s  eta {el/(n+1)*(len(idxs)-n-1):.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
