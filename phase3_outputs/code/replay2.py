"""Stage 2 v2 (GPU): replay cached frozen-model features through every method.   (supersedes replay.py)

What changed vs replay.py (all in PHASE3_CHANGES_LOG.md):
  * receiver-need feedback is CAUSAL, as Phase 2 specifies: the ego occupancy map of the PREVIOUS dataset frame, warped with the
    timestamped poses (previous ego pose, current collaborator pose); expired/lost feedback -> e = 0.
    The old same-frame version is kept as the ORACLE ablation A7.
  * B1 (uniform compression of the whole BEV), R1 (ours + one retransmission of lost packets, pays identical byte/time costs)
  * parameter overrides for the Phase 2 sensitivity grid: --eps --lam --tau --bits, custom links (--custom_conds), --outage_frac
  * timeline works for a slow ego sensor (camera ego): the cooperative result needs max(last arrival, ego encoding) + T_reserve
Nothing is trained or tuned; all defaults are the Phase 2 values.
"""
import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/cnet/ours_pkg"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ours import admission as ADM
from ours import link_replay as LR
from ours import selector as SEL
from ours import serializer as S
from ours import timeline as TL

import extract_features as E
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils.common_utils import compute_iou, convert_format, torch_tensor_to_numpy
from opencood.utils.transformation_utils import normalize_pairwise_tfm

CELL, X0, Y0 = 0.8, -70.4, -40.0
TH, TW = 50, 88
NT = TH * TW
TX = X0 + (np.arange(TW) + 0.5) * 1.6
TY = Y0 + (np.arange(TH) + 0.5) * 1.6
TILE_XY = np.stack(np.meshgrid(TX, TY), -1).reshape(-1, 2)      # tile centres in the SENDER frame; index = row*88 + col
CAM_GRID = 51.2
B_REF, S_REF = 0.40, 0.004          # camera proxy constants (fixed before any result)
N_REF = 12.0                        # LiDAR points per tile that counts as well observed (fixed before any result)
FB_MAP = (16, 32)                   # feedback map size (rows=y, cols=x)
FRAME_DT_MS = 100.0                 # OPV2V cadence 10 Hz (feedback age; expiry is 200 ms)
FB_METHODS = {"ours": "prev", "A1": "prev", "A3": "prev", "R1": "prev", "A7": "same", "A2": "none"}


# ---------------------------------------------------------------- tiles / quantisation
def to_tiles(feat):
    C, H, W = feat.shape
    return feat.reshape(C, H // 2, 2, W // 2, 2).permute(1, 3, 0, 2, 4).reshape(-1, C * 4)


def from_tiles(tiles, shape):
    C, H, W = shape
    return tiles.reshape(H // 2, W // 2, C, 2, 2).permute(2, 0, 3, 1, 4).reshape(C, H, W)


def quantize_tokens(tiles, bits=8):
    """Symmetric per-token quantisation with one fp16 scale; returns what the receiver sees after dequantisation."""
    if bits >= 16:
        return tiles.half().float()
    lv = 2 ** (bits - 1) - 1
    scale = (tiles.abs().amax(1) / lv).clamp_min(1e-8).half().float()
    return torch.round(tiles / scale[:, None]).clamp(-lv, lv) * scale[:, None]


# ---------------------------------------------------------------- scoring inputs
def camera_r(q):
    return np.clip(q["brightness"] / B_REF, 0, 1) * np.clip(np.sqrt(q["lapvar"] / S_REF), 0, 1)


def static_inputs(rec, ptm, outage):
    """p, r, a, c for the collaborator's tiles (sender frame) and the tile centres in the CURRENT ego frame."""
    mods = rec["data"]["agent_modality_list"]
    p = F.avg_pool2d(rec["p_col"].float()[None, None], 2)[0, 0].numpy().reshape(-1)
    xy = TILE_XY
    if mods[1] == "m1":                                         # LiDAR collaborator
        cnt = rec["lidar_cnt"][1].astype(float).reshape(-1)
        a = (cnt > 0).astype(float)
        r = np.clip(np.log1p(cnt) / np.log1p(N_REF), 0, 1)
    else:                                                       # camera collaborator
        a = ((np.abs(xy[:, 0]) < CAM_GRID) & (np.abs(xy[:, 1]) < CAM_GRID)).astype(float)
        q = rec["cam_q"]
        cam_id = np.argmax(np.cos(np.arctan2(xy[:, 1], xy[:, 0])[:, None] - q["yaw"][None, :]), axis=1)
        r = camera_r(q)[cam_id]
    if outage == "camera":                                      # collaborator sensor down
        a[:] = 0; r[:] = 0
    T = ptm
    xe = T[0, 0] * xy[:, 0] + T[0, 1] * xy[:, 1] + T[0, 3]
    ye = T[1, 0] * xy[:, 0] + T[1, 1] * xy[:, 1] + T[1, 3]
    c = SEL.critical_mask(np.stack([xe, ye], 1)).astype(float)
    return p, r, a, c, xe, ye


def feedback_lookup(p_ego_map, xq, yq):
    """Quantised 16 x 32 uint8 ego-confidence map (512 B payload), sampled at ego-frame positions (xq, yq); outside -> 0."""
    fb = F.adaptive_avg_pool2d(p_ego_map.float()[None, None], FB_MAP)[0, 0].numpy()
    fb = np.round(fb * 255) / 255.0
    j = np.floor((xq - X0) / 140.8 * FB_MAP[1]).astype(int)
    i = np.floor((yq - Y0) / 80.0 * FB_MAP[0]).astype(int)
    ok = (j >= 0) & (j < FB_MAP[1]) & (i >= 0) & (i < FB_MAP[0])
    e = np.zeros(len(xq))
    e[ok] = fb[i[ok], j[ok]]
    return e


def prev_frame_query(rec_prev, rec_cur):
    """Tile centres of the current collaborator frame expressed in the PREVIOUS ego frame (uses only past ego pose)."""
    pe = rec_prev["data"]["lidar_pose"].numpy()[0]              # x, y, z, roll, yaw(deg), pitch of ego at t-1
    pc = rec_cur["data"]["lidar_pose"].numpy()[1]               # collaborator at t
    yc, ye_ = math.radians(pc[4]), math.radians(pe[4])
    xy = TILE_XY
    wx = math.cos(yc) * xy[:, 0] - math.sin(yc) * xy[:, 1] + pc[0]
    wy = math.sin(yc) * xy[:, 0] + math.cos(yc) * xy[:, 1] + pc[1]
    dx, dy = wx - pe[0], wy - pe[1]
    return math.cos(ye_) * dx + math.sin(ye_) * dy, -math.sin(ye_) * dx + math.cos(ye_) * dy


def outage_for(scene_name, frame, n_frames, frac=0.10, seed=11):
    """`frac` of frames in 3-frame bursts, alternating collaborator ('camera' label) / ego ('lidar' label) sensor."""
    n_burst = max(1, round(frac * n_frames / 3))
    h = int.from_bytes(hashlib.sha256(f"{scene_name}|{seed}".encode()).digest()[:8], "big")
    rng = np.random.RandomState(h % (2 ** 32 - 1))
    starts, tries = [], 0
    gap = 4 if frac <= 0.12 else 3
    while len(starts) < n_burst and tries < 3000:
        s = int(rng.randint(0, max(1, n_frames - 3)))
        if all(abs(s - t) >= gap for t in starts):
            starts.append(s)
        tries += 1
    starts.sort()
    for k, s in enumerate(starts):
        if s <= frame < s + 3:
            return "camera" if k % 2 == 0 else "lidar"
    return None


# ---------------------------------------------------------------- detection + statistics
class Detector:
    def __init__(self, model, ds):
        self.m, self.ds = model, ds

    @torch.no_grad()
    def run(self, rec, feats, mods):
        d = rec["data"]
        dev = torch.device("cuda")
        affine = normalize_pairwise_tfm(d["pairwise_t_matrix"].to(dev), self.m.H, self.m.W, self.m.fake_voxel_size)
        n = len(mods)
        fused, _ = self.m.pyramid_backbone.forward_collab(feats, torch.tensor([n]), affine[:, :n, :n], mods, self.m.cam_crop_info)
        if self.m.shrink_flag:
            fused = self.m.shrink_conv(fused)
        out = {"ego": {"cls_preds": self.m.cls_head(fused), "reg_preds": self.m.reg_head(fused), "dir_preds": self.m.dir_head(fused)}}
        dd = {"ego": {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in d.items()}}
        return self.ds.post_process(dd, out)


def frame_stats(pred, score, gt):
    gtn = torch_tensor_to_numpy(gt) if gt is not None else np.zeros((0, 8, 3))
    ngt = gtn.shape[0]
    polys = list(convert_format(gtn)) if ngt else []
    ctr = gtn.mean(1)[:, :2] if ngt else np.zeros((0, 2))
    crit = (ctr[:, 0] > 0) & (ctr[:, 0] < 40) & (np.abs(ctr[:, 1]) < 10)
    det = torch_tensor_to_numpy(pred) if pred is not None else None
    res = {}
    for thr in (0.3, 0.5, 0.7):
        tps, scs = [], []
        matched = np.zeros(ngt, bool)
        if det is not None and len(det):
            s = torch_tensor_to_numpy(score)
            order = np.argsort(-s)
            dpoly = list(convert_format(det))
            remaining = list(range(ngt))
            for o in order:
                scs.append(float(s[o]))
                if not remaining:
                    tps.append(0)
                    continue
                ious = compute_iou(dpoly[o], [polys[k] for k in remaining])
                if np.max(ious) < thr:
                    tps.append(0)
                else:
                    tps.append(1)
                    j = int(np.argmax(ious))
                    matched[remaining[j]] = True
                    remaining.pop(j)
        if thr == 0.5:
            m05 = matched.copy()
        res[thr] = (np.array(tps, np.int8), np.array(scs, np.float32), ngt, int((matched & crit).sum()) if thr == 0.5 else None)
    res["gt_ctr"] = ctr.astype(np.float32)
    res["m05"] = m05
    res["crit_gt"] = int(crit.sum())
    res["n_det"] = 0 if det is None else len(det)
    return res


def empty_like(st):
    out = {thr: (np.zeros(0, np.int8), np.zeros(0, np.float32), st[thr][2], 0 if thr == 0.5 else None) for thr in (0.3, 0.5, 0.7)}
    out.update({"gt_ctr": st["gt_ctr"], "m05": np.zeros(len(st["gt_ctr"]), bool), "crit_gt": st["crit_gt"], "n_det": 0})
    return out


def ap_from(frames, thr):
    from opencood.utils.eval_utils import voc_ap
    tp = np.concatenate([f["st"][thr][0] for f in frames]) if frames else np.zeros(0)
    sc = np.concatenate([f["st"][thr][1] for f in frames]) if frames else np.zeros(0)
    gt = sum(f["st"][thr][2] for f in frames)
    if gt == 0 or len(tp) == 0:
        return 0.0
    o = np.argsort(-sc, kind="stable")
    tp = tp[o]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    return voc_ap((ctp / gt).tolist(), (ctp / (ctp + cfp)).tolist())[0]


# ---------------------------------------------------------------- link with optional single retransmission
def transmit_retry(k, link, t_ready, scene, frame, trace, bits, budget_left, deadline, t_ego_enc, t_res):
    """First pass as LR.transmit; then lost packets are re-sent once (independent loss draw), each only if it can still
    arrive in time for fusion and the byte budget is not exceeded. All retransmitted bytes are charged."""
    base = LR.transmit(k, link, t_ready, scene=scene, frame=frame, trace=trace, bits=bits)
    sizes = S.packet_sizes(k, bits)
    n = S.tokens_per_packet(bits)
    arrivals = list(base["packet_arrival_ms"])
    ok_pkt = [bool(base["token_delivered"][i * n]) for i in range(len(sizes))]
    t = base["last_arrival_ms"]
    extra = 0
    for i, sz in enumerate(sizes):
        if ok_pkt[i] or base["bytes_sent"] + extra + sz > budget_left:      # budget_left = per-frame budget minus feedback bytes
            continue
        t_new = t + 1000.0 * 8.0 * sz / link.rate_bps
        if max(t_new, t_ego_enc) + t_res > deadline:
            break
        t = t_new
        extra += sz
        if LR.loss_draw(scene, frame, trace, 1000 + i) >= link.loss:
            ok_pkt[i] = True
            arrivals[i] = t_new
    tok_ok = np.array([ok_pkt[i // n] for i in range(k)], bool)
    return {"packet_arrival_ms": arrivals, "token_delivered": tok_ok, "bytes_sent": base["bytes_sent"] + extra,
            "last_arrival_ms": max(arrivals) if arrivals else t_ready}


def pooled_tokens(tiles, f):
    """B1: average the whole BEV over f x f tile blocks -> one token per block (uniform spatial compression)."""
    grid = tiles.reshape(TH, TW, 256).permute(2, 0, 1)[None]
    return F.avg_pool2d(grid, f, ceil_mode=True)[0]          # [256, gh, gw]


def b1_factor(budget, bits):
    for f in range(1, 51):
        n = math.ceil(TH / f) * math.ceil(TW / f)
        if S.wire_cost(n, bits) <= budget:
            return f, n
    return 50, 1


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="opencood/logs/m1m1_infer")
    ap.add_argument("--cache_clean", default=os.path.expanduser("~/cnet/cache_ll_clean"))
    ap.add_argument("--cache_deg", default=os.path.expanduser("~/cnet/cache_ll_degraded"))
    ap.add_argument("--timing", default=os.path.expanduser("~/cnet/out/timing_ll.json"))
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--conds", default="C0,C3")
    ap.add_argument("--custom_conds", default="", help="name=state:rate_bps:loss:queue_ms,...  (state clean|degraded)")
    ap.add_argument("--methods", default="B0,B2,B3,U0,ours")
    ap.add_argument("--deadline", default="100")
    ap.add_argument("--budget", default="16384")
    ap.add_argument("--traces", type=int, default=3)
    ap.add_argument("--scenes", default="")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--out", default=os.path.expanduser("~/cnet/out/run2"))
    ap.add_argument("--keep_tokens", action="store_true")
    ap.add_argument("--eps", type=float, default=SEL.EPS)
    ap.add_argument("--lam", type=float, default=SEL.LAMBDA_C)
    ap.add_argument("--tau", type=float, default=SEL.TAU_MS)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--outage_frac", type=float, default=0.10)
    opt = ap.parse_args()

    for spec in [s for s in opt.custom_conds.split(",") if s]:
        name, rest = spec.split("=")
        state, rate, loss, q = rest.split(":")
        LR.CONDITIONS[name] = (state, LR.Link(float(rate), float(loss), float(q)))

    hy = E.load_hypes(argparse.Namespace(model_dir=opt.model_dir, range="70.4,40"))
    model = train_utils.create_model(hy)
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.cuda().eval()
    ds = build_dataset(hy, visualize=True, train=False)
    det = Detector(model, ds)
    dev = torch.device("cuda")
    lens = [0] + list(ds.len_record)

    def scene_of(idx):
        return max(i for i in range(len(lens) - 1) if lens[i] <= idx)

    files = sorted(os.listdir(opt.cache_clean))
    if opt.scenes:
        keep = set(int(x) for x in opt.scenes.split(","))
        files = [fn for fn in files if scene_of(int(fn[:-3])) in keep]
    files = files[:: opt.stride]
    if opt.max_frames:
        files = files[: opt.max_frames]

    def load(root, idx):
        return torch.load(os.path.join(root, f"{idx:05d}.pt"), weights_only=False)

    # ------------------------------------------------------------ calibration
    if opt.calibrate:
        te, tc, tcoop, tego, tsel, tocc = [], [], [], [], [], []
        for n, fn in enumerate(files[:120]):
            rec = load(opt.cache_clean, int(fn[:-3]))
            if rec["col_feat"] is None:
                continue
            mods = rec["data"]["agent_modality_list"]
            te.append(rec["t_enc_ms"][mods[0]]); tc.append(rec["t_enc_ms"][mods[1]])
            ego = rec["ego_feat"].to(dev).float(); col = rec["col_feat"].to(dev).float()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            E.occupancy(model, ego[None]); torch.cuda.synchronize(); tocc.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            p, r, a, c, xe, ye = static_inputs(rec, rec["data"]["pairwise_t_matrix"][0, 1, 0].numpy(), None)
            e = feedback_lookup(rec["p_ego"], xe, ye)
            idx = SEL.top_k(SEL.score(p, r, a, e, c, 50.0), 56)
            q = quantize_tokens(to_tiles(col)[torch.as_tensor(idx, device=dev)]); torch.cuda.synchronize()
            tsel.append((time.perf_counter() - t0) * 1000)
            if n < 4:
                det.run(rec, torch.stack([ego, col]), mods); det.run(rec, ego[None], mods[:1]); continue
            t0 = time.perf_counter()
            tl = torch.zeros(NT, 256, device=dev); tl[torch.as_tensor(idx, device=dev)] = q
            det.run(rec, torch.stack([ego, from_tiles(tl, col.shape)]), mods); torch.cuda.synchronize()
            tcoop.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            det.run(rec, ego[None], mods[:1]); torch.cuda.synchronize()
            tego.append((time.perf_counter() - t0) * 1000)
        tm = {"t_enc_ego_ms": float(np.median(te)), "t_enc_col_ms": float(np.median(tc)), "t_occ_ms": float(np.median(tocc)),
              "t_select_ms": float(np.median(tsel)), "T_reserve_coop_ms": float(np.percentile(tcoop, 95)),
              "T_coop_median_ms": float(np.median(tcoop)), "T_ego_det_ms": float(np.percentile(tego, 95)),
              "T_ego_median_ms": float(np.median(tego)), "n_frames": len(tcoop), "device": torch.cuda.get_device_name(0),
              "note": "encoders p50; receiver/detection p95 (Phase 2 T_reserve); fp32; RTX 5050 laptop"}
        json.dump(tm, open(opt.timing, "w"), indent=2)
        print(json.dumps(tm, indent=2))
        return

    # ------------------------------------------------------------ replay
    tm = json.load(open(opt.timing))
    conds, methods = opt.conds.split(","), opt.methods.split(",")
    deadlines = [float(x) for x in opt.deadline.split(",")]
    budgets = [int(x) for x in opt.budget.split(",")]
    bits = opt.bits
    nper = S.tokens_per_packet(bits)
    t_enc_ego, t_enc_col, T_res = tm["t_enc_ego_ms"], tm["t_enc_col_ms"], tm["T_reserve_coop_ms"]
    t_sel_ms = tm["t_select_ms"]
    ego_ready = t_enc_ego + tm["T_ego_det_ms"]
    t_ready = t_enc_col + t_sel_ms
    results, memo, est = [], {}, {}
    last_scene, t_start = None, time.time()

    for n, fn in enumerate(files):
        idx = int(fn[:-3])
        scene = scene_of(idx)
        fi, nfr = idx - lens[scene], lens[scene + 1] - lens[scene]
        rec_c, rec_d = load(opt.cache_clean, idx), load(opt.cache_deg, idx)
        sname = rec_c["scene_name"]
        prev_c = load(opt.cache_clean, idx - 1) if fi > 0 else None
        prev_d = load(opt.cache_deg, idx - 1) if fi > 0 else None
        if scene != last_scene:
            est, last_scene = {}, scene
        for cond in conds:
            state, link = LR.CONDITIONS[cond]
            rec, prev = (rec_c, prev_c) if state == "clean" else (rec_d, prev_d)
            outage = outage_for(sname, fi, nfr, opt.outage_frac) if state == "degraded" else None
            prev_outage = outage_for(sname, fi - 1, nfr, opt.outage_frac) if (state == "degraded" and fi > 0) else None
            mods = rec["data"]["agent_modality_list"]
            ego = rec["ego_feat"].to(dev).float()
            p_ego = rec["p_ego"]
            if outage == "lidar":
                ego = torch.zeros_like(ego); p_ego = torch.zeros_like(p_ego)
            have_col = (rec["col_feat"] is not None) and outage != "camera"
            k0 = (idx, cond, "ego")
            if k0 not in memo:
                memo[k0] = frame_stats(*det.run(rec, ego[None], mods[:1]))
            st_ego = memo[k0]
            if have_col:
                col = rec["col_feat"].to(dev).float()
                tiles = to_tiles(col)
                ptm = rec["data"]["pairwise_t_matrix"][0, 1, 0].numpy()
                p, r_, a, c, xe, ye = static_inputs(rec, ptm, outage)
                e_same = feedback_lookup(p_ego, xe, ye)
                if prev is not None:
                    pp = prev["p_ego"] if prev_outage != "lidar" else torch.zeros_like(prev["p_ego"])
                    xq, yq = prev_frame_query(prev, rec)
                    e_prev = feedback_lookup(pp, xq, yq)
                else:
                    e_prev = np.zeros(NT)
            for D in deadlines:
                for B in budgets:
                    for trace in ([0] if link.loss == 0 else range(opt.traces)):
                        for m in methods:
                            base = {"scene": scene, "scene_name": sname, "frame": fi, "idx": idx, "cond": cond, "method": m, "D": D,
                                    "budget": B, "trace": trace, "outage": outage}
                            if m == "B0" or not have_col:
                                on_time = (m == "U0") or ego_ready <= D
                                results.append({**base, "kind": "ego_only" if on_time else "empty", "ready_ms": ego_ready, "bytes": 0,
                                                "k_sel": 0, "k_used": 0, "k_lost": 0, "k_late": 0,
                                                "st": st_ego if on_time else empty_like(st_ego)})
                                continue
                            if m == "U0":
                                key = (idx, cond, "U0")
                                if key not in memo:
                                    memo[key] = frame_stats(*det.run(rec, torch.stack([ego, col]), mods))
                                results.append({**base, "kind": "cooperative_unconstrained", "ready_ms": float("nan"), "bytes": 0,
                                                "k_sel": NT, "k_used": NT, "k_lost": 0, "k_late": 0, "st": memo[key]})
                                continue
                            fbm = FB_METHODS.get(m, "none")
                            fb_bytes = S.FEEDBACK_BYTES if fbm != "none" else 0
                            k_cap = S.max_tokens_in_budget(B, bits, feedback=(fbm != "none"))
                            pooled_f = None
                            # ---------------- selection
                            if m == "B1":
                                pooled_f, k_sel_tok = b1_factor(B, bits)
                                idx_sel = np.arange(k_sel_tok)
                            elif m == "B3":
                                idx_sel = SEL.select_confidence_only(p, a, k_cap)
                            elif m == "B2":
                                h = int.from_bytes(hashlib.sha256(f"{sname}|{idx}|{cond}|rand".encode()).digest()[:8], "big")
                                idx_sel = SEL.select_random(a, k_cap, np.random.RandomState(h % (2 ** 32 - 1)))
                            else:
                                fb_ok = LR.loss_draw(sname, idx, trace, -1) >= link.loss
                                e_src = e_prev if fbm == "prev" else (e_same if fbm == "same" else np.zeros(NT))
                                e_eff = SEL.ego_confidence(e_src, FRAME_DT_MS if fb_ok else None) if fbm != "none" else np.zeros(NT)
                                r_use = np.ones(NT) if m == "A1" else r_
                                s = SEL.score(p, r_use, a, e_eff, c, t_ready, eps=opt.eps, lam_c=opt.lam, tau_ms=opt.tau)
                                if m == "A3":
                                    k_send = k_cap
                                else:
                                    ek = (m, cond, trace, D, B)
                                    est.setdefault(ek, ADM.RateEstimator())
                                    rate = est[ek].rate
                                    k_send = 0
                                    for cand in range(1, k_cap + 1):
                                        done = max(t_ready + link.queue_ms + 1000.0 * 8.0 * S.wire_cost(cand, bits) / rate, t_enc_ego) + T_res
                                        if done <= D:
                                            k_send = cand
                                        else:
                                            break
                                idx_sel = SEL.top_k(s, k_send)
                            k_sel = len(idx_sel)
                            # ---------------- link
                            if m == "R1":
                                res = transmit_retry(k_sel, link, t_ready, sname, idx, trace, bits, B - fb_bytes, D, t_enc_ego, T_res)
                            else:
                                res = LR.transmit(k_sel, link, t_ready, scene=sname, frame=idx, trace=trace, bits=bits)
                            total_bytes = res["bytes_sent"] + fb_bytes
                            assert total_bytes <= B, (total_bytes, B, m, k_sel)
                            if k_sel:
                                arr_tok = np.array([res["packet_arrival_ms"][i // nper] for i in range(k_sel)])
                                deliv = res["token_delivered"]
                                ok_mask = deliv & (np.maximum(arr_tok, t_enc_ego) + T_res <= D)
                            else:
                                arr_tok, deliv, ok_mask = np.zeros(0), np.zeros(0, bool), np.zeros(0, bool)
                            if m in ("ours", "A1", "A2", "A7", "R1") and k_sel:
                                sizes = S.packet_sizes(k_sel, bits)
                                svc = sum(8.0 * sz / link.rate_bps for sz in sizes)
                                got = sum(sz for i_, sz in enumerate(sizes) if LR.loss_draw(sname, idx, trace, i_) >= link.loss)
                                est[(m, cond, trace, D, B)].update(got, svc)
                            used = idx_sel[ok_mask] if k_sel else idx_sel
                            coop_ready = None
                            if len(used):
                                coop_ready = max(arr_tok[ok_mask].max(), t_enc_ego) + T_res
                            kind = TL.choose_output(coop_ready, ego_ready, D)
                            if kind == "cooperative":
                                key = (idx, cond, m if m == "B1" else "set", np.sort(used).tobytes())
                                if key not in memo:
                                    if m == "B1":
                                        pt = pooled_tokens(tiles, pooled_f)                     # [256, gh, gw]
                                        gh, gw = pt.shape[1:]
                                        qt = quantize_tokens(pt.reshape(256, -1).T, bits).T.reshape(256, gh, gw)
                                        msk = torch.zeros(gh * gw, device=dev); msk[torch.as_tensor(used, device=dev)] = 1
                                        up = F.interpolate((qt * msk.view(1, gh, gw))[None], scale_factor=pooled_f, mode="nearest")[0][:, :TH, :TW]
                                        full = up.permute(1, 2, 0).reshape(NT, 256)
                                    else:
                                        ui = torch.as_tensor(used, device=dev)
                                        full = torch.zeros(NT, 256, device=dev)
                                        full[ui] = quantize_tokens(tiles[ui], bits)
                                    memo[key] = frame_stats(*det.run(rec, torch.stack([ego, from_tiles(full, col.shape)]), mods))
                                st, ready = memo[key], coop_ready
                            elif kind == "ego_only":
                                st, ready = st_ego, ego_ready
                            else:
                                st, ready = empty_like(st_ego), float("nan")
                            rr = {**base, "kind": kind, "ready_ms": ready, "bytes": total_bytes, "k_sel": k_sel, "k_used": int(len(used)),
                                  "k_lost": int((~deliv).sum()) if k_sel else 0, "k_late": int((deliv & ~ok_mask).sum()) if k_sel else 0, "st": st}
                            if opt.keep_tokens:
                                rr["tok"] = {"idx": np.asarray(idx_sel, np.int32), "ok": np.asarray(ok_mask, bool), "deliv": np.asarray(deliv, bool)}
                            results.append(rr)
        if n % 100 == 0:
            el = time.time() - t_start
            print(f"{n}/{len(files)} frames  {el:.0f}s  eta {el/(n+1)*(len(files)-n-1):.0f}s  memo={len(memo)}", flush=True)
            if len(memo) > 9000:
                memo = {k: v for k, v in memo.items() if k[2] in ("ego", "U0")}

    os.makedirs(os.path.dirname(opt.out), exist_ok=True)
    with open(opt.out + ".pkl", "wb") as f:
        pickle.dump({"results": results, "timing": tm, "args": vars(opt)}, f)
    from collections import defaultdict
    g = defaultdict(list)
    for x in results:
        g[(x["cond"], x["method"], x["D"], x["budget"], x["trace"])].append(x)
    print("\ncond method D budget trace  AP@0.3 AP@0.5 AP@0.7   coop-share  bytes/frame")
    for k in sorted(g):
        fr = g[k]
        print(k, " ".join(f"{ap_from(fr, t):.3f}" for t in (0.3, 0.5, 0.7)),
              f"  {np.mean([f['kind'].startswith('coop') for f in fr]):.2f}  {np.mean([f['bytes'] for f in fr]):.0f}")
    print("saved", opt.out + ".pkl")


if __name__ == "__main__":
    main()
