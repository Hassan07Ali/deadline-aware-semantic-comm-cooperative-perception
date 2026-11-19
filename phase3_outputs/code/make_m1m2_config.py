"""Make a reduced m1 (LiDAR) + m2 (camera) config from the official HEAL_OPV2V config.

Usage (from ~/HEAL):
  python phase3_scripts/make_m1m2_config.py opencood/logs/final_infer opencood/logs/m1m2_infer [assignment.json]

What it changes (all recorded in the changes log):
  * drops the m3 / m4 encoders (we never use them) -> checkpoint loads with strict=False
  * heter.modality_setting keeps only m1 and m2; any other label (we use 'm9') = "excluded car"
  * heter.assignment_path -> our own per-scene ego/collaborator assignment (Option A)
  * the pretrained weights file is symlinked, not copied
"""
import copy
import os
import sys

import yaml


class NoAlias(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def main():
    src, dst = sys.argv[1], sys.argv[2]
    assignment = sys.argv[3] if len(sys.argv) > 3 else "opencood/logs/heter_modality_assign/ours_pair_assignment.json"
    ego_mod = sys.argv[4] if len(sys.argv) > 4 else None
    # plain loader (anchors resolved); floats like 1e-10 handled below
    with open(os.path.join(src, "config.yaml")) as f:
        txt = f.read()
    cfg = yaml.load(txt, Loader=yaml.Loader)
    cfg = copy.deepcopy(cfg)

    cfg["name"] = "m1m2_ours_frozen"
    heter = cfg["heter"]
    heter["assignment_path"] = assignment
    if ego_mod:
        heter["ego_modality"] = ego_mod
        cfg["model"]["args"]["ego_modality"] = ego_mod
    heter["lidar_channels_dict"] = {}
    heter["mapping_dict"] = {"m1": "m1", "m2": "m2", "m9": "m9"}
    for k in ("m3", "m4"):
        heter["modality_setting"].pop(k, None)
        cfg["model"]["args"].pop(k, None)
    # depth supervision is a training-only loss and needs depth GT images that are not in the download
    cfg["model"]["args"]["m2"]["encoder_args"]["depth_supervision"] = False
    cfg["input_source"] = ["lidar", "camera"]   # depth GT files are training-only and not downloaded

    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, Dumper=NoAlias, sort_keys=False)
    ck = [x for x in os.listdir(src) if x.endswith(".pth")]
    for x in ck:
        link = os.path.join(dst, x)
        if not os.path.exists(link):
            os.symlink(os.path.abspath(os.path.join(src, x)), link)
    print("wrote", os.path.join(dst, "config.yaml"), "weights:", ck)


if __name__ == "__main__":
    main()
