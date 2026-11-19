"""Choose ONE (ego, collaborator) car pair per test scene BEFORE any detection result is seen.

Rule (fixed in advance, uses only car poses, never labels or detections):
  * among all pairs of vehicle CAVs (id >= 0) in the scene, take the pair with the
    smallest mean ground-plane distance over the frames both cars have;
  * ego = the smaller CAV id of the pair (gets modality m1 = LiDAR), collaborator = the other (m2 = camera);
  * every other car gets the label m9 = excluded.
Writes:
  assignment json (HEAL format)  and  manifest json (scene, pair, frames, distance stats, share of frames in range).
"""
import itertools
import json
import math
import os
import sys


root = sys.argv[1]            # e.g. dataset/OPV2V/test
out_assign = sys.argv[2]
out_manifest = sys.argv[3]
COMM = 70.0
EGO_LABEL = sys.argv[5] if len(sys.argv) > 5 else "m1"
COLLAB_LABEL = sys.argv[4] if len(sys.argv) > 4 else "m2"   # m2 = camera collaborator (option A), m1 = LiDAR collaborator (option B)

import re
POSE_RE = re.compile(r"^lidar_pose:[ \t]*\r?\n- (\S+)\r?\n- (\S+)", re.M)   # first two numbers = x, y (plain text, no yaml objects executed)


def poses(cav_dir):
    d = {}
    for fn in sorted(os.listdir(cav_dir)):
        if fn.endswith(".yaml") and "additional" not in fn:
            with open(os.path.join(cav_dir, fn)) as f:
                m = POSE_RE.search(f.read())
            d[fn[:-5]] = (float(m.group(1)), float(m.group(2)))
    return d


assign, manifest = {}, {}
for scene in sorted(os.listdir(root)):
    sd = os.path.join(root, scene)
    if not os.path.isdir(sd):
        continue
    cavs = sorted([c for c in os.listdir(sd) if os.path.isdir(os.path.join(sd, c)) and int(c) >= 0], key=int)
    P = {c: poses(os.path.join(sd, c)) for c in cavs}
    best = None
    for a, b in itertools.combinations(cavs, 2):
        common = sorted(set(P[a]) & set(P[b]))
        if not common:
            continue
        dist = [math.dist(P[a][t], P[b][t]) for t in common]
        m = sum(dist) / len(dist)
        if best is None or m < best[0]:
            best = (m, a, b, common, dist)
    if best is None:
        manifest[scene] = {"cavs": cavs, "usable": False}
        continue
    m, ego, col, frames, dist = best
    assign[scene] = {c: "m9" for c in cavs}
    assign[scene][ego] = EGO_LABEL
    assign[scene][col] = COLLAB_LABEL
    manifest[scene] = {
        "cavs": cavs, "ego": ego, "collab": col, "n_frames": len(frames),
        "mean_dist_m": round(m, 2), "max_dist_m": round(max(dist), 2),
        "frames_in_range": sum(d <= COMM for d in dist),
        "frames_out_of_range": sum(d > COMM for d in dist),
        "first_frame": frames[0], "last_frame": frames[-1],
    }
    print(scene, manifest[scene])

json.dump(assign, open(out_assign, "w"), indent=2, sort_keys=True)
json.dump(manifest, open(out_manifest, "w"), indent=2, sort_keys=True)
print("scenes:", len(assign), "frames total:", sum(v.get("n_frames", 0) for v in manifest.values()))
