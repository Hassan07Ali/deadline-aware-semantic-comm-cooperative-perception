"""Phase 2 Step 11: why does `ours` miss critical-region objects? (C3, IoU 0.5)

Run replay.py with --keep_tokens first (methods B0,U0,ours; C3), then:
  python phase3_scripts/failure_analysis.py --pkl ... --cache_deg ~/cnet/cache_ll_degraded --out ~/cnet/out/report
Categories for each critical-region ground-truth object that `ours` does not detect:
  sensor outage / no sensor evidence / detector error even with all features (U0 also misses) /
  not selected / lost in transmission / late / used but still missed / (recovered: ours detects but B0 does not)
"""
import argparse, json, os, pickle
from collections import Counter, defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--pkl", nargs="+", required=True)
ap.add_argument("--cache_deg", default=os.path.expanduser("~/cnet/cache_ll_degraded"))
ap.add_argument("--out", required=True)
ap.add_argument("--D", type=float, default=100)
opt = ap.parse_args()
os.makedirs(opt.out, exist_ok=True)

F = defaultdict(dict)
for p in opt.pkl:
    for x in pickle.load(open(p, "rb"))["results"]:
        if x["cond"] == "C3" and x["D"] == opt.D and x["trace"] == 0 and x["method"] in ("B0", "U0", "ours"):
            F[x["idx"]][x["method"]] = x

cats = Counter()
recovered = 0
n_crit = 0
for idx, d in sorted(F.items()):
    if not all(m in d for m in ("B0", "U0", "ours")):
        continue
    o, u, b = d["ours"], d["U0"], d["B0"]
    gt = o["st"]["gt_ctr"]
    if len(gt) == 0:
        continue
    crit = (gt[:, 0] > 0) & (gt[:, 0] < 40) & (np.abs(gt[:, 1]) < 10)
    if not crit.any():
        continue
    rec = torch.load(os.path.join(opt.cache_deg, f"{idx:05d}.pt"), weights_only=False)
    cnt = rec["lidar_cnt"]
    ptm = rec["data"]["pairwise_t_matrix"][0, 0, 1].numpy() if rec["col_feat"] is not None else None   # ego -> collaborator
    tok = o.get("tok")
    sel = set(tok["idx"].tolist()) if tok is not None else set()
    ok = {int(t) for t, k in zip(tok["idx"], tok["ok"]) if k} if tok is not None else set()
    dl = {int(t) for t, k in zip(tok["idx"], tok["deliv"]) if k} if tok is not None else set()
    for i in np.flatnonzero(crit):
        n_crit += 1
        hit_o, hit_u, hit_b = o["st"]["m05"][i], u["st"]["m05"][i], b["st"]["m05"][i]
        if hit_o:
            if not hit_b:
                recovered += 1
            cats["detected"] += 1
            continue
        if o.get("outage"):
            cats["sensor outage frame"] += 1
            continue
        x, y = gt[i]
        te = (int((x + 70.4) // 1.6), int((y + 40) // 1.6))
        ev = cnt is not None and 0 <= te[0] < 88 and 0 <= te[1] < 50 and cnt[0][te[1], te[0]] > 0
        if ptm is not None:
            xc = ptm[0, 0] * x + ptm[0, 1] * y + ptm[0, 3]; yc = ptm[1, 0] * x + ptm[1, 1] * y + ptm[1, 3]
            tc = (int((xc + 70.4) // 1.6), int((yc + 40) // 1.6))
            if 0 <= tc[0] < 88 and 0 <= tc[1] < 50:
                ev = ev or cnt[1][tc[1], tc[0]] > 0
        else:
            tc = None
        if not ev:
            cats["no sensor evidence (no LiDAR points)"] += 1
            continue
        if not hit_u:
            cats["detector error (U0 with all features also misses)"] += 1
            continue
        near = set()
        if tc is not None:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ti, tj = tc[1] + di, tc[0] + dj
                    if 0 <= ti < 50 and 0 <= tj < 88:
                        near.add(ti * 88 + tj)
        if not (near & sel):
            cats["not selected"] += 1
        elif not (near & dl):
            cats["lost in transmission"] += 1
        elif not (near & ok):
            cats["late"] += 1
        else:
            cats["used but still missed (quantisation / fusion)"] += 1

print(f"critical-region objects examined (C3, D={opt.D:.0f} ms, trace 0): {n_crit}")
for k, v in cats.most_common():
    print(f"  {k:55s} {v:6d}  {100*v/n_crit:5.1f} %")
print(f"  objects detected by ours but missed by ego-only B0 (recovered by cooperation): {recovered}")
json.dump({"n_crit": n_crit, "categories": dict(cats), "recovered_by_cooperation": recovered}, open(os.path.join(opt.out, "failure_analysis.json"), "w"), indent=2)

miss = {k: v for k, v in cats.items() if k != "detected"}
if miss:
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ks = sorted(miss, key=miss.get)
    ax.barh(ks, [100 * miss[k] / n_crit for k in ks], color="#d95f02")
    ax.set_xlabel("% of critical-region objects (C3)"); ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(opt.out, "F7_failure_analysis.png"), dpi=170)
print("saved")
