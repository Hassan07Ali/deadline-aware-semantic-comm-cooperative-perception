"""Summarise sensitivity / stress / experiment-3 runs: one row per condition name (or parameter), mean over traces."""
import argparse, pickle, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "phase3_scripts")
from analyze import ap_frames  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("--pkl", nargs="+", required=True)
ap.add_argument("--labels", nargs="+", required=True, help="label per pkl, same order (e.g. rate=1Mb/s)")
ap.add_argument("--methods", default="B0,B2,B3,ours,U0")
ap.add_argument("--out", required=True)
opt = ap.parse_args()
methods = opt.methods.split(",")
rows = []
for p, lab in zip(opt.pkl, opt.labels):
    d = pickle.load(open(p, "rb"))
    g = defaultdict(lambda: defaultdict(list))
    for x in d["results"]:
        g[(x["cond"], x["method"], x["D"], x["budget"])][x["trace"]].append(x)
    for (cond, m, D, B), tr in sorted(g.items()):
        if m not in methods:
            continue
        aps = [ap_frames(f, 0.7) for f in tr.values()]
        allf = [f for v in tr.values() for f in v]
        rows.append((lab, cond, m, D, B, np.mean(aps), np.mean([f["bytes"] for f in allf]), np.mean([f["kind"].startswith("coop") for f in allf])))
print("label | cond | method | D | B | AP@0.7 | bytes/frame | coop share")
for r in rows:
    print(f"{r[0]:22s} {r[1]:5s} {r[2]:5s} {r[3]:5.0f} {r[4]:6d}  {r[5]:.3f}  {r[6]:7.0f}  {r[7]:.2f}")
import csv
with open(opt.out + ".csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["label", "cond", "method", "D_ms", "budget_B", "AP@0.7", "bytes_per_frame", "coop_share"]); w.writerows(rows)
