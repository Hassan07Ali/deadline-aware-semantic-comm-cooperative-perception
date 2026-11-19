"""Generic paired scene bootstrap: python paired.py --pkl a.pkl b.pkl --cond ST --A ours --B B3 [--D 100 --budget 16384 --thr 0.7]"""
import argparse, pickle, sys
from collections import defaultdict
sys.path.insert(0, "phase3_scripts")
from analyze import bootstrap_delta
ap = argparse.ArgumentParser()
ap.add_argument("--pkl", nargs="+", required=True); ap.add_argument("--cond", required=True)
ap.add_argument("--pairs", nargs="+", required=True, help="A:B pairs e.g. ours:B3 ours:A1")
ap.add_argument("--D", type=float, default=100); ap.add_argument("--budget", type=int, default=16384)
ap.add_argument("--thr", type=float, default=0.7); ap.add_argument("--nboot", type=int, default=2000)
o = ap.parse_args()
G = defaultdict(lambda: defaultdict(list))
for p in o.pkl:
    for x in pickle.load(open(p, "rb"))["results"]:
        G[(x["cond"], x["method"], x["D"], x["budget"])][x["trace"]].append(x)
for pr in o.pairs:
    a, b = pr.split(":")
    A, B = G[(o.cond, a, o.D, o.budget)], G[(o.cond, b, o.D, o.budget)]
    d, lo, hi, p = bootstrap_delta(A, B, o.thr, o.nboot)
    print(f"{o.cond} D={o.D:.0f} B={o.budget}: {a} - {b}: dAP@{o.thr} = {d*100:+.2f} pts  scene-95%CI [{lo*100:+.2f},{hi*100:+.2f}]  p={p:.3f}")
