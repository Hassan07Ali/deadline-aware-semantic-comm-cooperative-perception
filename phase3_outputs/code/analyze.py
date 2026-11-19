"""Stage 3: tables, paired scene-bootstrap, figures from replay result files.

  python phase3_scripts/analyze.py --pkl ~/cnet/out/run_main.pkl [more.pkl ...] --out ~/cnet/out/report --primary_D 150 --primary_B 16384
Everything is computed from the saved per-frame logs; nothing is edited by hand.
"""
import argparse
import csv
import json
import os
import pickle
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["B0", "B1", "B2", "B3", "ours", "U0", "A1", "A2", "A3", "A7", "R1"]
LABEL = {"B0": "B0 ego only", "B1": "B1 uniform BEV", "A7": "A7 oracle feedback", "R1": "R1 +1 retry", "B2": "B2 random", "B3": "B3 confidence", "ours": "Ours", "U0": "U0 unconstrained",
         "A1": "A1 no quality", "A2": "A2 no need", "A3": "A3 no admission"}
THR = (0.3, 0.5, 0.7)


def voc_ap(rec, prec):
    rec = [0.0] + list(rec) + [1.0]
    prec = [0.0] + list(prec) + [0.0]
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    ap = 0.0
    for i in range(1, len(rec)):
        if rec[i] != rec[i - 1]:
            ap += (rec[i] - rec[i - 1]) * prec[i]
    return ap


def ap_of(tp, sc, gt):
    if gt == 0 or len(tp) == 0:
        return 0.0
    o = np.argsort(-sc, kind="stable")
    tp = tp[o].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    return voc_ap(ctp / gt, ctp / (ctp + cfp))


def pooled(frames, thr):
    if not frames:
        return np.zeros(0), np.zeros(0), 0
    tp = np.concatenate([f["st"][thr][0] for f in frames])
    sc = np.concatenate([f["st"][thr][1] for f in frames])
    return tp, sc, sum(f["st"][thr][2] for f in frames)


def ap_frames(frames, thr):
    return ap_of(*pooled(frames, thr))


def crit_recall(frames):
    gt = sum(f["st"]["crit_gt"] for f in frames)
    hit = sum(f["st"][0.5][3] for f in frames)
    return hit / gt if gt else float("nan")


def per_scene(frames):
    d = defaultdict(list)
    for f in frames:
        d[f["scene"]].append(f)
    return d


def bootstrap_delta(cfgA, cfgB, thr, nboot=2000, seed=0, block=None):
    """Paired bootstrap of AP(A)-AP(B). cfg = {trace: frames}. Scene-level (default) or block-level within scenes."""
    rng = np.random.RandomState(seed)
    traces = sorted(set(cfgA) & set(cfgB))
    SA = {t: per_scene(cfgA[t]) for t in traces}
    SB = {t: per_scene(cfgB[t]) for t in traces}
    scenes = sorted(set(SA[traces[0]]) & set(SB[traces[0]]))
    base = np.mean([ap_frames(cfgA[t], thr) - ap_frames(cfgB[t], thr) for t in traces])
    deltas = []
    for _ in range(nboot):
        if block is None:
            pick = rng.choice(scenes, size=len(scenes), replace=True)
            vals = []
            for t in traces:
                fa = [f for s in pick for f in SA[t][s]]
                fb = [f for s in pick for f in SB[t][s]]
                vals.append(ap_frames(fa, thr) - ap_frames(fb, thr))
        else:                                     # block bootstrap over frames within the pooled sequence
            vals = []
            nfr = len(cfgA[traces[0]])
            nblk = int(np.ceil(nfr / block))
            starts = rng.randint(0, nblk, size=nblk) * block
            for t in traces:
                ia = [i for s in starts for i in range(s, min(s + block, nfr))]
                a_sorted = sorted(cfgA[t], key=lambda f: f["idx"]); b_sorted = sorted(cfgB[t], key=lambda f: f["idx"])
                vals.append(ap_frames([a_sorted[i] for i in ia], thr) - ap_frames([b_sorted[i] for i in ia], thr))
        deltas.append(np.mean(vals))
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_two = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return base, lo, hi, p_two


def holm(pvals):
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.zeros(m)
    run = 0.0
    for rank, i in enumerate(idx):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--primary_D", type=float, default=150)
    ap.add_argument("--primary_B", type=int, default=16384)
    ap.add_argument("--nboot", type=int, default=2000)
    opt = ap.parse_args()
    os.makedirs(opt.out, exist_ok=True)

    G = defaultdict(lambda: defaultdict(list))        # (cond, method, D, B) -> trace -> frames
    timing = None
    for p in opt.pkl:
        d = pickle.load(open(p, "rb"))
        timing = d["timing"]
        for x in d["results"]:
            G[(x["cond"], x["method"], x["D"], x["budget"])][x["trace"]].append(x)

    # ---------------- main table
    rows = []
    for k in sorted(G, key=lambda k: (k[0], k[2], k[3], ORDER.index(k[1]) if k[1] in ORDER else 99)):
        tr = G[k]
        aps = {t: [ap_frames(tr[tr_], t) for tr_ in sorted(tr)] for t in THR}
        allf = [f for tr_ in tr.values() for f in tr_]
        coop = [f for f in allf if f["k_sel"] > 0]
        ready = np.array([f["ready_ms"] for f in allf if not np.isnan(f["ready_ms"])])
        rows.append({
            "cond": k[0], "method": k[1], "D_ms": k[2], "budget_B": k[3], "n_traces": len(tr), "n_frames": len(allf) // len(tr),
            "AP@0.3": np.mean(aps[0.3]), "AP@0.5": np.mean(aps[0.5]), "AP@0.7": np.mean(aps[0.7]),
            "AP@0.7_sd_traces": np.std(aps[0.7]),
            "crit_recall@0.5": np.mean([crit_recall(tr[t]) for t in tr]),
            "bytes_per_frame": np.mean([f["bytes"] for f in allf]),
            "coop_share": np.mean([f["kind"].startswith("coop") for f in allf]),
            "miss_rate_when_sent": (np.mean([not f["kind"].startswith("coop") for f in coop]) if coop else float("nan")),
            "tokens_sent": np.mean([f["k_sel"] for f in allf]), "tokens_used": np.mean([f["k_used"] for f in allf]),
            "tokens_lost": np.mean([f["k_lost"] for f in allf]), "tokens_late": np.mean([f["k_late"] for f in allf]),
            "latency_p50_ms": float(np.median(ready)) if len(ready) else float("nan"),
            "latency_p95_ms": float(np.percentile(ready, 95)) if len(ready) else float("nan"),
        })
    with open(os.path.join(opt.out, "main_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("cond method   D    B    AP@0.3 AP@0.5 AP@0.7  crit-rec  bytes   coop  miss")
    for r in rows:
        print(f"{r['cond']:3s} {r['method']:5s} {r['D_ms']:4.0f} {r['budget_B']:6d} {r['AP@0.3']:.3f}  {r['AP@0.5']:.3f}  {r['AP@0.7']:.3f}  "
              f"{r['crit_recall@0.5']:.3f}  {r['bytes_per_frame']:6.0f}  {r['coop_share']:.2f}  {r['miss_rate_when_sent']:.2f}")

    # ---------------- hypothesis tests (paired scene bootstrap)
    D, B = opt.primary_D, opt.primary_B
    tests = []

    def cfg(c, m, d=D, b=B):
        return G.get((c, m, d, b))

    def add(name, c, a, b, thr, d=D, bb=B, c2=None):
        A, Bc = cfg(c, a, d, bb), cfg(c2 or c, b, d, bb)
        if A is None or Bc is None:
            return
        base, lo, hi, p = bootstrap_delta(A, Bc, thr, opt.nboot)
        _, lo_b, hi_b, _ = bootstrap_delta(A, Bc, thr, max(200, opt.nboot // 5), seed=1, block=20)
        tests.append({"test": name, "thr": thr, "delta_AP": base, "scene_CI_lo": lo, "scene_CI_hi": hi, "p_scene": p,
                      "frame_block_CI_lo": lo_b, "frame_block_CI_hi": hi_b})

    add("H1: ours - B3, C3", "C3", "ours", "B3", 0.7)
    add("H1b: ours - B3, C3 (AP@0.5)", "C3", "ours", "B3", 0.5)
    add("ours - B2, C3", "C3", "ours", "B2", 0.7)
    add("ours - B0 (ego only), C3", "C3", "ours", "B0", 0.7)
    add("B3 - B0, C3", "C3", "B3", "B0", 0.7)
    add("H2/A3: ours - no-admission, C3", "C3", "ours", "A3", 0.7)
    add("A1: ours - no-quality, C3", "C3", "ours", "A1", 0.7)
    add("A2: ours - no-need, C3", "C3", "ours", "A2", 0.7)
    add("H3: ours - B3, C0 (non-inferiority margin -0.01)", "C0", "ours", "B3", 0.7)
    add("U0 - B0, C0 (value of cooperation)", "C0", "U0", "B0", 0.7)
    add("ours - B1 (uniform compression), C3", "C3", "ours", "B1", 0.7)
    add("A7: oracle same-frame feedback - ours, C3", "C3", "A7", "ours", 0.7)
    add("R1: ours + one retry - ours, C3", "C3", "R1", "ours", 0.7)
    add("A2: ours - no-need, C0", "C0", "ours", "A2", 0.7)
    if tests:
        adj = holm([max(t["p_scene"], 1e-4) for t in tests])
        for t, a in zip(tests, adj):
            t["p_holm"] = a
        with open(os.path.join(opt.out, "tests.json"), "w") as f:
            json.dump(tests, f, indent=2, default=float)
        print("\nPaired bootstrap (2000 resamples over the 16 scenes; Holm over the family):")
        for t in tests:
            print(f"  {t['test']:55s} dAP@{t['thr']} = {t['delta_AP']*100:+6.2f} pts  scene-95%CI [{t['scene_CI_lo']*100:+.2f},{t['scene_CI_hi']*100:+.2f}]"
                  f"  block-95%CI [{t['frame_block_CI_lo']*100:+.2f},{t['frame_block_CI_hi']*100:+.2f}]  p_holm={t['p_holm']:.3f}")

    # H2: critical-region recall
    a, b = cfg("C3", "ours"), cfg("C3", "A3")
    if a and b:
        print("\nH2 critical-region recall@0.5 (C3): ours %.3f vs no-admission %.3f" % (
            np.mean([crit_recall(a[t]) for t in a]), np.mean([crit_recall(b[t]) for t in b])))

    # ---------------- figures
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"B1": "#6baed6", "A7": "#999999", "R1": "#17becf", "B0": "#8c8c8c", "B2": "#c9a227", "B3": "#d95f02", "ours": "#1b9e77", "U0": "#7570b3", "A1": "#66a61e", "A2": "#e7298a", "A3": "#a6761d"}

    def sel(c, d=D, b=B):
        return [r for r in rows if r["cond"] == c and r["D_ms"] == d and r["budget_B"] == b]

    # F1 main bars
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, c in zip(axes, ["C0", "C3"]):
        rr = [r for r in sel(c) if r["method"] in ("B0", "B1", "B2", "B3", "ours", "U0")]
        rr.sort(key=lambda r: ORDER.index(r["method"]))
        x = np.arange(len(rr))
        ax.bar(x - 0.2, [r["AP@0.5"] for r in rr], 0.4, label="AP@0.5", color="#9ecae1")
        ax.bar(x + 0.2, [r["AP@0.7"] for r in rr], 0.4, label="AP@0.7", color="#3182bd")
        ax.set_xticks(x); ax.set_xticklabels([LABEL[r["method"]].replace(" ", "\n", 1) for r in rr], fontsize=7)
        ax.set_title(f"{c}  (D={D:.0f} ms, {B//1024} KiB)")
    axes[0].set_ylabel("average precision"); axes[0].legend(frameon=False)
    plt.tight_layout(); plt.savefig(os.path.join(opt.out, "F1_main_comparison.png"), dpi=170); plt.close()

    # F2 AP vs deadline, F3 AP vs budget
    for tag, key, xs_label in (("F2_deadline_sweep", "D_ms", "deadline D (ms)"), ("F3_budget_sweep", "budget_B", "byte budget (bytes/frame)")):
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        any_line = False
        for m in ("B0", "B1", "B2", "B3", "ours", "U0"):
            pts = [(r[key], r["AP@0.7"]) for r in rows if r["cond"] == "C3" and r["method"] == m and
                   ((key == "D_ms" and r["budget_B"] == B) or (key == "budget_B" and r["D_ms"] == D))]
            pts.sort()
            if len(pts) > 1:
                ax.plot(*zip(*pts), marker="o", label=LABEL[m], color=colors[m]); any_line = True
        if any_line:
            ax.set_xlabel(xs_label); ax.set_ylabel("AP@0.7 (C3)"); ax.legend(frameon=False, fontsize=7)
            plt.tight_layout(); plt.savefig(os.path.join(opt.out, tag + ".png"), dpi=170)
        plt.close()

    # F4 ablations
    rr = [r for r in sel("C3") if r["method"] in ("ours", "A1", "A2", "A3", "A7", "R1", "B3")]
    if len(rr) > 1:
        rr.sort(key=lambda r: ORDER.index(r["method"]))
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        ax.bar([LABEL[r["method"]] for r in rr], [r["AP@0.7"] for r in rr], color=[colors[r["method"]] for r in rr])
        ax.set_ylabel("AP@0.7 (C3)"); plt.xticks(rotation=25, ha="right", fontsize=7)
        plt.tight_layout(); plt.savefig(os.path.join(opt.out, "F4_ablations.png"), dpi=170); plt.close()

    # F5 per-scene delta ours - B3 (C3)
    A, Bc = cfg("C3", "ours"), cfg("C3", "B3")
    if A and Bc:
        scenes = sorted({f["scene"] for f in A[0]})
        dl = []
        for s in scenes:
            va = np.mean([ap_frames([f for f in A[t] if f["scene"] == s], 0.7) for t in A])
            vb = np.mean([ap_frames([f for f in Bc[t] if f["scene"] == s], 0.7) for t in Bc])
            dl.append(va - vb)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar([str(s) for s in scenes], np.array(dl) * 100, color=["#1b9e77" if v >= 0 else "#d95f02" for v in dl])
        ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("test scene index"); ax.set_ylabel("AP@0.7 difference (points)")
        ax.set_title("Ours minus B3 per scene (C3)")
        plt.tight_layout(); plt.savefig(os.path.join(opt.out, "F5_per_scene_delta.png"), dpi=170); plt.close()

    # F6 bytes and cooperation share
    rr = [r for r in sel("C3") if r["method"] in ("B2", "B3", "ours", "A3")]
    if rr:
        fig, ax = plt.subplots(1, 2, figsize=(7.2, 3))
        ax[0].bar([LABEL[r["method"]] for r in rr], [r["bytes_per_frame"] for r in rr], color=[colors[r["method"]] for r in rr])
        ax[0].set_ylabel("wire bytes per frame"); ax[0].tick_params(axis="x", rotation=25, labelsize=7)
        ax[1].bar([LABEL[r["method"]] for r in rr], [r["coop_share"] for r in rr], color=[colors[r["method"]] for r in rr])
        ax[1].set_ylabel("share of frames with a cooperative result"); ax[1].tick_params(axis="x", rotation=25, labelsize=7)
        plt.tight_layout(); plt.savefig(os.path.join(opt.out, "F6_bytes_and_coop_share.png"), dpi=170); plt.close()
    print("figures + tables saved to", opt.out)


if __name__ == "__main__":
    main()
