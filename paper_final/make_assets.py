"""Builds every figure and every numeric table of the paper from the saved logs in phase3_outputs/data.
Nothing here is typed by hand: all values are read from CSV/JSON/TXT logs.
Run:  python make_assets.py
Writes: figs/*.pdf|png (300 dpi) and tables_auto.tex
"""
import json, os, re, math
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "phase3_outputs", "data")
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 7.5,
                     "axes.labelsize": 7.5, "axes.titlesize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.linewidth": 0.6, "figure.dpi": 100, "savefig.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = {"B0": "#7f7f7f", "B1": "#c9a227", "B2": "#b5651d", "B3": "#e07b00", "ours": "#0b5394", "U0": "#222222",
       "A1": "#6aa84f", "A2": "#8e7cc3", "A3": "#cc4125", "A7": "#45818e", "R1": "#a64d79"}
NAME = {"B0": "Ego only", "B1": "Uniform", "B2": "Random", "B3": "Confidence-only", "ours": "Ours", "U0": "Unconstrained"}
W1, W2 = 3.5, 7.16


def rd(sub):
    return pd.read_csv(os.path.join(DATA, sub, "main_table.csv"))


def pick(df, cond, method, D=100.0, B=16384):
    r = df[(df.cond == cond) & (df.method == method) & (df.D_ms == D) & (df.budget_B == B)]
    assert len(r) == 1, (cond, method, D, B, len(r))
    return r.iloc[0]


def tests(sub):
    return {t["test"]: t for t in json.load(open(os.path.join(DATA, sub, "tests.json")))}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


M, EC, CC = rd("main"), rd("ec"), rd("cc")
TM = json.load(open(os.path.join(DATA, "timing_ll.json")))
T_MAIN, T_EC, T_PRIM = tests("main"), tests("ec"), tests("prim10")

# ------------------------------------------------------------------ Fig. 1: architecture
fig, ax = plt.subplots(figsize=(W2, 1.55)); ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")


def box(x, y, w, h, text, fc="#eef3fa", ec="#0b5394", fs=5.9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(x0, y0, x1, y1, style="-|>", ls="-", col="#333333"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=7, lw=0.7, ls=ls, color=col))


ax.text(1, 28.3, "Sender (collaborator car B)", fontsize=7, fontweight="bold", color="#0b5394")
ax.text(52, 28.3, "Receiver (ego car A)", fontsize=7, fontweight="bold", color="#0b5394")
box(1, 18.5, 12.5, 7, "Sensor (LiDAR\nor camera),\nclean/degraded")
box(16.5, 18.5, 13.5, 7, "Frozen HEAL\nencoder, 64$\\times$100\n$\\times$176 BEV map")
box(33, 18.5, 15, 7, "Tile score $s_i$:\nquality, confidence,\nneed, critical region", fc="#fff2cc", ec="#b45f06")
box(33, 8.5, 15, 7, "Deadline admission:\nlargest $K$ that can\narrive before $D$", fc="#fff2cc", ec="#b45f06")
box(16.5, 8.5, 13.5, 7, "int8 tokens,\n4 per packet,\n+72 B header")
box(1, 8.5, 12.5, 7, "Emulated link:\nrate, queue,\nloss", fc="#f4cccc", ec="#990000")
box(55, 18.5, 13, 7, "Ego sensor +\nfrozen encoder")
box(55, 8.5, 13, 7, "Occupancy map,\nprevious-frame\nfeedback (584 B)", fc="#fff2cc", ec="#b45f06")
box(71, 13.5, 15, 7, "Frozen HEAL fusion\n+ detector (zeros\nfor missing tiles)")
box(88.5, 13.5, 11, 7, "Output at $D$:\ncooperative,\nego-only or empty", fc="#d9ead3", ec="#38761d")
arrow(13.5, 22, 16.5, 22); arrow(30, 22, 33, 22); arrow(40.5, 18.5, 40.5, 15.5); arrow(33, 12, 30, 12); arrow(16.5, 12, 13.5, 12)
arrow(7.2, 8.5, 7.2, 4.5); ax.plot([7.2, 78.5], [4.5, 4.5], color="#333333", lw=0.7); arrow(78.5, 4.5, 78.5, 13.5)
ax.text(42, 2.9, "tokens that arrive intact before $D - T_{\\mathrm{res}}$", fontsize=6, ha="center")
arrow(68, 22, 71, 18); arrow(68, 12, 71, 16); arrow(86, 17, 88.5, 17)
arrow(55, 12, 48, 21, ls="--", col="#b45f06"); ax.text(49.3, 17.4, "feedback", fontsize=5.8, rotation=52, color="#b45f06")
ax.text(50, 0.4, "Solid arrows: data path. Dashed: previous-frame feedback. Orange boxes are the proposed policy; all neural networks are frozen.",
        ha="center", fontsize=6.2, style="italic")
save(fig, "fig1_architecture")

# ------------------------------------------------------------------ Fig. 2: main comparison + ablation deltas
fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 1.7), gridspec_kw={"width_ratios": [1.15, 1]})
meths = ["B0", "B1", "B2", "B3", "ours", "U0"]; x = np.arange(len(meths)); w = 0.38
c0 = [pick(M, "C0", m)["AP@0.7"] for m in meths]; c3 = [pick(M, "C3", m)["AP@0.7"] for m in meths]
a.bar(x - w / 2, c0, w, color=[COL[m] for m in meths], alpha=0.45, label="C0 clean")
a.bar(x + w / 2, c3, w, color=[COL[m] for m in meths], label="C3 degraded sensors + bad link")
for xi, v in zip(x + w / 2, c3): a.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=5.8)
a.set_xticks(x); a.set_xticklabels([NAME[m].replace("-", "-\n") if m == "B3" else NAME[m] for m in meths], rotation=0, fontsize=6.2)
a.set_ylabel("AP@0.7 (LiDAR pair, D = 100 ms)"); a.set_ylim(0.5, 1.0); a.set_title("(a) Methods", loc="left")
from matplotlib.patches import Patch
a.legend(handles=[Patch(fc="#999999", alpha=0.45, label="C0 clean"), Patch(fc="#999999", label="C3 (main)")], loc="upper left", frameon=False)
# ablation deltas (variant - ours), points; CI flipped where the log reports ours - variant
rows = []
t = T_MAIN
for key, lab, flip in [("A1: ours - no-quality, C3", "A1 no quality", True), ("A2: ours - no-need, C3", "A2 no need", True),
                       ("H2/A3: ours - no-admission, C3", "A3 no admission", True), ("A7: oracle same-frame feedback - ours, C3", "A7 oracle feedback", False),
                       ("R1: ours + one retry - ours, C3", "R1 one retry", False)]:
    d, lo, hi = t[key]["delta_AP"] * 100, t[key]["scene_CI_lo"] * 100, t[key]["scene_CI_hi"] * 100
    rows.append((lab, -d if flip else d, -hi if flip else lo, -lo if flip else hi))
ys = np.arange(len(rows))[::-1]
for y, (lab, d, lo, hi) in zip(ys, rows):
    b.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o", color=COL["ours"], ms=3.5, capsize=2, lw=0.9)
b.axvline(0, color="#333333", lw=0.7); b.set_yticks(ys); b.set_yticklabels([r[0] for r in rows])
b.set_xlabel("Variant $-$ ours, AP@0.7 (points), 95% scene CI"); b.set_title("(b) Ablations (C3)", loc="left")
fig.tight_layout(); save(fig, "fig2_main_ablation")

# ------------------------------------------------------------------ Fig. 3: deadline and budget sweeps
fig, ax3 = plt.subplots(1, 3, figsize=(W2, 1.5))
a, b, c = ax3
Ds = [50, 75, 100, 150]
for m in ["B0", "B3", "ours", "U0"]:
    a.plot(Ds, [pick(M, "C3", m, D=float(d))["AP@0.7"] for d in Ds], marker="o", ms=3, lw=1, color=COL[m], label=NAME.get(m, "R1 one retry"))
a.set_xlabel("Deadline D (ms)"); a.set_ylabel("AP@0.7 (C3, 16 KiB)"); a.set_xticks(Ds); a.set_title("(a) Deadline", loc="left")
a.legend(frameon=False, loc="lower right", fontsize=5.6, ncol=2)
Bs = [8192, 16384, 32768]
for m in ["B1", "B2", "B3", "ours"]:
    b.plot([x / 1024 for x in Bs], [pick(M, "C3", m, B=bb)["AP@0.7"] for bb in Bs], marker="o", ms=3, lw=1, color=COL[m], label=NAME[m])
b.axhline(pick(M, "C3", "B0")["AP@0.7"], color=COL["B0"], ls="--", lw=0.8, label="Ego only")
b.set_xscale("log", base=2); b.set_xticks([8, 16, 32]); b.set_xticklabels(["8", "16", "32"]); b.set_xlabel("Byte budget per frame (KiB)")
b.set_ylabel("AP@0.7 (C3, D = 100 ms)"); b.set_title("(b) Budget: accuracy", loc="left"); b.set_ylim(0.60, 0.93); b.legend(frameon=False, fontsize=5.4, loc="upper center", ncol=3)
for m in ["B3", "ours"]:
    c.plot([x / 1024 for x in Bs], [pick(M, "C3", m, B=bb)["bytes_per_frame"] / 1024 for bb in Bs], marker="o", ms=3, lw=1, color=COL[m], label=NAME[m])
c.plot([8, 16, 32], [8, 16, 32], color="#999999", ls=":", lw=0.8, label="Budget")
c.set_xscale("log", base=2); c.set_yscale("log", base=2); c.set_xticks([8, 16, 32]); c.set_xticklabels(["8", "16", "32"])
c.set_yticks([8, 16, 32]); c.set_yticklabels(["8", "16", "32"]); c.set_xlabel("Byte budget per frame (KiB)"); c.set_ylabel("Bytes sent per frame (KiB)")
c.set_title("(c) Budget: bytes", loc="left"); c.legend(frameon=False, fontsize=5.8, loc="upper left")
fig.tight_layout(); save(fig, "fig3_sweeps")

# ------------------------------------------------------------------ Fig. 4: camera ego + failure analysis
fa = json.load(open(os.path.join(DATA, "main", "failure_analysis.json")))
fig, (a, b, c) = plt.subplots(1, 3, figsize=(W2, 1.6), gridspec_kw={"width_ratios": [1.1, 1.0, 1.25]})
meths = ["B0", "B2", "B3", "ours", "U0"]; x = np.arange(len(meths)); w = 0.38
v0 = [pick(EC, "C0", m, D=150.0)["AP@0.7"] for m in meths]; v3 = [pick(EC, "C3", m, D=150.0)["AP@0.7"] for m in meths]
a.bar(x - w / 2, v0, w, color=[COL[m] for m in meths], alpha=0.45); a.bar(x + w / 2, v3, w, color=[COL[m] for m in meths])
for xi, v in zip(x + w / 2, v3): a.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=5.6)
a.set_xticks(x); a.set_xticklabels(["Ego", "Rand.", "Conf.", "Ours", "Uncon."], fontsize=6)
a.set_ylabel("AP@0.7 (D = 150 ms)"); a.set_title("(a) Camera ego car", loc="left")
a.legend(handles=[Patch(fc="#999999", alpha=0.45, label="C0"), Patch(fc="#999999", label="C3")], frameon=False, loc="upper left")
# budget sweep (camera ego, C3, D=150): 16 KiB from ec main table, other budgets from budget_sweep_pooled.txt
pool = {}
for ln in open(os.path.join(DATA, "ec", "budget_sweep_pooled.txt")):
    mm = re.match(r"\('C3', '(\w+)', 150.0, (\d+)\) AP@0.7=([\d.]+) bytes=(\d+)", ln.strip())
    if mm: pool[(mm.group(1), int(mm.group(2)))] = float(mm.group(3))
bud = [8192, 16384, 32768, 65536]
def ecv(m, bb): return pool[(m, bb)] if bb != 16384 else round(pick(EC, "C3", m, D=150.0)["AP@0.7"], 3)
for m in ["B3", "ours"]:
    b.plot([bb / 1024 for bb in bud], [ecv(m, bb) for bb in bud], marker="o", ms=3, lw=1, color=COL[m], label=NAME[m])
b.axhline(pick(EC, "C3", "U0", D=150.0)["AP@0.7"], color=COL["U0"], ls="--", lw=0.8, label="Unconstrained")
b.set_xscale("log", base=2); b.set_xticks([8, 16, 32, 64]); b.set_xticklabels(["8", "16", "32", "64"])
b.set_xlabel("Byte budget (KiB)"); b.set_ylabel("AP@0.7 (C3)"); b.set_title("(b) Budget, camera ego", loc="left"); b.set_ylim(0.08, 0.60); b.legend(frameon=False, loc="lower right", fontsize=5.8)
order = ["detected", "no sensor evidence (no LiDAR points)", "sensor outage frame", "used but still missed (quantisation / fusion)",
         "detector error (U0 with all features also misses)", "lost in transmission", "not selected"]
short = {"no sensor evidence (no LiDAR points)": "no LiDAR points", "sensor outage frame": "sensor outage",
         "used but still missed (quantisation / fusion)": "used, still missed", "detector error (U0 with all features also misses)": "detector error",
         "lost in transmission": "lost in link", "not selected": "not selected"}
cats = order[1:]; vals = [fa["categories"][k] / fa["n_crit"] * 100 for k in cats]
yy = np.arange(len(cats))[::-1]
c.barh(yy, vals, color=["#7f7f7f", "#7f7f7f", "#7f7f7f", "#7f7f7f", COL["A3"], COL["A3"]], height=0.65)
for y, v, k in zip(yy, vals, cats): c.text(v + 0.05, y, f"{v:.1f}%", va="center", fontsize=6)
c.set_yticks(yy); c.set_yticklabels([short[k] for k in cats]); c.set_xlabel("Share of critical-region objects (%)")
c.set_title(f"(c) Misses, n = {fa['n_crit']:,}", loc="left"); c.set_xlim(0, 4.4)
fig.tight_layout(); save(fig, "fig4_camera_failure")

# ------------------------------------------------------------------ Fig. 5: timeline at the deadline (derived)
from math import ceil
TOK, HDR, NPP, FB = 262, 72, 4, 584
def wire(K): return 0 if K <= 0 else TOK * K + HDR * ceil(K / NPP)
tready = TM["t_enc_col_ms"] + TM["t_select_ms"]; q = 20.0; rate = 5e6; Tres = TM["T_reserve_coop_ms"]
Kmax = max(k for k in range(0, 57) if tready + q + 8000.0 * wire(k) / rate + Tres <= 100.0 or k == 0)
fig, ax = plt.subplots(figsize=(W1, 1.45))
def bar(label, y, K):
    ser = 8000.0 * wire(K) / rate
    segs = [("encode + select", tready, "#a4c2f4"), ("queue", q, "#d9d9d9"), ("serialise", ser, "#f6b26b"), ("decode + fuse + detect ($T_{res}$)", Tres, "#93c47d")]
    x0 = 0
    for nm, d, cc in segs:
        ax.barh(y, d, left=x0, color=cc, edgecolor="white", height=0.5, label=nm if y == 1 else None); x0 += d
    ax.text(x0 + 1.5, y, f"{x0:.1f} ms", va="center", fontsize=6)
    return x0
t56 = bar("", 1, 56); tk = bar("", 0, Kmax)
ax.axvline(100, color="#cc0000", lw=0.9, ls="--"); ax.text(99, 1.52, "D = 100 ms", color="#cc0000", fontsize=6, ha="right")
ax.set_yticks([1, 0]); ax.set_yticklabels([f"All 56 tokens\n(16 KiB budget)", f"Admitted $K$ = {Kmax}"], fontsize=6.2)
ax.set_xlabel("Time after the frame is captured (ms, derived from logged latencies)"); ax.set_xlim(0, 140); ax.set_ylim(-0.5, 1.75)
ax.legend(frameon=False, fontsize=5.4, loc="upper center", ncol=2, bbox_to_anchor=(0.45, -0.38))
save(fig, "fig5_timeline")
print("timeline: tready", tready, "T_res", Tres, "K_max admitted at C3 nominal rate:", Kmax, "t56", t56, "tK", tk)

# ------------------------------------------------------------------ automatic tables
def f3(v): return f"{v:.3f}"
def pts(v): return f"{v * 100:+.2f}"
def ci(t): return f"[{t['scene_CI_lo'] * 100:+.2f}, {t['scene_CI_hi'] * 100:+.2f}]"
def ph(p): return "$<$0.001" if p < 0.0005 else f"{p:.3f}"
def by(v): return f"{v:,.0f}"
out = []
# main table
out.append(r"\begin{table}[t]\caption{Main comparison, LiDAR ego car and LiDAR collaborator, condition C3 (degraded sensors and bad link), $D$ = 100 ms, 16 KiB budget. AP is the mean over 3 link traces. Critical recall is recall inside the critical region at IoU 0.5.}\label{tab:main}")
out.append(r"\centering\scriptsize\setlength{\tabcolsep}{2.6pt}\begin{tabular}{lccccc}\toprule")
out.append(r"Method & AP@0.5 & AP@0.7 & Crit.\ rec. & Bytes & C0 AP@0.7\\\midrule")
for m in ["B0", "B1", "B2", "B3", "ours", "U0"]:
    r3, r0 = pick(M, "C3", m), pick(M, "C0", m)
    name = NAME[m] if m != "ours" else r"\textbf{Ours}"
    cells = [f3(r3["AP@0.5"]), f3(r3["AP@0.7"]), f3(r3["crit_recall@0.5"]), "--" if m in ("U0",) else by(r3["bytes_per_frame"]), f3(r0["AP@0.7"])]
    if m == "ours": cells = [r"\textbf{" + c + "}" for c in cells]
    out.append(name + " & " + " & ".join(cells) + r"\\")
out.append(r"\bottomrule\end{tabular}\end{table}")
# paired tests
out.append(r"\begin{table}[t]\caption{Paired scene-level bootstrap (2,000 resamples over 16 scenes, Holm-adjusted $p$). $\Delta$ is the difference in AP@0.7 in percentage points. The first row uses 10 link traces, the others 3.}\label{tab:tests}")
out.append(r"\centering\scriptsize\setlength{\tabcolsep}{2.5pt}\begin{tabular}{lccc}\toprule Comparison & $\Delta$ (pts) & 95\% CI & Holm $p$\\\midrule")
tp = T_PRIM["H1: ours - B3, C3"]
out.append(f"H1: ours $-$ B3, C3 (10 traces) & {pts(tp['delta_AP'])} & {ci(tp)} & {ph(tp['p_holm'])}" + r"\\")
for key, lab in [("H1: ours - B3, C3", "ours $-$ B3, C3 (3 traces)"), ("H1b: ours - B3, C3 (AP@0.5)", "same, AP@0.5"),
                 ("ours - B0 (ego only), C3", "ours $-$ B0, C3"), ("ours - B2, C3", "ours $-$ B2, C3"), ("ours - B1 (uniform compression), C3", "ours $-$ B1, C3"),
                 ("H3: ours - B3, C0 (non-inferiority margin -0.01)", "H3: ours $-$ B3, C0"), ("U0 - B0, C0 (value of cooperation)", "U0 $-$ B0, C0")]:
    t = T_MAIN[key]; out.append(f"{lab} & {pts(t['delta_AP'])} & {ci(t)} & {ph(t['p_holm'])}" + r"\\")
out.append(r"\bottomrule\end{tabular}\end{table}")
# camera table
out.append(r"\begin{table}[t]\caption{AP@0.7 with one camera car. Experiment B: LiDAR ego, camera collaborator ($D$ = 250 ms). Experiment C: camera ego, LiDAR collaborator ($D$ = 150 ms). 16 KiB budget, 3 traces.}\label{tab:camera}")
out.append(r"\centering\scriptsize\setlength{\tabcolsep}{4pt}\begin{tabular}{lcccc}\toprule & \multicolumn{2}{c}{Exp.\ B (LiDAR ego)} & \multicolumn{2}{c}{Exp.\ C (camera ego)}\\\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
out.append(r"Method & C0 & C3 & C0 & C3\\\midrule")
for m in ["B0", "B1", "B2", "B3", "ours", "U0"]:
    n = NAME[m] if m != "ours" else r"\textbf{Ours}"
    cells = [f3(pick(CC, "C0", m, D=250.0)["AP@0.7"]), f3(pick(CC, "C3", m, D=250.0)["AP@0.7"]), f3(pick(EC, "C0", m, D=150.0)["AP@0.7"]), f3(pick(EC, "C3", m, D=150.0)["AP@0.7"])]
    if m == "ours": cells = [r"\textbf{" + c + "}" for c in cells]
    out.append(n + " & " + " & ".join(cells) + r"\\")
out.append(r"\bottomrule\end{tabular}\end{table}")
# ablation table
out.append(r"\begin{table}[t]\caption{Ablations in C3. AP@0.7 is from the LiDAR-pair run (Experiment A). The last two columns give variant $-$ ours (points, 95\% scene CI) for Experiment A (LiDAR pair) and Experiment C (camera ego).}\label{tab:abl}")
out.append(r"\centering\scriptsize\setlength{\tabcolsep}{1.4pt}\begin{tabular}{lccc}\toprule Variant & AP@0.7 & $\Delta$ Exp.\ A & $\Delta$ Exp.\ C\\\midrule")
def flipci(t): return f"{-t['delta_AP'] * 100:+.2f} [{-t['scene_CI_hi'] * 100:+.2f}, {-t['scene_CI_lo'] * 100:+.2f}]"
def dirci(t): return f"{t['delta_AP'] * 100:+.2f} [{t['scene_CI_lo'] * 100:+.2f}, {t['scene_CI_hi'] * 100:+.2f}]"
ec_a1, ec_a2, ec_a3 = T_EC["A1: ours - no-quality, C3"], T_EC["A2: ours - no-need, C3"], T_EC["H2/A3: ours - no-admission, C3"]
ab = [("A1", "A1 no quality", flipci(T_MAIN["A1: ours - no-quality, C3"]), flipci(ec_a1)), ("A2", "A2 no need", flipci(T_MAIN["A2: ours - no-need, C3"]), flipci(ec_a2)),
      ("A3", "A3 no admission", flipci(T_MAIN["H2/A3: ours - no-admission, C3"]), flipci(ec_a3)),
      ("A7", "A7 oracle feedback", dirci(T_MAIN["A7: oracle same-frame feedback - ours, C3"]), "--"), ("R1", "R1 one retry", dirci(T_MAIN["R1: ours + one retry - ours, C3"]), "--")]
r = pick(M, "C3", "ours"); out.append(f"Ours (full) & {f3(r['AP@0.7'])} & -- & --" + r"\\")
for k, lab, d1, d2 in ab:
    r = pick(M, "C3", k); out.append(f"{lab} & {f3(r['AP@0.7'])} & {d1} & {d2}" + r"\\")
out.append(r"\bottomrule\end{tabular}\end{table}")
# precision recall
PR = pd.read_csv(os.path.join(DATA, "main", "precision_recall.csv"))
out.append(r"\begin{table}[t]\caption{Precision (P), recall (R) and F1 in C3 at a fixed score threshold of 0.5, chosen before the runs. Pooled over all frames and 3 link traces (83,148 ground-truth vehicles). No intervals are computed.}\label{tab:pr}")
out.append(r"\centering\scriptsize\setlength{\tabcolsep}{3pt}\begin{tabular}{lcccccc}\toprule & \multicolumn{3}{c}{IoU 0.5} & \multicolumn{3}{c}{IoU 0.7}\\\cmidrule(lr){2-4}\cmidrule(lr){5-7}")
out.append(r"Method & P & R & F1 & P & R & F1\\\midrule")
for m in ["B0", "B1", "B2", "B3", "ours", "U0"]:
    cells = []
    for iou in (0.5, 0.7):
        rr = PR[(PR.cond == "C3") & (PR.method == m) & (PR.IoU == iou)].iloc[0]
        cells += [f"{rr['precision@score>=0.5']:.3f}", f"{rr['recall@score>=0.5']:.3f}", f"{rr['F1@score>=0.5']:.3f}"]
    out.append((NAME[m] if m != "ours" else r"\textbf{Ours}") + " & " + " & ".join(cells) + r"\\")
out.append(r"\bottomrule\end{tabular}\end{table}")
# link sensitivity
S = pd.read_csv(os.path.join(DATA, "main", "sens_link_table_v2.csv"))
def sl(lab, m): r = S[(S.cond == lab) & (S.method == m)].iloc[0]; return r
out.append(r"\begin{table}[t]\caption{Link sensitivity around C3 (one factor changed at a time; reported as a sweep, not used to pick any value). AP@0.7 of the confidence-only baseline and of ours, and bytes sent by ours; the baseline always sends 14,566 B.}\label{tab:link}")
out.append(r"\centering\scriptsize\setlength{\tabcolsep}{4pt}\begin{tabular}{lccc}\toprule Link setting & Conf.-only & Ours & Ours bytes\\\midrule")
for lab, name in [("Sbase", "5 Mb/s, 10\\% loss, 20 ms (C3)"), ("Srate1", "1 Mb/s"), ("Srate20", "20 Mb/s"), ("Sloss0", "0\\% loss"), ("Sloss20", "20\\% loss"), ("Sq40", "40 ms queue")]:
    a_, b_ = sl(lab, "B3"), sl(lab, "ours")
    out.append(f"{name} & {f3(a_['AP@0.7'])} & {f3(b_['AP@0.7'])} & {by(b_['bytes_per_frame'])}" + r"\\")
assert abs(S[S.method == "B3"].bytes_per_frame.round().iloc[0] - 14566) < 1
out.append(r"\bottomrule\end{tabular}\end{table}")
blob = ("\n".join(out) + "\n").replace("-0.00", "0.00").replace("+0.00", "0.00")
blob = re.sub(r"(?<=[\s\[,])-(?=\d)", "$-$", blob)
for blk in re.findall(r"\\begin\{table\}.*?\\end\{table\}", blob, re.S):
    lab = re.search(r"\\label\{tab:(\w+)\}", blk).group(1)
    open(os.path.join(HERE, f"tab_{lab}.tex"), "w", encoding="utf-8").write(blk + "\n")
    print("wrote tab_%s.tex" % lab)
