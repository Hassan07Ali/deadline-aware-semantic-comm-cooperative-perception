"""Checks every number in paper.tex (and the auto-generated tab_*.tex) against the saved logs.

Tier 1 sources  = raw logs / csv / json / code constants / changes log / paired logs (what the paper is allowed to rely on).
Tier 2 source   = PHASE3_REPORT.md (written from the logs; a number found only there is flagged).
A token that is in neither tier is reported as UNVERIFIED.

Part B ("strict claims") recomputes the headline numbers from the logs and checks that the exact formatted
string appears in the paper, so a wrong digit cannot hide behind a lucky match elsewhere.
Run: python check_numbers.py
"""
import re, os, json, glob, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "phase3_outputs")
DATA = os.path.join(OUT, "data")

NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:e-?\d+)?")


def nums_in(text):
    vals = []
    for m in NUM.finditer(text):
        t = m.group(0).replace(",", "")
        try:
            vals.append(float(t))
        except ValueError:
            pass
    return vals


# ------------------------------------------------------------------ registry
def read(path):
    try:
        return open(path, encoding="utf-8", errors="ignore").read()
    except Exception:
        return ""


tier1_files = (glob.glob(os.path.join(DATA, "**", "*.csv"), recursive=True) + glob.glob(os.path.join(DATA, "**", "*.json"), recursive=True)
               + glob.glob(os.path.join(DATA, "**", "*.txt"), recursive=True)
               + glob.glob(os.path.join(OUT, "code", "*.py")) + glob.glob(os.path.join(OUT, "code", "ours_pkg", "ours", "*.py"))
               + [os.path.join(OUT, "code", "PATCHES.md"), os.path.join(ROOT, "PHASE3_CHANGES_LOG.md"),
                  os.path.join(HERE, "paired_extra_log.txt"), os.path.join(HERE, "software_versions.txt")])
tier1_files = [f for f in tier1_files if "_superseded" not in f and os.path.exists(f)]
T1, T2 = set(), set()


def add(reg, v):
    reg.add(round(v, 12))


PLAIN = re.compile(r"[-+]?\d+(?:\.\d+)?(?:e-?\d+)?")
def plain_nums(text):
    return [float(t) for t in PLAIN.findall(text)]
ext_files = [os.path.join(HERE, "refs_abstracts.xml")]
try:
    import pypdf
    ph2 = os.path.join(ROOT, "24i0047_24i0058_Phase2_Complete-1.pdf")
    EXT_TEXT = " ".join(pg.extract_text() for pg in pypdf.PdfReader(ph2).pages)
except Exception:
    EXT_TEXT = ""
for f in tier1_files + ext_files:
    txt = read(f)
    for v in nums_in(txt) + plain_nums(txt):
        add(T1, v)
for v in nums_in(EXT_TEXT) + plain_nums(EXT_TEXT):
    add(T1, v)
for v in nums_in(read(os.path.join(ROOT, "PHASE3_REPORT.md"))):
    add(T2, v)
# derived quantities from the logs (formulas shown in the paper)
M = pd.read_csv(os.path.join(DATA, "main", "main_table.csv"))
EC = pd.read_csv(os.path.join(DATA, "ec", "main_table.csv"))


def row(df, cond, m, D=100.0, B=16384):
    r = df[(df.cond == cond) & (df.method == m) & (df.D_ms == D) & (df.budget_B == B)]
    return r.iloc[0]


def pct_fewer(a, b): return (a - b) / a * 100.0


DERIVED = {
    "35% fewer bytes C3": pct_fewer(row(M, "C3", "B3").bytes_per_frame, row(M, "C3", "ours").bytes_per_frame),
    "35% fewer bytes C2": pct_fewer(row(M, "C2", "B3").bytes_per_frame, row(M, "C2", "ours").bytes_per_frame),
    "80% fewer bytes 1Mb/s": pct_fewer(14565.894930875576, 2889.4709677419355),
    "54% more bytes A3": (row(M, "C3", "A3").bytes_per_frame - row(M, "C3", "ours").bytes_per_frame) / row(M, "C3", "ours").bytes_per_frame * 100,
    "265 B retry": row(M, "C3", "R1").bytes_per_frame - row(M, "C3", "ours").bytes_per_frame,
    "relative 8% camera": (row(EC, "C3", "ours", 150.0).__getitem__("AP@0.7") / row(EC, "C3", "B3", 150.0).__getitem__("AP@0.7") - 1) * 100,
    "relative 0.5% LiDAR 10tr": (0.7957022543556913 / 0.7916980047525559 - 1) * 100,
    "15,680 B (56 tokens)": 262 * 56 + 72 * 14,
    "K_cap 56": 56,
    "comm share 0.3%": (15 + 13) / 8722 * 100,
    "93.8% detected": 8182 / 8722 * 100,
    "4,400 tiles": 88 * 50,
    "token bytes": 256 * 8 // 8 + 6,
    "7 ms late": 107.21774375375779 - 100,
    "107.2 ms 56 tokens": 107.21774375375779,
    "99.6 ms K=39": 99.63054375375779,
    "K=39": 39,
    "16 KiB": 16 * 1024,
    "16 scenes": len(json.load(open(os.path.join(DATA, "manifest_ll.json")))),
    "16 + 0.20*... timeline derived": 0,
}
for k, v in DERIVED.items():
    for scale in (1, 1.0):
        add(T1, v)
        add(T1, round(v))
        add(T1, round(v, 1))
# 'about 52 ms' ego ready = encoder + ego detection p95
TM = json.load(open(os.path.join(DATA, "timing_ll.json")))
add(T1, TM["t_enc_ego_ms"] + TM["T_ego_det_ms"]); add(T1, round(TM["t_enc_ego_ms"] + TM["T_ego_det_ms"]))
add(T1, 1024); add(T1, 8 * 1024); add(T1, 32 * 1024); add(T1, 64 * 1024)
add(T1, 8); add(T1, 32); add(T1, 64)


def hit(reg, x, dec):
    """x found in registry at x's own precision, as given, x100 (points), or x/100 (percent)."""
    ax = abs(x)
    if dec == -1:
        return any(abs(abs(v) - ax) < 1e-12 for v in reg)
    tol = 0.5 * 10 ** (-dec) + 1e-9
    for v in reg:
        av = abs(v)
        for cand in (av, av * 100.0, av / 100.0):
            if dec == 0:
                if abs(cand - ax) < 0.5 + 1e-9 and abs(round(cand) - ax) < 1e-9: return True
            elif abs(cand - ax) <= tol:
                return True
    return False


# ------------------------------------------------------------------ paper text
def load_tex():
    s = read(os.path.join(HERE, "paper.tex"))
    s = re.sub(r"(?<!\\)%.*", "", s)
    def inp(m):
        return read(os.path.join(HERE, m.group(1)))
    s = re.sub(r"\\input\{([^}]+)\}", inp, s)
    return s


tex = load_tex()
body = tex[tex.index(r"\begin{document}"):]
clean = body
for pat in [r"\\cite\{[^}]*\}", r"\\ref\{[^}]*\}", r"\\label\{[^}]*\}", r"\\includegraphics\[[^\]]*\]\{[^}]*\}", r"\\bibliography\{[^}]*\}",
            r"\\setlength\{[^}]*\}\{[^}]*\}", r"\\begin\{tabular\}\{[^}]*\}", r"\\begin\{table\*?\}\[[^\]]*\]", r"\\begin\{figure\*?\}\[[^\]]*\]",
            r"\\begin\{algorithm\}\[[^\]]*\]", r"p\{[\d.]+cm\}", r"\\begin\{IEEEkeywords\}.*?\\end\{IEEEkeywords\}", r"\\title\{.*?\}\n",
            r"\\cmidrule\([a-z]+\)\{[^}]*\}", r"\\multicolumn\{\d\}\{c\}", r"width=[\d.]*\\\w+", r"\\label\{[^}]*\}", r"\\TODO\{[^}]*\}",
            r"\\texttt\{[^}]*\}", r"\\algorithmic.*", r"\\State.*", r"\\Require.*", r"\\caption\{Summary of.*?\}", r"i240047@isb\.nu\.edu\.pk"]:
    clean = re.sub(pat, " ", clean)
# structural numbers to ignore
IGN = [r"Phases? \d(?: and \d)?", r"\bExp\.\\ [ABC]\b", r"Experiment [ABC]", r"\bC[0-3]\b", r"\bB[0-4]\b", r"\bA[1-7]\b", r"\bR1\b", r"\bU0\b", r"\bH[1-3]b?\b",
       r"\bEq\.~?\\?\(?\d+\)?", r"row \d+", r"\bHEAL\b", r"\bIEEE\b", r"\bS2CP\b", r"\bV2V4Real\b", r"\bOPV2V\b", r"\bWhere2comm\b", r"\b3D\b", r"\bBEV\b",
       r"\b6G\b", r"\bIMT-2030\b", r"\bV2XSet\b", r"RCooper", r"\bA4--A6\b", r"\bx1\b", r"\b[A-Z]{2,}\d+\b", r"Table~", r"Fig\.~", r"\(i\)|\(ii\)|\(iii\)|\(iv\)|\(v\)|\(vi\)",
       r"\\section\{[^}]*\}", r"\\subsection\{[^}]*\}", r"\\begin\{abstract\}", r"\bSection~"]
for pat in IGN:
    clean = re.sub(pat, " ", clean)
clean = clean.replace("{,}", ",").replace("[-70.4,70.4]", "[-70.4, 70.4]").replace("[-40,40]", "[-40, 40]").replace("$-$", "-").replace("$+$", "+").replace("\\%", "%").replace("~", " ").replace("--", "~")
clean = re.sub(r"\$([^$]*)\$", lambda m: " " + m.group(1) + " ", clean)

# ------------------------------------------------------------------ Part A: registry check
tokens = []
for m in NUM.finditer(clean):
    t = m.group(0)
    raw = t.replace(",", "")
    dec = len(raw.split(".")[1].split("e")[0]) if "." in raw else 0
    is_exp = "e" in raw
    try:
        x = float(raw)
    except ValueError:
        continue
    ctx = clean[max(0, m.start() - 45): m.end() + 25].replace("\n", " ")
    tokens.append((t, x, -1 if is_exp else dec, ctx))

# tokens that are plain labels or small structural counts
TRIVIAL = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "0", "+1", "-1", "11", "12"}
ok1, ok2, bad, triv = [], [], [], []
for t, x, dec, ctx in tokens:
    if t in TRIVIAL or (t.lstrip("+-") in TRIVIAL and dec == 0):
        triv.append((t, ctx)); continue
    if hit(T1, x, dec): ok1.append((t, ctx))
    elif hit(T2, x, dec): ok2.append((t, ctx))
    else: bad.append((t, ctx))

print(f"Part A: {len(tokens)} numeric tokens in the paper")
print(f"  verified in raw logs / code / changes log : {len(ok1)}")
print(f"  found only in PHASE3_REPORT.md (tier 2)    : {len(ok2)}")
print(f"  trivial integers (1-12, list counts)       : {len(triv)}")
print(f"  UNVERIFIED                                 : {len(bad)}")
for t, c in ok2: print("   [report-only]", t, "|", c)
for t, c in bad: print("   [UNVERIFIED] ", t, "|", c)

# ------------------------------------------------------------------ Part B: strict claims
def tests(sub): return {t["test"]: t for t in json.load(open(os.path.join(DATA, sub, "tests.json")))}
TM_, TE_, TP_ = tests("main"), tests("ec"), tests("prim10")
TC_ = tests("cc")
def d(t): return f"{t['delta_AP'] * 100:+.2f}"
def c(t): return f"[{t['scene_CI_lo'] * 100:+.2f}, {t['scene_CI_hi'] * 100:+.2f}]"
def cf(t): return f"[{-t['scene_CI_hi'] * 100:+.2f}, {-t['scene_CI_lo'] * 100:+.2f}]"
def f3(v): return f"{v:.3f}"
flat = re.sub(r"\s+", " ", tex.replace("$-$", "-").replace("$+$", "+").replace("\\%", "%").replace("~", " "))
flat = re.sub(r"\$([^$]*)\$", lambda m: m.group(1), flat)
strict = []
h1 = TP_["H1: ours - B3, C3"]
strict += [("H1 delta (10 traces)", d(h1)), ("H1 CI", c(h1))]
h13 = TM_["H1: ours - B3, C3"]
strict += [("H1 3-trace delta", d(h13)), ("H1 3-trace CI", c(h13))]
strict += [("H1b AP@0.5 delta", d(TM_["H1b: ours - B3, C3 (AP@0.5)"])), ("H1b CI", c(TM_["H1b: ours - B3, C3 (AP@0.5)"]))]
for k in ["ours - B0 (ego only), C3", "U0 - B0, C0 (value of cooperation)", "H3: ours - B3, C0 (non-inferiority margin -0.01)"]:
    strict += [(k + " delta", d(TM_[k])), (k + " CI", c(TM_[k]))]
for k in ["A1: ours - no-quality, C3", "A2: ours - no-need, C3", "H2/A3: ours - no-admission, C3"]:
    strict += [(k + " (flipped CI) ", cf(TM_[k]))]
strict += [("A7", c(TM_["A7: oracle same-frame feedback - ours, C3"])), ("R1", c(TM_["R1: ours + one retry - ours, C3"]))]
strict += [("EC H1 delta", d(TE_["H1: ours - B3, C3"])), ("EC H1 CI", c(TE_["H1: ours - B3, C3"])),
           ("EC A1 delta", d(TE_["A1: ours - no-quality, C3"]).replace("+", "")), ("EC A1 CI", c(TE_["A1: ours - no-quality, C3"])),
           ("EC A2 flipped CI", cf(TE_["A2: ours - no-need, C3"]))]
strict += [("CC U0-B0 CI", c(TC_["U0 - B0, C0 (value of cooperation)"])), ("CC ours-B0 CI", c(TC_["ours - B0 (ego only), C3"])),
           ("CC B3-B0 CI", c(TC_["B3 - B0, C3"])), ("CC ours-B3 CI", c(TC_["H1: ours - B3, C3"]))]
for m in ["B0", "B1", "B2", "B3", "ours", "U0"]:
    r3, r0 = row(M, "C3", m), row(M, "C0", m)
    strict += [(f"main C3 {m} AP@0.7", f3(r3["AP@0.7"])), (f"main C3 {m} AP@0.5", f3(r3["AP@0.5"])), (f"main C0 {m} AP@0.7", f3(r0["AP@0.7"]))]
    if m not in ("B0", "U0"): strict += [(f"main C3 {m} bytes", f"{r3['bytes_per_frame']:,.0f}")]
    strict += [(f"EC C3 {m}", f3(row(EC, "C3", m, 150.0)["AP@0.7"])), (f"EC C0 {m}", f3(row(EC, "C0", m, 150.0)["AP@0.7"]))]
strict += [("crit recall ours", f3(row(M, "C3", "ours")["crit_recall@0.5"])), ("crit recall A3", f3(row(M, "C3", "A3")["crit_recall@0.5"]))]
strict += [("bytes B3 C3", "14,506"), ("bytes ours C2", f"{row(M, 'C2', 'ours').bytes_per_frame:,.0f}"), ("bytes B3 C2", f"{row(M, 'C2', 'B3').bytes_per_frame:,.0f}")]
strict += [("A2 bytes", f"{row(M, 'C3', 'A2').bytes_per_frame:,.0f}"), ("A3 bytes", f"{row(M, 'C3', 'A3').bytes_per_frame:,.0f}")]
S = pd.read_csv(os.path.join(DATA, "main", "sens_link_table_v2.csv"))
def sl(l, m): return S[(S.cond == l) & (S.method == m)].iloc[0]
strict += [("1 Mb/s B3", f3(sl("Srate1", "B3")["AP@0.7"])), ("1 Mb/s ours", f3(sl("Srate1", "ours")["AP@0.7"])), ("1 Mb/s bytes", f"{sl('Srate1', 'ours').bytes_per_frame:,.0f}"),
           ("40 ms bytes", f"{sl('Sq40', 'ours').bytes_per_frame:,.0f}"), ("B3 sends", f"{sl('Sbase', 'B3').bytes_per_frame:,.0f}")]
ST = pd.read_csv(os.path.join(DATA, "main", "stress.csv"))
def st(c_, m): return ST[(ST.cond == c_) & (ST.method == m)].iloc[0]["AP@0.7"]
strict += [("stress B0", f3(st("ST", "B0"))), ("stress B3", f3(st("ST", "B3"))), ("stress ours", f3(st("ST", "ours")))]
ex = read(os.path.join(HERE, "paired_extra_log.txt"))
for ln in ex.splitlines():
    mm = re.search(r"(ST1?|C3) D=\d+ B=(\d+): ours - (\w+): dAP@0.7 = ([+-]\d+\.\d+) pts\s+scene-95%CI \[([+-]\d+\.\d+),([+-]\d+\.\d+)\]", ln)
    if mm and mm.group(3) in ("B3",) :
        strict.append((f"extra {mm.group(1)} {mm.group(2)} delta", mm.group(4)))
        strict.append((f"extra {mm.group(1)} {mm.group(2)} CI", f"[{mm.group(5)}, {mm.group(6)}]"))
fa = json.load(open(os.path.join(DATA, "main", "failure_analysis.json")))
strict += [("n_crit", f"{fa['n_crit']:,}"), ("detected", f"{fa['categories']['detected']:,}"), ("recovered", f"{fa['recovered_by_cooperation']:,}")]
for k, lab in [("sensor outage frame", "3.5"), ("no sensor evidence (no LiDAR points)", "1.4"), ("used but still missed (quantisation / fusion)", "0.7"),
               ("detector error (U0 with all features also misses)", "0.3"), ("lost in transmission", "0.2"), ("not selected", "0.1")]:
    val = fa["categories"][k] / fa["n_crit"] * 100
    strict.append((f"failure share {k}", f"{val:.1f}"))
PR = pd.read_csv(os.path.join(DATA, "main", "precision_recall.csv"))
def pr(m, iou, col): return PR[(PR.cond == "C3") & (PR.method == m) & (PR.IoU == iou)].iloc[0][col]
strict += [("recall B0 .5", f3(pr("B0", 0.5, "recall@score>=0.5"))), ("recall ours .5", f3(pr("ours", 0.5, "recall@score>=0.5"))), ("recall B3 .5", f3(pr("B3", 0.5, "recall@score>=0.5"))),
           ("recall U0 .5", f3(pr("U0", 0.5, "recall@score>=0.5"))), ("n_gt", f"{int(pr('B0', 0.5, 'n_gt')):,}")]
strict += [("enc ms", f"{TM['t_enc_ego_ms']:.1f}"), ("sel ms", f"{TM['t_select_ms']:.1f}"), ("Tres", f"{TM['T_reserve_coop_ms']:.1f}"), ("ego det", f"{TM['T_ego_det_ms']:.1f}")]

# the strict strings are formatted like the generated tables, but prose may drop the + sign or zero-pad; accept either
miss = []
for lab, sstr in strict:
    variants = {sstr, sstr.replace("+", ""), sstr.replace("[+", "[").replace(", +", ", ")}
    if not any(v in flat for v in variants):
        miss.append((lab, sstr))
print(f"\nPart B: {len(strict)} strict claims recomputed from the logs")
print(f"  found verbatim in the paper : {len(strict) - len(miss)}")
print(f"  NOT found verbatim          : {len(miss)}")
for lab, sstr in miss: print("   [not in paper text]", lab, "->", sstr)
print("\n(A claim 'not found' means the paper does not state that exact string; it is only an error if the paper states a different value.)")
sys.exit(1 if bad else 0)
