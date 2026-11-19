"""Checks that every citation exists in refs.bib, every bib entry is cited, every label is referenced,
and every figure file used exists. Run: python check_refs_figs.py"""
import re, os, glob
H = os.path.dirname(os.path.abspath(__file__))
tex = open(os.path.join(H, "paper.tex"), encoding="utf-8").read()
alltex = tex + "".join(open(f, encoding="utf-8").read() for f in glob.glob(os.path.join(H, "tab_*.tex")))
cites = set()
for m in re.finditer(r"\\cite\{([^}]*)\}", alltex):
    cites |= {k.strip() for k in m.group(1).split(",")}
bib = set(re.findall(r"@\w+\{(\w+),", open(os.path.join(H, "refs.bib"), encoding="utf-8").read()))
print("cited but missing in bib :", sorted(cites - bib))
print("in bib but never cited   :", sorted(bib - cites))
labels = set(re.findall(r"\\label\{([^}]*)\}", alltex))
refs = set(m.group(1) for m in re.finditer(r"\\(?:ref|eqref)\{([^}]*)\}", alltex))
print("labels never referenced  :", sorted(labels - refs))
print("refs without a label     :", sorted(refs - labels))
figs = re.findall(r"includegraphics\[[^\]]*\]\{([^}]*)\}", tex)
for f in figs:
    print("figure", f, "exists:", os.path.exists(os.path.join(H, "figs", f)))
print("tables:", sorted(l for l in labels if l.startswith("tab:")))
print("figures:", sorted(l for l in labels if l.startswith("fig:")))
print("TODO markers:", len(re.findall(r"\\TODO\{", tex)))
for m in re.finditer(r"\\TODO\{([^}]*)\}", tex):
    print("   TODO:", m.group(1))
