# paper_final

IEEE conference-format paper (IEEEtran), compiled with Tectonic 0.17.0.

| File | What it is |
|---|---|
| `paper.tex`, `refs.bib` | LaTeX source and bibliography |
| `paper.pdf` | compiled paper (9 pages) |
| `figs/` | figures (PDF for the paper, 300-dpi PNG copies) |
| `tab_*.tex` | result tables, generated from the saved logs by `make_assets.py` (never typed by hand) |
| `make_assets.py` | builds every figure and table from `../phase3_outputs/data/` |
| `check_numbers.py` | checks every number in the paper against the logs (see its docstring) |
| `check_refs_figs.py` | checks citations, labels, figures, lists TODO markers |
| `paired_extra_log.txt`, `run_paired_extra.sh` | analysis-only bootstrap re-run for the stress test and the camera-ego budget sweep (the earlier outputs were only in the report; no model or replay was re-run) |
| `reference_check.md` | what was verified for each reference and what was not |
| `refs_abstracts.xml` | arXiv abstracts used for the numbers quoted in Table VII |
| `software_versions.txt` | versions used for the experiments |

Rebuild: `python make_assets.py`, then `tectonic paper.tex` (Overleaf also works: upload `paper.tex`, `refs.bib`, `tab_*.tex`, `figs/`).
