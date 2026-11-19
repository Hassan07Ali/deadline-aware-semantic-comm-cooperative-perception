# Deadline-aware semantic communication for cooperative perception

Course project for Computer Networks (FAST-NUCES, Department of Artificial Intelligence).
Authors: Hassan Ali (24I-0047) and Muhammad Saad (24I-0058). Instructor: Ms. Laraib Afzal.

**Paper title:** A Unified Task-Oriented Multimodal Semantic Communication Framework in 6G Networks: Deadline-Aware Camera--LiDAR Feature Sharing for Cooperative Perception

Two cars share bird's-eye-view (BEV) feature tiles over a link with a byte budget and a deadline, while their sensors may be degraded. This repository holds a training-free tile-selection and deadline-admission policy, an evaluation protocol that scores AP of the result ready at the deadline, and all logs behind the paper. The detector is the official HEAL_OPV2V checkpoint, used frozen. The link is emulated; nothing here measures real 6G or radio behaviour.

## Main result (OPV2V test split, 16 scenes, 2,170 frames)
Two LiDAR cars, degraded sensors and a bad link (5 Mb/s, 10% loss, 20 ms queue), deadline 100 ms, 16 KiB per frame, 10 link traces:
AP@0.7 is 0.796 for the policy against 0.792 for a confidence-only selector (+0.40 points, 95% scene-bootstrap interval [+0.23, +0.59]), with 35% fewer bytes sent. The quality term accounts for most of the gain; the receiver-need term and deadline admission do not raise accuracy. See the paper for all results, ablations and limits.

## Layout
| Path | Content |
|---|---|
| `paper_final/` | LaTeX source (`paper.tex`, `refs.bib`), compiled `paper.pdf`, figures, scripts that build every table and figure from the logs (`make_assets.py`), and the number and reference checkers |
| `phase3_outputs/code/` | feature extraction, replay engine (`replay2.py`), analysis, statistics, failure analysis; `ours_pkg/` is the selection, admission, link and timeline package; `PATCHES.md` lists the changes made to HEAL so it runs on an RTX 5050 |
| `phase3_outputs/data/` | result tables, bootstrap tests, timing measurements, manifests and fidelity-check logs |
| `phase3_outputs/figures/` | figures produced during Phase 3 |
| `PHASE3_REPORT.md`, `Phase3_Report_*.pdf` | Phase 3 report |
| `PHASE3_CHANGES_LOG.md` | every deviation from the Phase 2 plan, with the reason |

## Not included
The OPV2V dataset, the HEAL code and the HEAL_OPV2V checkpoint are not in this repository. Get them from their authors (OPV2V: arXiv 2109.07644; HEAL: arXiv 2401.13964, github.com/yifanlu0227/HEAL). `phase3_outputs/code/PATCHES.md` and `pyramid_fuse_PATCHED.py` describe our changes to HEAL, including a fix for a camera crop-mask bug that appears at non-square evaluation ranges.

## Reproducing
1. Run `extract_features.py` to cache each car's BEV features from the frozen checkpoint.
2. Run `replay2.py` to replay selection, byte budget, deadline admission, packet loss and fusion/detection per frame.
3. Run `analyze.py`, `paired.py`, `sens_table.py`, `failure_analysis.py` for tables, bootstrap tests and figures.
4. In `paper_final/`: `python make_assets.py`, then compile `paper.tex` (IEEEtran, for example with Tectonic or on Overleaf).

Experiments ran on an RTX 5050 Laptop GPU (8 GB) in WSL2 Ubuntu with Python 3.10, PyTorch 2.11 (CUDA 12.8) and NumPy 1.26.

## Limits
Frozen detector with no training, 16 test scenes, no validation split, emulated link and timing measured on one laptop GPU, hand-made quality proxies. Detection AP is a proxy for perception quality, not evidence about driving safety.
