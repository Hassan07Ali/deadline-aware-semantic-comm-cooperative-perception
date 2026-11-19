# Reference check (done 19 Nov 2026)

How each entry in `refs.bib` was checked, and what could not be checked. Nothing was added from memory without a lookup.

## Confirmed against Crossref (title, authors, journal, year, volume/issue/pages all match our Phase 2 list)
| Key | Paper | Venue | DOI |
|---|---|---|---|
| feng2025 | Feng et al., Semantic Communication for Edge Intelligence Enabled Autonomous Driving System | IEEE Network 39(2), 2025 | 10.1109/MNET.2024.3468328 |
| cao2025 | Cao et al., Task-Oriented Semantic Communication for Stereo-Vision 3D Object Detection | IEEE Trans. Commun. 73(9), 2025 | 10.1109/TCOMM.2025.3545687 |
| sheng2026 | Sheng et al., Semantic Communication for Cooperative Perception Using HARQ | IEEE TCCN 12, 2026 | 10.1109/TCCN.2025.3579540 |
| liu2025 | Liu et al., Semantic Communication Empowered Collaborative Perception in Constrained Networks | IEEE WCL 14(3), 2025 | 10.1109/LWC.2024.3520660 |
| zhang2026lane | Zhang et al., Multimodal Semantic Communications Empowered Lane-Change Planning | IEEE TVT 75(2), 2026 | 10.1109/TVT.2025.3599852 |
| yang2025 | Yang et al., Integrated Sensing, Computing and Semantic Communication for Vehicular Networks | IEEE TVT 74(11), 2025 | 10.1109/TVT.2025.3575243 |
| du2025 | Du et al., Task-Oriented Semantic Communication in LMM-Based Vehicle Networks | IEEE TMC 24(10), 2025 | 10.1109/TMC.2025.3564543 |
| hu2026prag | Hu et al., Pragmatic Communication in Multi-Agent Collaborative Perception | IEEE TPAMI 48(8), 2026 | 10.1109/TPAMI.2026.3680062 |
| zhang2025near | Zhang et al., Near-Sensor LiDAR and Visual Feature Extraction and Communication ... | IEEE IoT-J 12(17), 2025 | 10.1109/JIOT.2025.3583443 |
| li2025 | Li et al., Intelligent Cooperative Perception Technology ... V2V/V2I Semantic Communication | Electronics 14(24), 2025 | 10.3390/electronics14244969 |

## Confirmed against the arXiv record (title, authors, date match)
| Key | arXiv id | Venue as stated in the arXiv comment field |
|---|---|---|
| xu2022opv2v | 2109.07644 | ICRA 2022 |
| lu2024heal | 2401.13964 | ICLR 2024 (the code link in the comment is github.com/yifanlu0227/HEAL) |
| xu2023v2v4real | 2303.07601 | CVPR 2023 |
| philion2020 | 2008.05711 | ECCV 2020 |
| lang2019 | 1812.05784 | CVPR 2019 (v2 is the accepted version) |
| hu2022where2comm | 2209.12836 | NeurIPS 2022 |
| xu2025cosdh | 2503.03430 | CVPR 2025 |
| chen2026whisper | 2603.01708 | CVPR 2026 ("Accepted by CVPR26") |
| mukhopadhyay2026 | 2602.13287 | ICLR 2026 |
| zhu2026 | 2601.04542 | preprint, "submitted to the IEEE for possible publication" |
| wu2025fresh | 2502.07852 | preprint, no venue stated |
| wang2025jigsaw | 2511.17843 | preprint, no venue stated |

## Confirmed on the publisher page
| Key | Item | Check |
|---|---|---|
| itu2023 | ITU-R Recommendation M.2160-0, Nov 2023 | itu.int page title matches "Framework and overall objectives of the future development of IMT for 2030 and beyond" |

## What could NOT be fully verified (please check yourself)
1. **Venues of the arXiv items** come from the arXiv comment field, not from the conference proceedings. No page numbers are given for them for that reason.
2. **Three preprints** (zhu2026, wu2025fresh, wang2025jigsaw) are not peer reviewed as far as the arXiv pages show. The paper labels them "preprint".
3. **A wrong guess that was caught:** my first attempt at the HEAL arXiv id (2401.06635) turned out to be an unrelated maths paper. It was discarded. The correct HEAL paper was found by title search (2401.13964).
4. **Statements about what each related paper does** (Table I and Table VII in the paper) come from reading the abstracts and pages earlier in the project (Where2comm, CoSDH, WhisperNet, CooperTrim, Fresh2comm, Zhu et al., JigsawComm) and from our own Phase 2 reading (the S2CP and PragComm numbers). The abstract numbers quoted in Table VII (more than 100,000x for Where2comm, 80.28% and 72.52% for CooperTrim) were re-checked against the abstract text on 19 Nov 2026 (saved in `refs_abstracts.xml`). The "not stated" entries are limited to what the pages we read said. Spot-check a few before you submit.
5. Phase 2 references [11]-[13] were repository links; they are replaced here by the real papers (OPV2V, HEAL, V2V4Real). Phase 2 references [15] and [16] (journal quartile sources) are not cited in the paper.
