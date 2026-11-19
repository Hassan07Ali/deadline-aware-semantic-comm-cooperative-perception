# Phase 3 — Changes Log (deviations from the Phase 2 methodology)

Living document. Every change is dated, says WHY, and says what it does to the claims.
Rule: nothing here is a result. Numbers appear only after they come from a saved log.

## A. Things we do differently from Phase 2

| # | Phase 2 said | What we actually do | Why | Effect on claims |
|---|---|---|---|---|
| 1 | Train encoders, adapters, quality head, fusion head | **No training.** Frozen official HEAL_OPV2V checkpoint (`net_epoch1.pth`) | Laptop RTX 5050 8 GB, 2-day deadline | Say "frozen pretrained detector", never "we trained" |
| 2 | Learned quality head | **Hand-made quality proxy** per camera: r = min(1, brightness/0.40) x min(1, sqrt(Laplacian-variance/0.004)); constants fixed before any result was seen | no training | "proxy quality estimate" |
| 3 | Learned masked fusion head | HEAL's own pretrained pyramid fusion; missing tiles are zeros | no training | fusion is not tuned to dropped tiles; may hurt every method equally |
| 4 | Train / validation / test | **Only the official test split** (16 scenes). Nothing tuned. All settings = Phase 2 defaults | validation split not downloaded | sweeps are reported as sweeps, never used to choose a value |
| 5 | Camera + LiDAR on each car | **One sensor per car (HEAL design).** Option A: ego = LiDAR (m1), collaborator = camera (m2), for every method | pretrained checkpoint design | collaborator sensor is a camera only |
| 6 | Sensor assignment | The official assignment file gives LiDAR+camera pairs in only 7 of 16 test scenes. We **override it** (`ours_pair_assignment.json`): one fixed pair per scene, all other cars excluded (label m9). The encoders accept any car's raw data | keep all 16 scenes under option A | the checkpoint was trained with the official assignment, so cross-assignment is a mild distribution shift |
| 7 | Collaborator = nearest car per frame (handoff 11.2) | **One fixed pair per scene**, chosen before any detection result: the two cars with the smallest mean distance (uses poses only). Scene 2021_08_18_19_48_05 has only 2 cars and they are >70 m apart in 119 of 178 frames -> those frames are ego-only for every method | the dataset loader fixes the car list per scene | same pair for every method |
| 8 | "Seeds" = training seeds | Seeds only change link-loss draws (traces) and outage schedules | no training | say so |
| 9 | B1 uniform BEV compression, B4 PragComm | **Cut** (time). Baselines: B0 ego-only, B2 random tiles, B3 confidence-only, U0 unconstrained reference | 2-day deadline | disclose |
| 10 | A4 no freshness, A5 single modality, A6 no corruption training | A4 is a no-op in our setup (all tiles of one frame have the same age, so ranking cannot change). A5 N/A (one sensor per car). A6 N/A (no training) | structure of the experiment | report honestly |
| 11 | Latency from the full uncached pipeline on a lab GPU | Latencies **measured on the RTX 5050 laptop** (see `timing.json`) and used as constants in an emulated timeline | no lab GPU | "emulated system", report device |
| 12 | T_reserve = validation p95 | T_reserve = p95 measured on the cached dev frames | no validation split | disclose |
| 13 | Depth supervision in the camera encoder | **Disabled** (`depth_supervision: false`) | it is a training loss; depth ground-truth images are not in the download | none for inference |
| 14 | Original voxel pooling | **Equivalent scatter-add re-implementation** (`fast_voxel_pooling`). Verified: relative difference 4e-4 vs the original (summation order). Camera encoder 223 ms -> see timing.json | the original cumsum trick cost 179 ms on this GPU | features are not bit-identical to the stock run |
| 15 | Real feature shape assumed 64 ch, 88 x 50 | **Confirmed**: 64 x 100 x 176 cells at 0.8 m (range x +-70.4 m, y +-40 m) = 88 x 50 tiles of 2 x 2 cells = 4,400 tiles | checked on real tensors | none |
| 16 | Detection range | evaluation range x +-70.4 m, y +-40 m, as Phase 2 preprocessing | Phase 2 | official HEAL numbers use +-102.4 m, so they are not directly comparable |

## B. Software patches needed to run HEAL on an RTX 5050 (see also `PATCHES.md`)
torch 2.11+cu128 (Blackwell sm_120); numpy 1.26; `np.int` -> `int`; `spconv` unavailable -> own numba voxelizer (0 mismatches in 53,241 voxels);
`torch.load(weights_only=False)` for the official checkpoint; `pypcd` optional; `libusb` for open3d.

## C. Problems met and fixes (for the demo)
1. Camera encoder 223 ms per frame -> profiled: network 29 ms, voxel pooling 179 ms (cumsum trick); replaced by an equivalent scatter-add.
2. `depth` input files missing -> removed `depth` from input sources and disabled depth supervision (training-only).
3. Official sensor assignment gives few LiDAR+camera pairs -> own assignment file (change 6).
4. Background job died when the WSL session closed -> run long jobs as a background process owned by the desktop session.

## D. Checks done
- 22/22 unit tests of `ours_pkg` pass in WSL.
- Our encoder chain reproduces HEAL's stock classification output (max difference 0 on the probe frames; ~1e-3 with the faster pooling).
- Pilot, stock HEAL model on 20 frames of one scene: AP@0.3/0.5/0.7 = 0.82/0.79/0.71 (sanity only; official full-test numbers ~0.85/0.83/0.73).
- Geometry verified on real frames: BEV row = y, column = x, origin (-70.4, -40); `pairwise_t_matrix[1,0]` = collaborator -> ego (0.00 m error).

## E. Decisions made after looking at the measurements (14-15 Nov 2026)

**E1. [CORRECTED 15 Nov 2026] The "camera collaborator is a null result" claim was WRONG; it was caused by a masking bug.**
First diagnostic (camera collaborator, all 2,170 frames): sending everything gave exactly the same AP as ego-only (C0 0.783 vs 0.783; C3 0.674 vs 0.674). We concluded that frozen HEAL fusion ignores a camera collaborator and moved the main experiment to a LiDAR pair.
Later, a camera-ego experiment gave AP = 0.000 for a lone camera ego car. Tracing it showed the cause: HEAL's `pyramid_fuse.forward_collab` masks camera agents with `start_h = H//2 - crop_H//2`; at our non-square evaluation range
(x +-70.4 m, y +-40 m) with a +-51.2 m camera grid this is **-12**, which Python slicing treats as "12 from the end", so only 12 of 100 rows (8 % of the map) stayed unmasked. The camera agent was effectively removed.
**Fix:** clamp the indices to [0, H] / [0, W] (`PATCHES.md`; original file kept as `pyramid_fuse.py.orig_before_crop_fix`). LiDAR-only runs are unaffected (the mask applies to camera agents only), so the main LiDAR+LiDAR results stand unchanged.
All camera runs made before the fix are invalid and were moved to `INVALID_cropbug/` (never deleted). With the fix, a camera collaborator helps a LiDAR ego car on clean data (diagnostic, every 3rd frame: ego-only 0.784, everything sent 0.848, confidence-only 0.857 AP@0.7)
and adds nothing when degraded (0.675 / 0.675 / 0.670), so the camera experiments are rerun in full: **Experiment 1 = LiDAR ego + camera collaborator**, **Experiment 3 = camera ego + LiDAR collaborator**.
The main LiDAR+LiDAR experiment is kept as Experiment 2 because it isolates the selection policy from modality effects.

**E2. Deadline grid chosen from measured latency only (before any accuracy was seen).**
RTX 5050 (fp32), medians/p95: LiDAR encoder 7.9 ms, camera encoder 68.7 ms (after the voxel-pooling fix), selection+quantisation 12.7 ms,
receiver decode+fusion+detection p95 77.6 ms (=T_reserve), ego-only detection p95 50.8 ms. The Phase 2 primary deadline of 100 ms is therefore infeasible
for a camera collaborator on this hardware (earliest cooperative result ~165 ms even on the good link). The LiDAR collaborator is feasible from ~145 ms.
**LiDAR-pair latencies (main experiment, `timing_ll.json`):** LiDAR encoder 9.7 ms, selection+quantisation 2.0 ms, T_reserve (receiver decode+fusion+detection, p95) 50.4 ms,
ego-only detection p95 42.7 ms (ego result ready at ~52 ms). Cooperative result on the bad link with 56 tokens: 9.7+2.0+20 queue+26 serialisation+50.4 = ~108 ms. So the **Phase 2 primary deadline D = 100 ms is kept**
and is feasible; admission genuinely binds in C3. Sweep: D = 50, 75, 100, 150 ms (at 50 ms even the ego-only result is late -> every method outputs an empty prediction).

**E3. Quality proxy for a LiDAR collaborator** r = min(1, ln(1+n)/ln(1+12)), n = LiDAR points in the 1.6 m tile; availability a = (n > 0). Constant 12 fixed before any result.
Camera proxy (Experiment 1): r = min(1, brightness/0.40) * min(1, sqrt(Laplacian variance/0.004)).

## F. Fidelity fixes and additions made on 15 Nov 2026 (after the first full run)

**F1. Feedback is now causal (Phase 2: "feedback from the previous frame at 10 Hz").** The first full run used the CURRENT frame's ego occupancy map as the receiver-need feedback,
which a real system could not have (optimistic for our method). `replay2.py` uses the PREVIOUS dataset frame's ego map, warped into the collaborator's current frame with the timestamped
poses (previous ego pose, current collaborator pose; never a future pose); lost or expired (>200 ms) feedback sets e = 0. The old same-frame version is kept as ablation **A7 (oracle feedback)** so the cost of causality is reported.
All numbers in the report come from `replay2.py`; the first-run numbers (`report_main` v1) are superseded and not used.

**F2. Empty predictions keep their objects in the AP denominator** (Phase 2). Bug found in the first run (affected only the D = 50 ms rows); fixed and rerun.

**F3. Added baselines / ablations from Phase 2:** B1 uniform compression of the whole BEV (block-averaged tokens at the finest resolution that fits the byte budget); R1 = ours + one retransmission of lost packets (pays identical byte/time costs, only if it can still arrive in time).

**F4. Phase 2 sensitivity grid** (one factor at a time around C3): link rate 1/5/10/20 Mb/s, loss 0/0.05/0.10/0.20, queue 0/10/20/40 ms, quantisation 4/8/16 bits, epsilon, lambda_c.
Reported as sweeps; no value is selected from them. tau cannot change the ranking in our setup (all tiles of one frame have the same age), so it is not swept. Sensitivity runs use every 2nd frame and 2 link traces (laptop time); the baseline point is rerun with the same sampling.

**F5. Held-out stress test (Phase 2 G3):** LiDAR retention 0.3 (instead of 0.5) on both cars and outage bursts on 30 % of frames; the operating point is still the Phase 2 default (nothing tuned).

**F6. Experiment 3 (heterogeneous ego):** ego = camera (m2), collaborator = LiDAR (m1), same fixed car pairs. The cooperative result needs max(last packet arrival, ego camera encoding) + T_reserve.
This is the realistic multimodal case: a weak camera-only ego car receives LiDAR tiles.

**F7. Statistics:** primary comparison uses 10 link traces (Phase 2) instead of 3; scene-level paired bootstrap (2,000 resamples, 16 scenes) plus a frame-block bootstrap (labelled), Holm correction.

**F8. Limits that remain (cannot be done on this laptop / in this time):** no training (frozen HEAL), no PragComm baseline, no real-data (V2V4Real) transfer, emulated latency, 16 test scenes, no validation split, quality proxies are hand-made (constants fixed before any result).

## G. Final run (15 Nov 2026, after the camera-mask fix)

**G1. What the final numbers are built from.** LiDAR pair: `main2/` (replay2, causal feedback, 3 traces) plus `prim10/` (10 traces, H1 only), `sweep2/` (budget, deadline, C1/C2), `sens/`, `stress/`, `fail/fail2_C3.pkl`.
Camera: `cc/` (LiDAR ego + camera collaborator, D = 250 ms), `ec/` (camera ego + LiDAR collaborator, D = 100/150/200 ms), `ecb/` (budget sweep for the camera-ego case). The old `sweep/`, `main/`, `fail_C3.pkl` (replay v1) are superseded and not used.

**G2. Extra run added:** default-setting rerun on the same every-2nd-frame sample (`sens_default`) so the quantisation / epsilon / lambda_c rows are compared like with like.
**G3. Extra run added:** camera-ego budget sweep (8/32/64 KiB, D = 150 ms) because the 16 KiB budget is clearly binding in that experiment (ours 0.332 vs U0 0.550).
**G4. Fidelity check added:** HEAL's own official inference on the 16 test scenes (official config, official assignment, +-102.4 m) and on our LiDAR pair at our range, to compare with the replay engine's U0 row.

**G5. Hypotheses, decided as pre-declared:** H1 supported but small (+0.40 [+0.23, +0.59], 10 traces); H2 not supported (recall 0.937 vs 0.938); H3 supported for clean non-inferiority; the 25 % byte saving holds on the lossy link (35 %) but not on a clean link.
**G6. Things that came out against our design (reported, not hidden):** the receiver-need term does not improve accuracy and is slightly harmful on clean data; deadline admission does not improve accuracy; random/uniform sharing is worse than ego-only with the frozen detector; a degraded camera collaborator gives no cooperative benefit.
