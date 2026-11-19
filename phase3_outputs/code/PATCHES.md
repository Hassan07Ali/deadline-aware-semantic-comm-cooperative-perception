# Changes made to HEAL (for the Phase 3 changes log)
Purpose: run HEAL inference on an RTX 5050 (Blackwell, needs PyTorch >= 2.7 / CUDA 12.8). No model logic changed.

| File | Change | Why |
|---|---|---|
| bev_preprocessor.py, base_bev_backbone.py, att_bev_backbone.py, base_bev_backbone_resnet.py | `np.int` -> `int` | `np.int` was removed from NumPy |
| models/sub_modules/sparse_backbone_3d.py | spconv import wrapped in try/except | spconv is only needed by SECOND-style models, not PointPillars/LSS |
| tools/train_utils.py | `torch.load(..., weights_only=False)` | torch >= 2.6 changed the default; we load official HEAL checkpoints from the author's Hugging Face repo |
| utils/pcd_utils.py | pypcd import made optional | unused by our code path; fragile install |
| utils/setup.py | `distutils.core` -> `setuptools` | distutils removed in newer Python |
| data_utils/pre_processor/sp_voxel_preprocessor.py + new simple_voxelizer.py | if spconv is not installed, use our own numba voxel generator | the PointPillars config voxelizes through spconv's generator; the replacement returns the same arrays (checked against a brute-force reference on 120k random points: 0 mismatches) |
| (new) phase3_scripts/ | requirements_modern.txt, setup_wsl.sh, smoke_test.py | replaces HEAL's old pinned requirements |
Not changed: any model, loss, dataset or evaluation code.

## Crop-mask clamp (15 Nov 2026) — pyramid_fuse.py forward_collab
Camera agents are masked to the region their BEV grid covers. With a non-square evaluation range (x +-70.4, y +-40) and a camera grid of +-51.2 m,
`start_h = H//2 - crop_H//2` is negative (-12), which Python slicing treats as "from the end", so only 12 of 100 rows (8 % of the map) stayed unmasked.
Fix: clamp start/end to [0, H] / [0, W]. LiDAR-only experiments are unaffected (mask is applied to camera agents only). The original file is kept as pyramid_fuse.py.orig_before_crop_fix.
