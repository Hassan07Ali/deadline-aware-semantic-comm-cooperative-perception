"""Create opencood/tools/infer_subset.py = HEAL's inference.py + --scenes / --stride / timing (no other change)."""
s = open("opencood/tools/inference.py").read()

s = s.replace(
    "    parser.add_argument('--note', default=\"\", type=str, help=\"any other thing?\")",
    "    parser.add_argument('--note', default=\"\", type=str, help=\"any other thing?\")\n"
    "    parser.add_argument('--scenes', default=\"\", type=str, help='comma list of scene indices (0-based, sorted order); empty = all')\n"
    "    parser.add_argument('--stride', default=1, type=int, help='take every Nth frame inside each chosen scene')\n"
    "    parser.add_argument('--max_frames', default=0, type=int, help='stop after N frames (0 = no limit)')",
)

old = "    data_loader = DataLoader(opencood_dataset,"
new = (
    "    lens = [0] + list(opencood_dataset.len_record)\n"
    "    chosen = [int(x) for x in opt.scenes.split(',')] if opt.scenes else list(range(len(lens) - 1))\n"
    "    idxs = []\n"
    "    for sc in chosen:\n"
    "        idxs += list(range(lens[sc], lens[sc + 1], opt.stride))\n"
    "    if opt.max_frames:\n"
    "        idxs = idxs[:opt.max_frames]\n"
    "    print('SUBSET: scenes', chosen, 'stride', opt.stride, 'frames', len(idxs))\n"
    "    opencood_dataset_subset = Subset(opencood_dataset, idxs)\n"
    "    opencood_dataset_subset.collate_batch_test = opencood_dataset.collate_batch_test\n"
    "    data_loader = DataLoader(opencood_dataset_subset,"
)
assert old in s
s = s.replace(old, new, 1)

# timing + peak memory
s = s.replace("    for i, batch_data in enumerate(data_loader):",
              "    import time as _t\n    _t0 = _t.time(); _ft = []\n    for i, batch_data in enumerate(data_loader):\n        _fs = _t.time()", 1)
s = s.replace("        torch.cuda.empty_cache()\n",
              "        torch.cuda.synchronize(); _ft.append(_t.time() - _fs)\n        torch.cuda.empty_cache()\n", 1)
s = s.replace("    _, ap50, ap70 = eval_utils.eval_final_results(result_stat,",
              "    print('TIMING sec/frame (incl. data loading) mean %.3f median %.3f n=%d; peak GPU MB %.0f' % "
              "(float(np.mean(_ft)), float(np.median(_ft)), len(_ft), torch.cuda.max_memory_allocated() / 2**20))\n"
              "    _, ap50, ap70 = eval_utils.eval_final_results(result_stat,", 1)
open("opencood/tools/infer_subset.py", "w").write(s)
print("ok")
