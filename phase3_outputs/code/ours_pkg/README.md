# ours/ - Phase 3 method modules (numpy only)
Run the unit tests from this folder:  python -m unittest discover -s tests -v
serializer.py   token/packet byte accounting (Phase 2 numbers: 4 tokens = 1,120 B; 56 tokens = 15,680 B; +584 feedback = 16,264 B)
selector.py     tile score s = a r p (1+lam_c c)[eps+(1-eps)(1-e)]exp(-age/tau), critical region, B2/B3 baselines
admission.py    deadline rule t_ready + Q + 8 L(K)/R + T_reserve <= D, EMA rate estimator
link_replay.py  C0-C3 links, common-random-number packet loss, lost packets still cost bytes and time
timeline.py     cooperative vs ego-only vs empty output at the deadline
Still to write (need the real features): quality estimate, feature cache, fusion/detection, AP-at-deadline evaluator, runner.
