import unittest
import numpy as np
from ours import serializer as S, selector as SEL, admission as ADM, link_replay as LR, timeline as TL


class TestBytes(unittest.TestCase):
    def test_token_and_packet(self):
        self.assertEqual(S.token_bytes(8), 262)
        self.assertEqual(S.tokens_per_packet(8), 4)

    def test_phase2_unit_numbers(self):
        self.assertEqual(S.wire_cost(4), 1120)               # 4 tokens
        self.assertEqual(S.num_packets(56), 14)
        self.assertEqual(S.wire_cost(56), 15680)             # 56 tokens / 14 packets
        self.assertEqual(S.wire_cost(56) + S.FEEDBACK_BYTES, 16264)
        self.assertLess(16264, 16 * 1024)
        self.assertEqual(S.wire_cost(0), 0)

    def test_budget(self):
        self.assertEqual(S.max_tokens_in_budget(16 * 1024), 56)      # 57 tokens would exceed 16 KiB
        self.assertGreater(S.wire_cost(57) + S.FEEDBACK_BYTES, 16 * 1024)
        self.assertLess(S.max_tokens_in_budget(8 * 1024), 56)
        self.assertGreater(S.max_tokens_in_budget(32 * 1024), 56)

    def test_packet_sizes_sum(self):
        for k in (1, 3, 4, 5, 56, 57):
            self.assertEqual(sum(S.packet_sizes(k)), S.wire_cost(k))


class TestScore(unittest.TestCase):
    def base(self, **kw):
        a = dict(p=0.8, r=0.9, a=1, e=0.0, c=0, age_ms=0.0)
        a.update(kw)
        return float(SEL.score(**a))

    def test_critical_region_doubles(self):
        self.assertAlmostEqual(self.base(c=1) / self.base(c=0), 2.0)            # lambda_c = 1

    def test_ego_confidence_floor(self):
        self.assertAlmostEqual(self.base(e=1.0) / self.base(e=0.0), SEL.EPS)    # never fully excluded

    def test_age_decay(self):
        self.assertAlmostEqual(self.base(age_ms=100.0) / self.base(), np.exp(-1))

    def test_unavailable_is_zero(self):
        self.assertEqual(self.base(a=0), 0.0)

    def test_feedback_expiry(self):
        self.assertTrue(np.all(SEL.ego_confidence([0.9], 250.0) == 0.0))
        self.assertTrue(np.all(SEL.ego_confidence([0.9], None) == 0.0))
        self.assertTrue(np.all(SEL.ego_confidence([0.9], 100.0) == 0.9))

    def test_critical_mask(self):
        m = SEL.critical_mask([[10, 0], [-5, 0], [20, 15], [39.9, 9.9], [40.1, 0]])
        self.assertEqual(m.tolist(), [True, False, False, True, False])

    def test_selection_methods(self):
        p = np.array([0.9, 0.1, 0.8, 0.0]); a = np.array([1, 1, 1, 1])
        self.assertEqual(SEL.select_confidence_only(p, a, 2).tolist(), [0, 2])
        self.assertEqual(len(SEL.top_k([0, 0, 0], 2)), 0)               # zero-score tiles are never sent
        r1 = SEL.select_random(a, 2, np.random.default_rng(1)); r2 = SEL.select_random(a, 2, np.random.default_rng(1))
        self.assertEqual(r1.tolist(), r2.tolist())


class TestAdmission(unittest.TestCase):
    def test_monotone_in_queue_and_rate(self):
        kw = dict(k_wanted=56, t_ready_ms=40, t_reserve_ms=20)
        k_fast = ADM.max_admissible_k(queue_ms=0, rate_bps=20e6, **kw)
        k_slow = ADM.max_admissible_k(queue_ms=20, rate_bps=5e6, **kw)
        self.assertEqual(k_fast, 56)                      # 20 Mb/s carries 15.7 KB well inside 100 ms
        self.assertEqual(k_slow, 44)                      # 100-40-20-20 = 20 ms of wire time at 5 Mb/s = 12,500 B -> 44 tokens
        self.assertLessEqual(S.wire_cost(44), 12500); self.assertGreater(S.wire_cost(45), 12500)
        self.assertEqual(ADM.max_admissible_k(queue_ms=500, rate_bps=5e6, **kw), 0)   # nothing fits -> send nothing

    def test_rule_does_not_bind_when_processing_is_fast(self):
        # C3 link (5 Mb/s, 20 ms queue), fast processing: all 56 tokens fit inside 100 ms
        self.assertEqual(ADM.max_admissible_k(56, t_ready_ms=20, queue_ms=20, rate_bps=5e6, t_reserve_ms=15), 56)

    def test_rule_is_respected(self):
        k = ADM.max_admissible_k(56, 20, 20, 5e6, 15)
        self.assertLessEqual(ADM.predicted_completion_ms(k, 20, 20, 5e6, 15), ADM.D_MS)
        if k < 56:
            self.assertGreater(ADM.predicted_completion_ms(k + 1, 20, 20, 5e6, 15), ADM.D_MS)

    def test_budget_cap(self):
        self.assertEqual(ADM.max_admissible_k(100, 0, 0, 100e6, 0, budget_bytes=16 * 1024), 56)

    def test_rate_estimator(self):
        est = ADM.RateEstimator()
        self.assertEqual(est.rate, 5e6)
        est.update(delivered_bytes=1000, service_time_s=1000 * 8 / 20e6)   # observed 20 Mb/s
        self.assertAlmostEqual(est.rate, 0.8 * 5e6 + 0.2 * 20e6)


class TestLink(unittest.TestCase):
    def test_common_random_numbers(self):
        a = LR.loss_draw("sc", 3, 7, 2); b = LR.loss_draw("sc", 3, 7, 2)
        self.assertEqual(a, b)
        self.assertNotEqual(a, LR.loss_draw("sc", 3, 7, 3))

    def test_good_link_delivers_everything_fast(self):
        r = LR.transmit(56, LR.GOOD, t_ready_ms=0)
        self.assertTrue(r["token_delivered"].all())
        self.assertEqual(r["bytes_sent"], 15680)
        self.assertAlmostEqual(r["last_arrival_ms"], 1000 * 8 * 15680 / 20e6)

    def test_lost_packet_still_costs_bytes_and_time(self):
        lossy = LR.Link(5e6, 1.0, 20.0)                  # everything lost
        r = LR.transmit(8, lossy, 10)
        self.assertFalse(r["token_delivered"].any())
        self.assertEqual(r["bytes_sent"], S.wire_cost(8))
        self.assertGreater(r["last_arrival_ms"], 10 + 20)

    def test_loss_rate_is_about_ten_percent(self):
        n = 20000
        lost = sum(LR.loss_draw("x", f, 0, 0) < 0.10 for f in range(n)) / n
        self.assertAlmostEqual(lost, 0.10, delta=0.01)

    def test_late_packets_discarded(self):
        r = LR.transmit(56, LR.BAD, t_ready_ms=30)       # 5 Mb/s: 25 ms wire + 20 ms queue
        ok = LR.delivered_before(r, deadline_ms=60)
        self.assertLess(ok.sum(), r["token_delivered"].sum())


class TestTimeline(unittest.TestCase):
    def test_late_coop_loses_to_ego(self):
        self.assertEqual(TL.choose_output(112, 38, 100), "ego_only")      # Phase 2 unit test
        self.assertEqual(TL.choose_output(80, 38, 100), "cooperative")
        self.assertEqual(TL.choose_output(None, 120, 100), "empty")


if __name__ == "__main__":
    unittest.main()
