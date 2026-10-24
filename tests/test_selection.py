import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selection import automatic_scores, rank_rows


class SelectionTests(unittest.TestCase):
    def test_rms_balance(self):
        scores, info = automatic_scores([0.0, 1.0], [0.2, 0.0])
        self.assertAlmostEqual(info["lambda0"], math.sqrt(0.5) / (math.sqrt(0.02) + 1e-8))
        self.assertAlmostEqual(scores[1], 1.0)

    def test_no_flip_uses_raw_drop(self):
        scores, info = automatic_scores([0, 0], [0.1, 0.4])
        self.assertEqual(scores, [0.1, 0.4])
        self.assertTrue(info["drop_fallback"])
        self.assertEqual(info["lambda0"], 0.0)

    def test_both_zero(self):
        self.assertEqual(automatic_scores([0, 0], [0, 0])[0], [0, 0])

    def test_no_drop_preserves_flip(self):
        self.assertEqual(automatic_scores([0.125, 0.5], [0, 0])[0], [0.125, 0.5])

    def test_invalid_components(self):
        for f, d in (([], []), ([0], [0, 1]), ([math.nan], [0]), ([-1], [0]), ([2], [0])):
            with self.assertRaises(ValueError):
                automatic_scores(f, d)

    @staticmethod
    def bank():
        return [{"uid": f"synthetic-{i}", "entropy": 20 - i, "rsp_flip": 0,
                 "rsp_drop": i / 100, "random": i / 20} for i in range(12)]

    def test_prefilter_and_pool_order(self):
        _, info = rank_rows(self.bank(), 2, "rsp_drop")
        self.assertEqual(info["candidate_indices"], list(range(8)))
        self.assertEqual(info["selected_indices"], [6, 7])

    def test_full_pool_changes_scope(self):
        _, info = rank_rows(self.bank(), 2, "rsp_auto", True)
        self.assertEqual(info["selected_indices"], [10, 11])
        self.assertEqual(info["coefficient_scope"], "full_pool")

    def test_cutoff_ties_use_index(self):
        rows = self.bank()
        for r in rows:
            r["rsp_drop"] = 0
        self.assertEqual(rank_rows(rows, 2, "rsp_auto")[1]["selected_indices"], [0, 1])

    def test_random_reuses_stored_keys(self):
        self.assertEqual(rank_rows(self.bank(), 2, "random4b")[1]["selected_indices"], [6, 7])

    def test_random_name_cannot_mean_full_bank(self):
        with self.assertRaises(ValueError):
            rank_rows(self.bank(), 2, "random4b", True)

    def test_labels_and_explanations_are_irrelevant(self):
        a = self.bank()
        b = [dict(r, answer_letter="C", rationale="synthetic unused text") for r in a]
        self.assertEqual(rank_rows(a, 2), rank_rows(b, 2))

    def test_duplicate_uid_rejected(self):
        rows = self.bank()
        rows[1]["uid"] = rows[0]["uid"]
        with self.assertRaises(ValueError):
            rank_rows(rows, 2)


if __name__ == "__main__":
    unittest.main()
