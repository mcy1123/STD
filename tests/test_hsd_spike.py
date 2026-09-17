"""Cache and token contracts for the experimental nested cascade."""
import unittest


class HsdGreedyPrefixTests(unittest.TestCase):
    def test_full_accept_retains_the_bonus_prediction(self):
        from std_repro.hsd_spike import greedy_prefix
        self.assertEqual(greedy_prefix([2, 3, 4], [2, 3, 4, 5]), (3, 5))

    def test_rejection_uses_prediction_at_rejection_not_end_of_draft(self):
        from std_repro.hsd_spike import greedy_prefix
        self.assertEqual(greedy_prefix([2, 8, 9], [2, 3, 4, 5]), (1, 3))
        self.assertEqual(greedy_prefix([8, 9], [2, 3, 4]), (0, 2))

    def test_missing_bonus_is_rejected(self):
        from std_repro.hsd_spike import greedy_prefix
        with self.assertRaises(ValueError):
            greedy_prefix([2, 3], [2, 3])


if __name__ == "__main__":
    unittest.main()
