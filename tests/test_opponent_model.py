import unittest

from utils.opponent_model import OpponentProfileTracker


class OpponentModelTests(unittest.TestCase):
    def test_tracker_uses_balanced_profile_without_samples(self):
        tracker = OpponentProfileTracker()
        profile = tracker.get_profile()
        self.assertEqual(profile["sample_size"], 0)
        self.assertEqual(profile["style"], "balanced")
        self.assertEqual(profile["looseness"], 0.40)
        self.assertEqual(profile["aggression"], 0.35)

    def test_tracker_marks_repeated_pressure_as_aggressive(self):
        tracker = OpponentProfileTracker()
        for pot_size, to_call in [(0.30, 0.20), (0.90, 0.60), (1.50, 1.20), (2.00, 1.40)]:
            tracker.observe(
                {
                    "street": "flop",
                    "hole_cards": ["AH", "KD"],
                    "community_cards": ["7C", "4D", "2S"],
                    "pot_size": pot_size,
                    "to_call": to_call,
                    "available_actions": ["fold", "call", "raise"],
                    "buttons_confirmed": True,
                    "is_my_turn": True,
                    "num_players_remaining": 2,
                    "player_info": [{"role": "villain", "name": "dontseeyou1"}],
                }
            )
        profile = tracker.get_profile()
        self.assertEqual(profile["villain_key"], "dontseeyou1")
        self.assertGreater(profile["aggression"], 0.45)
        self.assertIn(profile["style"], {"tight_aggressive", "loose_aggressive", "balanced"})

    def test_tracker_resets_when_villain_changes(self):
        tracker = OpponentProfileTracker()
        tracker.observe(
            {
                "street": "preflop",
                "hole_cards": ["AH", "KD"],
                "community_cards": [],
                "pot_size": 0.10,
                "to_call": 0.10,
                "available_actions": ["fold", "call", "raise"],
                "buttons_confirmed": True,
                "is_my_turn": True,
                "num_players_remaining": 2,
                "player_info": [{"role": "villain", "name": "villain_one"}],
            }
        )
        tracker.observe(
            {
                "street": "preflop",
                "hole_cards": ["QS", "JD"],
                "community_cards": [],
                "pot_size": 0.10,
                "to_call": 0.0,
                "available_actions": ["fold", "check", "raise"],
                "buttons_confirmed": True,
                "is_my_turn": True,
                "num_players_remaining": 2,
                "player_info": [{"role": "villain", "name": "villain_two"}],
            }
        )
        profile = tracker.get_profile()
        self.assertEqual(profile["villain_key"], "villain_two")
        self.assertEqual(profile["sample_size"], 1)


if __name__ == "__main__":
    unittest.main()
