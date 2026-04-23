import unittest

from utils.action_abstraction import ActionAbstraction


class ActionAbstractionTests(unittest.TestCase):
    def test_build_actions_maps_live_raise_amount_to_nearest_bucket(self):
        abstraction = ActionAbstraction()
        bundle = abstraction.build_actions(
            {
                "pot_size": 1.20,
                "to_call": 0.40,
                "current_bet": 0.40,
                "hero_stack": 8.40,
                "available_actions": ["fold", "call", "raise"],
                "action_state": {"call_amount": 0.40, "raise_to_amount": 1.20},
            }
        )

        action_ids = [action["id"] for action in bundle["actions"]]
        self.assertIn("fold", action_ids)
        self.assertIn("call", action_ids)
        self.assertIn("bet_33", action_ids)
        self.assertIn("jam", action_ids)
        self.assertEqual(bundle["mapped_live_action"], "bet_66")

    def test_build_actions_uses_bet_sizes_for_check_spot(self):
        abstraction = ActionAbstraction()
        bundle = abstraction.build_actions(
            {
                "pot_size": 0.40,
                "to_call": 0.0,
                "current_bet": 0.0,
                "hero_stack": 9.40,
                "available_actions": ["fold", "check", "bet"],
                "action_state": {"call_amount": 0.0, "raise_to_amount": 0.20},
            }
        )

        labels = {action["id"]: action["amount"] for action in bundle["actions"]}
        self.assertEqual(bundle["default_action"], "check")
        self.assertGreater(labels["bet_33"], 0.0)
        self.assertGreater(labels["bet_pot"], labels["bet_33"])


if __name__ == "__main__":
    unittest.main()
