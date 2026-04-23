import unittest

from utils.subgame_solver import SubgameSolver


class SubgameSolverTests(unittest.TestCase):
    def test_solver_prefers_value_bet_with_strong_postflop_hand(self):
        solver = SubgameSolver()
        decision = solver.solve(
            game_state={"street": "flop"},
            monte_carlo_decision={
                "board_texture": {"texture": "wet"},
                "draws": {},
                "hand_details": {"rank_value": 4},
                "opponent_profile": {"aggression": 0.35, "fold_equity": 0.34},
                "action_evs": {"check": 0.12, "bet_33": 0.30, "bet_66": 0.34, "bet_pot": 0.28},
            },
            action_bundle={
                "actions": [
                    {"id": "check", "label": "Check", "kind": "check", "concrete_action": "check", "amount": 0.0},
                    {"id": "bet_33", "label": "33%", "kind": "aggressive", "concrete_action": "bet", "amount": 0.40, "pressure": 0.33},
                    {"id": "bet_66", "label": "66%", "kind": "aggressive", "concrete_action": "bet", "amount": 0.80, "pressure": 0.66},
                    {"id": "bet_pot", "label": "Pot", "kind": "aggressive", "concrete_action": "bet", "amount": 1.20, "pressure": 1.0},
                ]
            },
        )

        self.assertTrue(decision["enabled"])
        self.assertEqual(decision["recommended_action"], "bet")
        self.assertIn(decision["recommended_action_id"], {"bet_33", "bet_66", "bet_pot"})
        self.assertGreater(decision["confidence"], 0.50)

    def test_solver_stays_disabled_preflop(self):
        solver = SubgameSolver()
        decision = solver.solve(
            game_state={"street": "preflop"},
            monte_carlo_decision={"action_evs": {}},
            action_bundle={"actions": []},
        )
        self.assertFalse(decision["enabled"])
        self.assertEqual(decision["reason"], "solver_preflop_disabled")


if __name__ == "__main__":
    unittest.main()
