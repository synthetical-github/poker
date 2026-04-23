import unittest

from session_manager import SessionManager


class SessionManagerTests(unittest.TestCase):
    def test_micro_cash_stack_normalization_scales_down_false_decimal(self):
        manager = SessionManager()
        normalized = manager._normalize_stack_observation(
            raw_stack=66.0,
            reference_stack=6.9,
            history_reference=6.8,
            pot_size=0.15,
            to_call=0.05,
        )
        self.assertEqual(normalized, 6.6)

    def test_implausible_next_hand_stack_becomes_unknown(self):
        manager = SessionManager()
        manager.current_hand = {
            "initial_hero_stack": 6.9,
            "hero_stack_samples": [6.9, 6.88],
            "snapshots": [
                {"pot_size": 0.15, "to_call": 0.05},
                {"pot_size": 0.20, "to_call": 0.00},
            ],
            "recommendations": [],
        }
        self.assertFalse(manager._is_plausible_stack_delta(6.9, 66.0))

    def test_build_snapshot_prefers_sanitized_game_state_stacks(self):
        manager = SessionManager()
        snapshot = manager._build_snapshot(
            game_state={
                "street": "preflop",
                "hole_cards": ["AH", "TD"],
                "community_cards": [],
                "pot_size": 0.15,
                "to_call": 0.05,
                "current_bet": 0.05,
                "available_actions": ["fold", "call", "raise"],
                "buttons_confirmed": True,
                "is_my_turn": True,
                "position": "button",
                "hero_stack": 7.18,
                "villain_stack": 10.10,
                "player_info": [
                    {"role": "hero", "chips": 0.0},
                    {"role": "villain", "chips": 0.0},
                ],
            },
            strategy={
                "recommended_action": "call",
                "amount": 0.05,
                "confidence": 0.72,
                "reason": "Test",
                "hand_details": {"display_category": "High Card"},
                "board_texture": {"texture": "dry"},
                "draws": {},
                "opponent_profile": {"style": "balanced"},
                "range_summary": {"headline": "balanced range"},
                "monte_carlo": {"hand_equity": 0.54, "range_equity": 0.49},
                "solver_decision": {"recommended_action": "call"},
                "policy_source": "monte_carlo_mix",
                "parser_confidence": 0.77,
            },
        )
        self.assertEqual(snapshot["hero_stack"], 7.18)
        self.assertEqual(snapshot["villain_stack"], 10.10)
        self.assertTrue(snapshot["buttons_confirmed"])
        self.assertTrue(snapshot["is_my_turn"])
        self.assertEqual(snapshot["opponent_style"], "balanced")
        self.assertEqual(snapshot["policy_source"], "monte_carlo_mix")

    def test_new_hand_detected_when_board_resets_to_preflop_even_with_same_hole(self):
        manager = SessionManager()
        manager.current_hand = {
            "hole_cards": ["8H", "TS"],
            "final_board": ["QD", "4C", "9C"],
        }
        self.assertTrue(
            manager._is_new_hand(
                {
                    "hole_cards": ["8H", "TS"],
                    "community_cards": [],
                    "street": "preflop",
                    "buttons_confirmed": True,
                    "is_my_turn": True,
                }
            )
        )

    def test_new_hand_detected_when_same_hole_cards_reappear_with_new_flop(self):
        manager = SessionManager()
        manager.current_hand = {
            "hole_cards": ["4D", "TD"],
            "final_board": ["TS", "AC", "4H"],
        }
        self.assertTrue(
            manager._is_new_hand(
                {
                    "hole_cards": ["4D", "TD"],
                    "community_cards": ["7D", "4C", "2H"],
                    "street": "flop",
                    "buttons_confirmed": True,
                    "is_my_turn": True,
                }
            )
        )

    def test_unconfirmed_snapshot_is_not_added_as_actionable_recommendation(self):
        manager = SessionManager()
        manager.current_hand = {
            "hand_id": "test_hand",
            "started_at": 1.0,
            "finished_at": None,
            "hole_cards": ["AH", "KD"],
            "initial_hero_stack": 1.3,
            "final_hero_stack": 1.3,
            "hero_stack_delta": None,
            "outcome": "unknown",
            "outcome_source": "unknown",
            "position": "unknown",
            "recommendations": [],
            "snapshots": [],
            "hero_stack_samples": [],
            "final_board": [],
            "final_hand_category": "Preflop",
        }
        manager._update_current_hand(
            {
                "timestamp": 1.0,
                "street": "preflop",
                "community_cards": [],
                "pot_size": 0.03,
                "to_call": 0.0,
                "hero_stack": 1.3,
                "villain_stack": 2.13,
                "available_actions": [],
                "buttons_confirmed": False,
                "is_my_turn": False,
                "recommended_action": "raise",
                "recommended_amount": 0.4,
                "confidence": 0.87,
                "reason": "Test",
                "hand_category": "Preflop",
                "board_texture": "preflop",
                "draws": {},
            }
        )
        self.assertEqual(manager.current_hand["recommendations"], [])
        manager._update_current_hand(
            {
                "timestamp": 2.0,
                "street": "preflop",
                "community_cards": [],
                "pot_size": 0.08,
                "to_call": 0.03,
                "hero_stack": 1.3,
                "villain_stack": 2.13,
                "available_actions": ["fold", "call", "raise"],
                "buttons_confirmed": True,
                "is_my_turn": True,
                "recommended_action": "raise",
                "recommended_amount": 0.4,
                "confidence": 0.87,
                "reason": "Test",
                "hand_category": "Preflop",
                "board_texture": "preflop",
                "draws": {},
            }
        )
        self.assertEqual(len(manager.current_hand["recommendations"]), 1)
        self.assertEqual(manager.current_hand["recommendations"][0]["action"], "raise")


if __name__ == "__main__":
    unittest.main()
