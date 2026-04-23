import unittest

from strategy_engine import StrategyEngine
from utils.card_utils import Card, get_best_hand_details, get_hand_rank
from utils.monte_carlo_strategy import MonteCarloStrategy
from utils.poker_decision import analyze_board_texture, detect_draws, starting_hand_key


class PokerLogicTests(unittest.TestCase):
    def test_strategy_engine_uses_monte_carlo_mode(self):
        engine = StrategyEngine()
        self.assertEqual(engine.poker_model.strategy_mode, "monte_carlo")
        strategy = engine.calculate_strategy(
            hole_cards=[Card("A", "S"), Card("K", "D")],
            community_cards=[],
            table_info={
                "pot_size": 1.5,
                "to_call": 0.0,
                "current_bet": 0.0,
                "num_players_remaining": 2,
                "position": "button",
                "street": "preflop",
                "available_actions": ["fold", "check", "raise"],
                "buttons_confirmed": True,
                "is_my_turn": True,
                "player_info": [{"role": "villain", "name": "dontseeyou1"}],
            },
        )
        self.assertIn("range_summary", strategy)
        self.assertTrue(strategy["range_summary"])
        self.assertIn("abstract_actions", strategy)
        self.assertIn("solver_decision", strategy)
        self.assertIn("policy_source", strategy)

    def test_hand_evaluator_detects_straight_flush(self):
        hole_cards = [Card("A", "S"), Card("K", "S")]
        board = [Card("Q", "S"), Card("J", "S"), Card("T", "S"), Card("2", "D"), Card("3", "C")]
        rank_value, best_cards = get_hand_rank(hole_cards, board)
        details = get_best_hand_details(hole_cards, board)
        self.assertEqual(rank_value, 8)
        self.assertEqual(details["category"], "straight_flush")
        self.assertEqual([str(card) for card in best_cards], ["AS", "KS", "QS", "JS", "TS"])

    def test_board_texture_marks_wet_board(self):
        board = [Card("Q", "H"), Card("J", "H"), Card("T", "C")]
        texture = analyze_board_texture(board)
        self.assertEqual(texture["texture"], "wet")
        self.assertTrue(texture["connected"])
        self.assertEqual(texture["flush_pressure"], "draw_heavy")

    def test_draw_detection_finds_combo_draw(self):
        hole_cards = [Card("A", "H"), Card("K", "H")]
        board = [Card("Q", "H"), Card("J", "H"), Card("2", "C")]
        draws = detect_draws(hole_cards, board)
        self.assertTrue(draws["flush_draw"])
        self.assertTrue(draws["gutshot"] or draws["open_ended_straight_draw"])
        self.assertTrue(draws["combo_draw"])

    def test_strategy_engine_raises_premium_preflop(self):
        engine = StrategyEngine()
        strategy = engine.calculate_strategy(
            hole_cards=[Card("A", "S"), Card("A", "D")],
            community_cards=[],
            table_info={
                "pot_size": 3.0,
                "to_call": 1.0,
                "current_bet": 1.0,
                "num_players_remaining": 2,
                "position": "button",
                "street": "preflop",
                "available_actions": ["fold", "call", "raise"],
            },
        )
        self.assertEqual(strategy["recommended_action"], "raise")
        self.assertGreater(strategy["amount"], 0.0)
        self.assertEqual(starting_hand_key([Card("A", "S"), Card("A", "D")]), "AA")

    def test_heads_up_strategy_opens_wider_preflop(self):
        engine = StrategyEngine()
        strategy = engine.calculate_strategy(
            hole_cards=[Card("K", "S"), Card("7", "D")],
            community_cards=[],
            table_info={
                "pot_size": 1.5,
                "to_call": 0.0,
                "current_bet": 0.0,
                "num_players_remaining": 2,
                "position": "button",
                "street": "preflop",
                "available_actions": ["fold", "check", "raise"],
            },
        )
        self.assertEqual(strategy["recommended_action"], "raise")
        self.assertGreater(strategy["amount"], 0.0)

    def test_heads_up_strategy_folds_trash_vs_raise(self):
        engine = StrategyEngine()
        strategy = engine.calculate_strategy(
            hole_cards=[Card("7", "S"), Card("2", "D")],
            community_cards=[],
            table_info={
                "pot_size": 1.5,
                "to_call": 1.0,
                "current_bet": 1.0,
                "num_players_remaining": 2,
                "position": "blind",
                "street": "preflop",
                "available_actions": ["fold", "call", "raise"],
            },
        )
        self.assertEqual(strategy["recommended_action"], "fold")

    def test_heads_up_strategy_does_not_open_ten_three_off(self):
        engine = StrategyEngine()
        strategy = engine.calculate_strategy(
            hole_cards=[Card("T", "S"), Card("3", "D")],
            community_cards=[],
            table_info={
                "pot_size": 1.5,
                "to_call": 0.0,
                "current_bet": 0.0,
                "num_players_remaining": 2,
                "position": "button",
                "street": "preflop",
                "available_actions": ["fold", "check", "raise"],
            },
        )
        self.assertEqual(strategy["recommended_action"], "check")

    def test_heads_up_strategy_does_not_open_ten_two_suited(self):
        engine = StrategyEngine()
        strategy = engine.calculate_strategy(
            hole_cards=[Card("T", "C"), Card("2", "C")],
            community_cards=[],
            table_info={
                "pot_size": 1.5,
                "to_call": 0.0,
                "current_bet": 0.0,
                "num_players_remaining": 2,
                "position": "button",
                "street": "preflop",
                "available_actions": ["fold", "check", "raise"],
            },
        )
        self.assertEqual(strategy["recommended_action"], "check")

    def test_heads_up_strategy_defends_ace_two_off_for_small_price(self):
        engine = StrategyEngine()
        strategy = engine.calculate_strategy(
            hole_cards=[Card("A", "S"), Card("2", "H")],
            community_cards=[],
            table_info={
                "pot_size": 0.15,
                "to_call": 0.05,
                "current_bet": 0.05,
                "num_players_remaining": 2,
                "position": "blind",
                "street": "preflop",
                "available_actions": ["fold", "call", "raise"],
            },
        )
        self.assertEqual(strategy["recommended_action"], "call")

    def test_heads_up_strategy_folds_ace_two_off_for_one_blind_price(self):
        engine = StrategyEngine()
        strategy = engine.calculate_strategy(
            hole_cards=[Card("A", "D"), Card("2", "H")],
            community_cards=[],
            table_info={
                "pot_size": 0.10,
                "to_call": 0.10,
                "current_bet": 0.10,
                "num_players_remaining": 2,
                "position": "blind",
                "street": "preflop",
                "available_actions": ["fold", "call", "raise"],
            },
        )
        self.assertEqual(strategy["recommended_action"], "fold")

    def test_heads_up_strategy_folds_queen_five_off_for_one_blind_price(self):
        engine = StrategyEngine()
        strategy = engine.calculate_strategy(
            hole_cards=[Card("Q", "C"), Card("5", "D")],
            community_cards=[],
            table_info={
                "pot_size": 0.10,
                "to_call": 0.10,
                "current_bet": 0.10,
                "num_players_remaining": 2,
                "position": "blind",
                "street": "preflop",
                "available_actions": ["fold", "call", "raise"],
            },
        )
        self.assertEqual(strategy["recommended_action"], "fold")

    def test_heads_up_strategy_continues_nine_two_suited_for_small_price(self):
        engine = StrategyEngine()
        strategy = engine.calculate_strategy(
            hole_cards=[Card("9", "H"), Card("2", "H")],
            community_cards=[],
            table_info={
                "pot_size": 0.15,
                "to_call": 0.05,
                "current_bet": 0.05,
                "num_players_remaining": 2,
                "position": "blind",
                "street": "preflop",
                "available_actions": ["fold", "call", "raise"],
            },
        )
        self.assertEqual(strategy["recommended_action"], "call")

    def test_heads_up_micro_open_raise_stays_small(self):
        engine = StrategyEngine()
        strategy = engine.calculate_strategy(
            hole_cards=[Card("7", "S"), Card("7", "H")],
            community_cards=[],
            table_info={
                "pot_size": 0.20,
                "to_call": 0.0,
                "current_bet": 0.0,
                "num_players_remaining": 2,
                "position": "button",
                "street": "preflop",
                "available_actions": ["fold", "check", "raise"],
            },
        )
        self.assertEqual(strategy["recommended_action"], "raise")
        self.assertLessEqual(strategy["amount"], 0.40)

    def test_heads_up_small_pair_defends_small_reraise_price(self):
        engine = StrategyEngine()
        strategy = engine.calculate_strategy(
            hole_cards=[Card("5", "S"), Card("5", "H")],
            community_cards=[],
            table_info={
                "pot_size": 0.10,
                "to_call": 0.08,
                "current_bet": 0.08,
                "num_players_remaining": 2,
                "position": "blind",
                "street": "preflop",
                "available_actions": ["fold", "call", "raise"],
            },
        )
        self.assertEqual(strategy["recommended_action"], "call")

    def test_heads_up_strategy_folds_a5_off_vs_bad_cash_price(self):
        engine = StrategyEngine()
        strategy = engine.calculate_strategy(
            hole_cards=[Card("A", "D"), Card("5", "C")],
            community_cards=[],
            table_info={
                "pot_size": 0.10,
                "to_call": 0.30,
                "current_bet": 0.30,
                "num_players_remaining": 2,
                "position": "blind",
                "street": "preflop",
                "available_actions": ["fold", "call", "raise"],
            },
        )
        self.assertEqual(strategy["recommended_action"], "fold")

    def test_heads_up_strategy_continues_ats_vs_half_pot_price(self):
        engine = StrategyEngine()
        strategy = engine.calculate_strategy(
            hole_cards=[Card("A", "H"), Card("T", "H")],
            community_cards=[],
            table_info={
                "pot_size": 1.60,
                "to_call": 0.80,
                "current_bet": 0.80,
                "num_players_remaining": 2,
                "position": "blind",
                "street": "preflop",
                "available_actions": ["fold", "call", "raise"],
            },
        )
        self.assertEqual(strategy["recommended_action"], "call")

    def test_heads_up_strategy_folds_ajo_vs_bad_cash_reraise_price(self):
        engine = StrategyEngine()
        strategy = engine.calculate_strategy(
            hole_cards=[Card("A", "S"), Card("J", "H")],
            community_cards=[],
            table_info={
                "pot_size": 0.10,
                "to_call": 0.35,
                "current_bet": 0.35,
                "num_players_remaining": 2,
                "position": "blind",
                "street": "preflop",
                "available_actions": ["fold", "call", "raise"],
            },
        )
        self.assertEqual(strategy["recommended_action"], "fold")

    def test_postflop_strategy_uses_board_cards_for_hand_evaluation(self):
        engine = StrategyEngine()
        strategy = engine.calculate_strategy(
            hole_cards=[Card("A", "H"), Card("K", "H")],
            community_cards=[Card("Q", "H"), Card("J", "H"), Card("T", "H")],
            table_info={
                "pot_size": 1.20,
                "to_call": 0.0,
                "current_bet": 0.0,
                "num_players_remaining": 2,
                "position": "button",
                "street": "flop",
                "available_actions": ["fold", "check", "bet"],
            },
        )
        self.assertEqual(strategy["hand_details"]["display_category"], "Straight Flush")
        self.assertEqual(strategy["recommended_action"], "bet")
        self.assertTrue(strategy["solver_decision"].get("enabled"))

    def test_flop_ev_prefers_betting_combo_draw_over_check(self):
        strategy = MonteCarloStrategy()
        decision = strategy.analyze(
            {
                "hole_cards": [Card("A", "H"), Card("Q", "H")],
                "community_cards": [Card("J", "H"), Card("7", "H"), Card("2", "C")],
                "pot_size": 1.2,
                "to_call": 0.0,
                "current_bet": 0.0,
                "street": "flop",
                "position": "button",
                "num_players_remaining": 2,
                "hero_stack": 6.0,
                "available_actions": ["fold", "check", "bet"],
            },
            opponent_profile={"style": "balanced", "looseness": 0.40, "aggression": 0.35, "fold_equity": 0.34, "bluff_rate": 0.16},
        )
        self.assertGreater(decision["action_evs"]["bet_66"], decision["action_evs"]["check"])

    def test_turn_ev_keeps_jam_below_call_for_medium_one_pair(self):
        strategy = MonteCarloStrategy()
        decision = strategy.analyze(
            {
                "hole_cards": [Card("A", "H"), Card("9", "C")],
                "community_cards": [Card("A", "D"), Card("7", "S"), Card("4", "H"), Card("2", "C")],
                "pot_size": 1.8,
                "to_call": 0.6,
                "current_bet": 0.6,
                "street": "turn",
                "position": "blind",
                "num_players_remaining": 2,
                "hero_stack": 4.2,
                "available_actions": ["fold", "call", "raise"],
            },
            opponent_profile={"style": "balanced", "looseness": 0.40, "aggression": 0.35, "fold_equity": 0.34, "bluff_rate": 0.16},
        )
        self.assertGreater(decision["action_evs"]["call"], decision["action_evs"]["jam"])

    def test_river_ev_prefers_big_value_with_full_house(self):
        strategy = MonteCarloStrategy()
        decision = strategy.analyze(
            {
                "hole_cards": [Card("7", "H"), Card("7", "C")],
                "community_cards": [Card("A", "S"), Card("7", "D"), Card("2", "H"), Card("A", "C"), Card("2", "S")],
                "pot_size": 3.0,
                "to_call": 0.0,
                "current_bet": 0.0,
                "street": "river",
                "position": "button",
                "num_players_remaining": 2,
                "hero_stack": 8.0,
                "available_actions": ["fold", "check", "bet"],
            },
            opponent_profile={"style": "balanced", "looseness": 0.40, "aggression": 0.35, "fold_equity": 0.34, "bluff_rate": 0.16},
        )
        self.assertGreater(decision["action_evs"]["bet_pot"], decision["action_evs"]["check"])


if __name__ == "__main__":
    unittest.main()
