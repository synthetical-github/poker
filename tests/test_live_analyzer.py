import unittest
from unittest.mock import patch

from live_analyzer import LivePokerAnalyzer
from utils.card_utils import Card
from config import LIVE_CONFIG


class LiveAnalyzerStateTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = LivePokerAnalyzer(headless=True)
        self.analyzer.table_parser.layout_name = 'acipayam_heads_up'

    def test_heads_up_check_state_zeroes_current_bet(self):
        sanitized = self.analyzer._sanitize_game_state(
            {
                'hole_cards': [Card('9', 'H'), Card('5', 'C')],
                'community_cards': [],
                'pot_size': 0.4,
                'to_call': 0.0,
                'current_bet': 0.7,
                'num_players_remaining': 2,
                'available_actions': ['fold', 'check', 'raise'],
                'hero_stack': 3.99,
                'villain_stack': 4.21,
                'action_state': {'call_amount': 0.0, 'raise_to_amount': 0.4},
            }
        )
        self.assertEqual(sanitized['to_call'], 0.0)
        self.assertEqual(sanitized['current_bet'], 0.0)

    def test_heads_up_state_requires_two_players(self):
        self.assertFalse(
            self.analyzer._is_plausible_state(
                {
                    'hole_cards': [Card('A', 'S'), Card('K', 'D')],
                    'community_cards': [],
                    'pot_size': 0.15,
                    'to_call': 0.05,
                    'current_bet': 0.05,
                    'num_players_remaining': 1,
                    'available_actions': ['fold', 'call', 'raise'],
                    'hero_stack': 3.95,
                    'villain_stack': None,
                }
            )
        )

    def test_villain_stack_outlier_uses_previous_value(self):
        self.analyzer.table_parser.layout_name = 'heads_up'
        self.analyzer.current_game_state = {
            'hero_stack': 825.0,
            'villain_stack': 2095.0,
        }
        sanitized = self.analyzer._sanitize_game_state(
            {
                'hole_cards': [Card('A', 'H'), Card('K', 'C')],
                'community_cards': [],
                'pot_size': 20.0,
                'to_call': 0.0,
                'current_bet': 0.0,
                'num_players_remaining': 2,
                'available_actions': ['fold', 'check', 'raise'],
                'hero_stack': 825.0,
                'villain_stack': 21753.0,
                'action_state': {'call_amount': 0.0, 'raise_to_amount': 15.0},
            }
        )
        self.assertEqual(sanitized['villain_stack'], 2095.0)

    def test_current_bet_outlier_is_capped_to_to_call(self):
        sanitized = self.analyzer._sanitize_game_state(
            {
                'hole_cards': [Card('A', 'D'), Card('2', 'H')],
                'community_cards': [],
                'pot_size': 0.10,
                'to_call': 0.10,
                'current_bet': 1.90,
                'num_players_remaining': 2,
                'available_actions': ['fold', 'call', 'raise'],
                'hero_stack': 9.72,
                'villain_stack': 20.18,
                'action_state': {'call_amount': 0.10, 'raise_to_amount': 0.40},
            }
        )
        self.assertEqual(sanitized['current_bet'], 0.10)

    def test_missing_hero_stack_uses_last_valid_stack_fallback(self):
        self.analyzer.table_parser.layout_name = 'heads_up'
        self.analyzer.current_game_state = None
        self.analyzer.last_valid_stacks['hero'] = 220.0
        sanitized = self.analyzer._sanitize_game_state(
            {
                'hole_cards': [Card('A', 'D'), Card('7', 'S')],
                'community_cards': [],
                'pot_size': 30.0,
                'to_call': 10.0,
                'current_bet': 10.0,
                'num_players_remaining': 2,
                'available_actions': ['fold', 'call', 'raise'],
                'hero_stack': None,
                'villain_stack': 190.0,
                'action_state': {'call_amount': 10.0, 'raise_to_amount': 40.0},
            }
        )
        self.assertEqual(sanitized['hero_stack'], 220.0)

    def test_same_hand_hero_stack_does_not_jump_up(self):
        self.analyzer.table_parser.layout_name = 'heads_up'
        self.analyzer.current_game_state = {
            'hole_cards': [Card('5', 'H'), Card('J', 'D')],
            'community_cards': [],
            'street': 'preflop',
            'pot_size': 90.0,
            'to_call': 30.0,
            'current_bet': 30.0,
            'hero_stack': 230.0,
            'villain_stack': 630.0,
            'available_actions': ['fold', 'call', 'raise'],
            'buttons_confirmed': True,
            'is_my_turn': True,
            'action_state': {'call_amount': 30.0, 'raise_to_amount': 120.0},
        }
        stabilized = self.analyzer._stabilize_state_from_previous(
            {
                'hole_cards': [Card('5', 'H'), Card('J', 'D')],
                'community_cards': [Card('J', 'C'), Card('9', 'H'), Card('4', 'C')],
                'street': 'flop',
                'pot_size': 120.0,
                'to_call': 0.0,
                'current_bet': 0.0,
                'hero_stack': 250.0,
                'villain_stack': 630.0,
                'available_actions': ['fold', 'check', 'bet'],
                'buttons_confirmed': True,
                'is_my_turn': True,
                'action_state': {'call_amount': 0.0, 'raise_to_amount': 120.0},
            }
        )
        self.assertEqual(stabilized['hero_stack'], 230.0)

    def test_same_hand_villain_stack_outlier_drop_uses_previous_value(self):
        self.analyzer.table_parser.layout_name = 'heads_up'
        self.analyzer.current_game_state = {
            'hole_cards': [Card('6', 'H'), Card('5', 'D')],
            'community_cards': [Card('T', 'S'), Card('T', 'C'), Card('4', 'C')],
            'street': 'flop',
            'pot_size': 60.0,
            'to_call': 0.0,
            'current_bet': 0.0,
            'hero_stack': 220.0,
            'villain_stack': 645.0,
            'available_actions': ['fold', 'check', 'bet'],
            'buttons_confirmed': True,
            'is_my_turn': True,
            'action_state': {'call_amount': 0.0, 'raise_to_amount': 60.0},
        }
        stabilized = self.analyzer._stabilize_state_from_previous(
            {
                'hole_cards': [Card('6', 'H'), Card('5', 'D')],
                'community_cards': [Card('T', 'S'), Card('T', 'C'), Card('4', 'C')],
                'street': 'flop',
                'pot_size': 60.0,
                'to_call': 0.0,
                'current_bet': 0.0,
                'hero_stack': 220.0,
                'villain_stack': 70.0,
                'available_actions': ['fold', 'check', 'bet'],
                'buttons_confirmed': True,
                'is_my_turn': True,
                'action_state': {'call_amount': 0.0, 'raise_to_amount': 60.0},
            }
        )
        self.assertEqual(stabilized['villain_stack'], 645.0)

    def test_same_spot_reuses_previous_actions_when_ocr_drops_buttons(self):
        self.analyzer.table_parser.layout_name = 'heads_up'
        previous_state = {
            'hole_cards': [Card('T', 'C'), Card('3', 'H')],
            'community_cards': [],
            'street': 'preflop',
            'pot_size': 40.0,
            'to_call': 0.0,
            'hero_stack': 220.0,
            'villain_stack': 419.0,
            'available_actions': ['fold', 'check', 'raise'],
            'buttons_confirmed': True,
            'is_my_turn': True,
            'action_state': {'call_amount': 0.0, 'raise_to_amount': 30.0},
        }
        self.analyzer.current_game_state = previous_state
        stabilized = self.analyzer._stabilize_state_from_previous(
            {
                'hole_cards': [Card('T', 'C'), Card('3', 'H')],
                'community_cards': [],
                'street': 'preflop',
                'pot_size': 40.0,
                'to_call': 0.0,
                'hero_stack': 220.0,
                'villain_stack': 419.0,
                'available_actions': [],
                'buttons_confirmed': False,
                'is_my_turn': False,
                'action_state': {'call_amount': 0.0, 'raise_to_amount': 0.0},
            }
        )
        self.assertEqual(stabilized['available_actions'], ['fold', 'check', 'raise'])
        self.assertTrue(stabilized['buttons_confirmed'])
        self.assertTrue(stabilized['is_my_turn'])
        self.assertEqual(stabilized['action_state']['raise_to_amount'], 30.0)

    def test_postflop_transition_keeps_previous_hole_cards_on_flicker(self):
        self.analyzer.table_parser.layout_name = 'heads_up'
        self.analyzer.current_game_state = {
            'hole_cards': [Card('T', 'C'), Card('3', 'H')],
            'community_cards': [],
            'street': 'preflop',
            'pot_size': 40.0,
            'to_call': 0.0,
            'hero_stack': 220.0,
            'villain_stack': 419.0,
            'available_actions': ['fold', 'check', 'raise'],
            'buttons_confirmed': True,
            'is_my_turn': True,
            'action_state': {'call_amount': 0.0, 'raise_to_amount': 30.0},
        }
        stabilized = self.analyzer._stabilize_state_from_previous(
            {
                'hole_cards': [Card('T', 'S'), Card('3', 'H')],
                'community_cards': [Card('7', 'D'), Card('3', 'C'), Card('T', 'C')],
                'street': 'flop',
                'pot_size': 60.0,
                'to_call': 20.0,
                'hero_stack': 220.0,
                'villain_stack': 399.0,
                'available_actions': ['fold', 'call', 'raise'],
                'buttons_confirmed': True,
                'is_my_turn': True,
                'action_state': {'call_amount': 20.0, 'raise_to_amount': 60.0},
            }
        )
        self.assertEqual([str(card) for card in stabilized['hole_cards']], ['TC', '3H'])
        self.assertEqual([str(card) for card in stabilized['community_cards']], ['7D', '3C', 'TC'])

    def test_new_preflop_state_does_not_keep_previous_board(self):
        self.analyzer.table_parser.layout_name = 'heads_up'
        self.analyzer.current_game_state = {
            'hole_cards': [Card('8', 'H'), Card('T', 'S')],
            'community_cards': [Card('Q', 'D'), Card('4', 'C'), Card('9', 'C')],
            'street': 'flop',
            'pot_size': 180.0,
            'to_call': 60.0,
            'hero_stack': 440.0,
            'villain_stack': 380.0,
            'available_actions': ['fold', 'call', 'raise'],
            'buttons_confirmed': True,
            'is_my_turn': True,
            'action_state': {'call_amount': 60.0, 'raise_to_amount': 180.0},
        }
        stabilized = self.analyzer._stabilize_state_from_previous(
            {
                'hole_cards': [Card('A', 'D'), Card('7', 'S')],
                'community_cards': [],
                'street': 'preflop',
                'pot_size': 30.0,
                'to_call': 10.0,
                'hero_stack': 410.0,
                'villain_stack': 580.0,
                'available_actions': ['fold', 'call', 'raise'],
                'buttons_confirmed': True,
                'is_my_turn': True,
                'action_state': {'call_amount': 10.0, 'raise_to_amount': 40.0},
            }
        )
        self.assertEqual([str(card) for card in stabilized['hole_cards']], ['AD', '7S'])
        self.assertEqual(stabilized['community_cards'], [])
        self.assertEqual(stabilized['street'], 'preflop')

    def test_board_reset_to_preflop_is_plausible_new_hand_transition(self):
        previous_state = {
            'hole_cards': [Card('8', 'H'), Card('T', 'S')],
            'community_cards': [Card('Q', 'D'), Card('4', 'C'), Card('9', 'C')],
            'street': 'flop',
            'pot_size': 180.0,
            'to_call': 60.0,
            'hero_stack': 440.0,
            'villain_stack': 380.0,
            'available_actions': ['fold', 'call', 'raise'],
            'buttons_confirmed': True,
            'is_my_turn': True,
        }
        new_state = {
            'hole_cards': [Card('8', 'H'), Card('T', 'S')],
            'community_cards': [],
            'street': 'preflop',
            'pot_size': 40.0,
            'to_call': 0.0,
            'hero_stack': 300.0,
            'villain_stack': 600.0,
            'available_actions': ['fold', 'check', 'raise'],
            'buttons_confirmed': True,
            'is_my_turn': True,
        }
        self.assertTrue(self.analyzer._is_plausible_transition(previous_state, new_state))

    def test_different_flop_with_same_hole_cards_is_treated_as_new_hand(self):
        previous_state = {
            'hole_cards': [Card('4', 'D'), Card('T', 'D')],
            'community_cards': [Card('T', 'S'), Card('A', 'C'), Card('4', 'H')],
            'street': 'flop',
            'pot_size': 0.08,
            'to_call': 0.0,
            'hero_stack': 1.82,
            'villain_stack': 3.19,
            'available_actions': ['fold', 'check', 'bet'],
            'buttons_confirmed': True,
            'is_my_turn': True,
        }
        new_state = {
            'hole_cards': [Card('4', 'D'), Card('T', 'D')],
            'community_cards': [Card('7', 'D'), Card('4', 'C'), Card('2', 'H')],
            'street': 'flop',
            'pot_size': 0.10,
            'to_call': 0.76,
            'hero_stack': 1.48,
            'villain_stack': 1.71,
            'available_actions': ['fold', 'call', 'raise'],
            'buttons_confirmed': True,
            'is_my_turn': True,
        }
        self.assertTrue(self.analyzer._is_plausible_transition(previous_state, new_state))

    def test_unconfirmed_spot_shows_wait_in_recommendation_line(self):
        line = self.analyzer._build_recommendation_line(
            {
                'street': 'preflop',
                'hole_cards': [Card('A', 'S'), Card('7', 'D')],
                'is_my_turn': False,
                'buttons_confirmed': False,
                'available_actions': [],
            },
            {
                'recommended_action': 'raise',
                'amount': 0.4,
                'hand_details': {'display_category': 'Preflop', 'rank_value': 0},
            }
        )
        self.assertIn('NEXT: WARTEN', line)
        self.assertNotIn('RAISE 0.40', line)

    def test_confirmed_spot_uses_compact_readable_recommendation_line(self):
        line = self.analyzer._build_recommendation_line(
            {
                'street': 'preflop',
                'hole_cards': [Card('A', 'S'), Card('7', 'D')],
                'community_cards': [],
                'is_my_turn': True,
                'buttons_confirmed': True,
                'available_actions': ['fold', 'call', 'raise'],
            },
            {
                'recommended_action': 'call',
                'amount': 0.0,
                'pot_odds': 0.25,
                'equity_proxy': 0.336,
                'opponent_profile': {'style': 'loose_passive'},
                'range_summary': {'top_hands': [{'hand': 'AKo'}, {'hand': 'JJ'}, {'hand': 'QQ'}, {'hand': 'KK'}]},
                'solver_decision': {},
            }
        )
        self.assertIn('PREFLOP | AS 7D | NEXT: CALL', line)
        self.assertIn('EQ 33.6%', line)
        self.assertIn('ODDS 25.0%', line)
        self.assertIn('RANGE AKo JJ QQ KK', line)

    def test_live_summary_uses_clear_multiline_sections(self):
        summary = self.analyzer._build_live_summary(
            {
                'street': 'flop',
                'hole_cards': [Card('A', 'S'), Card('7', 'D')],
                'community_cards': [Card('Q', 'H'), Card('7', 'H'), Card('2', 'C')],
                'pot_size': 1.20,
                'to_call': 0.40,
                'available_actions': ['fold', 'call', 'raise'],
                'is_my_turn': True,
                'buttons_confirmed': True,
            },
            {
                'recommended_action': 'call',
                'amount': 0.0,
                'pot_odds': 0.25,
                'equity_proxy': 0.336,
                'hand_details': {'display_category': 'Pair', 'rank_value': 1},
                'board_texture': {'texture': 'semi_wet'},
                'draws': {'flush_draw': False, 'open_ended_straight_draw': False, 'gutshot': True, 'combo_draw': False},
                'opponent_profile': {'style': 'loose_passive'},
                'range_summary': {'top_hands': [{'hand': 'AKo'}, {'hand': 'JJ'}, {'hand': 'QQ'}, {'hand': 'KK'}]},
                'monte_carlo': {'hand_equity': 0.336, 'range_equity': 0.302},
                'solver_decision': {},
                'policy_source': 'monte_carlo_postflop',
                'parser_confidence': 0.78,
                'reason': 'mc sample reason',
            }
        )
        self.assertIn('=== Live Spot ===', summary)
        self.assertIn('FLOP | AS 7D | BOARD QH 7H 2C', summary)
        self.assertIn('Status: AM ZUG | Next: CALL', summary)
        self.assertIn('Equity hand 33.6% | range 30.2% | odds 25.0%', summary)
        self.assertIn('Villain loose_passive | Range AKo JJ QQ KK', summary)

    def test_heads_up_tournament_amounts_snap_to_blinds(self):
        self.analyzer.table_parser.layout_name = 'heads_up'
        LIVE_CONFIG['detected_window_title'] = "Heads Up (€2) 1159390467 | NL Hold'em | Level 2 | 15/30 | Game version 1.0"
        sanitized = self.analyzer._sanitize_game_state(
            {
                'hole_cards': [Card('A', 'D'), Card('9', 'H')],
                'community_cards': [],
                'pot_size': 40.0,
                'to_call': 10.0,
                'current_bet': 10.0,
                'num_players_remaining': 2,
                'available_actions': ['fold', 'call', 'raise'],
                'buttons_confirmed': True,
                'is_my_turn': True,
                'hero_stack': 1200.0,
                'villain_stack': 1300.0,
                'action_state': {'call_amount': 10.0, 'raise_to_amount': 40.0},
            }
        )
        self.assertEqual(sanitized['pot_size'], 45.0)
        self.assertEqual(sanitized['to_call'], 15.0)
        self.assertEqual(sanitized['current_bet'], 15.0)

    def test_heads_up_tournament_confirmed_spot_rejects_pot_smaller_than_to_call(self):
        self.analyzer.table_parser.layout_name = 'heads_up'
        LIVE_CONFIG['detected_window_title'] = "Heads Up (€2) 1159390467 | NL Hold'em | Level 2 | 15/30 | Game version 1.0"
        sanitized = self.analyzer._sanitize_game_state(
            {
                'hole_cards': [Card('A', 'D'), Card('9', 'H')],
                'community_cards': [],
                'pot_size': 7.0,
                'to_call': 10.0,
                'current_bet': 10.0,
                'num_players_remaining': 2,
                'available_actions': ['fold', 'call', 'raise'],
                'buttons_confirmed': True,
                'is_my_turn': True,
                'hero_stack': 1200.0,
                'villain_stack': 1300.0,
                'action_state': {'call_amount': 10.0, 'raise_to_amount': 40.0},
            }
        )
        self.assertFalse(self.analyzer._is_plausible_state(sanitized))

    def test_waiting_summary_repeats_after_heartbeat_interval(self):
        original_live_summary = LIVE_CONFIG.get('show_live_summary', False)
        original_reco_line = LIVE_CONFIG.get('show_recommendation_line', True)
        self.addCleanup(lambda: LIVE_CONFIG.__setitem__('show_live_summary', original_live_summary))
        self.addCleanup(lambda: LIVE_CONFIG.__setitem__('show_recommendation_line', original_reco_line))
        LIVE_CONFIG['show_live_summary'] = False
        LIVE_CONFIG['show_recommendation_line'] = True

        game_state = {
            'street': 'preflop',
            'hole_cards': [Card('A', 'S'), Card('7', 'D')],
            'community_cards': [],
            'is_my_turn': False,
            'buttons_confirmed': False,
            'available_actions': [],
        }
        strategy = {
            'recommended_action': 'call',
            'amount': 0.0,
            'opponent_profile': {'style': 'balanced'},
            'range_summary': {'top_hands': [{'hand': 'AKo'}]},
        }

        with patch('builtins.print') as mocked_print, patch('time.time', side_effect=[0.0, 6.0]):
            self.analyzer._print_live_summary(game_state, strategy)
            self.analyzer._print_live_summary(game_state, strategy)

        self.assertEqual(mocked_print.call_count, 2)


if __name__ == "__main__":
    unittest.main()
