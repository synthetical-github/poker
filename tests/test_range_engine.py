import unittest

from utils.card_utils import Card
from utils.range_engine import RangeEngine


class RangeEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = RangeEngine()
        self.deck = [
            Card(rank, suit)
            for rank in ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
            for suit in ['C', 'D', 'H', 'S']
        ]

    def test_weighted_range_excludes_dead_cards_and_emits_summary(self):
        dead_cards = {Card('A', 'S'), Card('K', 'D'), Card('Q', 'H'), Card('7', 'H')}
        deck = [card for card in self.deck if card not in dead_cards]
        weighted_range = self.engine.build_villain_range(
            deck=deck,
            community_cards=[Card('Q', 'H'), Card('7', 'H')],
            pot_size=1.2,
            to_call=0.4,
            street='flop',
            opponent_profile={
                'style': 'balanced',
                'looseness': 0.42,
                'aggression': 0.36,
                'bluff_rate': 0.16,
                'fold_equity': 0.34,
            },
        )
        self.assertGreater(len(weighted_range.candidates), 0)
        self.assertTrue(weighted_range.summary["top_hands"])
        self.assertEqual(weighted_range.summary["combo_count"], len(weighted_range.candidates))
        for combo in weighted_range.candidates:
            self.assertNotIn(Card('A', 'S'), combo)
            self.assertNotIn(Card('K', 'D'), combo)
            self.assertNotIn(Card('Q', 'H'), combo)
            self.assertNotIn(Card('7', 'H'), combo)

    def test_loose_aggressive_range_weights_connectors_more_than_tight_passive(self):
        deck = [card for card in self.deck if card not in {Card('A', 'S'), Card('K', 'D')}]
        loose = self.engine.build_villain_range(
            deck=deck,
            community_cards=[],
            pot_size=0.6,
            to_call=0.2,
            street='preflop',
            opponent_profile={
                'style': 'loose_aggressive',
                'looseness': 0.76,
                'aggression': 0.72,
                'bluff_rate': 0.28,
                'fold_equity': 0.20,
            },
        )
        tight = self.engine.build_villain_range(
            deck=deck,
            community_cards=[],
            pot_size=0.6,
            to_call=0.2,
            street='preflop',
            opponent_profile={
                'style': 'tight_passive',
                'looseness': 0.24,
                'aggression': 0.18,
                'bluff_rate': 0.08,
                'fold_equity': 0.48,
            },
        )

        def share(summary, category):
            for item in summary["top_categories"]:
                if item["category"] == category:
                    return item["share"]
            return 0.0

        self.assertGreaterEqual(share(loose.summary, "suited_connector"), share(tight.summary, "suited_connector"))
        self.assertGreater(loose.summary["total_weight"], 0.0)
        self.assertGreater(tight.summary["total_weight"], 0.0)


if __name__ == "__main__":
    unittest.main()
