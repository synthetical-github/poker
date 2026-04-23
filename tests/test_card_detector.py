import unittest

import numpy as np

from detectors.card_detector import CardDetector
from utils.card_utils import Card


class CardDetectorBoardSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.detector = CardDetector()
        self.detector.layout_name = "heads_up"

    def test_heads_up_ignores_partial_board_surfaces_before_card_ocr(self):
        self.detector.get_community_card_regions = lambda table_coords: [
            (0, 0, 20, 30),
            (20, 0, 20, 30),
            (40, 0, 20, 30),
            (60, 0, 20, 30),
            (80, 0, 20, 30),
        ]
        self.detector._extract_card_surface = lambda roi: None
        self.detector._is_left_edge_contaminated = lambda image: False
        surface_iter = iter([True, True, False, False, False])
        self.detector._has_community_card_surface = lambda roi: next(surface_iter)

        processed_contexts = []
        self.detector._process_card_roi = lambda roi, context="generic": processed_contexts.append(context) or Card("A", "S")

        screenshot = np.zeros((40, 120, 3), dtype=np.uint8)
        cards, surface_count = self.detector._detect_community_cards_for_current_layout(
            screenshot,
            (0, 0, 120, 40),
        )

        self.assertEqual(cards, [])
        self.assertEqual(surface_count, 0)
        self.assertEqual(processed_contexts, [])

    def test_heads_up_processes_board_once_flop_surfaces_are_present(self):
        self.detector.get_community_card_regions = lambda table_coords: [
            (0, 0, 20, 30),
            (20, 0, 20, 30),
            (40, 0, 20, 30),
            (60, 0, 20, 30),
            (80, 0, 20, 30),
        ]
        self.detector._extract_card_surface = lambda roi: None
        self.detector._is_left_edge_contaminated = lambda image: False
        surface_iter = iter([True, True, True, False, False])
        self.detector._has_community_card_surface = lambda roi: next(surface_iter)

        returned_cards = iter([Card("4", "C"), Card("6", "S"), Card("Q", "H")])
        processed_contexts = []
        self.detector._process_card_roi = lambda roi, context="generic": processed_contexts.append(context) or next(returned_cards)

        screenshot = np.zeros((40, 120, 3), dtype=np.uint8)
        cards, surface_count = self.detector._detect_community_cards_for_current_layout(
            screenshot,
            (0, 0, 120, 40),
        )

        self.assertEqual([str(card) for card in cards], ["4C", "6S", "QH"])
        self.assertEqual(surface_count, 3)
        self.assertEqual(processed_contexts, ["board1", "board2", "board3"])


if __name__ == "__main__":
    unittest.main()
