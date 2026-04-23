import unittest
from unittest import mock

from config import get_current_table_layout_name
from utils import screen_utils
from utils.screen_utils import _title_matches


class TestScreenUtils(unittest.TestCase):
    def setUp(self):
        self._window_cache = dict(screen_utils.window_cache)
        self._detected_title = screen_utils.LIVE_CONFIG.get('detected_window_title')

    def tearDown(self):
        screen_utils.window_cache.clear()
        screen_utils.window_cache.update(self._window_cache)
        screen_utils.LIVE_CONFIG['detected_window_title'] = self._detected_title

    def test_title_matches_minimal_nl_holdem_title(self):
        self.assertTrue(_title_matches("| NL Hold'em", "NL Hold'em"))

    def test_title_matches_full_heads_up_window_title(self):
        title = "Heads Up (€2) 1159232958 | NL Hold'em | Level 4 | 25/50 | Game version 1.0"
        self.assertTrue(_title_matches(title, "NL Hold'em"))

    def test_layout_detection_prefers_heads_up_hint(self):
        title = "Heads Up (€2) 1159232958 | NL Hold'em | Level 4 | 25/50 | Game version 1.0"
        self.assertEqual(get_current_table_layout_name(title), "heads_up")

    def test_layout_detection_keeps_cash_layout_for_arzon(self):
        title = "Arzon 817772815 | NL Hold'em | €0.10/€0.20 | Game version 1.0"
        self.assertEqual(get_current_table_layout_name(title), "acipayam_heads_up")

    def test_find_window_by_title_prefers_specific_table_over_generic_client(self):
        titles = {
            101: "Poker Swiss Casinos",
            202: "Arzon 817772815 | NL Hold'em | €0.10/€0.20 | Game version 1.0",
        }
        rects = {
            101: (0, 0, 1900, 1300),
            202: (50, 50, 1500, 1000),
        }

        def enum_windows(handler, _ctx):
            handler(101, None)
            handler(202, None)

        fake_gui = mock.Mock()
        fake_gui.EnumWindows.side_effect = enum_windows
        fake_gui.IsWindowVisible.side_effect = lambda hwnd: True
        fake_gui.IsIconic.side_effect = lambda hwnd: False
        fake_gui.GetWindowText.side_effect = lambda hwnd: titles[hwnd]
        fake_gui.GetWindowRect.side_effect = lambda hwnd: rects[hwnd]

        with mock.patch.object(screen_utils, "USE_WIN32", True), \
             mock.patch.object(screen_utils, "win32gui", fake_gui):
            hwnd = screen_utils._find_window_by_title(["NL Hold'em"], "Poker Swiss Casinos")

        self.assertEqual(hwnd, 202)
        self.assertEqual(
            screen_utils.LIVE_CONFIG.get('detected_window_title'),
            titles[202],
        )

    def test_generic_cached_title_keeps_last_specific_detected_title(self):
        screen_utils.window_cache.update({
            'hwnd': 202,
            'title': "Arzon 817772815 | NL Hold'em | €0.10/€0.20 | Game version 1.0",
            'seen_at': 100.0,
            'last_specific_hwnd': 202,
            'last_specific_title': "Arzon 817772815 | NL Hold'em | €0.10/€0.20 | Game version 1.0",
            'last_specific_seen_at': 100.0,
        })

        with mock.patch.object(screen_utils, "USE_WIN32", True), \
             mock.patch.object(screen_utils, "_cached_window_is_usable", return_value=True), \
             mock.patch.object(screen_utils, "_get_window_title", return_value="Poker Swiss Casinos"), \
             mock.patch.object(screen_utils.time, "time", return_value=110.0):
            hwnd = screen_utils._find_window_by_title(["NL Hold'em"], "Poker Swiss Casinos")

        self.assertEqual(hwnd, 202)
        self.assertEqual(
            screen_utils.LIVE_CONFIG.get('detected_window_title'),
            "Arzon 817772815 | NL Hold'em | €0.10/€0.20 | Game version 1.0",
        )


if __name__ == "__main__":
    unittest.main()
