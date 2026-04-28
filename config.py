# config.py
import os
import platform

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Logging Konfiguration
LOGGING = {
    'level': 'DEBUG',
    'console_enabled': False,
    'console_level': 'WARNING',
    'file': 'pokerbot.log',
    'format': '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
}

# Live Analyse Konfiguration
LIVE_CONFIG = {
    # Capture-Modus:
    # - 'window' = sucht ein sichtbares App-Fenster per Fenstertitel und liest dieses live aus
    # - 'screen' = benutzt eine feste Desktop-Region
    'capture_method': 'window',

    # Im Window-Modus soll KEINE feste Bildschirm-Region erzwungen werden
    'screen_region': None,

    # Hauptsuchbegriff für das App-Fenster
    'window_title_contains': "NL Hold'em",

    # Zusätzliche erlaubte Varianten für robustere Suche
    'window_title_aliases': [
        "NL Hold'em",
        "| NL Hold'em",
        'NL HOLDEM',
        'NDL HOLDEM',
        'NL Holdem',
        'NL HOLDEM',
    ],

    # Optionaler Fallback, falls der eigentliche Titel abweicht
    'window_title_fallback_contains': 'Poker Swiss Casinos',

    # Fenstersuche robuster machen
    'window_search_case_insensitive': True,
    'require_window_visible': True,
    'require_window_not_minimized': True,
    'prefer_client_area_capture': True,
    'allow_screen_fallback': False,

    # Layout-Erkennung
    'table_layout': 'auto',
    'default_table_layout': 'acipayam_heads_up',
    'table_layout_hints': {
        'heads up': 'heads_up',
        'acipayam': 'acipayam_heads_up',
        "nl hold'em": 'acipayam_heads_up',
        'poker swiss casinos': 'acipayam_heads_up',
    },

    # Laufzeit / Anzeige
    'voice_enabled': False,
    'show_overlay': True,
    'show_debug_window': False,
    'show_live_summary': False,
    'show_recommendation_line': True,
    'analysis_interval': 0.5,

    # Debug / Stabilität
    'debug_mode': True,
    'save_debug_images': True,

    # OCR
    'use_ocr_for_buttons': True,
    'use_ocr_for_numbers': True,
    'multi_pass_ocr': True,

    # Nur für Windows, falls nicht im PATH
    'tesseract_cmd': r'C:\Program Files\Tesseract-OCR\tesseract.exe',
}

# Poker spezifische Einstellungen
POKER_SETTINGS = {
    'suits': ['C', 'D', 'H', 'S'],
    'ranks': ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'],
    'rank_map': {rank: i for i, rank in enumerate(['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'])},
    'num_players': 9,
}

TABLE_LAYOUTS = {
    'ring': {
        'reference_size': (1945, 1379),
        'hero_hole_cards': [
            (852, 938, 112, 135),
            (946, 938, 112, 135),
        ],
        'community_cards': [
            (642, 500, 125, 160),
            (765, 500, 125, 160),
            (888, 500, 125, 160),
            (1011, 500, 125, 160),
            (1134, 500, 125, 160),
        ],
        'pot': (900, 429, 155, 66),
        'pot_stack': (1185, 1168, 240, 78),
        'hero_player_area': (751, 1110, 387, 173),
        'villain_player_area': (676, 89, 274, 123),
        'player_areas': [
            {'region': (751, 1110, 387, 173), 'role': 'hero'},
            {'region': (676, 89, 274, 123), 'role': 'top_center'},
            {'region': (78, 165, 322, 133), 'role': 'upper_left'},
            {'region': (32, 510, 307, 143), 'role': 'left_middle'},
            {'region': (212, 909, 343, 148), 'role': 'lower_left'},
            {'region': (1542, 907, 329, 146), 'role': 'lower_right'},
            {'region': (1625, 507, 303, 145), 'role': 'right_middle'},
            {'region': (1547, 165, 323, 132), 'role': 'upper_right'},
        ],
    },

    'heads_up': {
        'reference_size': (1927, 1364),
        'hero_hole_cards': [
            (804, 928, 96, 132),
            (910, 928, 96, 132),
        ],
        'community_cards': [
            (638, 496, 124, 158),
            (760, 496, 124, 158),
            (882, 496, 124, 158),
            (1004, 496, 124, 158),
            (1126, 496, 124, 158),
        ],
        'pot': (870, 430, 230, 82),
        'pot_stack': (923, 664, 60, 36),
        'hero_player_area': (803, 935, 380, 176),
        'villain_player_area': (780, 56, 362, 258),
        'player_areas': [
            {'region': (803, 935, 380, 176), 'role': 'hero', 'stack_region': (848, 1110, 208, 62)},
            {'region': (780, 56, 362, 258), 'role': 'villain', 'stack_region': (858, 270, 202, 72)},
        ],
    },

    'acipayam_heads_up': {
        'reference_size': (1935, 1369),
        'hero_hole_cards': [
            (836, 902, 92, 136),
            (944, 902, 92, 136),
        ],
        'community_cards': [
            (648, 514, 120, 174),
            (782, 514, 120, 174),
            (917, 514, 120, 174),
            (1052, 514, 120, 174),
            (1187, 514, 120, 174),
        ],
        'pot': (872, 432, 224, 82),
        'pot_stack': (926, 662, 78, 40),
        'hero_player_area': (804, 939, 384, 176),
        'villain_player_area': (782, 56, 366, 264),
        'player_areas': [
            {'region': (804, 939, 384, 176), 'role': 'hero', 'stack_region': (860, 1120, 190, 56)},
            {'region': (782, 56, 366, 264), 'role': 'villain', 'stack_region': (870, 284, 194, 68)},
        ],
    },
}

TABLE_ACTION_LAYOUTS = {
    'ring': {
        'reference_size': (1945, 1379),
        'fold_button': (1330, 1230, 250, 122),
        'check_button': (1565, 1230, 190, 122),
        'bet_button': (1745, 1230, 195, 122),
        'bet_input_field': (1251, 1167, 688, 60),
    },

    'heads_up': {
        'reference_size': (1927, 1364),
        'fold_button': (1242, 1210, 226, 132),
        'check_button': (1468, 1210, 225, 132),
        'bet_button': (1694, 1210, 226, 132),
        'bet_input_field': (1240, 1155, 680, 92),
    },

    'acipayam_heads_up': {
        'reference_size': (1935, 1369),
        'fold_button': (1230, 1170, 260, 190),
        'check_button': (1450, 1170, 260, 190),
        'bet_button': (1670, 1170, 280, 190),
        'bet_input_field': (1220, 1120, 720, 110),
    },
}


def get_current_table_layout_name(window_title: str = "") -> str:
    configured_layout = LIVE_CONFIG.get('table_layout', 'auto')
    if configured_layout and configured_layout != 'auto':
        return configured_layout if configured_layout in TABLE_LAYOUTS else LIVE_CONFIG.get('default_table_layout', 'ring')

    title_to_check = (window_title or LIVE_CONFIG.get('detected_window_title') or '').lower()
    for title_hint, layout_name in LIVE_CONFIG.get('table_layout_hints', {}).items():
        if title_hint.lower() in title_to_check and layout_name in TABLE_LAYOUTS:
            return layout_name

    return LIVE_CONFIG.get('default_table_layout', 'ring')


def get_table_rois(layout_name: str = None):
    layout_name = layout_name or get_current_table_layout_name()
    return TABLE_LAYOUTS.get(layout_name, TABLE_LAYOUTS[LIVE_CONFIG.get('default_table_layout', 'ring')])


def get_table_action_rois(layout_name: str = None):
    layout_name = layout_name or get_current_table_layout_name()
    return TABLE_ACTION_LAYOUTS.get(layout_name, TABLE_ACTION_LAYOUTS[LIVE_CONFIG.get('default_table_layout', 'ring')])


TABLE_ROIS = get_table_rois()
TABLE_ACTION_ROIS = get_table_action_rois()

# Strategie Konfiguration
STRATEGY_CONFIG = {
    'mode': 'monte_carlo',
    'default_action': 'fold',
    'bet_size_percentage': 0.5,
    'raise_size_percentage': 0.75,
    'min_bet_factor': 0.2,
    'min_raise_factor': 2.0,
    'monte_carlo_preflop_iterations': 280,
    'monte_carlo_postflop_iterations': 220,
}

# Pfade zu Templates und Assets
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
CARD_TEMPLATES_DIR = os.path.join(ASSETS_DIR, 'card_templates')
TABLE_TEMPLATE_PATH = os.path.join(ASSETS_DIR, 'table_template.png')

# Setze Tesseract Pfad, falls unter Windows
if platform.system() == "Windows" and 'tesseract_cmd' in LIVE_CONFIG:
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = LIVE_CONFIG['tesseract_cmd']
    except Exception as e:
        print(
            f"WARNUNG: Tesseract OCR ist nicht korrekt installiert oder der Pfad "
            f"'{LIVE_CONFIG['tesseract_cmd']}' ist falsch. Details: {e}"
        )
