import sys

import cv2

sys.path.insert(0, '.')
from detectors.card_detector import CardDetector

det = CardDetector()
det._apply_layout_profile('acipayam_heads_up')
print(
    f"Hero templates: {len(det.hero_card_templates)} -> "
    f"{sorted(det.hero_card_templates.keys())}"
)

tests = [
    ("Screenshot_30.png", ["5C", "7D"], ["2H", "8H", "JD", "AD", "9C"]),
    ("Screenshot_35.png", ["9S", "2C"], ["QC", "TH", "AH", "8S", "TS"]),
    ("Screenshot_15.png", ["KC", "6S"], ["JH", "QD", "5H", "7S", "TS"]),
    ("Screenshot_5.png", ["5H", "2C"], []),
]

for fname, h_exp, b_exp in tests:
    img = cv2.imread(f"C:/poker-1/Neuer Ordner/{fname}")
    h, w = img.shape[:2]
    tc = (0, 0, w, h)
    hole = det.detect_hole_cards(img, tc)
    comm = det.detect_community_cards(img, tc)
    hero_got = [str(c) for c in hole]
    board_got = [str(c) for c in comm]
    hero_ok = hero_got == h_exp
    board_ok = sorted(board_got) == sorted(b_exp)
    status_h = "OK" if hero_ok else "FAIL"
    status_b = "OK" if board_ok else "FAIL"
    print(f"{fname}:")
    print(f"  Hero:  got={hero_got} exp={h_exp} [{status_h}]")
    print(f"  Board: got={board_got} exp={b_exp} [{status_b}]")
