"""
Scannt alle 52 Screenshots und sammelt unerkannte Karten fuer Hero und Board.
Gibt aus welche Karten erkannt wurden und welche nicht (None).
"""
from collections import Counter

import cv2
from detectors.card_detector import CardDetector
from utils.config import TABLE_LAYOUTS

det = CardDetector()
det._apply_layout_profile('acipayam_heads_up')

folder = "C:/poker-1/Neuer Ordner"
layout = TABLE_LAYOUTS["acipayam_heads_up"]

all_hero = []
none_hero = []
all_board = []
none_board = []

for i in range(1, 53):
    fname = f"Screenshot_{i}.png"
    img = cv2.imread(f"{folder}/{fname}")
    if img is None:
        continue
    h, w = img.shape[:2]
    tc = (0, 0, w, h)

    hole = det.detect_hole_cards(img, tc)
    comm = det.detect_community_cards(img, tc)

    for card in hole:
        if card is None:
            none_hero.append((i, "hero", "None"))
        else:
            all_hero.append(str(card))

    for card in comm:
        if card is None:
            none_board.append((i, "board"))
        else:
            all_board.append(str(card))

print("=== Hero cards recognized ===")
hero_c = Counter(all_hero)
for k, v in sorted(hero_c.items()):
    print(f"  {k}: {v}x")

print(f"Hero Nones: {none_hero}")

print("\n=== Board cards recognized ===")
board_c = Counter(all_board)
for k, v in sorted(board_c.items()):
    print(f"  {k}: {v}x")

print(f"Board Nones: {none_board}")
