import os
import re
import sys

import cv2

sys.path.insert(0, '.')
from detectors.card_detector import CardDetector  # noqa: E402

det = CardDetector()
det._apply_layout_profile('acipayam_heads_up')

folder = 'auto_screenshots'
files = sorted(os.listdir(folder))

errors = 0
total = 0
for fname in files:
    m = re.match(r'\d+_([A-Z0-9]+)_([A-Z0-9]+)\.png', fname)
    if not m:
        continue
    hero_str, board_str = m.group(1), m.group(2)
    expected_hero = [hero_str[i:i+2] for i in range(0, len(hero_str), 2)]
    expected_board = [] if board_str == 'PRE' else [board_str[i:i+2] for i in range(0, len(board_str), 2)]

    img = cv2.imread(f'{folder}/{fname}')
    if img is None:
        continue
    h, w = img.shape[:2]
    tc = (0, 0, w, h)

    detected_hero = det.detect_hole_cards(img, tc)
    detected_board = det.detect_community_cards(img, tc)

    det_hero_str = [str(c) if c else 'None' for c in detected_hero]
    det_board_str = [str(c) if c else 'None' for c in detected_board if c]

    hero_ok = set(expected_hero) == set(det_hero_str)
    board_ok = (len(expected_board) == 0 and len(det_board_str) == 0) or set(expected_board) == set(det_board_str)

    total += 1
    if not hero_ok or not board_ok:
        errors += 1
        print(f'FAIL {fname}')
        if not hero_ok:
            print(f'  Hero:  expected={expected_hero}  got={det_hero_str}')
        if not board_ok:
            print(f'  Board: expected={expected_board}  got={det_board_str}')

print(f'\nTotal: {total}, Errors: {errors}, OK: {total-errors}')
