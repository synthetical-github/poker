"""Test detection over all roi_debug screenshots."""
import cv2
import os
import sys
sys.path.insert(0, '.')
import logging
logging.disable(logging.CRITICAL)

from detectors.card_detector import CardDetector

detector = CardDetector()

for folder in ['roi_debug', 'roi_debug2', 'roi_debug3', 'roi_debug4']:
    if not os.path.isdir(folder):
        continue
    pngs = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])
    for fname in pngs:
        img = cv2.imread(os.path.join(folder, fname))
        if img is None:
            continue
        h, w = img.shape[:2]
        result = detector._get_frame_detection(img, (0, 0, w, h))
        hole = [str(c) for c in result.get('hole_cards', [])]
        board = [str(c) for c in result.get('community_cards', [])]
        layout = result.get('layout_name', '?')
        street = 'preflop' if len(board) == 0 else ('flop' if len(board) == 3 else ('turn' if len(board) == 4 else 'river'))
        print(folder + '/' + fname + ': hole=' + str(hole) + ' board=' + str(board) + ' (' + street + ', layout=' + layout + ')')
