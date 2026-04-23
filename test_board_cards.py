"""Quick test: detect board cards from board_debug3 images."""
import cv2
import os
import sys
sys.path.insert(0, '.')

from detectors.card_detector import CardDetector

detector = CardDetector()
print("CardDetector loaded. Layout:", detector.layout_name)
print("Board card templates loaded:", len(detector.board_card_templates))
print()

folder = 'board_debug3'
for fname in sorted(os.listdir(folder)):
    if not fname.endswith('.png'):
        continue
    path = os.path.join(folder, fname)
    img = cv2.imread(path)
    if img is None:
        print(f'{fname}: COULD NOT LOAD')
        continue
    h, w = img.shape[:2]
    
    has_surf = detector._has_community_card_surface(img)
    card = detector._process_card_roi(img, context='board1')
    print(f'{fname} ({w}x{h}): has_surface={has_surf}, detected={card}')

print()
print("--- Testing full detection pipeline on board_debug3 parent ---")
# Find a screenshot that has the board in it (if roi_debug has any)
for folder2 in ['roi_debug', 'roi_debug2', 'roi_debug3', 'roi_debug4']:
    if os.path.isdir(folder2):
        imgs = sorted([f for f in os.listdir(folder2) if f.endswith('.png')])[:3]
        for fname in imgs:
            img = cv2.imread(os.path.join(folder2, fname))
            if img is None:
                continue
            h, w = img.shape[:2]
            result = detector._get_frame_detection(img, (0, 0, w, h))
            hole = [str(c) for c in result.get('hole_cards', [])]
            board = [str(c) for c in result.get('community_cards', [])]
            print(f'{folder2}/{fname} ({w}x{h}): hole={hole}, board={board}')
        break
