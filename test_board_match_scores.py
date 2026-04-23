"""Diagnose why board_debug3 cards return None from _process_card_roi."""
import cv2
import os
import sys
sys.path.insert(0, '.')

import logging
logging.basicConfig(level=logging.WARNING)  # suppress debug spam

from detectors.card_detector import CardDetector

detector = CardDetector()
print("Layout:", detector.layout_name)
print("Board card templates:", len(detector.board_card_templates))
print()

folder = 'board_debug3'
for fname in sorted(os.listdir(folder)):
    if not fname.endswith('.png'):
        continue
    img = cv2.imread(os.path.join(folder, fname))
    if img is None:
        continue
    
    idx = fname.replace('board_', '').replace('.png', '')
    context = 'board' + str(int(idx) + 1)

    # Extract card surface
    surface = detector._extract_card_surface(img)
    hero_surface = detector._extract_hero_card_surface(img, 'hero1') if hasattr(detector, '_extract_hero_card_surface') else None
    
    if surface is not None and surface.size > 0:
        match_gray = cv2.cvtColor(surface, cv2.COLOR_BGR2GRAY)
    else:
        match_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try board templates
    if detector.board_card_templates:
        tmap = detector.board_card_templates_by_context.get(context.lower()) or detector.board_card_templates
        name, score, second, _ = detector._get_card_template_stats_from_map(match_gray, tmap)
        gap = score - second if score and second else 0
        print(f"{fname} context={context}: board_template top={name} score={score:.3f} gap={gap:.3f} surface={'YES' if surface is not None else 'NO'}")
    
    # Try general card templates
    gen_name, gen_score, gen_second, _ = detector._get_card_template_stats(match_gray)
    gen_gap = gen_score - gen_second if gen_score and gen_second else 0
    print(f"  general_template top={gen_name} score={gen_score:.3f} gap={gen_gap:.3f}")

print()
print("Board card template threshold: score>=0.98 gap>=0.005 OR score>=0.78 gap>=0.02")
print("General card template threshold: score>=", detector.min_card_match_threshold, "gap>=", detector.min_card_match_gap)
