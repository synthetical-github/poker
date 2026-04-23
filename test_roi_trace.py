"""Trace exactly how roi_debug/Screenshot_27.png detects board cards - direct injection."""
import cv2
import sys
import logging
import os
sys.path.insert(0, '.')

logging.disable(logging.NOTSET)

from detectors.card_detector import CardDetector

detector = CardDetector()

# Directly test the community card ROIs from Screenshot_27
ss = cv2.imread('roi_debug/Screenshot_27.png')
if ss is None:
    print("NOT FOUND")
    sys.exit(1)

h, w = ss.shape[:2]
coords = (0, 0, w, h)
regions = detector.get_community_card_regions(coords)

print(f"Screenshot size: {w}x{h}")
print(f"Community card regions: {regions}")
print()

os.makedirs('debug_out', exist_ok=True)

for i, (rx, ry, rw, rh) in enumerate(regions):
    roi = ss[ry:ry+rh, rx:rx+rw]
    context = f'board{i+1}'
    
    # Extract surface
    surface = detector._extract_card_surface(roi)
    
    if surface is not None and surface.size > 0:
        match_gray = cv2.cvtColor(surface, cv2.COLOR_BGR2GRAY)
        surf_info = f"surface={surface.shape[:2]}"
    else:
        match_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        surf_info = "NO_SURFACE"
    
    has_surface = detector._has_community_card_surface(roi)
    
    # Try board templates
    board_name, board_score, board_second, _ = detector._get_card_template_stats_from_map(
        match_gray, detector.board_card_templates_by_context.get(context) or detector.board_card_templates
    )
    board_gap = board_score - board_second
    
    # General templates
    gen_name, gen_score, gen_second, _ = detector._get_card_template_stats(match_gray)
    gen_gap = gen_score - gen_second
    
    # Full process
    card = detector._process_card_roi(roi, context=context)
    
    print(f"ROI {i+1} ({rw}x{rh}) {surf_info} has_surface={has_surface}")
    print(f"  board_tpl: {board_name} score={board_score:.3f} gap={board_gap:.3f}")
    print(f"  general_tpl: {gen_name} score={gen_score:.3f} gap={gen_gap:.3f}")
    print(f"  _process_card_roi -> {card}")
    
    # Save images for visual inspection
    cv2.imwrite(f'debug_out/ss27_roi_{i+1}.png', roi)
    if surface is not None and surface.size > 0:
        cv2.imwrite(f'debug_out/ss27_surface_{i+1}.png', surface)

print()
print("Saved ROI images to debug_out/")
