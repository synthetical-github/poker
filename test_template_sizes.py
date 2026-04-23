"""Compare template sizes vs actual card surfaces."""
import cv2
import os
import sys
sys.path.insert(0, '.')
import logging
logging.disable(logging.CRITICAL)

from detectors.card_detector import CardDetector

detector = CardDetector()

# Show template dimensions
first_key = list(detector.board_card_templates.keys())[0]
first_tpl = detector.board_card_templates[first_key]
print(f"Board template sample: {first_key} shape={first_tpl.shape}")

# board_debug3 surface extraction
img = cv2.imread('board_debug3/board_0.png')
print(f"board_0.png shape: {img.shape}")
surface = detector._extract_card_surface(img)
if surface is not None:
    gray_s = cv2.cvtColor(surface, cv2.COLOR_BGR2GRAY)
    print(f"board_0 extracted surface shape: {gray_s.shape}")
else:
    gray_s = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"board_0 NO surface extracted, using full gray: {gray_s.shape}")

# Check roi_debug full screenshot
ss = cv2.imread('roi_debug/Screenshot_27.png')
if ss is not None:
    h, w = ss.shape[:2]
    coords = (0, 0, w, h)
    regions = detector.get_community_card_regions(coords)
    print(f"\nFull screenshot {w}x{h}, community_card_regions[0]={regions[0]}")
    rx, ry, rw, rh = regions[0]
    roi = ss[ry:ry+rh, rx:rx+rw]
    print(f"ROI shape: {roi.shape}")
    surf = detector._extract_card_surface(roi)
    if surf is not None:
        sg = cv2.cvtColor(surf, cv2.COLOR_BGR2GRAY)
        print(f"ROI surface shape: {sg.shape}")
        name, score, second, _ = detector._get_card_template_stats_from_map(sg, detector.board_card_templates)
        print(f"ROI match: {name} score={score:.3f} gap={score-second:.3f}")
        # Also save the surface for visual inspection
        cv2.imwrite('debug_out/board_roi_surface.png', surf)
        print("Saved surface to debug_out/board_roi_surface.png")
    else:
        rg = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        print(f"ROI NO surface, using full roi gray: {rg.shape}")
        name, score, second, _ = detector._get_card_template_stats_from_map(rg, detector.board_card_templates)
        print(f"ROI match: {name} score={score:.3f} gap={score-second:.3f}")
    
    # Also save the ROI
    os.makedirs('debug_out', exist_ok=True)
    cv2.imwrite('debug_out/board_roi_from_screenshot.png', roi)
    print("Saved ROI to debug_out/board_roi_from_screenshot.png")

# Also show board card templates by context info
for ctx in ['board1', 'board2', 'board3']:
    tmap = detector.board_card_templates_by_context.get(ctx, {})
    if tmap:
        k = list(tmap.keys())[0]
        print(f"\n{ctx} template sample: {k} shape={tmap[k].shape}, count={len(tmap)}")
