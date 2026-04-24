"""
Debug-Script: Analysiert hero + board Kartenerkennung auf einem Screenshot.
Speichert ROI-Crops und zeigt Matching-Scores.
"""
import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from detectors.card_detector import CardDetector
from config import TABLE_LAYOUTS

SCREENSHOT = r"c:\poker\auto_screenshots\1777041374_6C3D_PRE.png"  # Echte Karten: 7C 6S + Flop 6C K? Q?

def draw_roi(img, x, y, w, h, label, color=(0,255,0)):
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    cv2.putText(img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def main():
    img = cv2.imread(SCREENSHOT)
    if img is None:
        print(f"FEHLER: Kann {SCREENSHOT} nicht laden")
        return

    ih, iw = img.shape[:2]
    print(f"Screenshot Groesse: {iw}x{ih}")

    layout = TABLE_LAYOUTS['acipayam_heads_up']
    ref_w, ref_h = layout['reference_size']
    print(f"Referenz Groesse: {ref_w}x{ref_h}")
    print(f"Skalierungsfaktor X: {iw/ref_w:.3f}, Y: {ih/ref_h:.3f}")

    sx = iw / ref_w
    sy = ih / ref_h

    det = CardDetector()

    debug_img = img.copy()

    # --- Hero ROIs ---
    print("\n=== HERO KARTEN ===")
    for i, (rx, ry, rw, rh) in enumerate(layout['hero_hole_cards']):
        x = int(rx * sx); y = int(ry * sy); w = int(rw * sx); h = int(rh * sy)
        roi = img[y:y+h, x:x+w]
        context = f"hero{i+1}"
        draw_roi(debug_img, x, y, w, h, context, (0, 255, 0))

        # Save ROI
        cv2.imwrite(f"c:\\poker\\debug_out\\roi_{context}.png", roi)
        print(f"\n{context} ROI: ({x},{y},{w},{h})")

        # Extract surface
        surface = det._extract_hero_card_surface(roi, context)
        if surface is not None:
            cv2.imwrite(f"c:\\poker\\debug_out\\surface_{context}.png", surface)
            sg = cv2.cvtColor(surface, cv2.COLOR_BGR2GRAY)
            print(f"  Surface: {surface.shape}")

            # Hero template scores
            if det.hero_card_templates:
                name, score, second, scores = det._get_card_template_stats_from_map(sg, det.hero_card_templates)
                top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"  Hero-Templates Top5: {top5}")
                print(f"  Best: {name} score={score:.3f} gap={score-second:.3f} (threshold=0.80, gap_min=0.02)")

            # Corner detection
            rank, rs, rs2 = det._detect_rank_from_corner(surface, context=context)
            suit, ss, ss2 = det._detect_suit_from_corner(surface, context=context)
            print(f"  Corner rank: {rank} score={rs:.3f} gap={rs-rs2:.3f}")
            print(f"  Corner suit: {suit} score={ss:.3f} gap={ss-ss2:.3f}")

            # Component detection
            comp = det._detect_card_from_compact_hero_components(surface)
            print(f"  Component detection: {comp}")

        # Final result
        result = det._process_card_roi(roi, context=context)
        print(f"  FINAL RESULT: {result}")

    # --- Board ROIs ---
    print("\n=== BOARD KARTEN ===")
    for i, (rx, ry, rw, rh) in enumerate(layout['community_cards']):
        x = int(rx * sx); y = int(ry * sy); w = int(rw * sx); h = int(rh * sy)
        roi = img[y:y+h, x:x+w]
        context = f"board{i+1}"
        draw_roi(debug_img, x, y, w, h, context, (0, 0, 255))

        has_surface = det._has_community_card_surface(roi)
        cv2.imwrite(f"c:\\poker\\debug_out\\roi_{context}.png", roi)
        print(f"\n{context} ROI: ({x},{y},{w},{h}) -> has_surface={has_surface}")

        if has_surface:
            import cv2 as _cv2
            surface = det._extract_card_surface(roi)
            if surface is not None:
                sg = _cv2.cvtColor(surface, _cv2.COLOR_BGR2GRAY)
                if det.board_card_templates:
                    _, bscore, bsecond, bscores = det._get_card_template_stats_from_map(sg, det.board_card_templates)
                    top5 = sorted(bscores.items(), key=lambda x: x[1], reverse=True)[:5]
                    print(f"  Board-Template Top5: {top5}")
                    print(f"  gap={bscore-bsecond:.3f}")
            result = det._process_card_roi(roi, context=context)
            print(f"  RESULT: {result}")

    # Save annotated debug image
    os.makedirs("c:\\poker\\debug_out", exist_ok=True)
    cv2.imwrite("c:\\poker\\debug_out\\annotated.png", debug_img)
    print("\nAnnotiertes Bild: c:\\poker\\debug_out\\annotated.png")

    # Board surface count check
    print("\n=== BOARD SURFACE COUNT ===")
    table_coords = (0, 0, iw, ih)
    det._apply_layout_profile('acipayam_heads_up')
    community_regions = det.get_community_card_regions(table_coords)
    surface_count = 0
    for j, (x, y, w, h) in enumerate(community_regions, 1):
        roi = img[y:y+h, x:x+w]
        has = det._has_community_card_surface(roi)
        print(f"  Board{j} ({x},{y},{w},{h}): has_surface={has}")
        if has:
            surface_count += 1
    print(f"  Total surface_count={surface_count} (braucht >=3 fuer acipayam)")

if __name__ == "__main__":
    os.makedirs("c:\\poker\\debug_out", exist_ok=True)
    main()
