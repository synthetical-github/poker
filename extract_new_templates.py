"""
Extract new hero card templates and suit templates from the new-style screenshots.

Strategy:
- Rank: use hero_rank_templates (style-independent black numeral shapes)
- Suit: direct HSV color analysis (blue=D, red=H, dark=S/C)
         For S vs C disambiguation: use existing corner suit templates
- Avoids falling back to biased card template matching
"""
import re
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from detectors.card_detector import CardDetector  # noqa: E402
from config import TABLE_LAYOUTS  # noqa: E402

OUTPUT_CARD_DIR = Path("assets/hero_card_templates_v3")
OUTPUT_SUIT_DIR = Path("assets/hero_suit_templates_v3")
DEBUG_DIR = Path("debug_template_extraction")
DEBUG_DIR.mkdir(exist_ok=True)

SCREENSHOTS_DIR = Path("auto_screenshots")
NEW_PREFIX = "177748"


def get_hero_rois():
    layout = TABLE_LAYOUTS["acipayam_heads_up"]
    return layout.get("hero_hole_cards", [])


def is_card_visible(surface_bgr, threshold=90):
    if surface_bgr is None or surface_bgr.size == 0:
        return False
    gray = cv2.cvtColor(surface_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) > threshold


def detect_suit_by_color(surface_bgr, suit_roi_bgr, det: CardDetector):
    """
    Detect suit using direct HSV color analysis.
    New card style: blue=D, red=H, dark (no color)=S or C.
    For S vs C: use existing hero suit corner templates (shapes are style-independent).
    Returns (suit_letter, confidence) or (None, 0.0).
    """
    # Analyse the full card surface for color presence
    hsv_full = cv2.cvtColor(surface_bgr, cv2.COLOR_BGR2HSV)
    pix = hsv_full.reshape(-1, 3).astype(np.int32)
    total = len(pix)

    blue_count = int(np.sum(
        (pix[:, 0] >= 95) & (pix[:, 0] <= 135)
        & (pix[:, 1] > 50) & (pix[:, 2] > 80)
    ))
    red_count = int(np.sum(
        ((pix[:, 0] <= 12) | (pix[:, 0] >= 163))
        & (pix[:, 1] > 80) & (pix[:, 2] > 80)
    ))

    blue_ratio = blue_count / total
    red_ratio = red_count / total

    if blue_ratio > 0.008:
        return 'D', min(0.95, blue_ratio * 30)
    if red_ratio > 0.008:
        return 'H', min(0.95, red_ratio * 30)

    # Dark/achromatic suit: distinguish S vs C using existing corner templates
    if suit_roi_bgr is not None and suit_roi_bgr.size > 0:
        suit_gray = cv2.cvtColor(suit_roi_bgr, cv2.COLOR_BGR2GRAY)
        # Try hero suit templates (S and C shapes work across card styles)
        for tmpl_map in [det.hero_suit_templates, det.suit_corner_templates]:
            if not tmpl_map:
                continue
            best_suit, best_score, second = det._get_symbol_template_stats(suit_gray, tmpl_map)
            gap = best_score - second
            if best_suit in ('S', 'C') and best_score >= 0.30 and gap >= 0.02:
                return best_suit, best_score

    return None, 0.0


def detect_rank_hero(surface_bgr, det: CardDetector):
    """Detect rank using hero rank templates only — no card template fallback."""
    rank_roi, _ = det._extract_corner_regions(surface_bgr, context="hero")
    if rank_roi is None or rank_roi.size == 0:
        return None, 0.0
    rank_gray = cv2.cvtColor(rank_roi, cv2.COLOR_BGR2GRAY)
    best_rank, best_score, second = det._get_symbol_template_stats(rank_gray, det.hero_rank_templates)
    if best_rank and best_score >= 0.35 and (best_score - second) >= 0.02:
        return best_rank, best_score
    return None, 0.0


def get_existing_templates():
    existing = {}
    for f in OUTPUT_CARD_DIR.glob("*.png"):
        m = re.match(r'^([2-9AKQJT][CDHS])__', f.name)
        if m:
            existing.setdefault(m.group(1), []).append(f)
    return existing


def main():
    det = CardDetector()
    det._apply_layout_profile("acipayam_heads_up")

    hero_rois = get_hero_rois()
    print(f"Hero ROIs: {hero_rois}")

    existing = get_existing_templates()
    print(f"Cards already with templates: {sorted(existing.keys())}")

    all_cards = [f"{r}{s}" for r in "AKQJT98765432" for s in "SHDC"]
    missing = set(all_cards) - set(existing.keys())
    print(f"Cards MISSING templates ({len(missing)}): {sorted(missing)}")

    candidates = {}      # card_code -> [(surface, filename, rank_score, suit_score), ...]
    suit_patches = {}    # suit_letter -> [(patch_bgr, filename, card_idx), ...]

    pre_files = sorted([
        f for f in SCREENSHOTS_DIR.iterdir()
        if f.name.startswith(NEW_PREFIX) and "_PRE." in f.name and f.suffix == ".png"
    ])
    print(f"\nProcessing {len(pre_files)} new PRE screenshots...")

    for shot_file in pre_files:
        img = cv2.imread(str(shot_file))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]
        ref_w, ref_h = 1935, 1369
        sx, sy = w_img / ref_w, h_img / ref_h

        for card_idx, (rx, ry, rw, rh) in enumerate(hero_rois[:2]):
            x1, y1 = int(rx * sx), int(ry * sy)
            x2, y2 = int((rx + rw) * sx), int((ry + rh) * sy)
            roi_raw = img[y1:y2, x1:x2]
            if roi_raw is None or roi_raw.size == 0:
                continue

            # Use the detector's hero surface extractor
            surface = det._extract_hero_card_surface(roi_raw, f"hero{card_idx + 1}")
            if surface is None or surface.size == 0:
                surface = roi_raw
            if not is_card_visible(surface):
                continue

            # Extract suit ROI for S/C disambiguation
            _, suit_roi = det._extract_corner_regions(surface, context="hero")

            rank, rank_score = detect_rank_hero(surface, det)
            suit, suit_score = detect_suit_by_color(surface, suit_roi, det)

            if not rank or not suit:
                continue

            card_code = f"{rank}{suit}"
            candidates.setdefault(card_code, []).append(
                (surface.copy(), shot_file.name, rank_score, suit_score)
            )
            if suit_roi is not None and suit_roi.size > 0:
                suit_patches.setdefault(suit, []).append(
                    (suit_roi.copy(), shot_file.name, card_idx + 1, suit_score)
                )

    # Report
    print(f"\nDetected {len(candidates)} distinct card codes:")
    for code in sorted(candidates.keys()):
        surfs = candidates[code]
        flag = "EXISTING" if code in existing else "MISSING"
        best = max(s[2] + s[3] for s in surfs)
        print(f"  {code}: {len(surfs)} samples, best score={best:.2f}  [{flag}]")

    # Save templates for missing cards
    new_card_count = 0
    for card_code in sorted(missing):
        if card_code not in candidates:
            continue
        surfs = sorted(candidates[card_code], key=lambda x: x[2] + x[3], reverse=True)[:3]
        for i, (surface, fname, rs, ss) in enumerate(surfs):
            resized = cv2.resize(surface, (92, 136), interpolation=cv2.INTER_AREA)
            ts = re.sub(r'_PRE$', '', fname.replace(".png", ""))
            out_name = f"{card_code}__new_{ts}_{i + 1}.png"
            cv2.imwrite(str(OUTPUT_CARD_DIR / out_name), resized)
            print(f"  SAVED {out_name}  (rank={rs:.2f}, suit={ss:.2f})")
            new_card_count += 1

    # Save new D suit patches (old ones are wrong style)
    new_suit_count = 0
    for suit_letter in sorted(suit_patches.keys()):
        patches = sorted(suit_patches[suit_letter], key=lambda x: x[3], reverse=True)
        saved = 0
        for patch, fname, cidx, score in patches:
            if saved >= 5:
                break
            patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            if float(np.std(patch_gray)) < 8:
                continue
            resized = cv2.resize(patch, (24, 32), interpolation=cv2.INTER_AREA)
            ts = re.sub(r'_PRE$', '', fname.replace(".png", ""))
            out_name = f"{suit_letter}__new_{ts}_hero{cidx}.png"
            cv2.imwrite(str(OUTPUT_SUIT_DIR / out_name), resized)
            saved += 1
            new_suit_count += 1

    print("\n=== Summary ===")
    print(f"New card templates saved: {new_card_count}")
    print(f"New suit templates saved: {new_suit_count}")
    still_missing = sorted(missing - set(candidates.keys()))
    print(f"Still missing ({len(still_missing)}): {still_missing}")

    # Debug grid
    debug_items = []
    for code in sorted(candidates.keys()):
        s = candidates[code][0][0]
        s_small = cv2.resize(s, (60, 90))
        labeled = cv2.copyMakeBorder(s_small, 18, 0, 0, 0, cv2.BORDER_CONSTANT, value=(200, 200, 200))
        cv2.putText(labeled, code, (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 200) if code in missing else (0, 0, 0), 1)
        debug_items.append(labeled)

    if debug_items:
        rows = []
        for i in range(0, len(debug_items), 13):
            row = debug_items[i:i + 13]
            while len(row) < 13:
                row.append(np.ones((108, 60, 3), dtype=np.uint8) * 180)
            rows.append(np.hstack(row))
        cv2.imwrite(str(DEBUG_DIR / "detected_cards_overview.png"), np.vstack(rows))
        print(f"Debug grid: {DEBUG_DIR}/detected_cards_overview.png")


if __name__ == "__main__":
    main()
