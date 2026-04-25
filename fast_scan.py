"""
Schneller Scan: Nur Hero-Karten-Template-Matching ohne Tesseract OCR.
Zeigt welche Hero-Karten erkannt werden und welche None sind.
"""
from collections import Counter

import cv2
from detectors.card_detector import CardDetector
from utils.config import TABLE_LAYOUTS

det = CardDetector()
det._apply_layout_profile('acipayam_heads_up')

folder = "C:/poker-1/Neuer Ordner"
layout = TABLE_LAYOUTS["acipayam_heads_up"]
ref_w, ref_h = layout["reference_size"]

hero_results = []
none_positions = []

for i in range(1, 53):
    fname = f"Screenshot_{i}.png"
    img = cv2.imread(f"{folder}/{fname}")
    if img is None:
        continue
    h, w = img.shape[:2]
    sx = w / ref_w
    sy = h / ref_h

    row = []
    for ci, roi in enumerate(layout["hero_hole_cards"]):
        rx, ry, rw, rh = (
            int(roi[0] * sx),
            int(roi[1] * sy),
            int(roi[2] * sx),
            int(roi[3] * sy),
        )
        crop = img[ry : ry + rh, rx : rx + rw]
        if crop is None or crop.size == 0:
            row.append("ERR")
            continue

        ctx = f"hero{ci + 1}"
        surf = det._extract_hero_card_surface(crop, ctx)
        if surf is None or surf.size == 0:
            row.append("NO_SURF")
            continue

        gray = cv2.cvtColor(surf, cv2.COLOR_BGR2GRAY)

        scores = {}
        for name, tmpl_list in det.hero_card_templates.items():
            if isinstance(tmpl_list, list):
                s = max(det._score_card_template(gray, t) for t in tmpl_list)
            else:
                s = det._score_card_template(gray, tmpl_list)
            scores[name] = s

        top2 = sorted(scores.items(), key=lambda x: -x[1])[:2]
        best_name, best_score = top2[0]
        gap = best_score - top2[1][1] if len(top2) > 1 else 1.0

        if best_score >= 0.80 and gap >= 0.02:
            row.append(best_name)
        else:
            row.append(f"?({best_name}:{best_score:.2f},gap:{gap:.2f})")
            none_positions.append((i, ci + 1, best_name, best_score, gap))

    hero_results.append((i, row))

print("=== Hero Results per Screenshot ===")
for i, row in hero_results:
    print(f"  SS_{i:02d}: {row[0]:15s}  {row[1]:15s}")

print("\n=== Not recognized (score<0.80 or gap<0.02) ===")
for ss, pos, name, score, gap in none_positions:
    print(
        f"  SS_{ss:02d} hero{pos}: best={name} "
        f"score={score:.3f} gap={gap:.3f}"
    )

all_detected = [
    c
    for _, row in hero_results
    for c in row
    if not c.startswith("?") and c not in ("ERR", "NO_SURF")
]
print(f"\n=== Coverage: {Counter(sorted(all_detected))} ===")
