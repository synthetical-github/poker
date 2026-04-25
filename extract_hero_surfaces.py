"""
Extrahiert Hero-Surfaces aller 52 Screenshots in einen Debug-Ordner.
Dateinamen: hero1_SS{n}.png, hero2_SS{n}.png
So kann man visuell die Karten identifizieren.
"""
import os

import cv2

from detectors.card_detector import CardDetector
from utils.config import TABLE_LAYOUTS

det = CardDetector()
det._apply_layout_profile('acipayam_heads_up')

folder = "C:/poker-1/Neuer Ordner"
layout = TABLE_LAYOUTS["acipayam_heads_up"]
ref_w, ref_h = layout["reference_size"]
out = "hero_surfaces_debug"
os.makedirs(out, exist_ok=True)

for i in range(1, 53):
    fname = f"Screenshot_{i}.png"
    img = cv2.imread(f"{folder}/{fname}")
    if img is None:
        continue
    h, w = img.shape[:2]
    sx = w / ref_w
    sy = h / ref_h

    for ci, roi in enumerate(layout["hero_hole_cards"]):
        rx, ry, rw, rh = (
            int(roi[0] * sx),
            int(roi[1] * sy),
            int(roi[2] * sx),
            int(roi[3] * sy),
        )
        crop = img[ry : ry + rh, rx : rx + rw]
        if crop is None or crop.size == 0:
            continue
        ctx = f"hero{ci + 1}"
        surf = det._extract_hero_card_surface(crop, ctx)
        if surf is None or surf.size == 0:
            surf = crop
        cv2.imwrite(f"{out}/hero{ci + 1}_SS{i:02d}.png", surf)

print(f"Saved hero surfaces for 52 screenshots to {out}/")
