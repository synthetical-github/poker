"""Replace board templates with extracted card surfaces.

Run once after adding new board ROI crops.
"""
import os
import sys

import cv2

sys.path.insert(0, '.')
from detectors.card_detector import CardDetector
from utils.config import TABLE_LAYOUTS

det = CardDetector()
det._apply_layout_profile('acipayam_heads_up')

out_board = "assets/board_card_templates_v3"
layout = TABLE_LAYOUTS["acipayam_heads_up"]

sources = [
    ("Screenshot_30.png", ["2H", "8H", "JD", "AD", "9C"]),
    ("Screenshot_35.png", ["QC", "TH", "AH", "8S", "TS"]),
    ("Screenshot_15.png", ["JH", "QD", "5H", "7S", "TS"]),
]

for fname, board_cards in sources:
    img = cv2.imread(f"C:/poker-1/Neuer Ordner/{fname}")
    h, w = img.shape[:2]
    ref_w, ref_h = layout["reference_size"]
    sx = w / ref_w
    sy = h / ref_h
    base = os.path.splitext(fname)[0]

    for i, card_name in enumerate(board_cards):
        roi = layout["community_cards"][i]
        x, y, cw, ch = (
            int(roi[0] * sx),
            int(roi[1] * sy),
            int(roi[2] * sx),
            int(roi[3] * sy),
        )
        crop = img[y : y + ch, x : x + cw]

        # Extract card surface (same as detector does internally)
        surface = det._extract_card_surface(crop)
        if surface is None or surface.size == 0:
            surface = crop

        # Remove old full-ROI file if present
        for suffix in [f"_board{i}.png", f"_surf{i}.png"]:
            old = f"{out_board}/{card_name}__{base}{suffix}"
            if os.path.exists(old):
                os.remove(old)

        out = f"{out_board}/{card_name}__{base}_surf{i}.png"
        cv2.imwrite(out, surface)
        print(
            f"Saved board surface ({surface.shape[1]}x{surface.shape[0]}): {out}"
        )

print("Done.")
