"""
Extract hero card templates for missing card codes from new-style screenshots.
Uses the same surface extraction logic as the detector (old fixed-percentage code).
"""
import os
import re
import sys
from collections import defaultdict

import cv2

sys.path.insert(0, '.')
from detectors.card_detector import CardDetector  # noqa: E402
from config import TABLE_LAYOUTS  # noqa: E402

det = CardDetector()
det._apply_layout_profile('acipayam_heads_up')
rois = TABLE_LAYOUTS['acipayam_heads_up']['hero_hole_cards']
REF_W, REF_H = 1935, 1369

TMPL_DIR = 'assets/hero_card_templates_v3'
FOLDER = 'auto_screenshots'

# --- Find existing coverage ---
existing = set()
for f in os.listdir(TMPL_DIR):
    if f.endswith('.png'):
        existing.add(f.split('__')[0])

# --- Build card_code -> [(fname, card_index)] map from new-style screenshots ---
# New-style screenshots have timestamps starting with 177713... or 177748...
card_to_shots = defaultdict(list)
NEW_STYLE_PREFIXES = ('177713', '177748', '177749')
for fname in sorted(os.listdir(FOLDER)):
    m = re.match(r'(\d+)_([A-Z0-9]+)_PRE\.png', fname)
    if not m:
        continue
    ts = m.group(1)
    if not any(ts.startswith(p) for p in NEW_STYLE_PREFIXES):
        continue
    cards_str = m.group(2)
    for i in range(0, len(cards_str), 2):
        code = cards_str[i:i+2]
        ci = i // 2  # 0=hero1, 1=hero2
        card_to_shots[code].append((fname, ci))

# Extract templates for ALL cards in new-style screenshots that don't yet have
# a new-style template (filename contains '__new_').
new_style_existing = set()
for f in os.listdir(TMPL_DIR):
    if '__new_' in f and f.endswith('.png'):
        new_style_existing.add(f.split('__')[0])

missing = sorted(code for code in card_to_shots if code not in new_style_existing)
print(f'Cards needing new-style templates: {missing}')

saved = 0
for code in missing:
    shots = card_to_shots[code]
    # Save up to 3 templates per card code
    for shot_idx, (fname, ci) in enumerate(shots[:3]):
        img = cv2.imread(os.path.join(FOLDER, fname))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]
        sx, sy = w_img / REF_W, h_img / REF_H

        rx, ry, rw, rh = rois[ci]
        x1, y1 = int(rx * sx), int(ry * sy)
        x2, y2 = int((rx + rw) * sx), int((ry + rh) * sy)
        roi_raw = img[y1:y2, x1:x2]

        context = f'hero{ci + 1}'
        surface = det._extract_hero_card_surface(roi_raw, context)
        if surface is None or surface.size == 0:
            print(f'  SKIP {code} from {fname} card{ci+1}: no surface')
            continue

        # Verify the surface looks like a card (must have enough white pixels)
        import numpy as np
        gray = cv2.cvtColor(surface, cv2.COLOR_BGR2GRAY)
        white_ratio = float(np.mean(gray > 200))
        if white_ratio < 0.15:
            print(f'  SKIP {code} from {fname} card{ci+1}: too dark (white_ratio={white_ratio:.2f})')
            continue

        # Name: CODE__new_TIMESTAMP_HEROSTR_CARDIDX.png
        ts = re.match(r'(\d+)_', fname).group(1)
        hero_str = re.match(r'\d+_([A-Z0-9]+)_PRE', fname).group(1)
        out_name = f'{code}__new_{ts}_{hero_str}_{shot_idx+1}.png'
        out_path = os.path.join(TMPL_DIR, out_name)
        cv2.imwrite(out_path, surface)
        print(f'  Saved {out_name}  shape={surface.shape}  white={white_ratio:.2f}')
        saved += 1

print(f'\nTotal saved: {saved} templates for {len(missing)} missing codes')
