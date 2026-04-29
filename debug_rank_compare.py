import os
import sys

import cv2

sys.path.insert(0, '.')
from detectors.card_detector import CardDetector  # noqa: E402
from config import TABLE_LAYOUTS  # noqa: E402

det = CardDetector()
det._apply_layout_profile('acipayam_heads_up')
rois = TABLE_LAYOUTS['acipayam_heads_up']['hero_hole_cards']

shots = [
    ('auto_screenshots/1777041235_6CKD_PRE.png', 'OLD_6C_KD'),
    ('auto_screenshots/1777481943_KD4C_PRE.png', 'NEW_KD_4C'),
]

os.makedirs('debug_rank_size_compare', exist_ok=True)

for fname, label in shots:
    img = cv2.imread(fname)
    if img is None:
        continue
    h_img, w_img = img.shape[:2]
    sx, sy = w_img / 1935, h_img / 1369
    for ci, (rx, ry, rw, rh) in enumerate(rois[:2]):
        x1, y1 = int(rx * sx), int(ry * sy)
        x2, y2 = int((rx + rw) * sx), int((ry + rh) * sy)
        roi_raw = img[y1:y2, x1:x2]
        surface = det._extract_hero_card_surface(roi_raw, f'hero{ci+1}')
        if surface is None:
            surface = roi_raw
        rank_roi, _suit_roi = det._extract_corner_regions(surface, context='hero')
        rank_shape = rank_roi.shape if rank_roi is not None else None
        print(f'{label} card{ci+1}: surface={surface.shape}  rank_roi={rank_shape}')
        if rank_roi is not None:
            cv2.imwrite(f'debug_rank_size_compare/{label}_card{ci+1}_rank.png', rank_roi)
            rg = cv2.cvtColor(rank_roi, cv2.COLOR_BGR2GRAY)
            norm = det._normalize_rank_image(rg)
            print(f'  normalized: {norm.shape}')
            cv2.imwrite(f'debug_rank_size_compare/{label}_card{ci+1}_norm.png', norm)
            best, score, second = det._get_symbol_template_stats(rg, det.hero_rank_templates)
            print(f'  best match: {best} score={score:.3f} gap={score-second:.3f}')
