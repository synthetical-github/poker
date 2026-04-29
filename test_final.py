import time
import cv2
from detectors.card_detector import CardDetector

tests = [
    ('1777490928_4CTH_PRE.png',     '4C+TH',  ''),
    ('1777490782_QS8H_PRE.png',     '8H+QS',  ''),
    ('1777491258_4H4C_PRE.png',     '4C+4H',  ''),
    ('1777491365_TSTD_PRE.png',     'TD+TS',  ''),
    ('1777491033_TSTC_PRE.png',     'TC+TS',  ''),
    ('1777490994_6CAD_PRE.png',     '6C+AD',  ''),
    ('1777491110_6SAD_TCTH4C.png',  '6S+AD',  '4C+TC+TH'),
]

det = CardDetector()
for fname, exp_h, exp_b in tests:
    img = cv2.imread('auto_screenshots/' + fname)
    h, w = img.shape[:2]
    t0 = time.perf_counter()
    hole = det.detect_hole_cards(img, (0, 0, w, h))
    board = det.detect_community_cards(img, (0, 0, w, h))
    ms = (time.perf_counter()-t0)*1000
    got_h = '+'.join(sorted(str(c) for c in hole))
    got_b = '+'.join(sorted(str(c) for c in board))
    ok_h = 'OK' if got_h == exp_h else 'FAIL(exp='+exp_h+')'
    ok_b = '' if not exp_b else ('OK' if got_b == exp_b else 'FAIL(exp='+exp_b+')')
    print(f'{fname.split("_")[1]}: {ms:.0f}ms  hole={got_h} {ok_h}  board={got_b} {ok_b}')
