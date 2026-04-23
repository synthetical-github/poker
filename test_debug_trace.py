"""Trace exactly how roi_debug/Screenshot_27.png detects board cards."""
import cv2
import sys
import logging
sys.path.insert(0, '.')

# Enable DEBUG logging just for card_detector
logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s'
)
card_det_logger = logging.getLogger('detectors.card_detector')
card_det_logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter('%(message)s'))
card_det_logger.addHandler(handler)
card_det_logger.propagate = False

from detectors.card_detector import CardDetector

detector = CardDetector()
print(f"Layout: {detector.layout_name}")
print()

# Test roi_debug/Screenshot_27.png
ss = cv2.imread('roi_debug/Screenshot_27.png')
if ss is not None:
    h, w = ss.shape[:2]
    result = detector._get_frame_detection(ss, (0, 0, w, h))
    hole = [str(c) for c in result.get('hole_cards', [])]
    board = [str(c) for c in result.get('community_cards', [])]
    print(f"\n=== RESULT: hole={hole}, board={board}")
else:
    print("Screenshot_27.png not found")
