"""
Überprüft die Kartenerkennung anhand der automatischen Screenshots in auto_screenshots/.
Der Dateiname kodiert die erwarteten Karten: <ts>_<hole>_<board>.png
  - Hole: z.B. QS8H → QS, 8H
  - Board: z.B. 7SJCAC → 7S, JC, AC  (oder PRE für kein Board)
"""
import re
import sys
from pathlib import Path

import cv2

from detectors.card_detector import CardDetector

FOLDER = Path("auto_screenshots")
CARD_RE = re.compile(r'([2-9TJQKA][SHDC])')


def parse_filename(name: str):
    """Gibt (expected_hole, expected_board) als Listen von Strings zurück."""
    stem = Path(name).stem  # z.B. 1777490806_QS8H_7SJCAC
    parts = stem.split("_")
    if len(parts) < 2:
        return [], []
    hole_str = parts[1]
    board_str = parts[2] if len(parts) > 2 else "PRE"

    hole = CARD_RE.findall(hole_str)
    board = [] if board_str == "PRE" else CARD_RE.findall(board_str)
    return hole, board


def cards_to_strs(cards) -> list:
    return [str(c).upper() for c in cards if c is not None]


def normalize(card_str: str) -> str:
    """Normalisiere Karten-String für Vergleich (z.B. '9s' → '9S')."""
    return card_str.strip().upper()


def main():
    det = CardDetector()
    det._apply_layout_profile("acipayam_heads_up")

    files = sorted(FOLDER.glob("*.png"))
    if not files:
        print(f"Keine PNG-Dateien in {FOLDER} gefunden.")
        sys.exit(1)

    total = 0
    hole_ok = 0
    hole_fail = 0
    board_ok = 0
    board_fail = 0
    failures = []

    for img_path in files:
        expected_hole, expected_board = parse_filename(img_path.name)
        if not expected_hole:
            print(f"SKIP (kein Muster): {img_path.name}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"SKIP (nicht ladbar): {img_path.name}")
            continue

        h, w = img.shape[:2]
        tc = (0, 0, w, h)

        detected_hole = cards_to_strs(det.detect_hole_cards(img, tc))
        detected_board = cards_to_strs(det.detect_community_cards(img, tc))

        exp_hole_norm = [normalize(c) for c in expected_hole]
        exp_board_norm = [normalize(c) for c in expected_board]
        det_hole_norm = [normalize(c) for c in detected_hole]
        det_board_norm = [normalize(c) for c in detected_board]

        hole_match = sorted(det_hole_norm) == sorted(exp_hole_norm)
        board_match = sorted(det_board_norm) == sorted(exp_board_norm)

        total += 1
        if hole_match:
            hole_ok += 1
        else:
            hole_fail += 1
        if board_match:
            board_ok += 1
        else:
            board_fail += 1

        status = "OK" if (hole_match and board_match) else "FAIL"
        if not (hole_match and board_match):
            failures.append({
                "file": img_path.name,
                "exp_hole": exp_hole_norm,
                "det_hole": det_hole_norm,
                "exp_board": exp_board_norm,
                "det_board": det_board_norm,
            })

        hole_info = (
            f"Hole OK ({' '.join(det_hole_norm)})"
            if hole_match
            else f"Hole FAIL exp={exp_hole_norm} det={det_hole_norm}"
        )
        board_info = (
            f"Board OK ({' '.join(det_board_norm) or 'leer'})"
            if board_match
            else f"Board FAIL exp={exp_board_norm} det={det_board_norm}"
        )
        print(f"[{status}] {img_path.name}")
        print(f"       {hole_info}")
        print(f"       {board_info}")

    print("\n" + "=" * 60)
    print(f"Gesamt: {total} Screenshots")
    print(f"Hole Cards:  {hole_ok}/{total} korrekt  ({hole_fail} Fehler)")
    print(f"Board Cards: {board_ok}/{total} korrekt  ({board_fail} Fehler)")

    if failures:
        print(f"\n--- {len(failures)} fehlgeschlagene Screenshots ---")
        for f in failures:
            print(f"  {f['file']}")
            if sorted(f['det_hole']) != sorted(f['exp_hole']):
                print(f"    Hole  erwartet={f['exp_hole']}  erkannt={f['det_hole']}")
            if sorted(f['det_board']) != sorted(f['exp_board']):
                print(f"    Board erwartet={f['exp_board']}  erkannt={f['det_board']}")
    else:
        print("\nAlle Karten korrekt erkannt!")


if __name__ == "__main__":
    main()
