from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from detectors.card_detector import CardDetector


SLOT_NAMES = ("hero1", "hero2", "board1", "board2", "board3", "board4", "board5")


def _get_slot_region(detector: CardDetector, image, slot: str):
    table_coords = (0, 0, image.shape[1], image.shape[0])
    hero_regions = detector.get_hole_card_regions(table_coords)
    board_regions = detector.get_community_card_regions(table_coords)
    mapping = {
        "hero1": hero_regions[0],
        "hero2": hero_regions[1],
        "board1": board_regions[0],
        "board2": board_regions[1],
        "board3": board_regions[2],
        "board4": board_regions[3],
        "board5": board_regions[4],
    }
    return mapping[slot]


def _save_image(path: Path, image):
    if image is not None and getattr(image, "size", 0) > 0:
        cv2.imwrite(str(path), image)


def main():
    parser = argparse.ArgumentParser(description="Debuggt eine einzelne Karten-ROI fuer den CardDetector.")
    parser.add_argument("--image", required=True, help="Pfad zum Screenshot")
    parser.add_argument("--slot", required=True, choices=SLOT_NAMES, help="hero1/hero2/board1..board5")
    parser.add_argument(
        "--out-dir",
        default=str(Path("debug_card_roi_output")),
        help="Zielordner fuer Debug-Bilder",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Konnte Bild nicht laden: {image_path}")

    detector = CardDetector()
    region = _get_slot_region(detector, image, args.slot)
    x, y, w, h = region
    roi = image[y:y + h, x:x + w]
    analysis = detector.debug_analyze_card_roi(roi, context=args.slot)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    _save_image(out_dir / f"{stem}_{args.slot}_roi.png", roi)
    _save_image(out_dir / f"{stem}_{args.slot}_surface.png", analysis["surface"])
    _save_image(out_dir / f"{stem}_{args.slot}_rank.png", analysis["rank_patch"])
    _save_image(out_dir / f"{stem}_{args.slot}_suit.png", analysis["suit_patch"])

    print(f"image={image_path}")
    print(f"slot={args.slot}")
    print(f"region={region}")
    print(f"roi_shape={analysis['roi_shape']}")
    print(f"surface_shape={analysis['surface_shape']}")
    print(f"corner_rank={analysis['corner_rank']}")
    print(f"corner_suit={analysis['corner_suit']}")
    print(f"corner_card={analysis['corner_card']}")
    print(f"rank_top_matches={[(label, round(score, 3)) for label, score in analysis['rank_top_matches']]}")
    print(f"suit_top_matches={[(label, round(score, 3)) for label, score in analysis['suit_top_matches']]}")
    print(f"card_top_matches={[(label, round(score, 3)) for label, score in analysis['card_top_matches']]}")
    print(f"saved={out_dir}")


if __name__ == "__main__":
    main()
