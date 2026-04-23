import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2

from live_analyzer import LivePokerAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spielt Screenshot-Serien durch die Live-Analyse-Logik ab."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="roi_debug",
        help="Datei oder Ordner mit Screenshots",
    )
    parser.add_argument(
        "--pattern",
        default="Screenshot*.png",
        help="Glob-Muster für Screenshots in einem Ordner",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximale Anzahl Bilder, 0 = alle",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optionaler Pfad für einen JSON-Bericht",
    )
    return parser.parse_args()


def collect_image_files(input_path: Path, pattern: str, limit: int) -> List[Path]:
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(path for path in input_path.rglob(pattern) if path.is_file())
    return files[:limit] if limit > 0 else files


def build_record(path: Path, result: Dict[str, Any], elapsed_ms: float) -> Dict[str, Any]:
    game_state = result.get("game_state", {})
    strategy = result.get("strategy", {})
    return {
        "file": str(path),
        "hole_cards": strategy.get("hole_cards", []),
        "community_cards": strategy.get("community_cards", []),
        "street": game_state.get("street", "unknown"),
        "pot_size": game_state.get("pot_size", 0.0),
        "to_call": game_state.get("to_call", 0.0),
        "available_actions": game_state.get("available_actions", []),
        "recommended_action": strategy.get("recommended_action"),
        "amount": strategy.get("amount", 0.0),
        "confidence": strategy.get("confidence", 0.0),
        "reason": strategy.get("reason", ""),
        "hand_category": strategy.get("hand_details", {}).get("display_category")
        or strategy.get("hand_details", {}).get("category"),
        "board_texture": strategy.get("board_texture", {}).get("texture"),
        "draws": strategy.get("draws", {}),
        "elapsed_ms": round(elapsed_ms, 2),
    }


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Eingabepfad nicht gefunden: {input_path}")
        return 1

    files = collect_image_files(input_path, args.pattern, args.limit)
    if not files:
        print(f"Keine Screenshots gefunden unter {input_path} mit Muster {args.pattern}")
        return 1

    records: List[Dict[str, Any]] = []

    for index, image_path in enumerate(files, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[{index}/{len(files)}] Übersprungen: {image_path} konnte nicht geladen werden")
            continue

        analyzer = LivePokerAnalyzer(headless=True)
        start_time = time.perf_counter()
        result = analyzer.analyze_screenshot(image)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if not result:
            print(f"[{index}/{len(files)}] {image_path.name} | kein Ergebnis | {elapsed_ms:.2f} ms")
            continue

        record = build_record(image_path, result, elapsed_ms)
        records.append(record)
        print(
            f"[{index}/{len(files)}] {image_path.name} | "
            f"Hole={' '.join(record['hole_cards']) or '-'} | "
            f"Board={' '.join(record['community_cards']) or '-'} | "
            f"Action={record['recommended_action']} {record['amount']:.2f} | "
            f"{record['elapsed_ms']:.2f} ms"
        )

    if records:
        avg_ms = sum(record["elapsed_ms"] for record in records) / len(records)
        print(f"\nAnalysierte Frames: {len(records)} | Durchschnitt: {avg_ms:.2f} ms")
    else:
        print("\nKeine auswertbaren Frames erzeugt.")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"JSON-Bericht geschrieben: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
