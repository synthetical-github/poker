import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from utils.card_utils import Card
from utils.poker_decision import starting_hand_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wertet geloggte Hand-Empfehlungen aus.")
    parser.add_argument("logfile", nargs="?", help="Pfad zu einer hands_*.jsonl Datei")
    parser.add_argument("--top", type=int, default=10, help="Anzahl auffaelliger Haende")
    return parser.parse_args()


def latest_hands_log(logs_dir: Path) -> Path | None:
    candidates = sorted(logs_dir.glob("hands_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_hands(path: Path) -> List[Dict[str, Any]]:
    hands: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            hands.append(json.loads(line))
    return hands


def summarize(hands: List[Dict[str, Any]]) -> None:
    completed = [hand for hand in hands if hand.get("outcome") in {"won", "lost", "breakeven"}]
    won = sum(1 for hand in completed if hand.get("outcome") == "won")
    lost = sum(1 for hand in completed if hand.get("outcome") == "lost")
    breakeven = sum(1 for hand in completed if hand.get("outcome") == "breakeven")
    avg_delta = 0.0
    known_deltas = [float(hand["hero_stack_delta"]) for hand in completed if hand.get("hero_stack_delta") is not None]
    if known_deltas:
        avg_delta = sum(known_deltas) / len(known_deltas)

    print(f"Hands total: {len(hands)}")
    print(f"Hands with known outcome: {len(completed)}")
    print(f"Won: {won} | Lost: {lost} | Breakeven: {breakeven}")
    print(f"Average stack delta: {avg_delta:.2f}")


def _parse_hand_key(hole_cards: List[str]) -> str:
    cards: List[Card] = []
    for raw_card in hole_cards or []:
        raw_text = str(raw_card or "").strip().upper()
        if len(raw_text) < 2:
            continue
        rank = "T" if raw_text.startswith("10") else raw_text[0]
        cards.append(Card(rank, raw_text[-1]))
    return starting_hand_key(cards)


def _extract_score(reason: str) -> float | None:
    match = re.search(r"score=(\d+(?:\.\d+)?)", str(reason or ""))
    if not match:
        return None
    return float(match.group(1))


def _reasonable_hu_open(hand_key: str) -> bool:
    if not hand_key:
        return False
    if len(hand_key) == 2:
        return True
    high_rank = hand_key[0]
    low_rank = hand_key[1]
    suited = hand_key.endswith("s")
    if high_rank == "A":
        return True
    if high_rank == "K":
        return low_rank in "56789TJQ" or suited
    if high_rank == "Q":
        return low_rank in "789TJ" or suited
    if high_rank == "J":
        return low_rank in "89TQ" or suited
    if high_rank == "T":
        return low_rank in "789JQ" or (suited and low_rank in "567")
    if suited and hand_key[:2] in {"98", "97", "87", "86", "76", "75", "65", "64", "54"}:
        return True
    return False


def _reasonable_hu_defend(hand_key: str) -> bool:
    if not hand_key:
        return False
    if len(hand_key) == 2:
        return True
    high_rank = hand_key[0]
    low_rank = hand_key[1]
    suited = hand_key.endswith("s")
    if high_rank == "A":
        return True
    if high_rank == "K":
        return low_rank in "789TJQ" or (suited and low_rank in "456")
    if high_rank == "Q":
        return low_rank in "9TJ" or (suited and low_rank in "678")
    if high_rank == "J":
        return low_rank in "89TQ" or (suited and low_rank in "67")
    if high_rank == "T":
        return low_rank in "89JQ" or (suited and low_rank in "78")
    if suited and hand_key[:2] in {"98", "97", "87", "86", "76", "75", "65", "64", "54"}:
        return True
    return False


def _clear_hu_mistake(hand_key: str, action: str, reason: str, score: float | None) -> bool:
    reason_text = str(reason or "")
    if "HU " not in reason_text:
        return False
    if action in {"raise", "bet"}:
        return not _reasonable_hu_open(hand_key) and (score is None or score < 0.60)
    if action == "call":
        return not _reasonable_hu_defend(hand_key) and (score is None or score < 0.56)
    return False


def _clear_hu_bad_fold(hand_key: str, score: float | None, reason: str) -> bool:
    if "HU " not in str(reason or ""):
        return False
    return _reasonable_hu_open(hand_key) or (score is not None and score >= 0.55)


def find_questionable_hands(hands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    for hand in hands:
        outcome = hand.get("outcome")
        delta = hand.get("hero_stack_delta")
        recommendations = hand.get("recommendations", [])
        if outcome not in {"won", "lost"} or delta is None or not recommendations:
            continue

        last = recommendations[-1]
        action = str(last.get("action", ""))
        confidence = float(last.get("confidence", 0.0) or 0.0)
        hand_category = str(last.get("hand_category", "Unknown"))
        reason = str(last.get("reason", ""))
        hand_key = _parse_hand_key(hand.get("hole_cards", []))
        score = _extract_score(reason)

        suspicious = False
        tag = ""
        if (
            outcome == "lost"
            and confidence >= 0.76
            and action in {"call", "raise", "bet"}
            and hand_category in {"High Card", "Preflop"}
            and _clear_hu_mistake(hand_key, action, reason, score)
        ):
            suspicious = True
            tag = "too_loose"
        elif outcome == "won" and action == "fold" and _clear_hu_bad_fold(hand_key, score, reason):
            suspicious = True
            tag = "possible_bad_fold"
        elif outcome == "lost" and action == "raise" and confidence >= 0.82 and _clear_hu_mistake(hand_key, action, reason, score):
            suspicious = True
            tag = "aggressive_loss"

        if suspicious:
            findings.append(
                {
                    "hand_id": hand.get("hand_id"),
                    "outcome": outcome,
                    "delta": float(delta),
                    "hole_cards": hand.get("hole_cards", []),
                    "final_board": hand.get("final_board", []),
                    "final_hand_category": hand.get("final_hand_category"),
                    "hand_key": hand_key,
                    "action": action,
                    "confidence": confidence,
                    "reason": reason,
                    "tag": tag,
                }
            )

    findings.sort(key=lambda item: (abs(item["delta"]), item["confidence"]), reverse=True)
    return findings


def main() -> int:
    args = parse_args()
    logs_dir = Path(__file__).resolve().parent / "logs"
    path = Path(args.logfile) if args.logfile else latest_hands_log(logs_dir)
    if path is None or not path.exists():
        print("Keine Hands-Logdatei gefunden.")
        return 1

    hands = load_hands(path)
    print(f"Logfile: {path}")
    summarize(hands)

    findings = find_questionable_hands(hands)
    print("")
    print("Auffaellige Empfehlungen:")
    for finding in findings[: args.top]:
        print(
            f"- {finding['hand_id']} | {finding['tag']} | outcome={finding['outcome']} | "
            f"delta={finding['delta']:.2f} | hole={' '.join(finding['hole_cards'])} | "
            f"board={' '.join(finding['final_board'])} | action={finding['action']} | "
            f"conf={finding['confidence']:.2f} | hand={finding['final_hand_category']} | {finding['reason']}"
        )
    if not findings:
        print("- Keine klar auffaelligen Empfehlungen gefunden.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
