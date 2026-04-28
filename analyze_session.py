import json
import sys

fname = sys.argv[1] if len(sys.argv) > 1 else "logs/hands_20260428_135634.jsonl"
hands = []
with open(fname) as f:
    for line in f:
        hands.append(json.loads(line))

for i, h in enumerate(hands):
    hole = h.get("hole_cards", [])
    board = h.get("final_board", [])
    delta = h.get("hero_stack_delta", 0)
    outcome = h.get("outcome", "?")
    snaps = h.get("snapshots", [])
    print(f"\n=== Hand {i+1}: {hole} | Board: {board} | {outcome} | Delta: {delta:+.2f} ===")
    for s in snaps:
        street = s.get("street", "?")
        pot = s.get("pot_size", 0)
        to_call = s.get("to_call", 0)
        rec = s.get("recommended_action", "-")
        conf = s.get("confidence", 0)
        avail = s.get("available_actions", [])
        reason = s.get("reason", "")[:100]
        equity = s.get("mc_hand_equity", 0)
        print(f"  {street:8s} pot={pot:.2f} to_call={to_call:.2f} avail={avail} rec={rec}({conf:.0%}) eq={equity:.1%} | {reason}")
