from __future__ import annotations

from typing import Any, Dict, List, Optional


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


class SubgameSolver:
    def solve(
        self,
        game_state: Dict[str, Any],
        monte_carlo_decision: Dict[str, Any],
        action_bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        street = str(game_state.get("street", "preflop") or "preflop").lower()
        if street == "preflop":
            return {
                "enabled": False,
                "street": street,
                "recommended_action_id": None,
                "recommended_action": None,
                "recommended_amount": 0.0,
                "confidence": 0.0,
                "reason": "solver_preflop_disabled",
                "action_scores": {},
            }

        opponent_profile = monte_carlo_decision.get("opponent_profile", {}) or {}
        board_texture = monte_carlo_decision.get("board_texture", {}) or {}
        draws = monte_carlo_decision.get("draws", {}) or {}
        hand_details = monte_carlo_decision.get("hand_details", {}) or {}
        action_evs = monte_carlo_decision.get("action_evs", {}) or {}
        actions = action_bundle.get("actions", [])

        action_scores: Dict[str, float] = {}
        for action in actions:
            action_id = str(action["id"])
            score = float(action_evs.get(action_id, 0.0))
            score += self._heuristic_adjustment(
                street=street,
                action=action,
                board_texture=board_texture,
                draws=draws,
                hand_details=hand_details,
                opponent_profile=opponent_profile,
            )
            action_scores[action_id] = round(score, 4)

        if not action_scores:
            return {
                "enabled": True,
                "street": street,
                "recommended_action_id": None,
                "recommended_action": None,
                "recommended_amount": 0.0,
                "confidence": 0.0,
                "reason": "solver_no_actions",
                "action_scores": {},
            }

        ranked = sorted(action_scores.items(), key=lambda item: item[1], reverse=True)
        best_action_id, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else ranked[0][1]
        selected_action = next(action for action in actions if action["id"] == best_action_id)
        confidence = _clamp(0.52 + max(0.0, best_score - second_score) * 0.55, 0.52, 0.96)
        reason = (
            f"solver {street} | texture={board_texture.get('texture', 'unknown')} | "
            f"best={selected_action['label']} | score={best_score:.3f}"
        )
        return {
            "enabled": True,
            "street": street,
            "recommended_action_id": best_action_id,
            "recommended_action": selected_action["concrete_action"],
            "recommended_amount": round(float(selected_action.get("amount", 0.0) or 0.0), 2),
            "confidence": round(confidence, 2),
            "reason": reason,
            "action_scores": action_scores,
        }

    def _heuristic_adjustment(
        self,
        street: str,
        action: Dict[str, Any],
        board_texture: Dict[str, Any],
        draws: Dict[str, Any],
        hand_details: Dict[str, Any],
        opponent_profile: Dict[str, Any],
    ) -> float:
        label = str(action.get("id", ""))
        kind = str(action.get("kind", ""))
        pressure = float(action.get("pressure", 0.0) or 0.0)
        rank_value = int(hand_details.get("rank_value", 0) or 0)
        texture = str(board_texture.get("texture", "dry"))
        aggression = float(opponent_profile.get("aggression", 0.35) or 0.35)
        fold_equity = float(opponent_profile.get("fold_equity", 0.34) or 0.34)
        strong_draw = bool(
            draws.get("combo_draw")
            or draws.get("flush_draw")
            or draws.get("open_ended_straight_draw")
            or draws.get("double_gutshot")
        )
        weak_showdown = rank_value <= 1 and not strong_draw

        adjustment = 0.0
        if kind == "aggressive":
            if rank_value >= 4:
                adjustment += 0.12 + pressure * 0.03
            elif rank_value >= 2:
                adjustment += 0.05 + pressure * 0.02
            elif strong_draw and texture in {"wet", "semi_wet", "monotone"}:
                adjustment += 0.06 + fold_equity * 0.06
            elif weak_showdown and texture in {"dry", "dry_paired"} and fold_equity >= 0.25:
                adjustment += 0.03

            if label == "bet_pot" and (rank_value <= 1 and not strong_draw):
                adjustment -= 0.07
            if label == "jam" and street == "flop" and rank_value <= 2 and not strong_draw:
                adjustment -= 0.12
        elif label == "call":
            if strong_draw:
                adjustment += 0.04
            if rank_value >= 2:
                adjustment += 0.03
            if aggression >= 0.60 and weak_showdown:
                adjustment -= 0.03
        elif label == "check":
            if weak_showdown and aggression >= 0.55:
                adjustment += 0.02
            if strong_draw and street == "turn":
                adjustment -= 0.02
        elif label == "fold":
            if strong_draw or rank_value >= 1:
                adjustment -= 0.08

        return adjustment
