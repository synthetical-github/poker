from typing import Any, Dict, Tuple

from utils.logger import logger
from utils.poker_decision import analyze_spot
from utils.monte_carlo_strategy import MonteCarloStrategy
from utils.opponent_model import OpponentProfileTracker
from utils.subgame_solver import SubgameSolver
from config import STRATEGY_CONFIG


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


class PokerModel:
    """
    Zentrale Poker-Entscheidungslogik.
    Der Fokus liegt auf nachvollziehbaren Regeln statt auf Platzhaltern.
    """

    def __init__(self):
        self.strategy_mode = STRATEGY_CONFIG.get("mode", "monte_carlo")
        self.fallback_action = STRATEGY_CONFIG.get("default_action", "fold")
        self.last_decision: Dict[str, Any] = {}
        self.opponent_model = OpponentProfileTracker()
        self.monte_carlo_strategy = MonteCarloStrategy(
            preflop_iterations=STRATEGY_CONFIG.get("monte_carlo_preflop_iterations", 280),
            postflop_iterations=STRATEGY_CONFIG.get("monte_carlo_postflop_iterations", 220),
        )
        self.subgame_solver = SubgameSolver()

    def observe_game_state(self, game_state: Dict[str, Any]) -> None:
        self.opponent_model.observe(game_state)

    def analyze_game_state(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        if self.strategy_mode not in {"rule_based", "hybrid", "ml", "monte_carlo"}:
            logger.error(f"Unbekannter Strategie-Modus: {self.strategy_mode}")
            return {
                "action": self.fallback_action,
                "amount": 0.0,
                "confidence": 0.0,
                "reason": f"unknown_strategy_mode:{self.strategy_mode}",
                "hand_details": {"rank_value": 0, "category": "unknown", "display_category": "Unknown", "best_cards": [], "kickers": []},
                "board_texture": {},
                "draws": {},
            }

        hole_cards = [card for card in game_state.get("hole_cards", []) if card]
        if len(hole_cards) != 2:
            return {
                "action": self.fallback_action,
                "amount": 0.0,
                "confidence": 0.0,
                "reason": "missing_hole_cards",
                "hand_details": {"rank_value": 0, "category": "unknown", "display_category": "Unknown", "best_cards": hole_cards, "kickers": []},
                "board_texture": {},
                "draws": {},
            }

        if self.strategy_mode == "rule_based":
            decision = analyze_spot(game_state)
        else:
            if self.strategy_mode == "ml":
                logger.warning("ML-Modus ist nicht implementiert. Nutze Monte-Carlo-Entscheidung.")
            elif self.strategy_mode == "hybrid":
                logger.warning("Hybrid-Modus ist nicht implementiert. Nutze Monte-Carlo-Entscheidung.")
            decision = self.monte_carlo_strategy.analyze(
                game_state=game_state,
                opponent_profile=self.opponent_model.get_profile(),
            )
            parser_confidence = self._estimate_parser_confidence(game_state)
            solver_decision = self.subgame_solver.solve(
                game_state=game_state,
                monte_carlo_decision=decision,
                action_bundle=decision.get("action_bundle", {}) or {},
            )
            decision = self._mix_policies(
                game_state=game_state,
                monte_carlo_decision=decision,
                solver_decision=solver_decision,
                parser_confidence=parser_confidence,
            )
        self.last_decision = decision
        return decision

    def get_action(self, game_state: Dict[str, Any]) -> Tuple[str, float]:
        decision = self.analyze_game_state(game_state)
        action = str(decision.get("action", self.fallback_action) or self.fallback_action).lower()
        amount = float(decision.get("amount", 0.0) or 0.0)
        return action, amount

    def _estimate_parser_confidence(self, game_state: Dict[str, Any]) -> float:
        confidence = 0.18
        if len([card for card in game_state.get("hole_cards", []) if card]) == 2:
            confidence += 0.10
        if game_state.get("available_actions"):
            confidence += 0.16
        if game_state.get("buttons_confirmed", False):
            confidence += 0.18
        if game_state.get("is_my_turn", False):
            confidence += 0.10
        if game_state.get("hero_stack") is not None:
            confidence += 0.10
        if game_state.get("villain_stack") is not None:
            confidence += 0.08
        if int(game_state.get("num_players_remaining", 2) or 2) >= 2:
            confidence += 0.04

        street = str(game_state.get("street", "preflop") or "preflop").lower()
        board_count = len([card for card in game_state.get("community_cards", []) if card])
        expected_board = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}.get(street)
        if expected_board is not None and board_count == expected_board:
            confidence += 0.06

        return round(_clamp(confidence, 0.15, 0.96), 2)

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        values = [float(value) for value in scores.values()]
        minimum = min(values)
        maximum = max(values)
        if abs(maximum - minimum) < 1e-9:
            return {key: 0.5 for key in scores}
        return {
            key: (float(value) - minimum) / (maximum - minimum)
            for key, value in scores.items()
        }

    def _serialize_action(self, action: Dict[str, Any] | None) -> Dict[str, Any] | None:
        if not action:
            return None
        return {
            "id": str(action.get("id")),
            "label": str(action.get("label", action.get("id", "unknown"))),
            "kind": str(action.get("kind", "unknown")),
            "concrete_action": str(action.get("concrete_action", "fold")),
            "amount": round(float(action.get("amount", 0.0) or 0.0), 2),
            "hero_cost": round(float(action.get("hero_cost", 0.0) or 0.0), 2),
            "opponent_continue_cost": round(float(action.get("opponent_continue_cost", 0.0) or 0.0), 2),
            "pressure": round(float(action.get("pressure", 0.0) or 0.0), 2),
        }

    def _mix_policies(
        self,
        game_state: Dict[str, Any],
        monte_carlo_decision: Dict[str, Any],
        solver_decision: Dict[str, Any],
        parser_confidence: float,
    ) -> Dict[str, Any]:
        mixed = dict(monte_carlo_decision)
        action_bundle = monte_carlo_decision.get("action_bundle", {}) or {}
        actions = action_bundle.get("actions", []) or []
        action_lookup = {str(action.get("id")): action for action in actions}
        mc_scores = self._normalize_scores(monte_carlo_decision.get("action_evs", {}) or {})
        solver_scores = self._normalize_scores(solver_decision.get("action_scores", {}) or {})
        opponent_profile = monte_carlo_decision.get("opponent_profile", {}) or {}
        relative_strength = float((monte_carlo_decision.get("monte_carlo", {}) or {}).get("relative_strength", 0.0) or 0.0)
        mapped_live_action = str(monte_carlo_decision.get("mapped_live_action") or "")
        default_action = str(action_bundle.get("default_action") or "")
        solver_enabled = bool(solver_decision.get("enabled", False))
        combined_scores: Dict[str, float] = {}
        street = str(game_state.get("street", "preflop") or "preflop").lower()

        if street == "preflop" or not solver_enabled:
            mixed["parser_confidence"] = round(parser_confidence, 2)
            mixed["solver_decision"] = solver_decision
            mixed["policy_source"] = "monte_carlo_preflop" if street == "preflop" else "monte_carlo_postflop"
            mixed["policy_scores"] = monte_carlo_decision.get("action_evs", {}) or {}
            mixed["reason"] = (
                f"{monte_carlo_decision.get('reason', '')} | parser={parser_confidence:.2f} | "
                f"policy={mixed['policy_source']}"
            )
            mixed["confidence"] = round(
                _clamp(
                    max(float(monte_carlo_decision.get("confidence", 0.0) or 0.0), 0.42)
                    + parser_confidence * 0.08,
                    0.32,
                    0.98,
                ),
                2,
            )
            return mixed

        if action_lookup:
            for action_id, action in action_lookup.items():
                score = mc_scores.get(action_id, 0.0) * (0.64 + parser_confidence * 0.12)
                if solver_enabled:
                    score += solver_scores.get(action_id, 0.0) * (0.28 + parser_confidence * 0.12)
                if mapped_live_action and action_id == mapped_live_action:
                    score += 0.03
                if default_action and action_id == default_action and parser_confidence < 0.55:
                    score += 0.04

                action_kind = str(action.get("kind", "unknown"))
                if action_kind == "aggressive":
                    score += max(0.0, float(opponent_profile.get("fold_equity", 0.34) or 0.34) - 0.32) * 0.18
                    score += max(0.0, relative_strength) * 0.22
                    if parser_confidence < 0.45:
                        score -= 0.12
                    if not game_state.get("buttons_confirmed", False):
                        score -= 0.08
                elif action_id in {"check", "call"}:
                    score += max(0.0, -relative_strength) * 0.08
                elif action_id == "fold" and relative_strength > 0.05:
                    score -= 0.10

                combined_scores[action_id] = round(score, 4)

            best_action_id = max(combined_scores, key=combined_scores.get)
            best_action = action_lookup[best_action_id]
            solver_action_id = str(solver_decision.get("recommended_action_id") or "")
            mc_action_id = str((monte_carlo_decision.get("best_abstract_action") or {}).get("id") or "")
            if solver_enabled and best_action_id == solver_action_id == mc_action_id and mc_action_id:
                policy_source = "aligned_mc_solver"
            elif solver_enabled and best_action_id == solver_action_id and solver_action_id:
                policy_source = "solver_mix"
            elif best_action_id == mc_action_id and mc_action_id:
                policy_source = "monte_carlo_mix"
            else:
                policy_source = "policy_mix"

            mixed["action"] = str(best_action.get("concrete_action", mixed.get("action", self.fallback_action)) or self.fallback_action)
            mixed["amount"] = (
                round(float(best_action.get("amount", 0.0) or 0.0), 2)
                if mixed["action"] in {"bet", "raise"}
                else 0.0
            )
            mixed["best_abstract_action"] = self._serialize_action(best_action)
            mixed["policy_scores"] = combined_scores
        else:
            policy_source = "monte_carlo_fallback"
            mixed["policy_scores"] = {}

        mixed["parser_confidence"] = round(parser_confidence, 2)
        mixed["solver_decision"] = solver_decision
        mixed["policy_source"] = policy_source
        base_confidence = float(monte_carlo_decision.get("confidence", 0.0) or 0.0)
        solver_confidence = float(solver_decision.get("confidence", 0.0) or 0.0)
        mixed["confidence"] = round(
            _clamp(
                max(base_confidence, 0.42)
                + parser_confidence * 0.10
                + (0.05 if solver_enabled else 0.0)
                + max(0.0, solver_confidence - 0.55) * 0.08,
                0.32,
                0.98,
            ),
            2,
        )
        solver_reason = str(solver_decision.get("reason", "") or "")
        mixed["reason"] = (
            f"{monte_carlo_decision.get('reason', '')} | parser={parser_confidence:.2f} | "
            f"policy={policy_source}"
            + (f" | solver={solver_reason}" if solver_reason else "")
        )
        return mixed
