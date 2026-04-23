import json
import time
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional

from utils.logger import logger


class SessionManager:
    def __init__(self):
        self.session_start_time = time.time()
        self.hands_analyzed = 0
        self.hands_played = 0
        self.total_profit = 0.0

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logs_dir = Path(__file__).resolve().parent / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        self.analysis_log_path = self.logs_dir / f"analysis_events_{self.session_id}.jsonl"
        self.hands_log_path = self.logs_dir / f"hands_{self.session_id}.jsonl"

        self.current_hand: Optional[Dict[str, Any]] = None
        self.current_hand_id = 0
        self._last_valid_hero_stack: Optional[float] = None
        self._hero_stack_history: List[float] = []

    def record_analysis(self):
        self.hands_analyzed += 1

    def record_hand_played(self, action: str, amount: float, profit: float = 0.0):
        self.hands_played += 1
        self.total_profit += profit
        logger.info(f"Hand gespielt: {action.upper()} {amount:.2f}, Profit: {profit:.2f}")

    def record_live_decision(self, game_state: Dict[str, Any], strategy: Dict[str, Any]) -> None:
        snapshot = self._build_snapshot(game_state, strategy)
        snapshot["hero_stack"] = self._normalize_stack_observation(
            snapshot.get("hero_stack"),
            self._current_stack_reference(),
            self._history_stack_reference(),
            snapshot.get("pot_size", 0.0),
            snapshot.get("to_call", 0.0),
        )
        if snapshot["hero_stack"] is not None:
            self._last_valid_hero_stack = snapshot["hero_stack"]
            self._remember_hero_stack(snapshot["hero_stack"])
        self._append_jsonl(self.analysis_log_path, snapshot)

        if len(snapshot["hole_cards"]) != 2:
            return

        if self.current_hand is None:
            self._start_new_hand(snapshot)
            return

        if self._is_new_hand(snapshot):
            self._finalize_current_hand(next_snapshot=snapshot)
            self._start_new_hand(snapshot)
            return

        self._update_current_hand(snapshot)

    def close_session(self) -> None:
        if self.current_hand is not None:
            self._finalize_current_hand(next_snapshot=None)

    def get_session_stats(self) -> dict:
        elapsed_time = time.time() - self.session_start_time
        return {
            "start_time": self.session_start_time,
            "elapsed_time_seconds": elapsed_time,
            "hands_analyzed": self.hands_analyzed,
            "hands_played": self.hands_played,
            "total_profit": self.total_profit,
            "average_profit_per_hand": self.total_profit / self.hands_played if self.hands_played > 0 else 0,
            "analysis_log_path": str(self.analysis_log_path),
            "hands_log_path": str(self.hands_log_path),
        }

    def _build_snapshot(self, game_state: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
        hero_stack = self._prefer_numeric_stack(
            game_state.get("hero_stack"),
            self._extract_stack(game_state.get("player_info", []), "hero"),
        )
        villain_stack = self._prefer_numeric_stack(
            game_state.get("villain_stack"),
            self._extract_stack(game_state.get("player_info", []), "villain"),
        )
        hand_details = strategy.get("hand_details", {}) or {}
        board_texture = strategy.get("board_texture", {}) or {}
        draws = strategy.get("draws", {}) or {}
        opponent_profile = strategy.get("opponent_profile", {}) or {}
        range_summary = strategy.get("range_summary", {}) or {}
        monte_carlo = strategy.get("monte_carlo", {}) or {}
        abstract_actions = strategy.get("abstract_actions", {}) or {}
        solver_decision = strategy.get("solver_decision", {}) or {}
        best_abstract_action = strategy.get("best_abstract_action", {}) or {}
        return {
            "timestamp": time.time(),
            "street": str(game_state.get("street", "unknown")),
            "hole_cards": self._serialize_cards(game_state.get("hole_cards", [])),
            "community_cards": self._serialize_cards(game_state.get("community_cards", [])),
            "pot_size": float(game_state.get("pot_size", 0.0) or 0.0),
            "to_call": float(game_state.get("to_call", 0.0) or 0.0),
            "current_bet": float(game_state.get("current_bet", 0.0) or 0.0),
            "available_actions": list(dict.fromkeys(game_state.get("available_actions", []))),
            "buttons_confirmed": bool(game_state.get("buttons_confirmed", False)),
            "is_my_turn": bool(game_state.get("is_my_turn", False)),
            "position": str(game_state.get("position", "unknown")),
            "hero_stack": hero_stack,
            "villain_stack": villain_stack,
            "recommended_action": str(strategy.get("recommended_action", "fold") or "fold"),
            "recommended_amount": float(strategy.get("amount", 0.0) or 0.0),
            "confidence": float(strategy.get("confidence", 0.0) or 0.0),
            "reason": str(strategy.get("reason", "") or ""),
            "hand_category": str(hand_details.get("display_category") or hand_details.get("category") or "Unknown"),
            "board_texture": str(board_texture.get("texture", "unknown")),
            "draws": draws,
            "preflop_score": strategy.get("preflop_score"),
            "equity_proxy": strategy.get("equity_proxy"),
            "pot_odds": strategy.get("pot_odds"),
            "opponent_style": str(opponent_profile.get("style", "unknown")),
            "opponent_profile": opponent_profile,
            "range_summary": range_summary,
            "mapped_live_action": strategy.get("mapped_live_action"),
            "best_abstract_action": best_abstract_action,
            "abstract_actions": abstract_actions,
            "action_evs": strategy.get("action_evs", {}),
            "monte_carlo": monte_carlo,
            "solver_decision": solver_decision,
            "policy_source": str(strategy.get("policy_source", "monte_carlo")),
            "parser_confidence": float(strategy.get("parser_confidence", 0.0) or 0.0),
        }

    def _prefer_numeric_stack(self, primary: Optional[float], fallback: Optional[float]) -> Optional[float]:
        for value in (primary, fallback):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric > 0:
                return round(numeric, 2)
        return None

    def _serialize_cards(self, cards: List[Any]) -> List[str]:
        return [str(card) for card in cards if card]

    def _extract_stack(self, player_info: List[Dict[str, Any]], role: str) -> Optional[float]:
        for player in player_info or []:
            if str(player.get("role", "")).lower() == role.lower():
                chips = float(player.get("chips", 0.0) or 0.0)
                return chips if chips > 0 else None
        return None

    def _current_stack_reference(self) -> Optional[float]:
        if self.current_hand is not None:
            samples = [value for value in self.current_hand.get("hero_stack_samples", []) if value is not None]
            if samples:
                return round(float(median(samples[-3:])), 2)
        return self._history_stack_reference() or self._last_valid_hero_stack

    def _history_stack_reference(self) -> Optional[float]:
        if not self._hero_stack_history:
            return None
        return round(float(median(self._hero_stack_history[-7:])), 2)

    def _normalize_stack_observation(
        self,
        raw_stack: Optional[float],
        reference_stack: Optional[float],
        history_reference: Optional[float],
        pot_size: float,
        to_call: float,
    ) -> Optional[float]:
        if raw_stack is None:
            return None
        raw_value = round(float(raw_stack), 2)
        if raw_value <= 0:
            return None

        micro_cash_context = max(float(pot_size or 0.0), float(to_call or 0.0)) < 5.0
        target_reference = reference_stack
        if history_reference is not None and history_reference > 0:
            if (
                target_reference is None
                or target_reference <= 0
                or target_reference > history_reference * 3.0
                or target_reference < history_reference / 3.0
            ):
                target_reference = history_reference

        candidates = {raw_value}
        should_try_scaled_candidates = micro_cash_context or (
            target_reference is not None and target_reference < 25.0 and raw_value >= target_reference * 2.5
        )
        if should_try_scaled_candidates and raw_value >= 10:
            candidates.add(round(raw_value / 10.0, 2))
        if should_try_scaled_candidates and raw_value >= 100:
            candidates.add(round(raw_value / 100.0, 2))

        if target_reference is None or target_reference <= 0:
            sane_candidates = [value for value in candidates if 0 < value < 100000]
            if not sane_candidates:
                return None
            return min(sane_candidates) if micro_cash_context else raw_value

        best = min(candidates, key=lambda value: abs(value - target_reference))
        abs_diff = abs(best - target_reference)
        rel_diff = abs_diff / max(target_reference, 1.0)
        if micro_cash_context:
            abs_tolerance = max(1.0, target_reference * 0.30)
            rel_tolerance = 0.40
        else:
            abs_tolerance = max(120.0, target_reference * 0.45)
            rel_tolerance = 0.55
        if abs_diff <= abs_tolerance or rel_diff <= rel_tolerance:
            return round(best, 2)
        return None

    def _remember_hero_stack(self, hero_stack: float) -> None:
        baseline = self._history_stack_reference()
        if baseline is not None and baseline > 0:
            abs_diff = abs(hero_stack - baseline)
            rel_diff = abs_diff / max(baseline, 1.0)
            if abs_diff > max(1.5, baseline * 0.50) and rel_diff > 0.55:
                return
        self._hero_stack_history.append(round(float(hero_stack), 2))
        self._hero_stack_history = self._hero_stack_history[-20:]

    def _resolve_end_stack(self, next_snapshot: Optional[Dict[str, Any]]) -> tuple[Optional[float], str]:
        observed_samples = [value for value in self.current_hand.get("hero_stack_samples", []) if value is not None]
        observed_end = round(float(median(observed_samples[-3:])), 2) if observed_samples else None

        if next_snapshot:
            reference = observed_end or self.current_hand.get("initial_hero_stack") or self._last_valid_hero_stack
            next_stack = self._normalize_stack_observation(
                next_snapshot.get("hero_stack"),
                reference,
                self._history_stack_reference(),
                next_snapshot.get("pot_size", 0.0),
                next_snapshot.get("to_call", 0.0),
            )
            if next_stack is not None:
                return next_stack, "next_hand_stack"

        if observed_end is not None:
            return observed_end, "last_seen_stack"
        return None, "unknown"

    def _is_plausible_stack_delta(self, start_stack: Optional[float], end_stack: Optional[float]) -> bool:
        if start_stack is None or end_stack is None:
            return False
        snapshots = self.current_hand.get("snapshots", []) if self.current_hand else []
        max_pot = max((float(snapshot.get("pot_size", 0.0) or 0.0) for snapshot in snapshots), default=0.0)
        max_to_call = max((float(snapshot.get("to_call", 0.0) or 0.0) for snapshot in snapshots), default=0.0)
        max_exposure = max(max_pot + max_to_call, max_to_call, 0.1)
        abs_delta = abs(float(end_stack) - float(start_stack))
        tolerance = max(float(start_stack) * 1.2, max_exposure * 10.0, 2.0)
        return abs_delta <= tolerance

    def _start_new_hand(self, snapshot: Dict[str, Any]) -> None:
        self.current_hand_id += 1
        hand_id = f"{self.session_id}_{self.current_hand_id:05d}"
        self.current_hand = {
            "hand_id": hand_id,
            "started_at": snapshot["timestamp"],
            "finished_at": None,
            "hole_cards": snapshot["hole_cards"],
            "initial_hero_stack": snapshot["hero_stack"],
            "final_hero_stack": snapshot["hero_stack"],
            "hero_stack_delta": None,
            "outcome": "unknown",
            "outcome_source": "unknown",
            "position": snapshot["position"],
            "recommendations": [],
            "snapshots": [],
            "hero_stack_samples": [],
            "final_board": snapshot["community_cards"],
            "final_hand_category": snapshot["hand_category"],
        }
        self._update_current_hand(snapshot)

    def _update_current_hand(self, snapshot: Dict[str, Any]) -> None:
        if self.current_hand is None:
            return

        compact_snapshot = {
            "timestamp": snapshot["timestamp"],
            "street": snapshot["street"],
            "community_cards": snapshot["community_cards"],
            "pot_size": snapshot["pot_size"],
            "to_call": snapshot["to_call"],
            "hero_stack": snapshot["hero_stack"],
            "villain_stack": snapshot["villain_stack"],
            "available_actions": snapshot["available_actions"],
            "buttons_confirmed": snapshot["buttons_confirmed"],
            "is_my_turn": snapshot["is_my_turn"],
            "recommended_action": snapshot["recommended_action"],
            "recommended_amount": snapshot["recommended_amount"],
            "confidence": snapshot["confidence"],
            "reason": snapshot["reason"],
            "hand_category": snapshot["hand_category"],
            "board_texture": snapshot["board_texture"],
            "draws": snapshot["draws"],
            "opponent_style": snapshot.get("opponent_style", "unknown"),
            "range_headline": str((snapshot.get("range_summary", {}) or {}).get("headline", "")),
            "mc_hand_equity": float((snapshot.get("monte_carlo", {}) or {}).get("hand_equity", 0.0) or 0.0),
            "mc_range_equity": float((snapshot.get("monte_carlo", {}) or {}).get("range_equity", 0.0) or 0.0),
            "policy_source": snapshot.get("policy_source", "monte_carlo"),
            "parser_confidence": float(snapshot.get("parser_confidence", 0.0) or 0.0),
            "solver_action": str((snapshot.get("solver_decision", {}) or {}).get("recommended_action") or ""),
        }
        self.current_hand["snapshots"].append(compact_snapshot)
        self.current_hand["final_board"] = snapshot["community_cards"]
        self.current_hand["final_hand_category"] = snapshot["hand_category"]
        if snapshot["hero_stack"] is not None:
            self.current_hand["final_hero_stack"] = snapshot["hero_stack"]
            self.current_hand["hero_stack_samples"].append(snapshot["hero_stack"])

        actionable_snapshot = bool(
            snapshot["buttons_confirmed"]
            and snapshot["is_my_turn"]
            and snapshot["available_actions"]
        )
        last_recommendation = self.current_hand["recommendations"][-1] if self.current_hand["recommendations"] else None
        recommendation_signature = (
            snapshot["street"],
            tuple(snapshot["community_cards"]),
            snapshot["recommended_action"],
            round(snapshot["recommended_amount"], 2),
            round(snapshot["confidence"], 2),
        )
        if actionable_snapshot and (
            last_recommendation is None or last_recommendation.get("signature") != recommendation_signature
        ):
            self.current_hand["recommendations"].append(
                {
                    "timestamp": snapshot["timestamp"],
                    "street": snapshot["street"],
                    "board": snapshot["community_cards"],
                    "action": snapshot["recommended_action"],
                    "amount": snapshot["recommended_amount"],
                    "confidence": snapshot["confidence"],
                    "reason": snapshot["reason"],
                    "hand_category": snapshot["hand_category"],
                    "board_texture": snapshot["board_texture"],
                    "draws": snapshot["draws"],
                    "opponent_style": snapshot.get("opponent_style", "unknown"),
                    "range_headline": str((snapshot.get("range_summary", {}) or {}).get("headline", "")),
                    "mc_hand_equity": float((snapshot.get("monte_carlo", {}) or {}).get("hand_equity", 0.0) or 0.0),
                    "mc_range_equity": float((snapshot.get("monte_carlo", {}) or {}).get("range_equity", 0.0) or 0.0),
                    "policy_source": snapshot.get("policy_source", "monte_carlo"),
                    "parser_confidence": float(snapshot.get("parser_confidence", 0.0) or 0.0),
                    "solver_action": str((snapshot.get("solver_decision", {}) or {}).get("recommended_action") or ""),
                    "signature": recommendation_signature,
                }
            )

    def _is_new_hand(self, snapshot: Dict[str, Any]) -> bool:
        if self.current_hand is None:
            return False
        current_hole = tuple(self.current_hand.get("hole_cards", []))
        next_hole = tuple(snapshot.get("hole_cards", []))
        current_board = tuple(self.current_hand.get("final_board", []))
        next_board = tuple(snapshot.get("community_cards", []))
        if len(next_hole) != 2:
            return False
        if current_hole != next_hole and snapshot.get("street") == "preflop":
            return True
        if current_board and not next_board and snapshot.get("street") == "preflop":
            return True
        if len(current_board) >= 3 and len(next_board) >= 3 and current_board[:3] != next_board[:3]:
            return True
        return False

    def _finalize_current_hand(self, next_snapshot: Optional[Dict[str, Any]]) -> None:
        if self.current_hand is None:
            return

        self.current_hand["finished_at"] = time.time()
        start_stack = self.current_hand.get("initial_hero_stack")
        end_stack, outcome_source = self._resolve_end_stack(next_snapshot)
        self.current_hand["final_hero_stack"] = end_stack

        delta = None
        outcome = "unknown"
        if self._is_plausible_stack_delta(start_stack, end_stack):
            delta = round(float(end_stack) - float(start_stack), 2)
            if delta > 0.01:
                outcome = "won"
            elif delta < -0.01:
                outcome = "lost"
            else:
                outcome = "breakeven"
        elif end_stack is not None:
            observed_samples = [value for value in self.current_hand.get("hero_stack_samples", []) if value is not None]
            fallback_end = round(float(median(observed_samples[-3:])), 2) if observed_samples else None
            if fallback_end is not None and self._is_plausible_stack_delta(start_stack, fallback_end):
                end_stack = fallback_end
                delta = round(float(end_stack) - float(start_stack), 2)
                if delta > 0.01:
                    outcome = "won"
                elif delta < -0.01:
                    outcome = "lost"
                else:
                    outcome = "breakeven"
                outcome_source = "last_seen_stack"
                self.current_hand["final_hero_stack"] = end_stack
            else:
                self.current_hand["final_hero_stack"] = None
                outcome_source = "sanitized_unknown"

        self.current_hand["hero_stack_delta"] = delta
        self.current_hand["outcome"] = outcome
        self.current_hand["outcome_source"] = outcome_source

        recommendations = self.current_hand.get("recommendations", [])
        for recommendation in recommendations:
            recommendation.pop("signature", None)
        self.current_hand.pop("hero_stack_samples", None)

        self._append_jsonl(self.hands_log_path, self.current_hand)
        if delta is not None:
            self.record_hand_played(
                action=recommendations[-1]["action"] if recommendations else "unknown",
                amount=float(recommendations[-1]["amount"]) if recommendations else 0.0,
                profit=delta,
            )
        else:
            self.hands_played += 1

        logger.info(
            f"Hand geloggt: {self.current_hand['hand_id']} | Outcome={outcome} | "
            f"Delta={delta if delta is not None else 'unknown'}"
        )
        self.current_hand = None

    def _append_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
