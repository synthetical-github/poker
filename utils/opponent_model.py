from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


@dataclass
class OpponentProfileTracker:
    villain_key: str = "villain"
    observed_actionable_spots: int = 0
    preflop_opportunities: int = 0
    preflop_pressure_spots: int = 0
    preflop_small_pressure: int = 0
    preflop_large_pressure: int = 0
    preflop_checks: int = 0
    postflop_opportunities: int = 0
    postflop_pressure_spots: int = 0
    postflop_small_bets: int = 0
    postflop_large_bets: int = 0
    postflop_checks: int = 0
    _last_signature: Optional[Tuple[Any, ...]] = None

    def observe(self, game_state: Dict[str, Any]) -> None:
        if int(game_state.get("num_players_remaining", game_state.get("num_players", 2)) or 2) != 2:
            return
        if not game_state.get("is_my_turn", False):
            return
        if not game_state.get("buttons_confirmed", False):
            return
        if not game_state.get("available_actions", []):
            return

        villain_key = self._extract_villain_key(game_state)
        if villain_key != self.villain_key:
            self.reset(villain_key)

        signature = self._build_signature(game_state)
        if signature == self._last_signature:
            return
        self._last_signature = signature

        street = str(game_state.get("street", "preflop") or "preflop").lower()
        pot_size = float(game_state.get("pot_size", 0.0) or 0.0)
        to_call = max(0.0, float(game_state.get("to_call", 0.0) or 0.0))
        available_actions = set(str(action).lower() for action in game_state.get("available_actions", []))
        price_ratio = to_call / max(pot_size + to_call, 0.01)

        self.observed_actionable_spots += 1
        if street == "preflop":
            self.preflop_opportunities += 1
            if to_call <= 0 and "check" in available_actions:
                self.preflop_checks += 1
                return
            if to_call > 0:
                self.preflop_pressure_spots += 1
                if price_ratio >= 0.55:
                    self.preflop_large_pressure += 1
                else:
                    self.preflop_small_pressure += 1
            return

        self.postflop_opportunities += 1
        if to_call <= 0 and "check" in available_actions:
            self.postflop_checks += 1
            return
        if to_call > 0:
            self.postflop_pressure_spots += 1
            if price_ratio >= 0.55:
                self.postflop_large_bets += 1
            else:
                self.postflop_small_bets += 1

    def reset(self, villain_key: str = "villain") -> None:
        self.villain_key = villain_key
        self.observed_actionable_spots = 0
        self.preflop_opportunities = 0
        self.preflop_pressure_spots = 0
        self.preflop_small_pressure = 0
        self.preflop_large_pressure = 0
        self.preflop_checks = 0
        self.postflop_opportunities = 0
        self.postflop_pressure_spots = 0
        self.postflop_small_bets = 0
        self.postflop_large_bets = 0
        self.postflop_checks = 0
        self._last_signature = None

    def get_profile(self) -> Dict[str, Any]:
        if self.observed_actionable_spots <= 0:
            return {
                "villain_key": self.villain_key,
                "sample_size": 0,
                "preflop_pressure_rate": 0.0,
                "postflop_pressure_rate": 0.0,
                "large_bet_rate": 0.0,
                "check_rate": 0.0,
                "looseness": 0.40,
                "aggression": 0.35,
                "fold_equity": 0.34,
                "bluff_rate": 0.16,
                "style": "balanced",
            }

        preflop_pressure_rate = self.preflop_pressure_spots / max(self.preflop_opportunities, 1)
        postflop_pressure_rate = self.postflop_pressure_spots / max(self.postflop_opportunities, 1)
        large_bet_rate = (
            (self.preflop_large_pressure + self.postflop_large_bets)
            / max(self.preflop_pressure_spots + self.postflop_pressure_spots, 1)
        )
        check_rate = (
            self.preflop_checks + self.postflop_checks
        ) / max(self.preflop_opportunities + self.postflop_opportunities, 1)
        small_pressure_rate = (
            self.preflop_small_pressure + self.postflop_small_bets
        ) / max(self.preflop_pressure_spots + self.postflop_pressure_spots, 1)

        looseness = _clamp(
            0.28 + preflop_pressure_rate * 0.40 + small_pressure_rate * 0.18 - large_bet_rate * 0.08,
            0.18,
            0.88,
        )
        aggression = _clamp(
            0.24 + postflop_pressure_rate * 0.48 + large_bet_rate * 0.20 - check_rate * 0.08,
            0.12,
            0.94,
        )
        fold_equity = _clamp(
            0.44 - looseness * 0.20 - aggression * 0.10 + check_rate * 0.16,
            0.10,
            0.62,
        )
        bluff_rate = _clamp(
            0.10 + aggression * 0.24 + small_pressure_rate * 0.08 - large_bet_rate * 0.05,
            0.05,
            0.42,
        )

        sample_blend = min(1.0, self.observed_actionable_spots / 6.0)
        looseness = _clamp(0.40 + (looseness - 0.40) * sample_blend, 0.18, 0.88)
        aggression = _clamp(0.35 + (aggression - 0.35) * sample_blend, 0.12, 0.94)
        fold_equity = _clamp(0.34 + (fold_equity - 0.34) * sample_blend, 0.10, 0.62)
        bluff_rate = _clamp(0.16 + (bluff_rate - 0.16) * sample_blend, 0.05, 0.42)

        return {
            "villain_key": self.villain_key,
            "sample_size": self.observed_actionable_spots,
            "preflop_pressure_rate": round(preflop_pressure_rate, 3),
            "postflop_pressure_rate": round(postflop_pressure_rate, 3),
            "large_bet_rate": round(large_bet_rate, 3),
            "check_rate": round(check_rate, 3),
            "looseness": round(looseness, 3),
            "aggression": round(aggression, 3),
            "fold_equity": round(fold_equity, 3),
            "bluff_rate": round(bluff_rate, 3),
            "style": self._describe_style(looseness, aggression),
        }

    def _build_signature(self, game_state: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            str(game_state.get("street", "preflop")),
            tuple(str(card) for card in game_state.get("hole_cards", []) if card),
            tuple(str(card) for card in game_state.get("community_cards", []) if card),
            round(float(game_state.get("pot_size", 0.0) or 0.0), 2),
            round(float(game_state.get("to_call", 0.0) or 0.0), 2),
            tuple(str(action).lower() for action in game_state.get("available_actions", [])),
        )

    def _extract_villain_key(self, game_state: Dict[str, Any]) -> str:
        for player in game_state.get("player_info", []) or []:
            if str(player.get("role", "")).lower() != "villain":
                continue
            name = str(player.get("name", "") or "").strip()
            if name:
                return name
        return "villain"

    def _describe_style(self, looseness: float, aggression: float) -> str:
        if looseness >= 0.58 and aggression >= 0.55:
            return "loose_aggressive"
        if looseness >= 0.58:
            return "loose_passive"
        if aggression >= 0.60:
            return "tight_aggressive"
        if looseness <= 0.30 and aggression <= 0.30:
            return "tight_passive"
        return "balanced"
