from __future__ import annotations

from typing import Any, Dict, List, Optional


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


class ActionAbstraction:
    BET_SIZES = (
        ("bet_33", "33%", 0.33),
        ("bet_66", "66%", 0.66),
        ("bet_pot", "Pot", 1.00),
    )

    def build_actions(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        pot_size = float(game_state.get("pot_size", 0.0) or 0.0)
        to_call = max(0.0, float(game_state.get("to_call", 0.0) or 0.0))
        current_bet = max(0.0, float(game_state.get("current_bet", 0.0) or 0.0))
        hero_stack_raw = game_state.get("hero_stack")
        hero_stack = float(hero_stack_raw) if hero_stack_raw not in (None, "") else None
        available_actions = list(dict.fromkeys(str(action).lower() for action in game_state.get("available_actions", [])))
        action_state = game_state.get("action_state", {}) or {}
        live_raise_to_amount = max(0.0, float(action_state.get("raise_to_amount", 0.0) or 0.0))
        live_call_amount = max(0.0, float(action_state.get("call_amount", 0.0) or 0.0))
        call_amount = live_call_amount if live_call_amount > 0 else to_call

        actions: List[Dict[str, Any]] = []
        concrete_aggressive_action = self._resolve_aggressive_action(available_actions, to_call)
        unit = self._betting_unit(pot_size, to_call, current_bet)
        base_pot = max(pot_size, unit)

        if "fold" in available_actions:
            actions.append(
                {
                    "id": "fold",
                    "label": "Fold",
                    "kind": "fold",
                    "concrete_action": "fold",
                    "amount": 0.0,
                    "hero_cost": 0.0,
                    "opponent_continue_cost": 0.0,
                    "pressure": 0.0,
                }
            )
        if "check" in available_actions:
            actions.append(
                {
                    "id": "check",
                    "label": "Check",
                    "kind": "check",
                    "concrete_action": "check",
                    "amount": 0.0,
                    "hero_cost": 0.0,
                    "opponent_continue_cost": 0.0,
                    "pressure": 0.0,
                }
            )
        if "call" in available_actions:
            actions.append(
                {
                    "id": "call",
                    "label": "Call",
                    "kind": "call",
                    "concrete_action": "call",
                    "amount": 0.0,
                    "hero_cost": round(call_amount, 2),
                    "opponent_continue_cost": 0.0,
                    "pressure": 0.10 if call_amount > 0 else 0.0,
                }
            )

        if concrete_aggressive_action:
            for action_id, label, fraction in self.BET_SIZES:
                total_amount = self._target_total_amount(
                    base_pot=base_pot,
                    to_call=to_call,
                    current_bet=current_bet,
                    unit=unit,
                    hero_stack=hero_stack,
                    fraction=fraction,
                )
                hero_cost = total_amount if to_call <= 0 else max(0.0, total_amount - current_bet)
                opponent_continue_cost = total_amount if to_call <= 0 else max(0.0, total_amount - to_call)
                actions.append(
                    {
                        "id": action_id,
                        "label": label,
                        "kind": "aggressive",
                        "concrete_action": concrete_aggressive_action,
                        "amount": round(total_amount, 2),
                        "hero_cost": round(hero_cost, 2),
                        "opponent_continue_cost": round(opponent_continue_cost, 2),
                        "pressure": fraction,
                    }
                )

            if hero_stack is not None and hero_stack > 0:
                jam_amount = round(hero_stack, 2)
                jam_hero_cost = jam_amount if to_call <= 0 else max(0.0, jam_amount - current_bet)
                jam_continue_cost = jam_amount if to_call <= 0 else max(0.0, jam_amount - to_call)
                actions.append(
                    {
                        "id": "jam",
                        "label": "Jam",
                        "kind": "aggressive",
                        "concrete_action": concrete_aggressive_action,
                        "amount": jam_amount,
                        "hero_cost": round(jam_hero_cost, 2),
                        "opponent_continue_cost": round(jam_continue_cost, 2),
                        "pressure": 1.75,
                    }
                )

        mapped_live_action = self._map_live_amount_to_abstract(actions, available_actions, call_amount, live_raise_to_amount)
        default_action = self._default_action_id(actions)
        return {
            "actions": actions,
            "mapped_live_action": mapped_live_action,
            "default_action": default_action,
        }

    def to_serializable(self, action_bundle: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "mapped_live_action": action_bundle.get("mapped_live_action"),
            "default_action": action_bundle.get("default_action"),
            "actions": [
                {
                    "id": action["id"],
                    "label": action["label"],
                    "kind": action["kind"],
                    "concrete_action": action["concrete_action"],
                    "amount": round(float(action.get("amount", 0.0) or 0.0), 2),
                    "hero_cost": round(float(action.get("hero_cost", 0.0) or 0.0), 2),
                    "opponent_continue_cost": round(float(action.get("opponent_continue_cost", 0.0) or 0.0), 2),
                    "pressure": round(float(action.get("pressure", 0.0) or 0.0), 2),
                }
                for action in action_bundle.get("actions", [])
            ],
        }

    def select_action(self, action_bundle: Dict[str, Any], action_id: str) -> Optional[Dict[str, Any]]:
        for action in action_bundle.get("actions", []):
            if action.get("id") == action_id:
                return action
        return None

    def _resolve_aggressive_action(self, available_actions: List[str], to_call: float) -> Optional[str]:
        if to_call > 0:
            if "raise" in available_actions:
                return "raise"
            return None
        if "bet" in available_actions:
            return "bet"
        if "raise" in available_actions:
            return "raise"
        return None

    def _target_total_amount(
        self,
        base_pot: float,
        to_call: float,
        current_bet: float,
        unit: float,
        hero_stack: Optional[float],
        fraction: float,
    ) -> float:
        if to_call <= 0:
            target = max(base_pot * fraction, unit * 2.0)
        else:
            target = max(current_bet + max(base_pot * fraction, unit * 2.0), to_call + unit * 2.0)
        if hero_stack is not None and hero_stack > 0:
            target = min(target, hero_stack)
        return round(target, 2)

    def _map_live_amount_to_abstract(
        self,
        actions: List[Dict[str, Any]],
        available_actions: List[str],
        call_amount: float,
        live_raise_to_amount: float,
    ) -> Optional[str]:
        aggressive_actions = [action for action in actions if action["kind"] == "aggressive"]
        if live_raise_to_amount > 0 and aggressive_actions:
            best = min(aggressive_actions, key=lambda action: abs(action["amount"] - live_raise_to_amount))
            return str(best["id"])
        if "check" in available_actions and call_amount <= 0:
            return "check"
        if "call" in available_actions and call_amount > 0:
            return "call"
        if "fold" in available_actions and not aggressive_actions:
            return "fold"
        return None

    def _default_action_id(self, actions: List[Dict[str, Any]]) -> Optional[str]:
        if not actions:
            return None
        for preferred in ("check", "call", "bet_33", "bet_66", "bet_pot", "jam", "fold"):
            for action in actions:
                if action["id"] == preferred:
                    return preferred
        return actions[0]["id"]

    def _betting_unit(self, pot_size: float, to_call: float, current_bet: float) -> float:
        context = max(float(pot_size or 0.0), float(to_call or 0.0), float(current_bet or 0.0))
        if context < 1.0:
            return 0.05
        if context < 5.0:
            return 0.10
        return 1.0
