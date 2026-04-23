from typing import Any, Dict, List, Optional

from utils.logger import logger
from utils.card_utils import Card
from models.poker_model import PokerModel
from config import STRATEGY_CONFIG


class StrategyEngine:
    def __init__(self):
        self.poker_model = PokerModel()
        self.config = STRATEGY_CONFIG
        self.last_strategy_info = None

    def calculate_strategy(
        self,
        hole_cards: List[Optional[Card]],
        community_cards: List[Optional[Card]],
        table_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        valid_hole_cards = [card for card in hole_cards if card]
        valid_community_cards = [card for card in community_cards if card]

        pot_size = float(table_info.get("pot_size", 0.0) or 0.0)
        to_call = max(0.0, float(table_info.get("to_call", 0.0) or 0.0))
        current_bet = float(table_info.get("current_bet", 0.0) or 0.0)
        unit = self._betting_unit(pot_size, to_call, current_bet)
        if to_call > 0:
            max_reasonable_current_bet = max(to_call * 2.0, pot_size * 2.5, unit * 6.0)
            if current_bet > max_reasonable_current_bet:
                current_bet = to_call
        elif current_bet > max(pot_size * 2.5, unit * 6.0):
            current_bet = 0.0
        hero_stack = table_info.get("hero_stack")
        hero_stack_value = float(hero_stack) if hero_stack is not None else None
        if hero_stack_value is not None and hero_stack_value > 0:
            if to_call > hero_stack_value * 1.05:
                to_call = 0.0
            if current_bet > hero_stack_value * 1.05:
                current_bet = 0.0
        num_players = int(table_info.get("num_players_remaining", table_info.get("num_players", 2)) or 2)
        position = table_info.get("position", "unknown")
        street = table_info.get("street", "preflop")
        available_actions = list(dict.fromkeys(table_info.get("available_actions", [])))

        if len(valid_hole_cards) != 2:
            logger.warning("Ungültige oder fehlende Hole Cards. Fallback auf 'fold'.")
            return {
                "recommended_action": "fold",
                "amount": 0.0,
                "confidence": 0.0,
                "reason": "missing_hole_cards",
                "available_actions": available_actions,
            }

        game_state = {
            "hole_cards": valid_hole_cards,
            "community_cards": valid_community_cards,
            "pot_size": pot_size,
            "to_call": to_call,
            "current_bet": current_bet,
            "num_players_remaining": num_players,
            "position": position,
            "street": street,
            "available_actions": available_actions,
            "buttons_confirmed": bool(table_info.get("buttons_confirmed", False)),
            "is_my_turn": bool(table_info.get("is_my_turn", False)),
            "hero_stack": hero_stack_value,
            "villain_stack": table_info.get("villain_stack"),
            "player_info": table_info.get("player_info", []),
        }

        self.poker_model.observe_game_state(game_state)
        decision = self.poker_model.analyze_game_state(game_state)
        action = str(decision.get("action", "fold") or "fold").lower()
        amount = float(decision.get("amount", 0.0) or 0.0)

        action, amount = self._align_action_to_buttons(
            action,
            amount,
            to_call,
            pot_size,
            current_bet,
            available_actions,
            hero_stack_value,
        )

        strategy_info = {
            "recommended_action": action,
            "amount": round(amount, 2),
            "confidence": float(decision.get("confidence", 0.0) or 0.0),
            "reason": decision.get("reason", ""),
            "hole_cards": [str(card) for card in valid_hole_cards],
            "community_cards": [str(card) for card in valid_community_cards],
            "pot_size": pot_size,
            "to_call": to_call,
            "current_bet": current_bet,
            "street": street,
            "position": position,
            "available_actions": available_actions,
            "hand_details": decision.get("hand_details", {}),
            "board_texture": decision.get("board_texture", {}),
            "draws": decision.get("draws", {}),
            "preflop_score": decision.get("preflop_score"),
            "hand_key": decision.get("hand_key"),
            "equity_proxy": decision.get("equity_proxy"),
            "pot_odds": decision.get("pot_odds"),
            "opponent_profile": decision.get("opponent_profile", {}),
            "range_summary": decision.get("range_summary", {}),
            "abstract_actions": decision.get("abstract_actions", {}),
            "mapped_live_action": decision.get("mapped_live_action"),
            "best_abstract_action": decision.get("best_abstract_action"),
            "action_evs": decision.get("action_evs", {}),
            "monte_carlo": decision.get("monte_carlo", {}),
            "solver_decision": decision.get("solver_decision", {}),
            "policy_source": decision.get("policy_source", "monte_carlo"),
            "parser_confidence": decision.get("parser_confidence", 0.0),
            "policy_scores": decision.get("policy_scores", {}),
        }

        if strategy_info != self.last_strategy_info:
            logger.info(f"Strategie berechnet: {strategy_info}")
            self.last_strategy_info = strategy_info

        return strategy_info

    def _betting_unit(self, pot_size: float, to_call: float, current_bet: float) -> float:
        context = max(float(pot_size or 0.0), float(to_call or 0.0), float(current_bet or 0.0))
        if context < 1.0:
            return 0.05
        if context < 5.0:
            return 0.10
        return 1.0

    def _align_action_to_buttons(
        self,
        action: str,
        amount: float,
        to_call: float,
        pot_size: float,
        current_bet: float,
        available_actions: List[str],
        hero_stack: float | None,
    ) -> tuple[str, float]:
        if available_actions == ["fold"]:
            return "fold", 0.0

        if available_actions:
            if action == "call" and "call" not in available_actions and "check" in available_actions:
                action = "check"
            elif action == "bet" and "bet" not in available_actions and "raise" in available_actions:
                action = "raise"
            elif action == "raise" and "raise" not in available_actions:
                if "bet" in available_actions and to_call == 0:
                    action = "bet"
                elif "call" in available_actions:
                    action = "call"
                elif "check" in available_actions:
                    action = "check"
                elif "fold" in available_actions:
                    action = "fold"
            elif action == "fold" and to_call == 0 and "check" in available_actions:
                action = "check"

        if action in {"check", "fold", "call"}:
            return action, 0.0

        if action == "bet" and amount <= 0:
            amount = self._calculate_default_bet_size(pot_size, to_call, is_bet=True)
        elif action == "raise" and amount <= 0:
            amount = self._calculate_default_bet_size(pot_size, to_call, is_bet=False)

        unit = self._betting_unit(pot_size, to_call, current_bet)
        max_reasonable_amount = max(
            pot_size * 4.0,
            pot_size + (to_call * 4.0),
            current_bet * 4.0,
            to_call * 4.0,
            unit * 8.0,
        )
        amount = min(amount, max_reasonable_amount)
        min_bet_required = self._get_minimum_bet_or_raise(pot_size, current_bet, to_call)
        if action in {"bet", "raise"} and amount < min_bet_required:
            amount = min_bet_required
        if hero_stack is not None and hero_stack > 0:
            amount = min(amount, float(hero_stack))

        if action == "raise" and amount <= to_call and to_call > 0:
            return "call", 0.0
        if action == "bet" and to_call == 0 and amount <= 0:
            return "check", 0.0
        return action, amount

    def _calculate_default_bet_size(self, pot_size: float, to_call: float, is_bet: bool) -> float:
        unit = self._betting_unit(pot_size, to_call, 0.0 if is_bet else to_call)
        if is_bet:
            factor = self.config.get("bet_size_percentage", 0.5)
            size = max(pot_size, unit) * factor
        else:
            factor = self.config.get("raise_size_percentage", 0.75)
            size = (max(pot_size, unit) + to_call) * factor
        min_bet_raise = self._get_minimum_bet_or_raise(pot_size, 0.0 if is_bet else to_call, to_call)
        return max(size, min_bet_raise)

    def _get_minimum_bet_or_raise(self, pot_size: float, current_bet: float, to_call: float) -> float:
        min_bet = self._betting_unit(pot_size, to_call, current_bet)
        if current_bet > 0:
            min_raise = current_bet * self.config.get("min_raise_factor", 2.0)
            return max(min_raise, current_bet + min_bet, to_call + min_bet)
        min_bet_bet = max(pot_size, min_bet) * self.config.get("min_bet_factor", 0.2)
        return max(min_bet, min_bet_bet)
