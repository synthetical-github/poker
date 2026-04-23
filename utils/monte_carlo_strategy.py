from __future__ import annotations

import bisect
import hashlib
import random
from itertools import accumulate
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.card_utils import Card, RANKS, SUITS, get_best_hand_details
from utils.poker_decision import (
    _preflop_score,
    _price_ratio,
    _recommended_raise_size,
    analyze_board_texture,
    detect_draws,
    starting_hand_key,
)
from utils.action_abstraction import ActionAbstraction
from utils.range_engine import RangeEngine, WeightedRange


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _full_deck() -> List[Card]:
    return [Card(rank, suit) for rank in RANKS for suit in SUITS]


def _hand_value(hole_cards: Sequence[Card], community_cards: Sequence[Card]) -> Tuple[int, Tuple[int, ...]]:
    details = get_best_hand_details(list(hole_cards), list(community_cards))
    return int(details["rank_value"]), tuple(int(kicker) for kicker in details.get("kickers", []))


class MonteCarloStrategy:
    def __init__(self, preflop_iterations: int = 280, postflop_iterations: int = 220):
        self.preflop_iterations = max(80, int(preflop_iterations))
        self.postflop_iterations = max(80, int(postflop_iterations))
        self._cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self.action_abstraction = ActionAbstraction()
        self.range_engine = RangeEngine()

    def analyze(self, game_state: Dict[str, Any], opponent_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        hole_cards = [card for card in game_state.get("hole_cards", []) if card]
        community_cards = [card for card in game_state.get("community_cards", []) if card]
        pot_size = float(game_state.get("pot_size", 0.0) or 0.0)
        to_call = max(0.0, float(game_state.get("to_call", 0.0) or 0.0))
        street = str(game_state.get("street", "preflop") or "preflop").lower()
        position = str(game_state.get("position", "unknown") or "unknown").lower()
        num_players = int(game_state.get("num_players_remaining", game_state.get("num_players", 2)) or 2)
        profile = self._normalize_profile(opponent_profile)
        action_bundle = self.action_abstraction.build_actions(game_state)
        serialized_actions = self.action_abstraction.to_serializable(action_bundle)

        if len(hole_cards) != 2:
            return {
                "street": street,
                "action": "fold",
                "amount": 0.0,
                "confidence": 0.0,
                "reason": "monte_carlo_missing_hole_cards",
                "hand_details": {
                    "rank_value": 0,
                    "category": "unknown",
                    "display_category": "Unknown",
                    "best_cards": hole_cards,
                    "kickers": [],
                },
                "board_texture": analyze_board_texture(community_cards),
                "draws": detect_draws(hole_cards, community_cards),
                "pot_odds": round(_price_ratio(pot_size, to_call), 3),
                "equity_proxy": 0.0,
                "preflop_score": round(_preflop_score(hole_cards), 3),
                "hand_key": starting_hand_key(hole_cards),
                "opponent_profile": profile,
                "range_summary": {},
                "abstract_actions": serialized_actions,
                "action_bundle": action_bundle,
                "mapped_live_action": serialized_actions.get("mapped_live_action"),
                "best_abstract_action": None,
                "action_evs": {},
                "monte_carlo": {
                    "equity": 0.0,
                    "hand_equity": 0.0,
                    "range_equity": 0.0,
                    "relative_strength": 0.0,
                    "iterations": 0,
                    "range_iterations": 0,
                    "win_rate": 0.0,
                    "tie_rate": 0.0,
                    "range_win_rate": 0.0,
                    "range_tie_rate": 0.0,
                    "range_summary": {},
                    "hero_range_summary": {},
                },
            }

        board_texture = analyze_board_texture(community_cards)
        draws = detect_draws(hole_cards, community_cards)
        hand_details = self._build_hand_details(street, hole_cards, community_cards)
        pot_odds = _price_ratio(pot_size, to_call)
        preflop_score = _preflop_score(hole_cards)
        hand_key = starting_hand_key(hole_cards)
        monte_carlo = self._estimate_equity(
            hole_cards=hole_cards,
            community_cards=community_cards,
            pot_size=pot_size,
            to_call=to_call,
            street=street,
            position=position,
            num_players=num_players,
            opponent_profile=profile,
        )
        action_evs = self._estimate_action_evs(
            street=street,
            game_state=game_state,
            action_bundle=action_bundle,
            monte_carlo=monte_carlo,
            board_texture=board_texture,
            draws=draws,
            hand_details=hand_details,
            opponent_profile=profile,
        )
        fallback_action, fallback_amount, fallback_reason_tags = self._decide_action(
            street=street,
            equity=float(monte_carlo["equity"]),
            pot_odds=pot_odds,
            preflop_score=preflop_score,
            pot_size=pot_size,
            to_call=to_call,
            num_players=num_players,
            board_texture=board_texture,
            draws=draws,
            hand_details=hand_details,
            opponent_profile=profile,
        )
        if street == "preflop":
            selected_action = self._select_action_for_preflop(
                action_bundle=action_bundle,
                action_evs=action_evs,
                fallback_action=fallback_action,
                fallback_amount=fallback_amount,
            )
        else:
            selected_action = self._select_best_action(
                action_bundle=action_bundle,
                action_evs=action_evs,
                monte_carlo=monte_carlo,
                opponent_profile=profile,
            )

        if selected_action:
            action = str(selected_action.get("concrete_action", fallback_action) or fallback_action)
            amount = float(selected_action.get("amount", fallback_amount) or fallback_amount) if action in {"bet", "raise"} else 0.0
            best_abstract_action = self._serialize_action(selected_action)
            reason_tags = fallback_reason_tags + [
                f"best={selected_action.get('label', selected_action.get('id', 'unknown'))}",
                f"ev={action_evs.get(str(selected_action.get('id')), 0.0):.3f}",
            ]
        else:
            action, amount, reason_tags = fallback_action, fallback_amount, fallback_reason_tags
            best_abstract_action = None

        confidence = self._estimate_confidence(
            hand_details=hand_details,
            pot_odds=pot_odds,
            monte_carlo=monte_carlo,
            action_evs=action_evs,
        )
        reason = " | ".join(
            [
                f"MC hand={monte_carlo['hand_equity']:.3f}",
                f"range={monte_carlo['range_equity']:.3f}",
                f"rel={monte_carlo['relative_strength']:+.3f}",
                f"pot_odds={pot_odds:.3f}",
                f"villain={profile['style']}",
                f"agg={profile['aggression']:.2f}",
                f"loose={profile['looseness']:.2f}",
                f"fold_eq={profile['fold_equity']:.2f}",
                f"samples={monte_carlo['iterations']}/{monte_carlo['range_iterations']}",
                *reason_tags,
            ]
        )

        return {
            "street": street,
            "action": action,
            "amount": round(amount, 2),
            "confidence": round(confidence, 2),
            "reason": reason,
            "hand_details": hand_details,
            "board_texture": board_texture,
            "draws": draws,
            "pot_odds": round(pot_odds, 3),
            "equity_proxy": round(float(monte_carlo["hand_equity"]), 3),
            "preflop_score": round(preflop_score, 3),
            "hand_key": hand_key,
            "opponent_profile": profile,
            "range_summary": monte_carlo.get("range_summary", {}),
            "abstract_actions": serialized_actions,
            "action_bundle": action_bundle,
            "mapped_live_action": serialized_actions.get("mapped_live_action"),
            "best_abstract_action": best_abstract_action,
            "action_evs": {key: round(value, 4) for key, value in action_evs.items()},
            "monte_carlo": monte_carlo,
        }

    def _build_hand_details(
        self,
        street: str,
        hole_cards: List[Card],
        community_cards: List[Card],
    ) -> Dict[str, Any]:
        if street == "preflop":
            return {
                "rank_value": 0,
                "category": "preflop",
                "display_category": "Preflop",
                "best_cards": hole_cards,
                "kickers": [],
            }
        return get_best_hand_details(hole_cards, community_cards)

    def _normalize_profile(self, opponent_profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not opponent_profile:
            opponent_profile = {}
        return {
            "villain_key": str(opponent_profile.get("villain_key", "villain")),
            "sample_size": int(opponent_profile.get("sample_size", 0) or 0),
            "looseness": float(opponent_profile.get("looseness", 0.40) or 0.40),
            "aggression": float(opponent_profile.get("aggression", 0.35) or 0.35),
            "fold_equity": float(opponent_profile.get("fold_equity", 0.34) or 0.34),
            "bluff_rate": float(opponent_profile.get("bluff_rate", 0.16) or 0.16),
            "style": str(opponent_profile.get("style", "balanced") or "balanced"),
        }

    def _build_hero_reference_profile(
        self,
        street: str,
        position: str,
        pot_size: float,
        to_call: float,
    ) -> Dict[str, Any]:
        position_bonus = 0.04 if position in {"button", "dealer", "btn"} else 0.0
        if street == "preflop":
            looseness = 0.54 if to_call <= 0 else (0.46 if to_call <= max(0.1, pot_size * 0.45) else 0.36)
            aggression = 0.46 if to_call <= 0 else 0.35
            bluff_rate = 0.18 if to_call <= 0 else 0.11
        else:
            looseness = 0.44 if to_call <= 0 else 0.39
            aggression = 0.43 if to_call <= 0 else 0.37
            bluff_rate = 0.15

        return {
            "villain_key": "hero_reference",
            "sample_size": 999,
            "looseness": round(_clamp(looseness + position_bonus, 0.22, 0.72), 2),
            "aggression": round(_clamp(aggression + position_bonus * 0.6, 0.20, 0.66), 2),
            "fold_equity": 0.30,
            "bluff_rate": round(_clamp(bluff_rate + position_bonus * 0.4, 0.06, 0.28), 2),
            "style": "balanced",
        }

    def _estimate_equity(
        self,
        hole_cards: List[Card],
        community_cards: List[Card],
        pot_size: float,
        to_call: float,
        street: str,
        position: str,
        num_players: int,
        opponent_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        cache_key = (
            tuple(str(card) for card in hole_cards),
            tuple(str(card) for card in community_cards),
            round(pot_size, 2),
            round(to_call, 2),
            street,
            position,
            num_players,
            opponent_profile["style"],
            round(opponent_profile["looseness"], 2),
            round(opponent_profile["aggression"], 2),
            round(opponent_profile["fold_equity"], 2),
            round(opponent_profile["bluff_rate"], 2),
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        known_cards = set(hole_cards + community_cards)
        villain_deck = [card for card in _full_deck() if card not in known_cards]
        villain_range = self.range_engine.build_villain_range(
            deck=villain_deck,
            community_cards=community_cards,
            pot_size=pot_size,
            to_call=to_call,
            street=street,
            opponent_profile=opponent_profile,
        )
        if not villain_range.candidates:
            result = {
                "equity": 0.0,
                "hand_equity": 0.0,
                "range_equity": 0.0,
                "relative_strength": 0.0,
                "iterations": 0,
                "range_iterations": 0,
                "win_rate": 0.0,
                "tie_rate": 0.0,
                "range_win_rate": 0.0,
                "range_tie_rate": 0.0,
                "range_summary": villain_range.summary,
                "hero_range_summary": {},
            }
            self._cache[cache_key] = result
            return result

        iterations = self.preflop_iterations if street == "preflop" else self.postflop_iterations
        seed = self._build_seed(cache_key)
        hand_stats = self._simulate_hand_vs_range(
            hole_cards=hole_cards,
            community_cards=community_cards,
            deck=villain_deck,
            weighted_range=villain_range,
            iterations=iterations,
            seed=seed,
        )

        hero_reference_profile = self._build_hero_reference_profile(street, position, pot_size, to_call)
        hero_range = self.range_engine.build_villain_range(
            deck=[card for card in _full_deck() if card not in set(community_cards)],
            community_cards=community_cards,
            pot_size=pot_size,
            to_call=to_call,
            street=street,
            opponent_profile=hero_reference_profile,
        )
        range_stats = self._simulate_range_vs_range(
            hero_range=hero_range,
            villain_range=villain_range,
            community_cards=community_cards,
            iterations=max(60, iterations // 2),
            seed=seed + 7919,
        )

        hand_equity = float(hand_stats["equity"])
        range_equity = float(range_stats["equity"]) if range_stats["iterations"] > 0 else hand_equity
        if num_players > 2:
            penalty = max(0.32, 1.0 - 0.14 * (num_players - 2))
            hand_equity *= penalty
            range_equity *= penalty

        result = {
            "equity": round(_clamp(hand_equity, 0.0, 0.999), 4),
            "hand_equity": round(_clamp(hand_equity, 0.0, 0.999), 4),
            "range_equity": round(_clamp(range_equity, 0.0, 0.999), 4),
            "relative_strength": round(hand_equity - range_equity, 4),
            "iterations": hand_stats["iterations"],
            "range_iterations": range_stats["iterations"],
            "win_rate": round(hand_stats["win_rate"], 4),
            "tie_rate": round(hand_stats["tie_rate"], 4),
            "range_win_rate": round(range_stats["win_rate"], 4),
            "range_tie_rate": round(range_stats["tie_rate"], 4),
            "range_summary": villain_range.summary,
            "hero_range_summary": hero_range.summary,
        }
        self._cache[cache_key] = result
        return result

    def _simulate_hand_vs_range(
        self,
        hole_cards: List[Card],
        community_cards: List[Card],
        deck: List[Card],
        weighted_range: WeightedRange,
        iterations: int,
        seed: int,
    ) -> Dict[str, Any]:
        candidates = weighted_range.candidates
        weights = weighted_range.weights
        if not candidates or not weights:
            return {"equity": 0.0, "iterations": 0, "win_rate": 0.0, "tie_rate": 0.0}

        rng = random.Random(seed)
        cumulative_weights = list(accumulate(weights))
        total_weight = cumulative_weights[-1]
        wins = 0
        ties = 0
        missing_board_cards = max(0, 5 - len(community_cards))

        for _ in range(iterations):
            opponent_hole_cards = self._pick_weighted_candidate(candidates, cumulative_weights, total_weight, rng)
            runout_deck = [card for card in deck if card not in opponent_hole_cards]
            board_runout = list(community_cards)
            if missing_board_cards:
                board_runout.extend(rng.sample(runout_deck, missing_board_cards))

            hero_value = _hand_value(hole_cards, board_runout)
            villain_value = _hand_value(list(opponent_hole_cards), board_runout)
            if hero_value > villain_value:
                wins += 1
            elif hero_value == villain_value:
                ties += 1

        return {
            "equity": (wins + ties * 0.5) / max(iterations, 1),
            "iterations": iterations,
            "win_rate": wins / max(iterations, 1),
            "tie_rate": ties / max(iterations, 1),
        }

    def _simulate_range_vs_range(
        self,
        hero_range: WeightedRange,
        villain_range: WeightedRange,
        community_cards: List[Card],
        iterations: int,
        seed: int,
    ) -> Dict[str, Any]:
        if not hero_range.candidates or not villain_range.candidates:
            return {"equity": 0.0, "iterations": 0, "win_rate": 0.0, "tie_rate": 0.0}

        rng = random.Random(seed)
        hero_cumulative = list(accumulate(hero_range.weights))
        villain_cumulative = list(accumulate(villain_range.weights))
        hero_total = hero_cumulative[-1]
        villain_total = villain_cumulative[-1]
        full_deck = _full_deck()
        wins = 0
        ties = 0
        completed = 0
        missing_board_cards = max(0, 5 - len(community_cards))

        for _ in range(iterations):
            hero_combo = self._pick_weighted_candidate(hero_range.candidates, hero_cumulative, hero_total, rng)
            villain_combo = self._pick_non_overlapping_candidate(
                excluded_cards=set(hero_combo).union(community_cards),
                candidates=villain_range.candidates,
                cumulative_weights=villain_cumulative,
                total_weight=villain_total,
                rng=rng,
            )
            if villain_combo is None:
                continue

            dead_cards = set(hero_combo).union(villain_combo).union(community_cards)
            runout_deck = [card for card in full_deck if card not in dead_cards]
            if len(runout_deck) < missing_board_cards:
                continue

            board_runout = list(community_cards)
            if missing_board_cards:
                board_runout.extend(rng.sample(runout_deck, missing_board_cards))

            hero_value = _hand_value(list(hero_combo), board_runout)
            villain_value = _hand_value(list(villain_combo), board_runout)
            completed += 1
            if hero_value > villain_value:
                wins += 1
            elif hero_value == villain_value:
                ties += 1

        return {
            "equity": (wins + ties * 0.5) / max(completed, 1),
            "iterations": completed,
            "win_rate": wins / max(completed, 1),
            "tie_rate": ties / max(completed, 1),
        }

    def _estimate_action_evs(
        self,
        street: str,
        game_state: Dict[str, Any],
        action_bundle: Dict[str, Any],
        monte_carlo: Dict[str, Any],
        board_texture: Dict[str, Any],
        draws: Dict[str, Any],
        hand_details: Dict[str, Any],
        opponent_profile: Dict[str, Any],
    ) -> Dict[str, float]:
        actions = action_bundle.get("actions", [])
        if not actions:
            return {}

        pot_size = float(game_state.get("pot_size", 0.0) or 0.0)
        to_call = max(0.0, float(game_state.get("to_call", 0.0) or 0.0))
        hero_stack = max(0.0, float(game_state.get("hero_stack", 0.0) or 0.0))
        spr = hero_stack / max(pot_size, 0.01) if hero_stack > 0 else 0.0
        hand_equity = float(monte_carlo.get("hand_equity", 0.0) or 0.0)
        range_equity = float(monte_carlo.get("range_equity", hand_equity) or hand_equity)
        relative_strength = float(monte_carlo.get("relative_strength", hand_equity - range_equity) or 0.0)
        decision_equity = _clamp(hand_equity + relative_strength * 0.25, 0.01, 0.99)
        rank_value = int(hand_details.get("rank_value", 0) or 0)
        texture = str(board_texture.get("texture", "dry") or "dry")
        strong_draw = bool(
            draws.get("combo_draw")
            or draws.get("flush_draw")
            or draws.get("open_ended_straight_draw")
            or draws.get("double_gutshot")
        )
        showdown_value = rank_value >= 1
        fold_equity = float(opponent_profile.get("fold_equity", 0.34) or 0.34)
        aggression = float(opponent_profile.get("aggression", 0.35) or 0.35)
        looseness = float(opponent_profile.get("looseness", 0.40) or 0.40)
        bluff_rate = float(opponent_profile.get("bluff_rate", 0.16) or 0.16)
        street_context = {
            "street": street,
            "pot_size": pot_size,
            "to_call": to_call,
            "hero_stack": hero_stack,
            "spr": spr,
            "decision_equity": decision_equity,
            "hand_equity": hand_equity,
            "range_equity": range_equity,
            "relative_strength": relative_strength,
            "rank_value": rank_value,
            "texture": texture,
            "strong_draw": strong_draw,
            "showdown_value": showdown_value,
            "fold_equity": fold_equity,
            "aggression": aggression,
            "looseness": looseness,
            "bluff_rate": bluff_rate,
        }

        evs: Dict[str, float] = {}
        for action in actions:
            action_id = str(action["id"])
            if street == "preflop":
                ev = self._estimate_preflop_action_ev(action, street_context)
            elif street == "flop":
                ev = self._estimate_flop_action_ev(action, street_context)
            elif street == "turn":
                ev = self._estimate_turn_action_ev(action, street_context)
            else:
                ev = self._estimate_river_action_ev(action, street_context)
            evs[action_id] = round(ev, 4)

        return evs

    def _select_best_action(
        self,
        action_bundle: Dict[str, Any],
        action_evs: Dict[str, float],
        monte_carlo: Dict[str, Any],
        opponent_profile: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        actions = action_bundle.get("actions", [])
        if not actions or not action_evs:
            return None

        ranked_actions = sorted(
            actions,
            key=lambda action: (
                float(action_evs.get(str(action["id"]), float("-inf"))),
                -float(action.get("hero_cost", 0.0) or 0.0),
            ),
            reverse=True,
        )
        best = ranked_actions[0]
        best_id = str(best["id"])
        best_ev = float(action_evs.get(best_id, float("-inf")))
        second_ev = float(action_evs.get(str(ranked_actions[1]["id"]), best_ev)) if len(ranked_actions) > 1 else best_ev
        mapped_action_id = action_bundle.get("mapped_live_action")
        relative_strength = float(monte_carlo.get("relative_strength", 0.0) or 0.0)
        fold_equity = float(opponent_profile.get("fold_equity", 0.34) or 0.34)

        if mapped_action_id:
            mapped_action = self.action_abstraction.select_action(action_bundle, str(mapped_action_id))
            mapped_ev = float(action_evs.get(str(mapped_action_id), float("-inf")))
            if mapped_action is not None:
                close_gap = max(0.02, abs(best_ev) * 0.08, 0.04)
                aggressive_best = str(best.get("kind", "")) == "aggressive"
                aggressive_mapped = str(mapped_action.get("kind", "")) == "aggressive"
                if aggressive_best and aggressive_mapped and abs(best_ev - mapped_ev) <= close_gap:
                    return mapped_action
                if mapped_ev >= best_ev - close_gap and relative_strength <= 0.05 and fold_equity <= 0.38:
                    return mapped_action

        if best_ev <= 0.0:
            default_id = action_bundle.get("default_action")
            default_action = self.action_abstraction.select_action(action_bundle, str(default_id)) if default_id else None
            if default_action is not None and float(action_evs.get(str(default_action["id"]), float("-inf"))) >= best_ev - 0.04:
                return default_action

        if best_ev - second_ev < 0.03 and relative_strength < 0.04:
            conservative = next(
                (
                    action
                    for action in ranked_actions
                    if str(action.get("id")) in {"check", "call"}
                    and float(action_evs.get(str(action["id"]), float("-inf"))) >= best_ev - 0.05
                ),
                None,
            )
            if conservative is not None:
                return conservative

        return best

    def _select_action_for_preflop(
        self,
        action_bundle: Dict[str, Any],
        action_evs: Dict[str, float],
        fallback_action: str,
        fallback_amount: float,
    ) -> Optional[Dict[str, Any]]:
        actions = action_bundle.get("actions", [])
        if not actions:
            return None

        if fallback_action == "fold":
            return self.action_abstraction.select_action(action_bundle, "fold")
        if fallback_action == "check":
            return self.action_abstraction.select_action(action_bundle, "check")
        if fallback_action == "call":
            return self.action_abstraction.select_action(action_bundle, "call")

        aggressive_actions = [action for action in actions if str(action.get("kind", "")) == "aggressive"]
        if not aggressive_actions:
            return None

        mapped_live_action = action_bundle.get("mapped_live_action")
        mapped_action = self.action_abstraction.select_action(action_bundle, str(mapped_live_action)) if mapped_live_action else None
        if mapped_action is not None and str(mapped_action.get("kind", "")) == "aggressive":
            return mapped_action

        if fallback_amount > 0:
            return min(
                aggressive_actions,
                key=lambda action: (
                    abs(float(action.get("amount", 0.0) or 0.0) - fallback_amount),
                    -float(action_evs.get(str(action.get("id")), float("-inf"))),
                ),
            )

        return max(
            aggressive_actions,
            key=lambda action: float(action_evs.get(str(action.get("id")), float("-inf"))),
        )

    def _estimate_preflop_action_ev(self, action: Dict[str, Any], context: Dict[str, float | str | bool]) -> float:
        label = str(action.get("id", ""))
        kind = str(action.get("kind", "check"))
        pot_size = float(context["pot_size"])
        to_call = float(context["to_call"])
        decision_equity = float(context["decision_equity"])
        relative_strength = float(context["relative_strength"])
        fold_equity = float(context["fold_equity"])
        bluff_rate = float(context["bluff_rate"])
        aggression = float(context["aggression"])
        looseness = float(context["looseness"])
        hero_cost = float(action.get("hero_cost", 0.0) or 0.0)
        opponent_continue_cost = float(action.get("opponent_continue_cost", 0.0) or 0.0)
        pressure = float(action.get("pressure", 0.0) or 0.0)

        if label == "fold":
            return -min(to_call, pot_size * 0.12)
        if label == "check":
            return (decision_equity - 0.5) * pot_size * 0.18 + max(0.0, relative_strength) * 0.06
        if label == "call":
            future_pot = pot_size + hero_cost
            realization = _clamp(
                0.84 + max(0.0, relative_strength) * 0.18 - aggression * 0.04 - max(0.0, looseness - 0.5) * 0.05,
                0.58,
                0.96,
            )
            ev = decision_equity * future_pot * realization - hero_cost
            if to_call <= max(0.1, pot_size * 0.33):
                ev += pot_size * 0.03
            return ev

        fe = _clamp(
            fold_equity * (0.48 + pressure * 0.22) + bluff_rate * 0.08 - max(0.0, looseness - 0.45) * 0.10,
            0.05,
            0.58 if label == "jam" else 0.46,
        )
        eq_when_called = _clamp(
            decision_equity + max(0.0, relative_strength) * 0.10 - pressure * 0.05,
            0.01,
            0.99,
        )
        continue_pot = pot_size + hero_cost + opponent_continue_cost
        ev = fe * pot_size + (1.0 - fe) * (eq_when_called * continue_pot - hero_cost)
        if label == "bet_pot":
            ev -= pot_size * 0.12
        if label == "jam":
            ev -= pot_size * 0.35
        return ev

    def _estimate_flop_action_ev(self, action: Dict[str, Any], context: Dict[str, float | str | bool]) -> float:
        label = str(action.get("id", ""))
        kind = str(action.get("kind", "check"))
        pot_size = float(context["pot_size"])
        to_call = float(context["to_call"])
        decision_equity = float(context["decision_equity"])
        relative_strength = float(context["relative_strength"])
        rank_value = int(context["rank_value"])
        texture = str(context["texture"])
        strong_draw = bool(context["strong_draw"])
        showdown_value = bool(context["showdown_value"])
        fold_equity = float(context["fold_equity"])
        aggression = float(context["aggression"])
        looseness = float(context["looseness"])
        bluff_rate = float(context["bluff_rate"])
        hero_cost = float(action.get("hero_cost", 0.0) or 0.0)
        opponent_continue_cost = float(action.get("opponent_continue_cost", 0.0) or 0.0)
        pressure = float(action.get("pressure", 0.0) or 0.0)
        spr = float(context["spr"])

        if label == "fold":
            return -min(to_call, pot_size * 0.10)
        if label == "check":
            realization = _clamp(
                0.61
                + (0.12 if showdown_value else 0.0)
                + (0.12 if strong_draw else 0.0)
                + max(0.0, relative_strength) * 0.20
                - aggression * 0.08,
                0.45,
                0.92,
            )
            ev = (decision_equity - 0.5) * pot_size * realization
            if strong_draw and texture in {"wet", "semi_wet", "monotone"}:
                ev += pot_size * 0.05
            return ev
        if label == "call":
            future_pot = pot_size + hero_cost
            realization = _clamp(
                0.78
                + (0.08 if showdown_value else 0.0)
                + (0.10 if strong_draw else 0.0)
                + max(0.0, relative_strength) * 0.14
                - aggression * 0.05,
                0.56,
                0.96,
            )
            ev = decision_equity * future_pot * realization - hero_cost
            if strong_draw:
                ev += pot_size * 0.04
            if rank_value >= 2:
                ev += pot_size * 0.03
            return ev

        fe = _clamp(
            fold_equity * (0.54 + pressure * 0.30)
            + bluff_rate * 0.10
            - max(0.0, looseness - 0.45) * 0.10
            + (0.05 if strong_draw and texture in {"wet", "semi_wet", "monotone"} else 0.0),
            0.06,
            0.84 if label == "jam" else 0.72,
        )
        eq_when_called = _clamp(
            decision_equity
            + max(0.0, relative_strength) * 0.18
            + (0.05 if rank_value >= 3 else 0.0)
            + (0.05 if strong_draw else 0.0)
            - pressure * (0.04 if rank_value <= 1 and not strong_draw else 0.02),
            0.01,
            0.995,
        )
        continue_pot = pot_size + hero_cost + opponent_continue_cost
        ev = fe * pot_size + (1.0 - fe) * (eq_when_called * continue_pot - hero_cost)
        if label == "bet_33" and texture in {"dry", "dry_paired"} and rank_value <= 1:
            ev += pot_size * 0.05
        if label == "bet_66" and (rank_value >= 2 or strong_draw):
            ev += pot_size * 0.06
        if label == "bet_pot":
            ev += pot_size * (0.08 if (rank_value >= 4 or (strong_draw and texture in {"wet", "monotone"})) else -0.10)
        if label == "jam":
            if rank_value >= 5 and spr <= 1.4:
                ev += pot_size * 0.04
            elif strong_draw and spr <= 1.0:
                ev += pot_size * 0.02
            else:
                ev -= pot_size * 0.26 + hero_cost * 0.24
        return ev

    def _estimate_turn_action_ev(self, action: Dict[str, Any], context: Dict[str, float | str | bool]) -> float:
        label = str(action.get("id", ""))
        pot_size = float(context["pot_size"])
        to_call = float(context["to_call"])
        decision_equity = float(context["decision_equity"])
        relative_strength = float(context["relative_strength"])
        rank_value = int(context["rank_value"])
        texture = str(context["texture"])
        strong_draw = bool(context["strong_draw"])
        showdown_value = bool(context["showdown_value"])
        fold_equity = float(context["fold_equity"])
        aggression = float(context["aggression"])
        bluff_rate = float(context["bluff_rate"])
        hero_cost = float(action.get("hero_cost", 0.0) or 0.0)
        opponent_continue_cost = float(action.get("opponent_continue_cost", 0.0) or 0.0)
        pressure = float(action.get("pressure", 0.0) or 0.0)
        spr = float(context["spr"])

        if label == "fold":
            return -min(to_call, pot_size * 0.08)
        if label == "check":
            realization = _clamp(
                0.64
                + (0.14 if showdown_value else 0.0)
                + (0.06 if strong_draw else 0.0)
                + max(0.0, relative_strength) * 0.20
                - aggression * 0.10,
                0.46,
                0.93,
            )
            ev = (decision_equity - 0.5) * pot_size * realization
            if rank_value >= 2:
                ev += pot_size * 0.03
            return ev
        if label == "call":
            future_pot = pot_size + hero_cost
            realization = _clamp(
                0.74
                + (0.09 if showdown_value else 0.0)
                + (0.07 if strong_draw else 0.0)
                + max(0.0, relative_strength) * 0.12
                - aggression * 0.06,
                0.52,
                0.94,
            )
            ev = decision_equity * future_pot * realization - hero_cost
            if rank_value >= 2:
                ev += pot_size * 0.04
            return ev

        fe = _clamp(
            fold_equity * (0.49 + pressure * 0.28)
            + bluff_rate * 0.08
            + (0.04 if strong_draw else 0.0),
            0.04,
            0.78 if label == "jam" else 0.66,
        )
        eq_when_called = _clamp(
            decision_equity
            + max(0.0, relative_strength) * 0.18
            + (0.06 if rank_value >= 3 else 0.0)
            + (0.04 if strong_draw else 0.0)
            - pressure * (0.05 if rank_value <= 1 and not strong_draw else 0.02),
            0.01,
            0.995,
        )
        continue_pot = pot_size + hero_cost + opponent_continue_cost
        ev = fe * pot_size + (1.0 - fe) * (eq_when_called * continue_pot - hero_cost)
        if label == "bet_33" and rank_value >= 3:
            ev -= pot_size * 0.04
        if label == "bet_66" and (rank_value >= 2 or strong_draw):
            ev += pot_size * 0.05
        if label == "bet_pot":
            ev += pot_size * (0.10 if rank_value >= 4 else -0.06)
        if label == "jam":
            if rank_value >= 5 or (strong_draw and spr <= 1.2):
                ev += pot_size * 0.05
            elif rank_value <= 2 and not strong_draw:
                ev -= pot_size * 0.30 + hero_cost * 0.45
            else:
                ev -= pot_size * 0.22 + hero_cost * 0.18
        return ev

    def _estimate_river_action_ev(self, action: Dict[str, Any], context: Dict[str, float | str | bool]) -> float:
        label = str(action.get("id", ""))
        pot_size = float(context["pot_size"])
        to_call = float(context["to_call"])
        decision_equity = float(context["decision_equity"])
        relative_strength = float(context["relative_strength"])
        rank_value = int(context["rank_value"])
        showdown_value = bool(context["showdown_value"])
        fold_equity = float(context["fold_equity"])
        aggression = float(context["aggression"])
        bluff_rate = float(context["bluff_rate"])
        hero_cost = float(action.get("hero_cost", 0.0) or 0.0)
        opponent_continue_cost = float(action.get("opponent_continue_cost", 0.0) or 0.0)
        pressure = float(action.get("pressure", 0.0) or 0.0)
        spr = float(context["spr"])

        if label == "fold":
            return -min(to_call, pot_size * 0.06)
        if label == "check":
            realization = _clamp(
                0.70 + (0.16 if showdown_value else 0.0) + max(0.0, relative_strength) * 0.18 - aggression * 0.06,
                0.52,
                0.96,
            )
            return (decision_equity - 0.5) * pot_size * realization
        if label == "call":
            future_pot = pot_size + hero_cost
            realization = _clamp(
                0.84 + (0.10 if showdown_value else 0.0) + max(0.0, relative_strength) * 0.12 - aggression * 0.05,
                0.60,
                0.98,
            )
            ev = decision_equity * future_pot * realization - hero_cost
            if rank_value >= 2:
                ev += pot_size * 0.03
            return ev

        bluff_bonus = 0.05 if rank_value == 0 and fold_equity >= 0.42 and bluff_rate <= 0.28 else 0.0
        fe = _clamp(
            fold_equity * (0.46 + pressure * 0.26) + bluff_rate * 0.06 + bluff_bonus,
            0.03,
            0.72 if label == "jam" else 0.62,
        )
        eq_when_called = _clamp(
            decision_equity + max(0.0, relative_strength) * 0.20 + (0.08 if rank_value >= 4 else 0.0) - pressure * 0.05,
            0.01,
            0.995,
        )
        continue_pot = pot_size + hero_cost + opponent_continue_cost
        ev = fe * pot_size + (1.0 - fe) * (eq_when_called * continue_pot - hero_cost)
        if label == "bet_33":
            ev += pot_size * (0.03 if rank_value == 1 and relative_strength > 0.02 else -0.02 if rank_value >= 4 else 0.0)
        if label == "bet_66":
            ev += pot_size * (0.05 if rank_value >= 2 else 0.0)
        if label == "bet_pot":
            ev += pot_size * (0.10 if rank_value >= 4 else (-0.08 if rank_value <= 1 and bluff_bonus <= 0 else 0.02))
        if label == "jam":
            ev += pot_size * 0.08 if (rank_value >= 5 or (rank_value == 0 and bluff_bonus > 0 and spr <= 1.4)) else -pot_size * 0.20
        return ev

    def _estimate_confidence(
        self,
        hand_details: Dict[str, Any],
        pot_odds: float,
        monte_carlo: Dict[str, Any],
        action_evs: Dict[str, float],
    ) -> float:
        hand_equity = float(monte_carlo.get("hand_equity", 0.0) or 0.0)
        relative_strength = abs(float(monte_carlo.get("relative_strength", 0.0) or 0.0))
        rank_value = int(hand_details.get("rank_value", 0) or 0)
        if action_evs:
            ordered = sorted(action_evs.values(), reverse=True)
            gap = ordered[0] - ordered[1] if len(ordered) > 1 else abs(ordered[0])
        else:
            gap = 0.0
        return _clamp(
            0.52
            + abs(hand_equity - pot_odds) * 0.42
            + min(0.10, relative_strength * 0.45)
            + min(0.12, gap * 0.18)
            + min(0.10, rank_value * 0.02),
            0.38,
            0.98,
        )

    def _serialize_action(self, action: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
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

    def _pick_weighted_candidate(
        self,
        candidates: List[Tuple[Card, Card]],
        cumulative_weights: List[float],
        total_weight: float,
        rng: random.Random,
    ) -> Tuple[Card, Card]:
        pick = rng.random() * total_weight
        index = bisect.bisect_left(cumulative_weights, pick)
        return candidates[min(index, len(candidates) - 1)]

    def _pick_non_overlapping_candidate(
        self,
        excluded_cards: set[Card],
        candidates: List[Tuple[Card, Card]],
        cumulative_weights: List[float],
        total_weight: float,
        rng: random.Random,
    ) -> Optional[Tuple[Card, Card]]:
        for _ in range(10):
            candidate = self._pick_weighted_candidate(candidates, cumulative_weights, total_weight, rng)
            if excluded_cards.isdisjoint(candidate):
                return candidate
        for candidate in candidates:
            if excluded_cards.isdisjoint(candidate):
                return candidate
        return None

    def _build_seed(self, cache_key: Tuple[Any, ...]) -> int:
        payload = repr(cache_key).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        return int(digest[:16], 16)

    def _decide_action(
        self,
        street: str,
        equity: float,
        pot_odds: float,
        preflop_score: float,
        pot_size: float,
        to_call: float,
        num_players: int,
        board_texture: Dict[str, Any],
        draws: Dict[str, Any],
        hand_details: Dict[str, Any],
        opponent_profile: Dict[str, Any],
    ) -> Tuple[str, float, List[str]]:
        rank_value = int(hand_details.get("rank_value", 0) or 0)
        fold_equity = opponent_profile["fold_equity"]
        aggression = opponent_profile["aggression"]
        bluff_rate = opponent_profile["bluff_rate"]
        looseness = opponent_profile["looseness"]
        strong_draw = bool(
            draws.get("combo_draw")
            or draws.get("flush_draw")
            or draws.get("open_ended_straight_draw")
            or draws.get("double_gutshot")
        )
        multiway_penalty = 0.04 * max(0, num_players - 2)
        reason_tags: List[str] = []

        if to_call <= 0:
            if street == "preflop":
                raise_threshold = 0.46 - fold_equity * 0.06 + multiway_penalty
                if preflop_score < 0.46 and equity < raise_threshold + 0.06:
                    return "check", 0.0, ["mc_preflop_pass"]
                if equity >= raise_threshold and preflop_score >= 0.46:
                    return "raise", _recommended_raise_size(pot_size, 0.0, street), ["mc_open_raise"]
                return "check", 0.0, ["mc_check"]

            if rank_value >= 2 or equity >= 0.69 - looseness * 0.06 + multiway_penalty:
                size = _recommended_raise_size(pot_size, 0.0, street) * (1.00 + (0.10 if rank_value >= 4 else 0.0))
                return "bet", size, ["mc_value_bet"]
            if rank_value >= 1 and equity >= 0.56 - looseness * 0.05 + multiway_penalty:
                return "bet", _recommended_raise_size(pot_size, 0.0, street) * 0.85, ["mc_thin_value"]
            if strong_draw and fold_equity >= 0.16:
                return "bet", _recommended_raise_size(pot_size, 0.0, street) * 0.80, ["mc_semi_bluff"]
            if (
                equity >= 0.36
                and board_texture.get("texture") in {"dry", "dry_paired", "semi_wet"}
                and fold_equity >= 0.24
                and bluff_rate <= 0.35
            ):
                return "bet", _recommended_raise_size(pot_size, 0.0, street) * 0.65, ["mc_cbet_pressure"]
            return "check", 0.0, ["mc_check_back"]

        range_pressure = max(0.0, 0.42 - bluff_rate) * 0.10 + max(0.0, 0.38 - aggression) * 0.06
        call_threshold = min(0.94, pot_odds + range_pressure + multiway_penalty)
        raise_threshold = min(0.97, call_threshold + 0.18 - fold_equity * 0.06)

        if street == "preflop":
            if equity >= max(0.74, raise_threshold) and fold_equity >= 0.18:
                return "raise", _recommended_raise_size(pot_size, to_call, street), ["mc_preflop_iso_raise"]
            if equity >= call_threshold - 0.02:
                return "call", 0.0, ["mc_preflop_continue"]
            return "fold", 0.0, ["mc_preflop_fold"]

        if rank_value >= 4 and to_call <= max(pot_size * 1.2, 0.01):
            return "raise", _recommended_raise_size(pot_size, to_call, street), ["mc_nutted_raise"]
        if equity >= raise_threshold and (rank_value >= 2 or (strong_draw and fold_equity >= 0.22)):
            return "raise", _recommended_raise_size(pot_size, to_call, street), ["mc_value_or_pressure_raise"]
        if equity >= call_threshold - 0.02 or (strong_draw and equity + 0.06 >= pot_odds):
            return "call", 0.0, ["mc_profitable_continue"]
        return "fold", 0.0, ["mc_fold"]
