from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Sequence, Tuple

from utils.card_utils import Card, evaluate_hand_strength, get_best_hand_details
from utils.poker_decision import _preflop_score, _price_ratio, analyze_board_texture, detect_draws, starting_hand_key
from utils.range_catalog import categorize_hand_key, get_style_multiplier


@dataclass
class WeightedRange:
    candidates: List[Tuple[Card, Card]]
    weights: List[float]
    summary: Dict[str, Any]


class RangeEngine:
    def __init__(self):
        self._cache: Dict[Tuple[Any, ...], WeightedRange] = {}

    def build_villain_range(
        self,
        deck: Sequence[Card],
        community_cards: Sequence[Card],
        pot_size: float,
        to_call: float,
        street: str,
        opponent_profile: Dict[str, Any],
    ) -> WeightedRange:
        cache_key = (
            tuple(sorted(str(card) for card in deck)),
            tuple(str(card) for card in community_cards),
            round(float(pot_size or 0.0), 2),
            round(float(to_call or 0.0), 2),
            str(street or "preflop"),
            str(opponent_profile.get("style", "balanced")),
            round(float(opponent_profile.get("looseness", 0.4) or 0.4), 3),
            round(float(opponent_profile.get("aggression", 0.35) or 0.35), 3),
            round(float(opponent_profile.get("bluff_rate", 0.16) or 0.16), 3),
            round(float(opponent_profile.get("fold_equity", 0.34) or 0.34), 3),
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        board = [card for card in community_cards if card]
        price_ratio = _price_ratio(pot_size, to_call)
        board_texture = analyze_board_texture(board)
        hand_weight_totals: Dict[str, float] = defaultdict(float)
        category_weight_totals: Dict[str, float] = defaultdict(float)
        candidates: List[Tuple[Card, Card]] = []
        weights: List[float] = []

        for combo in combinations(deck, 2):
            hand_key = starting_hand_key(list(combo))
            category = categorize_hand_key(hand_key)
            weight = self._weight_combo(
                combo=combo,
                board=board,
                street=street,
                pot_size=pot_size,
                to_call=to_call,
                price_ratio=price_ratio,
                board_texture=board_texture,
                hand_key=hand_key,
                category=category,
                opponent_profile=opponent_profile,
            )
            if weight <= 0:
                continue
            rounded_weight = round(weight, 6)
            candidates.append(combo)
            weights.append(rounded_weight)
            hand_weight_totals[hand_key] += rounded_weight
            category_weight_totals[category] += rounded_weight

        summary = self._summarize_range(
            hand_weight_totals=hand_weight_totals,
            category_weight_totals=category_weight_totals,
            combo_count=len(candidates),
            street=street,
            price_ratio=price_ratio,
            opponent_profile=opponent_profile,
        )
        weighted_range = WeightedRange(candidates=candidates, weights=weights, summary=summary)
        self._cache[cache_key] = weighted_range
        return weighted_range

    def _weight_combo(
        self,
        combo: Tuple[Card, Card],
        board: List[Card],
        street: str,
        pot_size: float,
        to_call: float,
        price_ratio: float,
        board_texture: Dict[str, Any],
        hand_key: str,
        category: str,
        opponent_profile: Dict[str, Any],
    ) -> float:
        preflop_strength = _preflop_score(list(combo))
        looseness = float(opponent_profile.get("looseness", 0.40) or 0.40)
        aggression = float(opponent_profile.get("aggression", 0.35) or 0.35)
        bluff_rate = float(opponent_profile.get("bluff_rate", 0.16) or 0.16)
        style_multiplier = get_style_multiplier(str(opponent_profile.get("style", "balanced")), category)

        base = 0.02 + preflop_strength * (0.92 + looseness * 0.68)
        weight = base * style_multiplier

        if street == "preflop" or not board:
            if to_call <= 0:
                weight *= 0.82 + looseness * 0.28 + aggression * 0.08
            else:
                weight *= 0.70 + preflop_strength * (0.66 + aggression * 0.22)
                if price_ratio >= 0.45:
                    weight *= 0.72 + aggression * 0.20
                else:
                    weight *= 0.84 + looseness * 0.24
            return max(weight, 0.0005)

        postflop_strength = evaluate_hand_strength(list(combo), board)
        hand_details = get_best_hand_details(list(combo), board)
        rank_value = int(hand_details.get("rank_value", 0) or 0)
        draws = detect_draws(list(combo), board)
        draw_strength = (
            (0.20 if draws.get("flush_draw") else 0.0)
            + (0.16 if draws.get("open_ended_straight_draw") else 0.0)
            + (0.12 if draws.get("double_gutshot") else 0.0)
            + (0.09 if draws.get("gutshot") else 0.0)
            + (0.10 if draws.get("combo_draw") else 0.0)
        )
        made_hand_bonus = rank_value * 0.08 + postflop_strength * 1.30
        weight *= 0.42 + made_hand_bonus + draw_strength

        if to_call > 0:
            if rank_value == 0 and draw_strength <= 0.0:
                weight *= max(0.08, bluff_rate + aggression * 0.24)
            else:
                weight *= 0.80 + aggression * 0.22 - max(0.0, price_ratio - 0.35) * 0.18
        else:
            weight *= 0.84 + aggression * 0.10

        texture = str(board_texture.get("texture", "dry"))
        if texture in {"wet", "semi_wet"} and draw_strength > 0:
            weight *= 1.06
        if texture == "dry_paired" and rank_value == 0 and draw_strength <= 0.0:
            weight *= 0.92

        return max(weight, 0.0005)

    def _summarize_range(
        self,
        hand_weight_totals: Dict[str, float],
        category_weight_totals: Dict[str, float],
        combo_count: int,
        street: str,
        price_ratio: float,
        opponent_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        total_weight = sum(hand_weight_totals.values())
        sorted_hands = sorted(hand_weight_totals.items(), key=lambda item: item[1], reverse=True)
        sorted_categories = sorted(category_weight_totals.items(), key=lambda item: item[1], reverse=True)
        top_hands = [
            {"hand": hand_key, "weight": round(weight, 4), "share": round(weight / total_weight, 4)}
            for hand_key, weight in sorted_hands[:8]
        ] if total_weight > 0 else []
        top_categories = [
            {"category": category, "share": round(weight / total_weight, 4)}
            for category, weight in sorted_categories[:5]
        ] if total_weight > 0 else []

        top_hand_labels = ", ".join(item["hand"] for item in top_hands[:4]) if top_hands else "-"
        headline = (
            f"{opponent_profile.get('style', 'balanced')} | {street} | "
            f"price={price_ratio:.2f} | top={top_hand_labels}"
        )
        return {
            "combo_count": combo_count,
            "total_weight": round(total_weight, 4),
            "top_hands": top_hands,
            "top_categories": top_categories,
            "headline": headline,
        }
