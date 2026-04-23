from typing import Any, Dict, List, Optional, Tuple

from utils.card_utils import (
    Card,
    RANK_MAP,
    evaluate_hand_strength,
    get_best_hand_details,
)

POSITION_GROUPS = {
    "utg": "early",
    "ep": "early",
    "early": "early",
    "hj": "middle",
    "mp": "middle",
    "middle": "middle",
    "co": "late",
    "btn": "late",
    "button": "late",
    "late": "late",
    "sb": "blind",
    "bb": "blind",
    "blind": "blind",
    "blinds": "blind",
    "unknown": "unknown",
}


def canonical_position(position: str) -> str:
    return POSITION_GROUPS.get(str(position or "unknown").strip().lower(), "unknown")


def starting_hand_key(hole_cards: List[Card]) -> str:
    valid_cards = [card for card in hole_cards if card]
    if len(valid_cards) != 2:
        return ""

    first, second = sorted(valid_cards, key=lambda card: card.value, reverse=True)
    if first.rank == second.rank:
        return f"{first.rank}{second.rank}"
    suited_suffix = "s" if first.suit == second.suit else "o"
    return f"{first.rank}{second.rank}{suited_suffix}"


def _preflop_score(hole_cards: List[Card]) -> float:
    valid_cards = [card for card in hole_cards if card]
    if len(valid_cards) != 2:
        return 0.0

    high_card, low_card = sorted(valid_cards, key=lambda card: card.value, reverse=True)
    gap = high_card.value - low_card.value
    is_pair = high_card.rank == low_card.rank
    is_suited = high_card.suit == low_card.suit
    is_connector = gap <= 1
    broadway_count = sum(1 for card in valid_cards if card.value >= RANK_MAP["T"])
    wheel_bonus = 0.04 if {high_card.rank, low_card.rank} == {"A", "5"} else 0.0

    score = 0.18
    score += (high_card.value / 12.0) * 0.24
    score += (low_card.value / 12.0) * 0.14

    if is_pair:
        score += 0.34 + (high_card.value / 12.0) * 0.12
    if is_suited:
        score += 0.07
    if is_connector:
        score += 0.06
    elif gap == 2:
        score += 0.03

    score += broadway_count * 0.035
    score += wheel_bonus
    return min(score, 1.0)


def _preflop_features(hole_cards: List[Card]) -> Dict[str, Any]:
    valid_cards = [card for card in hole_cards if card]
    if len(valid_cards) != 2:
        return {}

    high_card, low_card = sorted(valid_cards, key=lambda card: card.value, reverse=True)
    gap = high_card.value - low_card.value
    broadway_count = sum(1 for card in valid_cards if card.value >= RANK_MAP["T"])
    return {
        "high_card": high_card,
        "low_card": low_card,
        "high_value": high_card.value,
        "low_value": low_card.value,
        "is_pair": high_card.rank == low_card.rank,
        "is_suited": high_card.suit == low_card.suit,
        "gap": gap,
        "is_connector": gap <= 1,
        "is_one_gapper": gap == 2,
        "has_ace": high_card.rank == "A" or low_card.rank == "A",
        "has_king": high_card.rank == "K" or low_card.rank == "K",
        "has_queen": high_card.rank == "Q" or low_card.rank == "Q",
        "has_jack": high_card.rank == "J" or low_card.rank == "J",
        "broadway_count": broadway_count,
        "wheel_ace": {high_card.rank, low_card.rank} == {"A", "5"},
    }


def _price_ratio(pot_size: float, to_call: float) -> float:
    return max(0.0, float(to_call or 0.0)) / max(float(pot_size or 0.0) + float(to_call or 0.0), 0.01)


def _sizing_unit(pot_size: float, to_call: float) -> float:
    context = max(float(pot_size or 0.0), float(to_call or 0.0))
    if context < 1.0:
        return 0.05
    if context < 5.0:
        return 0.10
    return 1.0


def _decide_heads_up_preflop(
    hole_cards: List[Card],
    score: float,
    pot_size: float,
    to_call: float,
) -> Dict[str, Any]:
    features = _preflop_features(hole_cards)
    hand_key = starting_hand_key(hole_cards)
    price_ratio = _price_ratio(pot_size, to_call)
    unit = _sizing_unit(pot_size, to_call)

    is_pair = bool(features.get("is_pair"))
    is_suited = bool(features.get("is_suited"))
    has_ace = bool(features.get("has_ace"))
    has_king = bool(features.get("has_king"))
    has_queen = bool(features.get("has_queen"))
    has_jack = bool(features.get("has_jack"))
    high_value = int(features.get("high_value", 0))
    low_value = int(features.get("low_value", 0))
    broadway_count = int(features.get("broadway_count", 0))
    is_connector = bool(features.get("is_connector"))
    is_one_gapper = bool(features.get("is_one_gapper"))
    cheap_defend_price = to_call > 0 and (
        to_call <= unit * 2.0
        or (pot_size <= unit * 4.0 and price_ratio <= 0.52)
    )

    action = "fold"
    amount = 0.0
    reason_tags: List[str] = [f"HU {hand_key or 'unknown'}", f"score={score:.2f}"]

    premium = is_pair and high_value >= RANK_MAP["8"] or hand_key in {"AKs", "AKo", "AQs", "AQo", "AJs", "KQs"}
    strong = (
        premium
        or (has_ace and low_value >= RANK_MAP["8"])
        or hand_key in {"ATs", "ATo", "A9s", "KJs", "KJo", "KTs", "QJs", "QTs", "JTs"}
        or (has_ace and (is_suited or low_value >= RANK_MAP["5"]))
        or (has_king and (is_suited or low_value >= RANK_MAP["7"]))
        or (has_queen and (is_suited and low_value >= RANK_MAP["8"]))
        or (broadway_count == 2 and high_value >= RANK_MAP["J"] and low_value >= RANK_MAP["9"])
    )
    open_raise = (
        strong
        or is_pair
        or has_ace
        or (is_suited and high_value >= RANK_MAP["9"] and low_value >= RANK_MAP["5"])
        or (has_king and (low_value >= RANK_MAP["5"] or is_suited))
        or (has_queen and (low_value >= RANK_MAP["7"] or (is_suited and low_value >= RANK_MAP["4"])))
        or (has_jack and (low_value >= RANK_MAP["8"] or (is_suited and low_value >= RANK_MAP["5"])))
        or (high_value == RANK_MAP["T"] and ((low_value >= RANK_MAP["7"]) or (is_suited and low_value >= RANK_MAP["5"])))
        or (is_suited and is_connector and high_value >= RANK_MAP["7"])
        or (is_one_gapper and is_suited and high_value >= RANK_MAP["9"])
    )
    playable_open = (
        open_raise
        or (is_suited and high_value >= RANK_MAP["8"])
        or (is_connector and high_value >= RANK_MAP["7"])
        or (has_king and low_value >= RANK_MAP["4"])
    )
    playable_defend = (
        strong
        or is_pair
        or has_ace
        or (has_king and (low_value >= RANK_MAP["7"] or (is_suited and low_value >= RANK_MAP["4"])))
        or (has_queen and (low_value >= RANK_MAP["9"] or (is_suited and low_value >= RANK_MAP["6"])))
        or (has_jack and (low_value >= RANK_MAP["8"] or (is_suited and low_value >= RANK_MAP["6"])))
        or (high_value == RANK_MAP["T"] and ((low_value >= RANK_MAP["8"]) or (is_suited and low_value >= RANK_MAP["7"])))
        or (is_suited and is_connector and high_value >= RANK_MAP["6"])
        or (is_suited and is_one_gapper and high_value >= RANK_MAP["8"])
        or hand_key in {"98o", "87o"}
    )
    cheap_defend = (
        has_ace
        or (has_king and low_value >= RANK_MAP["3"])
        or (has_queen and low_value >= RANK_MAP["5"])
        or (has_jack and low_value >= RANK_MAP["7"])
        or (high_value == RANK_MAP["T"] and low_value >= RANK_MAP["6"])
        or (is_suited and high_value >= RANK_MAP["8"] and low_value >= RANK_MAP["4"])
        or (is_connector and high_value >= RANK_MAP["6"])
        or (is_one_gapper and is_suited and high_value >= RANK_MAP["7"])
    )
    ace_continue = has_ace and (
        is_suited
        or low_value >= RANK_MAP["5"]
        or broadway_count == 2
    )
    strong_broadway_continue = broadway_count == 2 and (
        high_value >= RANK_MAP["A"]
        or low_value >= RANK_MAP["T"]
    )
    weak_offsuit_broadway = broadway_count == 1 and not has_ace and not is_suited and low_value <= RANK_MAP["4"]

    if to_call <= 0:
        if open_raise:
            action = "raise"
            amount = _recommended_raise_size(pot_size, 0.0, "preflop")
            reason_tags.append("hu_open_raise")
        elif playable_open:
            action = "check"
            reason_tags.append("hu_open_mix_check")
        else:
            action = "check"
            reason_tags.append("hu_check_back")
    else:
        if premium or (score >= 0.82 and price_ratio <= 0.45):
            action = "raise"
            amount = _recommended_raise_size(pot_size, to_call, "preflop")
            reason_tags.append("hu_value_3bet")
        elif is_pair and price_ratio <= 0.50:
            action = "call"
            reason_tags.append("hu_pair_defend")
        elif playable_defend and price_ratio <= 0.42 and not weak_offsuit_broadway:
            action = "call"
            reason_tags.append("hu_defend_call")
        elif ace_continue and price_ratio <= 0.80:
            action = "call"
            reason_tags.append("hu_ace_continue")
        elif strong_broadway_continue and price_ratio <= 0.55 and not weak_offsuit_broadway:
            action = "call"
            reason_tags.append("hu_broadway_continue")
        elif cheap_defend_price and cheap_defend and not weak_offsuit_broadway:
            action = "call"
            reason_tags.append("hu_cheap_defend")
        elif is_suited and (is_connector or is_one_gapper) and price_ratio <= 0.25:
            action = "call"
            reason_tags.append("hu_spec_call")
        else:
            action = "fold"
            reason_tags.append("hu_fold")

    confidence = min(0.98, 0.56 + score * 0.34 + (0.06 if premium else 0.0))
    return {
        "street": "preflop",
        "action": action,
        "amount": round(amount, 2),
        "confidence": round(confidence, 2),
        "reason": " | ".join(reason_tags),
        "hand_key": hand_key,
        "preflop_score": round(score, 3),
    }


def analyze_board_texture(community_cards: List[Card]) -> Dict[str, Any]:
    board = [card for card in community_cards if card]
    if not board:
        return {
            "texture": "preflop",
            "paired": False,
            "monotone": False,
            "two_tone": False,
            "connected": False,
            "broadway_count": 0,
            "high_card": None,
            "flush_pressure": "none",
            "straight_pressure": "none",
        }

    values = sorted((card.value for card in board), reverse=True)
    unique_values = sorted(set(values))
    suits = [card.suit for card in board]
    rank_counts = {}
    for value in values:
        rank_counts[value] = rank_counts.get(value, 0) + 1

    max_suit_count = max(suits.count(suit) for suit in set(suits))
    monotone = len(board) >= 3 and max_suit_count >= 3 and len(set(suits)) == 1
    two_tone = len(board) >= 3 and max_suit_count >= 2 and len(set(suits)) == 2
    paired = any(count >= 2 for count in rank_counts.values())
    broadway_count = sum(1 for value in values if value >= RANK_MAP["T"])

    gaps = [unique_values[index + 1] - unique_values[index] for index in range(len(unique_values) - 1)]
    connected = bool(gaps) and max(gaps) <= 2
    straight_windows = 0
    for start in range(13):
        window = set(range(start, min(start + 5, 13)))
        if len(window.intersection(unique_values)) >= 3:
            straight_windows += 1
    if 12 in unique_values and len({12, 0, 1}.intersection(unique_values)) >= 2:
        straight_windows += 1

    if max_suit_count >= 4:
        flush_pressure = "made"
    elif max_suit_count == 3:
        flush_pressure = "draw_heavy"
    elif len(board) == 3 and max_suit_count == 2:
        flush_pressure = "draw_heavy"
    elif max_suit_count == 2:
        flush_pressure = "possible"
    else:
        flush_pressure = "none"

    if straight_windows >= 2:
        straight_pressure = "heavy"
    elif straight_windows == 1 or connected:
        straight_pressure = "medium"
    else:
        straight_pressure = "light"

    wet_signals = sum(
        [
            1 if two_tone or monotone else 0,
            1 if connected else 0,
            1 if broadway_count >= 2 else 0,
            1 if straight_pressure == "heavy" else 0,
        ]
    )
    if monotone:
        texture = "monotone"
    elif wet_signals >= 3:
        texture = "wet"
    elif paired and not connected and max_suit_count <= 2:
        texture = "dry_paired"
    elif wet_signals <= 1:
        texture = "dry"
    else:
        texture = "semi_wet"

    return {
        "texture": texture,
        "paired": paired,
        "monotone": monotone,
        "two_tone": two_tone,
        "connected": connected,
        "broadway_count": broadway_count,
        "high_card": max(values),
        "flush_pressure": flush_pressure,
        "straight_pressure": straight_pressure,
    }


def _has_flush_draw(cards: List[Card]) -> Tuple[bool, bool]:
    suits = [card.suit for card in cards if card]
    if not suits:
        return False, False
    max_suit_count = max(suits.count(suit) for suit in set(suits))
    return max_suit_count >= 4, max_suit_count == 3


def _count_straight_out_patterns(values: List[int]) -> Dict[str, bool]:
    unique_values = sorted(set(values))
    extended = set(unique_values)
    if 12 in extended:
        extended.add(-1)

    open_ended = False
    gutshot = False
    double_gutshot = False
    gutshot_windows = 0

    for start in range(-1, 9):
        window = {start + offset for offset in range(5)}
        present = len(window.intersection(extended))
        missing = sorted(window.difference(extended))
        if present == 4 and len(missing) == 1:
            missing_card = missing[0]
            if missing_card == start or missing_card == start + 4:
                open_ended = True
            else:
                gutshot = True
                gutshot_windows += 1

    if gutshot_windows >= 2:
        double_gutshot = True

    return {
        "open_ended_straight_draw": open_ended,
        "gutshot": gutshot,
        "double_gutshot": double_gutshot,
    }


def detect_draws(hole_cards: List[Card], community_cards: List[Card]) -> Dict[str, Any]:
    hole = [card for card in hole_cards if card]
    board = [card for card in community_cards if card]
    all_cards = hole + board
    if len(all_cards) < 4:
        return {
            "flush_draw": False,
            "backdoor_flush_draw": False,
            "open_ended_straight_draw": False,
            "gutshot": False,
            "double_gutshot": False,
            "pair_plus_draw": False,
            "overcards": 0,
            "combo_draw": False,
        }

    flush_draw, backdoor_flush_draw = _has_flush_draw(all_cards)
    straight_patterns = _count_straight_out_patterns([card.value for card in all_cards])
    board_high = max((card.value for card in board), default=-1)
    overcards = sum(1 for card in hole if card.value > board_high)

    hand_details = get_best_hand_details(hole, board) if len(all_cards) >= 5 else {"rank_value": 0}
    made_pair_or_better = int(hand_details.get("rank_value", 0)) >= 1

    combo_draw = flush_draw and (
        straight_patterns["open_ended_straight_draw"]
        or straight_patterns["gutshot"]
        or straight_patterns["double_gutshot"]
    )

    return {
        "flush_draw": flush_draw,
        "backdoor_flush_draw": backdoor_flush_draw,
        "open_ended_straight_draw": straight_patterns["open_ended_straight_draw"],
        "gutshot": straight_patterns["gutshot"],
        "double_gutshot": straight_patterns["double_gutshot"],
        "pair_plus_draw": made_pair_or_better and (flush_draw or straight_patterns["gutshot"] or straight_patterns["open_ended_straight_draw"]),
        "overcards": overcards,
        "combo_draw": combo_draw,
    }


def _recommended_raise_size(pot_size: float, to_call: float, street: str) -> float:
    unit = _sizing_unit(pot_size, to_call)
    base_pot = max(float(pot_size or 0.0), unit)
    if street == "preflop":
        if base_pot < 1.0:
            unopened_target = max(base_pot * 2.0, unit * 8)
            facing_raise_target = max((to_call * 3.0) + base_pot * 0.5, unit * 8)
            return round(facing_raise_target if to_call > 0 else unopened_target, 2)
        return max(to_call * 3.0 if to_call > 0 else 3.0, base_pot * 0.75)
    if base_pot < 1.0:
        return round(max(base_pot * 0.60 + to_call, to_call * 2.5 if to_call > 0 else base_pot * 0.60, unit * 4), 2)
    return max(base_pot * 0.66 + to_call, to_call * 2.5 if to_call > 0 else base_pot * 0.6)


def decide_preflop_action(game_state: Dict[str, Any]) -> Dict[str, Any]:
    hole_cards = [card for card in game_state.get("hole_cards", []) if card]
    to_call = float(game_state.get("to_call", 0.0) or 0.0)
    pot_size = float(game_state.get("pot_size", 0.0) or 0.0)
    position = canonical_position(game_state.get("position", "unknown"))
    num_players = int(game_state.get("num_players_remaining", game_state.get("num_players", 2)) or 2)
    hand_key = starting_hand_key(hole_cards)
    score = _preflop_score(hole_cards)

    if num_players <= 2:
        return _decide_heads_up_preflop(hole_cards, score, pot_size, to_call)

    threshold_adjustments = {
        "early": 0.08,
        "middle": 0.04,
        "late": -0.03,
        "blind": 0.02,
        "unknown": 0.05,
    }
    adjustment = threshold_adjustments.get(position, 0.05)
    open_raise_threshold = 0.62 + adjustment
    call_threshold = 0.46 + adjustment
    defend_threshold = 0.38 + adjustment

    action = "fold"
    amount = 0.0
    reason = f"Preflop {hand_key or 'unknown'} Score {score:.2f}"

    if to_call <= 0:
        if score >= open_raise_threshold:
            action = "raise"
            amount = _recommended_raise_size(pot_size, 0.0, "preflop")
            reason += " | Open-Raise"
        elif score >= defend_threshold and position in {"late", "blind"}:
            action = "raise"
            amount = _recommended_raise_size(pot_size, 0.0, "preflop")
            reason += " | Steal/Raise"
        else:
            action = "check"
            reason += " | Freier Flop"
    else:
        price_ratio = _price_ratio(pot_size, to_call)
        if score >= open_raise_threshold:
            action = "raise"
            amount = _recommended_raise_size(pot_size, to_call, "preflop")
            reason += " | Value-Raise"
        elif score >= call_threshold and price_ratio <= 0.42:
            action = "call"
            reason += " | Guter Call"
        elif score >= defend_threshold and position in {"late", "blind"} and price_ratio <= 0.25:
            action = "call"
            reason += " | Blind-/Late-Defense"
        else:
            action = "fold"
            reason += " | Zu schwach"

    confidence = min(0.98, 0.52 + score * 0.40)
    return {
        "street": "preflop",
        "action": action,
        "amount": round(amount, 2),
        "confidence": round(confidence, 2),
        "reason": reason,
        "hand_key": hand_key,
        "preflop_score": round(score, 3),
    }


def _estimate_postflop_equity_proxy(
    hand_strength: float,
    draws: Dict[str, Any],
    board_texture: Dict[str, Any],
) -> float:
    equity = hand_strength
    if draws["flush_draw"]:
        equity += 0.16
    elif draws["backdoor_flush_draw"]:
        equity += 0.04

    if draws["open_ended_straight_draw"]:
        equity += 0.14
    elif draws["double_gutshot"]:
        equity += 0.13
    elif draws["gutshot"]:
        equity += 0.08

    if draws["pair_plus_draw"]:
        equity += 0.07
    if draws["combo_draw"]:
        equity += 0.08
    if draws["overcards"] == 2 and board_texture["texture"] in {"dry", "dry_paired"}:
        equity += 0.05

    return max(0.0, min(equity, 0.99))


def decide_postflop_action(game_state: Dict[str, Any]) -> Dict[str, Any]:
    hole_cards = [card for card in game_state.get("hole_cards", []) if card]
    community_cards = [card for card in game_state.get("community_cards", []) if card]
    pot_size = float(game_state.get("pot_size", 0.0) or 0.0)
    to_call = float(game_state.get("to_call", 0.0) or 0.0)
    street = str(game_state.get("street", "flop") or "flop").lower()
    num_players = int(game_state.get("num_players_remaining", game_state.get("num_players", 2)) or 2)
    heads_up = num_players <= 2

    hand_strength = evaluate_hand_strength(hole_cards, community_cards)
    hand_details = get_best_hand_details(hole_cards, community_cards)
    board_texture = analyze_board_texture(community_cards)
    draws = detect_draws(hole_cards, community_cards)
    pot_odds = _price_ratio(pot_size, to_call)
    equity_proxy = _estimate_postflop_equity_proxy(hand_strength, draws, board_texture)
    rank_value = int(hand_details["rank_value"])

    action = "fold"
    amount = 0.0
    reason_parts = [
        f"{hand_details.get('display_category', hand_details['category'])}",
        f"strength={hand_strength:.2f}",
        f"equity={equity_proxy:.2f}",
        f"pot_odds={pot_odds:.2f}",
        f"texture={board_texture['texture']}",
    ]

    if to_call <= 0:
        if rank_value >= 2 or hand_strength >= 0.67:
            action = "bet"
            amount = _recommended_raise_size(pot_size, 0.0, street)
            reason_parts.append("value_bet")
        elif heads_up and rank_value >= 1 and hand_strength >= 0.50:
            action = "bet"
            amount = _recommended_raise_size(pot_size, 0.0, street) * 0.85
            reason_parts.append("hu_thin_value")
        elif heads_up and board_texture["texture"] in {"dry", "semi_wet"} and (
            draws["overcards"] >= 1 or draws["backdoor_flush_draw"] or draws["gutshot"]
        ):
            action = "bet"
            amount = _recommended_raise_size(pot_size, 0.0, street) * 0.75
            reason_parts.append("hu_cbet_pressure")
        elif draws["combo_draw"] or draws["open_ended_straight_draw"] or draws["flush_draw"]:
            action = "bet" if board_texture["texture"] in {"wet", "semi_wet"} else "check"
            amount = _recommended_raise_size(pot_size, 0.0, street) if action == "bet" else 0.0
            reason_parts.append("semi_bluff" if action == "bet" else "take_free_card")
        else:
            action = "check"
            reason_parts.append("pot_control")
    else:
        if rank_value >= 4 or (rank_value >= 2 and hand_strength >= 0.72):
            if pot_odds <= 0.45:
                action = "raise"
                amount = _recommended_raise_size(pot_size, to_call, street)
                reason_parts.append("strong_value")
            else:
                action = "call"
                reason_parts.append("keep_range_wide")
        elif equity_proxy >= pot_odds + 0.08:
            if draws["combo_draw"] and pot_odds <= 0.35:
                action = "raise"
                amount = _recommended_raise_size(pot_size, to_call, street)
                reason_parts.append("aggressive_draw")
            else:
                action = "call"
                reason_parts.append("profitable_continue")
        elif heads_up and rank_value >= 1 and pot_odds <= 0.35:
            action = "call"
            reason_parts.append("hu_pair_continue")
        elif heads_up and draws["overcards"] == 2 and (draws["gutshot"] or draws["backdoor_flush_draw"]) and pot_odds <= 0.18:
            action = "call"
            reason_parts.append("hu_float_continue")
        elif hand_strength >= 0.58 and pot_odds <= 0.22:
            action = "call"
            reason_parts.append("thin_continue")
        else:
            action = "fold"
            reason_parts.append("insufficient_equity")

    confidence = min(0.97, 0.50 + abs(equity_proxy - pot_odds) * 0.65 + rank_value * 0.03)
    return {
        "street": street,
        "action": action,
        "amount": round(amount, 2),
        "confidence": round(confidence, 2),
        "reason": " | ".join(reason_parts),
        "hand_strength": round(hand_strength, 3),
        "hand_details": hand_details,
        "board_texture": board_texture,
        "draws": draws,
        "pot_odds": round(pot_odds, 3),
        "equity_proxy": round(equity_proxy, 3),
    }


def analyze_spot(game_state: Dict[str, Any]) -> Dict[str, Any]:
    street = str(game_state.get("street", "preflop") or "preflop").lower()
    if street == "preflop":
        result = decide_preflop_action(game_state)
        result["board_texture"] = analyze_board_texture([])
        result["draws"] = detect_draws(game_state.get("hole_cards", []), [])
        result["hand_details"] = {
            "rank_value": 0,
            "category": "preflop",
            "display_category": "Preflop",
            "best_cards": [card for card in game_state.get("hole_cards", []) if card],
            "kickers": [],
        }
        return result
    return decide_postflop_action(game_state)
