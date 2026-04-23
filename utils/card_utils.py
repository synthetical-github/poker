from collections import Counter
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from config import POKER_SETTINGS
from logger import logger

SUITS = POKER_SETTINGS["suits"]
RANKS = POKER_SETTINGS["ranks"]
RANK_MAP = POKER_SETTINGS["rank_map"]
VALUE_TO_RANK = {value: rank for rank, value in RANK_MAP.items()}

HAND_CATEGORY_NAMES = {
    8: "straight_flush",
    7: "four_of_a_kind",
    6: "full_house",
    5: "flush",
    4: "straight",
    3: "three_of_a_kind",
    2: "two_pair",
    1: "one_pair",
    0: "high_card",
}

HAND_CATEGORY_DISPLAY_NAMES = {
    "straight_flush": "Straight Flush",
    "four_of_a_kind": "Quads",
    "full_house": "Full House",
    "flush": "Flush",
    "straight": "Straight",
    "three_of_a_kind": "Trips",
    "two_pair": "Two Pair",
    "one_pair": "Pair",
    "high_card": "High Card",
    "preflop": "Preflop",
    "unknown": "Unknown",
}


class Card:
    def __init__(self, rank: str, suit: str):
        rank = rank.upper()
        suit = suit.upper()
        if rank not in RANKS:
            raise ValueError(f"Ungültiger Rang: {rank}. Gültige Ränge: {RANKS}")
        if suit not in SUITS:
            raise ValueError(f"Ungültige Farbe: {suit}. Gültige Farben: {SUITS}")

        self.rank = rank
        self.suit = suit
        self.value = RANK_MAP[rank]

    def __repr__(self) -> str:
        return f"{self.rank}{self.suit}"

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank == other.rank and self.suit == other.suit

    def __lt__(self, other) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.value < other.value

    def __hash__(self) -> int:
        return hash((self.rank, self.suit))


def parse_card_string(card_str: str) -> Optional[Card]:
    """Erstellt ein Card-Objekt aus einem String wie 'KH', 'AS', 'TD'."""
    if not card_str or len(card_str) < 2:
        return None

    card_str = card_str.upper().strip()
    rank_char = card_str[0]
    suit_char = card_str[1]

    rank_str = rank_char
    if rank_char == "1" and card_str.startswith("10"):
        rank_str = "T"
        suit_char = card_str[2] if len(card_str) > 2 else ""

    if rank_str in RANKS and suit_char in SUITS:
        try:
            return Card(rank_str, suit_char)
        except ValueError as exc:
            logger.warning(f"Fehler beim Erstellen der Karte aus '{card_str}': {exc}")
            return None
    return None


def get_card_name(card: Card) -> str:
    return str(card)


def _find_straight_high(values: List[int]) -> Optional[int]:
    unique_values = sorted(set(values), reverse=True)
    if 12 in unique_values:
        unique_values.append(-1)

    run_length = 1
    for index in range(1, len(unique_values)):
        if unique_values[index - 1] - 1 == unique_values[index]:
            run_length += 1
            if run_length >= 5:
                return unique_values[index - 4]
        else:
            run_length = 1
    return None


def _select_cards_by_values(cards: List[Card], ordered_values: List[int], limit: int = 5) -> List[Card]:
    selected: List[Card] = []
    remaining = list(cards)
    for value in ordered_values:
        for card in remaining:
            if card.value == value:
                selected.append(card)
                remaining.remove(card)
                break
        if len(selected) >= limit:
            break
    return selected


def _select_straight_cards(cards: List[Card], straight_high: int) -> List[Card]:
    target_values = [straight_high - offset for offset in range(5)]
    if straight_high == 3:
        target_values = [3, 2, 1, 0, 12]
    selected: List[Card] = []
    remaining = sorted(cards, key=lambda card: card.value, reverse=True)
    for value in target_values:
        for card in remaining:
            if card.value == value:
                selected.append(card)
                remaining.remove(card)
                break
    return selected


def _evaluate_five_card_hand(cards: List[Card]) -> Tuple[int, List[int], List[Card]]:
    sorted_cards = sorted(cards, key=lambda card: card.value, reverse=True)
    values = [card.value for card in sorted_cards]
    suits = [card.suit for card in sorted_cards]
    value_counts = Counter(values)
    counts_desc = sorted(value_counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
    is_flush = len(set(suits)) == 1
    straight_high = _find_straight_high(values)

    if is_flush and straight_high is not None:
        best_cards = _select_straight_cards(sorted_cards, straight_high)
        return 8, [straight_high], best_cards

    if counts_desc[0][1] == 4:
        quad_value = counts_desc[0][0]
        kicker = max(value for value, count in value_counts.items() if count == 1)
        ordered_values = [quad_value, quad_value, quad_value, quad_value, kicker]
        return 7, [quad_value, kicker], _select_cards_by_values(sorted_cards, ordered_values)

    if counts_desc[0][1] == 3 and counts_desc[1][1] == 2:
        triple_value = counts_desc[0][0]
        pair_value = counts_desc[1][0]
        ordered_values = [triple_value, triple_value, triple_value, pair_value, pair_value]
        return 6, [triple_value, pair_value], _select_cards_by_values(sorted_cards, ordered_values)

    if is_flush:
        return 5, values, sorted_cards[:5]

    if straight_high is not None:
        best_cards = _select_straight_cards(sorted_cards, straight_high)
        return 4, [straight_high], best_cards

    if counts_desc[0][1] == 3:
        triple_value = counts_desc[0][0]
        kickers = sorted((value for value, count in value_counts.items() if count == 1), reverse=True)
        ordered_values = [triple_value, triple_value, triple_value] + kickers[:2]
        return 3, [triple_value] + kickers[:2], _select_cards_by_values(sorted_cards, ordered_values)

    if counts_desc[0][1] == 2 and counts_desc[1][1] == 2:
        high_pair = max(counts_desc[0][0], counts_desc[1][0])
        low_pair = min(counts_desc[0][0], counts_desc[1][0])
        kicker = max(value for value, count in value_counts.items() if count == 1)
        ordered_values = [high_pair, high_pair, low_pair, low_pair, kicker]
        return 2, [high_pair, low_pair, kicker], _select_cards_by_values(sorted_cards, ordered_values)

    if counts_desc[0][1] == 2:
        pair_value = counts_desc[0][0]
        kickers = sorted((value for value, count in value_counts.items() if count == 1), reverse=True)
        ordered_values = [pair_value, pair_value] + kickers[:3]
        return 1, [pair_value] + kickers[:3], _select_cards_by_values(sorted_cards, ordered_values)

    return 0, values, sorted_cards[:5]


def describe_hand_rank(rank_value: int) -> str:
    return HAND_CATEGORY_NAMES.get(rank_value, "unknown")


def describe_hand_label(category: str) -> str:
    return HAND_CATEGORY_DISPLAY_NAMES.get(category, category.replace("_", " ").title())


def get_hand_rank(hole_cards: List[Card], community_cards: List[Card]) -> Tuple[int, List[Card]]:
    """
    Bestimmt den besten 5-Karten-Handrang aus 2-7 Karten.
    Rückgabe: (Rang 0-8, beste 5 Karten)
    """
    all_cards = [card for card in hole_cards + community_cards if card]
    if len(all_cards) < 5:
        best_cards = sorted(all_cards, key=lambda card: card.value, reverse=True)
        return 0, best_cards

    best_rank: Optional[Tuple[int, List[int], List[Card]]] = None
    for combo in combinations(all_cards, 5):
        current_rank = _evaluate_five_card_hand(list(combo))
        if best_rank is None or (current_rank[0], current_rank[1]) > (best_rank[0], best_rank[1]):
            best_rank = current_rank

    assert best_rank is not None
    return best_rank[0], best_rank[2]


def get_best_hand_details(hole_cards: List[Card], community_cards: List[Card]) -> Dict[str, object]:
    all_cards = [card for card in hole_cards + community_cards if card]
    if len(all_cards) < 5:
        best_cards = sorted(all_cards, key=lambda card: card.value, reverse=True)
        return {
            "rank_value": 0,
            "category": describe_hand_rank(0),
            "display_category": describe_hand_label(describe_hand_rank(0)),
            "best_cards": best_cards,
            "kickers": [card.value for card in best_cards],
        }

    best_rank: Optional[Tuple[int, List[int], List[Card]]] = None
    for combo in combinations(all_cards, 5):
        current_rank = _evaluate_five_card_hand(list(combo))
        if best_rank is None or (current_rank[0], current_rank[1]) > (best_rank[0], best_rank[1]):
            best_rank = current_rank

    assert best_rank is not None
    return {
        "rank_value": best_rank[0],
        "category": describe_hand_rank(best_rank[0]),
        "display_category": describe_hand_label(describe_hand_rank(best_rank[0])),
        "best_cards": best_rank[2],
        "kickers": best_rank[1],
    }


def evaluate_hand_strength(
    hole_cards: List[Card],
    community_cards: List[Card],
    num_opponent_cards: int = 0,
) -> float:
    """
    Schätzt die Handstärke als Wert zwischen 0 und 1.
    Das ist absichtlich heuristisch, aber deutlich robuster als die alte Platzhalter-Logik.
    """
    valid_hole_cards = [card for card in hole_cards if card]
    valid_community_cards = [card for card in community_cards if card]
    if len(valid_hole_cards) != 2:
        return 0.0

    if not valid_community_cards:
        high_card, low_card = sorted(valid_hole_cards, key=lambda card: card.value, reverse=True)
        gap = high_card.value - low_card.value
        is_pair = high_card.rank == low_card.rank
        is_suited = high_card.suit == low_card.suit
        is_connected = gap <= 1
        broadway_count = sum(1 for card in valid_hole_cards if card.value >= RANK_MAP["T"])

        score = 0.20
        score += (high_card.value / 12.0) * 0.22
        score += (low_card.value / 12.0) * 0.13
        if is_pair:
            score += 0.30 + (high_card.value / 12.0) * 0.16
        if is_suited:
            score += 0.08
        if is_connected:
            score += 0.06
        if gap == 2:
            score += 0.03
        score += broadway_count * 0.03
        return min(score, 1.0)

    details = get_best_hand_details(valid_hole_cards, valid_community_cards)
    rank_value = int(details["rank_value"])
    kickers = [int(value) for value in details["kickers"]]

    base_strength_by_rank = {
        0: 0.25,
        1: 0.46,
        2: 0.60,
        3: 0.69,
        4: 0.73,
        5: 0.79,
        6: 0.87,
        7: 0.95,
        8: 0.99,
    }
    strength = base_strength_by_rank.get(rank_value, 0.25)
    if kickers:
        strength += min(sum(kickers) / (len(kickers) * 12.0), 1.0) * 0.06

    max_community_cards = max(len(valid_community_cards), 1)
    street_boost = max_community_cards * 0.01
    opponent_penalty = min(num_opponent_cards, 8) * 0.015
    return max(0.0, min(strength + street_boost - opponent_penalty, 1.0))
