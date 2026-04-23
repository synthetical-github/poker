from __future__ import annotations

from typing import Dict

from utils.card_utils import RANK_MAP


STYLE_CATEGORY_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "tight_passive": {
        "premium_pair": 1.20,
        "medium_pair": 1.05,
        "small_pair": 0.95,
        "premium_broadway": 1.12,
        "strong_ace": 1.05,
        "weak_ace": 0.76,
        "strong_king": 0.92,
        "weak_king": 0.66,
        "broadway_mix": 0.90,
        "suited_connector": 0.72,
        "suited_gapper": 0.62,
        "suited_trash": 0.42,
        "offsuit_connector": 0.52,
        "trash": 0.34,
    },
    "balanced": {
        "premium_pair": 1.10,
        "medium_pair": 1.00,
        "small_pair": 0.98,
        "premium_broadway": 1.00,
        "strong_ace": 1.00,
        "weak_ace": 0.92,
        "strong_king": 0.96,
        "weak_king": 0.84,
        "broadway_mix": 0.96,
        "suited_connector": 0.96,
        "suited_gapper": 0.88,
        "suited_trash": 0.68,
        "offsuit_connector": 0.78,
        "trash": 0.56,
    },
    "loose_passive": {
        "premium_pair": 1.12,
        "medium_pair": 1.04,
        "small_pair": 1.00,
        "premium_broadway": 1.00,
        "strong_ace": 1.08,
        "weak_ace": 1.02,
        "strong_king": 1.00,
        "weak_king": 0.94,
        "broadway_mix": 0.98,
        "suited_connector": 1.08,
        "suited_gapper": 1.02,
        "suited_trash": 0.86,
        "offsuit_connector": 0.88,
        "trash": 0.72,
    },
    "tight_aggressive": {
        "premium_pair": 1.24,
        "medium_pair": 1.08,
        "small_pair": 0.98,
        "premium_broadway": 1.14,
        "strong_ace": 1.08,
        "weak_ace": 0.82,
        "strong_king": 1.02,
        "weak_king": 0.72,
        "broadway_mix": 1.00,
        "suited_connector": 0.92,
        "suited_gapper": 0.82,
        "suited_trash": 0.54,
        "offsuit_connector": 0.70,
        "trash": 0.42,
    },
    "loose_aggressive": {
        "premium_pair": 1.22,
        "medium_pair": 1.12,
        "small_pair": 1.06,
        "premium_broadway": 1.14,
        "strong_ace": 1.14,
        "weak_ace": 1.00,
        "strong_king": 1.06,
        "weak_king": 0.94,
        "broadway_mix": 1.04,
        "suited_connector": 1.20,
        "suited_gapper": 1.10,
        "suited_trash": 0.94,
        "offsuit_connector": 0.94,
        "trash": 0.78,
    },
}


def categorize_hand_key(hand_key: str) -> str:
    key = str(hand_key or "").strip()
    if not key:
        return "trash"

    if len(key) == 2 and key[0] == key[1]:
        pair_value = RANK_MAP.get(key[0], -1)
        if pair_value >= RANK_MAP["T"]:
            return "premium_pair"
        if pair_value >= RANK_MAP["6"]:
            return "medium_pair"
        return "small_pair"

    ranks = key[:2]
    suited = key.endswith("s")
    high_rank = ranks[0]
    low_rank = ranks[1]
    high_value = RANK_MAP.get(high_rank, -1)
    low_value = RANK_MAP.get(low_rank, -1)
    broadway_count = sum(1 for rank in ranks if RANK_MAP.get(rank, -1) >= RANK_MAP["T"])

    if high_rank == "A":
        if low_value >= RANK_MAP["T"] or (suited and low_value >= RANK_MAP["8"]):
            return "premium_broadway"
        if low_value >= RANK_MAP["6"] or suited:
            return "strong_ace"
        return "weak_ace"

    if broadway_count == 2:
        if high_rank in {"K", "Q"} and low_value >= RANK_MAP["J"]:
            return "premium_broadway"
        return "broadway_mix"

    if high_rank == "K":
        return "strong_king" if low_value >= RANK_MAP["8"] or suited else "weak_king"

    gap = high_value - low_value
    if suited and gap <= 1 and high_value >= RANK_MAP["6"]:
        return "suited_connector"
    if suited and gap == 2 and high_value >= RANK_MAP["7"]:
        return "suited_gapper"
    if suited and high_value >= RANK_MAP["8"]:
        return "suited_trash"
    if gap <= 1 and high_value >= RANK_MAP["7"]:
        return "offsuit_connector"
    return "trash"


def get_style_multiplier(style: str, category: str) -> float:
    normalized_style = str(style or "balanced").strip().lower()
    profile = STYLE_CATEGORY_MULTIPLIERS.get(normalized_style, STYLE_CATEGORY_MULTIPLIERS["balanced"])
    return float(profile.get(category, 1.0))
