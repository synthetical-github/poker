# live_analyzer.py
import cv2
import keyboard
import math
import re
import time
import threading
from typing import Any, Dict, Optional

from detectors.card_detector import CardDetector
from detectors.table_parser import TableParser
from action_executor import ActionExecutor
from strategy_engine import StrategyEngine
from voice_announcer import VoiceAnnouncer
from session_manager import SessionManager
from hotkeys import setup_hotkeys, remove_hotkeys
from overlay import OverlayWindow
from utils.logger import logger
from utils.config import LIVE_CONFIG
from utils.screen_utils import get_screenshot_for_processing, select_region_interactive


class LivePokerAnalyzer:
    def __init__(self, headless=False):
        self.headless = headless
        self.running = False
        self.paused = False
        self.auto_mode = False
        self.stop_event = threading.Event()

        self.card_detector = CardDetector()
        self.table_parser = TableParser()
        self.action_executor = ActionExecutor()
        self.strategy_engine = StrategyEngine()
        self.voice_announcer = VoiceAnnouncer()
        self.session_manager = SessionManager()

        self.analysis_count = 0
        self.error_count = 0
        self.last_action_taken = None
        self.current_game_state = None
        self.last_game_state_str = ""
        self.table_coords = None
        self.is_bot_thread_running = False
        self.last_screenshot = None
        self.last_summary_output = ""
        self.last_summary_time = 0.0
        self.summary_repeat_interval = 2.0
        self.last_state_signature = None
        self.pending_state_signature = None
        self.pending_state_count = 0
        self.required_stable_frames = 1
        self.last_strategy = None
        self.last_valid_stacks: Dict[str, Optional[float]] = {'hero': None, 'villain': None}

        # Screenshot-Aufnahme für N Runden (manuell via F9)
        self._capture_rounds_remaining = 0
        self._capture_round_count = 0
        self._capture_last_hole: tuple = ()
        self._capture_frame_index = 0
        # Auto-Screenshot bei jedem neuen Spielzustand
        self._auto_capture_last_sig: str = ""
        self._auto_capture_dir = 'auto_screenshots'
        import os as _os
        _os.makedirs(self._auto_capture_dir, exist_ok=True)

        self.overlay = OverlayWindow()
        if LIVE_CONFIG.get('show_overlay', True):
            self.overlay.start()

    def start_capture_rounds(self, rounds: int = 2):
        """Startet automatische Screenshot-Aufnahme für N komplette Runden."""
        import os
        self._capture_rounds_remaining = rounds
        self._capture_round_count = 0
        self._capture_last_hole = ()
        self._capture_frame_index = 0
        os.makedirs('capture_rounds', exist_ok=True)
        print(f"[BOT] Screenshot-Aufnahme gestartet: {rounds} Runden werden in capture_rounds/ gespeichert.", flush=True)

    def _maybe_capture_screenshot(self, game_state: dict):
        """Speichert Screenshot wenn Capture-Modus aktiv."""
        if self._capture_rounds_remaining <= 0:
            return
        if self.last_screenshot is None:
            return
        import cv2
        import time as _time

        # Neue Runde erkennen: Hole Cards haben sich geändert (neue Hand)
        current_hole = tuple(str(c) for c in game_state.get('hole_cards', []) if c)
        if current_hole and current_hole != self._capture_last_hole:
            if self._capture_last_hole:  # Nicht beim allerersten Frame
                self._capture_round_count += 1
                print(f"[BOT] Runde {self._capture_round_count} abgeschlossen.", flush=True)
                if self._capture_round_count >= self._capture_rounds_remaining + (
                    1 if self._capture_last_hole == () else 0
                ):
                    self._capture_rounds_remaining = 0
                    frames = self._capture_frame_index
                    print(
                        f"[BOT] Screenshot-Aufnahme beendet. {frames} Frames"
                        " gespeichert in capture_rounds/",
                        flush=True,
                    )
                    return
            self._capture_last_hole = current_hole

        ts = int(_time.time() * 1000)
        self._capture_frame_index += 1
        fname = f"capture_rounds/frame_{self._capture_frame_index:04d}_{ts}.png"
        cv2.imwrite(fname, self.last_screenshot)

    def _auto_capture_screenshot(self, game_state: dict):
        """Speichert automatisch einen Screenshot bei jedem neuen Spielzustand."""
        if self.last_screenshot is None:
            return
        import cv2 as _cv2
        import time as _time
        hole = tuple(str(c) for c in game_state.get('hole_cards', []) if c)
        board = tuple(str(c) for c in game_state.get('community_cards', []) if c)
        street = str(game_state.get('street', '') or '')
        sig = f"{hole}|{board}|{street}"
        if sig == self._auto_capture_last_sig:
            return
        self._auto_capture_last_sig = sig
        hole_str = ''.join(hole) if hole else 'XX'
        board_str = ''.join(board) if board else 'PRE'
        ts = int(_time.time())
        fname = f"{self._auto_capture_dir}/{ts}_{hole_str}_{board_str}.png"
        _cv2.imwrite(fname, self.last_screenshot)

    def _format_cards_list(self, cards) -> str:
        valid_cards = [str(card) for card in cards if card]
        return " ".join(valid_cards) if valid_cards else "-"

    def _infer_street_from_board_count(self, board_count: int) -> str:
        if board_count <= 0:
            return 'preflop'
        if board_count == 3:
            return 'flop'
        if board_count == 4:
            return 'turn'
        return 'river'

    def _format_cards(self, cards) -> str:
        if not cards:
            return "-"
        return " ".join(str(c) for c in cards)

    def _format_actions(self, actions) -> str:
        if not actions:
            return "-"
        return ", ".join(str(action).upper() for action in actions)

    def _format_percent(self, value: float) -> str:
        return f"{max(0.0, float(value or 0.0)) * 100:.1f}%"

    def _short_range_label(self, strategy: Dict[str, Any]) -> str:
        range_summary = strategy.get('range_summary', {}) or {}
        top_hands = range_summary.get('top_hands', []) or []
        if top_hands:
            labels = []
            for item in top_hands[:4]:
                if isinstance(item, dict):
                    labels.append(str(item.get('hand', '') or '').strip())
                else:
                    labels.append(str(item).strip())
            compact = " ".join(label for label in labels if label)
            if compact:
                return compact
        headline = str(range_summary.get('headline', '-') or '-')
        if '| top=' in headline:
            return headline.split('| top=', 1)[1].replace(',', ' ').strip()
        return headline

    def _get_hand_signal(self, strategy: Dict[str, Any]) -> str:
        hand_details = strategy.get('hand_details', {}) or {}
        rank_value = int(hand_details.get('rank_value', 0) or 0)
        hand_label = hand_details.get('display_category') or hand_details.get('category', 'Unknown')

        if rank_value >= 6:
            return f"MONSTER: {hand_label}"
        if rank_value >= 5:
            return f"SEHR STARK: {hand_label}"
        if rank_value >= 4:
            return f"STARK: {hand_label}"
        if rank_value >= 3:
            return f"GUT: {hand_label}"
        if rank_value >= 1:
            return f"HAND: {hand_label}"
        return f"HAND: {hand_label}"

    def _is_actionable_spot(self, game_state: Dict[str, Any]) -> bool:
        return bool(
            game_state.get('is_my_turn', False)
            and game_state.get('available_actions', [])
        )

    def _get_detected_blind_structure(self) -> Optional[tuple[float, float]]:
        title = str(LIVE_CONFIG.get('detected_window_title', '') or '')
        if not title:
            return None

        match = re.search(r'(\d+(?:[.,]\d+)?)\s*/\s*(\d+(?:[.,]\d+)?)', title)
        if not match:
            return None

        try:
            small_blind = float(match.group(1).replace(',', '.'))
            big_blind = float(match.group(2).replace(',', '.'))
        except ValueError:
            return None

        if small_blind <= 0 or big_blind <= 0:
            return None
        if big_blind < small_blind:
            small_blind, big_blind = big_blind, small_blind
        return round(small_blind, 2), round(big_blind, 2)

    def _get_blind_step(self, small_blind: float, big_blind: float) -> float:
        if small_blind <= 0 or big_blind <= 0:
            return 0.0

        if abs(small_blind - round(small_blind)) < 0.01 and abs(big_blind - round(big_blind)) < 0.01:
            step = math.gcd(int(round(small_blind)), int(round(big_blind)))
            return float(step or int(round(small_blind)) or 1)

        return round(min(small_blind, big_blind), 2)

    def _snap_to_blind_step(self, value: float, blind_step: float) -> float:
        numeric_value = max(0.0, float(value or 0.0))
        if numeric_value <= 0 or blind_step <= 0:
            return 0.0
        if numeric_value < blind_step * 0.6:
            return 0.0
        snapped = round(round(numeric_value / blind_step) * blind_step, 2)
        return snapped if snapped >= blind_step * 0.6 else 0.0

    def _build_live_summary(self, game_state: Dict[str, Any], strategy: Dict[str, Any]) -> str:
        hole_cards = self._format_cards(game_state.get('hole_cards', []))
        community_cards = self._format_cards(game_state.get('community_cards', []))
        available_actions = self._format_actions(game_state.get('available_actions', []))
        action = str(strategy.get('recommended_action', '?')).upper()
        amount = float(strategy.get('amount', 0.0) or 0.0)
        amount_suffix = f" {amount:.2f}" if amount > 0 else ""
        next_move = f"{action}{amount_suffix}"
        actionable_spot = self._is_actionable_spot(game_state)
        street = str(game_state.get('street', 'unknown')).upper()
        pot_size = float(game_state.get('pot_size', 0.0) or 0.0)
        to_call = float(game_state.get('to_call', 0.0) or 0.0)
        is_my_turn = "JA" if game_state.get('is_my_turn', False) else "NEIN"
        hand_category = strategy.get('hand_details', {}).get('display_category') or strategy.get('hand_details', {}).get('category', '-')
        board_texture = strategy.get('board_texture', {}).get('texture', '-')
        opponent_style = strategy.get('opponent_profile', {}).get('style', '-')
        range_label = self._short_range_label(strategy)
        monte_carlo = strategy.get('monte_carlo', {}) or {}
        hand_equity = float(monte_carlo.get('hand_equity', strategy.get('equity_proxy', 0.0)) or 0.0)
        range_equity = float(monte_carlo.get('range_equity', 0.0) or 0.0)
        pot_odds = float(strategy.get('pot_odds', game_state.get('to_call', 0.0)) or 0.0)
        solver_decision = strategy.get('solver_decision', {}) or {}
        solver_action = str(solver_decision.get('recommended_action', '-') or '-').upper()
        solver_amount = float(solver_decision.get('recommended_amount', 0.0) or 0.0)
        solver_suffix = f" {solver_amount:.2f}" if solver_amount > 0 else ""
        if solver_action not in {'', '-', 'NONE'}:
            solver_text = f"{solver_action}{solver_suffix}"
        else:
            solver_text = "preflop off" if street == "PREFLOP" else "n/a"
        policy_source = str(strategy.get('policy_source', '-') or '-')
        parser_confidence = float(strategy.get('parser_confidence', 0.0) or 0.0)

        draws = [
            label
            for label, enabled in {
                "FD": strategy.get('draws', {}).get('flush_draw', False),
                "OESD": strategy.get('draws', {}).get('open_ended_straight_draw', False),
                "GS": strategy.get('draws', {}).get('gutshot', False),
                "Combo": strategy.get('draws', {}).get('combo_draw', False),
            }.items()
            if enabled
        ]
        draw_text = ", ".join(draws) if draws else "-"
        reason = str(strategy.get('reason', '') or '-')
        hand_signal = self._get_hand_signal(strategy)
        board_segment = f" | BOARD {community_cards}" if community_cards != "-" else ""

        lines = [
            "",
            "=== Live Spot ===",
            f"{street} | {hole_cards}{board_segment}",
            hand_signal,
            f"Status: {'AM ZUG' if actionable_spot else 'WARTEN'} | Next: {next_move if actionable_spot else 'Beobachten'}",
            f"Pot {pot_size:.2f} | To Call {to_call:.2f} | Actions {available_actions} | My Turn {is_my_turn}",
            f"Hand {hand_category} | Texture {board_texture} | Draws {draw_text}",
            f"Equity hand {self._format_percent(hand_equity)}"
            f" | range {self._format_percent(range_equity)}"
            f" | odds {self._format_percent(pot_odds)}",
            f"Villain {opponent_style} | Range {range_label}",
            f"Solver {solver_text} | Policy {policy_source} | Parser {parser_confidence:.2f}",
            f"Why: {reason}",
        ]
        return "\n".join(lines)

    def _build_recommendation_line(self, game_state: Dict[str, Any], strategy: Dict[str, Any]) -> str:
        street = str(game_state.get('street', 'unknown') or 'unknown').upper()
        hole_cards = self._format_cards(game_state.get('hole_cards', []))
        community_cards = self._format_cards(game_state.get('community_cards', []))
        board_segment = f" | BOARD {community_cards}" if community_cards != "-" else ""
        action = str(strategy.get('recommended_action', '?')).upper()
        amount = float(strategy.get('amount', 0.0) or 0.0)
        amount_suffix = f" {amount:.2f}" if amount > 0 else ""
        next_move = f"{action}{amount_suffix}"
        opponent_style = str(strategy.get('opponent_profile', {}).get('style', '-') or '-')
        range_label = self._short_range_label(strategy)
        monte_carlo = strategy.get('monte_carlo', {}) or {}
        hand_equity = float(monte_carlo.get('hand_equity', strategy.get('equity_proxy', 0.0)) or 0.0)
        pot_odds = float(strategy.get('pot_odds', game_state.get('to_call', 0.0)) or 0.0)
        solver_decision = strategy.get('solver_decision', {}) or {}
        solver_action = str(solver_decision.get('recommended_action', '-') or '-').upper()
        solver_segment = f" | SOLVER {solver_action}" if solver_action not in {'', '-', 'NONE'} else ""
        prefix = f"{street} | {hole_cards}{board_segment}"
        if self._is_actionable_spot(game_state):
            return (
                f"{prefix} | NEXT: {next_move} | EQ {self._format_percent(hand_equity)} | "
                f"ODDS {self._format_percent(pot_odds)} | V {opponent_style} | RANGE {range_label}{solver_segment}"
            )
        return (
            f"{prefix} | NEXT: WARTEN | EQ {self._format_percent(hand_equity)} | "
            f"V {opponent_style} | RANGE {range_label} | Spot nicht bestaetigt"
        )

    def _print_live_summary(self, game_state: Dict[str, Any], strategy: Dict[str, Any]):
        if LIVE_CONFIG.get('show_live_summary', False):
            summary = self._build_live_summary(game_state, strategy)
        elif LIVE_CONFIG.get('show_recommendation_line', True):
            summary = self._build_recommendation_line(game_state, strategy)
        else:
            return

        now = time.time()
        repeat_interval = (
            self.summary_repeat_interval
            if self._is_actionable_spot(game_state)
            else max(self.summary_repeat_interval, 5.0)
        )
        should_repeat = (
            summary == self.last_summary_output
            and (now - self.last_summary_time) >= repeat_interval
        )

        if summary != self.last_summary_output or should_repeat:
            print(summary, flush=True)
            self.last_summary_output = summary
            self.last_summary_time = now

    def _update_overlay(self, game_state: Dict[str, Any], strategy: Dict[str, Any]):
        try:
            monte_carlo = strategy.get('monte_carlo', {}) or {}
            hand_equity = float(monte_carlo.get('hand_equity', strategy.get('equity_proxy', 0.0)) or 0.0)
            pot_odds = float(strategy.get('pot_odds', game_state.get('to_call', 0.0)) or 0.0)
            solver_decision = strategy.get('solver_decision', {}) or {}
            solver_action = str(solver_decision.get('recommended_action', '') or '')
            board = game_state.get('community_cards', [])
            street = str(game_state.get('street', self._infer_street_from_board_count(len(board)))).upper()
            hole = self._format_cards(game_state.get('hole_cards', []))
            board_str = self._format_cards(board) if board else ''
            self.overlay.update({
                'action': str(strategy.get('recommended_action', '---')),
                'amount': float(strategy.get('amount', 0.0) or 0.0),
                'is_my_turn': bool(self._is_actionable_spot(game_state)),
                'street': street,
                'hole': hole,
                'board': board_str,
                'equity': hand_equity,
                'odds': pot_odds,
                'solver_action': solver_action,
                'villain_style': str(strategy.get('opponent_profile', {}).get('style', '') or ''),
            })
        except Exception as e:
            logger.debug(f"Overlay update error: {e}")

    def toggle_overlay(self):
        if self.overlay._running:
            self.overlay.stop()
            logger.info("Overlay ausgeblendet.")
        else:
            self.overlay.start()
            logger.info("Overlay eingeblendet.")

    def _build_state_signature(self, game_state: Dict[str, Any]) -> tuple:
        return (
            tuple(str(card) for card in game_state.get('hole_cards', [])),
            tuple(str(card) for card in game_state.get('community_cards', [])),
            round(float(game_state.get('pot_size', 0.0) or 0.0), 2),
            round(float(game_state.get('to_call', 0.0) or 0.0), 2),
            str(game_state.get('street', 'unknown')),
            str(game_state.get('position', 'unknown')),
            tuple(sorted(str(action) for action in game_state.get('available_actions', []))),
            bool(game_state.get('is_my_turn', False)),
        )

    def _requires_confirmed_buttons(self, game_state: Dict[str, Any]) -> bool:
        return bool(game_state.get('is_my_turn', False))

    def _sanitize_stack_value(
        self,
        raw_stack: Optional[float],
        role: str,
        other_stack: Optional[float],
        pot_size: float,
    ) -> Optional[float]:
        if raw_stack is None:
            return None

        try:
            value = round(float(raw_stack), 2)
        except (TypeError, ValueError):
            return None
        if value <= 0:
            return None

        layout_name = getattr(self.table_parser, 'layout_name', '') or ''
        previous_state = self.current_game_state or {}
        previous_stack = previous_state.get(f'{role}_stack')
        try:
            previous_stack = round(float(previous_stack), 2) if previous_stack is not None else None
        except (TypeError, ValueError):
            previous_stack = None

        candidates = {value}
        if layout_name == 'heads_up' and value >= 1000:
            candidates.add(round(value / 10.0, 1))
            candidates.add(float(int(value // 10)))
            digits = ''.join(char for char in str(int(value)) if char.isdigit())
            if len(digits) >= 4:
                candidates.add(float(int(digits[:4])))

        if previous_stack and previous_stack > 0:
            inflation_cap = max(previous_stack * 3.0, float(other_stack or 0.0) * 3.0, 5000.0)
            if value > inflation_cap:
                plausible_candidates = [candidate for candidate in candidates if candidate <= inflation_cap]
                if plausible_candidates:
                    best = min(plausible_candidates, key=lambda candidate: abs(candidate - previous_stack))
                    if abs(best - previous_stack) <= max(80.0, previous_stack * 0.08):
                        return round(previous_stack, 2)
                    return round(best, 2)
                return round(previous_stack, 2)

        if layout_name == 'heads_up':
            static_cap = max(float(other_stack or 0.0) * 8.0, 5000.0)
            if value > static_cap:
                plausible_candidates = [candidate for candidate in candidates if candidate <= static_cap]
                if plausible_candidates:
                    best = min(plausible_candidates, key=lambda candidate: abs(candidate - (previous_stack or other_stack or candidate)))
                    return round(best, 2)
                return round(previous_stack, 2) if previous_stack else None

        return round(value, 2)

    def _fallback_previous_stack(
        self,
        role: str,
        raw_stack: Optional[float],
    ) -> Optional[float]:
        if raw_stack is not None:
            try:
                raw_value = round(float(raw_stack), 2)
            except (TypeError, ValueError):
                raw_value = None
            if raw_value and raw_value > 0:
                return raw_value

        if raw_stack is not None:
            return raw_stack

        previous_state = self.current_game_state or {}
        previous_stack = previous_state.get(f'{role}_stack')
        try:
            previous_value = round(float(previous_stack), 2) if previous_stack is not None else None
        except (TypeError, ValueError):
            previous_value = None
        if previous_value and previous_value > 0:
            return previous_value

        last_known_stack = self.last_valid_stacks.get(role)
        try:
            last_known_value = round(float(last_known_stack), 2) if last_known_stack is not None else None
        except (TypeError, ValueError):
            last_known_value = None
        return last_known_value if last_known_value and last_known_value > 0 else None

    def _remember_valid_stacks(
        self,
        hero_stack: Optional[float],
        villain_stack: Optional[float],
    ) -> None:
        for role, stack in (('hero', hero_stack), ('villain', villain_stack)):
            try:
                value = round(float(stack), 2) if stack is not None else None
            except (TypeError, ValueError):
                value = None
            if value and value > 0:
                self.last_valid_stacks[role] = value

    def _same_hole_cards(
        self,
        first_state: Optional[Dict[str, Any]],
        second_state: Optional[Dict[str, Any]],
    ) -> bool:
        if not first_state or not second_state:
            return False
        first_hole = tuple(str(card) for card in first_state.get('hole_cards', []) if card)
        second_hole = tuple(str(card) for card in second_state.get('hole_cards', []) if card)
        return len(first_hole) == 2 and first_hole == second_hole

    def _is_same_hand_stack_context(
        self,
        previous_state: Optional[Dict[str, Any]],
        new_state: Optional[Dict[str, Any]],
    ) -> bool:
        if not previous_state or not new_state:
            return False

        previous_hole = tuple(str(card) for card in previous_state.get('hole_cards', []) if card)
        new_hole = tuple(str(card) for card in new_state.get('hole_cards', []) if card)
        if len(previous_hole) != 2 or previous_hole != new_hole:
            return False

        previous_board = tuple(str(card) for card in previous_state.get('community_cards', []) if card)
        new_board = tuple(str(card) for card in new_state.get('community_cards', []) if card)
        prev_count = len(previous_board)
        new_count = len(new_board)

        if prev_count > 0 and new_count == 0:
            return False
        if prev_count == new_count and prev_count >= 3 and previous_board != new_board:
            return False
        if new_count < prev_count:
            return False
        if prev_count == 0 and new_count not in {0, 3}:
            return False
        if prev_count == 3 and new_count not in {3, 4}:
            return False
        if prev_count == 4 and new_count not in {4, 5}:
            return False
        if prev_count == 5 and new_count != 5:
            return False

        return True

    def _stabilize_stack_transition_value(
        self,
        role: str,
        previous_state: Dict[str, Any],
        new_state: Dict[str, Any],
    ) -> Optional[float]:
        previous_stack = previous_state.get(f'{role}_stack')
        current_stack = new_state.get(f'{role}_stack')

        try:
            previous_value = round(float(previous_stack), 2) if previous_stack is not None else None
        except (TypeError, ValueError):
            previous_value = None
        try:
            current_value = round(float(current_stack), 2) if current_stack is not None else None
        except (TypeError, ValueError):
            current_value = None

        if not previous_value or not current_value:
            return current_value

        if not self._is_same_hand_stack_context(previous_state, new_state):
            return current_value

        previous_pot = max(0.0, float(previous_state.get('pot_size', 0.0) or 0.0))
        current_pot = max(0.0, float(new_state.get('pot_size', 0.0) or 0.0))
        previous_to_call = max(0.0, float(previous_state.get('to_call', 0.0) or 0.0))
        current_to_call = max(0.0, float(new_state.get('to_call', 0.0) or 0.0))
        previous_bet = max(0.0, float(previous_state.get('current_bet', 0.0) or 0.0))
        current_bet = max(0.0, float(new_state.get('current_bet', 0.0) or 0.0))

        previous_action_state = previous_state.get('action_state', {}) or {}
        new_action_state = new_state.get('action_state', {}) or {}
        previous_raise_to = max(0.0, float(previous_action_state.get('raise_to_amount', 0.0) or 0.0))
        current_raise_to = max(0.0, float(new_action_state.get('raise_to_amount', 0.0) or 0.0))

        layout_name = getattr(self.table_parser, 'layout_name', '') or ''
        increase_tolerance = 1.0 if layout_name == 'heads_up' else 0.05
        if current_value > previous_value + increase_tolerance:
            return previous_value

        pot_growth = max(0.0, current_pot - previous_pot)
        stack_drop = previous_value - current_value
        if stack_drop <= 0:
            return current_value

        max_open_commit = max(
            previous_to_call,
            current_to_call,
            previous_bet,
            current_bet,
            previous_raise_to - previous_to_call if previous_raise_to > previous_to_call else 0.0,
            current_raise_to - current_to_call if current_raise_to > current_to_call else 0.0,
        )
        max_reasonable_drop = max(1.0, pot_growth + max_open_commit)
        if stack_drop > max_reasonable_drop + 1.0:
            return previous_value

        return current_value

    def _stabilize_state_from_previous(
        self,
        game_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        previous_state = self.current_game_state
        if not previous_state:
            self._remember_valid_stacks(game_state.get('hero_stack'), game_state.get('villain_stack'))
            return game_state

        stabilized = dict(game_state)
        previous_hole_cards = [card for card in previous_state.get('hole_cards', []) if card]
        current_hole_cards = [card for card in stabilized.get('hole_cards', []) if card]
        previous_board_cards = [card for card in previous_state.get('community_cards', []) if card]
        current_board_cards = [card for card in stabilized.get('community_cards', []) if card]
        previous_board_count = len(previous_board_cards)
        current_board_count = len(current_board_cards)

        if len(previous_hole_cards) == 2:
            board_grew = current_board_count > previous_board_count
            preserve_hole_cards = current_board_count > 0 and (board_grew or previous_board_count > 0)
            if len(current_hole_cards) != 2 and preserve_hole_cards:
                stabilized['hole_cards'] = previous_state.get('hole_cards', [])
                current_hole_cards = [card for card in stabilized.get('hole_cards', []) if card]
            elif current_hole_cards != previous_hole_cards and preserve_hole_cards:
                stabilized['hole_cards'] = previous_state.get('hole_cards', [])
                current_hole_cards = [card for card in stabilized.get('hole_cards', []) if card]

        same_hole_cards = (
            len(previous_hole_cards) == 2
            and len(current_hole_cards) == 2
            and tuple(str(card) for card in current_hole_cards) == tuple(str(card) for card in previous_hole_cards)
        )
        if same_hole_cards and previous_board_cards:
            if current_board_count == 0:
                stabilized['community_cards'] = []
            elif current_board_count not in {previous_board_count, previous_board_count + 1}:
                stabilized['community_cards'] = previous_state.get('community_cards', [])
            elif current_board_count < previous_board_count:
                stabilized['community_cards'] = previous_state.get('community_cards', [])

        stabilized['hero_stack'] = self._stabilize_stack_transition_value('hero', previous_state, stabilized)
        stabilized['villain_stack'] = self._stabilize_stack_transition_value('villain', previous_state, stabilized)

        stabilized_board_cards = [card for card in stabilized.get('community_cards', []) if card]
        stabilized['street'] = self._infer_street_from_board_count(len(stabilized_board_cards))

        previous_actions = list(previous_state.get('available_actions', []))
        same_street = str(previous_state.get('street', 'unknown')) == str(stabilized.get('street', 'unknown'))
        same_board = tuple(str(card) for card in previous_state.get('community_cards', []) if card) == tuple(
            str(card) for card in stabilized.get('community_cards', []) if card
        )
        previous_to_call = round(float(previous_state.get('to_call', 0.0) or 0.0), 2)
        current_to_call = round(float(stabilized.get('to_call', 0.0) or 0.0), 2)
        previous_pot = round(float(previous_state.get('pot_size', 0.0) or 0.0), 2)
        current_pot = round(float(stabilized.get('pot_size', 0.0) or 0.0), 2)
        previous_hero_stack = previous_state.get('hero_stack')
        current_hero_stack = stabilized.get('hero_stack')
        hero_stack_stable = (
            previous_hero_stack is not None
            and current_hero_stack is not None
            and abs(float(previous_hero_stack) - float(current_hero_stack)) <= max(1.0, float(previous_to_call or 0.0) * 2.0)
        ) or (previous_hero_stack is None or current_hero_stack is None)
        if (
            not stabilized.get('available_actions')
            and previous_state.get('buttons_confirmed', False)
            and previous_state.get('is_my_turn', False)
            and same_street
            and same_board
            and same_hole_cards
            and abs(previous_to_call - current_to_call) <= max(1.0, current_to_call, previous_to_call)
            and abs(previous_pot - current_pot) <= max(2.0, previous_to_call, current_to_call, previous_pot * 0.35)
            and hero_stack_stable
        ):
            stabilized['available_actions'] = previous_actions
            stabilized['buttons_confirmed'] = True
            stabilized['is_my_turn'] = True
            stabilized_action_state = dict(stabilized.get('action_state', {}) or {})
            previous_action_state = previous_state.get('action_state', {}) or {}
            stabilized_action_state.setdefault('available_actions', previous_actions)
            if float(stabilized_action_state.get('call_amount', 0.0) or 0.0) <= 0:
                stabilized_action_state['call_amount'] = previous_action_state.get('call_amount', 0.0)
            if float(stabilized_action_state.get('raise_to_amount', 0.0) or 0.0) <= 0:
                stabilized_action_state['raise_to_amount'] = previous_action_state.get('raise_to_amount', 0.0)
            stabilized_action_state['buttons_confirmed'] = True
            stabilized_action_state['is_my_turn'] = True
            stabilized['action_state'] = stabilized_action_state

        self._remember_valid_stacks(stabilized.get('hero_stack'), stabilized.get('villain_stack'))
        return stabilized

    def _sanitize_game_state(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = dict(game_state)
        layout_name = getattr(self.table_parser, 'layout_name', '') or ''

        # Entferne Board-Karten, die identisch zu einer Hero-Hole-Card sind (physikalisch unmöglich)
        hole_cards = [card for card in sanitized.get('hole_cards', []) if card]
        community_cards = [card for card in sanitized.get('community_cards', []) if card]
        if hole_cards and community_cards:
            hole_card_strs = {str(c) for c in hole_cards}
            seen_board: set = set()
            filtered_board = []
            for card in community_cards:
                card_str = str(card)
                if card_str in hole_card_strs:
                    logger.debug(f"Duplikat-Karte entfernt (auch in Hole Cards): {card_str}")
                    continue
                if card_str in seen_board:
                    logger.debug(f"Doppelte Board-Karte entfernt: {card_str}")
                    continue
                seen_board.add(card_str)
                filtered_board.append(card)
            sanitized['community_cards'] = filtered_board

        available_actions = list(dict.fromkeys(sanitized.get('available_actions', [])))
        hero_stack = sanitized.get('hero_stack')
        villain_stack = sanitized.get('villain_stack')
        num_players_remaining = int(sanitized.get('num_players_remaining', 0) or 0)
        pot_size = max(0.0, float(sanitized.get('pot_size', 0.0) or 0.0))
        to_call = max(0.0, float(sanitized.get('to_call', 0.0) or 0.0))
        current_bet = max(0.0, float(sanitized.get('current_bet', 0.0) or 0.0))
        action_state = sanitized.get('action_state', {}) or {}
        raise_to_amount = max(0.0, float(action_state.get('raise_to_amount', 0.0) or 0.0))
        call_amount = max(0.0, float(action_state.get('call_amount', 0.0) or 0.0))
        betting_context = max(pot_size, to_call, current_bet)
        if betting_context < 1.0:
            betting_unit = 0.05
        elif betting_context < 5.0:
            betting_unit = 0.10
        else:
            betting_unit = 1.0
        blind_structure = self._get_detected_blind_structure() if layout_name == 'heads_up' else None
        blind_step = 0.0
        if blind_structure:
            blind_step = self._get_blind_step(*blind_structure)
            if blind_step > 0:
                betting_unit = max(betting_unit, blind_step)

        hero_stack = self._fallback_previous_stack('hero', hero_stack)
        villain_stack = self._fallback_previous_stack('villain', villain_stack)
        hero_stack = self._sanitize_stack_value(hero_stack, 'hero', villain_stack, pot_size)
        villain_stack = self._sanitize_stack_value(villain_stack, 'villain', hero_stack, pot_size)
        sanitized['hero_stack'] = hero_stack
        sanitized['villain_stack'] = villain_stack

        if call_amount > 0:
            to_call = call_amount

        if 'check' in available_actions and 'call' not in available_actions:
            to_call = 0.0

        if blind_step > 0:
            pot_size = self._snap_to_blind_step(pot_size, blind_step)
            to_call = self._snap_to_blind_step(to_call, blind_step)
            current_bet = self._snap_to_blind_step(current_bet, blind_step)
            raise_to_amount = self._snap_to_blind_step(raise_to_amount, blind_step)

        if layout_name in {'heads_up', 'acipayam_heads_up'}:
            if 'check' in available_actions and to_call <= 0:
                current_bet = 0.0
            elif 'call' in available_actions and to_call > 0:
                current_bet = max(current_bet, to_call)

        if hero_stack and hero_stack > 0:
            max_stack_commit = max(hero_stack * 1.05, pot_size * 6.0, 1.0)
            if to_call > max_stack_commit:
                to_call = 0.0
            if current_bet > max_stack_commit:
                current_bet = 0.0
            if raise_to_amount > max_stack_commit:
                raise_to_amount = 0.0

        if villain_stack and villain_stack > 0 and to_call > villain_stack * 1.5:
            to_call = 0.0

        if to_call > 0:
            max_reasonable_current_bet = max(to_call * 2.0, pot_size * 2.5, betting_unit * 6.0)
            if current_bet > max_reasonable_current_bet:
                current_bet = to_call
        elif current_bet > max(pot_size * 2.5, betting_unit * 6.0):
            current_bet = 0.0

        if to_call > 0 and 'call' not in available_actions:
            if 'check' in available_actions:
                to_call = 0.0
            elif 'fold' in available_actions and len(available_actions) <= 2:
                available_actions = []

        if current_bet > 0 and (pot_size <= 0 or ('check' in available_actions and to_call <= 0)):
            current_bet = 0.0

        sanitized['pot_size'] = round(pot_size, 2)
        sanitized['to_call'] = round(to_call, 2)
        sanitized['current_bet'] = round(current_bet, 2)
        sanitized['available_actions'] = available_actions
        sanitized['num_players_remaining'] = num_players_remaining

        if 'action_state' in sanitized:
            sanitized['action_state'] = dict(action_state)
            sanitized['action_state']['call_amount'] = sanitized['to_call']
            sanitized['action_state']['raise_to_amount'] = round(raise_to_amount, 2)

        return sanitized

    def _is_valid_board_count(self, community_cards) -> bool:
        return len([card for card in community_cards if card]) in {0, 3, 4, 5}

    def _is_plausible_state(self, game_state: Dict[str, Any]) -> bool:
        layout_name = getattr(self.table_parser, 'layout_name', '') or ''
        hole_cards = [card for card in game_state.get('hole_cards', []) if card]
        community_cards = [card for card in game_state.get('community_cards', []) if card]
        num_players_remaining = int(game_state.get('num_players_remaining', 0) or 0)

        if len(hole_cards) != 2:
            return False
        if not self._is_valid_board_count(community_cards):
            return False
        if layout_name in {'heads_up', 'acipayam_heads_up'} and 0 < num_players_remaining < 2:
            return False

        hero_stack = game_state.get('hero_stack')
        villain_stack = game_state.get('villain_stack')
        pot_size = float(game_state.get('pot_size', 0.0) or 0.0)
        to_call = float(game_state.get('to_call', 0.0) or 0.0)
        current_bet = float(game_state.get('current_bet', 0.0) or 0.0)
        available_actions = set(game_state.get('available_actions', []))
        buttons_confirmed = bool(game_state.get('buttons_confirmed', False))
        blind_structure = self._get_detected_blind_structure() if layout_name == 'heads_up' else None
        blind_step = self._get_blind_step(*blind_structure) if blind_structure else 0.0

        if blind_step > 0 and buttons_confirmed:
            if 'check' in available_actions and to_call > 0:
                return False
            if 'call' in available_actions and to_call <= 0:
                return False
            if to_call > 0 and pot_size < to_call:
                return False
            if current_bet > 0 and current_bet < to_call:
                return False

        if hero_stack and hero_stack > 0:
            if to_call > hero_stack * 1.1:
                return False
            if current_bet > hero_stack * 2.5:
                return False
            if pot_size > max(hero_stack * 4.0, 20.0):
                return False

        if villain_stack and villain_stack > 0 and pot_size > (hero_stack or villain_stack) + villain_stack:
            return False

        return True

    def _is_plausible_transition(self, previous_state: Optional[Dict[str, Any]], new_state: Dict[str, Any]) -> bool:
        if previous_state is None:
            return True

        previous_hole = tuple(str(card) for card in previous_state.get('hole_cards', []) if card)
        new_hole = tuple(str(card) for card in new_state.get('hole_cards', []) if card)
        previous_board = [card for card in previous_state.get('community_cards', []) if card]
        new_board = [card for card in new_state.get('community_cards', []) if card]
        prev_count = len(previous_board)
        new_count = len(new_board)
        previous_board_signature = tuple(str(card) for card in previous_board)
        new_board_signature = tuple(str(card) for card in new_board)

        if prev_count > 0 and new_count == 0 and str(new_state.get('street', 'unknown')) == 'preflop':
            return True

        if new_hole != previous_hole:
            return len(new_board) == 0

        if prev_count == new_count and prev_count >= 3 and previous_board_signature != new_board_signature:
            if new_count == 3:
                return True
            return False

        if new_count < prev_count:
            return False
        if prev_count == 0 and new_count not in {0, 3}:
            return False
        if prev_count == 3 and new_count not in {3, 4}:
            return False
        if prev_count == 4 and new_count not in {4, 5}:
            return False
        if prev_count == 5 and new_count != 5:
            return False

        same_street = str(previous_state.get('street', 'unknown')) == str(new_state.get('street', 'unknown'))
        if same_street and prev_count == new_count:
            prev_actions = set(previous_state.get('available_actions', []))
            new_actions = set(new_state.get('available_actions', []))
            prev_to_call = round(float(previous_state.get('to_call', 0.0) or 0.0), 2)
            new_to_call = round(float(new_state.get('to_call', 0.0) or 0.0), 2)
            prev_hero_stack = previous_state.get('hero_stack')
            new_hero_stack = new_state.get('hero_stack')
            prev_my_turn = bool(previous_state.get('is_my_turn', False))
            new_my_turn = bool(new_state.get('is_my_turn', False))

            if prev_count == 0 and prev_my_turn and new_my_turn:
                preflop_reset = (
                    (prev_to_call > 0 and new_to_call == 0)
                    or (prev_to_call == 0 and new_to_call > 0)
                    or (('call' in prev_actions or 'check' in prev_actions) and 'raise' in new_actions and 'raise' not in prev_actions)
                    or (('raise' in prev_actions or 'bet' in prev_actions) and 'call' in new_actions and 'call' not in prev_actions)
                )
                if preflop_reset:
                    return False

            if prev_my_turn and new_my_turn:
                has_stack_change = (
                    prev_hero_stack is not None
                    and new_hero_stack is not None
                    and abs(float(prev_hero_stack) - float(new_hero_stack)) >= 0.08
                )
                changed_from_facing_bet = prev_to_call > 0 and new_to_call == 0
                changed_to_facing_bet = prev_to_call == 0 and new_to_call > 0

                if (changed_from_facing_bet or changed_to_facing_bet) and not has_stack_change:
                    return False
                if ('call' in prev_actions and 'call' not in new_actions and 'check' in new_actions and not has_stack_change):
                    return False
                if ('check' in prev_actions and 'call' in new_actions and not has_stack_change):
                    return False

        return True

    def _get_effective_table_region(self, screenshot) -> Optional[tuple]:
        if screenshot is None:
            return None

        h, w = screenshot.shape[:2]
        if w <= 0 or h <= 0:
            return None

        # Im Window-Modus soll der aktuelle Capture standardmäßig bereits das App-Fenster sein.
        # Deshalb nehmen wir den kompletten Screenshot als Arbeitsfläche.
        return (0, 0, w, h)

    def _get_game_state(self) -> Optional[Dict[str, Any]]:
        screenshot = get_screenshot_for_processing()
        if screenshot is None:
            logger.error("Konnte keinen Screenshot erstellen. Analyse wird übersprungen.")
            return None

        self.last_screenshot = screenshot
        return self._build_game_state_from_screenshot(screenshot)

    def _build_game_state_from_screenshot(self, screenshot) -> Optional[Dict[str, Any]]:
        if screenshot is None:
            return None

        effective_region = self._get_effective_table_region(screenshot)
        if not effective_region:
            logger.error("Konnte keinen gültigen Analysebereich bestimmen.")
            return None

        self.table_coords = effective_region
        frame_region = effective_region

        table_info = self.table_parser.parse_table(screenshot, self.table_coords)
        action_state = self.action_executor.read_action_state(screenshot, base_region=frame_region)

        hole_cards = self.card_detector.detect_hole_cards(screenshot, self.table_coords)
        community_cards = self.card_detector.detect_community_cards(screenshot, self.table_coords)

        community_count = len([card for card in community_cards if card])
        detected_street = table_info.get('street', 'preflop')
        if community_count == 0:
            detected_street = 'preflop'
        elif community_count == 3:
            detected_street = 'flop'
        elif community_count == 4:
            detected_street = 'turn'
        elif community_count >= 5:
            detected_street = 'river'

        game_state = {
            'hole_cards': hole_cards,
            'community_cards': community_cards,
            'pot_size': table_info.get('pot_size', 0.0),
            'to_call': action_state.get('call_amount', 0.0) or table_info.get('to_call', 0.0),
            'current_bet': table_info.get('current_bet', 0.0),
            'num_players_remaining': table_info.get('num_players', 9),
            'position': table_info.get('position', 'unknown'),
            'street': detected_street,
            'dealer_button_pos': table_info.get('dealer_button_pos'),
            'active_player_turn': table_info.get('active_player_turn'),
            'player_info': table_info.get('player_info', []),
            'roi_regions': table_info.get('roi_regions', {}),
            'card_regions': {
                'hero_hole_cards': self.card_detector.get_hole_card_regions(self.table_coords),
                'community_cards': self.card_detector.get_community_card_regions(self.table_coords),
            },
            'action_regions': self.action_executor.get_button_regions(base_region=frame_region),
            'available_actions': action_state.get('available_actions', []),
            'buttons_confirmed': bool(action_state.get('buttons_confirmed', False)),
            'action_state': action_state,
        }

        hero_stack = None
        villain_stack = None
        for player in game_state.get('player_info', []):
            role = str(player.get('role', '')).lower()
            chips = float(player.get('chips', 0.0) or 0.0)
            if chips <= 0:
                continue
            if role == 'hero':
                hero_stack = chips
            elif role == 'villain':
                villain_stack = chips

        game_state['hero_stack'] = hero_stack
        game_state['villain_stack'] = villain_stack

        active_player_turn = game_state.get('active_player_turn')
        game_state['is_my_turn'] = (
            active_player_turn is not None
            and active_player_turn >= 0
            and active_player_turn == self.get_my_player_index()
        )

        if action_state.get('is_my_turn'):
            game_state['is_my_turn'] = True

        sanitized_state = self._sanitize_game_state(game_state)
        return self._stabilize_state_from_previous(sanitized_state)

    def analyze_screenshot(self, screenshot) -> Optional[Dict[str, Any]]:
        game_state = self._build_game_state_from_screenshot(screenshot)
        if not game_state:
            return None

        strategy = self.strategy_engine.calculate_strategy(
            hole_cards=game_state['hole_cards'],
            community_cards=game_state['community_cards'],
            table_info=game_state,
        )

        return {
            'game_state': game_state,
            'strategy': strategy,
        }

    def analyze_loop(self):
        self.running = True
        self.is_bot_thread_running = True
        logger.info("Live-Analyse gestartet.")

        while self.running and not self.stop_event.is_set():
            if not self.paused:
                start_time = time.time()
                try:
                    game_state = self._get_game_state()
                except Exception as e:
                    print(f"[BOT ERROR] Fehler beim Screenshot/Analyse: {e}", flush=True)
                    logger.error(f"Fehler in analyze_loop: {e}", exc_info=True)
                    time.sleep(1)
                    continue

                if game_state:
                    if not self._is_plausible_state(game_state):
                        logger.debug("Verwerfe unplausiblen Spielzustand.")
                        if self.current_game_state and self.last_strategy and self.current_game_state.get('is_my_turn', False):
                            self._print_live_summary(self.current_game_state, self.last_strategy)
                        time.sleep(0.1)
                        continue

                    state_signature = self._build_state_signature(game_state)

                    # Neue Community-Karte erkannt → Strategie-Cache zurücksetzen
                    _new_board_count = len([c for c in game_state.get('community_cards', []) if c])
                    _prev_board_count = len([c for c in (self.current_game_state or {}).get('community_cards', []) if c])
                    if _new_board_count > _prev_board_count:
                        logger.info(
                            f"[BOT] Neue Boardkarte erkannt"
                            f" ({_prev_board_count}→{_new_board_count})."
                            " Strategie-Cache zurückgesetzt."
                        )
                        self.last_strategy = None
                        self.last_state_signature = None
                        self.last_summary_output = ""

                    if state_signature != self.pending_state_signature:
                        self.pending_state_signature = state_signature
                        self.pending_state_count = 1
                        print(f"[BOT] Neuer Zustand erkannt: {state_signature[:3]}", flush=True)
                    else:
                        self.pending_state_count += 1

                    required_frames = 1 if game_state.get('buttons_confirmed', False) else self.required_stable_frames
                    if self.pending_state_count < required_frames:
                        if self.current_game_state and self.last_strategy and self.current_game_state.get('is_my_turn', False):
                            self._print_live_summary(self.current_game_state, self.last_strategy)
                        time.sleep(0.05)
                        continue

                    if state_signature != self.last_state_signature:
                        if not self._is_plausible_transition(self.current_game_state, game_state):
                            logger.debug("Verwerfe unplausiblen Zustandswechsel innerhalb derselben Hand.")
                            self.pending_state_signature = None
                            self.pending_state_count = 0
                            if self.current_game_state and self.last_strategy and self.current_game_state.get('is_my_turn', False):
                                self._print_live_summary(self.current_game_state, self.last_strategy)
                            time.sleep(0.1)
                            continue

                        requires_confirmed_buttons = self._requires_confirmed_buttons(game_state)
                        if requires_confirmed_buttons and not game_state.get('buttons_confirmed', False):
                            logger.debug("Verwerfe Zustand ohne bestaetigte Buttons.")
                            self.pending_state_signature = None
                            self.pending_state_count = 0
                            continue

                        self.current_game_state = game_state
                        self.last_game_state_str = str(state_signature)
                        self.last_state_signature = state_signature
                        self.pending_state_signature = None
                        self.pending_state_count = 0

                        self._maybe_capture_screenshot(game_state)
                        self._auto_capture_screenshot(game_state)
                        self.session_manager.record_analysis()

                        try:
                            strategy = self.strategy_engine.calculate_strategy(
                                hole_cards=game_state['hole_cards'],
                                community_cards=game_state['community_cards'],
                                table_info=game_state
                            )
                        except Exception as e:
                            print(f"[BOT ERROR] Strategiefehler: {e}", flush=True)
                            logger.error(f"Strategiefehler: {e}", exc_info=True)
                            continue
                        self.last_strategy = strategy
                        self.session_manager.record_live_decision(game_state, strategy)

                        logger.debug(f"Strategie: {strategy}")
                        try:
                            self._print_live_summary(game_state, strategy)
                        except Exception as e:
                            print(f"[BOT ERROR] Ausgabefehler: {e}", flush=True)
                        self._update_overlay(game_state, strategy)

                        if LIVE_CONFIG.get('voice_enabled', False) and self._is_actionable_spot(game_state):
                            voice_payload = {
                                'strategy': strategy,
                                'hole_cards': strategy.get('hole_cards', []),
                                'community_cards': strategy.get('community_cards', []),
                            }
                            self.voice_announcer.announce(voice_payload)

                        if not self.headless and LIVE_CONFIG.get('show_debug_window', False) and self.last_screenshot is not None:
                            self._display_debug_image(self.last_screenshot, game_state, strategy)

                        # Auto-execute when auto_mode is enabled and it's our confirmed turn
                        if (
                            self.auto_mode
                            and self._is_actionable_spot(game_state)
                            and game_state.get('buttons_confirmed', False)
                            and self.last_action_taken != str(state_signature)
                        ):
                            rec_action = str(strategy.get('recommended_action', '')).lower()
                            rec_amount = float(strategy.get('amount', 0.0) or 0.0)
                            if rec_action in ('fold', 'call', 'check', 'bet', 'raise'):
                                logger.info(f"[AUTO] Führe aus: {rec_action.upper()} {rec_amount:.2f}")
                                print(f"[AUTO] {rec_action.upper()} {rec_amount:.2f}", flush=True)
                                try:
                                    self.execute_action(rec_action, rec_amount)
                                    self.last_action_taken = str(state_signature)
                                except Exception as e:
                                    logger.error(f"[AUTO] Fehler bei Ausführung: {e}")

                    elif self.current_game_state and self.last_strategy:
                        self._print_live_summary(self.current_game_state, self.last_strategy)

                else:
                    self.error_count += 1
                    logger.warning("Kein gültiger Spielzustand erkannt. Versuche erneut...")
                    time.sleep(1)

                elapsed_time = time.time() - start_time
                sleep_time = max(0, LIVE_CONFIG.get('analysis_interval', 0.5) - elapsed_time)
                time.sleep(sleep_time)

            else:
                time.sleep(0.5)

        logger.info("Live-Analyse beendet.")
        self.is_bot_thread_running = False

    def _display_debug_image(self, screenshot, game_state, strategy):
        img_vis = screenshot.copy()
        self._draw_region_group(img_vis, game_state.get('roi_regions', {}), (0, 255, 255))
        self._draw_region_list(img_vis, game_state.get('card_regions', {}).get('hero_hole_cards', []), "hole", (0, 255, 0))
        self._draw_region_list(img_vis, game_state.get('card_regions', {}).get('community_cards', []), "board", (255, 255, 0))
        self._draw_region_group(img_vis, game_state.get('action_regions', {}), (255, 128, 0))

        action = strategy.get('recommended_action', '?').upper()
        amount = strategy.get('amount', 0.0)
        text = f"Action: {action} {amount:.0f}"
        cv2.putText(img_vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cards_str = f"Hole: {game_state.get('hole_cards', ['?','?'])} Board: {game_state.get('community_cards', [])}"
        cv2.putText(img_vis, cards_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)

        pot = game_state.get('pot_size', 0)
        to_call = game_state.get('to_call', 0)
        pos = game_state.get('position', '?')
        pot_info = f"Pot: {pot:.0f} ToCall: {to_call:.0f} Pos: {pos}"
        cv2.putText(img_vis, pot_info, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1)

        cv2.imshow("Poker Bot Live Analysis", img_vis)
        cv2.waitKey(1)

    def _draw_region_group(self, image, regions, color):
        for name, region in regions.items():
            if not region:
                continue
            x, y, w, h = region
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, str(name), (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def _draw_region_list(self, image, regions, prefix, color):
        for index, region in enumerate(regions):
            x, y, w, h = region
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, f"{prefix}_{index}", (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def execute_action(self, action: str, amount: float):
        self.action_executor.execute_action(action, amount)

    def calibrate_rois(self):
        logger.info("Starte Kalibrierung der Regions of Interest...")
        capture_method = str(LIVE_CONFIG.get('capture_method', 'window')).strip().lower()

        if capture_method == 'window':
            # Im Window-Modus: Fenster-Cache zurücksetzen und neu suchen
            from utils.screen_utils import get_screenshot_for_processing
            LIVE_CONFIG.pop('detected_window_title', None)
            print("[Kalibrierung] Window-Modus: Suche Pokerfenster neu...", flush=True)
            screenshot = get_screenshot_for_processing()
            if screenshot is not None:
                title = LIVE_CONFIG.get('detected_window_title', '(unbekannt)')
                print(f"[Kalibrierung] Fenster gefunden: {title}", flush=True)
                logger.info(f"Fenster neu erkannt: {title}")
            else:
                print("[Kalibrierung] Kein Fenster gefunden. Sicherstellen dass das Spiel läuft.", flush=True)
                logger.warning("Fenster-Kalibrierung fehlgeschlagen: kein Fenster gefunden.")
        else:
            # Im Screen-Modus: interaktive Regionsauswahl per Screenshot
            print("Bitte wählen Sie nun den Bereich für den Spieltisch aus.", flush=True)
            selected_region = select_region_interactive()
            if selected_region:
                LIVE_CONFIG['screen_region'] = selected_region
                logger.info(f"Bildschirmregion aktualisiert auf: {selected_region}")
                print(f"Bildschirmregion aktualisiert auf: {selected_region}. Bitte speichern Sie diese in config.py.")
            else:
                logger.warning("Kalibrierung abgebrochen oder keine Region ausgewählt.")
                print("Kalibrierung abgebrochen.")

    def toggle_auto_mode(self):
        self.auto_mode = not self.auto_mode
        logger.info(f"Auto-Modus {'aktiviert' if self.auto_mode else 'deaktiviert'}.")
        print(f"--- Auto-Modus {'AN' if self.auto_mode else 'AUS'} ---")

    def toggle_pause(self):
        self.paused = not self.paused
        logger.info(f"Bot {'pausiert' if self.paused else 'fortgesetzt'}.")
        print(f"--- Bot {'PAUSIERT' if self.paused else 'AKTIV'} ---")

    def toggle_voice(self):
        self.voice_announcer.enabled = not self.voice_announcer.enabled
        LIVE_CONFIG['voice_enabled'] = self.voice_announcer.enabled
        logger.info(f"Sprachausgabe {'aktiviert' if self.voice_announcer.enabled else 'deaktiviert'}.")
        print(f"--- Sprachausgabe {'AN' if self.voice_announcer.enabled else 'AUS'} ---")
        if self.voice_announcer.enabled:
            self.voice_announcer.speak("Voice announcer enabled.")
        else:
            self.voice_announcer.speak("Voice announcer disabled.")

    def manual_fold(self):
        if self.current_game_state and self.current_game_state.get('is_my_turn', False):
            self.execute_action('fold', 0.0)
            self.last_action_taken = 'fold'
        else:
            logger.warning("Manuelles Fold angefordert, aber nicht am Zug oder kein gültiger Zustand.")

    def manual_call(self):
        if self.current_game_state and self.current_game_state.get('is_my_turn', False):
            self.execute_action('call', 0.0)
            self.last_action_taken = 'call'
        else:
            logger.warning("Manuelles Call angefordert, aber nicht am Zug oder kein gültiger Zustand.")

    def manual_raise(self):
        if self.current_game_state and self.current_game_state.get('is_my_turn', False):
            raise_amount = 0.0

            if self.current_game_state.get('street') != 'preflop' or self.current_game_state.get('to_call') == 0:
                strategy = self.strategy_engine.calculate_strategy(
                    self.current_game_state['hole_cards'],
                    self.current_game_state['community_cards'],
                    self.current_game_state
                )
                if strategy['recommended_action'] == 'raise':
                    raise_amount = strategy['amount']

            if raise_amount <= 0:
                pot = self.current_game_state.get('pot_size', 0)
                call_val = self.current_game_state.get('to_call', 0)
                raise_amount = (pot + call_val) * 2.5
                raise_amount = max(raise_amount, 10.0)

            max_all_in = self.current_game_state.get('pot_size', 0) + self.current_game_state.get('to_call', 0)
            raise_amount = min(raise_amount, max_all_in)

            if raise_amount > 0:
                self.execute_action('raise', raise_amount)
                self.last_action_taken = 'raise'
            else:
                logger.warning("Konnte keinen gültigen Raise-Betrag bestimmen.")
        else:
            logger.warning("Manuelles Raise angefordert, aber nicht am Zug oder kein gültiger Zustand.")

    def stop_bot(self):
        logger.info("Stopp-Signal empfangen.")
        self.running = False
        self.stop_event.set()
        self.overlay.stop()
        self.session_manager.close_session()
        remove_hotkeys()
        if not self.headless:
            cv2.destroyAllWindows()

    def get_my_player_index(self) -> int:
        logger.warning("get_my_player_index() nicht implementiert. Gehe von Spieler 0 aus.")
        return 0

    def run_analysis_thread(self):
        if not self.is_bot_thread_running:
            self.running = True
            self.stop_event.clear()
            self.analysis_thread = threading.Thread(target=self.analyze_loop, daemon=True)
            self.analysis_thread.start()
        else:
            logger.info("Analyse-Thread läuft bereits.")

    def run(self):
        logger.info("Initialisiere Live Poker Bot...")

        capture_method = str(LIVE_CONFIG.get('capture_method', 'screen')).strip().lower()

        if not self.headless:
            if capture_method != 'window' and LIVE_CONFIG.get('screen_region') is None:
                print("Keine Bildschirmregion in config.py definiert.")
                selected = select_region_interactive()
                if selected:
                    print(f"Region ausgewählt: {selected}. Bitte in config.py speichern.")
                    LIVE_CONFIG['screen_region'] = selected
                else:
                    print("WARNUNG: Keine Region ausgewählt. Verwende Standardeinstellungen.")

            setup_hotkeys(self)

            print("\n--- Live Poker Bot ---")
            print(f"Capture-Modus: {capture_method.upper()}")
            if capture_method == 'window':
                print(f"Fenstertitel-Suche: {LIVE_CONFIG.get('window_title_contains', '')}")
            print(f"Auto-Modus: {'AN' if self.auto_mode else 'AUS'}")
            print(f"Pausiert: {'JA' if self.paused else 'NEIN'}")
            print(f"Sprachausgabe: {'AN' if self.voice_announcer.enabled else 'AUS'}")
            print("----------------------\n")

        self.running = True
        self.stop_event.clear()
        self.run_analysis_thread()

        while self.running and not self.stop_event.is_set():
            try:
                if keyboard.is_pressed('ctrl+q'):
                    self.stop_bot()
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Fehler in Hauptschleife: {e}")
                self.stop_bot()

        logger.info("Bot Hauptschleife beendet.")
        if hasattr(self, 'analysis_thread') and self.analysis_thread.is_alive():
            logger.info("Warte auf Beendigung des Analyse-Threads...")
            self.analysis_thread.join(timeout=2.0)

        remove_hotkeys()
        if not self.headless:
            cv2.destroyAllWindows()
        logger.info("Live Poker Bot wurde gestoppt.")
