# action_executor.py
import re
import time
import copy
from difflib import SequenceMatcher
from typing import Dict, Tuple

import cv2
import numpy as np
import pyautogui
import pytesseract

from utils.logger import logger
from utils.config import LIVE_CONFIG
from utils.card_utils import Card # Nur für Typ-Hinweise
from config import get_current_table_layout_name, get_table_action_rois

class ActionExecutor:
    def __init__(self):
        self.screen_region = LIVE_CONFIG.get('screen_region')
        self.layout_name = ""
        self.action_rois = {}
        self.reference_width = 0
        self.reference_height = 0
        self._ocr_cache: Dict[tuple, list[str]] = {}
        self._ocr_cache_order: list[tuple] = []
        self._ocr_cache_limit = 128
        self._last_action_fingerprint: bytes | None = None
        self._last_action_state: Dict[str, object] | None = None
        if not self.screen_region:
            logger.warning("screen_region nicht in config gesetzt. Mausaktionen könnten fehlschlagen.")
            # Versuche, Bildschirmgröße dynamisch zu ermitteln
            try:
                screen_width = pyautogui.size().width
                screen_height = pyautogui.size().height
                self.screen_region = (0, 0, screen_width, screen_height)
            except Exception as e:
                 logger.error(f"Konnte Bildschirmgröße nicht ermitteln: {e}")
                 self.screen_region = (0, 0, 100, 100) # Minimaler Fallback

        self._refresh_layout()
        self.button_coords = self._load_button_coordinates() # Laden von Button-Positionen (MUSS ANGEPASST WERDEN!)

    def _refresh_layout(self):
        layout_name = get_current_table_layout_name()
        if layout_name == self.layout_name and self.action_rois:
            return
        self.layout_name = layout_name
        self.action_rois = get_table_action_rois(layout_name)
        self.reference_width, self.reference_height = self.action_rois['reference_size']

    def _load_button_coordinates(self) -> dict:
        """ Lädt die Koordinaten der Buttons (Fold, Call, Raise, Bet, Check). """
        button_regions = self.get_button_regions()
        coords = {
            'fold': self._center_of_region(button_regions['fold_button']),
            'call': self._center_of_region(button_regions['check_button']),
            'raise': self._center_of_region(button_regions['bet_button']),
            'bet': self._center_of_region(button_regions['bet_button']),
            'check': self._center_of_region(button_regions['check_button']),
            'bet_input_field': self._center_of_region(button_regions['bet_input_field']),
        }
                  
        logger.debug(f"Geladene Button-Koordinaten: {coords}")
        return coords

    def _scale_region(
        self,
        region: Tuple[int, int, int, int],
        base_region: Tuple[int, int, int, int] | None = None,
    ) -> Tuple[int, int, int, int]:
        if base_region is not None:
            base_x, base_y, base_w, base_h = base_region
        elif self.screen_region:
            base_x, base_y, base_w, base_h = self.screen_region
        else:
            size = pyautogui.size()
            base_x, base_y, base_w, base_h = 0, 0, size.width, size.height

        x, y, w, h = region
        scaled_x = base_x + int((x / self.reference_width) * base_w)
        scaled_y = base_y + int((y / self.reference_height) * base_h)
        scaled_w = max(1, int((w / self.reference_width) * base_w))
        scaled_h = max(1, int((h / self.reference_height) * base_h))
        return (scaled_x, scaled_y, scaled_w, scaled_h)

    def _center_of_region(self, region: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x, y, w, h = region
        return (x + w // 2, y + h // 2)

    def get_button_regions(
        self,
        base_region: Tuple[int, int, int, int] | None = None,
    ) -> Dict[str, Tuple[int, int, int, int]]:
        self._refresh_layout()
        return {
            'fold_button': self._scale_region(self.action_rois['fold_button'], base_region=base_region),
            'check_button': self._scale_region(self.action_rois['check_button'], base_region=base_region),
            'bet_button': self._scale_region(self.action_rois['bet_button'], base_region=base_region),
            'bet_input_field': self._scale_region(self.action_rois['bet_input_field'], base_region=base_region),
        }

    def _ocr_roi_text(self, roi_bgr: np.ndarray) -> str:
        if roi_bgr is None or roi_bgr.size == 0:
            return ""

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        enlarged = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        variants = [
            enlarged,
            cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.threshold(enlarged, 140, 255, cv2.THRESH_BINARY)[1],
        ]
        configs = [
            '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789€.,:',
            '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789€.,:',
            '--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789€.,:',
        ]

        texts = []
        for variant in variants:
            for config in configs:
                try:
                    text = pytesseract.image_to_string(variant, config=config)
                except Exception as exc:
                    logger.debug(f"Button-OCR fehlgeschlagen: {exc}")
                    continue
                cleaned = " ".join(text.replace("\n", " ").split()).strip()
                if cleaned:
                    texts.append(cleaned)
        return " | ".join(dict.fromkeys(texts))

    def _extract_button_text_regions(self, roi_bgr: np.ndarray, button_name: str) -> tuple[np.ndarray, np.ndarray]:
        h, w = roi_bgr.shape[:2]
        if button_name == 'bet_input_field':
            word_roi = roi_bgr[max(0, int(h * 0.02)):max(1, int(h * 0.24)), :]
            amount_roi = roi_bgr[
                max(0, int(h * 0.38)):max(1, int(h * 0.92)),
                0:max(1, int(w * 0.28)),
            ]
            return word_roi, amount_roi

        # Read only the action button body and avoid the slider/preset row above it.
        word_roi = roi_bgr[
            max(0, int(h * 0.42)):max(1, int(h * 0.78)),
            int(w * 0.10):max(int(w * 0.90), 1),
        ]
        amount_roi = roi_bgr[
            max(0, int(h * 0.66)):max(1, int(h * 0.98)),
            int(w * 0.18):max(int(w * 0.88), 1),
        ]
        return word_roi, amount_roi

    def _action_button_presence_score(self, roi_bgr: np.ndarray) -> float:
        if roi_bgr is None or roi_bgr.size == 0:
            return 0.0

        b = roi_bgr[:, :, 0].astype(np.int16)
        g = roi_bgr[:, :, 1].astype(np.int16)
        r = roi_bgr[:, :, 2].astype(np.int16)
        blue_mask = (b > 70) & (g > 50) & ((b - r) > 10)
        return float(np.mean(blue_mask))

    def _make_ocr_cache_key(self, roi_gray: np.ndarray, configs: list[str]) -> tuple | None:
        if roi_gray is None or roi_gray.size == 0:
            return None
        resized = cv2.resize(roi_gray, (96, 32), interpolation=cv2.INTER_AREA)
        return (tuple(configs), resized.tobytes())

    def _make_action_fingerprint(self, button_rois: Dict[str, np.ndarray]) -> bytes:
        parts: list[bytes] = []
        for key in ('fold_button', 'check_button', 'bet_button', 'bet_input_field'):
            roi = button_rois.get(key)
            if roi is None or roi.size == 0:
                parts.append(b'')
                continue
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (24, 10), interpolation=cv2.INTER_AREA)
            parts.append(resized.tobytes())
        return b'|'.join(parts)

    def _has_visible_action_panel(self, presence_scores: Dict[str, float]) -> bool:
        fold_score = float(presence_scores.get('fold_button', 0.0) or 0.0)
        middle_score = float(presence_scores.get('check_button', 0.0) or 0.0)
        right_score = float(presence_scores.get('bet_button', 0.0) or 0.0)
        input_score = float(presence_scores.get('bet_input_field', 0.0) or 0.0)

        if self.layout_name == 'heads_up':
            strong_threshold = 0.09
            weak_threshold = 0.05
        elif self.layout_name == 'acipayam_heads_up':
            strong_threshold = 0.10
            weak_threshold = 0.06
        else:
            strong_threshold = 0.12
            weak_threshold = 0.08

        return (
            max(fold_score, middle_score, right_score) >= strong_threshold
            or (fold_score >= weak_threshold and (middle_score >= weak_threshold or right_score >= weak_threshold))
            or input_score >= weak_threshold
        )

    def _store_ocr_cache(self, cache_key: tuple | None, texts: list[str]) -> None:
        if cache_key is None:
            return
        if cache_key not in self._ocr_cache:
            self._ocr_cache_order.append(cache_key)
        self._ocr_cache[cache_key] = texts
        while len(self._ocr_cache_order) > self._ocr_cache_limit:
            stale_key = self._ocr_cache_order.pop(0)
            self._ocr_cache.pop(stale_key, None)

    def _ocr_variants(self, roi_bgr: np.ndarray, configs: list[str]) -> list[str]:
        if roi_bgr is None or roi_bgr.size == 0:
            return []

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        cache_key = self._make_ocr_cache_key(gray, configs)
        if cache_key is not None and cache_key in self._ocr_cache:
            return self._ocr_cache[cache_key]

        enlarged = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        binary = cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        variants = [binary, enlarged]

        texts: list[str] = []
        for variant in variants:
            for config in configs:
                try:
                    text = pytesseract.image_to_string(variant, config=config)
                except Exception as exc:
                    logger.debug(f"Button-OCR fehlgeschlagen: {exc}")
                    continue
                cleaned = " ".join(text.replace("\n", " ").split()).strip()
                if cleaned:
                    texts.append(cleaned)
                    unique_texts = list(dict.fromkeys(texts))
                    if len(unique_texts) >= 2:
                        self._store_ocr_cache(cache_key, unique_texts)
                        return unique_texts
        unique_texts = list(dict.fromkeys(texts))
        self._store_ocr_cache(cache_key, unique_texts)
        return unique_texts

    def _normalize_keyword_text(self, text: str) -> str:
        normalized = re.sub(r'[^A-Z]', '', text.upper())
        normalized = (
            normalized.replace('0', 'O')
            .replace('1', 'I')
            .replace('5', 'S')
            .replace('€', 'E')
        )
        return normalized

    def _best_keyword_match(self, texts: list[str], candidates: list[str]) -> tuple[str | None, float]:
        best_keyword = None
        best_score = 0.0
        for text in texts:
            normalized_text = self._normalize_keyword_text(text)
            for candidate in candidates:
                score = SequenceMatcher(None, normalized_text, candidate).ratio()
                if candidate in normalized_text:
                    score = max(score, 0.99)
                if score > best_score:
                    best_score = score
                    best_keyword = candidate
        return best_keyword, best_score

    def _parse_amount_candidates(self, texts: list[str]) -> float:
        def cash_candidates_from_int(raw: str) -> list[float]:
            value = int(raw)
            if value <= 0:
                return []
            if self.layout_name == 'acipayam_heads_up':
                if len(raw) == 1:
                    return [value / 10.0]
                if len(raw) == 2:
                    if value % 10 == 0:
                        return [value / 100.0]
                        return [value / 10.0]
                if len(raw) in {3, 4}:
                    return [value / 100.0]
                return [float(value)]
            return [float(value)]

        candidates: list[float] = []
        for text in texts:
            for match in re.finditer(r'(\d+[.,]\d{1,2})', text):
                try:
                    candidates.append(float(match.group(1).replace(',', '.')))
                except ValueError:
                    pass

            plain_int_matches = re.findall(r'(?<![\d.,])(\d{1,4})(?![\d.,])', text)
            for raw in plain_int_matches:
                candidates.extend(cash_candidates_from_int(raw))

        if not candidates:
            return 0.0

        if self.layout_name == 'acipayam_heads_up':
            cash_candidates = [value for value in candidates if 0 < value < 50]
            if cash_candidates:
                return min(cash_candidates, key=lambda value: (value >= 10, value))

        if self.layout_name == 'heads_up':
            sane_candidates = [value for value in candidates if 0 < value < 100000]
            return min(sane_candidates) if sane_candidates else 0.0

        # Prefer smaller decimal values if present, otherwise the smallest sane OCR value.
        decimal_candidates = [value for value in candidates if 0 < value < 10]
        if decimal_candidates:
            return min(decimal_candidates)
        sane_candidates = [value for value in candidates if value < 10000]
        return min(sane_candidates) if sane_candidates else 0.0

    def _parse_button_amount(self, text: str) -> float:
        if not text:
            return 0.0
        return self._parse_amount_candidates([text])

    def read_action_state(
        self,
        screenshot_bgr: np.ndarray,
        base_region: Tuple[int, int, int, int] | None = None,
    ) -> Dict[str, object]:
        detected = {
            'is_my_turn': False,
            'buttons_confirmed': False,
            'panel_visible': False,
            'available_actions': [],
            'call_amount': 0.0,
            'raise_to_amount': 0.0,
            'bet_input_amount': 0.0,
            'button_text': {},
        }

        if screenshot_bgr is None or screenshot_bgr.size == 0:
            return detected

        if base_region is None:
            base_region = (0, 0, screenshot_bgr.shape[1], screenshot_bgr.shape[0])

        button_regions = self.get_button_regions(base_region=base_region)
        presence_scores: Dict[str, float] = {}
        button_rois: Dict[str, np.ndarray] = {}

        for name, region in button_regions.items():
            x, y, w, h = region
            roi = screenshot_bgr[y:y+h, x:x+w]
            button_rois[name] = roi
            presence_scores[name] = self._action_button_presence_score(roi)

        has_action_panel = self._has_visible_action_panel(presence_scores)
        detected['panel_visible'] = has_action_panel
        if not has_action_panel:
            return detected

        fingerprint = self._make_action_fingerprint(button_rois)
        if fingerprint == self._last_action_fingerprint and self._last_action_state is not None:
            return copy.deepcopy(self._last_action_state)

        if self.layout_name == 'heads_up':
            button_presence_threshold = 0.05
        elif self.layout_name == 'acipayam_heads_up':
            button_presence_threshold = 0.06
        else:
            button_presence_threshold = 0.08

        middle_button_present = presence_scores.get('check_button', 0.0) >= button_presence_threshold
        right_button_present = presence_scores.get('bet_button', 0.0) >= button_presence_threshold

        detected['button_text']['fold_button'] = 'FOLD' if presence_scores.get('fold_button', 0.0) >= button_presence_threshold else ''

        middle_word_roi, middle_amount_roi = self._extract_button_text_regions(button_rois['check_button'], 'check_button')
        middle_word_texts = self._ocr_variants(
            middle_word_roi,
            [
                '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
                '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
            ],
        )
        middle_keyword, middle_score = self._best_keyword_match(middle_word_texts, ['CHECK', 'CALL'])
        middle_amount_texts: list[str] = []
        if middle_keyword != 'CHECK' or middle_score < 0.80:
            middle_amount_texts = self._ocr_variants(
                middle_amount_roi,
                [
                    '--psm 7 -c tessedit_char_whitelist=0123456789€.,',
                    '--psm 8 -c tessedit_char_whitelist=0123456789€.,',
                ],
            )
        detected['button_text']['check_button'] = " | ".join(middle_word_texts + middle_amount_texts)
        inferred_middle_amount = self._parse_amount_candidates(middle_amount_texts)

        right_word_roi, right_amount_roi = self._extract_button_text_regions(button_rois['bet_button'], 'bet_button')
        right_word_texts = self._ocr_variants(
            right_word_roi,
            [
                '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
                '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
            ],
        )
        right_keyword, right_score = self._best_keyword_match(right_word_texts, ['BET', 'RAISETO', 'RAISE'])
        right_amount_texts: list[str] = []
        if right_keyword is not None or right_score < 0.80:
            right_amount_texts = self._ocr_variants(
                right_amount_roi,
                [
                    '--psm 7 -c tessedit_char_whitelist=0123456789€.,',
                    '--psm 8 -c tessedit_char_whitelist=0123456789€.,',
                ],
            )
        detected['button_text']['bet_button'] = " | ".join(right_word_texts + right_amount_texts)
        inferred_right_amount = self._parse_amount_candidates(right_amount_texts)

        input_texts: list[str] = []
        if presence_scores.get('bet_input_field', 0.0) >= 0.05:
            _, input_amount_roi = self._extract_button_text_regions(button_rois['bet_input_field'], 'bet_input_field')
            input_texts = self._ocr_variants(
                input_amount_roi,
                [
                    '--psm 7 -c tessedit_char_whitelist=0123456789€.,',
                    '--psm 8 -c tessedit_char_whitelist=0123456789€.,',
                ],
            )
        detected['button_text']['bet_input_field'] = " | ".join(input_texts)

        fold_text = detected['button_text'].get('fold_button', '')
        middle_text = detected['button_text'].get('check_button', '')
        right_text = detected['button_text'].get('bet_button', '')
        input_text = detected['button_text'].get('bet_input_field', '')

        # Left button on this client is the fold button whenever the action panel is visible.
        if has_action_panel and (middle_button_present or right_button_present):
            detected['available_actions'].append('fold')

        middle_keyword, middle_score = self._best_keyword_match(middle_word_texts or [middle_text], ['CHECK', 'CALL'])
        if middle_keyword == 'CHECK' and middle_score >= 0.55:
            detected['available_actions'].append('check')
        elif middle_keyword == 'CALL' and middle_score >= 0.55:
            detected['available_actions'].append('call')
            detected['call_amount'] = self._parse_button_amount(middle_text)
        elif middle_button_present:
            if inferred_middle_amount > 0:
                detected['available_actions'].append('call')
                detected['call_amount'] = inferred_middle_amount
            else:
                detected['available_actions'].append('check')

        right_keyword, right_score = self._best_keyword_match(right_word_texts or [right_text], ['BET', 'RAISETO', 'RAISE'])
        if right_keyword == 'BET' and right_score >= 0.55:
            detected['available_actions'].append('bet')
        elif right_keyword in {'RAISETO', 'RAISE'} and right_score >= 0.55:
            detected['available_actions'].append('raise')
            detected['raise_to_amount'] = self._parse_button_amount(right_text)
        elif right_button_present:
            if inferred_right_amount > 0 or detected['bet_input_amount'] > 0:
                if detected['call_amount'] > 0 or middle_keyword == 'CALL':
                    detected['available_actions'].append('raise')
                    detected['raise_to_amount'] = inferred_right_amount or detected['bet_input_amount']
                else:
                    detected['available_actions'].append('bet')
                    detected['raise_to_amount'] = inferred_right_amount or detected['bet_input_amount']

        detected['bet_input_amount'] = self._parse_button_amount(input_text)
        if detected['bet_input_amount'] > 0:
            # On both heads-up layouts the editable amount field is the most reliable source
            # for the actual bet/raise size. Prefer it over noisy button OCR.
            if 'raise' in detected['available_actions'] or 'bet' in detected['available_actions']:
                detected['raise_to_amount'] = detected['bet_input_amount']
            if 'call' in detected['available_actions']:
                if detected['call_amount'] <= 0 or detected['call_amount'] > detected['bet_input_amount']:
                    detected['call_amount'] = round(detected['bet_input_amount'] / 2.0, 2)

        # Fallbacks when OCR saw the buttons but missed the exact label.
        if 'fold' in detected['available_actions'] and 'check' not in detected['available_actions'] and 'call' not in detected['available_actions'] and middle_button_present:
            if inferred_middle_amount > 0:
                detected['available_actions'].append('call')
                detected['call_amount'] = inferred_middle_amount
            else:
                detected['available_actions'].append('check')
        if 'fold' in detected['available_actions'] and 'bet' not in detected['available_actions'] and 'raise' not in detected['available_actions'] and right_button_present:
            if detected['call_amount'] > 0:
                detected['available_actions'].append('raise')
                detected['raise_to_amount'] = inferred_right_amount or detected['bet_input_amount']
            elif inferred_right_amount > 0 or detected['bet_input_amount'] > 0:
                detected['available_actions'].append('bet')
                detected['raise_to_amount'] = inferred_right_amount or detected['bet_input_amount']

        if 'bet' in detected['available_actions'] and 'call' in detected['available_actions'] and middle_keyword != 'CALL':
            detected['available_actions'] = [action for action in detected['available_actions'] if action != 'call']
            if 'check' not in detected['available_actions'] and middle_button_present:
                detected['available_actions'].append('check')
            detected['call_amount'] = 0.0

        detected['available_actions'] = list(dict.fromkeys(detected['available_actions']))
        if detected['available_actions'] == ['fold']:
            detected['available_actions'] = []
        detected['is_my_turn'] = len(detected['available_actions']) >= 2
        valid_action_sets = {
            # 3-button combinations
            ('bet', 'check', 'fold'),
            ('check', 'fold', 'raise'),
            ('call', 'fold', 'raise'),
            ('bet', 'call', 'fold'),
            ('check', 'raise', 'fold'),
            ('bet', 'check', 'raise'),
            # 2-button combinations (common in heads-up)
            ('call', 'fold'),
            ('check', 'fold'),
            ('bet', 'fold'),
            ('check', 'raise'),
            ('call', 'raise'),
        }
        action_tuple = tuple(sorted(detected['available_actions']))
        detected['buttons_confirmed'] = (
            action_tuple in valid_action_sets
            or len(detected['available_actions']) >= 2
        )
        self._last_action_fingerprint = fingerprint
        self._last_action_state = copy.deepcopy(detected)
        return detected

    def _move_and_click(self, coords: Tuple[int, int], duration: float = 0.2):
        """ Bewegt die Maus zu den Koordinaten und klickt. """
        try:
            pyautogui.moveTo(coords[0], coords[1], duration=duration)
            pyautogui.click()
            logger.debug(f"Geklickt auf: {coords}")
            time.sleep(0.1) # Kleine Pause nach dem Klick
        except Exception as e:
            logger.error(f"Fehler bei Mausbewegung/Klick auf {coords}: {e}")

    def _type_amount(self, amount: float):
        """ Gibt den Betrag in das Eingabefeld ein. """
        try:
            coord = self.button_coords.get('bet_input_field')
            if not coord:
                 logger.error("Koordinaten für 'bet_input_field' nicht definiert.")
                 return
                 
            # Klicke zuerst ins Feld, um sicherzustellen, dass es aktiv ist
            self._move_and_click(coord, duration=0.1)
            
            # Lösche den aktuellen Inhalt (z.B. durch Strg+A, Entf)
            pyautogui.hotkey('ctrl', 'a')
            pyautogui.press('delete')
            time.sleep(0.1)
            
            # Gib den neuen Betrag ein
            amount_str = str(int(round(amount))) # Runde auf ganze Zahlen, da oft ganze Chips gesetzt werden
            pyautogui.write(amount_str, interval=0.05)
            logger.debug(f"Betrag eingegeben: {amount_str}")
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Fehler beim Eingeben des Betrags {amount}: {e}")

    def execute_action(self, action: str, amount: float = 0.0):
        """ Führt die gegebene Aktion aus. """
        self.button_coords = self._load_button_coordinates()
        action = action.lower()
        logger.info(f"Führe Aktion aus: {action.upper()} {amount:.2f}")

        if action == 'fold':
            coord = self.button_coords.get('fold')
            if coord: self._move_and_click(coord)
            else: logger.warning("Keine Koordinaten für 'fold' gefunden.")
        
        elif action == 'call':
            coord = self.button_coords.get('call')
            if coord: self._move_and_click(coord)
            else: logger.warning("Keine Koordinaten für 'call' gefunden.")

        elif action == 'check':
            coord = self.button_coords.get('check') # Annahme: Check ist oft der gleiche Button wie Fold
            if coord: self._move_and_click(coord)
            else: logger.warning("Keine Koordinaten für 'check' gefunden.")

        elif action == 'bet':
            # Setze den Betrag und klicke dann auf den Bet/Raise Button
            if amount > 0:
                self._type_amount(amount)
                coord = self.button_coords.get('bet') # Oder 'raise'
                if coord: self._move_and_click(coord)
                else: logger.warning("Keine Koordinaten für 'bet' gefunden.")
            else:
                 logger.warning("Aktion 'bet' gewählt, aber Betrag ist 0. Führe 'check' aus.")
                 self.execute_action('check')

        elif action == 'raise':
            # Setze den Betrag und klicke dann auf den Raise Button
            if amount > 0:
                self._type_amount(amount)
                coord = self.button_coords.get('raise')
                if coord: self._move_and_click(coord)
                else: logger.warning("Keine Koordinaten für 'raise' gefunden.")
            else:
                 logger.warning("Aktion 'raise' gewählt, aber Betrag ist 0. Führe 'call' aus.")
                 self.execute_action('call')
        
        else:
            logger.warning(f"Unbekannte Aktion: {action}")
