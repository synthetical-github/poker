# detectors/table_parser.py
import os
import re
import copy
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract

from utils.logger import logger
from config import (
    TABLE_TEMPLATE_PATH,
    get_current_table_layout_name,
    get_table_rois,
)


class TableParser:
    def __init__(self):
        self.table_template = self._load_template(TABLE_TEMPLATE_PATH)
        self.table_detection_threshold = 0.8  # Konfigurierbar machen?
        self.layout_name = ""
        self.reference_width = 0
        self.reference_height = 0
        self.player_area_definitions = []
        self.num_players = 0
        self.roi_definitions = {}
        self._ocr_cache: Dict[tuple, List[str]] = {}
        self._ocr_cache_order: List[tuple] = []
        self._ocr_cache_limit = 128
        self._last_parse_fingerprint: bytes | None = None
        self._last_parsed_data: Optional[Dict[str, Any]] = None
        self._refresh_layout()

    def _refresh_layout(self):
        layout_name = get_current_table_layout_name()
        if layout_name == self.layout_name and self.roi_definitions:
            return

        table_rois = get_table_rois(layout_name)
        self.layout_name = layout_name
        self.reference_width, self.reference_height = table_rois['reference_size']
        configured_player_areas = table_rois.get('player_areas')
        if configured_player_areas:
            self.player_area_definitions = configured_player_areas
        else:
            self.player_area_definitions = [
                {'region': table_rois['hero_player_area'], 'type': 'player_info', 'role': 'hero'},
                {'region': table_rois['villain_player_area'], 'type': 'player_info', 'role': 'villain'},
            ]
        self.num_players = len(self.player_area_definitions)
        self.roi_definitions = {
            'pot': {'region': table_rois['pot'], 'type': 'text'},
            'player_areas': self.player_area_definitions,
        }

    def _load_template(self, path: str) -> Optional[np.ndarray]:
        """ Lädt ein Bild-Template (Graustufen). """
        if not path or not os.path.exists(path):
            logger.warning(f"Template nicht gefunden: {path}")
            return None
        template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            logger.error(f"Konnte Template nicht laden: {path}")
        return template

    def _find_template_on_image(
        self,
        image_gray: np.ndarray,
        template: Optional[np.ndarray],
        threshold: float,
    ) -> Optional[Tuple[int, int, int, int]]:
        """ Findet ein Template im Bild und gibt die Bounding Box zurück. """
        if template is None or image_gray is None:
            return None

        # Sicherstellen, dass Template nicht größer als Bild ist
        if template.shape[0] > image_gray.shape[0] or template.shape[1] > image_gray.shape[1]:
            return None

        res = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val >= threshold:
            # Gibt (x, y, width, height) der gefundenen Übereinstimmung zurück
            return (max_loc[0], max_loc[1], template.shape[1], template.shape[0])
        return None

    def _get_absolute_coords(self, table_coords: Tuple[int, int, int, int], relative_roi: Dict[str, Any]) -> Tuple[int, int, int, int]:
        """ Konvertiert feste Referenz-ROIs in absolute Bildkoordinaten. """
        table_x, table_y, table_w, table_h = table_coords

        region = relative_roi['region']
        x, y, w, h = region
        x = table_x + int((x / self.reference_width) * table_w)
        y = table_y + int((y / self.reference_height) * table_h)
        w = max(1, int((w / self.reference_width) * table_w))
        h = max(1, int((h / self.reference_height) * table_h))

        return x, y, w, h

    def _fingerprint_region(self, screenshot_bgr: np.ndarray, region: Tuple[int, int, int, int]) -> bytes:
        x, y, w, h = region
        roi = screenshot_bgr[y:y+h, x:x+w]
        if roi is None or roi.size == 0:
            return b''
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (24, 10), interpolation=cv2.INTER_AREA)
        return resized.tobytes()

    def _make_parse_fingerprint(
        self,
        screenshot_bgr: np.ndarray,
        table_coords: Tuple[int, int, int, int],
    ) -> bytes:
        regions: list[Tuple[int, int, int, int]] = []
        pot_definition = self.roi_definitions.get('pot')
        if pot_definition:
            regions.append(self._get_absolute_coords(table_coords, pot_definition))

        for player_definition in self.player_area_definitions:
            stack_region = player_definition.get('stack_region')
            if stack_region:
                regions.append(self._get_absolute_coords(table_coords, {'region': stack_region}))
            else:
                regions.append(self._get_absolute_coords(table_coords, player_definition))

        parts = [self.layout_name.encode('utf-8')]
        for region in regions:
            parts.append(self._fingerprint_region(screenshot_bgr, region))
        return b'|'.join(parts)

    def _parse_global_elements(self, screenshot_bgr: np.ndarray) -> Dict[str, Any]:
        return {
            'pot_size': 0.0,
            'current_bet': 0.0,
            'to_call': 0.0,
            'num_players': self.num_players,
            'position': 'unknown',
            'street': 'unknown',
            'dealer_button_pos': None,
            'active_player_turn': -1,
            'player_info': [{} for _ in range(self.num_players)],
            'community_cards_detected': [],
            'roi_regions': {},
        }

    def parse_table(self, screenshot_bgr: np.ndarray, table_coords: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """ Parst Tisch-Informationen wie Pot, Einsätze, Spielerpositionen etc. """
        self._refresh_layout()
        if table_coords is None:
            # Versuche, den Tisch zu finden, wenn keine Koordinaten gegeben sind
            # Dies würde eine Funktion wie find_table() aus image_processor.py benötigen
            logger.warning("Keine Tischkoordinaten übergeben. Tisch-spezifische ROIs können nicht angewendet werden.")
            # Hier könnte man versuchen, eine globale Suche durchzuführen, aber das ist weniger effizient.
            # Fürs Erste: Rückgabe von Standardwerten oder Fehler.
            return self._parse_global_elements(screenshot_bgr)  # Versuch, globale Elemente zu finden

        table_x, table_y, table_w, table_h = table_coords
        parse_fingerprint = self._make_parse_fingerprint(screenshot_bgr, table_coords)
        if parse_fingerprint == self._last_parse_fingerprint and self._last_parsed_data is not None:
            return copy.deepcopy(self._last_parsed_data)

        parsed_data = {
            'pot_size': 0.0,
            'current_bet': 0.0,  # Aktueller Höchsteinsatz
            'to_call': 0.0,  # Was muss der Bot zahlen?
            'num_players': self.num_players,  # Standardwert
            'position': 'unknown',  # Position des Bots (z.B. 'button', 'early', 'middle', 'late')
            'street': 'unknown',  # 'preflop', 'flop', 'turn', 'river'
            'dealer_button_pos': None,  # Position des Dealers (0 bis num_players-1)
            'active_player_turn': -1,  # Index des Spielers, der am Zug ist (-1 = unbekannt)
            'player_info': [{} for _ in range(self.num_players)],  # Infos für jeden Spieler
            'community_cards_detected': [],  # Karten, die als Community-Karten erkannt wurden
            'roi_regions': {},
        }
        amount_cache: Dict[Tuple[str, int, int, int, int], float] = {}

        img_gray = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2GRAY)

        # --- Parse Elemente basierend auf ROIs ---
        for name, definition in self.roi_definitions.items():
            try:
                if name == 'player_areas':
                    for player_index, player_definition in enumerate(definition):
                        player_definition = {'type': 'player_info', **player_definition}
                        abs_coords = self._get_absolute_coords(table_coords, player_definition)
                        parsed_data['roi_regions'][f"player_{player_index}"] = abs_coords
                        x, y, w, h = abs_coords

                        img_h, img_w = screenshot_bgr.shape[:2]
                        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
                            continue

                        roi_img = screenshot_bgr[y:y+h, x:x+w]
                        stack_roi_img = None
                        stack_region = player_definition.get('stack_region')
                        if stack_region:
                            stack_abs_coords = self._get_absolute_coords(table_coords, {'region': stack_region})
                            parsed_data['roi_regions'][f"player_{player_index}_stack"] = stack_abs_coords
                            sx, sy, sw, sh = stack_abs_coords
                            if sx >= 0 and sy >= 0 and sx + sw <= img_w and sy + sh <= img_h:
                                stack_roi_img = screenshot_bgr[sy:sy+sh, sx:sx+sw]
                        parsed_data['player_info'][player_index] = self._parse_player_details(
                            roi_img,
                            player_index,
                            player_definition.get('role', f'player_{player_index}'),
                            stack_roi_img,
                        )
                    continue

                abs_coords = self._get_absolute_coords(table_coords, definition)
                parsed_data['roi_regions'][name] = abs_coords
                x, y, w, h = abs_coords

                # Stelle sicher, dass die ROI innerhalb des Screenshots liegt
                img_h, img_w = screenshot_bgr.shape[:2]
                if x < 0 or y < 0 or x+w > img_w or y+h > img_h:
                    logger.debug(f"ROI '{name}' liegt außerhalb des Bildbereichs. Übersprungen.")
                    continue

                roi_img = screenshot_bgr[y:y+h, x:x+w]
                roi_gray = img_gray[y:y+h, x:x+w]

                element_type = definition['type']

                if element_type == 'text':
                    amount_cache_key = ('pot' if name == 'pot' else 'amount', x, y, w, h)
                    if amount_cache_key not in amount_cache:
                        amount_cache[amount_cache_key] = self._ocr_amount_roi(roi_gray, name)
                    amount = amount_cache[amount_cache_key]
                    if name == 'pot':
                        parsed_data['pot_size'] = amount
                    elif name == 'to_call':
                        parsed_data['to_call'] = amount
                    elif name == 'current_bet':
                        parsed_data['current_bet'] = amount

            except Exception as e:
                logger.error(f"Fehler beim Parsen von ROI '{name}': {e}", exc_info=True)

        # --- Zusätzliche Logik ---
        # - Bestimme die Position des Bots (z.B. relativ zum Dealer-Button)
        # - Bestimme die Street basierend auf der Anzahl der Community Cards (müssten hier erkannt werden)
        # - Kombiniere Informationen (z.B. wenn to_call > 0 und current_bet > to_call, dann ist current_bet der Betrag)

        community_count = len(parsed_data.get('community_cards_detected', []))
        if community_count == 0:
            parsed_data['street'] = 'preflop'
        elif community_count == 3:
            parsed_data['street'] = 'flop'
        elif community_count == 4:
            parsed_data['street'] = 'turn'
        elif community_count == 5:
            parsed_data['street'] = 'river'

        active_players = [player for player in parsed_data['player_info'] if player.get('active')]
        if active_players:
            parsed_data['num_players'] = len(active_players)
        parsed_data['current_bet'] = max(
            (float(player.get('current_bet', 0.0) or 0.0) for player in parsed_data['player_info']),
            default=0.0,
        )

        logger.debug(f"Geparste Tisch-Infos: {parsed_data}")
        self._last_parse_fingerprint = parse_fingerprint
        self._last_parsed_data = copy.deepcopy(parsed_data)
        return parsed_data

    def _parse_amount(self, text: str) -> float:
        """ Parst Text zu einem Geldbetrag (float). """
        try:
            cleaned = ''.join(c for c in text if c.isdigit() or c in ['.', ','])
            cleaned = cleaned.replace(',', '.')
            # Entferne ggf. Tausendertrennzeichen, wenn sie vor dem Dezimalpunkt stehen
            if '.' in cleaned and cleaned.count('.') > 1:
                parts = cleaned.split('.')
                if len(parts[-1]) <= 3:  # Annahme: Letzter Teil ist Dezimalzahl
                    cleaned = ".".join(parts[:-1]) + "." + parts[-1]
                else:  # Annahme: Punkte sind Tausendertrenner
                    cleaned = cleaned.replace('.', '')

            if not cleaned:
                return 0.0
            return float(cleaned)
        except ValueError:
            logger.warning(f"Konnte Betrag nicht parsen: '{text}'")
            return 0.0

    def _read_text_variants(self, roi_gray: np.ndarray, configs: List[str]) -> List[str]:
        texts: List[str] = []
        if roi_gray is None or roi_gray.size == 0:  # noqa: E701
            return texts

        resized_for_key = cv2.resize(roi_gray, (96, 32), interpolation=cv2.INTER_AREA)
        cache_key = (tuple(configs), resized_for_key.tobytes())
        cached_texts = self._ocr_cache.get(cache_key)
        if cached_texts is not None:
            return cached_texts

        enlarged = cv2.resize(roi_gray, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        image_variants = [
            cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            enlarged,
        ]

        for image_variant in image_variants:
            for config in configs:
                try:
                    text = pytesseract.image_to_string(image_variant, config=config)
                except Exception as e:
                    logger.debug(f"OCR-Variante fehlgeschlagen: {e}")
                    continue
                cleaned = self._clean_ocr_text(text)
                if cleaned:
                    texts.append(cleaned)
                    if len(dict.fromkeys(texts)) >= 2:
                        break
            else:
                continue
            break
        unique_texts = list(dict.fromkeys(texts))
        if cache_key not in self._ocr_cache:
            self._ocr_cache_order.append(cache_key)
        self._ocr_cache[cache_key] = unique_texts
        while len(self._ocr_cache_order) > self._ocr_cache_limit:
            stale_key = self._ocr_cache_order.pop(0)
            self._ocr_cache.pop(stale_key, None)
        return unique_texts

    def _ocr_amount_roi(self, roi_gray: np.ndarray, name: str) -> float:
        if roi_gray is None or roi_gray.size == 0:
            return 0.0

        if name == 'stack':
            return self._ocr_stack_roi(roi_gray)
        if name == 'pot' and self.layout_name == 'heads_up':
            heads_up_pot = self._ocr_heads_up_pot_roi(roi_gray)
            if heads_up_pot > 0:
                return heads_up_pot

        def parse_amount_candidates(text: str) -> List[float]:
            def cash_candidates_from_int(raw: str) -> List[float]:
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

            candidates: List[float] = []
            for match in re.finditer(r'(\d+[.,]\d{1,2})', text):
                try:
                    candidates.append(float(match.group(1).replace(',', '.')))
                except ValueError:
                    pass

            for raw in re.findall(r'(?<![\d.,])(\d{1,4})(?![\d.,])', text):
                candidates.extend(cash_candidates_from_int(raw))
            return candidates

        padded_roi = cv2.copyMakeBorder(roi_gray, 8, 8, 10, 10, cv2.BORDER_REPLICATE)
        if name == 'pot':
            configs = [
                '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789€.,:',
                '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789€.,:',
                '--psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789€.,:',
            ]
            candidate_texts = self._read_text_variants(padded_roi, configs)
        else:
            configs = [
                '--psm 7 -c tessedit_char_whitelist=0123456789€.,',
                '--psm 8 -c tessedit_char_whitelist=0123456789€.,',
            ]
            candidate_texts = self._read_text_variants(padded_roi, configs)

        candidates: List[float] = []
        for text in candidate_texts:
            candidates.extend(parse_amount_candidates(text))

        if not candidates:
            return 0.0

        if self.layout_name == 'acipayam_heads_up':
            cash_candidates = [value for value in candidates if 0 < value < 50]
            if cash_candidates:
                return min(cash_candidates, key=lambda value: (value >= 10, value))
        if self.layout_name == 'heads_up':
            sane_candidates = [value for value in candidates if 0 < value < 100000]
            if not sane_candidates:
                return 0.0
            if name == 'pot':
                return max(sane_candidates)
            return max(sane_candidates)

        sane_candidates = [value for value in candidates if 0 < value < 100000]
        return min(sane_candidates) if sane_candidates else 0.0

    def _ocr_heads_up_pot_roi(self, roi_gray: np.ndarray) -> float:
        """Read only the numeric part of the tournament pot label, excluding 'Pot:' and the heart icon."""
        if roi_gray is None or roi_gray.size == 0:
            return 0.0

        height, width = roi_gray.shape[:2]
        numeric_crop = roi_gray[
            max(0, int(height * 0.08)):max(1, int(height * 0.88)),
            max(0, int(width * 0.34)):max(1, int(width * 0.82)),
        ]
        if numeric_crop.size == 0:
            return 0.0

        padded_roi = cv2.copyMakeBorder(numeric_crop, 8, 8, 10, 10, cv2.BORDER_REPLICATE)
        candidate_texts = self._read_text_variants(
            padded_roi,
            [
                '--psm 7 -c tessedit_char_whitelist=0123456789',
                '--psm 8 -c tessedit_char_whitelist=0123456789',
                '--psm 13 -c tessedit_char_whitelist=0123456789',
            ],
        )

        candidates: List[float] = []
        for text in candidate_texts:
            digits = ''.join(char for char in text if char.isdigit())
            if 1 <= len(digits) <= 4:
                candidates.append(float(int(digits)))

        # Numeric-only OCR on this label often returns the correct pot plus an inflated
        # duplicate such as 850 for 450 or 1200 for 120. Prefer the smallest sane match.
        sane_candidates = [value for value in candidates if 0 < value < 100000]
        return min(sane_candidates) if sane_candidates else 0.0

    def _ocr_stack_roi(self, roi_gray: np.ndarray) -> float:
        padded_roi = cv2.copyMakeBorder(roi_gray, 8, 8, 10, 10, cv2.BORDER_REPLICATE)
        contrast_roi = cv2.threshold(padded_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        candidate_texts: List[str] = []
        for candidate_roi in (padded_roi, contrast_roi):
            candidate_texts.extend(
                self._read_text_variants(
                    candidate_roi,
                    [
                        '--psm 7 -c tessedit_char_whitelist=0123456789€.,',
                        '--psm 8 -c tessedit_char_whitelist=0123456789€.,',
                        '--psm 13 -c tessedit_char_whitelist=0123456789€.,',
                    ],
                )
            )

        for text in dict.fromkeys(candidate_texts):
            value = self._parse_stack_text(text)
            if value > 0:
                return value
        return 0.0

    def _parse_stack_text(self, text: str) -> float:
        cleaned = self._clean_ocr_text(text).replace(' ', '').replace('€', '')
        if not cleaned:
            return 0.0

        thousand_match = re.search(r'(?<!\d)(\d{1,3}[.,]\d{3})(?!\d)', cleaned)
        if thousand_match:
            digits = re.sub(r'\D', '', thousand_match.group(1))
            if digits:
                try:
                    return float(int(digits))
                except ValueError:
                    pass

        decimal_match = re.search(r'(?<!\d)(\d{1,3}[.,]\d{2})(?!\d)', cleaned)
        if decimal_match:
            try:
                return float(decimal_match.group(1).replace(',', '.'))
            except ValueError:
                pass

        bare_match = re.search(r'(\d{1,5})', cleaned)
        if not bare_match:
            return 0.0

        digits = bare_match.group(1)
        try:
            value = int(digits)
        except ValueError:
            return 0.0

        if self.layout_name == 'acipayam_heads_up':
            if len(digits) >= 3:
                return value / 100.0
            return float(value)

        return float(value)

    def _clean_ocr_text(self, text: str) -> str:
        """ Bereinigt den von OCR gelesenen Text. """
        cleaned = ''.join(c for c in text if c.isalnum() or c in ['.', ',', ' '])
        cleaned = cleaned.replace('O', '0').replace('o', '0')
        cleaned = cleaned.replace('l', '1').replace('I', '1')
        cleaned = cleaned.replace('S', '5').replace('s', '5')
        cleaned = cleaned.replace(',', '.')  # Standardisiere Dezimaltrennzeichen
        cleaned = ' '.join(cleaned.split())
        return cleaned.strip()

    def _normalize_stack_value(self, value: float) -> float:
        if value <= 0:
            return 0.0
        if self.layout_name == 'heads_up':
            return value if 50 <= value <= 100000 else 0.0
        if self.layout_name == 'acipayam_heads_up':
            return value if 1.0 <= value <= 1000 else 0.0
        return value if 0.1 <= value <= 100000 else 0.0

    def _focus_stack_roi(self, stack_roi_bgr: np.ndarray, role: str) -> np.ndarray:
        if stack_roi_bgr is None or stack_roi_bgr.size == 0:
            return stack_roi_bgr

        h, w = stack_roi_bgr.shape[:2]
        role_name = str(role or '').lower()
        if role_name == 'hero':
            y1, y2 = int(h * 0.08), int(h * 0.88)
            x1, x2 = int(w * 0.10), int(w * 0.90)
        elif role_name == 'villain':
            y1, y2 = int(h * 0.10), int(h * 0.86)
            x1, x2 = int(w * 0.08), int(w * 0.92)
        else:
            y1, y2 = int(h * 0.10), int(h * 0.90)
            x1, x2 = int(w * 0.10), int(w * 0.90)

        focused = stack_roi_bgr[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]
        return focused if focused.size > 0 else stack_roi_bgr

    def _parse_player_details(
        self,
        player_roi_bgr: np.ndarray,
        player_index: int,
        role: str,
        stack_roi_bgr: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """ Liefert Spielerinformationen inkl. OCR-Text, Chipstand und semantischer Rolle. """
        text = ""
        chips = 0.0
        current_bet = 0.0
        role_name = str(role or '').lower()
        try:
            h, w = player_roi_bgr.shape[:2]
            stack_roi = stack_roi_bgr
            if stack_roi is None or stack_roi.size == 0:
                if role_name == 'villain':
                    stack_roi = player_roi_bgr[
                        max(0, int(h * 0.56)):max(1, int(h * 0.88)),
                        max(0, int(w * 0.18)):max(1, int(w * 0.82)),
                    ]
                else:
                    stack_roi = player_roi_bgr[
                        max(0, int(h * 0.36)):max(1, int(h * 0.92)),
                        max(0, int(w * 0.12)):max(1, int(w * 0.88)),
                    ]
            if stack_roi.size > 0:
                stack_roi = self._focus_stack_roi(stack_roi, role)
                stack_gray = cv2.cvtColor(stack_roi, cv2.COLOR_BGR2GRAY)
                chips = self._ocr_amount_roi(stack_gray, 'stack')

            bet_roi = player_roi_bgr[
                max(0, int(h * 0.08)):max(1, int(h * 0.42)),
                max(0, int(w * 0.25)):max(1, int(w * 0.75)),
            ]
            if bet_roi.size > 0:
                bet_gray = cv2.cvtColor(bet_roi, cv2.COLOR_BGR2GRAY)
                current_bet = self._ocr_amount_roi(bet_gray, 'current_bet')
        except Exception as e:
            logger.debug(f"OCR fuer Spieler {player_index} fehlgeschlagen: {e}")

        chips = self._normalize_stack_value(chips)
        current_bet = self._normalize_stack_value(current_bet)

        if self.layout_name in {'heads_up', 'acipayam_heads_up'}:
            active = True
        else:
            gray = cv2.cvtColor(player_roi_bgr, cv2.COLOR_BGR2GRAY)
            text = self._clean_ocr_text(pytesseract.image_to_string(gray, config='--psm 6'))
            normalized_text = text.lower()
            active = "fold" not in normalized_text and "sit out" not in normalized_text

        return {
            'name': f'Player_{player_index}',
            'chips': chips,
            'current_bet': current_bet,
            'active': active,
            'role': role,
            'ocr_text': text,
        }
