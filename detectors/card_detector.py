# detectors/card_detector.py
import os
import re
import platform
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract

from utils.card_utils import Card, parse_card_string, RANK_MAP
from utils.logger import logger
from config import (
    ASSETS_DIR,
    CARD_TEMPLATES_DIR,
    LIVE_CONFIG,
    POKER_SETTINGS,
    get_current_table_layout_name,
    get_table_rois,
)

class CardDetector:
    def __init__(self):
        # Initialize Tesseract path for Windows
        self._init_tesseract()
        
        self.ranks = POKER_SETTINGS['ranks']
        self.suits = POKER_SETTINGS['suits']
        self.rank_map = POKER_SETTINGS['rank_map']
        self.card_templates = self._load_card_templates()
        self.rank_templates = self._load_rank_templates()
        self.rank_corner_templates, self.suit_corner_templates = self._load_corner_templates()
        self.hero_rank_templates = self._load_merged_symbol_templates(
            ["hero_rank_templates", "hero_rank_templates_v2", "hero_rank_templates_v3"],
            valid_labels=set(self.ranks),
        )
        self.hero_suit_templates = self._load_merged_symbol_templates(
            ["hero_suit_templates", "hero_suit_templates_v2", "hero_suit_templates_v3"],
            valid_labels=set(self.suits),
        )
        self.board_rank_templates = self._load_merged_symbol_templates(
            ["board_rank_templates_v2", "board_rank_templates_v3"],
            valid_labels=set(self.ranks),
        )
        self.board_suit_templates = self._load_merged_symbol_templates(
            ["board_suit_templates_v2", "board_suit_templates_v3"],
            valid_labels=set(self.suits),
        )
        (
            self.board_card_templates,
            self.board_card_templates_by_context,
        ) = self._load_merged_card_template_maps(
            ["board_card_templates_v2", "board_card_templates_v3"]
        )
        self.hero_card_templates = self._load_merged_card_templates(
            ["hero_card_templates_v3"]
        )
        # Validate all templates are properly loaded
        self._validate_templates()
        
        # ADJUSTED THRESHOLDS: Lowered from 0.82 to 0.70 to accept valid card matches
        # at 0.75-0.81 range that were previously rejected
        self.min_card_match_threshold = 0.70
        # Reduced gap from 0.03 to 0.02 for more realistic differentiation
        self.min_card_match_gap = 0.02
        # Improved rank matching threshold
        self.min_rank_match_threshold = 0.50
        # Relaxed corner detection thresholds to reduce false negatives
        self.min_corner_rank_match_threshold = 0.38
        self.min_corner_suit_match_threshold = 0.32
        self.layout_name = ""
        self.reference_width = 0
        self.reference_height = 0
        self.table_rois = {}
        self._roi_result_cache: Dict[Tuple[str, str, int, int, bytes], Optional[Card]] = {}
        self._roi_cache_order: List[Tuple[str, str, int, int, bytes]] = []
        self._roi_cache_limit = 512
        self._frame_detection_cache: Dict[Tuple[int, int, bytes], Dict[str, object]] = {}
        self._frame_detection_order: List[Tuple[int, int, bytes]] = []
        self._frame_detection_limit = 64
        self._refresh_layout()

    def _init_tesseract(self) -> None:
        """Initialize Tesseract path for Windows systems."""
        if platform.system() == "Windows":
            common_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Users\Admin\AppData\Local\Tesseract-OCR\tesseract.exe",
            ]
            for path in common_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.pytesseract_cmd = path
                    logger.info(f"Tesseract OCR initialized: {path}")
                    return
            logger.warning("Tesseract OCR not found. OCR fallback will be unavailable. Relying on template matching only.")
        else:
            logger.info("Tesseract OCR should be available via system PATH on Linux/Mac")

    def _validate_templates(self) -> None:
        """Validate that all required card templates are loaded."""
        if not self.card_templates:
            logger.error("CRITICAL: No card templates loaded! Check CARD_TEMPLATES_DIR configuration.")
            return
        
        # Check if we have basic ranks and suits
        if not self.ranks or not self.suits:
            logger.error("CRITICAL: Ranks or suits configuration is empty!")
            return
        
        loaded_count = len(self.card_templates)
        expected_count = len(self.ranks) * len(self.suits)  # 13 ranks * 4 suits = 52 cards
        
        if loaded_count < expected_count * 0.5:
            logger.warning(
                f"Template loading INCOMPLETE: Only {loaded_count} templates loaded "
                f"(expected ~{expected_count}). Card detection will be impaired."
            )
        else:
            logger.info(f"Template validation OK: {loaded_count} card templates loaded")

    def _refresh_layout(self):
        layout_name = get_current_table_layout_name()
        if layout_name == self.layout_name and self.table_rois:
            return
        self._apply_layout_profile(layout_name)

    def _apply_layout_profile(self, layout_name: str):
        self.layout_name = layout_name
        self.table_rois = get_table_rois(layout_name)
        self.reference_width, self.reference_height = self.table_rois['reference_size']

    def _load_card_templates(self) -> Dict[str, np.ndarray]:
        """ Lädt Karten-Templates aus dem Verzeichnis. """
        templates = {}
        if not CARD_TEMPLATES_DIR or not os.path.isdir(CARD_TEMPLATES_DIR):
            logger.warning(f"Karten-Template-Verzeichnis nicht gefunden oder konfiguriert: {CARD_TEMPLATES_DIR}")
            return templates
            
        logger.info(f"Lade Karten-Templates aus: {CARD_TEMPLATES_DIR}")
        for filename in os.listdir(CARD_TEMPLATES_DIR):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                raw_name = os.path.splitext(filename)[0]
                card_name = self._normalize_template_name(raw_name)
                if not card_name:
                    logger.debug(f"Überspringe unbekannt benanntes Template: {raw_name}")
                    continue
                path = os.path.join(CARD_TEMPLATES_DIR, filename)
                template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    templates[card_name] = template
                else:
                    logger.warning(f"Konnte Template nicht laden: {path}")
        logger.info(f"{len(templates)} Karten-Templates geladen.")
        return templates

    def _load_rank_templates(self) -> Dict[str, np.ndarray]:
        templates = {}
        rank_templates_dir = os.path.join(ASSETS_DIR, 'rank_templates')
        if not os.path.isdir(rank_templates_dir):
            return templates

        for filename in os.listdir(rank_templates_dir):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            rank_name = os.path.splitext(filename)[0].strip().upper()
            if rank_name == '10':
                rank_name = 'T'
            if rank_name not in self.ranks:
                continue
            template = cv2.imread(os.path.join(rank_templates_dir, filename), cv2.IMREAD_GRAYSCALE)
            if template is not None:
                templates[rank_name] = self._normalize_rank_image(template)
        if templates:
            logger.info(f"{len(templates)} Rank-Templates geladen.")
        return templates

    def _load_symbol_templates_from_dir(self, subdir: str, valid_labels: set[str]) -> Dict[str, List[np.ndarray]]:
        templates: Dict[str, List[np.ndarray]] = {}
        directory = os.path.join(ASSETS_DIR, subdir)
        if not os.path.isdir(directory):
            return templates

        for filename in os.listdir(directory):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            raw_label = os.path.splitext(filename)[0].split("__", 1)[0].strip().upper()
            if raw_label == "10":
                raw_label = "T"
            if raw_label not in valid_labels:
                continue
            image = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            normalized = self._normalize_rank_image(image)
            if normalized is None or normalized.size == 0:
                continue
            templates.setdefault(raw_label, []).append(normalized)

        if templates:
            logger.info(
                f"{sum(len(v) for v in templates.values())} Templates aus {subdir} geladen."
            )
        return templates

    def _merge_symbol_template_maps(
        self,
        *template_maps: Dict[str, List[np.ndarray]],
    ) -> Dict[str, List[np.ndarray]]:
        merged: Dict[str, List[np.ndarray]] = {}
        for template_map in template_maps:
            for label, templates in template_map.items():
                merged.setdefault(label, []).extend(templates)
        return merged

    def _load_merged_symbol_templates(
        self,
        subdirs: List[str],
        valid_labels: set[str],
    ) -> Dict[str, List[np.ndarray]]:
        loaded = [
            self._load_symbol_templates_from_dir(subdir, valid_labels=valid_labels)
            for subdir in subdirs
        ]
        return self._merge_symbol_template_maps(*loaded)

    def _load_card_template_map_from_dir(
        self,
        subdir: str,
        split_context: bool = False,
    ) -> Dict[str, List[np.ndarray]] | Dict[str, Dict[str, List[np.ndarray]]]:
        templates: Dict[str, List[np.ndarray]] = {}
        contextual_templates: Dict[str, Dict[str, List[np.ndarray]]] = {}
        directory = os.path.join(ASSETS_DIR, subdir)
        if not os.path.isdir(directory):
            return contextual_templates if split_context else templates

        for filename in os.listdir(directory):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            stem = os.path.splitext(filename)[0]
            raw_label = stem.split("__", 1)[0].strip().upper()
            card_name = self._normalize_template_name(raw_label)
            if not card_name:
                continue
            image = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            if image is None or image.size == 0:
                continue
            if split_context:
                match = re.search(r"(board\d+)$", stem, re.IGNORECASE)
                if not match:
                    continue
                context_name = match.group(1).lower()
                contextual_templates.setdefault(context_name, {}).setdefault(card_name, []).append(image)
            else:
                templates.setdefault(card_name, []).append(image)

        if split_context and contextual_templates:
            logger.info(
                f"{sum(sum(len(v) for v in ctx.values()) for ctx in contextual_templates.values())} "
                f"Slot-Vollkarten-Templates aus {subdir} geladen."
            )
        elif templates:
            logger.info(
                f"{sum(len(v) for v in templates.values())} Vollkarten-Templates aus {subdir} geladen."
            )
        return contextual_templates if split_context else templates

    def _merge_card_template_maps(
        self,
        *template_maps: Dict[str, List[np.ndarray]],
    ) -> Dict[str, List[np.ndarray]]:
        merged: Dict[str, List[np.ndarray]] = {}
        for template_map in template_maps:
            for label, templates in template_map.items():
                merged.setdefault(label, []).extend(templates)
        return merged

    def _merge_contextual_card_template_maps(
        self,
        *template_maps: Dict[str, Dict[str, List[np.ndarray]]],
    ) -> Dict[str, Dict[str, List[np.ndarray]]]:
        merged: Dict[str, Dict[str, List[np.ndarray]]] = {}
        for template_map in template_maps:
            for context_name, cards in template_map.items():
                context_bucket = merged.setdefault(context_name, {})
                for label, templates in cards.items():
                    context_bucket.setdefault(label, []).extend(templates)
        return merged

    def _load_merged_card_template_maps(
        self,
        subdirs: List[str],
    ) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, Dict[str, List[np.ndarray]]]]:
        full_maps = [
            self._load_card_template_map_from_dir(subdir)
            for subdir in subdirs
        ]
        contextual_maps = [
            self._load_card_template_map_from_dir(subdir, split_context=True)
            for subdir in subdirs
        ]
        return (
            self._merge_card_template_maps(*full_maps),
            self._merge_contextual_card_template_maps(*contextual_maps),
        )

    def _load_merged_card_templates(
        self,
        subdirs: List[str],
    ) -> Dict[str, List[np.ndarray]]:
        full_maps = [
            self._load_card_template_map_from_dir(subdir)
            for subdir in subdirs
        ]
        return self._merge_card_template_maps(*full_maps)

    def _extract_template_corner_regions(self, image_gray: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if image_gray is None or image_gray.size == 0:
            return None, None

        rank_roi = self._extract_symbol_patch(
            image_gray,
            (0.00, 0.00, 0.38, 0.28),
            max_x_ratio=0.72,
            min_area=12,
            prefer_upper=True,
        )
        suit_roi = self._extract_symbol_patch(
            image_gray,
            (0.00, 0.10, 0.34, 0.42),
            max_x_ratio=0.70,
            min_area=10,
            prefer_upper=False,
        )
        if rank_roi.size == 0:
            rank_roi = None
        if suit_roi.size == 0:
            suit_roi = None
        return rank_roi, suit_roi

    def _build_symbol_mask(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)

        if len(image.shape) == 2:
            gray = image
            _, mask = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY_INV)
        else:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            yellow_mask = cv2.inRange(
                hsv,
                np.array([14, 55, 80], dtype=np.uint8),
                np.array([55, 255, 255], dtype=np.uint8),
            )
            color_mask = cv2.inRange(
                hsv,
                np.array([0, 35, 35], dtype=np.uint8),
                np.array([180, 255, 255], dtype=np.uint8),
            )
            color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(yellow_mask))
            dark_mask = cv2.inRange(gray, 0, 185)
            mask = cv2.bitwise_or(color_mask, dark_mask)

        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask

    def _extract_symbol_patch(
        self,
        image: np.ndarray,
        search_box: Tuple[float, float, float, float],
        *,
        max_x_ratio: float,
        min_area: int,
        prefer_upper: bool,
    ) -> Optional[np.ndarray]:
        if image is None or image.size == 0:
            return None

        h, w = image.shape[:2]
        x1 = int(w * search_box[0])
        y1 = int(h * search_box[1])
        x2 = max(x1 + 1, int(w * search_box[2]))
        y2 = max(y1 + 1, int(h * search_box[3]))
        search = image[y1:y2, x1:x2]
        if search.size == 0:
            return None

        mask = self._build_symbol_mask(search)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return search

        sh, sw = search.shape[:2]
        boxes = []
        for contour in contours:
            bx, by, bw, bh = cv2.boundingRect(contour)
            area = bw * bh
            if area < min_area:
                continue
            if bx > sw * max_x_ratio:
                continue
            if bw < max(3, int(sw * 0.04)):
                continue
            if bh > sh * 0.45 and (bw / float(max(bh, 1))) < 0.12:
                continue
            score = area / float(sw * sh)
            score += (1.0 - (bx / max(sw, 1))) * 0.2
            if prefer_upper:
                score += (1.0 - (by / max(sh, 1))) * 0.2
            boxes.append((score, bx, by, bw, bh))

        if not boxes:
            return search

        boxes.sort(reverse=True)
        _, bx, by, bw, bh = boxes[0]
        include = [(bx, by, bw, bh)]
        for _, ox, oy, ow, oh in boxes[1:]:
            same_band = abs(oy - by) <= max(6, int(sh * 0.10))
            close_x = ox <= bx + bw + max(8, int(sw * 0.12))
            overlap = not (ox + ow < bx or ox > bx + bw or oy + oh < by or oy > by + bh)
            vertical_gap = min(abs(oy - (by + bh)), abs(by - (oy + oh)))
            close_vertical = vertical_gap <= max(5, int(sh * 0.08))
            if (prefer_upper and (same_band or close_x or overlap)) or ((not prefer_upper) and (overlap or (close_x and close_vertical))):
                include.append((ox, oy, ow, oh))

        min_x = max(0, min(b[0] for b in include) - 2)
        min_y = max(0, min(b[1] for b in include) - 2)
        max_x = min(sw, max(b[0] + b[2] for b in include) + 2)
        max_y = min(sh, max(b[1] + b[3] for b in include) + 2)
        patch = search[min_y:max_y, min_x:max_x]
        if patch.size == 0:
            return search
        return patch

    def _load_corner_templates(self) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]]]:
        rank_templates: Dict[str, List[np.ndarray]] = {}
        suit_templates: Dict[str, List[np.ndarray]] = {}

        for card_name, template in self.card_templates.items():
            parsed = parse_card_string(card_name)
            if not parsed:
                continue

            rank_roi, suit_roi = self._extract_template_corner_regions(template)
            if rank_roi is not None:
                normalized_rank = self._normalize_rank_image(rank_roi)
                if normalized_rank is not None and normalized_rank.size > 0:
                    rank_templates.setdefault(parsed.rank, []).append(normalized_rank)

            if suit_roi is not None:
                normalized_suit = self._normalize_rank_image(suit_roi)
                if normalized_suit is not None and normalized_suit.size > 0:
                    suit_templates.setdefault(parsed.suit, []).append(normalized_suit)

        if rank_templates or suit_templates:
            logger.info(
                f"Ecken-Templates geladen: {sum(len(v) for v in rank_templates.values())} Rang, "
                f"{sum(len(v) for v in suit_templates.values())} Suit."
            )
        return rank_templates, suit_templates

    def _normalize_rank_image(self, image_gray: np.ndarray) -> np.ndarray:
        if image_gray is None or image_gray.size == 0:
            return image_gray
        enlarged = cv2.resize(image_gray, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            pad = 4
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(thresh.shape[1], x + w + pad)
            y2 = min(thresh.shape[0], y + h + pad)
            thresh = thresh[y1:y2, x1:x2]
        return thresh

    def _get_symbol_template_stats(
        self,
        candidate: np.ndarray,
        template_map: Dict[str, List[np.ndarray]],
    ) -> Tuple[Optional[str], float, float]:
        if candidate is None or candidate.size == 0 or not template_map:
            return None, -1.0, -1.0

        normalized_candidate = self._normalize_rank_image(candidate)
        if normalized_candidate is None or normalized_candidate.size == 0:
            return None, -1.0, -1.0

        best_label = None
        best_score = -1.0
        second_best = -1.0
        for label, templates in template_map.items():
            label_best = -1.0
            for template in templates:
                resized = cv2.resize(
                    normalized_candidate,
                    (template.shape[1], template.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(result)
                label_best = max(label_best, float(score))
            if label_best > best_score:
                second_best = best_score
                best_score = label_best
                best_label = label
            elif label_best > second_best:
                second_best = label_best

        return best_label, float(best_score), float(second_best)

    def _get_top_symbol_matches(
        self,
        candidate: np.ndarray,
        template_map: Dict[str, List[np.ndarray]],
        topn: int = 5,
    ) -> List[Tuple[str, float]]:
        if candidate is None or candidate.size == 0 or not template_map:
            return []

        normalized_candidate = self._normalize_rank_image(candidate)
        if normalized_candidate is None or normalized_candidate.size == 0:
            return []

        scores: List[Tuple[str, float]] = []
        for label, templates in template_map.items():
            label_best = -1.0
            for template in templates:
                resized = cv2.resize(
                    normalized_candidate,
                    (template.shape[1], template.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(result)
                label_best = max(label_best, float(score))
            scores.append((label, label_best))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:topn]

    def _match_symbol_templates(
        self,
        candidate: np.ndarray,
        template_map: Dict[str, List[np.ndarray]],
        min_threshold: float,
    ) -> Optional[str]:
        best_label, best_score, second_best = self._get_symbol_template_stats(candidate, template_map)
        if best_label and best_score >= min_threshold and (best_score - second_best) >= 0.02:
            return best_label
        return None

    def _normalize_template_name(self, name: str) -> Optional[str]:
        normalized = name.strip().upper()
        parsed_direct = parse_card_string(normalized)
        if parsed_direct:
            return str(parsed_direct)

        if "_OF_" not in normalized:
            return None

        rank_part, suit_part = normalized.split("_OF_", 1)
        rank_map = {
            "ACE": "A",
            "KING": "K",
            "QUEEN": "Q",
            "JACK": "J",
            "10": "T",
            "9": "9",
            "8": "8",
            "7": "7",
            "6": "6",
            "5": "5",
            "4": "4",
            "3": "3",
            "2": "2",
        }
        suit_map = {
            "CLUBS": "C",
            "DIAMONDS": "D",
            "HEARTS": "H",
            "SPADES": "S",
        }

        rank = rank_map.get(rank_part)
        suit = suit_map.get(suit_part)
        if not rank or not suit:
            return None
        return f"{rank}{suit}"

    def _crop_fractional_box(
        self,
        image: np.ndarray,
        box: Tuple[float, float, float, float],
    ) -> Optional[np.ndarray]:
        if image is None or image.size == 0:
            return None
        h, w = image.shape[:2]
        x1 = max(0, min(w - 1, int(w * box[0])))
        x2 = max(x1 + 1, min(w, int(w * box[1])))
        y1 = max(0, min(h - 1, int(h * box[2])))
        y2 = max(y1 + 1, min(h, int(h * box[3])))
        crop = image[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def _parse_rank_text(self, text: str) -> Optional[str]:
        cleaned = ''.join(ch for ch in text.upper() if ch.isalnum())
        if not cleaned:
            return None
        if cleaned.startswith('10'):
            return 'T'
        if cleaned == '1':
            return 'T'
        rank = cleaned[0]
        rank_aliases = {'0': 'Q', 'O': 'Q', 'I': 'T', 'L': 'T', '1': 'T'}
        rank = rank_aliases.get(rank, rank)
        return rank if rank in self.ranks else None

    def _trim_left_highlight_border(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return image

        h, w = image.shape[:2]
        if w < 24:
            return image

        sample = image[:max(1, int(h * 0.55)), :]
        hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(
            hsv,
            np.array([0, 0, 165], dtype=np.uint8),
            np.array([180, 85, 255], dtype=np.uint8),
        )
        yellow_mask = cv2.inRange(
            hsv,
            np.array([14, 55, 85], dtype=np.uint8),
            np.array([55, 255, 255], dtype=np.uint8),
        )

        cut_x = 0
        last_highlight_x = -1
        scan_limit = min(w, max(10, int(w * 0.34)))
        for x in range(scan_limit):
            white_ratio = float(np.mean(white_mask[:, x] > 0))
            yellow_ratio = float(np.mean(yellow_mask[:, x] > 0))
            if yellow_ratio > 0.18 and white_ratio < 0.60:
                last_highlight_x = x
            if x >= 2 and white_ratio > 0.42 and yellow_ratio < 0.15:
                cut_x = max(0, x - 1)
                break

        if cut_x == 0 and last_highlight_x >= 0:
            cut_x = min(w - 1, last_highlight_x + 2)

        if cut_x <= 0:
            return image

        trimmed = image[:, cut_x:]
        return trimmed if trimmed.size > 0 else image

    def _is_left_edge_contaminated(self, image: np.ndarray) -> bool:
        if image is None or image.size == 0:
            return False

        h, w = image.shape[:2]
        edge = image[:, :max(1, int(w * 0.12))]
        hsv = cv2.cvtColor(edge, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        total = max(1, edge.shape[0] * edge.shape[1])
        white_ratio = float(np.sum((gray > 180) & (sat < 60))) / total
        red_ratio = float(np.sum((((hue <= 12) | (hue >= 170)) & (sat > 50) & (val > 40)))) / total
        return white_ratio < 0.55 or red_ratio > 0.02

    def _detect_rank_from_focus_boxes(
        self,
        image: np.ndarray,
        boxes: List[Tuple[float, float, float, float]],
    ) -> Optional[str]:
        for box in boxes:
            crop = self._crop_fractional_box(image, box)
            if crop is None:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            big = cv2.resize(gray, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)
            thresh = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            try:
                text = pytesseract.image_to_string(
                    thresh,
                    config='--psm 10 -c tessedit_char_whitelist=A23456789TJQK10',
                )
            except Exception:
                text = ""
            parsed_rank = self._parse_rank_text(text)
            if parsed_rank:
                return parsed_rank

            matched_rank = self._match_rank_template(gray)
            if matched_rank:
                return matched_rank
        return None

    def _detect_rank_from_focus_boxes_ocr_first(
        self,
        image: np.ndarray,
        boxes: List[Tuple[float, float, float, float]],
    ) -> Optional[str]:
        crops: List[np.ndarray] = []
        for box in boxes:
            crop = self._crop_fractional_box(image, box)
            if crop is None:
                continue
            crops.append(crop)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            big = cv2.resize(gray, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)
            thresh = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            try:
                text = pytesseract.image_to_string(
                    thresh,
                    config='--psm 10 -c tessedit_char_whitelist=A23456789TJQK10',
                )
            except Exception:
                text = ""
            parsed_rank = self._parse_rank_text(text)
            if parsed_rank:
                return parsed_rank

        for crop in crops:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            matched_rank = self._match_rank_template(gray)
            if matched_rank:
                return matched_rank
        return None

    def _detect_suit_from_focus_boxes(
        self,
        image: np.ndarray,
        boxes: List[Tuple[float, float, float, float]],
    ) -> Optional[str]:
        best_label = None
        best_score = 0
        for box in boxes:
            crop = self._crop_fractional_box(image, box)
            if crop is None:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            hue = hsv[:, :, 0]
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]

            green_count = int(np.sum((hue >= 40) & (hue <= 95) & (sat > 50) & (val > 40)))
            blue_count = int(np.sum((hue >= 90) & (hue <= 140) & (sat > 50) & (val > 40)))
            red_count = int(np.sum((((hue <= 12) | (hue >= 170)) & (sat > 50) & (val > 40))))
            black_count = int(np.sum((gray < 80) & (val < 130)))

            color_scores = {
                'C': green_count,
                'D': blue_count,
                'H': red_count,
            }
            color_label = max(color_scores, key=color_scores.get)
            color_score = color_scores[color_label]
            suit_scores = {
                'C': green_count,
                'D': blue_count,
                'H': red_count,
                'S': black_count,
            }
            if color_score >= 60 and color_score >= (black_count * 0.45):
                label = color_label
                score = color_score
            else:
                label = max(suit_scores, key=suit_scores.get)
                score = suit_scores[label]
            if score > best_score:
                best_score = score
                best_label = label

        if best_label and best_score >= 80:
            return best_label
        return None

    def _detect_card_from_compact_hero_components(
        self,
        card_surface_bgr: np.ndarray,
    ) -> Optional[Card]:
        if card_surface_bgr is None or card_surface_bgr.size == 0:
            return None

        h, w = card_surface_bgr.shape[:2]
        top = card_surface_bgr[:max(1, int(h * 0.58)), :]
        mask = self._build_symbol_mask(top)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        min_x = int(w * 0.35)
        min_width = max(10, int(w * 0.12))
        boxes: List[Tuple[int, int, int, int, int]] = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            area = bw * bh
            if area < 20 or x < min_x:
                continue
            boxes.append((x, y, bw, bh, area))
        if not boxes:
            return None

        upper_boxes = [
            box for box in sorted(boxes, key=lambda item: item[0])
            if box[1] <= int(top.shape[0] * 0.18) and box[2] >= min_width
        ]
        if not upper_boxes:
            return None

        merged_groups: List[List[Tuple[int, int, int, int, int]]] = []
        for box in upper_boxes:
            if not merged_groups:
                merged_groups.append([box])
                continue
            prev = merged_groups[-1][-1]
            same_band = abs(box[1] - prev[1]) <= max(6, int(top.shape[0] * 0.10))
            close_x = box[0] <= (prev[0] + prev[2] + max(6, int(w * 0.08)))
            if same_band and close_x:
                merged_groups[-1].append(box)
            else:
                merged_groups.append([box])

        rank_group = max(
            merged_groups,
            key=lambda group: (
                max(item[0] + item[2] for item in group),
                sum(item[4] for item in group),
            ),
        )
        rank_x1 = max(0, min(item[0] for item in rank_group) - 2)
        rank_y1 = max(0, min(item[1] for item in rank_group) - 2)
        rank_x2 = min(top.shape[1], max(item[0] + item[2] for item in rank_group) + 2)
        rank_y2 = min(top.shape[0], max(item[1] + item[3] for item in rank_group) + 2)
        rank_patch = top[rank_y1:rank_y2, rank_x1:rank_x2]
        if rank_patch.size == 0:
            return None

        lower_boxes = [
            box for box in boxes
            if box[1] >= int(top.shape[0] * 0.45) and box[0] >= int(w * 0.40)
        ]
        suit_patch = None
        if lower_boxes:
            suit_box = max(lower_boxes, key=lambda item: item[4])
            sx, sy, sw, sh, _ = suit_box
            suit_patch = top[max(0, sy - 2):min(top.shape[0], sy + sh + 2), max(0, sx - 2):min(top.shape[1], sx + sw + 2)]

        rank = self._detect_rank_from_focus_boxes(rank_patch, [(0.0, 1.0, 0.0, 1.0)])
        suit = None
        if suit_patch is not None and suit_patch.size > 0:
            suit = self._detect_suit_from_focus_boxes(suit_patch, [(0.0, 1.0, 0.0, 1.0)])
        if rank and suit:
            return parse_card_string(f"{rank}{suit}")
        return None

    def _detect_card_from_compact_focus(
        self,
        card_surface_bgr: np.ndarray,
        context: str,
    ) -> Optional[Card]:
        if card_surface_bgr is None or card_surface_bgr.size == 0:
            return None

        focus = card_surface_bgr
        if context == "hero1":
            component_card = self._detect_card_from_compact_hero_components(focus)
            if component_card:
                return component_card
            rank_boxes = [
                (0.24, 0.84, 0.00, 0.26),
                (0.28, 0.88, 0.00, 0.30),
                (0.32, 0.92, 0.00, 0.34),
                (0.36, 0.96, 0.00, 0.38),
            ]
            suit_boxes = [
                (0.18, 0.86, 0.24, 0.92),
                (0.24, 0.90, 0.26, 0.94),
            ]
        elif context == "hero2":
            component_card = self._detect_card_from_compact_hero_components(focus)
            if component_card:
                return component_card
            rank_boxes = [
                (0.36, 0.80, 0.00, 0.26),
                (0.42, 0.86, 0.00, 0.30),
                (0.46, 0.90, 0.00, 0.34),
                (0.50, 0.94, 0.00, 0.38),
            ]
            suit_boxes = [
                (0.40, 0.92, 0.28, 0.92),
                (0.46, 0.96, 0.26, 0.90),
                (0.52, 0.98, 0.26, 0.92),
            ]
        else:
            board_index = 0
            if context.startswith("board"):
                try:
                    board_index = int(context[5:])
                except ValueError:
                    board_index = 0
            if board_index >= 3:
                focus_gray = cv2.cvtColor(focus, cv2.COLOR_BGR2GRAY)
                template_map = self.board_card_templates_by_context.get(context.lower()) or self.board_card_templates
                template_name, template_score, template_second, _ = self._get_card_template_stats_from_map(
                    focus_gray,
                    template_map,
                )
                if (
                    not self._is_left_edge_contaminated(focus)
                    and template_name
                    and template_score >= 0.72
                    and (template_score - template_second) >= 0.08
                ):
                    template_card = parse_card_string(template_name)
                    if template_card:
                        return template_card
                rank_boxes = [
                    (0.08, 0.26, 0.00, 0.30),
                    (0.12, 0.30, 0.00, 0.30),
                    (0.16, 0.34, 0.00, 0.30),
                    (0.20, 0.38, 0.00, 0.30),
                    (0.24, 0.42, 0.00, 0.30),
                    (0.28, 0.46, 0.00, 0.30),
                    (0.32, 0.50, 0.00, 0.30),
                    (0.36, 0.54, 0.00, 0.30),
                    (0.40, 0.58, 0.00, 0.30),
                ]
                suit_boxes = [
                    (0.08, 0.26, 0.18, 0.52),
                    (0.14, 0.32, 0.18, 0.52),
                    (0.20, 0.38, 0.18, 0.52),
                    (0.26, 0.44, 0.18, 0.52),
                    (0.32, 0.50, 0.18, 0.52),
                    (0.38, 0.56, 0.18, 0.52),
                ]
                rank = self._detect_rank_from_focus_boxes_ocr_first(focus, rank_boxes)
                suit = self._detect_suit_from_focus_boxes(focus, suit_boxes)
                if rank and suit:
                    return parse_card_string(f"{rank}{suit}")
            rank_boxes = [
                (0.00, 0.34, 0.00, 0.42),
                (0.04, 0.38, 0.00, 0.44),
                (0.08, 0.42, 0.00, 0.46),
                (0.12, 0.46, 0.00, 0.46),
                (0.16, 0.50, 0.00, 0.46),
                (0.20, 0.54, 0.00, 0.46),
                (0.24, 0.58, 0.00, 0.46),
                (0.28, 0.60, 0.00, 0.46),
            ]
            suit_boxes = [
                (0.00, 0.18, 0.18, 0.52),
                (0.06, 0.24, 0.18, 0.52),
                (0.12, 0.30, 0.18, 0.52),
                (0.18, 0.36, 0.18, 0.52),
                (0.24, 0.42, 0.18, 0.52),
                (0.30, 0.48, 0.18, 0.52),
                (0.34, 0.52, 0.18, 0.52),
            ]
            if board_index in {1, 2}:
                corner_rank, corner_score, corner_second = self._detect_rank_from_corner(focus, context=context)
                suit = self._detect_suit_from_focus_boxes(focus, suit_boxes)
                if (
                    corner_rank
                    and suit
                    and corner_score >= 0.46
                    and (corner_score - corner_second) >= 0.04
                ):
                    return parse_card_string(f"{corner_rank}{suit}")

        rank = self._detect_rank_from_focus_boxes(focus, rank_boxes)
        suit = self._detect_suit_from_focus_boxes(focus, suit_boxes)
        if rank and suit:
            return parse_card_string(f"{rank}{suit}")
        return None

    def _find_template(self, image_gray: np.ndarray, template: np.ndarray) -> Optional[Tuple[int, int, float]]:
        """ Findet ein Template in einem Bild und gibt Ort und Konfidenz zurück. """
        if template is None or image_gray is None or template.shape[0] > image_gray.shape[0] or template.shape[1] > image_gray.shape[1]:
            return None
            
        res = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if max_val >= self.min_card_match_threshold:
            return max_loc[0], max_loc[1], max_val # x, y, confidence
        return None

    def _score_card_template(self, card_gray: np.ndarray, template: np.ndarray) -> float:
        if card_gray is None or card_gray.size == 0 or template is None or template.size == 0:
            return -1.0
        resized = cv2.resize(card_gray, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(result)
        return float(score)

    def _make_roi_cache_key(
        self,
        roi_bgr: np.ndarray,
        context: str,
    ) -> Optional[Tuple[str, str, int, int, bytes]]:
        if roi_bgr is None or roi_bgr.size == 0:
            return None
        small = cv2.resize(roi_bgr, (32, 48), interpolation=cv2.INTER_AREA)
        return (self.layout_name, context, roi_bgr.shape[1], roi_bgr.shape[0], small.tobytes())

    def _get_cached_roi_result(
        self,
        roi_bgr: np.ndarray,
        context: str,
    ) -> Tuple[bool, Optional[Card], Optional[Tuple[str, str, int, int, bytes]]]:
        cache_key = self._make_roi_cache_key(roi_bgr, context)
        if cache_key is None:
            return False, None, None
        if cache_key in self._roi_result_cache:
            return True, self._roi_result_cache[cache_key], cache_key
        return False, None, cache_key

    def _store_cached_roi_result(
        self,
        cache_key: Optional[Tuple[str, str, int, int, bytes]],
        card: Optional[Card],
    ) -> None:
        if cache_key is None:
            return
        if cache_key not in self._roi_result_cache:
            self._roi_cache_order.append(cache_key)
        self._roi_result_cache[cache_key] = card
        while len(self._roi_cache_order) > self._roi_cache_limit:
            stale_key = self._roi_cache_order.pop(0)
            self._roi_result_cache.pop(stale_key, None)

    def _clear_roi_cache(self) -> None:
        self._roi_result_cache.clear()
        self._roi_cache_order.clear()

    def _make_frame_cache_key(self, screenshot_bgr: np.ndarray) -> Optional[Tuple[int, int, bytes]]:
        if screenshot_bgr is None or screenshot_bgr.size == 0:
            return None
        small = cv2.resize(screenshot_bgr, (80, 48), interpolation=cv2.INTER_AREA)
        return (screenshot_bgr.shape[1], screenshot_bgr.shape[0], small.tobytes())

    def _infer_layout_hint_from_title(self, screenshot_bgr: np.ndarray) -> Optional[str]:
        if screenshot_bgr is None or screenshot_bgr.size == 0:
            return None

        h, w = screenshot_bgr.shape[:2]
        crop = screenshot_bgr[:max(1, int(h * 0.08)), :max(1, int(w * 0.55))]
        if crop.size == 0:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        enlarged = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        try:
            text = pytesseract.image_to_string(thresh, config='--psm 6')
        except Exception:
            return None

        normalized = text.upper()
        if "HEADS UP" in normalized:
            return "heads_up"
        if "ARZON" in normalized:
            return "acipayam_heads_up"
        return None

    def _store_frame_detection(
        self,
        frame_key: Optional[Tuple[int, int, bytes]],
        detection: Dict[str, object],
    ) -> None:
        if frame_key is None:
            return
        if frame_key not in self._frame_detection_cache:
            self._frame_detection_order.append(frame_key)
        self._frame_detection_cache[frame_key] = detection
        while len(self._frame_detection_order) > self._frame_detection_limit:
            stale_key = self._frame_detection_order.pop(0)
            self._frame_detection_cache.pop(stale_key, None)

    def _candidate_layout_profiles(self) -> List[str]:
        if self.layout_name in {"acipayam_heads_up", "heads_up"}:
            return ["acipayam_heads_up", "heads_up"]
        return [self.layout_name]

    def _detect_hole_cards_for_current_layout(
        self,
        screenshot_bgr: np.ndarray,
        table_coords: Tuple[int, int, int, int],
    ) -> List[Card]:
        hole_card_regions = self.get_hole_card_regions(table_coords)
        cards: List[Optional[Card]] = []
        hero_contexts = ["hero1", "hero2"]
        for region, hero_context in zip(hole_card_regions, hero_contexts):
            x, y, w, h = region
            if w <= 0 or h <= 0:
                cards.append(None)
                continue
            img_h, img_w = screenshot_bgr.shape[:2]
            x = max(0, x)
            y = max(0, y)
            w = min(w, img_w - x)
            h = min(h, img_h - y)
            if w <= 0 or h <= 0:
                cards.append(None)
                continue
            roi = screenshot_bgr[y:y+h, x:x+w]
            cards.append(self._process_card_roi(roi, context=hero_context))
        unique_cards: List[Card] = []
        for card in cards:
            if card and card not in unique_cards:
                unique_cards.append(card)
        return unique_cards[:2]

    def _detect_community_cards_for_current_layout(
        self,
        screenshot_bgr: np.ndarray,
        table_coords: Tuple[int, int, int, int],
    ) -> Tuple[List[Card], int]:
        community_card_regions = self.get_community_card_regions(table_coords)
        normalized_regions: List[Tuple[int, int, int, int]] = []
        surface_flags: List[bool] = []
        surface_count = 0
        compact_heads_up_layout = self.layout_name in {"heads_up", "acipayam_heads_up"}

        img_h, img_w = screenshot_bgr.shape[:2]
        for region in community_card_regions:
            x, y, w, h = region
            if w <= 0 or h <= 0:
                normalized_regions.append((0, 0, 0, 0))
                surface_flags.append(False)
                continue

            x = max(0, x)
            y = max(0, y)
            w = min(w, img_w - x)
            h = min(h, img_h - y)
            if w <= 0 or h <= 0:
                normalized_regions.append((0, 0, 0, 0))
                surface_flags.append(False)
                continue

            normalized_regions.append((x, y, w, h))
            roi = screenshot_bgr[y:y+h, x:x+w]
            has_surface = self._has_community_card_surface(roi)
            surface_flags.append(has_surface)
            if has_surface:
                surface_count += 1

        # Im kompakten Heads-up-Layout sind einzelne helle UI-Elemente haeufige
        # False-Positives. Teure OCR/Template-Loops starten wir deshalb erst,
        # wenn mindestens ein plausibler Flop-Block sichtbar ist.
        if compact_heads_up_layout and surface_count < 3:
            return [], 0

        detected_cards: List[Card] = []
        saw_gap = False
        for index, region in enumerate(normalized_regions, start=1):
            x, y, w, h = region
            if w <= 0 or h <= 0:
                continue

            roi = screenshot_bgr[y:y+h, x:x+w]
            if not surface_flags[index - 1]:
                if detected_cards:
                    saw_gap = True
                continue

            card = self._process_card_roi(roi, context=f"board{index}")
            if self.layout_name == "heads_up" and index >= 3:
                surface = self._extract_card_surface(roi)
                needs_expanded_retry = card is None
                if surface is not None and surface.size > 0 and self._is_left_edge_contaminated(surface):
                    needs_expanded_retry = True
                if needs_expanded_retry:
                    expanded_x = max(0, x - 18)
                    expanded_w = min(img_w - expanded_x, w + 24)
                    detect_roi = screenshot_bgr[y:y+h, expanded_x:expanded_x+expanded_w]
                    expanded_card = self._process_card_roi(detect_roi, context=f"board{index}")
                    if expanded_card:
                        card = expanded_card
            if card:
                if card not in detected_cards:
                    detected_cards.append(card)
            elif saw_gap:
                break
        return detected_cards, surface_count

    def _detect_cards_for_layout_profile(
        self,
        screenshot_bgr: np.ndarray,
        table_coords: Tuple[int, int, int, int],
        layout_name: str,
    ) -> Dict[str, object]:
        previous_layout = self.layout_name
        previous_rois = self.table_rois
        previous_ref = (self.reference_width, self.reference_height)
        self._apply_layout_profile(layout_name)
        try:
            hole_cards = self._detect_hole_cards_for_current_layout(screenshot_bgr, table_coords)
            community_cards, surface_count = self._detect_community_cards_for_current_layout(screenshot_bgr, table_coords)
            score = (len(hole_cards) * 12.0) + (len(community_cards) * 6.0) + (surface_count * 2.0)
            if len(hole_cards) == 2:
                score += 3.0
            if surface_count and not community_cards:
                score -= 4.0
            return {
                "layout_name": layout_name,
                "hole_cards": hole_cards,
                "community_cards": community_cards,
                "surface_count": surface_count,
                "score": score,
            }
        finally:
            self.layout_name = previous_layout
            self.table_rois = previous_rois
            self.reference_width, self.reference_height = previous_ref

    def _get_frame_detection(
        self,
        screenshot_bgr: np.ndarray,
        table_coords: Tuple[int, int, int, int],
    ) -> Dict[str, object]:
        self._refresh_layout()
        frame_key = self._make_frame_cache_key(screenshot_bgr)
        if frame_key is not None and frame_key in self._frame_detection_cache:
            return self._frame_detection_cache[frame_key]

        self._clear_roi_cache()

        layout_hint = self._infer_layout_hint_from_title(screenshot_bgr)
        primary_layout = layout_hint or self.layout_name
        primary = self._detect_cards_for_layout_profile(screenshot_bgr, table_coords, primary_layout)
        best = primary

        primary_board_count = len(primary["community_cards"])
        primary_is_complete = (
            len(primary["hole_cards"]) == 2
            and primary_board_count in {0, 3, 4, 5}
        )
        primary_needs_backup = (
            not primary_is_complete
            or (len(primary["hole_cards"]) == 0 and primary_board_count == 0)
        )

        if primary_needs_backup:
            candidate_layouts = []
            for layout_name in self._candidate_layout_profiles():
                if layout_name == primary_layout:
                    continue
                candidate_layouts.append(layout_name)

            for layout_name in candidate_layouts:
                candidate = self._detect_cards_for_layout_profile(screenshot_bgr, table_coords, layout_name)
                if len(best["hole_cards"]) < 2 and len(candidate["hole_cards"]) > len(best["hole_cards"]):
                    best = candidate
                    continue
                if (
                    len(best["hole_cards"]) == len(candidate["hole_cards"]) == 2
                    and len(best["community_cards"]) == 0
                    and len(candidate["community_cards"]) >= 3
                ):
                    best = candidate
                    continue
                if (
                    len(best["community_cards"]) < 3
                    and len(candidate["hole_cards"]) >= len(best["hole_cards"])
                    and len(candidate["community_cards"]) >= len(best["community_cards"]) + 2
                ):
                    best = candidate
                    continue
                if (
                    len(candidate["hole_cards"]) >= len(best["hole_cards"])
                    and len(best["hole_cards"]) < 2
                    and len(candidate["community_cards"]) >= len(best["community_cards"]) + 2
                ):
                    best = candidate
                    continue
                if (
                    len(best["hole_cards"]) == 0
                    and len(candidate["community_cards"]) > len(best["community_cards"]) + 1
                ):
                    best = candidate
                    continue
                if candidate["score"] > best["score"] + 14 and len(candidate["hole_cards"]) >= len(best["hole_cards"]) + 1:
                    best = candidate

        self._apply_layout_profile(best["layout_name"])
        detection = {
            "layout_name": best["layout_name"],
            "hole_cards": best["hole_cards"],
            "community_cards": best["community_cards"],
        }
        self._store_frame_detection(frame_key, detection)
        return detection

    def _get_card_template_stats(self, card_gray: np.ndarray) -> Tuple[Optional[str], float, float, Dict[str, float]]:
        scores: Dict[str, float] = {}
        best_name = None
        best_score = -1.0
        second_best = -1.0

        for card_name, template in self.card_templates.items():
            score = self._score_card_template(card_gray, template)
            scores[card_name] = score
            if score > best_score:
                second_best = best_score
                best_score = score
                best_name = card_name
            elif score > second_best:
                second_best = score

        return best_name, float(best_score), float(second_best), scores

    def _get_card_template_stats_from_map(
        self,
        card_gray: np.ndarray,
        template_map: Dict[str, List[np.ndarray]],
    ) -> Tuple[Optional[str], float, float, Dict[str, float]]:
        scores: Dict[str, float] = {}
        best_name = None
        best_score = -1.0
        second_best = -1.0

        for card_name, templates in template_map.items():
            label_best = -1.0
            for template in templates:
                score = self._score_card_template(card_gray, template)
                label_best = max(label_best, score)
            scores[card_name] = label_best
            if label_best > best_score:
                second_best = best_score
                best_score = label_best
                best_name = card_name
            elif label_best > second_best:
                second_best = label_best

        return best_name, float(best_score), float(second_best), scores

    def _get_top_card_matches(self, card_gray: np.ndarray, topn: int = 5) -> List[Tuple[str, float]]:
        _, _, _, scores = self._get_card_template_stats(card_gray)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:topn]

    def debug_analyze_card_roi(self, roi_bgr: np.ndarray, context: str = "generic") -> Dict[str, object]:
        result: Dict[str, object] = {
            'roi_shape': None if roi_bgr is None else tuple(roi_bgr.shape),
            'surface_shape': None,
            'corner_rank': None,
            'corner_suit': None,
            'corner_card': None,
            'rank_top_matches': [],
            'suit_top_matches': [],
            'card_top_matches': [],
            'rank_patch': None,
            'suit_patch': None,
            'surface': None,
        }
        if roi_bgr is None or roi_bgr.size == 0:
            return result

        if context in {"hero1", "hero2"}:
            surface = self._extract_hero_card_surface(roi_bgr, context)
        else:
            surface = self._extract_card_surface(roi_bgr)
        if surface is None or surface.size == 0:
            return result

        result['surface'] = surface
        result['surface_shape'] = tuple(surface.shape)
        rank_patch, suit_patch = self._extract_corner_regions(surface, context=context)
        result['rank_patch'] = rank_patch
        result['suit_patch'] = suit_patch

        if rank_patch is not None and rank_patch.size > 0:
            if context.startswith("hero") and self.hero_rank_templates:
                rank_template_map = self.hero_rank_templates
            elif context.startswith("board") and self.board_rank_templates:
                rank_template_map = self.board_rank_templates
            else:
                rank_template_map = self.rank_corner_templates
            result['rank_top_matches'] = self._get_top_symbol_matches(
                cv2.cvtColor(rank_patch, cv2.COLOR_BGR2GRAY) if len(rank_patch.shape) == 3 else rank_patch,
                rank_template_map,
            )
        if suit_patch is not None and suit_patch.size > 0:
            if context.startswith("hero") and self.hero_suit_templates:
                suit_template_map = self.hero_suit_templates
            elif context.startswith("board") and self.board_suit_templates:
                suit_template_map = self.board_suit_templates
            else:
                suit_template_map = self.suit_corner_templates
            result['suit_top_matches'] = self._get_top_symbol_matches(
                cv2.cvtColor(suit_patch, cv2.COLOR_BGR2GRAY) if len(suit_patch.shape) == 3 else suit_patch,
                suit_template_map,
            )

        rank, _, _ = self._detect_rank_from_corner(surface, context=context)
        suit, _, _ = self._detect_suit_from_corner(surface, context=context)
        result['corner_rank'] = rank
        result['corner_suit'] = suit
        if rank and suit:
            result['corner_card'] = f"{rank}{suit}"

        card_gray = cv2.cvtColor(surface, cv2.COLOR_BGR2GRAY)
        result['card_top_matches'] = self._get_top_card_matches(card_gray)
        return result

    def _extract_card_surface(self, roi_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Versucht die eigentliche Kartenflaeche innerhalb der ROI zu isolieren."""
        def default_inset_crop() -> np.ndarray:
            h, w = roi_bgr.shape[:2]
            mx = max(2, int(w * 0.08))
            my = max(2, int(h * 0.06))
            return roi_bgr[my:h-my, mx:w-mx]

        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 120], dtype=np.uint8), np.array([180, 120, 255], dtype=np.uint8))
        white_ratio = cv2.countNonZero(white_mask) / float(white_mask.size)
        if white_ratio < 0.08:
            # Fallback: ROI ist bereits gut platziert, also nur den Rand abziehen.
            return default_inset_crop()

        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return default_inset_crop()

        roi_h, roi_w = roi_bgr.shape[:2]
        roi_center_x = roi_w / 2.0
        best_bbox = None
        best_score = -1.0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < roi_w * 0.18 or h < roi_h * 0.40:
                continue

            area_ratio = (w * h) / float(roi_w * roi_h)
            aspect_ratio = w / float(max(h, 1))
            center_x = x + (w / 2.0)
            center_penalty = abs(center_x - roi_center_x) / max(roi_w, 1)

            score = area_ratio
            if 0.45 <= aspect_ratio <= 0.95:
                score += 0.35
            elif 0.30 <= aspect_ratio <= 1.10:
                score += 0.10
            score -= center_penalty * 0.40

            if score > best_score:
                best_score = score
                best_bbox = (x, y, w, h)

        if best_bbox is None:
            return default_inset_crop()
        x, y, w, h = best_bbox

        x = max(0, x)
        y = max(0, y)
        w = min(w, roi_bgr.shape[1] - x)
        h = min(h, roi_bgr.shape[0] - y)
        inset_x = max(1, int(w * 0.04))
        inset_y = max(1, int(h * 0.03))
        cropped = roi_bgr[y+inset_y:y+h-inset_y, x+inset_x:x+w-inset_x]
        if cropped.size == 0:
            return default_inset_crop()

        # Wenn die Segmentierung nur eine schmale Kartenecke erwischt, nutze lieber die ROI selbst.
        if cropped.shape[1] < roi_bgr.shape[1] * 0.72 or cropped.shape[0] < roi_bgr.shape[0] * 0.72:
            return default_inset_crop()
        return cropped

    def _trim_hero_card_surface(self, card_surface_bgr: np.ndarray) -> np.ndarray:
        if card_surface_bgr is None or card_surface_bgr.size == 0:
            return card_surface_bgr

        h, w = card_surface_bgr.shape[:2]
        if w < 20:
            return card_surface_bgr

        sample = card_surface_bgr[:max(1, int(h * 0.42)), :]
        hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(
            hsv,
            np.array([14, 60, 90], dtype=np.uint8),
            np.array([50, 255, 255], dtype=np.uint8),
        )
        white_mask = cv2.inRange(
            hsv,
            np.array([0, 0, 175], dtype=np.uint8),
            np.array([180, 70, 255], dtype=np.uint8),
        )

        cut_x = 0
        scan_limit = min(w, max(10, int(w * 0.24)))
        last_yellow_x = -1
        for x in range(scan_limit):
            yellow_ratio = float(np.mean(yellow_mask[:, x] > 0))
            white_ratio = float(np.mean(white_mask[:, x] > 0))
            if yellow_ratio > 0.22:
                last_yellow_x = x
            if x >= 3 and white_ratio > 0.55 and yellow_ratio < 0.18:
                cut_x = max(0, x - 1)
                break

        if cut_x == 0 and last_yellow_x >= 0:
            cut_x = min(w - 1, last_yellow_x + 2)

        if cut_x <= 0:
            return card_surface_bgr

        trimmed = card_surface_bgr[:, cut_x:]
        return trimmed if trimmed.size > 0 else card_surface_bgr

    def _extract_hero_card_surface(self, roi_bgr: np.ndarray, context: str) -> Optional[np.ndarray]:
        if roi_bgr is None or roi_bgr.size == 0:
            return None

        h, w = roi_bgr.shape[:2]
        top = max(1, int(h * 0.02))
        bottom = max(top + 1, h - max(1, int(h * 0.04)))

        if context == "hero1":
            left = max(0, int(w * 0.02))
            right = min(w, max(left + 1, int(w * 0.82)))
        else:
            # Use 29% instead of 22% to reliably eliminate the red/orange selection
            # indicator border that appears on the left edge of hero2 cards
            left = max(0, int(w * 0.29))
            right = min(w, max(left + 1, int(w * 0.96)))

        cropped = roi_bgr[top:bottom, left:right]
        if cropped.size == 0:
            return roi_bgr
        if self.layout_name == "acipayam_heads_up":
            return cropped

        isolated = self._extract_card_surface(cropped)
        if isolated is None or isolated.size == 0:
            isolated = cropped
        return isolated

    def _has_community_card_surface(self, roi_bgr: np.ndarray) -> bool:
        """Prueft strikt, ob in einer Board-ROI wirklich eine Kartenflaeche liegt."""
        if roi_bgr is None or roi_bgr.size == 0:
            return False

        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # Swiss-Casino-Boardkarten sind hell/grau mit wenig Sättigung.
        bright_low_sat = cv2.inRange(
            hsv,
            np.array([0, 0, 105], dtype=np.uint8),
            np.array([180, 95, 255], dtype=np.uint8),
        )

        # Nimmt auch pink umrandete Karten mit, ohne komplett auf UI-Farben hereinzufallen.
        pink_mask = cv2.inRange(
            hsv,
            np.array([135, 45, 105], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )

        mask = cv2.bitwise_or(bright_low_sat, pink_mask)
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_ratio = float(np.mean(mask > 0))
        if mask_ratio >= 0.45:
            return True

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False

        roi_h, roi_w = roi_bgr.shape[:2]
        roi_area = float(roi_h * roi_w)
        contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(contour)
        if contour_area < roi_area * 0.16:
            return False

        x, y, w, h = cv2.boundingRect(contour)
        if w < roi_w * 0.38 or h < roi_h * 0.52:
            return False

        aspect_ratio = w / float(max(h, 1))
        if not 0.42 <= aspect_ratio <= 0.95:
            return False

        bbox = gray[y:y+h, x:x+w]
        if bbox.size == 0:
            return False

        bright_ratio = float(np.mean(bbox > 105))
        if bright_ratio < 0.45:
            return False

        return True

    def _prepare_rank_candidates(self, card_surface_bgr: np.ndarray) -> List[np.ndarray]:
        h, w = card_surface_bgr.shape[:2]
        x1 = int(w * 0.18)
        x2 = max(x1 + 1, int(w * 0.50))
        y2 = max(1, int(h * 0.42))
        rank_roi = card_surface_bgr[0:y2, x1:x2]
        hsv = cv2.cvtColor(rank_roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(rank_roi, cv2.COLOR_BGR2GRAY)
        yellow_mask = (
            (hsv[:, :, 0] >= 14) & (hsv[:, :, 0] <= 55) &
            (hsv[:, :, 1] >= 55) &
            (hsv[:, :, 2] >= 80)
        )

        non_white_mask = (
            (((hsv[:, :, 1] > 35) & (hsv[:, :, 2] > 40) & (~yellow_mask))) |
            (gray < 170)
        ).astype(np.uint8) * 255

        contours, _ = cv2.findContours(non_white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        bounding_boxes = []
        for contour in contours:
            x, y, ww, hh = cv2.boundingRect(contour)
            if ww < 4 or hh < 8:
                continue
            if x > rank_roi.shape[1] * 0.55:
                continue
            area = ww * hh
            bounding_boxes.append((x, y, ww, hh, area))

        bounding_boxes.sort(key=lambda item: (item[0], -item[4]))
        if bounding_boxes:
            x, y, ww, hh, _ = bounding_boxes[0]
            pad = 3
            x1b = max(0, x - pad)
            y1b = max(0, y - pad)
            x2b = min(rank_roi.shape[1], x + ww + pad)
            y2b = min(rank_roi.shape[0], y + hh + pad)
            glyph_roi = rank_roi[y1b:y2b, x1b:x2b]
            glyph_gray = cv2.cvtColor(glyph_roi, cv2.COLOR_BGR2GRAY)
            glyph_thresh = self._normalize_rank_image(glyph_gray)
            candidates.append(glyph_thresh)

        upscaled_mask = self._normalize_rank_image(non_white_mask)
        thresh_gray = self._normalize_rank_image(gray)
        candidates.append(upscaled_mask)
        candidates.append(thresh_gray)

        return candidates

    def _extract_corner_regions(
        self,
        card_surface_bgr: np.ndarray,
        context: str = "generic",
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if card_surface_bgr is None or card_surface_bgr.size == 0:
            return None, None

        surface = card_surface_bgr
        if context.startswith("hero"):
            h, w = surface.shape[:2]
            rank_x1 = int(w * 0.02)
            rank_x2 = max(rank_x1 + 1, int(w * 0.28))
            rank_y1 = int(h * 0.00)
            rank_y2 = max(rank_y1 + 1, int(h * 0.28))
            suit_x1 = int(w * 0.04)
            suit_x2 = max(suit_x1 + 1, int(w * 0.26))
            suit_y1 = int(h * 0.18)
            suit_y2 = max(suit_y1 + 1, int(h * 0.48))
            rank_raw = surface[rank_y1:rank_y2, rank_x1:rank_x2]
            suit_raw = surface[suit_y1:suit_y2, suit_x1:suit_x2]
            rank_roi = self._extract_symbol_patch(
                rank_raw,
                (0.00, 0.00, 1.00, 1.00),
                max_x_ratio=0.95,
                min_area=8,
                prefer_upper=True,
            )
            suit_roi = self._extract_symbol_patch(
                suit_raw,
                (0.00, 0.00, 1.00, 1.00),
                max_x_ratio=0.95,
                min_area=8,
                prefer_upper=False,
            )
            if rank_roi is None or rank_roi.size == 0:
                rank_roi = rank_raw
            if suit_roi is None or suit_roi.size == 0:
                suit_roi = suit_raw
        else:
            if context.startswith("board"):
                rank_box = (0.00, 0.00, 0.30, 0.26)
                suit_box = (0.04, 0.24, 0.26, 0.60)
                rank_max_x = 0.60
                suit_max_x = 0.58
            else:
                rank_box = (0.00, 0.00, 0.38, 0.28)
                suit_box = (0.00, 0.10, 0.34, 0.42)
                rank_max_x = 0.72
                suit_max_x = 0.70
            rank_roi = self._extract_symbol_patch(
                surface,
                rank_box,
                max_x_ratio=rank_max_x,
                min_area=16,
                prefer_upper=True,
            )
            suit_roi = self._extract_symbol_patch(
                surface,
                suit_box,
                max_x_ratio=suit_max_x,
                min_area=12,
                prefer_upper=False,
            )
        if rank_roi.size == 0:
            rank_roi = None
        if suit_roi.size == 0:
            suit_roi = None
        return rank_roi, suit_roi

    def _detect_rank_from_corner(
        self,
        card_surface_bgr: np.ndarray,
        context: str = "generic",
    ) -> Tuple[Optional[str], float, float]:
        rank_roi, _ = self._extract_corner_regions(card_surface_bgr, context=context)
        if rank_roi is None:
            return None, -1.0, -1.0

        rank_gray = cv2.cvtColor(rank_roi, cv2.COLOR_BGR2GRAY)
        template_maps: List[Dict[str, List[np.ndarray]]] = []
        if context.startswith("hero") and self.layout_name == "acipayam_heads_up" and self.hero_rank_templates:
            template_maps.append(self.hero_rank_templates)
        if context.startswith("board") and self.layout_name == "acipayam_heads_up" and self.board_rank_templates:
            template_maps.append(self.board_rank_templates)
        if self.rank_corner_templates:
            template_maps.append(self.rank_corner_templates)

        matched_rank = None
        best_score = -1.0
        second_best = -1.0
        for template_map in template_maps:
            candidate_rank, candidate_best, candidate_second = self._get_symbol_template_stats(
                rank_gray,
                template_map,
            )
            if candidate_best > best_score:
                matched_rank = candidate_rank
                best_score = candidate_best
                second_best = candidate_second
        if matched_rank and best_score >= self.min_corner_rank_match_threshold and (best_score - second_best) >= 0.02:
            return matched_rank, best_score, second_best

        candidates = self._prepare_rank_candidates(rank_roi)
        for candidate in candidates:
            matched_rank = self._match_rank_template(candidate)
            if matched_rank:
                return matched_rank, best_score, second_best
        return None, best_score, second_best

    def _detect_suit_from_corner(
        self,
        card_surface_bgr: np.ndarray,
        context: str = "generic",
    ) -> Tuple[Optional[str], float, float]:
        _, suit_roi = self._extract_corner_regions(card_surface_bgr, context=context)
        if suit_roi is None:
            return None, -1.0, -1.0

        hsv = cv2.cvtColor(suit_roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(suit_roi, cv2.COLOR_BGR2GRAY)
        pixels = hsv.reshape(-1, 3)
        gray_pixels = gray.reshape(-1)

        green_count = 0
        blue_count = 0
        red_count = 0
        black_count = 0

        for (hue, sat, val), gray_val in zip(pixels, gray_pixels):
            if gray_val < 70 or val < 70:
                black_count += 1
                continue
            if sat < 35:
                continue
            if 35 <= hue <= 90:
                green_count += 1
            elif 90 < hue <= 135:
                blue_count += 1
            elif hue <= 15 or hue >= 170:
                red_count += 1

        color_pixels = green_count + blue_count + red_count
        if not context.startswith("hero") and not context.startswith("board") and color_pixels >= 8:
            if green_count >= blue_count and green_count >= red_count:
                return 'C', 1.0, 0.0
            if blue_count >= green_count and blue_count >= red_count:
                return 'D', 1.0, 0.0
            return 'H', 1.0, 0.0

        suit_gray = cv2.cvtColor(suit_roi, cv2.COLOR_BGR2GRAY)
        template_maps: List[Dict[str, List[np.ndarray]]] = []
        if context.startswith("hero") and self.layout_name == "acipayam_heads_up" and self.hero_suit_templates:
            template_maps.append(self.hero_suit_templates)
        if context.startswith("board") and self.layout_name == "acipayam_heads_up" and self.board_suit_templates:
            template_maps.append(self.board_suit_templates)
        if self.suit_corner_templates:
            template_maps.append(self.suit_corner_templates)

        matched_suit = None
        best_score = -1.0
        second_best = -1.0
        for template_map in template_maps:
            candidate_suit, candidate_best, candidate_second = self._get_symbol_template_stats(
                suit_gray,
                template_map,
            )
            if candidate_best > best_score:
                matched_suit = candidate_suit
                best_score = candidate_best
                second_best = candidate_second
        if matched_suit and best_score >= self.min_corner_suit_match_threshold and (best_score - second_best) >= 0.02:
            return matched_suit, best_score, second_best

        total = len(gray_pixels) if len(gray_pixels) else 1
        if black_count / total > 0.10:
            return 'S', 0.45, 0.0
        return None, best_score, second_best

    def _detect_card_from_corner(self, card_surface_bgr: np.ndarray, context: str = "generic") -> Optional[Card]:
        rank, rank_score, rank_second = self._detect_rank_from_corner(card_surface_bgr, context=context)
        suit, suit_score, suit_second = self._detect_suit_from_corner(card_surface_bgr, context=context)
        if not rank or not suit:
            return None

        card_code = f"{rank}{suit}"
        template = self.card_templates.get(card_code)
        if template is not None:
            card_gray = cv2.cvtColor(card_surface_bgr, cv2.COLOR_BGR2GRAY)
            best_name, best_score, second_best, scores = self._get_card_template_stats(card_gray)
            exact_score = scores.get(card_code, -1.0)
            gap_to_best = best_score - exact_score
            rank_gap = rank_score - rank_second
            suit_gap = suit_score - suit_second

            if context.startswith("hero"):
                hero_accept = (
                    rank_score >= 0.95
                    and suit_score >= self.min_corner_suit_match_threshold
                    and rank_gap >= 0.01
                    and suit_gap >= 0.01
                )
                if hero_accept:
                    card = parse_card_string(card_code)
                    if card:
                        logger.debug(f"Hero-Karte ueber Hero-Templates erkannt: {card}")
                        return card

            if context.startswith("board"):
                board_accept = (
                    (rank_score >= 0.90 or (rank_score >= 0.48 and rank_gap >= 0.15))
                    and suit_score >= self.min_corner_suit_match_threshold
                    and rank_gap >= 0.01
                    and suit_gap >= 0.01
                )
                if board_accept:
                    card = parse_card_string(card_code)
                    if card:
                        logger.debug(f"Board-Karte ueber Board-Templates erkannt: {card}")
                        return card

            generic_corner_accept = (
                self.layout_name != "acipayam_heads_up"
                and rank_score >= max(0.48, self.min_corner_rank_match_threshold)
                and suit_score >= max(0.34, self.min_corner_suit_match_threshold)
                and rank_gap >= 0.01
                and suit_gap >= 0.0
            )
            if generic_corner_accept:
                card = parse_card_string(card_code)
                if card:
                    logger.debug(f"Karte ueber generische Ecke erkannt: {card}")
                    return card

            accept_exact = (
                exact_score >= 0.56
                and (
                    best_name == card_code
                    or gap_to_best <= 0.06
                    or exact_score >= second_best + 0.03
                    or (
                        rank_score >= self.min_corner_rank_match_threshold
                        and suit_score >= self.min_corner_suit_match_threshold
                        and rank_gap >= 0.02
                        and suit_gap >= 0.02
                        and exact_score >= 0.48
                    )
                )
            )
            if not accept_exact and exact_score < 0.64:
                return None

        card = parse_card_string(card_code)
        if card:
            logger.debug(f"Karte ueber feste Ecke erkannt: {card}")
        return card

    def _match_rank_template(self, candidate: np.ndarray) -> Optional[str]:
        if not self.rank_templates:
            return None

        best_rank = None
        best_score = -1.0
        for rank, template in self.rank_templates.items():
            normalized_candidate = self._normalize_rank_image(candidate)
            resized_candidate = cv2.resize(normalized_candidate, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_NEAREST)
            result = cv2.matchTemplate(resized_candidate, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(result)
            if score > best_score:
                best_rank = rank
                best_score = score

        if best_rank and best_score >= self.min_rank_match_threshold:
            return best_rank
        return None

    def _ocr_rank(self, card_surface_bgr: np.ndarray) -> Optional[str]:
        candidates = self._prepare_rank_candidates(card_surface_bgr)

        for candidate in candidates:
            matched_rank = self._match_rank_template(candidate)
            if matched_rank:
                return matched_rank

        config = '--psm 10 -c tessedit_char_whitelist=A23456789TJQK10'
        for candidate in candidates:
            text = pytesseract.image_to_string(candidate, config=config)
            cleaned = ''.join(ch for ch in text.upper() if ch.isalnum())
            if cleaned.startswith('10'):
                return 'T'
            if cleaned:
                rank = cleaned[0]
                rank_aliases = {'0': 'Q', 'O': 'Q', 'I': 'T', 'L': 'T'}
                rank = rank_aliases.get(rank, rank)
                if rank in self.ranks:
                    return rank
        return None

    def _detect_suit(self, card_surface_bgr: np.ndarray) -> Optional[str]:
        h, w = card_surface_bgr.shape[:2]
        x1 = int(w * 0.22)
        x2 = max(x1 + 1, int(w * 0.50))
        y1 = int(h * 0.22)
        y2 = max(y1 + 1, int(h * 0.56))
        suit_roi = card_surface_bgr[y1:y2, x1:x2]
        hsv = cv2.cvtColor(suit_roi, cv2.COLOR_BGR2HSV)
        bgr = suit_roi.reshape(-1, 3)
        hsv_flat = hsv.reshape(-1, 3)

        green_count = 0
        blue_count = 0
        red_count = 0
        dark_pixels = 0
        total_pixels = len(bgr)

        for (b, g, r), (hue, sat, val) in zip(bgr, hsv_flat):
            if val < 80:
                dark_pixels += 1
                continue
            if sat < 40:
                continue
            if 35 <= hue <= 90:
                green_count += 1
            elif 90 < hue <= 135:
                blue_count += 1
            elif hue <= 15 or hue >= 170:
                red_count += 1

        colored_total = green_count + blue_count + red_count
        colored_ratio = (colored_total / total_pixels) if total_pixels else 0.0

        if colored_total and colored_ratio > 0.02:
            if green_count >= blue_count and green_count >= red_count:
                return 'C'
            if blue_count >= green_count and blue_count >= red_count:
                return 'D'
            if red_count >= green_count and red_count >= blue_count:
                return 'H'

        if total_pixels and dark_pixels / total_pixels > 0.12:
            return 'S'
        return None

    def _ocr_card_from_surface(self, card_surface_bgr: np.ndarray) -> Optional[Card]:
        rank = self._ocr_rank(card_surface_bgr)
        suit = self._detect_suit(card_surface_bgr)
        if rank and suit:
            card = parse_card_string(f"{rank}{suit}")
            if card:
                logger.debug(f"Karte ueber OCR/Farb-Fallback erkannt: {card}")
                return card
        return None

    def _process_card_roi(self, roi_bgr: np.ndarray, context: str = "generic") -> Optional[Card]:
        """ Verarbeitet einen Region-of-Interest (ROI) zur Kartenerkennung. """
        if roi_bgr is None or roi_bgr.size == 0:
            return None

        cache_hit, cached_card, cache_key = self._get_cached_roi_result(roi_bgr, context)
        if cache_hit:
            return cached_card

        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        if context in {"hero1", "hero2"}:
            card_surface = self._extract_hero_card_surface(roi_bgr, context)
        else:
            card_surface = self._extract_card_surface(roi_bgr)
        if card_surface is not None and card_surface.size > 0:
            match_gray = cv2.cvtColor(card_surface, cv2.COLOR_BGR2GRAY)
        else:
            match_gray = roi_gray

        # Fallback hero template match (score >= 0.80 but gap too small for immediate return)
        _hero_fallback_name: Optional[str] = None
        _hero_fallback_score: float = -1.0

        if self.layout_name == 'acipayam_heads_up' and card_surface is not None and card_surface.size > 0:
            if context.startswith("hero") and self.hero_card_templates:
                hero_name, hero_score, hero_second, _ = self._get_card_template_stats_from_map(
                    match_gray,
                    self.hero_card_templates,
                )
                hero_gap = hero_score - hero_second
                if (
                    hero_name
                    and hero_score >= 0.80
                    and hero_gap >= 0.02
                ):
                    card = parse_card_string(hero_name)
                    if card:
                        logger.debug(
                            f"Hero-Karte ueber Vollkarten-Templates erkannt: {card} "
                            f"(Konfidenz: {hero_score:.2f}, Gap: {hero_gap:.2f})"
                        )
                        self._store_cached_roi_result(cache_key, card)
                        return card
                # Save as fallback for when gap is small but score is decent
                if hero_name and hero_score >= 0.80:
                    _hero_fallback_name = hero_name
                    _hero_fallback_score = hero_score
            if context.startswith("board") and self.board_card_templates:
                # Use all board templates (not slot-specific) for complete coverage across positions
                board_name, board_score, board_second, _ = self._get_card_template_stats_from_map(
                    match_gray,
                    self.board_card_templates,
                )
                if (
                    board_name
                    and (
                        (board_score >= 0.98 and (board_score - board_second) >= 0.005)
                        or (board_score >= 0.74 and (board_score - board_second) >= 0.02)
                    )
                ):
                    card = parse_card_string(board_name)
                    if card:
                        logger.debug(
                            f"Board-Karte ueber Vollkarten-Templates erkannt: {card} "
                            f"(Konfidenz: {board_score:.2f}, Gap: {board_score - board_second:.2f})"
                        )
                        self._store_cached_roi_result(cache_key, card)
                        return card

        if (
            self.layout_name == 'heads_up'
            and card_surface is not None
            and card_surface.size > 0
            and (context.startswith("hero") or context.startswith("board"))
        ):
            focus_card = self._detect_card_from_compact_focus(card_surface, context=context)
            if focus_card:
                self._store_cached_roi_result(cache_key, focus_card)
                return focus_card

        if card_surface is not None and card_surface.size > 0:
            corner_card = self._detect_card_from_corner(card_surface, context=context)
            if corner_card:
                # For hero cards in acipayam: if hero template had a decent score
                # but corner detection disagrees, use suit to disambiguate:
                # - Same suit but different rank → trust corner (corner rank is reliable)
                # - Different suit → trust hero template (corner suit detection is error-prone)
                if (
                    _hero_fallback_name
                    and self.layout_name == 'acipayam_heads_up'
                    and context.startswith("hero")
                    and str(corner_card) != _hero_fallback_name
                ):
                    corner_suit = str(corner_card)[1] if len(str(corner_card)) == 2 else ""
                    fallback_suit = _hero_fallback_name[1] if len(_hero_fallback_name) == 2 else ""
                    if corner_suit != fallback_suit:
                        # Suit disagreement → hero template is more reliable for suit
                        fallback_card = parse_card_string(_hero_fallback_name)
                        if fallback_card:
                            logger.debug(
                                f"Hero-Karte: corner={corner_card}(suit={corner_suit}) vs "
                                f"hero-template={_hero_fallback_name}(suit={fallback_suit}) – "
                                f"Suit-Konflikt, waehle hero-template"
                            )
                            self._store_cached_roi_result(cache_key, fallback_card)
                            return fallback_card
                    # Same suit but different rank → trust corner's rank detection
                    logger.debug(
                        f"Hero-Karte: corner={corner_card} und hero-template={_hero_fallback_name} "
                        f"gleicher Suit, corner-Rang bevorzugt"
                    )
                self._store_cached_roi_result(cache_key, corner_card)
                return corner_card

        best_match_card, best_match_confidence, second_best_confidence, _ = self._get_card_template_stats(match_gray)

        if (
            best_match_card
            and best_match_confidence >= self.min_card_match_threshold
            and (best_match_confidence - second_best_confidence) >= self.min_card_match_gap
        ):
            card = parse_card_string(best_match_card)
            if card:
                logger.debug(
                    f"Karte erkannt: {card} (Konfidenz: {best_match_confidence:.2f}, Gap: {best_match_confidence - second_best_confidence:.2f})"
                )
                self._store_cached_roi_result(cache_key, card)
                return card

        if self.layout_name == 'acipayam_heads_up':
            # Use hero fallback if generic templates also failed
            if _hero_fallback_name:
                fallback_card = parse_card_string(_hero_fallback_name)
                if fallback_card:
                    logger.debug(
                        f"Hero-Karte letzter Fallback: {fallback_card} (score={_hero_fallback_score:.2f})"
                    )
                    self._store_cached_roi_result(cache_key, fallback_card)
                    return fallback_card
            self._store_cached_roi_result(cache_key, None)
            return None

        if card_surface is not None and card_surface.size > 0:
            ocr_card = self._ocr_card_from_surface(card_surface)
            if ocr_card:
                self._store_cached_roi_result(cache_key, ocr_card)
                return ocr_card

        # Letzter Fallback: OCR auf der gesamten ROI
        try:
            config = '--psm 10' # Annahme: einzelne Zeichen
            text = pytesseract.image_to_string(roi_gray, config=config)
            cleaned_text = ''.join(filter(str.isalnum, text)).upper()
            
            # Einfache Heuristik: Suche nach Rang+Farbe Kombinationen
            possible_cards = []
            for rank in self.ranks:
                 for suit in self.suits:
                      card_str = f"{rank}{suit}"
                      if card_str in cleaned_text:
                           parsed_card = parse_card_string(card_str)
                           if parsed_card: possible_cards.append(parsed_card)

            if possible_cards:
                 logger.debug(f"OCR Fallback fand Karten: {possible_cards}")
                 # Gib die erste gefundene Karte zurück (oder eine robustere Auswahl)
                 self._store_cached_roi_result(cache_key, possible_cards[0])
                 return possible_cards[0]
                 
        except Exception as e:
            logger.warning(f"OCR Fallback für Karten fehlgeschlagen: {e}")

        self._store_cached_roi_result(cache_key, None)
        return None

    def detect_cards_in_regions(
        self,
        screenshot_bgr: np.ndarray,
        regions: List[Tuple[int, int, int, int]],
        context: str = "generic",
    ) -> List[Optional[Card]]:
        """ Erkennt Karten in einer Liste von definierten Regionen (ROIs). """
        detected_cards = []
        for i, region in enumerate(regions):
            x, y, w, h = region
            if w <= 0 or h <= 0: 
                detected_cards.append(None)
                continue

            # Stelle sicher, dass die ROI innerhalb des Bildes liegt
            img_h, img_w = screenshot_bgr.shape[:2]
            x = max(0, x)
            y = max(0, y)
            w = min(w, img_w - x)
            h = min(h, img_h - y)

            if w <= 0 or h <= 0:
                 detected_cards.append(None)
                 continue

            roi = screenshot_bgr[y:y+h, x:x+w]
            card = self._process_card_roi(roi, context=context)
            detected_cards.append(card)
            
        return detected_cards

    def _scale_fixed_region(
        self,
        region: Tuple[int, int, int, int],
        table_coords: Tuple[int, int, int, int],
    ) -> Tuple[int, int, int, int]:
        table_x, table_y, table_w, table_h = table_coords
        x, y, w, h = region
        if (
            table_x == 0
            and table_y == 0
            and abs(table_w - self.reference_width) <= 10
            and abs(table_h - self.reference_height) <= 2
        ):
            return region
        scaled_x = table_x + int((x / self.reference_width) * table_w)
        scaled_y = table_y + int((y / self.reference_height) * table_h)
        scaled_w = max(1, int((w / self.reference_width) * table_w))
        scaled_h = max(1, int((h / self.reference_height) * table_h))
        return (scaled_x, scaled_y, scaled_w, scaled_h)

    def detect_hole_cards(self, screenshot_bgr: np.ndarray, table_coords: Tuple[int, int, int, int]) -> List[Card]:
        """ Erkennt die beiden Hole Cards des Spielers mit automatischer Heads-up-Profilwahl. """
        detection = self._get_frame_detection(screenshot_bgr, table_coords)
        return detection["hole_cards"]  # type: ignore[return-value]

    def detect_community_cards(self, screenshot_bgr: np.ndarray, table_coords: Tuple[int, int, int, int]) -> List[Card]:
        """ Erkennt die Community Cards (Flop, Turn, River) mit automatischer Heads-up-Profilwahl. """
        detection = self._get_frame_detection(screenshot_bgr, table_coords)
        return detection["community_cards"]  # type: ignore[return-value]

    def get_hole_card_regions(self, table_coords: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
        return [
            self._scale_fixed_region(region, table_coords)
            for region in self.table_rois['hero_hole_cards']
        ]

    def get_community_card_regions(self, table_coords: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
        return [
            self._scale_fixed_region(region, table_coords)
            for region in self.table_rois['community_cards']
        ]
