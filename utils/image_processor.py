import cv2
import numpy as np
import os
import pytesseract
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
from PIL import ImageGrab # Alternative für Screenshots
import logging

# Importiere Konfiguration und Logger
try:
    from config import (
        LIVE_CONFIG,
        POKER_SETTINGS,
        TABLE_TEMPLATE_PATH,
        CARD_TEMPLATES_DIR,
    )
    from logger import logger
except ImportError:
    # Fallback für grundlegende Funktionalität ohne Konfiguration
    LIVE_CONFIG = {}
    POKER_SETTINGS = {'suits': ['C', 'D', 'H', 'S'], 'ranks': ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']}
    TABLE_TEMPLATE_PATH = None
    CARD_TEMPLATES_DIR = None
    logger = logging.getLogger(__name__)

IMAGE_PROCESSING = {
    'tesseract_cmd': LIVE_CONFIG.get('tesseract_cmd'),
    'table_template_path': TABLE_TEMPLATE_PATH,
    'card_templates_dir': CARD_TEMPLATES_DIR,
    'screenshot_region': LIVE_CONFIG.get('screen_region'),
}
    
# Stelle Tesseract Pfad ein, falls nötig (z.B. unter Windows)
if os.name == 'nt' and 'tesseract_cmd' in IMAGE_PROCESSING:
    try:
        pytesseract.pytesseract.tesseract_cmd = IMAGE_PROCESSING['tesseract_cmd']
    except pytesseract.TesseractNotFoundError:
        logger.error(f"Tesseract OCR ist nicht korrekt installiert oder der Pfad '{IMAGE_PROCESSING['tesseract_cmd']}' ist falsch.")

# Möglicherweise 'mss' verwenden, wenn es installiert ist
try:
    import mss
    USE_MSS = True
except ImportError:
    logger.warning("Bibliothek 'mss' nicht gefunden. Nutze Pillow für Screenshots (kann langsamer sein). Installieren Sie 'mss' für bessere Performance: pip install mss")
    USE_MSS = False

class ImageProcessor:
    def __init__(self):
        self.tesseract_cmd = IMAGE_PROCESSING.get('tesseract_cmd')
        self.table_template_path = IMAGE_PROCESSING.get('table_template_path')
        self.card_templates_dir = IMAGE_PROCESSING.get('card_templates_dir')
        self.chip_colors = IMAGE_PROCESSING.get('chip_color_thresholds', {})
        self.screenshot_region = IMAGE_PROCESSING.get('screenshot_region')
        self.card_detection_threshold = IMAGE_PROCESSING.get('card_detection_threshold', 0.75)
        self.table_detection_threshold = IMAGE_PROCESSING.get('table_detection_threshold', 0.8)
        self.player_area_ratio = IMAGE_PROCESSING.get('player_area_ratio', 0.15)
        self.pot_area_ratio = IMAGE_PROCESSING.get('pot_area_ratio', 0.10)
        
        self.table_template = None
        self.card_templates = {}
        self.suits = POKER_SETTINGS.get('suits', ['C', 'D', 'H', 'S'])
        self.ranks = POKER_SETTINGS.get('ranks', ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'])
        
        self._load_templates()

    def _load_templates(self):
        """Lädt Tisch- und Karten-Templates."""
        logger.info("Lade Bild-Templates...")
        
        # Tisch-Template
        if self.table_template_path and os.path.exists(self.table_template_path):
            self.table_template = cv2.imread(self.table_template_path, cv2.IMREAD_GRAYSCALE)
            if self.table_template is None:
                logger.error(f"Fehler: Konnte Tisch-Template nicht laden: {self.table_template_path}")
            else:
                logger.info(f"Tisch-Template geladen: {self.table_template_path}")
        else:
            logger.warning("Tisch-Template Pfad nicht konfiguriert oder Datei nicht gefunden.")

        # Karten-Templates
        if self.card_templates_dir and os.path.isdir(self.card_templates_dir):
            for filename in os.listdir(self.card_templates_dir):
                card_name = os.path.splitext(filename)[0] 
                path = os.path.join(self.card_templates_dir, filename)
                template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    self.card_templates[card_name] = template
                else:
                    logger.warning(f"Konnte Karten-Template nicht laden: {path}")
            logger.info(f"{len(self.card_templates)} Karten-Templates geladen.")
        else:
            logger.warning("Karten-Template Verzeichnis nicht konfiguriert oder ungültig.")

    def get_screenshot(self) -> Optional[np.ndarray]:
        """Nimmt einen Screenshot auf (nutzt mss oder Pillow)."""
        bbox = None
        if self.screenshot_region:
            x, y, w, h = self.screenshot_region
            # Konvertiere zu (left, top, right, bottom) für Pillow/mss
            bbox = {"top": y, "left": x, "width": w, "height": h}

        try:
            if USE_MSS:
                with mss.mss() as sct:
                    # Wähle Monitor (0 = alle, 1 = primär, etc.)
                    monitor_index = 1 if bbox is None else 0 # Wenn bbox definiert, nimm den ersten Monitor, sonst den primären
                    mon = sct.monitors[monitor_index]

                    # Passe bbox an, falls nicht als dict übergeben
                    if bbox and isinstance(bbox, dict):
                         grab_bbox = bbox
                    elif bbox and not isinstance(bbox, dict): # Falls als (x,y,w,h) übergeben
                         grab_bbox = {"top": bbox[1], "left": bbox[0], "width": bbox[2], "height": bbox[3]}
                    else: # Ganzer Monitor
                         grab_bbox = mon
                         
                    # Stelle sicher, dass bbox innerhalb der Monitor-Grenzen liegt
                    grab_bbox["top"] = max(mon["top"], grab_bbox.get("top", 0))
                    grab_bbox["left"] = max(mon["left"], grab_bbox.get("left", 0))
                    grab_bbox["width"] = min(mon["width"] - grab_bbox["left"], grab_bbox.get("width", mon["width"]))
                    grab_bbox["height"] = min(mon["height"] - grab_bbox["top"], grab_bbox.get("height", mon["height"]))
                    
                    img_mss = sct.grab(grab_bbox)
                    img = np.array(img_mss)
                    # mss gibt BGRA zurück, OpenCV erwartet BGR
                    if img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    return img
            else: # Pillow als Fallback
                if bbox:
                    # Pillow erwartet (left, top, right, bottom)
                    pil_bbox = (bbox['left'], bbox['top'], bbox['left'] + bbox['width'], bbox['top'] + bbox['height'])
                else:
                    pil_bbox = None # Ganzer Bildschirm
                
                img_pil = ImageGrab.grab(bbox=pil_bbox)
                img = np.array(img_pil)
                # Pillow gibt RGB zurück, OpenCV erwartet BGR
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return img_bgr
                
        except Exception as e:
            logger.error(f"Fehler beim Erstellen des Screenshots: {e}")
            return None

    def find_template(self, image_bgr: np.ndarray, template_gray: np.ndarray, threshold: float) -> Optional[Tuple[int, int]]:
        """Findet ein Template in einem Bild mittels Template Matching."""
        if template_gray is None or image_bgr is None:
            return None
            
        img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Sicherstellen, dass Template nicht größer als Bild ist
        if template_gray.shape[0] > img_gray.shape[0] or template_gray.shape[1] > img_gray.shape[1]:
             return None
             
        res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if max_val >= threshold:
            return max_loc 
        return None

    def find_table(self, screenshot: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Findet den Spieltisch im Screenshot."""
        if self.table_template is None:
            logger.warning("Kein Tisch-Template zum Finden vorhanden.")
            return None
            
        location = self.find_template(screenshot, self.table_template, self.table_detection_threshold)
        
        if location:
            logger.debug(f"Tisch gefunden bei: {location}")
            return (location[0], location[1], self.table_template.shape[1], self.table_template.shape[0])
        else:
            logger.debug("Tisch-Template nicht gefunden.")
            return None

    def detect_player_areas(self, table_coords: Tuple[int, int, int, int], num_players: int) -> List[Tuple[int, int, int, int]]:
        """ Schätzt die Bereiche für Spieler-Infos. (HEURISTIK - Muss angepasst werden!) """
        table_x, table_y, table_w, table_h = table_coords
        player_areas = []
        info_height = int(table_h * self.player_area_ratio) 
        info_y_start = table_y + table_h
        info_width = table_w
        
        if num_players > 0:
            segment_width = info_width / num_players
            for i in range(num_players):
                area_x = table_x + int(i * segment_width)
                area_w = int(segment_width)
                area_h = int(info_height * 0.8)
                area_y = info_y_start + int(info_height * 0.1) 
                player_areas.append((area_x, area_y, area_w, area_h))
        logger.debug(f"{num_players} Spielerbereiche geschätzt.")
        return player_areas

    def get_pot_area(self, table_coords: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """ Schätzt den Pot-Bereich. (HEURISTIK - Muss angepasst werden!) """
        table_x, table_y, table_w, table_h = table_coords
        pot_area_height = int(table_h * self.pot_area_ratio)
        pot_area_width = int(table_w * 0.2) 
        pot_x = table_x + (table_w - pot_area_width) // 2
        pot_y = table_y + (table_h - pot_area_height) // 2 - int(table_h * 0.1)
        logger.debug(f"Pot-Bereich geschätzt: ({pot_x}, {pot_y}, {pot_area_width}, {pot_area_height})")
        return (pot_x, pot_y, pot_area_width, pot_area_height)

    def detect_chips_in_area(self, image: np.ndarray, area: Tuple[int, int, int, int]) -> Dict[str, int]:
        """ Erkennt Chip-Stapel in einem Bereich anhand der Farbe. """
        x, y, w, h = area
        if w <= 0 or h <= 0: return {}
        
        roi = image[y:y+h, x:x+w]
        if roi.size == 0: return {}
        
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        chip_counts = defaultdict(int)
        
        for color_name, thresholds in self.chip_colors.items():
            lower_bound = np.array(thresholds['hsv_lower'], dtype=np.uint8)
            upper_bound = np.array(thresholds['hsv_upper'], dtype=np.uint8)
            
            mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)
            pixel_count = cv2.countNonZero(mask)
            
            if pixel_count > (w * h * 0.01): # Mindestens 1% der Fläche muss die Farbe haben
                chip_counts[color_name] += 1
                
        logger.debug(f"Chip-Farben erkannt: {dict(chip_counts)}")
        return dict(chip_counts)

    def detect_cards(self, image: np.ndarray, table_coords: Tuple[int, int, int, int]) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """ Versucht, Karten (Hole Cards, Community Cards) zu finden. """
        table_x, table_y, table_w, table_h = table_coords
        detected_cards = []
        
        # Bounds checking for table coordinates
        img_h, img_w = image.shape[:2]
        if table_y < 0 or table_x < 0 or table_y + table_h > img_h or table_x + table_w > img_w:
            logger.warning(
                f"Table coordinates {table_coords} exceed image bounds {(img_w, img_h)}. "
                f"Adjusting to fit within image."
            )
            table_w = min(table_w, img_w - table_x)
            table_h = min(table_h, img_h - table_y)
            if table_w <= 0 or table_h <= 0:
                logger.error(f"Invalid table coordinates after adjustment. Cannot detect cards.")
                return detected_cards
        
        # --- DEFINIERE BEREICHE FÜR KARTENERKENNUNG (MUSS ANGEPASST WERDEN!) ---
        # Diese Koordinaten sind stark abhängig vom Tisch-Layout.
        
        # Beispiel: Annahme für Hole Cards (unterer Bereich)
        hole_card_y_start = table_y + int(table_h * 0.7) 
        hole_card_height = int(table_h * 0.15)
        hole_card_width_approx = int(table_w * 0.08) 
        hc1_x = table_x + int(table_w * 0.2) # Linke Hole Card
        hc2_x = table_x + int(table_w * 0.4) # Rechte Hole Card
        hole_card_areas = [
            (hc1_x, hole_card_y_start, hole_card_width_approx, hole_card_height),
            (hc2_x, hole_card_y_start, hole_card_width_approx, hole_card_height)
        ]
        
        # Beispiel: Annahme für Community Cards (mittlerer/oberer Bereich)
        community_y_start = table_y + int(table_h * 0.4)
        community_card_height = int(table_h * 0.12)
        community_card_width_approx = int(table_w * 0.07)
        community_spacing = int(table_w * 0.01)
        community_x_start = table_x + (table_w - (5 * community_card_width_approx + 4 * community_spacing)) // 2 # Zentriert
        
        community_card_areas = []
        for i in range(5):
            x = community_x_start + i * (community_card_width_approx + community_spacing)
            area = (x, community_y_start, community_card_width_approx, community_card_height)
            community_card_areas.append(area)

        card_search_areas = hole_card_areas + community_card_areas
        
        # --- Suche nach Karten in den definierten Bereichen ---
        for i, area in enumerate(card_search_areas):
            x, y, w, h = area
            if w <= 0 or h <= 0:
                logger.debug(f"Card area {i} has invalid dimensions: {area}")
                continue
            
            # Add bounds checking before slicing
            if y + h > img_h or x + w > img_w:
                logger.debug(
                    f"Card area {i} {area} exceeds image bounds {(img_w, img_h)}. "
                    f"Clipping to fit."
                )
                w = min(w, img_w - x)
                h = min(h, img_h - y)
                if w <= 0 or h <= 0:
                    logger.debug(f"Card area {i} clipped to zero size. Skipping.")
                    continue
            
            card_roi = image[y:y+h, x:x+w]
            
            best_match_card_name = None
            best_match_score = -1.0
            best_match_loc_in_roi = None

            for card_name, template in self.card_templates.items():
                # Skaliere Template ggf., wenn Karten verschieden groß sind
                # template_resized = cv2.resize(template, (w, h)) 
                
                # Finde das Template in der ROI
                match_loc = self.find_template(card_roi, template, self.card_detection_threshold)
                
                if match_loc:
                    # Hole die Konfidenz (max_val aus find_template)
                    # Um die Konfidenz zu bekommen, müssten wir find_template anpassen oder neu berechnen
                    # Hier vereinfacht: Wir nehmen einfach die erste gefundene Übereinstimmung
                    
                    # Einfache Überlappungsprüfung (rudimentär)
                    is_overlapping = False
                    current_match_abs_loc = (x + match_loc[0], y + match_loc[1])
                    current_match_w, current_match_h = template.shape[1], template.shape[0]

                    # Prüfe gegen bereits gefundene Karten
                    for existing_card_name, existing_loc in detected_cards:
                        ex, ey, ew, eh = existing_loc
                        # Einfache Überlappungsprüfung: Prüfen, ob Mittelpunkte zu nah beieinander liegen
                        # Eine bessere Prüfung wäre die IOU (Intersection over Union)
                        center_x_curr = current_match_abs_loc[0] + current_match_w / 2
                        center_y_curr = current_match_abs_loc[1] + current_match_h / 2
                        center_x_exist = ex + ew / 2
                        center_y_exist = ey + eh / 2

                        dist_sq = (center_x_curr - center_x_exist)**2 + (center_y_curr - center_y_exist)**2
                        if dist_sq < (min(current_match_w, ew) * min(current_match_h, eh) * 0.25): # Wenn Mittelpunkte < 50% der kleineren Kartenbreite/Höhe entfernt sind
                           is_overlapping = True
                           break
                           
                    if not is_overlapping:
                        detected_cards.append((card_name, (current_match_abs_loc[0], current_match_abs_loc[1], template.shape[1], template.shape[0])))
                        logger.debug(f"Karte erkannt: {card_name} in Area {i} bei {current_match_abs_loc}")
                        # Wir brechen hier ab, um nicht dieselbe Karte mehrmals zu erkennen,
                        # aber eine robustere Lösung würde die beste Übereinstimmung über alle Templates suchen.
                        break 

        logger.debug(f"{len(detected_cards)} Karten im Bild gefunden.")
        return detected_cards

    def read_text(self, image: np.ndarray, area: Optional[Tuple[int, int, int, int]] = None, lang='eng', config='--psm 6') -> str:
        """ Liest Text aus einem Bildbereich mittels Tesseract OCR. """
        if image is None or image.size == 0: return ""

        roi = image
        if area:
            x, y, w, h = area
            if w > 0 and h > 0 and y+h <= image.shape[0] and x+w <= image.shape[1]:
                roi = image[y:y+h, x:x+w]
            else: 
                logger.debug(f"Ungültiger OCR-Bereich: {area}, Bildgröße: {image.shape}")
                return ""

        if roi.size == 0: return ""

        try:
            custom_config = f'--oem 3 {config}' 
            text = pytesseract.image_to_string(roi, lang=lang, config=custom_config)
            cleaned_text = self._clean_text(text)
            logger.debug(f"Gelesener Text (Area={area}): '{cleaned_text}'")
            return cleaned_text
        except Exception as e:
            logger.error(f"Fehler bei OCR: {e}")
            return ""

    def _clean_text(self, text: str) -> str:
        """ Bereinigt den von OCR gelesenen Text. """
        cleaned = ''.join(c for c in text if c.isalnum() or c in ['.', ',', ' '])
        cleaned = cleaned.replace('O', '0').replace('o', '0')
        cleaned = cleaned.replace('l', '1').replace('I', '1')
        cleaned = cleaned.replace('S', '5').replace('s', '5')
        cleaned = cleaned.replace('B', '8')
        # Ersetze Kommas durch Punkte für Dezimalzahlen
        cleaned = cleaned.replace(',', '.')
        # Entferne mehrere Leerzeichen hintereinander
        cleaned = ' '.join(cleaned.split())
        return cleaned.strip()

    def draw_elements(self, image: np.ndarray, table_coords: Optional[Tuple[int, int, int, int]], 
                      player_areas: List[Tuple[int, int, int, int]], 
                      pot_area: Optional[Tuple[int, int, int, int]], 
                      detected_cards: List[Tuple[str, Tuple[int, int, int, int]]]) -> np.ndarray:
        """ Zeichnet erkannte Elemente auf das Bild (für Debugging). """
        img_copy = image.copy()
        
        if table_coords:
            x, y, w, h = table_coords
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2) # Grün
            
        for area in player_areas:
            x, y, w, h = area
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 1) # Blau
            
        if pot_area:
            x, y, w, h = pot_area
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 2) # Rot
            
        for card_name, loc in detected_cards:
            x, y, w, h = loc
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 255, 0), 2) # Cyan
            cv2.putText(img_copy, card_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
        return img_copy

