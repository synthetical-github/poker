# bot_logic.py
import cv2
import time
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

# Importiere benötigte Klassen
try:
    from utils.image_processor import ImageProcessor
    from strategy_engine import StrategyEngine
    from utils.card_utils import Card, parse_card_string, get_hand_rank
    from config import POKER_SETTINGS
    from logger import logger
except ImportError:
    logger = logging.getLogger(__name__)
    logger.error("Konnte benötigte Module für bot_logic.py nicht importieren.")
    # Dummy-Definitionen für den Fall eines Importfehlers
    class ImageProcessor:
        def get_screenshot(self): return None
        def find_table(self, img): return None
        def detect_player_areas(self, coords, num): return []
        def get_pot_area(self, coords): return (0,0,0,0)
        def detect_cards(self, img, coords): return []
        def read_text(self, img, area): return ""
        def draw_elements(self, img, *args): return img
    class StrategyEngine:
        def decide_action(self, gs): return "fold", 0.0
    class Card: pass
    def parse_card_string(s): return None
    def get_hand_rank(h, c): return (0, [])
    POKER_SETTINGS = {'num_players': 9}


class PokerBotVision:
    """ Verarbeitet Bilddaten und extrahiert den Spielzustand. """
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.current_table_coords: Optional[Tuple[int, int, int, int]] = None
        self.num_players = POKER_SETTINGS.get('num_players', 9)
        self.table_found = False
        self.last_game_state = None # Zum Vergleich von Zustandsänderungen

    def update_game_state(self) -> Optional[Dict[str, Any]]:
        """ Erfasst das Spielgeschehen vom Bildschirm und gibt den aktuellen Zustand zurück. """
        screenshot = self.image_processor.get_screenshot()
        if screenshot is None:
            logger.error("Konnte keinen Screenshot erstellen. Überprüfen Sie die Screenshot-Konfiguration und Berechtigungen.")
            self.table_found = False # Tisch kann nicht gefunden werden ohne Screenshot
            return None

        # 1. Tisch finden (wiederholtes Finden nur, wenn nötig oder wenn Tisch verloren ging)
        if not self.table_found or self.current_table_coords is None:
            self.current_table_coords = self.image_processor.find_table(screenshot)
            if self.current_table_coords is None:
                logger.warning("Konnte den Spieltisch nicht finden. Stelle sicher, dass das Template korrekt ist und der Tisch sichtbar ist.")
                self.table_found = False
                # Optional: Screenshot anzeigen, um Tisch nicht gefunden zu debuggen
                # cv2.imshow("Screenshot - Tisch nicht gefunden", screenshot)
                # cv2.waitKey(100)
                return None 
            else:
                logger.info(f"Spiel-Tisch gefunden und erkannt: {self.current_table_coords}")
                self.table_found = True

        # 2. Bereiche definieren
        player_areas = self.image_processor.detect_player_areas(self.current_table_coords, self.num_players)
        pot_area = self.image_processor.get_pot_area(self.current_table_coords)
        
        # 3. Informationen extrahieren
        game_state: Dict[str, Any] = {
            'hole_cards': [], 
            'community_cards': [], 
            'pot_size': 0.0,
            'current_bet': 0.0, # Der aktuelle Höchsteinsatz im Spiel
            'to_call': 0.0,     # Betrag, den wir zahlen müssten, um im Spiel zu bleiben
            'num_players_remaining': self.num_players, # Muss dynamisch ermittelt werden!
            'player_info': {}, # Infos zu anderen Spielern (Name, Chips, Position)
            'street': 'preflop', 
            'position': 'unknown', # Eigene Position am Tisch
            'current_player_turn': False, # Ist dieser Bot gerade am Zug? (Muss ermittelt werden!)
            'all_in_possible': False # Ist ein All-In möglich?
        }

        try:
            # --- Pot-Größe lesen ---
            pot_text = self.image_processor.read_text(screenshot, pot_area)
            game_state['pot_size'] = self._parse_amount(pot_text) 

            # --- Karten lesen ---
            detected_cards_with_loc = self.image_processor.detect_cards(screenshot, self.current_table_coords)
            
            # Filtere Community Cards und Hole Cards basierend auf Position (muss angepasst werden!)
            community_card_y_threshold = self.current_table_coords[1] + int(self.current_table_coords[3] * 0.5) 
            
            hole_cards_found = []
            community_cards_found = []
            
            # Annahme: Hole Cards sind die ersten beiden erkannten Karten mit niedriger y-Koordinate
            # Dies ist eine HEURISTIK und muss verbessert werden!
            # Man braucht eine Methode, um die eigenen Hole Cards eindeutig zu identifizieren.
            
            # Sortiere erkannte Karten nach X-Position, um Reihenfolge zu bestimmen
            detected_cards_with_loc.sort(key=lambda item: item[1][0]) 

            for card_name, loc in detected_cards_with_loc:
                 card_obj = parse_card_string(card_name)
                 if card_obj:
                      # Einfache Positionsbestimmung (muss überarbeitet werden!)
                      if loc[1] < community_card_y_threshold: # Karte ist im oberen/mittleren Bereich -> Community Card
                           community_cards_found.append(card_obj)
                      else: # Karte ist im unteren Bereich -> Hole Card (Annahme!)
                           hole_cards_found.append(card_obj)
                           
            game_state['community_cards'] = community_cards_found
            game_state['hole_cards'] = hole_cards_found[:2] # Nur die ersten beiden als Hole Cards

            # --- Street bestimmen ---
            game_state['street'] = self._determine_street(game_state['community_cards'])

            # --- Spieler-Infos (Anzahl Spieler, Position, Einsätze) ---
            # Dies ist der komplexeste Teil und erfordert genaue Kenntnis des Tisch-Layouts.
            # Hier werden nur Platzhalter verwendet.
            game_state['num_players_remaining'] = self._determine_num_players(screenshot, player_areas) # Implementierung fehlt
            game_state['position'] = self._determine_my_position(self.current_table_coords, game_state['hole_cards']) # Implementierung fehlt
            
            # Lesen von aktuellen Einsätzen, zu-call-Beträgen und Spieler-Chips
            # Dies erfordert OCR auf Spieler-Info-Bereichen und Analyse der Chip-Stapel.
            # game_state['current_bet'], game_state['to_call'], game_state['player_info'] = self._parse_player_data(screenshot, player_areas)
            
            # Fürs Erste: Setze Platzhalter
            game_state['current_bet'] = 10.0 # Annahme
            game_state['to_call'] = 10.0 # Annahme
            game_state['player_info'] = {} # Leere Spielerinfos
            game_state['current_player_turn'] = True # Annahme: Immer am Zug für Debugging

            # --- Debugging: Zeichne Elemente auf den Screenshot ---
            debug_image = self.image_processor.draw_elements(
                screenshot, 
                self.current_table_coords, 
                player_areas, 
                pot_area, 
                [(c.rank + c.suit, (loc[0], loc[1], loc[2], loc[3])) for c, loc in zip(game_state['hole_cards'], detected_cards_with_loc[:len(game_state['hole_cards'])])] + \
                [(c.rank + c.suit, (loc[0], loc[1], loc[2], loc[3])) for c, loc in zip(game_state['community_cards'], detected_cards_with_loc[len(game_state['hole_cards']):])]
            )
            # Zeige das Debug-Bild an (optional, kann Performance beeinträchtigen)
            # cv2.imshow("Bot Vision Debug", debug_image)
            # cv2.waitKey(1) 

        except Exception as e:
            logger.error(f"Fehler bei der Analyse des Spielzustands: {e}", exc_info=True)
            # Zeige Screenshot für Debugging, falls Fehler auftritt
            # cv2.imshow("Error Screenshot", screenshot)
            # cv2.waitKey(1000)
            return None 

        # Verhindere unnötige Verarbeitung, wenn sich der Zustand nicht geändert hat
        # Dies ist wichtig, wenn der Bot schnell in einer Schleife läuft.
        # if game_state == self.last_game_state:
        #     return None # Kein neuer Zustand
        # self.last_game_state = game_state
        
        return game_state

    def _parse_amount(self, text: str) -> float:
        """ Parst Text zu einem Geldbetrag (float). """
        try:
            cleaned = ''.join(c for c in text if c.isdigit() or c in ['.', ','])
            cleaned = cleaned.replace(',', '.')
            cleaned = cleaned.strip('.')
            if not cleaned: return 0.0
            # Entferne tausender Trennzeichen, falls vorhanden (z.B. 1.234,56 -> 1234.56)
            if '.' in cleaned and ',' in cleaned: # Wahrscheinlich deutsches Format
                 parts = cleaned.split(',')
                 integer_part = parts[0].replace('.', '')
                 decimal_part = parts[1]
                 cleaned = f"{integer_part}.{decimal_part}"
            elif '.' in cleaned and cleaned.count('.') > 1: # Mehrere Punkte, wahrscheinlich Tausender-Trennzeichen
                 cleaned = cleaned.replace('.', '')

            return float(cleaned)
        except ValueError:
            logger.warning(f"Konnte Betrag nicht parsen: '{text}'")
            return 0.0

    def _determine_street(self, community_cards: List[Card]) -> str:
        """ Bestimmt die aktuelle Street des Spiels. """
        num_cards = len(community_cards)
        if num_cards == 0: return 'preflop'
        if num_cards == 3: return 'flop'
        if num_cards == 4: return 'turn'
        if num_cards == 5: return 'river'
        return 'unknown'

    # --- Platzhalter für komplexere Zustandsbestimmung ---
    
    def _determine_num_players(self, screenshot: np.ndarray, player_areas: List[Tuple[int, int, int, int]]) -> int:
        """ Ermittelt die Anzahl der aktiven Spieler in der Hand. (Muss implementiert werden!) """
        # Dies könnte durch OCR der Spieler-Infos oder durch Analyse der aktiven Spielpositionen geschehen.
        # Fürs Erste wird die konfigurierte Anzahl zurückgegeben.
        return self.num_players 

    def _determine_my_position(self, table_coords: Tuple[int, int, int, int], hole_cards: List[Card]) -> str:
        """ Ermittelt die Position des Bots am Tisch (z.B. 'early', 'middle', 'late', 'button'). (Muss implementiert werden!) """
        # Solange keine Dealer- und Seat-Erkennung implementiert ist, bleibt das
        # bewusst bei einem sicheren Platzhalter statt auf undefinierte Variablen
        # zuzugreifen.
        return "unknown"
        
    def _parse_player_data(self, screenshot: np.ndarray, player_areas: List[Tuple[int, int, int, int]]) -> Tuple[float, float, Dict[str, Dict]]:
        """ Liest Daten für alle Spieler (Chips, Einsätze, Namen). (Muss implementiert werden!) """
        current_bet = 0.0
        to_call = 0.0
        player_info = {}
        
        # Hier müsste für jeden Spielerbereich:
        # 1. Name per OCR gelesen werden.
        # 2. Chip-Betrag per OCR oder Farb-/Mustererkennung gelesen werden.
        # 3. Aktueller Einsatz für diese Runde gelesen werden.
        # 4. Ermittelt werden, ob dieser Spieler gerade am Zug ist.
        # 5. `current_bet` und `to_call` müssten aus den Spielerdaten aggregiert werden.
        
        logger.debug("Spielerdatenanalyse nicht implementiert.")
        return current_bet, to_call, player_info

