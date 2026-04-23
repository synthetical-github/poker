# voice_announcer.py
import pyttsx3
import time
from typing import Any, Dict

from utils.logger import logger
from utils.config import LIVE_CONFIG

class VoiceAnnouncer:
    def __init__(self):
        self.enabled = LIVE_CONFIG.get('voice_enabled', False)
        self.engine = None
        self.last_announced_action = None
        self.last_announced_state = None
        
        if self.enabled:
            try:
                self.engine = pyttsx3.init()
                # Stimmen-Konfiguration (optional)
                # voices = self.engine.getProperty('voices')
                # self.engine.setProperty('voice', voices[1].id) # Wähle eine Stimme
                logger.info("Sprachausgabe initialisiert.")
            except Exception as e:
                logger.error(f"Fehler bei der Initialisierung der Sprachausgabe: {e}")
                self.enabled = False

    def announce(self, analysis_result: Dict[str, Any]):
        """ Gibt die Analyseergebnisse per Sprache aus. """
        if not self.enabled or not self.engine:
            return

        try:
            strategy = analysis_result.get('strategy', analysis_result)
            action = strategy.get('recommended_action')
            amount = strategy.get('amount', 0.0)
            hole_cards = analysis_result.get('hole_cards', [])
            community_cards = analysis_result.get('community_cards', [])
            hand_details = strategy.get('hand_details', {}) or {}
            hand_label = hand_details.get('display_category') or hand_details.get('category', '')
            rank_value = int(hand_details.get('rank_value', 0) or 0)
            
            # Nur ankündigen, wenn sich die Aktion oder der Zustand geändert hat
            current_state_key = (tuple(hole_cards), tuple(community_cards), action, amount, hand_label)
            
            if current_state_key != self.last_announced_state:
                message = ""
                if rank_value >= 6 and hand_label:
                    message = f"Monster. {hand_label}. "
                elif rank_value >= 5 and hand_label:
                    message = f"Very strong. {hand_label}. "
                elif rank_value >= 4 and hand_label:
                    message = f"Strong hand. {hand_label}. "
                elif rank_value >= 1 and hand_label:
                    message = f"{hand_label}. "

                if action in ['fold', 'check', 'call']:
                    message += f"{action.upper()}"
                elif action == 'bet':
                    message += f"Bet {amount:.0f}"
                elif action == 'raise':
                     # Betrag ist der Gesamtbetrag, nicht nur der Raise-Anteil
                     message += f"Raise to {amount:.0f}" 
                     
                # Füge Karten hinzu, wenn sie erkannt wurden
                if hole_cards:
                     message += f" with {hole_cards[0]} and {hole_cards[1]}"
                if community_cards:
                     message += f" - Board: {' '.join(community_cards)}"
                     
                logger.info(f"Sprachausgabe: {message}")
                self.engine.say(message)
                self.engine.runAndWait()
                
                self.last_announced_state = current_state_key
                
        except Exception as e:
            logger.error(f"Fehler bei der Sprachausgabe: {e}")

    def speak(self, text: str):
         """ Spricht einen gegebenen Text aus. """
         if self.enabled and self.engine:
              self.engine.say(text)
              self.engine.runAndWait()
