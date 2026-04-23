# hotkeys.py
import keyboard
from utils.logger import logger

# Annahme: analyzer ist eine Instanz von LivePokerAnalyzer
def setup_hotkeys(analyzer):
    """ Richtet Hotkeys ein, um Aktionen im LivePokerAnalyzer auszulösen. """
    
    hotkeys = {
        'f2': {'action': analyzer.manual_fold, 'description': 'Manual Fold'},
        'f3': {'action': analyzer.manual_call, 'description': 'Manual Call'},
        'f4': {'action': analyzer.manual_raise, 'description': 'Manual Raise (Amount needs definition)'},
        'f5': {'action': analyzer.calibrate_rois, 'description': 'Calibrate ROIs'}, # Benötigt Implementierung
        'f6': {'action': analyzer.toggle_pause, 'description': 'Toggle Pause'},
        'f7': {'action': analyzer.toggle_voice, 'description': 'Toggle Voice Announcer'},
        'ctrl+q': {'action': analyzer.stop_bot, 'description': 'Stop Bot'}, # Hotkey zum Beenden
    }
    
    logger.info("Richte Hotkeys ein...")
    for key, data in hotkeys.items():
        try:
            keyboard.add_hotkey(key, data['action'])
            logger.info(f"Hotkey '{key}' registriert für: {data['description']}")
        except Exception as e:
            logger.error(f"Konnte Hotkey '{key}' nicht registrieren: {e}")
            
    print("\n--- Hotkeys ---")
    for key, data in hotkeys.items():
        print(f"- {key}: {data['description']}")
    print("---------------\n")

def remove_hotkeys():
    """ Entfernt alle registrierten Hotkeys. """
    keyboard.unhook_all()
    logger.info("Alle Hotkeys entfernt.")
