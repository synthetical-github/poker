# bot.py
import time
import cv2
import pyautogui

from live_analyzer import LivePokerAnalyzer
from hotkeys import setup_hotkeys
from utils.logger import logger
from utils.config import LIVE_CONFIG
from utils.screen_utils import select_region_interactive


def main():
    """Hauptfunktion zum Starten des Bots."""

    capture_method = str(LIVE_CONFIG.get('capture_method', 'screen')).strip().lower()
    screen_region = LIVE_CONFIG.get('screen_region')

    # Nur im klassischen Screen-Modus eine Region erzwingen
    if capture_method != 'window':
        current_screen_size = None
        try:
            size = pyautogui.size()
            current_screen_size = (0, 0, size.width, size.height)
        except Exception as e:
            logger.warning(f"Konnte Bildschirmgröße nicht automatisch bestimmen: {e}")

        needs_calibration = (
            not screen_region or
            (current_screen_size is not None and screen_region == current_screen_size)
        )

        if needs_calibration:
            print("\nWARNUNG: 'screen_region' ist in config.py nicht definiert oder None.")
            print("Oder sie zeigt noch auf den gesamten Desktop statt auf das Pokerfenster.")
            print("Starte interaktive Regionsauswahl...")

            selected = select_region_interactive()
            if selected:
                LIVE_CONFIG['screen_region'] = selected
                print(f"Region ausgewählt: {selected}. Bitte speichern Sie diese manuell in config.py!")
            else:
                print("WARNUNG: Keine Region ausgewählt. Der Bot versucht, den gesamten Hauptmonitor zu verwenden.")
                try:
                    size = pyautogui.size()
                    LIVE_CONFIG['screen_region'] = (0, 0, size.width, size.height)
                except Exception as e:
                    logger.error(f"Konnte Bildschirmgröße nicht ermitteln: {e}. Screenshot wird fehlschlagen.")
                    LIVE_CONFIG['screen_region'] = (0, 0, 100, 100)
    else:
        title_hint = LIVE_CONFIG.get('window_title_contains', "NL Hold'em")
        print(f"Window-Capture aktiv. Suche App-Fenster mit Titel passend zu: {title_hint}")

    headless = bool(LIVE_CONFIG.get('headless', False))
    bot = LivePokerAnalyzer(headless=headless)

    setup_hotkeys(bot)
    bot.run_analysis_thread()

    show_overlay = not headless and bool(LIVE_CONFIG.get('show_overlay', True))

    try:
        if show_overlay:
            # Tkinter muss im Hauptthread laufen (Windows-Anforderung)
            bot.overlay.run_blocking(bot.stop_event)
        else:
            while bot.running and not bot.stop_event.is_set():
                time.sleep(0.2)

    except KeyboardInterrupt:
        logger.info("Strg+C erkannt. Beende Bot...")
        bot.stop_bot()

    except Exception as e:
        logger.critical(f"Unerwarteter Fehler im Hauptprogramm: {e}", exc_info=True)
        bot.stop_bot()

    finally:
        try:
            if not bot.headless:
                try:
                    if cv2.getWindowProperty("Poker Bot Live Analysis", cv2.WND_PROP_VISIBLE) >= 0:
                        cv2.destroyAllWindows()
                except Exception:
                    cv2.destroyAllWindows()
        except Exception:
            pass

        logger.info("Programm beendet.")


if __name__ == "__main__":
    main()
