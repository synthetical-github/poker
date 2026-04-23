# utils/screen_selector.py
import cv2
import numpy as np
import pyautogui
from typing import Tuple, Optional, List

# Versuche, mss und Pillow zu importieren
try:
    import mss
    USE_MSS = True
except ImportError:
    USE_MSS = False
    try:
        from PIL import ImageGrab
    except ImportError:
        ImageGrab = None # Weder mss noch Pillow verfügbar

class ScreenSelector:
    def __init__(self):
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.roi = None
        self.window_name = "Select Region - Press SPACE to confirm, ESC to cancel"

    def select_region(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Öffnet ein Fenster, um einen Bereich auf dem Bildschirm interaktiv auszuwählen.
        Gibt die Koordinaten (x, y, width, height) des ausgewählten Bereichs zurück
        oder None, wenn die Auswahl abgebrochen wurde.
        """
        
        # Screenshot machen
        img = self._get_screenshot()
        if img is None:
            print("Fehler: Konnte keinen Screenshot für die Regionsauswahl erstellen.")
            return None

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._draw_rectangle)

        print("Ziehen Sie mit der Maus einen Bereich auf dem Bildschirm auf.")
        print("Drücken Sie SPACE, um den Bereich zu bestätigen.")
        print("Drücken Sie ESC, um abzubrechen.")

        while True:
            img_copy = img.copy()
            if self.start_point and self.end_point:
                cv2.rectangle(img_copy, self.start_point, self.end_point, (0, 255, 0), 2)
            
            cv2.imshow(self.window_name, img_copy)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '): # Space-Taste zum Bestätigen
                if self.start_point and self.end_point:
                    x1 = min(self.start_point[0], self.end_point[0])
                    y1 = min(self.start_point[1], self.end_point[1])
                    x2 = max(self.start_point[0], self.end_point[0])
                    y2 = max(self.start_point[1], self.end_point[1])
                    
                    width = x2 - x1
                    height = y2 - y1
                    
                    if width > 0 and height > 0:
                        self.roi = (x1, y1, width, height)
                        print(f"Region ausgewählt: {self.roi}")
                        break
                    else:
                        print("Ungültige Region (Breite oder Höhe ist 0). Bitte erneut versuchen.")
                        self.start_point = None
                        self.end_point = None

            elif key == 27: # ESC-Taste zum Abbrechen
                print("Regionsauswahl abgebrochen.")
                self.roi = None
                break
        
        cv2.destroyWindow(self.window_name)
        return self.roi

    def _get_screenshot(self) -> Optional[np.ndarray]:
        """ Holt einen Screenshot (ähnlich wie in ImageProcessor). """
        bbox_dict = None # Für mss
        pil_bbox = None # Für Pillow

        if USE_MSS:
            with mss.mss() as sct:
                monitor = sct.monitors[1] # Primärer Monitor
                bbox_dict = monitor # Ganzer Monitor
                try:
                    img_mss = sct.grab(bbox_dict)
                    img = np.array(img_mss)
                    if img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    return img
                except Exception as e:
                    print(f"mss Screenshot fehlgeschlagen, nutze Pillow-Fallback: {e}")
        elif ImageGrab:
             screen_width = pyautogui.size().width
             screen_height = pyautogui.size().height
             pil_bbox = (0, 0, screen_width, screen_height)
        else:
            print("Fehler: Keine Screenshot-Bibliothek (mss oder Pillow) gefunden.")
            return None

        try:
            if ImageGrab and pil_bbox:
                img_pil = ImageGrab.grab(bbox=pil_bbox)
                img = np.array(img_pil)
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                 return None
        except Exception as e:
            print(f"Fehler beim Erstellen des Screenshots: {e}")
            return None

    def _draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.end_point = (x, y)
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.end_point = (x, y)
            self.drawing = False

# --- Beispiel zur Verwendung ---
if __name__ == "__main__":
    selector = ScreenSelector()
    region = selector.select_region()
    
    if region:
        print(f"\nSie haben folgende Region ausgewählt (x, y, width, height): {region}")
        print("Diese Koordinaten können Sie in die 'config.py' unter 'LIVE_CONFIG[\"screen_region\"]' eintragen.")
        
        # Optional: Screenshot des ausgewählten Bereichs anzeigen
        try:
            import mss
            with mss.mss() as sct:
                 monitor = {"top": region[1], "left": region[0], "width": region[2], "height": region[3]}
                 img = np.array(sct.grab(monitor))
                 if img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                 
                 cv2.imshow("Ausgewaehlter Bereich", img)
                 print("Fenster mit ausgewaehltem Bereich wird angezeigt. Schliessen Sie es manuell.")
                 cv2.waitKey(0)
                 cv2.destroyAllWindows()
        except Exception as e:
             print(f"Konnte Screenshot des ausgewaehlten Bereichs nicht anzeigen: {e}")
             
    else:
        print("Keine Region ausgewählt.")
