import os
from typing import Optional, Tuple

import cv2

from config import LIVE_CONFIG, TABLE_TEMPLATE_PATH
from utils.logger import logger
from utils.screen_utils import get_screenshot_for_processing, select_region_interactive


def _save_region_as_template(image, region: Tuple[int, int, int, int], output_path: str) -> Optional[str]:
    x, y, w, h = region
    crop = image[y:y + h, x:x + w]
    if crop is None or crop.size == 0:
        return None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not cv2.imwrite(output_path, crop):
        return None
    return output_path


def main() -> None:
    print("Kalibrierung fuer screen_region und table_template")
    print("1. Stelle sicher, dass der Pokertisch sichtbar und nicht minimiert ist.")
    print("2. Wenn die Regionsauswahl erscheint, ziehe exakt den Tischrahmen auf.")
    print("3. Danach wird ein Template aus genau diesem Bereich gespeichert.")

    selected_region = select_region_interactive()
    if not selected_region:
        print("Keine Region ausgewaehlt. Abbruch.")
        return

    LIVE_CONFIG["screen_region"] = selected_region
    print(f"Ausgewaehlte screen_region: {selected_region}")

    screenshot = get_screenshot_for_processing()
    if screenshot is None:
        print("Screenshot nach der Kalibrierung unbrauchbar.")
        print("Wahrscheinliche Ursache: schwarzer Capture durch Berechtigungen, minimiertes Fenster oder falsche Anzeigequelle.")
        print(f"Trage diese Region spaeter manuell ein: LIVE_CONFIG['screen_region'] = {selected_region}")
        return

    template_path = _save_region_as_template(
        screenshot,
        (0, 0, screenshot.shape[1], screenshot.shape[0]),
        TABLE_TEMPLATE_PATH,
    )
    if not template_path:
        print("Template konnte nicht gespeichert werden.")
        return

    print(f"Template gespeichert: {template_path}")
    print(f"Trage diese Region in config.py ein: LIVE_CONFIG['screen_region'] = {selected_region}")
    logger.info(f"Kalibrierung abgeschlossen. Region={selected_region}, Template={template_path}")


if __name__ == "__main__":
    main()
