from __future__ import annotations

import os
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np
import pyautogui

from config import LIVE_CONFIG

try:
    import mss
except ImportError:
    mss = None

try:
    from PIL import ImageGrab
except ImportError:
    ImageGrab = None


ImageType = Optional[np.ndarray]
RegionType = Optional[Tuple[int, int, int, int]]


def _stats(image: ImageType) -> Dict[str, object]:
    if image is None:
        return {
            "ok": False,
            "shape": None,
            "min": None,
            "max": None,
            "mean": None,
            "nonzero": None,
            "usable": False,
        }

    nonzero = int(np.count_nonzero(image))
    mean = float(image.mean())
    return {
        "ok": True,
        "shape": tuple(int(x) for x in image.shape),
        "min": int(image.min()),
        "max": int(image.max()),
        "mean": round(mean, 3),
        "nonzero": nonzero,
        "usable": nonzero > 0 and mean > 1.0,
    }


def _save_image(output_dir: str, name: str, image: ImageType) -> Optional[str]:
    if image is None:
        return None

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.png")
    if cv2.imwrite(path, image):
        return path
    return None


def _grab_mss(region: RegionType) -> ImageType:
    if mss is None:
        raise RuntimeError("mss nicht installiert")

    with mss.mss() as sct:
        monitors = sct.monitors
        monitor_index = 1 if len(monitors) > 1 else 0
        monitor = monitors[monitor_index]

        if region:
            left, top, width, height = region
            bbox = {
                "left": left,
                "top": top,
                "width": width,
                "height": height,
            }
        else:
            bbox = {
                "left": monitor["left"],
                "top": monitor["top"],
                "width": monitor["width"],
                "height": monitor["height"],
            }

        img = np.array(sct.grab(bbox))
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img


def _grab_pillow(region: RegionType) -> ImageType:
    if ImageGrab is None:
        raise RuntimeError("Pillow ImageGrab nicht verfuegbar")

    if region:
        left, top, width, height = region
        bbox = (left, top, left + width, top + height)
    else:
        size = pyautogui.size()
        bbox = (0, 0, size.width, size.height)

    img = np.array(ImageGrab.grab(bbox=bbox))
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def _run_method(name: str, fn: Callable[[RegionType], ImageType], region: RegionType, output_dir: str) -> None:
    print(f"\n[{name}]")
    try:
        image = fn(region)
        stats = _stats(image)
        saved = _save_image(output_dir, name, image)
        print(f"stats={stats}")
        print(f"saved={saved}")
    except Exception as exc:
        print(f"error={type(exc).__name__}: {exc}")


def main() -> None:
    output_dir = os.path.join(os.getcwd(), "capture_diagnostics")
    configured_region = LIVE_CONFIG.get("screen_region")
    full_screen_region = None

    screen_size = pyautogui.size()
    print("Capture-Diagnose")
    print(f"screen_size=({screen_size.width}, {screen_size.height})")
    print(f"configured_region={configured_region}")
    print(f"output_dir={output_dir}")

    print("\nTests mit konfigurierter Region")
    _run_method("mss_configured_region", _grab_mss, configured_region, output_dir)
    _run_method("pillow_configured_region", _grab_pillow, configured_region, output_dir)

    print("\nTests mit Vollbild")
    _run_method("mss_full_screen", _grab_mss, full_screen_region, output_dir)
    _run_method("pillow_full_screen", _grab_pillow, full_screen_region, output_dir)

    print("\nHinweise")
    print("- Wenn nur mss schwarze Bilder liefert, ist oft Hardwarebeschleunigung oder eine geschuetzte Surface die Ursache.")
    print("- Wenn nur Pillow fehlschlaegt, ist das in manchen Windows-Sitzungen normal.")
    print("- Wenn beide fehlschlagen oder schwarz sind, liegt das Problem meist an Berechtigungen, minimiertem Fenster oder der Desktop-Session.")


if __name__ == "__main__":
    main()
