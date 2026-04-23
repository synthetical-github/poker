# utils/screen_utils.py
import cv2
import ctypes
import numpy as np
import time
from typing import Tuple, Optional, List
import pyautogui

try:
    import win32con
    import win32gui
    import win32ui
    USE_WIN32 = True
except ImportError:
    win32con = None
    win32gui = None
    win32ui = None
    USE_WIN32 = False

try:
    import mss
    USE_MSS = True
except ImportError:
    USE_MSS = False

try:
    from PIL import ImageGrab
except ImportError:
    ImageGrab = None

from utils.config import LIVE_CONFIG
from utils.logger import logger


drawing_data = {
    'drawing': False,
    'start_point': None,
    'end_point': None,
    'roi': None
}

window_cache = {
    'hwnd': None,
    'title': '',
    'seen_at': 0.0,
    'last_specific_hwnd': None,
    'last_specific_title': '',
    'last_specific_seen_at': 0.0,
}

PW_RENDERFULLCONTENT = 0x00000002


def _normalize_title(value: str) -> str:
    if not value:
        return ""
    value = value.lower()
    allowed = []
    for ch in value:
        if ch.isalnum():
            allowed.append(ch)
    return "".join(allowed)


def _is_generic_client_title(title: str) -> bool:
    normalized = _normalize_title(title or "")
    fallback = _normalize_title(LIVE_CONFIG.get('window_title_fallback_contains', ''))
    return bool(normalized and fallback and normalized.startswith(fallback))


def _get_title_filters(include_fallback: bool = True) -> List[str]:
    filters: List[str] = []
    primary = LIVE_CONFIG.get('window_title_contains')
    fallback = LIVE_CONFIG.get('window_title_fallback_contains')
    aliases = LIVE_CONFIG.get('window_title_aliases') or []

    items = [primary, *aliases]
    if include_fallback:
        items.append(fallback)

    for item in items:
        if isinstance(item, str) and item.strip():
            filters.append(item.strip())

    result: List[str] = []
    seen = set()
    for item in filters:
        key = _normalize_title(item)
        if key and key not in seen:
            seen.add(key)
            result.append(item)

    return result


def _is_usable_screenshot(image: Optional[np.ndarray]) -> bool:
    if image is None or image.size == 0:
        return False
    if np.count_nonzero(image) == 0:
        return False
    return float(image.mean()) > 1.0


def _title_matches(window_title: str, title_filter: str) -> bool:
    raw_title = (window_title or "").strip()
    raw_filter = (title_filter or "").strip()

    if not raw_title or not raw_filter:
        return False

    if raw_filter.lower() in raw_title.lower():
        return True

    normalized_title = _normalize_title(raw_title)
    normalized_filter = _normalize_title(raw_filter)

    if not normalized_title or not normalized_filter:
        return False

    if normalized_filter in normalized_title:
        return True

    # toleriert z. B. NDL HOLDEM statt NL HOLDEM
    typo_variant = normalized_filter.replace("d", "", 1)
    if typo_variant and typo_variant in normalized_title:
        return True

    return False


def _remember_window_match(hwnd: int, title: str) -> None:
    window_cache['hwnd'] = hwnd
    window_cache['title'] = title or ''
    window_cache['seen_at'] = time.time()
    if title and not _is_generic_client_title(title):
        window_cache['last_specific_hwnd'] = hwnd
        window_cache['last_specific_title'] = title
        window_cache['last_specific_seen_at'] = window_cache['seen_at']


def _cached_window_is_usable(hwnd: Optional[int]) -> bool:
    if not USE_WIN32 or not hwnd:
        return False
    try:
        if not win32gui.IsWindow(hwnd):
            return False
        if LIVE_CONFIG.get('require_window_visible', True) and not win32gui.IsWindowVisible(hwnd):
            return False
        if LIVE_CONFIG.get('require_window_not_minimized', True) and win32gui.IsIconic(hwnd):
            return False
        return True
    except Exception:
        return False


def _find_window_by_title(title_filters: List[str], fallback_filter: str = "") -> Optional[int]:
    if not USE_WIN32 or not title_filters:
        return None

    cached_hwnd = window_cache.get('hwnd')
    if _cached_window_is_usable(cached_hwnd):
        cached_title = _get_window_title(cached_hwnd)
        recent_cache = (time.time() - float(window_cache.get('seen_at', 0.0) or 0.0)) <= 20.0
        if cached_title and (
            any(_title_matches(cached_title, title_filter) for title_filter in title_filters)
            or (
                recent_cache
                and _is_generic_client_title(cached_title)
                and cached_hwnd == window_cache.get('last_specific_hwnd')
                and bool(window_cache.get('last_specific_title'))
            )
        ):
            detected_title = (
                window_cache.get('last_specific_title')
                if _is_generic_client_title(cached_title) and window_cache.get('last_specific_title')
                else cached_title
            )
            LIVE_CONFIG['detected_window_title'] = detected_title
            _remember_window_match(cached_hwnd, cached_title)
            return cached_hwnd

    matches: List[Tuple[int, int, str]] = []

    def enum_handler(hwnd, _ctx):
        try:
            if LIVE_CONFIG.get('require_window_visible', True) and not win32gui.IsWindowVisible(hwnd):
                return

            if LIVE_CONFIG.get('require_window_not_minimized', True) and win32gui.IsIconic(hwnd):
                return

            title = win32gui.GetWindowText(hwnd).strip()
            if not title:
                return

            for title_filter in title_filters:
                if _title_matches(title, title_filter):
                    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                    area = max(0, right - left) * max(0, bottom - top)
                    if area > 0:
                        matches.append((hwnd, area, title))
                    return
        except Exception:
            return

    try:
        win32gui.EnumWindows(enum_handler, None)
    except Exception as exc:
        logger.debug(f"EnumWindows fehlgeschlagen: {exc}")

    if not matches:
        return None

    matches.sort(key=lambda item: item[1], reverse=True)
    hwnd, _, title = matches[0]
    LIVE_CONFIG['detected_window_title'] = title
    _remember_window_match(hwnd, title)
    return hwnd


def _get_window_title(hwnd: int) -> str:
    if not USE_WIN32 or not hwnd:
        return ""
    try:
        return win32gui.GetWindowText(hwnd).strip()
    except Exception:
        return ""


def _get_window_rect(hwnd: int) -> Optional[Tuple[int, int, int, int]]:
    if not USE_WIN32 or not hwnd:
        return None
    try:
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = max(0, right - left)
        height = max(0, bottom - top)
        if width <= 0 or height <= 0:
            return None
        return (left, top, width, height)
    except Exception as exc:
        logger.debug(f"WindowRect konnte nicht gelesen werden: {exc}")
        return None


def _get_window_client_box(hwnd: int) -> Optional[Tuple[int, int, int, int]]:
    if not USE_WIN32 or not hwnd:
        return None
    try:
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        top_left = win32gui.ClientToScreen(hwnd, (left, top))
        bottom_right = win32gui.ClientToScreen(hwnd, (right, bottom))
        x1, y1 = top_left
        x2, y2 = bottom_right
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        if width <= 0 or height <= 0:
            return None
        return (x1, y1, width, height)
    except Exception as exc:
        logger.debug(f"Client-Box konnte nicht gelesen werden: {exc}")
        return None


def _resolve_window_crop(
    window_rect: Tuple[int, int, int, int],
    client_box: Optional[Tuple[int, int, int, int]],
    region: Optional[Tuple[int, int, int, int]],
) -> Optional[Tuple[int, int, int, int]]:
    wx, wy, ww, wh = window_rect

    if not region:
        if client_box and LIVE_CONFIG.get('prefer_client_area_capture', True):
            cx, cy, cw, ch = client_box
            return (max(0, cx - wx), max(0, cy - wy), cw, ch)
        return None

    rx, ry, rw, rh = region
    if rw <= 0 or rh <= 0:
        return None

    # Region bereits fensterrelativ
    if 0 <= rx < ww and 0 <= ry < wh and rx + rw <= ww and ry + rh <= wh:
        return (rx, ry, rw, rh)

    # Region in Desktop-Koordinaten -> in Fensterbereich umrechnen
    overlap_left = max(wx, rx)
    overlap_top = max(wy, ry)
    overlap_right = min(wx + ww, rx + rw)
    overlap_bottom = min(wy + wh, ry + rh)

    if overlap_right > overlap_left and overlap_bottom > overlap_top:
        return (
            overlap_left - wx,
            overlap_top - wy,
            overlap_right - overlap_left,
            overlap_bottom - overlap_top,
        )

    if client_box and LIVE_CONFIG.get('prefer_client_area_capture', True):
        cx, cy, cw, ch = client_box
        return (max(0, cx - wx), max(0, cy - wy), cw, ch)

    return None


def _capture_window(hwnd: int, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
    if not USE_WIN32:
        return None

    window_rect = _get_window_rect(hwnd)
    if not window_rect:
        return None

    left, top, width, height = window_rect

    try:
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(bitmap)

        try:
            result = ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), PW_RENDERFULLCONTENT)
            if result != 1:
                save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)

            bmpinfo = bitmap.GetInfo()
            bmpstr = bitmap.GetBitmapBits(True)
            image = np.frombuffer(bmpstr, dtype=np.uint8).reshape(
                (bmpinfo["bmHeight"], bmpinfo["bmWidth"], 4)
            )
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            client_box = _get_window_client_box(hwnd)
            crop = _resolve_window_crop(window_rect, client_box, region)

            if crop:
                x, y, w, h = crop
                img_h, img_w = image.shape[:2]
                x = max(0, min(x, img_w))
                y = max(0, min(y, img_h))
                w = max(0, min(w, img_w - x))
                h = max(0, min(h, img_h - y))
                if w > 0 and h > 0:
                    image = image[y:y + h, x:x + w]

            return image if _is_usable_screenshot(image) else None

        finally:
            win32gui.DeleteObject(bitmap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwnd_dc)

    except Exception as exc:
        logger.debug(f"Fenster-Capture fehlgeschlagen: {exc}")
        return None


def _get_full_desktop_region() -> Tuple[int, int, int, int]:
    try:
        size = pyautogui.size()
        return (0, 0, size.width, size.height)
    except Exception:
        return (0, 0, 1920, 1080)


def _capture_desktop(region: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
    if USE_MSS:
        try:
            with mss.mss() as sct:
                if region:
                    bbox = {
                        "top": region[1],
                        "left": region[0],
                        "width": region[2],
                        "height": region[3],
                    }
                else:
                    bbox = sct.monitors[0]

                img_mss = sct.grab(bbox)
                img = np.array(img_mss)
                if img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                if _is_usable_screenshot(img):
                    return img
                logger.debug("mss lieferte leeres oder schwarzes Bild.")
        except Exception as exc:
            logger.warning(f"mss Screenshot fehlgeschlagen, nutze Pillow-Fallback: {exc}")

    if ImageGrab:
        try:
            if region:
                bbox = (region[0], region[1], region[0] + region[2], region[1] + region[3])
            else:
                x, y, w, h = _get_full_desktop_region()
                bbox = (x, y, x + w, y + h)

            img_pil = ImageGrab.grab(bbox=bbox)
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            return img if _is_usable_screenshot(img) else None
        except Exception as exc:
            logger.error(f"Fehler beim Erstellen des Screenshots per Pillow: {exc}")
            return None

    logger.error("Keine funktionierende Screenshot-Bibliothek (mss oder Pillow) gefunden.")
    return None


def _get_screenshot(
    region: Optional[Tuple[int, int, int, int]] = None,
    force_desktop: bool = False
) -> Optional[np.ndarray]:
    capture_method = str(LIVE_CONFIG.get('capture_method', 'window')).strip().lower()
    allow_screen_fallback = bool(LIVE_CONFIG.get('allow_screen_fallback', False))

    if not force_desktop and capture_method == 'window':
        title_filters = _get_title_filters(include_fallback=False)
        fallback_filter = str(LIVE_CONFIG.get('window_title_fallback_contains', '') or '').strip()
        hwnd = _find_window_by_title(title_filters, fallback_filter)

        if hwnd:
            actual_title = _get_window_title(hwnd)
            if actual_title:
                LIVE_CONFIG['detected_window_title'] = actual_title

            logger.debug(f"Verwende Fenster-Capture fuer '{actual_title or title_filters[0]}'.")
            image = _capture_window(hwnd, region)

            if _is_usable_screenshot(image):
                return image

            logger.warning("Fenster-Capture fehlgeschlagen.")

        else:
            logger.warning("Kein passendes App-Fenster gefunden.")

        if not allow_screen_fallback:
            return None

        logger.warning("Nutze Desktop-Fallback nach fehlgeschlagenem Window-Capture.")

    return _capture_desktop(region)


def select_region_interactive() -> Optional[Tuple[int, int, int, int]]:
    window_name = "Select Region - Press SPACE to confirm, ESC to cancel"

    # Fuer die Regionsauswahl IMMER den kompletten Desktop nehmen
    img = _get_screenshot(_get_full_desktop_region(), force_desktop=True)
    if img is None:
        logger.error("Konnte keinen Screenshot für die Regionsauswahl erstellen.")
        return None

    drawing_data['drawing'] = False
    drawing_data['start_point'] = None
    drawing_data['end_point'] = None
    drawing_data['roi'] = None

    def draw_rectangle_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing_data['start_point'] = (x, y)
            drawing_data['end_point'] = (x, y)
            drawing_data['drawing'] = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing_data['drawing']:
            drawing_data['end_point'] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing_data['end_point'] = (x, y)
            drawing_data['drawing'] = False

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_rectangle_callback)

    roi = None

    while True:
        img_copy = img.copy()

        if drawing_data['start_point'] and drawing_data['end_point']:
            cv2.rectangle(
                img_copy,
                drawing_data['start_point'],
                drawing_data['end_point'],
                (0, 255, 0),
                2
            )

        cv2.imshow(window_name, img_copy)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if drawing_data['start_point'] and drawing_data['end_point']:
                x1 = min(drawing_data['start_point'][0], drawing_data['end_point'][0])
                y1 = min(drawing_data['start_point'][1], drawing_data['end_point'][1])
                x2 = max(drawing_data['start_point'][0], drawing_data['end_point'][0])
                y2 = max(drawing_data['start_point'][1], drawing_data['end_point'][1])

                width = x2 - x1
                height = y2 - y1

                if width > 0 and height > 0:
                    roi = (x1, y1, width, height)
                    break

                drawing_data['start_point'] = None
                drawing_data['end_point'] = None

        elif key == 27:
            break

    cv2.destroyWindow(window_name)
    return roi


def get_screenshot_for_processing() -> Optional[np.ndarray]:
    region = LIVE_CONFIG.get('screen_region')
    capture_method = str(LIVE_CONFIG.get('capture_method', 'window')).strip().lower()

    if capture_method == 'window':
        screenshot = _get_screenshot(region)
    else:
        if region is None:
            region = _get_full_desktop_region()
        screenshot = _get_screenshot(region)

    if not _is_usable_screenshot(screenshot):
        logger.error(
            "Screenshot ist leer oder schwarz. "
            "Pruefen Sie, ob das App-Fenster sichtbar, nicht minimiert und per Titel auffindbar ist."
        )
        return None

    return screenshot
