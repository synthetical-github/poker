from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import win32con
import win32gui
import win32ui


def _list_visible_windows() -> List[Tuple[int, str]]:
    windows: List[Tuple[int, str]] = []

    def enum_handler(hwnd: int, _ctx) -> None:
        try:
            if not win32gui.IsWindowVisible(hwnd):
                return

            title = win32gui.GetWindowText(hwnd).strip()
            if not title:
                return

            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            if right - left <= 0 or bottom - top <= 0:
                return

            windows.append((hwnd, title))
        except Exception:
            return

    try:
        win32gui.EnumWindows(enum_handler, None)
    except Exception:
        try:
            hwnd = win32gui.GetForegroundWindow()
            if hwnd:
                title = win32gui.GetWindowText(hwnd).strip()
                if title:
                    windows.append((hwnd, title))
        except Exception:
            pass
    return windows


def _find_window(title_filter: str) -> Optional[Tuple[int, str]]:
    title_filter = title_filter.lower()
    for hwnd, title in _list_visible_windows():
        if title_filter in title.lower():
            return hwnd, title
    return None


def _stats(image: Optional[np.ndarray]) -> str:
    if image is None:
        return "image=None"
    return (
        f"shape={tuple(int(x) for x in image.shape)} "
        f"min={int(image.min())} max={int(image.max())} "
        f"mean={float(image.mean()):.3f} nonzero={int(np.count_nonzero(image))}"
    )


def _capture_with_printwindow(hwnd: int) -> Optional[np.ndarray]:
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        return None

    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()
    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
    save_dc.SelectObject(bitmap)

    try:
        result = win32gui.PrintWindow(hwnd, save_dc.GetSafeHdc(), 2)
        if result != 1:
            return None

        bmpinfo = bitmap.GetInfo()
        bmpstr = bitmap.GetBitmapBits(True)
        image = np.frombuffer(bmpstr, dtype=np.uint8)
        image = image.reshape((bmpinfo["bmHeight"], bmpinfo["bmWidth"], 4))
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image
    finally:
        win32gui.DeleteObject(bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)


def _capture_with_window_dc(hwnd: int) -> Optional[np.ndarray]:
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        return None

    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()
    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
    save_dc.SelectObject(bitmap)

    try:
        save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)
        bmpinfo = bitmap.GetInfo()
        bmpstr = bitmap.GetBitmapBits(True)
        image = np.frombuffer(bmpstr, dtype=np.uint8)
        image = image.reshape((bmpinfo["bmHeight"], bmpinfo["bmWidth"], 4))
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image
    finally:
        win32gui.DeleteObject(bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", help="Teilstring des Fenstertitels fuer die Suche")
    parser.add_argument("--list", action="store_true", help="Nur sichtbare Fenster auflisten")
    args = parser.parse_args()

    if args.list or not args.title:
        print("Sichtbare Fenster:")
        for hwnd, title in _list_visible_windows():
            print(f"{hwnd}: {title}")
        if not args.title:
            return

    match = _find_window(args.title)
    if not match:
        print(f"Kein sichtbares Fenster mit Titelteil '{args.title}' gefunden.")
        return

    hwnd, title = match
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    print(f"Fenster gefunden: hwnd={hwnd} title={title}")
    print(f"rect=({left}, {top}, {right}, {bottom}) size=({right-left}, {bottom-top})")

    output_dir = os.path.join(os.getcwd(), "capture_diagnostics")
    os.makedirs(output_dir, exist_ok=True)

    img_printwindow = _capture_with_printwindow(hwnd)
    print(f"printwindow {_stats(img_printwindow)}")
    if img_printwindow is not None:
        path = os.path.join(output_dir, "window_printwindow.png")
        cv2.imwrite(path, img_printwindow)
        print(f"saved={path}")

    img_dc = _capture_with_window_dc(hwnd)
    print(f"window_dc {_stats(img_dc)}")
    if img_dc is not None:
        path = os.path.join(output_dir, "window_dc.png")
        cv2.imwrite(path, img_dc)
        print(f"saved={path}")


if __name__ == "__main__":
    main()
