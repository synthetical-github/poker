from __future__ import annotations

import argparse
import ctypes
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pyautogui
import win32con
import win32gui
import win32ui

PW_RENDERFULLCONTENT = 0x00000002


def _list_windows() -> List[Tuple[int, str, Tuple[int, int, int, int]]]:
    windows: List[Tuple[int, str, Tuple[int, int, int, int]]] = []

    def enum_handler(hwnd: int, _ctx) -> None:
        try:
            if not win32gui.IsWindowVisible(hwnd):
                return

            title = win32gui.GetWindowText(hwnd).strip()
            if not title:
                return

            rect = win32gui.GetWindowRect(hwnd)
            left, top, right, bottom = rect
            if right - left <= 0 or bottom - top <= 0:
                return

            windows.append((hwnd, title, rect))
        except Exception:
            return

    try:
        win32gui.EnumWindows(enum_handler, None)
    except Exception:
        pass

    return windows


def _filter_windows(query: str) -> List[Tuple[int, str, Tuple[int, int, int, int]]]:
    query = query.lower()
    return [item for item in _list_windows() if query in item[1].lower()]


def _image_stats(image: Optional[np.ndarray]) -> dict:
    if image is None:
        return {"usable": False, "shape": None, "min": None, "max": None, "mean": None, "nonzero": None}

    nonzero = int(np.count_nonzero(image))
    mean = float(image.mean())
    return {
        "usable": nonzero > 0 and mean > 1.0,
        "shape": tuple(int(x) for x in image.shape),
        "min": int(image.min()),
        "max": int(image.max()),
        "mean": round(mean, 3),
        "nonzero": nonzero,
    }


def _capture_printwindow(hwnd: int) -> Optional[np.ndarray]:
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
        result = ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), PW_RENDERFULLCONTENT)
        if result != 1:
            return None

        bmpinfo = bitmap.GetInfo()
        bmpstr = bitmap.GetBitmapBits(True)
        image = np.frombuffer(bmpstr, dtype=np.uint8).reshape((bmpinfo["bmHeight"], bmpinfo["bmWidth"], 4))
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    finally:
        win32gui.DeleteObject(bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)


def _capture_window_dc(hwnd: int) -> Optional[np.ndarray]:
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
        image = np.frombuffer(bmpstr, dtype=np.uint8).reshape((bmpinfo["bmHeight"], bmpinfo["bmWidth"], 4))
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    finally:
        win32gui.DeleteObject(bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)


def _save(output_dir: str, name: str, image: Optional[np.ndarray]) -> Optional[str]:
    if image is None:
        return None
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.png")
    if cv2.imwrite(path, image):
        return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contains", default="Swiss", help="Titelteil fuer Fenstersuche")
    args = parser.parse_args()

    output_dir = os.path.join(os.getcwd(), "capture_diagnostics")
    screen_size = pyautogui.size()
    candidates = _filter_windows(args.contains)

    print("Auto Detect Capture Target")
    print(f"screen_size=({screen_size.width}, {screen_size.height})")
    print(f"query={args.contains!r}")

    if not candidates:
        print("Keine passenden sichtbaren Fenster gefunden.")
        print("Tipp: App sichtbar oeffnen und einen breiteren Suchbegriff verwenden.")
        return

    for index, (hwnd, title, rect) in enumerate(candidates, start=1):
        left, top, right, bottom = rect
        width = right - left
        height = bottom - top
        print(f"\n[{index}] hwnd={hwnd} title={title}")
        print(f"rect=({left}, {top}, {right}, {bottom}) size=({width}, {height})")

        img_pw = _capture_printwindow(hwnd)
        stats_pw = _image_stats(img_pw)
        path_pw = _save(output_dir, f"candidate_{index}_printwindow", img_pw)
        print(f"printwindow={stats_pw}")
        print(f"saved_printwindow={path_pw}")

        img_dc = _capture_window_dc(hwnd)
        stats_dc = _image_stats(img_dc)
        path_dc = _save(output_dir, f"candidate_{index}_windowdc", img_dc)
        print(f"window_dc={stats_dc}")
        print(f"saved_window_dc={path_dc}")

        best_usable = stats_pw["usable"] or stats_dc["usable"]
        suggested_title = title[:60]
        suggested_region = (0, 0, width, height)

        print("suggested config:")
        print(f"  window_title_contains = {suggested_title!r}")
        print("  capture_method = 'window'")
        print(f"  screen_region = {suggested_region}")
        print(f"  usable_capture = {best_usable}")


if __name__ == "__main__":
    main()
