from __future__ import annotations

import argparse
from typing import List, Tuple

import win32gui


def list_windows() -> List[Tuple[int, str, Tuple[int, int, int, int]]]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contains", help="Nur Fenster anzeigen, deren Titel diesen Text enthaelt")
    args = parser.parse_args()

    windows = list_windows()
    query = (args.contains or "").lower()

    if query:
        windows = [item for item in windows if query in item[1].lower()]

    if not windows:
        if query:
            print(f"Keine sichtbaren Fenster mit Titelteil '{args.contains}' gefunden.")
        else:
            print("Keine sichtbaren Fenster gefunden.")
        return

    print("Fenster:")
    for hwnd, title, rect in windows:
        left, top, right, bottom = rect
        width = right - left
        height = bottom - top
        suggested = title[:40]
        print(f"hwnd={hwnd} size=({width}x{height}) title={title}")
        print(f"suggested window_title_contains: {suggested}")


if __name__ == "__main__":
    main()
