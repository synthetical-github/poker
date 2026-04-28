# overlay.py ÔÇö Always-on-top overlay window showing bot recommendation
import threading
import tkinter as tk
from typing import Optional


class OverlayWindow:
    """
    A small always-on-top window that shows the current recommendation.
    Runs in its own thread so it doesn't block the analysis loop.
    """

    _BG = "#1a1a2e"
    _FG_ACTION = "#00ff88"
    _FG_INFO = "#c0c0c0"
    _FG_WAIT = "#888888"
    _FG_SOLVER = "#ffcc00"
    _FONT_ACTION = ("Consolas", 18, "bold")
    _FONT_INFO = ("Consolas", 11)
    _FONT_SMALL = ("Consolas", 9)
    _WIDTH = 420
    _HEIGHT = 150

    def __init__(self):
        self._root: Optional[tk.Tk] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._pending: Optional[dict] = None
        self._running = False
        self._visible = True
        self._stop_event = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._root:
            try:
                self._root.after(0, self._root.destroy)
            except Exception:
                pass

    def run_blocking(self, stop_event=None):
        """Run overlay in the current (main) thread. Blocks until closed.
        Must be called from the main thread on Windows."""
        self._stop_event = stop_event
        self._running = True
        self._run()

    def update(self, data: dict):
        """Thread-safe update. data keys: action, amount, street, hole, board,
        equity, odds, solver_action, is_my_turn."""
        with self._lock:
            self._pending = data
        if self._root:
            try:
                self._root.after(0, self._apply_pending)
            except Exception:
                pass

    # ------------------------------------------------------------------ #

    def _run(self):
        root = tk.Tk()
        self._root = root
        root.title("Poker Bot")
        root.update_idletasks()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        pos_x = screen_w - self._WIDTH - 20
        pos_y = screen_h - self._HEIGHT - 60  # 60px Abstand von Taskbar
        root.geometry(f"{self._WIDTH}x{self._HEIGHT}+{pos_x}+{pos_y}")
        root.configure(bg=self._BG)
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.88)
        root.overrideredirect(True)   # borderless

        # Drag support so user can reposition the window
        root.bind("<ButtonPress-1>", self._drag_start)
        root.bind("<B1-Motion>", self._drag_move)

        # Close button (X) top-right corner
        close_btn = tk.Label(root, text=" X ", bg=self._BG, fg="#ff4444",
                             font=("Consolas", 10, "bold"), cursor="hand2")
        close_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-2, y=2)
        close_btn.bind("<Button-1>", lambda _: self.stop())

        # Line 1: action (big)
        self._lbl_action = tk.Label(root, text="---", bg=self._BG,
                                    fg=self._FG_WAIT, font=self._FONT_ACTION,
                                    anchor="w")
        self._lbl_action.place(x=10, y=10, width=self._WIDTH - 20)

        # Line 2: cards + street
        self._lbl_cards = tk.Label(root, text="", bg=self._BG,
                                   fg=self._FG_INFO, font=self._FONT_INFO,
                                   anchor="w")
        self._lbl_cards.place(x=10, y=48, width=self._WIDTH - 20)

        # Line 3: equity / odds
        self._lbl_eq = tk.Label(root, text="", bg=self._BG,
                                fg=self._FG_INFO, font=self._FONT_INFO,
                                anchor="w")
        self._lbl_eq.place(x=10, y=72, width=self._WIDTH - 20)

        # Line 4: solver hint
        self._lbl_solver = tk.Label(root, text="", bg=self._BG,
                                    fg=self._FG_SOLVER, font=self._FONT_SMALL,
                                    anchor="w")
        self._lbl_solver.place(x=10, y=96, width=self._WIDTH - 20)

        # Line 5: status bar
        self._lbl_status = tk.Label(root, text="Warte auf Spiel...", bg=self._BG,
                                    fg=self._FG_WAIT, font=self._FONT_SMALL,
                                    anchor="w")
        self._lbl_status.place(x=10, y=118, width=self._WIDTH - 20)

        root.after(100, self._tick)
        root.mainloop()
        self._running = False

    def _tick(self):
        if not self._running:
            if self._root:
                try:
                    self._root.quit()
                except Exception:
                    pass
            return
        if self._stop_event is not None and self._stop_event.is_set():
            self.stop()
            return
        self._apply_pending()
        if self._root:
            self._root.after(200, self._tick)

    def _apply_pending(self):
        with self._lock:
            data = self._pending
            self._pending = None
        if data is None:
            return

        action = str(data.get("action", "---")).upper()
        amount = float(data.get("amount", 0.0) or 0.0)
        is_my_turn = bool(data.get("is_my_turn", False))
        street = str(data.get("street", "")).upper()
        hole = str(data.get("hole", ""))
        board = str(data.get("board", ""))
        equity = float(data.get("equity", 0.0) or 0.0)
        odds = float(data.get("odds", 0.0) or 0.0)
        solver_action = str(data.get("solver_action", "") or "")
        villain_style = str(data.get("villain_style", "") or "")

        # Action line
        if not is_my_turn:
            action_text = "  WARTEN..."
            action_color = self._FG_WAIT
        else:
            amount_str = f"  {amount:.0f}" if amount > 0 else ""
            action_text = f"  {action}{amount_str}"
            action_color = self._FG_ACTION

        self._lbl_action.config(text=action_text, fg=action_color)

        # Cards line
        board_part = f"  BOARD {board}" if board and board != "-" else ""
        self._lbl_cards.config(text=f"  {street}  {hole}{board_part}")

        # Equity line
        eq_str = f"{equity*100:.1f}%" if equity > 0 else "--"
        odds_str = f"{odds*100:.1f}%" if odds > 0 else "--"
        self._lbl_eq.config(text=f"  EQ {eq_str}   ODDS {odds_str}   V: {villain_style}")

        # Solver hint
        if solver_action and solver_action not in {"-", "NONE", ""}:
            self._lbl_solver.config(text=f"  Solver: {solver_action.upper()}")
        else:
            self._lbl_solver.config(text="")

        # Status bar
        turn_str = "AM ZUG" if is_my_turn else "Beobachten"
        self._lbl_status.config(text=f"  {turn_str}", fg=self._FG_ACTION if is_my_turn else self._FG_WAIT)

    def _drag_start(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _drag_move(self, event):
        if self._root:
            x = self._root.winfo_x() + (event.x - self._drag_x)
            y = self._root.winfo_y() + (event.y - self._drag_y)
            self._root.geometry(f"+{x}+{y}")
