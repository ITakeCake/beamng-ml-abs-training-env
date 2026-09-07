"""Live training monitor: a window instead of a wall of console text.

The trainer's console is a scrolling log; the questions actually being asked of
it are "is it getting better", "how much longer", and "did something break".
This answers those three from the episode CSV the trainer already writes.

The chart is a plain Tk canvas rather than matplotlib: matplotlib is installed
here but is not in requirements.txt, and a monitor window is not worth pinning
a plotting stack over. A scatter with a rolling-mean line is all the shape this
data has anyway.

Run it standalone against a finished run to see the layout without training:

    python train_monitor.py --demo runs/PPO-05
"""
import os
import tkinter as tk
from tkinter import ttk

import train_progress as tp

BG = "#ffffff"
GRID = "#e6e6e6"
DOT = "#b9c6d6"
LINE = "#1f6fb2"
BEST = "#1a9850"
WORST = "#d73027"
MUTED = "#666666"

PAD = dict(padx=12, pady=4)


class TrainMonitor:
    """A Toplevel that re-reads the episode log on a timer.

    Owns no training state: it is given a callable returning TrainStats, so the
    same window serves a live run and a replay without knowing which it has.
    """

    def __init__(self, parent, stats_source, title="Training", total_steps=0,
                 on_stop=None, poll_ms=2000):
        self.stats_source = stats_source
        self.on_stop = on_stop
        self.poll_ms = poll_ms
        self._alive = True
        self._stats = tp.TrainStats(total_steps=total_steps)

        self.win = tk.Toplevel(parent) if parent is not None else tk.Tk()
        self.win.title(title)
        self.win.geometry("860x620")
        self.win.minsize(680, 520)

        self._build()
        self.win.protocol("WM_DELETE_WINDOW", self.hide)
        self._tick()

    # ------------------------------------------------------------ layout
    def _build(self):
        head = ttk.Frame(self.win)
        head.pack(fill="x", **PAD)
        self.run_var = tk.StringVar(value="run: --")
        ttk.Label(head, textvariable=self.run_var,
                  font=("Segoe UI", 13, "bold")).pack(side="left")
        self.pct_var = tk.StringVar(value="")
        ttk.Label(head, textvariable=self.pct_var, foreground=MUTED,
                  font=("Segoe UI", 11)).pack(side="right")

        self.bar = ttk.Progressbar(self.win, mode="determinate", maximum=100)
        self.bar.pack(fill="x", **PAD)

        self.steps_var = tk.StringVar(value="steps: --")
        ttk.Label(self.win, textvariable=self.steps_var,
                  font=("Segoe UI", 10)).pack(anchor="w", **PAD)

        # Two columns so the eye can find a number without reading a paragraph.
        cards = ttk.Frame(self.win)
        cards.pack(fill="x", **PAD)
        cards.columnconfigure(0, weight=1)
        cards.columnconfigure(1, weight=1)

        self.best_var = tk.StringVar(value="best avg_g: --")
        self.worst_var = tk.StringVar(value="worst avg_g: --")
        self.roll_var = tk.StringVar(value="rolling-20 avg_g: --")
        self.last_var = tk.StringVar(value="last episode: --")
        self.trend_var = tk.StringVar(value="trend: --")
        self.eta_var = tk.StringVar(value="ETA: --")
        self.rate_var = tk.StringVar(value="rate: --")
        self.crash_var = tk.StringVar(value="non-STOP outcomes: --")

        left = [(self.best_var, BEST), (self.worst_var, WORST),
                (self.roll_var, None), (self.last_var, None)]
        right = [(self.trend_var, None), (self.eta_var, None),
                 (self.rate_var, None), (self.crash_var, None)]
        for r, (var, colour) in enumerate(left):
            ttk.Label(cards, textvariable=var, foreground=colour or "black",
                      font=("Segoe UI", 10)).grid(row=r, column=0, sticky="w", pady=1)
        for r, (var, colour) in enumerate(right):
            ttk.Label(cards, textvariable=var, foreground=colour or "black",
                      font=("Segoe UI", 10)).grid(row=r, column=1, sticky="w", pady=1)

        chart_box = ttk.LabelFrame(self.win, text="avg_g per episode")
        chart_box.pack(fill="both", expand=True, padx=12, pady=8)
        self.canvas = tk.Canvas(chart_box, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=6, pady=6)
        self.canvas.bind("<Configure>", lambda e: self._draw_chart())

        self.note_var = tk.StringVar(value="")
        ttk.Label(self.win, textvariable=self.note_var, foreground=MUTED,
                  font=("Segoe UI", 9)).pack(anchor="w", padx=12)

        btns = ttk.Frame(self.win)
        btns.pack(fill="x", side="bottom", **PAD)
        ttk.Button(btns, text="Hide", command=self.hide).pack(side="right")
        if self.on_stop is not None:
            ttk.Button(btns, text="Stop training (graceful)",
                       command=self.on_stop).pack(side="right", padx=6)

    # ------------------------------------------------------------- update
    def hide(self):
        self.win.withdraw()

    def show(self):
        self.win.deiconify()
        self.win.lift()

    def close(self):
        self._alive = False
        try:
            self.win.destroy()
        except tk.TclError:
            pass

    def _tick(self):
        if not self._alive:
            return
        try:
            self._stats = self.stats_source() or self._stats
        except Exception as e:                      # a monitor must never kill a run
            self.note_var.set(f"monitor read failed: {type(e).__name__}: {e}")
        else:
            self._render()
        try:
            self.win.after(self.poll_ms, self._tick)
        except tk.TclError:
            self._alive = False

    def _render(self):
        s = self._stats
        self.run_var.set(f"run: {s.run_name or '--'}")
        self.pct_var.set(f"{s.percent:.1f}%")
        self.bar["value"] = s.percent
        total = f"{s.total_steps:,}" if s.total_steps else "?"
        self.steps_var.set(
            f"steps: {s.steps_done:,} of {total}      episodes: {s.episodes:,}")

        if s.best:
            self.best_var.set(f"best avg_g: {s.best[0]:.4f}  (episode {s.best[1]})")
        if s.worst:
            self.worst_var.set(f"worst avg_g: {s.worst[0]:.4f}  (episode {s.worst[1]})")
        if s.rolling_g is not None:
            self.roll_var.set(f"rolling-{tp.RATE_WINDOW} avg_g: {s.rolling_g:.4f}")
        if s.last_g is not None:
            self.last_var.set(
                f"last episode: {s.last_g:.4f}  ({s.last_outcome})")

        self.trend_var.set(tp.format_trend(s))
        self.eta_var.set(f"ETA to {total} steps: {tp.format_eta(s.eta_seconds)}")
        self.rate_var.set(
            "rate: --" if not s.steps_per_sec
            else f"rate: {s.steps_per_sec:.1f} steps/s")
        self.crash_var.set(f"non-STOP outcomes: {s.crashes}")
        self._draw_chart()

    # -------------------------------------------------------------- chart
    def _draw_chart(self):
        c = self.canvas
        c.delete("all")
        w = c.winfo_width() or 800
        h = c.winfo_height() or 260
        left, right, top, bottom = 56, 14, 12, 28
        pw, ph = w - left - right, h - top - bottom
        series = self._stats.series
        if pw < 40 or ph < 40:
            return
        if len(series) < 2:
            c.create_text(w / 2, h / 2, text="waiting for episodes...",
                          fill=MUTED, font=("Segoe UI", 10))
            return

        xs = [e for e, _ in series]
        ys = [g for _, g in series]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        # A flat run would otherwise divide by zero and, worse, render a
        # straight line through the middle that looks like a real measurement.
        span = max(y1 - y0, 0.02)
        y0, y1 = y0 - span * 0.12, y1 + span * 0.12
        if x1 == x0:
            x1 = x0 + 1

        def px(e):
            return left + (e - x0) / float(x1 - x0) * pw

        def py(g):
            return top + (y1 - g) / float(y1 - y0) * ph

        for i in range(5):
            gv = y0 + (y1 - y0) * i / 4.0
            yy = py(gv)
            c.create_line(left, yy, left + pw, yy, fill=GRID)
            c.create_text(left - 8, yy, text=f"{gv:.3f}", anchor="e",
                          fill=MUTED, font=("Segoe UI", 8))
        c.create_line(left, top, left, top + ph, fill="#bbbbbb")
        c.create_line(left, top + ph, left + pw, top + ph, fill="#bbbbbb")
        for frac in (0.0, 0.5, 1.0):
            e = int(x0 + (x1 - x0) * frac)
            c.create_text(px(e), top + ph + 12, text=str(e), fill=MUTED,
                          font=("Segoe UI", 8))

        for e, g in series:
            x, y = px(e), py(g)
            c.create_oval(x - 1.6, y - 1.6, x + 1.6, y + 1.6, fill=DOT,
                          outline="")

        # Rolling mean: the per-episode scatter is dominated by the pedal draw,
        # so the line is what the eye should follow, not the dots.
        win_n = max(3, min(tp.RATE_WINDOW, len(ys) // 4 or 3))
        pts = []
        run_sum = 0.0
        for i, g in enumerate(ys):
            run_sum += g
            if i >= win_n:
                run_sum -= ys[i - win_n]
            if i >= win_n - 1:
                pts += [px(xs[i]), py(run_sum / win_n)]
        if len(pts) >= 4:
            c.create_line(*pts, fill=LINE, width=2, smooth=True)

        if self._stats.best:
            bg, be = self._stats.best
            c.create_oval(px(be) - 4, py(bg) - 4, px(be) + 4, py(bg) + 4,
                          outline=BEST, width=2)
        if self._stats.worst:
            wg, we = self._stats.worst
            c.create_oval(px(we) - 4, py(wg) - 4, px(we) + 4, py(wg) + 4,
                          outline=WORST, width=2)
        c.create_text(left + pw, top + 2,
                      text=f"rolling mean of {win_n}", anchor="ne",
                      fill=LINE, font=("Segoe UI", 8))


# ------------------------------------------------------------ data sources
def _csv_path(target, episode_filename="episode_log_env0.csv"):
    """Accept a run directory or the CSV itself.

    The per-run copy is a mirror of logs/episode_log_env0.csv, so a run whose
    name was reused has a short file while the master mirror still holds the
    full history, being able to point at either is what makes a demo of a
    long run possible after that has happened."""
    if os.path.isfile(target):
        return target
    return os.path.join(target, episode_filename)


def file_source(run_dir, total_steps=0, run_name="",
                episode_filename="episode_log_env0.csv"):
    """Read the run's episode CSV every call. Missing file reads as 'no episodes
    yet' rather than an error: the file appears only after episode 1 finishes."""
    import csv
    path = _csv_path(run_dir, episode_filename)

    def read():
        rows = []
        if os.path.exists(path):
            try:
                with open(path, newline="", encoding="utf-8",
                          errors="ignore") as fh:
                    rows = list(csv.DictReader(fh))
            except (OSError, csv.Error):
                rows = []
        return tp.summarize(rows, total_steps=total_steps,
                            run_name=run_name or os.path.basename(run_dir))
    return read


def replay_source(run_dir, total_steps=0, step=4, delay_ticks=1):
    """Reveal a finished run's episodes a few at a time, so the layout can be
    reviewed (and the chart seen animating) without starting a training run."""
    import csv
    path = _csv_path(run_dir)
    with open(path, newline="", encoding="utf-8", errors="ignore") as fh:
        all_rows = list(csv.DictReader(fh))
    name = os.path.basename(os.path.dirname(path)
                            if os.path.isfile(run_dir) else
                            run_dir.rstrip("/\\")) or "run"
    state = {"n": min(6, len(all_rows))}

    def read():
        rows = all_rows[:state["n"]]
        if state["n"] < len(all_rows):
            state["n"] = min(len(all_rows), state["n"] + step)
        return tp.summarize(rows, total_steps=total_steps,
                            run_name=f"{name}  (DEMO REPLAY)")
    return read


def _demo(run_dir):
    root = tk.Tk()
    root.withdraw()
    total = 1_000_000
    mon = TrainMonitor(root, replay_source(run_dir, total_steps=total),
                       title=f"Training monitor - DEMO ({os.path.basename(run_dir)})",
                       total_steps=total, poll_ms=180)
    mon.note_var.set(
        "DEMO: replaying a finished run's episode log. No training is running.")
    mon.win.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", metavar="RUN_DIR",
                    help="replay a finished run's episode log into the window")
    a = ap.parse_args()
    if a.demo:
        _demo(a.demo)
    else:
        raise SystemExit("nothing to do: pass --demo RUN_DIR")
