"""Launcher/monitor for residual ABS training (SAC or PPO). Shells out to
train_residual.py -- never reimplements training logic here."""
import csv
import os
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gui_cmd import build_cmd, validate_settings
from residual_log import setup_logging, tail_lines

log = setup_logging(os.path.join(HERE, "logs", "gui.log"), component="gui")

try:
    import beamngpy
    _BNG_VER = beamngpy.__version__.strip()
    BEAMNGPY_OK = _BNG_VER == "1.34.1"
except ImportError:
    _BNG_VER = None
    BEAMNGPY_OK = False
log.info("=== gui start: pid=%d python=%s beamngpy=%s ok=%s",
         os.getpid(), sys.executable, _BNG_VER, BEAMNGPY_OK)

TRAIN_LOG_TAIL = 20   # lines of the run's train.log shown when the trainer dies

SAC_DEFAULTS = dict(lr=1e-4, buffer_size=100_000, tau=0.005,
                    target_entropy=-2.0, learning_starts=5_000)
PPO_DEFAULTS = dict(lr=1e-4, n_steps=2048, batch_size=512, n_epochs=10,
                    clip_range=0.2, gae_lambda=0.95)

SAC_LABELS = dict(lr="learning rate", buffer_size="buffer size", tau="tau",
                  target_entropy="target entropy", learning_starts="learning starts")
PPO_LABELS = dict(lr="learning rate", n_steps="n steps", batch_size="batch size",
                  n_epochs="n epochs", clip_range="clip range", gae_lambda="GAE lambda")


class ResidualTrainerGUI:
    def __init__(self, root):
        self.root = root
        root.title("Residual ABS Trainer")
        self.proc = None
        self.field_vars = {}
        self._active_run = None          # run name of the process we launched
        self._log_seen = False           # per-run log-file-appeared transition
        self._last_parse_err = None      # dedupe repeating monitor parse errors
        # tkinter swallows exceptions raised inside callbacks -- sys.excepthook
        # never sees them -- so route them into gui.log explicitly.
        root.report_callback_exception = self._tk_exception

        pad = dict(padx=6, pady=4)

        # Row 1: algo dropdown
        row1 = ttk.Frame(root)
        row1.grid(row=0, column=0, sticky="ew", **pad)
        ttk.Label(row1, text="Algorithm:").pack(side="left")
        self.algo_var = tk.StringVar(value="sac")
        algo_box = ttk.Combobox(row1, textvariable=self.algo_var, values=["sac", "ppo"],
                                state="readonly", width=8)
        algo_box.pack(side="left", padx=6)
        algo_box.bind("<<ComboboxSelected>>", lambda e: self._swap_algo_panel())

        self.algo_panel = ttk.Frame(root)
        self.algo_panel.grid(row=1, column=0, sticky="ew", **pad)
        self._swap_algo_panel()

        # Row 2: speeds + map label
        row2 = ttk.Frame(root)
        row2.grid(row=2, column=0, sticky="ew", **pad)
        ttk.Label(row2, text="Speeds (mph, comma-separated):").pack(side="left")
        self.speeds_var = tk.StringVar(value="60")
        ttk.Entry(row2, textvariable=self.speeds_var, width=20).pack(side="left", padx=6)
        ttk.Label(row2, text='Map: smallgrid ("Grid, Small, Pure")').pack(side="left", padx=12)

        # Row 3: pedal randomization
        row3 = ttk.Frame(root)
        row3.grid(row=3, column=0, sticky="ew", **pad)
        self.pedal_random_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row3, text="Randomize pedal", variable=self.pedal_random_var,
                        command=self._toggle_pedal_entry).pack(side="left")
        self.pedal_spec_var = tk.StringVar(value="0.4-1.0")
        self.pedal_entry = ttk.Entry(row3, textvariable=self.pedal_spec_var, width=12,
                                     state="disabled")
        self.pedal_entry.pack(side="left", padx=6)

        # Row 4: deferred/dummy switches
        row4 = ttk.Frame(root)
        row4.grid(row=4, column=0, sticky="ew", **pad)
        ttk.Checkbutton(row4, text="Turn angles (needs steering reward design)",
                        state="disabled").pack(side="left")
        ttk.Checkbutton(row4, text="Pedal patterns ramp/pump (V4)",
                        state="disabled").pack(side="left", padx=12)

        # Row 5: run name / resume / total steps
        row5 = ttk.Frame(root)
        row5.grid(row=5, column=0, sticky="ew", **pad)
        ttk.Label(row5, text="Run name:").pack(side="left")
        self.run_name_var = tk.StringVar(value="residual_run1")
        ttk.Entry(row5, textvariable=self.run_name_var, width=20).pack(side="left", padx=6)
        ttk.Label(row5, text="Total steps:").pack(side="left", padx=(12, 0))
        self.total_steps_var = tk.StringVar(value="200000")
        ttk.Entry(row5, textvariable=self.total_steps_var, width=10).pack(side="left", padx=6)
        self.resume_var = tk.StringVar(value="")
        ttk.Button(row5, text="Resume from...", command=self._pick_resume).pack(side="left", padx=(12, 0))
        self.resume_label = ttk.Label(row5, text="(none)")
        self.resume_label.pack(side="left", padx=6)

        # Row 6: start/stop/status
        row6 = ttk.Frame(root)
        row6.grid(row=6, column=0, sticky="ew", **pad)
        ttk.Button(row6, text="START", command=self.start).pack(side="left")
        ttk.Button(row6, text="GRACEFUL STOP", command=self.stop).pack(side="left", padx=6)
        self.status_var = tk.StringVar(value="idle")
        ttk.Label(row6, textvariable=self.status_var).pack(side="left", padx=12)

        # Row 7: monitor
        row7 = ttk.LabelFrame(root, text="Monitor")
        row7.grid(row=7, column=0, sticky="ew", **pad)
        self.monitor_var = tk.StringVar(value="episodes: -- | rolling-20 avg_g: -- | best avg_g: -- | last outcome: --")
        ttk.Label(row7, textvariable=self.monitor_var).pack(side="left", padx=6, pady=4)

        self.root.after(1000, self._poll_monitor)

    def _tk_exception(self, exc_type, exc, tb):
        log.error("tkinter callback exception: %s: %s", exc_type.__name__, exc,
                  exc_info=(exc_type, exc, tb))
        messagebox.showerror("GUI error", f"{exc_type.__name__}: {exc}\n\nSee logs\\gui.log")

    def _swap_algo_panel(self):
        for child in self.algo_panel.winfo_children():
            child.destroy()
        self.field_vars = {}
        defaults = SAC_DEFAULTS if self.algo_var.get() == "sac" else PPO_DEFAULTS
        labels = SAC_LABELS if self.algo_var.get() == "sac" else PPO_LABELS
        for key, default in defaults.items():
            frame = ttk.Frame(self.algo_panel)
            frame.pack(side="left", padx=4)
            ttk.Label(frame, text=labels[key] + ":").pack(side="top")
            var = tk.StringVar(value=str(default))
            ttk.Entry(frame, textvariable=var, width=10).pack(side="top")
            self.field_vars[key] = var

    def _toggle_pedal_entry(self):
        self.pedal_entry.configure(state="normal" if self.pedal_random_var.get() else "disabled")

    def _pick_resume(self):
        path = filedialog.askopenfilename(filetypes=[("SB3 checkpoint", "*.zip")])
        if path:
            self.resume_var.set(path)
            self.resume_label.configure(text=os.path.basename(path))

    def _collect_settings(self):
        settings = dict(
            algo=self.algo_var.get(),
            speeds=self.speeds_var.get(),
            pedal_random=self.pedal_random_var.get(),
            pedal_spec=self.pedal_spec_var.get(),
            total_steps=self.total_steps_var.get(),
            run_name=self.run_name_var.get(),
            resume=self.resume_var.get(),
        )
        for key, var in self.field_vars.items():
            settings[key] = var.get()
        return settings

    def _run_dir(self):
        return os.path.join(HERE, "runs", self.run_name_var.get())

    def start(self):
        if not BEAMNGPY_OK:
            log.error("START refused: beamngpy=%s (need 1.34.1)", _BNG_VER)
            messagebox.showerror("beamngpy version",
                                 "beamngpy is missing or does not match the pinned "
                                 "1.34.1 this project's BeamNG.tech install speaks.")
            return

        settings = self._collect_settings()
        log.info("START pressed: settings=%s", settings)
        problems = validate_settings(settings)
        if problems:
            log.warning("START refused: invalid settings: %s", problems)
            messagebox.showerror("Invalid settings", "\n".join(problems))
            return

        pid_path = os.path.join(self._run_dir(), "pid.txt")
        if os.path.exists(pid_path):
            log.warning("START refused: %s exists (run already active or stale)", pid_path)
            messagebox.showerror("Run already active",
                                 f"{pid_path} exists -- a training process for this "
                                 f"run name may already be running. Choose a different "
                                 f"run name, or delete that file if it is stale.")
            return

        stop_path = os.path.join(HERE, "STOP_TRAINING.txt")
        if os.path.exists(stop_path):
            os.remove(stop_path)
            log.info("removed stale stop file %s", stop_path)

        cmd = build_cmd(settings)
        creationflags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
        try:
            self.proc = subprocess.Popen(cmd, cwd=HERE, creationflags=creationflags)
        except OSError as e:
            log.error("launch FAILED: %s: %s  cmd=%s", type(e).__name__, e, cmd)
            messagebox.showerror("Launch failed", f"{e}\n\nSee logs\\gui.log")
            return
        self._active_run = settings["run_name"]
        self._log_seen = False
        self._last_parse_err = None
        log.info("launched trainer pid=%d run=%s cmd=%s", self.proc.pid, self._active_run, cmd)
        self.status_var.set(f"running (pid {self.proc.pid})")

    def stop(self):
        stop_path = os.path.join(HERE, "STOP_TRAINING.txt")
        with open(stop_path, "w") as fh:
            fh.write("stop")
        log.info("GRACEFUL STOP pressed: wrote %s (trainer pid=%s)",
                 stop_path, self.proc.pid if self.proc else None)
        self.status_var.set("stop requested -- waiting for graceful save")

    def _on_trainer_exit(self, code):
        train_log = os.path.join(HERE, "runs", str(self._active_run), "train.log")
        if code == 0:
            log.info("trainer exited cleanly (run=%s)", self._active_run)
            self.status_var.set("finished")
            return
        tail = tail_lines(train_log, TRAIN_LOG_TAIL)
        log.error("trainer exited with code %s (run=%s); last %d lines of %s:\n%s",
                  code, self._active_run, len(tail), train_log, "\n".join(tail))
        self.status_var.set(f"exited (code {code})")
        messagebox.showerror(
            f"Trainer exited (code {code})",
            f"Run '{self._active_run}' died. Last lines of train.log:\n\n"
            + ("\n".join(tail) if tail else "(no train.log found)")
            + "\n\nFull detail: logs\\gui.log and that run's train.log")

    def _poll_monitor(self):
        if self.proc is not None:
            code = self.proc.poll()
            if code is not None:
                self.proc = None
                self._on_trainer_exit(code)

        log_path = os.path.join(self._run_dir(), "episode_log_env0.csv")
        if os.path.exists(log_path):
            if not self._log_seen:
                self._log_seen = True
                log.info("episode log appeared: %s", log_path)
            try:
                with open(log_path, newline="") as fh:
                    rows = list(csv.DictReader(fh))
            except (OSError, csv.Error) as e:
                rows = []
                err = f"{type(e).__name__}: {e}"
                if err != self._last_parse_err:      # log a repeating error once
                    self._last_parse_err = err
                    log.warning("monitor: cannot read %s: %s", log_path, err)
            if rows:
                tail = rows[-20:]
                gs = [float(r["avg_g"]) for r in tail if r.get("avg_g")]
                best = max((float(r["avg_g"]) for r in rows if r.get("avg_g")), default=0.0)
                mean = sum(gs) / len(gs) if gs else 0.0
                last_outcome = rows[-1].get("outcome", "--")
                self.monitor_var.set(
                    f"episodes: {len(rows)} | rolling-20 avg_g: {mean:.3f} | "
                    f"best avg_g: {best:.3f} | last outcome: {last_outcome}")
        self.root.after(1000, self._poll_monitor)


def main():
    root = tk.Tk()
    ResidualTrainerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
