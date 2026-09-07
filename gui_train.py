"""Launcher/monitor for current co-sim PPO and legacy residual SAC/PPO.

The GUI only translates settings, spawns the selected trainer, reads its files,
and writes its run-local stop marker; environment/training logic stays in the
backend entry points.
"""
import csv
import json
import os
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gui_help
import train_monitor
import gui_state
import calibration_progress as calprog
from gui_cmd import (
    build_cmd, build_cosim_cmd, build_cosim_config, build_calibration_cmd,
    validate_settings, validate_calibration_settings,
)
from residual_log import setup_logging, tail_lines
from sim_config import (
    SimConfig, load as load_sim_config, save as save_sim_config,
    validate as validate_sim_config, find_exe, detect_game_version,
    default_userpath, content_userpath,
)
from vehicle_scanner import (
    scan_models, scan_trims, scan_custom_configs, resolve_part_config,
    unique_model_labels, check_ml_abs_car, SUPPORTED_ML_ABS_MODELS,
)
from compat import check_compat
import asset_installer
import mod_output
from model_registry import list_finished_runs
from reward_spec import PRESETS
from experiment_io import append_jsonl, update_run_state, utc_now
from simulator_guard import require_beamng_available

CUSTOM_TRIM_LABEL = "Custom..."

SETTINGS_PATH = os.path.join(HERE, "settings.json")
# Separate from settings.json, which holds the simulator config and is read
# by train_residual.py at launch, the GUI's own field memory has no
# business in a file another process parses.
GUI_STATE_PATH = os.path.join(HERE, "gui_state.json")
ASSETS_DIR = os.path.join(HERE, "assets")
RUNS_DIR = os.path.join(HERE, "runs")

log = setup_logging(os.path.join(HERE, "logs", "gui.log"), component="gui")

try:
    import beamngpy
    _BNG_VER = beamngpy.__version__.strip()
except ImportError:
    _BNG_VER = None
log.info("=== gui start: pid=%d python=%s beamngpy=%s",
         os.getpid(), sys.executable, _BNG_VER)

TRAIN_LOG_TAIL = 20   # lines of the run's train.log shown when the trainer dies

SAC_DEFAULTS = dict(lr=1e-4, buffer_size=100_000, tau=0.005,
                    target_entropy=-2.0, learning_starts=5_000, train_freq=2)
PPO_DEFAULTS = dict(lr=1e-4, n_steps=8192, batch_size=512, n_epochs=4,
                    clip_range=0.2, gae_lambda=0.95, gamma=0.995, ent_coef=0.0,
                    initial_release=0.50, log_std_init=0.0, target_kl=0.02)

SAC_LABELS = dict(lr="learning rate", buffer_size="buffer size", tau="tau",
                  target_entropy="target entropy", learning_starts="learning starts",
                  train_freq="train freq (steps)")
PPO_LABELS = dict(lr="learning rate", n_steps="n steps", batch_size="batch size",
                  n_epochs="n epochs", clip_range="clip range", gae_lambda="GAE lambda",
                  gamma="gamma", ent_coef="entropy coef", initial_release="initial release",
                  log_std_init="initial log std", target_kl="target KL")

COSIM_REWARDS = tuple(
    name for name, factory in PRESETS.items() if not factory().normalize)


def next_run_name(runs_dir, prefix="PPO"):
    """Next numeric run name, considering finished and unfinished directories."""
    import re
    highest = 0
    try:
        names = [entry.name for entry in os.scandir(runs_dir) if entry.is_dir()]
    except OSError:
        names = []
    pattern = re.compile(r"^" + re.escape(prefix) + r"-(\d+)$", re.I)
    for name in names:
        match = pattern.match(name)
        if match:
            highest = max(highest, int(match.group(1)))
    return f"{prefix}-{highest + 1:02d}"


class ResidualTrainerGUI:
    def __init__(self, root):
        self.root = root
        root.title("BeamNG ML-ABS Trainer")
        self.proc = None
        self._monitor = None
        self.field_vars = {}
        self._active_run = None          # run name of the spawned process
        self._log_seen = False           # per-run log-file-appeared transition
        self._last_parse_err = None      # dedupe repeating monitor parse errors
        self._active_backend = None
        self._residual_only_rows = []
        self._settings_backend = None
        # tkinter swallows exceptions raised inside callbacks, sys.excepthook
        # never sees them, so route them into gui.log explicitly.
        root.report_callback_exception = self._tk_exception

        self.state = gui_state.load(GUI_STATE_PATH)
        log.info("gui state: %d run fields, algos=%s",
                 len(self.state.get("run", {})), sorted(self.state.get("algo", {})))

        pad = dict(padx=6, pady=4)

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)
        self.training_tab = ttk.Frame(self.notebook)
        self.sim_tab = ttk.Frame(self.notebook)
        self.output_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.training_tab, text="Training")
        self.notebook.add(self.sim_tab, text="Simulator")
        self.notebook.add(self.output_tab, text="Output")

        self._build_simulator_tab(pad)
        self._build_output_tab(pad)

        # Row 1: backend + algo. Co-sim is the current 100 Hz training contract;
        # residual keeps the older in-car experiment available for comparison.
        row1 = ttk.Frame(self.training_tab)
        row1.pack(fill="x", **pad)
        ttk.Label(row1, text="Backend:").pack(side="left")
        self.backend_var = tk.StringVar(
            value=gui_state.run_value(self.state, "backend", "cosim"))
        self.backend_box = ttk.Combobox(
            row1, textvariable=self.backend_var, values=["cosim", "residual"],
            state="readonly", width=10)
        self.backend_box.pack(side="left", padx=6)
        self.backend_box.bind("<<ComboboxSelected>>", lambda e: self._sync_backend())
        gui_help.attach(self.backend_box, None, "backend")
        old_reward = gui_state.run_value(self.state, "reward", "v6.0")
        old_run_name = gui_state.run_value(self.state, "run_name", "residual_run1")
        self._reward_by_backend = {
            "cosim": gui_state.run_value(self.state, "cosim_reward", "v6.0"),
            "residual": gui_state.run_value(
                self.state, "residual_reward", old_reward),
        }
        self._run_name_by_backend = {
            "cosim": gui_state.run_value(
                self.state, "cosim_run_name", next_run_name(RUNS_DIR)),
            "residual": gui_state.run_value(
                self.state, "residual_run_name", old_run_name),
        }

        ttk.Label(row1, text="Algorithm:").pack(side="left")
        self.algo_var = tk.StringVar(value=gui_state.run_value(self.state, "algo", "sac"))
        self.algo_box = ttk.Combobox(
            row1, textvariable=self.algo_var, values=["sac", "ppo"],
            state="readonly", width=8)
        self.algo_box.pack(side="left", padx=6)
        self.algo_box.bind("<<ComboboxSelected>>", lambda e: self._swap_algo_panel())
        gui_help.attach(self.algo_box, None, "algo")

        ttk.Label(row1, text="Network:").pack(side="left", padx=(18, 0))
        self._net_arch_by_backend = {
            "cosim": gui_state.run_value(self.state, "cosim_net_arch", "3x256"),
            "residual": gui_state.run_value(
                self.state, "residual_net_arch",
                gui_state.run_value(self.state, "net_arch", "3x256")),
        }
        self.net_arch_var = tk.StringVar(
            value=self._net_arch_by_backend.get(self.backend_var.get(), "3x256"))
        e = ttk.Entry(row1, textvariable=self.net_arch_var, width=16)
        e.pack(side="left", padx=6)
        gui_help.attach(e, None, "net_arch")
        ttk.Label(row1, text='layers x width, or "512,256,128"').pack(side="left")

        ttk.Label(row1, text="Run-up:").pack(side="left", padx=(18, 0))
        self.runup_speed_var = tk.StringVar(value=str(
            gui_state.run_value(self.state, "runup_speed_factor", "4")))
        self.runup_speed_entry = ttk.Entry(
            row1, textvariable=self.runup_speed_var, width=6)
        self.runup_speed_entry.pack(side="left", padx=6)
        gui_help.attach(self.runup_speed_entry, None, "runup_speed_factor")
        ttk.Label(row1, text="x (co-sim acceleration only)").pack(side="left")

        self.backend_note_var = tk.StringVar(value="")
        ttk.Label(self.training_tab, textvariable=self.backend_note_var,
                  foreground="#555555").pack(fill="x", padx=6, pady=(0, 2))

        self.algo_panel = ttk.Frame(self.training_tab)
        self.algo_panel.pack(fill="x", **pad)
        self._swap_algo_panel()

        # Row 2: speeds + map label
        row2 = ttk.Frame(self.training_tab)
        row2.pack(fill="x", **pad)
        ttk.Label(row2, text="Speeds (mph, comma-separated):").pack(side="left")
        self.speeds_var = tk.StringVar(
            value=gui_state.run_value(self.state, "speeds", "60"))
        e = ttk.Entry(row2, textvariable=self.speeds_var, width=20)
        e.pack(side="left", padx=6)
        gui_help.attach(e, None, "speeds")
        ttk.Label(row2, text="(map is set in the Simulator tab)").pack(side="left", padx=12)

        self._build_car_picker(pad)

        # Row 3: pedal randomization
        row3 = ttk.Frame(self.training_tab)
        row3.pack(fill="x", **pad)
        self._residual_only_rows.append(row3)
        self.pedal_random_var = tk.BooleanVar(
            value=bool(gui_state.run_value(self.state, "pedal_random", False)))
        cb = ttk.Checkbutton(row3, text="Randomize pedal", variable=self.pedal_random_var,
                             command=self._toggle_pedal_entry)
        cb.pack(side="left")
        gui_help.attach(cb, None, "pedal_random")
        self.pedal_spec_var = tk.StringVar(
            value=gui_state.run_value(self.state, "pedal_spec", "0.4-1.0"))
        self.pedal_entry = ttk.Entry(row3, textvariable=self.pedal_spec_var, width=12,
                                     state="disabled")
        self.pedal_entry.pack(side="left", padx=6)
        gui_help.attach(self.pedal_entry, None, "pedal_spec")

        # Row 4: surface + corner. Both key a calibration row, so the text here
        # has to match what the reference runner was run with.
        row4 = ttk.Frame(self.training_tab)
        row4.pack(fill="x", **pad)
        self._residual_only_rows.append(row4)
        ttk.Label(row4, text="Tire grip:").pack(side="left")
        self.grip_var = tk.StringVar(
            value=gui_state.run_value(self.state, "grip", "off"))
        e = ttk.Entry(row4, textvariable=self.grip_var, width=14)
        e.pack(side="left", padx=6)
        gui_help.attach(e, None, "grip")
        ttk.Label(row4, text='off = stock  |  "0.6"  |  "0.5,0.75,1.0"  |  "0.4-1.0"'
                  ).pack(side="left")

        row4b = ttk.Frame(self.training_tab)
        row4b.pack(fill="x", **pad)
        self._residual_only_rows.append(row4b)
        ttk.Label(row4b, text="Corner radius:").pack(side="left")
        self.corner_var = tk.StringVar(
            value=gui_state.run_value(self.state, "corner", "straight"))
        e = ttk.Entry(row4b, textvariable=self.corner_var, width=14)
        e.pack(side="left", padx=6)
        gui_help.attach(e, None, "corner")
        ttk.Label(row4b, text='straight  |  "50"  |  "50L"  |  "50R"  (metres; '
                  "needs a calibrated steering angle)").pack(side="left")

        # Row 4c: reward preset
        row_fast = ttk.Frame(self.training_tab)
        row_fast.pack(fill="x", **pad)
        self._residual_only_rows.append(row_fast)
        self.fast_cal_var = tk.BooleanVar(
            value=bool(gui_state.run_value(self.state, "fast_calibration", True)))
        cbf = ttk.Checkbutton(row_fast, text="Fast calibration",
                              variable=self.fast_cal_var)
        cbf.pack(side="left")
        gui_help.attach(cbf, None, "fast_calibration")
        self.speed_factor_var = tk.StringVar(
            value=str(gui_state.run_value(self.state, "speed_factor", "10")))
        e = ttk.Entry(row_fast, textvariable=self.speed_factor_var, width=6)
        e.pack(side="left", padx=6)
        gui_help.attach(e, None, "speed_factor")
        ttk.Label(row_fast, text="x speed  (~11x faster per stop; affects "
                  "calibration only, not training)").pack(side="left")

        row_det = ttk.Frame(self.training_tab)
        row_det.pack(fill="x", **pad)
        self._residual_only_rows.append(row_det)
        self.determ_var = tk.BooleanVar(
            value=bool(gui_state.run_value(self.state, "deterministic", True)))
        cbd = ttk.Checkbutton(row_det, text="Deterministic training",
                              variable=self.determ_var,
                              command=self._sync_determinism_row)
        cbd.pack(side="left")
        gui_help.attach(cbd, None, "deterministic")
        # Built once and packed/forgotten, rather than created on demand: a
        # rebuilt widget loses the value was typed into it.
        self.freerun_frame = ttk.Frame(row_det)
        ttk.Label(self.freerun_frame, text="Engine speed:").pack(side="left",
                                                                 padx=(12, 0))
        self.train_speed_var = tk.StringVar(
            value=str(gui_state.run_value(self.state, "train_speed_factor", "1")))
        e = ttk.Entry(self.freerun_frame, textvariable=self.train_speed_var, width=6)
        e.pack(side="left", padx=6)
        gui_help.attach(e, None, "train_speed_factor")
        ttk.Label(self.freerun_frame,
                  text="x  (EXPERIMENTAL: free-running. avg_g stays valid; "
                       "step counts do not)").pack(side="left")
        self._sync_determinism_row()

        row4c = ttk.Frame(self.training_tab)
        row4c.pack(fill="x", **pad)
        ttk.Label(row4c, text="Reward:").pack(side="left")
        self.reward_var = tk.StringVar(
            value=self._reward_by_backend.get(self.backend_var.get(), "v6.0"))
        self.reward_combo = ttk.Combobox(
            row4c, textvariable=self.reward_var, width=14, state="readonly",
            values=list(COSIM_REWARDS) + ["normalized"])
        self.reward_combo.pack(side="left", padx=6)
        gui_help.attach(self.reward_combo, None, "reward")
        self.reward_row = row4c
        ttk.Label(row4c, text="v6.0 = sustained braking G with consistency, no "
                  "target or upper plateau").pack(side="left")
        ttk.Checkbutton(row4c, text="Pedal patterns ramp/pump (V4)",
                        state="disabled").pack(side="left", padx=12)

        # Row 5: run name / resume / total steps
        row5 = ttk.Frame(self.training_tab)
        row5.pack(fill="x", **pad)
        ttk.Label(row5, text="Run name:").pack(side="left")
        self.run_name_var = tk.StringVar(
            value=self._run_name_by_backend.get(self.backend_var.get(),
                                                next_run_name(RUNS_DIR)))
        self.run_name_entry = ttk.Entry(
            row5, textvariable=self.run_name_var, width=20)
        self.run_name_entry.pack(side="left", padx=6)
        gui_help.attach(self.run_name_entry, None, "run_name")
        ttk.Label(row5, text="Total steps:").pack(side="left", padx=(12, 0))
        self.total_steps_var = tk.StringVar(
            value=gui_state.run_value(self.state, "total_steps", "200000"))
        e = ttk.Entry(row5, textvariable=self.total_steps_var, width=10)
        e.pack(side="left", padx=6)
        gui_help.attach(e, None, "total_steps")
        self.resume_var = tk.StringVar(value="")
        self.resume_button = ttk.Button(
            row5, text="Resume from...", command=self._pick_resume)
        self.resume_button.pack(side="left", padx=(12, 0))
        gui_help.attach(self.resume_button, None, "resume")
        self.resume_label = ttk.Label(row5, text="(none)")
        self.resume_label.pack(side="left", padx=6)
        ttk.Label(row5, text="Seed:").pack(side="left", padx=(12, 0))
        self.seed_var = tk.StringVar(
            value=str(gui_state.run_value(self.state, "seed", "auto")))
        seed_entry = ttk.Entry(row5, textvariable=self.seed_var, width=11)
        seed_entry.pack(side="left", padx=6)

        # Row 6: start/stop/status
        row6 = ttk.Frame(self.training_tab)
        row6.pack(fill="x", **pad)
        self.start_button = ttk.Button(row6, text="START", command=self.start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(
            row6, text="GRACEFUL STOP", command=self.stop)
        self.stop_button.pack(side="left", padx=6)
        ttk.Button(row6, text="Show monitor",
                   command=self._show_monitor).pack(side="left", padx=6)
        self.calibrate_button = ttk.Button(
            row6, text="Calibrate baselines", command=self.calibrate)
        self.calibrate_button.pack(side="left", padx=(18, 0))
        self.status_var = tk.StringVar(value="idle")
        ttk.Label(row6, textvariable=self.status_var).pack(side="left", padx=12)

        # Row 7: monitor
        row7 = ttk.LabelFrame(root, text="Monitor")
        row7.pack(fill="x", **pad)
        self.monitor_var = tk.StringVar(value="episodes:, | rolling-20 avg_g:, | best avg_g:, | last outcome: --")
        ttk.Label(row7, textvariable=self.monitor_var).pack(side="left", padx=6, pady=4)

        self._sync_backend()
        self._reattach_active_run()
        self.root.after(1000, self._poll_monitor)

    def _open_monitor(self, settings=None):
        """Show the live monitor for the current run, reusing the window if one
        is already open, a second Toplevel would poll the same file twice and
        leave the user guessing which is current."""
        try:
            total = int(str((settings or self._collect_settings())["total_steps"]))
        except (KeyError, TypeError, ValueError):
            total = 0
        run_name = self._active_run or self.run_name_var.get()
        if self._monitor is not None:
            try:
                self._monitor.close()
            except Exception:
                pass
            self._monitor = None
        try:
            self._monitor = train_monitor.TrainMonitor(
                self.root,
                train_monitor.file_source(self._run_dir(), total_steps=total,
                                          run_name=run_name,
                                          episode_filename=self._episode_filename()),
                title=f"Training monitor - {run_name}",
                total_steps=total, on_stop=self.stop)
        except Exception as e:
            # A monitor is a convenience; failing to build one must never take
            # the run down with it.
            log.warning("monitor window failed to open: %s: %s",
                        type(e).__name__, e)
            self._monitor = None

    def _show_monitor(self):
        if self._monitor is None:
            self._open_monitor()
        else:
            self._monitor.show()

    def _sync_determinism_row(self):
        """Show the engine-speed box only when it can do anything.

        A speed factor is meaningless while stepping: physics advances only when
        Python asks, so there is no free-running clock to multiply. Leaving the
        box visible-but-inert would invite exactly the misreading it describes."""
        if self.determ_var.get():
            self.freerun_frame.pack_forget()
        else:
            self.freerun_frame.pack(side="left")

    def _sync_backend(self):
        """Make backend-specific controls honest instead of merely ignored."""
        backend = self.backend_var.get()
        if backend not in ("cosim", "residual"):
            backend = "cosim"
            self.backend_var.set(backend)
        if self._settings_backend in ("cosim", "residual"):
            self._net_arch_by_backend[self._settings_backend] = self.net_arch_var.get()
            if hasattr(self, "reward_var"):
                self._reward_by_backend[self._settings_backend] = self.reward_var.get()
            if hasattr(self, "run_name_var"):
                self._run_name_by_backend[self._settings_backend] = self.run_name_var.get()
        self.net_arch_var.set(self._net_arch_by_backend[backend])
        if hasattr(self, "reward_var"):
            self.reward_var.set(self._reward_by_backend[backend])
        if hasattr(self, "run_name_var"):
            self.run_name_var.set(self._run_name_by_backend[backend])
        self._settings_backend = backend
        if backend == "cosim":
            if self.algo_var.get() != "ppo":
                self.algo_var.set("ppo")
            if getattr(self, "_panel_state_key", None) != self._algo_state_key():
                self._swap_algo_panel()
            self.algo_box.configure(state="disabled")
            self.runup_speed_entry.configure(state="normal")
            self.reward_combo.configure(values=COSIM_REWARDS)
            if self.reward_var.get() == "normalized":
                self.reward_var.set("v6.0")
            self.resume_label.configure(text="(none)" if not self.resume_var.get()
                                        else os.path.basename(self.resume_var.get()))
            self.resume_button.configure(state="normal")
            self.calibrate_button.configure(state="disabled")
            for row in self._residual_only_rows:
                row.pack_forget()
            self.backend_note_var.set(
                "Co-sim PPO: straight line, full pedal, fixed grip; 35x64 past-frame obs "
                "and two bounded axle-release actions at 100 Hz.")
        else:
            if getattr(self, "_panel_state_key", None) != self._algo_state_key():
                self._swap_algo_panel()
            self.algo_box.configure(state="readonly")
            self.runup_speed_entry.configure(state="disabled")
            self.reward_combo.configure(values=tuple(PRESETS))
            self.resume_label.configure(text="(none)" if not self.resume_var.get()
                                        else os.path.basename(self.resume_var.get()))
            self.resume_button.configure(state="normal")
            self.calibrate_button.configure(state="normal")
            for row in self._residual_only_rows:
                if not row.winfo_manager():
                    row.pack(fill="x", padx=6, pady=4, before=self.reward_row)
            self._sync_determinism_row()
            self.backend_note_var.set(
                "Residual in-car backend: legacy comparison path with pedal, grip, "
                "corner, calibration, SAC/PPO, and optional frame stepping.")

    def _build_simulator_tab(self, pad):
        cfg = load_sim_config(SETTINGS_PATH)
        t = self.sim_tab

        row = ttk.Frame(t); row.pack(fill="x", **pad)
        ttk.Label(row, text="Game:").pack(side="left")
        self.sim_game_var = tk.StringVar(value=cfg.game)
        for val, text in (("tech", "BeamNG.tech"), ("drive", "BeamNG.drive")):
            ttk.Radiobutton(row, text=text, value=val, variable=self.sim_game_var,
                           command=self._on_sim_game_changed).pack(side="left", padx=6)

        row = ttk.Frame(t); row.pack(fill="x", **pad)
        ttk.Label(row, text="Game folder:").pack(side="left")
        self.sim_folder_var = tk.StringVar(value=cfg.game_folder)
        ttk.Entry(row, textvariable=self.sim_folder_var, width=50).pack(side="left", padx=6)
        ttk.Button(row, text="Browse...", command=self._pick_game_folder).pack(side="left")
        self.sim_folder_status_var = tk.StringVar(value="")
        ttk.Label(row, textvariable=self.sim_folder_status_var).pack(side="left", padx=8)

        row = ttk.Frame(t); row.pack(fill="x", **pad)
        ttk.Label(row, text="Userpath:").pack(side="left")
        self.sim_userpath_var = tk.StringVar(value=cfg.userpath)
        ttk.Entry(row, textvariable=self.sim_userpath_var, width=50).pack(side="left", padx=6)
        ttk.Button(row, text="Browse...", command=self._pick_userpath).pack(side="left")
        ttk.Button(row, text="Auto", command=self._auto_userpath).pack(side="left", padx=4)

        row = ttk.Frame(t); row.pack(fill="x", **pad)
        self.sim_headless_var = tk.BooleanVar(value=cfg.headless)
        self.sim_headless_check = ttk.Checkbutton(
            row, text="Headless (no window, no GPU rendering, BeamNG.tech only)",
            variable=self.sim_headless_var)
        self.sim_headless_check.pack(side="left")

        row = ttk.Frame(t); row.pack(fill="x", **pad)
        ttk.Label(row, text="Port:").pack(side="left")
        self.sim_port_var = tk.StringVar(value=str(cfg.port))
        ttk.Entry(row, textvariable=self.sim_port_var, width=8).pack(side="left", padx=6)
        ttk.Label(row, text="Map:").pack(side="left", padx=(12, 0))
        self.sim_map_var = tk.StringVar(value=cfg.map)
        ttk.Entry(row, textvariable=self.sim_map_var, width=16).pack(side="left", padx=6)

        row = ttk.Frame(t); row.pack(fill="x", **pad)
        self.sim_cpu_pinning_var = tk.BooleanVar(value=cfg.cpu_pinning)
        ttk.Checkbutton(row, text="Pin CPU cores (advanced; off by default)",
                       variable=self.sim_cpu_pinning_var).pack(side="left")

        row = ttk.Frame(t); row.pack(fill="x", **pad)
        ttk.Button(row, text="Save Simulator Settings",
                  command=self._save_sim_settings).pack(side="left")
        self.sim_status_var = tk.StringVar(value="")
        ttk.Label(row, textvariable=self.sim_status_var).pack(side="left", padx=8)

        row = ttk.LabelFrame(t, text="Compatibility"); row.pack(fill="x", **pad)
        self.sim_compat_var = tk.StringVar(value="")
        ttk.Label(row, textvariable=self.sim_compat_var, wraplength=560,
                 justify="left").pack(side="left", padx=6, pady=4)
        self.sim_proceed_anyway_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="Proceed even if incompatible",
                       variable=self.sim_proceed_anyway_var).pack(side="right", padx=6)

        row = ttk.LabelFrame(t, text="Game Assets (ML-ABS mod + reference car configs)")
        row.pack(fill="x", **pad)
        ttk.Button(row, text="Check Status",
                  command=self._check_asset_status).pack(side="left", padx=6, pady=4)
        ttk.Button(row, text="Install / Update Assets",
                  command=self._install_assets).pack(side="left", padx=6)
        self.sim_asset_status_var = tk.StringVar(value="")
        ttk.Label(row, textvariable=self.sim_asset_status_var).pack(side="left", padx=8)

        self._on_sim_game_changed()
        self._refresh_sim_folder_status()
        self.sim_folder_var.trace_add("write", lambda *a: self._refresh_compat_status())
        self._refresh_compat_status()

    def _refresh_compat_status(self):
        game_version = detect_game_version(self.sim_game_var.get(), self.sim_folder_var.get())
        compat = check_compat(game_version, _BNG_VER)
        self.sim_compat_var.set(compat.message)

    def _check_asset_status(self):
        userpath = content_userpath(self._collect_sim_config())
        status = asset_installer.installed_status(ASSETS_DIR, userpath)
        counts = {}
        for v in status.values():
            counts[v] = counts.get(v, 0) + 1
        log.info("asset status check: userpath=%s counts=%s", userpath, counts)
        self.sim_asset_status_var.set(
            ", ".join(f"{v}: {n}" for v, n in sorted(counts.items())) or "no assets")

    def _install_assets(self):
        userpath = content_userpath(self._collect_sim_config())
        report = asset_installer.install_assets(ASSETS_DIR, userpath)
        counts = {}
        for _path, action in report:
            counts[action] = counts.get(action, 0) + 1
        log.info("assets installed to %s: %s", userpath, counts)
        self.sim_asset_status_var.set(
            ", ".join(f"{a}: {n}" for a, n in sorted(counts.items())))
        messagebox.showinfo("Assets installed",
                            f"Installed to {userpath}:\n\n"
                            + "\n".join(f"{a}: {n}" for a, n in sorted(counts.items())))

    def _on_sim_game_changed(self):
        # .drive has no -gfx null; headless only makes sense for .tech.
        if self.sim_game_var.get() == "drive":
            self.sim_headless_var.set(False)
            self.sim_headless_check.configure(state="disabled")
        else:
            self.sim_headless_check.configure(state="normal")
        self._refresh_sim_folder_status()

    def _refresh_sim_folder_status(self):
        folder = self.sim_folder_var.get()
        game = self.sim_game_var.get()
        exe = find_exe(folder, game) if folder else None
        if not folder:
            self.sim_folder_status_var.set("")
        elif exe:
            ver = detect_game_version(game, folder)
            self.sim_folder_status_var.set(f"found (version: {ver or 'unknown'})")
        else:
            self.sim_folder_status_var.set("EXE NOT FOUND for this game")

    def _pick_game_folder(self):
        path = filedialog.askdirectory(title="BeamNG install folder")
        if path:
            self.sim_folder_var.set(path)
            self._refresh_sim_folder_status()

    def _pick_userpath(self):
        path = filedialog.askdirectory(title="BeamNG userpath")
        if path:
            self.sim_userpath_var.set(path)

    def _auto_userpath(self):
        self.sim_userpath_var.set(default_userpath(self.sim_game_var.get()))

    def _collect_sim_config(self):
        try:
            port = int(self.sim_port_var.get())
        except ValueError:
            port = SimConfig().port
        return SimConfig(
            game=self.sim_game_var.get(),
            game_folder=self.sim_folder_var.get(),
            userpath=self.sim_userpath_var.get(),
            headless=self.sim_headless_var.get(),
            port=port,
            map=self.sim_map_var.get() or "smallgrid",
            cpu_pinning=self.sim_cpu_pinning_var.get(),
        )

    def _save_sim_settings(self):
        cfg = self._collect_sim_config()
        problems = validate_sim_config(cfg)
        save_sim_config(cfg, SETTINGS_PATH)
        log.info("simulator settings saved: %s (problems=%s)", cfg, problems)
        if problems:
            self.sim_status_var.set("saved, with issues, see below")
            messagebox.showwarning("Simulator settings saved, with issues",
                                   "\n".join(problems))
        else:
            self.sim_status_var.set(f"saved to {os.path.basename(SETTINGS_PATH)}")

    # ---------------------------------------------------------- car picker
    def _build_car_picker(self, pad):
        row = ttk.Frame(self.training_tab); row.pack(fill="x", **pad)
        ttk.Label(row, text="Model:").pack(side="left")
        self.car_model_var = tk.StringVar(
            value=gui_state.run_value(self.state, "car_model", ""))
        self.car_model_combo = ttk.Combobox(row, textvariable=self.car_model_var,
                                            state="readonly", width=22)
        self.car_model_combo.pack(side="left", padx=6)
        self.car_model_combo.bind("<<ComboboxSelected>>",
                                  lambda e: self._on_car_model_changed())

        ttk.Label(row, text="Trim:").pack(side="left", padx=(12, 0))
        self.car_trim_var = tk.StringVar(
            value=gui_state.run_value(self.state, "car_trim", ""))
        self.car_trim_combo = ttk.Combobox(row, textvariable=self.car_trim_var,
                                           state="readonly", width=22)
        self.car_trim_combo.pack(side="left", padx=6)
        self.car_trim_combo.bind("<<ComboboxSelected>>",
                                 lambda e: self._on_car_trim_changed())

        ttk.Label(row, text="Custom:").pack(side="left", padx=(12, 0))
        self.car_custom_var = tk.StringVar(
            value=gui_state.run_value(self.state, "car_custom", ""))
        self.car_custom_combo = ttk.Combobox(row, textvariable=self.car_custom_var,
                                             state="disabled", width=22)
        self.car_custom_combo.pack(side="left", padx=6)
        self.car_custom_combo.bind("<<ComboboxSelected>>",
                                   lambda e: self._update_resolved_car())

        row_b = ttk.Frame(self.training_tab); row_b.pack(fill="x", **pad)
        self.car_resolved_var = tk.StringVar(value="(select a model)")
        ttk.Label(row_b, text="partConfig:").pack(side="left")
        ttk.Label(row_b, textvariable=self.car_resolved_var).pack(side="left", padx=6)
        self.car_warning_var = tk.StringVar(value="")
        ttk.Label(row_b, textvariable=self.car_warning_var, foreground="#b8860b").pack(
            side="left", padx=12)

        self._car_model_by_display = {}
        self._car_trim_by_display = {}
        self._car_custom_by_display = {}
        self.sim_folder_var.trace_add("write", lambda *a: self._refresh_car_models())
        self._refresh_car_models()

    def _refresh_car_models(self):
        folder = self.sim_folder_var.get()
        vehicles_dir = os.path.join(folder, "content", "vehicles") if folder else ""
        if not folder or not os.path.isdir(vehicles_dir):
            self.car_model_combo["values"] = []
            self.car_model_var.set("")
            self.car_resolved_var.set("(set Game folder in the Simulator tab)")
            return
        models = scan_models(vehicles_dir)
        self._car_model_by_display = unique_model_labels(models)
        self.car_model_combo["values"] = sorted(self._car_model_by_display)
        if self.car_model_var.get() not in self._car_model_by_display and models:
            # Only fall back when the remembered model is genuinely gone (a
            # different install, a removed car), otherwise the restored
            # selection would be overwritten on every startup.
            self.car_model_var.set(sorted(self._car_model_by_display)[0])
        self._on_car_model_changed()

    def _on_car_model_changed(self):
        model = self._car_model_by_display.get(self.car_model_var.get())
        if model is None:
            self.car_trim_combo["values"] = []
            self.car_trim_var.set("")
            self._update_resolved_car()
            return
        trims = scan_trims(model.zip_path, model.name)
        self._car_trim_by_display = {t.display_name: t for t in trims}
        values = sorted(self._car_trim_by_display) + [CUSTOM_TRIM_LABEL]
        self.car_trim_combo["values"] = values
        if self.car_trim_var.get() not in values:
            self.car_trim_var.set(values[0])
        self._on_car_trim_changed()

    def _on_car_trim_changed(self):
        model = self._car_model_by_display.get(self.car_model_var.get())
        if self.car_trim_var.get() == CUSTOM_TRIM_LABEL and model is not None:
            userpath = content_userpath(self._collect_sim_config())
            customs = scan_custom_configs(userpath, model.name)
            self._car_custom_by_display = {c.display_name: c for c in customs}
            self.car_custom_combo["values"] = sorted(self._car_custom_by_display)
            self.car_custom_combo.configure(state="readonly" if customs else "disabled")
            if customs and self.car_custom_var.get() not in self._car_custom_by_display:
                self.car_custom_var.set(sorted(self._car_custom_by_display)[0])
            if not customs:
                self.car_custom_var.set("")
        else:
            self._car_custom_by_display = {}
            self.car_custom_combo["values"] = []
            self.car_custom_var.set("")
            self.car_custom_combo.configure(state="disabled")
        self._update_resolved_car()

    def _update_resolved_car(self):
        model = self._car_model_by_display.get(self.car_model_var.get())
        if model is None:
            self.car_resolved_var.set("(select a model)")
            self.car_warning_var.set("")
            self._resolved_vehicle_pc = None
            return
        if self.car_trim_var.get() == CUSTOM_TRIM_LABEL:
            trim = self._car_custom_by_display.get(self.car_custom_var.get())
        else:
            trim = self._car_trim_by_display.get(self.car_trim_var.get())
        if trim is None:
            self.car_resolved_var.set("(no custom configs found for this model)"
                                     if self.car_trim_var.get() == CUSTOM_TRIM_LABEL
                                     else "(select a trim)")
            self._resolved_vehicle_pc = None
        else:
            self._resolved_vehicle_pc = resolve_part_config(model.name, trim.pc_name)
            self.car_resolved_var.set(self._resolved_vehicle_pc)
            self._car_abs_problem = self._check_car_has_ml_abs(
                model.name, trim.pc_name)
        if model.name not in SUPPORTED_ML_ABS_MODELS:
            self.car_warning_var.set(
                f"No ML-ABS part ships for {model.name} yet, only "
                f"{', '.join(sorted(SUPPORTED_ML_ABS_MODELS))} is currently supported.")
        else:
            self.car_warning_var.set("")

    # ------------------------------------------------------------- output tab
    def _build_output_tab(self, pad):
        t = self.output_tab

        row = ttk.LabelFrame(t, text="Deploy to game")
        row.pack(fill="x", **pad)
        ttk.Button(row, text="Generate ML ABS for all cars",
                  command=self._generate_all_cars).pack(side="left", padx=6, pady=4)
        self.output_generate_status_var = tk.StringVar(value="")
        ttk.Label(row, textvariable=self.output_generate_status_var).pack(
            side="left", padx=8)

        row = ttk.LabelFrame(t, text="Trained models")
        row.pack(fill="both", expand=True, **pad)

        cols = ("run_name", "algo", "model", "interface", "best_avg_g")
        self.output_tree = ttk.Treeview(row, columns=cols, show="headings", height=10)
        for c, label, w in (("run_name", "Run", 190), ("algo", "Algo", 55),
                           ("model", "Car", 80), ("interface", "Deploy contract", 165),
                           ("best_avg_g", "Best avg_g", 85)):
            self.output_tree.heading(c, text=label)
            self.output_tree.column(c, width=w)
        self.output_tree.pack(side="left", fill="both", expand=True, padx=6, pady=4)

        btns = ttk.Frame(row)
        btns.pack(side="left", fill="y", padx=6)
        ttk.Button(btns, text="Refresh", command=self._refresh_output_models).pack(
            fill="x", pady=2)
        ttk.Button(btns, text="Export to Game", command=self._export_selected_model).pack(
            fill="x", pady=2)
        ttk.Button(btns, text="Remove from Game", command=self._remove_selected_model).pack(
            fill="x", pady=2)

        self.output_status_var = tk.StringVar(value="")
        ttk.Label(t, textvariable=self.output_status_var, wraplength=560,
                 justify="left").pack(fill="x", **pad)

        self._output_runs = {}   # run_name -> RunInfo, refreshed by _refresh_output_models
        self._refresh_output_models()

    def _output_mod_dir(self):
        return mod_output.mod_dir_for(content_userpath(self._collect_sim_config()))

    def _generate_all_cars(self):
        cfg = self._collect_sim_config()
        vehicles_dir = os.path.join(cfg.game_folder, "content", "vehicles")
        if not cfg.game_folder or not os.path.isdir(vehicles_dir):
            messagebox.showerror("Generate ML ABS",
                                 "Set 'Game folder' on the Simulator tab first.")
            self.notebook.select(self.sim_tab)
            return
        written, skipped = mod_output.generate_all_cars(vehicles_dir, self._output_mod_dir())
        log.info("generate_all_cars: written=%d skipped=%s", len(written), skipped)
        self.output_generate_status_var.set(
            f"{len(written)} car(s) generated, {len(skipped)} skipped (no ABS slot)")
        messagebox.showinfo("ML ABS generated",
                            f"Generated for {len(written)} car(s).\n\n"
                            f"Skipped (no ABS slot in this car): "
                            f"{', '.join(skipped) if skipped else '(none)'}")

    def _refresh_output_models(self):
        for row in self.output_tree.get_children():
            self.output_tree.delete(row)
        self._output_runs = {}
        for run in list_finished_runs(RUNS_DIR):
            self._output_runs[run.run_name] = run
            g = f"{run.best_avg_g:.3f}" if run.best_avg_g is not None else "--"
            self.output_tree.insert("", "end", iid=run.run_name,
                                    values=(run.run_name, run.algo, run.model,
                                            run.deployment_interface, g))
        log.info("output tab: %d finished run(s) found", len(self._output_runs))

    def _selected_run(self):
        sel = self.output_tree.selection()
        if not sel:
            messagebox.showwarning("No model selected", "Select a run in the table first.")
            return None
        return self._output_runs.get(sel[0])

    def _export_selected_model(self):
        run = self._selected_run()
        if run is None:
            return
        self.output_status_var.set(f"Exporting {run.run_name}...")
        self.root.update_idletasks()
        ok, msg = mod_output.export_model_to_game(run, self._output_mod_dir())
        log.info("export %s -> %s: ok=%s msg=%s", run.run_name, run.model, ok, msg[:300])
        if ok:
            self.output_status_var.set(f"Exported {run.run_name} to {run.model}: {msg}")
        else:
            self.output_status_var.set(f"Export FAILED for {run.run_name}")
            messagebox.showerror("Export failed", f"{run.run_name}:\n\n{msg}")

    def _remove_selected_model(self):
        run = self._selected_run()
        if run is None:
            return
        mod_output.remove_model_from_game(run, self._output_mod_dir())
        log.info("removed %s from game (%s)", run.run_name, run.model)
        self.output_status_var.set(f"Removed {run.run_name} from the game.")

    def save_state(self):
        """Write the Training tab down. Called on START, on CALIBRATE and on
        close, cheap enough to do often, and doing it on launch means a run
        that crashes still leaves its settings behind."""
        values = {
            "backend": self.backend_var.get(),
            "algo": self.algo_var.get(),
            "cosim_net_arch": (self.net_arch_var.get()
                               if self.backend_var.get() == "cosim"
                               else self._net_arch_by_backend["cosim"]),
            "residual_net_arch": (self.net_arch_var.get()
                                  if self.backend_var.get() == "residual"
                                  else self._net_arch_by_backend["residual"]),
            "cosim_reward": (self.reward_var.get()
                             if self.backend_var.get() == "cosim"
                             else self._reward_by_backend["cosim"]),
            "residual_reward": (self.reward_var.get()
                                if self.backend_var.get() == "residual"
                                else self._reward_by_backend["residual"]),
            "cosim_run_name": (self.run_name_var.get()
                               if self.backend_var.get() == "cosim"
                               else self._run_name_by_backend["cosim"]),
            "residual_run_name": (self.run_name_var.get()
                                  if self.backend_var.get() == "residual"
                                  else self._run_name_by_backend["residual"]),
            "speeds": self.speeds_var.get(),
            "pedal_random": self.pedal_random_var.get(),
            "pedal_spec": self.pedal_spec_var.get(),
            "grip": self.grip_var.get(),
            "corner": self.corner_var.get(),
            "reward": self.reward_var.get(),
            "deterministic": self.determ_var.get(),
            "train_speed_factor": self.train_speed_var.get(),
            "runup_speed_factor": self.runup_speed_var.get(),
            "run_name": self.run_name_var.get(),
            "total_steps": self.total_steps_var.get(),
            "seed": self.seed_var.get(),
            "car_model": self.car_model_var.get(),
            "car_trim": self.car_trim_var.get(),
            "car_custom": self.car_custom_var.get(),
            "fast_calibration": self.fast_cal_var.get(),
            "speed_factor": self.speed_factor_var.get(),
        }
        gui_state.remember_run(self.state, values)
        if self.field_vars:
            gui_state.remember_algo(self.state, self._algo_state_key(),
                                    {k: v.get() for k, v in self.field_vars.items()})
        try:
            gui_state.save(self.state, GUI_STATE_PATH)
        except OSError as e:
            log.warning("could not save gui state: %s", e)

    def on_close(self):
        self.save_state()
        self.root.destroy()

    def _tk_exception(self, exc_type, exc, tb):
        log.error("tkinter callback exception: %s: %s", exc_type.__name__, exc,
                  exc_info=(exc_type, exc, tb))
        messagebox.showerror("GUI error", f"{exc_type.__name__}: {exc}\n\nSee logs\\gui.log")

    def _swap_algo_panel(self):
        # Whatever is on screen belongs to the algorithm that WAS selected, so
        # bank it before rebuilding, otherwise switching to the other
        # algorithm and back silently restores defaults over tuned values.
        if self.field_vars and getattr(self, "_panel_state_key", None):
            gui_state.remember_algo(self.state, self._panel_state_key,
                                    {k: v.get() for k, v in self.field_vars.items()})
        for child in self.algo_panel.winfo_children():
            child.destroy()
        self.field_vars = {}
        self._panel_algo = self.algo_var.get()
        self._panel_state_key = self._algo_state_key()
        defaults = SAC_DEFAULTS if self.algo_var.get() == "sac" else PPO_DEFAULTS
        labels = SAC_LABELS if self.algo_var.get() == "sac" else PPO_LABELS
        values = gui_state.algo_values(self.state, self._panel_state_key, defaults)
        for key, default in defaults.items():
            frame = ttk.Frame(self.algo_panel)
            frame.pack(side="left", padx=4)
            algo = self.algo_var.get()
            lbl = ttk.Label(frame, text=labels[key] + ":")
            lbl.pack(side="top")
            var = tk.StringVar(value=values[key])
            entry = ttk.Entry(frame, textvariable=var, width=10)
            entry.pack(side="top")
            # Both label and box: the label is the wider target, and it is what
            # a reader hovers when the NAME is the confusing part ("tau").
            gui_help.attach(lbl, algo, key)
            gui_help.attach(entry, algo, key)
            self.field_vars[key] = var

    def _algo_state_key(self):
        backend = self.backend_var.get() if hasattr(self, "backend_var") else "residual"
        return "cosim_ppo" if backend == "cosim" else self.algo_var.get()

    def _toggle_pedal_entry(self):
        self.pedal_entry.configure(state="normal" if self.pedal_random_var.get() else "disabled")

    def _pick_resume(self):
        path = filedialog.askopenfilename(filetypes=[("SB3 checkpoint", "*.zip")])
        if path:
            self.resume_var.set(path)
            self.resume_label.configure(text=os.path.basename(path))

    def _collect_settings(self):
        settings = dict(
            backend=self.backend_var.get(),
            algo=self.algo_var.get(),
            speeds=self.speeds_var.get(),
            pedal_random=self.pedal_random_var.get(),
            pedal_spec=self.pedal_spec_var.get(),
            total_steps=self.total_steps_var.get(),
            run_name=self.run_name_var.get(),
            resume=self.resume_var.get(),
            vehicle_pc=getattr(self, "_resolved_vehicle_pc", None),
            net_arch=self.net_arch_var.get(),
            fast_calibration=self.fast_cal_var.get(),
            speed_factor=self.speed_factor_var.get(),
            grip=self.grip_var.get(),
            corner=self.corner_var.get(),
            reward=self.reward_var.get(),
            deterministic=self.determ_var.get(),
            train_speed_factor=self.train_speed_var.get(),
            runup_speed_factor=self.runup_speed_var.get(),
            seed=self.seed_var.get(),
        )
        for key, var in self.field_vars.items():
            settings[key] = var.get()
        return settings

    def _run_dir(self):
        return os.path.join(HERE, "runs", self.run_name_var.get())

    def _find_active_runs(self, only_run=None):
        """Return [(run_name, backend, pid)] for live trainer PID markers."""
        try:
            import psutil
        except ImportError:
            return []
        found = []
        try:
            entries = [entry for entry in os.scandir(RUNS_DIR) if entry.is_dir()]
        except OSError:
            return found
        for entry in entries:
            if only_run is not None and entry.name != only_run:
                continue
            pid_path = os.path.join(entry.path, "pid.txt")
            try:
                with open(pid_path, encoding="ascii") as fh:
                    pid = int(fh.read().strip())
            except (OSError, ValueError):
                continue
            if not psutil.pid_exists(pid):
                continue
            try:
                command = " ".join(psutil.Process(pid).cmdline()).lower()
            except (psutil.Error, OSError):
                command = ""
            if (not command or
                    ("train_cosim.py" not in command and
                     "train_residual.py" not in command)):
                continue
            backend = ("cosim" if os.path.isfile(
                os.path.join(entry.path, "config.json")) else "residual")
            found.append((entry.name, backend, pid))
        return sorted(found)

    def _reattach_active_run(self):
        active = self._find_active_runs()
        if not active:
            return
        run_name, backend, pid = active[0]
        if len(active) > 1:
            log.warning("multiple active trainer PID files found: %s", active)
        self._active_run = run_name
        self._active_backend = backend
        self.backend_var.set(backend)
        self._sync_backend()
        self.run_name_var.set(run_name)
        self._run_name_by_backend[backend] = run_name
        self.status_var.set(f"attached to active run (pid {pid})")
        self._set_running_ui(True)
        log.info("reattached GUI controls to run=%s backend=%s pid=%d",
                 run_name, backend, pid)

    def _set_running_ui(self, running):
        self.start_button.configure(state="disabled" if running else "normal")
        self.backend_box.configure(state="disabled" if running else "readonly")
        self.run_name_entry.configure(state="disabled" if running else "normal")
        if not running:
            self._sync_backend()

    def _advance_after_run(self):
        """Move co-sim to a fresh immutable name after any terminal exit."""
        if self._active_backend == "cosim":
            fresh = next_run_name(RUNS_DIR)
            self._run_name_by_backend["cosim"] = fresh
            self.run_name_var.set(fresh)
        self._active_run = None
        self._active_backend = None

    def _episode_filename(self):
        backend = self._active_backend or self.backend_var.get()
        return "episode_log.csv" if backend == "cosim" else "episode_log_env0.csv"

    def start(self):
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showwarning("Training already active",
                                   "Stop the active run before starting another.")
            return
        active = self._find_active_runs()
        if active:
            messagebox.showwarning(
                "Training already active",
                f"Run {active[0][0]} is still active (PID {active[0][2]}). "
                "Use GRACEFUL STOP before starting another.")
            return
        settings = self._collect_settings()
        self.save_state()
        log.info("START pressed: settings=%s", settings)
        problems = validate_settings(settings)
        if problems:
            log.warning("START refused: invalid settings: %s", problems)
            messagebox.showerror("Invalid settings", "\n".join(problems))
            return

        abs_problem = (getattr(self, "_car_abs_problem", None)
                       if settings.get("backend") == "residual" else None)
        if abs_problem:
            log.warning("START refused: selected car cannot run in-car training: %s",
                        abs_problem)
            messagebox.showerror("Car has no ML ABS part", abs_problem)
            return

        sim_cfg = self._collect_sim_config()
        sim_problems = validate_sim_config(sim_cfg)
        if settings.get("backend") == "cosim" and sim_cfg.game != "tech":
            sim_problems.append("co-sim training currently requires BeamNG.tech")
        if sim_problems:
            log.warning("START refused: invalid simulator settings: %s", sim_problems)
            messagebox.showerror("Invalid simulator settings (Simulator tab)",
                                 "\n".join(sim_problems))
            self.notebook.select(self.sim_tab)
            return
        try:
            require_beamng_available(game=sim_cfg.game)
        except RuntimeError as exc:
            messagebox.showwarning("BeamNG is in use", str(exc))
            return
        save_sim_config(sim_cfg, SETTINGS_PATH)  # train_residual.py reads this by default
        settings["headless"] = sim_cfg.headless
        settings["port"] = sim_cfg.port

        game_version = detect_game_version(sim_cfg.game, sim_cfg.game_folder)
        compat = check_compat(game_version, _BNG_VER)
        log.info("compat check: game_version=%s beamngpy=%s ok=%s",
                 game_version, _BNG_VER, compat.ok)
        if compat.ok is False and not self.sim_proceed_anyway_var.get():
            log.warning("START refused: %s", compat.message)
            messagebox.showerror(
                "beamngpy / BeamNG version mismatch (Simulator tab)",
                compat.message + (f"\n\nFix: {compat.fix_command}" if compat.fix_command else "")
                + "\n\nOr tick 'Proceed even if incompatible' on the Simulator tab.")
            self.notebook.select(self.sim_tab)
            return
        elif compat.ok is not True:
            log.warning("proceeding despite compat check: %s", compat.message)

        if settings.get("backend") == "cosim" and os.path.exists(self._run_dir()):
            messagebox.showerror(
                "Run name already exists",
                f"{self._run_dir()} already exists. Co-sim runs are immutable; "
                "choose a new run name.")
            return

        stop_path = os.path.join(self._run_dir(), "STOP_TRAINING.txt")
        if os.path.exists(stop_path):
            os.remove(stop_path)
            log.info("removed stale stop file %s", stop_path)
        settings["stop_file"] = stop_path

        if settings.get("backend") == "cosim":
            config_dir = os.path.join(HERE, ".gui-configs")
            os.makedirs(config_dir, exist_ok=True)
            config_path = os.path.join(config_dir,
                                       str(settings["run_name"]) + ".json")
            with open(config_path, "w", encoding="utf-8") as fh:
                json.dump(build_cosim_config(settings), fh, indent=2)
            cmd = build_cosim_cmd(config_path)
        else:
            cmd = build_cmd(settings)
        creationflags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
        try:
            self.proc = subprocess.Popen(cmd, cwd=HERE, creationflags=creationflags)
        except OSError as e:
            log.error("launch FAILED: %s: %s  cmd=%s", type(e).__name__, e, cmd)
            messagebox.showerror("Launch failed", f"{e}\n\nSee logs\\gui.log")
            return
        self._active_run = settings["run_name"]
        self._active_backend = settings.get("backend", "residual")
        self._log_seen = False
        self._last_parse_err = None
        log.info("launched trainer pid=%d run=%s cmd=%s", self.proc.pid, self._active_run, cmd)
        self.status_var.set(f"running (pid {self.proc.pid})")
        self._set_running_ui(True)
        self._open_monitor(settings)

    def _check_car_has_ml_abs(self, model_name, pc_name):
        """Problem string if the selected configuration cannot run in-car
        training, else None.

        Training spawns the car, slams the brakes, and waits for the ML ABS
        controller to report in. A .pc without that part never reports, so the
        wait times out with "active=None", an error naming the symptom, two
        minutes after the game booted. Reading the .pc costs nothing and says
        what is actually wrong."""
        for root in (os.path.join(ASSETS_DIR, "cars", model_name),
                     os.path.join(content_userpath(self._collect_sim_config()),
                                  "vehicles", model_name)):
            path = os.path.join(root, pc_name if pc_name.endswith(".pc")
                                else pc_name + ".pc")
            if os.path.isfile(path):
                return check_ml_abs_car(path)
        return None      # not found locally, let the trainer be the judge

    def calibrate(self):
        """Measure the slam/stock references for the configuration currently set
        on this tab, so the normalized reward has anchors for it.

        Deliberately reuses the Training tab's own speeds/grip/corner rather than
        offering its own: a reference measured on a different configuration than
        it scores is worse than a missing one, because training consumes it
        without complaint. Same simulator validation and launch path as START."""
        settings = self._collect_settings()
        self.save_state()
        problems = validate_calibration_settings(settings)
        if problems:
            log.warning("CALIBRATE refused: %s", problems)
            messagebox.showerror("Invalid settings", "\n".join(problems))
            return

        model_info = self._car_model_by_display.get(self.car_model_var.get())
        # .name, not the ModelInfo itself: this becomes --car, which becomes the
        # output filename calibration/<car>.json, which is what training looks
        # up. Interpolating the object writes a file nothing will ever read.
        model = model_info.name if model_info else None
        if not model:
            log.warning("CALIBRATE refused: no car model selected")
            messagebox.showerror(
                "No car selected",
                "Pick a car model first, calibration is measured per car and "
                "written to calibration/<model>.json.")
            return

        sim_cfg = self._collect_sim_config()
        sim_problems = validate_sim_config(sim_cfg)
        if sim_problems:
            log.warning("CALIBRATE refused: invalid simulator settings: %s", sim_problems)
            messagebox.showerror("Invalid simulator settings (Simulator tab)",
                                 "\n".join(sim_problems))
            self.notebook.select(self.sim_tab)
            return
        try:
            require_beamng_available(game=sim_cfg.game)
        except RuntimeError as exc:
            messagebox.showwarning("BeamNG is in use", str(exc))
            return
        save_sim_config(sim_cfg, SETTINGS_PATH)   # reference_runner.py reads this

        cmd = build_calibration_cmd(settings, car=model)
        corner = settings.get("corner") or "straight"
        planned = self._planned_stops(settings)
        regime = self._regime_label(settings)
        mixed = self._mixed_regime_warning(model, regime)
        if not messagebox.askokcancel(
                "Calibrate baselines",
                f"Measure slam and stock ABS references for {model}.\n\n"
                f"speeds: {settings['speeds']}\n"
                f"grip: {settings.get('grip') or 'stock'}\n"
                f"corner: {corner}\n\n"
                f"{planned} stops in total, measured {regime}.{mixed}\n\n"
                f"This drives the car repeatedly (a "
                f"corner also seeks its steering angle first). Results are cached "
                f"in calibration/{model}.json, it only needs running once per "
                f"configuration.\n\nStart?"):
            log.info("calibration cancelled by user")
            return

        # No console: progress comes from the log, which the runner writes
        # anyway, so a black window full of beamngpy chatter signals
        # nothing they cannot see better in the bar below.
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        log_path = os.path.join(HERE, "logs", "calibration.log")
        try:
            mark = os.path.getsize(log_path) if os.path.exists(log_path) else 0
            proc = subprocess.Popen(cmd, cwd=HERE, creationflags=creationflags)
        except OSError as e:
            log.error("calibration launch FAILED: %s: %s  cmd=%s", type(e).__name__, e, cmd)
            messagebox.showerror("Launch failed", f"{e}\n\nSee logs\\gui.log")
            return
        log.info("launched calibration pid=%d car=%s stops=%d cmd=%s",
                 proc.pid, model, planned, cmd)
        self.status_var.set(f"calibrating {model} (pid {proc.pid})")
        self._open_calibration_window(proc, model, planned, log_path, mark)

    def _regime_label(self, settings):
        from calibration import regime_name
        if not settings.get("fast_calibration"):
            return "deterministic (stepped)"
        try:
            factor = float(settings.get("speed_factor") or 10)
        except ValueError:
            factor = 10.0
        return f"live at {factor:g}x speed"

    def _mixed_regime_warning(self, car, regime_desc):
        """A calibration table is the ruler the reward divides by, so rows
        measured different ways are not strictly comparable. Say so before the
        run rather than leaving it to be discovered in the JSON."""
        from calibration import CalibrationTable, table_regimes, regime_name
        path = os.path.join(HERE, "calibration", f"{car}.json")
        if not os.path.exists(path):
            return ""
        try:
            existing = table_regimes(CalibrationTable.load(path))
        except (OSError, ValueError):
            return ""
        if not existing:
            return ""
        s = self._collect_settings()
        new = regime_name(float(s.get("speed_factor") or 10),
                          bool(s.get("fast_calibration")))
        if new in existing and len(existing) == 1:
            return ""
        return (f"\n\nNOTE: {car}.json already holds rows measured "
                f"{', '.join(existing)}. Mixing regimes in one table means its "
                f"rows are not strictly comparable to each other.")

    def _planned_stops(self, settings):
        """How many stops the chosen configuration implies."""
        from residual_core import (parse_speeds, parse_grip_spec, parse_pedal_spec)
        speeds = parse_speeds(settings["speeds"])
        grip = parse_grip_spec(settings.get("grip") or "off")
        grips = (grip.levels() if grip else None) or [1.0]
        pedals = [1.0]
        if settings.get("pedal_random") and settings.get("pedal_spec"):
            spec = parse_pedal_spec(settings["pedal_spec"])
            pedals = (spec.levels() if spec else None) or [1.0]
        reps = 3      # reference_runner's own default
        return calprog.planned_stops(speeds, grips, pedals, reps)

    def _open_calibration_window(self, proc, car, planned, log_path, mark):
        win = tk.Toplevel(self.root)
        win.title(f"Calibrating {car}")
        win.geometry("620x260")
        pad = dict(padx=10, pady=6)

        ttk.Label(win, text=f"Measuring slam and stock references for {car}",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", **pad)

        bar = ttk.Progressbar(win, mode="determinate", maximum=planned)
        bar.pack(fill="x", **pad)

        status = tk.StringVar(value="starting BeamNG...")
        ttk.Label(win, textvariable=status).pack(anchor="w", **pad)

        detail = tk.StringVar(value="")
        ttk.Label(win, textvariable=detail, foreground="#555").pack(anchor="w", **pad)

        rows = tk.StringVar(value="")
        ttk.Label(win, textvariable=rows, foreground="#555").pack(anchor="w", **pad)

        btns = ttk.Frame(win)
        btns.pack(fill="x", side="bottom", **pad)
        # Calibration has no graceful-stop file, so cancelling really does mean
        # killing it. Rows already written to the table survive, put() saves
        # per row, so a cancelled run keeps what it measured.
        ttk.Button(btns, text="Cancel",
                   command=lambda: self._cancel_calibration(proc, win)).pack(side="right")
        ttk.Button(btns, text="Hide", command=win.withdraw).pack(side="right", padx=6)

        state = {"proc": proc, "win": win, "planned": planned,
                 "log_path": log_path, "mark": mark, "car": car,
                 "bar": bar, "status": status, "detail": detail, "rows": rows}
        self._calibration = state
        win.protocol("WM_DELETE_WINDOW", win.withdraw)
        self._poll_calibration()

    def _cancel_calibration(self, proc, win):
        if not messagebox.askyesno(
                "Cancel calibration",
                "Stop the calibration run?\n\nRows already measured are kept, "
                "the table is written after each one, so cancelling loses only "
                "the configuration currently being measured."):
            return
        try:
            proc.terminate()
        except OSError:
            pass
        log.info("calibration cancelled by user")
        win.destroy()
        self._calibration = None
        self.status_var.set("calibration cancelled")

    def _poll_calibration(self):
        st = getattr(self, "_calibration", None)
        if not st:
            return
        try:
            with open(st["log_path"], encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except OSError:
            text = ""
        # Read the whole file and let parse_progress find the last run marker,
        # rather than seeking to a byte offset captured at launch: the runner's
        # log handler may truncate or reopen the file, which leaves that offset
        # pointing past everything since written and freezes the bar at "1 of N".
        prog = calprog.parse_progress(text, total_stops=st["planned"])
        st["bar"]["value"] = prog["stops_done"]
        st["status"].set(calprog.describe(prog))
        if prog["seconds_per_stop"]:
            st["detail"].set(f"{prog['seconds_per_stop']:.0f}s per stop  |  "
                             f"{prog['rows_done']} configuration(s) written")
        st["rows"].set(f"log: {os.path.basename(st['log_path'])}")

        alive = st["proc"].poll() is None
        if alive:
            self.root.after(2000, self._poll_calibration)
            return

        if prog["finished"]:
            st["bar"]["value"] = st["planned"]
            st["status"].set(f"done, {prog['stops_done']} stops measured")
            self.status_var.set(f"calibration finished ({st['car']})")
            log.info("calibration finished: %s stops, wrote %s",
                     prog["stops_done"], prog["out_path"])
            self._refresh_output_models()
        else:
            why = prog["failed"][1] if prog["failed"] else "see logs\\calibration.log"
            st["status"].set("FAILED, " + str(why)[:120])
            self.status_var.set("calibration failed")
            log.error("calibration failed: %s", why)
        self._calibration = None

    def stop(self):
        run_name = self._active_run or self.run_name_var.get()
        run_dir = os.path.join(HERE, "runs", str(run_name))
        launched_alive = self.proc is not None and self.proc.poll() is None
        detached_alive = bool(self._find_active_runs(only_run=run_name))
        if not launched_alive and not detached_alive:
            self.status_var.set("no active training process")
            return
        stop_path = os.path.join(run_dir, "STOP_TRAINING.txt")
        os.makedirs(os.path.dirname(stop_path), exist_ok=True)
        with open(stop_path, "w") as fh:
            fh.write("stop")
        log.info("GRACEFUL STOP pressed: wrote %s (trainer pid=%s)",
                 stop_path, self.proc.pid if self.proc else None)
        self.status_var.set("stop requested, waiting for graceful save")

    def _on_trainer_exit(self, code):
        run_name = self._active_run
        backend = self._active_backend
        run_dir = os.path.join(HERE, "runs", str(run_name))
        train_log = os.path.join(run_dir, "train.log")
        if backend == "cosim" and os.path.isdir(run_dir):
            state_path = os.path.join(run_dir, "run_state.json")
            state = update_run_state(
                state_path, gui_observed_exit_code=int(code),
                gui_observed_exit_utc=utc_now())
            if code != 0 and state.get("status") in ("starting", "training"):
                update_run_state(
                    state_path, status="terminated",
                    shutdown_reason="process_exit_code_%s" % code,
                    exit_code=int(code))
            append_jsonl(os.path.join(run_dir, "run_events.jsonl"), {
                "event": "gui_observed_process_exit", "utc": utc_now(),
                "exit_code": int(code), "run_name": run_name,
            })
        if code == 0:
            log.info("trainer exited cleanly (run=%s)", run_name)
            self.status_var.set("finished")
            self._advance_after_run()
            self._set_running_ui(False)
            return
        tail = tail_lines(train_log, TRAIN_LOG_TAIL)
        log.error("trainer exited with code %s (run=%s); last %d lines of %s:\n%s",
                  code, run_name, len(tail), train_log, "\n".join(tail))
        self.status_var.set(f"exited (code {code})")
        self._advance_after_run()
        self._set_running_ui(False)
        messagebox.showerror(
            f"Trainer exited (code {code})",
            f"Run '{run_name}' died. Last lines of train.log:\n\n"
            + ("\n".join(tail) if tail else "(no train.log found)")
            + "\n\nFull detail: logs\\gui.log and that run's train.log")

    def _poll_monitor(self):
        if self.proc is not None:
            code = self.proc.poll()
            if code is not None:
                self.proc = None
                self._on_trainer_exit(code)
        elif self._active_run is not None and not self._find_active_runs(
                only_run=self._active_run):
            if str(self.status_var.get()).startswith("attached"):
                self.status_var.set("attached run ended")
                self._advance_after_run()
                self._set_running_ui(False)

        log_path = os.path.join(self._run_dir(), self._episode_filename())
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
    app = ResidualTrainerGUI(root)
    # Without this the window manager destroys the window directly and the
    # settings are never written.
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
