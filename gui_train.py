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

from gui_cmd import (build_cmd, build_calibration_cmd, validate_settings,
                     validate_calibration_settings)
from residual_log import setup_logging, tail_lines
from sim_config import (
    SimConfig, load as load_sim_config, save as save_sim_config,
    validate as validate_sim_config, find_exe, detect_game_version,
    default_userpath, content_userpath,
)
from vehicle_scanner import (
    scan_models, scan_trims, scan_custom_configs, resolve_part_config,
    unique_model_labels, SUPPORTED_ML_ABS_MODELS,
)
from compat import check_compat
import asset_installer
import mod_output
from model_registry import list_finished_runs

CUSTOM_TRIM_LABEL = "Custom..."

SETTINGS_PATH = os.path.join(HERE, "settings.json")
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
PPO_DEFAULTS = dict(lr=1e-4, n_steps=2048, batch_size=512, n_epochs=10,
                    clip_range=0.2, gae_lambda=0.95, ent_coef=0.005)

SAC_LABELS = dict(lr="learning rate", buffer_size="buffer size", tau="tau",
                  target_entropy="target entropy", learning_starts="learning starts",
                  train_freq="train freq (steps)")
PPO_LABELS = dict(lr="learning rate", n_steps="n steps", batch_size="batch size",
                  n_epochs="n epochs", clip_range="clip range", gae_lambda="GAE lambda",
                  ent_coef="entropy coef")


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

        # Row 1: algo dropdown
        row1 = ttk.Frame(self.training_tab)
        row1.pack(fill="x", **pad)
        ttk.Label(row1, text="Algorithm:").pack(side="left")
        self.algo_var = tk.StringVar(value="sac")
        algo_box = ttk.Combobox(row1, textvariable=self.algo_var, values=["sac", "ppo"],
                                state="readonly", width=8)
        algo_box.pack(side="left", padx=6)
        algo_box.bind("<<ComboboxSelected>>", lambda e: self._swap_algo_panel())

        self.algo_panel = ttk.Frame(self.training_tab)
        self.algo_panel.pack(fill="x", **pad)
        self._swap_algo_panel()

        # Row 2: speeds + map label
        row2 = ttk.Frame(self.training_tab)
        row2.pack(fill="x", **pad)
        ttk.Label(row2, text="Speeds (mph, comma-separated):").pack(side="left")
        self.speeds_var = tk.StringVar(value="60")
        ttk.Entry(row2, textvariable=self.speeds_var, width=20).pack(side="left", padx=6)
        ttk.Label(row2, text="(map is set in the Simulator tab)").pack(side="left", padx=12)

        self._build_car_picker(pad)

        # Row 3: pedal randomization
        row3 = ttk.Frame(self.training_tab)
        row3.pack(fill="x", **pad)
        self.pedal_random_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row3, text="Randomize pedal", variable=self.pedal_random_var,
                        command=self._toggle_pedal_entry).pack(side="left")
        self.pedal_spec_var = tk.StringVar(value="0.4-1.0")
        self.pedal_entry = ttk.Entry(row3, textvariable=self.pedal_spec_var, width=12,
                                     state="disabled")
        self.pedal_entry.pack(side="left", padx=6)

        # Row 4: surface + corner. Both key a calibration row, so the text here
        # has to match what the reference runner was run with.
        row4 = ttk.Frame(self.training_tab)
        row4.pack(fill="x", **pad)
        ttk.Label(row4, text="Tire grip:").pack(side="left")
        self.grip_var = tk.StringVar(value="off")
        ttk.Entry(row4, textvariable=self.grip_var, width=14).pack(side="left", padx=6)
        ttk.Label(row4, text='off = stock  |  "0.6"  |  "0.5,0.75,1.0"  |  "0.4-1.0"'
                  ).pack(side="left")

        row4b = ttk.Frame(self.training_tab)
        row4b.pack(fill="x", **pad)
        ttk.Label(row4b, text="Corner radius:").pack(side="left")
        self.corner_var = tk.StringVar(value="straight")
        ttk.Entry(row4b, textvariable=self.corner_var, width=14).pack(side="left", padx=6)
        ttk.Label(row4b, text='straight  |  "50"  |  "50L"  |  "50R"  (metres; '
                  "needs a calibrated steering angle)").pack(side="left")

        # Row 4c: reward preset
        row4c = ttk.Frame(self.training_tab)
        row4c.pack(fill="x", **pad)
        ttk.Label(row4c, text="Reward:").pack(side="left")
        self.reward_var = tk.StringVar(value="v5.0")
        ttk.Combobox(row4c, textvariable=self.reward_var, width=14, state="readonly",
                     values=["v5.0", "normalized"]).pack(side="left", padx=6)
        ttk.Label(row4c, text="normalized = anchored on this config's measured "
                  "slam/stock references").pack(side="left")
        ttk.Checkbutton(row4c, text="Pedal patterns ramp/pump (V4)",
                        state="disabled").pack(side="left", padx=12)

        # Row 5: run name / resume / total steps
        row5 = ttk.Frame(self.training_tab)
        row5.pack(fill="x", **pad)
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
        row6 = ttk.Frame(self.training_tab)
        row6.pack(fill="x", **pad)
        ttk.Button(row6, text="START", command=self.start).pack(side="left")
        ttk.Button(row6, text="GRACEFUL STOP", command=self.stop).pack(side="left", padx=6)
        ttk.Button(row6, text="Calibrate baselines",
                   command=self.calibrate).pack(side="left", padx=(18, 0))
        self.status_var = tk.StringVar(value="idle")
        ttk.Label(row6, textvariable=self.status_var).pack(side="left", padx=12)

        # Row 7: monitor
        row7 = ttk.LabelFrame(root, text="Monitor")
        row7.pack(fill="x", **pad)
        self.monitor_var = tk.StringVar(value="episodes: -- | rolling-20 avg_g: -- | best avg_g: -- | last outcome: --")
        ttk.Label(row7, textvariable=self.monitor_var).pack(side="left", padx=6, pady=4)

        self.root.after(1000, self._poll_monitor)

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
            row, text="Headless (no window, no GPU rendering -- BeamNG.tech only)",
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
            self.sim_status_var.set("saved, with issues -- see below")
            messagebox.showwarning("Simulator settings saved, with issues",
                                   "\n".join(problems))
        else:
            self.sim_status_var.set(f"saved to {os.path.basename(SETTINGS_PATH)}")

    # ---------------------------------------------------------- car picker
    def _build_car_picker(self, pad):
        row = ttk.Frame(self.training_tab); row.pack(fill="x", **pad)
        ttk.Label(row, text="Model:").pack(side="left")
        self.car_model_var = tk.StringVar(value="")
        self.car_model_combo = ttk.Combobox(row, textvariable=self.car_model_var,
                                            state="readonly", width=22)
        self.car_model_combo.pack(side="left", padx=6)
        self.car_model_combo.bind("<<ComboboxSelected>>",
                                  lambda e: self._on_car_model_changed())

        ttk.Label(row, text="Trim:").pack(side="left", padx=(12, 0))
        self.car_trim_var = tk.StringVar(value="")
        self.car_trim_combo = ttk.Combobox(row, textvariable=self.car_trim_var,
                                           state="readonly", width=22)
        self.car_trim_combo.pack(side="left", padx=6)
        self.car_trim_combo.bind("<<ComboboxSelected>>",
                                 lambda e: self._on_car_trim_changed())

        ttk.Label(row, text="Custom:").pack(side="left", padx=(12, 0))
        self.car_custom_var = tk.StringVar(value="")
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
        if model.name not in SUPPORTED_ML_ABS_MODELS:
            self.car_warning_var.set(
                f"No ML-ABS part ships for {model.name} yet -- only "
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

        cols = ("run_name", "algo", "model", "best_avg_g")
        self.output_tree = ttk.Treeview(row, columns=cols, show="headings", height=10)
        for c, label, w in (("run_name", "Run", 200), ("algo", "Algo", 60),
                           ("model", "Car", 100), ("best_avg_g", "Best avg_g", 90)):
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
                                    values=(run.run_name, run.algo, run.model, g))
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
            vehicle_pc=getattr(self, "_resolved_vehicle_pc", None),
            grip=self.grip_var.get(),
            corner=self.corner_var.get(),
            reward=self.reward_var.get(),
        )
        for key, var in self.field_vars.items():
            settings[key] = var.get()
        return settings

    def _run_dir(self):
        return os.path.join(HERE, "runs", self.run_name_var.get())

    def start(self):
        settings = self._collect_settings()
        log.info("START pressed: settings=%s", settings)
        problems = validate_settings(settings)
        if problems:
            log.warning("START refused: invalid settings: %s", problems)
            messagebox.showerror("Invalid settings", "\n".join(problems))
            return

        sim_cfg = self._collect_sim_config()
        sim_problems = validate_sim_config(sim_cfg)
        if sim_problems:
            log.warning("START refused: invalid simulator settings: %s", sim_problems)
            messagebox.showerror("Invalid simulator settings (Simulator tab)",
                                 "\n".join(sim_problems))
            self.notebook.select(self.sim_tab)
            return
        save_sim_config(sim_cfg, SETTINGS_PATH)  # train_residual.py reads this by default

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

    def calibrate(self):
        """Measure the slam/stock references for the configuration currently set
        on this tab, so the normalized reward has anchors for it.

        Deliberately reuses the Training tab's own speeds/grip/corner rather than
        offering its own: a reference measured on a different configuration than
        it scores is worse than a missing one, because training consumes it
        without complaint. Same simulator validation and launch path as START."""
        settings = self._collect_settings()
        problems = validate_calibration_settings(settings)
        if problems:
            log.warning("CALIBRATE refused: %s", problems)
            messagebox.showerror("Invalid settings", "\n".join(problems))
            return

        model = self._car_model_by_display.get(self.car_model_var.get())
        if not model:
            log.warning("CALIBRATE refused: no car model selected")
            messagebox.showerror(
                "No car selected",
                "Pick a car model first -- calibration is measured per car and "
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
        save_sim_config(sim_cfg, SETTINGS_PATH)   # reference_runner.py reads this

        cmd = build_calibration_cmd(settings, car=model)
        corner = settings.get("corner") or "straight"
        if not messagebox.askokcancel(
                "Calibrate baselines",
                f"Measure slam and stock ABS references for {model}.\n\n"
                f"speeds: {settings['speeds']}\n"
                f"grip: {settings.get('grip') or 'stock'}\n"
                f"corner: {corner}\n\n"
                f"This drives the car repeatedly and takes a while (a corner also "
                f"seeks its steering angle first). Results are cached in "
                f"calibration/{model}.json -- it only needs running once per "
                f"configuration.\n\nStart?"):
            log.info("calibration cancelled by user")
            return

        creationflags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
        try:
            proc = subprocess.Popen(cmd, cwd=HERE, creationflags=creationflags)
        except OSError as e:
            log.error("calibration launch FAILED: %s: %s  cmd=%s", type(e).__name__, e, cmd)
            messagebox.showerror("Launch failed", f"{e}\n\nSee logs\\gui.log")
            return
        log.info("launched calibration pid=%d car=%s cmd=%s", proc.pid, model, cmd)
        self.status_var.set(f"calibrating {model} (pid {proc.pid})")

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
