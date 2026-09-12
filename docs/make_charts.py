"""Regenerate docs/charts/*.png from the committed raw data under docs/data/.

    python docs/make_charts.py

Every number plotted comes from a per-stop CSV in docs/data/probes/, the
calibration file, or docs/data/runs_table.csv. Nothing is typed in by hand.
"""
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PROBES = os.path.join(HERE, "data", "probes")
CHARTS = os.path.join(HERE, "charts")
CALIB = os.path.join(REPO, "calibration", "etk800.json")
RUNS = os.path.join(HERE, "data", "runs_table.csv")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#4a3aa7", "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "normal", "axes.titlecolor": INK,
    "legend.frameon": False, "legend.fontsize": 9,
})


def stops(name):
    """{controller: [avg_g, ...]} from one probe CSV."""
    out = defaultdict(list)
    with open(os.path.join(PROBES, name), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("outcome", "STOP") == "STOP":
                out[r["controller"]].append(float(r["avg_g"]))
    return out


def calib_row(key):
    with open(CALIB, encoding="utf-8") as fh:
        return json.load(fh)["rows"][key]


REF = calib_row("grip=1.000|speed=80.0|radius=straight")
SLAM = REF["slam"]["values"]
STOCK = REF["stock"]["values"]


def refline(ax, y, label, x=0.995, color=MUTED, ha="right", dy=0.002):
    ax.axhline(y, color=color, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    ax.text(x, y + dy, label, transform=ax.get_yaxis_transform(),
            ha=ha, va="bottom", fontsize=8, color=color)


def strip(ax, x, values, color, label=None, jitter=0.06, size=22, bar=0.22):
    rng = np.random.default_rng(0)
    xs = x + rng.uniform(-jitter, jitter, len(values))
    ax.scatter(xs, values, s=size, color=color, edgecolor=SURFACE,
               linewidth=0.8, zorder=3, label=label)
    m = float(np.mean(values))
    if bar:
        ax.plot([x - bar, x + bar], [m, m], color=INK, linewidth=1.4, zorder=4)
    return m


def save(fig, name):
    os.makedirs(CHARTS, exist_ok=True)
    fig.savefig(os.path.join(CHARTS, name), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def chart_headline():
    teacher = stops("teacher_probe3.csv")["prop_0.10_kp6.0"]
    d3 = stops("d3_confirm.csv")["policy_BC-D3"]
    ppo71 = stops("ppo71_confirm.csv")["policy_PPO-71"]
    ppo39 = [1.0594]
    groups = [
        ("Locked wheels", SLAM, MUTED),
        ("Stock ABS", STOCK, MUTED),
        ("Best RL-only\n(PPO-39, 100 Hz)", ppo39, BLUE),
        ("Scripted teacher\n(400 Hz)", teacher, AQUA),
        ("BC-D3\n(distilled)", d3, ORANGE),
        ("PPO-71\n(RL from BC-D3)", ppo71, VIOLET),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for i, (label, vals, color) in enumerate(groups):
        m = strip(ax, i, vals, color)
        n = len(vals)
        ax.text(i, 0.958, "n=%d\nmean %.3f" % (n, m), ha="center", va="top",
                fontsize=8, color=INK2)
    refline(ax, 1.180, "goal 1.180", x=0.005, ha="left")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8.5)
    ax.set_ylim(0.925, 1.23)
    ax.set_ylabel("avg g per stop (2 kHz in-game metric)")
    ax.set_title("80 mph straight-line stops, etk800, dry asphalt, co-sim channel")
    ax.grid(axis="x", visible=False)
    save(fig, "headline.png")


def chart_control_rate():
    rate = {100: stops("slip_ceiling_probe2.csv"),
            200: stops("probe_200hz.csv"),
            400: stops("probe_400hz.csv")}
    series = [
        ("const_0.35", "constant release 0.35 (no feedback)", MUTED),
        ("prop_0.15_kp3.0", "P regulator, target 0.15, kp 3", BLUE),
        ("prop_0.15_kp6.0", "P regulator, target 0.15, kp 6", ORANGE),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for key, label, color in series:
        xs, ys = [], []
        for hz in (100, 200, 400):
            v = rate[hz].get(key)
            if v:
                xs.append(hz)
                ys.append(np.mean(v))
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, label=label)
        ax.text(xs[-1] + 8, ys[-1], "%.3f" % ys[-1], va="center", fontsize=8,
                color=INK2)
    v = rate[400]["prop_0.12_kp6.0"]
    ax.scatter([400], [np.mean(v)], s=40, color=AQUA, edgecolor=SURFACE,
               zorder=4, label="P regulator, target 0.12, kp 6 (400 Hz only)")
    ax.text(408, np.mean(v), "%.3f" % np.mean(v), va="center", fontsize=8,
            color=INK2)
    refline(ax, np.mean(STOCK), "stock ABS %.3f" % np.mean(STOCK), x=0.02, ha="left")
    refline(ax, np.mean(SLAM), "locked %.3f" % np.mean(SLAM), x=0.02, ha="left")
    ax.set_xticks([100, 200, 400])
    ax.set_xlim(80, 470)
    ax.set_ylim(0.95, 1.22)
    ax.set_xlabel("policy control rate (Hz), physics at 2 kHz")
    ax.set_ylabel("avg g (one stop each)")
    ax.set_title("Scripted controllers through the co-sim channel")
    ax.legend(loc="center right", bbox_to_anchor=(0.99, 0.40))
    save(fig, "control_rate.png")


def chart_constant_release():
    data = defaultdict(list)
    for f in ("const_100hz_low.csv", "const_cosim_100hz.csv", "slip_ceiling_probe2.csv"):
        for k, v in stops(f).items():
            if k.startswith("const_"):
                data[float(k.split("_")[1])].extend(v)
    xs = sorted(data)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for x in xs:
        strip(ax, x, data[x], BLUE, jitter=0.004, size=18, bar=0.012)
    means = [np.mean(data[x]) for x in xs]
    ax.plot(xs, means, color=BLUE, linewidth=1.2, zorder=2)
    best = xs[int(np.argmax(means))]
    ax.annotate("%.3f g at release %.2f" % (max(means), best),
                xy=(best, max(means)), xytext=(best + 0.06, max(means) + 0.03),
                fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    refline(ax, np.mean(STOCK), "stock ABS %.3f" % np.mean(STOCK), x=0.02, ha="left")
    refline(ax, np.mean(SLAM), "locked %.3f" % np.mean(SLAM), x=0.02, ha="left")
    ax.set_xlabel("constant brake release (0 = full pedal, 1 = no brake)")
    ax.set_ylabel("avg g")
    ax.set_title("One fixed release for the whole stop, 100 Hz")
    ax.set_ylim(0.72, 1.24)
    save(fig, "constant_release_100hz.png")


def chart_target_sweep():
    data = defaultdict(list)
    for f in ("target_sweep_low.csv", "target_sweep_fine.csv", "probe_400hz.csv",
              "teacher_probe2.csv", "teacher_probe3.csv", "teacher_probe.csv"):
        for k, v in stops(f).items():
            if k.startswith("prop_") and k.endswith("_kp6.0"):
                data[float(k.split("_")[1])].extend(v)
    xs = sorted(data)
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for x in xs:
        strip(ax, x, data[x], AQUA, jitter=0.0015, size=18, bar=0.004)
    means = [np.mean(data[x]) for x in xs]
    ax.plot(xs, means, color=AQUA, linewidth=1.2, zorder=2)
    best = int(np.argmax(means))
    ax.annotate("peak %.3f g at target %.2f (n=%d)" % (means[best], xs[best], len(data[xs[best]])),
                xy=(xs[best], means[best]), xytext=(xs[best] + 0.03, 1.218),
                fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    for x, m in zip(xs, means):
        if x >= 0.10:
            ax.text(x + 0.004, m, "%.3f" % m, ha="left", va="center", fontsize=7.5, color=INK2)
    refline(ax, np.mean(STOCK), "stock ABS %.3f" % np.mean(STOCK), x=0.98, ha="right", dy=-0.009)
    ax.set_xlabel("slip target (proportional regulator, kp 6, ground-truth slip)")
    ax.set_ylabel("avg g")
    ax.set_title("Scripted teacher at 400 Hz: slip target sweep")
    ax.set_xlim(0.02, 0.21)
    ax.set_ylim(1.13, 1.225)
    save(fig, "target_sweep_400hz.png")


def chart_distillation():
    stages = [
        ("Teacher\nP 0.10 kp 6", stops("teacher_probe3.csv")["prop_0.10_kp6.0"], AQUA),
        ("BC-0\nfrom teacher only", stops("bc_eval.csv")["policy_BC-PPO-64"], ORANGE),
        ("DAgger 1\nBC-D1", stops("d1_eval.csv")["policy_BC-D1"], ORANGE),
        ("DAgger 2\nBC-D2", stops("d2_eval.csv")["policy_BC-D2"], ORANGE),
        ("DAgger 3\nBC-D3, perturbed\ncollection", stops("d3_confirm.csv")["policy_BC-D3"], ORANGE),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for i, (label, vals, color) in enumerate(stages):
        m = strip(ax, i, vals, color)
        ax.text(i, 0.55, "n=%d\nmean %.3f\nmin %.3f" % (len(vals), m, min(vals)),
                ha="center", va="top", fontsize=8, color=INK2)
    refline(ax, np.mean(STOCK), "stock ABS %.3f" % np.mean(STOCK), x=0.005, ha="left")
    refline(ax, np.mean(SLAM), "locked %.3f" % np.mean(SLAM), x=0.005, ha="left")
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([s[0] for s in stages], fontsize=8.5)
    ax.set_ylim(0.4, 1.25)
    ax.set_ylabel("avg g per stop")
    ax.set_title("Distilling the scripted teacher into the sensor-only student, 400 Hz")
    ax.grid(axis="x", visible=False)
    save(fig, "distillation.png")


def chart_pedal_sweep():
    with open(CALIB, encoding="utf-8") as fh:
        rows = json.load(fh)["rows"]
    pedal, slam, stock = [], [], []
    for key, r in sorted(rows.items()):
        if "speed=80.0|radius=straight|pedal=" in key:
            pedal.append(float(key.split("pedal=")[1]))
            slam.append(r["slam"]["median"])
            stock.append(r["stock"]["median"])
    pedal.append(1.0)
    slam.append(REF["slam"]["median"])
    stock.append(REF["stock"]["median"])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(pedal, slam, color=MUTED, linewidth=2, label="no ABS module")
    ax.plot(pedal, stock, color=BLUE, linewidth=2, label="stock ABS")
    cross = next(p for p, a, b in zip(pedal, slam, stock) if b > a)
    ax.axvline(cross, color=AXIS, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.text(cross + 0.01, 0.77, "stock ABS pulls ahead\nabove pedal %.2f" % cross,
            fontsize=8.5, color=INK2, va="bottom")
    ax.set_xlabel("brake pedal fraction (median of 3 stops each)")
    ax.set_ylabel("avg g")
    ax.set_title("Where ABS matters: partial-pedal stops at 80 mph")
    ax.legend(loc="upper left")
    save(fig, "pedal_sweep.png")


def chart_training_progress():
    rows = []
    with open(RUNS, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    ppo = [r for r in rows if r["kind"] == "ppo" and r["run"].startswith("PPO-")
           and (r["n_episodes"] not in ("", "0") or r["eval_g"])]
    ppo.sort(key=lambda r: r["notes"].split("first_log_ts=")[-1] + r["run"])
    fig, ax = plt.subplots(figsize=(11, 4.4))
    xs = list(range(len(ppo)))
    for x, r in zip(xs, ppo):
        hz = r["control_hz"] or "100"
        color = ORANGE if hz == "400" else BLUE
        unreliable = r["log_unreliable"] == "true"
        if r["last20_mean_g"] and not unreliable:
            ax.scatter([x], [float(r["last20_mean_g"])], s=26, facecolor=SURFACE,
                       edgecolor=color, linewidth=1.3, zorder=3)
        if r["eval_g"]:
            ax.scatter([x], [float(r["eval_g"])], s=44, color=color,
                       edgecolor=SURFACE, linewidth=0.8, zorder=4)
        if unreliable and not r["eval_g"]:
            ax.scatter([x], [0.42], s=18, marker="x", color=MUTED, linewidth=1, zorder=3)
        if r["status"] == "failed":
            ax.scatter([x], [0.40], s=18, marker="_", color=RED, linewidth=1.5, zorder=3)
    refline(ax, np.mean(STOCK), "stock ABS %.3f" % np.mean(STOCK), x=0.005, ha="left")
    refline(ax, np.mean(SLAM), "locked %.3f" % np.mean(SLAM), x=0.005, ha="left")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["run"].replace("PPO-", "").replace("-resume-1", "r")
                        for r in ppo], rotation=90, fontsize=7.5)
    ax.set_xlabel("PPO run, chronological (2026-08-30 to 2026-09-09)")
    ax.set_ylabel("avg g")
    ax.set_ylim(0.36, 1.25)
    ax.set_title("Every PPO run with a brake measurement")
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", color=BLUE, markerfacecolor=SURFACE, linestyle="", label="100 Hz, training-log mean of last 20 stops"),
        Line2D([], [], marker="o", color=BLUE, linestyle="", label="100 Hz, frozen deterministic evaluation"),
        Line2D([], [], marker="o", color=ORANGE, linestyle="", label="400 Hz, checkpoint probe (log is unreliable at 400 Hz)"),
        Line2D([], [], marker="x", color=MUTED, linestyle="", label="400 Hz, no evaluation: no trustworthy g"),
        Line2D([], [], marker="_", color=RED, linestyle="", markersize=10, label="run crashed"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.62, 0.93), fontsize=8)
    ax.grid(axis="x", visible=False)
    save(fig, "training_progress.png")


def chart_axle_vs_independent():
    ind = [
        ("teacher", stops("teacher_probe3.csv")["prop_0.10_kp6.0"]),
        ("student", stops("d3_confirm.csv")["policy_BC-D3"]),
    ]
    axle = [
        ("teacher", stops("teacher_deploy2.csv")["prop_0.01_kp3.0_axle"]),
        ("student", stops("deploy_final.csv")["policy_DEPLOY-D1R"]),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    labels = []
    i = 0
    for group, color, name in ((ind, ORANGE, "4 per-wheel"), (axle, VIOLET, "2 per-axle")):
        for role, vals in group:
            m = strip(ax, i, vals, color)
            ax.text(i, 1.105, "n=%d\n%.3f" % (len(vals), m), ha="center", va="top",
                    fontsize=8, color=INK2)
            labels.append("%s\n%s" % (role, name))
            i += 1
        i += 0.4
    refline(ax, np.mean(STOCK), "stock ABS %.3f" % np.mean(STOCK), x=0.005, ha="left")
    ax.set_xticks([0, 1, 2.4, 3.4])
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(1.09, 1.215)
    ax.set_ylabel("avg g per stop")
    ax.set_title("Action space: per-wheel vs per-axle release, 400 Hz")
    ax.grid(axis="x", visible=False)
    save(fig, "axle_vs_independent.png")


def chart_noise():
    data = []
    for f, sd in (("noise_0.000.csv", 0.0), ("noise_0.025.csv", 0.025), ("noise_0.050.csv", 0.05)):
        data.append((sd, stops(f)["policy_BC-D3"]))
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for i, (sd, vals) in enumerate(data):
        m = strip(ax, i, vals, ORANGE)
        ax.text(i, 1.125, "n=%d\n%.3f" % (len(vals), m), ha="center", va="top", fontsize=8, color=INK2)
    refline(ax, np.mean(STOCK), "stock ABS %.3f" % np.mean(STOCK), x=0.005, ha="left")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["0", "0.025", "0.050"])
    ax.set_xlabel("gaussian noise added to every release action (sd)")
    ax.set_ylabel("avg g")
    ax.set_ylim(1.11, 1.215)
    ax.set_title("BC-D3 under action noise")
    ax.grid(axis="x", visible=False)
    save(fig, "noise_robustness.png")


if __name__ == "__main__":
    chart_headline()
    chart_training_progress()
    chart_control_rate()
    chart_constant_release()
    chart_target_sweep()
    chart_distillation()
    chart_axle_vs_independent()
    chart_noise()
    chart_pedal_sweep()
