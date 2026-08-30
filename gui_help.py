"""Hover help for every setting on the Training tab.

Written for someone who has not read a reinforcement-learning paper. Each entry
says what the knob does in plain terms, what happens if it is too high or too
low, and -- where there is one -- the concrete symptom to watch for in this
project. Jargon is spelled out on first use rather than assumed; a label like
"tau" or "GAE lambda" tells a reader nothing on its own, which is exactly why
these exist.

Kept out of gui_train.py so the text can be read, reviewed and tested as prose
rather than buried in widget construction.
"""

# --- what the car is asked to do -------------------------------------------
RUN_HELP = {
    "speeds": (
        "How fast the car is going when the brakes slam on, in mph.\n\n"
        "One number (60) trains at that speed every time. A list (60,90,120)\n"
        "picks one at random each attempt, which makes the controller work at\n"
        "any speed instead of memorising one.\n\n"
        "Every speed you list needs its own measured baseline before you can\n"
        "use the normalized reward -- press 'Calibrate baselines' first."
    ),
    "pedal_random": (
        "Vary how hard the driver presses the brake pedal, ONE value per\n"
        "attempt.\n\n"
        "Off: the pedal is mashed to the floor (100%) every time.\n"
        "On:  each attempt picks a random amount from the range and HOLDS it\n"
        "     for that whole stop -- 62% this time, 87% the next. The pedal\n"
        "     does not move during a stop.\n\n"
        "Real drivers do not always slam the pedal, so this teaches the\n"
        "controller to work from a half-pressed pedal as well as a floored\n"
        "one. It sees the current pedal position, so it can adapt rather than\n"
        "guess.\n\n"
        "Learning is slower, because it has to handle every pedal position\n"
        "instead of just one."
    ),
    "pedal_spec": (
        "Which pedal positions to draw from. One is picked per attempt and\n"
        "held for the whole stop.\n\n"
        "0.5-1.0       a range -- every 0.01 step between them (51 levels)\n"
        "0.5,0.75,1.0  a list -- only these\n"
        "0.6           one fixed level\n\n"
        "Values snap to 2 decimals, because each distinct level needs its own\n"
        "measured baseline before the normalized reward can score it.\n\n"
        "With 'Fast calibration' on, the whole 0.5-1.0 range is about 20\n"
        "minutes; stepped it is nearer three hours. A list is still cheaper\n"
        "when you do not need the resolution.\n\n"
        "Press 'Calibrate baselines' after setting this -- it measures exactly\n"
        "the levels this implies."
    ),
    "grip": (
        "How slippery the road is, by scaling tyre grip.\n\n"
        "off      = leave the tyres alone (normal dry road)\n"
        "0.6      = 60% of normal grip, roughly wet\n"
        "0.5,0.75,1.0 = pick one of these at random each attempt\n"
        "0.4-1.0  = pick any value in that range\n\n"
        "Grip changes at the exact moment braking starts, so the car always\n"
        "reaches its target speed normally first.\n\n"
        "The normalized reward needs a measured baseline per grip level, so\n"
        "use a LIST of levels (not a range) if you are using it -- a range can\n"
        "produce a value nothing was measured at."
    ),
    "corner": (
        "Brake while turning, instead of in a straight line.\n\n"
        "straight = no turn (the normal brake test)\n"
        "150      = a 150 metre radius left turn\n"
        "150L / 150R = same, explicitly left or right\n\n"
        "Bigger number = gentler curve. Too tight and the car cannot hold the\n"
        "turn at all: at 60 mph this car tops out around 64 m, and even that\n"
        "uses all the grip, leaving none for braking. 150 m is a sensible\n"
        "test -- it uses about half the grip.\n\n"
        "Needs 'Calibrate baselines' run first: the tool has to measure what\n"
        "steering angle actually holds that curve."
    ),
    "reward": (
        "How the controller is scored -- what it is trying to get better at.\n\n"
        "v5.0       = fixed targets. A stop is scored against absolute\n"
        "             numbers that were hand-picked for dry tarmac at 60 mph.\n"
        "             On a wet road a genuinely good stop scores badly.\n\n"
        "normalized = scored against THIS car on THIS surface. 0 means 'no\n"
        "             better than locking the wheels', 1 means 'as good as the\n"
        "             car's own factory ABS', above 1 means better than\n"
        "             factory. Comparable across any surface or speed.\n\n"
        "Use normalized. It needs baselines measured first."
    ),
    "run_name": (
        "A folder name for this attempt, under runs\\.\n\n"
        "Everything from the run lands there: the log, the trained model, and\n"
        "the episode-by-episode results. Use a new name for each experiment,\n"
        "or you will not be able to tell two results apart later."
    ),
    "total_steps": (
        "How long to train, counted in physics steps (200 per second).\n\n"
        "500,000 is roughly a night's run and about 450 braking attempts.\n"
        "Below ~50,000 the controller has barely started learning.\n\n"
        "You can stop early at any point with GRACEFUL STOP -- the model is\n"
        "saved on the way out, so a long number is not a commitment."
    ),
    "resume": (
        "Carry on training a model you already have, instead of starting from\n"
        "scratch.\n\n"
        "Pick the final.zip from an earlier run. Useful for continuing a run\n"
        "you stopped, or for fine-tuning a good model on a harder setting."
    ),
    "algo": (
        "Which learning method to use.\n\n"
        "SAC = learns from a stored memory of past attempts. Gets results from\n"
        "      far fewer attempts, which matters here because every attempt\n"
        "      needs a real ~70-second stop in the game. Start with this.\n\n"
        "PPO = learns only from recent attempts. Steadier, but needs many more\n"
        "      of them, so it is slower in wall-clock time for this project."
    ),
    "fast_calibration": (
        "Measure calibration stops free-running under a physics speed factor,\n"
        "instead of stepping the simulation one tick at a time.\n\n"
        "About 11x faster per stop (~4s instead of ~35s). Measured head to\n"
        "head on 2026-08-30, interleaved, 5 reps per regime, 60 mph, full\n"
        "pedal:\n"
        "   deterministic 1.0135      live 4x  1.0214\n"
        "   live 10x      1.0091      live 25x 1.0066\n\n"
        "All within 0.8%, which is smaller than any single regime's own\n"
        "run-to-run spread (~0.02) -- so they are indistinguishable there.\n\n"
        "That was ONE configuration though, not a proof for all of them, so\n"
        "every row records how it was measured and you are warned before\n"
        "mixing regimes in one table.\n\n"
        "Affects calibration only. Training is untouched."
    ),
    "speed_factor": (
        "How much faster than real time to run physics during calibration.\n\n"
        "10 is the default. Higher numbers are accepted but buy nothing: the\n"
        "engine caps out near 5x real time on this machine (measured), and\n"
        "what remains is the acceleration run-up, not the stop. 4x, 10x and\n"
        "25x all measured about 4s per stop.\n\n"
        "Only used when 'Fast calibration' is ticked."
    ),
    "net_arch": (
        "The size of the controller's 'brain' -- how many layers of how many\n"
        "units.\n\n"
        "3x256 = three layers of 256 units (the default, and what every\n"
        "        result in this project so far used)\n"
        "22x128 = twenty-two layers of 128\n"
        "512,256,128 = three layers of decreasing width\n\n"
        "Bigger is not better here. Braking is a fairly simple reaction --\n"
        "wheel speeds in, brake pressure out -- so a deep network mostly adds\n"
        "training time and a slower controller, without learning anything the\n"
        "small one could not.\n\n"
        "There is a hard reason to stay modest: the trained network is\n"
        "exported into a Lua controller that runs it BY HAND every 0.5 ms\n"
        "inside the game. A network too slow to finish in time just misses\n"
        "ticks, with nothing logged. Capped at 24 layers and 2048 wide.\n\n"
        "Changing this makes results incomparable to earlier runs, and a\n"
        "resumed run must match the shape its checkpoint was trained with."
    ),
    "vehicle_pc": (
        "Which car and configuration to train on.\n\n"
        "Pick the model, then a factory trim, or 'Custom' for your own saved\n"
        "configurations of that model.\n\n"
        "Baselines are measured per car, so switching cars means calibrating\n"
        "again."
    ),
}

# --- the learning knobs (shared) -------------------------------------------
_LR = (
    "Learning rate: how big a correction the controller makes after each\n"
    "lesson.\n\n"
    "Too high and it over-reacts to a single lucky or unlucky stop and never\n"
    "settles. Too low and it improves so slowly the run finishes first.\n\n"
    "0.0001 is the tuned value for this project -- it was lowered from 0.0003\n"
    "because the higher value made the car crash often after resuming a run.\n"
    "Leave it alone unless you have a reason."
)

SAC_HELP = {
    "lr": _LR,
    "buffer_size": (
        "How many past braking attempts to keep in memory and re-learn from.\n\n"
        "SAC does not just learn from the stop it has just done -- it keeps a\n"
        "library of old ones and revisits them, which is why it needs far\n"
        "fewer real attempts than PPO.\n\n"
        "Bigger remembers more but uses more RAM. 100,000 steps is about 90\n"
        "attempts' worth, which comfortably covers a night's run."
    ),
    "tau": (
        "How fast the controller's 'stable copy' catches up to the one being\n"
        "trained.\n\n"
        "SAC keeps a slow-moving copy of itself to compare against, so its own\n"
        "improvements do not chase their own tail. This is how quickly that\n"
        "copy updates.\n\n"
        "Small (0.005) = very smooth and stable. Larger = faster but can\n"
        "wobble. Rarely worth changing."
    ),
    "target_entropy": (
        "How much random experimentation to keep doing.\n\n"
        "Entropy here just means randomness. Without it the controller settles\n"
        "on the first thing that half-works -- for this project that is\n"
        "'stand on the brakes and lock the wheels' -- and never discovers\n"
        "anything better.\n\n"
        "More negative = less experimenting, more sticking to what it knows.\n"
        "-2.0 matches the number of things it controls (front and rear)."
    ),
    "learning_starts": (
        "How many steps to just mess about randomly before learning starts.\n\n"
        "It needs some experience in memory before drawing conclusions.\n"
        "5,000 steps is about 5 braking attempts.\n\n"
        "Setting this ABOVE your total steps means it never learns at all --\n"
        "useful for a deliberate random-behaviour control run, and a silent\n"
        "waste of a night otherwise."
    ),
    "train_freq": (
        "How often to stop and learn, in steps.\n\n"
        "2 = review after every 2 physics steps. Lower learns more per\n"
        "attempt but runs slower; higher is faster but wastes experience.\n\n"
        "The game is the slow part here, not the learning, so low is fine."
    ),
}

PPO_HELP = {
    "lr": _LR,
    "n_steps": (
        "How much experience to gather before each learning session.\n\n"
        "PPO collects a batch of driving, learns from it, then throws it away\n"
        "and collects more. 2048 steps is about 2 braking attempts.\n\n"
        "Bigger = steadier lessons, but longer between improvements."
    ),
    "batch_size": (
        "How much of that experience to look at in one go while learning.\n\n"
        "Bigger = smoother, more reliable updates and better GPU use.\n"
        "Must not exceed 'n steps'. 512 is the tuned value here."
    ),
    "n_epochs": (
        "How many times to re-read the same batch of experience before\n"
        "throwing it away.\n\n"
        "More squeezes more out of each attempt, which is valuable when\n"
        "attempts are expensive -- but too many and it over-fits to that\n"
        "batch and forgets how to generalise. 10 is standard."
    ),
    "clip_range": (
        "A safety limit on how much the controller may change in one lesson.\n\n"
        "This is PPO's whole trick: it refuses to move too far from what it\n"
        "was doing, so one weird batch cannot wreck a working controller.\n\n"
        "0.2 = at most a 20% shift. Rarely worth changing."
    ),
    "gae_lambda": (
        "How much credit to give earlier actions for how the stop finally\n"
        "turned out.\n\n"
        "Braking is a chain of decisions and only the end result is scored, so\n"
        "the controller has to work out which earlier choices deserve the\n"
        "credit. Higher (0.95) shares credit further back.\n\n"
        "Too high gets noisy, too low ignores early decisions that mattered."
    ),
    "ent_coef": (
        "How strongly to encourage experimenting rather than repeating what\n"
        "already works.\n\n"
        "Same idea as SAC's target entropy. Too low and it locks onto the\n"
        "brake-slamming habit early; too high and it never commits to\n"
        "anything. 0.005 is a mild nudge."
    ),
}


def help_for(algo, key):
    """Tooltip text for one field, or None when there is nothing written for
    it -- callers must treat a missing entry as 'no tooltip', never an error."""
    if key in RUN_HELP:
        return RUN_HELP[key]
    table = SAC_HELP if algo == "sac" else PPO_HELP
    return table.get(key)


class Tooltip:
    """Hover help for a widget. Plain tkinter -- no ttk tooltip exists.

    Shows after a short delay so sweeping the mouse across the form does not
    flash popups, and hides on leave, click, or when the widget goes away.
    Deliberately dumb: it owns one toplevel and destroys it rather than trying
    to reuse one, because a stale reused window that outlives its widget is the
    classic way these leak visible artefacts."""

    DELAY_MS = 450
    WRAP_PX = 460

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        self._after = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")
        widget.bind("<Destroy>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel()
        self._after = self.widget.after(self.DELAY_MS, self._show)

    def _cancel(self):
        if self._after is not None:
            try:
                self.widget.after_cancel(self._after)
            except Exception:
                pass
            self._after = None

    def _show(self):
        import tkinter as tk
        if self.tip is not None or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 12
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        except Exception:
            return                      # widget already gone
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)   # no title bar or border
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify="left",
                 background="#ffffe0", foreground="#000000",
                 relief="solid", borderwidth=1, wraplength=self.WRAP_PX,
                 font=("Segoe UI", 9), padx=8, pady=6).pack()

    def _hide(self, _event=None):
        self._cancel()
        if self.tip is not None:
            try:
                self.tip.destroy()
            except Exception:
                pass
            self.tip = None


def attach(widget, algo, key):
    """Give `widget` the hover help for `key`, if any is written."""
    text = help_for(algo, key)
    if text:
        Tooltip(widget, text)
    return widget
