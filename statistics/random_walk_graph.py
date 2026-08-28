import tkinter as tk
from tkinter import ttk, messagebox
import random
import time
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# house palette, shared with the other Numerical Methods Laboratory tools
RP = {
    "base": "#0a0a0a", "surface": "#111111", "overlay": "#1a1a1a",
    "muted": "#555555", "subtle": "#888888", "text": "#f0e020",
    "love": "#ff2d78", "gold": "#ff6600", "rose": "#c04239",
    "pine": "#ff0000", "foam": "#ff9900", "iris": "#ffce64",
    "hl_low": "#141414", "hl_med": "#222222", "hl_high": "#2e2e2e",
}

FONT_MONO = ("Lucida Console", 11)
FONT_MONO_SM = ("Lucida Console", 9)
FONT_HEAD = ("Lucida Console", 13)
FONT_TITLE = ("Lucida Console", 20)
FONT_LABEL = ("Lucida Console", 10)


def generate_walk(p, n, rng):
    # two-variable random walk: an independent +-1 draw for x and for y at every step
    # P(step = 1) = p, P(step = -1) = 1 - p, P(step = anything else) = 0
    x, y = [0], [0]  # walk starts at the origin
    for _ in range(n):
        dx = 1 if rng.random() < p else -1  # x-axis step
        dy = 1 if rng.random() < p else -1  # y-axis step, independent draw
        x.append(x[-1] + dx)
        y.append(y[-1] + dy)
    return x, y


class GraphPopup(tk.Toplevel):
    # a standalone popup window holding one walk's plotted, scaled graph
    def __init__(self, master, x, y, seed, p, n, popup_index):
        super().__init__(master)
        self.title(f"Random Walk Graph #{popup_index}  —  seed {seed}")
        self.configure(bg=RP["base"])
        self.geometry("760x760")
        self.minsize(520, 520)

        header = ttk.Label(
            self,
            text=f"p = {p}    N = {n}    seed = {seed}",
            style="Stat.TLabel",
        )
        header.pack(side=tk.TOP, anchor="w", padx=12, pady=(10, 4))

        fig = Figure(figsize=(6.5, 6.5), dpi=100, facecolor=RP["surface"])
        ax = fig.add_subplot(111)
        ax.set_facecolor("#ffffff")  # white graph surface, house convention

        # plot the steps as points on the x/y axes, joined step-to-step
        ax.plot(x, y, linewidth=1.2, color=RP["foam"], zorder=2)
        ax.scatter(x, y, s=14, color=RP["iris"], zorder=3, label="step")
        ax.plot(x[0], y[0], marker="o", markersize=9, color=RP["pine"], zorder=5, label="start")
        ax.plot(x[-1], y[-1], marker="*", markersize=15, color=RP["love"], zorder=5, label="end")

        # scale both axes to this walk's own bounding box, with padding, so the
        # full path is visible in the popup no matter how many steps were taken
        pad_x = max((max(x) - min(x)) * 0.1, 1)
        pad_y = max((max(y) - min(y)) * 0.1, 1)
        ax.set_xlim(min(x) - pad_x, max(x) + pad_x)
        ax.set_ylim(min(y) - pad_y, max(y) + pad_y)
        ax.set_aspect("equal", adjustable="box")

        ax.set_title(f"2D Random Walk  (N = {n} steps)", fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        canvas.draw()


class RandomWalkLauncher:
    # main window: only holds the experiment parameters and the popup trigger
    def __init__(self, root):
        self.root = root
        self.root.title("2D Random Walk — Graph Launcher")
        self.root.configure(bg=RP["base"])
        self.root.geometry("420x360")
        self.root.minsize(380, 320)

        self.rng = random.Random()  # dedicated RNG instance, reseeded per experiment
        self.popup_count = 0

        self._build_style()
        self._build_layout()

    # ------------------------------------------------------------------ style

    def _build_style(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TFrame", background=RP["base"])
        style.configure("Panel.TFrame", background=RP["surface"])

        style.configure(
            "TLabel", background=RP["surface"], foreground=RP["subtle"], font=FONT_LABEL
        )
        style.configure(
            "Head.TLabel", background=RP["surface"], foreground=RP["text"], font=FONT_HEAD
        )
        style.configure(
            "Title.TLabel", background=RP["base"], foreground=RP["text"], font=FONT_TITLE
        )
        style.configure(
            "Stat.TLabel", background=RP["surface"], foreground=RP["iris"], font=FONT_MONO_SM
        )

        style.configure(
            "TEntry",
            fieldbackground=RP["hl_low"],
            foreground=RP["text"],
            insertcolor=RP["text"],
            bordercolor=RP["hl_high"],
        )

        style.configure(
            "Regen.TButton",
            background=RP["love"],
            foreground=RP["base"],
            font=FONT_HEAD,
            borderwidth=0,
            padding=(14, 10),
        )
        style.map("Regen.TButton", background=[("active", RP["gold"])])

    # ----------------------------------------------------------------- layout

    def _build_layout(self):
        title = ttk.Label(self.root, text="RANDOM WALK  ·  GRAPH", style="Title.TLabel")
        title.pack(side=tk.TOP, anchor="w", padx=18, pady=(16, 8))

        panel = ttk.Frame(self.root, style="Panel.TFrame", padding=16)
        panel.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=18, pady=(0, 18))

        ttk.Label(panel, text="PARAMETERS", style="Head.TLabel").pack(anchor="w", pady=(0, 12))

        ttk.Label(panel, text="p   —  P(step = +1)").pack(anchor="w")
        self.p_var = tk.StringVar(value="0.5")
        ttk.Entry(panel, textvariable=self.p_var, width=14, font=FONT_MONO).pack(
            anchor="w", pady=(2, 12)
        )

        ttk.Label(panel, text="N   —  number of steps").pack(anchor="w")
        self.n_var = tk.StringVar(value="500")
        ttk.Entry(panel, textvariable=self.n_var, width=14, font=FONT_MONO).pack(
            anchor="w", pady=(2, 18)
        )

        gen_btn = ttk.Button(
            panel, text="GENERATE GRAPH", style="Regen.TButton", command=self._generate_graph
        )
        gen_btn.pack(anchor="w", fill=tk.X, pady=(0, 20))

        self.last_lbl = ttk.Label(panel, text="popups opened: 0", style="Stat.TLabel")
        self.last_lbl.pack(anchor="w", pady=2)

    # --------------------------------------------------------------- actions

    def _read_params(self):
        try:
            p = float(self.p_var.get())
        except ValueError:
            raise ValueError("p must be a number between 0 and 1.")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must lie in [0, 1].")

        try:
            n = int(self.n_var.get())
        except ValueError:
            raise ValueError("N must be a whole number.")
        if n <= 0:
            raise ValueError("N must be a positive integer.")

        return p, n

    def _generate_graph(self):
        try:
            p, n = self._read_params()
        except ValueError as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return

        # fresh seed per experiment, drawn from system entropy + high-res clock
        seed = random.SystemRandom().randint(0, 2**32 - 1) ^ int(time.time_ns() & 0xFFFFFFFF)
        self.rng.seed(seed)

        x, y = generate_walk(p, n, self.rng)

        self.popup_count += 1
        GraphPopup(self.root, x, y, seed, p, n, self.popup_count)
        self.last_lbl.config(text=f"popups opened: {self.popup_count}")


def main():
    root = tk.Tk()
    RandomWalkLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
