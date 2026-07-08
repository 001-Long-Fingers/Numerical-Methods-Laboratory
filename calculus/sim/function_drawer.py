"""
function_grapher.py
===================
Enter a function, see it graphed. Scale is adjustable via sliders.
Requires function_parser.py in the same directory.

Run:
    python function_grapher.py
"""

import sys
import tkinter as tk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

sys.path.insert(0, ".")
from taylor import function_parser, function

# Palette

RP = {
    "base":    "#0a0a0a",
    "surface": "#111111",
    "overlay": "#1a1a1a",
    "muted":   "#555555",
    "subtle":  "#888888",
    "text":    "#f0e020",
    "love":    "#ff2d78",
    "gold":    "#ff6600",
    "rose":    "#c04239",
    "pine":    "#ff0000",
    "foam":    "#ff9900",
    "iris":    "#ffce64",
    "hl_low":  "#141414",
    "hl_med":  "#222222",
    "hl_high": "#2e2e2e",
}

FONT_MONO    = ("Lucida Console", 11)
FONT_MONO_SM = ("Lucida Console", 9)
FONT_HEAD    = ("Lucida Console", 13)
FONT_TITLE   = ("Lucida Console", 18)
FONT_LABEL   = ("Lucida Console", 10)

CURVE_COLOR  = "#e63946"   # red curve on white graph


# Safe vectorised eval

def _eval_array(parsed, xs):
    ys = np.empty_like(xs, dtype=float)
    for i, x in enumerate(xs):
        try:
            ys[i] = function(parsed, float(x))
        except Exception:
            ys[i] = np.nan
    return ys


# App

class GrapherApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Function Grapher")
        self.configure(bg=RP["base"])
        self.resizable(True, True)

        self._parsed   = None
        self._x_lo     = -10.0
        self._x_hi     =  10.0
        self._y_lo     = -10.0
        self._y_hi     =  10.0
        self._auto_y   = True   # auto-scale y by default

        self._build_ui()
        self._draw_empty()

    #build

    def _build_ui(self):
        # left strip (controls)
        left = tk.Frame(self, bg=RP["surface"], padx=14, pady=14, width=280)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        tk.Label(left, text="GRAPHER", font=FONT_TITLE,
                 fg=RP["pine"], bg=RP["surface"]).pack(anchor="w", pady=(0, 14))

        # function entry
        self._divider(left, "FUNCTION")

        tk.Label(left, text="y =", font=FONT_MONO,
                 fg=RP["foam"], bg=RP["surface"]).pack(anchor="w", pady=(6, 2))

        self._fn_var = tk.StringVar()
        self._fn_entry = tk.Entry(
            left, textvariable=self._fn_var, width=26,
            font=FONT_MONO, bg=RP["hl_high"], fg=RP["text"],
            insertbackground=RP["text"], relief=tk.FLAT,
            bd=0, highlightthickness=1,
            highlightcolor=RP["gold"],
            highlightbackground=RP["muted"],
        )
        self._fn_entry.pack(anchor="w", pady=(0, 2))
        self._fn_entry.bind("<Return>", lambda e: self._plot())

        # parsed expression echo
        self._parsed_lbl = tk.Label(
            left, text="", font=FONT_MONO_SM,
            fg=RP["muted"], bg=RP["surface"],
            wraplength=240, justify=tk.LEFT,
        )
        self._parsed_lbl.pack(anchor="w", pady=(0, 6))

        # error label
        self._err_lbl = tk.Label(
            left, text="", font=FONT_MONO_SM,
            fg=RP["love"], bg=RP["surface"],
            wraplength=240, justify=tk.LEFT,
        )
        self._err_lbl.pack(anchor="w")

        # plot button
        tk.Button(
            left, text="  PLOT  ", font=FONT_HEAD,
            fg=RP["base"], bg=RP["gold"],
            activebackground=RP["foam"], activeforeground=RP["base"],
            relief=tk.FLAT, bd=0, cursor="hand2",
            command=self._plot,
        ).pack(anchor="w", pady=(8, 14))

        # x range
        self._divider(left, "X  RANGE")

        self._xlo_var = tk.DoubleVar(value=-10.0)
        self._xhi_var = tk.DoubleVar(value=10.0)

        self._slider(left, "x min", self._xlo_var, -100, 0,   self._on_range)
        self._slider(left, "x max", self._xhi_var,   0, 100,  self._on_range)

        # y range
        self._divider(left, "Y  RANGE")

        self._auto_y_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            left, text="Auto-scale Y", variable=self._auto_y_var,
            font=FONT_LABEL, fg=RP["iris"], bg=RP["surface"],
            selectcolor=RP["overlay"],
            activebackground=RP["surface"],
            activeforeground=RP["text"],
            command=self._toggle_auto_y,
        ).pack(anchor="w", pady=(6, 4))

        self._ylo_var = tk.DoubleVar(value=-10.0)
        self._yhi_var = tk.DoubleVar(value=10.0)

        self._ylo_sl = self._slider(left, "y min", self._ylo_var, -200, 0,   self._on_range)
        self._yhi_sl = self._slider(left, "y max", self._yhi_var,   0, 200,  self._on_range)
        self._set_y_sliders_state(tk.DISABLED)

        # hint
        self._divider(left, "SYNTAX")
        hints = [
            "sinx^2 + log(cosx)/2",
            "x^3 - 2*x + e^x",
            "sin(x)*cos(x)",
            "sqrt(1 - x^2)",
            "tan(x)",
        ]
        for h in hints:
            lbl = tk.Label(left, text=h, font=FONT_MONO_SM,
                           fg=RP["subtle"], bg=RP["surface"], cursor="hand2")
            lbl.pack(anchor="w")
            lbl.bind("<Button-1>", lambda e, v=h: self._load_hint(v))

        # right: graph
        right = tk.Frame(self, bg=RP["base"])
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        self._fig = Figure(figsize=(7, 6), dpi=100, facecolor="white")
        self._ax  = self._fig.add_subplot(111)
        self._fig.subplots_adjust(left=0.09, right=0.97, top=0.94, bottom=0.09)

        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # widget helpers

    def _divider(self, parent, title):
        f = tk.Frame(parent, bg=RP["surface"])
        f.pack(fill=tk.X, pady=(10, 0))
        tk.Label(f, text=title, font=FONT_MONO_SM,
                 fg=RP["iris"], bg=RP["surface"]).pack(side=tk.LEFT)
        tk.Frame(f, height=1, bg=RP["muted"]).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0), pady=6)

    def _slider(self, parent, label, var, lo, hi, cmd):
        row = tk.Frame(parent, bg=RP["surface"])
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text=f"{label:6s}", font=FONT_MONO_SM,
                 fg=RP["foam"], bg=RP["surface"], width=7).pack(side=tk.LEFT)
        val_lbl = tk.Label(row, text=f"{var.get():>7.1f}", font=FONT_MONO_SM,
                           fg=RP["text"], bg=RP["surface"], width=7)
        val_lbl.pack(side=tk.RIGHT)

        sl = tk.Scale(
            parent, variable=var, from_=lo, to=hi,
            orient=tk.HORIZONTAL, resolution=0.5,
            bg=RP["surface"], fg=RP["subtle"],
            troughcolor=RP["hl_high"], activebackground=RP["gold"],
            highlightthickness=0, bd=0, showvalue=False,
            command=lambda v, lbl=val_lbl, dv=var: (
                lbl.config(text=f"{float(v):>7.1f}"), cmd()
            ),
        )
        sl.pack(fill=tk.X, pady=(0, 2))
        return sl

    def _set_y_sliders_state(self, state):
        self._ylo_sl.config(state=state)
        self._yhi_sl.config(state=state)

    # callbacks

    def _toggle_auto_y(self):
        self._auto_y = self._auto_y_var.get()
        self._set_y_sliders_state(tk.DISABLED if self._auto_y else tk.NORMAL)
        self._replot()

    def _on_range(self):
        self._replot()

    def _load_hint(self, text):
        self._fn_var.set(text)
        self._plot()

    def _plot(self):
        raw = self._fn_var.get().strip()
        if not raw:
            return
        self._err_lbl.config(text="")
        try:
            self._parsed = function_parser(raw)
            self._parsed_lbl.config(
                text=f"→ {self._parsed.expression}", fg=RP["muted"])
        except (SyntaxError, ValueError) as e:
            self._err_lbl.config(text=str(e))
            self._parsed_lbl.config(text="")
            self._parsed = None
            return
        self._replot()

    def _replot(self):
        if self._parsed is None:
            return

        x_lo = self._xlo_var.get()
        x_hi = self._xhi_var.get()
        if x_lo >= x_hi:
            x_hi = x_lo + 0.1

        xs = np.linspace(x_lo, x_hi, 800)
        ys = _eval_array(self._parsed, xs)

        ax = self._ax
        ax.clear()

        # axes style (white graph)
        ax.set_facecolor("white")
        ax.tick_params(colors="#555555", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#cccccc")
        ax.set_xlabel("x", color="#555555", fontsize=9)
        ax.set_ylabel("y", color="#555555", fontsize=9)
        ax.grid(True, color="#eeeeee", linewidth=0.6, zorder=0)
        ax.axhline(0, color="#bbbbbb", linewidth=0.9, zorder=1)
        ax.axvline(0, color="#bbbbbb", linewidth=0.9, zorder=1)

        # y limits
        if self._auto_y:
            finite = ys[np.isfinite(ys)]
            if len(finite) > 1:
                lo, hi = np.min(finite), np.max(finite)
                pad    = max((hi - lo) * 0.08, 0.5)
                ax.set_ylim(lo - pad, hi + pad)
            # else leave matplotlib default
        else:
            y_lo = self._ylo_var.get()
            y_hi = self._yhi_var.get()
            if y_lo < y_hi:
                ax.set_ylim(y_lo, y_hi)

        ax.set_xlim(x_lo, x_hi)

        # plot curve — split at discontinuities
        # mask nan/inf so matplotlib doesn't bridge gaps (e.g. tan asymptotes)
        mask  = np.isfinite(ys)
        # also clip to y-window if manual, to avoid huge spikes
        if not self._auto_y:
            y_lo = self._ylo_var.get()
            y_hi = self._yhi_var.get()
            mask &= (ys >= y_lo - 1) & (ys <= y_hi + 1)

        xs_plot = np.where(mask, xs, np.nan)
        ys_plot = np.where(mask, ys, np.nan)

        expr = self._parsed.expression
        lbl  = f"y = {expr[:48]}{'…' if len(expr)>48 else ''}"
        ax.plot(xs_plot, ys_plot, color=CURVE_COLOR, linewidth=1.8,
                label=lbl, zorder=3)

        ax.set_title(lbl, color="#333333", fontsize=9, pad=6)
        self._canvas.draw()

    # empty state

    def _draw_empty(self):
        ax = self._ax
        ax.set_facecolor("white")
        ax.tick_params(colors="#555555", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#cccccc")
        ax.set_xlabel("x", color="#555555", fontsize=9)
        ax.set_ylabel("y", color="#555555", fontsize=9)
        ax.grid(True, color="#eeeeee", linewidth=0.6)
        ax.axhline(0, color="#bbbbbb", linewidth=0.9)
        ax.axvline(0, color="#bbbbbb", linewidth=0.9)
        ax.set_title("Enter a function and press PLOT  (or press Enter)",
                     color="#999999", fontsize=9)
        self._canvas.draw()



if __name__ == "__main__":
    app = GrapherApp()
    app.mainloop()