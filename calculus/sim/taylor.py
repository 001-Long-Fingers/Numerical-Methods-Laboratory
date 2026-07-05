"""
function_ui.py
==============
Interactive GUI for function_parser.py.

Layout
------
Left panel  : inputs (function, taylor terms, expansion center, x value)
              + results readout
Right panel : matplotlib graph (white background) showing f(x), T(x),
              vertical line at x, and neighbourhood window around x.

Run from the same directory as function_parser.py:
    python function_ui.py
"""

import sys
import math
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# import our parser from the same directory
sys.path.insert(0, ".")
from taylor_prototype import function_parser, function, taylor_series, compare, is_polynomial # pyright: ignore[reportMissingImports]

# Palette & fonts

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
FONT_TITLE   = ("Lucida Console", 20)
FONT_LABEL   = ("Lucida Console", 10)

# graph colours (on white background)
GRAPH_F      = "#e63946"   # red   – exact function
GRAPH_T      = "#1d6fa4"   # blue  – Taylor polynomial
GRAPH_XLINE  = "#f4a261"   # amber – vertical x marker


# Helper: safe vectorised evaluation

def _eval_safe(parsed, xs):
    """Evaluate parsed function over a numpy array, returning nan for failures."""
    ys = np.empty_like(xs)
    for i, x in enumerate(xs):
        try:
            ys[i] = function(parsed, float(x))
        except Exception:
            ys[i] = np.nan
    return ys

def _eval_taylor_safe(ts, xs):
    ys = np.empty_like(xs)
    for i, x in enumerate(xs):
        try:
            ys[i] = ts.evaluate(float(x))
        except Exception:
            ys[i] = np.nan
    return ys


# Main application

class FunctionApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Function Parser  +  Taylor Series")
        self.configure(bg=RP["base"])
        self.resizable(True, True)

        # state
        self._parsed = None
        self._ts     = None

        self._build_ui()

    # UI construction

    def _build_ui(self):
        # root is a single horizontal paned window
        pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg=RP["base"],
                              sashwidth=4, sashrelief=tk.FLAT)
        pane.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        left  = self._build_left(pane)
        right = self._build_right(pane)

        pane.add(left,  minsize=340)
        pane.add(right, minsize=480)

    def _lbl(self, parent, text, font=FONT_LABEL, fg=None, **kw):
        return tk.Label(parent, text=text, font=font,
                        fg=fg or RP["foam"], bg=RP["surface"], **kw)

    def _entry(self, parent, textvariable=None, width=28):
        e = tk.Entry(parent, textvariable=textvariable, width=width,
                     font=FONT_MONO, bg=RP["hl_high"], fg=RP["text"],
                     insertbackground=RP["text"], relief=tk.FLAT,
                     bd=0, highlightthickness=1,
                     highlightcolor=RP["gold"],
                     highlightbackground=RP["muted"])
        return e

    def _section(self, parent, title):
        """A labelled divider line."""
        f = tk.Frame(parent, bg=RP["surface"])
        tk.Label(f, text=f"  {title}  ", font=FONT_MONO_SM,
                 fg=RP["iris"], bg=RP["surface"]).pack(side=tk.LEFT)
        tk.Frame(f, height=1, bg=RP["muted"]).pack(side=tk.LEFT, fill=tk.X, expand=True)
        return f

    def _build_left(self, parent):
        frame = tk.Frame(parent, bg=RP["surface"], padx=14, pady=14)

        # title
        tk.Label(frame, text="FUNCTION PARSER", font=FONT_TITLE,
                 fg=RP["pine"], bg=RP["surface"]).pack(anchor="w", pady=(0, 10))

        # Function inputs
        self._section(frame, "FUNCTIONS").pack(fill=tk.X, pady=(6, 4))

        self._fn_vars = []
        for i in range(3):
            row = tk.Frame(frame, bg=RP["surface"])
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=f"f{i+1}(x) =", font=FONT_MONO,
                     fg=RP["foam"], bg=RP["surface"], width=8).pack(side=tk.LEFT)
            v = tk.StringVar()
            self._fn_vars.append(v)
            e = self._entry(row, textvariable=v, width=24)
            e.pack(side=tk.LEFT, padx=4)
            # placeholder hint
            hints = ["sinx^2 + log(cosx)/2", "x^3 - 2*x + e^x", "sqrt(x^2 + 1)"]
            e.insert(0, hints[i])
            e.config(fg=RP["subtle"])
            self._bind_placeholder(e, v, hints[i])

        # active function selector
        row = tk.Frame(frame, bg=RP["surface"])
        row.pack(fill=tk.X, pady=(6, 2))
        tk.Label(row, text="Active :", font=FONT_LABEL,
                 fg=RP["foam"], bg=RP["surface"]).pack(side=tk.LEFT)
        self._active_fn = tk.IntVar(value=1)
        for i in range(1, 4):
            tk.Radiobutton(row, text=f"f{i}", variable=self._active_fn, value=i,
                           font=FONT_LABEL, fg=RP["iris"], bg=RP["surface"],
                           selectcolor=RP["overlay"],
                           activebackground=RP["surface"],
                           activeforeground=RP["text"]).pack(side=tk.LEFT, padx=6)

        # Taylor parameters
        self._section(frame, "TAYLOR PARAMETERS").pack(fill=tk.X, pady=(10, 4))

        self._n_terms_var = tk.StringVar(value="8")
        self._center_var  = tk.StringVar(value="0")

        for label, var in [("Terms  :", self._n_terms_var),
                            ("Center :", self._center_var)]:
            row = tk.Frame(frame, bg=RP["surface"])
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=label, font=FONT_MONO,
                     fg=RP["foam"], bg=RP["surface"], width=8).pack(side=tk.LEFT)
            self._entry(row, textvariable=var, width=10).pack(side=tk.LEFT, padx=4)

        # Evaluate at x
        self._section(frame, "EVALUATE").pack(fill=tk.X, pady=(10, 4))

        row = tk.Frame(frame, bg=RP["surface"])
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="x      :", font=FONT_MONO,
                 fg=RP["foam"], bg=RP["surface"], width=8).pack(side=tk.LEFT)
        self._x_var = tk.StringVar(value="1.0")
        self._entry(row, textvariable=self._x_var, width=10).pack(side=tk.LEFT, padx=4)

        # compute button
        tk.Button(frame, text="  COMPUTE  ", font=FONT_HEAD,
                  fg=RP["base"], bg=RP["gold"],
                  activebackground=RP["foam"], activeforeground=RP["base"],
                  relief=tk.FLAT, bd=0, cursor="hand2",
                  command=self._compute).pack(pady=12, anchor="w")

        # Results readout
        self._section(frame, "RESULTS").pack(fill=tk.X, pady=(4, 4))

        self._result_text = tk.Text(frame, font=FONT_MONO_SM,
                                    bg=RP["hl_low"], fg=RP["text"],
                                    relief=tk.FLAT, bd=0,
                                    height=16, width=38,
                                    state=tk.DISABLED,
                                    wrap=tk.WORD,
                                    highlightthickness=0)
        self._result_text.pack(fill=tk.BOTH, expand=True, pady=4)

        # configure colour tags for the text widget
        self._result_text.tag_configure("head",  foreground=RP["iris"],  font=FONT_MONO_SM)
        self._result_text.tag_configure("ok",    foreground=RP["text"],  font=FONT_MONO_SM)
        self._result_text.tag_configure("val",   foreground=RP["gold"],  font=FONT_MONO_SM)
        self._result_text.tag_configure("err",   foreground=RP["love"],  font=FONT_MONO_SM)
        self._result_text.tag_configure("poly",  foreground=RP["foam"],  font=FONT_MONO_SM)
        self._result_text.tag_configure("dim",   foreground=RP["muted"], font=FONT_MONO_SM)

        return frame

    def _build_right(self, parent):
        frame = tk.Frame(parent, bg=RP["base"])

        tk.Label(frame, text="GRAPH", font=FONT_HEAD,
                 fg=RP["iris"], bg=RP["base"]).pack(anchor="w", padx=8, pady=(8, 2))

        # legend strip
        leg = tk.Frame(frame, bg=RP["base"])
        leg.pack(anchor="w", padx=10, pady=2)
        for colour, label in [(GRAPH_F, "f(x)  exact"),
                               (GRAPH_T, "T(x)  taylor"),
                               (GRAPH_XLINE, "x  marker")]:
            tk.Label(leg, text="━━", font=FONT_LABEL,
                     fg=colour, bg=RP["base"]).pack(side=tk.LEFT)
            tk.Label(leg, text=f" {label}    ", font=FONT_LABEL,
                     fg=RP["subtle"], bg=RP["base"]).pack(side=tk.LEFT)

        # matplotlib figure — white background as requested
        self._fig = Figure(figsize=(6, 5), dpi=100, facecolor="white")
        self._ax  = self._fig.add_subplot(111)
        self._fig.subplots_adjust(left=0.1, right=0.97, top=0.93, bottom=0.1)

        self._canvas = FigureCanvasTkAgg(self._fig, master=frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self._draw_empty_graph()
        return frame

    # Placeholder behaviour for entry fields

    def _bind_placeholder(self, entry, var, hint):
        def on_focus_in(e):
            if var.get() == hint:
                entry.delete(0, tk.END)
                entry.config(fg=RP["text"])
        def on_focus_out(e):
            if not var.get():
                entry.insert(0, hint)
                entry.config(fg=RP["subtle"])
        entry.bind("<FocusIn>",  on_focus_in)
        entry.bind("<FocusOut>", on_focus_out)

    # Graph helper

    def _draw_empty_graph(self):
        ax = self._ax
        ax.set_facecolor("white")
        ax.tick_params(colors="#333333")
        ax.spines[:].set_color("#cccccc")
        ax.set_xlabel("x", color="#333333", fontsize=9)
        ax.set_ylabel("y", color="#333333", fontsize=9)
        ax.set_title("Enter a function and press COMPUTE", color="#888888", fontsize=9)
        ax.axhline(0, color="#cccccc", linewidth=0.8)
        ax.axvline(0, color="#cccccc", linewidth=0.8)
        self._canvas.draw()

    def _draw_graph(self, parsed, ts, x_val, center):
        ax = self._ax
        ax.clear()
        ax.set_facecolor("white")
        ax.tick_params(colors="#444444", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#cccccc")
        ax.set_xlabel("x", color="#444444", fontsize=9)
        ax.set_ylabel("y", color="#444444", fontsize=9)

        # neighbourhood window: ±2 units around x, but at least ±1
        half_w = max(abs(x_val - center) * 1.6, 2.0)
        x_lo   = x_val - half_w
        x_hi   = x_val + half_w
        xs     = np.linspace(x_lo, x_hi, 600)

        # evaluate both curves
        ys_f = _eval_safe(parsed, xs)
        ys_t = _eval_taylor_safe(ts, xs)

        # clip wild Taylor excursions so plot stays readable
        y_f_finite = ys_f[np.isfinite(ys_f)]
        y_t_finite = ys_t[np.isfinite(ys_t)]
        if len(y_f_finite):
            f_lo, f_hi = np.nanmin(y_f_finite), np.nanmax(y_f_finite)
            span = max(f_hi - f_lo, 1.0)
            clip_lo, clip_hi = f_lo - span, f_hi + span
            ys_t = np.clip(ys_t, clip_lo, clip_hi)

        ax.plot(xs, ys_f, color=GRAPH_F, linewidth=2.0, label="f(x)")
        ax.plot(xs, ys_t, color=GRAPH_T, linewidth=1.6,
                linestyle="--", label=f"T_{ts.n_terms}(x)")

        # vertical line at x
        ax.axvline(x=x_val, color=GRAPH_XLINE, linewidth=1.2,
                   linestyle=":", label=f"x = {x_val}")

        # dot at (x, f(x))
        try:
            y_dot = function(parsed, x_val)
            ax.plot(x_val, y_dot, "o", color=GRAPH_F, markersize=6, zorder=5)
            ax.plot(x_val, ts.evaluate(x_val), "s",
                    color=GRAPH_T, markersize=5, zorder=5)
        except Exception:
            pass

        # zero lines
        ax.axhline(0, color="#cccccc", linewidth=0.7)
        ax.axvline(0, color="#cccccc", linewidth=0.7)

        expr_label = parsed.expression[:40] + ("…" if len(parsed.expression) > 40 else "")
        ax.set_title(f"f(x) = {expr_label}   |   T_{ts.n_terms}  a={center}",
                     color="#333333", fontsize=8, pad=6)
        ax.legend(fontsize=8, loc="upper left",
                  facecolor="white", edgecolor="#cccccc", labelcolor="#333333")
        ax.grid(True, color="#eeeeee", linewidth=0.6)

        self._canvas.draw()

    # Compute

    def _write(self, text, tag="ok"):
        self._result_text.config(state=tk.NORMAL)
        self._result_text.insert(tk.END, text, tag)
        self._result_text.config(state=tk.DISABLED)

    def _clear_results(self):
        self._result_text.config(state=tk.NORMAL)
        self._result_text.delete("1.0", tk.END)
        self._result_text.config(state=tk.DISABLED)

    def _compute(self):
        self._clear_results()

        # read active function string
        idx      = self._active_fn.get() - 1
        fn_str   = self._fn_vars[idx].get().strip()
        hints    = ["sinx^2 + log(cosx)/2", "x^3 - 2*x + e^x", "sqrt(x^2 + 1)"]
        if fn_str == hints[idx] or not fn_str:
            self._write("[!] Enter a function in the active slot.\n", "err")
            return

        # parse
        try:
            parsed = function_parser(fn_str)
        except (SyntaxError, ValueError) as e:
            self._write(f"[PARSE ERROR]\n{e}\n", "err")
            return

        self._write("PARSED\n", "head")
        self._write(f"  expr  : {parsed.expression}\n", "ok")

        #taylor parameters
        try:
            n_terms = int(self._n_terms_var.get())
            center  = float(self._center_var.get())
        except ValueError:
            self._write("[!] Terms must be int, center must be float.\n", "err")
            return

        #compute taylor series
        try:
            ts = taylor_series(parsed, n_terms, center)
        except Exception as e:
            self._write(f"[TAYLOR ERROR]\n{e}\n", "err")
            return

        if ts.is_poly:
            self._write(f"\n[POLYNOMIAL  deg={ts.poly_degree}]\n", "poly")
            self._write("  Exact coefficients — no stencil used.\n", "dim")
            self._write("  Higher terms are exactly zero.\n", "dim")

        self._write(f"\nTAYLOR  (n={n_terms}, a={center})\n", "head")
        repr_str = repr(ts)
        # split after every + so it wraps nicely
        parts = repr_str.replace("T(x) = ", "").split(" + ")
        self._write("  T(x) =\n", "ok")
        for p in parts:
            self._write(f"    + {p}\n", "val")

        # ── evaluate at x ─────────────────────────────────────────────────
        try:
            x_val = float(self._x_var.get())
        except ValueError:
            self._write("[!] x must be a number.\n", "err")
            return

        try:
            result = compare(parsed, ts, x_val)
        except Exception as e:
            self._write(f"[EVAL ERROR]\n{e}\n", "err")
            return

        self._write(f"\nRESULTS  at x = {x_val}\n", "head")
        self._write(f"  {'─'*32}\n", "dim")
        self._write(f"  Exact  f(x)    = ", "ok")
        self._write(f"{result['exact']:.10f}\n", "val")
        self._write(f"  Taylor T_{n_terms}(x)  = ", "ok")
        self._write(f"{result['taylor']:.10f}", "val")
        if ts.is_poly:
            self._write("  [poly]\n", "poly")
        else:
            self._write("\n", "ok")
        self._write(f"  Abs error      = ", "ok")
        self._write(f"{result['abs_error']:.6e}", "err" if result['abs_error'] > 1e-3 else "val")
        if ts.is_poly:
            self._write("  (rounding)\n", "dim")
        else:
            self._write("\n", "ok")
        self._write(f"  Rel error      = ", "ok")
        self._write(f"{result['rel_error_%']:.4f} %\n", "val")
        self._write(f"  {'─'*32}\n", "dim")

        # store for redraw
        self._parsed = parsed
        self._ts     = ts

        # ── draw graph ────────────────────────────────────────────────────
        try:
            self._draw_graph(parsed, ts, x_val, center)
        except Exception as e:
            self._write(f"\n[GRAPH ERROR]\n{e}\n", "err")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = FunctionApp()
    app.mainloop()
=======
"""
function_ui.py
==============
Interactive GUI for function_parser.py.

Layout
------
Left panel  : inputs (function, taylor terms, expansion center, x value)
              + results readout
Right panel : matplotlib graph (white background) showing f(x), T(x),
              vertical line at x, and neighbourhood window around x.

Run from the same directory as function_parser.py:
    python function_ui.py
"""

import sys
import math
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# import our parser from the same directory
sys.path.insert(0, ".")
from taylor_prototype import function_parser, function, taylor_series, compare, is_polynomial # pyright: ignore[reportMissingImports]

# Palette & fonts

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
FONT_TITLE   = ("Lucida Console", 20)
FONT_LABEL   = ("Lucida Console", 10)

# graph colours (on white background)
GRAPH_F      = "#e63946"   # red   – exact function
GRAPH_T      = "#1d6fa4"   # blue  – Taylor polynomial
GRAPH_XLINE  = "#f4a261"   # amber – vertical x marker


# Helper: safe vectorised evaluation

def _eval_safe(parsed, xs):
    """Evaluate parsed function over a numpy array, returning nan for failures."""
    ys = np.empty_like(xs)
    for i, x in enumerate(xs):
        try:
            ys[i] = function(parsed, float(x))
        except Exception:
            ys[i] = np.nan
    return ys

def _eval_taylor_safe(ts, xs):
    ys = np.empty_like(xs)
    for i, x in enumerate(xs):
        try:
            ys[i] = ts.evaluate(float(x))
        except Exception:
            ys[i] = np.nan
    return ys


# Main application

class FunctionApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Function Parser  +  Taylor Series")
        self.configure(bg=RP["base"])
        self.resizable(True, True)

        # state
        self._parsed = None
        self._ts     = None

        self._build_ui()

    # UI construction

    def _build_ui(self):
        # root is a single horizontal paned window
        pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg=RP["base"],
                              sashwidth=4, sashrelief=tk.FLAT)
        pane.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        left  = self._build_left(pane)
        right = self._build_right(pane)

        pane.add(left,  minsize=340)
        pane.add(right, minsize=480)

    def _lbl(self, parent, text, font=FONT_LABEL, fg=None, **kw):
        return tk.Label(parent, text=text, font=font,
                        fg=fg or RP["foam"], bg=RP["surface"], **kw)

    def _entry(self, parent, textvariable=None, width=28):
        e = tk.Entry(parent, textvariable=textvariable, width=width,
                     font=FONT_MONO, bg=RP["hl_high"], fg=RP["text"],
                     insertbackground=RP["text"], relief=tk.FLAT,
                     bd=0, highlightthickness=1,
                     highlightcolor=RP["gold"],
                     highlightbackground=RP["muted"])
        return e

    def _section(self, parent, title):
        """A labelled divider line."""
        f = tk.Frame(parent, bg=RP["surface"])
        tk.Label(f, text=f"  {title}  ", font=FONT_MONO_SM,
                 fg=RP["iris"], bg=RP["surface"]).pack(side=tk.LEFT)
        tk.Frame(f, height=1, bg=RP["muted"]).pack(side=tk.LEFT, fill=tk.X, expand=True)
        return f

    def _build_left(self, parent):
        frame = tk.Frame(parent, bg=RP["surface"], padx=14, pady=14)

        # title
        tk.Label(frame, text="FUNCTION PARSER", font=FONT_TITLE,
                 fg=RP["pine"], bg=RP["surface"]).pack(anchor="w", pady=(0, 10))

        # Function inputs
        self._section(frame, "FUNCTIONS").pack(fill=tk.X, pady=(6, 4))

        self._fn_vars = []
        for i in range(3):
            row = tk.Frame(frame, bg=RP["surface"])
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=f"f{i+1}(x) =", font=FONT_MONO,
                     fg=RP["foam"], bg=RP["surface"], width=8).pack(side=tk.LEFT)
            v = tk.StringVar()
            self._fn_vars.append(v)
            e = self._entry(row, textvariable=v, width=24)
            e.pack(side=tk.LEFT, padx=4)
            # placeholder hint
            hints = ["sinx^2 + log(cosx)/2", "x^3 - 2*x + e^x", "sqrt(x^2 + 1)"]
            e.insert(0, hints[i])
            e.config(fg=RP["subtle"])
            self._bind_placeholder(e, v, hints[i])

        # active function selector
        row = tk.Frame(frame, bg=RP["surface"])
        row.pack(fill=tk.X, pady=(6, 2))
        tk.Label(row, text="Active :", font=FONT_LABEL,
                 fg=RP["foam"], bg=RP["surface"]).pack(side=tk.LEFT)
        self._active_fn = tk.IntVar(value=1)
        for i in range(1, 4):
            tk.Radiobutton(row, text=f"f{i}", variable=self._active_fn, value=i,
                           font=FONT_LABEL, fg=RP["iris"], bg=RP["surface"],
                           selectcolor=RP["overlay"],
                           activebackground=RP["surface"],
                           activeforeground=RP["text"]).pack(side=tk.LEFT, padx=6)

        # Taylor parameters
        self._section(frame, "TAYLOR PARAMETERS").pack(fill=tk.X, pady=(10, 4))

        self._n_terms_var = tk.StringVar(value="8")
        self._center_var  = tk.StringVar(value="0")

        for label, var in [("Terms  :", self._n_terms_var),
                            ("Center :", self._center_var)]:
            row = tk.Frame(frame, bg=RP["surface"])
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=label, font=FONT_MONO,
                     fg=RP["foam"], bg=RP["surface"], width=8).pack(side=tk.LEFT)
            self._entry(row, textvariable=var, width=10).pack(side=tk.LEFT, padx=4)

        # Evaluate at x
        self._section(frame, "EVALUATE").pack(fill=tk.X, pady=(10, 4))

        row = tk.Frame(frame, bg=RP["surface"])
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="x      :", font=FONT_MONO,
                 fg=RP["foam"], bg=RP["surface"], width=8).pack(side=tk.LEFT)
        self._x_var = tk.StringVar(value="1.0")
        self._entry(row, textvariable=self._x_var, width=10).pack(side=tk.LEFT, padx=4)

        # compute button
        tk.Button(frame, text="  COMPUTE  ", font=FONT_HEAD,
                  fg=RP["base"], bg=RP["gold"],
                  activebackground=RP["foam"], activeforeground=RP["base"],
                  relief=tk.FLAT, bd=0, cursor="hand2",
                  command=self._compute).pack(pady=12, anchor="w")

        # Results readout
        self._section(frame, "RESULTS").pack(fill=tk.X, pady=(4, 4))

        self._result_text = tk.Text(frame, font=FONT_MONO_SM,
                                    bg=RP["hl_low"], fg=RP["text"],
                                    relief=tk.FLAT, bd=0,
                                    height=16, width=38,
                                    state=tk.DISABLED,
                                    wrap=tk.WORD,
                                    highlightthickness=0)
        self._result_text.pack(fill=tk.BOTH, expand=True, pady=4)

        # configure colour tags for the text widget
        self._result_text.tag_configure("head",  foreground=RP["iris"],  font=FONT_MONO_SM)
        self._result_text.tag_configure("ok",    foreground=RP["text"],  font=FONT_MONO_SM)
        self._result_text.tag_configure("val",   foreground=RP["gold"],  font=FONT_MONO_SM)
        self._result_text.tag_configure("err",   foreground=RP["love"],  font=FONT_MONO_SM)
        self._result_text.tag_configure("poly",  foreground=RP["foam"],  font=FONT_MONO_SM)
        self._result_text.tag_configure("dim",   foreground=RP["muted"], font=FONT_MONO_SM)

        return frame

    def _build_right(self, parent):
        frame = tk.Frame(parent, bg=RP["base"])

        tk.Label(frame, text="GRAPH", font=FONT_HEAD,
                 fg=RP["iris"], bg=RP["base"]).pack(anchor="w", padx=8, pady=(8, 2))

        # legend strip
        leg = tk.Frame(frame, bg=RP["base"])
        leg.pack(anchor="w", padx=10, pady=2)
        for colour, label in [(GRAPH_F, "f(x)  exact"),
                               (GRAPH_T, "T(x)  taylor"),
                               (GRAPH_XLINE, "x  marker")]:
            tk.Label(leg, text="━━", font=FONT_LABEL,
                     fg=colour, bg=RP["base"]).pack(side=tk.LEFT)
            tk.Label(leg, text=f" {label}    ", font=FONT_LABEL,
                     fg=RP["subtle"], bg=RP["base"]).pack(side=tk.LEFT)

        # matplotlib figure — white background as requested
        self._fig = Figure(figsize=(6, 5), dpi=100, facecolor="white")
        self._ax  = self._fig.add_subplot(111)
        self._fig.subplots_adjust(left=0.1, right=0.97, top=0.93, bottom=0.1)

        self._canvas = FigureCanvasTkAgg(self._fig, master=frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self._draw_empty_graph()
        return frame

    # Placeholder behaviour for entry fields

    def _bind_placeholder(self, entry, var, hint):
        def on_focus_in(e):
            if var.get() == hint:
                entry.delete(0, tk.END)
                entry.config(fg=RP["text"])
        def on_focus_out(e):
            if not var.get():
                entry.insert(0, hint)
                entry.config(fg=RP["subtle"])
        entry.bind("<FocusIn>",  on_focus_in)
        entry.bind("<FocusOut>", on_focus_out)

    # Graph helpers

    def _draw_empty_graph(self):
        ax = self._ax
        ax.set_facecolor("white")
        ax.tick_params(colors="#333333")
        ax.spines[:].set_color("#cccccc")
        ax.set_xlabel("x", color="#333333", fontsize=9)
        ax.set_ylabel("y", color="#333333", fontsize=9)
        ax.set_title("Enter a function and press COMPUTE", color="#888888", fontsize=9)
        ax.axhline(0, color="#cccccc", linewidth=0.8)
        ax.axvline(0, color="#cccccc", linewidth=0.8)
        self._canvas.draw()

    def _draw_graph(self, parsed, ts, x_val, center):
        ax = self._ax
        ax.clear()
        ax.set_facecolor("white")
        ax.tick_params(colors="#444444", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#cccccc")
        ax.set_xlabel("x", color="#444444", fontsize=9)
        ax.set_ylabel("y", color="#444444", fontsize=9)

        # neighbourhood window: ±2 units around x, but at least ±1
        half_w = max(abs(x_val - center) * 1.6, 2.0)
        x_lo   = x_val - half_w
        x_hi   = x_val + half_w
        xs     = np.linspace(x_lo, x_hi, 600)

        # evaluate both curves
        ys_f = _eval_safe(parsed, xs)
        ys_t = _eval_taylor_safe(ts, xs)

        # clip wild Taylor excursions so plot stays readable
        y_f_finite = ys_f[np.isfinite(ys_f)]
        y_t_finite = ys_t[np.isfinite(ys_t)]
        if len(y_f_finite):
            f_lo, f_hi = np.nanmin(y_f_finite), np.nanmax(y_f_finite)
            span = max(f_hi - f_lo, 1.0)
            clip_lo, clip_hi = f_lo - span, f_hi + span
            ys_t = np.clip(ys_t, clip_lo, clip_hi)

        ax.plot(xs, ys_f, color=GRAPH_F, linewidth=2.0, label="f(x)")
        ax.plot(xs, ys_t, color=GRAPH_T, linewidth=1.6,
                linestyle="--", label=f"T_{ts.n_terms}(x)")

        # vertical line at x
        ax.axvline(x=x_val, color=GRAPH_XLINE, linewidth=1.2,
                   linestyle=":", label=f"x = {x_val}")

        # dot at (x, f(x))
        try:
            y_dot = function(parsed, x_val)
            ax.plot(x_val, y_dot, "o", color=GRAPH_F, markersize=6, zorder=5)
            ax.plot(x_val, ts.evaluate(x_val), "s",
                    color=GRAPH_T, markersize=5, zorder=5)
        except Exception:
            pass

        # zero lines
        ax.axhline(0, color="#cccccc", linewidth=0.7)
        ax.axvline(0, color="#cccccc", linewidth=0.7)

        expr_label = parsed.expression[:40] + ("…" if len(parsed.expression) > 40 else "")
        ax.set_title(f"f(x) = {expr_label}   |   T_{ts.n_terms}  a={center}",
                     color="#333333", fontsize=8, pad=6)
        ax.legend(fontsize=8, loc="upper left",
                  facecolor="white", edgecolor="#cccccc", labelcolor="#333333")
        ax.grid(True, color="#eeeeee", linewidth=0.6)

        self._canvas.draw()

    # Compute

    def _write(self, text, tag="ok"):
        self._result_text.config(state=tk.NORMAL)
        self._result_text.insert(tk.END, text, tag)
        self._result_text.config(state=tk.DISABLED)

    def _clear_results(self):
        self._result_text.config(state=tk.NORMAL)
        self._result_text.delete("1.0", tk.END)
        self._result_text.config(state=tk.DISABLED)

    def _compute(self):
        self._clear_results()

        # read active function string
        idx      = self._active_fn.get() - 1
        fn_str   = self._fn_vars[idx].get().strip()
        hints    = ["sinx^2 + log(cosx)/2", "x^3 - 2*x + e^x", "sqrt(x^2 + 1)"]
        if fn_str == hints[idx] or not fn_str:
            self._write("[!] Enter a function in the active slot.\n", "err")
            return

        # parse
        try:
            parsed = function_parser(fn_str)
        except (SyntaxError, ValueError) as e:
            self._write(f"[PARSE ERROR]\n{e}\n", "err")
            return

        self._write("PARSED\n", "head")
        self._write(f"  expr  : {parsed.expression}\n", "ok")

        # taylor parameters
        try:
            n_terms = int(self._n_terms_var.get())
            center  = float(self._center_var.get())
        except ValueError:
            self._write("[!] Terms must be int, center must be float.\n", "err")
            return

        # compute taylor series
        try:
            ts = taylor_series(parsed, n_terms, center)
        except Exception as e:
            self._write(f"[TAYLOR ERROR]\n{e}\n", "err")
            return

        if ts.is_poly:
            self._write(f"\n[POLYNOMIAL  deg={ts.poly_degree}]\n", "poly")
            self._write("  Exact coefficients — no stencil used.\n", "dim")
            self._write("  Higher terms are exactly zero.\n", "dim")

        self._write(f"\nTAYLOR  (n={n_terms}, a={center})\n", "head")
        repr_str = repr(ts)
        # split after every + so it wraps nicely
        parts = repr_str.replace("T(x) = ", "").split(" + ")
        self._write("  T(x) =\n", "ok")
        for p in parts:
            self._write(f"    + {p}\n", "val")

        # evaluate at x
        try:
            x_val = float(self._x_var.get())
        except ValueError:
            self._write("[!] x must be a number.\n", "err")
            return

        try:
            result = compare(parsed, ts, x_val)
        except Exception as e:
            self._write(f"[EVAL ERROR]\n{e}\n", "err")
            return

        self._write(f"\nRESULTS  at x = {x_val}\n", "head")
        self._write(f"  {'─'*32}\n", "dim")
        self._write(f"  Exact  f(x)    = ", "ok")
        self._write(f"{result['exact']:.10f}\n", "val")
        self._write(f"  Taylor T_{n_terms}(x)  = ", "ok")
        self._write(f"{result['taylor']:.10f}", "val")
        if ts.is_poly:
            self._write("  [poly]\n", "poly")
        else:
            self._write("\n", "ok")
        self._write(f"  Abs error      = ", "ok")
        self._write(f"{result['abs_error']:.6e}", "err" if result['abs_error'] > 1e-3 else "val")
        if ts.is_poly:
            self._write("  (rounding)\n", "dim")
        else:
            self._write("\n", "ok")
        self._write(f"  Rel error      = ", "ok")
        self._write(f"{result['rel_error_%']:.4f} %\n", "val")
        self._write(f"  {'─'*32}\n", "dim")

        # store for redraw
        self._parsed = parsed
        self._ts     = ts

        # draw graph
        try:
            self._draw_graph(parsed, ts, x_val, center)
        except Exception as e:
            self._write(f"\n[GRAPH ERROR]\n{e}\n", "err")


# Entry point

if __name__ == "__main__":
    app = FunctionApp()
    app.mainloop()
>>>>>>> dbc3abd0a2f05b2ffc5525d11bba3ed751c5a6a3
