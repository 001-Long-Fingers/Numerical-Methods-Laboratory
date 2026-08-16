"""
Fourier Series Visualizer -- Signal Module
Computational Physics Laboratory

A desktop tool: type a periodic function f(x), set the half-period L and the
number of terms N, and see the original signal plotted against its N-term
Fourier partial sum. Any single harmonic n can be isolated and drawn on the
same axes.

Dependencies:
    pip install numpy sympy matplotlib
"""

import tkinter as tk
from tkinter import ttk

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
)

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ----------------------------------------------------------------------------
# Palette / typography  (oscilloscope: dark housing, phosphor-green readouts)
# ----------------------------------------------------------------------------
SP = {
    "base":        "#05070a",
    "bezel":       "#0d1013",
    "bezel_hi":    "#171b1f",
    "screen":      "#030905",
    "grid":        "#39ff9e",
    "phosphor":    "#39ff9e",
    "phosphor_dim":"#1f8f5e",
    "text_hi":     "#eafff1",
    "text_mid":    "#7fa38f",
    "text_dim":    "#465a51",
    "danger":      "#ff5d5d",
    "sig_original":"#4ce3ff",
    "sig_sum":     "#ffb454",
    "sig_nth":     "#ff4fd8",
}

FONT_MONO   = ("Consolas", 10)
FONT_MONO_SM= ("Consolas", 9)
FONT_HEAD   = ("Consolas", 11, "bold")
FONT_TITLE  = ("Consolas", 17, "bold")
FONT_LABEL  = ("Consolas", 9)
FONT_SANS   = ("Segoe UI", 9)

INTEG_POINTS = 2001          # odd -> even number of intervals for Simpson
PLOT_POINTS  = 900
X_SYM = sp.symbols("x")

TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


# ----------------------------------------------------------------------------
# Engine: parsing, numerical integration, Fourier coefficients
# ----------------------------------------------------------------------------
class FourierEngine:
    """Owns parsing + numerical Fourier machinery. No UI knowledge."""

    def __init__(self):
        self.expr = None
        self.func = None          # numpy-vectorized callable
        self.latex = ""
        self.L = np.pi
        self.N = 8
        self.a0 = 0.0
        self.an = np.array([])
        self.bn = np.array([])
        self.had_undefined = False

    # -- parsing ---------------------------------------------------------
    def parse(self, expr_str):
        """Parse expr_str into a sympy expression and a numpy-callable.
        Raises on failure; caller handles the exception."""
        expr = parse_expr(expr_str, transformations=TRANSFORMS, local_dict={"x": X_SYM})
        func = sp.lambdify(X_SYM, expr, modules=["numpy"])
        self.expr = expr
        self.func = func
        self.latex = sp.latex(expr)
        return self.latex

    # -- safe vectorized evaluation --------------------------------------
    def _eval(self, x_arr):
        """Return (y_plot, y_calc, had_undefined).
        y_plot keeps NaN for undefined points (so matplotlib gaps the line).
        y_calc replaces NaN/Inf with 0.0 (for integration)."""
        with np.errstate(all="ignore"):
            y = np.asarray(self.func(x_arr), dtype=float)
            if y.ndim == 0:
                y = np.full_like(x_arr, float(y))
        finite = np.isfinite(y)
        had_undefined = not np.all(finite)
        y_plot = np.where(finite, y, np.nan)
        y_calc = np.where(finite, y, 0.0)
        return y_plot, y_calc, had_undefined

    @staticmethod
    def _simpson(y, dx):
        n = len(y) - 1
        if n % 2 == 1:
            y = y[:-1]
            n -= 1
        s = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2])
        return s * dx / 3.0

    # -- coefficient computation ------------------------------------------
    def compute(self, L, N):
        self.L, self.N = L, N
        x = np.linspace(-L, L, INTEG_POINTS)
        dx = x[1] - x[0]
        _, y_calc, had_undef = self._eval(x)
        self.had_undefined = had_undef

        self.a0 = (1.0 / L) * self._simpson(y_calc, dx)

        an = np.zeros(N)
        bn = np.zeros(N)
        for k in range(1, N + 1):
            ck = np.cos(k * np.pi * x / L)
            sk = np.sin(k * np.pi * x / L)
            an[k - 1] = (1.0 / L) * self._simpson(y_calc * ck, dx)
            bn[k - 1] = (1.0 / L) * self._simpson(y_calc * sk, dx)
        self.an, self.bn = an, bn
        return self.a0, an, bn, had_undef

    def single_harmonic(self, L, n):
        """Coefficients for one harmonic n, computed fresh (used when n > N)."""
        x = np.linspace(-L, L, INTEG_POINTS)
        dx = x[1] - x[0]
        _, y_calc, _ = self._eval(x)
        ck = np.cos(n * np.pi * x / L)
        sk = np.sin(n * np.pi * x / L)
        a = (1.0 / L) * self._simpson(y_calc * ck, dx)
        b = (1.0 / L) * self._simpson(y_calc * sk, dx)
        return a, b

    # -- sampling for plotting --------------------------------------------
    def sample_original(self, L, samples=PLOT_POINTS):
        x = np.linspace(-L, L, samples)
        y_plot, _, _ = self._eval(x)
        return x, y_plot

    def sample_partial_sum(self, L, N, a0, an, bn, samples=PLOT_POINTS):
        x = np.linspace(-L, L, samples)
        y = np.full_like(x, a0 / 2.0)
        for k in range(1, N + 1):
            y += an[k - 1] * np.cos(k * np.pi * x / L) + bn[k - 1] * np.sin(k * np.pi * x / L)
        return x, y

    def sample_isolated(self, L, n, a, b, samples=PLOT_POINTS):
        x = np.linspace(-L, L, samples)
        y = a * np.cos(n * np.pi * x / L) + b * np.sin(n * np.pi * x / L)
        return x, y


# ----------------------------------------------------------------------------
# UI: math preview strip (renders LaTeX via a small embedded Matplotlib figure)
# ----------------------------------------------------------------------------
class MathPreview(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=SP["screen"], highlightthickness=1,
                          highlightbackground=SP["bezel_hi"])
        self.fig = Figure(figsize=(3.0, 0.55), dpi=110, facecolor=SP["screen"])
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_facecolor(SP["screen"])
        self.ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.show_text("f(x) = ...", color=SP["text_dim"], math=False)

    def show_text(self, text, color, math=True):
        self.ax.clear()
        self.ax.axis("off")
        self.ax.set_facecolor(SP["screen"])
        payload = f"${text}$" if math else text
        try:
            self.ax.text(0.02, 0.5, payload, color=color, fontsize=13,
                         ha="left", va="center", transform=self.ax.transAxes)
        except Exception:
            self.ax.text(0.02, 0.5, text, color=color, fontsize=10,
                         family="monospace", ha="left", va="center",
                         transform=self.ax.transAxes)
        self.canvas.draw_idle()


# ----------------------------------------------------------------------------
# UI: oscilloscope plot panel
# ----------------------------------------------------------------------------
class ScopePanel(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=SP["bezel"])
        self.fig = Figure(figsize=(8.6, 5.2), dpi=100, facecolor=SP["bezel"])
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        self._style_axes()

    def _style_axes(self):
        ax = self.ax
        ax.set_facecolor(SP["screen"])
        ax.grid(True, color=SP["grid"], alpha=0.12, linewidth=0.8)
        ax.axhline(0, color=SP["grid"], alpha=0.30, linewidth=1.0)
        ax.axvline(0, color=SP["grid"], alpha=0.30, linewidth=1.0)
        for spine in ax.spines.values():
            spine.set_color(SP["bezel_hi"])
        ax.tick_params(colors=SP["text_mid"], labelsize=8)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily("monospace")

    def _glow_plot(self, x, y, color, base_lw=1.8):
        """Approximate an oscilloscope glow: stacked translucent strokes."""
        ax = self.ax
        ax.plot(x, y, color=color, linewidth=base_lw + 5, alpha=0.10, solid_capstyle="round")
        ax.plot(x, y, color=color, linewidth=base_lw + 2.5, alpha=0.18, solid_capstyle="round")
        ax.plot(x, y, color=color, linewidth=base_lw, alpha=1.0, solid_capstyle="round")

    def render(self, original, partial_sum, isolated, L):
        self.ax.clear()
        self._style_axes()

        handles = []
        x0, y0 = original
        self._glow_plot(x0, y0, SP["sig_original"])
        handles.append(("f(x), original", SP["sig_original"]))

        if partial_sum is not None:
            x1, y1 = partial_sum
            self._glow_plot(x1, y1, SP["sig_sum"], base_lw=1.6)
            handles.append(("partial sum, N terms", SP["sig_sum"]))

        if isolated is not None:
            x2, y2 = isolated
            self._glow_plot(x2, y2, SP["sig_nth"], base_lw=1.4)
            handles.append(("isolated nth term", SP["sig_nth"]))

        self.ax.set_xlim(-L, L)
        self.ax.set_xlabel("x", color=SP["text_mid"], fontsize=9, family="monospace")
        self.ax.set_ylabel("y", color=SP["text_mid"], fontsize=9, family="monospace")

        legend_lines = [matplotlib.lines.Line2D([0], [0], color=c, lw=2) for _, c in handles]
        legend_labels = [t for t, _ in handles]
        leg = self.ax.legend(legend_lines, legend_labels, loc="upper right",
                              facecolor=SP["bezel"], edgecolor=SP["bezel_hi"],
                              labelcolor=SP["text_mid"], fontsize=8, framealpha=0.9)
        for text in leg.get_texts():
            text.set_fontfamily("monospace")

        self.canvas.draw_idle()


class FourierApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fourier Series Visualizer -- Signal Module")
        self.geometry("1280x780")
        self.minsize(980, 620)
        self.configure(bg=SP["base"])

        self.engine = FourierEngine()

        self._build_header()

        body = tk.Frame(self, bg=SP["base"])
        body.pack(fill="both", expand=True, padx=24, pady=(0, 20))
        body.columnconfigure(0, weight=0, minsize=320)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self.controls = tk.Frame(body, bg=SP["base"])
        self.controls.grid(row=0, column=0, sticky="nsew", padx=(0, 16))

        self.scope = ScopePanel(body)
        self.scope.grid(row=0, column=1, sticky="nsew")

        self._build_function_panel()
        self._build_domain_panel()
        self._build_isolate_panel()
        self._build_warning_label()
        self._build_readout()

        self.plot_signal()

    # -- header ------------------------------------------------------------
    def _build_header(self):
        header = tk.Frame(self, bg=SP["base"])
        header.pack(fill="x", padx=24, pady=(20, 14))
        tk.Label(header, text="COMPUTATIONAL PHYSICS LAB \u2014 SIGNAL MODULE",
                  font=FONT_LABEL, fg=SP["phosphor_dim"], bg=SP["base"]).pack(anchor="w")
        tk.Label(header, text="Fourier Series Visualizer",
                  font=FONT_TITLE, fg=SP["text_hi"], bg=SP["base"]).pack(anchor="w", pady=(2, 0))
        tk.Label(header,
                  text="Type a periodic function, set the half-period and term count, and watch\n"
                       "the partial-sum reconstruction converge on the original signal.",
                  font=FONT_SANS, fg=SP["text_mid"], bg=SP["base"], justify="left").pack(anchor="w", pady=(4, 0))

    def _panel(self, title):
        outer = tk.Frame(self.controls, bg=SP["bezel"], highlightthickness=1,
                          highlightbackground=SP["bezel_hi"])
        outer.pack(fill="x", pady=(0, 14))
        inner = tk.Frame(outer, bg=SP["bezel"])
        inner.pack(fill="x", padx=14, pady=12)
        tk.Label(inner, text=title, font=FONT_LABEL, fg=SP["text_dim"], bg=SP["bezel"]
                  ).pack(anchor="w", pady=(0, 8))
        return inner

    def _entry(self, parent, textvariable, width=None):
        e = tk.Entry(parent, textvariable=textvariable, font=FONT_MONO,
                     bg=SP["screen"], fg=SP["phosphor"], insertbackground=SP["phosphor"],
                     relief="flat", highlightthickness=1,
                     highlightbackground=SP["bezel_hi"], highlightcolor=SP["phosphor"])
        if width:
            e.configure(width=width)
        return e

    def _button(self, parent, text, command, secondary=False):
        b = tk.Button(parent, text=text, command=command, font=(FONT_MONO_SM[0], 9, "bold"),
                      bg=(SP["bezel"] if secondary else SP["phosphor"]),
                      fg=(SP["sig_nth"] if secondary else "#04120a"),
                      activebackground=SP["bezel_hi"] if secondary else SP["phosphor_dim"],
                      activeforeground=(SP["sig_nth"] if secondary else "#04120a"),
                      relief="flat", bd=0, padx=10, pady=8, cursor="hand2",
                      highlightthickness=1,
                      highlightbackground=(SP["sig_nth"] if secondary else SP["phosphor"]))
        return b

    # -- function panel ------------------------------------------------------
    def _build_function_panel(self):
        panel = self._panel("Function \u2014 f(x)")
        self.fn_var = tk.StringVar(value="sin(x)**2 * log(cos(x))")
        entry = self._entry(panel, self.fn_var)
        entry.pack(fill="x")
        entry.bind("<KeyRelease>", lambda e: self.update_preview())
        entry.bind("<Return>", lambda e: self.plot_signal())

        tk.Label(panel, justify="left", wraplength=270,
                  text="Use ** or ^ for powers, * for multiplication, names sin cos tan "
                       "log sqrt exp Abs. log is natural log. Example: sin(x)**2 * log(cos(x)) "
                       "reads as sin x whole squared, times log of cos x.",
                  font=(FONT_SANS[0], 8), fg=SP["text_dim"], bg=SP["bezel"]
                  ).pack(anchor="w", pady=(8, 8), fill="x")

        self.preview = MathPreview(panel)
        self.preview.pack(fill="x")
        self.update_preview()

    def update_preview(self):
        expr_str = self.fn_var.get().strip()
        try:
            latex = self.engine.parse(expr_str) if expr_str else "..."
            self.preview.show_text(f"f(x) = {latex}", color=SP["text_hi"])
        except Exception as exc:
            self.preview.show_text(f"parse error: {exc}", color=SP["danger"], math=False)

    # -- domain / reconstruction panel ---------------------------------------
    def _build_domain_panel(self):
        panel = self._panel("Domain & Reconstruction")
        row = tk.Frame(panel, bg=SP["bezel"])
        row.pack(fill="x")
        row.columnconfigure(0, weight=1)
        row.columnconfigure(1, weight=1)

        col1 = tk.Frame(row, bg=SP["bezel"]); col1.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        col2 = tk.Frame(row, bg=SP["bezel"]); col2.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        tk.Label(col1, text="Half-period L", font=FONT_LABEL, fg=SP["text_mid"], bg=SP["bezel"]).pack(anchor="w")
        self.l_var = tk.StringVar(value=f"{np.pi:.8f}")
        self._entry(col1, self.l_var).pack(fill="x", pady=(4, 0))

        tk.Label(col2, text="Terms N (1-60)", font=FONT_LABEL, fg=SP["text_mid"], bg=SP["bezel"]).pack(anchor="w")
        self.n_var = tk.StringVar(value="8")
        self._entry(col2, self.n_var).pack(fill="x", pady=(4, 0))

        tk.Label(panel, text="Series is built on [-L, L], period 2L. Default L = \u03c0.",
                  font=(FONT_SANS[0], 8), fg=SP["text_dim"], bg=SP["bezel"]).pack(anchor="w", pady=(8, 10))

        self._button(panel, "PLOT SIGNAL", self.plot_signal).pack(fill="x")

    # -- isolate harmonic panel ----------------------------------------------
    def _build_isolate_panel(self):
        panel = self._panel("Isolate Harmonic")
        tk.Label(panel, text="n", font=FONT_LABEL, fg=SP["text_mid"], bg=SP["bezel"]).pack(anchor="w")
        self.iso_var = tk.StringVar(value="3")
        self._entry(panel, self.iso_var).pack(fill="x", pady=(4, 10))
        self._button(panel, "DRAW NTH TERM", self.draw_isolated, secondary=True).pack(fill="x")
        tk.Label(panel, text="Draws a_n cos(n\u03c0x/L) + b_n sin(n\u03c0x/L) alone, no DC offset.",
                  font=(FONT_SANS[0], 8), fg=SP["text_dim"], bg=SP["bezel"], wraplength=270,
                  justify="left").pack(anchor="w", pady=(8, 0))

    def _build_warning_label(self):
        self.warn_var = tk.StringVar(value="")
        self.warn_label = tk.Label(self.controls, textvariable=self.warn_var, font=(FONT_MONO_SM[0], 8),
                                     fg=SP["danger"], bg=SP["base"], wraplength=300, justify="left")
        self.warn_label.pack(fill="x", pady=(0, 6))

    # -- readout --------------------------------------------------------------
    def _build_readout(self):
        outer = tk.Frame(self, bg=SP["bezel"], highlightthickness=1, highlightbackground=SP["bezel_hi"])
        outer.pack(fill="x", padx=24, pady=(0, 20))
        self.readout_frame = tk.Frame(outer, bg=SP["bezel"])
        self.readout_frame.pack(fill="x", padx=14, pady=10)

    def _refresh_readout(self, cells):
        for w in self.readout_frame.winfo_children():
            w.destroy()
        for i, (label, value) in enumerate(cells):
            cell = tk.Frame(self.readout_frame, bg=SP["screen"], highlightthickness=1,
                             highlightbackground=SP["bezel_hi"])
            cell.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 8, 0))
            self.readout_frame.columnconfigure(i, weight=1)
            tk.Label(cell, text=label, font=(FONT_MONO_SM[0], 8), fg=SP["text_dim"], bg=SP["screen"]
                      ).pack(anchor="w", padx=10, pady=(6, 0))
            tk.Label(cell, text=value, font=(FONT_MONO[0], 12, "bold"), fg=SP["phosphor"], bg=SP["screen"]
                      ).pack(anchor="w", padx=10, pady=(0, 6))

    # -- actions --------------------------------------------------------------
    def _set_warning(self, msg):
        self.warn_var.set(msg or "")

    def plot_signal(self):
        expr_str = self.fn_var.get().strip()
        try:
            self.engine.parse(expr_str)
        except Exception as exc:
            self._set_warning(f"Could not parse f(x): {exc}")
            return

        try:
            L = float(sp.sympify(self.l_var.get()))
        except Exception:
            self._set_warning("Half-period L must be a positive number.")
            return
        if L <= 0:
            self._set_warning("Half-period L must be a positive number.")
            return

        try:
            N = max(1, min(60, int(float(self.n_var.get()))))
        except Exception:
            N = 8
        self.n_var.set(str(N))

        a0, an, bn, had_undef = self.engine.compute(L, N)

        original = self.engine.sample_original(L)
        partial = self.engine.sample_partial_sum(L, N, a0, an, bn)
        self.scope.render(original, partial, None, L)

        self._set_warning(
            "f(x) is undefined over part of [-L, L] (e.g. log of a non-positive value). "
            "Those points are gapped in the plot and treated as zero for the integrals."
            if had_undef else ""
        )

        self._last_partial = partial
        self._last_original = original
        self._last_isolated = None

        self._refresh_readout([
            ("a\u2080", f"{a0:.5f}"),
            ("N terms", str(N)),
            ("L", f"{L:.5f}"),
            ("a\u2081", f"{an[0]:.5f}" if N >= 1 else "--"),
            ("b\u2081", f"{bn[0]:.5f}" if N >= 1 else "--"),
        ])

    def draw_isolated(self):
        if self.engine.func is None:
            self._set_warning("Plot a signal first.")
            return
        try:
            n = max(1, int(float(self.iso_var.get())))
        except Exception:
            n = 1
        self.iso_var.set(str(n))

        if n <= self.engine.N:
            a, b = self.engine.an[n - 1], self.engine.bn[n - 1]
        else:
            a, b = self.engine.single_harmonic(self.engine.L, n)

        isolated = self.engine.sample_isolated(self.engine.L, n, a, b)
        self._last_isolated = isolated
        self.scope.render(self._last_original, self._last_partial, isolated, self.engine.L)

        self._refresh_readout([
            ("a\u2080", f"{self.engine.a0:.5f}"),
            ("N terms", str(self.engine.N)),
            ("L", f"{self.engine.L:.5f}"),
            (f"a_{n}", f"{a:.5f}"),
            (f"b_{n}", f"{b:.5f}"),
        ])


if __name__ == "__main__":
    app = FourierApp()
    app.mainloop()
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
