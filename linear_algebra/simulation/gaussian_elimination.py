import numpy as np  # pyright: ignore[reportMissingImports]
import tkinter as tk
from tkinter import ttk, messagebox
import time

# Neon Retro VHS palette preset
RP = {
    "base":      "#0a0a0a",
    "surface":   "#111111",
    "overlay":   "#1a1a1a",
    "muted":     "#555555",
    "subtle":    "#888888",
    "text":      "#f0e020",   # yellow – primary text
    "love":      "#ff2d78",   # hot pink – pivot rows
    "gold":      "#ff6600",   # neon orange – elimination highlight
    "rose":      "#c04239",   # neon red – result
    "pine":      "#ff0000",   # neon scarlet – header
    "foam":      "#ff9900",   # neon flame color – accent labels
    "iris":      "#ffce64",   # neon yellow – step titles
    "hl_low":    "#141414",
    "hl_med":    "#222222",
    "hl_high":   "#2e2e2e",
}

FONT_MONO = ("Lucida Console", 11)
FONT_MONO_SM = ("Lucida Consol", 9)
FONT_HEAD = ("Lucida Consol", 13 )
FONT_TITLE = ("Lucida Consol", 20 )
FONT_LABEL = ("Lucida Consol", 10)

# core algorithm (Same as prototype)

def solve_system_of_equations(A, b):
    try:
        solution = np.linalg.solve(A, b)
        return solution
    except np.linalg.LinAlgError:
        return "System has no unique solutions"


def gaussian_elimination(A, b):
    n = len(A)
    augmented = np.column_stack((A, b)).astype(float)
    steps = []

    def snap(label, highlight_rows=None, highlight_col=None, color=None):
        steps.append({
            "label": label,
            "matrix": augmented.copy(),
            "highlight_rows": highlight_rows or [],
            "highlight_col": highlight_col,
            "color": color or RP["foam"],
        })

    snap("Initial augmented matrix [A | b]")

    # Forward elimination
    for i in range(n):
        # Partial pivoting
        pivot_row = i
        max_val = abs(augmented[i][i])
        for j in range(i + 1, n):
            if abs(augmented[j][i]) > max_val:
                max_val = abs(augmented[j][i])
                pivot_row = j

        if pivot_row != i:
            augmented[i], augmented[pivot_row] = (
                augmented[pivot_row].copy(),
                augmented[i].copy(),
            )
            snap(
                f"Swap R{i+1} ↔ R{pivot_row+1}  (partial pivoting)",
                highlight_rows=[i, pivot_row],
                color=RP["gold"],
            )

        if abs(augmented[i][i]) < 1e-12:
            steps.append({"label": "SINGULAR: no unique solution", "matrix": augmented.copy(),
                          "highlight_rows": [i], "highlight_col": i, "color": RP["love"]})
            return "System has no unique solutions", steps

        # Normalize pivot row
        pivot = augmented[i][i]
        augmented[i] = augmented[i] / pivot
        snap(
            f"R{i+1}  ←  R{i+1} / {pivot:.4g}   (normalize pivot)",
            highlight_rows=[i],
            highlight_col=i,
            color=RP["love"],
        )

        # Eliminate below pivot
        for k in range(i + 1, n):
            factor = augmented[k][i]
            if abs(factor) > 1e-12:
                augmented[k] = augmented[k] - factor * augmented[i]
                snap(
                    f"R{k+1}  ←  R{k+1} − ({factor:.4g})·R{i+1}",
                    highlight_rows=[k, i],
                    highlight_col=i,
                    color=RP["iris"],
                )

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = augmented[i][-1]
        for j in range(i + 1, n):
            x[i] -= augmented[i][j] * x[j]
        snap(
            f"Back-substitute: x{i+1} = {x[i]:.6g}",
            highlight_rows=[i],
            color=RP["rose"],
        )

    snap("Row-echelon form achieved  ✓", color=RP["foam"])
    return x, steps


# Default example
A = np.array([[3, 2, -1], [4, 2, 0], [0, -2, 3]])
b = np.array([2, 8, 9])


# GUI Preset

class GaussApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Gaussian Elimination  ·  VHS")
        self.configure(bg=RP["base"])
        self.resizable(True, True)
        self.geometry("1100x740")
        self.minsize(900, 620)

        # State
        self._steps = []
        self._step_idx = 0
        self._playing = False
        self._after_id = None

        self._build_ui()
        self._populate_defaults()
        self._run()

    # Layout

    def _build_ui(self):
        # ── Top bar ──
        top = tk.Frame(self, bg=RP["surface"], pady=10)
        top.pack(fill="x")

        tk.Label(top, text="1.10 Gaussian Elimination", font=FONT_TITLE,
                 bg=RP["surface"], fg=RP["pine"]).pack(side="left", padx=20)

        tk.Label(top, text="Augmented matrix  ·  elementary row operations",
                 font=FONT_MONO_SM, bg=RP["surface"], fg=RP["muted"]).pack(side="left")

        # Main body
        body = tk.Frame(self, bg=RP["base"])
        body.pack(fill="both", expand=True, padx=0, pady=0)

        # Left panel: input
        left = tk.Frame(body, bg=RP["surface"], width=310)
        left.pack(side="left", fill="y", padx=0, pady=0)
        left.pack_propagate(False)
        self._build_input_panel(left)

        # Right panel: steps + matrix display
        right = tk.Frame(body, bg=RP["base"])
        right.pack(side="left", fill="both", expand=True)
        self._build_right_panel(right)

        # ── Bottom bar ──
        self._build_bottom_bar()

    def _build_input_panel(self, parent):
        tk.Label(parent, text="System  AX = b", font=FONT_HEAD,
                 bg=RP["surface"], fg=RP["pine"]).pack(pady=(18, 4), padx=16, anchor="w")

        # Dimension selector
        dim_row = tk.Frame(parent, bg=RP["surface"])
        dim_row.pack(fill="x", padx=16, pady=(0, 10))
        tk.Label(dim_row, text="Dimension:", font=FONT_LABEL,
                 bg=RP["surface"], fg=RP["subtle"]).pack(side="left")
        self._dim_var = tk.IntVar(value=3)
        self._dim_spin = tk.Spinbox(
            dim_row, from_=2, to=10, textvariable=self._dim_var,
            command=self._on_dim_change, font=FONT_MONO,
            bg=RP["overlay"], fg=RP["text"], buttonbackground=RP["hl_med"],
            insertbackground=RP["text"], relief="flat",
            highlightthickness=1, highlightcolor=RP["pine"],
            highlightbackground=RP["hl_med"], width=4, justify="center"
        )
        self._dim_spin.pack(side="left", padx=6)
        self._dim_var.trace_add("write", lambda *_: self.after(100, self._on_dim_change))

        text_cfg = dict(
            font=FONT_MONO, bg=RP["overlay"], fg=RP["text"],
            insertbackground=RP["text"], relief="flat",
            highlightthickness=1, highlightcolor=RP["pine"],
            highlightbackground=RP["hl_med"],
            selectbackground=RP["hl_high"], selectforeground=RP["pine"],
            wrap="none", padx=6, pady=4
        )

        # Matrix A text box
        tk.Label(parent, text="Matrix A  (one row per line)",
                 font=FONT_LABEL, bg=RP["surface"], fg=RP["iris"]).pack(anchor="w", padx=16)
        self._A_text = tk.Text(parent, height=6, **text_cfg)
        self._A_text.pack(fill="x", padx=16, pady=(2, 8))

        # Vector b text box
        tk.Label(parent, text="Vector b  (one value per line)",
                 font=FONT_LABEL, bg=RP["surface"], fg=RP["iris"]).pack(anchor="w", padx=16)
        self._b_text = tk.Text(parent, height=6, **text_cfg)
        self._b_text.pack(fill="x", padx=16, pady=(2, 8))

        # Run button
        run_btn = tk.Button(parent, text="▶  Solve & Animate",
                            font=FONT_MONO, bg=RP["pine"], fg=RP["base"],
                            activebackground=RP["foam"], activeforeground=RP["base"],
                            relief="flat", bd=0, padx=14, pady=8,
                            cursor="hand2", command=self._run)
        run_btn.pack(fill="x", padx=16, pady=(4, 6))

        # np.linalg result label
        tk.Label(parent, text="NumPy reference answer:",
                 font=FONT_MONO_SM, bg=RP["surface"], fg=RP["muted"]).pack(anchor="w", padx=16, pady=(10, 0))
        self._ref_var = tk.StringVar(value="—")
        tk.Label(parent, textvariable=self._ref_var, font=FONT_MONO_SM,
                 bg=RP["surface"], fg=RP["rose"], wraplength=260,
                 justify="left").pack(anchor="w", padx=20)

        # Solution label
        tk.Label(parent, text="Elimination answer:",
                 font=FONT_MONO_SM, bg=RP["surface"], fg=RP["muted"]).pack(anchor="w", padx=16, pady=(8, 0))
        self._sol_var = tk.StringVar(value="—")
        tk.Label(parent, textvariable=self._sol_var, font=FONT_MONO_SM,
                 bg=RP["surface"], fg=RP["text"], wraplength=260,
                 justify="left").pack(anchor="w", padx=20)

    def _on_dim_change(self):
        try:
            n = self._dim_var.get()
            if n < 2:
                return
        except tk.TclError:
            return
        self._populate_defaults()
        self._step_idx = 0
        self._steps = []
        self._matrix_canvas.delete("all")
        self._step_label_var.set("Enter values and press Solve")
        self._counter_var.set("")
        self._sol_var.set("—")
        self._ref_var.set("—")

    def _build_right_panel(self, parent):
        # Step title
        self._step_label_var = tk.StringVar(value="Enter values and press  ▶  Solve & Animate")
        step_lbl = tk.Label(parent, textvariable=self._step_label_var,
                            font=FONT_HEAD, bg=RP["base"], fg=RP["iris"],
                            anchor="w", pady=10)
        step_lbl.pack(fill="x", padx=20)

        # Matrix canvas
        canvas_frame = tk.Frame(parent, bg=RP["overlay"],
                                highlightthickness=1, highlightbackground=RP["pine"])
        canvas_frame.pack(fill="both", expand=True, padx=20, pady=(0, 8))
        self._matrix_canvas = tk.Canvas(canvas_frame, bg=RP["overlay"],
                                        highlightthickness=0)
        self._matrix_canvas.pack(fill="both", expand=True, padx=10, pady=10)

        # Step log
        tk.Label(parent, text="Step log", font=FONT_MONO_SM,
                 bg=RP["base"], fg=RP["muted"]).pack(anchor="w", padx=22)
        log_frame = tk.Frame(parent, bg=RP["surface"],
                             highlightthickness=1, highlightbackground=RP["hl_med"])
        log_frame.pack(fill="x", padx=20, pady=(2, 0))
        self._log_text = tk.Text(log_frame, height=5, font=FONT_MONO_SM,
                                 bg=RP["surface"], fg=RP["subtle"],
                                 insertbackground=RP["text"], relief="flat",
                                 state="disabled", wrap="word")
        log_scroll = tk.Scrollbar(log_frame, command=self._log_text.yview,
                                  bg=RP["overlay"])
        self._log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side="right", fill="y")
        self._log_text.pack(fill="both", expand=True, padx=6, pady=4)

    def _build_bottom_bar(self):
        bar = tk.Frame(self, bg=RP["surface"], pady=8)
        bar.pack(fill="x", side="bottom")

        btn_cfg = dict(font=FONT_MONO, bg=RP["overlay"], fg=RP["text"],
                       activebackground=RP["hl_high"], activeforeground=RP["pine"],
                       relief="flat", bd=0, padx=12, pady=5, cursor="hand2")

        self._prev_btn = tk.Button(bar, text="◀  Prev", command=self._prev_step, **btn_cfg)
        self._prev_btn.pack(side="left", padx=(16, 4))

        self._play_btn = tk.Button(bar, text="⏵  Play", command=self._toggle_play, **btn_cfg)
        self._play_btn.pack(side="left", padx=4)

        self._next_btn = tk.Button(bar, text="Next  ▶", command=self._next_step, **btn_cfg)
        self._next_btn.pack(side="left", padx=4)

        # Speed
        tk.Label(bar, text="Speed:", font=FONT_MONO_SM,
                 bg=RP["surface"], fg=RP["muted"]).pack(side="left", padx=(20, 4))
        self._speed_var = tk.DoubleVar(value=0.7)
        speed_scale = tk.Scale(bar, from_=0.1, to=2.0, resolution=0.1,
                               orient="horizontal", variable=self._speed_var,
                               bg=RP["surface"], fg=RP["subtle"], troughcolor=RP["overlay"],
                               highlightthickness=0, font=FONT_MONO_SM, length=120,
                               showvalue=False)
        speed_scale.pack(side="left")

        self._counter_var = tk.StringVar(value="")
        tk.Label(bar, textvariable=self._counter_var, font=FONT_MONO_SM,
                 bg=RP["surface"], fg=RP["muted"]).pack(side="right", padx=20)

    # Populate defaults

    def _populate_defaults(self):
        defaults_A = {
            3: [[3, 2, -1], [4, 2, 0], [0, -2, 3]],
        }
        defaults_b = {
            3: [2, 8, 9],
        }
        n = self._dim_var.get()
        self._A_text.delete("1.0", "end")
        self._b_text.delete("1.0", "end")
        if n in defaults_A:
            for row in defaults_A[n]:
                self._A_text.insert("end", "  ".join(str(v) for v in row) + "\n")
            for val in defaults_b[n]:
                self._b_text.insert("end", str(val) + "\n")
        else:
            for _ in range(n):
                self._A_text.insert("end", "  ".join(["0"] * n) + "\n")
                self._b_text.insert("end", "0\n")

    # Read inputs
    def _read_inputs(self):
        try:
            n = self._dim_var.get()
            a_lines = [l for l in self._A_text.get("1.0", "end").splitlines() if l.strip()]
            b_lines = [l for l in self._b_text.get("1.0", "end").splitlines() if l.strip()]
            if len(a_lines) != n:
                messagebox.showerror("Input error", f"Matrix A must have {n} rows.")
                return None, None
            if len(b_lines) != n:
                messagebox.showerror("Input error", f"Vector b must have {n} values.")
                return None, None
            A = np.array([[float(v) for v in row.split()] for row in a_lines])
            b = np.array([float(v) for v in b_lines])
            if A.shape != (n, n):
                messagebox.showerror("Input error", f"Each row of A must have {n} values.")
                return None, None
            return A, b
        except ValueError:
            messagebox.showerror("Input error", "All entries must be valid numbers.")
            return None, None

    def _run(self):
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        self._playing = False
        self._play_btn.config(text="⏵  Play")

        A, b = self._read_inputs()
        if A is None:
            return

        ref = solve_system_of_equations(A.copy(), b.copy())
        if isinstance(ref, str):
            self._ref_var.set(ref)
        else:
            self._ref_var.set("  ".join(f"x{i+1}={v:.4g}" for i, v in enumerate(ref)))

        sol, self._steps = gaussian_elimination(A.copy(), b.copy())
        if isinstance(sol, str):
            self._sol_var.set(sol)
        else:
            self._sol_var.set("  ".join(f"x{i+1}={v:.4g}" for i, v in enumerate(sol)))

        self._log_clear()
        self._step_idx = 0
        self._draw_step(self._step_idx)

    def _prev_step(self):
        if self._steps and self._step_idx > 0:
            self._step_idx -= 1
            self._draw_step(self._step_idx)

    def _next_step(self):
        if self._steps and self._step_idx < len(self._steps) - 1:
            self._step_idx += 1
            self._draw_step(self._step_idx)

    def _toggle_play(self):
        if not self._steps:
            return
        self._playing = not self._playing
        if self._playing:
            self._play_btn.config(text="⏸  Pause")
            self._auto_advance()
        else:
            self._play_btn.config(text="⏵  Play")
            if self._after_id:
                self.after_cancel(self._after_id)
                self._after_id = None

    def _auto_advance(self):
        if not self._playing:
            return
        if self._step_idx < len(self._steps) - 1:
            self._step_idx += 1
            self._draw_step(self._step_idx)
            delay = int(1000 / self._speed_var.get())
            self._after_id = self.after(delay, self._auto_advance)
        else:
            self._playing = False
            self._play_btn.config(text="⏵  Play")

    # Drawing
    def _draw_step(self, idx):
        step = self._steps[idx]
        self._step_label_var.set(f"Step {idx + 1}/{len(self._steps)}  ·  {step['label']}")
        self._counter_var.set(f"step {idx + 1} / {len(self._steps)}")
        self._draw_matrix(step)
        self._log_append(idx, step["label"])

    def _draw_matrix(self, step):
        c = self._matrix_canvas
        c.delete("all")
        c.update_idletasks()
        W = c.winfo_width() or 600
        H = c.winfo_height() or 340

        mat = step["matrix"]
        rows, cols = mat.shape
        h_rows = step["highlight_rows"]
        h_col = step["highlight_col"]
        color = step["color"]

        CELL_W = min(90, (W - 80) // cols)
        CELL_H = min(48, (H - 60) // rows)
        total_w = CELL_W * cols
        total_h = CELL_H * rows
        ox = (W - total_w) // 2
        oy = (H - total_h) // 2

        # Draw bracket lines
        bk = RP["pine"]
        bw = 2
        lx, rx = ox - 14, ox + total_w + 14
        ty, by = oy - 4, oy + total_h + 4
        # Left bracket
        c.create_line(lx + 8, ty, lx, ty, lx, by, lx + 8, by, width=bw, fill=bk)
        # Right bracket (before augmented sep)
        c.create_line(rx - 8, ty, rx, ty, rx, by, rx - 8, by, width=bw, fill=bk)
        # Separator before last column (augmented)
        sep_x = ox + CELL_W * (cols - 1) - 4
        c.create_line(sep_x, ty, sep_x, by, width=1, fill=RP["hl_high"], dash=(4, 4))

        cell_font = FONT_MONO if rows <= 7 else FONT_MONO_SM

        for i in range(rows):
            for j in range(cols):
                x0 = ox + j * CELL_W
                y0 = oy + i * CELL_H
                x1 = x0 + CELL_W
                y1 = y0 + CELL_H

                # Background
                if i in h_rows and (h_col is None or j == h_col or j == cols - 1):
                    bg = color
                    fg = RP["base"]
                elif i in h_rows:
                    bg = RP["hl_low"]
                    fg = color
                else:
                    bg = RP["overlay"] if i % 2 == 0 else RP["hl_low"]
                    fg = RP["text"]

                c.create_rectangle(x0 + 1, y0 + 1, x1 - 1, y1 - 1,
                                   fill=bg, outline="")

                val = mat[i][j]
                txt = f"{val:+.3g}" if abs(val) < 1000 else f"{val:.2e}"
                c.create_text((x0 + x1) // 2, (y0 + y1) // 2,
                              text=txt, font=cell_font,
                              fill=fg, anchor="center")

        # Row labels
        for i in range(rows):
            y_mid = oy + i * CELL_H + CELL_H // 2
            clr = color if i in h_rows else RP["muted"]
            c.create_text(ox - 22, y_mid, text=f"R{i+1}",
                          font=FONT_MONO_SM, fill=clr, anchor="e")

        # Col labels
        for j in range(cols):
            x_mid = ox + j * CELL_W + CELL_W // 2
            lbl = f"x{j+1}" if j < cols - 1 else "b"
            c.create_text(x_mid, oy - 14, text=lbl,
                          font=FONT_MONO_SM, fill=RP["muted"], anchor="center")

    def _log_clear(self):
        self._log_text.config(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.config(state="disabled")

    def _log_append(self, idx, label):
        self._log_text.config(state="normal")
        entry = f"[{idx + 1:02d}] {label}\n"
        self._log_text.insert("end", entry)
        self._log_text.see("end")
        self._log_text.config(state="disabled")

#EntryPoint
if __name__ == "__main__":
    app = GaussApp()
    app.mainloop()
