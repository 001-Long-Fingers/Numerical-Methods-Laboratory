# Fourier Series Visualizer

A Tkinter desktop app that plots a function and its Fourier series approximation,
and lets you isolate and view any single harmonic term.

## Setup

```bash
pip install numpy scipy sympy matplotlib
python fourier_series_app.py
```

(Tkinter ships with standard Python on Windows/macOS; on Linux you may need
`sudo apt install python3-tk`.)

## How to use

1. **f(x)** — type the function using explicit parentheses around every
   argument and `**` or `^` for powers:
   - `sin(x)^2*log(cos(x))` → (sin x)² · ln(cos x)
   - `sin(x)**2 * log(cos(x))` → same
   - `x**2 + sin(2*x)`, `exp(-x**2)`, `sqrt(1+x**2)` all work.
   - "Implicit multiplication" like `2x` or `2 sin(x)` is allowed, but
     `sinx` (no parentheses/space) is **not** recognized as `sin(x)` —
     always parenthesize function arguments.

2. **Number of terms N** — how many harmonics (n = 1…N) to sum for the
   Fourier series approximation.

3. Click **Plot Function + Series** — this computes the Fourier
   coefficients (a₀, aₙ, bₙ) numerically over one period `[-π, π]` and
   draws both the original function and the N-term approximation.

4. **Plot individual term n** — type any harmonic index `n` (0 = the
   constant/DC term a₀/2, 1, 2, 3, …) up to the N you already computed,
   then click **Plot nth term** to overlay just that single
   `aₙcos(nx) + bₙsin(nx)` term on top of the function and the full
   series, so you can see each harmonic's individual contribution.

5. **Clear** resets the plot.

## Notes on the math

The Fourier series is built on the standard period `[-π, π]`:

```
f(x) ≈ a0/2 + Σ_{n=1}^{N} [ aₙ cos(nx) + bₙ sin(nx) ]
```

with coefficients computed by numerical integration (`scipy.integrate.quad`):

```
a0 = (1/π) ∫ f(x) dx           over [-π, π]
aₙ = (1/π) ∫ f(x) cos(nx) dx   over [-π, π]
bₙ = (1/π) ∫ f(x) sin(nx) dx   over [-π, π]
```

This works for arbitrary (even non-periodic) functions typed in — the
series is simply fit to whatever `f(x)` looks like on `[-π, π]`, which is
standard practice for Fourier analysis of a function over an interval.
Points where `f(x)` is undefined (e.g. `log(cos(x))` at `x = ±π/2`) are
skipped automatically during integration and plotting.
