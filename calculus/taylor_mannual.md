# `function_parser.py` — Complete Explanation

A walkthrough of every component: what it does, why it exists, and how
execution flows from the user typing a string to a Taylor comparison being printed.

---

## Table of Contents

1. [High-level overview](#1-high-level-overview)
2. [Data types](#2-data-types)
3. [Stage 1 — Normalisation (`_normalise`)](#3-stage-1--normalisation-_normalise)
4. [Stage 2 — Parsing and safety (`function_parser`)](#4-stage-2--parsing-and-safety-function_parser)
5. [Stage 3 — Evaluation (`function`)](#5-stage-3--evaluation-function)
6. [Stage 4 — Numerical differentiation (`_nth_derivative`)](#6-stage-4--numerical-differentiation-_nth_derivative)
7. [Stage 5 — Taylor series (`taylor_series`)](#7-stage-5--taylor-series-taylor_series)
8. [Stage 6 — Comparison (`compare`)](#8-stage-6--comparison-compare)
9. [Full execution flow (end to end)](#9-full-execution-flow-end-to-end)
10. [Worked example: `y = sinx^2 + log(cosx)/2`](#10-worked-example-y--sinx2--logcosx2)

---

## 1. High-level overview

The program does four things in sequence:

```
User string  →  [normalise]  →  Python expression string
             →  [parse]      →  AST (abstract syntax tree)
             →  [eval]       →  f(x)  for any float x
             →  [differentiate + accumulate]  →  Taylor polynomial T(x)
             →  [compare]    →  |f(x) - T(x)|
```

None of these stages knows about the others. Each takes a well-defined
input type and returns a well-defined output type, which is why the
code is structured around two dataclasses rather than a bundle of loose
variables.

---

## 2. Data types

Two dataclasses carry state through the program.

### `ParsedFunction`

```python
@dataclass
class ParsedFunction:
    original   : str        # raw user input, e.g. "y = sinx^2 + log(cosx)/2"
    expression : str        # normalised Python string, e.g. "sin(x)**2+log(cos(x))/2"
    tree       : code       # compiled bytecode object, ready for eval()
```

`original` is kept purely for display. `expression` is the result after
normalisation — a valid Python expression. `tree` is the compiled code
object produced by `compile(ast.parse(...))`. It is **not** the raw AST;
it is already compiled to bytecode so that every call to `eval(tree, env)`
skips re-parsing and re-compiling, which matters if you evaluate in a loop.

### `TaylorSeries`

```python
@dataclass
class TaylorSeries:
    center       : float        # expansion point a
    n_terms      : int          # number of terms computed
    coefficients : List[float]  # c[k] = f^(k)(a) / k!
```

`coefficients[k]` is the coefficient of the `(x - a)^k` term. The full
polynomial is:

```
T(x) = c[0] + c[1]·(x-a) + c[2]·(x-a)² + c[3]·(x-a)³ + …
```

It also carries two methods:

- `__repr__` — prints the polynomial in human-readable form, e.g.
  `T(x) = 0 + 1·(x-0) + 0·(x-0)^2 + -0.166667·(x-0)^3 + …`
- `evaluate(x)` — evaluates the polynomial at `x` using Horner's method
  (explained in Stage 7).

---

## 3. Stage 1 — Normalisation (`_normalise`)

**Input:** a raw string like `"y = sinx^2 + log(cosx)/2"`  
**Output:** a valid Python expression like `"sin(x)**2+log(cos(x))/2"`

This is the messiest stage because human math notation and Python syntax
differ in several ways. Six transformations are applied in order.

### Step 1 — Strip `y =`

```
"y = sinx^2 + log(cosx)/2"  →  "sinx^2 + log(cosx)/2"
```

A regex `^[yY]\s*=\s*` matches an optional `y =` or `Y =` at the start
and removes it. The `\s*` allows spaces around the equals sign.

### Step 2 — Remove all whitespace

```
"sinx^2 + log(cosx)/2"  →  "sinx^2+log(cosx)/2"
```

A simple `str.replace(' ', '')`. All subsequent steps work on a
compact string with no spaces.

### Step 3 — Replace `^` with `**`

```
"sinx^2+log(cosx)/2"  →  "sinx**2+log(cosx)/2"
```

Python uses `**` for exponentiation. The caret is standard mathematical
notation but not valid Python.

### Step 4 — Replace `ln` with `log`

```
"ln(x)"  →  "log(x)"
```

Both `ln` and `log` map to `math.log` (natural logarithm). The
substitution is done at the string level before any other processing so
that `ln` does not confuse the function-name recogniser in step 5.

### Step 5 — Expand bare function application

```
"sinx"   →  "sin(x)"
"cosx"   →  "cos(x)"
"sin2x"  →  "sin(2*x)"
```

A regex matches any recognised function name (`sin`, `cos`, `tan`,
`log`, `sqrt`, `exp`, `abs`, etc.) that is **not** followed by `(`.
The captured suffix (the argument without parentheses) is wrapped:

```python
def _expand_bare_func(m):
    fname = m.group(1)    # e.g. "sin"
    rest  = m.group(2)    # e.g. "x" or "2x"
    rest  = re.sub(r'(\d)(x)', r'\1*\2', rest)   # "2x" → "2*x"
    return f"{fname}({rest})"                     # "sin(2*x)"
```

Function names are listed longest-first in `_FUNC_NAMES` so that
`sqrt` is matched before `s` and `asin` before `sin`.

### Step 6 — Insert implicit multiplication

Four sub-rules, applied in order:

| Pattern | Example | Result |
|---|---|---|
| digit before `x` | `2x` | `2*x` |
| `x` before digit | `x2` | `x*2` |
| `)` before `(` or letter | `)(`, `)x` | `*(`, `)*x` |
| digit before `(` | `2(` | `2*(` |

Each is a one-line `re.sub`. The order matters: the digit-before-`x`
rule must run before the `)` rules so that `2x` inside a larger
expression is fixed first.

---

## 4. Stage 2 — Parsing and safety (`function_parser`)

**Input:** raw user string  
**Output:** `ParsedFunction` dataclass

```python
def function_parser(s: str) -> ParsedFunction:
    original = s.strip()
    py_expr  = _normalise(original)
    tree     = ast.parse(py_expr, mode='eval')
    _check_safe(tree)
    ast.fix_missing_locations(tree)
    compiled = compile(tree, filename="<expr>", mode="eval")
    return ParsedFunction(original=original, expression=py_expr, tree=compiled)
```

### `ast.parse(py_expr, mode='eval')`

Converts the Python expression string into an AST. `mode='eval'` tells
the parser to expect a single expression (not statements), which
matches our use case. The result is an `ast.Expression` node whose
`.body` is the root of the expression tree.

### `_check_safe(tree)`

Walks every node in the tree recursively and raises `ValueError` if
any node type is not in the whitelist:

```python
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call,
    ast.Constant, ast.Name, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
    ast.USub, ast.UAdd,
)
```

This blocks anything that could execute arbitrary code: `ast.Import`,
`ast.Assign`, `ast.Attribute` (which could access `.__class__.__mro__`),
`ast.Lambda`, and so on. The expression `__import__('os').system('rm -rf /')`,
for example, would contain an `ast.Call` with an `ast.Attribute` node and
would be rejected here before `eval` is ever called.

### `compile(...)`

Turns the checked AST into a Python code object (bytecode). This is
the object stored in `ParsedFunction.tree`. Compiling once and reusing
is faster than passing a string to `eval` every time, since `eval` of a
string re-parses and re-compiles on every call.

---

## 5. Stage 3 — Evaluation (`function`)

**Input:** `ParsedFunction`, `float x`  
**Output:** `float y`

```python
def function(parsed: ParsedFunction, x: float) -> float:
    env = {**_SAFE_ENV, "x": x}
    return float(eval(parsed.tree, env))
```

`eval(code_object, env)` executes the bytecode with `env` as the
namespace. Every `Name` node in the tree is resolved against `env`:

- `"x"` → the float passed in
- `"sin"` → `math.sin`
- `"pi"` → `math.pi`
- `"__builtins__"` → `{}` (explicitly emptied to block `open`, `exec`, etc.)

`_SAFE_ENV` is a module-level dict defined once. Each call to `function`
creates a shallow copy with `x` added, so the base dict is never mutated.

---

## 6. Stage 4 — Numerical differentiation (`_nth_derivative`)

**Input:** `ParsedFunction`, point `a`, order `n`  
**Output:** approximation of `f^(n)(a)`

This is the mathematical core of the Taylor computation. No symbolic
manipulation of the AST is done — derivatives are estimated purely from
function values.

### The central difference idea

The derivative is defined as:

```
f'(a) = lim_{h→0} [f(a+h) - f(a)] / h
```

We do not take the limit. We pick a small but finite `h` and accept a
small approximation error. The **central** form is used instead of the
one-sided form:

```
f'(a) ≈ [f(a+h) - f(a-h)] / (2h)
```

Because the function is sampled symmetrically around `a`, the odd-order
error terms in the Taylor expansion of `f(a±h)` cancel, leaving an
error of order `h²` rather than `h`.

### The general n-th order stencil

Applying the same idea for the second derivative:

```
f''(a) ≈ [f(a+h) - 2·f(a) + f(a-h)] / h²
```

The weights `[+1, -2, +1]` are row 2 of Pascal's triangle with
alternating signs. For the third derivative the weights are
`[-1, +3, -3, +1]`. The pattern generalises to:

```
f^(n)(a) ≈ (1/hⁿ) · Σ_{k=0}^{n}  (-1)^(n-k) · C(n,k) · f(a + (k - n/2)·h)
```

where `C(n,k)` is the binomial coefficient `n! / (k!(n-k)!)`.

The `n+1` evaluation points are symmetric around `a`:

```
k = 0    →    a - (n/2)·h        ← leftmost
k = 1    →    a - (n/2 - 1)·h
  ...
k = n/2  →    a                  ← centre (only exists when n is even)
  ...
k = n    →    a + (n/2)·h        ← rightmost
```

### Why the binomial coefficients appear

Define the shift operator `E_s` such that `E_s[f](a) = f(a + s)`.
The central difference operator is `δ = E_{h/2} - E_{-h/2}`.
Applying it `n` times:

```
δⁿ = (E_{h/2} - E_{-h/2})ⁿ = Σ_{k=0}^{n} C(n,k) · (-1)^(n-k) · E_{(k-n/2)h}
```

This is just the binomial theorem on the shift operators. Applied to
`f` at `a`, it produces exactly the stencil formula above.

### Stencil weights by order

| Order | Weights | Division |
|---|---|---|
| n=1 | `-1, +1` | `h` |
| n=2 | `+1, -2, +1` | `h²` |
| n=3 | `-1, +3, -3, +1` | `h³` |
| n=4 | `+1, -4, +6, -4, +1` | `h⁴` |

### Choosing the step size h

Two errors compete as `h` changes:

**Truncation error** — the stencil formula is not exact; the Taylor
remainder gives an error proportional to `h²`. Smaller `h` reduces this.

**Cancellation error** — when `h` is very small, `f(a+h)` and `f(a-h)`
are nearly equal, and their difference loses precision to floating-point
cancellation. After dividing by `hⁿ`, this error is approximately
`ε / hⁿ` where `ε ≈ 1e-15` is machine epsilon. Smaller `h` makes
this worse.

Setting the two errors equal to find the optimal `h`:

```
h² ≈ ε / hⁿ
h^(n+2) ≈ ε
h ≈ ε^(1/(n+2))
```

In code:

```python
eps = 1e-15
h   = max(eps ** (1.0 / (n + 2)), 1e-4)
```

The floor at `1e-4` prevents `h` going so small that `h^n` underflows
to zero for low `n`. The result is that higher-order derivatives use a
**larger** `h` — the cancellation problem dominates more severely when
you divide by `h^n` for large `n`.

### Code walkthrough

```python
if n == 0:
    return function(parsed, a)          # base case: 0th derivative is f itself
```

```python
binom = [1] * (n + 1)
for k in range(1, n + 1):
    binom[k] = binom[k - 1] * (n - k + 1) // k
```

Builds Pascal's row `n` via the multiplicative recurrence
`C(n,k) = C(n,k-1) · (n-k+1) / k`. Integer arithmetic throughout;
no factorials computed.

```python
half = n / 2.0
for k in range(n + 1):
    sign = (-1) ** (n - k)
    x_k  = a + (k - half) * h
    total += sign * binom[k] * function(parsed, x_k)
return total / (h ** n)
```

Each iteration evaluates `f` at one stencil point and accumulates the
weighted sum. The division by `hⁿ` at the end recovers the derivative scale.

---

## 7. Stage 5 — Taylor series (`taylor_series`)

**Input:** `ParsedFunction`, number of terms `n`, expansion point `center`  
**Output:** `TaylorSeries` dataclass

### The Taylor polynomial

Any smooth function can be written as an infinite power series around a
point `a`:

```
f(x) = f(a) + f'(a)·(x-a) + f''(a)/2!·(x-a)² + f'''(a)/3!·(x-a)³ + …
```

Truncating after `n` terms gives the Taylor polynomial:

```
T(x) = Σ_{k=0}^{n-1}  c[k] · (x - a)^k

where  c[k] = f^(k)(a) / k!
```

When `a = 0` this is specifically called a Maclaurin series.

### How each coefficient is built

The loop runs `k = 0, 1, 2, …, n-1`. At each step:

```
k=0:  dk = f(a)       factorial = 1   →  c[0] = f(a) / 1
k=1:  dk = f'(a)      factorial = 1   →  c[1] = f'(a) / 1
k=2:  dk = f''(a)     factorial = 2   →  c[2] = f''(a) / 2
k=3:  dk = f'''(a)    factorial = 6   →  c[3] = f'''(a) / 6
k=4:  dk = f''''(a)   factorial = 24  →  c[4] = f''''(a) / 24
```

`factorial` is maintained as a **running product**:

```
factorial_0 = 1
factorial_k = factorial_{k-1} · k
```

This is a recurrence relation — each value depends only on the previous
one. It avoids recomputing `k!` from scratch each iteration, which would
cost `O(k)` multiplications per step instead of `O(1)`.

The derivative at each `k` is computed independently by `_nth_derivative`,
which runs a fresh stencil calibrated for that specific order.

### Polynomial evaluation — Horner's method

`TaylorSeries.evaluate(x)` does not compute each `c[k] · (x-a)^k`
separately and sum. Instead it uses Horner's method, which rewrites
the polynomial to eliminate all explicit powers:

```
T(x) = c[0] + (x-a)·(c[1] + (x-a)·(c[2] + (x-a)·(c[3] + …)))
```

In code, iterating over the reversed coefficient list:

```python
h = x - self.center
result = 0.0
for c in reversed(self.coefficients):
    result = result * h + c
```

For `n` terms, naive evaluation requires `O(n²)` multiplications (one
power per term, each power built from scratch). Horner's method requires
exactly `n-1` multiplications and `n` additions — `O(n)` — and is also
numerically better conditioned because it avoids computing large powers
that may overflow or lose precision.

---

## 8. Stage 6 — Comparison (`compare`)

**Input:** `ParsedFunction`, `TaylorSeries`, float `x`  
**Output:** dict with exact value, Taylor value, and two error measures

```python
exact   = function(parsed, x)
approx  = ts.evaluate(x)
abs_dev = abs(exact - approx)
rel_dev = abs_dev / abs(exact) * 100
```

**Absolute error** `|f(x) - T(x)|` is the raw difference in function
value. It tells you the error in the same units as `f`.

**Relative error** `|f(x) - T(x)| / |f(x)| × 100` expresses the error
as a percentage of the exact value. This is more informative when `f(x)`
is very large or very small — an absolute error of `0.001` means
something very different if `f(x) = 0.002` versus `f(x) = 1000`.

---

## 9. Full execution flow (end to end)

```
main()
│
├─ input: "y = sinx^2 + log(cosx)/2"
│
├─ function_parser("y = sinx^2 + log(cosx)/2")
│   ├─ _normalise(...)
│   │   ├─ strip "y ="          →  "sinx^2 + log(cosx)/2"
│   │   ├─ remove spaces        →  "sinx^2+log(cosx)/2"
│   │   ├─ ^ → **               →  "sinx**2+log(cosx)/2"
│   │   ├─ ln → log             →  (no change here)
│   │   ├─ expand sinx          →  "sin(x)**2+log(cosx)/2"
│   │   ├─ expand cosx          →  "sin(x)**2+log(cos(x))/2"
│   │   └─ implicit *           →  (no change here)
│   ├─ ast.parse(...)           →  AST tree
│   ├─ _check_safe(tree)        →  validates all nodes are math-safe
│   ├─ compile(tree)            →  bytecode code object
│   └─ returns ParsedFunction(original, expression, tree)
│
├─ input: n_terms = 6, center = 0.0
│
├─ taylor_series(parsed, 6, 0.0)
│   ├─ k=0: _nth_derivative(parsed, 0.0, 0)  →  f(0)          c[0] = f(0)/1
│   ├─ k=1: _nth_derivative(parsed, 0.0, 1)  →  f'(0)         c[1] = f'(0)/1
│   ├─ k=2: _nth_derivative(parsed, 0.0, 2)  →  f''(0)        c[2] = f''(0)/2
│   ├─ k=3: _nth_derivative(parsed, 0.0, 3)  →  f'''(0)       c[3] = f'''(0)/6
│   ├─ k=4: _nth_derivative(parsed, 0.0, 4)  →  f''''(0)      c[4] = f''''(0)/24
│   ├─ k=5: _nth_derivative(parsed, 0.0, 5)  →  f'''''(0)     c[5] = f'''''(0)/120
│   └─ returns TaylorSeries(center=0, n_terms=6, coefficients=[c0,c1,c2,c3,c4,c5])
│
│   Inside each _nth_derivative call (e.g. k=2):
│       ├─ h = max(1e-15^(1/4), 1e-4) ≈ 1.78e-4
│       ├─ binom = [1, 2, 1]
│       ├─ stencil points: [0 - h, 0, 0 + h]
│       ├─ weights:        [+1,   -2,  +1]
│       ├─ total = +1·f(-h) - 2·f(0) + 1·f(+h)
│       └─ return total / h²
│
├─ input: x = 0.5
│
├─ compare(parsed, ts, 0.5)
│   ├─ exact  = function(parsed, 0.5)   →  eval bytecode with x=0.5
│   ├─ approx = ts.evaluate(0.5)        →  Horner's method on coefficients
│   ├─ abs_error = |exact - approx|
│   └─ rel_error = abs_error / |exact| × 100
│
└─ print results
```

---

## 10. Worked example: `y = sinx^2 + log(cosx)/2`

### Normalisation trace

```
input  :  "y = sinx^2 + log(cosx)/2"
step 1 :  "sinx^2 + log(cosx)/2"       strip y=
step 2 :  "sinx^2+log(cosx)/2"         remove spaces
step 3 :  "sinx**2+log(cosx)/2"        ^ → **
step 4 :  (no change, no ln)
step 5 :  "sin(x)**2+log(cos(x))/2"    expand sinx, cosx
step 6 :  (no implicit * needed here)
output :  "sin(x)**2+log(cos(x))/2"
```

### AST structure

```
Expression
└── BinOp [Add]                      ← the top-level +
    ├── BinOp [Pow]                  ← sin(x) ** 2
    │   ├── Call sin(x)
    │   │   └── arg: Name 'x'
    │   └── Constant 2
    └── BinOp [Div]                  ← log(cos(x)) / 2
        ├── Call log(cos(x))
        │   └── arg: Call cos(x)
        │            └── arg: Name 'x'
        └── Constant 2
```

### Taylor coefficients at `a = 0` (6 terms)

The stencil estimates each derivative numerically:

| k | `f^(k)(0)` (approx) | `k!` | `c[k]` |
|---|---|---|---|
| 0 | 0.0 | 1 | 0.0 |
| 1 | 0.0 | 1 | 0.0 |
| 2 | 1.5 | 2 | 0.75 |
| 3 | 0.0 | 6 | 0.0 |
| 4 | −4.5 | 24 | −0.1875 |
| 5 | 0.0 | 120 | 0.0 |

So `T(x) ≈ 0.75x² − 0.1875x⁴` (odd terms vanish because the function
is even).

### Comparison at `x = 0.5`

```
exact  f(0.5) = sin(0.5)² + log(cos(0.5))/2 = 0.1645567...
T_6(0.5)       = 0.75·0.25 − 0.1875·0.0625  = 0.1875 − 0.01171875 ≈ 0.16406...
absolute error = |0.16456 − 0.16406| = 4.9 × 10⁻⁴
relative error = 0.30 %
```

Adding more terms (n=8) brings the absolute error down to `2.6 × 10⁻⁵`,
and n=10 to around `10⁻⁶`, showing the polynomial converging toward the
exact function as the radius of the truncated series grows.