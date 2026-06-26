"""
function_parser.py
==================
This program is for parsing human-readable math expressions into an AST, evaluate at float x,
and compute Taylor series with polynomial detection.

Supported: sin cos tan log ln sqrt exp abs, constants pi e
Operators: + - * / ^ (unary -)
Implicit:  sinx→sin(x), 2x→2*x, x^2→x**2
"""

import ast
import math
import re
from dataclasses import dataclass
from typing import List


@dataclass
class TaylorSeries:
    center      : float
    n_terms     : int
    coefficients: List[float]
    is_poly     : bool = False
    poly_degree : int  = -1

    def __repr__(self):
        terms = []
        for k, c in enumerate(self.coefficients):
            if k == 0:
                terms.append(f"{c:.6g}")
            elif k == 1:
                a = self.center
                terms.append(f"{c:.6g}·(x{'-' if a>=0 else '+'}{abs(a):.6g})")
            else:
                a = self.center
                terms.append(f"{c:.6g}·(x{'-' if a>=0 else '+'}{abs(a):.6g})^{k}")
        return "T(x) = " + " + ".join(terms)

    def evaluate(self, x: float) -> float:
        # Horner's method: avoids explicit powers, O(n) multiplications
        h = x - self.center
        result = 0.0
        for c in reversed(self.coefficients):
            result = result * h + c
        return result


@dataclass
class ParsedFunction:
    original  : str   # raw user input
    expression: str   # normalised Python expression
    tree      : object  # compiled bytecode, ready for eval()

    def __repr__(self):
        return f"ParsedFunction(expr={self.expression!r})"


# Normaliser

_FUNC_NAMES = ["sqrt", "sinh", "cosh", "tanh", "asin", "acos", "atan",
               "sin", "cos", "tan", "log", "ln", "exp", "abs"]
_FUNC_PAT   = "|".join(_FUNC_NAMES)


def _normalise(expr: str) -> str:
    s = expr.strip()
    s = re.sub(r'^[yY]\s*=\s*', '', s)   # strip "y ="
    s = s.replace(' ', '')                # remove spaces
    s = s.replace('^', '**')             # caret → power
    s = s.replace('ln', 'log')           # ln → log

    # expand bare function application: sinx → sin(x), sin2x → sin(2*x)
    def _expand_bare_func(m):
        fname, rest = m.group(1), m.group(2)
        if not rest:
            return fname
        rest = re.sub(r'(\d)(x)', r'\1*\2', rest)
        rest = re.sub(r'(x)(\d)', r'\1*\2', rest)
        return f"{fname}({rest})"

    s = re.sub(r'(' + _FUNC_PAT + r')(?!\()([0-9]*\.?[0-9]*x?|x[0-9]*)',
               _expand_bare_func, s)

    # implicit multiplication rules
    s = re.sub(r'(\d)(x)',      r'\1*\2', s)   # 2x → 2*x
    s = re.sub(r'(x)(\d)',      r'\1*\2', s)   # x2 → x*2
    s = re.sub(r'\)(\()',       r')*\1',  s)   # )( → )*(
    s = re.sub(r'\)([a-zA-Z])', r')*\1',  s)   # )x → )*x
    s = re.sub(r'(\d)\(',       r'\1*(', s)    # 2( → 2*(
    return s


#Safe eval environment

_SAFE_ENV = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "log": math.log, "exp": math.exp, "sqrt": math.sqrt, "abs": abs,
    "pi": math.pi, "e": math.e,
    "__builtins__": {},  # block all builtins
}

#AST safety check

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call,
    ast.Constant, ast.Name, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
    ast.USub, ast.UAdd,
)

def _check_safe(node: ast.AST):
    # reject any node type not in the math-only whitelist
    if not isinstance(node, _ALLOWED_NODES):
        raise ValueError(f"Unsafe construct: {type(node).__name__}")
    for child in ast.iter_child_nodes(node):
        _check_safe(child)


#Polynomial detection

def _is_polynomial_node(node: ast.AST) -> bool:
    # Constant (any number) → always polynomial
    if isinstance(node, ast.Constant):
        return True
    # Name: only 'x' is the polynomial variable; e, pi etc. are not
    if isinstance(node, ast.Name):
        return node.id == 'x'
    # unary minus/plus: polynomial if operand is
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _is_polynomial_node(node.operand)
    if isinstance(node, ast.BinOp):
        op = node.op
        if isinstance(op, (ast.Add, ast.Sub)):   # sum/diff of polynomials
            return _is_polynomial_node(node.left) and _is_polynomial_node(node.right)
        if isinstance(op, ast.Mult):              # product of polynomials
            return _is_polynomial_node(node.left) and _is_polynomial_node(node.right)
        if isinstance(op, ast.Div):              # p(x)/constant only
            return _is_polynomial_node(node.left) and isinstance(node.right, ast.Constant)
        if isinstance(op, ast.Pow):              # x^n: exponent must be non-negative int
            exp_ok = (isinstance(node.right, ast.Constant)
                      and isinstance(node.right.value, int)
                      and node.right.value >= 0)
            return _is_polynomial_node(node.left) and exp_ok
    return False  # ast.Call (sin, log, ...) and everything else


def _polynomial_degree(node: ast.AST) -> int:
    if isinstance(node, ast.Constant): return 0
    if isinstance(node, ast.Name):     return 1          # must be 'x'
    if isinstance(node, ast.UnaryOp):  return _polynomial_degree(node.operand)
    if isinstance(node, ast.BinOp):
        op = node.op
        if isinstance(op, (ast.Add, ast.Sub)):
            return max(_polynomial_degree(node.left), _polynomial_degree(node.right))
        if isinstance(op, ast.Mult):
            return _polynomial_degree(node.left) + _polynomial_degree(node.right)
        if isinstance(op, ast.Div):   return _polynomial_degree(node.left)
        if isinstance(op, ast.Pow):   return node.right.value
    return 0


def _extract_poly_coeffs(node: ast.AST, degree: int) -> List[float]:
    # Evaluate p at degree+1 integer points, then solve the Vandermonde
    # system V·c = y to recover exact algebraic coefficients.
    n      = degree + 1
    points = list(range(n))
    tmp    = compile(ast.fix_missing_locations(ast.Expression(body=node)), "<poly>", "eval")
    y_vals = [float(eval(tmp, {**_SAFE_ENV, "x": float(xi)})) for xi in points]

    # build augmented matrix [V | y], V[i][j] = xi^j
    M = [[float(xi ** j) for j in range(n)] + [y_vals[i]] for i, xi in enumerate(points)]

    # Gaussian elimination with partial pivoting
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[pivot_row] = M[pivot_row], M[col]
        piv = M[col][col]
        if abs(piv) < 1e-12:
            continue
        M[col] = [v / piv for v in M[col]]
        for row in range(n):
            if row == col: continue
            f = M[row][col]
            M[row] = [M[row][j] - f * M[col][j] for j in range(n + 1)]

    return [M[k][n] for k in range(n)]


def is_polynomial(parsed: "ParsedFunction") -> "tuple[bool, int]":
    # re-parse the normalised string to get a walkable AST (tree is bytecode)
    body = ast.parse(parsed.expression, mode='eval').body
    if _is_polynomial_node(body):
        return True, _polynomial_degree(body)
    return False, -1


# Public API

def function_parser(s: str) -> ParsedFunction:
    original = s.strip()
    py_expr  = _normalise(original)
    try:
        tree = ast.parse(py_expr, mode='eval')
    except SyntaxError as exc:
        raise SyntaxError(f"Parse failed: {py_expr!r}\n  {exc}") from exc
    _check_safe(tree)
    ast.fix_missing_locations(tree)
    compiled = compile(tree, "<expr>", "eval")
    return ParsedFunction(original=original, expression=py_expr, tree=compiled)


def function(parsed: ParsedFunction, x: float) -> float:
    return float(eval(parsed.tree, {**_SAFE_ENV, "x": x}))


def _nth_derivative(parsed: ParsedFunction, a: float, n: int) -> float:
    """
    n-th derivative at a via central-difference stencil with binomial weights.

    Formula:  f^(n)(a) ≈ (1/h^n) · Σ_{k=0}^{n} (-1)^(n-k) · C(n,k) · f(a + (k-n/2)·h)

    Weights are Pascal's row n with alternating signs:
        n=1: [-1,+1]   n=2: [+1,-2,+1]   n=3: [-1,+3,-3,+1]  ...

    Step size h = eps^(1/(n+2)) balances truncation error O(h²) against
    cancellation error O(eps/h^n). Higher n → larger h is optimal.
    """
    if n == 0:
        return function(parsed, a)

    eps = 1e-15
    h   = max(eps ** (1.0 / (n + 2)), 1e-4)

    # Pascal's row n via multiplicative recurrence: C(n,k) = C(n,k-1)*(n-k+1)/k
    binom = [1] * (n + 1)
    for k in range(1, n + 1):
        binom[k] = binom[k - 1] * (n - k + 1) // k

    total = 0.0
    half  = n / 2.0
    for k in range(n + 1):
        sign = (-1) ** (n - k)          # alternates: leftmost gets (-1)^n
        x_k  = a + (k - half) * h      # symmetric around a
        total += sign * binom[k] * function(parsed, x_k)

    return total / (h ** n)


def taylor_series(parsed: ParsedFunction, n_terms: int, center: float = 0.0) -> TaylorSeries:
    """
    Build Taylor polynomial T(x) = Σ c[k]·(x-a)^k,  c[k] = f^(k)(a)/k!

    Polynomial fast path: if f is already a polynomial, derivatives beyond
    its degree are exactly zero — numerical stencils would return ~1e-9 noise.
    Instead, exact coefficients are extracted via a Vandermonde solve and
    higher terms are padded with 0.0. Stencil loop is skipped entirely.
    """
    poly_flag, poly_deg = is_polynomial(parsed)
    if poly_flag and center == 0.0:
        raw_body     = ast.parse(parsed.expression, mode='eval').body
        exact_coeffs = _extract_poly_coeffs(raw_body, poly_deg)
        padded       = (exact_coeffs + [0.0] * n_terms)[:n_terms]
        return TaylorSeries(center=center, n_terms=n_terms, coefficients=padded,
                            is_poly=True, poly_degree=poly_deg)

    # General path: compute each c[k] = f^(k)(a)/k! numerically
    coeffs    = []
    factorial = 1   # maintained as running product: factorial_k = factorial_{k-1} * k
    for k in range(n_terms):
        if k > 0:
            factorial *= k
        dk = _nth_derivative(parsed, center, k)
        coeffs.append(dk / factorial)

    return TaylorSeries(center=center, n_terms=n_terms, coefficients=coeffs,
                        is_poly=False, poly_degree=-1)


def compare(parsed: ParsedFunction, ts: TaylorSeries, x: float) -> dict:
    exact   = function(parsed, x)
    approx  = ts.evaluate(x)
    abs_dev = abs(exact - approx)
    rel_dev = abs_dev / abs(exact) * 100 if exact != 0 else float('inf')
    return {"x": x, "exact": exact, "taylor": approx,
            "abs_error": abs_dev, "rel_error_%": rel_dev}


# CLI

def main():
    print("=" * 60)
    print("  Math Function Evaluator  +  Taylor Series")
    print("  Supports: sin cos tan log sqrt exp ^ pi e")
    print("=" * 60)
    print("  Example: y = sinx^2 + log(cosx)/2")
    print("           y = x^3 - 2*x + e^x")
    print()

    expr_str = input("Enter function        > ").strip()
    try:
        parsed = function_parser(expr_str)
    except (SyntaxError, ValueError) as err:
        print(f"\n[Parse error] {err}"); return
    print(f"Parsed expression     : {parsed.expression}\n")

    try:
        n_terms = int(input("Number of Taylor terms> ").strip())
        center  = float(input("Expansion center a    > [default 0]: ").strip() or "0")
    except ValueError:
        print("[Error] n_terms must be int, center must be float."); return

    print("\nComputing Taylor coefficients...", end=" ", flush=True)
    ts = taylor_series(parsed, n_terms, center)
    print("done.\n")

    if ts.is_poly:
        print(f"[POLYNOMIAL DETECTED]  degree = {ts.poly_degree}")
        print(f"  Taylor series is f itself — exact coefficients from AST, no stencil.")
        print(f"  Terms beyond degree {ts.poly_degree} are exactly zero.\n")

    print(ts)
    print()

    x_str = input("Enter x value         > ").strip()
    try:
        x_val = float(x_str)
    except ValueError:
        print("[Error] x must be a number."); return

    result = compare(parsed, ts, x_val)

    print()
    print("─" * 44)
    print(f"  x              = {result['x']}")
    print(f"  Exact  f(x)    = {result['exact']:.10f}")
    if ts.is_poly:
        print(f"  Taylor T_{n_terms}(x)  = {result['taylor']:.10f}  [= f(x), poly bypass]")
        print(f"  Absolute error = {result['abs_error']:.6e}  (float rounding only)")
    else:
        print(f"  Taylor T_{n_terms}(x)  = {result['taylor']:.10f}")
        print(f"  Absolute error = {result['abs_error']:.6e}")
    print(f"  Relative error = {result['rel_error_%']:.4f} %")
    print("─" * 44)


if __name__ == "__main__":
    main()