<div align="Left">
  
## Random Walk
<img width="706" height="462" alt="image" src="https://github.com/user-attachments/assets/a68e0c7b-bcfa-429a-a70d-c4e0c6c28fcd" />

*A one-dimensional random walk models a particle that takes discrete, independent ±1 steps. Despite its simplicity, it is the foundation for diffusion, Brownian motion, and much of stochastic process theory.*

</div>

---

## `[ 01 ]` Definition

Let $X_1, X_2, \dots, X_N$ be a sequence of independent, identically distributed random variables such that

$$
X_i =
\begin{cases}
+1 & \text{with probability } p \\
-1 & \text{with probability } q = 1-p \\
\ 0 & \text{with probability } 0
\end{cases}
$$

The position of the walker after $N$ steps, starting from the origin, is the partial sum

$$
S_0 = 0, \qquad S_N = \sum_{i=1}^{N} X_i
$$

Each $X_i$ is drawn independently at every step — this is exactly what `generate_walk(p, n, rng)` does in the program, calling `rng.random()` once per step and comparing it against `p`.

---

## `[ 02 ]` Mean and Variance

### Single-step moments

$$
E[X_i] = p(1) + q(-1) = p - q = 2p - 1
$$

$$
E[X_i^2] = p(1)^2 + q(-1)^2 = 1
$$

$$
\text{Var}(X_i) = E[X_i^2] - \big(E[X_i]\big)^2 = 1 - (2p-1)^2 = 4pq
$$

### After $N$ steps

Because the $X_i$ are independent, expectation and variance are additive:

$$
E[S_N] = N(2p - 1)
$$

$$
\text{Var}(S_N) = N \cdot 4pq
$$

For the **symmetric walk** ($p = q = \tfrac12$):

$$
E[S_N] = 0, \qquad \text{Var}(S_N) = N
$$

so the *typical* displacement grows like $\sqrt{N}$ — the signature scaling of diffusive motion, not $N$ as it would for ballistic motion.

---

## `[ 03 ]` Exact Distribution

After $N$ steps, if $m$ of them were $+1$ and $N-m$ were $-1$, the position is

$$
S_N = m - (N - m) = 2m - N
$$

So landing on $S_N = k$ requires $m = \dfrac{N+k}{2}$, which is only an integer when $N$ and $k$ share the same parity. The probability follows a binomial law:

$$
P(S_N = k) = \binom{N}{m} \, p^{m} q^{\,N-m}, \qquad m = \frac{N+k}{2}
$$

This is the exact, finite-$N$ distribution — a walk of $N$ steps can only ever land on one of $N+1$ possible sites, spaced two apart.

---

## `[ 04 ]` Generating Function

The moment generating function of a single step is

$$
E\!\left[e^{tX_i}\right] = p e^{t} + q e^{-t}
$$

By independence, the generating function of $S_N$ is the single-step MGF raised to the $N$-th power:

$$
E\!\left[e^{tS_N}\right] = \left(p e^{t} + q e^{-t}\right)^{N}
$$

Differentiating this at $t=0$ recovers the mean and variance derived in `[ 02 ]` directly, and expanding it in powers of $N$ is the standard route to the Central Limit Theorem result below.

---

## `[ 05 ]` Central Limit Theorem — the Gaussian Limit

Standardizing $S_N$ by its own mean and standard deviation,

$$
Z_N = \frac{S_N - N(2p-1)}{\sqrt{4Npq}}
$$

the Central Limit Theorem gives, as $N \to \infty$,

$$
Z_N \ \xrightarrow{d}\ \mathcal{N}(0, 1)
$$

So for large $N$, the walk's position is approximately Gaussian:

$$
S_N \ \approx\ \mathcal{N}\big(N(2p-1),\ 4Npq\big)
$$

This is the bridge between the **Random Walk Model** and the **Gaussian Analysis** module — the walk is a concrete, simulable example of the CLT in action, and running many independent walks and histogramming their endpoints is the natural experiment to visualize it.

---

## `[ 06 ]` Diffusion Limit

Rescale the walk so each step has size $\Delta x$ and takes time $\Delta t$, and let $\Delta x, \Delta t \to 0$ while holding

$$
D = \frac{(\Delta x)^2}{2\,\Delta t}
$$

fixed. In this continuum limit the discrete walk converges to **Brownian motion**, and its probability density $P(x, t)$ obeys the diffusion (heat) equation:

$$
\frac{\partial P}{\partial t} = D \, \frac{\partial^2 P}{\partial x^2}
$$

with the Gaussian from `[ 05 ]` as its fundamental solution. This is the theoretical endpoint the discrete simulation is approximating: a single simulated walk is one sample path of that limiting diffusion process.

---

## `[ 07 ]` Program Workflow

### Core step: `generate_walk(p, n, rng)`

```
  pos ← [0]
  for i in 1..N:
      draw u ~ Uniform(0, 1) from rng
      step ← +1 if u < p else -1
      pos.append(pos[-1] + step)
  return pos
```

Every step draws its own random number — the walk is never precomputed from a formula, it is built one Bernoulli trial at a time, exactly as the derivation in `[ 01 ]` describes.

### Experiment cycle (both program variants)

```
  ① read p, N from the input fields
  ② validate: 0 ≤ p ≤ 1, N a positive integer
  ③ draw a fresh seed (system entropy ⊕ high-resolution clock)
  ④ seed the dedicated RNG instance
  ⑤ run generate_walk(p, N, rng) → position sequence
  ⑥ plot position vs. step index
  ⑦ autoscale the y-axis to the walk's own min/max (+ padding)
  ⑧ display seed, final position, distance from origin, step count
```

### Two interface variants

| File | Behaviour |
|---|---|
| `RandomWalk1D.py` | Plot is embedded directly in the main window; **Regenerate** reruns steps ③–⑧ in place |
| `RandomWalkGraphPopup1D.py` | Main window holds only the parameters; **Generate Graph** runs steps ③–⑧ inside a brand-new popup window, so multiple walks can be kept open and compared side by side |

In both variants, the x-axis is naturally bounded to $[0, N]$, and only the y-axis needs autoscaling — a direct consequence of moving from the two-variable version (where both axes depended on the walk) to the one-dimensional model, where the step index is deterministic and only the position is random.

---

## `[ 08 ]` Status

> **◆ Work In Progress**
>
> Implemented: one-dimensional walk generation, position-vs-step visualization, autoscaling, seed-based
> regeneration in both embedded and popup interface variants.
>
> Not yet implemented: multi-walk ensembles for empirically verifying the CLT limit in `[ 05 ]`,
> first-passage/return time statistics, and biased-walk drift visualization.

---

<div align="center">

`Random Walk Model` · `Statistics` · `Numerical Methods Laboratory` · `◆ Work In Progress`

</div>
