"""
Piston Kinetic Theory Simulator — FastAPI backend.

Holds the authoritative thermodynamic state (T, P, V, Q, U, W) for a gas of
N ~ 10^23 particles being heated in a piston, either at constant pressure
(free piston, isobaric) or constant volume (locked piston, isochoric).

The frontend (static/index.html) renders a representative sample of tracer
particles for visual intuition and polls this backend for the exact physics,
which is computed here from the ideal gas law + first law of thermodynamics
rather than by literally integrating 10^23 particle trajectories.

Run:
    pip install -r requirements.txt
    uvicorn main:app --reload
    open http://127.0.0.1:8000
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Literal, List, Tuple

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# ---------------------------------------------------------------- constants
R = 8.314462618      # J / (mol K)
NA = 6.02214076e23   # particles / mol
CAL = 4.184          # J / cal
ATM = 101325.0       # Pa
MAX_TRACE_POINTS = 400


# ---------------------------------------------------------------- schemas
class SimParams(BaseModel):
    N_coef: float = Field(6.022, ge=0.1, le=9.999)
    N_exp: int = Field(23, ge=10, le=30)
    gas_type: Literal["mono", "di"] = "di"
    V0_L: float = Field(10.0, gt=0)
    T0_K: float = Field(300.0, gt=0)
    mode: Literal["free", "locked"] = "free"
    heat_amount: float = Field(5000.0, ge=0)
    heat_unit: Literal["J", "cal"] = "J"
    duration_s: float = Field(6.0, gt=0)


class StepRequest(BaseModel):
    dt: float = Field(..., gt=0, le=0.5)


class SimState(BaseModel):
    elapsed: float
    duration: float
    Q_delivered_J: float
    Q_total_J: float
    T_K: float
    P_Pa: float
    P_atm: float
    V_L: float
    U_J: float
    W_J: float
    n_mol: float
    N_particles: float
    Cv: float
    Cp: float
    f_dof: int
    mode: str
    done: bool
    pv_trace: List[Tuple[float, float]]  # (V in L, P in Pa)


# ---------------------------------------------------------------- physics state
class _Sim:
    """Mutable simulation state, recomputed on /api/reset."""
    def __init__(self):
        self.params: SimParams = SimParams()
        self.reset(self.params)

    def reset(self, p: SimParams):
        self.params = p
        f_dof = 3 if p.gas_type == "mono" else 5
        self.f_dof = f_dof
        self.Cv = (f_dof / 2) * R
        self.Cp = self.Cv + R
        self.C_active = self.Cp if p.mode == "free" else self.Cv

        N = p.N_coef * (10 ** p.N_exp)
        self.N_particles = N
        self.n_mol = N / NA

        self.V0 = p.V0_L / 1000.0  # m^3
        self.T0 = p.T0_K
        self.P0 = self.n_mol * R * self.T0 / self.V0

        self.T = self.T0
        self.V = self.V0
        self.P = self.P0

        self.Q_total = p.heat_amount if p.heat_unit == "J" else p.heat_amount * CAL
        self.duration = p.duration_s
        self.power = self.Q_total / self.duration if self.duration > 0 else 0.0

        self.Q_delivered = 0.0
        self.elapsed = 0.0
        self.done = self.Q_total <= 0
        self.pv_trace = [(self.V * 1000.0, self.P)]

    def step(self, dt: float):
        if self.done:
            return
        dQ = self.power * dt
        if self.Q_delivered + dQ >= self.Q_total:
            dQ = self.Q_total - self.Q_delivered
        self.Q_delivered += dQ
        dT = dQ / (self.n_mol * self.C_active) if self.n_mol * self.C_active > 0 else 0.0
        self.T += dT

        if self.params.mode == "free":
            self.P = self.P0
            self.V = self.n_mol * R * self.T / self.P0
        else:
            self.V = self.V0
            self.P = self.n_mol * R * self.T / self.V

        self.elapsed += dt
        self.pv_trace.append((self.V * 1000.0, self.P))
        if len(self.pv_trace) > MAX_TRACE_POINTS:
            self.pv_trace = self.pv_trace[::2]

        if self.Q_delivered >= self.Q_total - 1e-9 or self.elapsed >= self.duration:
            self.done = True

    def snapshot(self) -> SimState:
        U = self.n_mol * self.Cv * self.T
        W = self.n_mol * R * (self.T - self.T0) if self.params.mode == "free" else 0.0
        return SimState(
            elapsed=round(self.elapsed, 4),
            duration=self.duration,
            Q_delivered_J=self.Q_delivered,
            Q_total_J=self.Q_total,
            T_K=self.T,
            P_Pa=self.P,
            P_atm=self.P / ATM,
            V_L=self.V * 1000.0,
            U_J=U,
            W_J=W,
            n_mol=self.n_mol,
            N_particles=self.N_particles,
            Cv=self.Cv,
            Cp=self.Cp,
            f_dof=self.f_dof,
            mode=self.params.mode,
            done=self.done,
            pv_trace=self.pv_trace,
        )


sim = _Sim()

# ---------------------------------------------------------------- app
app = FastAPI(title="Piston Kinetic Theory Simulator")


@app.post("/api/reset", response_model=SimState)
def api_reset(params: SimParams):
    try:
        sim.reset(params)
    except ZeroDivisionError:
        raise HTTPException(400, "Invalid parameters (division by zero)")
    return sim.snapshot()


@app.post("/api/step", response_model=SimState)
def api_step(req: StepRequest):
    sim.step(req.dt)
    return sim.snapshot()


@app.get("/api/state", response_model=SimState)
def api_state():
    return sim.snapshot()


# Serve the frontend last so it doesn't shadow the /api routes.
# Uses an absolute path (relative to this file, not the shell's cwd) so it
# works no matter what directory you launch `uvicorn` from.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
