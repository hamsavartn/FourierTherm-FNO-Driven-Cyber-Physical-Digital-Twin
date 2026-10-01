"""Scenario grid generator (PRD §5.1): supervised training data for the FNO
surrogate, straight from the numpy twin. Each scenario is a 1440-min
trajectory under a sampled environment; the FNO dataset class slices
context/horizon windows at training time.

Sampling (v3.2-v3.5 requirements):
- seasonal Kusuda boundary {T_mean, T_amp, t_shift} -- slow and physical
- baseload level, EV duty-cycle pulse profiles per phase
- lognormal parameter perturbations (twin parametrization)
- a share of scenarios driven into soil dry-out (T_x crossing T_crit)

Outputs outputs/scenarios.npz with per-scenario arrays. Generation cost is
CPU-bound: ~1-2 s per scenario (twin rollout); N=1000 is a ~30 min run.
"""
from __future__ import annotations

import numpy as np

from dtr.twin.rc_model import DigitalTwin, TwinParams


def sample_params(base: TwinParams, rng: np.random.Generator, sigma: float) -> TwinParams:
    d = base.__dict__.copy()
    for k in ("R_core", "R2", "R3", "R4_wet", "C_core", "C_i", "C_s", "C_x"):
        d[k] = float(d[k] * rng.lognormal(0.0, sigma))
    return TwinParams(**d)


def sample_ev_profile(rng: np.random.Generator, n_min: int, n_blocks: int,
                      peak_kw_total: float) -> np.ndarray:
    """Random duty-cycle block profile, (n_min, 3) kW, phases roughly balanced."""
    p = np.zeros((n_min, 3))
    for _ in range(n_blocks):
        start = int(rng.integers(0, n_min - 60))
        dur = int(rng.integers(60, 480))
        amp = peak_kw_total * rng.uniform(0.3, 1.0) / 3.0
        skew = rng.dirichlet(np.ones(3))          # mild phase skew
        p[start:start + dur] += amp * skew
    return p


def generate_scenarios(base_params: TwinParams, n_scenarios: int = 1000,
                       seed: int = 0, sigma: float = 0.15,
                       dryout_fraction: float = 0.2,
                       n_min: int = 1440) -> dict:
    rng = np.random.default_rng(seed)
    n = n_min
    arrays = {
        "P_phase": np.zeros((n_scenarios, n, 3)),      # kW per phase (base+EV)
        "EV": np.zeros((n_scenarios, n, 3)),           # EV-only profile
        "T_inf": np.zeros((n_scenarios, n)),
        "T_cores": np.zeros((n_scenarios, n, 3)),
        "T_nodes": np.zeros((n_scenarios, n, 4)),      # Ti, Ts, Tx, s
        "R4_eff": np.zeros((n_scenarios,)),            # initial effective R4
        "params": np.zeros((n_scenarios, 8)),          # perturbed thermal params
        "peak_kw": np.zeros((n_scenarios,)),
        "dryout": np.zeros((n_scenarios,), dtype=bool),
    }
    for i in range(n_scenarios):
        p = sample_params(base_params, rng, sigma)
        twin = DigitalTwin(p)
        t_shift = float(rng.uniform(0.0, 365.0))
        t0 = float(rng.uniform(0.0, 365.0)) * 86400.0
        T_inf = np.array([twin.kusuda_T_inf(t0 + m * 60.0) for m in range(n)])

        base_ph = rng.uniform(800.0, 1900.0)          # kW per phase
        ev = sample_ev_profile(rng, n, int(rng.integers(2, 6)),
                               rng.uniform(600.0, 2200.0))
        if i < int(dryout_fraction * n_scenarios):
            ev += sample_ev_profile(rng, n, 3, 2600.0)   # sustained overload
        P = np.clip(base_ph + ev, 0.0, None)

        x0 = twin.steady_state([base_ph] * 3, float(T_inf[0]))
        out = twin.rollout(x0, P, T_inf, 60.0)
        arrays["P_phase"][i] = P
        arrays["EV"][i] = ev
        arrays["T_inf"][i] = T_inf
        arrays["T_cores"][i] = out["T_cores"]
        arrays["T_nodes"][i] = out["states"][:, 3:]
        arrays["R4_eff"][i] = p.R4_wet
        arrays["params"][i] = [p.R_core, p.R2, p.R3, p.R4_wet,
                               p.C_core, p.C_i, p.C_s, p.C_x]
        arrays["peak_kw"][i] = P.sum(axis=1).max()
        arrays["dryout"][i] = out["s"].max() > 0.05
    return arrays


def save_scenarios(arrays: dict, path: str) -> None:
    np.savez_compressed(path, **arrays)


def load_scenarios(path: str) -> dict:
    return dict(np.load(path))
