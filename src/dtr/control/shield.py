"""Safety shield (PRD §5.4, twin-only controller (c); the guarded hybrid wraps
this in Step 5). Receding-horizon bisection over the demand-scaling fraction
alpha, with the v3.6 ROUTING FIX: feasibility composes the allocator, so the
twin verifies the per-phase profile that will actually be dispatched.
Invariant: max over the three cores and the horizon <= T_max - margin."""
from __future__ import annotations

from collections import deque

import numpy as np

from dtr.control.allocator import allocate, desired_kw, phase_aggregate


class Shield:
    def __init__(self, twin, phase_map: np.ndarray, H: int = 30, dt: float = 60.0,
                 iters: int = 9, T_max: float = 90.0, margin: float = 2.0,
                 beta: float = 0.6):
        self.twin = twin
        self.phase_map = phase_map
        self.H = H
        self.dt = dt
        self.iters = iters
        self.T_max = T_max
        self.margin = margin
        self.beta = beta

    # ---------------- horizon rollout with allocator composed (v3.6) ----------------
    def horizon_max_core(self, x0, t_gmin: int, alpha: float, jobs,
                         baseload_phase: np.ndarray, t_inf_fn) -> float:
        """Max core temperature over the next H minutes if the allocator
        dispatches with allowed = alpha * desired(t). Job copies are mutated
        hypothetically (delivered energy, completion)."""
        p = self.twin.p
        j = jobs.copy()
        x = np.asarray(x0, dtype=float).copy()
        pm_j = self.phase_map[j.station]   # phase of each JOB's station (v3.6)
        mx = -np.inf
        for h in range(self.H):
            t_now = int(t_gmin) + h
            des = desired_kw(j, t_now)
            allowed = alpha * float(des.sum())
            sp = allocate(allowed, j, t_now, beta=self.beta)
            p_phase = baseload_phase[t_now % baseload_phase.shape[0]] \
                + phase_aggregate(sp, pm_j)
            x = self.twin.step(x, p_phase, t_inf_fn(t_now), self.dt)
            mx = max(mx, float(x[:3].max()))
            act = j.active(t_now)
            j.delivered[act] += sp[act] * self.dt / 3600.0   # kW x s -> kWh
        return mx

    # ---------------- bisection (PRD §5.4) ----------------
    def choose_alpha(self, x0, t_gmin: int, jobs, baseload_phase: np.ndarray,
                     t_inf_fn):
        """Returns (alpha, mode). modes: 'ok' (no curtailment), 'curtailed',
        'base_over' (baseload alone breaches the margin -- uncontrollable)."""
        Tlim = self.T_max - self.margin
        t0 = t_gmin

        def feasible(a: float) -> bool:
            return self.horizon_max_core(x0, t0, a, jobs, baseload_phase,
                                         t_inf_fn) <= Tlim

        des0 = desired_kw(jobs, t0)
        if float(des0.sum()) <= 0.0:
            return 1.0, "ok"          # nothing to allocate
        if feasible(1.0):
            return 1.0, "ok"
        if not feasible(0.0):
            return 0.0, "base_over"   # uncontrollable load alone is too hot
        lo, hi = 0.0, 1.0
        for _ in range(self.iters):
            mid = 0.5 * (lo + hi)
            if feasible(mid):
                lo = mid
            else:
                hi = mid
        return lo, "curtailed"
