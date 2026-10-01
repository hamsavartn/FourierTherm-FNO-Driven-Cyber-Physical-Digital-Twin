"""Guarded hybrid shield (PRD v3.2 §5.3b + v3.4 monotonicity watchdog + v3.6
routing composition): the FNO surrogate runs the alpha-bisection; the numeric
twin verifies ONCE per minute (safety authority) and grades optimality
(two-sided temperature consistency, K_RECOV slack check, setpoint-jump guard);
escalation to full twin bisection on any trigger.

Feature building mirrors WindowDataset exactly: slots 0-59 realized context,
60-89 the CANDIDATE allocator-composed dispatch + T_inf forecast, states held
at last value. The candidate profile is composed through the ALLOCATOR (v3.6
routing fix), so the surrogate verifies the profile that would be dispatched.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from dtr.control.allocator import allocate, desired_kw, phase_aggregate
from dtr.control.shield import Shield
from dtr.surrogate.fno_1d import FNO1d
from dtr.surrogate.train import IN_CH, T_CTX, T_OUT

MODES = ("surrogate_ok", "recovered", "fallback", "nonmono", "base_over")


class GuardedHybrid:
    def __init__(self, twin, phase_map, models: list,
                 H: int = 30, dt: float = 60.0, iters: int = 9,
                 T_max: float = 90.0, margin: float = 2.0, beta: float = 0.6,
                 dT_tol: float = 1.5, k_recov: float = 3.0, jump_tol: float = 0.10,
                 eps_mono: float = 0.2,
                 mean=None, std=None):
        self.twin = twin
        self.phase_map = phase_map
        self.models = [m["model"] if isinstance(m, dict) else m for m in models]
        self.mean = mean if mean is not None else (
            models[0]["mean"] if isinstance(models[0], dict) else None)
        self.std = std if std is not None else (
            models[0]["std"] if isinstance(models[0], dict) else None)
        self.y_mean = models[0]["y_mean"] if isinstance(models[0], dict) else None
        self.y_std = models[0]["y_std"] if isinstance(models[0], dict) else None
        self.H, self.dt, self.iters = H, dt, iters
        self.T_max, self.margin, self.beta = T_max, margin, beta
        self.dT_tol, self.k_recov, self.jump_tol = dT_tol, k_recov, jump_tol
        self.eps_mono = eps_mono
        self.bias_ewma = 0.0
        self.a_prev = 1.0

    # ---------------- features ----------------
    def _features(self, x, t_gmin: int, alpha: float, jobs,
                  baseload_phase, t_inf_fn, P_hist, Tinf_hist) -> np.ndarray:
        """(12, 90) window for one candidate alpha (allocator-composed).
        Slots 0-59: REALIZED context (P_hist per-phase kW, Tinf_hist, states
        from x -- matches WindowDataset training distribution exactly).
        Slots 60-89: candidate future dispatch + T_inf forecast, states held."""
        j = jobs.copy()
        pm_j = self.phase_map[j.station]
        P_fut = np.zeros((self.H, 3))
        for h in range(self.H):
            t_now = int(t_gmin) + h
            des = desired_kw(j, t_now)
            sp = allocate(alpha * float(des.sum()), j, t_now, beta=self.beta)
            P_fut[h] = baseload_phase[t_now % baseload_phase.shape[0]] \
                + phase_aggregate(sp, pm_j)
            act = j.active(t_now)
            j.delivered[act] += sp[act] * self.dt / 3600.0
        Tinf_fut = np.array([t_inf_fn(int(t_gmin) + h) for h in range(self.H)])
        hold = lambda v: np.full(self.H, float(v))
        ctx_states = [np.full(T_CTX, float(v)) for v in x[:7]]
        return np.stack([
            np.concatenate([P_hist[:, 0] / 1000.0, P_fut[:, 0] / 1000.0]),
            np.concatenate([P_hist[:, 1] / 1000.0, P_fut[:, 1] / 1000.0]),
            np.concatenate([P_hist[:, 2] / 1000.0, P_fut[:, 2] / 1000.0]),
            np.concatenate([Tinf_hist, Tinf_fut]),
            np.concatenate([ctx_states[0], hold(x[0])]),
            np.concatenate([ctx_states[1], hold(x[1])]),
            np.concatenate([ctx_states[2], hold(x[2])]),
            np.concatenate([ctx_states[3], hold(x[3])]),
            np.concatenate([ctx_states[4], hold(x[4])]),
            np.concatenate([ctx_states[5], hold(x[5])]),
            np.concatenate([ctx_states[6], hold(x[6])]),
            np.full(T_CTX + self.H, self.twin.p.R4_wet),
        ]).astype(np.float32)[:IN_CH]

    def _surrogate_maxT(self, x, t_gmin, alpha, jobs, baseload_phase,
                        t_inf_fn, P_hist=None, Tinf_hist=None):
        if P_hist is None:
            P_hist = np.tile(
                baseload_phase[int(t_gmin) % baseload_phase.shape[0]], (T_CTX, 1))
        if Tinf_hist is None:
            Tinf_hist = np.full(T_CTX, t_inf_fn(int(t_gmin)))
        feats = torch.from_numpy(
            self._features(x, t_gmin, alpha, jobs, baseload_phase, t_inf_fn,
                           P_hist, Tinf_hist)
        ).unsqueeze(0)
        if self.mean is not None:
            feats = (feats - self.mean) / self.std
        preds = [m(feats).squeeze(0).detach().numpy() for m in self.models]
        if self.y_mean is not None:
            preds = [p * float(self.y_std) + float(self.y_mean) for p in preds]
        return float(np.max(preds)), float(np.std(preds) + 1e-9), preds[0]

    # ---------------- bisection on the surrogate ----------------
    def choose_alpha(self, x, t_gmin, jobs, baseload_phase, t_inf_fn,
                     P_hist=None, Tinf_hist=None):
        Tlim = self.T_max - self.margin
        if P_hist is None:
            P_hist = np.tile(baseload_phase[int(t_gmin) % baseload_phase.shape[0]],
                             (T_CTX, 1))
        if Tinf_hist is None:
            Tinf_hist = np.full(T_CTX, t_inf_fn(int(t_gmin)))
        queries: list[tuple[float, float]] = []

        def s_feasible(a: float) -> bool:
            mT, spread, _ = self._surrogate_maxT(x, t_gmin, a, jobs,
                                                 baseload_phase, t_inf_fn,
                                                 P_hist, Tinf_hist)
            queries.append((a, mT))
            return mT <= Tlim

        des0 = desired_kw(jobs, int(t_gmin))
        if float(des0.sum()) <= 0.0:
            return 1.0, "surrogate_ok", 0.0, 0.0
        if not s_feasible(0.0):
            return 0.0, "base_over", 0.0, 0.0
        lo, hi = 0.0, 1.0
        if s_feasible(1.0):
            a_hat = 1.0
        else:
            for _ in range(self.iters):
                mid = 0.5 * (lo + hi)
                if s_feasible(mid):
                    lo = mid
                else:
                    hi = mid
            a_hat = lo

        # monotonicity watchdog (v3.4): near-isotonicity over the query history
        qs = sorted(queries)
        nonmono = any(qs[i + 1][1] < qs[i][1] - self.eps_mono
                      for i in range(len(qs) - 1))

        # twin verification: safety + two-sided temperature grading (v3.2)
        sh = Shield(self.twin, self.phase_map, H=self.H, dt=self.dt,
                    T_max=self.T_max, margin=self.margin, beta=self.beta)
        twin_maxT = sh.horizon_max_core(x, t_gmin, a_hat, jobs,
                                        baseload_phase, t_inf_fn)
        s_maxT = max(mT for _, mT in qs)
        self.bias_ewma = 0.9 * self.bias_ewma + 0.1 * (twin_maxT - s_maxT)
        t0 = time.perf_counter()

        if twin_maxT > Tlim:
            a, mode = self._twin_bisect(x, t_gmin, jobs, baseload_phase,
                                        t_inf_fn), "fallback"
        elif (abs(twin_maxT - s_maxT) > self.dT_tol
              or twin_maxT <= Tlim - (self.k_recov - 1.0) * self.margin
              or abs(a_hat - self.a_prev) > self.jump_tol
              or nonmono):
            a, mode = self._twin_bisect(x, t_gmin, jobs, baseload_phase,
                                        t_inf_fn), "recovered"
        else:
            a, mode = a_hat, "surrogate_ok"
        self.a_prev = a
        del t0
        return a, mode, twin_maxT, s_maxT

    def _twin_bisect(self, x, t_gmin, jobs, baseload_phase, t_inf_fn) -> float:
        sh = Shield(self.twin, self.phase_map, H=self.H, dt=self.dt,
                    iters=self.iters, T_max=self.T_max, margin=self.margin,
                    beta=self.beta)
        a, mode = sh.choose_alpha(x, t_gmin, jobs, baseload_phase, t_inf_fn)
        return a
