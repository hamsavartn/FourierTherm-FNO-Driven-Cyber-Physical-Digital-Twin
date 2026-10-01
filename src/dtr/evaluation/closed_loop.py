"""Closed-loop day simulator (PRD §5.6).

Controllers: 'no_control' (a), 'static_cap' (b), 'shield' (c, twin-only).
The guarded hybrid (d) wraps the shield in Step 5. Log schema matches
outputs/sim_<controller>_<date>.parquet semantics (per-minute rows); job
outcomes follow §9 metrics (completion ratio by deadline, Jain index).

Thermal preconditioning note (v3.2 §5.6): tests initialize at the analytic
steady state of the baseload; the full 48 h spin-up profile ships with the
evaluation harness when real multi-day data arrives."""
from __future__ import annotations

import time

import numpy as np

from dtr.control.allocator import allocate, desired_kw, phase_aggregate
from dtr.control.shield import Shield

DT = 60.0
CONTROLLERS = ("no_control", "static_cap", "shield", "guarded_hybrid")


def jain_index(x) -> float:
    x = np.asarray(x, dtype=float)
    s = float(x.sum())
    if s <= 0.0:
        return 1.0
    return float(s * s / (len(x) * float(np.square(x).sum())))


def run_day(twin, day, controller: str, p_static_kw: float | None = None,
            x0=None, t0_gmin: float = 0.0, margin: float = 2.0,
            beta: float = 0.6, recheck_every: int = 0, models=None) -> dict:
    if controller not in CONTROLLERS:
        raise ValueError(controller)
    n = 1440
    jobs = day.jobs.copy()
    if x0 is None:
        x0 = twin.steady_state(day.baseload_phase[0].tolist(),
                               twin.kusuda_T_inf(t0_gmin * 60.0))
    x = np.asarray(x0, dtype=float).copy()
    shield = Shield(twin, day.phase_map, margin=margin, beta=beta) \
        if controller == "shield" else None
    hybrid = None
    P_hist = None
    if controller == "guarded_hybrid":
        from dtr.control.guarded_hybrid import GuardedHybrid
        if not models:
            raise ValueError("guarded_hybrid requires trained models")
        hybrid = GuardedHybrid(twin, day.phase_map, models, margin=margin,
                               beta=beta)
        # rolling realized-context buffer (the twin observer's memory)
        P_hist = np.tile(day.baseload_phase[0], (60, 1)).astype(float)
        Tinf_hist = np.full(60, twin.kusuda_T_inf(t0_gmin * 60.0))
    Tlim = twin.p.T_max - margin

    logs = {k: np.zeros(n) for k in
            ("demand_kw", "allowed_kw", "alpha", "max_core", "s",
             "violations_cum", "step_ms")}
    cores_log = np.zeros((n, 3))
    pm_j = day.phase_map[jobs.station]   # phase of each JOB's station (v3.6)
    modes = []
    nJ = len(jobs.energy_req)
    completed_by_deadline = np.zeros(nJ, dtype=bool)
    recheck_fail = 0
    viol = 0
    E = 0.0
    t_start = time.perf_counter()

    for m in range(n):
        tc0 = time.perf_counter()
        t_now = m                      # day-local minutes (single-day runs)
        des = desired_kw(jobs, t_now)
        demand = float(des.sum())

        if controller == "no_control":
            alpha, allowed, mode = 1.0, demand, "ok"
        elif controller == "static_cap":
            allowed = min(float(p_static_kw), demand)
            alpha = allowed / demand if demand > 0 else 1.0
            mode = "ok"
        else:
            fn = lambda g: twin.kusuda_T_inf(g * 60.0)
            if controller == "guarded_hybrid":
                alpha, mode, tw_mT, s_mT = hybrid.choose_alpha(
                    x, t0_gmin + m, jobs, day.baseload_phase, fn,
                    P_hist=P_hist, Tinf_hist=Tinf_hist)
            else:
                alpha, mode = shield.choose_alpha(
                    x, t0_gmin + m, jobs, day.baseload_phase, fn)
            allowed = alpha * demand
            if recheck_every and m % recheck_every == 0 \
                    and controller != "guarded_hybrid":
                rc = shield.horizon_max_core(
                    x, t0_gmin + m, alpha, jobs, day.baseload_phase,
                    lambda g: twin.kusuda_T_inf(g * 60.0))
                if rc > Tlim + 1e-6:
                    recheck_fail += 1

        sp = allocate(allowed, jobs, t_now, beta=beta)
        p_phase = day.baseload_phase[m] + phase_aggregate(sp, pm_j)
        x = twin.step(x, p_phase, twin.kusuda_T_inf((t0_gmin + m) * 60.0), DT)
        if P_hist is not None:
            P_hist[:-1] = P_hist[1:]
            P_hist[-1] = p_phase
            Tinf_hist[:-1] = Tinf_hist[1:]
            Tinf_hist[-1] = twin.kusuda_T_inf((t0_gmin + m) * 60.0)

        act = jobs.active(t_now)
        step_kwh = sp[act] * DT / 3600.0   # kW x seconds -> kWh
        jobs.delivered[act] += step_kwh
        E += float(step_kwh.sum())

        crossed = jobs.t_deadline == t_now
        if crossed.any():
            completed_by_deadline[crossed] = (
                jobs.delivered[crossed] >= 0.95 * jobs.energy_req[crossed])

        mc = float(x[:3].max())
        if mc > twin.p.T_max + 1e-6:
            viol += 1

        logs["demand_kw"][m] = demand
        logs["allowed_kw"][m] = allowed
        logs["alpha"][m] = alpha
        logs["max_core"][m] = mc
        logs["s"][m] = x[6]
        logs["violations_cum"][m] = viol
        logs["step_ms"][m] = (time.perf_counter() - tc0) * 1000.0
        cores_log[m] = x[:3]
        modes.append(mode)

    summary = {
        "controller": controller,
        "violations": viol,
        "E_delivered_kwh": E,
        "completion_ratio": float(completed_by_deadline.mean()) if nJ else 1.0,
        "jain_index": jain_index(jobs.delivered),
        "max_core": float(cores_log.max()),
        "recheck_failures": recheck_fail,
        "wall_s": time.perf_counter() - t_start,
        "modes": {m_: modes.count(m_) for m_ in set(modes)},
    }
    return {"logs": logs, "cores": cores_log, "modes": modes,
            "summary": summary, "jobs": jobs, "x_final": x}
