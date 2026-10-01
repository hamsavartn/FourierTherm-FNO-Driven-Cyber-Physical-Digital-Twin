"""Phase-11 REAL-DATA rerun (PRD §10 step 4): closed-loop evaluation on the
fetched ACN-Data jobs (outputs/jobs_caltech_*.csv), chained across days
(v3.3), with controllers no_control / static_cap / shield (+ guarded_hybrid
when the surrogate artifact passes its gate).

Scenario framing (documented assumptions, PRD-consistent):
- The EV hub hangs on an 11 kV feeder whose non-EV baseload follows a smooth
  diurnal profile (peak ~10:00, matching the hub's own arrival pattern).
- Baseload scale factor: the analytic steady state at (baseload + EV peak,
  summer far-field T_inf) is set to ~97 degC -- hot enough that no-control
  violates, cool enough that the static cap is a meaningful baseline
  (mirrors the validated synthetic stress design; the twin's absolute MW
  scale is arbitrary, the stress PATTERN is what is evaluated).
- Station -> phase: round-robin over stationIDs (ACN-Data publishes no site
  wiring; documented per PRD §2.2).
- Summer heatwave boundary: Kusuda day-of-year 200 (validated fixture choice).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from dtr.baselines.static_cap import compute_static_cap_kw
from dtr.data.synthetic import EVSE_CAP_KW, JobSet
from dtr.evaluation.closed_loop import run_day
from dtr.twin.rc_model import DigitalTwin, TwinParams

ROOT = os.path.join(os.path.dirname(__file__), "..")
T0_SUMMER_GMIN = 200 * 1440


def load_jobs(day: str, station_phase: dict, ev_scale: float = 20.0) -> tuple[JobSet, list[str]]:
    df = pd.read_csv(os.path.join(ROOT, "outputs", f"jobs_caltech_{day}.csv"))
    stations, starts, deadlines, reqs, deliv, caps = [], [], [], [], [], []
    for _, r in df.iterrows():
        stations.append(station_phase.setdefault(r.stationID, len(station_phase)))
        starts.append(int(r.t_start_min))
        deadlines.append(int(r.t_deadline_min))
        reqs.append(float(r.energy_requested_kwh) * ev_scale)
        deliv.append(0.0)
        caps.append(EVSE_CAP_KW * ev_scale)
    jobs = JobSet(station=np.array(stations), t_start=np.array(starts),
                  t_deadline=np.array(deadlines), energy_req=np.array(reqs),
                  delivered=np.array(deliv), evse_cap=np.array(caps))
    return jobs, list(df.stationID)


def build_day(twin: DigitalTwin, day: str, station_phase: dict,
              base_target_T: float = 86.0, ev_scale: float = 20.0) -> tuple:
    jobs, sids = load_jobs(day, station_phase, ev_scale)
    phase_map = np.arange(len(station_phase)) % 3
    # EV demand profile from the jobs themselves (allocator-consistent:
    # same desired_kw the shield/allocator use at runtime)
    from dtr.control.allocator import desired_kw, phase_aggregate
    ev = np.zeros((1440, 3))
    pm_j = phase_map[jobs.station]
    for m in range(1440):
        des = desired_kw(jobs, m)
        if des.any():
            ev[m] = phase_aggregate(des, pm_j)
    # baseload calibration: base-ONLY analytic steady state at the summer
    # eval boundary = base_target_T (86 degC) -- static-safe with margin, so
    # the STATIC CAP is positive and EV curtailment is the decisive lever
    # (the previous framing made the BASELOAD the violator, which no EV
    # controller can fix). Adoption scaling: each real session represents
    # `ev_scale` identical sessions (documented scenario construction).
    T_eval = twin.kusuda_T_inf(T0_SUMMER_GMIN * 60.0)
    lo, hi = 0.0, 5000.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if twin.steady_state([mid] * 3, T_eval)[:3].max() <= base_target_T:
            lo = mid
        else:
            hi = mid
    base_kw = lo
    hrs = np.arange(1440) / 60.0
    shape = 0.75 + 0.25 * np.exp(-((hrs - 10.0) / 4.0) ** 2)   # morning peak
    baseload = np.stack([base_kw * shape] * 3, axis=1)

    class Day:
        pass
    d = Day()
    d.jobs, d.phase_map, d.baseload_phase, d.n_stations = jobs, phase_map, baseload, len(station_phase)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+",
                    default=["2019-10-15", "2019-10-16", "2019-10-17",
                             "2019-10-18", "2019-10-19"])
    ap.add_argument("--controllers", nargs="+",
                    default=["no_control", "static_cap", "shield"])
    ap.add_argument("--target", type=float, default=86.0)
    ap.add_argument("--ev-scale", type=float, default=20.0)
    args = ap.parse_args()

    twin = DigitalTwin(TwinParams.from_yaml(os.path.join(ROOT, "configs", "twin.yaml")))
    station_phase: dict = {}
    days = {d: build_day(twin, d, station_phase, args.target, args.ev_scale)
            for d in args.days}
    # SEASONAL static rating basis: worst far-field T_inf within the
    # evaluation window (documented; annual-max basis yields cap=0 when the
    # framing puts base+EV peaks near the limit -- see phase11_real.log)
    T_eval_max = max(twin.kusuda_T_inf((T0_SUMMER_GMIN + m) * 60.0)
                     for m in range(1440))
    p_static = compute_static_cap_kw(twin, float(max(
        d.baseload_phase.max() for d in days.values())), T_worst=T_eval_max)
    print(f"static cap: {p_static:.1f} kW total EV | {len(station_phase)} stations")

    models = None
    art = os.path.join(ROOT, "outputs", "fno_model.pt")
    if "guarded_hybrid" in args.controllers and os.path.exists(art):
        from dtr.surrogate.train import load_model
        models = [load_model(art)]

    # chained spin-up under static cap (v3.3 preconditioning)
    x0 = None
    for _ in range(2):
        x0 = run_day(twin, days[args.days[0]], "static_cap",
                     p_static_kw=p_static, x0=x0,
                     t0_gmin=T0_SUMMER_GMIN)["x_final"]

    rows = []
    for c in args.controllers:
        x = x0
        for day in args.days:
            t0 = time.perf_counter()
            r = run_day(twin, days[day], c, p_static_kw=p_static, x0=x,
                        t0_gmin=T0_SUMMER_GMIN, models=models)
            x = r["x_final"]
            s = r["summary"]
            rows.append({"day": day, "controller": c,
                         "violations": s["violations"],
                         "E_delivered_kwh": round(s["E_delivered_kwh"], 1),
                         "completion_ratio": round(s["completion_ratio"], 3),
                         "jain_index": round(s["jain_index"], 4),
                         "max_core": round(s["max_core"], 2),
                         "wall_s": round(time.perf_counter() - t0, 1),
                         "modes": str(s.get("modes", {}))})
            print(f"{day} {c:14s} viol={s['violations']:3d} "
                  f"E={s['E_delivered_kwh']:8.1f} kWh "
                  f"compl={s['completion_ratio']:.2f} "
                  f"jain={s['jain_index']:.3f} maxT={s['max_core']:.1f}")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ROOT, "outputs", "phase_summary_real.csv"),
               index=False)
    print(f"\nwrote outputs/phase_summary_real.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
