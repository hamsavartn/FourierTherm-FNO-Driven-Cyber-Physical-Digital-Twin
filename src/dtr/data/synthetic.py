"""SYNTHETIC demand-day generator -- TEST FIXTURE ONLY.

Stands in for ACN-Data jobs (Steps 1-2) so the control layer can be built and
validated before the API token is available. NOT used for reported results:
all reported metrics come from scripts/01..02 on real ACN data (PRD FR1/FR2).

Design (PRD §5.5 / Round-7 test design):
- n_stations single-phase EVSEs, round-robin phase map (documented assumption)
- jobs concentrated in an evening window, one EV per station at a time
- background baseload calibrated so that the balanced steady state at peak EV
  reaches a target core temperature (default 94 degC) -- guarantees the
  no-control baseline violates 90 degC while the shield can still act.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dtr.control.allocator import desired_kw
from dtr.twin.rc_model import DigitalTwin

EVSE_CAP_KW = 7.2


@dataclass
class JobSet:
    """Vectorized job container (PRD §3.5 jobs.csv semantics in-memory)."""
    station: np.ndarray      # (J,) station index
    t_start: np.ndarray      # (J,) minute of day
    t_deadline: np.ndarray   # (J,) minute of day
    energy_req: np.ndarray   # (J,) kWh
    delivered: np.ndarray    # (J,) kWh
    evse_cap: np.ndarray     # (J,) pilot cap kW

    def copy(self) -> "JobSet":
        return JobSet(self.station.copy(), self.t_start.copy(),
                      self.t_deadline.copy(), self.energy_req.copy(),
                      self.delivered.copy(), self.evse_cap)

    def active(self, t_now: int) -> np.ndarray:
        return ((self.delivered < self.energy_req - 1e-9)
                & (t_now >= self.t_start))


@dataclass
class DemandDay:
    jobs: JobSet
    phase_map: np.ndarray        # (S,) station -> phase 0/1/2
    baseload_phase: np.ndarray   # (1440, 3) kW -- exogenous, not controllable
    n_stations: int


def _desired_kw(jobs: JobSet, t_now: int) -> np.ndarray:
    return desired_kw(jobs, t_now)


def make_synthetic_day(twin: DigitalTwin, n_stations: int = 200, n_jobs: int = 300,
                       seed: int = 0, target_peak_T: float = 94.0,
                       start_hour: int = 14, end_hour: int = 24) -> DemandDay:
    rng = np.random.default_rng(seed)
    phase_map = np.arange(n_stations) % 3

    # ---- jobs: sustained afternoon/evening block, one EV per station at a
    # time. Sized so the EV fleet alone (~1.6 MW peak) stresses the feeder
    # over multiple days -- single-evening loads cannot out-heat the ~88 h
    # soil time constant (measured; see tests/test_control.py warm-up). ----
    stations, t_start, t_deadline, energy = [], [], [], []
    free_at = np.zeros(n_stations)
    lo, hi = start_hour * 60, end_hour * 60
    for _ in range(n_jobs):
        start = int(rng.integers(lo, hi - 60))
        dur = int(rng.integers(120, 361))
        cand = np.where(free_at <= start)[0]
        if cand.size == 0:
            continue
        st = int(rng.choice(cand))
        free_at[st] = start + dur
        stations.append(st)
        t_start.append(start)
        t_deadline.append(min(start + int(rng.integers(180, 481)), 1439))
        energy.append(float(rng.uniform(30.0, 75.0)))
    jobs = JobSet(
        station=np.array(stations), t_start=np.array(t_start),
        t_deadline=np.array(t_deadline), energy_req=np.array(energy),
        delivered=np.zeros(len(stations)),
        evse_cap=np.full(len(stations), EVSE_CAP_KW),
    )

    # ---- peak EV demand (balanced-equivalent) and baseload calibration ----
    peak_ev_kw = 0.0
    for t in range(lo, 1440):
        peak_ev_kw = max(peak_ev_kw, float(_desired_kw(jobs, t).sum()))
    # calibrate against the ACTUAL far-field temperature at the load peak
    # (Kusuda is seasonal: Jan-1 evening is ~6 degC below T_mean)
    t_peak_gmin = 0.5 * (start_hour + end_hour) * 60
    T_inf_day = twin.kusuda_T_inf(t_peak_gmin * 60.0)
    i_ev = peak_ev_kw / 3.0 * 1000.0 / (twin.p.V_ll / math_sqrt3() * twin.p.pf)
    i_target = ((target_peak_T - T_inf_day)
                / (3.0 * twin.p.R_ac * (twin.p.R2 + twin.p.R3 + twin.p.R4_wet)
                   + twin.p.R_ac * twin.p.R_core)) ** 0.5
    i_base = i_target - i_ev
    base_ph_kw = i_base * (twin.p.V_ll / math_sqrt3()) * twin.p.pf / 1000.0
    baseload = np.full((1440, 3), base_ph_kw)
    return DemandDay(jobs=jobs, phase_map=phase_map, baseload_phase=baseload,
                     n_stations=n_stations)


def math_sqrt3() -> float:
    return 3.0 ** 0.5
