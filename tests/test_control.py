"""Step 4 validation (PRD §10): allocator sanity, AT-3 (0 violations),
AT-4 (shield invariants), G2 ordering E_dyn > E_static. Runs on the SYNTHETIC
fixture day (src/dtr/data/synthetic.py) until real ACN data lands."""
import os

import numpy as np
import pytest

from dtr.baselines.static_cap import compute_static_cap_kw
from dtr.control.allocator import allocate, desired_kw, phase_aggregate
from dtr.data.synthetic import JobSet, make_synthetic_day
from dtr.evaluation.closed_loop import jain_index, run_day
from dtr.twin.rc_model import DigitalTwin, TwinParams

CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "twin.yaml")


@pytest.fixture(scope="module")
def twin():
    return DigitalTwin(TwinParams.from_yaml(CONFIG))


@pytest.fixture(scope="module")
def day(twin):
    # Stress fixture: ~1.85 MW sustained EV block (14:00-24:00) on a SUMMER
    # day (Kusuda day ~200, T_inf ~28 degC -- the E7(d) heatwave case). The
    # winter boundary (13.7 degC) can't produce violations: the 88 h soil
    # node keeps daily-cycle peaks ~14 degC under the analytic steady state
    # (measured). Summer + high evening load => genuine no-control breach;
    # baseload-only noon steady stays ~73 degC (static-safe).
    return make_synthetic_day(twin, n_stations=200, n_jobs=300, seed=0,
                              target_peak_T=100.0, start_hour=14, end_hour=24)


T0_SUMMER_GMIN = 200 * 1440   # mid-July: Kusuda far-field ~28 degC


@pytest.fixture(scope="module")
def p_static(twin, day):
    return compute_static_cap_kw(twin, float(day.baseload_phase.max()))


# ---------------- allocator unit behavior ----------------
def _mini_jobs():
    return JobSet(
        station=np.array([0, 1]), t_start=np.array([0, 0]),
        t_deadline=np.array([400, 400]), energy_req=np.array([40.0, 40.0]),
        delivered=np.array([0.0, 0.0]), evse_cap=np.array([7.2, 7.2]))


def test_allocator_no_scarcity_serves_full_and_fair():
    jobs = _mini_jobs()
    sp = allocate(20.0, jobs, t_now=0)          # 20 >= 14.4 desired
    assert np.allclose(sp, [7.2, 7.2])
    assert jain_index(sp) >= 0.999              # PRD §5.5 sanity (>= 0.95)


def test_allocator_urgency_priority():
    jobs = _mini_jobs()
    jobs.t_deadline = np.array([15, 400])       # job 0 urgent (< 30 min boost)
    sp = allocate(7.2, jobs, t_now=0)           # half the demand
    assert sp[0] > 2.0 * sp[1]                  # urgent job clearly prioritized


def test_allocator_respects_evse_cap():
    jobs = _mini_jobs()
    sp = allocate(1000.0, jobs, t_now=0)
    assert sp.max() <= 7.2 + 1e-9


def test_phase_aggregate_round_robin():
    pm = np.arange(6) % 3
    out = phase_aggregate(np.ones(6), pm)
    assert np.allclose(out, [2.0, 2.0, 2.0])    # balanced by construction


# ---------------- AT-3: violations ----------------
@pytest.fixture(scope="module")
def results(twin, day, p_static):
    # Thermal preconditioning (PRD §5.6 / trap 14): 48 h summer spin-up under
    # STANDARD operation (static cap -- how the feeder would really have run
    # before evaluation day; uncontrolled spin-up hands the controllers an
    # already-overheated cable that no alpha can rescue). Soil needs multiple
    # days to warm (tau ~ 88 h), so carry the final state into every run.
    x0 = None
    for _ in range(2):
        x0 = run_day(twin, day, "static_cap", p_static_kw=p_static, x0=x0,
                     t0_gmin=T0_SUMMER_GMIN)["x_final"]
    return {c: run_day(twin, day, c, p_static_kw=p_static, x0=x0,
                       t0_gmin=T0_SUMMER_GMIN, recheck_every=15)
            for c in ("no_control", "static_cap", "shield")}


def test_at3_no_control_violates(twin, day, results):
    assert results["no_control"]["summary"]["violations"] > 0
    assert results["no_control"]["summary"]["max_core"] > 90.0


def test_at3_static_zero_violations(twin, day, p_static, results):
    assert p_static > 0.0                        # sanity: cap exists on this day
    assert results["static_cap"]["summary"]["violations"] == 0
    assert results["shield"]["summary"]["violations"] == 0       # G1


def test_at4_alpha_bounds_and_recheck(results):
    r = results["shield"]
    assert np.all((r["logs"]["alpha"] >= 0.0) & (r["logs"]["alpha"] <= 1.0))
    assert r["summary"]["recheck_failures"] == 0  # twin re-verification holds


def test_g2_energy_ordering(results):
    E = {c: results[c]["summary"]["E_delivered_kwh"] for c in results}
    assert E["no_control"] >= E["shield"]                     # sanity
    assert E["shield"] > E["static_cap"]                      # G2 improvement


def test_g3_fairness_and_completion_reported(results):
    for c in ("static_cap", "shield"):
        s = results[c]["summary"]
        assert 0.0 <= s["jain_index"] <= 1.0
        assert 0.0 <= s["completion_ratio"] <= 1.0
    # shield throttles less than static -> more jobs complete by deadline
    assert results["shield"]["summary"]["completion_ratio"] \
        >= results["static_cap"]["summary"]["completion_ratio"]
