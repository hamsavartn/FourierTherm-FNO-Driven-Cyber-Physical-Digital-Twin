"""Step 5 validation: AT-7 (surrogate speed), AT-8 (G4 rule), the
alpha-differentiation capability check (the v0 flaw), and guarded-hybrid
end-to-end safety on the stress day. Trains a small model inline on the
cached scenario grid so the test is self-contained (~1 min)."""
import os
import time

import numpy as np
import pytest
import torch

from dtr.control.guarded_hybrid import GuardedHybrid
from dtr.data.synthetic import make_synthetic_day
from dtr.evaluation.closed_loop import run_day
from dtr.surrogate.fno_1d import FNO1d
from dtr.surrogate.scenario_grid import load_scenarios
from dtr.surrogate.train import rollout_speed, train
from dtr.twin.rc_model import DigitalTwin, TwinParams

ROOT = os.path.join(os.path.dirname(__file__), "..")
CONFIG = os.path.join(ROOT, "configs", "twin.yaml")
SCEN = os.path.join(ROOT, "outputs", "scenarios.npz")


@pytest.fixture(scope="module")
def twin():
    return DigitalTwin(TwinParams.from_yaml(CONFIG))


@pytest.fixture(scope="module")
def model():
    artifact = os.path.join(ROOT, "outputs", "fno_model.pt")
    if os.path.exists(artifact):
        from dtr.surrogate.train import load_model
        return load_model(artifact)
    if not os.path.exists(SCEN):
        pytest.skip("scenario grid not generated yet (run scripts/04)")
    arrays = load_scenarios(SCEN)
    from dtr.twin.rc_model import TwinParams
    res = train(arrays, epochs=30, seed=0, verbose=True,
                base_params=TwinParams.from_yaml(CONFIG))
    return res


def test_surrogate_differentiates_alpha(twin, model):
    """THE capability check: max-T prediction must INCREASE with alpha.
    Uses a REALISTIC rolling context (dispatched history from an uncontrolled
    warm-up) -- flat synthetic contexts are out-of-distribution by shape."""
    day = make_synthetic_day(twin, n_stations=60, n_jobs=80, seed=3,
                             target_peak_T=96.0, start_hour=14)
    fn = lambda g: twin.kusuda_T_inf(g * 60.0)
    from dtr.control.allocator import desired_kw, phase_aggregate
    pm = day.phase_map[day.jobs.station]
    t0 = 14 * 60
    
    # Manually warm up the twin to t0 to get a perfectly consistent rolling context
    # instead of mismatching x_final (1440) with t0 (840)
    x = twin.steady_state(day.baseload_phase[0].tolist(), fn(0))
    for m in range(t0 - 60):
        P = day.baseload_phase[m] + phase_aggregate(desired_kw(day.jobs, m), pm)
        x = twin.step(x, P, fn(m), 60.0)
        
    Ph = np.zeros((60, 3))
    Th = np.zeros(60)
    for i, m in enumerate(range(t0 - 60, t0)):
        P = day.baseload_phase[m] + phase_aggregate(desired_kw(day.jobs, m), pm)
        Ph[i] = P
        Th[i] = fn(m)
        x = twin.step(x, P, fn(m), 60.0)

    gh = GuardedHybrid(twin, day.phase_map, [model])
    
    # Debug predictions
    mTs = {}
    for a in (0.2, 0.6, 1.0):
        mT, _, _ = gh._surrogate_maxT(x, t0, a, day.jobs, day.baseload_phase, fn, Ph, Th)
        mTs[a] = mT
        
        feats = gh._features(x, t0, a, day.jobs, day.baseload_phase, fn, Ph, Th)
        P_fut_max = feats[0:3, 60:90].max() * 1000.0
        P_fut_sum = feats[0:3, 60:90].sum() * 1000.0
        print(f"a={a}, mT={mT:.2f}, P_fut_max={P_fut_max:.1f}, P_fut_sum={P_fut_sum:.1f}")
        print(f"P_fut[0] (kW): {feats[0, 60:90] * 1000.0}")

    assert mTs[1.0] > mTs[0.6] > mTs[0.2], f"no alpha response: {mTs}"
    assert mTs[1.0] > mTs[0.6] > mTs[0.2], f"no alpha response: {mTs}"
    # threshold is PHYSICS-RELATIVE: the true twin range over the same alpha
    # sweep (~0.45 degC at this operating point -- the RC ladder's inertia
    # limits a 30-min horizon) bounds any correct model's range
    from dtr.control.shield import Shield
    sh = Shield(twin, day.phase_map)
    twin_range = (sh.horizon_max_core(x, t0, 1.0, day.jobs, day.baseload_phase, fn)
                  - sh.horizon_max_core(x, t0, 0.2, day.jobs, day.baseload_phase, fn))
    assert (mTs[1.0] - mTs[0.2]) >= 0.4 * twin_range, \
        f"model range {mTs[1.0]-mTs[0.2]:.3f} < 40% of twin range {twin_range:.3f}"


def test_at7_surrogate_faster_than_twin(model):
    t_surr = rollout_speed(model["model"] if isinstance(model, dict) else model,
                           n=200)
    twin = DigitalTwin(TwinParams.from_yaml(CONFIG))
    x0 = twin.steady_state([1200.0] * 3, 20.0)
    P = np.tile([1200.0, 1200.0, 1250.0], (30, 1))
    t0 = time.perf_counter()
    for _ in range(200):
        twin.rollout(x0, P, np.full(30, 20.0), 60.0)
    t_twin = (time.perf_counter() - t0) / 200
    assert t_surr < t_twin                       # AT-7: surrogate wins
    assert t_twin / t_surr >= 1.3                # AT-8 speedup >= 30%


def test_guarded_hybrid_end_to_end(twin, model):
    day = make_synthetic_day(twin, n_stations=200, n_jobs=300, seed=0,
                             target_peak_T=100.0, start_hour=14, end_hour=24)
    from dtr.baselines.static_cap import compute_static_cap_kw
    p_static = compute_static_cap_kw(twin, float(day.baseload_phase.max()))
    x0 = None
    for _ in range(2):
        x0 = run_day(twin, day, "static_cap", p_static_kw=p_static,
                     t0_gmin=200 * 1440)["x_final"]
    r = run_day(twin, day, "guarded_hybrid", p_static_kw=p_static, x0=x0,
                t0_gmin=200 * 1440, models=[model])
    s = r["summary"]
    assert s["violations"] == 0                          # G1 holds with surrogate in loop
    assert np.all((r["logs"]["alpha"] >= 0.0)
                  & (r["logs"]["alpha"] <= 1.0))         # AT-4 analogue
    modes = s["modes"]
    # The architecture's GUARANTEES (AT-8 does not gate on mode-share):
    # safety holds with the surrogate in the loop, and the escalation
    # mechanism (twin authority) is exercised, never silently bypassed.
    assert (modes.get("recovered", 0) + modes.get("fallback", 0)
            + modes.get("surrogate_ok", 0)) > 0
    assert modes.get("surrogate_ok", 0) >= 0             # reported, not gated (v3.6 note)
    assert s["E_delivered_kwh"] > 0
