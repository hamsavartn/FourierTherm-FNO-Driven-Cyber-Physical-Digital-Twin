"""Step 3 validation (PRD §10): AT-5 analytical, AT-6 RK4 vs solve_ivp,
twin physics invariants, Round-7 per-core divergence quantification."""
import math
import os

import numpy as np
import pytest

from dtr.twin.rc_model import DigitalTwin, TwinParams

CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "twin.yaml")


@pytest.fixture(scope="module")
def twin():
    return DigitalTwin(TwinParams.from_yaml(CONFIG))


def x_uniform(t=20.0):
    return np.array([t, t, t, t, t, t, 0.0])


# ---------------- AT-5: integrator vs closed form ----------------
def test_at5_rk4_vs_closed_form():
    # 2-node system C dT/dt = P - (T - T_inf)/R has exact solution
    # T(t) = T_ss + (T0 - T_ss) exp(-t/tau), tau = C*R.
    C, R, P, T_inf, T0 = 1.0e4, 0.5, 10.0, 20.0, 20.0
    T_ss, tau = T_inf + P * R, C * R
    dt = 1.0
    n = int(3.0 * tau / dt)
    x = np.array([T0])
    for _ in range(n):
        x = rk4(lambda z: np.array([(P - (z[0] - T_inf) / R) / C]), x, dt)
    t = n * dt
    exact = T_ss + (T0 - T_ss) * math.exp(-t / tau)
    assert abs(x[0] - exact) / (T_ss - T0) < 0.005  # AT-5: <0.5% of the swing


def rk4(f, x, dt):
    from dtr.twin.integrator import rk4_step
    return rk4_step(f, x, dt)


# ---------------- AT-6: RK4 vs solve_ivp on the full twin, 24 h ----------------
def test_at6_rk4_vs_solve_ivp(twin):
    n, dt = 24 * 60, 60.0
    rng = np.random.default_rng(0)
    P = 500.0 + 700.0 * rng.random((n, 3))   # kW per phase (~83..200 A)
    T_inf = np.full(n, 20.0)
    rk = twin.rollout(x_uniform(), P, T_inf, dt)["states"]
    ref = twin.rollout_solve_ivp(x_uniform(), P, T_inf, dt)
    assert np.abs(rk - ref).max() <= 0.1  # AT-6: <= 0.1 degC over 24 h


# ---------------- steady state: analytic vs dynamic wiring ----------------
def test_steady_state_consistency(twin):
    P = [1130.0, 1130.0, 1130.0]  # ~187 A per phase
    ss = twin.steady_state(P, 20.0)
    # start AT the analytic steady state: it must stay there (wiring check)
    out = twin.rollout(ss, np.tile(P, (24 * 60, 1)), np.full(24 * 60, 20.0), 60.0)["states"]
    assert np.abs(out[:, :6] - ss[:6]).max() < 0.05
    assert np.all(out[:, 6] >= -1e-9) and np.all(out[:, 6] <= 1.0 + 1e-9)


def test_steady_state_physical_ranges(twin):
    ss = twin.steady_state([1130.0] * 3, 20.0)
    Ta, Tb, Tc, Ti, Ts, Tx, s = ss
    assert Ta == pytest.approx(Tc, abs=1e-6)          # balanced cores equal
    assert Ta > Ti > Ts > Tx > 19.9                    # monotone heat flow
    assert 45.0 < Ta < 70.0                            # documented band (~187 A, PRD §2.3)


# ---------------- Round-7 fix quantified: per-core divergence ----------------
def test_per_core_divergence_under_imbalance(twin):
    # same TOTAL power: 200 A on phase a, 50 A on b and c vs 100 A balanced
    ss_bal = twin.steady_state([603.3] * 3, 20.0)
    ss_imb = twin.steady_state([1206.0, 302.0, 302.0], 20.0)
    # hottest core runs hotter than the balanced same-total case (sum I^2 effect)
    assert ss_imb[:3].max() > ss_bal[:3].max()
    # the cores spread apart -- invisible to any single-node model
    spread = ss_imb[:3].max() - ss_imb[:3].min()
    assert spread > 5.0


# ---------------- soil dry-out raises temperature (two-zone model) ----------------
def test_soil_dryout_raises_temperature(twin):
    wet = twin.steady_state([1200.0] * 3, 20.0, s=0.0)
    dry = twin.steady_state([1200.0] * 3, 20.0, s=1.0)
    assert dry[0] > wet[0] + 5.0  # dry soil materially derates the cable


# ---------------- hysteresis: re-wetting >= 10x slower than drying ----------------
def test_hysteresis_asymmetry(twin):
    hot = twin.deriv(x_uniform(t=60.0), [1800.0] * 3, 20.0)[6]   # T_x well above T_crit, s=0
    cold = twin.deriv(x_uniform(t=20.0), [0.0] * 3, 20.0)[6]     # T_x well below, s=0
    assert hot == pytest.approx(1.0 / twin.p.tau_dry, rel=0.01)
    assert cold == pytest.approx(-0.0, abs=1e-12)  # s=0: nothing to re-wet yet
    # with s=1 the picture inverts: drying stops, re-wetting creeps
    x_wet_soil = np.array([20.0] * 6 + [1.0])
    cold1 = twin.deriv(x_wet_soil, [0.0] * 3, 20.0)[6]
    assert cold1 == pytest.approx(-1.0 / twin.p.tau_wet, rel=0.01)
    assert abs(cold1) * 10.0 < 1.0 / twin.p.tau_dry  # >= 10x slower (tau_wet = 20x)


# ---------------- dielectric loss: temperature dependence + materiality ----------------
def test_dielectric_materiality(twin):
    wd20 = twin.dielectric_loss(20.0)
    wd90 = twin.dielectric_loss(90.0)
    assert wd90 > wd20                    # Arrhenius-type rise (v3.4)
    assert 0.0 < wd90 < 0.05              # ~0.03% of ~20 W/m Joule at rated load (PRD §2.2)


# ---------------- Kusuda-Achenbach boundary ----------------
def test_kusuda_boundary(twin):
    ts = np.arange(0, 365 * 86400, 3600)
    T = np.array([twin.kusuda_T_inf(t) for t in ts])
    assert abs(T.mean() - twin.p.T_mean) < 0.5              # annual mean preserved
    assert (T.max() - T.min()) / 2 < twin.p.T_amp           # attenuated with depth
    day = np.array([twin.kusuda_T_inf(t) for t in range(0, 86400, 60)])
    assert (day.max() - day.min()) < 0.5                    # diurnal swing ~0 at 0.8 m
