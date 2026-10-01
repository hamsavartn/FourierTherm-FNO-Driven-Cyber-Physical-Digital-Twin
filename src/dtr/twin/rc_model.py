"""Thermal digital twin: per-core 3-phase RC ladder + soil-moisture hysteresis.

Implements PRD v3.6 section 2:
- three conductor nodes T_a, T_b, T_c, star-coupled through the common
  insulation/filler node (R_core = 3*R1 so the balanced-load limit equals the
  classic single-node ladder; a single node would average away the hottest
  core under phase imbalance -- Round-7 finding, trap 27)
- Van Wormer 50/50 dielectric-loss split (cores / filler node)
- sheath/armour eddy loss P_extra deposited in the sheath node
- state-dependent soil resistivity with dry-fraction s(t) and smooth sigmoid
  hysteresis (tau_wet >= 10 * tau_dry)
- Kusuda-Achenbach annual far-field boundary T_inf (NBS Report 8972)

State vector: x = [T_a, T_b, T_c, T_i, T_s, T_x, s].
Load input: per-phase feeder active power in kW (single-phase EVSEs mapped to
their phase; 3-phase stations split equally across phases).
All thermal quantities are per metre of cable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import yaml


@dataclass(frozen=True)
class TwinParams:
    # electrical
    V_ll: float
    pf: float
    R_ac: float
    lam_zs: float
    # per-core thermal
    R_core: float
    C_core: float
    # common chain
    R2: float
    C_i: float
    R3: float
    C_s: float
    C_x: float
    R4_wet: float
    # soil hysteresis
    rho_wet: float
    rho_dry: float
    T_crit: float
    k_sig: float
    tau_dry: float
    tau_wet: float
    # dielectric
    C_f: float  # F per metre
    U0: float
    tan_delta_20: float
    beta: float
    freq: float
    # Kusuda-Achenbach boundary
    T_mean: float
    T_amp: float
    alpha_soil: float
    depth: float
    t_shift_days: float
    # limits
    T_max: float = 90.0

    @classmethod
    def from_yaml(cls, path: str) -> "TwinParams":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        e, ct, k, so, di, bo, lim = (
            raw["electrical"], raw["core_thermal"], raw["common_thermal"],
            raw["soil"], raw["dielectric"], raw["boundary"], raw["limits"],
        )
        return cls(
            V_ll=float(e["V_ll"]), pf=float(e["pf"]),
            R_ac=float(e["R_ac"]), lam_zs=float(e["lam_zs"]),
            R_core=float(ct["R_core"]), C_core=float(ct["C_core"]),
            R2=float(k["R2"]), C_i=float(k["C_i"]), R3=float(k["R3"]),
            C_s=float(k["C_s"]), C_x=float(k["C_x"]), R4_wet=float(k["R4_wet"]),
            rho_wet=float(so["rho_wet"]), rho_dry=float(so["rho_dry"]),
            T_crit=float(so["T_crit"]), k_sig=float(so["k_sig"]),
            tau_dry=float(so["tau_dry"]), tau_wet=float(so["tau_wet"]),
            C_f=float(di["capacitance_per_m"]), U0=float(di["U0"]),
            tan_delta_20=float(di["tan_delta_20"]), beta=float(di["beta"]),
            freq=float(di["freq"]),
            T_mean=float(bo["T_mean"]), T_amp=float(bo["T_amp"]),
            alpha_soil=float(bo["alpha_soil"]), depth=float(bo["depth"]),
            t_shift_days=float(bo["t_shift_days"]),
            T_max=float(lim["T_max"]),
        )


def phase_currents(P_phases_kw, V_ll: float, pf: float):
    """Per-phase RMS current [A] from per-phase active power [kW]."""
    V_ph = V_ll / math.sqrt(3.0)
    return [P * 1000.0 / (V_ph * pf) for P in P_phases_kw]


class DigitalTwin:
    STATE_DIM = 7  # [T_a, T_b, T_c, T_i, T_s, T_x, s]

    def __init__(self, params: TwinParams):
        self.p = params
        self.omega = 2.0 * math.pi * params.freq

    # ---------------- boundary ----------------
    def kusuda_T_inf(self, t_sec: float) -> float:
        """Far-field soil temperature [degC] at t seconds since Jan 1."""
        p = self.p
        year_sec = 365.0 * 86400.0
        dr = p.depth * math.sqrt(math.pi / (p.alpha_soil * year_sec))
        phase = 2.0 * math.pi * (t_sec - p.t_shift_days * 86400.0) / year_sec - dr
        return p.T_mean - p.T_amp * math.exp(-dr) * math.cos(phase)

    # ---------------- losses ----------------
    def dielectric_loss(self, T_ref: float) -> float:
        """Total dielectric loss [W/m] at reference core temperature."""
        p = self.p
        tan = p.tan_delta_20 * math.exp(p.beta * (T_ref - 20.0))
        return self.omega * p.C_f * p.U0 ** 2 * tan

    # ---------------- dynamics ----------------
    def deriv(self, x: np.ndarray, P_phases_kw, T_inf: float) -> np.ndarray:
        p = self.p
        Ta, Tb, Tc, Ti, Ts, Tx, s = x
        I = phase_currents(P_phases_kw, p.V_ll, p.pf)
        p_core = [p.R_ac * i * i for i in I]  # per-core Joule, W/m
        p_extra = (p.lam_zs - 1.0) * p.R_ac * sum(i * i for i in I)
        cores = (Ta, Tb, Tc)
        wd = [self.dielectric_loss(T) for T in cores]
        wd_common = self.dielectric_loss(sum(cores) / 3.0)
        rho = p.rho_wet + s * (p.rho_dry - p.rho_wet)
        R4 = p.R4_wet * rho / p.rho_wet

        d = np.empty(self.STATE_DIM)
        for k in range(3):
            d[k] = (p_core[k] + wd[k] / 6.0 - (cores[k] - Ti) / p.R_core) / p.C_core
        d[3] = (sum((cores[k] - Ti) / p.R_core for k in range(3))
                + wd_common / 2.0 - (Ti - Ts) / p.R2) / p.C_i
        d[4] = ((Ti - Ts) / p.R2 + p_extra - (Ts - Tx) / p.R3) / p.C_s
        d[5] = ((Ts - Tx) / p.R3 - (Tx - T_inf) / R4) / p.C_x

        hot = 1.0 / (1.0 + math.exp(-p.k_sig * (Tx - p.T_crit)))
        cool = 1.0 / (1.0 + math.exp(-p.k_sig * (p.T_crit - Tx)))
        d[6] = hot * (1.0 - s) / p.tau_dry - cool * s / p.tau_wet
        return d

    def step(self, x, P_phases_kw, T_inf, dt: float) -> np.ndarray:
        from dtr.twin.integrator import rk4_step
        return rk4_step(lambda xx: self.deriv(xx, P_phases_kw, T_inf), np.asarray(x, float), dt)

    def rollout(self, x0, P_series, T_inf_series, dt: float) -> dict:
        """Integrate over a profile. P_series: (N,3) kW; T_inf_series: (N,) degC."""
        n = len(P_series)
        x = np.asarray(x0, dtype=float).copy()
        out = np.empty((n, self.STATE_DIM))
        for i in range(n):
            x = self.step(x, P_series[i], T_inf_series[i], dt)
            out[i] = x
        return {
            "states": out,
            "T_cores": out[:, :3],
            "max_core": out[:, :3].max(axis=1),
            "s": out[:, 6],
        }

    def rollout_solve_ivp(self, x0, P_series, T_inf_series, dt: float) -> np.ndarray:
        """Reference integration (piecewise-constant inputs, matching RK4 semantics)."""
        from scipy.integrate import solve_ivp
        n = len(P_series)
        P = np.asarray(P_series, dtype=float)
        T = np.asarray(T_inf_series, dtype=float)

        def f(t, x):
            i = min(int(t // dt), n - 1)
            return self.deriv(x, P[i], T[i])

        t_eval = np.arange(1, n + 1) * dt
        sol = solve_ivp(f, (0.0, n * dt), np.asarray(x0, dtype=float),
                        method="RK45", rtol=1e-8, atol=1e-10, t_eval=t_eval)
        return sol.y.T

    # ---------------- steady state ----------------
    def steady_state(self, P_phases_kw, T_inf: float, s: float = 0.0) -> np.ndarray:
        """Analytic steady state at fixed s (linear solve; tan delta at 20 degC --
        iteration immaterial at 11 kV, asserted by test_dielectric_materiality)."""
        p = self.p
        I = phase_currents(P_phases_kw, p.V_ll, p.pf)
        pj = [p.R_ac * i * i for i in I]
        p_extra = (p.lam_zs - 1.0) * p.R_ac * sum(i * i for i in I)
        wd = self.dielectric_loss(20.0)
        rho = p.rho_wet + s * (p.rho_dry - p.rho_wet)
        R4 = p.R4_wet * rho / p.rho_wet

        A = np.zeros((6, 6))
        b = np.zeros(6)
        for k in range(3):
            A[k, k] = 1.0 / p.R_core
            A[k, 3] = -1.0 / p.R_core
            b[k] = pj[k] + wd / 6.0
        A[3, 0] = A[3, 1] = A[3, 2] = 1.0 / p.R_core
        A[3, 3] = -(3.0 / p.R_core + 1.0 / p.R2)
        A[3, 4] = 1.0 / p.R2
        b[3] = -wd / 2.0
        A[4, 3] = 1.0 / p.R2
        A[4, 4] = -(1.0 / p.R2 + 1.0 / p.R3)
        A[4, 5] = 1.0 / p.R3
        b[4] = -p_extra
        A[5, 4] = 1.0 / p.R3
        A[5, 5] = -(1.0 / p.R3 + 1.0 / R4)
        b[5] = -T_inf / R4
        return np.append(np.linalg.solve(A, b), s)
