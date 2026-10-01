"""Static-cap baseline (PRD §5.7): the largest CONSTANT total EV power that
keeps the twin's analytic steady state at or below T_max, evaluated at the
worst-case far-field soil temperature (Kusuda annual maximum) and the day's
peak baseload. This is the 'rate for the worst case' philosophy -- computed
once, never adapted. Note: steady_state evaluates tan delta at 20 degC;
the difference is ~0.003 degC at 11 kV (PRD v3.4 materiality note)."""
from __future__ import annotations

import math


def kusuda_annual_max(twin) -> float:
    p = twin.p
    dr = p.depth * math.sqrt(math.pi / (p.alpha_soil * 365.0 * 86400.0))
    return p.T_mean + p.T_amp * math.exp(-dr)


def compute_static_cap_kw(twin, base_peak_per_phase_kw: float,
                          T_max: float | None = None,
                          T_worst: float | None = None) -> float:
    """Max total EV charging power [kW] with zero thermal headroom risk under
    worst-case assumptions. Returns 0 if the baseload alone already breaches
    the limit at worst-case soil temperature. T_worst overrides the annual
    maximum (e.g. a SEASONAL rating basis, standard utility practice)."""
    T_worst = kusuda_annual_max(twin) if T_worst is None else T_worst
    T_max = twin.p.T_max if T_max is None else T_max

    def max_core(ev_total_kw: float) -> float:
        per = base_peak_per_phase_kw + ev_total_kw / 3.0
        return twin.steady_state([per] * 3, T_worst)[:3].max()

    if max_core(0.0) > T_max:
        return 0.0
    hi = 1.0
    while max_core(hi) <= T_max:          # exponential upper-bound search
        hi *= 2.0
        if hi > 1e7:
            break
    lo = hi / 2.0
    for _ in range(40):                   # bisection to ~1 W precision
        mid = 0.5 * (lo + hi)
        if max_core(mid) <= T_max:
            lo = mid
        else:
            hi = mid
    return lo
