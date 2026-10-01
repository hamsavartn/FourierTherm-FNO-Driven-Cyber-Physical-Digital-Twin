"""Fixed-step RK4 integrator (PRD §5.1). Validated against closed-form
solutions in tests/test_twin.py (AT-5) and scipy solve_ivp (AT-6)."""
import numpy as np


def rk4_step(f, x, dt: float) -> np.ndarray:
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
