"""Step 5 entry point: generate (or reuse) the scenario grid, train the FNO
surrogate, and report val RMSE + surrogate-vs-twin horizon-rollout speed
(AT-7 preview). Full run: python scripts/04_train_surrogate.py [--n 800]
[--epochs 60]. Smoke defaults keep the whole path under ~2 min on CPU."""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from dtr.surrogate.scenario_grid import generate_scenarios, load_scenarios, save_scenarios
from dtr.surrogate.train import rollout_speed, train
from dtr.twin.rc_model import DigitalTwin, TwinParams

ROOT = os.path.join(os.path.dirname(__file__), "..")
SCEN = os.path.join(ROOT, "outputs", "scenarios.npz")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--modes", type=int, default=16)
    args = ap.parse_args()

    if os.path.exists(SCEN):
        arrays = load_scenarios(SCEN)
        if arrays["P_phase"].shape[0] >= args.n:
            arrays = {k: v[: args.n] for k, v in arrays.items()}
            print(f"loaded {args.n} scenarios from cache")
        else:
            arrays = None
    else:
        arrays = None
    if arrays is None:
        t0 = time.perf_counter()
        params = TwinParams.from_yaml(os.path.join(ROOT, "configs", "twin.yaml"))
        arrays = generate_scenarios(params, n_scenarios=args.n, seed=args.seed)
        os.makedirs(os.path.dirname(SCEN), exist_ok=True)
        save_scenarios(arrays, SCEN)
        print(f"generated {args.n} scenarios in "
              f"{time.perf_counter() - t0:.0f} s "
              f"(dry-out share: {arrays['dryout'].mean():.0%})")

    res = train(arrays, epochs=args.epochs, seed=args.seed,
                width=args.width, modes_n=args.modes,
                base_params=TwinParams.from_yaml(
                    os.path.join(ROOT, "configs", "twin.yaml")))
    from dtr.surrogate.train import save_model
    save_model(res, os.path.join(ROOT, "outputs", "fno_model.pt"))
    print(f"\nval RMSE: {res['val_rmse']:.3f} degC | "
          f"windows: {res['n_windows_train']} train / {res['n_windows_val']} val | "
          f"train time {res['train_s']:.0f} s")

    # AT-7 preview: twin horizon rollout (30 RK4 steps) vs surrogate forward
    params = TwinParams.from_yaml(os.path.join(ROOT, "configs", "twin.yaml"))
    twin = DigitalTwin(params)
    x0 = twin.steady_state([1200.0] * 3, 20.0)
    P = np.tile([1200.0, 1200.0, 1250.0], (30, 1))
    t0 = time.perf_counter()
    for _ in range(200):
        twin.rollout(x0, P, np.full(30, 20.0), 60.0)
    t_twin = (time.perf_counter() - t0) / 200
    t_fno = rollout_speed(res["model"])
    print(f"horizon rollout: twin {t_twin * 1e3:.1f} ms vs surrogate "
          f"{t_fno * 1e3:.2f} ms -> speedup x{t_twin / t_fno:.0f}")


if __name__ == "__main__":
    main()
