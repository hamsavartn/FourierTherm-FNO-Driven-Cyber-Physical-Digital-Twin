"""E5 ablation (PRD §6): persistence vs MLP vs FNO vs GSNO on identical held-out
realized windows. Diagnoses the surrogate accuracy plateau (issue 1) and
evaluates the GSNO (Graph Spectral Neural Operator) for over-squashing
mitigation against the FNO baseline. Output: outputs/e5_ablation.csv"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from torch import nn

from dtr.surrogate.train import WindowDataset, fit_standardizer, load_model

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main() -> None:
    arr = dict(np.load(os.path.join(ROOT, "outputs", "scenarios.npz")))
    rng = np.random.default_rng(7)
    n = arr["P_phase"].shape[0]
    val_ids = rng.permutation(n)[int(0.8 * n):]
    ds = WindowDataset(arr, val_ids, stride=20, augment_frac=0.0)
    x, y = ds.x, ds.y
    print(f"val windows: {len(y)} (realized only, {n - int(0.8*n)} scenarios)")

    # ---- persistence: hold the last observed max-core value ----
    pers = x[:, 4, -1].unsqueeze(1).repeat(1, 30)
    rmse = lambda p: float((p - y).pow(2).mean().sqrt())
    rows = [("persistence", rmse(pers), 0.0)]

    # ---- MLP baseline: flattened window -> 30 outputs ----
    torch.manual_seed(0)
    mean, std = fit_standardizer(ds)
    xs = (x - mean) / std
    ys = (y - y.mean().view(1, 1)) / y.std().clamp_min(1e-6).view(1, 1)
    n_tr = int(0.85 * len(xs))
    mlp = nn.Sequential(nn.Flatten(), nn.Linear(12 * 90, 256), nn.GELU(),
                        nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 30))
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    t0 = time.perf_counter()
    for ep in range(120):
        sel = torch.randperm(n_tr)[:512]
        opt.zero_grad()
        loss = ((mlp(xs[sel]) - ys[sel]) ** 2).mean()
        loss.backward()
        opt.step()
    train_s = time.perf_counter() - t0
    with torch.no_grad():
        pred = mlp(xs[n_tr:]) * y.std() + y.mean()
    mlp_rmse = float((pred - y[n_tr:]).pow(2).mean().sqrt())
    rows.append(("MLP", mlp_rmse, train_s))

    # ---- FNO (the shipped artifact) ----
    art = os.path.join(ROOT, "outputs", "fno_model.pt")
    if os.path.exists(art):
        bundle = load_model(art)
        fno = bundle["model"]
        with torch.no_grad():
            pf = torch.cat([fno(((x[i:i+512] - bundle["mean"]) / bundle["std"]))
                            for i in range(0, len(x), 512)])
        pf = pf * bundle["y_std"] + bundle["y_mean"]
        rows.append(("FNO (shipped)", rmse(pf), bundle["val_rmse"]))

    # ---- GSNO (Graph Spectral Neural Operator) ----
    gsno_art = os.path.join(ROOT, "outputs", "gsno_model.pt")
    if os.path.exists(gsno_art):
        try:
            from dtr.surrogate.gsno_thermal import load_gsno
            bundle = load_gsno(gsno_art)
            gsno = bundle["model"]
            with torch.no_grad():
                pg = torch.cat([gsno(((x[i:i+512] - bundle["mean"]) / bundle["std"]))
                                for i in range(0, len(x), 512)])
            pg = pg * bundle["y_std"] + bundle["y_mean"]
            rows.append(("GSNO", rmse(pg), bundle["val_rmse"]))
        except ImportError:
            rows.append(("GSNO", float('nan'), 0.0))
    else:
        rows.append(("GSNO", float('nan'), 0.0))

    import csv
    out = os.path.join(ROOT, "outputs", "e5_ablation.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "val_rmse_degC", "note"])
        for name, r, note in rows:
            w.writerow([name, round(r, 3), round(note, 1)])
            print(f"{name:14s} val RMSE {r:.3f} degC  (note: {note})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
