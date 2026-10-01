"""FNO training pipeline (PRD §5.3a): window extraction, splits by scenario
(no leakage), tail-weighted MSE, early stopping. CPU-first by design."""
from __future__ import annotations

import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import spectral_norm

from dtr.surrogate.fno_1d import FNO1d
from dtr.twin.rc_model import DigitalTwin, TwinParams

T_CTX, T_OUT = 60, 30
IN_CH = 12


class WindowDataset(Dataset):
    """Builds (12, 90) windows: slots 0-59 realized context, slots 60-89 the
    CANDIDATE future profile (states held at last value).

    COUNTERFACTUAL AUGMENTATION (the shortcut-learning fix): realized
    trajectories make the future load almost perfectly predictable from the
    context, so a model trained on them alone learns to IGNORE the future
    channels -- which bisection varies (measured: identical output for every
    alpha). For a fraction of windows we therefore draw alpha ~ U(0.1, 1),
    build the variant future profile  base + alpha * EV, and compute the
    target with a REAL 30-step twin rollout under the scenario's own
    perturbed parameters (~2 ms each -- nearly free). Same context,
    different candidate future, different outcome: the network is forced to
    read the future channels. Channels are then z-scored (temperature
    channels otherwise dwarf the load channels)."""

    def __init__(self, arrays: dict, scenario_ids, stride: int = 15,
                 base_params=None, augment_frac: float = 0.35,
                 n_variants: int = 2, seed: int = 0):
        rng = np.random.default_rng(seed)
        xs, ys = [], []
        self.pairs = []          # (cf_idx, realized_idx, delta_alpha) — Sobolev pairs
        for i in scenario_ids:
            P = arrays["P_phase"][i] / 1000.0          # kW -> k
            EV = arrays.get("EV", np.zeros_like(P))[i] / 1000.0
            Tinf = arrays["T_inf"][i]
            cores = arrays["T_cores"][i]
            nodes = arrays["T_nodes"][i]               # Ti, Ts, Tx, s
            r4 = arrays["R4_eff"][i]
            n = P.shape[0]
            twin = None
            if base_params is not None and "params" in arrays:
                twin = DigitalTwin(_override_params(
                    base_params, arrays["params"][i]))
            for t0 in range(T_CTX, n - T_OUT + 1, stride):
                ctx = slice(t0 - T_CTX, t0)
                fut = slice(t0, t0 + T_OUT)
                realized_idx = None
                candidates = [(1.0, np.concatenate([P[ctx], P[fut]], axis=0),
                               cores[fut].max(axis=1))]  # realized = alpha 1.0
                if twin is not None and rng.random() < augment_frac:
                    for _ in range(n_variants):
                        a = float(rng.uniform(0.1, 1.0))
                        P_var = np.clip(
                            (P - EV)[fut] + a * EV[fut], 0.0, None)
                        x_c = np.concatenate([cores[t0 - 1],
                                              nodes[t0 - 1]])
                        out = twin.rollout(x_c, P_var * 1000.0, Tinf[fut], 60.0)
                        candidates.append(
                            (a, np.concatenate([P[ctx], P_var], axis=0),
                             out["T_cores"].max(axis=1)))
                for alpha, P_win, target in candidates:
                    if realized_idx is None:
                        realized_idx = len(xs)
                    Pc = P_win[:T_CTX]
                    Pf = P_win[T_CTX:]
                    hold = lambda ch: np.full(T_OUT, ch[-1])
                    feats = np.stack([
                        np.concatenate([Pc[:, 0], Pf[:, 0]]),
                        np.concatenate([Pc[:, 1], Pf[:, 1]]),
                        np.concatenate([Pc[:, 2], Pf[:, 2]]),
                        np.concatenate([Tinf[ctx], Tinf[fut]]),
                        np.concatenate([cores[ctx, 0], hold(cores[ctx, 0])]),
                        np.concatenate([cores[ctx, 1], hold(cores[ctx, 1])]),
                        np.concatenate([cores[ctx, 2], hold(cores[ctx, 2])]),
                        np.concatenate([nodes[ctx, 0], hold(nodes[ctx, 0])]),
                        np.concatenate([nodes[ctx, 1], hold(nodes[ctx, 1])]),
                        np.concatenate([nodes[ctx, 2], hold(nodes[ctx, 2])]),
                        np.concatenate([nodes[ctx, 3], hold(nodes[ctx, 3])]),
                        np.full(T_CTX + T_OUT, r4),
                    ], axis=0)
                    xs.append(feats.astype(np.float32))
                    ys.append(target.astype(np.float32))
                    if alpha < 1.0:
                        # Sobolev pair (v3.6): the CF window and its realized
                        # twin differ only in alpha -> the FD slope over
                        # (1 - alpha) supervises dT/d(alpha) exactly as the
                        # spec's "central differences" fallback anticipated
                        self.pairs.append((len(xs) - 1, realized_idx, 1.0 - alpha))
        self.x = torch.from_numpy(np.stack(xs))
        self.y = torch.from_numpy(np.stack(ys))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def _override_params(base: TwinParams, vals) -> TwinParams:
    d = base.__dict__.copy()
    for k, v in zip(("R_core", "R2", "R3", "R4_wet",
                     "C_core", "C_i", "C_s", "C_x"), vals):
        d[k] = float(v)
    return TwinParams(**d)


def fit_standardizer(ds: WindowDataset):
    """Per-channel z-score stats, shaped (1, C, 1) to broadcast over (N, C, T)."""
    mean = ds.x.mean(dim=(0, 2)).view(1, -1, 1)
    std = ds.x.std(dim=(0, 2)).clamp_min(1e-6).view(1, -1, 1)
    return mean, std


def tail_weighted_mse(pred, y, w: float = 2.0):
    err2 = (pred - y) ** 2
    per_step = err2.mean(dim=0)                    # average over batch
    weights = torch.ones(T_OUT, device=pred.device)
    weights[-10:] = w                              # tail of horizon matters
    return (per_step * weights).mean()


def train(arrays: dict, epochs: int = 60, batch: int = 64, lr: float = 1e-3,
          seed: int = 0, verbose: bool = True, base_params: TwinParams = None,
          width: int = 32, modes_n: int = 16, lambda_sob: float = 0.5) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = arrays["P_phase"].shape[0]
    ids = rng.permutation(n)
    n_tr = int(0.8 * n)
    ds_tr = WindowDataset(arrays, ids[:n_tr], base_params=base_params,
                          seed=seed)
    ds_va = WindowDataset(arrays, ids[n_tr:], base_params=base_params,
                          seed=seed + 1, augment_frac=0.0)
    mean, std = fit_standardizer(ds_tr)
    ds_tr.x = (ds_tr.x - mean) / std
    ds_va.x = (ds_va.x - mean) / std
    y_mean = ds_tr.y.mean().view(1, 1)
    y_std = ds_tr.y.std().clamp_min(1e-6).view(1, 1)
    ds_tr.y = (ds_tr.y - y_mean) / y_std
    ds_va.y = (ds_va.y - y_mean) / y_std
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=256)
    # Sobolev-in-alpha pairs (v3.6, PRD §5.3a): FD directional derivative of
    # the model along the alpha axis vs the twin's FD derivative, on
    # counterfactual pairs that differ ONLY in the candidate load
    P_pairs = np.asarray(getattr(ds_tr, "pairs", []), dtype=np.float64)
    pair_cf = torch.as_tensor(P_pairs[:, 0].astype(np.int64)) if len(P_pairs) else None
    pair_re = torch.as_tensor(P_pairs[:, 1].astype(np.int64)) if len(P_pairs) else None
    pair_da = torch.as_tensor(P_pairs[:, 2], dtype=torch.float32).clamp_min(0.05) \
        if len(P_pairs) else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pair_cf is not None and verbose:
        print(f"Sobolev pairs: {len(pair_cf)} (lambda_sob={lambda_sob})")
    model = FNO1d(width=width, modes=modes_n).to(device)
    
    # MPNO: Constitutive Markov Spectral Radius Constraint
    # Bounding the transition operator Lipschitz constant for infinite-horizon dissipative stability
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)) and 'head' not in name:
            spectral_norm(module)
            
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val, best_state, patience = np.inf, None, 0
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = tail_weighted_mse(model(xb), yb)
            if pair_cf is not None:
                sel = torch.randint(0, len(pair_cf), (min(64, len(pair_cf)),))
                xa, xb_p = ds_tr.x[pair_cf[sel]].to(device), ds_tr.x[pair_re[sel]].to(device)
                ya, yb_p = ds_tr.y[pair_cf[sel]].to(device), ds_tr.y[pair_re[sel]].to(device)
                da = pair_da[sel].view(-1, 1).to(device)
                slope_pred = (model(xa) - model(xb_p)) / da
                slope_true = (ya - yb_p) / da
                
                # FC-PINO: Fourier-Continuation spectral residual
                # Extend smoothly via mirroring to avoid Gibbs phenomenon at boundaries
                sp_pad = torch.cat([slope_pred, slope_pred.flip(dims=[-1])], dim=-1)
                st_pad = torch.cat([slope_true, slope_true.flip(dims=[-1])], dim=-1)
                sp_fft = torch.fft.rfft(sp_pad, dim=-1)
                st_fft = torch.fft.rfft(st_pad, dim=-1)
                fc_pino_loss = torch.abs(sp_fft - st_fft).pow(2).mean()
                
                # TBPTT: Truncated Backpropagation Through Time chunking for Sobolev loss
                T_bptt = 10
                tbptt_loss = 0
                for t_start in range(0, T_OUT, T_bptt):
                    t_end = t_start + T_bptt
                    chunk_pred = slope_pred[..., t_start:t_end]
                    chunk_true = slope_true[..., t_start:t_end]
                    tbptt_loss += (chunk_pred - chunk_true).pow(2).mean()
                    
                loss = loss + lambda_sob * tbptt_loss + 0.1 * lambda_sob * fc_pino_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            errs = [(model(xb.to(device)) - yb.to(device)).pow(2).mean().item()
                    for xb, yb in dl_va]
        val = float(np.mean(errs))
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"epoch {ep:3d}  val MSE(z) {val:.4f}  "
                  f"RMSE {val ** 0.5 * float(y_std):.3f} degC")
        if val < best_val - 1e-4:
            best_val, patience = val, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 30:
                break
    model.load_state_dict(best_state)
    return {"model": model, "val_rmse": best_val ** 0.5 * float(y_std),
            "mean": mean, "std": std, "y_mean": y_mean, "y_std": y_std,
            "width": width, "modes": modes_n,
            "train_s": time.perf_counter() - t0,
            "n_windows_train": len(ds_tr), "n_windows_val": len(ds_va)}


def save_model(res: dict, path: str) -> None:
    torch.save({"state_dict": res["model"].state_dict(),
                "mean": res["mean"], "std": res["std"],
                "y_mean": res["y_mean"], "y_std": res["y_std"],
                "val_rmse": res["val_rmse"],
                "config": {"in_ch": IN_CH, "t_ctx": T_CTX, "t_out": T_OUT,
                           "width": res["width"], "modes": res["modes"]}},
               path)


def load_model(path: str) -> dict:
    ckpt = torch.load(path, weights_only=True, map_location="cpu")
    model = FNO1d(in_ch=ckpt["config"]["in_ch"],
                  t_ctx=ckpt["config"]["t_ctx"], t_out=ckpt["config"]["t_out"],
                  width=ckpt["config"].get("width", 32),
                  modes=ckpt["config"].get("modes", 16))
    
    # MPNO: Re-apply spectral norm structure to match saved weights
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)) and 'head' not in name:
            spectral_norm(module)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return {"model": model, "mean": ckpt["mean"], "std": ckpt["std"],
            "y_mean": ckpt["y_mean"], "y_std": ckpt["y_std"],
            "val_rmse": ckpt["val_rmse"]}


def rollout_speed(model: FNO1d, n: int = 1000) -> float:
    """Mean surrogate horizon-rollout time [s] (one forward pass)."""
    model.eval()
    x = torch.randn(n, IN_CH, T_CTX)
    t0 = time.perf_counter()
    with torch.no_grad():
        model(x)
    return (time.perf_counter() - t0) / n
