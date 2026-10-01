"""1-D Fourier Neural Operator surrogate (PRD §5.3a, v3.3/v3.5 spec).

Maps a 60-minute context window to the next 30 minutes of max-core
temperature. Implementation: the time axis is padded with 30 zero slots
(context || future) and the Fourier layers run on the full uniform
90-point grid -- the spectral kernel is global, so context information
propagates into the future slots, and the pointwise head is read only
there (the pad-and-mask pattern from the v3.3 ISSUE-03 adjudication;
FFT size 90 -> Nyquist 45 > 16 modes, aliasing-free). This preserves the
operator's grid-invariance property while producing off-grid-horizon
predictions in one shot -- no autoregressive stepping.

Input channels (12): P_a, P_b, P_c [kW/1e3], T_inf, T_a, T_b, T_c,
T_i, T_s, T_x, s, R4_eff [per-m]. Output (B, 30): max-core temperature.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, modes: int):
        super().__init__()
        self.modes = modes
        self.scale = 1.0 / (in_ch * out_ch)
        self.weights = nn.Parameter(
            self.scale * torch.randn(in_ch, out_ch, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T = x.shape
        xf = torch.fft.rfft(x, dim=-1)                 # (B, C, T//2+1)
        m = min(self.modes, xf.shape[-1])
        out = torch.einsum("bcm,com->bom", xf[..., :m], self.weights[..., :m])
        of = torch.zeros(B, self.weights.shape[1], xf.shape[-1],
                         dtype=torch.cfloat, device=x.device)
        of[..., :m] = out
        return torch.fft.irfft(of, n=T, dim=-1)


class FNO1d(nn.Module):
    def __init__(self, in_ch: int = 12, width: int = 32, modes: int = 16,
                 n_layers: int = 4, t_ctx: int = 60, t_out: int = 30):
        super().__init__()
        self.t_ctx, self.t_out = t_ctx, t_out
        self.lift = nn.Linear(in_ch, width)
        self.specs = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.ws = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                  nn.Linear(width, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_ch, t_ctx + t_out = 90) -> (B, t_out)
        # The 90-slot grid carries the REALIZED context (slots 0-59) and the
        # CANDIDATE future profile (slots 60-89: the alpha-scaled dispatch the
        # bisection is querying, plus the known T_inf forecast). The spectral
        # kernel is global, so context mixes into future slots; the head reads
        # only the future slice. FFT size 90 -> Nyquist 45 > 16 modes.
        h = self.lift(x.transpose(1, 2)).transpose(1, 2)      # (B, W, 90)
        for spec, w in zip(self.specs, self.ws):
            h = F.gelu(spec(h) + w(h))
        fut = h[:, :, self.t_ctx:]                            # (B, W, t_out)
        return self.head(fut.transpose(1, 2)).squeeze(-1)     # (B, t_out)
