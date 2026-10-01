import torch
from torch.utils.data import DataLoader
from dtr.surrogate.train import WindowDataset

ds = WindowDataset("outputs/scenarios.npz")
print(f"Dataset size: {len(ds)}")

for pair_cf, pair_re, da in ds.pairs[:10]:
    xa = ds.x[pair_cf]
    ya = ds.y[pair_cf]
    xb = ds.x[pair_re]
    yb = ds.y[pair_re]
    
    print(f"alpha diff: {da}")
    print(f"xa P_fut mean: {xa[0:3, 60:90].mean().item():.4f}, xb P_fut mean: {xb[0:3, 60:90].mean().item():.4f}")
    print(f"ya mean: {ya.mean().item():.4f}, yb mean: {yb.mean().item():.4f}")
    print(f"ya max: {ya.max().item():.4f}, yb max: {yb.max().item():.4f}")
    print(f"slope_true max: {((ya - yb)/da).max().item():.4f}, slope_true min: {((ya - yb)/da).min().item():.4f}")
