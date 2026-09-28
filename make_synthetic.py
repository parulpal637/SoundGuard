"""SMOKE TEST ONLY: generates fake fan-like audio so you can check the pipeline runs.
Never use results from this data in your submission - record real data with record.py."""
import os, numpy as np
from features import SR, WIN_SEC
rng = np.random.default_rng(0)
t = np.arange(SR * WIN_SEC) / SR

def fan(load, fault=False):
    level = 0.004 + 0.02 * load
    x = rng.normal(0, level, t.size)
    x = np.convolve(x, np.ones(4) / 4, mode="same")             # low-pass-ish noise
    x += level * 0.8 * np.sin(2 * np.pi * (90 + 120 * load) * t)  # blade-pass tone
    if fault:
        x += 0.01 * np.sin(2 * np.pi * 2600 * t) * (1 + np.sin(2 * np.pi * 7 * t))  # rattle
    return x.astype(np.float32)

def session(folder, name, load, n, fault=False):
    os.makedirs(folder, exist_ok=True)
    a = np.stack([fan(load + rng.normal(0, .03), fault) for _ in range(n)])
    c = np.clip(load * 100 + rng.normal(0, 3, n), 0, 100).astype(np.float32)
    np.savez_compressed(os.path.join(folder, name), audio=a, cpu=c)

for i, (nm, ld) in enumerate([("idle", .05), ("medium", .5), ("high", .95)]):
    session("data/normal", f"{nm}.npz", ld, 80)
session("data/abnormal", "rattle_high.npz", .95, 30, fault=True)
print("synthetic data written")
