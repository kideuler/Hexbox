"""Stress test: many random parity-valid assignments, report validity + quality."""
import sys, time; sys.path.insert(0, ".")
import numpy as np
from hexbox.build import build_block
from hexbox.sheets import parity_ok
from hexbox.smooth import scaled_jacobian
rng = np.random.default_rng(0)
lo, hi, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
rows = []
t0 = time.time()
while len(rows) < N:
    n = rng.integers(lo, hi + 1, 12)
    if not parity_ok(n):
        continue
    try:
        m = build_block(n, verbose=False)
        sj = scaled_jacobian(m.X, m.H)
        rows.append((sj.min(), sj.mean(), len(m.H)))
        if sj.min() <= 0: print('INVALID', n.tolist(), sj.min())
    except Exception as e:
        rows.append((np.nan, np.nan, 0)); print("FAIL", n, e)
r = np.array(rows)
print(f"counts in [{lo},{hi}]: {N} blocks, {time.time()-t0:.0f}s; valid (minSJ>0): {(r[:,0]>0).sum()}/{N}; "
      f"minSJ p5/median {np.nanpercentile(r[:,0],5):.2f}/{np.nanmedian(r[:,0]):.2f}; meanSJ median {np.nanmedian(r[:,1]):.2f}")
print("worst minSJ:", np.sort(r[:,0])[:5].round(3))
