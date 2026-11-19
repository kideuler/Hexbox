"""Random parity-valid 12-edge assignments (single-block mode)."""
import sys; sys.path.insert(0, ".")
import numpy as np, matplotlib.pyplot as plt
from hexbox.build import build_block
from hexbox.sheets import parity_ok, EDGES
from hexbox.viz import plot_net, plot_3d
from hexbox.smooth import scaled_jacobian
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
found = []
while len(found) < 3:
    n = rng.integers(6, 15, 12)
    if parity_ok(n):
        found.append(n)
fig = plt.figure(figsize=(15, 10))
for i, n in enumerate(found):
    print("counts", dict(zip(["x%d%d" % e[1:] if e[0]==0 else ("y%d%d" % e[1:] if e[0]==1 else "z%d%d" % e[1:]) for e in EDGES], n)))
    m = build_block(n)
    plot_net(m, fig.add_subplot(2, 3, i + 1), " ".join(map(str, n)))
    ax = fig.add_subplot(2, 3, i + 4)
    ax.hist(scaled_jacobian(m.X, m.H), bins=40, range=(0, 1)); ax.set_xlabel("scaled Jacobian"); ax.set_yscale("log")
fig.savefig("figs/03_random.png", dpi=100, bbox_inches="tight")
