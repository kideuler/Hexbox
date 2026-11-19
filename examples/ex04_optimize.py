"""Laplacian vs untangle+smooth optimisation on the hardest random case."""
import sys; sys.path.insert(0, ".")
import numpy as np, matplotlib.pyplot as plt
from hexbox.build import build_block
from hexbox.smooth import optimize, scaled_jacobian
from hexbox.viz import plot_net
n = [6, 14, 6, 13, 13, 12, 7, 13, 14, 13, 12, 8]
m = build_block(n)
sj0 = scaled_jacobian(m.X, m.H)
fig = plt.figure(figsize=(14, 8))
plot_net(m, fig.add_subplot(1, 3, 1), f"Laplacian: minSJ {sj0.min():.2f}")
optimize(m, verbose=True)
sj1 = scaled_jacobian(m.X, m.H)
plot_net(m, fig.add_subplot(1, 3, 2), f"optimised: minSJ {sj1.min():.2f}")
ax = fig.add_subplot(1, 3, 3)
ax.hist([sj0, sj1], bins=40, range=(-1, 1), label=["Laplacian", "optimised"]); ax.set_yscale("log"); ax.legend(); ax.set_xlabel("scaled Jacobian")
fig.savefig("figs/04_optimize.png", dpi=100, bbox_inches="tight")
print("min SJ: %.3f -> %.3f   mean %.3f -> %.3f" % (sj0.min(), sj1.min(), sj0.mean(), sj1.mean()))
