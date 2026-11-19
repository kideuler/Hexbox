import sys; sys.path.insert(0, ".")
import numpy as np, matplotlib.pyplot as plt
from hexbox.mesh import HexMesh
from hexbox.smooth import laplacian, respace_block_edges, scaled_jacobian
from hexbox.viz import plot_net, plot_3d
m = HexMesh.structured(6, 6, 6)
b = m.base
S = (b[:,0] >= 3) & (b[:,1] < 3) & (b[:,2] < 3)       # corner box at (1,0,0)
m.pillow(S)
respace_block_edges(m); laplacian(m)
sj = scaled_jacobian(m.X, m.H)
print("hexes", len(m.H), "min SJ %.3f" % sj.min())
fig = plt.figure(figsize=(12, 7))
plot_net(m, fig.add_subplot(1, 2, 1), "corner sheet at (1,0,0): net")
plot_3d(m, fig.add_subplot(1, 2, 2, projection="3d"), "3D", azim=-50)
fig.savefig("figs/01_corner_sheet.png", dpi=110, bbox_inches="tight")
