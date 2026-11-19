"""Two blocks meshed independently (e.g. on different ranks) share a face: identical face meshes."""
import sys; sys.path.insert(0, ".")
import numpy as np, matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from hexbox.deterministic import build_block_det, face_template
from hexbox.faceiso import same_topology, match_nodes
from hexbox.viz import face_of_quads, FACE_COL

# edge order: x-edges (y,z)=00,01,10,11 | y-edges (x,z)=00,01,10,11 | z-edges (x,y)=00,01,10,11
A = [11, 7, 7, 7,  9, 9, 9, 5,  6, 6, 8, 8]        # upper block: its bottom face z=0 is shared
B = [13, 11, 9, 7, 9, 9, 13, 9, 7, 7, 7, 7]        # lower block: its top face z=1 is shared
mA, mB = build_block_det(A), build_block_det(B)

def face(m, name):
    Q = m.boundary_quads(); return Q[face_of_quads(m.X, Q) == name]
QA, QB = face(mA, "z0"), face(mB, "z1")
same = same_topology(mA.X[:, :2], QA, mB.X[:, :2], QB)
UVt, Qt = face_template(11, 9, 7, 9)
M = match_nodes(mA.X[:, :2], QA, mB.X[:, :2], QB)
src, dst = np.fromiter(M, int), np.fromiter(M.values(), int)
dpos = np.abs(mA.X[src, :2] - mB.X[dst, :2]).max()
print("max node position difference on the shared face: %.1e" % dpos)
print("A's face == B's face:", same, "| == stand-alone template(11,9,7,9):", same_topology(mA.X[:, :2], QA, UVt, Qt))

fig = plt.figure(figsize=(16, 6))
ax = fig.add_subplot(1, 3, 1, projection="3d")
for m, dz, cut in ((mA, 1.0, None), (mB, 0.0, "z1")):
    Q = m.boundary_quads(); names = face_of_quads(m.X, Q)
    keep = names != cut if cut else np.ones(len(Q), bool)
    P = m.X[Q[keep]].copy(); P[:, :, 2] += dz
    ax.add_collection3d(Poly3DCollection(P, facecolors=[FACE_COL[n] for n in names[keep]], edgecolors="k", linewidths=0.3, alpha=0.95))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 2); ax.set_box_aspect((1, 1, 2)); ax.view_init(20, -60)
ax.set_title("block A (top) on block B (bottom)", fontsize=10)
for i, (m, Q, t) in enumerate(((mA, QA, "A's bottom face (built by A alone)"), (mB, QB, "B's top face (built by B alone)"))):
    ax = fig.add_subplot(1, 3, 2 + i)
    ax.add_collection(PolyCollection(m.X[Q][:, :, :2], facecolors="#dbe9f6", edgecolors="k", linewidths=0.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(t + f"\ncounts 11 (y=0), 9, 7 (y=1), 9 | same topology: {same}, max node gap {dpos:.0e}", fontsize=10)
fig.savefig("figs/09_two_blocks.png", dpi=100, bbox_inches="tight")
