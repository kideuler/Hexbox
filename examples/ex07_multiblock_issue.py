"""Why face meshes must be a function of the face's own counts in a multi-block setting."""
import sys; sys.path.insert(0, ".")
import numpy as np, matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from hexbox.build import build_block
from hexbox.viz import face_of_quads

def face_poly(m, name, drop):
    Q = m.boundary_quads(); Q = Q[face_of_quads(m.X, Q) == name]
    return m.X[Q][:, :, [d for d in range(3) if d != drop]]

A = build_block([7,6,7,6, 7,6,7,6, 6,7,7,6], verbose=False)   # case B: corners (1,0,0)+(0,1,0)
B = build_block([7]*4 + [7]*4 + [5]*4, verbose=False)          # structured block underneath
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for ax, P, t in [(axs[0], face_poly(A, "z0", 2), "block A, bottom face (7,7,7,7)\nfrom A's 3D decomposition"),
                 (axs[1], face_poly(B, "z1", 2), "block B, top face (7,7,7,7)\nsame physical face, B's view")]:
    ax.add_collection(PolyCollection(P, facecolors="#dbe9f6", edgecolors="k", linewidths=0.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.set_title(t, fontsize=10); ax.axis("off")
fig.suptitle("Same shared face, same 4 counts, different meshes -> not conformal across blocks", fontsize=11)
fig.savefig("figs/07_multiblock_issue.png", dpi=100, bbox_inches="tight")
