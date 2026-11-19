"""The deterministic face templates: mesh = f(a, b, c, d) only."""
import sys; sys.path.insert(0, ".")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from hexbox.deterministic import face_template
cases = [(7, 7, 7, 7), (9, 7, 7, 7), (13, 7, 7, 7), (9, 9, 7, 7), (11, 9, 7, 7), (14, 10, 6, 6), (8, 12, 8, 6), (13, 5, 5, 9)]
fig, axs = plt.subplots(2, 4, figsize=(16, 8.4))
for ax, c in zip(axs.ravel(), cases):
    UV, Q = face_template(*c)
    ax.add_collection(PolyCollection(UV[Q], facecolors="#dbe9f6", edgecolors="k", linewidths=0.6))
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("bottom %d, right %d, top %d, left %d" % c, fontsize=10)
fig.suptitle("Deterministic faces: structured core min(a,c) x min(b,d) + nested U-turns on the longer side "
             "(collar L=2, depth D=2)", fontsize=11)
fig.savefig("figs/10_face_templates.png", dpi=100, bbox_inches="tight")
