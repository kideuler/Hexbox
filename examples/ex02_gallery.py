"""Gallery of single-sheet-type transitions."""
import sys; sys.path.insert(0, ".")
import matplotlib.pyplot as plt
from hexbox.build import build_block
from hexbox.viz import plot_net, plot_3d
# EDGE ORDER: x-edges (y,z)=00,01,10,11 | y-edges (x,z)=00,01,10,11 | z-edges (x,y)=00,01,10,11
cases = {
    "wrap around y-edge (x=1,z=0)": [7,6,7,6, 6,6,6,6, 6,6,7,7],
    "uwrap on face z=0 (x-dir)":    [10,6,10,6, 6,6,6,6, 6,6,6,6],
    "uturn on x-edge (y=0,z=0)":    [8,6,6,6, 6,6,6,6, 6,6,6,6],
    "case B: corners (1,0,0)+(0,1,0)": [7,6,7,6, 7,6,7,6, 6,7,7,6],
}
fig = plt.figure(figsize=(16, 9))
for i, (name, n) in enumerate(cases.items()):
    print(name); m = build_block(n)
    plot_net(m, fig.add_subplot(2, 4, i + 1), name)
    plot_3d(m, fig.add_subplot(2, 4, i + 5, projection="3d"), azim=-50)
fig.savefig("figs/02_gallery.png", dpi=100, bbox_inches="tight")
