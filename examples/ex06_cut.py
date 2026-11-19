"""Cut-away views of the interior (faces coloured by scaled Jacobian of their hex)."""
import sys; sys.path.insert(0, ".")
import matplotlib.pyplot as plt
from hexbox.build import build_block
from hexbox.viz import plot_net, plot_cut
cases = {
    "corner +3 at (1,0,0)": [9,6,6,6, 6,6,9,6, 6,6,9,6],
    "wrap x2 around y-edge (1,*,0)": [8,6,8,6, 6,6,6,6, 6,6,8,8],
    "random: 6 14 7 9 13 8 8 9 12 11 8 7": [6,14,7,9, 13,8,8,9, 12,11,8,7],
}
fig = plt.figure(figsize=(16, 11))
for i, (name, n) in enumerate(cases.items()):
    print(name); m = build_block(n)
    plot_net(m, fig.add_subplot(2, 3, i + 1), name)
    # remove the octant around the corner (1,0,0)/(x>0.5,y<0.5,z<0.5) to look inside
    plot_cut(m, fig.add_subplot(2, 3, i + 4, projection="3d"), "cut-away (colour = scaled Jacobian)",
             keep=lambda c: ~((c[:, 0] > 0.5) & (c[:, 1] < 0.5) & (c[:, 2] > 0.35)), azim=-55)
fig.savefig("figs/06_cutaway.png", dpi=100, bbox_inches="tight")
