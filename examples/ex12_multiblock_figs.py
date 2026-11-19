"""Pictures for the multi-block tests: figs/11_grid.png, figs/12_ball.png, figs/13_free_mode_control.png"""
import sys
sys.path.insert(0, ".")
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from hexbox.viz import draw_hexes, draw_exploded, draw_shared_face, block_face_quads, BLOCK_COLORS, equal_box
from hexbox.mesh import HEX_FACES
from hexbox.multiblock import _corner, FACE_CYCLE
from hexbox.sheets import other
from ex11_multiblock_tests import GRID, H_GRID, SPHERE, H_SPHERE


def hex_colors(parts):
    return np.vstack([np.repeat(BLOCK_COLORS[b % 20][None], len(p["H"]), 0) for b, p in enumerate(parts)])


def most_irregular_shared_face(mb):
    best = None
    for pair in mb.block_faces()[1]:
        (b, f, s) = pair[0]
        a, bb, c, d = mb.canonical_frame(b, f, s)[3]
        score = abs(a - c) + abs(bb - d)
        if best is None or score > best[0]:
            best = (score, pair, (a, bb, c, d))
    return best[1], best[2]


def canonical_corners(mb, b, f, s):
    p, q = other(f)
    gid = [int(mb.blocks[b][_corner(f, s, p, q, u, v)]) for u, v in FACE_CYCLE]
    o = int(np.argmin(gid)); n1, n2 = (o + 1) % 4, (o - 1) % 4
    nu, nv = (n1, n2) if gid[n1] < gid[n2] else (n2, n1)
    return mb.V[gid[o]], mb.V[gid[nu]], mb.V[gid[nv]]


def overlay(ax, mb, parts, pair, counts, what):
    (b1, f1, s1), (b2, f2, s2) = pair
    draw_shared_face(ax, parts[b1], (f1, s1), parts[b2], (f2, s2), corners=canonical_corners(mb, b1, f1, s1), title=
                     f"{what}: the face between blocks {b1} and {b2}\ncounts {counts}; "
                     f"blue = block {b1}'s mesh, red dashed = block {b2}'s")


def unmatched_quads(mb):
    F = np.sort(mb.H[:, HEX_FACES].reshape(-1, 4), axis=1)
    uq, cnt = np.unique(F, axis=0, return_counts=True)
    once = {tuple(f) for f in uq[cnt == 1]}
    bnd, _ = mb.block_faces()
    expect = set()
    for b, f, s in bnd:
        Q = block_face_quads(mb.parts[b], f, s)
        expect |= {tuple(sorted(mb.node_of[q + mb.off[b]])) for q in Q}
    bad = once - expect
    return np.array(sorted(bad)) if bad else np.zeros((0, 4), int)


def grid_figure():
    mb = GRID(); mb.assign_counts(H_GRID)
    parts = mb.mesh_blocks(); mb.assemble(parts); rep = mb.verify()
    fig = plt.figure(figsize=(18, 6.2))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    draw_exploded(ax, parts, gap=0.45, lw=0.2); ax.view_init(22, -130); ax.set_axis_off()
    ax.set_title("12 blocks, each meshed on its own (exploded)", fontsize=10)
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    c = mb.X[mb.H].mean(1)
    keep = ~((c[:, 2] > 1.5) & (c[:, 1] < 1.5))
    draw_hexes(ax, mb.X, mb.H, hex_colors(parts), keep, lw=0.2); ax.view_init(22, -130); ax.set_axis_off()
    ax.set_title(f"assembled, cut open through the top blocks, coloured by block\n{rep['hexes']} hexes, conformal: "
                 f"{rep['conformal']}, unmatched quads: {rep['unmatched_quads']}, min SJ {rep['min_sj']:.2f}", fontsize=10)
    ax = fig.add_subplot(1, 3, 3)
    pair, counts = most_irregular_shared_face(mb)
    overlay(ax, mb, parts, pair, counts, "deterministic faces")
    fig.savefig("figs/11_grid.png", dpi=110, bbox_inches="tight")
    return mb, parts


def ball_figure():
    mb = SPHERE(); mb.assign_counts(H_SPHERE)
    parts = mb.mesh_blocks(); mb.assemble(parts); rep = mb.verify()
    fig = plt.figure(figsize=(18, 6.2))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    draw_exploded(ax, parts, gap=0.6, lw=0.15); ax.view_init(20, -40); ax.set_axis_off()
    ax.set_title("7 blocks (exploded): centre cube + 6 shell blocks\ncurved sphere edges; 3 blocks meet at each radial edge", fontsize=10)
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    c = mb.X[mb.H].mean(1)
    keep = c[:, 2] < 0
    draw_hexes(ax, mb.X, mb.H, hex_colors(parts), keep, lw=0.15); ax.view_init(58, -70); ax.set_axis_off()
    ax.set_title(f"assembled, upper half removed; finer toward +x (right)\n{rep['hexes']} hexes, conformal: {rep['conformal']}, "
                 f"unmatched quads: {rep['unmatched_quads']}, min SJ {rep['min_sj']:.2f}", fontsize=10)
    ax = fig.add_subplot(1, 3, 3)
    pair, counts = most_irregular_shared_face(mb)
    overlay(ax, mb, parts, pair, counts, "curved interface")
    fig.savefig("figs/12_ball.png", dpi=110, bbox_inches="tight")


def free_mode_figure():
    mb = GRID(); mb.assign_counts(H_GRID)
    parts = mb.mesh_blocks(mode="free"); mb.assemble(parts, by="position"); rep = mb.verify()
    bad = {tuple(x) for x in unmatched_quads(mb)}
    # unmatched quads seen from the first block of every shared face (drawing both sides z-fights)
    best, red = None, []
    for pair in mb.block_faces()[1]:
        (b1, f1, s1) = pair[0]
        Q = block_face_quads(parts[b1], f1, s1)
        miss = [q for q in Q if tuple(sorted(mb.node_of[q + mb.off[b1]])) in bad]
        red += [parts[b1]["X"][q] for q in miss]
        if best is None or len(miss) > best[0]:
            best = (len(miss), pair)
    fig = plt.figure(figsize=(13, 6.2))
    ax = fig.add_subplot(1, 2, 1)
    (b1, f1, s1) = best[1][0]
    overlay(ax, mb, parts, best[1], mb.canonical_frame(b1, f1, s1)[3], "free mode (control)")
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    Q, _ = None, None
    from hexbox.viz import surface_of
    Q, _ = surface_of(mb.H, np.ones(len(mb.H), bool))
    ax.add_collection3d(Poly3DCollection(mb.X[Q], facecolors=(0.8, 0.8, 0.8, 0.05), edgecolors=(0, 0, 0, 0.08), linewidths=0.2))
    if red:
        ax.add_collection3d(Poly3DCollection(np.array(red), facecolors="#d62728", edgecolors="k", linewidths=0.25))
    equal_box(ax, mb.X); ax.view_init(22, -130); ax.set_axis_off()
    ax.set_title(f"free mode: {rep['unmatched_quads']} interface quads have no partner (red)\n"
                 f"each block chose its own face meshes", fontsize=10)
    fig.savefig("figs/13_free_mode_control.png", dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    which = sys.argv[1:] or ["grid", "ball", "free"]
    if "grid" in which: grid_figure()
    if "ball" in which: ball_figure()
    if "free" in which: free_mode_figure()
