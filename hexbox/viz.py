"""Pictures: unfolded cube net of the 6 face meshes, 3D surface view, quality."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# cube face -> (fixed axis, value, map (x,y,z)->(u,v) in the cross-shaped net)
NET = {
    "z0": (2, 0, lambda p: (1 + p[:, 0], 1 + p[:, 1])),
    "y0": (1, 0, lambda p: (1 + p[:, 0], 1 - p[:, 2])),
    "y1": (1, 1, lambda p: (1 + p[:, 0], 2 + p[:, 2])),
    "x0": (0, 0, lambda p: (1 - p[:, 2], 1 + p[:, 1])),
    "x1": (0, 1, lambda p: (2 + p[:, 2], 1 + p[:, 1])),
    "z1": (2, 1, lambda p: (1 + p[:, 0], 4 - p[:, 1])),
}
FACE_COL = {"z0": "#dbe9f6", "z1": "#dbe9f6", "y0": "#fde5cf", "y1": "#fde5cf", "x0": "#e0f0dc", "x1": "#e0f0dc"}


def face_of_quads(X, Q):
    """Name of the cube face each boundary quad lies on."""
    P = X[Q]                                   # (n,4,3)
    out = np.empty(len(Q), object)
    for name, (ax, val, _) in NET.items():
        out[np.all(np.abs(P[:, :, ax] - val) < 1e-9, axis=1)] = name
    return out


def irregular_nodes(mesh):
    """Interior/face nodes whose valence (#incident hexes) differs from the regular value."""
    val = np.bincount(mesh.H.ravel(), minlength=len(mesh.X))
    nfix = mesh.fixed.sum(1)
    regular = np.array([8, 4, 2, 1])[nfix]
    return np.where(val != regular)[0], val


def plot_net(mesh, ax=None, title=None, counts=None):
    Q = mesh.boundary_quads()
    names = face_of_quads(mesh.X, Q)
    ax = ax or plt.figure(figsize=(6, 8)).gca()
    for name, (_, _, f) in NET.items():
        q = Q[names == name]
        if not len(q):
            continue
        u, v = f(mesh.X[q.ravel()])
        polys = np.stack([u, v], 1).reshape(-1, 4, 2)
        ax.add_collection(PolyCollection(polys, facecolors=FACE_COL[name], edgecolors="k", linewidths=0.5))
        c = polys.reshape(-1, 2).mean(0)
        ax.text(*np.floor(c) + 0.5, name, ha="center", va="center", fontsize=14, color="0.6", alpha=0.6)
    # mark irregular face vertices (valence != 4 on the face) in red
    deg = {}
    for q in Q:
        for n in q:
            deg[n] = deg.get(n, 0) + 1
    fx = mesh.fixed.sum(1)
    for n, d in deg.items():
        if fx[n] == 1 and d != 4:
            ax_, = np.where(mesh.fixed[n])
            name = [k for k, (a, v, _) in NET.items() if a == ax_[0] and abs(mesh.X[n, a] - v) < 1e-9][0]
            u, v = NET[name][2](mesh.X[n][None])
            ax.plot(u, v, "o", ms=5, color="red" if d == 3 else "blue")
    ax.set_xlim(-0.05, 3.05); ax.set_ylim(-0.05, 4.05); ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def plot_3d(mesh, ax=None, title=None, elev=25, azim=-60, alpha=0.9):
    Q = mesh.boundary_quads()
    names = face_of_quads(mesh.X, Q)
    if ax is None:
        ax = plt.figure(figsize=(6, 6)).add_subplot(projection="3d")
    cols = [FACE_COL[n] for n in names]
    ax.add_collection3d(Poly3DCollection(mesh.X[Q], facecolors=cols, edgecolors="k", linewidths=0.4, alpha=alpha))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev, azim); ax.set_box_aspect((1, 1, 1))
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def plot_cut(mesh, ax=None, title=None, keep=lambda c: (c[:, 0] < 0.55) | (c[:, 1] > 0.5) | (c[:, 2] < 0.45),
             elev=25, azim=-60):
    """3D view of the hexes whose centroid satisfies `keep` (a cut-away), coloured by quality."""
    from .smooth import scaled_jacobian
    from .mesh import HEX_FACES
    c = mesh.X[mesh.H].mean(1)
    sel = np.where(keep(c))[0]
    sub = mesh.H[sel]
    F = sub[:, HEX_FACES].reshape(-1, 4)
    key = np.sort(F, 1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    vis = cnt[inv.ravel()] == 1                       # faces on the surface of the kept set
    q = np.repeat(scaled_jacobian(mesh.X, mesh.H)[sel], 6)[vis]
    if ax is None:
        ax = plt.figure(figsize=(6, 6)).add_subplot(projection="3d")
    cmap = plt.get_cmap("viridis")
    ax.add_collection3d(Poly3DCollection(mesh.X[F[vis]], facecolors=cmap(np.clip(q, 0, 1)), edgecolors="k", linewidths=0.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1); ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev, azim); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    if title:
        ax.set_title(title, fontsize=10)
    return ax


# ---------------------------------------------------------------------------------------------
# multi-block pictures (physical coordinates)
# ---------------------------------------------------------------------------------------------
BLOCK_COLORS = plt.get_cmap("tab20")(np.arange(20) % 20)


def surface_of(H, keep):
    """Outer quads of the hex subset `keep` and the hex each one belongs to."""
    from .mesh import HEX_FACES
    sel = np.where(keep)[0]
    F = H[sel][:, HEX_FACES].reshape(-1, 4)
    _, inv, cnt = np.unique(np.sort(F, 1), axis=0, return_inverse=True, return_counts=True)
    vis = cnt[inv.ravel()] == 1
    return F[vis], np.repeat(sel, 6)[vis]


def equal_box(ax, P):
    lo, hi = P.min(0), P.max(0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(np.maximum(hi - lo, 1e-9))


def draw_hexes(ax, X, H, colors, keep=None, lw=0.25, alpha=1.0):
    """Surface of the kept hexes; colors: (M,4) per hex."""
    keep = np.ones(len(H), bool) if keep is None else keep
    Q, owner = surface_of(H, keep)
    ax.add_collection3d(Poly3DCollection(X[Q], facecolors=colors[owner], edgecolors="k", linewidths=lw, alpha=alpha))
    equal_box(ax, X[Q].reshape(-1, 3))


def draw_exploded(ax, parts, gap=0.35, lw=0.25):
    """Each block's own boundary mesh, pushed away from the domain centre."""
    cen = np.vstack([p["X"] for p in parts]).mean(0)
    allP = []
    for b, p in enumerate(parts):
        Q = p["ref"].boundary_quads()
        shift = gap * (p["X"].mean(0) - cen)
        P = p["X"][Q] + shift
        ax.add_collection3d(Poly3DCollection(P, facecolors=BLOCK_COLORS[b % 20], edgecolors="k", linewidths=lw))
        allP.append(P.reshape(-1, 3))
    equal_box(ax, np.vstack(allP))


def block_face_quads(part, f, s):
    ref = part["ref"]
    Q = ref.boundary_quads()
    return Q[np.all(np.abs(ref.X[Q][:, :, f] - s) < 1e-9, axis=1)]


def draw_shared_face(ax, partA, faceA, partB, faceB, title=None, corners=None):
    """One interface drawn from both sides (projected to the plane of its corners):
    block A thick blue, block B thin dashed red.  Identical meshes look like one blue-red mesh."""
    QA, QB = block_face_quads(partA, *faceA), block_face_quads(partB, *faceB)
    PA, PB = partA["X"][QA], partB["X"][QB]
    pts = PA.reshape(-1, 3)
    c = pts.mean(0)
    if corners is None:                     # best-fit plane
        _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
        e1, e2 = Vt[0], Vt[1]
    else:                                   # (origin, U-end, V-end) corners: canonical orientation
        o, cu, cv = map(np.asarray, corners)
        e1 = (cu - o) / np.linalg.norm(cu - o)
        e2 = (cv - o) - ((cv - o) @ e1) * e1; e2 /= np.linalg.norm(e2)
    proj = lambda P: np.stack([(P - c) @ e1, (P - c) @ e2], -1)
    ax.add_collection(PolyCollection(proj(PA), facecolors="#eef4fb", edgecolors="#1f5fa8", linewidths=2.2))
    ax.add_collection(PolyCollection(proj(PB), facecolors="none", edgecolors="#d62728", linewidths=0.9, linestyles="--"))
    ax.autoscale(); ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
