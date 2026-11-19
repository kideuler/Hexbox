"""Reference-space node placement: uniform block-edge nodes, constrained Laplacian smoothing,
and hex quality (scaled Jacobian)."""
import numpy as np
from .mesh import HEX_EDGES

# corner -> its 3 neighbours in VTK hex ordering (for corner Jacobians)
_NB = np.array([(1, 3, 4), (2, 0, 5), (3, 1, 6), (0, 2, 7), (7, 5, 0), (4, 6, 1), (5, 7, 2), (6, 4, 3)])


def scaled_jacobian(X, H):
    P = X[H]                                            # (M,8,3)
    e = P[:, _NB, :] - P[:, :, None, :]                 # (M,8,3,3)
    e /= np.linalg.norm(e, axis=3, keepdims=True) + 1e-300
    return np.linalg.det(e).min(axis=1)


def edge_list(H):
    E = np.sort(H[:, HEX_EDGES].reshape(-1, 2), axis=1)
    return np.unique(E, axis=0)


def respace_block_edges(mesh):
    """Put the nodes of every block edge at uniform spacing (keeps their current order)."""
    fx = mesh.fixed
    X = mesh.X
    on_edge = fx.sum(1) == 2
    for free in range(3):
        sel = on_edge & ~fx[:, free]
        for a in (0, 1):
            for b in (0, 1):
                o = [d for d in range(3) if d != free]
                ids = np.where(sel & (np.abs(X[:, o[0]] - a) < 1e-9) & (np.abs(X[:, o[1]] - b) < 1e-9))[0]
                ids = ids[np.argsort(X[ids, free])]
                X[ids, free] = np.arange(1, len(ids) + 1) / (len(ids) + 1)


def laplacian(mesh, iters=200, relax=0.6, move_faces=True):
    """Jacobi Laplacian.  Interior nodes use all edge neighbours; face nodes use only neighbours on
    the same cube face (so they slide in-plane); edge/corner nodes are frozen."""
    X = mesh.X
    fx = mesh.fixed
    nfix = fx.sum(1)
    E = edge_list(mesh.H)
    # a boundary-face node only listens to neighbours that share its fixed coordinate
    def keep(a, b):
        ok = nfix[a] == 0
        f = nfix[a] == 1
        ax = np.argmax(fx[a], axis=1)
        ok |= f & fx[b, ax] & (np.abs(X[b, ax] - X[a, ax]) < 1e-9)
        return ok
    src = np.r_[E[:, 0], E[:, 1]]; dst = np.r_[E[:, 1], E[:, 0]]
    m = keep(src, dst)
    src, dst = src[m], dst[m]
    deg = np.bincount(src, minlength=len(X)).astype(float)
    move = ((nfix <= 1) if move_faces else (nfix == 0)) & (deg > 0)
    for _ in range(iters):
        acc = np.zeros_like(X)
        np.add.at(acc, src, X[dst])
        avg = acc[move] / deg[move, None]
        new = X[move] + relax * (avg - X[move])
        new[fx[move]] = X[move][fx[move]]
        X[move] = new
    return mesh


def _objective(X, H, delta):
    """Sum over hex corners of the inverse mean ratio |J|_F^2 / (3 h(detJ)^(2/3)), with the
    regularised determinant h(D) = (D + sqrt(D^2 + 4 delta^2))/2 so inverted corners still have
    finite value and a gradient pointing toward untangling (Escobar et al.)."""
    P = X[H]
    e = P[:, _NB, :] - P[:, :, None, :]                    # (M,8,3 edges,3 coords)
    F = (e ** 2).sum(axis=(2, 3))
    c1 = np.cross(e[:, :, 1], e[:, :, 2]); c2 = np.cross(e[:, :, 2], e[:, :, 0]); c3 = np.cross(e[:, :, 0], e[:, :, 1])
    D = (e[:, :, 0] * c1).sum(-1)
    s = np.sqrt(D ** 2 + 4 * delta ** 2)
    h = 0.5 * (D + s); dh = 0.5 * (1 + D / s)
    f = F / (3 * h ** (2 / 3))
    a = 2 / (3 * h ** (2 / 3))                             # d f / d e_i = a e_i - b cof_i
    b = (2 / 9) * F * h ** (-5 / 3) * dh
    g = a[..., None, None] * e - b[..., None, None] * np.stack([c1, c2, c3], 2)
    G = np.zeros_like(X)
    for k in range(3):
        np.add.at(G, H[:, _NB[:, k]], g[:, :, k])
    np.add.at(G, H, -g.sum(2))
    return f.sum(), G, D.min()


def optimize(mesh, schedule=(0.3, 0.1, 0.03, 0.01, 0.001), maxiter=1000, verbose=False, move_faces=True):
    """Untangle + smooth by continuation on the determinant regularisation delta (large -> small).
    Free dofs: all coords of interior nodes, in-plane coords of face nodes; block-edge nodes fixed."""
    from scipy.optimize import minimize
    fx = mesh.fixed
    free = (~fx) & ((fx.sum(1) <= 1) if move_faces else (fx.sum(1) == 0))[:, None]
    if not free.any():                      # nothing to move (e.g. a tiny structured block)
        return mesh
    P = mesh.X[mesh.H]
    scale = np.median(np.abs(np.linalg.det(P[:, [1, 3, 4]] - P[:, [0]])))
    for t in schedule:
        delta = t * scale
        def fun(z):
            X = mesh.X.copy(); X[free] = z
            v, G, _ = _objective(X, mesh.H, delta)
            return v, G[free]
        res = minimize(fun, mesh.X[free], jac=True, method="L-BFGS-B",
                       bounds=[(0, 1)] * int(free.sum()), options={"maxiter": maxiter})
        mesh.X[free] = res.x
        if verbose:
            print(f"  delta/scale {t:g}: minSJ {scaled_jacobian(mesh.X, mesh.H).min():.3f}")
    return mesh


def _obj2d(UV, Q, delta):
    """2D inverse mean ratio over quad corners, |e1|^2+|e2|^2 / (2 h(cross(e1,e2)))."""
    P = UV[Q]
    e1 = np.roll(P, -1, axis=1) - P
    e2 = np.roll(P, 1, axis=1) - P
    D = e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0]
    F = (e1 ** 2).sum(-1) + (e2 ** 2).sum(-1)
    s = np.sqrt(D ** 2 + 4 * delta ** 2); h = 0.5 * (D + s); dh = 0.5 * (1 + D / s)
    f = F / (2 * h)
    a = 1 / h; b = F / (2 * h ** 2) * dh
    g1 = a[..., None] * e1 - b[..., None] * np.stack([e2[..., 1], -e2[..., 0]], -1)
    g2 = a[..., None] * e2 - b[..., None] * np.stack([-e1[..., 1], e1[..., 0]], -1)
    G = np.zeros_like(UV)
    np.add.at(G, np.roll(Q, -1, axis=1), g1); np.add.at(G, np.roll(Q, 1, axis=1), g2)
    np.add.at(G, Q, -(g1 + g2))
    return f.sum(), G


def optimize2d(UV, Q, schedule=(0.3, 0.1, 0.01, 0.001), maxiter=1000):
    """Untangle+smooth a quad mesh of the unit square; boundary nodes fixed.  Deterministic."""
    from scipy.optimize import minimize
    UV = UV.copy()
    P = UV[Q]
    area = 0.5 * np.sum(P[:, :, 0] * np.roll(P[:, :, 1], -1, 1) - np.roll(P[:, :, 0], -1, 1) * P[:, :, 1], axis=1)
    if np.median(area) < 0:
        Q = Q[:, ::-1]
    onb = (np.abs(UV) < 1e-12) | (np.abs(UV - 1) < 1e-12)
    used = np.zeros(len(UV), bool); used[Q.ravel()] = True
    free = used & ~onb.any(1)
    if not free.any():
        return UV
    scale = np.median(np.abs(area))
    for t in schedule:
        def fun(z):
            X = UV.copy(); X[free] = z.reshape(-1, 2)
            v, G = _obj2d(X, Q, t * scale)
            return v, G[free].ravel()
        r = minimize(fun, UV[free].ravel(), jac=True, method="L-BFGS-B",
                     bounds=[(0, 1)] * (2 * int(free.sum())), options={"maxiter": maxiter})
        UV[free] = r.x.reshape(-1, 2)
    return UV
