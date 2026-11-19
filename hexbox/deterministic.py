"""Deterministic faces (multi-block / parallel safe).

Rule: inside a block, the 4 parallel edges of each direction have the same parity.
Then every face mesh is a fixed function of that face's 4 counts:
    structured core  min(a,c) x min(b,d)
    + (larger - smaller)/2 nested U-turns on the longer side of each opposite pair,
      spanning the side minus a collar of L cells at each end, depth D cells (D <= L).
The block interior is built from a structured base (per-direction minimum) plus "U-sheets" from a
skyline decomposition of the excess on the 4 parallel edges; their traces reproduce the templates.
"""
import functools
import numpy as np
from .mesh import HexMesh
from .sheets import EIDX, other
from .smooth import respace_block_edges, laplacian, optimize, optimize2d, scaled_jacobian
from .faceiso import match_nodes

CYC = [(0, 0), (1, 0), (1, 1), (0, 1)]          # parallel edges in cyclic order around their axis


def rule_ok(n):
    n = np.asarray(n)
    return all(len({int(n[EIDX[(d, a, b)]]) % 2 for a, b in CYC}) == 1 for d in range(3))


def _rng(N, side, w):
    return (0, w) if side == 0 else (N - w, N)


def skyline(k):
    """Excess per edge (CYC order) -> list of (level, run of CYC indices); runs are contiguous arcs."""
    out = []
    for lev in range(1, max(k) + 1):
        act = [ki >= lev for ki in k]
        s = act.index(False)                       # min excess is 0, so some edge is inactive
        run = []
        for t in range(1, 5):
            i = (s + t) % 4
            if act[i]:
                run.append(i)
            elif run:
                out.append((lev, run)); run = []
    return out


def u_sheets(n, L, D):
    """Base sizes and the list of U-sheets (level, direction, run)."""
    n = np.asarray(n)
    B = [min(int(n[EIDX[(d, a, b)]]) for a, b in CYC) for d in range(3)]
    sheets = []
    for d in range(3):
        k = [(int(n[EIDX[(d, a, b)]]) - B[d]) // 2 for a, b in CYC]
        for lev, run in skyline(k):
            sheets.append((lev, d, run))
    return B, sorted(sheets, key=lambda s: s[0])


def _cross_section(base_p, base_q, Bp, Bq, run, D):
    """Cells (in the plane normal to the sheet direction) enclosed by the sheet: D-deep squares at
    the run's corners plus D-deep strips along the faces joining consecutive corners."""
    R = np.zeros(len(base_p), bool)
    inr = lambda x, r: (x >= r[0]) & (x < r[1])
    for i in run:
        a, b = CYC[i]
        R |= inr(base_p, _rng(Bp, a, D)) & inr(base_q, _rng(Bq, b, D))
    for i, j in zip(run, run[1:]):
        (a1, b1), (a2, b2) = CYC[i], CYC[j]
        R |= inr(base_p, _rng(Bp, a1, D)) if a1 == a2 else inr(base_q, _rng(Bq, b1, D))
    return R


def build_block_det(n, L=2, D=2, opt=True, verbose=True, frames=None, smooth=True):
    n = np.asarray(n, int)
    if not rule_ok(n):
        raise ValueError("deterministic mode needs equal parity on the 4 parallel edges of each direction")
    B, sheets = u_sheets(n, L, D)
    for lev, d, run in sheets:
        if B[d] < 2 * L + 1:
            raise ValueError(f"direction {d}: min count {B[d]} too small for a transition (need {2*L+1})")
        for p in other(d):
            if B[p] < D + 1:
                raise ValueError(f"direction {p}: min count {B[p]} too small to cross a direction-{d} "
                                 f"transition (need {D+1})")
    mesh = HexMesh.structured(*B)
    for lev, d, run in sheets:
        p, q = other(d)
        b = mesh.base
        S = (b[:, d] >= L) & (b[:, d] < B[d] - L) & _cross_section(b[:, p], b[:, q], B[p], B[q], run, D)
        mesh.pillow(S)
    respace_block_edges(mesh)
    stamp_faces(mesh, n, L, D, frames)         # face nodes <- deterministic 2D templates (frozen)
    if smooth:
        laplacian(mesh, move_faces=False)
    if opt and smooth:
        optimize(mesh, move_faces=False)
    if verbose:
        sj = scaled_jacobian(mesh.X, mesh.H)
        print(f"base {B}, {len(sheets)} U-sheets, hexes {len(mesh.H)}, minSJ {sj.min():.3f}, meanSJ {sj.mean():.3f}")
    return mesh


@functools.lru_cache(maxsize=None)
def face_template(a, b, c, d, L=2, D=2):
    """Deterministic 2D mesh of a face: sides a (v=0), b (u=1), c (v=1), d (u=0).
    Returns (UV nodes, quads).  Built as a 1-cell slab so the 3D pillow code can be reused."""
    if (a - c) % 2 or (b - d) % 2:
        raise ValueError("opposite sides must have equal parity")
    nu, nv = min(a, c), min(b, d)
    m = HexMesh.structured(nu, nv, 1)
    for side, cnt in ((0, a), (2, c), (1, b), (3, d)):
        opp = {0: c, 2: a, 1: d, 3: b}[side]
        for _ in range(max(0, cnt - opp) // 2):
            bs = m.base
            if side in (0, 2):
                S = (bs[:, 0] >= L) & (bs[:, 0] < nu - L) & ((bs[:, 1] < D) if side == 0 else (bs[:, 1] >= nv - D))
            else:
                S = (bs[:, 1] >= L) & (bs[:, 1] < nv - L) & ((bs[:, 0] >= nu - D) if side == 1 else (bs[:, 0] < D))
            m.pillow(S)
    respace_block_edges(m)
    laplacian(m)
    Q = m.boundary_quads()
    Q = Q[np.all(np.abs(m.X[Q][:, :, 2]) < 1e-9, axis=1)]
    return optimize2d(m.X[:, :2], Q), Q


def _ekey(d, c):
    o = other(d)
    return (d, c[o[0]], c[o[1]])


def face_counts(n, f, s):
    """Counts (a,b,c,d) of face x_f = s in its local frame u = x_p, v = x_q, (p,q) = other(f)."""
    p, q = other(f)
    return [int(n[EIDX[_ekey(p, {f: s, q: 0})]]), int(n[EIDX[_ekey(q, {f: s, p: 1})]]),
            int(n[EIDX[_ekey(p, {f: s, q: 1})]]), int(n[EIDX[_ekey(q, {f: s, p: 0})]])]


def stamp_faces(mesh, n, L=2, D=2, frames=None):
    """Overwrite every face's node positions with its stand-alone template (topology must match).

    frames[(f, s)] = (Po, Eu, Ev, counts): the face's canonical frame expressed in the block's local
    face coordinates (u, v) = Po + U*Eu + V*Ev, and the face's counts listed in that canonical frame.
    Without frames the local frame is used.  With frames built from global vertex ids
    (see multiblock.canonical_frame) every block sharing the face evaluates the *same* template call
    and gets identical node positions, however it is oriented."""
    Q = mesh.boundary_quads()
    mesh.face_map = {}                         # (f, s) -> (template node ids, block node ids)
    for f in range(3):
        p, q = other(f)
        for s in (0, 1):
            Qf = Q[np.all(np.abs(mesh.X[Q][:, :, f] - s) < 1e-9, axis=1)]
            if frames is not None:
                Po, Eu, Ev, cnt = frames[(f, s)]
            else:
                Po, Eu, Ev, cnt = (0, 0), (1, 0), (0, 1), face_counts(n, f, s)
            UVc, Qt = face_template(*map(int, cnt), L=L, D=D)
            UVt = np.asarray(Po, float) + UVc[:, :1] * np.asarray(Eu, float) + UVc[:, 1:] * np.asarray(Ev, float)
            M = match_nodes(UVt, Qt, mesh.X[:, [p, q]], Qf)
            if M is None:
                raise RuntimeError(f"face {(f, s)} does not match its template")
            src = np.fromiter(M.keys(), int); dst = np.fromiter(M.values(), int)
            mesh.X[dst[:, None], [p, q]] = UVt[src]
            mesh.face_map[(f, s)] = (src, dst)
