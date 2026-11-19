"""Block builder: 12 edge counts -> sheet decomposition -> structured base + box pillows -> smooth."""
import numpy as np
from .mesh import HexMesh
from .sheets import decompose, EDGES, EIDX, other, parity_ok
from .smooth import respace_block_edges, laplacian, scaled_jacobian, optimize

ORDER = {"wrap": 0, "uwrap": 1, "corner": 2, "uturn": 3}


def _near(N, side, pad):
    k = int(np.clip(round(pad * N), 1, N - 1))
    return (0, k) if side == 0 else (N - k, N)


def _middle(N, frac=1 / 3):
    m1 = int(np.clip(round(frac * N), 1, N - 2))
    return (m1, N - m1) if N - m1 > m1 else (m1, m1 + 1)


def shrink_box(kind, key, N, pad=0.4):
    """Per-axis cell ranges [lo, hi) of the shrink set that generates this sheet."""
    R = [None] * 3
    if kind == "corner":
        for d in range(3):
            R[d] = _near(N[d], key[d], pad)
    elif kind in ("wrap", "uturn"):
        d, a, b = key
        o = other(d)
        R[o[0]], R[o[1]] = _near(N[o[0]], a, pad), _near(N[o[1]], b, pad)
        R[d] = (0, N[d]) if kind == "wrap" else _middle(N[d])
    elif kind == "uwrap":
        f, s, d = key
        g = [k for k in range(3) if k not in (f, d)][0]
        R[f], R[d], R[g] = _near(N[f], s, pad), _middle(N[d]), (0, N[g])
    return R


def edge_counts(mesh):
    fx, X = mesh.fixed, mesh.X
    out = np.zeros(12, int)
    for i, (d, a, b) in enumerate(EDGES):
        o = other(d)
        on = fx[:, o[0]] & fx[:, o[1]] & (np.abs(X[:, o[0]] - a) < 1e-9) & (np.abs(X[:, o[1]] - b) < 1e-9)
        out[i] = on.sum() - 1
    return out


def check_conformal(mesh):
    """Every face shared by <=2 hexes; faces used once lie on the cube boundary; no repeated nodes."""
    F, owner, inv, cnt = mesh.face_adjacency()
    assert cnt.max() <= 2, "non-manifold face"
    bq = F[cnt[inv] == 1]
    P = mesh.X[bq]
    onb = np.zeros(len(bq), bool)
    for d in range(3):
        for v in (0, 1):
            onb |= np.all(np.abs(P[:, :, d] - v) < 1e-9, axis=1)
    assert onb.all(), "boundary face inside the cube"
    assert all(len(set(h)) == 8 for h in mesh.H), "degenerate hex"
    return True


def build_block(n, verbose=True, target=0.2, **kw):
    """Build with retries over (pad, min_belt); keep the best mesh by min scaled Jacobian."""
    best = None
    for pad, mb in [(0.4, 5), (0.3, 5), (0.5, 5), (0.4, 7), (0.35, 4), (0.45, 3)]:
        try:
            m = _build(n, pad=pad, min_belt=mb, verbose=False, **kw)
        except ValueError:
            raise
        except Exception:
            continue
        q = scaled_jacobian(m.X, m.H).min()
        if best is None or q > best[0]:
            best = (q, m, pad, mb)
        if q >= target:
            break
    if best is None:
        raise RuntimeError('could not fit the transitions: some edge counts are too small (need ~4+)')
    q, m, pad, mb = best
    if verbose:
        used = {k: int(v) for k, v in m.sheets.items()}
        sj = scaled_jacobian(m.X, m.H)
        print(f"sheets {used}\n  hexes {len(m.H)}  counts ok {np.array_equal(edge_counts(m), np.asarray(n))}  "
              f"minSJ {sj.min():.3f}  meanSJ {sj.mean():.3f}  (pad {pad}, min_belt {mb})")
    return m


def _build(n, pad=0.4, min_belt=5, smooth_iters=300, sheets=None, opt=True, verbose=True):
    n = np.asarray(n, int)
    if not parity_ok(n):
        raise ValueError("parity violated: every block face needs an even edge-count sum")
    sheets = sheets or decompose(n, min_belt=min_belt)
    if sheets is None:
        raise ValueError("no sheet decomposition found")
    N = [int(sheets[("belt", d)]) for d in range(3)]
    mesh = HexMesh.structured(*N)
    for (kind, key), mult in sorted(sheets.items(), key=lambda kv: ORDER.get(kv[0][0], -1)):
        if kind == "belt":
            continue
        R = shrink_box(kind, key, N, pad)
        for _ in range(mult):
            b = mesh.base
            S = np.ones(len(b), bool)
            for d in range(3):
                S &= (b[:, d] >= R[d][0]) & (b[:, d] < R[d][1])
            mesh.pillow(S)
    respace_block_edges(mesh)
    laplacian(mesh, iters=smooth_iters)
    if opt:
        optimize(mesh)
    got = edge_counts(mesh)
    if not np.array_equal(got, n):             # a shrink box came out empty: block too small for it
        raise RuntimeError(f"counts not reproduced: got {got.tolist()}")
    check_conformal(mesh)
    sj = scaled_jacobian(mesh.X, mesh.H)
    if verbose:
        used = {k: int(v) for k, v in sheets.items()}
        print(f"sheets {used}\n  hexes {len(mesh.H)}  counts ok {np.array_equal(got, n)}  "
              f"minSJ {sj.min():.3f}  meanSJ {sj.mean():.3f}")
    mesh.sheets = sheets
    return mesh
