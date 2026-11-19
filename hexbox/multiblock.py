"""Multi-block driver.

  1. topology     block vertices V, 8 global vertex ids per block (VTK order), optional curved edges
  2. chains       global edges linked by "parallel in some block" (the structured edge chains)
  3. counts       one integer per global edge, given or derived from a size field;
                  the deterministic rule asks for one parity per chain
  4. per block    mesh_block_job(job): the block's 12 counts, 6 canonical face frames, 12 edge curves.
                  Nothing else is needed, so blocks can run in any order, on any process or rank.
  5. assembly     global node numbering from canonical node keys (no geometric search),
                  or, for comparison, by merging coincident points
  6. verification conformity, exact edge counts, Jacobians, key/position agreement
"""
import itertools
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import cKDTree
from .sheets import EDGES, EIDX, other
from .deterministic import build_block_det
from .build import build_block, edge_counts
from .smooth import scaled_jacobian
from .tfi import tfi, Line, Arc, Reversed
from .mesh import HEX_FACES

CORNER_XYZ = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
VTK = {c: i for i, c in enumerate(CORNER_XYZ)}
FACE_CYCLE = [(0, 0), (1, 0), (1, 1), (0, 1)]          # (u,v) corners of a face, cyclic


def _corner(f, s, p, q, u, v):
    x = [0, 0, 0]; x[f] = s; x[p] = u; x[q] = v
    return VTK[tuple(x)]


def _round_parity(w, par, nmin):
    m = 2 * int(round((w - par) / 2)) + par
    while m < nmin:
        m += 2
    return m


# ------------------------------------------------------------------------------------------------
# per-block work: block-local data in, mesh + canonical node labels out
# ------------------------------------------------------------------------------------------------
def mesh_block_job(job):
    """Mesh one block from block-local data only.  Picklable, so it runs in a worker process.

    job: b, n (12 counts), gids (8 global corner ids), frames (6 canonical face frames),
         curves (12 edge curves), L, D, opt, mode:
           'det'        deterministic faces in canonical frames (the multi-block method)
           'det-local'  deterministic faces in each block's own local frames (control)
           'free'       single-block free mode, faces chosen by the block's 3D sheets (control)"""
    n = np.asarray(job["n"], int)
    if job["mode"] == "free":
        ref = build_block(n, verbose=False)
    else:
        frames = job["frames"] if job["mode"] == "det" else None
        ref = build_block_det(n, L=job["L"], D=job["D"], opt=job["opt"], verbose=False, frames=frames)
    if not np.array_equal(edge_counts(ref), n):
        raise RuntimeError(f"block {job['b']}: edge counts not reproduced")
    keys = node_keys(ref, job["gids"], n, job["b"]) if job["mode"] == "det" else None
    return dict(X=tfi(job["curves"], ref.X), H=ref.H, ref=ref, keys=keys)


def node_keys(ref, G, n, b):
    """Canonical label of every node, computed from block-local data only:
        block corner     ('v', global vertex id)
        block-edge node  ('e', lower id, higher id, i)   i = segment index from the lower-id end
        face node        ('f', 4 sorted ids, j)          j = node index in the canonical template
        interior node    ('c', block, local id)
    Two blocks that share an entity label the same node identically."""
    X, fx = ref.X, ref.fixed
    nf = fx.sum(1)
    keys = [("c", b, j) for j in range(len(X))]
    for (f, s), (src, dst) in ref.face_map.items():
        p, q = other(f)
        fkey = tuple(sorted(int(G[_corner(f, s, p, q, u, v)]) for u, v in FACE_CYCLE))
        for i, j in zip(src, dst):
            if nf[j] == 1:
                keys[j] = ("f", fkey, int(i))
    for j in np.where(nf >= 2)[0]:
        c = np.rint(X[j]).astype(int)
        if nf[j] == 3:
            keys[j] = ("v", int(G[VTK[tuple(c)]]))
            continue
        d = int(np.where(~fx[j])[0][0])
        o = other(d)
        c0 = list(c); c0[d] = 0
        c1 = list(c); c1[d] = 1
        ga, gb = int(G[VTK[tuple(c0)]]), int(G[VTK[tuple(c1)]])
        m = int(n[EIDX[(d, int(c[o[0]]), int(c[o[1]]))]])
        i = int(round(X[j, d] * m))
        keys[j] = ("e", min(ga, gb), max(ga, gb), i if ga < gb else m - i)
    return keys


def _merge_by_position(X, tol):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    pairs = cKDTree(X).query_pairs(tol, output_type="ndarray")
    A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(X), len(X)))
    return connected_components(A, directed=False)[1]


def same_partition(a, b):
    """True if two node numberings group the raw (per-block) nodes identically."""
    return len(set(zip(a.tolist(), b.tolist()))) == len(set(a.tolist())) == len(set(b.tolist()))


# ------------------------------------------------------------------------------------------------
class MultiBlock:
    """V: (nv,3) block vertices.  blocks: (nb,8) global vertex ids in VTK order (each block's local
    reference cube).  curves: {(ga, gb): curve} geometry of edge ga -> gb (default: straight)."""

    def __init__(self, V, blocks, curves=None, L=2, D=2):
        self.V = np.asarray(V, float)
        self.blocks = np.asarray(blocks, int)
        self.curves = dict(curves or {})
        self.L, self.D = L, D
        self.counts = {}

    # ---- topology ------------------------------------------------------------------------------
    def local_edge(self, b, e):
        """Global (start, end) vertex ids of local edge e = (d, a, c) of block b."""
        d, a, c = e
        o = other(d)
        x0 = [0, 0, 0]; x0[o[0]] = a; x0[o[1]] = c
        x1 = list(x0); x1[d] = 1
        G = self.blocks[b]
        return int(G[VTK[tuple(x0)]]), int(G[VTK[tuple(x1)]])

    def edges(self):
        return sorted({frozenset(self.local_edge(b, e)) for b in range(len(self.blocks)) for e in EDGES},
                      key=sorted)

    def chains(self):
        """Union-find of global edges linked by 'parallel in some block'."""
        parent = {e: e for e in self.edges()}

        def find(e):
            while parent[e] != e:
                parent[e] = parent[parent[e]]
                e = parent[e]
            return e
        for b in range(len(self.blocks)):
            for d in range(3):
                es = [frozenset(self.local_edge(b, e)) for e in EDGES if e[0] == d]
                for e in es[1:]:
                    parent[find(e)] = find(es[0])
        groups = {}
        for e in parent:
            groups.setdefault(find(e), []).append(e)
        return list(groups.values())

    def block_faces(self):
        """Domain-boundary block faces [(b, f, s)] and shared ones [[(b, f, s), (b', f', s')]]."""
        key = {}
        for b in range(len(self.blocks)):
            for f in range(3):
                p, q = other(f)
                for s in (0, 1):
                    k = frozenset(int(self.blocks[b][_corner(f, s, p, q, u, v)]) for u, v in FACE_CYCLE)
                    key.setdefault(k, []).append((b, f, s))
        return [v[0] for v in key.values() if len(v) == 1], [v for v in key.values() if len(v) == 2]

    # ---- geometry ------------------------------------------------------------------------------
    def curve(self, ga, gb):
        if (ga, gb) in self.curves:
            return self.curves[(ga, gb)]
        if (gb, ga) in self.curves:
            return Reversed(self.curves[(gb, ga)])
        return Line(self.V[ga], self.V[gb])

    def edge_length(self, e):
        ga, gb = sorted(e)
        P = self.curve(ga, gb)(np.linspace(0, 1, 33))
        return np.linalg.norm(np.diff(P, axis=0), axis=1).sum()

    # ---- interval assignment -------------------------------------------------------------------
    def assign_counts(self, size, nmin=5):
        """n_e ~ length / h(midpoint), rounded to one parity per chain: the parity with the smaller
        total log-error.  Magnitudes stay free along the chain."""
        want = {}
        for e in self.edges():
            ga, gb = sorted(e)
            mid = self.curve(ga, gb)(np.array([0.5]))[0]
            want[e] = max(nmin, self.edge_length(e) / size(mid))
        for chain in self.chains():
            best = None
            for par in (0, 1):
                cnt = {e: _round_parity(want[e], par, nmin) for e in chain}
                err = sum(np.log(cnt[e] / want[e]) ** 2 for e in chain)
                if best is None or err < best[0]:
                    best = (err, cnt)
            self.counts.update(best[1])
        return self.counts

    def random_counts(self, lo=5, hi=13, seed=0):
        """Adversarial assignment: a random parity per chain, then an independent random count in
        [lo, hi] with that parity on every edge (neighbouring edges can differ by any even amount)."""
        rng = np.random.default_rng(seed)
        for chain in self.chains():
            par = int(rng.integers(2))
            vals = [m for m in range(lo, hi + 1) if m % 2 == par]
            for e in chain:
                self.counts[e] = int(rng.choice(vals))
        return self.counts

    def block_counts(self, b):
        return np.array([self.counts[frozenset(self.local_edge(b, e))] for e in EDGES])

    # ---- per-block jobs ------------------------------------------------------------------------
    def canonical_frame(self, b, f, s):
        """Frame of local face (f, s) fixed by global ids: origin at the corner with the smallest id,
        U axis toward its smaller-id neighbour.  Returns origin and axes in the block's local face
        coordinates, plus the face counts listed in the canonical frame."""
        p, q = other(f)
        G = self.blocks[b]
        gid = [int(G[_corner(f, s, p, q, u, v)]) for u, v in FACE_CYCLE]
        o = int(np.argmin(gid))
        n1, n2 = (o + 1) % 4, (o - 1) % 4
        nu, nv = (n1, n2) if gid[n1] < gid[n2] else (n2, n1)
        opp = (o + 2) % 4
        P = np.array(FACE_CYCLE, float)
        cnt = lambda i, j: self.counts[frozenset((gid[i], gid[j]))]
        counts = (cnt(o, nu), cnt(nu, opp), cnt(nv, opp), cnt(o, nv))
        return P[o], P[nu] - P[o], P[nv] - P[o], counts

    def job(self, b, mode="det", opt=True):
        return dict(b=b, n=self.block_counts(b), gids=self.blocks[b].copy(),
                    frames={(f, s): self.canonical_frame(b, f, s) for f in range(3) for s in (0, 1)},
                    curves=[self.curve(*self.local_edge(b, e)) for e in EDGES],
                    L=self.L, D=self.D, mode=mode, opt=opt)

    def mesh_blocks(self, mode="det", opt=True, workers=0):
        jobs = [self.job(b, mode, opt) for b in range(len(self.blocks))]
        if workers:
            with ProcessPoolExecutor(workers) as ex:
                return list(ex.map(mesh_block_job, jobs))
        return [mesh_block_job(j) for j in jobs]

    # ---- assembly + verification ---------------------------------------------------------------
    def assemble(self, parts, by="keys"):
        """Global mesh.  by='keys': numbering from canonical node keys (no search, no communication);
        by='position': merge points closer than 1e-8 x domain size."""
        X = np.vstack([p["X"] for p in parts])
        off = np.cumsum([0] + [len(p["X"]) for p in parts])
        H = np.vstack([p["H"] + off[i] for i, p in enumerate(parts)])
        self.tol = 1e-8 * np.ptp(X, axis=0).max()
        if by == "keys":
            table = {}
            gid = np.array([table.setdefault(k, len(table)) for p in parts for k in p["keys"]])
        else:
            gid = _merge_by_position(X, self.tol)
        Xg = np.zeros((gid.max() + 1, 3))
        np.add.at(Xg, gid, X)
        Xg /= np.bincount(gid)[:, None]
        self.parts, self.off, self.node_of = parts, off, gid
        self.spread = float(np.linalg.norm(X - Xg[gid], axis=1).max())
        self.X, self.H = Xg, gid[H]
        return self.X, self.H

    def verify(self):
        """Every quad used by <= 2 hexes; the quads used once are exactly the quads of domain-boundary
        block faces (anything else is an unmatched interface quad); no degenerate hexes; no two
        distinct nodes at the same point; per-key position spread; scaled Jacobians."""
        F = np.sort(self.H[:, HEX_FACES].reshape(-1, 4), axis=1)
        uq, cnt = np.unique(F, axis=0, return_counts=True)
        once = set(map(tuple, uq[cnt == 1]))
        expect = set()
        bnd, shared = self.block_faces()
        for b, f, s in bnd:
            ref = self.parts[b]["ref"]
            Q = ref.boundary_quads()
            Q = Q[np.all(np.abs(ref.X[Q][:, :, f] - s) < 1e-9, axis=1)]
            expect |= set(map(tuple, np.sort(self.node_of[Q + self.off[b]], axis=1)))
        Hs = np.sort(self.H, axis=1)
        sj = scaled_jacobian(self.X, self.H)
        rep = dict(blocks=len(self.blocks), shared_faces=len(shared), hexes=len(self.H), nodes=len(self.X),
                   max_face_use=int(cnt.max()), unmatched_quads=len(once - expect),
                   missing_boundary_quads=len(expect - once),
                   degenerate_hexes=int(np.any(np.diff(Hs, axis=1) == 0, axis=1).sum()),
                   coincident_nodes=len(cKDTree(self.X).query_pairs(self.tol)),
                   spread=self.spread, min_sj=float(sj.min()), mean_sj=float(sj.mean()))
        rep["conformal"] = (rep["max_face_use"] <= 2 and rep["unmatched_quads"] == 0 and
                            rep["missing_boundary_quads"] == 0 and rep["degenerate_hexes"] == 0 and
                            rep["coincident_nodes"] == 0)
        return rep


# ------------------------------------------------------------------------------------------------
# test domains and transformations
# ------------------------------------------------------------------------------------------------
def box_grid(nb=(3, 2, 2), size=(3.0, 2.0, 2.0), jitter=0.0, seed=0):
    """Box split into nb[0] x nb[1] x nb[2] hex blocks; interior block vertices jittered."""
    rng = np.random.default_rng(seed)
    I, J, K = np.meshgrid(*[np.arange(m + 1) for m in nb], indexing="ij")
    V = np.c_[I.ravel() * size[0] / nb[0], J.ravel() * size[1] / nb[1], K.ravel() * size[2] / nb[2]]
    inner = ((I > 0) & (I < nb[0]) & (J > 0) & (J < nb[1]) & (K > 0) & (K < nb[2])).ravel()
    V[inner] += jitter * rng.uniform(-1, 1, (inner.sum(), 3)) * np.array(size) / np.array(nb)
    vid = lambda i, j, k: (i * (nb[1] + 1) + j) * (nb[2] + 1) + k
    blocks = [[vid(i + c[0], j + c[1], k + c[2]) for c in CORNER_XYZ]
              for i in range(nb[0]) for j in range(nb[1]) for k in range(nb[2])]
    return MultiBlock(V, blocks)


def cubed_sphere(a=0.3, R=1.0):
    """Unit ball as 7 blocks: a central cube [-a,a]^3 and 6 blocks out to the sphere.  The 12 sphere
    edges are great-circle arcs; the 8 radial edges are each shared by 3 blocks."""
    cs = list(itertools.product((0, 1), repeat=3))
    idx = {c: i for i, c in enumerate(cs)}
    V = np.array([a * (2 * np.array(c) - 1) for c in cs] + [R * (2 * np.array(c) - 1) / np.sqrt(3) for c in cs])
    blocks = [[idx[c] for c in CORNER_XYZ]]
    for k in range(3):
        p, q = (k + 1) % 3, (k + 2) % 3
        for s in (0, 1):
            u_ax, v_ax = (p, q) if s == 1 else (q, p)        # keeps every block right-handed
            blk = []
            for u, v, w in CORNER_XYZ:
                c = [0, 0, 0]; c[k] = s; c[u_ax] = u; c[v_ax] = v
                blk.append(idx[tuple(c)] + 8 * w)
            blocks.append(blk)
    curves = {(8 + i, 8 + j): Arc(V[8 + i], V[8 + j], (0, 0, 0))
              for i, ci in enumerate(cs) for j, cj in enumerate(cs) if i < j and sum(x != y for x, y in zip(ci, cj)) == 1}
    return MultiBlock(V, blocks, curves)


def cube_rotations():
    """The 24 orientation-preserving symmetries of the unit cube, as maps on corner tuples."""
    out = []
    for perm in itertools.permutations(range(3)):
        sgn = round(np.linalg.det(np.eye(3)[list(perm)]))
        for flips in itertools.product((0, 1), repeat=3):
            if sgn * (-1) ** sum(flips) > 0:
                out.append((perm, flips))
    return out


def scrambled(mb, seed=0):
    """Same domain and counts, but every block gets a random local orientation (one of the 24 cube
    rotations) and all global vertex ids are randomly permuted."""
    rng = np.random.default_rng(seed)
    rots = cube_rotations()
    pi = rng.permutation(len(mb.V))
    V = np.empty_like(mb.V); V[pi] = mb.V
    blocks = []
    for G in mb.blocks:
        perm, flips = rots[rng.integers(len(rots))]
        blocks.append([pi[G[VTK[tuple(c[perm[i]] ^ flips[i] for i in range(3))]]] for c in CORNER_XYZ])
    curves = {(int(pi[a]), int(pi[b])): c for (a, b), c in mb.curves.items()}
    out = MultiBlock(V, blocks, curves, mb.L, mb.D)
    out.counts = {frozenset(int(pi[v]) for v in e): n for e, n in mb.counts.items()}
    return out
