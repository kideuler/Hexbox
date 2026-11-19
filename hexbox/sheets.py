"""Sheet vocabulary for one hex block and the integer program that decomposes 12 edge counts.

Edges are keyed (d, a, b): direction d in {0,1,2}, (a, b) = position along the two other axes
(in increasing axis order).  Corners are (x, y, z) in {0,1}^3.

Every sheet (dual surface) of the final hex mesh crosses the block edges in a fixed pattern:
  belt[d]          +1 on the 4 edges of direction d            (structured layer, no irregularity)
  wrap[e]          +1 on the 4 other edges at e's two ends     (2D 3-5 pair, extruded along e)
  corner[v]        +1 on the 3 edges at corner v               (3-5 pairs on the 3 faces at v)
  uwrap[f,s,d]     +2 on the 2 edges of face (x_f=s) parallel d (2D U-turn extruded across face)
  uturn[e]         +2 on edge e only                           (local 3-to-1 style refinement)
"""
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

EDGES = [(d, a, b) for d in range(3) for a in (0, 1) for b in (0, 1)]
EIDX = {e: i for i, e in enumerate(EDGES)}
CORNERS = list(itertools.product((0, 1), repeat=3))


def other(d):
    return [k for k in range(3) if k != d]


def edge_at(v, d):
    """Edge of direction d through corner v."""
    o = other(d)
    return (d, v[o[0]], v[o[1]])


def edge_ends(e):
    d, a, b = e
    o = other(d)
    ends = []
    for t in (0, 1):
        v = [0, 0, 0]; v[d] = t; v[o[0]] = a; v[o[1]] = b
        ends.append(tuple(v))
    return ends


def catalog():
    """List of (kind, key, contribution vector over the 12 edges)."""
    S = []
    for d in range(3):
        c = np.zeros(12); c[[EIDX[e] for e in EDGES if e[0] == d]] = 1
        S.append(("belt", d, c))
    for e in EDGES:
        c = np.zeros(12)
        for v in edge_ends(e):
            for d in other(e[0]):
                c[EIDX[edge_at(v, d)]] += 1
        S.append(("wrap", e, c))
    for v in CORNERS:
        c = np.zeros(12)
        for d in range(3):
            c[EIDX[edge_at(v, d)]] += 1
        S.append(("corner", v, c))
    for f in range(3):
        for s in (0, 1):
            for d in other(f):
                c = np.zeros(12)
                for e in EDGES:
                    if e[0] == d:
                        o = other(d)
                        pos = dict(zip(o, e[1:]))
                        if pos[f] == s:
                            c[EIDX[e]] = 2
                S.append(("uwrap", (f, s, d), c))
    for e in EDGES:
        c = np.zeros(12); c[EIDX[e]] = 2
        S.append(("uturn", e, c))
    return S


COST = {"belt": 0.0, "wrap": 2.0, "corner": 3.0, "uwrap": 3.0, "uturn": 4.0}


def parity_ok(n):
    """Every block face must carry an even number of boundary edges."""
    n = np.asarray(n)
    for f in range(3):
        for s in (0, 1):
            tot = sum(n[EIDX[e]] for e in EDGES if e[0] != f and dict(zip(other(e[0]), e[1:]))[f] == s)
            if tot % 2:
                return False
    return True


def decompose(n, min_belt=3, cost=COST):
    """Nonnegative integer sheet multiplicities reproducing the 12 counts n (ordered as EDGES).

    Objective: minimise sum(cost_k * [x_k > 0]) (number of distinct irregular sheet families,
    weighted) + small * total irregular multiplicity, i.e. keep as much as possible structured.
    Returns dict {(kind, key): multiplicity} or None if infeasible.
    """
    n = np.asarray(n, float)
    S = catalog()
    K = len(S)
    A = np.array([c for _, _, c in S]).T             # 12 x K
    big = n.max() + 1
    # variables: x_0..x_{K-1} (integer mult), y_0..y_{K-1} (binary "used")
    c = np.r_[[1e-3 * (k != "belt") for k, _, _ in S], [cost[k] for k, _, _ in S]]
    cons = [LinearConstraint(np.c_[A, np.zeros((12, K))], n, n),
            LinearConstraint(np.c_[np.eye(K), -big * np.eye(K)], -np.inf, 0)]
    for lb in sorted({min_belt, 2, 1, 0}, reverse=True):
        if lb > min_belt:
            continue
        lo = np.r_[[lb if k == "belt" else 0 for k, _, _ in S], np.zeros(K)]
        hi = np.r_[np.full(K, big), np.ones(K)]
        r = milp(c, constraints=cons, integrality=np.ones(2 * K), bounds=Bounds(lo, hi))
        if r.status == 0:
            x = np.round(r.x[:K]).astype(int)
            return {(S[i][0], S[i][1]): x[i] for i in range(K) if x[i] > 0 or S[i][0] == "belt"}
    return None
