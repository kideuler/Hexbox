"""Canonical (face-local) rules and the parity patterns they admit.

For multi-block conformity every face mesh must be a function of that face's 4 counts only.
Rule CR2: a face gets one turning chord at corner v  <=>  both of its sides at v have odd count.
(All remaining count differences are even and are absorbed by U-turn chords.)
"""
import itertools
import numpy as np
from .sheets import EDGES, EIDX, CORNERS, edge_at, edge_ends, other


def odd_edges_of(cls):
    """cls: corner -> 0/1 colouring.  Odd edges = edges joining different colours (a cut)."""
    return np.array([cls[edge_ends(e)[0]] != cls[edge_ends(e)[1]] for e in EDGES], int)


def cr2_turnings(odd):
    """T[(f, v)] = 1 if face x_f = v[f] turns at corner v."""
    return {(f, v): int(all(odd[EIDX[edge_at(v, d)]] for d in other(f))) for f in range(3) for v in CORNERS}


def cr2_sheets(odd):
    """Corner/wrap sheets implied by CR2 turnings, or None if they don't close up into sheets."""
    T = cr2_turnings(odd)
    C = {v: int(all(T[(f, v)] for f in range(3))) for v in CORNERS}
    W = {}
    for e in EDGES:
        v0, v1 = edge_ends(e)
        w0, w1 = T[(e[0], v0)] - C[v0], T[(e[0], v1)] - C[v1]
        if w0 != w1 or w0 < 0:
            return None
        W[e] = w0
    return C, W


def parity_class(n):
    """Corner colouring whose cut = odd edges of n (colour of corner (0,0,0) fixed to 0)."""
    odd = np.asarray(n) % 2
    cls = {(0, 0, 0): 0}
    stack = [(0, 0, 0)]
    while stack:
        v = stack.pop()
        for d in range(3):
            w = list(v); w[d] ^= 1; w = tuple(w)
            c = cls[v] ^ odd[EIDX[edge_at(v, d)]]
            if w in cls and cls[w] != c:
                return None                              # face parity violated
            if w not in cls:
                cls[w] = c; stack.append(w)
    return cls


def enumerate_patterns():
    out = []
    for bits in itertools.product((0, 1), repeat=7):
        cls = dict(zip(CORNERS, (0,) + bits))
        odd = odd_edges_of(cls)
        out.append((cls, odd, cr2_sheets(odd)))
    return out


def cr_turnings(odd):
    """Rule CR: only *odd* faces (opposite sides of different parity) turn, at the corner between
    their two odd sides.  Balanced faces never get turning chords."""
    T = {}
    for f in range(3):
        for v in CORNERS:
            ds = other(f)
            at_v = [odd[EIDX[edge_at(v, d)]] for d in ds]
            # opposite sides: shift v along the other in-face direction
            opp = []
            for i, d in enumerate(ds):
                w = list(v); w[ds[1 - i]] ^= 1
                opp.append(odd[EIDX[edge_at(tuple(w), d)]])
            face_odd = at_v[0] != opp[0]
            T[(f, v)] = int(face_odd and all(at_v))
    return T


def rule_sheets(odd, rule=cr_turnings):
    T = rule(odd)
    C = {v: int(all(T[(f, v)] for f in range(3))) for v in CORNERS}
    W = {}
    for e in EDGES:
        v0, v1 = edge_ends(e)
        w0, w1 = T[(e[0], v0)] - C[v0], T[(e[0], v1)] - C[v1]
        if w0 != w1 or w0 < 0:
            return None
        W[e] = w0
    return C, W
