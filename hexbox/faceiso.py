"""Boundary-anchored isomorphism test for two quad meshes of the unit square.

Boundary nodes are matched by position (block-edge nodes are uniformly spaced in both meshes);
the match is then propagated quad by quad.  If an isomorphism fixing the boundary exists it is unique,
so this decides whether two face meshes have identical topology."""
import numpy as np


def _edge_quads(Q):
    em = {}
    for qi, q in enumerate(Q):
        for i in range(4):
            em.setdefault(frozenset((q[i], q[(i + 1) % 4])), []).append(qi)
    return em


def _rot(q, a, b):
    q = list(q); i = q.index(a)
    if q[(i + 1) % 4] == b:
        return [a, b, q[(i + 2) % 4], q[(i + 3) % 4]]
    return [a, b, q[(i + 2) % 4], q[(i + 1) % 4]]


def same_topology(UV1, Q1, UV2, Q2, tol=1e-7):
    return match_nodes(UV1, Q1, UV2, Q2, tol) is not None


def match_nodes(UV1, Q1, UV2, Q2, tol=1e-7):
    """Node map mesh1 -> mesh2 (dict) for a boundary-fixing isomorphism, or None."""
    if len(Q1) != len(Q2):
        return None
    e1, e2 = _edge_quads(Q1), _edge_quads(Q2)
    bnd1 = [e for e, qs in e1.items() if len(qs) == 1]
    nodes1 = sorted({v for e in bnd1 for v in e}); nodes2 = sorted({v for e, qs in e2.items() if len(qs) == 1 for v in e})
    if len(nodes1) != len(nodes2):
        return None
    from scipy.spatial import cKDTree
    dist, idx = cKDTree(UV2[nodes2]).query(UV1[nodes1])
    if dist.max() > tol or len(set(idx.tolist())) != len(nodes1):
        return None
    M = {v: nodes2[i] for v, i in zip(nodes1, idx)}
    e = next(iter(bnd1)); a, b = tuple(e)
    stack = [(e1[e][0], e2[frozenset((M[a], M[b]))][0], a, b)]
    match = {}
    while stack:
        q1, q2, a, b = stack.pop()
        if q1 in match:
            if match[q1] != q2:
                return None
            continue
        match[q1] = q2
        r1, r2 = _rot(Q1[q1], a, b), _rot(Q2[q2], M[a], M[b])
        for x, y in zip(r1[2:], r2[2:]):
            if M.setdefault(x, y) != y:
                return None
        for i in range(4):
            u, w = r1[i], r1[(i + 1) % 4]
            n1 = [q for q in e1[frozenset((u, w))] if q != q1]
            n2 = [q for q in e2.get(frozenset((M[u], M[w])), []) if q != q2]
            if len(n1) != len(n2):
                return None
            if n1:
                stack.append((n1[0], n2[0], u, w))
    ok = len(match) == len(Q1) and len(set(match.values())) == len(Q2)
    return M if ok else None
