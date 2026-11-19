"""Edge-based transfinite interpolation (TFI) of a hex block from its 12 edge curves.

    X(u,v,w) =   sum_{x-edges (j,k)} phi_j(v) phi_k(w) E^x_jk(u)
               + sum_{y-edges (i,k)} phi_i(u) phi_k(w) E^y_ik(v)
               + sum_{z-edges (i,j)} phi_i(u) phi_j(v) E^z_ij(w)
               - 2 sum_{corners ijk} phi_i(u) phi_j(v) phi_k(w) C_ijk,      phi_0 = 1-t, phi_1 = t.

On every block face this reduces to the 2D Coons patch of that face's 4 edges, so two blocks that
share a face (and its edge curves) map their common face nodes to the same physical points.
"""
import numpy as np
from .sheets import EDGES, EIDX, other


def _phi(t, side):
    return t if side else 1.0 - t


def tfi(curves, P):
    """curves[i](t) -> (m,3) for local edge EDGES[i] = (d,a,b), t in [0,1] running along x_d from
    the corner with x_d = 0.  P: (N,3) reference coordinates.  Returns (N,3) physical coordinates."""
    P = np.asarray(P, float)
    X = np.zeros((len(P), 3))
    for i, (d, a, b) in enumerate(EDGES):
        o = other(d)
        X += (_phi(P[:, o[0]], a) * _phi(P[:, o[1]], b))[:, None] * curves[i](P[:, d])
    for x in (0, 1):
        for y in (0, 1):
            for z in (0, 1):
                C = curves[EIDX[(0, y, z)]](np.array([float(x)]))[0]
                X -= 2 * (_phi(P[:, 0], x) * _phi(P[:, 1], y) * _phi(P[:, 2], z))[:, None] * C
    return X


class Line:
    """Straight edge from A to B, uniform in t."""
    def __init__(self, A, B):
        self.A, self.B = np.asarray(A, float), np.asarray(B, float)

    def __call__(self, t):
        return self.A + np.asarray(t, float)[:, None] * (self.B - self.A)


class Arc:
    """Circular arc from A to B about `center` (A, B at the same radius), uniform in angle."""
    def __init__(self, A, B, center):
        self.c = np.asarray(center, float)
        self.a = np.asarray(A, float) - self.c
        b = np.asarray(B, float) - self.c
        r = np.linalg.norm(self.a)
        self.th = np.arccos(np.clip(self.a @ b / (r * np.linalg.norm(b)), -1, 1))
        e2 = b - (self.a @ b) / (r * r) * self.a
        self.e2 = e2 / np.linalg.norm(e2) * r

    def __call__(self, t):
        t = np.asarray(t, float)[:, None] * self.th
        return self.c + np.cos(t) * self.a + np.sin(t) * self.e2


class Reversed:
    """The same curve traversed from the other end."""
    def __init__(self, curve):
        self.curve = curve

    def __call__(self, t):
        return self.curve(1.0 - np.asarray(t, float))
