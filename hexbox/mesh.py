"""Unstructured hex mesh in the reference cube [0,1]^3 + pillowing (sheet insertion).

Hex node ordering is VTK: 0-3 bottom (counter-clockwise seen from +z), 4-7 top.
"""
import numpy as np

# outward-oriented local faces of a VTK hex
HEX_FACES = np.array([(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
                      (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
HEX_EDGES = np.array([(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                      (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)])


class HexMesh:
    def __init__(self, X, H, base):
        self.X = np.asarray(X, float)        # (N,3) node coordinates
        self.H = np.asarray(H, int)          # (M,8) connectivity
        self.base = np.asarray(base, int)    # (M,3) base-grid cell (i,j,k) each hex descends from

    @property
    def fixed(self):
        """(N,3) bool: node lies on the cube face x_d = 0 or 1."""
        return (np.abs(self.X) < 1e-12) | (np.abs(self.X - 1) < 1e-12)

    @staticmethod
    def structured(nx, ny, nz):
        I, J, K = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1), indexing="ij")
        X = np.c_[I.ravel() / nx, J.ravel() / ny, K.ravel() / nz]
        nid = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k
        i, j, k = [a.ravel() for a in np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")]
        H = np.c_[nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                  nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)]
        return HexMesh(X, H, np.c_[i, j, k])

    def faces(self):
        """All hex faces: (M*6,4) node ids (outward), owner hex, sorted-key -> list."""
        F = self.H[:, HEX_FACES].reshape(-1, 4)
        owner = np.repeat(np.arange(len(self.H)), 6)
        return F, owner

    def face_adjacency(self):
        F, owner = self.faces()
        key = np.sort(F, axis=1)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        return F, owner, inv.ravel(), cnt

    def boundary_quads(self):
        F, owner, inv, cnt = self.face_adjacency()
        return F[cnt[inv] == 1]

    def pillow(self, S, shrink=0.35):
        """Insert one dual sheet around shrink set S (bool mask over hexes).

        Interface = faces between S and not-S.  Interface nodes are duplicated, S keeps the
        copies, and one new hex is created per interface face.  Returns #new hexes.
        """
        S = np.asarray(S, bool)
        F, owner, inv, cnt = self.face_adjacency()
        inS = S[owner]
        # a face key is interface iff it has 2 owners, exactly one in S
        nS = np.bincount(inv, weights=inS, minlength=cnt.size)
        is_if = (cnt[inv] == 2) & (nS[inv] == 1) & inS
        IF = F[is_if]                      # interface faces as seen from S (outward from S)
        IFbase = self.base[owner[is_if]]
        nodes = np.unique(IF)
        new_id = np.full(len(self.X), -1)
        new_id[nodes] = len(self.X) + np.arange(len(nodes))
        # move S hexes onto the copies
        HS = self.H[S]
        m = new_id[HS] >= 0
        HS[m] = new_id[HS[m]]
        H = self.H.copy(); H[S] = HS
        # new node positions: pulled toward the centroid of adjacent S hexes
        cen = self.X[self.H[S]].mean(axis=1)
        acc = np.zeros((len(self.X), 3)); w = np.zeros(len(self.X))
        np.add.at(acc, self.H[S].ravel(), np.repeat(cen, 8, axis=0)); np.add.at(w, self.H[S].ravel(), 1)
        Xn = self.X[nodes] + shrink * (acc[nodes] / w[nodes, None] - self.X[nodes])
        fx = self.fixed[nodes]
        Xn[fx] = self.X[nodes][fx]         # stay on the cube faces/edges the old node was on
        newH = np.c_[new_id[IF], IF]       # bottom = S-side copy, top = old face
        self.X = np.vstack([self.X, Xn])
        self.H = np.vstack([H, newH])
        self.base = np.vstack([self.base, IFbase])
        return len(newH)
