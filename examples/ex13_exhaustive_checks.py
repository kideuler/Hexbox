"""Exhaustive checks behind deterministic faces.

1. Face traces.  For every combination of per-edge excess levels 0..LMAX on the 4 x-edges and the
   4 y-edges (min level 0 per direction), build the block (topology only) and require every face to
   be isomorphic, with the boundary fixed, to the stand-alone template of its 4 counts.  stamp_faces
   raises if any face differs.  Sheets of the third direction never touch the faces normal to it,
   and the code treats the three directions alike, so this covers every in-plane combination of a face.
2. Template symmetry.  For every face tuple with sides in 5..11, rotating or reflecting the counts must
   rotate or reflect the template (D4 equivariance of the topology).  Blocks rely on this when they
   take a template computed in the face's canonical frame.  The node positions are also compared:
   they agree only to optimiser tolerance, which is why the canonical frame is needed.

    python examples/ex13_exhaustive_checks.py [LMAX=2]
"""
import sys, time, itertools
sys.path.insert(0, ".")
import numpy as np
from hexbox.deterministic import build_block_det, face_template
from hexbox.faceiso import match_nodes
from hexbox.sheets import EIDX

CYC = [(0, 0), (1, 0), (1, 1), (0, 1)]


def face_traces(LMAX=2, B=(5, 6, 7)):
    pats = [k for k in itertools.product(range(LMAX + 1), repeat=4) if min(k) == 0]
    bad, t = [], time.time()
    for kx in pats:
        for ky in pats:
            n = np.zeros(12, int)
            for d, K in enumerate((kx, ky, (0, 0, 0, 0))):
                for (a, b), k in zip(CYC, K):
                    n[EIDX[(d, a, b)]] = B[d] + 2 * k
            try:
                build_block_det(n, smooth=False, verbose=False)
            except RuntimeError as e:
                bad.append((kx, ky, str(e)))
    return len(pats) ** 2, bad, time.time() - t


def d4():
    rot = (lambda P: np.c_[1 - P[:, 1], P[:, 0]], lambda c: (c[3], c[0], c[1], c[2]))
    ref = (lambda P: np.c_[1 - P[:, 0], P[:, 1]], lambda c: (c[0], c[3], c[2], c[1]))
    group = []
    for k in range(4):
        for r in range(2):
            def gP(P, k=k, r=r):
                P = ref[0](P) if r else P
                for _ in range(k):
                    P = rot[0](P)
                return P
            def gc(c, k=k, r=r):
                c = ref[1](c) if r else c
                for _ in range(k):
                    c = rot[1](c)
                return c
            group.append((gP, gc))
    vals = {0: [6, 8, 10], 1: [5, 7, 9, 11]}
    tuples = [(a, b, c, d) for pa in (0, 1) for pb in (0, 1) for a in vals[pa] for c in vals[pa]
              for b in vals[pb] for d in vals[pb]]
    checked, bad, dpos = 0, [], 0.0
    for cnt in tuples:
        UV, Q = face_template(*cnt)
        for gP, gc in group[1:]:
            UV2, Q2 = face_template(*gc(cnt))
            M = match_nodes(gP(UV), Q, UV2, Q2)
            checked += 1
            if M is None:
                bad.append((cnt, gc(cnt)))
                continue
            src, dst = np.fromiter(M.keys(), int), np.fromiter(M.values(), int)
            dpos = max(dpos, float(np.abs(gP(UV)[src] - UV2[dst]).max()))
    return len(tuples), checked, bad, dpos


if __name__ == "__main__":
    LMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    nb, bad, dt = face_traces(LMAX)
    print(f"1. face traces: {nb} blocks (excess levels 0..{LMAX} on x- and y-edges), {6 * nb} faces, "
          f"{len(bad)} mismatches, {dt:.0f}s")
    for b in bad[:10]:
        print("   ", b)
    nt, nc, bad, dpos = d4()
    print(f"2. template symmetry: {nt} face tuples x 7 non-trivial symmetries = {nc} checks, "
          f"{len(bad)} topology mismatches; node positions differ by up to {dpos:.1e}")
    for b in bad[:10]:
        print("   ", b)
