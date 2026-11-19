"""Verify: every face of a deterministic-mode block == the face template built from its 4 counts alone."""
import sys, time; sys.path.insert(0, ".")
import numpy as np
from hexbox.deterministic import build_block_det, face_template
from hexbox.sheets import EIDX, other
from hexbox.faceiso import same_topology
from hexbox.smooth import scaled_jacobian

def ekey(d, c):
    o = other(d); return (d, c[o[0]], c[o[1]])

def block_faces(m, n):
    """Yield (counts a,b,c,d ; UV ; quads) for the 6 faces in their local frames."""
    Q = m.boundary_quads()
    for f in range(3):
        p, q = other(f)
        for s in (0, 1):
            on = np.all(np.abs(m.X[Q][:, :, f] - s) < 1e-9, axis=1)
            cnt = [n[EIDX[ekey(p, {f: s, q: 0})]], n[EIDX[ekey(q, {f: s, p: 1})]],
                   n[EIDX[ekey(p, {f: s, q: 1})]], n[EIDX[ekey(q, {f: s, p: 0})]]]
            yield (f, s), cnt, m.X[:, [p, q]], Q[on]

rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
ok_faces = tot_faces = 0; worst = 1; t0 = time.time()
for it in range(N):
    n = np.zeros(12, int)
    for d in range(3):
        par = rng.integers(2)
        for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            n[EIDX[(d, a, b)]] = 2 * rng.integers(3, 7) + par        # 6..13 with fixed parity
    m = build_block_det(n, verbose=False)
    worst = min(worst, scaled_jacobian(m.X, m.H).min())
    for face, cnt, UV, Q in block_faces(m, n):
        UVt, Qt = face_template(*map(int, cnt))
        same = same_topology(UV, Q, UVt, Qt)
        ok_faces += same; tot_faces += 1
        if not same:
            print("MISMATCH", n.tolist(), "face", face, "counts", cnt)
print(f"{N} blocks, {tot_faces} faces: {ok_faces} identical to the stand-alone template; "
      f"worst minSJ {worst:.3f}; {time.time()-t0:.0f}s")
