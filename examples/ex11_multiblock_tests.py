"""Multi-block tests.  Every block is meshed on its own from block-local data (optionally in separate
processes), then the blocks are assembled and the global mesh is verified.

    python examples/ex11_multiblock_tests.py          -> table on stdout + figs/11_multiblock_tests.md
"""
import sys, time
sys.path.insert(0, ".")
import numpy as np
from hexbox.multiblock import box_grid, cubed_sphere, scrambled, same_partition, _merge_by_position

GRID = lambda: box_grid((3, 2, 2), (3.0, 2.0, 2.0), jitter=0.15, seed=1)
H_GRID = lambda x: 0.05 + 0.06 * np.linalg.norm(x)          # fine near the origin corner
SPHERE = lambda: cubed_sphere(a=0.4)
H_SPHERE = lambda x: 0.1 - 0.035 * x[0]                      # fine toward the +x pole
CUBE27 = lambda: box_grid((3, 3, 3), (3.0, 3.0, 3.0), jitter=0.2, seed=2)

rows = []


def run(name, mb, mode="det", by="keys", workers=0, expect=True):
    t = time.time()
    parts = mb.mesh_blocks(mode=mode, workers=workers)
    dt = time.time() - t
    mb.assemble(parts, by=by)
    r = mb.verify()
    if by == "keys":
        gp = _merge_by_position(np.vstack([p["X"] for p in parts]), mb.tol)
        r["keys_eq_merge"] = same_partition(mb.node_of, gp)
    c = np.array(list(mb.counts.values()))
    r.update(name=name, mode=mode, by=by, time=dt, counts=f"{c.min()}-{c.max()}", ok=(expect is None or r["conformal"] == expect))
    rows.append(r)
    print(f"{name:34s} conformal={r['conformal']!s:5s} unmatched={r['unmatched_quads']:4d} hexes={r['hexes']:6d} "
          f"minSJ={r['min_sj']:.2f} meanSJ={r['mean_sj']:.2f} spread={r['spread']:.1e} "
          f"keys==merge={r.get('keys_eq_merge', '-')} {dt:.1f}s", flush=True)
    return parts


if __name__ == "__main__":
    # 1. block grid, counts from a graded size field
    g = GRID(); g.assign_counts(H_GRID)
    serial = run("grid 3x2x2, size field", g)
    # 2. same blocks meshed in 4 worker processes: must be bitwise identical to the serial run
    par = run("grid 3x2x2, 4 processes", g, workers=4)
    rows[-1]["bitwise_eq_serial"] = all(np.array_equal(a["X"], b["X"]) and np.array_equal(a["H"], b["H"])
                                        for a, b in zip(serial, par))
    print("   4-process result bitwise identical to serial:", rows[-1]["bitwise_eq_serial"])
    # 3. random block orientations + random global vertex numbering
    for seed in range(5):
        run(f"grid scrambled (seed {seed})", scrambled(g, seed))
    # 4. 7-block ball: curved edges, 3 blocks around each radial edge
    s = SPHERE(); s.assign_counts(H_SPHERE)
    run("ball 7 blocks, size field", s)
    for seed in range(3):
        run(f"ball scrambled (seed {seed})", scrambled(s, seed))
    # 5. adversarial counts: random parity per chain, independent random counts 5..13 per edge
    for seed in range(3):
        c = CUBE27(); c.random_counts(5, 13, seed)
        run(f"27 blocks, random counts (seed {seed})", c)
    # 6. controls: the same assemblies without the ingredients that make faces agree
    run("control: free mode (grid)", g, mode="free", by="position", expect=False)
    run("control: local frames (grid scrambled)", scrambled(g, 0), mode="det-local", by="position", expect=None)

    with open("figs/11_multiblock_tests.md", "w") as fh:
        fh.write("| test | counts | hexes | conformal | unmatched quads | min SJ | mean SJ | keys = merge | time |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r['name']} | {r['counts']} | {r['hexes']} | {r['conformal']} | {r['unmatched_quads']} | "
                     f"{r['min_sj']:.2f} | {r['mean_sj']:.2f} | {r.get('keys_eq_merge', '-')} | {r['time']:.1f}s |\n")
    bad = [r["name"] for r in rows if r["ok"] is False]
    print("ALL EXPECTATIONS MET" if not bad else f"UNEXPECTED: {bad}")
