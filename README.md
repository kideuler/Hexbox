# Hexbox: reference-space hex meshing of one block from 12 arbitrary edge counts

Given the number of mesh edges on each of the 12 edges of a hex block, build an all-hex,
conformal mesh of the reference cube [0,1]^3 (faces + interior). TFI maps it to the physical block
afterwards.

## 1. The only constraint the user must respect

**Every block face must carry an even number of boundary edges** (the sum of its 4 counts is even).

This condition is necessary: a quad mesh of a disk exists only if its boundary edge count is even.
The construction below shows it is also sufficient for one block. Equivalent statements:

* the set of odd-count edges is a *cut* of the cube graph: colour the 8 corners black/white; an edge
  must be odd exactly when its two corners have different colours;
* so the parity of the 12 counts has 7 free bits (128 patterns out of 4096), and the magnitudes are
  completely free.

Opposite edges are not required to be equal. Two parallel edges of the same block can be 6 and 14.

## 2. Design: think in dual sheets

Every layer ("sheet") of a hex mesh is a surface. Its boundary is a loop on the cube surface, and that
loop crosses each block edge once per mesh edge the sheet contributes there. A structured block has only
*belts* (flat layers). To decouple the counts we allow five sheet types, each with a fixed signature on
the 12 edge counts:

| sheet | edge counts it adds | irregularity it creates |
|---|---|---|
| belt (direction d) | +1 on the 4 edges of direction d | none (structured layer) |
| wrap (around edge e) | +1 on the 4 other edges at e's two ends | one 3-5 pair, extruded along e (2.5D) |
| corner (at corner v) | +1 on the 3 edges at v | one 3-5 pair on each of the 3 faces at v |
| U-wrap (face F, dir d) | +2 on both edges of F parallel to d | U-turn (3-to-1 type) extruded across F |
| U-turn (edge e) | +2 on e only | local 3-to-1 type refinement |

Every parity-valid count vector is a nonnegative combination of these sheets. Corner sheets generate
all 128 parity patterns: their stars span the cube graph's cut space. U-turns absorb any even remainder,
and belts carry the bulk. Several copies of the same sheet nest, so a corner +5 still produces only one
3-5 pair per face.

## 3. Algorithm (`hexbox/`)

1. **Decompose** (`sheets.py`): a small MILP picks multiplicities. It minimises the weighted number of
   *distinct* irregular sheet families, so the mesh stays as structured as possible, and requires at
   least 5 belts per direction when feasible, so the transitions have room.
2. **Build** (`build.py`, `mesh.py`): start from a structured belt grid (Bx, By, Bz). Insert every other
   sheet by *pillowing* a box of base cells (corner box, edge prism, mid-edge box, ...), once per
   multiplicity. Pillowing duplicates the box's interface nodes and adds one hex per interface face. It
   is always conformal, and nodes on cube faces/edges stay there.
3. **Place nodes** (`smooth.py`): block-edge nodes are spaced uniformly. Then a constrained Laplacian
   runs, followed by an untangle-and-smooth optimisation. That is an inverse mean-ratio objective with a
   regularised determinant (Escobar et al.), using continuation in the regularisation and L-BFGS-B.
   Face nodes slide in their face plane, and interior nodes are free.
4. **Retry** over (box size, minimum belts) when the mesh is inverted, keeping the best result.
5. **Verify**: exact edge counts; every face shared by at most 2 hexes; every unshared face lies on the
   cube; min scaled Jacobian.

Stress test (`examples/ex05_stress.py`) on fully random parity-valid counts, each edge independent, up
to 2.3:1 between parallel edges:

| counts | valid | min SJ (p5 / median) | mean SJ (median) |
|---|---|---|---|
| 6..14 | 40/40 | 0.30 / 0.47 | 0.88 |
| 4..8  | 40/40 | 0.23 / 0.46 | 0.85 |
| 2..5  | 38/40 (2 had no room) | 0.36 / 0.51 | 0.78 |

Real assignments, which have a few transitions rather than 12 independent random counts, are better.
A single corner +3 gives min SJ 0.73, and an edge-wrap x2 gives 0.85.

## 4. Multi-block caveat: who decides the face mesh?

Here the face meshes come from the block's 3D decomposition. A shared face must be meshed identically
by both blocks (`figs/07_multiblock_issue.png`). If face meshes must instead be a function of the
face's own 4 counts (purely local, order-independent), some patterns become impossible. Example: two odd
corners on a face diagonal. The shared face is (7,7,7,7) and fully symmetric, so no local rule can put
the needed two 3-5 pairs on one diagonal.

`hexbox/canonical.py` enumerates this. With the best local rule found ("an odd face turns once, at the
corner between its two odd sides"), 44 of the 128 parity patterns stay fillable by clean sheets.
Exact matching of the 3-5 positions adds further conditions.

The alternative keeps full freedom: one global integer program over all blocks decides each shared
face's transition pattern, and each block then builds its interior to match. This fits naturally where
interval assignment already happens.

## 5. Usage

```bash
python3 -m venv .venv && .venv/bin/pip install numpy scipy matplotlib
.venv/bin/python examples/ex03_random.py      # random blocks -> figs/03_random.png
```

```python
from hexbox.build import build_block
# edge order: x-edges (y,z)=00,01,10,11 | y-edges (x,z)=00,01,10,11 | z-edges (x,y)=00,01,10,11
m = build_block([9,6,6,6, 6,6,9,6, 6,6,9,6])   # corner (1,0,0) refined by +3
m.X, m.H                                       # nodes (N,3) in [0,1]^3, hexes (M,8) VTK order
```

## 6. Known limits / next steps

* Transitions need room. Counts of 2-3 on an edge that must host a transition can fail ("no room").
* Shared-face canonicalisation (section 4) is analysed but not built yet.
* Block-edge nodes are uniform in reference space. Grading belongs in the edge parametrisation
  before TFI.
