# Hexbox: all-hex meshing of a block from its 12 edge counts

The user prescribes the number of mesh edges on each of the 12 edges of a hex block. Hexbox fills the
block's faces and interior with a conformal all-hex mesh that has exactly those counts. All meshing
happens in the reference cube [0,1]^3; transfinite interpolation (TFI) then maps the result onto the
physical block.

There are two modes:

| | free mode (`build.py`) | deterministic mode (`deterministic.py`) |
|---|---|---|
| rule on the 12 counts | every face's 4 counts sum to an even number | the 4 parallel edges of each direction share parity |
| parity patterns allowed | 128 of 4096 | 8 (one parity bit per direction) |
| magnitudes | free | free (parallel edges may differ by any even amount) |
| a face's mesh is decided by | the block's own 3D construction | the face's own 4 counts |
| several blocks | neighbours generally disagree on shared faces | conformal by construction; each block meshed alone, in parallel |

Deterministic mode plus `multiblock.py` is the multi-block method. Free mode is the most permissive
single-block mesher and the source of the ideas.

**Contents.** 1 Conventions · 2 What every filling must satisfy · 3 Dual sheets · 4 Free mode ·
5 Multi-block: who decides a shared face? · 6 Deterministic mode · 7 The multi-block pipeline ·
8 Tests and results · 9 Usage · 10 Limitations and next steps · 11 File map

---

## 1. Problem and conventions

**Input:** 12 integers n_e >= 1, the number of mesh edges on block edge e.

**Output:** an all-hex mesh of [0,1]^3 in which every quad is shared by at most two hexes and every
quad used by only one hex lies on the cube's surface (no hanging nodes, no AMR-style refinement). Block
edge e carries exactly n_e mesh edges, spaced uniformly.

Conventions used throughout the code:

* **Edges** are keyed `(d, a, b)`: direction d in {0,1,2} (x, y, z), and (a, b) in {0,1}^2, the
  position along the two other axes in increasing axis order. Count vectors are ordered
  `x-edges (y,z)=00,01,10,11 | y-edges (x,z)=00,01,10,11 | z-edges (x,y)=00,01,10,11`.
* **Hexes** use VTK node order (0-3 bottom, counter-clockwise seen from +z; 4-7 top).
* **Faces** are `(f, s)`, the face x_f = s. Its local coordinates are (u, v) = (x_p, x_q), with (p, q)
  the other two axes in increasing order. A face's counts are listed (a, b, c, d) = sides v=0, u=1,
  v=1, u=0 (bottom, right, top, left).
* A node is a **face, edge or corner node** when 1, 2 or 3 of its coordinates are exactly 0 or 1.

## 2. What every filling must satisfy

**Faces.** A quad mesh of a disk has an even number of boundary edges: counting quad-edge
incidences gives 4Q = 2E_interior + E_boundary. Every block face is a quad-meshed disk, so
**the 4 counts of every face must sum to an even number.** For a single block this is also enough
(section 4).

Two equivalent statements:

* The odd edges form a *cut* of the cube's graph. Colour the 8 corners black and white; an edge must
  be odd exactly when its two corners have different colours.
* The parities therefore have 7 free bits (2^8 colourings, each cut obtained twice): 128 of the 4096
  parity patterns are allowed. Magnitudes are unrestricted, so parallel edges of one block may be 6
  and 14.

**Blocks.** A hex mesh has an even number of boundary quads, since 6H = 2F_interior + F_boundary.
This second parity fact limits face-local rules (section 5.2).

## 3. Dual sheets

Take any mesh edge, jump to the opposite edge of a quad that contains it, and repeat. The edges
reached form a class. The hexes containing them form a **sheet**: a layer of hexes whose mid-surface
(the dual surface) cuts every edge of the class once. In a structured grid the sheets are the i-, j-
and k-layers. Every mesh edge on a block edge belongs to exactly one sheet, so **each count n_e is the
number of times the sheets' mid-surfaces cut block edge e.** On a block face, a mid-surface leaves a
curve called a **chord** of the face's quad mesh. Choosing counts therefore means choosing sheets with
known cutting patterns.

Free mode uses five sheet types:

| sheet | counts it adds | what the block faces show |
|---|---|---|
| belt in direction d | +1 on the 4 edges of direction d | nothing irregular (a structured layer) |
| wrap around edge e | +1 on the 4 other edges at e's two ends | a 3-5 vertex pair, extruded along e |
| corner at v | +1 on the 3 edges at v | a 3-5 pair on each of the 3 faces at v |
| U-wrap (face F, direction d) | +2 on both edges of F parallel to d | a U-turn on the 2 faces next to F |
| U-turn at edge e | +2 on e only | a U-turn on each of the 2 faces at e |

(A *U-turn* is a chord that enters a face from one side and returns to the same side, the 2D
analogue of 3-to-1 refinement. A *turning* chord connects two adjacent sides and shows up as a 3-5
vertex pair.)

Every face-even count vector is a nonnegative integer combination of these sheets:

* Corner stars span the cut space, so they produce every allowed parity pattern.
* U-turns absorb any even remainder.
* Belts carry the bulk.

Several copies of one sheet nest inside each other, so a corner sheet used 5 times still gives a
single 3-5 pair per face.

Deterministic mode uses belts plus one more family, the **U-sheets** of section 6. A U-sheet adds +2
to a contiguous run of 1, 2 or 3 of the 4 parallel edges of one direction. The U-turn and U-wrap above
are the runs of length 1 and 2.

---

## 4. Free mode: one block, maximal freedom

Files: `sheets.py`, `mesh.py`, `build.py`, `smooth.py`. Free mode accepts every face-even assignment,
given some room (4.5).

### 4.1 Decompose the counts into sheets (`sheets.decompose`)

A small integer program runs over the 47-sheet catalogue: 3 belts, 12 wraps, 8 corners, 12 U-wraps
and 12 U-turns.

* **Variables:** a multiplicity x_k >= 0 (integer) and a binary y_k ("sheet k is used").
* **Constraints:**
  * sum_k x_k c_k = n, where c_k is the sheet's 12-vector of added counts;
  * x_k <= big · y_k;
  * every belt multiplicity is at least min_belt. The default is 5; if that is infeasible it is
    relaxed to 2, then 1, then 0. This leaves structured room around the transitions.
* **Objective:** sum_k cost_k y_k + 0.001 · sum_{k not a belt} x_k, with costs belt 0, wrap 2,
  corner 3, U-wrap 3, U-turn 4. This first minimises the number of distinct irregular sheet families
  (each is one kind of irregularity on the faces), then their total multiplicity.

`scipy.optimize.milp` (HiGHS) solves it in milliseconds.

### 4.2 Build the sheets by pillowing (`HexMesh.pillow`, `build._build`)

1. Start from a structured grid of N_x x N_y x N_z cells, where N_d is the number of belts in direction d.
2. Create every other sheet by **pillowing** a *shrink set* S of hexes:
   * the interface faces are the interior faces with exactly one of their two hexes in S (faces on
     the cube surface have only one hex and are never interface faces);
   * every interface node is duplicated, and the hexes of S switch to the copies;
   * one new hex is inserted on each interface face, between its old nodes and their copies.

   The result is always a conformal hex mesh, and the new layer is exactly one new sheet wrapped
   around S. The sheet ends where S touches the cube surface, so its chord on each block face is the
   outline of S's footprint there. Copies keep any coordinate that was exactly 0 or 1, so cube faces
   and edges stay flat.
3. **Base cells.** Every hex remembers the structured-grid cell it descends from, and a new pillow hex
   inherits the base cell of its S-side neighbour. Shrink sets are boxes of base cells, so a later box
   automatically contains the pillow hexes that earlier sheets created inside it. Later sheets
   therefore wrap around earlier ones, and repeating a box (multiplicity > 1) nests the copies outward.
4. **Shrink boxes.** "Near" means the k = round(pad · N) cells next to a side, clamped to [1, N-1]
   (pad = 0.4; the retries in 4.4 vary it).
   "Middle" means the middle third.

   | sheet | shrink box |
   |---|---|
   | corner at v | near v in all 3 axes |
   | wrap around e = (d, a, b) | near e in the 2 other axes; all cells along d |
   | U-turn at e | near e in the 2 other axes; middle along d |
   | U-wrap (f, s, d) | near the face x_f = s; middle along d; all cells along the third axis |

   The sheets are inserted in this order: wraps, U-wraps, corners, U-turns.

### 4.3 Place the nodes (`smooth.py`)

1. Space the block-edge nodes uniformly (`respace_block_edges`).
2. Run a Jacobi Laplacian (relaxation 0.6). Interior nodes average all their neighbours. Face nodes
   average only neighbours on the same face, so they slide within it. Edge and corner nodes are frozen.
3. Untangle and smooth (`optimize`) by minimising, over the free coordinates, the sum over every hex
   corner of the inverse mean ratio

       f = |J|_F^2 / (3 h(det J)^(2/3)),     h(D) = (D + sqrt(D^2 + 4 delta^2)) / 2

   where J holds the 3 edge vectors at the corner.
   * The regularised determinant h stays positive for inverted corners, so f is finite everywhere and
     its gradient pushes toward untangling (Escobar et al., CMAME 2003).
   * delta is lowered step by step: 0.3, 0.1, 0.03, 0.01, 0.001 times the median |det J|.
   * Each step is solved by L-BFGS-B with coordinates bounded to [0,1].
   * The analytic gradient matches finite differences to 1e-8.

### 4.4 Retries and checks (`build_block`)

`build_block` tries (pad, min_belt) = (0.4,5), (0.3,5), (0.5,5), (0.4,7), (0.35,4), (0.45,3). It keeps
the mesh with the best minimum scaled Jacobian and stops early once that reaches 0.2.

Every attempt must pass these checks:
* exact edge counts (a shrink box that came out empty raises);
* every quad shared by at most 2 hexes;
* every unshared quad on the cube surface;
* no repeated node within a hex.

If no attempt passes, it raises "could not fit the transitions".

### 4.5 Results (`examples/ex05_stress.py`)

Random face-even counts, each edge drawn independently (up to 2.3 : 1 between parallel edges, far
harsher than real assignments). "Valid" means exact counts, conformal, and min scaled Jacobian > 0.

| counts | valid | min SJ (p5 / median) | worst min SJ | mean SJ (median) |
|---|---|---|---|---|
| 6..14 | 40/40 | 0.30 / 0.47 | 0.29 | 0.88 |
| 4..8  | 40/40 | 0.42 / 0.51 | 0.38 | 0.85 |
| 2..5  | 8/40, the other 32 raise "could not fit" | 0.49 / 0.53 | 0.48 | 0.78 |

Transitions need room. With every count in 2..5 there are too few cells for the shrink boxes. Real
assignments, with a few transitions instead of 12 independent counts, do better: a single corner +3
gives min SJ 0.73 and an edge-wrap x2 gives 0.85.

Figures: `01_corner_sheet`, `02_gallery`, `03_random`, `04_optimize`, `06_cutaway`.

---

## 5. Multi-block: who decides a shared face?

### 5.1 The problem

In free mode a face's mesh is a by-product of the block's 3D sheet choice. Two blocks that share a
face see the same 4 counts but generally pick different sheets, so their face meshes differ and the
assembled mesh is not conformal (`figs/07_multiblock_issue.png`, `figs/13_free_mode_control.png`).

There are two ways out:

1. **Global:** one integer program over all blocks fixes the pattern of every shared face, and each
   block then builds its interior to match. This keeps the face-even freedom but couples all blocks
   through one global solve.
2. **Local:** each face mesh is a fixed function of that face's own 4 counts. Blocks then need no
   coordination: they can be meshed in any order, on any process. **Hexbox implements this.**

### 5.2 A local rule must restrict the counts

Assume a face's mesh depends only on its counts, up to rotation and reflection of the face. In other
words it does not depend on how the vertices happen to be numbered. Assume also that balanced faces
(a = c, b = d) are structured grids, so 7,7,7,7 is always the 7 x 7 grid. Now take the block

    [7,6,7,6, 7,6,7,6, 6,7,7,6]    bottom 7,7,7,7 | top 6,6,6,6 | each side 7,7,6,6

* Every face sum is even, so free mode meshes it.
* On each side face the two 7s meet at a bottom corner. The four sides are rotations of each other,
  so they get congruent meshes with the same quad count q.
* The block's boundary would then have 49 + 36 + 4q quads. That is odd, and no hex mesh has an odd
  number of boundary quads (section 2).

So any such rule must reject some assignments that free mode accepts.

### 5.3 Which restriction

`hexbox/canonical.py` explores local rules that still allow odd faces. The best one found is: an odd
face turns once, at the corner between its two odd sides, and balanced faces never turn. It admits 44
of the 128 parity patterns, and exact placement of the turning chords adds further conditions.

Deterministic mode takes the 8 of those 44 patterns that need no turning chord at all: equal parity
on parallel edges. Within them the magnitudes are completely free and every assignment works
(section 6). The other 36 patterns (isolated corner and edge-wrap transitions) are a possible
extension (section 10).

---

## 6. Deterministic mode (`hexbox/deterministic.py`)

### 6.1 The rule

**Within a block, the 4 edges of each direction have the same parity.** Opposite edges may differ by
any even amount (6 vs 14, 7 vs 13). This is the structured rule relaxed from "equal" to "equal mod 2",
and it implies the face-even rule.

Across a multi-block mesh, the rule means one parity per **chain**. A chain is a set of edges linked
by "parallel in some block": exactly the edges a structured mesher would force to be equal. The
counts along a chain vary freely in steps of 2.

**Sizes.** Defaults are L = 2 and D = 2 (6.2).
* A direction that carries a transition needs its smallest count to be at least 2L+1 = 5.
* The two directions crossing that transition need at least D+1 = 3.
* Blocks without transitions accept any counts.

Both size rules raise a clear `ValueError`.

### 6.2 Face templates

`face_template(a, b, c, d)` is the face mesh, a pure function of the 4 counts:

1. A structured core of min(a,c) x min(b,d) cells.
2. On the longer side of each opposite pair, |a - c|/2 (resp. |b - d|/2) nested **U-turns**. Each is a
   pillow of the core cells within D = 2 cells of that side and at least L = 2 cells from both of its
   ends (the *collar*). Its chord leaves the side L cells from one corner, runs D cells deep and
   returns L cells from the other corner.
3. Node positions come from uniform boundary nodes, a Laplacian, then a 2D version of the optimiser
   in 4.3:
   * the objective is the inverse mean ratio (|e1|^2 + |e2|^2) / (2 h(e1 x e2)) at every quad corner;
   * delta steps through 0.3, 0.1, 0.01, 0.001 times the median quad area;
   * L-BFGS-B, with boundary nodes fixed.

The template is built as a one-cell-thick slab, reusing the 3D pillow code, and is cached per count
tuple. 7,7,7,7 is always the plain 7 x 7 grid (`figs/10_face_templates.png`).

**Symmetry.** The construction treats the 4 sides alike. Features from different sides meet at most at
one point (because D <= L), so the order in which sides are processed doesn't matter. The topology is
therefore **D4-equivariant**: rotating or reflecting the counts rotates or reflects the mesh. Node
positions are equivariant only up to the optimiser's tolerance; they differ by up to 7e-5 when a
template is computed in a rotated frame. That is why multi-block meshing evaluates every face in a
canonical frame (7.3), so both blocks run the identical computation.

### 6.3 Building the block

For counts n that satisfy the rule:

1. **Base grid.** B_d = the smallest of the 4 counts in direction d. Start from the structured
   B_x x B_y x B_z grid.
2. **Excess.** Each edge's excess is k = (n - B_d)/2, an integer. At least one edge per direction
   has k = 0.
3. **Skyline.**
   * Order the 4 edges of direction d cyclically around the d-axis: (0,0) -> (1,0) -> (1,1) -> (0,1)
     in the other two coordinates. Consecutive edges share a block face.
   * For each level l = 1 .. max k, the edges with k >= l form one or more maximal cyclic runs. They
     are never all four, since some k is 0.
   * Each run is one **U-sheet** (l, d, run). An edge with excess k lies in exactly one run at each
     level 1..k, so it gains exactly 2k.
4. **Regions.** The shrink set of U-sheet (l, d, run) is the base cells with L <= x_d < B_d - L,
   crossed with a cross-section in the other two directions. The cross-section is the D x D corner
   square at every edge of the run, plus the D-deep strip along every block face shared by two
   consecutive edges of the run.

   | run length | cross-section | trace on the 4 side faces |
   |---|---|---|
   | 1 (U-turn) | one corner square | U-turns on the 2 faces at the edge |
   | 2 (U-wrap) | a strip along one face | straight chords across that face; U-turns on its 2 neighbours |
   | 3 (L-arc) | an L of two strips | straight chords across 2 faces; U-turns on the other 2 |

   The region spans L <= x_d < B_d - L, so it never touches the two end faces x_d = 0 and 1.
5. **Pillow** the U-sheets in increasing level. The sort is stable, so within a level the order is
   x, y, z.
6. **Stamp the faces.** Each face's mesh is matched to its template by a boundary-anchored isomorphism
   (`faceiso.match_nodes`): boundary nodes are matched by position, then the match is propagated quad
   by quad. With the boundary fixed the map is unique. The template's node positions are then copied
   onto the face. A mismatch raises.
7. **Interior.** Run the Laplacian and the optimiser of 4.3 with every face, edge and corner node
   frozen.

### 6.4 Why every face equals its template

Take a face F whose two edges i and j run in direction d and whose other two edges run in direction q.
Sheets of the third direction never touch F. Go through the d-sheets level by level:

* **l <= min(k_i, k_j):** i and j are both active and consecutive, so they lie in the same run. The
  region covers F's whole band L <= x_d < B_d - L. Its trace on F is two **straight chords**, at
  x_d = L and x_d = B_d - L, running from i to j.
* **min < l <= max:** only the larger edge, say i, is active. The run contains i but not j, and the
  region meets F only in the D-deep band along i. The trace is a **U-turn** on i, spanning the band
  and D cells deep.

So F gets min(k_i, k_j) pairs of straight chords, which is the template's core:
min(n_i, n_j) = B_d + 2 · min(k). It also gets |k_i - k_j| = |n_i - n_j| / 2 U-turns on the larger
edge. Both match the template.

The positions match too. Pillowing in increasing level puts every later sheet outside the earlier
ones wherever they share a boundary (base-cell inheritance, 4.2). Walking along edge i from a corner,
you meet:
1. L base columns (the collar);
2. the U-turn ends, from the highest level down;
3. the straight chords, from the highest level down;
4. the remaining base columns.

So each U-turn spans exactly the core columns [L, core - L) with an L-column collar, as in the
template. Pillowing the high levels first would put the straight chords outside the U-turns and widen
the collar.

The two directions do not interfere. A d-U-turn reaches D <= L cells into F, while every q-feature
starts L cells away from F's d-edges, so they meet at most at a point, exactly as in the template.

Section 8.2 backs this argument with exhaustive checks.

---

## 7. The multi-block pipeline (`hexbox/multiblock.py`, `hexbox/tfi.py`)

### 7.1 Input

* `V`: block vertex coordinates.
* `blocks`: 8 global vertex ids per block, in VTK order and right-handed. Any local orientation is
  fine; neighbouring blocks need not agree on it.
* Optional curved edges `{(ga, gb): curve}`. A curve maps t in [0,1] from vertex ga to vertex gb
  (`tfi.Line`, `tfi.Arc`). Unlisted edges are straight, and an edge used from its other end is reversed
  automatically.

### 7.2 Chains and counts

`chains()` groups the global edges by union-find over "parallel in some block". The deterministic
rule asks for one parity per chain. Any assignment with one parity per chain is valid, and magnitudes
are unconstrained. Two helpers produce counts:

* `assign_counts(h)` turns a target size field into counts:
  * the wanted count is n_e ~ length(e) / h(midpoint);
  * each chain gets the parity with the smaller total squared log-error;
  * each edge is rounded to the nearest count of that parity, at least nmin = 5.
* `random_counts` draws a random parity per chain and an independent random count per edge, for
  testing.

### 7.3 Canonical face frames

Two blocks sharing a face may see it rotated or mirrored relative to each other. So every face gets a
frame fixed by global vertex ids:

* the origin is the corner with the smallest id;
* the U axis points toward the neighbouring corner with the smaller id;
* the V axis points toward the other neighbour.

The counts are listed in that frame, so both blocks make the identical call `face_template(counts)` and
get bit-identical (U, V) positions. Each block then maps them into its local face coordinates by
(u, v) = P_o + U E_u + V E_v, with E_u, E_v in {±e1, ±e2}.

### 7.4 The per-block job

`mesh_block_job` needs only the block's own data:
* its 12 counts;
* its 8 global vertex ids, for the canonical frames;
* its 12 edge curves.

It builds the reference mesh (section 6), checks the edge counts, maps the nodes by TFI and labels them
(7.6). It needs no other block's data or result, so blocks can be meshed in any order, in separate
processes or on separate ranks. `mesh_blocks(workers=n)` uses a process pool.

### 7.5 TFI

Edge-based transfinite interpolation from the 12 edge curves E, with phi_0(t) = 1 - t and
phi_1(t) = t:

    X(u,v,w) =   sum over x-edges (j,k):       phi_j(v) phi_k(w) E^x_jk(u)
               + sum over y-edges (i,k):       phi_i(u) phi_k(w) E^y_ik(v)
               + sum over z-edges (i,j):       phi_i(u) phi_j(v) E^z_ij(w)
               - 2 sum over corners (i,j,k):   phi_i(u) phi_j(v) phi_k(w) C_ijk

* It interpolates all 12 edges and reproduces the trilinear map when the edges are straight (checked
  to 6e-16).
* On each block face it reduces to the bilinearly blended Coons patch of that face's 4 edges.
* The Coons patch does not depend on which of the 8 symmetries of the square parametrises the face.

Two blocks sharing a face therefore send the same reference node to the same physical point. The
observed differences are at round-off level, 4e-16 to 2e-15.

### 7.6 Global numbering without search

Every node gets a label computed from block-local data only:

| node | label |
|---|---|
| block corner | its global vertex id |
| on a block edge | (lower id, higher id, segment index counted from the lower-id end) |
| on a block face | (the face's 4 sorted ids, node index in the canonical template) |
| interior | (block, local index) |

Blocks sharing a vertex, edge or face give the same labels to the same nodes. Global numbering is
therefore a dictionary lookup, with no geometric search and no communication. That is
`assemble(parts, by="keys")`. `by="position"` merges points closer than 1e-8 x the domain size and is
used only as a cross-check.

### 7.7 Verification (`MultiBlock.verify`)

* Every quad is used by at most 2 hexes.
* The quads used once are exactly the quads of the block faces on the domain boundary. Any other quad
  used once is an **unmatched interface quad**, i.e. a hanging face.
* No hex is degenerate, and no two distinct nodes sit at the same point.
* Each block's edge counts are exact (checked inside the job).
* The key numbering groups nodes exactly as the position merge does, and the spread of positions
  among nodes with the same key is reported.
* Scaled Jacobians are reported.

---

## 8. Tests and results

### 8.1 Multi-block (`examples/ex11_multiblock_tests.py`; figures from `ex12_multiblock_figs.py`)

The test domains:

* **grid:** a 3 x 2 x 2 block box with jittered interior vertices. The size field 0.05 + 0.06|x| is
  fine at one corner, giving counts 5..13.
* **scrambled:** the same blocks and counts, but every block gets a random local orientation (one of
  the 24 cube rotations) and all global vertex ids are randomly permuted.
* **ball:** 7 blocks, a centre cube [-0.4, 0.4]^3 plus 6 shell blocks. Sphere edges are great-circle
  arcs, and 3 blocks meet around each radial edge. The size field is finer toward +x, so one shell
  block goes from 9 to 17 cells between its inner and outer face.
* **27 blocks, random counts:** a jittered 3 x 3 x 3 box with a random parity per chain and an
  independent random count in 5..13 on every edge.
* **controls:** the grid in free mode, and the scrambled grid in deterministic mode but with each
  block using its own local face frames instead of the canonical ones.

| test | counts | hexes | conformal | unmatched quads | min SJ | mean SJ | keys = merge | time |
|---|---|---|---|---|---|---|---|---|
| grid 3x2x2, size field | 5-13 | 2834 | True | 0 | 0.33 | 0.91 | True | 1.5s |
| grid 3x2x2, 4 processes | 5-13 | 2834 | True | 0 | 0.33 | 0.91 | True | 0.9s |
| grid scrambled (seed 0) | 5-13 | 2834 | True | 0 | 0.33 | 0.91 | True | 1.5s |
| grid scrambled (seed 1) | 5-13 | 2834 | True | 0 | 0.33 | 0.91 | True | 1.4s |
| grid scrambled (seed 2) | 5-13 | 2834 | True | 0 | 0.33 | 0.91 | True | 1.4s |
| grid scrambled (seed 3) | 5-13 | 2834 | True | 0 | 0.33 | 0.91 | True | 1.4s |
| grid scrambled (seed 4) | 5-13 | 2834 | True | 0 | 0.33 | 0.91 | True | 1.4s |
| ball 7 blocks, size field | 5-17 | 4618 | True | 0 | 0.25 | 0.73 | True | 3.4s |
| ball scrambled (seed 0) | 5-17 | 4618 | True | 0 | 0.25 | 0.73 | True | 3.4s |
| ball scrambled (seed 1) | 5-17 | 4620 | True | 0 | 0.25 | 0.73 | True | 3.3s |
| ball scrambled (seed 2) | 5-17 | 4618 | True | 0 | 0.25 | 0.73 | True | 3.3s |
| 27 blocks, random counts (seed 0) | 5-13 | 17243 | True | 0 | 0.19 | 0.87 | True | 13.4s |
| 27 blocks, random counts (seed 1) | 5-13 | 16986 | True | 0 | 0.27 | 0.86 | True | 13.8s |
| 27 blocks, random counts (seed 2) | 5-13 | 17379 | True | 0 | 0.20 | 0.87 | True | 12.9s |
| control: free mode (grid) | 5-13 | 2883 | False | 1282 | 0.51 | 0.94 | - | 2.6s |
| control: local frames (grid scrambled) | 5-13 | 2834 | False | 650 | 0.33 | 0.91 | - | 1.4s |

What the table shows:

* **Every deterministic assembly is conformal:** 15 of them, covering grids, scrambled orientations
  and numbering, the ball, and random counts. None has an unmatched quad, and the numbering by keys
  groups nodes exactly as the geometric merge does.
* **Parallel runs are bit-identical:** the 4-process run gives the same arrays as the serial run.
* **Orientation and numbering don't matter:** scrambling both leaves the faces unchanged; only the
  round-off spread moves, from 5e-16 to 2e-15.
* **The controls fail as expected:**
  * free mode leaves 1,282 interface quads without a partner;
  * local face frames give the right topology but node positions that differ by up to the optimiser
    tolerance, so 650 quads fail to match. The canonical frame is what makes positions agree exactly.

Figures: `figs/11_grid.png`, `figs/12_ball.png`, `figs/13_free_mode_control.png`.

### 8.2 Deterministic faces (`examples/ex13_exhaustive_checks.py`, `ex08_deterministic_check.py`)

| check | cases | failures |
|---|---|---|
| face trace = template, every excess pattern with levels 0..2 on the x- and y-edges, B = (5,6,7) | 4,225 blocks, 25,350 faces | 0 |
| the same with levels 0..3 | 30,625 blocks, 183,750 faces | 0 |
| levels 0..2 with the crossing count at its minimum, B = (5,5,3) | 4,225 blocks | 0 |
| the same with crossing count 2, below the minimum | 4,225 blocks | 4,200 (why the size check exists) |
| template symmetry: all face tuples with sides 5..11 under the 7 non-trivial symmetries | 4,375 | 0 (positions differ by up to 7.2e-5) |
| random blocks, counts 6..13 with one parity per direction, faces vs stand-alone templates | 60 blocks, 360 faces | 0; worst min SJ 0.24 |

Sheets of the third direction never touch the faces normal to it, and the code treats the three
directions alike. Varying two directions therefore covers every in-plane combination a face can see.

### 8.3 Speed

A deterministic block of a few hundred hexes takes about 0.1 s; one with many transitions takes
0.5-1 s, mostly in the optimiser. The 27-block cases (about 17,000 hexes) take 13-14 s serially. A
topology-only build (no smoothing) takes about 5 ms.

---

## 9. Usage

```bash
python3 -m venv .venv && .venv/bin/pip install numpy scipy matplotlib
.venv/bin/python examples/ex11_multiblock_tests.py      # multi-block tests -> figs/11_multiblock_tests.md
.venv/bin/python examples/ex12_multiblock_figs.py       # figs/11_grid.png, 12_ball.png, 13_free_mode_control.png
.venv/bin/python examples/ex13_exhaustive_checks.py 2   # exhaustive face checks
```

Multi-block:

```python
import numpy as np
from hexbox.multiblock import MultiBlock, box_grid
mb = box_grid((3, 2, 2), (3.0, 2.0, 2.0))          # or MultiBlock(V, blocks, curves)
mb.assign_counts(lambda x: 0.05 + 0.06 * np.linalg.norm(x))
# or set your own: mb.counts[frozenset((ga, gb))] = n, one parity per chain (mb.chains())
parts = mb.mesh_blocks(workers=4)                   # every block independently (in a script: under
                                                    # if __name__ == "__main__", for the process pool)
X, H = mb.assemble(parts, by="keys")                # global nodes (N,3), hexes (M,8) VTK order
print(mb.verify())                                  # conformal, unmatched_quads, min_sj, ...
ref = parts[0]["ref"]                               # block 0 in reference space: ref.X, ref.H
```

A single block in deterministic mode, in its own local face frames:

```python
from hexbox.deterministic import build_block_det, face_template
# edge order: x-edges (y,z)=00,01,10,11 | y-edges (x,z)=00,01,10,11 | z-edges (x,y)=00,01,10,11
m = build_block_det([11, 7, 7, 7,  9, 9, 9, 5,  6, 6, 8, 8])
UV, Q = face_template(11, 9, 7, 9)                  # the mesh of any face with these counts
```

A single block in free mode:

```python
from hexbox.build import build_block
m = build_block([9, 6, 6, 6,  6, 6, 9, 6,  6, 6, 9, 6])   # corner (1,0,0) refined by +3
```

---

## 10. Limitations and next steps

* **Odd jumps.** Deterministic mode allows only even differences between parallel edges, i.e. one
  parity per chain. The other 36 parity patterns admitted by the local rule of 5.3 (isolated corner and
  edge-wrap transitions) would need odd-face templates with turning chords, which are not implemented.
  The global route of 5.1 is the alternative if full face-even freedom is needed across blocks.
* **Fixed transition placement.** L and D are global constants (2, 2) shared by all blocks. U-turns
  always sit at the collar distance from the corners, centred on the edge. A face cannot shift its
  refinement toward one end.
* **Quality is optimised in reference space.** After TFI, strongly non-cubic blocks lose quality. In a
  ball with a smaller centre cube (half-size 0.3), a shell block has min/mean SJ 0.58/0.84 when
  structured but 0.29/0.63 with transitions (6 -> 12). Optimising the interior nodes in physical
  space after TFI, with faces frozen, would recover much of this without affecting conformity or
  parallelism.
* **Interior topology depends on local orientation.** Same-level sheets of different directions are
  pillowed in axis order, so a rotated block can get a slightly different interior (4,618 vs 4,620
  hexes in one scrambled ball). Faces are unaffected.
* **Faces come from edge-only TFI** (Coons patches), so curved faces such as the sphere are only
  approximated. A per-face projection would be deterministic and keep conformity.
* **Floating point across machines.** Bit-identical shared nodes rely on the template optimiser giving
  identical results everywhere. Numbering by keys does not depend on positions. Computing the templates
  once and distributing them (they are a table indexed by 4 integers) removes even that dependence.
* **Grading.** Block-edge nodes are uniform in reference space; grading belongs in the edge curves'
  parametrisation, which neighbouring blocks share.
* **Free mode needs room.** With counts in 2..5 most random blocks fail with "could not fit". The fix
  would be smaller boxes or sheet types sized for 1-2 belts.

## 11. File map

| file | contents |
|---|---|
| `hexbox/mesh.py` | `HexMesh`, structured grids, face adjacency, `pillow` |
| `hexbox/sheets.py` | edge/corner indexing, the sheet catalogue, face-even check, MILP decomposition |
| `hexbox/build.py` | free-mode block builder with shrink boxes and retries; `edge_counts`, `check_conformal` |
| `hexbox/smooth.py` | edge respacing, constrained Laplacian, 3D and 2D untangle-and-smooth optimisers, scaled Jacobian |
| `hexbox/deterministic.py` | equal-parity rule, face templates, skyline U-sheets, face stamping |
| `hexbox/faceiso.py` | boundary-anchored isomorphism of two quad meshes (`same_topology`, `match_nodes`) |
| `hexbox/canonical.py` | analysis of local face rules for odd faces (rule CR: 44/128 parity patterns) |
| `hexbox/tfi.py` | edge-based TFI; `Line`, `Arc`, `Reversed` curves |
| `hexbox/multiblock.py` | chains, count assignment, canonical frames, per-block jobs, node keys, assembly, verification, test domains |
| `hexbox/viz.py` | cube-net, 3D, cut-away, exploded and shared-face plots |
| `examples/ex01`-`ex07` | free mode: corner sheet, gallery, random blocks, optimiser, stress test, cut-aways, multi-block issue |
| `examples/ex08`-`ex10` | deterministic faces: random check, two independent blocks, template gallery |
| `examples/ex11`-`ex13` | multi-block tests, multi-block figures, exhaustive checks |
| `figs/` | figures and logs: `stress_*.txt`, `11_multiblock_tests.md`/`.log`, `13_exhaustive_checks*.txt` |
