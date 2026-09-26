# MetalSim changes on this branch (`metalsim`)

This branch is the MuJoCo Warp build that [MetalSim](https://github.com/pulipakaa24/MetalSim) runs on (see the
section "What was built where" in its [README](https://github.com/pulipakaa24/MetalSim/blob/main/README.md)).

Lineage: [google-deepmind/mujoco_warp](https://github.com/google-deepmind/mujoco_warp) v3.14.0 (`88af9cc`) →
branch `metal` (`8ce5bb0`, the Metal device patch) → this branch.

Licence: Apache License 2.0, as upstream. `LICENSE` is unchanged from google-deepmind/mujoco_warp (same blob).
The changes below modify Apache-2.0 code and are offered under the same licence.

## Commits (on top of v3.14.0 `88af9cc`)

| commit | change |
|---|---|
| `8ce5bb0` | Metal device support: David Dobas's patch ([DavidDobas/mujoco_warp#1](https://github.com/DavidDobas/mujoco_warp/pull/1), Apache-2.0), applied unchanged. Also branch `metal`. |
| `f2716b4` | Heightfield collision: contacts against each prism's top plane per triangle. Upstream's single-witness GJK/EPA against the prism column gave inverted normals and 2 m penetrations for thin convex meshes (G1 feet on rough terrain). Switch: `HFIELD_PLANE_CONTACTS` in `collision_convex.py`. |
| `284dcd1` | `plane_convex`: MuJoCo C's `mjc_PlaneConvex` contact set (support vertex plus the max-area quad of the most anti-aligned adjacent face), margin-aware. Upstream kept 2 contacts where MuJoCo C keeps 4 on a tilted foot. |
| `a8e6485` | `MJW_PLANE_CONVEX=legacy` restores upstream's heuristic for A/B comparisons. |
| `b2e9ea5` | Flex contacts as MuJoCo C: per-(body, flex) `MJ_MAXCONPAIR` cap with C's selection order; all box-triangle contacts. |
| `a277152` | Flex element contacts as MuJoCo C: flex-local element ids for every flex after the first, capsule elements for cables, radius-inflated contact points, float32 ties. |
| `bbe19bb` | Flex contact selection: tighter float32 tie tolerances. |
| `6234396` | Flex volumes: only active-layer elements collide with geoms, as MuJoCo C. |
| `fa2ff1d` | Mesh-flex contact normal: MuJoCo C's EPA direction at edges and corners. |
| `5da74f1` | Flex workspace: no block-parallel FPS scratch in the serial selection modes (memory). |
| `9c6e09f` | Flex edge equality rows in MuJoCo C's order on every device (fixes `FlexConstraintTest` on Metal). |
| `361f11f` | Flex filter and SAP sorts and scans on the device on Metal (`FLEX_DEVICE_SORT`). |
| `dfa5d30` | Mesh-flex normal: keep the face normal for touching contacts (witness separation below 0.1 mm). |
| `ad22120` | Metal: sparse L'DL factor and solve with one world per thread; gravity-compensation and tendon-damping launches skipped when the model has none. |
| `ccfaffb` | Metal: incremental Newton-Hessian update fused into the register Cholesky launch (`MJW_METAL_FUSE_H_CHOLESKY=0` restores two launches). |
| `1791414` | `MJW_METAL_DENSE_CHOL_MAX` knob for the off-CUDA dense vs blocked Hessian Cholesky threshold (default 64, unchanged). |
| `8fbf965` | Merge of `metalsim-flex` (`b2e9ea5..dfa5d30`); the tree MetalSim's results were produced with. |
| `c301880` | Metal: elliptic-cone Newton Hessian without capacity-sized launches or one-lane groups. Dense Jacobian: `MJW_JTCJ_MODE` = `world2` (default; per-world cone list, then one thread per (world, Hessian entry), no atomics), `world`, `contact` (upstream's CUDA form sized from `Device.sm_count`, `MJW_JTCJ_SM_FACTOR`), `capacity` (upstream's off-CUDA fallback, `dim_block = naconmax`). Sparse Jacobian: `MJW_JTDAJ_ELLIPTIC_LANES` 32 (default; per-lane cone terms) / 1 (previous one-lane groups), `MJW_JTDAJ_GROUPS_PER_WORLD`. `MJW_CCD_GRID_WAVES` (off-CUDA CCD grid, default 0 = previous). Go2 elliptic 240 K → 580 K steps/s at 4096 worlds; MetalSim G1 task elliptic 11.0 K → 55.4 K env-steps/s. Measurements and the physics checks: MetalSim `docs/research/elliptic_cones_2026-09-25.md`. |
| `b630530` | Heightfield plane contacts for meshes only (`HFIELD_PLANE_CONTACTS_PRIMITIVES`, default False). The per-triangle path of `f2716b4` tests every vertex of a mesh of at most 256 vertices against the triangle's column, but for primitives it only has the single support point along the triangle normal, which misses the contact when that point's footprint lies outside the triangle (upstream's `test_hfield_maxconpair`: a 2 m box on a 0.2 m heightfield, 0 contacts instead of 4). Primitives now take upstream's GJK/EPA against the prism; `True` restores the previous form. Fork `-k hfield` 3 passed on CPU and Metal; MetalSim's G1-on-heightfield tests unchanged (mesh feet). |
| `a33c731`, `b895bff`, `b0150ac` | Elliptic cones on the incremental Newton path (`MJW_ELLIPTIC_INCREMENTAL`, dense Jacobian, fused register Cholesky): MuJoCo C's structure (`engine_solver.c` `HessianIncremental` + `HessianCone`), `ctx.h` keeps M + J'DJ over QUADRATIC rows updated by the flipped rows, the cone contacts' term is rebuilt every iteration; mode 2 (default) assembles deltas + cone term into `ctx.htot` per (world, entry) and factorizes with the plain register Cholesky, mode 1 adds the cone buffer inside the fused launch (slower, kept), 0 = the previous full rebuild. CONE-state rows count as a state change (no stable-state fast path for their worlds). `MJW_CONE_SKIP_ZERO` (off) and `MJW_FUSE_H_CHOLESKY_CPU` (tests). G1 task elliptic 55 K → 65 K env-steps/s (1.21× pyramidal), Go2 / humanoid / SO-101 +4–7 %. Measurements: MetalSim `docs/research/elliptic_cones_2026-09-25.md` §8. |

Every flex change and every solver launch-form change is behind a module-level flag or environment variable that restores upstream's behaviour; the details and measurements are in
the commit messages and in MetalSim's `scripts/diagnostics/deformable/UPSTREAM.md`. The same sixteen code commits are
exported as MetalSim's `patches/mujoco_warp/0001-0016` (flex commits rebased after the throughput ones; the result
is the tree of `8fbf965`).

Only on `metalsim-flex`, not merged here: implicit flex damping (`c013067`) and `FLEX_MAXCONPAIR` (`64a1ea5`).
