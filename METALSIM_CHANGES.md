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
| `dd42c23` | Metal: the five per-iteration Newton launches between the line search and the Hessian update (zero the change counters, `_update_constraint_efc`, `qfrc_constraint = J^T force`, `_update_gradient_zero_grad_dot`, `_update_gradient_grad`) fused into one 32-lane kernel per world, `_update_constraint_gradient_fused` (dense Jacobian, pyramidal cones, incremental path): 9 -> 5 launches per iteration, G1 flat step +3.9 %; per-row / per-dof arithmetic and order unchanged, `grad_dot` a SIMD sum instead of atomics. `MJW_METAL_FUSE_UPDATE=0` restores the five launches. |
| `7245794`, `778cf9e`, `441f67c` | Metal: lane-parallel sparse L'DL kernels (`_factor_i_sparse_lanes` / `_solve_LD_sparse_lanes`, one world per SIMD group, lanes over the (row, element) pairs of a tree level; bitwise the serial kernels' arithmetic). Measured slower than the one-world-per-thread kernels of `ad22120` (factor 0.57 vs 0.45 ms per 4096 G1 worlds); off, `MJW_METAL_LDL_LANES=1` for A/B. |
| `cdc4fd4`, `e068734`, `a1daf88` | Metal: chain-parallel sparse L'DL (`_factor_i_sparse_chains` / `_solve_LD_sparse_chains`, one lane per single-child chain of the dof tree per level; the solve keeps the serial order bitwise, the factor's cross-chain updates are atomic as upstream's CUDA `_qLD_acc`). Measured slower (factor 0.65 vs 0.45 ms); off, `MJW_METAL_LDL_CHAINS=1` for A/B. Schedule tables `qLD_updates_byrow`, `qLD_lane_*`, `qLD_chain_*`, `qLD_updates_bysrc`, `qLD_src_adr`, `qLD_row_adr` are built in `put_model` regardless. |
| `4ef3da3`, `2d37...`, `a1daf88` (`MJW_ELLIPTIC_CONE_UPDATE`) | Elliptic cone term as rank-1 updates of the stored factor of h (MuJoCo C's `HessianConeUpdate`: `_cone_vectors`, `_cone_update_prepare`, `_update_gradient_cholesky_cone_update`, Warp's `tile_cholesky_update_inplace`; worlds above `MJW_CONE_UPDATE_KMAX` cone rows keep the per-entry path; `MJW_CONE_UPDATE_REUSE`). **Archived off**: physics within the run-to-run floors but measured 5-24 % slower on every model (G1 task ellip10 -16 %, Go2 -5 %, humanoid -24 %, SO-101 -11 %): on a 32-lane register factorization the rank-1 recurrence has the factorization's latency and the factor pass through `htot` adds a launch. `MJW_ELLIPTIC_CONE_UPDATE=1` enables it. |
| `f4276b0`, `05ad3f0` | Metal: per-model unrolled register L'DL factorization of M (`_factor_i_sparse_unrolled`, `MJW_METAL_LDL_UNROLLED`, default on for models with <= 32 entries per row and <= 64 dofs): lane j of a 32-lane world holds position j of every row of the factor in registers, the model's update list (`put_model` builds `qLD_unrolled`) is emitted as straight-line native code with two SIMD shuffles per update, fp contraction off. Bitwise the serial kernel of `ad22120` (factor and D; G1 task, Go2, humanoid); factor 0.453 -> 0.314 ms per 4096 G1 worlds, physics-only step +3.6 %. The unrolled solve (`_solve_LD_sparse_unrolled`) measured 0.79 vs 0.21 ms and stays off (`MJW_METAL_LDL_UNROLLED_SOLVE=1`). `MJW_METAL_CHOL_LANES` selects the Newton Cholesky launches' lanes per world (32; 64 for the Warp fork's `metal_register_cholesky64`, measured no faster at n = 43). |

Every flex change and every solver launch-form change is behind a module-level flag or environment variable that restores upstream's behaviour; the details and measurements are in
the commit messages and in MetalSim's `scripts/diagnostics/deformable/UPSTREAM.md`. The same sixteen code commits are
exported as MetalSim's `patches/mujoco_warp/0001-0016` (flex commits rebased after the throughput ones; the result
is the tree of `8fbf965`).

Only on `metalsim-flex`, not merged here: implicit flex damping (`c013067`) and `FLEX_MAXCONPAIR` (`64a1ea5`).
