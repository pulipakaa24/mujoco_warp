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

Every flex change is behind a module-level flag that restores upstream's behaviour; the details and measurements are in
the commit messages and in MetalSim's `scripts/diagnostics/deformable/UPSTREAM.md`. The same sixteen code commits are
exported as MetalSim's `patches/mujoco_warp/0001-0016` (flex commits rebased after the throughput ones; the result
is the tree of `8fbf965`).

Only on `metalsim-flex`, not merged here: implicit flex damping (`c013067`) and `FLEX_MAXCONPAIR` (`64a1ea5`).
