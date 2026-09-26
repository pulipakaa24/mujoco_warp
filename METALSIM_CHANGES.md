# MetalSim changes on this branch (`metalsim-flex`)

Deformable (flex) branch of the MuJoCo Warp fork used by [MetalSim](https://github.com/pulipakaa24/MetalSim)
(`scripts/setup_flex.sh` installs it; see "What was built where" in the
[README](https://github.com/pulipakaa24/MetalSim/blob/main/README.md)). Its first thirteen commits are merged into
`metalsim` (merge `8fbf965`); the last two are only here. This branch does not carry the Metal throughput commits
of `metalsim` (`ad22120`, `ccfaffb`, `1791414`).

Lineage: [google-deepmind/mujoco_warp](https://github.com/google-deepmind/mujoco_warp) v3.14.0 (`88af9cc`) →
branch `metal` (`8ce5bb0`) → this branch.

Licence: Apache License 2.0, as upstream. `LICENSE` is unchanged from google-deepmind/mujoco_warp. The changes below
modify Apache-2.0 code and are offered under the same licence.

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
| `c013067` | Flex: implicit (backward-Euler) elasticity damping, opt-in (`flex_damping.ENABLE`); MuJoCo's explicit damping diverges at PhysX's default damping. |
| `64a1ea5` | Flex: configurable per-pair contact cap for the serial selection (`FLEX_MAXCONPAIR`, default `mjMAXCONPAIR` = 50, as MuJoCo C). A 33 x 33 cloth on a box falls through with 50 contacts per pair in MuJoCo C too. |
