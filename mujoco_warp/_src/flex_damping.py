# Copyright 2026 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implicit (backward-Euler) flex elasticity damping (MetalSim).

MuJoCo's flex elasticity damping is stiffness-proportional (Rayleigh) damping: the element strain uses
elongation = d^2 - L0^2 + (d^2 - d_prev^2) * damping / dt, i.e. the damping force is f = -D v with
D v = damping * sum_{e1,e2} metric[e1,e2] * (2 g_e1 . v_e1) * dg_e2/dx (g_e = x_a - x_b, metric the element's
StVK edge metric). It is integrated explicitly, which bounds damping * dt * omega_max^2 < 2: at dt = 0.5 ms a
0.5 m rod with E = 1e5 diverges above damping ~1e-4 (PhysX's default elasticity damping is 5e-3).

With ENABLE (and Euler integration) the explicit damping term is dropped from the passive force and, after
forward(), the flex vertex velocities are advanced with the damping taken implicitly:
    (M + dt D) v' = M (qvel + dt qacc)
solved per world by Jacobi-preconditioned conjugate gradients (fixed ITERATIONS, graph-capturable; D is
symmetric positive semidefinite, so M + dt D is SPD). qacc (and efc.Ma) of the flex vertex DOFs are replaced by
(v' - qvel) / dt; euler() then integrates as usual. Unconditionally stable in the damping; for damping*dt*omega^2
<< 1 it agrees with the explicit form to O(dt).
Scope: non-interpolated flexes (dof="full") whose vertices are bodies with three orthogonal slide joints.
"""

import numpy as np
import warp as wp

from mujoco_warp._src.types import Data
from mujoco_warp._src.types import Model

ENABLE: bool = False
ITERATIONS: int = 30

_EDGES = np.array([[0, 1], [1, 2], [2, 0], [2, 3], [0, 3], [1, 3]], np.int32)   # dim 3; dim 2 uses rows 0-2 as (1,2),(2,0),(0,1)


@wp.kernel
def _gather(
  vert_dof: wp.array[int],
  vert_axes: wp.array[wp.mat33],
  vert_mass: wp.array[float],
  qvel: wp.array2d[float],
  qacc: wp.array2d[float],
  dt: float,
  # out
  v0: wp.array2d[wp.vec3],
  x: wp.array2d[wp.vec3],
  b: wp.array2d[wp.vec3],
):
  w, i = wp.tid()
  adr = vert_dof[i]
  if adr < 0:
    v0[w, i] = wp.vec3(0.0)
    x[w, i] = wp.vec3(0.0)
    b[w, i] = wp.vec3(0.0)
    return
  A = vert_axes[i]
  vq = wp.vec3(qvel[w, adr], qvel[w, adr + 1], qvel[w, adr + 2])
  aq = wp.vec3(qacc[w, adr], qacc[w, adr + 1], qacc[w, adr + 2])
  vc = A * vq
  vp = A * (vq + aq * dt)
  v0[w, i] = vc
  x[w, i] = vp
  b[w, i] = vp * vert_mass[i]


@wp.kernel
def _apply_D(
  # element data (flattened over the damped flexes)
  elem_verts: wp.array[wp.vec4i],
  elem_dim: wp.array[int],
  elem_metric_adr: wp.array[int],
  elem_beta: wp.array[float],
  flex_stiffness: wp.array[float],
  edges: wp.array2d[int],
  vert_dof: wp.array[int],
  xpos: wp.array2d[wp.vec3],
  vert_global: wp.array[int],
  p: wp.array2d[wp.vec3],
  scale: float,
  # out (accumulated)
  y: wp.array2d[wp.vec3],
):
  w, e = wp.tid()
  dim = elem_dim[e]
  nvert = dim + 1
  nedge = nvert * (nvert - 1) / 2
  ev = elem_verts[e]
  beta = elem_beta[e] * scale
  # per edge: g_e and s_e = 2 g_e . (p_a - p_b)
  g = wp.matrix(0.0, shape=(6, 3))
  s = wp.vector(0.0, length=6)
  for k in range(nedge):
    a = ev[edges[dim, 2 * k]]
    c = ev[edges[dim, 2 * k + 1]]
    xa = xpos[w, vert_global[a]]
    xc = xpos[w, vert_global[c]]
    gv = xa - xc
    for t in range(3):
      g[k, t] = gv[t]
    pa = p[w, a]
    pc = p[w, c]
    if vert_dof[a] < 0:
      pa = wp.vec3(0.0)
    if vert_dof[c] < 0:
      pc = wp.vec3(0.0)
    s[k] = 2.0 * wp.dot(gv, pa - pc)
  adr = elem_metric_adr[e]
  for k2 in range(nedge):
    acc = float(0.0)
    for k1 in range(nedge):
      lo = wp.min(k1, k2)
      hi = wp.max(k1, k2)
      # packed upper triangle, row-major (as flex_stiffness: for ed1, for ed2 >= ed1)
      idx = lo * nedge - lo * (lo - 1) / 2 + (hi - lo)
      acc += s[k1] * flex_stiffness[adr + idx]
    a = ev[edges[dim, 2 * k2]]
    c = ev[edges[dim, 2 * k2 + 1]]
    gv = wp.vec3(g[k2, 0], g[k2, 1], g[k2, 2])
    f = gv * (beta * acc)
    if vert_dof[a] >= 0:
      wp.atomic_add(y, w, a, f)
    if vert_dof[c] >= 0:
      wp.atomic_sub(y, w, c, f)


@wp.kernel
def _add_mass(vert_dof: wp.array[int], vert_mass: wp.array[float], p: wp.array2d[wp.vec3], y: wp.array2d[wp.vec3]):
  w, i = wp.tid()
  if vert_dof[i] < 0:
    y[w, i] = wp.vec3(0.0)
    return
  y[w, i] = y[w, i] + p[w, i] * vert_mass[i]


@wp.kernel
def _residual(b: wp.array2d[wp.vec3], ax: wp.array2d[wp.vec3], vert_mass: wp.array[float], vert_dof: wp.array[int],
              r: wp.array2d[wp.vec3], z: wp.array2d[wp.vec3], p: wp.array2d[wp.vec3], rz: wp.array[float]):
  w, i = wp.tid()
  if vert_dof[i] < 0:
    r[w, i] = wp.vec3(0.0); z[w, i] = wp.vec3(0.0); p[w, i] = wp.vec3(0.0)
    return
  rr = b[w, i] - ax[w, i]
  zz = rr / vert_mass[i]
  r[w, i] = rr
  z[w, i] = zz
  p[w, i] = zz
  wp.atomic_add(rz, w, wp.dot(rr, zz))


@wp.kernel
def _dot(a: wp.array2d[wp.vec3], b: wp.array2d[wp.vec3], out: wp.array[float]):
  w, i = wp.tid()
  wp.atomic_add(out, w, wp.dot(a[w, i], b[w, i]))


@wp.kernel
def _update_xr(x: wp.array2d[wp.vec3], r: wp.array2d[wp.vec3], z: wp.array2d[wp.vec3], p: wp.array2d[wp.vec3],
               ap: wp.array2d[wp.vec3], rz: wp.array[float], pap: wp.array[float], vert_mass: wp.array[float],
               vert_dof: wp.array[int], rz_new: wp.array[float]):
  w, i = wp.tid()
  if vert_dof[i] < 0:
    return
  alpha = float(0.0)
  if pap[w] > 1.0e-30:
    alpha = rz[w] / pap[w]
  x[w, i] = x[w, i] + p[w, i] * alpha
  rr = r[w, i] - ap[w, i] * alpha
  zz = rr / vert_mass[i]
  r[w, i] = rr
  z[w, i] = zz
  wp.atomic_add(rz_new, w, wp.dot(rr, zz))


@wp.kernel
def _update_p(z: wp.array2d[wp.vec3], p: wp.array2d[wp.vec3], rz: wp.array[float], rz_new: wp.array[float],
              vert_dof: wp.array[int]):
  w, i = wp.tid()
  if vert_dof[i] < 0:
    return
  beta = float(0.0)
  if rz[w] > 1.0e-30:
    beta = rz_new[w] / rz[w]
  p[w, i] = z[w, i] + p[w, i] * beta


@wp.kernel
def _swap(rz: wp.array[float], rz_new: wp.array[float], pap: wp.array[float]):
  w = wp.tid()
  rz[w] = rz_new[w]
  rz_new[w] = 0.0
  pap[w] = 0.0


@wp.kernel
def _scatter(vert_dof: wp.array[int], vert_axes: wp.array[wp.mat33], vert_mass: wp.array[float], x: wp.array2d[wp.vec3],
             v0: wp.array2d[wp.vec3], dt: float, qacc: wp.array2d[float], ma: wp.array2d[float]):
  w, i = wp.tid()
  adr = vert_dof[i]
  if adr < 0:
    return
  a = wp.transpose(vert_axes[i]) * ((x[w, i] - v0[w, i]) / dt)
  for t in range(3):
    qacc[w, adr + t] = a[t]
    ma[w, adr + t] = a[t] * vert_mass[i]


class _Plan:
  pass


_PLANS = {}


def _plan(m: Model, mjm_arrays):
  """Host-side element / vertex tables (built once per Model from the MjModel copy kept on it)."""
  key = id(m)
  if key in _PLANS:
    return _PLANS[key]
  P = _Plan()
  (flex_dim, flex_interp, flex_damping, flex_vertadr, flex_vertnum, flex_elemadr, flex_elemnum, flex_elemdataadr,
   flex_elem, flex_stiffnessadr, flex_vertbodyid, body_mass, body_dofnum, body_dofadr, jnt_axis, body_jntadr,
   body_quat, dof_armature, dt0) = mjm_arrays
  nv_total = int(flex_vertnum.sum())
  vert_dof = -np.ones(nv_total, np.int32)
  vert_axes = np.tile(np.eye(3, dtype=np.float32), (nv_total, 1, 1))
  vert_mass = np.ones(nv_total, np.float32)
  vert_global = np.arange(nv_total, dtype=np.int32)
  ev, ed, ea, eb = [], [], [], []
  for f in range(len(flex_dim)):
    if flex_interp[f] != 0 or flex_damping[f] <= 0.0 or flex_stiffnessadr[f] < 0:
      continue
    dim = int(flex_dim[f])
    for v in range(flex_vertnum[f]):
      gi = flex_vertadr[f] + v
      bid = flex_vertbodyid[gi]
      if bid <= 0 or body_dofnum[bid] != 3:
        continue
      q = body_quat[bid]
      w_, x_, y_, z_ = q
      R = np.array([[1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - z_ * w_), 2 * (x_ * z_ + y_ * w_)],
                    [2 * (x_ * y_ + z_ * w_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - x_ * w_)],
                    [2 * (x_ * z_ - y_ * w_), 2 * (y_ * z_ + x_ * w_), 1 - 2 * (x_ * x_ + y_ * y_)]])
      ja = body_jntadr[bid]
      A = R @ np.stack([jnt_axis[ja], jnt_axis[ja + 1], jnt_axis[ja + 2]], 1)
      vert_dof[gi] = body_dofadr[bid]
      vert_axes[gi] = A
      vert_mass[gi] = body_mass[bid] + dof_armature[body_dofadr[bid]]
    for e in range(flex_elemnum[f]):
      base = flex_elemdataadr[f] + e * (dim + 1)
      vv = [int(flex_vertadr[f] + flex_elem[base + k]) for k in range(dim + 1)] + [0] * (3 - dim)
      ev.append(vv); ed.append(dim)
      nedge = (dim + 1) * dim // 2
      ea.append(int(flex_stiffnessadr[f] + e * 21))
      eb.append(float(flex_damping[f]))
  P.nelem = len(ev)
  P.nvert = nv_total
  P.dt = float(dt0)
  if P.nelem == 0:
    _PLANS[key] = P
    return P
  edges = np.zeros((4, 12), np.int32)
  edges[1, :2] = [0, 1]
  edges[2, :6] = [1, 2, 2, 0, 0, 1]
  edges[3, :12] = _EDGES.reshape(-1)
  P.elem_verts = wp.array(np.array(ev, np.int32), dtype=wp.vec4i)
  P.elem_dim = wp.array(np.array(ed, np.int32), dtype=int)
  P.elem_metric_adr = wp.array(np.array(ea, np.int32), dtype=int)
  P.elem_beta = wp.array(np.array(eb, np.float32), dtype=float)
  P.edges = wp.array(edges, dtype=int)
  P.vert_dof = wp.array(vert_dof, dtype=int)
  P.vert_axes = wp.array(vert_axes, dtype=wp.mat33)
  P.vert_mass = wp.array(vert_mass, dtype=float)
  P.vert_global = wp.array(vert_global, dtype=int)
  _PLANS[key] = P
  return P


def attach(m: Model, mjm) -> None:
  """Build the implicit-damping tables for Model m from its MjModel (call once after put_model)."""
  _PLANS.pop(id(m), None)
  _plan(m, (mjm.flex_dim, mjm.flex_interp, mjm.flex_damping, mjm.flex_vertadr, mjm.flex_vertnum, mjm.flex_elemadr,
            mjm.flex_elemnum, mjm.flex_elemdataadr, mjm.flex_elem, mjm.flex_stiffnessadr, mjm.flex_vertbodyid,
            mjm.body_mass, mjm.body_dofnum, mjm.body_dofadr, mjm.jnt_axis, mjm.body_jntadr, mjm.body_quat,
            mjm.dof_armature, mjm.opt.timestep))


def active(m: Model) -> bool:
  P = _PLANS.get(id(m))
  return bool(ENABLE and P is not None and P.nelem > 0)


def apply(m: Model, d: Data) -> None:
  """Replace the flex vertex DOFs' qacc (and efc.Ma) by the backward-Euler damped update (see module doc)."""
  P = _PLANS[id(m)]
  nw, nv = d.nworld, P.nvert
  dt = P.dt
  v0 = wp.empty((nw, nv), dtype=wp.vec3)
  x = wp.empty((nw, nv), dtype=wp.vec3)
  b = wp.empty((nw, nv), dtype=wp.vec3)
  r = wp.empty_like(x); z = wp.empty_like(x); p = wp.empty_like(x); ap = wp.zeros_like(x)
  rz = wp.zeros(nw, dtype=float); rz_new = wp.zeros(nw, dtype=float); pap = wp.zeros(nw, dtype=float)

  def A(src, out):
    out.zero_()
    wp.launch(_apply_D, dim=(nw, P.nelem),
              inputs=[P.elem_verts, P.elem_dim, P.elem_metric_adr, P.elem_beta, m.flex_stiffness, P.edges, P.vert_dof,
                      d.flexvert_xpos, P.vert_global, src, dt], outputs=[out])
    wp.launch(_add_mass, dim=(nw, nv), inputs=[P.vert_dof, P.vert_mass, src], outputs=[out])

  wp.launch(_gather, dim=(nw, nv), inputs=[P.vert_dof, P.vert_axes, P.vert_mass, d.qvel, d.qacc, dt], outputs=[v0, x, b])
  A(x, ap)
  wp.launch(_residual, dim=(nw, nv), inputs=[b, ap, P.vert_mass, P.vert_dof], outputs=[r, z, p, rz])
  for _ in range(ITERATIONS):
    A(p, ap)
    pap.zero_()
    wp.launch(_dot, dim=(nw, nv), inputs=[p, ap], outputs=[pap])
    rz_new.zero_()
    wp.launch(_update_xr, dim=(nw, nv), inputs=[x, r, z, p, ap, rz, pap, P.vert_mass, P.vert_dof], outputs=[rz_new])
    wp.launch(_update_p, dim=(nw, nv), inputs=[z, p, rz, rz_new, P.vert_dof])
    wp.launch(_swap, dim=nw, inputs=[rz, rz_new, pap])
  wp.launch(_scatter, dim=(nw, nv), inputs=[P.vert_dof, P.vert_axes, P.vert_mass, x, v0, dt], outputs=[d.qacc, d.efc.Ma])
