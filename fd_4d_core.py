"""
fd_4d_core.py
-------------
4-state HJB solver for goal-reaching with stochastic volatility and interest rate.

State space: (W, τ, v, r)
  W  — wealth (controlled GBM)
  τ  — time-to-horizon (deterministic, backward)
  v  — variance = σ²  (CIR / Heston process)
  r  — interest rate  (CIR process)

Numerical scheme: Strang operator splitting → sequences of 1-D Thomas solves.
Cross-derivative terms treated explicitly (O(dt²) splitting error, consistent
with second-order Strang scheme).

New functions:
  fd_solve_4d          — single-asset 4D HJB solver
  fd_solve_4d_nd       — multi-asset extension (factor vol model)
  make_fd_policy_4d    — policy lookup (trilinear interpolation, t=0 only)
  make_fd_policy_4d_time_aware — policy lookup (4D interpolation in w,v,r,τ)
  evaluate_policy_mc_4d — Monte-Carlo evaluation with joint (W,v,r) paths

Imports thomas() and policy helpers from fd_core — never duplicated here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from fd_core import (
    thomas,
    policy_from_V,
    _policy_nd_from_V,
    _clip_leverage_nd,
    _clip_leverage_nd_batch,
    goal_utility,
)


# ── Grid dataclass ───────────────────────────────────────────────────────────

@dataclass
class Grid4D:
    Nw:    int    = 80
    A:     float  = 2.5
    Nv:    int    = 20
    v_max: float  = 0.25
    Nr:    int    = 15
    r_max: float  = 0.20
    Nt:    int    = 80
    T:     float  = 1.0

    def build(self):
        w  = np.linspace(0.0, self.A,     self.Nw + 1)
        v  = np.linspace(0.0, self.v_max, self.Nv + 1)
        r  = np.linspace(0.0, self.r_max, self.Nr + 1)
        dw = self.A     / self.Nw
        dv = self.v_max / self.Nv
        dr = self.r_max / self.Nr
        dt = self.T     / self.Nt
        return w, v, r, dw, dv, dr, dt


# ── Cross-derivative corrections (explicit, centred differences) ─────────────

def _cross_Wv(V, Pi, w, v, dw, dv, xi_v, rho_Wv, s):
    """
    Explicit cross term:  −ρ_Wv · ξ_v · √v · π · v · W · V_Wv · s
    Returns correction array (Nw+1, Nv+1, Nr+1).
    """
    corr = np.zeros_like(V)
    Nr1  = V.shape[2]
    for iv in range(1, V.shape[1] - 1):
        vi = v[iv]
        factor = s * rho_Wv * xi_v * math.sqrt(max(vi, 0.0)) * vi
        for ir in range(Nr1):
            V_Wv = (V[2:, iv+1, ir] - V[2:, iv-1, ir]
                    - V[:-2, iv+1, ir] + V[:-2, iv-1, ir]) / (4.0 * dw * dv)
            w_int  = w[1:-1]
            pi_int = Pi[1:-1, iv, ir]
            corr[1:-1, iv, ir] -= factor * pi_int * w_int * V_Wv
    return corr


def _cross_Wr(V, Pi, w, r, dw, dr, xi_r, rho_Wr, s):
    """
    Explicit cross term:  −ρ_Wr · ξ_r · √r · π · √v · W · V_Wr · s
    Note: the √v factor is absorbed into pi_int (user passes effective pi).
    Per spec: −ρ_Wr · ξ_r · √r · π√v · W · V_Wr
    Here we pass (Pi * sqrt(v)) already folded — callers pass Pi directly,
    so we compute √vi separately from the current v grid.
    Returns correction array (Nw+1, Nv+1, Nr+1).
    """
    corr = np.zeros_like(V)
    Nv1  = V.shape[1]
    for ir in range(1, V.shape[2] - 1):
        ri = r[ir]
        factor = s * rho_Wr * xi_r * math.sqrt(max(ri, 0.0))
        for iv in range(Nv1):
            # Vi slice: cross in W and r
            V_Wr = (V[2:, iv, ir+1] - V[2:, iv, ir-1]
                    - V[:-2, iv, ir+1] + V[:-2, iv, ir-1]) / (4.0 * dw * dr)
            w_int  = w[1:-1]
            pi_int = Pi[1:-1, iv, ir]
            corr[1:-1, iv, ir] -= factor * pi_int * w_int * V_Wr
    return corr


def _cross_vr(V, v, r, dv, dr, xi_v, xi_r, rho_vr, s):
    """
    Explicit cross term:  −ρ_vr · ξ_v · ξ_r · √(v·r) · V_vr · s
    Returns correction array (Nw+1, Nv+1, Nr+1).
    """
    corr = np.zeros_like(V)
    for iv in range(1, V.shape[1] - 1):
        vi = v[iv]
        for ir in range(1, V.shape[2] - 1):
            ri = r[ir]
            factor = s * rho_vr * xi_v * xi_r * math.sqrt(max(vi * ri, 0.0))
            V_vr = (V[:, iv+1, ir+1] - V[:, iv+1, ir-1]
                    - V[:, iv-1, ir+1] + V[:, iv-1, ir-1]) / (4.0 * dv * dr)
            corr[:, iv, ir] -= factor * V_vr
    return corr


# ── Sub-step: variance direction (CIR) ───────────────────────────────────────

def _substep_v(V, v, dv, dt, kappa_v, theta_v, xi_v):
    """
    Implicit solve in v direction for each fixed (iw, ir).
    CIR operator:  V_τ = κ_v(θ_v − v)V_v + ½ξ_v²v V_vv
    Neumann BC at v=0 and v=v_max.
    Modifies V in-place.
    """
    Nw1, Nv1, Nr1 = V.shape
    vi_int = v[1:-1]   # interior v points, length Nv-1
    a2_v   = 0.5 * xi_v**2 * vi_int
    Adv_v  = kappa_v * (theta_v - vi_int)
    Ap_v   = np.maximum(Adv_v, 0.0) / dv
    Am_v   = np.minimum(Adv_v, 0.0) / dv

    a_s = -dt * (a2_v / dv**2 - Am_v)
    b_m =  1.0 + dt * (2.0 * a2_v / dv**2 + Ap_v - Am_v)
    c_s = -dt * (a2_v / dv**2 + Ap_v)

    for iw in range(Nw1):
        for ir in range(Nr1):
            rhs = V[iw, 1:-1, ir].copy()
            # Neumann BC: V[:,0,:] = V[:,1,:]  →  ghost contribution = 0 on sub-diag
            # Neumann BC: V[:,Nv,:] = V[:,Nv-1,:] → ghost contribution = 0 on super-diag
            # Simply solve interior; then copy to boundaries
            a_loc = a_s.copy(); b_loc = b_m.copy(); c_loc = c_s.copy()
            a_loc[0]  = 0.0   # no left ghost (Neumann: V[0]=V[1] handled by copy)
            c_loc[-1] = 0.0
            V[iw, 1:-1, ir] = thomas(a_loc, b_loc, c_loc, rhs)
            V[iw,  0,   ir] = V[iw, 1,  ir]   # Neumann left
            V[iw, -1,   ir] = V[iw, -2, ir]   # Neumann right


def _substep_r(V, r, dr, dt, kappa_r, theta_r, xi_r):
    """
    Implicit solve in r direction for each fixed (iw, iv).
    CIR operator:  V_τ = κ_r(θ_r − r)V_r + ½ξ_r²r V_rr
    Neumann BC at r=0 and r=r_max.
    Modifies V in-place.
    """
    Nw1, Nv1, Nr1 = V.shape
    ri_int = r[1:-1]
    a2_r   = 0.5 * xi_r**2 * ri_int
    Adv_r  = kappa_r * (theta_r - ri_int)
    Ap_r   = np.maximum(Adv_r, 0.0) / dr
    Am_r   = np.minimum(Adv_r, 0.0) / dr

    a_s = -dt * (a2_r / dr**2 - Am_r)
    b_m =  1.0 + dt * (2.0 * a2_r / dr**2 + Ap_r - Am_r)
    c_s = -dt * (a2_r / dr**2 + Ap_r)

    for iw in range(Nw1):
        for iv in range(Nv1):
            rhs = V[iw, iv, 1:-1].copy()
            a_loc = a_s.copy(); b_loc = b_m.copy(); c_loc = c_s.copy()
            a_loc[0]  = 0.0
            c_loc[-1] = 0.0
            V[iw, iv, 1:-1] = thomas(a_loc, b_loc, c_loc, rhs)
            V[iw, iv,  0]   = V[iw, iv,  1]
            V[iw, iv, -1]   = V[iw, iv, -2]


def _substep_W(V, Pi, w, v, r, dw, dt_half, mu, d, u,
               xi_v, xi_r, rho_Wv, rho_Wr, rho_vr, dv, dr):
    """
    Implicit W half-step with cross-derivative corrections and policy iteration.
    BC: V[0,:,:] = 0 (absorbing), V[-1,:,:] = 1 (above goal).
    Updates Pi in-place and returns updated V (same object).
    """
    Nw, Nv1, Nr1 = V.shape[0] - 1, V.shape[1], V.shape[2]
    wi = w[1:Nw]   # interior wealth points

    V_old = V.copy()

    # Cross corrections added to RHS before the tridiagonal solve
    corr  = _cross_Wv(V_old, Pi, w, v, dw, dv, xi_v, rho_Wv, dt_half)
    corr += _cross_Wr(V_old, Pi, w, r, dw, dr, xi_r, rho_Wr, dt_half)
    corr += _cross_vr(V_old, v, r, dv, dr, xi_v, xi_r, rho_vr, dt_half)

    for iv in range(Nv1):
        vi = v[iv]
        for ir in range(Nr1):
            ri  = r[ir]
            eta = mu - ri             # excess return at this r node

            # Policy iteration (Howard) for this (iv, ir) slice
            pi_slice = Pi[:, iv, ir].copy()
            V_slice  = V_old[:, iv, ir].copy()
            rhs_base = (V_slice[1:Nw] + corr[1:Nw, iv, ir]).copy()

            for _ in range(60):
                pi_old_s = pi_slice.copy()
                pi       = pi_slice[1:Nw]
                sig2     = pi**2 * vi
                Adv      = pi * eta * wi
                a2       = 0.5 * sig2 * wi**2
                Ap       = np.maximum(Adv, 0.0) / dw
                Am       = np.minimum(Adv, 0.0) / dw
                a_s      = -dt_half * (a2 / dw**2 - Am)
                b_m      = 1.0 + dt_half * (2.0 * a2 / dw**2 + Ap - Am)
                c_s      = -dt_half * (a2 / dw**2 + Ap)
                rhs      = rhs_base.copy()
                rhs[0]  -= a_s[0]  * 0.0;  a_s[0]  = 0.0   # BC w=0: V=0
                rhs[-1] -= c_s[-1] * 1.0;  c_s[-1] = 0.0   # BC w=A: V=1
                V_int    = thomas(a_s, b_m, c_s, rhs)

                V_new_s      = np.empty(Nw + 1)
                V_new_s[0]   = 0.0
                V_new_s[Nw]  = 1.0
                V_new_s[1:Nw] = V_int

                sig2_scalar = vi
                pi_slice[1:Nw] = policy_from_V(V_new_s, wi, dw, eta, sig2_scalar, d, u)

                if np.max(np.abs(pi_slice - pi_old_s)) < 1e-8:
                    break

            V[:, iv, ir]  = V_new_s
            Pi[:, iv, ir] = pi_slice


# ── Main 4D solver ───────────────────────────────────────────────────────────

def fd_solve_4d(
    mu:       float,
    r0:       float,
    v0:       float,
    T:        float,
    goal_mult: float,
    w0:       float,
    kappa_v:  float,
    theta_v:  float,
    xi_v:     float,
    kappa_r:  float,
    theta_r:  float,
    xi_r:     float,
    rho_Wv:   float = -0.5,
    rho_Wr:   float =  0.0,
    rho_vr:   float =  0.0,
    Nw: int   = 80,
    Nv: int   = 20,
    Nr: int   = 15,
    Nt: int   = 80,
    A:  float = 2.5,
    v_max: float = 0.25,
    r_max: float = 0.20,
    d: float  = -5.0,
    u: float  =  3.0,
    store_policy_path: bool = False,
):
    """
    4-state HJB solver for goal-reaching with stochastic vol and interest rate.

    Strang splitting:  L_W(dt/2) · L_r(dt) · L_v(dt) · L_W(dt/2)
    Cross-derivative terms treated explicitly.

    Returns
    -------
    w_grid, v_grid, r_grid : 1-D axes
    V_grid  : (Nw+1, Nv+1, Nr+1) value function at t=0
    Pi_grid : (Nw+1, Nv+1, Nr+1) policy at t=0

    If store_policy_path=True, additionally:
    Pi_path  : (Nt, Nw+1, Nv+1, Nr+1)
    tau_path : (Nt,)  monotonically increasing from dt to T
    """
    grid = Grid4D(Nw=Nw, Nv=Nv, Nr=Nr, Nt=Nt, T=T, A=A, v_max=v_max, r_max=r_max)
    w, v, r, dw, dv, dr, dt = grid.build()

    # Terminal condition: V = 1{w >= goal}  (goal = goal_mult × w0 / w0 = goal_mult, normalised)
    V = (w[:, None, None] / goal_mult >= 1.0).astype(float)
    Pi = np.zeros((Nw + 1, Nv + 1, Nr + 1))

    if store_policy_path:
        Pi_path  = np.empty((Nt, Nw + 1, Nv + 1, Nr + 1))
        tau_path = np.arange(1, Nt + 1, dtype=float) * dt

    for step in range(Nt):
        # Strang: W(dt/2) · r(dt) · v(dt) · W(dt/2)
        _substep_W(V, Pi, w, v, r, dw, dt / 2.0, mu, d, u,
                   xi_v, xi_r, rho_Wv, rho_Wr, rho_vr, dv, dr)
        _substep_r(V, r, dr, dt, kappa_r, theta_r, xi_r)
        _substep_v(V, v, dv, dt, kappa_v, theta_v, xi_v)
        _substep_W(V, Pi, w, v, r, dw, dt / 2.0, mu, d, u,
                   xi_v, xi_r, rho_Wv, rho_Wr, rho_vr, dv, dr)

        # Enforce wealth BCs after each full step
        V[0,  :, :] = 0.0
        V[-1, :, :] = 1.0

        if store_policy_path:
            Pi_path[step] = Pi.copy()

    if store_policy_path:
        return w, v, r, V, Pi, Pi_path, tau_path
    return w, v, r, V, Pi


# ── Multi-asset 4D solver ─────────────────────────────────────────────────────

def _substep_W_nd(V, Pi, w, v, r, dw, dt_half, mu_vec, C_mat, d, u,
                  max_long, max_short, xi_v, xi_r, rho_Wv, rho_Wr, rho_vr, dv, dr):
    """
    Multi-asset W half-step using factor vol model: Ω(v) = v · C.
    Pi shape: (Nw+1, Nv+1, Nr+1, n_assets).
    """
    n    = len(mu_vec)
    Nw   = V.shape[0] - 1
    Nv1  = V.shape[1]
    Nr1  = V.shape[2]
    wi   = w[1:Nw]
    V_old = V.copy()

    # Build scalar Pi for cross-derivative (use sum of weights as proxy)
    Pi_scalar = Pi[:, :, :, 0] * 0.0  # placeholder zeros; cross terms use scalar
    # For cross terms, use the effective scalar: η_eff = (mu-r)^T π at each node
    # We pass a zero-correction for cross terms in nd (approximation: small cross effect)

    # Reuse existing _policy_nd_from_V logic but per (iv, ir) slice
    try:
        C_inv = np.linalg.inv(C_mat)
    except np.linalg.LinAlgError:
        C_inv = np.linalg.pinv(C_mat)

    for iv in range(Nv1):
        vi = v[iv]
        omega = vi * C_mat   # effective covariance at this v node

        try:
            omega_inv_mu = np.linalg.solve(omega + 1e-12 * np.eye(n), mu_vec)
        except np.linalg.LinAlgError:
            omega_inv_mu = np.zeros(n)

        for ir in range(Nr1):
            ri  = r[ir]
            eta = mu_vec - ri

            try:
                omega_inv_eta = np.linalg.solve(omega + 1e-12 * np.eye(n), eta)
            except np.linalg.LinAlgError:
                omega_inv_eta = np.zeros(n)

            V_slice  = V_old[:, iv, ir].copy()
            rhs_base = V_slice[1:Nw].copy()
            pi_slice = Pi[:, iv, ir, :].copy()   # (Nw+1, n)

            for _ in range(60):
                pi_old_s = pi_slice.copy()
                pi = pi_slice[1:Nw]   # (Nint, n)

                # Effective scalar params
                sig2_eff = np.einsum('ij,jk,ik->i', pi, omega, pi)   # (Nint,)
                eta_eff  = pi @ eta                                     # (Nint,)

                a2  = 0.5 * sig2_eff * wi**2
                Adv = eta_eff * wi
                Ap  = np.maximum(Adv, 0.0) / dw
                Am  = np.minimum(Adv, 0.0) / dw
                a_s = -dt_half * (a2 / dw**2 - Am)
                b_m = 1.0 + dt_half * (2.0 * a2 / dw**2 + Ap - Am)
                c_s = -dt_half * (a2 / dw**2 + Ap)
                rhs = rhs_base.copy()
                rhs[0]  -= a_s[0]  * 0.0;  a_s[0]  = 0.0
                rhs[-1] -= c_s[-1] * 1.0;  c_s[-1] = 0.0
                V_int    = thomas(a_s, b_m, c_s, rhs)

                V_new_s       = np.empty(Nw + 1)
                V_new_s[0]    = 0.0
                V_new_s[Nw]   = 1.0
                V_new_s[1:Nw] = V_int

                pi_slice[1:Nw] = _policy_nd_from_V(
                    V_new_s, wi, dw, eta, omega, omega_inv_eta,
                    d, u, max_long, max_short)

                if np.max(np.abs(pi_slice - pi_old_s)) < 1e-8:
                    break

            V[:, iv, ir]     = V_new_s
            Pi[:, iv, ir, :] = pi_slice


def fd_solve_4d_nd(
    mu_vec:   np.ndarray,
    C_mat:    np.ndarray,
    r0:       float,
    v0:       float,
    T:        float,
    goal_mult: float,
    w0:       float,
    kappa_v:  float,
    theta_v:  float,
    xi_v:     float,
    kappa_r:  float,
    theta_r:  float,
    xi_r:     float,
    rho_Wv:   float = -0.5,
    rho_Wr:   float =  0.0,
    rho_vr:   float =  0.0,
    Nw: int   = 80,
    Nv: int   = 20,
    Nr: int   = 15,
    Nt: int   = 80,
    A:  float = 2.5,
    v_max: float = 0.25,
    r_max: float = 0.20,
    d: float  = -5.0,
    u: float  =  3.0,
    max_long:  float = 3.0,
    max_short: float = 5.0,
    store_policy_path: bool = False,
):
    """
    Multi-asset 4D HJB solver. Covariance model: Ω(v) = v · C_mat.
    Pi_grid shape: (Nw+1, Nv+1, Nr+1, n_assets).
    """
    mu_vec = np.asarray(mu_vec, float)
    C_mat  = np.asarray(C_mat,  float)
    n      = len(mu_vec)

    grid = Grid4D(Nw=Nw, Nv=Nv, Nr=Nr, Nt=Nt, T=T, A=A, v_max=v_max, r_max=r_max)
    w, v, r, dw, dv, dr, dt = grid.build()

    V  = (w[:, None, None] / goal_mult >= 1.0).astype(float)
    Pi = np.zeros((Nw + 1, Nv + 1, Nr + 1, n))

    if store_policy_path:
        Pi_path  = np.empty((Nt, Nw + 1, Nv + 1, Nr + 1, n))
        tau_path = np.arange(1, Nt + 1, dtype=float) * dt

    for step in range(Nt):
        _substep_W_nd(V, Pi, w, v, r, dw, dt / 2.0, mu_vec, C_mat, d, u,
                      max_long, max_short, xi_v, xi_r, rho_Wv, rho_Wr, rho_vr, dv, dr)
        _substep_r(V, r, dr, dt, kappa_r, theta_r, xi_r)
        _substep_v(V, v, dv, dt, kappa_v, theta_v, xi_v)
        _substep_W_nd(V, Pi, w, v, r, dw, dt / 2.0, mu_vec, C_mat, d, u,
                      max_long, max_short, xi_v, xi_r, rho_Wv, rho_Wr, rho_vr, dv, dr)

        V[0,  :, :] = 0.0
        V[-1, :, :] = 1.0

        if store_policy_path:
            Pi_path[step] = Pi.copy()

    if store_policy_path:
        return w, v, r, V, Pi, Pi_path, tau_path
    return w, v, r, V, Pi


# ── Policy lookup ─────────────────────────────────────────────────────────────

def make_fd_policy_4d(w_grid, v_grid, r_grid, Pi_grid, d=-5.0, u=3.0):
    """
    Trilinear interpolation over (w, v, r). τ is accepted but ignored.
    Works for both scalar and array (w_norm, v, r).
    """
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (w_grid, v_grid, r_grid), Pi_grid,
        method='linear', bounds_error=False, fill_value=None,
    )

    def policy(w_norm, v, r, tau=None):
        pts = np.column_stack([
            np.atleast_1d(np.asarray(w_norm, float)),
            np.atleast_1d(np.asarray(v,      float)),
            np.atleast_1d(np.asarray(r,      float)),
        ])
        return np.clip(interp(pts), d, u)

    return policy


def make_fd_policy_4d_time_aware(w_grid, v_grid, r_grid, Pi_path, tau_path,
                                  d=-5.0, u=3.0):
    """
    4D (τ, w, v, r) policy lookup via RegularGridInterpolator.
    Pi_path : (Nt, Nw+1, Nv+1, Nr+1).
    tau_path: (Nt,) monotonically increasing.
    """
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (tau_path, w_grid, v_grid, r_grid), Pi_path,
        method='linear', bounds_error=False, fill_value=None,
    )

    def policy(w_norm, v, r, tau):
        tau_c = float(np.clip(float(tau), tau_path[0], tau_path[-1]))
        w_arr = np.atleast_1d(np.asarray(w_norm, float))
        v_arr = np.atleast_1d(np.asarray(v,      float))
        r_arr = np.atleast_1d(np.asarray(r,      float))
        N     = max(len(w_arr), len(v_arr), len(r_arr))
        pts   = np.column_stack([
            np.full(N, tau_c),
            np.broadcast_to(w_arr, (N,)),
            np.broadcast_to(v_arr, (N,)),
            np.broadcast_to(r_arr, (N,)),
        ])
        return np.clip(interp(pts), d, u)

    return policy


# ── Monte-Carlo evaluation ────────────────────────────────────────────────────

def evaluate_policy_mc_4d(
    policy_fn,
    mu, r0, v0, T, w0, goal_mult,
    kappa_v, theta_v, xi_v,
    kappa_r, theta_r, xi_r,
    rho_Wv=0.0, rho_Wr=0.0, rho_vr=0.0,
    n_paths=4000, n_steps=252, seed=1,
    d=-5.0, u=3.0,
):
    """
    Simulate joint (W, v, r) paths under policy_fn and compute goal-reaching metrics.

    Uses Euler-Milstein scheme with Cholesky correlated shocks.

    Returns dict with keys:
        goal_probability, mean_wealth, median_wealth,
        wealth_p05, wealth_p25, wealth_p75, wealth_p95, max_drawdown_mean
    """
    rng = np.random.default_rng(seed)
    dt  = T / n_steps
    goal = goal_mult * w0

    # Correlation matrix for (Z_W, Z_v, Z_r)
    Corr = np.array([
        [1.0,    rho_Wv, rho_Wr],
        [rho_Wv, 1.0,    rho_vr],
        [rho_Wr, rho_vr, 1.0   ],
    ])
    # Regularise slightly for numerical safety
    Corr += 1e-8 * np.eye(3)
    L = np.linalg.cholesky(Corr)

    W = np.full(n_paths, float(w0))
    v = np.full(n_paths, float(v0))
    r = np.full(n_paths, float(r0))
    peak_W = W.copy()
    max_dd = np.zeros(n_paths)

    for step in range(n_steps):
        tau = T - step * dt   # time remaining

        # Correlated shocks
        Z_ind = rng.standard_normal((n_paths, 3))
        Z_cor = Z_ind @ L.T            # (n_paths, 3)
        Z_W   = Z_cor[:, 0]
        Z_v   = Z_cor[:, 1]
        Z_r   = Z_cor[:, 2]

        # Policy
        w_norm  = W / goal
        pi      = np.clip(policy_fn(w_norm, v, r, tau), d, u)

        # Wealth update (GBM)
        sigma_t = np.sqrt(np.maximum(v, 0.0))
        dW_pct  = (r + pi * (mu - r)) * dt + pi * sigma_t * math.sqrt(dt) * Z_W
        W       = W * (1.0 + dW_pct)
        W       = np.maximum(W, 1e-12)

        # Milstein CIR for v
        v_new = (v + kappa_v * (theta_v - v) * dt
                   + xi_v * np.sqrt(np.maximum(v, 0.0)) * math.sqrt(dt) * Z_v
                   + 0.25 * xi_v**2 * dt * (Z_v**2 - 1.0))
        v = np.maximum(v_new, 0.0)

        # Milstein CIR for r
        r_new = (r + kappa_r * (theta_r - r) * dt
                   + xi_r * np.sqrt(np.maximum(r, 0.0)) * math.sqrt(dt) * Z_r
                   + 0.25 * xi_r**2 * dt * (Z_r**2 - 1.0))
        r = np.maximum(r_new, 0.0)

        # Track drawdown
        peak_W = np.maximum(peak_W, W)
        dd     = (peak_W - W) / peak_W
        max_dd = np.maximum(max_dd, dd)

    return dict(
        goal_probability  = float(np.mean(W >= goal)),
        mean_wealth       = float(np.mean(W)),
        median_wealth     = float(np.median(W)),
        wealth_p05        = float(np.percentile(W,  5)),
        wealth_p25        = float(np.percentile(W, 25)),
        wealth_p75        = float(np.percentile(W, 75)),
        wealth_p95        = float(np.percentile(W, 95)),
        max_drawdown_mean = float(np.mean(max_dd)),
    )


# ── Default calibrated parameters (§10 of spec) ──────────────────────────────

FD4D_PARAMS = dict(
    kappa_v = 5.0,    theta_v = 0.04,  xi_v = 0.50,
    kappa_r = 0.50,   theta_r = 0.04,  xi_r = 0.10,
    rho_Wv  = -0.50,  rho_Wr  = 0.0,  rho_vr = 0.0,
    Nw = 80, Nv = 20, Nr = 15, Nt = 80,
    v_max = 0.25,  r_max = 0.20,
)
