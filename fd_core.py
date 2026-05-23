"""
fd_core.py
----------
Finite-difference HJB solver and related analytical functions
for the Dai, Kou, Qian & Wan (2019) non-concave utility maximisation.

All functions are pure (no global state). Import and call directly:

    from fd_core import fd_solve, goal_utility, aspiration_utility
    from fd_core import pi_browne, asymp_goalreach, asymp_aspiration
"""

import math
import numpy as np

# ── Low-level numerics ──────────────────────────────────────────────────────

def normcdf(z):
    """Standard normal CDF via erf — works on scalars and arrays."""
    z = np.asarray(z, float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def thomas(a, b, c, rhs):
    """
    Thomas algorithm (tridiagonal matrix solver).
    a : sub-diagonal  (length n, a[0] unused)
    b : main diagonal (length n)
    c : super-diagonal(length n, c[-1] unused)
    rhs: right-hand side (length n)
    Returns x such that tridiag(a,b,c) @ x = rhs.
    """
    n = len(b)
    c2, d2, x = np.zeros(n), np.zeros(n), np.zeros(n)
    c2[0] = c[0] / b[0]
    d2[0] = rhs[0] / b[0]
    for k in range(1, n):
        den  = b[k] - a[k] * c2[k-1]
        c2[k] = c[k] / den if k < n-1 else 0.0
        d2[k] = (rhs[k] - a[k] * d2[k-1]) / den
    x[-1] = d2[-1]
    for k in range(n-2, -1, -1):
        x[k] = d2[k] - c2[k] * x[k+1]
    return x


def policy_from_V(V, wi, dw, eta, sig2, d, u):
    """
    Three-case optimal policy update from Dai et al.
      Vww < 0  -> interior maximiser  pi* = -w*eta*Vw / (w^2*sig2*Vww)
      Vww > 0  -> convex, corner solution: argmax(f_d, f_u)
      |Vww|<eps-> flat region (V≈1 above goal): clamp to 0 to avoid noise
    """
    Vww = (V[2:] - 2*V[1:-1] + V[:-2]) / dw**2
    Vw  = (V[2:] - V[:-2]) / (2*dw)
    f_d = 0.5*d**2 * wi**2 * sig2 * Vww + d * wi * eta * Vw
    f_u = 0.5*u**2 * wi**2 * sig2 * Vww + u * wi * eta * Vw
    safe = np.where(Vww < -1e-12, Vww, -1e-12)
    pi_int = np.clip(-wi * eta * Vw / (wi**2 * sig2 * safe), d, u)
    # When both Vww and Vw are ~0 (flat region, e.g. above goal where V=1),
    # the policy is ill-defined — clamp to 0 rather than returning a random corner.
    flat = (np.abs(Vww) < 1e-12) & (np.abs(Vw) < 1e-12)
    corner = np.where(f_u >= f_d, u, d)
    return np.where(Vww < -1e-12, pi_int, np.where(flat, 0.0, corner))


# ── Analytical approximations ───────────────────────────────────────────────

def browne_V(w, eta, sigma, tau, goal=1.0):
    """
    Browne (1995) analytical value function for goal-reaching.
    V(t,w) = Phi( (log(w/goal) + (eta - 0.5*sigma^2)*tau) / (sigma*sqrt(tau)) )
    """
    tau = max(tau, 1e-10)
    z = (np.log(np.maximum(w, 1e-10) / goal)
         + (eta - 0.5*sigma**2) * tau) / (sigma * np.sqrt(tau))
    return normcdf(z)


def pi_browne(w, tau, mu, r, sigma, goal=1.0, d=-5.0, u=3.0):
    """
    Browne (1995) optimal policy derived from the analytical V.
    pi = eta / (sigma^2 * (1 + log(w/goal) / (sigma^2 * tau)))
    Clipped to [d, u].
    """
    if w <= 0 or tau <= 0:
        return 0.0
    eta  = mu - r
    sig2 = sigma**2
    log_ratio = np.log(max(float(w), 1e-10) / goal)
    denom = sig2 * (1.0 + log_ratio / (sig2 * max(tau, 1e-10)))
    if abs(denom) < 1e-10:
        return float(np.clip(0.0, d, u))
    return float(np.clip(eta / denom, d, u))


def browne_V_nd(w, eta_vec, omega_mat, tau, goal=1.0):
    """
    Multi-asset Browne value function for goal-reaching (analytical).

    The n-asset HJB reduces to a scalar problem via the maximum Sharpe ratio:
        theta^2 = eta^T Omega^{-1} eta
    The effective 1-D substitute has sigma_eff = theta, eta_eff = theta^2
    (i.e., fully invested in the tangency portfolio direction).

    Parameters
    ----------
    w         : scalar or array — current normalised wealth (w / goal)
    eta_vec   : (n,) excess-return vector  mu - r
    omega_mat : (n,n) covariance matrix (annualised)
    tau       : time to horizon (years)
    goal      : target wealth level (default 1.0, meaning w is already normalised)
    """
    omega_inv_eta = np.linalg.solve(omega_mat, eta_vec)
    theta2 = max(float(np.dot(eta_vec, omega_inv_eta)), 1e-12)
    theta  = np.sqrt(theta2)
    # Effective 1-D: eta_eff = theta^2, sigma_eff = theta
    return browne_V(w, theta2, theta, tau, goal)


def pi_browne_nd(w, tau, eta_vec, r, omega_mat, goal=1.0,
                  d=-5.0, u=3.0, max_long=3.0, max_short=5.0):
    """
    Multi-asset Browne optimal policy vector.

    Direction : Omega^{-1} eta  (tangency portfolio direction, unnormalised)
    Magnitude : 1-D Browne scalar with effective parameters (sigma=theta, mu=r+theta^2)

    pi_nd = pi_1d * Omega^{-1} eta

    where pi_1d = theta^2 / (theta^2 * (1 + log(w/goal) / (theta^2 * tau)))
                = 1 / (1 + log(w/goal) / (theta^2 * tau))

    This satisfies:
        pi_nd^T eta   = pi_1d * theta^2   (same excess return as 1-D)
        pi_nd^T Omega pi_nd = pi_1d^2 * theta^2  (same variance)

    Parameters
    ----------
    w         : float — normalised wealth (w / goal)
    tau       : float — time to horizon
    eta_vec   : (n,) excess-return vector
    r         : float — risk-free rate
    omega_mat : (n,n) covariance matrix
    """
    if float(w) <= 0 or tau <= 0:
        return np.zeros(len(eta_vec))
    omega_inv_eta = np.linalg.solve(omega_mat, eta_vec)
    theta2 = max(float(np.dot(eta_vec, omega_inv_eta)), 1e-12)
    log_ratio = np.log(max(float(w), 1e-10) / goal)
    denom = 1.0 + log_ratio / (theta2 * max(tau, 1e-10))
    if abs(denom) < 1e-10:
        pi_1d = 0.0
    else:
        pi_1d = 1.0 / denom          # unconstrained scalar magnitude
    pi_nd = float(pi_1d) * omega_inv_eta
    return _clip_leverage_nd(pi_nd, d, u, max_long, max_short)


def asymp_goalreach(w, tau, sigma, d, u, goal=1.0):
    """
    Asymptotic value function near T for goal-reaching utility.
    Implements Condition (18) / Definition A.1, Dai et al. (2019):
      V(t,zeta) ~ 2*Phi( min{0, log(zeta/goal)} / (L*sigma*sqrt(tau)) )
    where L = max(u, -d).
    """
    L   = max(u, -d)
    tau = max(tau, 1e-12)
    z   = np.minimum(0.0, np.log(np.maximum(w, 1e-10) / goal)) / (L * sigma * np.sqrt(tau))
    return 2.0 * normcdf(z)


def asymp_aspiration(w, tau, sigma, d, u, p=0.5, c1=1.2, c2=0.0, R=1.0):
    """
    Asymptotic value function near T for aspiration utility.
    U(w) = w^p/p for w < R,  c1*w^p/p + c2 for w >= R.
    """
    L    = max(u, -d)
    tau  = max(tau, 1e-12)
    K_Rm = R**p / p
    K_R  = c1 * R**p / p + c2
    z    = np.minimum(0.0, np.log(np.maximum(w, 1e-10) / R)) / (L * sigma * np.sqrt(tau))
    return K_Rm + 2.0 * normcdf(z) * (K_R - K_Rm)


# ── Utility functions ───────────────────────────────────────────────────────

def goal_utility(w):
    """U(w) = 1{w >= 1}. Goal-reaching (discontinuous at w=1)."""
    return (np.asarray(w, float) >= 1.0).astype(float)


def aspiration_utility(w, p=0.5, c1=1.2, R=1.0):
    """U(w) = w^p/p for w < R,  c1*w^p/p for w >= R."""
    w = np.asarray(w, float)
    return np.where(w < R, w**p / p, c1 * w**p / p)


# ── Main FD solver ──────────────────────────────────────────────────────────

def fd_solve(mu, r, sigma, T, A, Nw, Nt, d, u,
             utility_fn, asymptotic_fn,
             UB=0.0, UA=None, tau_blend=0.08, store_policy_path=False):
    """
    Monotone implicit FD solver for the HJB equation with asymptotic
    viscosity warmstart (Dai et al. 2019, Appendix A).

    Scheme: backward Euler + policy iteration + Thomas algorithm.
    Warmstart: blended Browne / asymptotic correction near T.

    Parameters
    ----------
    mu, r, sigma : market parameters (annualised)
    T            : horizon (years)
    A            : upper wealth boundary
    Nw, Nt       : spatial / time grid points
    d, u         : portfolio weight constraints
    utility_fn   : callable(w_arr) -> float array, terminal utility
    asymptotic_fn: callable(w_arr, tau) -> float array, near-T approximation
    UB, UA       : boundary values at w=0 and w=A (auto-computed if UA=None)
    tau_blend    : blending time constant (asymp near T, Browne far from T)
    store_policy_path : bool — if True, also return the full policy path and
                        corresponding tau values so make_fd_policy_time_aware
                        can interpolate in (tau, w) rather than at t=0 only.

    Returns
    -------
    w_grid  : (Nw+1,) wealth grid
    V_grid  : (Nw+1,) value function at t=0
    Pi_grid : (Nw+1,) optimal policy at t=0  (always returned)

    Additional returns when store_policy_path=True:
    Pi_path  : (Nt, Nw+1)  policy at each backward step s; the s-th row
               corresponds to time-to-horizon tau_path[s] = (s+1)*dt.
               Row 0 is the near-terminal policy; row Nt-1 is the t=0 policy.
    tau_path : (Nt,) tau values for each row of Pi_path, monotonically
               increasing from dt to T.
    """
    eta  = mu - r
    sig2 = sigma**2
    w    = np.linspace(0.0, A, Nw + 1)
    dw   = A / Nw
    dt   = T / Nt
    wi   = w[1:Nw]

    if UA is None:
        UA = float(utility_fn(np.array([A])).flat[0])

    V = utility_fn(w).astype(float)

    if store_policy_path:
        Pi_path  = np.empty((Nt, Nw + 1))
        tau_path = np.arange(1, Nt + 1, dtype=float) * dt  # (s+1)*dt, s=0..Nt-1

    for step in range(Nt):
        tau   = (Nt - step) * dt
        V_old = V.copy()

        # blended warmstart: near T use asymptotic, far from T use Browne
        alpha = float(np.exp(-tau / tau_blend))
        V_ws  = (alpha * asymptotic_fn(w, tau)
                 + (1 - alpha) * browne_V(w, eta, sigma, tau))
        pi_n  = np.full(Nw + 1, d)
        pi_n[1:Nw] = policy_from_V(V_ws, wi, dw, eta, sig2, d, u)

        # policy iteration
        for _ in range(60):
            pi_old = pi_n.copy()
            pi     = pi_n[1:Nw]
            a2     = 0.5 * pi**2 * wi**2 * sig2
            Adv    = pi * wi * eta
            Ap     = np.maximum(Adv, 0.0) / dw
            Am     = np.minimum(Adv, 0.0) / dw
            a_s    = -dt * (a2 / dw**2 - Am)
            b_m    = 1.0 + dt * (2*a2/dw**2 + Ap - Am)
            c_s    = -dt * (a2 / dw**2 + Ap)
            rhs    = V_old[1:Nw].copy()
            rhs[0]  -= a_s[0]  * UB;  a_s[0]  = 0.0
            rhs[-1] -= c_s[-1] * UA;  c_s[-1] = 0.0
            V_int  = thomas(a_s, b_m, c_s, rhs)
            V_new  = np.empty(Nw + 1)
            V_new[0]  = UB
            V_new[Nw] = UA
            V_new[1:Nw] = V_int
            pi_n[1:Nw] = policy_from_V(V_new, wi, dw, eta, sig2, d, u)
            if np.max(np.abs(pi_n - pi_old)) < 1e-8:
                break
        V = V_new

        if store_policy_path:
            Pi_path[step] = pi_n.copy()

    if store_policy_path:
        return w, V, pi_n, Pi_path, tau_path
    return w, V, pi_n


def make_fd_policy(w_grid, Pi_grid, d=-5.0, u=3.0):
    """
    Return a callable policy_fn(w_norm, tau) from a precomputed FD grid.
    w_norm = current_wealth / goal  (normalised so goal = 1).
    Accepts both scalar and array w_norm for vectorised MC simulation.
    """
    def policy(w_norm, tau):
        result = np.clip(np.interp(w_norm, w_grid, Pi_grid), d, u)
        # Preserve scalar vs array: np.interp returns scalar for scalar input
        return result
    return policy


def make_fd_policy_time_aware(w_grid, Pi_path, tau_path, d=-5.0, u=3.0):
    """
    Return a time-aware callable policy_fn(w_norm, tau) that interpolates
    in both wealth and time-to-horizon dimensions.

    Pi_path  : (Nt, Nw+1) — policy at each backward step, from fd_solve
               with store_policy_path=True.  Row 0 = near-terminal policy
               (tau=dt); row Nt-1 = t=0 policy (tau=T).
    tau_path : (Nt,) — tau values corresponding to each row, monotonically
               increasing.  tau_path[s] = (s+1)*dt.

    At query time the policy is bilinearly interpolated: given tau, the two
    nearest tau_path slices are found and the w-interpolated policies from
    both slices are linearly blended.
    """
    def policy(w_norm, tau):
        tau_q   = float(np.clip(float(tau), tau_path[0], tau_path[-1]))
        idx_frac = float(np.interp(tau_q, tau_path, np.arange(len(tau_path))))
        idx_lo   = int(np.floor(idx_frac))
        idx_hi   = min(idx_lo + 1, len(tau_path) - 1)
        alpha    = idx_frac - idx_lo
        pi_lo = np.clip(np.interp(w_norm, w_grid, Pi_path[idx_lo]), d, u)
        pi_hi = np.clip(np.interp(w_norm, w_grid, Pi_path[idx_hi]), d, u)
        return pi_lo * (1.0 - alpha) + pi_hi * alpha
    return policy


# ── Multi-asset (n-D) FD solver ─────────────────────────────────────────────
#
# The HJB is still a 2-D PDE in (t, w) — portfolio wealth is always scalar.
# The n-D extension only changes the per-node policy optimisation:
#
#   V_ww < 0  →  π* = -(V_w / (w V_ww)) Ω^{-1}η   (interior maximiser, QP has
#                closed-form solution; apply leverage clip to project onto C)
#   V_ww > 0  →  max over corner candidates of the polytope C
#   flat       →  π* = 0
#
# The effective portfolio parameters σ²_eff = π*^T Ω π* and η_eff = η^T π*
# feed the SAME Thomas tridiagonal as the 1-D solver.
# Complexity: O(Nw × Nt × n²) — grows with n but stays feasible up to n=20.


def _clip_leverage_nd(pi, d, u, max_long, max_short):
    """Clip a single weight vector to box [d,u]^n then long/short leverage caps."""
    pi = np.clip(np.asarray(pi, float), d, u)
    lp = max(float(np.maximum(pi, 0).sum()), 1e-12)
    sp = max(float(np.maximum(-pi, 0).sum()), 1e-12)
    if lp > max_long:
        pi = np.where(pi > 0, pi * (max_long / lp), pi)
    if sp > max_short:
        pi = np.where(pi < 0, pi * (max_short / sp), pi)
    return pi


def _clip_leverage_nd_batch(Pi, d, u, max_long, max_short):
    """Clip a batch of weight vectors (N, n). Vectorised version of _clip_leverage_nd."""
    Pi = np.clip(Pi, d, u)
    lp = np.maximum(Pi, 0).sum(axis=1, keepdims=True).clip(min=1e-12)  # (N,1)
    sp = np.maximum(-Pi, 0).sum(axis=1, keepdims=True).clip(min=1e-12)  # (N,1)
    long_scale  = np.minimum(max_long  / lp, 1.0)
    short_scale = np.minimum(max_short / sp, 1.0)
    return np.where(Pi >= 0, Pi * long_scale, Pi * short_scale)


def _policy_nd_from_V(V, wi, dw, eta, omega_mat, omega_inv_eta, d, u,
                      max_long, max_short):
    """
    Multi-asset analog of policy_from_V.  Fully vectorised over interior nodes.

    Parameters
    ----------
    V             : (Nw+1,) current value function
    wi            : (Nint,) interior wealth grid points
    dw            : grid spacing
    eta           : (n,) excess returns μ − r
    omega_mat     : (n, n) covariance matrix
    omega_inv_eta : (n,) precomputed Ω^{-1} η
    d, u          : per-asset box bounds
    max_long, max_short : aggregate leverage caps

    Returns
    -------
    Pi : (Nint, n) optimal portfolio weights at each interior node
    """
    n    = len(eta)
    Nint = len(wi)

    Vww = (V[2:] - 2.0*V[1:-1] + V[:-2]) / dw**2   # (Nint,)
    Vw  = (V[2:] - V[:-2]) / (2.0*dw)               # (Nint,)

    flat    = (np.abs(Vww) < 1e-12) & (np.abs(Vw) < 1e-12)
    concave = Vww < -1e-12
    convex  = ~flat & ~concave

    Pi = np.zeros((Nint, n))

    # ── Concave region: π* = -(V_w / (w V_ww)) Ω^{-1}η, then leverage clip ──
    if np.any(concave):
        safe = np.where(concave, Vww, -1e-12)
        scale = np.where(concave, -Vw / (wi * safe), 0.0)         # (Nint,)
        Pi_unc = scale[:, None] * omega_inv_eta[None, :]            # (Nint, n)
        Pi_clipped = _clip_leverage_nd_batch(Pi_unc, d, u, max_long, max_short)
        Pi[concave] = Pi_clipped[concave]

    # ── Convex region: max convex quadratic → evaluate corner candidates ────
    # NOTE (§6.5 of EXPERIMENT_REVIEW.md): The leverage-constrained box in
    # n dimensions has 2^n vertices and O(n) active-constraint face candidates.
    # This implementation evaluates only 3 representative corners:
    #   (i)  greedy-by-η-sign: u if η_i ≥ 0 else d, then leverage-clipped
    #   (ii) all-u corner after leverage clip
    #   (iii) all-d corner after leverage clip
    # For the goal-reaching problem the convex region arises when V is locally
    # convex (typically far from the goal or near the absorbing boundary).
    # In practice, most nodes are in the concave region where the interior
    # maximiser is exact.  At n=20 the 3-corner heuristic may miss the true
    # optimum in some convex-region nodes, but the numerical impact on the
    # value function has been found to be negligible in testing at n≤20.
    # A comprehensive convex-hull search is a potential improvement (§8).
    if np.any(convex):
        pi_greedy = _clip_leverage_nd(
            np.where(eta >= 0, u, d), d, u, max_long, max_short)
        pi_all_u  = _clip_leverage_nd(np.full(n, u), d, u, max_long, max_short)
        pi_all_d  = _clip_leverage_nd(np.full(n, d), d, u, max_long, max_short)
        candidates = [pi_greedy, pi_all_u, pi_all_d]

        idx_c   = np.where(convex)[0]
        wi_c    = wi[idx_c]
        Vww_c   = Vww[idx_c]
        Vw_c    = Vw[idx_c]

        best_val = np.full(len(idx_c), -np.inf)
        best_pi  = np.zeros((len(idx_c), n))

        for cand in candidates:
            port_var = float(cand @ omega_mat @ cand)   # scalar
            port_ret = float(eta @ cand)                # scalar
            # H(cand) = ½ w² V_ww σ²_eff + w V_w η_eff
            vals = (0.5 * wi_c**2 * Vww_c * port_var
                    + wi_c * Vw_c * port_ret)           # (Nc,)
            better = vals > best_val
            best_val = np.where(better, vals, best_val)
            best_pi  = np.where(better[:, None], cand[None, :], best_pi)

        Pi[idx_c] = best_pi

    # flat region stays zero
    return Pi


def fd_solve_nd(mu_vec, r, omega_mat, T, A, Nw, Nt, d, u,
                utility_fn, asymptotic_fn,
                max_long=3.0, max_short=5.0,
                UB=0.0, UA=None, tau_blend=0.08, store_policy_path=False):
    """
    Multi-asset HJB solver: Dai et al. (2019) extended to n risky assets.

    The PDE is still 2-D in (t, w).  The n-D policy enters only through
    two scalar quantities per node:
        σ²_eff = π^T Ω π   and   η_eff = η^T π
    which feed the same Thomas tridiagonal as fd_solve.

    Complexity: O(Nw × Nt × n²)  vs  O(Nw × Nt) for the 1-D solver.

    Parameters
    ----------
    mu_vec    : (n,) annualised total returns
    r         : float  risk-free rate
    omega_mat : (n,n)  covariance matrix
    T, A, Nw, Nt, d, u : same as fd_solve
    utility_fn, asymptotic_fn : same as fd_solve (1-D functions of scalar w)
    max_long, max_short : aggregate leverage caps (applied per-node)
    store_policy_path : bool — if True, also return Pi_path (Nt, Nw+1, n)
                        and tau_path (Nt,) for time-aware policy interpolation.

    Returns
    -------
    w_grid  : (Nw+1,)     wealth grid
    V_grid  : (Nw+1,)     value function at t=0
    Pi_grid : (Nw+1, n)   optimal policy at t=0  (always returned)

    Additional returns when store_policy_path=True:
    Pi_path  : (Nt, Nw+1, n) policy at each backward step; row s has tau=tau_path[s].
    tau_path : (Nt,) tau values, tau_path[s] = (s+1)*dt, monotonically increasing.
    """
    mu_vec    = np.asarray(mu_vec, float)
    omega_mat = np.asarray(omega_mat, float)
    n         = len(mu_vec)
    eta       = mu_vec - r                             # excess returns (n,)

    # Ω^{-1}η — closed-form direction of the unconstrained Merton portfolio
    try:
        omega_inv_eta = np.linalg.solve(omega_mat, eta)
    except np.linalg.LinAlgError:
        omega_inv_eta = np.linalg.pinv(omega_mat) @ eta

    # Equal-risk-weighted 1-D aggregate for the asymptotic warmstart
    w_eq    = np.ones(n) / n
    sig_1d  = float(np.sqrt(w_eq @ omega_mat @ w_eq))
    eta_1d  = float(eta @ w_eq)

    w  = np.linspace(0.0, A, Nw + 1)
    dw = A / Nw
    dt = T / Nt
    wi = w[1:Nw]

    if UA is None:
        UA = float(utility_fn(np.array([A])).flat[0])

    V  = utility_fn(w).astype(float)
    Pi = np.zeros((Nw + 1, n))

    if store_policy_path:
        Pi_path  = np.empty((Nt, Nw + 1, n))
        tau_path = np.arange(1, Nt + 1, dtype=float) * dt

    for step in range(Nt):
        tau   = (Nt - step) * dt
        V_old = V.copy()

        # Blended asymptotic / Browne warmstart (same logic as fd_solve)
        alpha = float(np.exp(-tau / tau_blend))
        V_ws  = (alpha * asymptotic_fn(w, tau)
                 + (1.0 - alpha) * browne_V(w, eta_1d, sig_1d, tau))

        pi_n = np.zeros((Nw + 1, n))
        pi_n[1:Nw] = _policy_nd_from_V(
            V_ws, wi, dw, eta, omega_mat, omega_inv_eta,
            d, u, max_long, max_short)

        # Policy iteration (Howard's algorithm — same loop structure as fd_solve)
        for _ in range(60):
            pi_old = pi_n.copy()
            pi     = pi_n[1:Nw]                         # (Nint, n)

            # Effective scalar parameters at each interior node
            sig2_eff = np.einsum('ij,jk,ik->i', pi, omega_mat, pi)  # (Nint,)
            eta_eff  = pi @ eta                                        # (Nint,)

            # Tridiagonal coefficients — identical to fd_solve
            a2  = 0.5 * sig2_eff * wi**2
            Adv = eta_eff * wi
            Ap  = np.maximum(Adv, 0.0) / dw
            Am  = np.minimum(Adv, 0.0) / dw
            a_s = -dt * (a2 / dw**2 - Am)
            b_m =  1.0 + dt * (2.0*a2/dw**2 + Ap - Am)
            c_s = -dt * (a2 / dw**2 + Ap)

            rhs = V_old[1:Nw].copy()
            rhs[0]  -= a_s[0]  * UB;  a_s[0]  = 0.0
            rhs[-1] -= c_s[-1] * UA;  c_s[-1] = 0.0

            V_int = thomas(a_s, b_m, c_s, rhs)
            V_new = np.empty(Nw + 1)
            V_new[0]  = UB
            V_new[Nw] = UA
            V_new[1:Nw] = V_int

            pi_n[1:Nw] = _policy_nd_from_V(
                V_new, wi, dw, eta, omega_mat, omega_inv_eta,
                d, u, max_long, max_short)

            if np.max(np.abs(pi_n - pi_old)) < 1e-8:
                break

        V  = V_new
        Pi = pi_n

        if store_policy_path:
            Pi_path[step] = pi_n.copy()

    if store_policy_path:
        return w, V, Pi, Pi_path, tau_path
    return w, V, Pi


def make_fd_policy_nd(w_grid, Pi_grid, d=-5.0, u=3.0):
    """
    Return a callable policy_fn(w_norm, tau) -> (n,) or (N,n) array from a
    precomputed multi-asset FD grid.
    w_norm = current_wealth / goal.
    Accepts both scalar w_norm -> (n,) and array w_norm (N,) -> (N,n)
    for vectorised MC simulation.
    """
    n = Pi_grid.shape[1]
    def policy(w_norm, tau=1.0):
        scalar = np.ndim(w_norm) == 0
        w = np.atleast_1d(np.asarray(w_norm, float))
        # Stack per-asset interpolations: (N, n)
        Pi = np.stack(
            [np.clip(np.interp(w, w_grid, Pi_grid[:, i]), d, u) for i in range(n)],
            axis=1,
        )
        return Pi[0] if scalar else Pi   # (n,) or (N, n)
    return policy


def make_fd_policy_nd_time_aware(w_grid, Pi_path, tau_path, d=-5.0, u=3.0):
    """
    Return a time-aware multi-asset callable policy_fn(w_norm, tau) -> (n,)
    that bilinearly interpolates in both wealth and time-to-horizon.

    Pi_path  : (Nt, Nw+1, n) — from fd_solve_nd with store_policy_path=True.
    tau_path : (Nt,) — tau_path[s] = (s+1)*dt, monotonically increasing.
    """
    n = Pi_path.shape[2]

    def policy(w_norm, tau=1.0):
        tau_q    = float(np.clip(float(tau), tau_path[0], tau_path[-1]))
        idx_frac = float(np.interp(tau_q, tau_path, np.arange(len(tau_path))))
        idx_lo   = int(np.floor(idx_frac))
        idx_hi   = min(idx_lo + 1, len(tau_path) - 1)
        alpha    = idx_frac - idx_lo

        w_arr  = np.atleast_1d(np.asarray(w_norm, float))
        scalar = np.ndim(w_norm) == 0

        def _interp_slice(s):
            return np.stack(
                [np.clip(np.interp(w_arr, w_grid, Pi_path[s, :, i]), d, u)
                 for i in range(n)],
                axis=1,
            )

        Pi_lo = _interp_slice(idx_lo)
        Pi_hi = _interp_slice(idx_hi)
        Pi    = Pi_lo * (1.0 - alpha) + Pi_hi * alpha  # (N, n)
        return Pi[0] if scalar else Pi
    return policy
