"""
backtest_core.py
----------------
Walk-forward backtest engine and multi-year rolling re-optimisation.

All functions are pure (no global state) — market params are explicit args.
"""

import numpy as np

# ── Single-year simulators ──────────────────────────────────────────────────

def simulate_one_year_1asset(W_start, goal, policy_fn, mu, r, sigma,
                              days=252, rng=None):
    """
    Simulate one year of a 1-asset strategy under GBM.

    policy_fn : callable(w_norm, tau) -> pi  where w_norm = W/goal
    Returns
    -------
    W_path  : (days+1,) wealth path
    pi_path : (days,)   daily portfolio weight
    """
    if rng is None:
        rng = np.random.default_rng()
    dt            = 1.0 / days
    log_ret_stock = ((mu - 0.5*sigma**2)*dt
                     + sigma*np.sqrt(dt)*rng.standard_normal(days))
    r_daily       = r * dt

    W_path  = np.empty(days + 1);  W_path[0] = W_start
    pi_path = np.empty(days)

    for t in range(days):
        tau           = max(1.0 - t*dt, dt)
        w_norm        = W_path[t] / goal
        pi            = float(policy_fn(w_norm, tau))
        pi_path[t]    = pi
        ret_excess    = pi*(np.exp(log_ret_stock[t]) - 1) + (1 - pi)*(np.exp(r_daily) - 1)
        W_path[t+1]   = max(W_path[t] * (1 + ret_excess), 1e-6)

    return W_path, pi_path


def simulate_one_year_5asset(W_start, goal, nn_policy_fn, mu_vec, Omega_mat, r,
                              d=-5.0, u=3.0,
                              target_vol=None,   # deprecated — pass None to disable
                              days=252, rng=None):
    """
    Simulate one year of a multi-asset NN strategy.

    Variance normalisation (target_vol) is disabled by default so that
    the NN policy is evaluated as-trained, making its comparison with the FD
    benchmark fair.  Pass ``target_vol=0.25`` only for standalone risk-budget
    studies where you explicitly want vol targeting.

    nn_policy_fn : callable(W_current, goal) -> pi_vec (n_assets,)
    Returns
    -------
    W_path    : (days+1,) wealth path
    pi_matrix : (days, n_assets) weight matrix
    """
    if rng is None:
        rng = np.random.default_rng()
    n      = len(mu_vec)
    dt     = 1.0 / days
    L_chol = np.linalg.cholesky(Omega_mat)
    Z      = rng.standard_normal((days, n))
    log_rets = ((mu_vec - 0.5*np.diag(Omega_mat))*dt
                + (Z @ L_chol.T)*np.sqrt(dt))
    r_daily = r * dt

    W_path    = np.empty(days + 1);  W_path[0] = W_start
    pi_matrix = np.empty((days, n))

    for t in range(days):
        pi = np.array(nn_policy_fn(W_path[t], goal), dtype=float)

        # Optional vol-targeting — disabled by default to avoid confounding
        # the NN vs FD comparison (see nn_core.py note on volatility normalisation).
        if target_vol is not None:
            port_var = float(pi @ Omega_mat @ pi)
            if port_var > 1e-8:
                scale = min(target_vol / np.sqrt(port_var), 2.0)
                pi    = np.clip(pi * scale, d, u)

        pi_matrix[t] = pi
        gross         = np.exp(log_rets[t]) - 1
        excess        = np.dot(pi, gross - r_daily)
        W_path[t+1]   = max(W_path[t] * (1 + r_daily + excess), 1e-6)

    return W_path, pi_matrix


# ── Walk-forward backtest (used by notebook 05) ────────────────────────────

def run_backtest_1d(log_returns, r_daily, strategy_fn, strategy_kwargs,
                    T_horizon=1.0, W0=1.0, d=-5.0, u=3.0):
    """
    Daily-rebalanced walk-forward backtest (1-asset).
    strategy_fn: callable(w, tau, **strategy_kwargs) -> scalar pi
    Returns (wealth_path, pi_path).
    """
    n       = len(log_returns)
    wealth  = np.zeros(n + 1);  wealth[0] = W0
    pi_path = np.zeros(n)
    dt_d    = 1 / 252

    for t in range(n):
        w          = wealth[t]
        tau        = max(T_horizon - t*dt_d, dt_d)
        pi         = float(np.clip(strategy_fn(w, tau, **strategy_kwargs), d, u))
        pi_path[t] = pi
        daily_ret  = np.exp(log_returns[t]) - 1
        wealth[t+1] = max(w * (1 + r_daily + pi*(daily_ret - r_daily)), 1e-6)

    return wealth, pi_path


def compute_metrics(wealth, r_daily=0.03/252, target_return=1.10, freq=252):
    """
    Standard performance metrics for a wealth path.
    Returns dict with ann_ret, ann_vol, sharpe, max_dd, goal_rate.
    """
    ret     = np.diff(wealth) / wealth[:-1]
    ann_ret = (wealth[-1] / wealth[0]) ** (freq / len(ret)) - 1
    ann_vol = ret.std() * np.sqrt(freq)
    sharpe  = (ann_ret - r_daily*freq) / (ann_vol + 1e-8)
    peak    = np.maximum.accumulate(wealth)
    max_dd  = ((wealth - peak) / (peak + 1e-8)).min()
    n_win   = max(1, len(wealth) - freq)
    gr      = sum(wealth[t+freq] >= target_return * wealth[t]
                  for t in range(n_win)) / n_win
    return dict(ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
                max_dd=max_dd, goal_rate=gr)


# ── Multi-year rolling engine ───────────────────────────────────────────────

def run_multi_year(n_years, W0, annual_target, strategy,
                   fd_gr_policy=None, fd_as_policy=None,
                   nn_gr_nets=None, nn_as_nets=None,
                   nn_policy_fn=None,
                   mu=0.12, r=0.03, sigma=0.18,
                   mu_vec=None, Omega_mat=None,
                   d=-5.0, u=3.0, days=252, seed=42):
    """
    Rolling re-optimisation over n_years.

    strategy : 'fd_goalreach' | 'fd_aspiration' | 'browne' | 'kelly' | 'nn_5asset'

    For 'fd_*'      pass fd_gr_policy / fd_as_policy (callables from make_fd_policy)
    For 'nn_5asset' pass nn_gr_nets / nn_as_nets (list of nets) and nn_policy_fn
    For 'browne'/'kelly' only mu, r, sigma are needed.

    Returns dict:
      W          : (n_years*days+1,) wealth path
      pi         : (n_years*days,) or (n_years*days, n) weights
      goals      : (n_years,) absolute goal each year
      year_end_W : (n_years,) end-of-year wealth
      goal_hit   : (n_years,) bool
    """
    rng   = np.random.default_rng(seed)
    W_all = [W0]
    pi_all, goals, year_end_W = [], [], []
    W_cur = W0

    for yr in range(n_years):
        goal = W_cur * annual_target
        goals.append(goal)

        if strategy == 'fd_goalreach':
            W_path, pi_path = simulate_one_year_1asset(
                W_cur, goal, fd_gr_policy, mu, r, sigma, days=days, rng=rng)
            pi_all.append(pi_path)

        elif strategy == 'fd_aspiration':
            W_path, pi_path = simulate_one_year_1asset(
                W_cur, goal, fd_as_policy, mu, r, sigma, days=days, rng=rng)
            pi_all.append(pi_path)

        elif strategy == 'browne':
            eta  = mu - r
            sig2 = sigma**2
            def _pi_browne(w_norm, tau, _eta=eta, _s2=sig2, _s=sigma,
                           _d=d, _u=u):
                f   = _s * np.sqrt(max(tau, 1/252))
                z   = (np.log(max(w_norm, 1e-6)) + (_eta - 0.5*_s2)*max(tau, 1/252)) / f
                den = _s2 * (1.0 + z/f)
                raw = _eta/den if abs(den) > 1e-8 else (_u if _eta > 0 else _d)
                return float(np.clip(raw, _d, _u))
            W_path, pi_path = simulate_one_year_1asset(
                W_cur, goal, _pi_browne, mu, r, sigma, days=days, rng=rng)
            pi_all.append(pi_path)

        elif strategy == 'kelly':
            kelly_pi = float(np.clip((mu - r)/sigma**2, d, u))
            W_path, pi_path = simulate_one_year_1asset(
                W_cur, goal, lambda w, t: kelly_pi, mu, r, sigma, days=days, rng=rng)
            pi_all.append(pi_path)

        elif strategy == 'nn_5asset':
            assert mu_vec is not None and Omega_mat is not None
            net = nn_gr_nets[yr] if nn_gr_nets is not None else nn_as_nets[yr]
            W_path, pi_mat = simulate_one_year_5asset(
                W_cur, goal, lambda W, g: nn_policy_fn(net, W, g),
                mu_vec, Omega_mat, r, d=d, u=u, days=days, rng=rng)
            pi_all.append(pi_mat)

        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        W_all.extend(W_path[1:].tolist())
        year_end_W.append(float(W_path[-1]))
        W_cur = W_path[-1]

    is_5asset = (strategy == 'nn_5asset')
    pi_arr    = np.vstack(pi_all) if is_5asset else np.concatenate(pi_all)

    return {
        'W':          np.array(W_all),
        'pi':         pi_arr,
        'goals':      np.array(goals),
        'year_end_W': np.array(year_end_W),
        'goal_hit':   np.array(year_end_W) >= np.array(goals),
    }
