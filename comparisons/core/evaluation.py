import time
from typing import Dict, List

import numpy as np

from backtest_core import run_backtest_1d
from fd_core import (
    asymp_goalreach, fd_solve, fd_solve_nd,
    goal_utility, make_fd_policy, make_fd_policy_nd,
    make_fd_policy_time_aware, make_fd_policy_nd_time_aware,
)
from comparisons.core.nn_models import (
    policy_weights as numpy_policy_weights,
    train_numpy_policy_net,
    _apply_leverage_constraint_numpy,
)
from real_data_loader import MarketData, agg_1d, load_portfolio

HAS_TORCH = False
TORCH_ARCHITECTURES: dict = {}
try:
    import torch  # noqa: F401
    from comparisons.core.torch_nn_models import (
        TORCH_ARCHITECTURES,
        policy_weights as torch_policy_weights,
        train_torch_policy_net,
    )
    HAS_TORCH = True
except Exception:
    torch = None
    torch_policy_weights = None
    train_torch_policy_net = None

from .artifacts import artifact_filename, save_fd_artifact, save_torch_model_artifact
from .config import BenchmarkConfig
from .io import ensure_dir, save_run_result, save_summary_csv
from .metrics import (
    compute_drawdown_series,
    compute_path_risk_metrics,
    compute_target_metrics,
    compute_weight_metrics,
)
from .reporting import save_comparison_tables, save_plots

REQUIRED_RESULT_KEYS = (
    "method_family", "method_name", "n_assets", "seed",
    "initial_wealth", "target_wealth", "train_time_sec",
    "solve_time_sec", "eval_time_sec", "wealth_path", "weight_path",
    "terminal_wealth", "goal_hit", "gross_leverage_path",
    "net_exposure_path", "concentration_path", "drawdown_path",
)

_MARKET_CAP_WEIGHTS = {
    "IVV": 0.24, "QQQ": 0.18, "IWM": 0.06, "VEA": 0.07, "VWO": 0.05,
    "VNQ": 0.05, "TLT": 0.08, "IEF": 0.05, "SHY": 0.02, "TIP": 0.03,
    "LQD": 0.03, "HYG": 0.02, "GLD": 0.04, "SLV": 0.01, "DBC": 0.01,
    "USO": 0.01, "XLK": 0.03, "XLF": 0.01, "XLV": 0.01, "UUP": 0.01,
}


# ---------------------------------------------------------------
# Leverage constraint helper
# ---------------------------------------------------------------

def apply_leverage_constraint(weights, d, u, max_long, max_short):
    """
    Enforce per-asset box bounds [d, u] then aggregate long/short caps.
    Long  side: sum(max(w_i, 0)) <= max_long
    Short side: sum(max(-w_i, 0)) <= max_short
    Never amplifies; safe for 1-D vectors or 2-D (B x n) batches.
    """
    weights = np.clip(weights, d, u)
    squeeze = weights.ndim == 1
    if squeeze:
        weights = weights[None, :]
    long_lev = np.maximum(weights, 0.0).sum(axis=1, keepdims=True).clip(min=1e-12)
    short_lev = np.maximum(-weights, 0.0).sum(axis=1, keepdims=True).clip(min=1e-12)
    long_scale = np.minimum(max_long / long_lev, 1.0)
    short_scale = np.minimum(max_short / short_lev, 1.0)
    weights = np.where(weights >= 0, weights * long_scale, weights * short_scale)
    return weights.squeeze(0) if squeeze else weights


# ---------------------------------------------------------------
# Result schema helpers
# ---------------------------------------------------------------

def validate_result_schema(result):
    missing = [k for k in REQUIRED_RESULT_KEYS if k not in result]
    if missing:
        raise KeyError(f"Result missing required keys: {missing}")
    wp = np.asarray(result["wealth_path"], dtype=float)
    wt = np.asarray(result["weight_path"], dtype=float)
    if wp.ndim != 1:
        raise ValueError("wealth_path must be 1D")
    if wt.ndim not in (1, 2):
        raise ValueError("weight_path must be 1D or 2D")
    if len(wp) != len(wt) + 1:
        raise ValueError("wealth_path length must equal weight_path length + 1")
    return True


def _cash_weighted_growth(weights, gross_returns, r_daily):
    cash_weight = 1.0 - float(np.sum(weights))
    return cash_weight * r_daily + float(np.dot(weights, gross_returns))


def _build_result(method_family, method_name, market_data, initial_wealth,
                  target_wealth, wealth_path, weight_path, seed=1,
                  train_time_sec=0.0, solve_time_sec=0.0, eval_time_sec=0.0):
    weight_path = np.asarray(weight_path, dtype=float)
    if weight_path.ndim == 1:
        weight_path = weight_path[:, None]
    wealth_path = np.asarray(wealth_path, dtype=float)
    result = {
        "method_family": method_family,
        "method_name": method_name,
        "n_assets": int(weight_path.shape[1]),
        "seed": int(seed),
        "initial_wealth": float(initial_wealth),
        "target_wealth": float(target_wealth),
        "train_time_sec": float(train_time_sec),
        "solve_time_sec": float(solve_time_sec),
        "eval_time_sec": float(eval_time_sec),
        "wealth_path": wealth_path,
        "weight_path": weight_path,
        "terminal_wealth": np.array([float(wealth_path[-1])]),
        "goal_hit": np.array([wealth_path[-1] >= target_wealth], dtype=bool),
        "gross_leverage_path": np.abs(weight_path).sum(axis=1),
        "net_exposure_path": weight_path.sum(axis=1),
        "concentration_path": np.square(weight_path).sum(axis=1),
        "drawdown_path": compute_drawdown_series(wealth_path),
        "dates": np.asarray(market_data.dates),
        "tickers": np.asarray(market_data.tickers[:weight_path.shape[1]]),
    }
    validate_result_schema(result)
    return result


def result_to_summary_row(result):
    target_metrics = compute_target_metrics(result["terminal_wealth"], result["target_wealth"])
    weight_metrics = compute_weight_metrics(result["weight_path"])
    risk_metrics = compute_path_risk_metrics(result["wealth_path"])
    return {
        "method_name": result["method_name"],
        "method_family": result["method_family"],
        "n_assets": result["n_assets"],
        "seed": result["seed"],
        "initial_wealth": result["initial_wealth"],
        "target_wealth": result["target_wealth"],
        "train_time_sec": result["train_time_sec"],
        "solve_time_sec": result["solve_time_sec"],
        "eval_time_sec": result["eval_time_sec"],
        **target_metrics,
        **weight_metrics,
        **risk_metrics,
    }


# ---------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------

def evaluate_static_portfolio(method_name, method_family, market_data, weights,
                               initial_wealth=1.0, target_multiplier=1.10, seed=1):
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or len(weights) != market_data.n:
        raise ValueError("weights must be 1-D with length market_data.n")
    target_wealth = initial_wealth * target_multiplier
    r_daily = market_data.r / 252.0
    gross_returns = np.exp(market_data.log_ret) - 1.0
    start = time.perf_counter()
    wealth = np.empty(len(gross_returns) + 1, dtype=float)
    wealth[0] = initial_wealth
    for idx, daily_gross in enumerate(gross_returns):
        wealth[idx + 1] = max(
            wealth[idx] * (1.0 + _cash_weighted_growth(weights, daily_gross, r_daily)),
            1e-6,
        )
    weight_path = np.repeat(weights[None, :], len(gross_returns), axis=0)
    return _build_result(
        method_family=method_family, method_name=method_name,
        market_data=market_data, initial_wealth=initial_wealth,
        target_wealth=target_wealth, wealth_path=wealth,
        weight_path=weight_path, seed=seed,
        eval_time_sec=time.perf_counter() - start,
    )


def market_cap_weights(tickers):
    raw = np.array([_MARKET_CAP_WEIGHTS.get(t, 0.01) for t in tickers], dtype=float)
    s = raw.sum()
    return raw / s if s > 0 else np.ones(len(tickers)) / len(tickers)


def max_sharpe_weights(mu_vec, omega_mat, r):
    eta = np.asarray(mu_vec, dtype=float) - float(r)
    raw = np.clip(np.linalg.pinv(omega_mat) @ eta, 0.0, None)
    s = raw.sum()
    return raw / s if s > 1e-12 else np.ones(len(mu_vec)) / len(mu_vec)


# ---------------------------------------------------------------
# FD 1-D proxy benchmark
# ---------------------------------------------------------------

def evaluate_fd_benchmark(market_data, config, initial_wealth=1.0, seed=1,
                          eval_market_data=None):
    """
    Corrected HJB viscosity FD solver.

    n == 1  →  exact 1-D Dai et al. (2019) scheme (fd_solve, O(Nw·Nt))
    n  > 1  →  proper multi-asset extension (fd_solve_nd, O(Nw·Nt·n²)):
               policy π*(w,t) ∈ R^n at each node via closed-form QP;
               evaluated on the full n-asset historical return series.

    Parameters
    ----------
    market_data      : MarketData — calibration window (used for FD solve)
    eval_market_data : MarketData or None — evaluation window used for the
                       historical backtest.  If None, market_data is used for
                       both (legacy / in-sample behaviour).
    """
    # Use a separate evaluation window if provided (out-of-sample backtest)
    backtest_data = eval_market_data if eval_market_data is not None else market_data
    n = market_data.n

    # ── 1-D path (n=1 or deliberate 1D-proxy mode) ──────────────────────────
    if n == 1:
        mu_1d  = float(market_data.mu_ann[0])
        sig_1d = float(market_data.sigma[0])
        log_ret_1d = backtest_data.log_ret[:, 0]

        t0 = time.perf_counter()
        w_grid, _, pi_grid, Pi_path_1d, tau_path_1d = fd_solve(
            mu=mu_1d, r=market_data.r, sigma=sig_1d, T=1.0,
            A=config.fd_wealth_max, Nw=config.fd_nw, Nt=config.fd_nt,
            d=config.weight_lower_bound, u=config.weight_upper_bound,
            utility_fn=goal_utility,
            asymptotic_fn=lambda w, tau: asymp_goalreach(
                w, tau, sig_1d,
                config.weight_lower_bound, config.weight_upper_bound),
            UB=0.0, UA=1.0, store_policy_path=True,
        )
        solve_time_sec = time.perf_counter() - t0

        policy = make_fd_policy_time_aware(
            w_grid, Pi_path_1d, tau_path_1d,
            d=config.weight_lower_bound, u=config.weight_upper_bound)
        t1 = time.perf_counter()
        wealth_path, pi_path = run_backtest_1d(
            log_returns=log_ret_1d, r_daily=market_data.r / 252.0,
            strategy_fn=lambda w, tau, p: p(
                w / (initial_wealth * config.target_multiplier), tau),
            strategy_kwargs={"p": policy}, T_horizon=1.0, W0=initial_wealth,
            d=config.weight_lower_bound, u=config.weight_upper_bound,
        )
        result = _build_result(
            method_family="fd", method_name="fd_1d",
            market_data=backtest_data, initial_wealth=initial_wealth,
            target_wealth=initial_wealth * config.target_multiplier,
            wealth_path=wealth_path, weight_path=pi_path[:, None], seed=seed,
            solve_time_sec=solve_time_sec,
            eval_time_sec=time.perf_counter() - t1,
        )
        result["_fd_artifact"] = {
            "w_grid": w_grid, "pi_grid": pi_grid,
            "Pi_path": Pi_path_1d, "tau_path": tau_path_1d,
            "metadata": {
                "d": config.weight_lower_bound,
                "u": config.weight_upper_bound,
                "goal": float(config.target_multiplier),
                "n_assets": 1,
            },
        }
        return result

    # ── Multi-asset path (n > 1) — full n-D FD solver ────────────────────────
    # Aggregate σ for the asymptotic warmstart  (equal-risk-weighted)
    w_eq   = np.ones(n) / n
    sig_agg = float(np.sqrt(w_eq @ market_data.omega @ w_eq))

    t0 = time.perf_counter()
    w_grid, _, Pi_grid, Pi_path_nd, tau_path_nd = fd_solve_nd(
        mu_vec=market_data.mu_ann, r=market_data.r,
        omega_mat=market_data.omega, T=1.0,
        A=config.fd_wealth_max, Nw=config.fd_nw, Nt=config.fd_nt,
        d=config.weight_lower_bound, u=config.weight_upper_bound,
        utility_fn=goal_utility,
        asymptotic_fn=lambda w, tau: asymp_goalreach(
            w, tau, sig_agg,
            config.weight_lower_bound, config.weight_upper_bound),
        max_long=config.max_long_leverage,
        max_short=config.max_short_leverage,
        UB=0.0, UA=1.0, store_policy_path=True,
    )
    solve_time_sec = time.perf_counter() - t0

    policy_nd = make_fd_policy_nd_time_aware(
        w_grid, Pi_path_nd, tau_path_nd,
        d=config.weight_lower_bound, u=config.weight_upper_bound)

    # Walk-forward backtest on the evaluation window (out-of-sample if split)
    target_wealth  = initial_wealth * config.target_multiplier
    gross_returns  = np.exp(backtest_data.log_ret) - 1.0   # (T, n)
    r_daily        = market_data.r / 252.0
    T_horizon      = 1.0

    t1 = time.perf_counter()
    wealth      = np.empty(len(gross_returns) + 1, dtype=float)
    wealth[0]   = initial_wealth
    weight_path = np.empty((len(gross_returns), n), dtype=float)

    for idx, daily_gross in enumerate(gross_returns):
        tau         = max(T_horizon - idx / 252.0, 1.0 / 252.0)
        w_norm      = wealth[idx] / target_wealth
        pi          = policy_nd(w_norm, tau)              # (n,)
        # Apply leverage cap at eval time (same as NN methods)
        pi          = apply_leverage_constraint(
            pi, config.weight_lower_bound, config.weight_upper_bound,
            config.max_long_leverage, config.max_short_leverage)
        weight_path[idx] = pi
        wealth[idx + 1]  = max(
            wealth[idx] * (1.0 + _cash_weighted_growth(pi, daily_gross, r_daily)),
            1e-6)

    result = _build_result(
        method_family="fd", method_name="fd_nd",
        market_data=backtest_data, initial_wealth=initial_wealth,
        target_wealth=target_wealth,
        wealth_path=wealth, weight_path=weight_path, seed=seed,
        solve_time_sec=solve_time_sec,
        eval_time_sec=time.perf_counter() - t1,
    )
    result["_fd_artifact"] = {
        "w_grid": w_grid, "pi_grid": Pi_grid,
        "Pi_path": Pi_path_nd, "tau_path": tau_path_nd,
        "metadata": {
            "d": config.weight_lower_bound,
            "u": config.weight_upper_bound,
            "goal": float(config.target_multiplier),
            "n_assets": n,
        },
    }
    return result


# ---------------------------------------------------------------
# Merton multi-asset FD benchmark (proper n-D analytical proxy)
# ---------------------------------------------------------------

def _merton_weights(market_data, config):
    """
    Ω^{-1}(μ − r·1) — Merton maximum-growth portfolio normalised to unit
    gross leverage, then projected onto the feasible set with the same
    long/short leverage caps applied to all neural methods.
    """
    excess = np.asarray(market_data.mu_ann, dtype=float) - market_data.r
    try:
        raw = np.linalg.solve(market_data.omega, excess)
    except np.linalg.LinAlgError:
        raw = np.linalg.pinv(market_data.omega) @ excess
    gross_raw = np.abs(raw).sum()
    if gross_raw > 1e-10:
        raw = raw / gross_raw
    return apply_leverage_constraint(
        raw, config.weight_lower_bound, config.weight_upper_bound,
        config.max_long_leverage, config.max_short_leverage,
    )


def evaluate_merton_benchmark(market_data, config, initial_wealth=1.0, seed=1):
    """
    Multi-asset FD benchmark using the Merton analytical optimal policy.
    Uses the full n × n covariance matrix — the proper multi-dimensional
    reference point that the FD 1-D proxy cannot provide.
    """
    t0 = time.perf_counter()
    weights = _merton_weights(market_data, config)
    solve_time_sec = time.perf_counter() - t0

    target_wealth = initial_wealth * config.target_multiplier
    gross_returns = np.exp(market_data.log_ret) - 1.0
    r_daily = market_data.r / 252.0
    t1 = time.perf_counter()
    wealth = np.empty(len(gross_returns) + 1, dtype=float)
    wealth[0] = initial_wealth
    for idx, daily_gross in enumerate(gross_returns):
        wealth[idx + 1] = max(
            wealth[idx] * (1.0 + _cash_weighted_growth(weights, daily_gross, r_daily)),
            1e-6,
        )
    weight_path = np.repeat(weights[None, :], len(gross_returns), axis=0)
    return _build_result(
        method_family="fd", method_name="fd_merton_multi",
        market_data=market_data, initial_wealth=initial_wealth,
        target_wealth=target_wealth, wealth_path=wealth,
        weight_path=weight_path, seed=seed,
        solve_time_sec=solve_time_sec, eval_time_sec=time.perf_counter() - t1,
    )


# ---------------------------------------------------------------
# Neural network evaluation
# ---------------------------------------------------------------

def evaluate_nn_portfolio(market_data, config, architecture_name,
                          initial_wealth=1.0, seed=1,
                          device=None, compile_model=False,
                          eval_market_data=None):
    """
    Parameters
    ----------
    market_data      : MarketData — used for NN training (mu, omega, r)
    eval_market_data : MarketData or None — used for the historical backtest.
                       If None, market_data is used (legacy in-sample behaviour).
    """
    backtest_data = eval_market_data if eval_market_data is not None else market_data
    target_wealth = initial_wealth * config.target_multiplier
    gross_returns = np.exp(backtest_data.log_ret) - 1.0
    r_daily = market_data.r / 252.0

    total_steps = len(gross_returns)   # number of daily backtest steps
    T_horizon   = getattr(config, 'nn_horizon_years', 1.0)

    t0 = time.perf_counter()
    train_meta = {}
    if HAS_TORCH and architecture_name in TORCH_ARCHITECTURES:
        arch_kind = TORCH_ARCHITECTURES.get(architecture_name, {}).get('kind', '')
        # historical_replay uses calib-window returns for block-bootstrap training
        calib_gross = np.exp(market_data.log_ret) - 1.0
        hist_rets = calib_gross if arch_kind == 'historical_replay' else None

        net, train_meta = train_torch_policy_net(
            mu_vec=market_data.mu_ann, omega_mat=market_data.omega, r=market_data.r,
            architecture_name=architecture_name, w0=initial_wealth,
            goal_mult=config.target_multiplier, n_paths=config.nn_paths,
            n_iters=config.nn_iters, n_steps=config.nn_steps, lr=3e-3,
            d=config.weight_lower_bound, u=config.weight_upper_bound,
            max_long_leverage=config.max_long_leverage,
            max_short_leverage=config.max_short_leverage, seed=seed,
            # ── new training improvements ──────────────────────────────────
            pretrain_iters=getattr(config, 'nn_pretrain_iters', 100),
            antithetic=getattr(config, 'nn_antithetic', True),
            p_curriculum=getattr(config, 'nn_p_curriculum', 0.30),
            patience=getattr(config, 'nn_patience', 60),
            T=T_horizon,
            historical_returns=hist_rets,
            # ── GPU / compile options ──────────────────────────────────────
            device=device,
            compile_model=compile_model,
        )
        # Pass step_idx AND wealth history so the network receives the correct
        # tau and rolling features at each step.
        wealth_history_list = []

        def infer_weights(w, step_idx=0):
            hist_arg = wealth_history_list if arch_kind == 'historical_replay' else None
            raw = np.asarray(
                torch_policy_weights(net, w, target_wealth,
                                     history=hist_arg,
                                     step_idx=step_idx,
                                     total_steps=total_steps),
                dtype=float,
            )
            wealth_history_list.append(float(w))
            return apply_leverage_constraint(raw, config.weight_lower_bound,
                                             config.weight_upper_bound,
                                             config.max_long_leverage,
                                             config.max_short_leverage)
    else:
        net, train_meta = train_numpy_policy_net(
            mu_vec=market_data.mu_ann, omega_mat=market_data.omega, r=market_data.r,
            architecture_name=architecture_name, w0=initial_wealth,
            goal_mult=config.target_multiplier, n_paths=config.nn_paths,
            n_iters=config.nn_iters, population_size=config.nn_population_size,
            elite_frac=config.nn_elite_frac, n_steps=config.nn_steps,
            d=config.weight_lower_bound, u=config.weight_upper_bound,
            max_long_leverage=config.max_long_leverage,
            max_short_leverage=config.max_short_leverage, seed=seed,
        )
        def infer_weights(w, step_idx=0):
            raw = np.asarray(numpy_policy_weights(net, w, target_wealth), dtype=float)
            return apply_leverage_constraint(raw, config.weight_lower_bound,
                                             config.weight_upper_bound,
                                             config.max_long_leverage,
                                             config.max_short_leverage)
    train_time_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    wealth = np.empty(len(gross_returns) + 1, dtype=float)
    wealth[0] = initial_wealth
    weight_path = np.empty((len(gross_returns), market_data.n), dtype=float)
    for idx, daily_gross in enumerate(gross_returns):
        w = infer_weights(wealth[idx], step_idx=idx)   # tau-aware query
        weight_path[idx] = w
        wealth[idx + 1] = max(
            wealth[idx] * (1.0 + _cash_weighted_growth(w, daily_gross, r_daily)), 1e-6)

    result = _build_result(
        method_family="nn", method_name=architecture_name,
        market_data=backtest_data, initial_wealth=initial_wealth,
        target_wealth=target_wealth, wealth_path=wealth,
        weight_path=weight_path, seed=seed,
        train_time_sec=train_time_sec, eval_time_sec=time.perf_counter() - t1,
    )
    if train_meta:
        result["nn_param_count"] = int(train_meta.get("param_size", 0))
        if "backend" in train_meta:
            result["nn_backend"] = train_meta["backend"]
        # Training / validation / test curves
        if "loss_history" in train_meta:
            result["train_history"] = train_meta["loss_history"]
        if "val_history" in train_meta:
            result["val_history"]   = train_meta["val_history"]
            result["val_iters"]     = train_meta.get("val_iters", [])
        if "test_u" in train_meta:
            result["test_u"] = train_meta["test_u"]
    if HAS_TORCH and architecture_name in TORCH_ARCHITECTURES:
        result["_model_artifact"] = {
            "model": net,
            "metadata": {
                "architecture_name": architecture_name,
                "n_assets": market_data.n,
                "n_steps": config.nn_steps,
                "d": config.weight_lower_bound,
                "u": config.weight_upper_bound,
                **train_meta,
            },
        }
    return result


# ---------------------------------------------------------------
# ES-GRPO evaluator (mirrors evaluate_nn_portfolio)
# ---------------------------------------------------------------

def evaluate_grpo_portfolio(
    market_data,
    config,
    architecture_name: str = 'es_grpo',
    initial_wealth: float = 1.0,
    seed: int = 1,
    device=None,
    compile_model: bool = False,
    eval_market_data=None,
):
    """
    Train an ES-GRPO policy on calibrated GBM (μ̂, Ω̂, r) and evaluate
    on the historical return series.

    Parameters
    ----------
    market_data      : MarketData — used for training (mu, omega, r)
    config           : BenchmarkConfig
    architecture_name: str — must be in TORCH_ARCHITECTURES with kind='es_grpo'
    eval_market_data : MarketData or None — used for the historical backtest.
                       If None, market_data is used (legacy in-sample behaviour).

    Returns
    -------
    result dict matching REQUIRED_RESULT_KEYS (same schema as evaluate_nn_portfolio)
    """
    from comparisons.core.grpo_torch import train_es_grpo_torch, grpo_policy_weights

    backtest_data = eval_market_data if eval_market_data is not None else market_data
    target_wealth = initial_wealth * config.target_multiplier
    gross_returns = np.exp(backtest_data.log_ret) - 1.0
    r_daily       = market_data.r / 252.0
    total_steps   = len(gross_returns)
    T_horizon     = getattr(config, 'nn_horizon_years', 1.0)

    # Read ES-GRPO hyper-params from config (with defaults matching the spec)
    G           = getattr(config, 'es_grpo_G',        32)
    sigma_es    = getattr(config, 'es_grpo_sigma',    0.030)
    es_lr       = getattr(config, 'es_grpo_lr',       3e-3)
    n_iters     = getattr(config, 'es_grpo_iters',    config.nn_iters)
    n_paths     = getattr(config, 'es_grpo_paths',    config.nn_paths)
    n_steps     = config.nn_steps
    pretrain    = getattr(config, 'es_grpo_pretrain', 0)
    patience    = getattr(config, 'es_grpo_patience', config.nn_patience)

    t0 = time.perf_counter()
    net, train_meta = train_es_grpo_torch(
        mu_vec=market_data.mu_ann,
        omega_mat=market_data.omega,
        r=market_data.r,
        architecture_name=architecture_name,
        w0=initial_wealth,
        goal_mult=config.target_multiplier,
        G=G,
        sigma_es=sigma_es,
        lr=es_lr,
        n_paths=n_paths,
        n_iters=n_iters,
        n_steps=n_steps,
        d=config.weight_lower_bound,
        u=config.weight_upper_bound,
        max_long_leverage=config.max_long_leverage,
        max_short_leverage=config.max_short_leverage,
        pretrain_iters=pretrain,
        p_curriculum=getattr(config, 'nn_p_curriculum', 0.30),
        patience=patience,
        T=T_horizon,
        seed=seed,
        device=device,
    )
    train_time_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    wealth      = np.empty(len(gross_returns) + 1, dtype=float)
    wealth[0]   = initial_wealth
    weight_path = np.empty((len(gross_returns), market_data.n), dtype=float)

    for idx, daily_gross in enumerate(gross_returns):
        pi  = grpo_policy_weights(net, wealth[idx], target_wealth,
                                   step_idx=idx, total_steps=total_steps)
        pi  = apply_leverage_constraint(pi, config.weight_lower_bound,
                                        config.weight_upper_bound,
                                        config.max_long_leverage,
                                        config.max_short_leverage)
        weight_path[idx] = pi
        wealth[idx + 1]  = max(
            wealth[idx] * (1.0 + _cash_weighted_growth(pi, daily_gross, r_daily)), 1e-6
        )

    result = _build_result(
        method_family="nn", method_name=architecture_name,
        market_data=backtest_data, initial_wealth=initial_wealth,
        target_wealth=target_wealth, wealth_path=wealth,
        weight_path=weight_path, seed=seed,
        train_time_sec=train_time_sec, eval_time_sec=time.perf_counter() - t1,
    )
    if train_meta:
        result["nn_param_count"] = int(train_meta.get("param_size", 0))
        result["nn_backend"]     = train_meta.get("backend", "torch_es_grpo")
        if "loss_history" in train_meta:
            result["train_history"] = train_meta["loss_history"]
        if "val_history" in train_meta:
            result["val_history"] = train_meta["val_history"]
            result["val_iters"]   = train_meta.get("val_iters", [])
        if "test_u" in train_meta:
            result["test_u"] = train_meta["test_u"]
    result["_model_artifact"] = {
        "model"   : net,
        "metadata": {
            "architecture_name": architecture_name,
            "n_assets"         : market_data.n,
            "n_steps"          : n_steps,
            **train_meta,
        },
    }
    return result


# ---------------------------------------------------------------
# Monte Carlo evaluation helpers
# ---------------------------------------------------------------

def evaluate_policy_mc(policy_fn, market_data, initial_wealth, target_multiplier,
                       n_mc=500, seed=42, vol_scale_target=None):
    """
    Evaluate a policy using Monte Carlo simulation under GBM.

    For n=1: policy_fn(w_norm, tau) -> scalar pi
             where w_norm = W / goal
    For n>1: policy_fn(W, goal) -> np.array of shape (n,)

    Parameters
    ----------
    policy_fn       : callable — see above convention
    market_data     : MarketData
    initial_wealth  : float
    target_multiplier: float  (goal = initial_wealth * target_multiplier)
    n_mc            : int     — number of simulation paths
    seed            : int
    vol_scale_target: float or None — if given, rescale NN output so portfolio
                      vol matches this target (matches backtest_core convention)

    Returns
    -------
    dict with keys:
        terminal_wealth  (n_mc,)
        goal_hit         (n_mc,) bool
        wealth_paths     (n_mc, days+1)
        weight_paths     (n_mc, days, n)
        target_hit_prob  float
        mean_terminal_wealth float
        median_terminal_wealth float
        shortfall_mean   float  E[max(goal-W_T, 0)]
        sharpe_mean      float
    """
    from backtest_core import simulate_one_year_1asset, simulate_one_year_5asset
    goal = initial_wealth * target_multiplier
    rng  = np.random.default_rng(seed)

    n = market_data.n
    days = 252

    terminal_wealth = np.empty(n_mc)
    goal_hit        = np.empty(n_mc, dtype=bool)
    wealth_paths    = np.empty((n_mc, days + 1))
    weight_paths    = np.empty((n_mc, days, n))

    for i in range(n_mc):
        path_rng = np.random.default_rng(rng.integers(0, 2**31))

        if n == 1:
            mu_1d   = float(market_data.mu_ann[0])
            sig_1d  = float(market_data.sigma[0])
            r       = market_data.r
            W_path, pi_path = simulate_one_year_1asset(
                W_start=initial_wealth, goal=goal,
                policy_fn=policy_fn,
                mu=mu_1d, r=r, sigma=sig_1d,
                days=days, rng=path_rng,
            )
            wealth_paths[i]    = W_path
            weight_paths[i]    = pi_path[:, None]
        else:
            # For n>1 policies: policy_fn(W, goal) -> (n,) weights
            mu_1d, sig_1d, _ = agg_1d(market_data)

            # Wrap so simulate_one_year_5asset can call it
            def _nn_policy(W, g, _fn=policy_fn):
                return _fn(W, g)

            W_path, pi_mat = simulate_one_year_5asset(
                W_start=initial_wealth, goal=goal,
                nn_policy_fn=_nn_policy,
                mu_vec=market_data.mu_ann,
                Omega_mat=market_data.omega,
                r=market_data.r,
                target_vol=vol_scale_target if vol_scale_target else 0.25,
                days=days, rng=path_rng,
            )
            wealth_paths[i]  = W_path
            weight_paths[i]  = pi_mat

        terminal_wealth[i] = W_path[-1]
        goal_hit[i]        = W_path[-1] >= goal

    # Summary statistics
    shortfall = np.maximum(goal - terminal_wealth, 0.0)
    returns   = np.diff(wealth_paths, axis=1) / (wealth_paths[:, :-1] + 1e-8)
    ann_ret   = (terminal_wealth / initial_wealth) - 1.0
    ann_vol   = returns.std(axis=1) * np.sqrt(days)
    sharpe    = (ann_ret - market_data.r) / (ann_vol + 1e-8)

    return {
        "terminal_wealth":        terminal_wealth,
        "goal_hit":               goal_hit,
        "wealth_paths":           wealth_paths,
        "weight_paths":           weight_paths,
        "target_hit_prob":        float(goal_hit.mean()),
        "mean_terminal_wealth":   float(terminal_wealth.mean()),
        "median_terminal_wealth": float(np.median(terminal_wealth)),
        "shortfall_mean":         float(shortfall.mean()),
        "sharpe_mean":            float(sharpe.mean()),
    }


def fd_policy_1d(w_grid, pi_grid, d=-5.0, u=3.0):
    """
    Return a 1-D FD policy callable of the form policy_fn(w_norm, tau).
    Compatible with evaluate_policy_mc and simulate_one_year_1asset.

    w_norm = W / goal  (so goal=1 is assumed — call with normalised wealth).
    """
    from fd_core import make_fd_policy
    policy = make_fd_policy(w_grid, pi_grid, d=d, u=u)
    return policy


def browne_policy_1d(mu, r, sigma, d=-5.0, u=3.0):
    """
    Analytical Browne (1995) goal-reaching policy.
    policy_fn(w_norm, tau) -> scalar pi  OR  (N,) array when w_norm is (N,).
    w_norm = W / goal.
    Accepts both scalar and array inputs for use with vectorised MC runners.
    """
    eta  = mu - r
    sig2 = sigma ** 2

    def _pi(w_norm, tau):
        w_arr  = np.asarray(w_norm, float)
        scalar = w_arr.ndim == 0
        w_arr  = np.atleast_1d(w_arr)
        tau_s  = max(float(tau), 1.0 / 252)
        f      = sigma * np.sqrt(tau_s) + 1e-12
        w_safe = np.maximum(w_arr, 1e-6)
        z      = (np.log(w_safe) + (eta - 0.5 * sig2) * tau_s) / f
        den    = sig2 * (1.0 + z / f)
        raw    = np.where(np.abs(den) > 1e-8, eta / den,
                          np.where(eta > 0, u, d))
        result = np.clip(raw, d, u)
        return float(result[0]) if scalar else result

    return _pi


# ---------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------

def _summary_array(rows):
    if not rows:
        return np.array([], dtype=[])
    dtype = []
    first = rows[0]
    for key, value in first.items():
        if isinstance(value, str):
            width = max(len(str(row[key])) for row in rows)
            dtype.append((key, f"U{max(width, 1)}"))
        elif isinstance(value, (int, np.integer)):
            dtype.append((key, "i8"))
        else:
            dtype.append((key, "f8"))
    data = np.empty(len(rows), dtype=dtype)
    for idx, row in enumerate(rows):
        data[idx] = tuple(row[key] for key in first.keys())
    return data


# ---------------------------------------------------------------
# Main comparison runner
# ---------------------------------------------------------------

def run_real_data_portfolio_comparison(config):
    from comparisons.core.nn_models import ARCHITECTURES as _NP_ARCHS

    raw_dir = ensure_dir(config.results_dir / "raw")
    artifacts_dir = ensure_dir(config.results_dir / "artifacts")
    summary_dir = ensure_dir(config.results_dir / "summary")

    results = []
    summary_rows = []

    for n_assets in config.n_assets_list:
        print(f"\n=== n_assets={n_assets} ===", flush=True)
        market_data = load_portfolio(n_assets, start=config.start_date,
                                     end=config.end_date)

        baselines = [
            ("equal_weight", np.ones(market_data.n) / market_data.n),
            ("max_sharpe", max_sharpe_weights(market_data.mu_ann,
                                              market_data.omega, market_data.r)),
            ("market_cap", market_cap_weights(market_data.tickers)),
        ]

        for seed in config.random_seeds:
            for w0 in config.initial_wealth_levels:

                # Baselines
                for mname, wts in baselines:
                    r = evaluate_static_portfolio(mname, "baseline", market_data,
                                                  wts, w0, config.target_multiplier, seed)
                    results.append(r); summary_rows.append(result_to_summary_row(r))
                    save_run_result(raw_dir / f"{mname}_n{n_assets}_seed{seed}_w{w0:.2f}.npz", r)

                # FD 1-D proxy
                if config.include_fd_benchmark:
                    r = evaluate_fd_benchmark(market_data, config, w0, seed)
                    results.append(r); summary_rows.append(result_to_summary_row(r))
                    save_run_result(raw_dir / f"fd_1d_proxy_n{n_assets}_seed{seed}_w{w0:.2f}.npz", r)
                    fd_artifact = r.pop("_fd_artifact", None)
                    if fd_artifact is not None:
                        save_fd_artifact(
                            artifacts_dir / f"{artifact_filename('fd_1d_proxy', n_assets, seed, w0, 'fd_policy')}.npz",
                            fd_artifact["w_grid"],
                            fd_artifact["pi_grid"],
                            fd_artifact["metadata"],
                        )

                # Merton multi-asset FD benchmark
                if config.include_merton_benchmark:
                    r = evaluate_merton_benchmark(market_data, config, w0, seed)
                    results.append(r); summary_rows.append(result_to_summary_row(r))
                    save_run_result(raw_dir / f"fd_merton_multi_n{n_assets}_seed{seed}_w{w0:.2f}.npz", r)

                # Neural methods
                if config.include_nn:
                    for arch in list(config.nn_architectures):
                        if not HAS_TORCH and arch not in _NP_ARCHS:
                            print(f"  [skip] {arch}: requires PyTorch", flush=True)
                            continue
                        print(f"  training {arch} n={n_assets} seed={seed}...", flush=True)
                        arch_kind = TORCH_ARCHITECTURES.get(arch, {}).get('kind', '')
                        if arch_kind in ('es_grpo', 'reinforce_grpo'):
                            r = evaluate_grpo_portfolio(market_data, config, arch, w0, seed)
                        else:
                            r = evaluate_nn_portfolio(market_data, config, arch, w0, seed)
                        results.append(r); summary_rows.append(result_to_summary_row(r))
                        save_run_result(
                            raw_dir / f"{arch}_n{n_assets}_seed{seed}_w{w0:.2f}.npz", r)
                        model_artifact = r.pop("_model_artifact", None)
                        if model_artifact is not None:
                            save_torch_model_artifact(
                                artifacts_dir / f"{artifact_filename(arch, n_assets, seed, w0, 'model')}.pt",
                                model_artifact["model"],
                                model_artifact["metadata"],
                            )

    save_summary_csv(summary_dir / "main_results.csv", summary_rows)
    tables = save_comparison_tables(summary_rows, summary_dir)
    plots = save_plots(summary_rows, config.results_dir / "plots")
    return {"results": results, "summary": _summary_array(summary_rows),
            "tables": tables, "plots": plots}
