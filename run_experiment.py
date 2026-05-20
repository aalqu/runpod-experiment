"""
run_experiment.py
-----------------
Full comparison: FD HJB solver vs all neural network architectures,
across multiple asset counts.

Outputs (all written to RESULTS_DIR)
─────────────────────────────────────
results/
├── summary.csv              ← one row per method × n_assets × seed
├── goal_probability.png     ← P(W_T ≥ goal) by method and asset count
├── terminal_wealth.png      ← distribution of W_T by method
├── train_time.png           ← solve / train time by method
├── convergence/
│   └── <arch>_n<k>.png     ← E[U] training curve per architecture
├── weights/
│   └── <method>_n<k>_<ticker>.png  ← weight time-series per ticker
└── weight_dist/
    └── <method>_n<k>.png   ← box-plot of weights across assets

Usage
─────
    cd "/path/to/Claude Code"
    python run_experiment.py              # full run
    python run_experiment.py --quick      # fast sanity check (1 seed, tiny NN)
    python run_experiment.py --no-nn      # FD + baselines only
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "comparisons"))

RESULTS_DIR = ROOT / "results" / "experiment"

# ── Configuration ─────────────────────────────────────────────────────────────

_BASE_ARCHS = [
    "nn_mlp_small",
    "nn_mlp_small_v2",
    "nn_mlp_deep",
    "nn_policy_net",
    "nn_policy_net_v2",
    "nn_ste_goalreach",
    "nn_ste_goalreach_v2",
    "nn_digital_hedge",
    "nn_policy_long_only",
    "nn_ste_long_only",
    "nn_digital_hedge_long_only",
    "nn_historical_replay",
    "nn_historical_replay_long_only",
    "deep_bsde",
    "deep_bsde_v2",
    "pinn",
    "actor_critic",
    "actor_critic_v2",
    "lstm",
    "lstm_v2",
    "transformer",
    "transformer_v2",
]

_GRPO_ARCHS = ["es_grpo", "es_grpo_long_only"]


def _default_archs(include_grpo: bool = False) -> list:
    """Return the default architecture list, optionally appending ES-GRPO."""
    archs = list(_BASE_ARCHS)
    if include_grpo:
        archs.extend(_GRPO_ARCHS)
    return archs


def make_config(quick=False, no_nn=False, n_assets_list=None, seeds=None,
                nn_archs=None, include_grpo=False, in_sample=False):
    """Build a BenchmarkConfig with sensible defaults."""
    from comparisons.core.config import BenchmarkConfig

    return BenchmarkConfig(
        n_assets_list  = n_assets_list or ([1, 5] if quick else [1, 5, 10, 20]),
        random_seeds   = seeds         or ([1]  if quick else [1, 2, 3, 4, 5]),
        start_date     = "2015-01-01",
        end_date       = "2024-12-31",
        calib_start     = "2015-01-01",
        calib_end       = "2020-12-31",
        eval_start      = "2021-01-01",
        eval_end        = "2024-12-31",
        initial_wealth_levels = [1.0],
        target_multiplier     = 1.10,

        # FD solver
        fd_wealth_max = 2.5,
        fd_nw         = 120,
        fd_nt         = 80,

        # NN training
        include_nn        = not no_nn,
        nn_architectures  = nn_archs or _default_archs(include_grpo),
        nn_paths          = 64  if quick else 512,
        nn_iters          = 10  if quick else 200,
        nn_steps          = 16  if quick else 40,
        nn_pretrain_iters = 0,   # pretrain does nothing — always 0
        nn_antithetic     = True,
        nn_p_curriculum   = 0.30,
        nn_patience       = 30  if quick else 60,
        nn_horizon_years  = 1.0,

        # ES-GRPO hyper-params (used when include_grpo=True)
        es_grpo_G        = 16 if quick else 32,
        es_grpo_sigma    = 0.030,
        es_grpo_lr       = 3e-3,
        es_grpo_iters    = 10 if quick else 300,
        es_grpo_paths    = 32 if quick else 256,
        es_grpo_pretrain = 0  if quick else 100,
        es_grpo_patience = 30 if quick else 60,

        # Leverage
        weight_lower_bound = -5.0,
        weight_upper_bound =  3.0,
        max_long_leverage  =  3.0,
        max_short_leverage =  5.0,

        # Single goal — no multiplier sweep in new experiment design
        goal_multipliers = [1.10],

        # In-sample: skip calib/eval split when requested
        use_calib_split = not in_sample,

        results_dir = RESULTS_DIR,
    )


# ── OOS experiment helpers ────────────────────────────────────────────────────

CALIB_START = "2015-01-01"
CALIB_END   = "2020-12-31"

# Main tier: 5 key architectures + v2 counterparts for the paper's primary table
MAIN_ARCHS = [
    "fd_nd",
    "nn_mlp_small", "nn_mlp_small_v2",
    "deep_bsde", "deep_bsde_v2",
    "nn_digital_hedge",
    "nn_historical_replay",
]

# Appendix tier: all architectures including v2 enriched-feature variants
APPENDIX_ARCHS = [
    "fd_nd",
    "nn_mlp_small", "nn_mlp_small_v2",
    "nn_mlp_deep",
    "nn_policy_net", "nn_policy_net_v2",
    "nn_ste_goalreach", "nn_ste_goalreach_v2",
    "nn_digital_hedge",
    "nn_policy_long_only", "nn_ste_long_only", "nn_digital_hedge_long_only",
    "nn_historical_replay", "nn_historical_replay_long_only",
    "deep_bsde", "deep_bsde_v2",
    "pinn",
    "actor_critic", "actor_critic_v2",
    "lstm", "lstm_v2",
    "transformer", "transformer_v2",
]


def make_oos_windows(T: int):
    """
    Return list of (eval_start, eval_end) strings for T-year sliding OOS windows.

    Fixed calibration: 2015-01-01 → 2020-12-31
    Sliding eval start: 2021, 2022, ... until eval_end > 2024

    T=1 → 4 windows  (2021, 2022, 2023, 2024)
    T=2 → 3 windows  (2021-22, 2022-23, 2023-24)
    T=3 → 2 windows  (2021-23, 2022-24)
    """
    windows = []
    for year_s in range(2021, 2025):
        year_e = year_s + T - 1
        if year_e > 2024:
            break
        windows.append((f"{year_s}-01-01", f"{year_e}-12-31"))
    return windows


def make_config_for_horizon(T: int, tier: str = 'main', quick: bool = False,
                            include_grpo: bool = False):
    """
    Build BenchmarkConfig for horizon T ∈ {1, 2, 3}.

    Parameters
    ----------
    T     : investment horizon in years
    tier  : 'main'     → 5 archs, n=[1, 20, 100], 2 seeds
            'appendix' → 15 archs, n=[1, 5, 10, 20], 2 seeds
    quick : bool       → minimal iters for smoke test

    Scaling rules
    -------------
    fd_nt        = 80 × T
    nn_steps     = 40 × T
    nn_iters     = 200 × T  (or 10 × T if quick)
    nn_horizon_years = T    (→ tau normalisation + _gbm_mc_eval days = 252*T)
    goal             = 1.10 ** T  (annualised 10 % per year)
    pretrain         = 0  (pretrain does nothing per empirical observation)
    """
    from comparisons.core.config import BenchmarkConfig

    goal_mult = 1.10 ** T

    if tier == 'main':
        n_list = [1, 20, 100]   # 100 = synthetic GBM — answers tractability question
        archs  = list(MAIN_ARCHS)
    else:
        n_list = [1, 5, 10, 20]
        archs  = list(APPENDIX_ARCHS)
    if include_grpo:
        archs.extend(_GRPO_ARCHS)

    return BenchmarkConfig(
        n_assets_list   = n_list,
        random_seeds    = [1, 2],
        start_date      = CALIB_START,
        end_date        = "2024-12-31",
        use_calib_split = True,
        calib_start     = CALIB_START,
        calib_end       = CALIB_END,
        eval_start      = "2021-01-01",  # overridden per window in run_oos_experiment
        eval_end        = "2024-12-31",  # overridden per window

        initial_wealth_levels = [1.0],
        target_multiplier     = goal_mult,
        goal_multipliers      = [goal_mult],  # single goal, no sweep

        # FD solver — scaled with T
        fd_wealth_max = 2.5,
        fd_nw         = 120,
        fd_nt         = int(80 * T),

        # NN training — scaled with T
        include_nn        = True,
        nn_architectures  = archs,
        nn_paths          = 64  if quick else 512,
        nn_iters          = int(10 * T) if quick else int(200 * T),
        nn_steps          = int(40 * T),
        nn_pretrain_iters = 0,
        nn_antithetic     = True,
        nn_p_curriculum   = 0.30,
        nn_patience       = 30  if quick else 60,
        nn_horizon_years  = float(T),   # used for tau normalisation & MC horizon

        # ES-GRPO hyper-params (scaled with T, active when include_grpo=True)
        es_grpo_G        = 16 if quick else 32,
        es_grpo_sigma    = 0.030,
        es_grpo_lr       = 3e-3,
        es_grpo_iters    = int(10 * T) if quick else int(300 * T),
        es_grpo_paths    = 32 if quick else 256,
        es_grpo_pretrain = 0  if quick else int(100 * T),
        es_grpo_patience = 30 if quick else 60,

        # Leverage
        weight_lower_bound = -5.0,
        weight_upper_bound =  3.0,
        max_long_leverage  =  3.0,
        max_short_leverage =  5.0,

        results_dir = RESULTS_DIR,
    )


def run_oos_experiment(
    horizons=(1, 2, 3),
    tier: str = 'main',
    quick: bool = False,
    device=None,
    resume: bool = False,
    compile_model: bool = False,
    include_grpo: bool = False,
):
    """
    Run rolling OOS evaluation for each T in horizons.

    For each T, iterates over make_oos_windows(T), copies the base config with
    the correct eval window, and calls run_all().  Each result dict is tagged
    with T_horizon, window_idx, eval_start, eval_end.

    Returns
    -------
    df        : combined summary DataFrame (all T, all windows)
    histories : merged training-curve dict keyed by 'T{T}_w{w+1}_{arch}_...'
    """
    import copy

    all_results   = []
    all_histories = {}

    for T in horizons:
        windows     = make_oos_windows(T)
        config_base = make_config_for_horizon(T, tier=tier, quick=quick,
                                              include_grpo=include_grpo)

        for win_idx, (eval_start, eval_end) in enumerate(windows):
            print(f"\n{'#'*60}")
            print(f"  T={T}yr  window {win_idx+1}/{len(windows)}  "
                  f"eval: {eval_start} → {eval_end}")
            print(f"{'#'*60}")

            cfg             = copy.copy(config_base)
            cfg.eval_start  = eval_start
            cfg.eval_end    = eval_end
            cfg.results_dir = RESULTS_DIR / f"T{T}_w{win_idx+1}"
            cfg.results_dir.mkdir(parents=True, exist_ok=True)

            results, hist = run_all(
                cfg,
                device        = device,
                resume        = resume,
                compile_model = compile_model,
            )

            # Tag each result with window metadata for downstream grouping
            for r in results:
                r.update({
                    "T_horizon"  : T,
                    "window_idx" : win_idx,
                    "eval_start" : eval_start,
                    "eval_end"   : eval_end,
                })
            all_results.extend(results)

            prefix = f"T{T}_w{win_idx + 1}"
            for k, v in hist.items():
                all_histories[f"{prefix}_{k}"] = v

    df = build_summary(all_results)
    return df, all_histories


# ── Data & evaluation imports ─────────────────────────────────────────────────

def _imports():
    from real_data_loader import load_portfolio, load_portfolio_split
    from comparisons.core.evaluation import (
        evaluate_fd_benchmark,
        evaluate_nn_portfolio,
        evaluate_merton_benchmark,
        apply_leverage_constraint,
    )
    return (load_portfolio, load_portfolio_split,
            evaluate_fd_benchmark, evaluate_nn_portfolio,
            evaluate_merton_benchmark, apply_leverage_constraint)


# ── MC evaluation helpers ─────────────────────────────────────────────────────

def _gbm_mc_eval(policy_fn, market_data, initial_wealth, target_multiplier,
                 n_mc=2000, seed=42, T_horizon: float = 1.0):
    """
    Evaluate policy_fn via Monte Carlo under calibrated GBM.

    policy_fn interface: policy_fn(w_norm, tau) -> weights (n,)
    where w_norm = W / goal  and  tau = time remaining (years).
    Works for both 1-D scalar returns and n-D vector returns.

    T_horizon: investment horizon in years — scales the number of MC steps.

    Returns a dict with MC metrics.
    """
    n      = market_data.n
    goal   = initial_wealth * target_multiplier
    days   = int(T_horizon * 252)   # scale MC horizon with T (was hardcoded 252)
    dt     = 1.0 / 252              # daily step always 1/252; days controls total steps
    r      = market_data.r
    mu     = np.asarray(market_data.mu_ann, float)
    omega  = np.asarray(market_data.omega, float)
    L_chol = np.linalg.cholesky(omega + 1e-10 * np.eye(n))
    rng    = np.random.default_rng(seed)

    terminal_wealth = np.empty(n_mc)
    goal_hit        = np.empty(n_mc, dtype=bool)
    all_weights     = np.empty((n_mc, days, n))

    Z_all = rng.standard_normal((n_mc, days, n))
    log_ret_mat = ((mu - 0.5 * np.diag(omega)) * dt
                   + (Z_all @ L_chol.T) * np.sqrt(dt))   # (n_mc, days, n)

    r_daily = r * dt
    for i in range(n_mc):
        W = float(initial_wealth)
        for t in range(days):
            tau  = (days - t) * dt          # time-to-horizon
            w_norm = W / goal
            pi   = np.asarray(policy_fn(w_norm, tau), dtype=float).ravel()
            all_weights[i, t] = pi
            gross = np.exp(log_ret_mat[i, t]) - 1.0
            excess = float(np.dot(pi, gross - r_daily))
            W = max(W * (1.0 + r_daily + excess), 1e-6)
        terminal_wealth[i] = W
        goal_hit[i]        = W >= goal

    shortfall  = np.maximum(goal - terminal_wealth, 0.0)
    pct        = np.percentile(terminal_wealth, [5, 25, 50, 75, 95])
    gross_lev  = np.abs(all_weights).sum(axis=2)  # (n_mc, days)

    # Bootstrap 95% CI on goal probability
    boot_probs = np.array([
        rng.choice(goal_hit, size=len(goal_hit), replace=True).mean()
        for _ in range(500)
    ])

    return {
        "mc_goal_prob":           float(goal_hit.mean()),
        "mc_goal_prob_ci_lo":     float(np.percentile(boot_probs, 2.5)),
        "mc_goal_prob_ci_hi":     float(np.percentile(boot_probs, 97.5)),
        "mc_shortfall_mean":      float(shortfall.mean()),
        "mc_mean_wealth":         float(terminal_wealth.mean()),
        "mc_median_wealth":       float(pct[2]),
        "mc_wealth_p05":          float(pct[0]),
        "mc_wealth_p25":          float(pct[1]),
        "mc_wealth_p75":          float(pct[3]),
        "mc_wealth_p95":          float(pct[4]),
        "mc_mean_gross_leverage": float(gross_lev.mean()),
    }


def _extract_policy_fn(result, market_data, config):
    """
    Reconstruct a policy_fn(w_norm, tau) -> weights callable from a result dict.
    Used to feed _gbm_mc_eval without re-running the solver/trainer.
    """
    from fd_core import make_fd_policy_time_aware, make_fd_policy_nd_time_aware

    family = result.get("method_family", "")
    n      = int(result.get("n_assets", market_data.n))

    if family == "fd":
        art = result.get("_fd_artifact", {})
        w_grid   = art.get("w_grid")
        Pi_path  = art.get("Pi_path")
        tau_path = art.get("tau_path")
        if w_grid is None or Pi_path is None:
            return None
        d, u = config.weight_lower_bound, config.weight_upper_bound
        if n == 1:
            return make_fd_policy_time_aware(w_grid, Pi_path, tau_path, d=d, u=u)
        return make_fd_policy_nd_time_aware(w_grid, Pi_path, tau_path, d=d, u=u)

    if family == "nn":
        model_art = result.get("_model_artifact")
        if model_art is None:
            return None
        try:
            from comparisons.core.torch_nn_models import policy_weights as torch_pw
            net = model_art["model"]
            meta = model_art.get("metadata", {})
            n_steps = meta.get("n_steps", config.nn_steps)
            goal_val = float(result.get("target_wealth",
                                         config.target_multiplier))
            total_steps = 252   # backtest horizon in days

            def nn_policy(w_norm, tau):
                W = float(w_norm) * goal_val
                step_idx = max(0, int(round(total_steps * (1.0 - tau))))
                weights  = np.asarray(
                    torch_pw(net, W, goal_val,
                             step_idx=step_idx,
                             total_steps=total_steps),
                    dtype=float)
                from comparisons.core.evaluation import apply_leverage_constraint
                return apply_leverage_constraint(
                    weights.ravel(), config.weight_lower_bound,
                    config.weight_upper_bound,
                    config.max_long_leverage, config.max_short_leverage)
            return nn_policy
        except Exception:
            return None

    return None


# ── Single-run orchestrator ───────────────────────────────────────────────────

def _ckpt_path(results_dir, method, n, seed, gm):
    """Return the per-run checkpoint path (used for --resume)."""
    safe = method.replace("/", "_")
    return results_dir / "checkpoints" / f"{safe}_n{n}_s{seed}_gm{gm:.2f}.npz"


def _save_ckpt(results_dir, res, n, seed, gm):
    """Persist a lightweight checkpoint so --resume can skip completed runs."""
    path = _ckpt_path(results_dir, res["method_name"], n, seed, gm)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        method_name   = np.array(res["method_name"]),
        goal_hit      = np.array(res["goal_hit"]),
        wealth_path   = np.array(res.get("wealth_path", [])),
        train_time    = np.array(res.get("train_time_sec", 0.0)),
        solve_time    = np.array(res.get("solve_time_sec", 0.0)),
    )


def run_all(config, device=None, resume=False, compile_model=False):
    """
    Run every method for every (n_assets, seed) combination.

    Parameters
    ----------
    config        : BenchmarkConfig
    device        : str | None  — PyTorch device string ('cuda', 'cpu', …)
    resume        : bool        — skip runs that already have a checkpoint
    compile_model : bool        — pass compile=True to eval_nn (torch.compile)

    Returns
    -------
    results  : list[dict]  — one result dict per (method, n_assets, seed)
    histories: dict        — {key: {"train", "val", "val_iters", "test_u"}}
                             keyed by f"{arch}_n{n}_s{seed}_gm{gm:.2f}"
    """
    T_horizon = float(getattr(config, 'nn_horizon_years', 1.0))
    (load_portfolio, load_portfolio_split,
     eval_fd, eval_nn, eval_merton, _) = _imports()

    results   = []
    histories = {}    # train/val/test curves per arch×n×seed

    for n in config.n_assets_list:
        print(f"\n{'='*60}")
        print(f"  n_assets = {n}")
        print(f"{'='*60}")

        use_split = getattr(config, 'use_calib_split', False)
        if use_split:
            mkt, eval_mkt = load_portfolio_split(
                n,
                calib_start = getattr(config, 'calib_start', "2015-01-01"),
                calib_end   = getattr(config, 'calib_end',   "2020-12-31"),
                eval_start  = getattr(config, 'eval_start',  "2021-01-01"),
                eval_end    = getattr(config, 'eval_end',    "2024-12-31"),
            )
            print(f"  [SPLIT] calib: {mkt.dates[0]}→{mkt.dates[-1]} ({len(mkt.dates)} days)  "
                  f"eval: {eval_mkt.dates[0]}→{eval_mkt.dates[-1]} ({len(eval_mkt.dates)} days)")
        else:
            mkt = load_portfolio(n, start=config.start_date, end=config.end_date)
            eval_mkt = mkt

        print(f"  Tickers : {list(mkt.tickers)}")
        print(f"  Returns : {[f'{m:.1%}' for m in mkt.mu_ann]}")
        print(f"  Dates   : {mkt.dates[0]} → {mkt.dates[-1]}  ({len(mkt.dates)} days)")

        goal_multipliers = getattr(config, 'goal_multipliers', [config.target_multiplier])
        import copy

        # ── FD and Merton are deterministic (no seed dependency) ─────────────
        # Solve once per (n_assets, goal_mult) and reuse the result for every
        # seed.  This eliminates len(seeds) − 1 redundant FD solves per goal_mult
        # (50 solves → 10 for 5 seeds × 5 goal_mults).
        #
        # The result dict is shallow-copied per seed with only the "seed" field
        # updated so that downstream groupby(["method","n_assets","seed"]) works.
        fd_cache      = {}   # gm → result dict (seed-agnostic)
        merton_cache  = {}   # gm → result dict (seed-agnostic)

        for gm in goal_multipliers:
            cfg_gm = copy.copy(config)
            cfg_gm.target_multiplier = gm

            if config.include_fd_benchmark:
                _ck = _ckpt_path(config.results_dir, "fd_nd", n, 0, gm)
                if resume and _ck.exists():
                    print(f"  [FD n={n} goal={gm:.2f}] skipped (checkpoint exists)")
                else:
                    print(f"  [FD n={n} goal={gm:.2f}] solving HJB ...", end=" ", flush=True)
                    t0  = time.perf_counter()
                    res = eval_fd(mkt, cfg_gm, initial_wealth=1.0, seed=0,
                                  eval_market_data=eval_mkt if use_split else None)
                    # MC evaluation under calibrated GBM (primary metric)
                    _pfn = _extract_policy_fn(res, mkt, cfg_gm)
                    if _pfn is not None:
                        mc = _gbm_mc_eval(_pfn, mkt, 1.0, gm, n_mc=2000, seed=0,
                                          T_horizon=T_horizon)
                        res = {**res, **mc}
                    print(f"done ({time.perf_counter()-t0:.1f}s)  "
                          f"goal_hit={res['goal_hit'][0]}  "
                          f"mc_P(goal)={res.get('mc_goal_prob', float('nan')):.3f}")
                    fd_cache[gm] = {**res, "n_assets": n, "goal_mult": gm}

            if config.include_merton_benchmark:
                _ck = _ckpt_path(config.results_dir, "merton", n, 0, gm)
                if resume and _ck.exists():
                    print(f"  [Merton n={n} goal={gm:.2f}] skipped (checkpoint exists)")
                else:
                    print(f"  [Merton n={n} goal={gm:.2f}] ...", end=" ", flush=True)
                    res = eval_merton(mkt, cfg_gm, initial_wealth=1.0, seed=0)
                    print(f"done  goal_hit={res['goal_hit'][0]}")
                    merton_cache[gm] = {**res, "n_assets": n, "goal_mult": gm}

        for seed in config.random_seeds:
            print(f"\n  seed={seed}")

            for gm in goal_multipliers:
                cfg_gm = copy.copy(config)
                cfg_gm.target_multiplier = gm

                # Stamp cached FD/Merton results with this seed (shallow copy)
                if gm in fd_cache:
                    res_fd = {**fd_cache[gm], "seed": seed}
                    results.append(res_fd)
                    _save_ckpt(config.results_dir, res_fd, n, seed, gm)

                if gm in merton_cache:
                    res_m = {**merton_cache[gm], "seed": seed}
                    results.append(res_m)
                    _save_ckpt(config.results_dir, res_m, n, seed, gm)

                if config.include_nn:
                    for arch in config.nn_architectures:
                        _ck = _ckpt_path(config.results_dir, arch, n, seed, gm)
                        if resume and _ck.exists():
                            print(f"    [{arch} goal={gm:.2f}] skipped (checkpoint exists)")
                            continue
                        print(f"    [{arch} goal={gm:.2f}] training ...", end=" ", flush=True)
                        t0 = time.perf_counter()
                        try:
                            # Pass device and compile_model through if eval_nn accepts them
                            _nn_kwargs = dict(
                                architecture_name=arch,
                                initial_wealth=1.0,
                                seed=seed,
                                eval_market_data=eval_mkt if use_split else None,
                            )
                            if device is not None:
                                _nn_kwargs["device"] = device
                            if compile_model:
                                _nn_kwargs["compile_model"] = True
                            res = eval_nn(mkt, cfg_gm, **_nn_kwargs)
                            # MC evaluation under calibrated GBM (primary metric)
                            _pfn = _extract_policy_fn(res, mkt, cfg_gm)
                            if _pfn is not None:
                                mc = _gbm_mc_eval(_pfn, mkt, 1.0, gm,
                                                  n_mc=2000, seed=seed,
                                                  T_horizon=T_horizon)
                                res = {**res, **mc}
                            elapsed = time.perf_counter() - t0
                            test_u  = res.get("test_u", float("nan"))
                            print(f"done ({elapsed:.1f}s)  "
                                  f"goal_hit={res['goal_hit'][0]}  "
                                  f"mc_P(goal)={res.get('mc_goal_prob', float('nan')):.3f}  "
                                  f"test_E[U]={test_u:.4f}")
                            res = {**res, "n_assets": n, "seed": seed, "goal_mult": gm}
                            results.append(res)
                            _save_ckpt(config.results_dir, res, n, seed, gm)

                            key = f"{arch}_n{n}_s{seed}_gm{gm:.2f}"
                            histories[key] = {
                                "arch"      : arch,
                                "n_assets"  : n,
                                "seed"      : seed,
                                "goal_mult" : gm,
                                "train"     : res.get("loss_history", []),
                                "val"       : res.get("val_history",   []),
                                "val_iters" : res.get("val_iters",     []),
                                "test_u"    : test_u,
                            }

                        except Exception as exc:
                            import traceback
                            print(f"FAILED: {exc}")
                            if not resume:
                                traceback.print_exc()

    return results, histories


# ── Summary table ─────────────────────────────────────────────────────────────

def build_summary(results):
    """
    Build a tidy DataFrame with one row per (method, n_assets, seed).

    Historical-backtest columns (single-path, robustness check only):
      hist_goal_hit, hist_terminal_wealth, hist_shortfall, hist_max_drawdown

    MC evaluation columns (primary metrics, N=2000 GBM paths):
      mc_goal_prob, mc_goal_prob_ci_lo, mc_goal_prob_ci_hi,
      mc_shortfall_mean, mc_mean_wealth, mc_median_wealth,
      mc_wealth_p05, mc_wealth_p95, mc_mean_gross_leverage

    Weight / leverage metrics (Table 3):
      mean_gross_leverage, max_gross_leverage,
      mean_net_exposure, max_single_name_weight, turnover,
      wealth_vol

    Other:
      test_u, train_time_sec, solve_time_sec, nn_param_count
    """
    rows = []
    _nan = float("nan")
    for r in results:
        wp      = np.asarray(r["wealth_path"])
        wt_path = np.asarray(r["weight_path"])
        if wt_path.ndim == 1:
            wt_path = wt_path[:, None]
        goal    = r.get("target_wealth", 1.1)

        # Weight metrics
        gross_lev  = np.abs(wt_path).sum(axis=1)
        net_exp    = wt_path.sum(axis=1)
        max_single = np.abs(wt_path).max(axis=1).mean()
        turnover   = float(np.abs(np.diff(wt_path, axis=0)).sum()) if len(wt_path) > 1 else 0.0

        rows.append({
            # Identity
            "method"               : r["method_name"],
            "family"               : r.get("method_family", "?"),
            "n_assets"             : r.get("n_assets", 0),
            "seed"                 : r.get("seed", 0),
            "goal_mult"            : float(r.get("goal_mult", 1.10)),
            # Historical backtest (single path — robustness check)
            "hist_goal_hit"        : bool(r["goal_hit"][0]),
            "hist_terminal_wealth" : float(wp[-1]),
            "hist_shortfall"       : float(max(goal - wp[-1], 0.0)),
            "hist_max_drawdown"    : float(np.min(r.get("drawdown_path", [0]))),
            # Kept for backward compat with old plots
            "goal_probability"     : float(r.get("mc_goal_prob", float(r["goal_hit"][0]))),
            "terminal_wealth"      : float(wp[-1]),
            # MC primary metrics
            "mc_goal_prob"         : float(r.get("mc_goal_prob",      _nan)),
            "mc_goal_prob_ci_lo"   : float(r.get("mc_goal_prob_ci_lo", _nan)),
            "mc_goal_prob_ci_hi"   : float(r.get("mc_goal_prob_ci_hi", _nan)),
            "mc_shortfall_mean"    : float(r.get("mc_shortfall_mean",  _nan)),
            "mc_mean_wealth"       : float(r.get("mc_mean_wealth",     _nan)),
            "mc_median_wealth"     : float(r.get("mc_median_wealth",   _nan)),
            "mc_wealth_p05"        : float(r.get("mc_wealth_p05",      _nan)),
            "mc_wealth_p25"        : float(r.get("mc_wealth_p25",      _nan)),
            "mc_wealth_p75"        : float(r.get("mc_wealth_p75",      _nan)),
            "mc_wealth_p95"        : float(r.get("mc_wealth_p95",      _nan)),
            # Weight / leverage (Table 3)
            "mean_gross_leverage"  : float(np.mean(gross_lev)),
            "max_gross_leverage"   : float(np.max(gross_lev)),
            "mean_net_exposure"    : float(np.mean(net_exp)),
            "max_single_name_weight": float(max_single),
            "turnover"             : float(turnover),
            "wealth_vol"           : float(np.std(np.diff(np.log(wp + 1e-8))) * np.sqrt(252)),
            # NN training
            "test_u"               : float(r.get("test_u", _nan)),
            "train_time_sec"       : float(r.get("train_time_sec", 0.0)),
            "solve_time_sec"       : float(r.get("solve_time_sec", 0.0)),
            "nn_param_count"       : int(r.get("nn_param_count", 0)),
        })
    return pd.DataFrame(rows)


# ── FD / Merton deduplication ─────────────────────────────────────────────────

def deduplicate_deterministic_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    FD (and Merton) solvers are deterministic: `run_all` shallow-copies the
    single result once per seed so that downstream ``groupby`` on seed works,
    but this inflates the row count for those methods.

    §4 of EXPERIMENT_REVIEW.md: the seed dimension is meaningful only for
    stochastic NN training — for FD/Merton it is a fake replicate.

    This helper keeps ONE row per (method, family, n_assets, goal_mult) for
    deterministic families ('fd', 'merton') and leaves all NN rows untouched.

    Use this before computing seed-aggregated statistics (mean, std, CI) to
    avoid triple-counting FD in tables and plots.

    Parameters
    ----------
    df : DataFrame produced by build_summary()

    Returns
    -------
    df_deduped : copy with duplicate deterministic rows removed
    """
    _deterministic_families = {'fd', 'merton'}
    deduped_rows = []
    seen_det_keys: set = set()

    for _, row in df.iterrows():
        fam = str(row.get('family', ''))
        if fam in _deterministic_families:
            key = (row['method'], row['n_assets'], float(row.get('goal_mult', 0.0)))
            if key in seen_det_keys:
                continue    # skip duplicate — keep first seed only
            seen_det_keys.add(key)
        deduped_rows.append(row)

    return pd.DataFrame(deduped_rows).reset_index(drop=True)


# ── Table 2: Value-function error ‖V_NN − V_FD‖ ──────────────────────────────

def compute_value_function_errors(results, config, n_eval_points=200,
                                   n_mc_per_point=500, tau_eval=0.0):
    """
    Compute Table 2: ‖V_NN(w,τ) − V_FD(w,τ)‖ on the wealth grid at t=0.

    For n=1 only (where the 1-D fd_solve gives the exact viscosity solution).
    V_NN(w0, τ=T) is estimated by running n_mc_per_point MC paths from w0
    and computing E[1{W_T ≥ goal}] under the NN policy.

    Returns a DataFrame with columns:
        method, n_assets, seed, goal_mult,
        vf_err_L1, vf_err_L2, vf_err_Linf,
        vf_err_at_below_goal, vf_err_at_goal, vf_err_at_above_goal
    """
    from fd_core import make_fd_policy_time_aware

    rows = []
    # Build a lookup of FD results at n=1 for each (goal_mult)
    fd_by_gm = {}
    for r in results:
        if r.get("method_family") == "fd" and int(r.get("n_assets", 0)) == 1:
            gm  = float(r.get("goal_mult", 1.1))
            art = r.get("_fd_artifact", {})
            if art.get("w_grid") is not None and art.get("Pi_path") is not None:
                fd_by_gm[gm] = art   # store just once per goal_mult

    if not fd_by_gm:
        return pd.DataFrame()

    rng = np.random.default_rng(1337)

    for r in results:
        if r.get("method_family") != "nn":
            continue
        if int(r.get("n_assets", 0)) != 1:
            continue

        gm  = float(r.get("goal_mult", 1.1))
        art = fd_by_gm.get(gm)
        if art is None:
            continue

        # FD value function on its own wealth grid (already normalised by goal)
        w_fd   = np.asarray(art["w_grid"])   # (Nw+1,)
        V_fd   = None
        # V_fd is not directly stored, but we can recover it from the artifact
        # if we re-run; for now, skip architectures where fd V is unavailable.
        # Instead estimate V_FD via MC as well — both get the same MC estimator.
        # Build eval grid (same range as FD solver)
        goal   = 1.0 * gm
        w_eval = np.linspace(0.05, min(float(config.fd_wealth_max), 2.5), n_eval_points)

        # FD policy  (time-aware)
        fd_policy = make_fd_policy_time_aware(
            art["w_grid"], art["Pi_path"], art["tau_path"],
            d=config.weight_lower_bound, u=config.weight_upper_bound)

        # NN policy
        nn_policy = _extract_policy_fn(r, None, config)
        if nn_policy is None:
            continue

        # Estimate V at each w_eval by MC (shared noise for variance reduction)
        n    = 1
        days = 252
        dt   = 1.0 / days
        mu1  = float(r.get("_mu1", 0.08))    # fallback if not stored
        sig1 = float(r.get("_sig1", 0.20))
        r_f  = float(config.__dict__.get("r", 0.03))

        V_fd_mc = np.empty(n_eval_points)
        V_nn_mc = np.empty(n_eval_points)

        Z_shared = rng.standard_normal((n_eval_points, n_mc_per_point, days))

        for k, w0 in enumerate(w_eval):
            w_norm0 = w0 / goal
            for method_idx, (policy_fn, V_out) in enumerate(
                [(fd_policy, V_fd_mc), (nn_policy, V_nn_mc)]
            ):
                hit = 0
                for j in range(n_mc_per_point):
                    W = float(w0)
                    for t in range(days):
                        tau = (days - t) * dt
                        pi  = float(np.asarray(policy_fn(W / goal, tau)).ravel()[0])
                        pi  = float(np.clip(pi, config.weight_lower_bound,
                                            config.weight_upper_bound))
                        log_r = ((mu1 - 0.5 * sig1**2) * dt
                                 + sig1 * Z_shared[k, j, t] * np.sqrt(dt))
                        gross = np.exp(log_r) - 1.0
                        W = max(W * (1.0 + r_f * dt + pi * (gross - r_f * dt)), 1e-6)
                    hit += int(W >= goal)
                V_out[k] = hit / n_mc_per_point

        diff     = np.abs(V_fd_mc - V_nn_mc)
        goal_idx  = np.searchsorted(w_eval, 1.0)    # w_norm ≈ 1
        below_idx = max(0, goal_idx - n_eval_points // 10)
        above_idx = min(n_eval_points - 1, goal_idx + n_eval_points // 10)

        rows.append({
            "method"              : r["method_name"],
            "n_assets"            : 1,
            "seed"                : r.get("seed", 0),
            "goal_mult"           : gm,
            "vf_err_L1"           : float(np.mean(diff)),
            "vf_err_L2"           : float(np.sqrt(np.mean(diff**2))),
            "vf_err_Linf"         : float(np.max(diff)),
            "vf_err_at_below_goal": float(diff[below_idx]),
            "vf_err_at_goal"      : float(diff[goal_idx]),
            "vf_err_at_above_goal": float(diff[above_idx]),
        })

    return pd.DataFrame(rows)


# ── Effect-size table (replaces BH pairwise tests) ───────────────────────────

def compute_effect_size_table(df, n_bootstrap: int = 2000, rng_seed: int = 0):
    """
    Compute Δ = P̄_method − P̄_fd_nd per (n_assets, T_horizon) with 95 %
    bootstrap CI across OOS windows.

    Parameters
    ----------
    df          : summary DataFrame produced by build_summary() for OOS run,
                  must have columns: method, n_assets, T_horizon, window_idx,
                  mc_goal_prob (or goal_probability), seed
    n_bootstrap : number of bootstrap resamples of windows

    Returns
    -------
    DataFrame with columns:
        method, n_assets, T_horizon,
        mean_goal_prob, delta_vs_fd, ci_lo, ci_hi,
        n_windows
    """
    mc_col = "mc_goal_prob" if "mc_goal_prob" in df.columns else "goal_probability"
    if "T_horizon" not in df.columns:
        df = df.copy()
        df["T_horizon"] = 1

    rng = np.random.default_rng(rng_seed)
    rows = []

    for (n, T), grp in df.groupby(["n_assets", "T_horizon"]):
        windows = sorted(grp["window_idx"].unique()) if "window_idx" in grp.columns \
                  else [0]

        # Per-window, per-method mean across seeds
        pivot = (
            grp.groupby(["method", "window_idx"] if "window_idx" in grp.columns
                        else ["method"])[mc_col]
            .mean()
            .unstack("window_idx" if "window_idx" in grp.columns else "method")
        )

        # FD reference row (fd_nd)
        fd_key = "fd_nd"
        if fd_key not in pivot.index:
            continue  # can't compute Δ without FD reference
        fd_by_window = pivot.loc[fd_key].values.astype(float)  # (n_windows,)

        for method in pivot.index:
            m_by_window = pivot.loc[method].values.astype(float)
            delta_by_w  = m_by_window - fd_by_window          # per-window Δ

            mean_gp = float(np.nanmean(m_by_window))
            delta   = float(np.nanmean(delta_by_w))
            n_win   = int(np.sum(~np.isnan(delta_by_w)))

            # Bootstrap CI over windows (resample window indices)
            valid = delta_by_w[~np.isnan(delta_by_w)]
            if len(valid) >= 2:
                boot = np.array([
                    np.mean(rng.choice(valid, size=len(valid), replace=True))
                    for _ in range(n_bootstrap)
                ])
                ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
            else:
                ci_lo = ci_hi = delta   # single window: no CI

            rows.append({
                "method"        : method,
                "n_assets"      : n,
                "T_horizon"     : T,
                "mean_goal_prob": round(mean_gp, 4),
                "delta_vs_fd"   : round(delta,   4),
                "ci_lo"         : round(float(ci_lo), 4),
                "ci_hi"         : round(float(ci_hi), 4),
                "n_windows"     : n_win,
            })

    return pd.DataFrame(rows).sort_values(
        ["T_horizon", "n_assets", "delta_vs_fd"], ascending=[True, True, False]
    ).reset_index(drop=True)


def compute_robustness_score(df):
    """
    Compute robustness = min_window_goal_prob / mean_window_goal_prob per
    (method, n_assets, T_horizon).

    A score near 1.0 means the method performs consistently across windows.
    Values near 0.0 indicate at least one window where performance collapses.

    Returns DataFrame with columns:
        method, n_assets, T_horizon, mean_goal_prob, min_goal_prob, robustness
    """
    mc_col = "mc_goal_prob" if "mc_goal_prob" in df.columns else "goal_probability"
    if "T_horizon" not in df.columns:
        df = df.copy()
        df["T_horizon"] = 1
    if "window_idx" not in df.columns:
        df = df.copy()
        df["window_idx"] = 0

    # Mean across seeds first, then aggregate across windows
    per_win = (
        df.groupby(["method", "n_assets", "T_horizon", "window_idx"])[mc_col]
        .mean()
        .reset_index()
    )

    rows = []
    for (method, n, T), grp in per_win.groupby(["method", "n_assets", "T_horizon"]):
        vals = grp[mc_col].values
        mean_v = float(np.nanmean(vals))
        min_v  = float(np.nanmin(vals))
        score  = float(min_v / mean_v) if mean_v > 1e-6 else 0.0
        rows.append({
            "method"         : method,
            "n_assets"       : n,
            "T_horizon"      : T,
            "mean_goal_prob" : round(mean_v, 4),
            "min_goal_prob"  : round(min_v,  4),
            "robustness"     : round(score,  4),
        })

    return pd.DataFrame(rows).sort_values(
        ["T_horizon", "n_assets", "robustness"], ascending=[True, True, False]
    ).reset_index(drop=True)


# ── Plots ──────────────────────────────────────────────────────────────────────

PALETTE = {
    "fd"       : "#1f77b4",
    "nn"       : "#ff7f0e",
    "baseline" : "#2ca02c",
}

METHOD_ORDER = [
    "fd_1d_proxy", "fd_nd", "fd_merton_multi",
    "nn_mlp_small", "nn_mlp_deep", "nn_policy_net", "nn_ste_goalreach",
    "nn_digital_hedge",
    "nn_policy_long_only", "nn_ste_long_only", "nn_digital_hedge_long_only",
    "nn_historical_replay", "nn_historical_replay_long_only",
    "deep_bsde", "pinn", "actor_critic", "lstm", "transformer",
    "equal_weight", "merton", "market_cap",
]

# Primary architecture shortlist (reviewer-recommended, main comparison table).
# All other architectures are relegated to the appendix.
PRIMARY_ARCHS = {
    "fd_nd",                  # FD HJB benchmark
    "nn_mlp_small",           # MLP baseline
    "deep_bsde",              # Deep BSDE
    "nn_digital_hedge",       # digital-hedge
    "nn_historical_replay",   # historical-replay
    "nn_policy_long_only",    # long-only twin
}


def _sorted_methods(df):
    present = df["method"].unique().tolist()
    return [m for m in METHOD_ORDER if m in present] + \
           [m for m in present if m not in METHOD_ORDER]


def plot_goal_vs_multiplier(df, out_dir):
    """
    Line chart: P(goal) vs goal multiplier for each method, one panel per n_assets.
    Secondary right axis shows mean_gross_leverage at each multiplier per method.
    """
    if "goal_mult" not in df.columns:
        return
    mults = sorted(df["goal_mult"].unique())
    if len(mults) <= 1:
        return

    n_list = sorted(df["n_assets"].unique())
    fig, axes = plt.subplots(1, len(n_list), figsize=(7 * len(n_list), 5), sharey=True)
    if len(n_list) == 1:
        axes = [axes]

    KEY_METHODS = list(PRIMARY_ARCHS) + ["equal_weight", "merton"]

    for ax, n in zip(axes, n_list):
        sub = df[df["n_assets"] == n]
        present = [m for m in KEY_METHODS if m in sub["method"].unique()]
        cmap = plt.get_cmap("tab10")

        ax2 = ax.twinx()
        ax2.set_ylabel("Mean gross leverage", fontsize=8, color="#555")
        ax2.tick_params(axis='y', labelcolor="#555", labelsize=7)

        for i, method in enumerate(present):
            ms    = sub[sub["method"] == method]
            color = cmap(i % 10)

            gp    = ms.groupby("goal_mult")["goal_probability"]
            means = gp.mean().reindex(mults)
            stds  = gp.std().reindex(mults).fillna(0)
            ax.plot(mults, means.values, marker='o', lw=1.8, label=method, color=color)
            ax.fill_between(mults,
                            (means - stds).values.clip(0, 1),
                            (means + stds).values.clip(0, 1),
                            alpha=0.12, color=color)

            # Secondary axis: mean gross leverage per multiplier
            if "mean_gross_leverage" in ms.columns:
                lev_mean = ms.groupby("goal_mult")["mean_gross_leverage"].mean().reindex(mults)
                ax2.plot(mults, lev_mean.values, linestyle='--', lw=0.9,
                         color=color, alpha=0.55, marker='s', ms=3)

        ax.set_xlabel("Goal multiplier (target / initial wealth)")
        ax.set_ylabel("P(reach goal)" if n == n_list[0] else "")
        ax.set_title(f"n_assets = {n}")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(mults)
        ax.set_xticklabels([f"{m:.2f}×" for m in mults])
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle("P(goal) vs Goal Difficulty  (dashed = leverage, right axis)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "goal_vs_multiplier.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def plot_goal_probability(df, out_dir):
    """
    Bar chart: P(goal) by method, one panel per n_assets.
    Secondary right axis shows mean_gross_leverage per method (dot+dashed line).
    Primary methods (PRIMARY_ARCHS) are shown with a star marker on the leverage axis.
    """
    if "goal_mult" in df.columns and len(df["goal_mult"].unique()) > 1:
        df = df[df["goal_mult"] == 1.10]
    n_list = sorted(df["n_assets"].unique())
    fig, axes = plt.subplots(1, len(n_list), figsize=(6 * len(n_list), 5),
                             sharey=True)
    if len(n_list) == 1:
        axes = [axes]

    for ax, n in zip(axes, n_list):
        sub = df[df["n_assets"] == n]
        methods = _sorted_methods(sub)
        means = [sub[sub["method"] == m]["goal_probability"].mean() for m in methods]
        errs  = [sub[sub["method"] == m]["goal_probability"].std()  for m in methods]
        levs  = [sub[sub["method"] == m]["mean_gross_leverage"].mean()
                 if "mean_gross_leverage" in sub.columns else float("nan")
                 for m in methods]
        colors = [PALETTE.get(sub[sub["method"] == m]["family"].iloc[0]
                              if len(sub[sub["method"] == m]) else "baseline",
                              "#9467bd") for m in methods]

        # Hatching for primary vs appendix
        hatches = ["" if m in PRIMARY_ARCHS else "/" for m in methods]

        bars = ax.bar(range(len(methods)), means, yerr=errs, color=colors,
                      alpha=0.85, capsize=4, ecolor="grey")
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        # Secondary axis: mean gross leverage
        ax2 = ax.twinx()
        xs   = [i for i, v in enumerate(levs) if not np.isnan(v)]
        ys   = [v for v in levs if not np.isnan(v)]
        if xs:
            ax2.plot(xs, ys, 'k^--', ms=5, lw=1.0, alpha=0.65,
                     label="mean gross leverage")
            ax2.set_ylabel("Mean gross leverage", fontsize=8, color="#555")
            ax2.tick_params(axis='y', labelcolor="#555", labelsize=7)
            ax2.set_ylim(0, max(ys) * 1.6 if ys else 4)

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=40, ha="right", fontsize=8)
        ax.set_title(f"n_assets = {n}", fontsize=11)
        ax.set_ylabel("P(W_T ≥ goal)" if n == n_list[0] else "")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.axhline(0.5, color="grey", lw=0.8, ls="--")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Goal Probability by Method  (hatched = appendix-only, △ = leverage)",
        fontsize=12,
    )
    fig.tight_layout()
    path = out_dir / "goal_probability.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def plot_terminal_wealth(df, out_dir):
    """
    Box plot of terminal wealth by method, one panel per n_assets.
    Annotates each box with mean_gross_leverage (right axis, triangle markers).
    """
    n_list = sorted(df["n_assets"].unique())
    fig, axes = plt.subplots(1, len(n_list), figsize=(6 * len(n_list), 5),
                             sharey=False)
    if len(n_list) == 1:
        axes = [axes]

    for ax, n in zip(axes, n_list):
        sub = df[df["n_assets"] == n]
        methods = _sorted_methods(sub)
        data = [sub[sub["method"] == m]["terminal_wealth"].values for m in methods]
        levs = [sub[sub["method"] == m]["mean_gross_leverage"].mean()
                if "mean_gross_leverage" in sub.columns else float("nan")
                for m in methods]

        bp = ax.boxplot(data, tick_labels=methods, patch_artist=True,
                        medianprops=dict(color="black", lw=2))

        # Colour boxes by family
        for patch, method in zip(bp["boxes"], methods):
            fam_rows = sub[sub["method"] == method]
            fam = fam_rows["family"].iloc[0] if len(fam_rows) else "baseline"
            patch.set_facecolor(PALETTE.get(fam, "#9467bd"))
            patch.set_alpha(0.7)
            if method not in PRIMARY_ARCHS:
                patch.set_hatch("/")

        ax.set_xticklabels(methods, rotation=40, ha="right", fontsize=8)
        ax.set_title(f"n_assets = {n}")
        ax.set_ylabel("Terminal wealth" if n == n_list[0] else "")
        ax.axhline(1.10, color="red", lw=1.0, ls="--", label="goal (1.10)")
        ax.grid(axis="y", alpha=0.3)

        # Secondary axis: leverage
        ax2 = ax.twinx()
        xs = [i + 1 for i, v in enumerate(levs) if not np.isnan(v)]  # boxplot is 1-indexed
        ys = [v for v in levs if not np.isnan(v)]
        if xs:
            ax2.plot(xs, ys, 'k^--', ms=5, lw=1.0, alpha=0.65)
            ax2.set_ylabel("Mean gross leverage", fontsize=8, color="#555")
            ax2.tick_params(axis='y', labelcolor="#555", labelsize=7)
            ax2.set_ylim(0, max(ys) * 1.6 if ys else 4)

        ax.legend(fontsize=8)

    fig.suptitle(
        "Terminal Wealth Distribution  (hatched = appendix-only, △ = leverage)",
        fontsize=12,
    )
    fig.tight_layout()
    path = out_dir / "terminal_wealth.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def plot_train_time(df, out_dir):
    """Solve/train time vs method, stacked bars."""
    sub = df.groupby("method").agg(
        train=("train_time_sec", "mean"),
        solve=("solve_time_sec", "mean"),
        family=("family", "first"),
    ).reset_index()
    sub = sub.sort_values("train", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(sub))
    ax.bar(x, sub["solve"], label="FD solve", color="#1f77b4", alpha=0.85)
    ax.bar(x, sub["train"], bottom=sub["solve"], label="NN train",
           color="#ff7f0e", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(sub["method"].tolist(), rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Wall-clock seconds")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Training / Solve Time by Method", fontsize=13)
    fig.tight_layout()
    path = out_dir / "train_time.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def plot_convergence(conv, out_dir):
    """E[U] training curves for each NN architecture × n_assets."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, history in conv.items():
        if not history:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history, lw=1.5, color="#ff7f0e")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("E[U]  (goal-reaching)")
        ax.set_title(f"Convergence: {key}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = out_dir / f"{key}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    if conv:
        print(f"  → {out_dir}/ ({len(conv)} curves)")


def plot_weight_paths(results, out_dir):
    """
    Per-ticker weight time-series for each method × n_assets.
    One file per (method, n_assets): all tickers overlaid.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        wt_path = np.asarray(r.get("weight_path", []))
        tickers = np.asarray(r.get("tickers", []))
        if wt_path.ndim != 2 or wt_path.shape[1] == 0:
            continue

        n      = wt_path.shape[1]
        method = r["method_name"]
        n_ast  = r.get("n_assets", n)
        seed   = r.get("seed", 0)

        fig, ax = plt.subplots(figsize=(12, 4))
        for i in range(n):
            label = tickers[i] if i < len(tickers) else f"asset_{i}"
            ax.plot(wt_path[:, i], lw=0.9, label=label, alpha=0.8)

        ax.axhline(0, color="grey", lw=0.6, ls="--")
        ax.set_xlabel("Trading day")
        ax.set_ylabel("Portfolio weight")
        ax.set_title(f"{method}  |  n={n_ast}  |  seed={seed}")
        ax.legend(ncol=min(n, 5), fontsize=7, loc="upper right")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = out_dir / f"{method}_n{n_ast}_s{seed}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    print(f"  → {out_dir}/")


def plot_weight_distribution(results, out_dir):
    """
    Box plot of time-averaged weights per asset, one panel per ticker.
    One file per (method, n_assets, seed).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        wt_path = np.asarray(r.get("weight_path", []))
        tickers = list(r.get("tickers", []))
        if wt_path.ndim != 2 or wt_path.shape[1] == 0:
            continue

        method = r["method_name"]
        n_ast  = r.get("n_assets", wt_path.shape[1])
        seed   = r.get("seed", 0)

        fig, ax = plt.subplots(figsize=(max(8, wt_path.shape[1] * 0.9), 4))
        ax.boxplot(
            [wt_path[:, i] for i in range(wt_path.shape[1])],
            tick_labels=tickers or [f"A{i}" for i in range(wt_path.shape[1])],
            patch_artist=True,
            medianprops=dict(color="black", lw=2),
        )
        ax.axhline(0, color="grey", lw=0.7, ls="--")
        ax.set_ylabel("Weight")
        ax.set_title(f"Weight distribution — {method}  n={n_ast}  seed={seed}")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path = out_dir / f"{method}_n{n_ast}_s{seed}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    print(f"  → {out_dir}/")


def plot_training_curves(histories, out_dir):
    """
    For each NN run, plot training E[U], validation E[U], and the scalar
    test E[U] (horizontal dashed line) on the same axes.

    One file per (arch × n_assets): overlays all seeds with shading.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group by (arch, n_assets)
    groups = {}
    for key, h in histories.items():
        gk = (h["arch"], h["n_assets"])
        groups.setdefault(gk, []).append(h)

    for (arch, n), runs in groups.items():
        fig, ax = plt.subplots(figsize=(9, 4))

        for h in runs:
            seed  = h["seed"]
            train = h["train"]
            val   = h["val"]
            vit   = h["val_iters"]
            test  = h["test_u"]

            if not train:
                continue

            iters = range(len(train))
            ax.plot(iters, train, lw=1.2, alpha=0.7,
                    color="#ff7f0e", label=f"train (s={seed})" if seed == runs[0]["seed"] else "_")
            if val and vit:
                ax.plot(vit, val, "o--", ms=4, lw=1.0, alpha=0.85,
                        color="#1f77b4", label=f"val (s={seed})" if seed == runs[0]["seed"] else "_")
            if not np.isnan(test):
                ax.axhline(test, lw=1.2, ls=":", alpha=0.7,
                           color="#2ca02c",
                           label=f"test (s={seed})" if seed == runs[0]["seed"] else "_")

        ax.set_xlabel("Training iteration")
        ax.set_ylabel("E[U]  (goal-reaching utility)")
        ax.set_title(f"{arch}  |  n_assets={n}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Annotate gap between final train and test to show overfit
        for h in runs:
            if h["train"] and not np.isnan(h["test_u"]):
                gap = h["train"][-1] - h["test_u"]
                ax.annotate(
                    f"gap={gap:+.3f}",
                    xy=(len(h["train"]) - 1, h["train"][-1]),
                    xytext=(-60, 10), textcoords="offset points",
                    fontsize=7, color="grey",
                    arrowprops=dict(arrowstyle="->", color="grey", lw=0.8),
                )
                break   # annotate once per arch

        fig.tight_layout()
        path = out_dir / f"{arch}_n{n}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)

    if histories:
        print(f"  → {out_dir}/ ({len(groups)} curve files)")


def plot_weights_vs_wealth(results, config, out_dir):
    """
    Plot the policy function π*(w) as a function of wealth, at several
    fixed time-to-horizon values.

    For FD   : interpolate the saved policy grid at (w, τ).
    For NN   : forward-pass over a wealth grid with τ encoded in the input.

    One file per (method × n_assets × seed): one panel per asset, curves
    for τ ∈ {T, T/2, T/4, T/8}.  This directly reveals:
      - How aggressively each method bets near the goal boundary
      - Whether the NN learned the bang-bang behaviour near τ≈0
      - How leverage varies with wealth above vs below the goal
    """
    try:
        import torch
        from comparisons.core.torch_nn_models import policy_weights as torch_pw
        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False

    out_dir.mkdir(parents=True, exist_ok=True)

    T         = getattr(config, "nn_horizon_years", 1.0)
    goal_mult = config.target_multiplier
    goal      = 1.0 * goal_mult           # normalised: w0 = 1.0
    w_grid    = np.linspace(0.4 * goal, 1.8 * goal, 200)
    tau_vals  = [T, T / 2, T / 4, T / 8]
    tau_labels = [f"τ={t:.2f}y" for t in tau_vals]
    colors     = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for r in results:
        method  = r["method_name"]
        n_ast   = r.get("n_assets", 1)
        seed    = r.get("seed", 0)
        tickers = list(r.get("tickers", [f"A{i}" for i in range(n_ast)]))
        n       = len(tickers)

        # ── FD policy ─────────────────────────────────────────────────────
        if r.get("method_family") == "fd" and "_fd_artifact" in r:
            art = r["_fd_artifact"]
            # The artifact should contain grids; try to extract policy
            grids = art.get("grids", {})
            w_fd  = grids.get("w_grid")
            pi_fd = grids.get("Pi_grid")   # (Nw+1,) for 1D or (Nw+1, n) for nD
            if w_fd is None or pi_fd is None:
                continue
            pi_fd = np.atleast_2d(pi_fd) if pi_fd.ndim == 1 else pi_fd
            # FD is time-stationary at t=0; plot the single solved policy
            fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
            for i, ticker in enumerate(tickers):
                ax = axes[0, i]
                col = pi_fd[:, i] if pi_fd.shape[1] > i else pi_fd[:, 0]
                ax.plot(w_fd, col, lw=1.8, color="#1f77b4", label="FD t=0")
                ax.axvline(goal, color="grey", lw=0.8, ls="--", label="goal")
                ax.set_xlabel("Wealth w")
                ax.set_ylabel(f"π* ({ticker})")
                ax.set_title(ticker)
                ax.legend(fontsize=7)
                ax.grid(alpha=0.25)
            fig.suptitle(f"{method}  |  n={n_ast}  |  seed={seed}  — π*(w) at t=0",
                         fontsize=11)
            fig.tight_layout()
            path = out_dir / f"{method}_n{n_ast}_s{seed}.png"
            fig.savefig(path, dpi=130, bbox_inches="tight")
            plt.close(fig)

        # ── NN policy ─────────────────────────────────────────────────────
        elif r.get("method_family") == "nn" and HAS_TORCH:
            model_art = r.get("_model_artifact")
            if model_art is None:
                continue
            net        = model_art["model"]
            meta       = model_art.get("metadata", {})
            n_steps    = meta.get("n_steps", 40)
            total_steps = n_steps

            fig, axes = plt.subplots(
                len(tau_vals), n,
                figsize=(4 * n, 3 * len(tau_vals)),
                squeeze=False,
            )

            for ti, (tau, tlabel, col) in enumerate(
                zip(tau_vals, tau_labels, colors)
            ):
                # Map tau to step_idx: tau = T * (1 - step_idx / total_steps)
                step_idx = max(0, int(round(total_steps * (1.0 - tau / T))))

                pi_grid = np.array([
                    torch_pw(net, w, goal,
                             step_idx=step_idx,
                             total_steps=total_steps)
                    for w in w_grid
                ])   # (200, n)

                for i, ticker in enumerate(tickers):
                    ax = axes[ti, i]
                    ax.plot(w_grid, pi_grid[:, i] if pi_grid.ndim > 1 else pi_grid,
                            lw=1.6, color=col)
                    ax.axvline(goal, color="grey", lw=0.8, ls="--")
                    ax.axhline(0,    color="grey", lw=0.5, ls=":")
                    ax.set_xlabel("Wealth w" if ti == len(tau_vals) - 1 else "")
                    ax.set_ylabel(f"π* ({ticker})" if i == 0 else "")
                    ax.set_title(f"{ticker} | {tlabel}" if ti == 0 else tlabel)
                    ax.grid(alpha=0.2)

            fig.suptitle(
                f"{method}  |  n={n_ast}  |  seed={seed}  — π*(w) at fixed τ",
                fontsize=11,
            )
            fig.tight_layout()
            path = out_dir / f"{method}_n{n_ast}_s{seed}.png"
            fig.savefig(path, dpi=130, bbox_inches="tight")
            plt.close(fig)

    print(f"  → {out_dir}/")


def plot_fd_vs_nn_scatter(df, out_dir):
    """
    Scatter: goal_probability vs mean_gross_leverage, coloured by family.
    One panel per n_assets.
    """
    n_list = sorted(df["n_assets"].unique())
    fig, axes = plt.subplots(1, len(n_list), figsize=(6 * len(n_list), 5))
    if len(n_list) == 1:
        axes = [axes]

    for ax, n in zip(axes, n_list):
        sub = df[df["n_assets"] == n]
        for fam, grp in sub.groupby("family"):
            ax.scatter(grp["mean_gross_leverage"], grp["goal_probability"],
                       label=fam, color=PALETTE.get(fam, "#9467bd"),
                       s=70, alpha=0.8)
            for _, row in grp.iterrows():
                ax.annotate(row["method"][:10],
                            (row["mean_gross_leverage"], row["goal_probability"]),
                            fontsize=6, alpha=0.75)

        ax.set_xlabel("Mean gross leverage")
        ax.set_ylabel("Goal probability")
        ax.set_title(f"n_assets = {n}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle("Goal Probability vs Leverage (by family)", fontsize=13)
    fig.tight_layout()
    path = out_dir / "goal_vs_leverage.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def plot_regime_heatmap(df, T: int, out_dir):
    """
    Regime heatmap: cell = mean(terminal_wealth / goal) averaged over seeds.

    Rows = methods (sorted by mean across windows, descending).
    Cols = eval windows (labelled by eval_start year).
    Colour: diverging red–white–green centred at 1.0 (the goal).
    Black contour marks cells where mean_terminal_wealth == goal exactly;
    the boundary is approximated by drawing borders on cells above 1.0.

    One figure per n_assets.
    """
    mc_col = "mc_mean_wealth" if "mc_mean_wealth" in df.columns else "terminal_wealth"
    if "T_horizon" not in df.columns:
        return
    df_t = df[df["T_horizon"] == T].copy()
    if df_t.empty:
        return

    # Compute terminal_wealth / goal
    if "goal_mult" in df_t.columns:
        df_t["tw_over_goal"] = df_t[mc_col] / df_t["goal_mult"]
    else:
        df_t["tw_over_goal"] = df_t[mc_col] / (1.10 ** T)

    n_list = sorted(df_t["n_assets"].unique())
    for n in n_list:
        sub = df_t[df_t["n_assets"] == n].copy()
        if "window_idx" not in sub.columns:
            continue

        # Mean across seeds for each (method, window)
        pivot = (
            sub.groupby(["method", "window_idx"])["tw_over_goal"]
            .mean()
            .unstack("window_idx")
        )
        if pivot.empty:
            continue

        # Label columns by eval_start year if available
        if "eval_start" in sub.columns:
            win_labels = {
                w: sub[sub["window_idx"] == w]["eval_start"].iloc[0][:4]
                for w in pivot.columns
            }
            pivot.columns = [win_labels.get(c, str(c)) for c in pivot.columns]

        # Sort rows by mean across windows, descending
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

        fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.4),
                                        max(4, len(pivot) * 0.55)))
        vmax = max(2.0, float(pivot.values[~np.isnan(pivot.values)].max()))
        vmin = 0.0

        import matplotlib.colors as mcolors
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "rg_centered",
            [(0.0, "#d62728"), (1.0 / vmax, "#ffffff"), (1.0, "#2ca02c")],
        )
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        plt.colorbar(im, ax=ax, label="Terminal wealth / goal")

        # Cell values
        for row_i, row in enumerate(pivot.index):
            for col_j, col in enumerate(pivot.columns):
                val = pivot.loc[row, col]
                if not np.isnan(val):
                    color = "black" if 0.4 < val / vmax < 0.85 else "white"
                    ax.text(col_j, row_i, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, color=color)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_xlabel(f"Eval window start (T={T}yr)")
        ax.set_title(f"Regime heatmap  T={T}yr  n={n}  "
                     f"(terminal wealth / {1.10**T:.3f}×goal)", fontsize=11)
        fig.tight_layout()

        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"regime_heatmap_T{T}_n{n}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {path}")


def plot_tractability(df, out_dir):
    """
    Tractability plot: solve/train time vs n_assets for each method.

    One panel per T_horizon.  Log scale on y-axis.
    Focuses on the primary architectures (MAIN_ARCHS) plus fd_nd.
    """
    time_col = "train_time_sec"
    if time_col not in df.columns:
        return

    T_vals = sorted(df["T_horizon"].unique()) if "T_horizon" in df.columns else [1]
    fig, axes = plt.subplots(1, len(T_vals), figsize=(6 * len(T_vals), 4),
                             sharey=True)
    if len(T_vals) == 1:
        axes = [axes]

    KEY_METHODS = list(dict.fromkeys(MAIN_ARCHS))   # preserve order, deduplicate
    cmap = plt.get_cmap("tab10")

    for ax, T in zip(axes, T_vals):
        sub = df[df["T_horizon"] == T] if "T_horizon" in df.columns else df
        n_list = sorted(sub["n_assets"].unique())

        for i, method in enumerate(KEY_METHODS):
            ms = sub[sub["method"] == method]
            if ms.empty:
                continue
            times = ms.groupby("n_assets")[time_col].mean().reindex(n_list)
            ax.plot(n_list, times.values, marker="o", lw=1.8, label=method,
                    color=cmap(i % 10))

        ax.set_yscale("log")
        ax.set_xlabel("n_assets")
        ax.set_ylabel("Mean train / solve time (s)" if T == T_vals[0] else "")
        ax.set_title(f"T = {T} yr")
        ax.set_xticks(n_list)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("Computational tractability: time vs n_assets",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "tractability.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run FD vs NN portfolio experiment")
    parser.add_argument("--quick",    action="store_true",
                        help="fast sanity check (scaled-down iters)")
    parser.add_argument("--no-nn",   action="store_true",
                        help="skip NN training, FD and baselines only")
    parser.add_argument("--n-assets", default=None,
                        help="comma-separated list, e.g. '5,10'")
    parser.add_argument("--seeds",    default=None,
                        help="comma-separated list, e.g. '1,2'")
    parser.add_argument("--archs",    default=None,
                        help="comma-separated NN architectures to run")
    parser.add_argument("--include-grpo", action="store_true",
                        help="add es_grpo and es_grpo_long_only to the architecture list")
    parser.add_argument("--in-sample",   action="store_true",
                        help="train and evaluate on the same window (no calib/eval split)")
    # ── OOS rolling-window flags (new experiment design) ─────────────────────
    parser.add_argument("--oos",      action="store_true",
                        help="run rolling OOS experiment (T=1,2,3 sliding windows)")
    parser.add_argument("--horizon",  default=None,
                        help="comma-separated horizons for OOS, e.g. '1,2,3' "
                             "(default: all three)")
    parser.add_argument("--tier",     default="main",
                        choices=["main", "appendix"],
                        help="OOS tier: 'main' (5 archs, n=[1,20,100]) or "
                             "'appendix' (15 archs, n=[1,5,10,20])")
    # ── GPU / misc flags ─────────────────────────────────────────────────────
    parser.add_argument("--device",   default=None,
                        help="PyTorch device string, e.g. 'cuda', 'cpu'")
    parser.add_argument("--resume",   action="store_true",
                        help="skip runs that already have a checkpoint")
    parser.add_argument("--compile",  action="store_true",
                        help="enable torch.compile() on NN models (PyTorch ≥ 2.0)")
    parser.add_argument("--results-dir", default=None,
                        help="override output directory (default: results/experiment/)")
    args = parser.parse_args()

    # Allow CLI override of the module-level RESULTS_DIR constant.
    if args.results_dir is not None:
        global RESULTS_DIR
        RESULTS_DIR = Path(args.results_dir)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Detect device ─────────────────────────────────────────────────────────
    _device_str = args.device
    if _device_str is None:
        try:
            import torch
            if torch.cuda.is_available():
                _device_str = "cuda"
                print(f"\n  GPU detected : {torch.cuda.get_device_name(0)}")
                print(f"  VRAM         : "
                      f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                _device_str = "cpu"
        except ImportError:
            _device_str = "cpu"

    # ══════════════════════════════════════════════════════════════════════════
    # OOS ROLLING WINDOW EXPERIMENT  (--oos flag)
    # ══════════════════════════════════════════════════════════════════════════
    if args.oos:
        horizons = (
            [int(h) for h in args.horizon.split(",")]
            if args.horizon else [1, 2, 3]
        )

        print("\n" + "="*60)
        print("  OOS ROLLING WINDOW EXPERIMENT")
        print("="*60)
        print(f"  horizons       : {horizons}")
        print(f"  tier           : {args.tier}")
        print(f"  quick          : {args.quick}")
        print(f"  device         : {_device_str}")
        print(f"  resume         : {args.resume}")
        for T in horizons:
            cfg_ex = make_config_for_horizon(T, tier=args.tier, quick=args.quick,
                                             include_grpo=args.include_grpo)
            wins   = make_oos_windows(T)
            print(f"\n  T={T}yr  goal={cfg_ex.target_multiplier:.4f}  "
                  f"fd_nt={cfg_ex.fd_nt}  nn_steps={cfg_ex.nn_steps}  "
                  f"nn_iters={cfg_ex.nn_iters}  windows={len(wins)}")
            for i, (ws, we) in enumerate(wins):
                print(f"    window {i+1}: {ws} → {we}")
        print("="*60)

        t_start = time.perf_counter()
        df_oos, histories = run_oos_experiment(
            horizons      = horizons,
            tier          = args.tier,
            quick         = args.quick,
            device        = _device_str,
            resume        = args.resume,
            compile_model = args.compile,
            include_grpo  = args.include_grpo,
        )
        print(f"\nTotal OOS run time: {time.perf_counter() - t_start:.1f}s")

        if df_oos.empty:
            print("No results — check data loader and config.")
            return

        # ── Save combined summary ─────────────────────────────────────────────
        oos_csv = RESULTS_DIR / "oos_summary.csv"
        df_oos.to_csv(oos_csv, index=False)
        print(f"\nOOS summary → {oos_csv}")

        # ── Effect-size table (replaces BH tests) ─────────────────────────────
        print("\nComputing effect-size table (Δ vs fd_nd, 95% CI across windows) ...")
        df_eff = compute_effect_size_table(df_oos)
        if not df_eff.empty:
            eff_path = RESULTS_DIR / "effect_size_table.csv"
            df_eff.to_csv(eff_path, index=False)
            print(f"  Effect sizes → {eff_path}")
            with pd.option_context("display.float_format", "{:.4f}".format,
                                   "display.max_rows", 60):
                print(df_eff.to_string(index=False))

        # ── Robustness scores ─────────────────────────────────────────────────
        print("\nComputing robustness scores ...")
        df_rob = compute_robustness_score(df_oos)
        if not df_rob.empty:
            rob_path = RESULTS_DIR / "robustness_scores.csv"
            df_rob.to_csv(rob_path, index=False)
            print(f"  Robustness → {rob_path}")
            with pd.option_context("display.float_format", "{:.4f}".format,
                                   "display.max_rows", 60):
                print(df_rob.to_string(index=False))

        # ── Plots ─────────────────────────────────────────────────────────────
        print("\nGenerating OOS plots ...")
        plot_goal_probability(df_oos, RESULTS_DIR)
        plot_train_time(df_oos,       RESULTS_DIR)
        plot_tractability(df_oos,     RESULTS_DIR)
        for T in horizons:
            plot_regime_heatmap(df_oos, T, RESULTS_DIR / "heatmaps")
        plot_training_curves(histories, RESULTS_DIR / "training_curves")

        print(f"\nAll OOS outputs saved to: {RESULTS_DIR}")
        print("Done.")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # LEGACY SINGLE-WINDOW RUN  (default, no --oos flag)
    # ══════════════════════════════════════════════════════════════════════════

    n_assets_list = (
        [int(x) for x in args.n_assets.split(",")]
        if args.n_assets else None
    )
    seeds = (
        [int(x) for x in args.seeds.split(",")]
        if args.seeds else None
    )
    nn_archs = (
        [x.strip() for x in args.archs.split(",")]
        if args.archs else None
    )

    config = make_config(
        quick         = args.quick,
        no_nn         = args.no_nn,
        n_assets_list = n_assets_list,
        seeds         = seeds,
        nn_archs      = nn_archs,
        include_grpo  = args.include_grpo,
        in_sample     = args.in_sample,
    )

    print("\n" + "="*60)
    print("  EXPERIMENT CONFIGURATION (legacy single-window)")
    print("="*60)
    print(f"  n_assets       : {config.n_assets_list}")
    print(f"  seeds          : {config.random_seeds}")
    print(f"  device         : {_device_str}")
    print(f"  resume         : {args.resume}")
    print(f"  compile        : {args.compile}")
    print(f"  FD grid        : Nw={config.fd_nw}, Nt={config.fd_nt}")
    print(f"  NN iters       : {config.nn_iters}")
    print(f"  NN paths       : {config.nn_paths}")
    print(f"  NN pretrain    : {config.nn_pretrain_iters}")
    print(f"  NN antithetic  : {config.nn_antithetic}")
    print(f"  NN curriculum  : {config.nn_p_curriculum:.0%}")
    print(f"  NN patience    : {config.nn_patience}")
    print(f"  Architectures  : {config.nn_architectures if config.include_nn else 'none'}")
    print(f"  Results → {RESULTS_DIR}")
    print("="*60)

    t_start = time.perf_counter()
    results, histories = run_all(
        config,
        device        = _device_str,
        resume        = args.resume,
        compile_model = args.compile,
    )
    print(f"\nTotal run time: {time.perf_counter() - t_start:.1f}s")

    if not results:
        print("No results — check data loader and config.")
        return

    # ── Summary table ─────────────────────────────────────────────────────────
    df = build_summary(results)
    csv_path = RESULTS_DIR / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSummary table → {csv_path}")

    # ── Table 2: value-function error (n=1 NN vs FD) ─────────────────────────
    print("\nComputing Table 2 (value-function errors, n=1 only) ...")
    try:
        df_vf = compute_value_function_errors(results, config)
        if not df_vf.empty:
            vf_path = RESULTS_DIR / "table2_vf_errors.csv"
            df_vf.to_csv(vf_path, index=False)
            print(f"  Table 2 → {vf_path}")
            print(df_vf.groupby("method")[["vf_err_L1","vf_err_L2","vf_err_Linf"]].mean().to_string())
        else:
            print("  (skipped — no n=1 NN runs with model artifacts)")
    except Exception as _e:
        print(f"  Table 2 skipped: {_e}")

    # ── Effect-size table (replaces BH pairwise tests) ───────────────────────
    print("\nComputing effect-size table (Δ vs fd_nd) ...")
    try:
        df_eff = compute_effect_size_table(df)
        if not df_eff.empty:
            eff_path = RESULTS_DIR / "effect_size_table.csv"
            df_eff.to_csv(eff_path, index=False)
            print(f"  Effect sizes → {eff_path}")
            with pd.option_context("display.float_format", "{:.4f}".format,
                                   "display.max_rows", 60):
                print(df_eff.to_string(index=False))
        else:
            print("  (no fd_nd reference found — effect sizes skipped)")
    except Exception as _e:
        print(f"  Effect sizes skipped: {_e}")

    # Print quick console summary
    print("\n" + "="*60)
    print("  MC GOAL PROBABILITY (mean across seeds)")
    print("="*60)
    mc_col = "mc_goal_prob" if "mc_goal_prob" in df.columns else "goal_probability"
    pivot = (
        df.groupby(["method", "n_assets"])[mc_col]
        .mean()
        .unstack("n_assets")
    )
    first_col = pivot.columns[0]
    pivot = pivot.sort_values(first_col, ascending=False)
    with pd.option_context("display.float_format", "{:.1%}".format,
                           "display.max_rows", 50):
        print(pivot.to_string())

    print("\n" + "="*60)
    print("  TRAINING TIME (seconds)")
    print("="*60)
    time_df = df.groupby("method")[["train_time_sec", "solve_time_sec"]].mean()
    time_df["total_sec"] = time_df["train_time_sec"] + time_df["solve_time_sec"]
    print(time_df.sort_values("total_sec", ascending=False).to_string())

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_goal_probability(df,        RESULTS_DIR)
    plot_goal_vs_multiplier(df,      RESULTS_DIR)
    plot_terminal_wealth(df,         RESULTS_DIR)
    plot_train_time(df,              RESULTS_DIR)
    plot_fd_vs_nn_scatter(df,        RESULTS_DIR)
    plot_tractability(df,            RESULTS_DIR)
    plot_training_curves(histories,  RESULTS_DIR / "training_curves")
    plot_weights_vs_wealth(results,  config, RESULTS_DIR / "weights_vs_wealth")
    plot_weight_paths(results,       RESULTS_DIR / "weights")
    plot_weight_distribution(results, RESULTS_DIR / "weight_dist")

    print(f"\nAll outputs saved to: {RESULTS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
