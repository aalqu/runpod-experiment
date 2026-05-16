"""
run_full_experiment.py
----------------------
Full FD vs NN comparison experiment across n=1, 5, 10, 20 assets.

Usage
-----
    python3.9 comparisons/experiments/run_full_experiment.py

Options (edit CONFIG block below or pass via argparse):
    --n-assets   comma-separated list, e.g. 1,5,10,20
    --seeds      comma-separated, e.g. 1,2,3
    --n-mc       number of MC simulation paths (default 500)
    --no-nn      skip neural network training
    --quick      fast mode: n_mc=100, nn_iters=10, 1 seed

Results saved to:  comparisons/results/full_experiment/
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ── Path bootstrap (mirrors notebook Cell 1) ──────────────────────────────
ROOT = Path(__file__).resolve()
while not (ROOT / "comparisons").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Core imports ──────────────────────────────────────────────────────────
import comparisons  # noqa – triggers path bootstrap in comparisons/__init__.py
from fd_core import (
    asymp_goalreach, fd_solve, goal_utility, make_fd_policy,
)
from real_data_loader import agg_1d, load_portfolio
from comparisons.core.evaluation import (
    browne_policy_1d, evaluate_policy_mc, fd_policy_1d,
)

HAS_TORCH = False
try:
    import torch  # noqa
    from comparisons.core.torch_nn_models import (
        TORCH_ARCHITECTURES,
        policy_weights as torch_policy_weights,
        train_torch_policy_net,
    )
    HAS_TORCH = True
    if torch.cuda.is_available():
        _dev = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        _dev = torch.device('mps')
    else:
        _dev = torch.device('cpu')
    print(f"[torch] available — device: {_dev}")
except ImportError:
    print("[torch] not available — NN methods will use numpy fallback")
    from comparisons.core.nn_models import (
        ARCHITECTURES as _NP_ARCHS,
        policy_weights as numpy_policy_weights,
        train_numpy_policy_net,
    )

# ── Argument parsing ───────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Full FD vs NN experiment")
    p.add_argument("--n-assets", default="1,5,10,20",
                   help="Comma-separated asset counts")
    p.add_argument("--seeds", default="1,2,3",
                   help="Comma-separated random seeds")
    p.add_argument("--n-mc", type=int, default=500,
                   help="Monte Carlo paths per evaluation")
    p.add_argument("--no-nn", action="store_true",
                   help="Skip neural network training")
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: n_mc=100, 1 seed, few NN iters")
    p.add_argument("--nn-archs", default="nn_mlp_small,nn_mlp_deep",
                   help="Comma-separated NN architecture names")
    return p.parse_args()


# ── Config ─────────────────────────────────────────────────────────────────
TARGET_MULT    = 1.10    # goal = 1.10 × initial wealth
INITIAL_WEALTH = 1.0
D, U           = -5.0, 3.0   # per-asset bounds
MAX_LONG       = 3.0
MAX_SHORT      = 5.0

# FD grid settings
FD_NW, FD_NT   = 120, 80
FD_WEALTH_MAX  = 2.5

# NN training settings (full)
NN_ITERS       = 50
NN_PATHS       = 512
NN_STEPS       = 32
NN_LR          = 3e-3

OUT_DIR = ROOT / "comparisons" / "results" / "full_experiment"


# ── Leverage constraint ────────────────────────────────────────────────────
def clip_leverage(weights, d=D, u=U, max_long=MAX_LONG, max_short=MAX_SHORT):
    weights = np.clip(weights, d, u)
    squeeze = weights.ndim == 1
    if squeeze:
        weights = weights[None, :]
    lp = np.maximum(weights, 0).sum(1, keepdims=True).clip(min=1e-12)
    sp = np.maximum(-weights, 0).sum(1, keepdims=True).clip(min=1e-12)
    weights = np.where(weights >= 0,
                       weights * np.minimum(max_long / lp, 1.0),
                       weights * np.minimum(max_short / sp, 1.0))
    return weights.squeeze(0) if squeeze else weights


# ── FD solver wrapper ──────────────────────────────────────────────────────
def solve_fd_1d(mu_1d, sig_1d, r, goal=TARGET_MULT, verbose=True):
    """Solve FD HJB and return (w_grid, pi_grid, policy_fn, solve_sec)."""
    t0 = time.perf_counter()
    w_grid, _, pi_grid = fd_solve(
        mu=mu_1d, r=r, sigma=sig_1d, T=1.0,
        A=FD_WEALTH_MAX, Nw=FD_NW, Nt=FD_NT,
        d=D, u=U,
        utility_fn=goal_utility,
        asymptotic_fn=lambda w, tau: asymp_goalreach(w, tau, sig_1d, D, U),
        UB=0.0, UA=1.0,
    )
    dt = time.perf_counter() - t0
    if verbose:
        print(f"    FD solved in {dt:.2f}s  (Nw={FD_NW}, Nt={FD_NT})")
    policy = make_fd_policy(w_grid, pi_grid, d=D, u=U)
    return w_grid, pi_grid, policy, dt


# ── NN trainer wrapper ─────────────────────────────────────────────────────
def train_nn(arch, market_data, initial_wealth, seed, n_iters):
    """Train one NN and return (infer_fn, train_sec, n_params)."""
    goal = initial_wealth * TARGET_MULT
    t0 = time.perf_counter()
    if HAS_TORCH and arch in TORCH_ARCHITECTURES:
        net, meta = train_torch_policy_net(
            mu_vec=market_data.mu_ann, omega_mat=market_data.omega,
            r=market_data.r, architecture_name=arch,
            w0=initial_wealth, goal_mult=TARGET_MULT,
            n_paths=NN_PATHS, n_iters=n_iters, n_steps=NN_STEPS, lr=NN_LR,
            d=D, u=U, max_long_leverage=MAX_LONG, max_short_leverage=MAX_SHORT,
            seed=seed,
        )
        def infer_fn(W, g, _net=net):
            raw = np.asarray(torch_policy_weights(_net, W, g), dtype=float)
            return clip_leverage(raw)
    else:
        net, meta = train_numpy_policy_net(
            mu_vec=market_data.mu_ann, omega_mat=market_data.omega,
            r=market_data.r, architecture_name=arch,
            w0=initial_wealth, goal_mult=TARGET_MULT,
            n_paths=NN_PATHS, n_iters=n_iters,
            population_size=24, elite_frac=0.25, n_steps=NN_STEPS,
            d=D, u=U, max_long_leverage=MAX_LONG, max_short_leverage=MAX_SHORT,
            seed=seed,
        )
        def infer_fn(W, g, _net=net):
            raw = np.asarray(numpy_policy_weights(_net, W, g), dtype=float)
            return clip_leverage(raw)
    train_sec = time.perf_counter() - t0
    n_params  = int(meta.get("param_size", 0))
    return infer_fn, train_sec, n_params


# ── MC wrapper for 1-D FD policy ──────────────────────────────────────────
def fd_policy_wrapper_1d(policy):
    """Wraps make_fd_policy callable for evaluate_policy_mc (n=1)."""
    def _fn(w_norm, tau):
        return policy(w_norm, tau)
    return _fn


# ── MC wrapper for n>1 NN policy ──────────────────────────────────────────
def nn_policy_wrapper_nd(infer_fn, goal):
    """Wraps infer_fn(W, goal) for evaluate_policy_mc (n>1)."""
    def _fn(W, g):
        return infer_fn(W, g)
    return _fn


# ── Main experiment loop ───────────────────────────────────────────────────
def run_experiment(n_assets_list, seeds, n_mc, include_nn, nn_archs, quick):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nn_iters = 10 if quick else NN_ITERS

    all_rows = []
    print(f"\n{'='*60}")
    print(f"Full FD vs NN Experiment")
    print(f"n_assets: {n_assets_list}  seeds: {seeds}  n_mc: {n_mc}")
    print(f"include_nn: {include_nn}  nn_archs: {nn_archs}")
    print(f"{'='*60}\n")

    for n in n_assets_list:
        print(f"\n--- n_assets = {n} ---")
        mkt = load_portfolio(n)
        mu_1d, sig_1d, _ = agg_1d(mkt)
        r = mkt.r
        goal = INITIAL_WEALTH * TARGET_MULT

        # ── Solve FD once per n (same 1D aggregate, seed-independent) ──
        w_grid, pi_grid, fd_policy, fd_solve_sec = solve_fd_1d(
            mu_1d, sig_1d, r, verbose=True)

        # Browne analytical solution (1D)
        browne_policy = browne_policy_1d(mu_1d, r, sig_1d, d=D, u=U)

        for seed in seeds:
            print(f"  seed={seed}", flush=True)
            base_row = dict(n_assets=n, seed=seed, initial_wealth=INITIAL_WEALTH,
                            target_wealth=goal, n_mc=n_mc)

            # ── FD 1D proxy ──
            fd_mc = evaluate_policy_mc(
                policy_fn=fd_policy_wrapper_1d(fd_policy),
                market_data=load_portfolio(1) if n > 1 else mkt,
                initial_wealth=INITIAL_WEALTH, target_multiplier=TARGET_MULT,
                n_mc=n_mc, seed=seed,
            )
            all_rows.append({**base_row, "method": "fd_1d_proxy",
                              "solve_sec": fd_solve_sec, "train_sec": 0.0,
                              **_mc_metrics(fd_mc)})
            print(f"    FD  hit={fd_mc['target_hit_prob']:.3f}  "
                  f"mean_W={fd_mc['mean_terminal_wealth']:.4f}")

            # ── Browne analytical ──
            br_mc = evaluate_policy_mc(
                policy_fn=browne_policy,
                market_data=load_portfolio(1) if n > 1 else mkt,
                initial_wealth=INITIAL_WEALTH, target_multiplier=TARGET_MULT,
                n_mc=n_mc, seed=seed,
            )
            all_rows.append({**base_row, "method": "browne_1d",
                              "solve_sec": 0.0, "train_sec": 0.0,
                              **_mc_metrics(br_mc)})
            print(f"    Browne hit={br_mc['target_hit_prob']:.3f}  "
                  f"mean_W={br_mc['mean_terminal_wealth']:.4f}")

            # ── Equal-weight static ──
            ew_weights = np.ones(n) / n
            ew_mc = _eval_static_mc(ew_weights, mkt, seed, n_mc)
            all_rows.append({**base_row, "method": "equal_weight",
                              "solve_sec": 0.0, "train_sec": 0.0,
                              **_mc_metrics(ew_mc)})
            print(f"    EW   hit={ew_mc['target_hit_prob']:.3f}  "
                  f"mean_W={ew_mc['mean_terminal_wealth']:.4f}")

            # ── Neural networks ──
            if include_nn:
                for arch in nn_archs:
                    print(f"    training {arch}...", end=" ", flush=True)
                    try:
                        infer_fn, train_sec, n_params = train_nn(
                            arch, mkt, INITIAL_WEALTH, seed, nn_iters)
                        nn_goal = goal

                        if n == 1:
                            # Wrap NN output for 1D evaluate_policy_mc signature
                            def nn_1d_policy(w_norm, tau, _fn=infer_fn, _g=goal,
                                             _w0=INITIAL_WEALTH):
                                W = w_norm * _g
                                raw = _fn(W, _g)
                                return float(np.asarray(raw).ravel()[0])
                            nn_mc = evaluate_policy_mc(
                                policy_fn=nn_1d_policy,
                                market_data=mkt,
                                initial_wealth=INITIAL_WEALTH,
                                target_multiplier=TARGET_MULT,
                                n_mc=n_mc, seed=seed,
                            )
                        else:
                            def nn_nd_policy(W, g, _fn=infer_fn):
                                return _fn(W, g)
                            nn_mc = evaluate_policy_mc(
                                policy_fn=nn_nd_policy,
                                market_data=mkt,
                                initial_wealth=INITIAL_WEALTH,
                                target_multiplier=TARGET_MULT,
                                n_mc=n_mc, seed=seed,
                            )

                        all_rows.append({**base_row, "method": arch,
                                         "solve_sec": 0.0, "train_sec": train_sec,
                                         "n_params": n_params,
                                         **_mc_metrics(nn_mc)})
                        print(f"hit={nn_mc['target_hit_prob']:.3f}  "
                              f"mean_W={nn_mc['mean_terminal_wealth']:.4f}  "
                              f"({train_sec:.0f}s)")
                    except Exception as exc:
                        print(f"FAILED: {exc}")
                        all_rows.append({**base_row, "method": arch,
                                         "solve_sec": 0.0, "train_sec": 0.0,
                                         "error": str(exc)})

    # ── Save results ───────────────────────────────────────────────────────
    import csv
    csv_path = OUT_DIR / "mc_results.csv"
    if all_rows:
        fieldnames = sorted({k for row in all_rows for k in row})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in all_rows:
                w.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"\nResults saved → {csv_path}")
    return all_rows


def _mc_metrics(mc):
    return {
        "target_hit_prob":        mc["target_hit_prob"],
        "mean_terminal_wealth":   mc["mean_terminal_wealth"],
        "median_terminal_wealth": mc["median_terminal_wealth"],
        "shortfall_mean":         mc["shortfall_mean"],
        "sharpe_mean":            mc["sharpe_mean"],
    }


def _eval_static_mc(weights, mkt, seed, n_mc):
    """Monte Carlo evaluation of a fixed weight vector."""
    from backtest_core import simulate_one_year_5asset, simulate_one_year_1asset
    n = mkt.n
    goal = INITIAL_WEALTH * TARGET_MULT
    rng = np.random.default_rng(seed)
    terminal_wealth = np.empty(n_mc)
    for i in range(n_mc):
        path_rng = np.random.default_rng(rng.integers(0, 2**31))
        if n == 1:
            W_path, _ = simulate_one_year_1asset(
                W_start=INITIAL_WEALTH, goal=goal,
                policy_fn=lambda w_norm, tau: float(weights[0]),
                mu=float(mkt.mu_ann[0]), r=mkt.r,
                sigma=float(mkt.sigma[0]), rng=path_rng,
            )
        else:
            def _static_policy(W, g, _w=weights):
                return _w
            W_path, _ = simulate_one_year_5asset(
                W_start=INITIAL_WEALTH, goal=goal,
                nn_policy_fn=_static_policy,
                mu_vec=mkt.mu_ann, Omega_mat=mkt.omega,
                r=mkt.r, target_vol=0.25, rng=path_rng,
            )
        terminal_wealth[i] = W_path[-1]
    goal_hit = terminal_wealth >= goal
    shortfall = np.maximum(goal - terminal_wealth, 0.0)
    ann_ret = terminal_wealth / INITIAL_WEALTH - 1.0
    sharpe = (ann_ret - mkt.r) / (terminal_wealth.std() + 1e-8)
    return {
        "terminal_wealth":        terminal_wealth,
        "goal_hit":               goal_hit,
        "target_hit_prob":        float(goal_hit.mean()),
        "mean_terminal_wealth":   float(terminal_wealth.mean()),
        "median_terminal_wealth": float(np.median(terminal_wealth)),
        "shortfall_mean":         float(shortfall.mean()),
        "sharpe_mean":            float(sharpe.mean()),
    }


if __name__ == "__main__":
    args = parse_args()
    n_assets_list = [int(x) for x in args.n_assets.split(",")]
    seeds         = [int(x) for x in args.seeds.split(",")]
    nn_archs      = [x.strip() for x in args.nn_archs.split(",")]
    n_mc          = 100 if args.quick else args.n_mc
    if args.quick:
        seeds = seeds[:1]
    run_experiment(n_assets_list, seeds, n_mc,
                   include_nn=not args.no_nn,
                   nn_archs=nn_archs,
                   quick=args.quick)
