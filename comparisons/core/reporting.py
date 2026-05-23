import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .io import ensure_dir, save_summary_csv

os.environ.setdefault('MPLCONFIGDIR', str(Path('/tmp') / 'matplotlib-codex'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


_NUMERIC_FIELDS = {
    'train_time_sec', 'solve_time_sec', 'eval_time_sec', 'target_hit_rate',
    'mean_terminal_wealth', 'median_terminal_wealth', 'terminal_wealth_p05',
    'expected_shortfall', 'mean_gross_leverage', 'max_gross_leverage',
    'mean_net_exposure', 'mean_concentration', 'max_single_name_weight',
    'turnover', 'wealth_volatility', 'max_drawdown', 'initial_wealth', 'target_wealth'
}


def aggregate_summary_rows(rows: List[Dict], group_fields=('method_name', 'method_family', 'n_assets')):
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(row)

    aggregated = []
    for key, items in groups.items():
        base = {field: value for field, value in zip(group_fields, key)}
        base['n_runs'] = len(items)
        for field in _NUMERIC_FIELDS:
            vals = [float(item[field]) for item in items if field in item]
            if vals:
                base[field] = float(np.mean(vals))
        aggregated.append(base)

    aggregated.sort(key=lambda row: (row['n_assets'], row['method_family'], row['method_name']))
    return aggregated


def filter_rows(rows: Iterable[Dict], **criteria):
    out = []
    for row in rows:
        if all(row.get(k) == v for k, v in criteria.items()):
            out.append(row)
    return out


_FD_METHODS = {'fd_1d_proxy', 'fd_merton_multi', 'fd_goalreach_proxy'}
_NN_METHODS = {'deep_bsde', 'pinn', 'actor_critic', 'lstm', 'transformer',
               'nn_mlp_small', 'nn_mlp_deep',
               # v2 enriched-feature variants (§3.2 of NN_FEATURE_REDESIGN.md)
               'nn_mlp_small_v2', 'nn_policy_net_v2', 'nn_ste_goalreach_v2',
               'deep_bsde_v2', 'actor_critic_v2', 'lstm_v2', 'transformer_v2'}


def save_comparison_tables(summary_rows: List[Dict], summary_dir: Path):
    summary_dir = ensure_dir(summary_dir)
    aggregated = aggregate_summary_rows(summary_rows)
    neural_rows = [row for row in aggregated if row['method_family'] == 'nn']
    fd_vs_nn_rows = [row for row in aggregated if row['method_name'] in
                     _FD_METHODS | _NN_METHODS]

    save_summary_csv(summary_dir / 'aggregated_results.csv', aggregated)
    save_summary_csv(summary_dir / 'neural_family_results.csv', neural_rows)
    save_summary_csv(summary_dir / 'fd_vs_neural_results.csv', fd_vs_nn_rows)
    return {
        'aggregated': aggregated,
        'neural_family': neural_rows,
        'fd_vs_neural': fd_vs_nn_rows,
    }


_COLOR_MAP = {
    'fd_1d_proxy': '#1f2937',
    'fd_merton_multi': '#374151',
    'deep_bsde': '#b91c1c',
    'pinn': '#0f766e',
    'actor_critic': '#1d4ed8',
    'lstm': '#a16207',
    'transformer': '#7c3aed',
    'nn_mlp_small': '#6d28d9',
    'nn_mlp_deep': '#0369a1',
    # keep old name for backward-compat with any cached rows
    'fd_goalreach_proxy': '#1f2937',
    # v2 enriched-feature variants (§5 of NN_FEATURE_REDESIGN.md)
    # Use lighter shades of the v1 colours so v1 and v2 are visually paired.
    'nn_mlp_small_v2':     '#a78bfa',   # lighter purple  (nn_mlp_small = #6d28d9)
    'nn_policy_net_v2':    '#93c5fd',   # lighter blue    (no v1 entry)
    'nn_ste_goalreach_v2': '#6ee7b7',   # lighter green   (no v1 entry)
    'deep_bsde_v2':        '#fca5a5',   # lighter red     (deep_bsde = #b91c1c)
    'actor_critic_v2':     '#93c5fd',   # lighter blue    (actor_critic = #1d4ed8)
    'lstm_v2':             '#fde68a',   # lighter amber   (lstm = #a16207)
    'transformer_v2':      '#c4b5fd',   # lighter violet  (transformer = #7c3aed)
}


def _line_plot(ax, rows, methods, metric, ylabel, title):
    for method in methods:
        pts = [row for row in rows if row['method_name'] == method]
        if not pts:
            continue
        pts = sorted(pts, key=lambda row: row['n_assets'])
        x = [row['n_assets'] for row in pts]
        y = [row[metric] for row in pts]
        ax.plot(x, y, marker='o', linewidth=2, label=method, color=_COLOR_MAP.get(method))
    ax.set_xlabel('Number of assets')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)


def save_plots(summary_rows: List[Dict], plots_dir: Path):
    plots_dir = ensure_dir(plots_dir)
    aggregated = aggregate_summary_rows(summary_rows)
    target_methods = [
        'fd_1d_proxy', 'fd_merton_multi',
        'deep_bsde', 'pinn', 'actor_critic', 'lstm', 'transformer',
        'nn_mlp_small', 'nn_mlp_deep',
        # v2 enriched-feature variants
        'nn_mlp_small_v2', 'nn_policy_net_v2', 'nn_ste_goalreach_v2',
        'deep_bsde_v2', 'actor_critic_v2', 'lstm_v2', 'transformer_v2',
        # backward-compat
        'fd_goalreach_proxy',
    ]
    compare_rows = [row for row in aggregated if row['method_name'] in target_methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    _line_plot(ax, compare_rows, target_methods, 'target_hit_rate', 'Target hit rate', 'FD vs Neural: Target Hit Rate')
    fig.tight_layout()
    fig.savefig(plots_dir / 'fd_vs_neural_target_hit_rate.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    _line_plot(ax, compare_rows, target_methods, 'mean_terminal_wealth', 'Mean terminal wealth', 'FD vs Neural: Mean Terminal Wealth')
    fig.tight_layout()
    fig.savefig(plots_dir / 'fd_vs_neural_mean_terminal_wealth.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    _line_plot(ax, compare_rows, target_methods, 'train_time_sec', 'Train / solve time (s)', 'FD vs Neural: Computational Cost')
    fig.tight_layout()
    fig.savefig(plots_dir / 'fd_vs_neural_runtime.png', dpi=160)
    plt.close(fig)

    all_method_rows = [row for row in aggregated if row['method_family'] in ('nn', 'fd')]
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ['fd_1d_proxy', 'fd_merton_multi', 'deep_bsde', 'pinn',
                   'actor_critic', 'lstm', 'transformer', 'nn_mlp_small', 'nn_mlp_deep']:
        pts = [row for row in all_method_rows if row['method_name'] == method]
        if not pts:
            continue
        ax.scatter(
            [row['mean_gross_leverage'] for row in pts],
            [row['target_hit_rate'] for row in pts],
            s=[40 + 12 * row['n_assets'] for row in pts],
            alpha=0.8,
            label=method,
            color=_COLOR_MAP.get(method),
        )
    ax.set_xlabel('Mean gross leverage')
    ax.set_ylabel('Target hit rate')
    ax.set_title('FD & Neural: Risk vs Performance')
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / 'neural_risk_vs_performance.png', dpi=160)
    plt.close(fig)

    return [
        plots_dir / 'fd_vs_neural_target_hit_rate.png',
        plots_dir / 'fd_vs_neural_mean_terminal_wealth.png',
        plots_dir / 'fd_vs_neural_runtime.png',
        plots_dir / 'neural_risk_vs_performance.png',
    ]


# ── Pairwise bootstrap test + Benjamini-Hochberg FDR correction ───────────────

def _bh_correction(p_values: np.ndarray, q: float = 0.10):
    """
    Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : array of raw p-values (length m)
    q        : target FDR level (default 0.10)

    Returns
    -------
    p_adj      : BH-adjusted p-values (same length, same order)
    significant: bool array — True where null is rejected at level q
    """
    m = len(p_values)
    if m == 0:
        return np.array([]), np.array([], dtype=bool)

    order    = np.argsort(p_values)
    ranks    = np.empty(m, dtype=int)
    ranks[order] = np.arange(1, m + 1)

    # Step-up: find largest k s.t. p_(k) ≤ k/m * q
    p_sorted = p_values[order]
    thresholds = (np.arange(1, m + 1) / m) * q
    below = np.where(p_sorted <= thresholds)[0]
    k_max = below[-1] if len(below) > 0 else -1

    significant_sorted = np.zeros(m, dtype=bool)
    if k_max >= 0:
        significant_sorted[: k_max + 1] = True

    significant = significant_sorted[ranks - 1]

    # Adjusted p-values (Yekutieli–Benjamini step-up formula)
    p_adj_sorted = np.minimum(1.0, np.minimum.accumulate(
        (m / np.arange(1, m + 1))[::-1] * p_sorted[::-1]
    )[::-1])
    p_adj = p_adj_sorted[ranks - 1]
    return p_adj, significant


def pairwise_bh_test(
    df: "pd.DataFrame",
    metric: str = "mc_goal_prob",
    n_bootstrap: int = 2000,
    q: float = 0.10,
    seed: int = 0,
    n_assets_filter: int = None,
    goal_mult_filter: float = 1.10,
) -> "pd.DataFrame":
    """
    Pairwise permutation bootstrap test between all method pairs, with
    Benjamini-Hochberg FDR correction at level q.

    For each pair (A, B) at a given n_assets the test statistic is
    |mean_A − mean_B| on `metric` across seeds.  The null distribution
    is obtained by 2000 random permutations of the pooled sample.

    Parameters
    ----------
    df               : summary DataFrame from build_summary()
    metric           : column to compare (default "mc_goal_prob")
    n_bootstrap      : number of permutation replicates (default 2000)
    q                : BH FDR level (default 0.10)
    seed             : RNG seed
    n_assets_filter  : restrict to a single n_assets value (None = all)
    goal_mult_filter : restrict to a single goal_mult (None = skip filter)

    Returns
    -------
    DataFrame with columns:
        n_assets, method_a, method_b, mean_a, mean_b, stat,
        p_raw, p_adj, significant
    Sorted by p_raw ascending within each n_assets group.
    """
    if metric not in df.columns:
        raise ValueError(f"metric '{metric}' not found in df.columns={list(df.columns)}")

    rng = np.random.default_rng(seed)
    work = df.copy()

    if goal_mult_filter is not None and "goal_mult" in work.columns:
        work = work[np.isclose(work["goal_mult"].astype(float), goal_mult_filter)]
    if n_assets_filter is not None:
        work = work[work["n_assets"] == n_assets_filter]

    rows = []
    for n, grp in work.groupby("n_assets"):
        methods = sorted(grp["method"].unique())
        pairs = [(a, b) for i, a in enumerate(methods) for b in methods[i + 1:]]

        raw_ps = []
        pair_stats = []
        for a, b in pairs:
            xa = grp.loc[grp["method"] == a, metric].dropna().values.astype(float)
            xb = grp.loc[grp["method"] == b, metric].dropna().values.astype(float)
            if len(xa) == 0 or len(xb) == 0:
                raw_ps.append(float("nan"))
                pair_stats.append((float("nan"), float("nan"), float("nan")))
                continue
            obs_stat = abs(xa.mean() - xb.mean())
            pooled = np.concatenate([xa, xb])
            na = len(xa)
            count_ge = 0
            for _ in range(n_bootstrap):
                rng.shuffle(pooled)
                perm_stat = abs(pooled[:na].mean() - pooled[na:].mean())
                if perm_stat >= obs_stat:
                    count_ge += 1
            p_raw = (count_ge + 1) / (n_bootstrap + 1)   # +1 smoothing
            raw_ps.append(p_raw)
            pair_stats.append((xa.mean(), xb.mean(), obs_stat))

        valid_mask  = np.array([not np.isnan(p) for p in raw_ps])
        raw_arr     = np.array([p for p in raw_ps if not np.isnan(p)], dtype=float)
        p_adj_arr, sig_arr = _bh_correction(raw_arr, q=q)

        adj_iter = iter(zip(p_adj_arr, sig_arr))
        for (a, b), p_raw, (mean_a, mean_b, stat), is_valid in zip(
            pairs, raw_ps, pair_stats, valid_mask
        ):
            if is_valid:
                p_adj, sig = next(adj_iter)
            else:
                p_adj, sig = float("nan"), False
            rows.append({
                "n_assets"  : int(n),
                "method_a"  : a,
                "method_b"  : b,
                "mean_a"    : float(mean_a),
                "mean_b"    : float(mean_b),
                "stat"      : float(stat),
                "p_raw"     : float(p_raw),
                "p_adj"     : float(p_adj),
                "significant": bool(sig),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["n_assets", "p_raw"]).reset_index(drop=True)
    return out
