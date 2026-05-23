import numpy as np


def compute_drawdown_series(wealth_path):
    wealth_path = np.asarray(wealth_path, dtype=float)
    peak = np.maximum.accumulate(wealth_path)
    return wealth_path / np.maximum(peak, 1e-12) - 1.0


def compute_target_metrics(terminal_wealth, target):
    terminal_wealth = np.asarray(terminal_wealth, dtype=float)
    shortfall = np.maximum(target - terminal_wealth, 0.0)
    return {
        "target_hit_rate": float(np.mean(terminal_wealth >= target)),
        "mean_terminal_wealth": float(np.mean(terminal_wealth)),
        "median_terminal_wealth": float(np.median(terminal_wealth)),
        "terminal_wealth_p05": float(np.percentile(terminal_wealth, 5)),
        "expected_shortfall": float(np.mean(shortfall)),
    }


def compute_weight_metrics(weight_path):
    weight_path = np.asarray(weight_path, dtype=float)
    if weight_path.ndim == 1:
        weight_path = weight_path[:, None]

    gross = np.abs(weight_path).sum(axis=1)
    net = weight_path.sum(axis=1)
    concentration = np.square(weight_path).sum(axis=1)
    max_single = np.abs(weight_path).max(axis=1)
    turnover = np.abs(np.diff(weight_path, axis=0)).sum() if len(weight_path) > 1 else 0.0

    return {
        "mean_gross_leverage": float(np.mean(gross)),
        "max_gross_leverage": float(np.max(gross)),
        "mean_net_exposure": float(np.mean(net)),
        "mean_concentration": float(np.mean(concentration)),
        "max_single_name_weight": float(np.max(max_single)),
        "turnover": float(turnover),
    }


def compute_path_risk_metrics(wealth_path, trading_days=252):
    wealth_path = np.asarray(wealth_path, dtype=float)
    rets = np.diff(wealth_path) / np.maximum(wealth_path[:-1], 1e-12)
    drawdown = compute_drawdown_series(wealth_path)
    return {
        "wealth_volatility": float(np.std(rets) * np.sqrt(trading_days)) if len(rets) else 0.0,
        "max_drawdown": float(np.min(drawdown)) if len(drawdown) else 0.0,
    }
