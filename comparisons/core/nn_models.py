import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


ARCHITECTURES: Dict[str, Tuple[int, ...]] = {
    'nn_linear': tuple(),
    'nn_mlp_small': (32, 32),        # matches torch_nn_models TORCH_ARCHITECTURES
    'nn_mlp_deep': (64, 64, 32),     # matches torch_nn_models TORCH_ARCHITECTURES
    # ── Long-only variants (numpy fallback hidden sizes) ──────────────────────
    # These are trained with PyTorch in the full experiment; the numpy entries
    # exist only so evaluate_nn_portfolio does not crash when PyTorch is absent.
    'nn_policy_long_only':              (128, 128, 128),
    'nn_ste_long_only':                 (128, 128, 128),
    'nn_digital_hedge_long_only':       (128, 128, 128),
    # ── Historical-replay variants ────────────────────────────────────────────
    'nn_historical_replay':             (128, 128, 128),
    'nn_historical_replay_long_only':   (128, 128, 128),
}


@dataclass
class NumpyPolicyNet:
    n_assets: int
    hidden_layers: Tuple[int, ...] = (16,)
    d: float = -5.0
    u: float = 3.0
    params: np.ndarray = None

    def __post_init__(self):
        self.hidden_layers = tuple(self.hidden_layers)
        self.layer_sizes = (1,) + self.hidden_layers + (self.n_assets,)
        self.param_size = 0
        for inp, out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            self.param_size += inp * out + out
        if self.params is None:
            self.params = np.zeros(self.param_size, dtype=float)
        else:
            self.params = np.asarray(self.params, dtype=float).copy()
            if self.params.size != self.param_size:
                raise ValueError('Parameter size does not match architecture')

    def _unpack(self, params=None):
        flat = self.params if params is None else np.asarray(params, dtype=float)
        idx = 0
        layers = []
        for inp, out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            w = flat[idx:idx + inp * out].reshape(inp, out)
            idx += inp * out
            b = flat[idx:idx + out]
            idx += out
            layers.append((w, b))
        return layers

    def forward(self, x, params=None):
        h = np.asarray(x, dtype=float)
        if h.ndim == 1:
            h = h[:, None]
        for layer_idx, (w, b) in enumerate(self._unpack(params)):
            h = h @ w + b
            if layer_idx < len(self.layer_sizes) - 2:
                h = np.tanh(h)
        return 0.5 * (self.u - self.d) * np.tanh(h) + 0.5 * (self.u + self.d)


def policy_weights(net: NumpyPolicyNet, W_current, goal: float):
    """Return portfolio weights for a given wealth level.

    Supports both scalar and batched array inputs:
      - Scalar W_current  → returns shape (n_assets,)
      - Array  W_current  → returns shape (N, n_assets)
    """
    W = np.asarray(W_current, dtype=float)
    scalar_input = W.ndim == 0
    W_flat = W.ravel()                                   # (N,)
    x = (W_flat / max(goal, 1e-12))[:, None]             # (N, 1)
    out = net.forward(x)                                 # (N, n_assets)
    if scalar_input:
        return out.reshape(-1)                           # (n_assets,)
    return out                                           # (N, n_assets)


def _apply_leverage_constraint_numpy(weights, d: float, u: float,
                                      max_long: float, max_short: float):
    """
    Per-asset clip [d, u] followed by aggregate long/short leverage caps.
    Long  side : sum(max(w_i, 0)) <= max_long
    Short side : sum(max(-w_i, 0)) <= max_short
    Applied per-path (weights shape: B x n_assets).
    """
    weights = np.clip(weights, d, u)
    long_lev = np.maximum(weights, 0.0).sum(axis=1, keepdims=True).clip(min=1e-12)
    short_lev = np.maximum(-weights, 0.0).sum(axis=1, keepdims=True).clip(min=1e-12)
    long_scale = np.minimum(max_long / long_lev, 1.0)
    short_scale = np.minimum(max_short / short_lev, 1.0)
    weights = np.where(weights >= 0, weights * long_scale, weights * short_scale)
    return weights


def _portfolio_step(weights, mu_vec, chol, r, dt, rng, d, u, max_long, max_short):
    z = rng.standard_normal((weights.shape[0], len(mu_vec)))
    dS = mu_vec * dt + (z @ chol.T) * math.sqrt(dt)
    # Apply the SAME leverage constraint used at evaluation time.
    weights = _apply_leverage_constraint_numpy(weights, d, u, max_long, max_short)
    bond = r * dt
    excess = np.sum(weights * (dS - bond), axis=1)
    return weights, excess, bond


def evaluate_candidate(params, net_template: NumpyPolicyNet, mu_vec, omega_mat, r,
                       w0, goal_mult, n_paths, n_steps, max_long, max_short,
                       utility, asp_p, asp_c1, asp_R, seed):
    rng = np.random.default_rng(seed)
    mu_vec = np.asarray(mu_vec, dtype=float)
    chol = np.linalg.cholesky(np.asarray(omega_mat, dtype=float) + 1e-10 * np.eye(len(mu_vec)))
    goal = w0 * goal_mult
    wealth = np.full(n_paths, w0, dtype=float)
    dt = 1.0 / n_steps

    for _ in range(n_steps):
        weights = net_template.forward((wealth / goal)[:, None], params=params)
        weights, excess, bond = _portfolio_step(
            weights, mu_vec, chol, r, dt, rng,
            net_template.d, net_template.u, max_long, max_short,
        )
        wealth = np.maximum(wealth * (1.0 + bond + excess), 1e-6)

    w_norm = wealth / goal
    if utility == 'goalreach':
        utility_vals = 1.0 / (1.0 + np.exp(-(w_norm - 1.0) / 0.05))
    elif utility == 'aspiration':
        utility_vals = np.where(w_norm < asp_R, w_norm**asp_p / asp_p, asp_c1 * w_norm**asp_p / asp_p)
    else:
        raise ValueError(f'Unknown utility: {utility}')

    return float(np.mean(utility_vals)), wealth


def train_numpy_policy_net(mu_vec, omega_mat, r, architecture_name='nn_mlp_small',
                           w0=1.0, goal_mult=1.10, n_paths=256, n_iters=20,
                           population_size=24, elite_frac=0.25, n_steps=24,
                           max_long_leverage=3.0, max_short_leverage=5.0,
                           utility='goalreach', asp_p=0.5,
                           asp_c1=1.2, asp_R=1.0, d=-5.0, u=3.0, seed=1):
    if architecture_name not in ARCHITECTURES:
        raise ValueError(f'Unknown architecture: {architecture_name}')

    net = NumpyPolicyNet(
        n_assets=len(mu_vec),
        hidden_layers=ARCHITECTURES[architecture_name],
        d=d,
        u=u,
    )
    rng = np.random.default_rng(seed)
    mean = rng.normal(scale=0.05, size=net.param_size)
    std = np.full(net.param_size, 0.15, dtype=float)
    elite_count = max(2, int(round(population_size * elite_frac)))

    best_score = -np.inf
    best_params = mean.copy()

    for it in range(n_iters):
        candidates = mean + rng.standard_normal((population_size, net.param_size)) * std
        scores = np.empty(population_size, dtype=float)
        for idx, cand in enumerate(candidates):
            score, _ = evaluate_candidate(
                cand, net, mu_vec, omega_mat, r, w0, goal_mult,
                n_paths=n_paths, n_steps=n_steps,
                max_long=max_long_leverage, max_short=max_short_leverage,
                utility=utility, asp_p=asp_p, asp_c1=asp_c1, asp_R=asp_R,
                seed=seed * 1000 + it * population_size + idx,
            )
            scores[idx] = score
        elite_idx = np.argsort(scores)[-elite_count:]
        elite = candidates[elite_idx]
        mean = elite.mean(axis=0)
        std = np.maximum(elite.std(axis=0), 0.02)
        if scores[elite_idx[-1]] > best_score:
            best_score = float(scores[elite_idx[-1]])
            best_params = candidates[elite_idx[-1]].copy()

    net.params = best_params
    return net, {
        'architecture_name': architecture_name,
        'best_objective': best_score,
        'param_size': net.param_size,
        'hidden_layers': net.hidden_layers,
    }
