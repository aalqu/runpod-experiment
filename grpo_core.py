"""
grpo_core.py
------------
Pure-NumPy reference implementation of ES-GRPO (Evolution-Strategies with
Group-Relative-Policy-Optimisation).

This file is the single source of truth for the algorithm.  The PyTorch CUDA
trainer in `comparisons/core/grpo_torch.py` is a vectorised port of this.

See GRPO_INTEGRATION_SPEC.md §2.1 and §3 for the algorithm description.

No PyTorch dependency — safe to import anywhere.
"""

import math
import numpy as np
from typing import Callable, Optional, Tuple


# ── NumPy MLP helpers ─────────────────────────────────────────────────────────

def _mlp_forward_np(x: np.ndarray, params: np.ndarray, layer_shapes) -> np.ndarray:
    """
    Forward pass through a simple MLP with tanh hidden activations and
    no output activation (raw logits, used for policy weights before
    leverage clamping).

    Parameters
    ----------
    x            : (B, n_in) — input batch
    params       : (P,) flat parameter vector
    layer_shapes : list of (W_shape, b_shape) per layer

    Returns
    -------
    (B, n_out) — output
    """
    out = x
    idx = 0
    for i, (w_shape, b_shape) in enumerate(layer_shapes):
        nw = int(np.prod(w_shape))
        nb = int(np.prod(b_shape))
        W = params[idx: idx + nw].reshape(w_shape)
        idx += nw
        b = params[idx: idx + nb].reshape(b_shape)
        idx += nb
        out = out @ W.T + b
        if i < len(layer_shapes) - 1:   # hidden layers: tanh
            out = np.tanh(out)
    return out


def _build_layer_shapes(n_in: int, hidden_layers: tuple, n_out: int):
    """Return list of (W_shape, b_shape) for each layer of the MLP."""
    dims = [n_in] + list(hidden_layers) + [n_out]
    return [((dims[i + 1], dims[i]), (dims[i + 1],)) for i in range(len(dims) - 1)]


def _count_params(layer_shapes) -> int:
    total = 0
    for w_shape, b_shape in layer_shapes:
        total += int(np.prod(w_shape)) + int(np.prod(b_shape))
    return total


def _init_params_np(layer_shapes, rng: np.random.Generator) -> np.ndarray:
    """Kaiming-uniform-like initialisation."""
    params = []
    for w_shape, b_shape in layer_shapes:
        fan_in = w_shape[1]
        std = math.sqrt(2.0 / fan_in)
        params.append(rng.standard_normal(int(np.prod(w_shape))).astype(np.float32) * std)
        params.append(np.zeros(int(np.prod(b_shape)), dtype=np.float32))
    return np.concatenate(params)


# ── Leverage constraint (NumPy) ───────────────────────────────────────────────

def _apply_leverage_np(pi: np.ndarray, d: float, u: float,
                       max_long: float, max_short: float) -> np.ndarray:
    """Per-asset box then aggregate long/short caps. Mirrors evaluation.py."""
    pi = np.clip(pi, d, u)
    squeeze = pi.ndim == 1
    if squeeze:
        pi = pi[None, :]
    long_lev  = np.maximum(pi, 0.0).sum(axis=1, keepdims=True).clip(min=1e-12)
    short_lev = np.maximum(-pi, 0.0).sum(axis=1, keepdims=True).clip(min=1e-12)
    pi = np.where(pi >= 0, pi * np.minimum(max_long / long_lev, 1.0),
                           pi * np.minimum(max_short / short_lev, 1.0))
    return pi.squeeze(0) if squeeze else pi


# ── GBM simulator ─────────────────────────────────────────────────────────────

def simulate_gbm_paths_np(
    params: np.ndarray,
    layer_shapes,
    mu: np.ndarray,          # (n,) annualised excess returns
    L_chol: np.ndarray,      # (n, n) Cholesky of ANNUAL Omega
    r: float,
    n_paths: int,
    n_steps: int,
    goal: float,
    w0: float = 1.0,
    T_horizon: float = 1.0,
    d: float = -5.0,
    u: float = 3.0,
    max_long: float = 3.0,
    max_short: float = 5.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Simulate `n_paths` GBM trajectories of length `n_steps` under the MLP
    policy defined by `params`.

    Uses the annualised arithmetic-return convention matching torch_nn_models
    and fd_core (Bug 2 / Issue 10 of READINESS_AUDIT.md):
        dS = mu_ann * dt + chol(Omega) @ Z * sqrt(dt)

    The policy receives a 2-feature input: [w_norm, tau_norm].
    Returns (n_paths,) terminal wealth array.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_assets = len(mu)
    dt       = T_horizon / n_steps
    sqrt_dt  = math.sqrt(dt)
    r_dt     = r * dt

    W    = np.full(n_paths, w0, dtype=np.float64)
    Z    = rng.standard_normal((n_paths, n_steps, n_assets))  # pre-generate all noise

    for s in range(n_steps):
        tau_norm = max((n_steps - s) / n_steps, 1.0 / n_steps)
        w_norm   = (W / max(goal, 1e-12)).clip(1e-6, 10.0)
        feats    = np.column_stack([w_norm, np.full(n_paths, tau_norm)])  # (B, 2)
        pi_raw   = _mlp_forward_np(feats.astype(np.float32), params, layer_shapes)
        pi       = _apply_leverage_np(pi_raw.astype(np.float64), d, u, max_long, max_short)

        # Arithmetic GBM step — annualised convention
        dS     = mu * dt + (Z[:, s, :] @ L_chol.T) * sqrt_dt   # (n_paths, n)
        excess = (pi * (dS - r_dt)).sum(axis=1)
        W      = np.maximum(W * (1.0 + r_dt + excess), 1e-6)

    return W


# ── Core ES-GRPO step ─────────────────────────────────────────────────────────

def es_grpo_step_numpy(
    params: np.ndarray,
    forward_fn: Callable,
    rewards_fn: Callable,
    sigma_es: float,
    lr: float,
    G: int,
    n_paths: int,
    n_steps: int,
    goal: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float, float]:
    """
    One ES-GRPO update step.

    Algorithm (§2.1 of GRPO_INTEGRATION_SPEC.md):
      1. Draw G/2 antithetic noise vectors ε_i ~ N(0, I) of shape (P,).
      2. Form G candidate parameters: θ_i = θ + σ_es · ε_i.
      3. Evaluate r_i = E[U(W_T) | θ_i] for each candidate via forward_fn.
      4. Group-normalise: A_i = (r_i - mean(r)) / (std(r) + 1e-8).
      5. ES gradient: g = (1 / (G · σ_es)) · Σ_i A_i · ε_i.
      6. Update: θ ← θ + lr · g.

    Parameters
    ----------
    params     : (P,) current parameter vector
    forward_fn : Callable[[np.ndarray, np.random.Generator, int, int], np.ndarray]
                 forward_fn(params_i, rng, n_paths, n_steps) -> (n_paths,) terminal W
    rewards_fn : Callable[[np.ndarray, float], float]
                 rewards_fn(terminal_W, goal) -> scalar reward
    sigma_es   : perturbation std
    lr         : learning rate
    G          : group size (must be even for antithetic)
    n_paths    : MC paths per candidate
    n_steps    : trajectory length
    goal       : wealth target
    rng        : numpy Generator (state is advanced in-place)

    Returns
    -------
    new_params : (P,)
    mean_reward: float
    std_reward : float
    """
    assert G % 2 == 0, f"G must be even for antithetic noise, got G={G}"
    P    = params.shape[0]
    half = G // 2

    # Step 1 & 2 — antithetic candidate parameters
    eps_half = rng.standard_normal((half, P)).astype(np.float32)
    eps      = np.concatenate([eps_half, -eps_half], axis=0)    # (G, P)
    cand_params = params[None, :] + sigma_es * eps               # (G, P)

    # Step 3 — evaluate each candidate
    rewards = np.empty(G, dtype=np.float64)
    for i in range(G):
        wT         = forward_fn(cand_params[i], rng, n_paths, n_steps)
        rewards[i] = rewards_fn(wT, goal)

    # Step 4 — group-normalised advantages
    mu_r = float(rewards.mean())
    sd_r = float(max(rewards.std(), 1e-8))
    adv  = (rewards - mu_r) / sd_r    # (G,)

    # Step 5 & 6 — ES gradient and update
    g          = (adv[:, None] * eps).mean(axis=0) / sigma_es  # (P,)
    new_params = params + lr * g

    return new_params, mu_r, sd_r


# ── Browne warm-start ─────────────────────────────────────────────────────────

def _browne_policy_np_grpo(
    w_norm: np.ndarray,     # (B,)
    tau: np.ndarray,        # (B,) or scalar
    omega_inv_eta: np.ndarray,  # (n,)
    theta2: float,
    d: float = -5.0,
    u: float = 3.0,
) -> np.ndarray:
    """
    Browne (1995) optimal policy scaled to the unit-leverage direction.
    Mirrors `_browne_policy_np` from `torch_nn_models.py`.
    """
    tau = np.asarray(tau, dtype=float)
    log_w_norm = np.log(np.clip(w_norm, 1e-8, None))
    denom = 1.0 + log_w_norm / np.clip(theta2 * tau, 1e-8, None)
    scale_1d = np.clip(1.0 / np.clip(denom, 1e-3, None), -3.0, 3.0)  # (B,)
    pi = scale_1d[:, None] * omega_inv_eta[None, :]                    # (B, n)
    return np.clip(pi, d, u)


def _pretrain_browne_np(
    params: np.ndarray,
    layer_shapes,
    omega_inv_eta: np.ndarray,
    theta2: float,
    n_iters: int,
    lr: float = 1e-3,
    batch_size: int = 512,
    d: float = -5.0,
    u: float = 3.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Supervised pre-training to initialise the MLP near the Browne policy.
    Uses SGD with MSE loss.

    Returns updated params.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if n_iters <= 0:
        return params

    P = len(params)
    for _ in range(n_iters):
        w_norm = rng.uniform(0.5, 1.5, batch_size).astype(np.float32)
        tau_v  = rng.uniform(0.05, 1.0, batch_size).astype(np.float32)
        feats  = np.column_stack([w_norm, tau_v])

        target = _browne_policy_np_grpo(w_norm, tau_v, omega_inv_eta, theta2, d, u)

        pred   = _mlp_forward_np(feats, params, layer_shapes)  # (B, n)

        # MSE gradient (closed-form for last layer; approximate with finite diff for full net)
        # Use a simple numerical gradient for the NumPy reference
        # (the PyTorch version uses autograd)
        eps_fd = 1e-3
        grad   = np.zeros_like(params)
        base_loss = np.mean((pred - target.astype(np.float32)) ** 2)
        for j in range(P):
            params[j] += eps_fd
            pred_p     = _mlp_forward_np(feats, params, layer_shapes)
            loss_p     = np.mean((pred_p - target.astype(np.float32)) ** 2)
            grad[j]    = (loss_p - base_loss) / eps_fd
            params[j] -= eps_fd
        params -= lr * grad

    return params


# ── Rewards function ──────────────────────────────────────────────────────────

def binary_goal_reward(terminal_wealth: np.ndarray, goal: float) -> float:
    """Binary indicator: E[1{W_T >= goal}]. The exact objective, no surrogate."""
    return float((terminal_wealth >= goal).mean())


# ── Main training loop ────────────────────────────────────────────────────────

def train_es_grpo_numpy(
    init_params: Optional[np.ndarray],
    forward_fn: Callable,
    rewards_fn: Callable,
    *,
    sigma_es: float = 0.03,
    lr: float = 3e-3,
    G: int = 32,
    n_iters: int = 300,
    n_paths: int = 256,
    n_steps: int = 40,
    goal: float = 1.10,
    layer_shapes=None,
    omega_inv_eta: Optional[np.ndarray] = None,
    theta2: float = 0.04,
    d: float = -5.0,
    u: float = 3.0,
    pretrain_iters: int = 0,
    w0_sampler: Optional[Callable] = None,
    seed: int = 1,
    patience: int = 60,
    val_fn: Optional[Callable] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Train ES-GRPO from scratch.

    Parameters
    ----------
    init_params    : (P,) initial flat params; if None, random init via layer_shapes
    forward_fn     : (params_i, rng, n_paths, n_steps) -> (n_paths,) terminal W
    rewards_fn     : (terminal_W, goal) -> scalar
    layer_shapes   : list of (W_shape, b_shape) — required if init_params is None
    omega_inv_eta  : (n,) for Browne warm-start; skipped if None or pretrain_iters=0
    val_fn         : called every 10 iters with (params) -> scalar val reward

    Returns
    -------
    (final_params, history_dict)
    history_dict keys: 'loss_history', 'reward_std', 'val_history', 'val_iters',
                       'iters_run', 'test_u'
    """
    rng = np.random.default_rng(seed)

    # Initialise params
    if init_params is None:
        assert layer_shapes is not None, "layer_shapes required when init_params is None"
        params = _init_params_np(layer_shapes, rng)
    else:
        params = init_params.copy().astype(np.float32)

    # Browne supervised pre-training (optional)
    if pretrain_iters > 0 and layer_shapes is not None and omega_inv_eta is not None:
        params = _pretrain_browne_np(
            params, layer_shapes, omega_inv_eta, theta2,
            n_iters=pretrain_iters, lr=1e-3, rng=rng, d=d, u=u,
        )

    loss_history = []
    reward_std   = []
    val_history  = []
    val_iters    = []
    best_val     = -math.inf
    best_params  = params.copy()
    no_improve   = 0

    for it in range(n_iters):
        params, mu_r, sd_r = es_grpo_step_numpy(
            params, forward_fn, rewards_fn,
            sigma_es=sigma_es, lr=lr, G=G,
            n_paths=n_paths, n_steps=n_steps,
            goal=goal, rng=rng,
        )
        loss_history.append(mu_r)
        reward_std.append(sd_r)

        # Validation & early stopping every 10 iters
        if it % 10 == 0:
            val_r = val_fn(params) if val_fn is not None else mu_r
            val_history.append(val_r)
            val_iters.append(it)

            if val_r > best_val + 1e-5:
                best_val    = val_r
                best_params = params.copy()
                no_improve  = 0
            else:
                no_improve += 10

        if no_improve >= patience:
            break

    # Restore best
    params = best_params

    # Final test evaluation (held-out estimate)
    test_rng = np.random.default_rng(seed + 9999)
    wT_test  = forward_fn(params, test_rng, 1024, n_steps)
    test_u   = float(rewards_fn(wT_test, goal))

    history = {
        'loss_history': loss_history,
        'reward_std'  : reward_std,
        'val_history' : val_history,
        'val_iters'   : val_iters,
        'iters_run'   : len(loss_history),
        'test_u'      : test_u,
        'backend'     : 'numpy_es_grpo',
    }
    return params, history
