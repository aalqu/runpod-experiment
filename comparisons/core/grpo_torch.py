"""
comparisons/core/grpo_torch.py
-------------------------------
PyTorch-CUDA ES-GRPO trainer.

Vectorises over the G candidate axis using `torch.func.vmap` +
`torch.func.functional_call` (PyTorch ≥ 2.0).  Falls back to a GPU-vectorised
Python loop on older runtimes — still fast because each loop iteration is a
fully-batched GBM simulation.

See GRPO_INTEGRATION_SPEC.md §2.2 and §4.

Public API:
    train_es_grpo_torch(mu_vec, omega_mat, r, ...)  ->  (model, history_dict)
    grpo_policy_weights(net, W_current, goal, ...)  ->  np.ndarray  (weights)
"""

import math
import time
import warnings
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# vmap + functional_call available in PyTorch ≥ 2.0
try:
    from torch.func import vmap, functional_call  # type: ignore
    HAS_VMAP = True
except ImportError:
    HAS_VMAP = False

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _require_torch():
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for grpo_torch.")


# ── Internal helpers ─────────────────────────────────────────────────────────

def _device(device):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _params_to_flat(model: "nn.Module"):
    """Extract all trainable parameters as a single flat tensor + metadata."""
    names, shapes, sizes = [], [], []
    chunks = []
    for name, p in model.named_parameters():
        names.append(name)
        shapes.append(p.shape)
        sizes.append(p.numel())
        chunks.append(p.data.detach().flatten())
    flat = torch.cat(chunks) if chunks else torch.empty(0)
    return flat, names, shapes, sizes


def _flat_to_params_dict(flat: "torch.Tensor", names, shapes, sizes):
    """Reconstruct named parameter dict from a flat vector."""
    result = {}
    idx = 0
    for name, shape, size in zip(names, shapes, sizes):
        result[name] = flat[idx: idx + size].reshape(shape)
        idx += size
    return result


def _apply_leverage_torch(pi: "torch.Tensor", d: float, u: float,
                           max_long: float, max_short: float) -> "torch.Tensor":
    """Per-asset clamp + aggregate leverage caps. (B, n) → (B, n)."""
    pi        = pi.clamp(d, u)
    long_lev  = pi.clamp(min=0.0).sum(dim=1, keepdim=True).clamp(min=1e-12)
    short_lev = (-pi).clamp(min=0.0).sum(dim=1, keepdim=True).clamp(min=1e-12)
    long_scale  = (max_long  / long_lev).clamp(max=1.0)
    short_scale = (max_short / short_lev).clamp(max=1.0)
    return torch.where(pi >= 0, pi * long_scale, pi * short_scale)


def _rollout_loop(
    model: "nn.Module",
    params_flat: "torch.Tensor",
    names, shapes, sizes,
    Z: "torch.Tensor",        # (G, n_paths, n_steps, n_assets)
    mu_t: "torch.Tensor",     # (n_assets,)  — annualised excess returns
    sig_t: "torch.Tensor",    # (n_assets, n_assets) — Cholesky of ANNUAL Omega
    r: float,
    dt: float,                # years per step  (= T / n_steps)
    goal: float,
    w0: float,
    n_steps: int,
    d: float, u: float,
    max_long: float, max_short: float,
) -> "torch.Tensor":          # (G, n_paths)
    """
    Simulate G candidate policies via a Python loop over G.

    Uses the annualised arithmetic-return convention that matches
    torch_nn_models.train_torch_policy_net and fd_core:
        dS = mu_ann * dt + chol(Omega) @ Z * sqrt(dt)
    (Issues 2, 4, 10 of READINESS_AUDIT.md)
    """
    G, n_paths = Z.shape[0], Z.shape[1]
    dev       = params_flat.device
    cand_params = params_flat  # (G, P)

    W_all = torch.empty(G, n_paths, device=dev)
    r_dt    = r * dt
    sqrt_dt = math.sqrt(dt)

    for i in range(G):
        params_dict_i = _flat_to_params_dict(cand_params[i], names, shapes, sizes)
        W = torch.full((n_paths,), w0, dtype=params_flat.dtype, device=dev)
        Z_i = Z[i]   # (n_paths, n_steps, n_assets)

        for s in range(n_steps):
            tau_norm = max((n_steps - s) / n_steps, 1.0 / n_steps)
            w_norm   = (W / max(goal, 1e-12)).clamp(1e-6, 10.0)
            feats    = torch.stack([w_norm, w_norm.new_full(w_norm.shape, tau_norm)], dim=1)
            with torch.no_grad():
                raw_pi = functional_call(model, params_dict_i, (feats,))
            pi = _apply_leverage_torch(raw_pi, d, u, max_long, max_short)

            # Arithmetic GBM step — matches torch_nn_models convention
            dS    = mu_t * dt + (Z_i[:, s, :] @ sig_t.T) * sqrt_dt   # (n_paths, n)
            excess = (pi * (dS - r_dt)).sum(dim=1)
            W      = (W * (1.0 + r_dt + excess)).clamp(min=1e-6)

        W_all[i] = W.detach()
    return W_all


def _rollout_vmap(
    model: "nn.Module",
    cand_params_flat: "torch.Tensor",   # (G, P)
    names, shapes, sizes,
    Z: "torch.Tensor",                  # (G, n_paths, n_steps, n_assets)
    mu_t: "torch.Tensor",               # (n_assets,) annualised excess returns
    sig_t: "torch.Tensor",              # (n_assets, n_assets) annual Cholesky
    r: float, dt: float,
    goal: float, w0: float, n_steps: int,
    d: float, u: float,
    max_long: float, max_short: float,
) -> "torch.Tensor":                    # (G, n_paths)
    """
    Simulate G candidates in one batched vmap call.

    Uses annualised arithmetic-return convention to match _rollout_loop,
    torch_nn_models, and fd_core (Issues 2, 4, 10 of READINESS_AUDIT.md).
    """
    sqrt_dt = math.sqrt(dt)
    r_dt    = r * dt

    # Buffer keys for model (buffers don't change across candidates)
    buffers = dict(model.named_buffers())

    def _single_candidate(theta_flat_i: "torch.Tensor",
                          Z_i: "torch.Tensor") -> "torch.Tensor":
        # theta_flat_i: (P,)  Z_i: (n_paths, n_steps, n_assets)
        params_d = _flat_to_params_dict(theta_flat_i, names, shapes, sizes)

        W = torch.full((Z_i.shape[0],), w0,
                       dtype=theta_flat_i.dtype, device=theta_flat_i.device)

        for s in range(n_steps):
            tau_norm = float(max((n_steps - s) / n_steps, 1.0 / n_steps))
            w_norm   = (W / max(goal, 1e-12)).clamp(1e-6, 10.0)
            feats    = torch.stack(
                [w_norm, torch.full_like(w_norm, tau_norm)], dim=1
            )
            raw_pi = functional_call(model, (params_d, buffers), (feats,))
            pi     = _apply_leverage_torch(raw_pi, d, u, max_long, max_short)

            # Arithmetic GBM step — matches torch_nn_models convention
            dS    = mu_t * dt + (Z_i[:, s, :] @ sig_t.T) * sqrt_dt
            excess = (pi * (dS - r_dt)).sum(dim=1)
            W      = (W * (1.0 + r_dt + excess)).clamp(min=1e-6)

        return W  # (n_paths,)

    # vmap over (candidate params, candidate noise)
    return vmap(_single_candidate, in_dims=(0, 0))(cand_params_flat, Z)


# ── Supervised Browne warm-start (mirrors torch_nn_models.py) ────────────────

def _pretrain_browne_torch(
    model: "nn.Module",
    omega_inv_eta: "torch.Tensor",
    theta2: float,
    n_iters: int,
    lr: float = 1e-3,
    batch_size: int = 512,
    d: float = -5.0,
    u: float = 3.0,
    dev: "torch.device" = None,
    verbose: bool = False,
):
    """Supervised MSE towards the Browne (1995) policy for `n_iters` steps."""
    if n_iters <= 0:
        return
    if dev is None:
        dev = next(model.parameters()).device

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    oin = omega_inv_eta.to(dev)   # (n,)

    for pit in range(n_iters):
        w_norm = torch.rand(batch_size, 1, device=dev) * 1.5 + 0.5
        tau    = torch.rand(batch_size, 1, device=dev).clamp(min=0.05)

        log_w = torch.log(w_norm.clamp(min=1e-8))
        denom = 1.0 + log_w / (theta2 * tau).clamp(min=1e-8)
        scale = (1.0 / denom.clamp(min=1e-3)).clamp(-3.0, 3.0)
        target = (scale * oin.unsqueeze(0)).clamp(d, u)   # (B, n)

        feats = torch.cat([w_norm, tau], dim=1)            # (B, 2)
        pred  = model(feats)
        loss  = (pred - target).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    if verbose:
        print(f"  [ES-GRPO pretrain] final MSE: {loss.item():.4f}")


# ── Main trainer ─────────────────────────────────────────────────────────────

def train_es_grpo_torch(
    mu_vec,
    omega_mat,
    r: float,
    *,
    architecture_name: str = 'es_grpo',
    w0: float = 1.0,
    goal_mult: float = 1.10,
    G: int = 32,
    sigma_es: float = 0.03,
    lr: float = 3e-3,
    n_paths: int = 256,
    n_iters: int = 300,
    n_steps: int = 40,
    d: float = -5.0,
    u: float = 3.0,
    max_long_leverage: float = 3.0,
    max_short_leverage: float = 5.0,
    pretrain_iters: int = 100,
    p_curriculum: float = 0.30,
    patience: int = 60,
    T: float = 1.0,
    seed: int = 1,
    device=None,
    use_amp: bool = True,
    verbose: bool = False,
):
    """
    Train an ES-GRPO policy network on calibrated GBM.

    Parameters
    ----------
    mu_vec, omega_mat, r : market calibration (annualised excess returns, cov, rf)
    architecture_name    : key in TORCH_ARCHITECTURES (must have kind='es_grpo')
    G                    : group size (must be even for antithetic noise)
    sigma_es             : perturbation std for ES
    lr                   : ES learning rate (applied to ES gradient, not via Adam)
    n_paths              : GBM paths per candidate per iter
    n_iters              : total ES iterations
    n_steps              : steps per trajectory (scaled with T)
    pretrain_iters       : supervised Browne warm-start iterations
    p_curriculum         : fraction of rollout paths near goal boundary
    patience             : early-stopping patience (iter count, not val-check count)
    T                    : horizon in years
    seed                 : random seed
    use_amp              : enable AMP for forward rollout on CUDA
    verbose              : print training progress

    Returns
    -------
    (model, history_dict) — same shape as train_torch_policy_net.
    history_dict keys: loss_history, val_history, val_iters, test_u,
                       param_size, backend, device, architecture_name.
    """
    _require_torch()
    assert G % 2 == 0, f"G must be even for antithetic noise; got G={G}"

    # ── Setup ─────────────────────────────────────────────────────────────────
    from comparisons.core.torch_nn_models import (
        TORCH_ARCHITECTURES, _build_model, _apply_leverage_constraint_torch,
    )

    dev = _device(device)
    torch.manual_seed(seed)
    if dev.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    # Issue 7 of READINESS_AUDIT.md: use a SEPARATE generator for validation so
    # the training-noise RNG state is not advanced by validation calls, preserving
    # reproducibility when validation is toggled on/off.
    gen_val = torch.Generator(device=dev)
    gen_val.manual_seed(seed + 1)

    mu_np    = np.asarray(mu_vec, dtype=np.float32)
    omega_np = np.asarray(omega_mat, dtype=np.float32)
    n_assets = mu_np.shape[0]
    goal     = float(w0 * goal_mult)

    # Market tensors — ANNUALISED convention, matching torch_nn_models and fd_core
    # (Bug 2 / Issue 10 of READINESS_AUDIT.md: previous code used daily μ and daily
    # Chol combined with annual dt, causing vol to be √252× too small per step)
    eta              = mu_np - r                          # (n,) annualised excess return
    L_chol_ann       = np.linalg.cholesky(
        omega_np + 1e-10 * np.eye(n_assets)
    ).astype(np.float32)                                  # annual Cholesky
    omega_inv_eta_np = np.linalg.solve(omega_np, eta)
    theta2 = max(float(np.dot(eta, omega_inv_eta_np)), 1e-12)

    mu_t   = torch.tensor(mu_np, device=dev)             # (n,) annualised excess return
    sig_t  = torch.tensor(L_chol_ann, device=dev)        # (n, n) annual Cholesky
    oin_t  = torch.tensor(omega_inv_eta_np, device=dev)

    dt     = T / n_steps
    r_dt   = r * dt

    # ── Build model ───────────────────────────────────────────────────────────
    # ES-GRPO uses TorchPolicyNet (basic 2-feature MLP) — identical class to
    # gradient NN methods; the training algorithm differs.
    arch_spec = TORCH_ARCHITECTURES.get(architecture_name, {})
    arch_constraints = arch_spec.get('constraints', {})
    if arch_constraints:
        d                  = arch_constraints.get('d',         d)
        u                  = arch_constraints.get('u',         u)
        max_long_leverage  = arch_constraints.get('max_long',  max_long_leverage)
        max_short_leverage = arch_constraints.get('max_short', max_short_leverage)

    model = _build_model(architecture_name, n_assets, n_steps, d, u)
    model = model.to(dev).float()
    model.eval()   # ES doesn't use dropout / batchnorm train mode

    # ── Phase 1: Browne supervised warm-start ─────────────────────────────────
    if pretrain_iters > 0:
        if verbose:
            print(f"  [ES-GRPO] Phase 1: Browne pretrain ({pretrain_iters} iters)")
        _pretrain_browne_torch(
            model, oin_t, theta2, n_iters=pretrain_iters,
            lr=1e-3, batch_size=512, d=d, u=u, dev=dev, verbose=verbose,
        )

    # ── Flatten params ────────────────────────────────────────────────────────
    theta, param_names, param_shapes, param_sizes = _params_to_flat(model)
    theta = theta.to(dev)

    # ── Phase 2: ES-GRPO training ─────────────────────────────────────────────
    loss_history = []
    val_history  = []
    val_iters    = []
    best_val     = -math.inf
    best_theta   = theta.clone()
    no_improve   = 0

    half = G // 2
    amp_ctx = (
        torch.amp.autocast(device_type='cuda')
        if (dev.type == 'cuda' and use_amp)
        else __import__('contextlib').nullcontext()
    )

    if verbose:
        print(f"  [ES-GRPO] Phase 2: ES training (G={G}, σ={sigma_es}, "
              f"lr={lr}, iters={n_iters}, paths={n_paths}, steps={n_steps})")

    for it in range(n_iters):
        # ── Antithetic perturbations ──────────────────────────────────────────
        P = theta.numel()
        eps_half = torch.randn(half, P, generator=gen, device=dev)
        eps      = torch.cat([eps_half, -eps_half], dim=0)   # (G, P)

        cand_params = theta.unsqueeze(0) + sigma_es * eps    # (G, P)

        # NOTE (Issue 6 of READINESS_AUDIT.md): per-path curriculum w0 sampling
        # requires threading a (n_paths,) w0 vector through the rollout functions.
        # The vmap path maps over (G, n_paths) with a scalar w0; adding a per-path
        # dimension changes the vmap signature non-trivially.
        # For now, all paths start at the fixed w0; curriculum is a future improvement.

        # ── Pre-generate all GBM noise: (G, n_paths, n_steps, n_assets) ────
        with amp_ctx:
            Z = torch.randn(G, n_paths, n_steps, n_assets,
                            generator=gen, device=dev)

            # Rollout: (G, n_paths) terminal wealth
            if HAS_VMAP and dev.type in ('cuda', 'cpu'):
                try:
                    W_T = _rollout_vmap(
                        model, cand_params, param_names, param_shapes, param_sizes,
                        Z, mu_t, sig_t, r, dt, goal, w0, n_steps,
                        d, u, max_long_leverage, max_short_leverage,
                    )
                except (RuntimeError, torch.autograd.graph.GradCheckError) as _vmap_err:
                    # Issue 5 of READINESS_AUDIT.md: log vmap fallback so failures
                    # are visible in CI and RunPod logs rather than silently degrading
                    # to the slower loop path.
                    warnings.warn(
                        f"[ES-GRPO] vmap rollout failed ({type(_vmap_err).__name__}: "
                        f"{_vmap_err}); falling back to Python loop.",
                        RuntimeWarning, stacklevel=2,
                    )
                    W_T = _rollout_loop(
                        model, cand_params, param_names, param_shapes, param_sizes,
                        Z, mu_t, sig_t, r, dt, goal, w0, n_steps,
                        d, u, max_long_leverage, max_short_leverage,
                    )
            else:
                W_T = _rollout_loop(
                    model, cand_params, param_names, param_shapes, param_sizes,
                    Z, mu_t, sig_t, r, dt, goal, w0, n_steps,
                    d, u, max_long_leverage, max_short_leverage,
                )

        # ── Binary rewards (exact indicator, no surrogate) ─────────────────
        rewards = (W_T >= goal).float().mean(dim=1)   # (G,)

        mu_r = rewards.mean()
        sd_r = rewards.std(unbiased=False).clamp(min=1e-8)
        adv  = (rewards - mu_r) / sd_r               # (G,)

        # ── ES gradient + update ──────────────────────────────────────────────
        # g = (1 / (G * sigma_es)) * sum_i A_i * eps_i
        # (mean over G, so the 1/G factor is already in .mean(dim=0))
        g     = (adv.unsqueeze(1) * eps).mean(dim=0) / sigma_es   # (P,)
        theta = theta + lr * g

        loss_history.append(float(mu_r))

        # ── Validation & early stopping every 10 iters ────────────────────────
        if it % 10 == 0:
            with torch.no_grad():
                # Load best-so-far params for validation
                params_d_val = _flat_to_params_dict(theta, param_names,
                                                    param_shapes, param_sizes)
                _load_params(model, params_d_val)

                Z_val = torch.randn(1, 256, n_steps, n_assets,
                                    generator=gen_val, device=dev)
                W_val = _rollout_loop(
                    model, theta.unsqueeze(0), param_names, param_shapes, param_sizes,
                    Z_val, mu_t, sig_t, r, dt, goal, w0, n_steps,
                    d, u, max_long_leverage, max_short_leverage,
                ).squeeze(0)   # (256,)
            val_r = float((W_val >= goal).float().mean())
            val_history.append(val_r)
            val_iters.append(it)

            if val_r > best_val + 1e-5:
                best_val   = val_r
                best_theta = theta.clone()
                no_improve = 0
            else:
                no_improve += 10

            if verbose and it % 50 == 0:
                print(f"  [ES-GRPO] iter {it:4d}  train={float(mu_r):.4f}  "
                      f"val={val_r:.4f}  best_val={best_val:.4f}")

        if no_improve >= patience:
            if verbose:
                print(f"  [ES-GRPO] early stop at iter {it+1}")
            break

    # ── Restore best params ───────────────────────────────────────────────────
    params_d_best = _flat_to_params_dict(best_theta, param_names,
                                         param_shapes, param_sizes)
    _load_params(model, params_d_best)
    model.eval()

    # ── Final test evaluation ─────────────────────────────────────────────────
    with torch.no_grad():
        gen_test = torch.Generator(device=dev)
        gen_test.manual_seed(seed + 9999)
        Z_test   = torch.randn(1, 1024, n_steps, n_assets,
                               generator=gen_test, device=dev)
        W_test   = _rollout_loop(
            model, best_theta.unsqueeze(0), param_names, param_shapes, param_sizes,
            Z_test, mu_t, sig_t, r, dt, goal, w0, n_steps,
            d, u, max_long_leverage, max_short_leverage,
        ).squeeze(0)
    test_u = float((W_test >= goal).float().mean())

    param_count = sum(p.numel() for p in model.parameters())
    history = {
        'architecture_name': architecture_name,
        'backend'          : 'torch_es_grpo',
        'device'           : str(dev),
        'param_size'       : int(param_count),
        'loss_history'     : loss_history,
        'val_history'      : val_history,
        'val_iters'        : val_iters,
        'test_u'           : test_u,
        # ES-specific diagnostics
        'G'                : G,
        'sigma_es'         : sigma_es,
        'iters_run'        : len(loss_history),
    }
    return model, history


def _load_params(model: "nn.Module", params_dict: dict):
    """Load a params dict back into the model's parameter tensors."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in params_dict:
                p.copy_(params_dict[name])


# ── Inference helper ──────────────────────────────────────────────────────────

def grpo_policy_weights(
    net: "nn.Module",
    W_current,
    goal: float,
    step_idx: int = 0,
    total_steps: int = 252,
    device=None,
) -> "np.ndarray":
    """
    Query the trained ES-GRPO policy at the current wealth level.

    Parameters
    ----------
    net        : trained model returned by train_es_grpo_torch
    W_current  : float or (N,) — current wealth
    goal       : float — wealth target
    step_idx   : int — current step index (for tau computation)
    total_steps: int — total steps in the horizon

    Returns
    -------
    (n_assets,) or (N, n_assets) portfolio weights
    """
    _require_torch()
    dev = _device(device) if device is not None else next(net.parameters()).device
    W_np = np.asarray(W_current, dtype=np.float32).ravel()
    is_scalar = W_np.size == 1

    with torch.no_grad():
        W = torch.tensor(W_np, dtype=torch.float32, device=dev).unsqueeze(1)  # (N,1)
        tau_norm = max((total_steps - step_idx) / total_steps, 1.0 / total_steps)
        w_norm   = (W / max(goal, 1e-12)).clamp(1e-6, 10.0)
        feats    = torch.cat([w_norm,
                              w_norm.new_full(w_norm.shape, tau_norm)], dim=1)
        pi = net(feats).detach().cpu().numpy()

    return pi.squeeze(0) if is_scalar else pi
