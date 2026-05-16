from pathlib import Path
from typing import Dict

import numpy as np

from .io import ensure_dir

HAS_TORCH = False
try:
    import torch
    from .torch_nn_models import TORCH_ARCHITECTURES, _build_model, _device
    HAS_TORCH = True
except ImportError:
    torch = None
    TORCH_ARCHITECTURES = {}
    _build_model = None
    _device = None


def artifact_filename(method_name: str, n_assets: int, seed: int, initial_wealth: float, suffix: str):
    return f"{method_name}_n{n_assets}_seed{seed}_w{initial_wealth:.2f}_{suffix}"


def save_fd_artifact(path, w_grid, pi_grid, metadata: Dict):
    path = Path(path)
    ensure_dir(path.parent)
    payload = {'w_grid': np.asarray(w_grid, dtype=float), 'pi_grid': np.asarray(pi_grid, dtype=float)}
    payload.update(metadata)
    np.savez_compressed(path, **payload)
    return path


def load_fd_artifact(path):
    data = np.load(Path(path), allow_pickle=True)
    out = {}
    for key in data.files:
        value = data[key]
        out[key] = value.item() if value.ndim == 0 else value
    return out


def save_torch_model_artifact(path, model, metadata: Dict):
    if not HAS_TORCH:
        raise ImportError("PyTorch is required to save model artifacts.")
    path = Path(path)
    ensure_dir(path.parent)
    payload = {
        'state_dict': model.state_dict(),
        'metadata': metadata,
    }
    torch.save(payload, path)
    return path


def load_torch_model_artifact(path, map_location=None):
    if not HAS_TORCH:
        raise ImportError("PyTorch is required to load model artifacts.")
    payload = torch.load(Path(path), map_location=map_location or _device('cpu'))
    meta = dict(payload['metadata'])
    architecture_name = meta['architecture_name']
    if architecture_name not in TORCH_ARCHITECTURES:
        raise ValueError(f'Unknown architecture in artifact: {architecture_name}')
    model = _build_model(
        architecture_name=architecture_name,
        n_assets=int(meta['n_assets']),
        n_steps=int(meta.get('n_steps', 32)),
        d=float(meta.get('d', -5.0)),
        u=float(meta.get('u', 3.0)),
    )
    model.load_state_dict(payload['state_dict'])
    model.eval()
    return model, meta
