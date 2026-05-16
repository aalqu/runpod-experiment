import csv
from pathlib import Path

from fd_core import make_fd_policy

from .artifacts import load_fd_artifact, load_torch_model_artifact
from .io import load_run_result

try:
    from .torch_nn_models import policy_weights as torch_policy_weights
    _HAS_TORCH_API = True
except ImportError:
    torch_policy_weights = None
    _HAS_TORCH_API = False


def _results_dir(results_dir=None):
    if results_dir is None:
        return Path(__file__).resolve().parents[1] / 'results'
    return Path(results_dir)


def load_summary_table(name='main_results', results_dir=None):
    path = _results_dir(results_dir) / 'summary' / f'{name}.csv'
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def list_available_runs(results_dir=None):
    raw_dir = _results_dir(results_dir) / 'raw'
    runs = []
    for path in sorted(raw_dir.glob('*.npz')):
        stem = path.stem
        parts = stem.split('_')
        # method may contain underscores; parse from tail
        w = parts[-1]
        seed = parts[-2]
        n = parts[-3]
        method = '_'.join(parts[:-3])
        runs.append({
            'method_name': method,
            'n_assets': int(n[1:]),
            'seed': int(seed.replace('seed', '')),
            'initial_wealth': float(w[1:]),
            'path': str(path),
        })
    return runs


def _run_path(method_name, n_assets, seed, initial_wealth, results_dir=None):
    return _results_dir(results_dir) / 'raw' / f'{method_name}_n{n_assets}_seed{seed}_w{initial_wealth:.2f}.npz'


def _artifact_dir(results_dir=None):
    return _results_dir(results_dir) / 'artifacts'


def load_fd_policy_bundle(n_assets, seed=1, initial_wealth=1.0, results_dir=None):
    result = load_run_result(_run_path('fd_1d_proxy', n_assets, seed, initial_wealth, results_dir))
    artifact_path = _artifact_dir(results_dir) / f'fd_1d_proxy_n{n_assets}_seed{seed}_w{initial_wealth:.2f}_fd_policy.npz'
    artifact = load_fd_artifact(artifact_path)
    policy = make_fd_policy(artifact['w_grid'], artifact['pi_grid'], d=float(artifact['d']), u=float(artifact['u']))
    return {'result': result, 'artifact': artifact, 'policy': policy}


def load_nn_model_bundle(method_name, n_assets, seed=1, initial_wealth=1.0, results_dir=None):
    if not _HAS_TORCH_API:
        raise ImportError("PyTorch is required to load NN model bundles.")
    result = load_run_result(_run_path(method_name, n_assets, seed, initial_wealth, results_dir))
    artifact_path = _artifact_dir(results_dir) / f'{method_name}_n{n_assets}_seed{seed}_w{initial_wealth:.2f}_model.pt'
    model, metadata = load_torch_model_artifact(artifact_path)
    return {
        'result': result,
        'model': model,
        'metadata': metadata,
        'weights_fn': lambda current_wealth, goal, history=None, step_idx=0, total_steps=252: torch_policy_weights(
            model,
            current_wealth,
            goal,
            history=history,
            step_idx=step_idx,
            total_steps=total_steps,
        ),
    }
