from dataclasses import dataclass, field
from pathlib import Path
from typing import List


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


@dataclass
class BenchmarkConfig:
    n_assets_list: List[int] = field(default_factory=lambda: [5, 10, 20])
    start_date: str = None
    end_date: str = None
    initial_wealth_levels: List[float] = field(default_factory=lambda: [1.0])
    target_multiplier: float = 1.10
    random_seeds: List[int] = field(default_factory=lambda: [1, 2, 3])
    results_dir: Path = DEFAULT_RESULTS_DIR
    include_fd_benchmark: bool = True
    include_merton_benchmark: bool = True   # multi-asset Merton FD proxy
    include_nn: bool = False
    nn_architectures: List[str] = field(default_factory=lambda: [
        "nn_mlp_small",
        "nn_mlp_deep",
        "deep_bsde",
        "pinn",
        "actor_critic",
        "lstm",
        "transformer",
    ])
    fd_wealth_max: float = 2.5
    fd_nw: int = 120
    fd_nt: int = 80
    nn_hidden: int = 64
    nn_iters: int = 40          # increased from 18 for better convergence
    nn_paths: int = 512         # increased from 384
    nn_steps: int = 32
    nn_population_size: int = 24
    nn_elite_frac: float = 0.25
    weight_lower_bound: float = -5.0
    weight_upper_bound: float = 3.0
    # ------------------------------------------------------------------ #
    # Leverage constraints (applied identically at BOTH train and eval)   #
    #   max_long_leverage  : sum of all positive weights  <= 3.0          #
    #   max_short_leverage : |sum of all negative weights| <= 5.0         #
    # These match the per-asset box bounds [d, u] = [-5, 3] by design.    #
    # ------------------------------------------------------------------ #
    max_long_leverage: float = 3.0
    max_short_leverage: float = 5.0
    # ── New training improvements ──────────────────────────────────────────── #
    nn_pretrain_iters: int = 100      # supervised Browne warm-start iterations
    nn_antithetic: bool = True        # antithetic variates (Z / -Z pairs)
    nn_p_curriculum: float = 0.30     # fraction of paths near goal boundary
    nn_patience: int = 60             # early-stopping patience (iters)
    nn_horizon_years: float = 1.0     # T used for tau normalisation
    # Sweep over multiple goal multipliers (produces goal_vs_multiplier.png)
    goal_multipliers: List[float] = field(default_factory=lambda: [1.10])

    # Calibration / evaluation split (§5 of experiment review)
    # When True: estimate μ/Ω on [calib_start, calib_end] and run the
    # historical backtest on [eval_start, eval_end].
    use_calib_split: bool = True
    calib_start: str = "2015-01-01"
    calib_end:   str = "2020-12-31"
    eval_start:  str = "2021-01-01"
    eval_end:    str = "2024-12-31"

    # ── ES-GRPO hyper-parameters (§13 of GRPO_INTEGRATION_SPEC) ─────────── #
    es_grpo_G: int = 32              # population size (must be even)
    es_grpo_sigma: float = 0.030     # ES perturbation std-dev
    es_grpo_lr: float = 3e-3         # ES learning rate
    es_grpo_iters: int = 300         # max training iterations
    es_grpo_paths: int = 256         # MC paths per candidate per iter
    es_grpo_pretrain: int = 100      # Browne warm-start supervised iters
    es_grpo_patience: int = 60       # early-stopping patience (iters)
    # ── MC evaluation parameters ─────────────────────────────────────────── #
    mc_eval_paths: int = 2000        # paths for OOS MC evaluation
    mc_eval_seed: int = 4242         # RNG seed for MC evaluation

    # ── Enriched-feature controls (§3.9 of NN_FEATURE_REDESIGN.md) ──────── #
    # nn_feature_set_default : override per-arch 'feature_set' ('basic'|'enriched')
    # nn_empirical_cdf_bootstrap : block-bootstrap paths for empirical-d lookup
    # nn_empirical_cdf_block     : block length in days (~1 month)
    # nn_rolling_window          : window for realized-vol / rolling stats
    nn_feature_set_default: str = 'basic'
    nn_empirical_cdf_bootstrap: int = 1000
    nn_empirical_cdf_block: int = 21
    nn_rolling_window: int = 20

    def __post_init__(self):
        self.results_dir = Path(self.results_dir)
