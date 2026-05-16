"""
real_data_loader.py
-------------------
Loads pre-extracted ETF data (real_etf_data.npz) and exposes
ready-to-use parameter sets for the HJB / NN notebooks.

Usage
-----
    from real_data_loader import load_portfolio, PORTFOLIOS

    # Drop-in for synthetic params
    mkt = load_portfolio(1)           # 1-asset (IVV only)
    mkt = load_portfolio(5)          # 5-asset
    mkt = load_portfolio(10)         # 10-asset
    mkt = load_portfolio(20)         # 20-asset
    mkt = load_portfolio(100)        # 100-asset (synthetic, calibrated to real market)

    mkt.mu        # annualised excess returns (shape: n,)
    mkt.sigma     # annualised vols           (shape: n,)
    mkt.rho       # correlation matrix        (shape: n,n)
    mkt.omega     # covariance matrix         (shape: n,n)
    mkt.log_ret   # daily log returns matrix  (shape: T, n)
    mkt.tickers   # list of ticker strings
    mkt.n         # number of assets

Portfolios
----------
  5-asset  (2010-2026): IVV, QQQ, TLT, GLD, VNQ
 10-asset  (2010-2026): IVV, QQQ, IWM, VEA, VNQ, TLT, IEF, LQD, GLD, XLK
 20-asset  (2010-2026): IVV, QQQ, IWM, VEA, VWO, VNQ, TLT, IEF, SHY, TIP,
                        LQD, HYG, GLD, SLV, DBC, USO, XLK, XLF, XLV, UUP
100-asset  (synthetic): factor-model calibrated to real market statistics
                        (mu ~6% excess, sigma ~20%, 3-factor correlation ~0.30)
"""

from pathlib import Path
import numpy as np

_NPZ = Path(__file__).parent / 'real_etf_data.npz'

PORTFOLIOS = {
    1:  ['IVV'],
    5:  ['IVV', 'QQQ', 'TLT', 'GLD', 'VNQ'],
    10: ['IVV', 'QQQ', 'IWM', 'VEA', 'VNQ', 'TLT', 'IEF', 'LQD', 'GLD', 'XLK'],
    20: ['IVV', 'QQQ', 'IWM', 'VEA', 'VWO', 'VNQ', 'TLT', 'IEF', 'SHY', 'TIP',
         'LQD', 'HYG', 'GLD', 'SLV', 'DBC', 'USO', 'XLK', 'XLF', 'XLV', 'UUP'],
}

R_FREE_ANN = 0.03   # assumed risk-free rate (3%)


class MarketData:
    """Container for a portfolio's calibrated parameters."""
    def __init__(self, n, tickers, mu_ann, sig_ann, rho, log_ret, dates):
        self.n       = n
        self.tickers = list(tickers)
        self.mu      = mu_ann - R_FREE_ANN   # excess return
        self.mu_ann  = mu_ann                # total return
        self.sigma   = sig_ann
        self.rho     = rho
        self.omega   = np.outer(sig_ann, sig_ann) * rho
        self.log_ret = log_ret               # shape (T, n)
        self.dates   = dates                 # int YYYYMMDD
        self.r       = R_FREE_ANN

    def __repr__(self):
        lines = [f"MarketData({self.n}-asset, {len(self.dates)} days)"]
        for t, m, s in zip(self.tickers, self.mu_ann, self.sigma):
            lines.append(f"  {t:<6}: mu={m*100:+5.1f}%  sigma={s*100:4.1f}%")
        return '\n'.join(lines)


def _synthetic_market(n_assets: int, start: str = None, end: str = None,
                      seed: int = 42) -> MarketData:
    """
    Generate a synthetic MarketData for n_assets using a 3-factor model
    calibrated to real equity-market statistics:
      - excess returns:  mean ~6 %, std ~3 %
      - annualised vols: mean ~20 %
      - pairwise corr:   ~0.30 (via 3 common factors)

    A simulated daily log-return series is drawn to match (mu, Omega).
    The date axis mirrors the 20-asset real dataset for compatibility.
    """
    rng = np.random.default_rng(seed)

    # --- Parameters --------------------------------------------------------
    r   = R_FREE_ANN
    eta = np.abs(rng.normal(0.06, 0.03, n_assets)).clip(0.01, 0.20)
    mu  = r + eta                          # total annualised return

    # Factor covariance: 3 common factors, idiosyncratic residual
    k = 3
    B = rng.normal(0, 0.08, (n_assets, k))
    D = np.diag(rng.uniform(0.03, 0.06, n_assets))
    Omega = B @ B.T + D

    # Ensure positive definite
    lam_min = np.linalg.eigvalsh(Omega).min()
    if lam_min < 1e-6:
        Omega += (abs(lam_min) + 1e-6) * np.eye(n_assets)

    sig = np.sqrt(np.diag(Omega))                       # annualised vols
    D_inv = np.diag(1.0 / sig)
    rho   = D_inv @ Omega @ D_inv                       # correlation matrix

    # --- Simulate daily log-returns ----------------------------------------
    # Load the 20-asset date axis for a realistic trading-day calendar
    try:
        d     = np.load(_NPZ, allow_pickle=True)
        dates = d['dates20']                             # int YYYYMMDD
    except Exception:
        # Fallback: generate ~3780 Mon-Fri trading days (2010-01-04 → 2024-12-31)
        import datetime
        day   = datetime.date(2010, 1, 4)
        end_d = datetime.date(2025, 1, 1)
        cal   = []
        while day < end_d:
            if day.weekday() < 5:   # Mon=0 … Fri=4
                cal.append(int(day.strftime('%Y%m%d')))
            day += datetime.timedelta(days=1)
        dates = np.array(cal, dtype=int)

    T_days  = len(dates)
    dt      = 1.0 / 252
    mu_d    = (mu - 0.5 * sig ** 2) * dt               # daily drift
    # Cholesky for correlated draws
    L       = np.linalg.cholesky(Omega * dt)
    Z       = rng.standard_normal((T_days, n_assets))
    log_ret = mu_d[None, :] + Z @ L.T                  # (T, n)

    # Optional date slicing (mirror real loader behaviour)
    if start is not None:
        s = int(start.replace('-', ''))
        mask = dates >= s
        log_ret = log_ret[mask]; dates = dates[mask]
    if end is not None:
        e = int(end.replace('-', ''))
        mask = dates <= e
        log_ret = log_ret[mask]; dates = dates[mask]

    tickers = [f"SYN{i:03d}" for i in range(n_assets)]
    return MarketData(n_assets, tickers, mu, sig, rho, log_ret, dates)


def load_portfolio(n_assets: int, start: str = None, end: str = None) -> MarketData:
    """
    Load calibrated parameters for the n_assets-asset portfolio.

    Parameters
    ----------
    n_assets : 5 | 10 | 20
    start    : 'YYYY-MM-DD' to slice the return history (optional)
    end      : 'YYYY-MM-DD' to slice the return history (optional)

    Returns
    -------
    MarketData object with mu, sigma, rho, omega, log_ret, tickers
    """
    # For explicitly supported large asset counts, fall back to a synthetic
    # market calibrated to match real market statistics.
    _REAL_N      = (1, 5, 10, 20)
    _SYNTHETIC_N = (50, 100, 200)
    if n_assets in _SYNTHETIC_N:
        return _synthetic_market(n_assets, start=start, end=end)
    if n_assets not in _REAL_N:
        raise ValueError(
            f"n_assets must be one of {_REAL_N + _SYNTHETIC_N}, got {n_assets}"
        )

    d = np.load(_NPZ, allow_pickle=True)
    # n=1: load IVV by slicing the first column of the 5-asset dataset
    if n_assets == 1:
        k = '5'
        mu      = d[f'mu{k}'][:1]
        sig     = d[f'sig{k}'][:1]
        rho     = d[f'rho{k}'][:1, :1]
        log_ret = d[f'log_ret{k}'][:, :1]   # (T, 1)
        dates   = d[f'dates{k}']
        tickers = ['IVV']
    else:
        k = str(n_assets)
        mu      = d[f'mu{k}']
        sig     = d[f'sig{k}']
        rho     = d[f'rho{k}']
        log_ret = d[f'log_ret{k}']    # (T, n)
        dates   = d[f'dates{k}']      # int YYYYMMDD
        tickers = list(d[f'tickers{k}'])

    # Optional date slicing
    if start is not None:
        s = int(start.replace('-', ''))
        mask = dates >= s
        log_ret = log_ret[mask]; dates = dates[mask]
    if end is not None:
        e = int(end.replace('-', ''))
        mask = dates <= e
        log_ret = log_ret[mask]; dates = dates[mask]

    return MarketData(n_assets, tickers, mu, sig, rho, log_ret, dates)


def load_portfolio_split(
    n_assets: int,
    calib_start: str = "2015-01-01",
    calib_end: str   = "2020-12-31",
    eval_start: str  = "2021-01-01",
    eval_end: str    = "2024-12-31",
) -> tuple:
    """
    Return (calib_data, eval_data) where:
    - calib_data : MarketData whose mu/sigma/omega were RE-ESTIMATED from the
                   [calib_start, calib_end] log-return slice — NOT from the
                   pre-computed full-sample NPZ values (Bug 1 of READINESS_AUDIT.md).
    - eval_data  : MarketData with the SAME mu/sigma/omega as calib_data but
                   log_ret covers [eval_start, eval_end].

    Using calib_data for FD solving / NN training and eval_data for the
    historical backtest removes the in-sample leakage identified in §5 of
    EXPERIMENT_REVIEW.md.
    """
    # Load raw slice — log_ret and dates are correctly windowed by load_portfolio,
    # but mu/sigma/rho come from the pre-computed full-sample NPZ (Bug 1 fix below).
    calib_raw = load_portfolio(n_assets, start=calib_start, end=calib_end)

    # ── Re-estimate statistics from the calibration-window returns ────────────
    # This is the key fix: overwrite the NPZ-sourced parameters with values
    # estimated exclusively from the calibration window's log_ret.
    lr_c = calib_raw.log_ret                                      # (T_calib, n)
    mu_re_ann  = lr_c.mean(axis=0) * 252 + 0.5 * lr_c.var(axis=0) * 252
    sig_re_ann = lr_c.std(axis=0) * np.sqrt(252)
    # Guard against degenerate windows (constant returns → sig=0)
    sig_re_ann = np.maximum(sig_re_ann, 1e-6)
    rho_re     = np.corrcoef(lr_c.T) if n_assets > 1 else np.array([[1.0]])

    # Rebuild calib_data with correct statistics
    calib_data = MarketData(
        n        = calib_raw.n,
        tickers  = calib_raw.tickers,
        mu_ann   = mu_re_ann,
        sig_ann  = sig_re_ann,
        rho      = rho_re,
        log_ret  = calib_raw.log_ret,
        dates    = calib_raw.dates,
    )

    # Build eval_data: eval-window returns, but SAME calibrated statistics.
    eval_raw  = load_portfolio(n_assets, start=eval_start, end=eval_end)
    eval_data = MarketData(
        n        = calib_data.n,
        tickers  = calib_data.tickers,
        mu_ann   = mu_re_ann,           # calib-window estimates only
        sig_ann  = sig_re_ann,
        rho      = rho_re,
        log_ret  = eval_raw.log_ret,    # out-of-sample returns
        dates    = eval_raw.dates,
    )
    return calib_data, eval_data


def agg_1d(mkt: MarketData, weights=None):
    """
    Collapse an n-asset MarketData to a 1D S&P-like aggregate.
    weights: (n,) array; default = equal-weight.
    Returns (mu_1d, sigma_1d, log_ret_1d).
    """
    n = mkt.n
    w = np.ones(n) / n if weights is None else np.asarray(weights, float)
    w /= w.sum()
    log_ret_1d = mkt.log_ret @ w
    mu_1d      = mkt.mu_ann @ w
    sig_1d     = np.sqrt(w @ mkt.omega @ w)
    return float(mu_1d), float(sig_1d), log_ret_1d


if __name__ == '__main__':
    for n in (1, 5, 10, 20):
        mkt = load_portfolio(n)
        print(mkt)
        print()
