"""
AUTHOR: Kharanshu Bhojak, Manpreet Singh, Tanvir H. Pantha, Sourav Dutta
University of Texas at Dallas, USA

Date: September 2026

Function library for Bayesian Active Learning (BAL) of process-material parameters.
"""

import re
import warnings

import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# Build GP Kernel
# ─────────────────────────────────────────────────────────────────────────────
def build_kernel(n_dims: int):
    length_scale_bounds = (1e-10, 1e5)

    amplitude = C(
        constant_value=1.0,
        constant_value_bounds=(1e-3, 1e3),
    )
    matern = Matern(
        nu=2.5,
        length_scale=np.ones(n_dims),
        length_scale_bounds=[length_scale_bounds] * n_dims,
    )
    noise = WhiteKernel(
        noise_level=1e-2,
        noise_level_bounds=(1e-5, 5e-1),
    )
    return amplitude * matern + noise


# ─────────────────────────────────────────────────────────────────────────────
# GP fitter
# ─────────────────────────────────────────────────────────────────────────────
def fit_gp(X_train, y_train, n_dims: int,
           prev_kernel=None, n_restarts: int = 10):
    kernel = build_kernel(n_dims)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-8,
        normalize_y=True,
        n_restarts_optimizer=n_restarts,
        optimizer="fmin_l_bfgs_b",
    )
    if prev_kernel is not None:
        try:
            gp.kernel.theta = prev_kernel.theta
        except (ValueError, AttributeError):
            pass
    gp.fit(X_train, y_train)
    return gp


# ─────────────────────────────────────────────────────────────────────────────
# Acquisition functions- EI minimize, EI maximize, LCB minimize
# ─────────────────────────────────────────────────────────────────────────────
def ei_maximize(mu, sigma, f_best):
    z  = (mu - f_best) / np.clip(sigma, 1e-12, None)
    ei = (mu - f_best) * norm.cdf(z) + sigma * norm.pdf(z)
    return np.clip(ei, 0, None), int(np.argmax(ei))


def ei_minimize(mu, sigma, f_best):
    z  = (f_best - mu) / np.clip(sigma, 1e-12, None)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    return np.clip(ei, 0, None), int(np.argmax(ei))

def lcb_minimize_abs(mu, sigma, iteration, total_iters,
                     kappa_max=2.0, decay=3.0,
                     queried_mask=None):

    t      = iteration
    T      = max(total_iters - 1, 1)
    kappa  = kappa_max * np.exp(-decay * t / T)

    acq    = np.abs(mu) - kappa * sigma

    if queried_mask is not None:
        acq[queried_mask] = np.inf

    if np.all(np.isinf(acq)):
        idx = int(np.argmax(sigma))
    else:
        idx = int(np.argmin(acq))

    return acq, idx, kappa

# ─────────────────────────────────────────────────────────────────────────────
# Acquisition function- UCB maximize
# ─────────────────────────────────────────────────────────────────────────────
def ucb_maximize(mu, sigma, kappa=2.0):
    ucb = mu + kappa * sigma
    return ucb, int(np.argmax(ucb))


# ─────────────────────────────────────────────────────────────────────────────
# Figure of Merit
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_name(name) -> str:
    """'|delta(Vt_PBTI)|' -> 'deltavtpbti'  (case/space/symbol-insensitive key)."""
    s = str(name).lower().replace("δ", "delta")
    return re.sub(r"[^a-z0-9]", "", s)


# Common parameters, stored in normalized form (see _normalize_name).
_VT_NAMES = {"vt", "vth", "thresholdvoltage"}
_MINIMIZE_NAMES = _VT_NAMES | {
    "deltavtpbti", "dvtpbti", "vtshiftpbti",
    "deltavtnbti", "dvtnbti", "vtshiftnbti",
    "ss", "subthresholdswing",
    "latency",
    "readtime", "writetime",
}
_MAXIMIZE_NAMES = {
    "ion", "oncurrent",
    "mobility", "mu", "fieldeffectmobility",
    "retentiontime", "retention",
    "throughput",
    "readmargin",
}


def classify_parameters(columns, minimize=None, maximize=None) -> dict:

    minimize = list(minimize or [])
    maximize = list(maximize or [])

    both = set(minimize) & set(maximize)
    if both:
        raise ValueError(f"Parameters listed as both minimize and maximize: {sorted(both)}")

    directions = {}
    for col in columns:
        key = _normalize_name(col)
        if col in minimize:
            directions[col] = "min"
        elif col in maximize:
            if key in _VT_NAMES:
                raise ValueError(
                    f"'{col}' is a threshold voltage: it is always treated as "
                    "|Vt - Vt_target| and minimized; it cannot be maximized."
                )
            directions[col] = "max"
        elif key in _MINIMIZE_NAMES:
            directions[col] = "min"
        elif key in _MAXIMIZE_NAMES:
            directions[col] = "max"
        else:
            raise ValueError(
                f"Cannot tell whether '{col}' should be minimized or maximized. "
                "List it explicitly under `minimize` or `maximize`."
            )
    return directions


def define_FOM(df, objective_columns=None, minimize=None, maximize=None,
               vt_target=0.0, eps=1e-9, return_details=False):
    columns = []
    for col in list(objective_columns or []) + list(minimize or []) + list(maximize or []):
        if col not in columns:
            columns.append(col)
    if not columns:
        raise ValueError("No optimization parameters given.")

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in data: {missing}. "
                       f"Available columns: {list(df.columns)}")

    directions = classify_parameters(columns, minimize, maximize)
    values = df[columns].apply(pd.to_numeric, errors="raise")

    normalized = pd.DataFrame(index=df.index, columns=columns, dtype=float)
    for col in columns:
        x = values[col]
        if _normalize_name(col) in _VT_NAMES:
            x = x - vt_target
        x = x.abs()

        if directions[col] == "min":
            if (x < eps).any():
                warnings.warn(
                    f"'{col}' has values below eps={eps:g} (ideal value reached); "
                    "they are clipped to eps before taking 1/|param|, which "
                    "dominates the min-max scale.", stacklevel=2)
            score = 1.0 / x.clip(lower=eps)
        else:
            score = x

        lo, hi = score.min(), score.max()
        if not np.isfinite(lo) or hi == lo:
            warnings.warn(f"'{col}' is constant (or empty) in the data; its "
                          "normalized contribution is set to 0.", stacklevel=2)
            normalized[col] = np.where(score.isna(), np.nan, 0.0)
        else:
            normalized[col] = (score - lo) / (hi - lo)

    fom = normalized.sum(axis=1, skipna=False).rename("FOM")
    if return_details:
        return fom, normalized, directions
    return fom


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(path, sheet_name=0) -> pd.DataFrame:
    """Read the experiment sheet. A missing file raises FileNotFoundError."""
    return pd.read_excel(path, sheet_name=sheet_name)


# ─────────────────────────────────────────────────────────────────────────────
# GP helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_length_scales(gp, n_dims: int):
    """Learned Matern length scales of a GP built by build_kernel()."""
    try:
        # kernel_ = Sum(Product(C, Matern), White) -> k1 = Product -> k1.k2 = Matern
        return np.atleast_1d(gp.kernel_.k1.k2.length_scale).copy()
    except AttributeError:
        return np.full(n_dims, np.nan)


def generate_grid(lower, upper, n_grid=7, max_candidates=2000):
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    d = len(lower)
    counts = [int(n_grid)] * d if np.isscalar(n_grid) else [int(n) for n in n_grid]
    if len(counts) != d:
        raise ValueError(f"n_grid has {len(counts)} entries but there are {d} "
                         "process parameters; give one value or one per parameter.")
    if min(counts) < 2:
        raise ValueError("Every axis needs at least 2 grid points.")
    total = int(np.prod(counts, dtype=np.int64))
    if total > max_candidates:
        raise ValueError(f"The grid {counts} is {total:,} candidates "
                         f"(limit {max_candidates:,}). Use fewer points per axis "
                         "or raise max_candidates.")
    axes = [np.linspace(lo, hi, n) for lo, hi, n in zip(lower, upper, counts)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([m.ravel() for m in mesh])


# ─────────────────────────────────────────────────────────────────────────────
# Next-experiment suggestion
# ─────────────────────────────────────────────────────────────────────────────
def suggest_next_point(df, process_params, objective_columns=None,
                       minimize=None, maximize=None, vt_target=0.0,
                       bounds=None, kappa=2.0, n_grid=7,
                       n_restarts=15, min_distance=1e-2, top_k=5,
                       eps=1e-9, max_candidates=2000):
    process_params = list(process_params)
    if not process_params:
        raise ValueError("At least one process parameter is required.")
    n_dims = len(process_params)

    missing = [c for c in process_params if c not in df.columns]
    if missing:
        raise KeyError(f"Process parameters not found in data: {missing}. "
                       f"Available columns: {list(df.columns)}")

    obj_cols = []
    for col in list(objective_columns or []) + list(minimize or []) + list(maximize or []):
        if col not in obj_cols:
            obj_cols.append(col)
    if not obj_cols:
        raise ValueError("No optimization parameters given.")
    missing = [c for c in obj_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Optimization parameters not found in data: {missing}. "
                       f"Available columns: {list(df.columns)}")

    data = df[process_params + obj_cols].dropna().reset_index(drop=True)
    if len(data) < 3:
        raise ValueError(f"Need at least 3 complete experiments, found {len(data)}.")

    fom, normalized, directions = define_FOM(
        data, obj_cols, minimize, maximize, vt_target=vt_target, eps=eps,
        return_details=True)

    X_exp = data[process_params].to_numpy(dtype=float)
    y_exp = fom.to_numpy(dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_exp)

    try:
        gp = fit_gp(X_scaled, y_exp, n_dims, n_restarts=n_restarts)
    except np.linalg.LinAlgError as err:
        raise RuntimeError("GP fit failed (singular kernel matrix).") from err

    # search box
    lower = X_exp.min(axis=0)
    upper = X_exp.max(axis=0)
    for name, (lo, hi) in (bounds or {}).items():
        if name not in process_params:
            raise KeyError(f"Bounds given for unknown process parameter '{name}'.")
        if lo > hi:
            raise ValueError(f"Bounds for '{name}': low ({lo}) > high ({hi}).")
        j = process_params.index(name)
        lower[j], upper[j] = lo, hi

    cand = generate_grid(lower, upper, n_grid, max_candidates=max_candidates)
    cand_scaled = scaler.transform(cand)

    # drop candidates sitting on top of an already-measured point (chunked: low memory)
    d_min = np.concatenate([
        cdist(cand_scaled[i:i + 100_000], X_scaled).min(axis=1)
        for i in range(0, len(cand_scaled), 100_000)])
    keep = d_min >= min_distance
    if not keep.any():
        raise RuntimeError("All candidate points are within min_distance of "
                           "measured points; widen the bounds or lower min_distance.")
    cand, cand_scaled = cand[keep], cand_scaled[keep]

    mu, sigma = gp.predict(cand_scaled, return_std=True)
    ucb, best = ucb_maximize(mu, sigma, kappa=kappa)

    order = np.argsort(-ucb)[:top_k]
    top = pd.DataFrame(cand[order], columns=process_params)
    top["predicted_FOM"] = mu[order]
    top["predicted_std"] = sigma[order]
    top["UCB"] = ucb[order]

    used = data.copy()
    for col in obj_cols:
        used[f"norm({col})"] = normalized[col]
    used["FOM"] = fom

    return {
        "next_point": pd.Series(cand[best], index=process_params),
        "predicted_fom": float(mu[best]),
        "predicted_std": float(sigma[best]),
        "ucb": float(ucb[best]),
        "top_candidates": top,
        "data": used,
        "directions": directions,
        "gp": gp,
        "log_marginal_likelihood": float(gp.log_marginal_likelihood_value_),
        "length_scales": get_length_scales(gp, n_dims),
    }
