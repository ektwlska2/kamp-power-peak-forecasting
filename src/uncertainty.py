"""Probability calibration and conformal prediction intervals.

Calibration: a monotone map fitted on out-of-fold training scores
(none / Platt / isotonic). Monotone maps keep the ranking, so an alert
cutoff mapped through the same function flags exactly the same hours.

Conformal intervals: forecast +/- q, where q is the 90% quantile of past
absolute errors of the deployed forecaster (completed hours only).
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


# ------------------------------------------------------------------ calibration
def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_calibrator(kind, p, y):
    """Returns a function mapping raw scores to calibrated probabilities."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    if kind == "none":
        return lambda s: np.asarray(s, float)
    if kind == "platt":
        lr = LogisticRegression(C=1e6).fit(_logit(p).reshape(-1, 1), y)
        return lambda s: lr.predict_proba(_logit(s).reshape(-1, 1))[:, 1]
    if kind == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(p, y)
        return lambda s: iso.predict(np.asarray(s, float))
    raise ValueError(kind)


def log_loss(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, int)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, int)) ** 2))


def ece(p, y, bins=10):
    """Expected calibration error with equal-width bins."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    out = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(out)


# ------------------------------------------------------------------ conformal
HOUR_BANDS = {"night 0-6": range(0, 7), "start-up 7-8": range(7, 9), "day 9-16": range(9, 17), "evening 17-23": range(17, 24)}


def band_of(hour):
    for name, hrs in HOUR_BANDS.items():
        if hour in hrs:
            return name
    raise ValueError(hour)


def conformal_q(abs_err, level=0.9):
    """Split-conformal quantile with the finite-sample correction."""
    a = np.sort(np.asarray(abs_err, float))
    n = len(a)
    k = min(n - 1, int(np.ceil((n + 1) * level)) - 1)
    return float(a[k])


def rolling_q(ts, hist_ts, hist_err, level=0.9, window_days=28, min_n=24 * 7, warm=None):
    """For each issue time in `ts`: conformal quantile over errors of hours that
    were complete before it (hist_ts < issue time) within the last window."""
    hist = pd.Series(np.asarray(hist_err, float), index=pd.DatetimeIndex(hist_ts)).sort_index()
    out = []
    for t in pd.DatetimeIndex(ts):
        w = hist[(hist.index < t) & (hist.index >= t - pd.Timedelta(days=window_days))]
        out.append(conformal_q(w, level) if len(w) >= min_n else warm)
    return np.array(out, float)


def deployed_interval_q(df, cal_pred, ev_pred, level=0.9, quarters=("q1", "q2", "q3", "q4")):
    """Rolling conformal half-width for each hour of `ev_pred` (the deployed
    model's forecasts). History = errors of the previous model on the previous
    month (`cal_pred`, warm start) + the deployed model's own errors on hours
    that were complete before the issue time. Same method as experiment 16."""
    q = list(quarters)
    cal_err = (cal_pred - df.loc[cal_pred.index, q]).abs()
    ev_err = (ev_pred - df.loc[ev_pred.index, q]).abs()
    warm = conformal_q(cal_err.to_numpy().ravel(), level)
    hist_ts = pd.concat([df.loc[cal_err.index, "ts"], df.loc[ev_err.index, "ts"]])
    flat_ts = np.repeat(hist_ts.to_numpy() + np.timedelta64(1, "h"), 4)
    flat_err = pd.concat([cal_err, ev_err]).to_numpy().ravel()
    out = rolling_q(df.loc[ev_pred.index, "ts"], flat_ts, flat_err, level, warm=warm)
    return pd.Series(out, index=ev_pred.index)


def block_bootstrap_ci(values, blocks, n=4000, seed=0):
    """95% CI of the mean of `values`, resampling whole blocks (e.g. weeks)."""
    v, b = np.asarray(values, float), np.asarray(blocks)
    ub = np.unique(b)
    idx = {u: np.flatnonzero(b == u) for u in ub}
    rng = np.random.default_rng(seed)
    stats = [v[np.concatenate([idx[u] for u in rng.choice(ub, len(ub))])].mean() for _ in range(n)]
    return [float(x) for x in np.percentile(stats, [2.5, 97.5])]
