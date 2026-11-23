"""Peak-risk detector utilities.

A peak hour is an hour with any quarter reading >= PEAK. The detector
estimates P(peak) for hour t+lead, issued at the start of hour t with rows
completed through t-1. Stacked inputs (state probabilities, forecaster
output) are out-of-fold on training rows.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import Q
from .metrics import PEAK

FC_PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                 random_state=0, verbose=-1, n_jobs=4)
CLF_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=0, verbose=-1, n_jobs=4)


def peak_label(df: pd.DataFrame, peak=PEAK) -> pd.Series:
    return (df[Q].max(axis=1) >= peak).astype(int)


def episode_onset(lbl: pd.Series) -> pd.Series:
    """1 on the first hour of each contiguous run of peak hours."""
    return ((lbl == 1) & (lbl.shift(1, fill_value=0) == 0)).astype(int)


def oof_forecast(df, X, train_mask, n_blocks=6) -> pd.DataFrame:
    """Direct next-hour forecasts of q1..q4: out-of-fold (date blocks) on
    training rows, fit-on-all-training elsewhere."""
    out = pd.DataFrame(np.nan, index=df.index, columns=Q)
    dates = np.sort(df.loc[train_mask, "date"].unique())
    jobs = [(train_mask & ~df.date.isin(b), train_mask & df.date.isin(b))
            for b in np.array_split(dates, n_blocks)]
    jobs.append((train_mask, ~train_mask))
    for fit_m, pred_m in jobs:
        for q in Q:
            m = fit_m & df[q].notna()
            out.loc[pred_m, q] = lgb.LGBMRegressor(**FC_PARAMS).fit(X[m], df.loc[m, q]).predict(X[pred_m])
    return out


def context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Past-only peak context (all shifted >= 1 hour)."""
    lbl = peak_label(df)
    f = pd.DataFrame(index=df.index)
    f["peaks_last24"] = lbl.shift(1).rolling(24).sum()
    f["peak_l1"] = lbl.shift(1)
    f["peak_l24"] = lbl.shift(24)
    last_peak_idx = pd.Series(np.where(lbl == 1, np.arange(len(df)), np.nan), index=df.index).ffill().shift(1)
    f["hours_since_peak"] = np.minimum(np.arange(len(df)) - last_peak_idx, 24 * 7)
    daymax = df.groupby("date")["peak_q"].max()
    f["yday_max"] = (df["date"] - pd.Timedelta(days=1)).map(daymax).to_numpy()
    f["sofar_max"] = df.groupby("date")["peak_q"].transform(lambda x: x.shift(1).cummax())
    # manufacturing risk window found in the analysis: 08-11 and 13-14 with prior mean >= 100
    f["risk_window"] = (df["hour"].isin([8, 9, 10, 11, 13, 14]) & (df["mean_load"].shift(1) >= 100)).astype(int)
    return f


def best_f1_cutoff(p: np.ndarray, y: np.ndarray) -> float:
    grid = np.unique(np.round(p, 4))
    best, cut = -1, 0.5
    for c in grid:
        f = p >= c
        tp = (f & (y == 1)).sum(); fp = (f & (y == 0)).sum(); fn = (~f & (y == 1)).sum()
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0
        if f1 > best:
            best, cut = f1, c
    return cut


def detection_scores(p, y, onset, cutoff) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss
    flag = p >= cutoff
    y = y.astype(bool); onset = onset.astype(bool)
    tp, fp, fn = (flag & y).sum(), (flag & ~y).sum(), (~flag & y).sum()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    r = {"peak_hours": int(y.sum()), "TP": int(tp), "FP": int(fp), "FN": int(fn),
         "precision": prec, "recall": rec, "F1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
         "onsets": int(onset.sum()), "onsets_caught": int((flag & onset).sum()),
         "continuing_caught": int((flag & y & ~onset).sum()), "continuing": int((y & ~onset).sum())}
    if 0 < y.sum() < len(y) and np.all((p >= 0) & (p <= 1)):
        r["PR_AUC"] = average_precision_score(y, p)
        r["Brier"] = brier_score_loss(y, p)
    return r
