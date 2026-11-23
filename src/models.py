"""Final model components used by run_all.py.

- next-hour forecaster  : LightGBM per quarter, base features (calendar + recent load)
- peak detector         : LightGBM classifier, copy weights 1/n, cutoff = max F1 on
                          out-of-fold training scores
- two-tier alert        : RED = detector >= cutoff; YELLOW = risk-window rule, not red
- day-ahead ladder      : strict / +plan / +shutdown calendar / ...
- response what-if      : monotone day-ahead load model, production reshuffling
- peak-cap simulation   : energy-conserving deferral above a cap
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import Q
from .metrics import PEAK

SEED = 0
FC_PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                 random_state=SEED, verbose=-1, n_jobs=4)
CLF_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=SEED, verbose=-1, n_jobs=4)
RISK_HOURS = (8, 9, 10, 11, 13, 14)


# ------------------------------------------------------------------ forecaster
def fit_quarters(X, df, mask, weight=None, **extra):
    w = None if weight is None else weight[mask]
    return {q: lgb.LGBMRegressor(**FC_PARAMS, **extra).fit(X[mask], df.loc[mask, q], sample_weight=w)
            for q in Q}


def predict_quarters(models, X, mask):
    return pd.DataFrame({q: m.predict(X[mask]) for q, m in models.items()}, index=X.index[mask])


# ------------------------------------------------------------------ detector
def best_f1_cutoff(p, y):
    order = np.argsort(-p)
    p, y = p[order], y[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y); P = y.sum()
    f1 = 2 * tp / (tp + fp + P)
    # only evaluate at the last index of each distinct score
    last = np.r_[p[1:] != p[:-1], True]
    f = np.where(last, f1, -1)
    i = len(f) - 1 - np.argmax(f[::-1])      # ties -> lowest cutoff (same as exp04)
    return float(p[i])


def fit_detector(X, y, train_mask, weight, n_blocks=6, dates=None, return_oof=False):
    """Returns (final model, cutoff). Cutoff maximises F1 on out-of-fold
    training scores from contiguous date blocks. With return_oof=True also
    returns the out-of-fold scores (used to fit a probability calibrator)."""
    m = train_mask & y.notna() & (weight > 0)
    oof = pd.Series(np.nan, index=X.index)
    for blk in np.array_split(np.sort(dates[m].unique()), n_blocks):
        hold = m & dates.isin(blk)
        fit = m & ~hold
        oof[hold] = lgb.LGBMClassifier(**CLF_PARAMS).fit(
            X[fit], y[fit], sample_weight=weight[fit]).predict_proba(X[hold])[:, 1]
    model = lgb.LGBMClassifier(**CLF_PARAMS).fit(X[m], y[m], sample_weight=weight[m])
    cut = best_f1_cutoff(oof[m].to_numpy(), y[m].to_numpy().astype(int))
    return (model, cut, oof[m]) if return_oof else (model, cut)


def risk_rule(df, lead=0):
    hour = (df["hour"] + lead) % 24
    return (hour.isin(RISK_HOURS) & (df["mean_load"].shift(1) >= 100)).astype(int)


def two_tier(score, cutoff, rule):
    red = score >= cutoff
    yellow = (rule == 1) & ~red
    return pd.Series(np.select([red, yellow], ["RED", "YELLOW"], "GREEN"), index=score.index)


def tier_scores(tier, y, onset, dates, opday):
    """Alert-level summary: y/onset aligned to the target hour of each alert."""
    y = y.astype(bool); onset = onset.astype(bool)
    out = {"peak_hours": int(y.sum()), "onsets": int(onset.sum())}
    for name, m in (("red", tier == "RED"), ("yellow", tier == "YELLOW"), ("any", tier != "GREEN")):
        out[f"{name}_hours"] = int(m.sum())
        out[f"{name}_peaks"] = int((m & y).sum())
        out[f"{name}_precision"] = float((m & y).sum() / m.sum()) if m.sum() else np.nan
        out[f"{name}_onsets"] = int((m & onset).sum())
    red = tier == "RED"
    out["red_F1"] = 2 * out["red_peaks"] / (out["red_hours"] + out["peak_hours"])
    out["missed_peaks"] = int((y & (tier == "GREEN")).sum())
    out["recall_any"] = out["any_peaks"] / out["peak_hours"]
    n_op = int(opday.groupby(dates).first().sum())
    out["red_per_operating_day"] = out["red_hours"] / n_op
    out["yellow_per_operating_day"] = out["yellow_hours"] / n_op
    out["false_red_per_operating_day"] = float((red & ~y).sum() / n_op)
    return out


# ------------------------------------------------------------------ what-if
PROD_FEATS = ["prod_h", "prod_prev", "prod_next"]


def plan_feats(prod, date):
    g = prod.groupby(date)
    return pd.DataFrame({"prod_h": prod, "prod_prev": g.shift(1).fillna(0),
                         "prod_next": g.shift(-1).fillna(0), "prod_day": g.transform("sum")})


def reschedule_day(rows, prod, base_X, models, cols, budget, chunk_share=0.05):
    """Greedy move of production chunks from the highest-predicted-peak hour to
    allowed hours (inside the day's production span, outside RISK_HOURS,
    capacity = day's max hourly production). Daily total unchanged."""
    prod = prod.astype(float).copy()
    total = prod.sum()
    hrs = np.arange(len(rows))
    active = np.flatnonzero(prod.to_numpy() > 0)
    lo, hi = active.min(), active.max()
    cap = prod.max()
    recv_ok = np.array([(lo <= h <= hi) and h not in RISK_HOURS for h in hrs])
    chunk, moved = chunk_share * total, 0.0
    date = pd.Series(0, index=rows)

    def pred(pr):
        Xd = pd.concat([base_X.loc[rows], plan_feats(pr, date)], axis=1)[cols]
        return np.column_stack([m.predict(Xd) for m in models.values()]).max(axis=1)

    cur = pred(prod)
    start = cur.max()
    while moved + chunk <= budget * total + 1e-9:
        donors = [i for i in np.argsort(-cur) if prod.iloc[i] >= chunk]
        if not donors:
            break
        d, best = donors[0], None
        for r in np.flatnonzero(recv_ok):
            if r == d or prod.iloc[r] + chunk > cap:
                continue
            trial = prod.copy(); trial.iloc[d] -= chunk; trial.iloc[r] += chunk
            mx = pred(trial).max()
            if best is None or mx < best[0]:
                best = (mx, trial)
        if best is None or best[0] >= cur.max() - 1e-6:
            break
        prod = best[1]; cur = pred(prod); moved += chunk
    return start, cur.max(), moved / total if total else 0.0


# ------------------------------------------------------------------ peak cap
def cap_day(readings, cap):
    """readings: 96 quarter values. Returns (capped series, excess energy,
    leftover backlog, longest delay in minutes). Energy in recorded unit x hour."""
    out, B, start, longest = [], 0.0, None, 0
    for i, x in enumerate(readings):
        want = x + B
        o = min(want, cap)
        nb = want - o
        if nb > 0 and start is None:
            start = i
        if nb == 0 and start is not None:
            longest = max(longest, i - start + 1); start = None
        out.append(o); B = nb
    if start is not None:
        longest = max(longest, len(readings) - start)
    excess = np.clip(readings - cap, 0, None).sum() / 4
    return np.array(out), excess, B / 4, longest * 15
