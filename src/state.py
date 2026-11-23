"""Operating-state model.

Daily state (from a completed day's mean reading, thresholds set from the
bimodal histogram of daily means: gap at 30-40 and 60-70):
    0 idle     mean < 35   (flat ~22 profile, max ~26)
    1 partial  35 <= mean < 70
    2 operating mean >= 70

At issuance hour t on day D (hour h), a classifier estimates P(state of D)
from information available then: calendar of D, states of completed days,
and today's readings for hours 0..h-1. Out-of-fold probabilities are used on
training rows so the forecaster never sees in-sample classifier outputs.

State-matched seasonal lag: for each state s, the four readings at hour h of
the most recent *completed* day whose state is s. The mixture
sum_s P(s) * profile_s(h) replaces "same hour last week", which fails when
last week was in a different state (shutdown / restart).
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import Q
from .features import HOLIDAYS_2021

IDLE, PARTIAL, OPER = 0, 1, 2
STATES = (IDLE, PARTIAL, OPER)
CLF_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                  random_state=0, verbose=-1, n_jobs=4)


def daily_state(df: pd.DataFrame) -> pd.Series:
    m = df.groupby("date")["mean_load"].mean()
    return pd.Series(np.select([m < 35, m < 70], [IDLE, PARTIAL], OPER), index=m.index, name="state")


def _date_shift(s: pd.Series, dates: pd.Series, days: int) -> np.ndarray:
    return s.reindex(dates - pd.Timedelta(days=days)).to_numpy()


def classifier_features(df: pd.DataFrame, st: pd.Series) -> pd.DataFrame:
    d = df["date"]
    f = pd.DataFrame(index=df.index)
    f["hour"] = df["hour"]
    f["dow"] = df["dow"]
    f["offday"] = ((df["dow"] >= 6) | d.isin(HOLIDAYS_2021)).astype(int)
    f["st_d1"] = _date_shift(st, d, 1)
    f["st_d7"] = _date_shift(st, d, 7)
    f["st_d2"] = _date_shift(st, d, 2)
    # run length of yesterday's state (completed days only)
    runs = (st != st.shift()).cumsum()
    run_len = st.groupby(runs).cumcount() + 1
    f["run_d1"] = np.minimum(_date_shift(run_len, d, 1), 3)   # cap: long runs unseen in training
    # today so far (hours 0..h-1) — NaN at hour 0
    grp = df.groupby("date")["mean_load"]
    csum = grp.cumsum() - df["mean_load"]
    f["sofar_mean"] = csum / df["hour"].replace(0, np.nan)
    f["sofar_max"] = df.groupby("date")["peak_q"].transform(lambda x: x.shift(1).cummax())
    f["last"] = df["mean_load"].shift(1)
    return f


def fit_state_probs(df: pd.DataFrame, train_mask: pd.Series, n_blocks=6) -> pd.DataFrame:
    """P(state of today) for every row. Rows in `train_mask` get out-of-fold
    probabilities (contiguous date blocks); all other rows get predictions from
    a model fit on all training rows."""
    st = daily_state(df)
    X = classifier_features(df, st)
    y = df["date"].map(st)
    P = pd.DataFrame(np.nan, index=df.index, columns=[f"p_{s}" for s in STATES])

    def fit(m):
        clf = lgb.LGBMClassifier(**CLF_PARAMS).fit(X[m], y[m])
        return clf

    def predict(clf, m):
        pr = clf.predict_proba(X[m])
        out = np.zeros((m.sum(), 3))
        for j, c in enumerate(clf.classes_):
            out[:, int(c)] = pr[:, j]
        return out

    tr_dates = np.sort(df.loc[train_mask, "date"].unique())
    for blk in np.array_split(tr_dates, n_blocks):
        hold = train_mask & df["date"].isin(blk)
        P.loc[hold] = predict(fit(train_mask & ~hold), hold)
    rest = ~train_mask
    P.loc[rest] = predict(fit(train_mask), rest)
    return P


def state_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """For every row (day D, hour h) and each state s: readings at hour h of the
    most recent completed day before D with state s, plus that day's age."""
    st = daily_state(df)
    dates = st.index
    wide = {q: df.pivot(index="date", columns="hour", values=q) for q in Q}
    out = {}
    for s in STATES:
        last_date = pd.Series(np.where(st == s, dates, pd.NaT), index=dates).ffill().shift(1)
        src = df["date"].map(last_date)
        for q in Q:
            w = wide[q].stack()
            key = pd.MultiIndex.from_arrays([src, df["hour"]])
            out[f"{q}_s{s}"] = w.reindex(key).to_numpy()
        out[f"age_s{s}"] = (df["date"] - src).dt.days.to_numpy()
    return pd.DataFrame(out, index=df.index)


def state_features(df: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
    prof = state_profiles(df)
    # age_s* (days since last day in state s) is left out: during a long
    # shutdown it exceeds every training value and trees extrapolate badly.
    f = pd.concat([P, prof.drop(columns=[c for c in prof if c.startswith("age_")])], axis=1)
    for q in Q:
        mix = sum(P[f"p_{s}"] * prof[f"{q}_s{s}"].fillna(prof[f"{q}_s{OPER}"]) for s in STATES)
        f[f"{q}_smix"] = mix
    f["smix_mean"] = f[[f"{q}_smix" for q in Q]].mean(axis=1)
    f["p_change"] = 1 - P.max(axis=1)          # classifier uncertainty
    return f
