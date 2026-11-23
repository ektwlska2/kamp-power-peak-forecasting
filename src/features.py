"""Strict next-hour features.

Forecast contract: issued at the start of hour t, targets the four readings
q1..q4 of hour t, and may only use rows completed through t-1 plus the
calendar of t. Every feature below is built from values shifted by >= 1 hour
on a complete hourly grid, so `lag k` always means "k hours earlier".
"""
import numpy as np
import pandas as pd

from .data import Q

HOLIDAYS_2021 = pd.to_datetime([
    "2021-01-01", "2021-02-11", "2021-02-12", "2021-02-13", "2021-03-01",
    "2021-05-05", "2021-05-19", "2021-06-06", "2021-08-15", "2021-08-16"])


def calendar(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["hour"] = df["hour"]
    f["dow"] = df["dow"]
    f["weekend"] = (df["dow"] >= 6).astype(int)
    f["holiday"] = df["date"].isin(HOLIDAYS_2021).astype(int)
    f["offday"] = (f.weekend | f.holiday).astype(int)
    return f


def recent_load(df: pd.DataFrame) -> pd.DataFrame:
    s = df[Q]
    m = df["mean_load"]
    f = pd.DataFrame(index=df.index)
    for k in (1, 2, 3):
        for c in Q:
            f[f"{c}_l{k}"] = s[c].shift(k)
    f["mean_l1"] = m.shift(1)
    f["max_l1"] = s.max(axis=1).shift(1)
    f["d_l1"] = m.shift(1) - m.shift(2)            # last hourly change
    f["intra_slope_l1"] = s["q4"].shift(1) - s["q1"].shift(1)
    f["roll3"] = m.shift(1).rolling(3).mean()
    f["roll6"] = m.shift(1).rolling(6).mean()
    f["roll24"] = m.shift(1).rolling(24).mean()
    f["max24"] = s.max(axis=1).shift(1).rolling(24).max()
    f["mean_l24"] = m.shift(24)                     # same hour yesterday
    f["d_l24"] = m.shift(24) - m.shift(25)          # yesterday's change into this hour
    return f


def weekly(df: pd.DataFrame) -> pd.DataFrame:
    s = df[Q]
    m = df["mean_load"]
    f = pd.DataFrame(index=df.index)
    for c in Q:
        f[f"{c}_l168"] = s[c].shift(168)
    f["mean_l168"] = m.shift(168)
    f["d_l168"] = m.shift(168) - m.shift(169)       # last week's change into this hour
    return f


def past_exog(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["prod_l1"] = df["production"].shift(1)
    f["prod_roll3"] = df["production"].shift(1).rolling(3).mean()
    f["temp_l1"] = df["temp"].shift(1)
    return f


def build(df: pd.DataFrame, use_weekly=True, use_exog=False, use_labor=False) -> pd.DataFrame:
    parts = [calendar(df), recent_load(df)]
    if use_weekly:
        parts.append(weekly(df))
    if use_exog:
        parts.append(past_exog(df))
    X = pd.concat(parts, axis=1)
    if use_labor:
        X["labor_rate"] = df["labor_rate"]   # target-hour value: a published clock rule
    return X


def assert_no_future_leak(df: pd.DataFrame, **kw) -> None:
    """Mutation check: altering hour t's measurements must not change hour t's features."""
    X0 = build(df, **kw)
    d2 = df.copy()
    i = len(d2) // 2
    d2.loc[i, Q + ["mean_load", "production", "temp"]] += 1000
    X1 = build(d2, **kw)
    changed = (X0.loc[i].fillna(-1) != X1.loc[i].fillna(-1))
    if kw.get("use_labor"):
        changed = changed.drop("labor_rate")
    assert not changed.any(), f"leak via {list(changed[changed].index)}"
