"""Day-ahead features: issued at midnight before day D, forecasting all 24
hours (q1..q4) of D.

strict   : calendar of D + electricity through D-1 23:00 (no target-day data)
+plan    : strict + day-D production plan, "assumes the plan is executed as
           recorded" (the recorded hourly production of D is used as the plan)
+opcal   : strict + day-D operating state (idle/partial/operating) only, i.e.
           the factory's own shutdown/operating calendar is known a day ahead
+weather : adds day-D temperature/humidity (recorded values as a proxy for a
           weather forecast; optimistic)
"""
import numpy as np
import pandas as pd

from .data import Q
from .features import calendar


def strict(df: pd.DataFrame, use_weekly=False) -> pd.DataFrame:
    f = calendar(df)
    d = df["date"]
    for q in Q:
        f[f"{q}_d1"] = df[q].shift(24)                       # same hour yesterday
    f["mean_d1"] = df["mean_load"].shift(24)
    daily = df.groupby("date").agg(dmean=("mean_load", "mean"), dmax=("peak_q", "max"))
    last = df[df.hour == 23].set_index("date")["mean_load"]
    for k in (1, 2):
        f[f"daymean_d{k}"] = (d - pd.Timedelta(days=k)).map(daily.dmean).to_numpy()
        f[f"daymax_d{k}"] = (d - pd.Timedelta(days=k)).map(daily.dmax).to_numpy()
    f["last_reading"] = (d - pd.Timedelta(days=1)).map(last).to_numpy()   # D-1 23:00
    # yesterday's hourly profile smoothed around this hour
    # (grouped by D so the centred window never reaches past D-1 23:00)
    f["d1_win3"] = df["mean_load"].shift(24).groupby(d).transform(
        lambda x: x.rolling(3, center=True, min_periods=1).mean())
    if use_weekly:
        for q in Q:
            f[f"{q}_d7"] = df[q].shift(168)
        f["daymean_d7"] = (d - pd.Timedelta(days=7)).map(daily.dmean).to_numpy()
    return f


def plan(df: pd.DataFrame) -> pd.DataFrame:
    p = df["production"]
    g = df.groupby("date")["production"]
    f = pd.DataFrame(index=df.index)
    f["plan_h"] = p
    f["plan_prev"] = g.shift(1)
    f["plan_next"] = g.shift(-1)
    f["plan_day_total"] = g.transform("sum")
    f["plan_cum"] = g.cumsum()
    f["plan_active_hours"] = g.transform(lambda x: (x > 0).sum())
    f["plan_share"] = p / f["plan_day_total"].replace(0, np.nan)
    return f


def opcal(df: pd.DataFrame, levels=3) -> pd.DataFrame:
    """levels=3: idle/partial/operating; levels=2: idle vs running only
    (a plain shutdown calendar, the more defensible assumption)."""
    from .state import IDLE, daily_state
    st = daily_state(df)
    if levels == 2:
        st = (st != IDLE).astype(int)
    return pd.DataFrame({"state_D": df["date"].map(st)}, index=df.index)


def weather(df: pd.DataFrame) -> pd.DataFrame:
    return df[["temp", "humidity"]].copy()


def build(df, scenario="strict", use_weekly=False):
    parts = [strict(df, use_weekly)]
    if scenario == "opcal":
        parts.append(opcal(df))
    if scenario in ("opcal2", "opcal2+plan"):
        parts.append(opcal(df, levels=2))
    if scenario in ("plan", "plan+weather", "opcal2+plan"):
        parts.append(plan(df))
    if scenario in ("weather", "plan+weather"):
        parts.append(weather(df))
    return pd.concat(parts, axis=1)


def assert_strict_no_target_day(df, use_weekly=False):
    """Changing any electricity reading on day D must not change strict features on day D."""
    D = df["date"].iloc[len(df) // 2]
    X0 = strict(df, use_weekly)
    d2 = df.copy()
    m = d2["date"] == D
    d2.loc[m, Q + ["mean_load", "peak_q", "production", "temp"]] += 1000
    X1 = strict(d2, use_weekly)
    same = (X0[m].fillna(-1) == X1[m].fillna(-1)).all().all()
    assert same, "strict day-ahead features use target-day data"
