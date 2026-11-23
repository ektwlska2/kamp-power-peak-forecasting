"""Where do weekly lags hurt in Aug-Sep? Per-week and per-condition breakdown
(direct target, all-days policy, same settings as 02)."""
import sys
from pathlib import Path
import lightgbm as lgb, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import Q, load, with_copy_status
from src.features import build
PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              random_state=0, verbose=-1, n_jobs=4)
df = with_copy_status(load())
tr = (df.ts >= "2021-01-08") & (df.ts < "2021-08-01"); te = (df.ts >= "2021-08-01") & (df.ts < "2021-09-15")
ae = {}
for wk in (False, True):
    X = build(df, use_weekly=wk)
    p = pd.DataFrame({c: lgb.LGBMRegressor(**PARAMS).fit(X[tr], df.loc[tr, c]).predict(X[te]) for c in Q}, index=df.index[te])
    ae[wk] = (p - df.loc[te, Q]).abs().mean(axis=1)
d = df.loc[te, ["ts", "date", "hour", "dow", "mean_load"]].assign(off=ae[False], on=ae[True])
d["diff"] = d.on - d.off
d["week"] = (d.date - pd.to_timedelta(d.dow - 1, unit="D")).dt.strftime("%m-%d")
lw = df.set_index("ts").mean_load.shift(168, freq="h")
d["lastweek_mean"] = d.ts.map(lw)
daymean = df.groupby("date").mean_load.mean()
d["lastweek_day_mean"] = (d.date - pd.Timedelta(days=7)).map(daymean)
d["day_mean"] = d.date.map(daymean)
print(d.groupby("week")[["off", "on", "diff"]].mean().round(2))
d["regime_change"] = (d.day_mean - d.lastweek_day_mean).abs() > 30
print("\nby |today's daily mean - same weekday last week| > 30:")
print(d.groupby("regime_change")[["off", "on", "diff"]].agg(["mean"]).round(2), d.groupby("regime_change").date.nunique())
print("\ndays with regime change:", sorted(d[d.regime_change].date.dt.strftime("%m-%d(%a)").unique()))
