"""Rolling-origin check of weekly lags on unique-day-rich months only vs all months.
Expanding training from 2021-01-08, monthly score windows Apr..Jul and Aug-Sep.
Also scores each fold separately on copied vs unique target days."""
import sys
from pathlib import Path
import lightgbm as lgb, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import Q, ROOT, load, with_copy_status
from src.features import build
PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              random_state=0, verbose=-1, n_jobs=4)
df = with_copy_status(load())
folds = [("2021-04-01", "2021-05-01"), ("2021-05-01", "2021-06-01"), ("2021-06-01", "2021-07-01"),
         ("2021-07-01", "2021-08-01"), ("2021-08-01", "2021-09-15")]
Xs = {False: build(df, use_weekly=False), True: build(df, use_weekly=True)}
rows = []
for a, b in folds:
    tr = (df.ts >= "2021-01-08") & (df.ts < a); te = (df.ts >= a) & (df.ts < b)
    for wk, X in Xs.items():
        for tgt in ("direct", "residual"):
            p = {}
            for q in Q:
                off = X[f"{q}_l1"] if tgt == "residual" else 0
                y = df[q] - off; m = tr & y.notna()
                p[q] = lgb.LGBMRegressor(**PARAMS).fit(X[m], y[m]).predict(X[te]) + (off[te] if tgt == "residual" else 0)
            ae = (pd.DataFrame(p, index=df.index[te]) - df.loc[te, Q]).abs().mean(axis=1)
            for st, g in ae.groupby(df.loc[te, "copy_status"].replace("first", "copy")):
                rows.append({"fold": a[5:7], "weekly": wk, "target": tgt, "day": st, "hours": len(g), "MAE": g.mean()})
            rows.append({"fold": a[5:7], "weekly": wk, "target": tgt, "day": "all", "hours": len(ae), "MAE": ae.mean()})
r = pd.DataFrame(rows)
r.to_csv(ROOT / "outputs" / "exp02" / "rolling_weekly.csv", index=False)
t = r.pivot_table(index=["target", "fold", "day"], columns="weekly", values="MAE").round(2)
t["effect(on-off)"] = (t[True] - t[False]).round(2)
t["hours"] = r[r.weekly].set_index(["target", "fold", "day"]).hours
with pd.option_context("display.max_rows", 100):
    print(t)
