"""Rescheduling what-if (Chapter 4): model-estimated peak reduction from moving
production within a day. Not a validated intervention result.

Load-response model (day-ahead information only): calendar, shutdown calendar
(idle vs running), yesterday's electricity, production plan of day D
(hour h, h-1, h+1, day total) and temperature (recorded value as a forecast
proxy). Production features are monotone increasing. Repeated/copied days are
down-weighted because copying decoupled production from electricity.

Search per operating day: repeatedly move a chunk (5% of daily production)
from the production hour with the highest predicted 15-min max to the allowed
hour that keeps the predicted daily max lowest. Allowed receivers: hours inside
that day's production span, outside the peak-risk window (08-11, 13-14),
with receiver production capped at the day's own maximum hourly production.
Daily total is unchanged. Budget: at most 10% / 20% / 30% of daily production moved.

Run: .venv/bin/python experiments/10_reschedule_whatif.py
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import Q, ROOT, load, with_copy_status  # noqa: E402
from src.dayahead import opcal, strict  # noqa: E402
from src.metrics import score  # noqa: E402
from src.state import IDLE, daily_state  # noqa: E402

OUT = ROOT / "outputs" / "exp10"
OUT.mkdir(parents=True, exist_ok=True)
PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              random_state=0, verbose=-1, n_jobs=4)
RISK = {8, 9, 10, 11, 13, 14}
PROD_FEATS = ["prod_h", "prod_prev", "prod_next"]


def plan_feats(prod: pd.Series, date: pd.Series) -> pd.DataFrame:
    g = prod.groupby(date)
    return pd.DataFrame({"prod_h": prod, "prod_prev": g.shift(1).fillna(0),
                         "prod_next": g.shift(-1).fillna(0), "prod_day": g.transform("sum")})


df = with_copy_status(load())
base_X = pd.concat([strict(df), opcal(df, levels=2), df[["temp"]]], axis=1)
X = pd.concat([base_X, plan_feats(df.production, df.date)], axis=1)
mono = [1 if c in PROD_FEATS else 0 for c in X.columns]


def fit(train_mask, w):
    return {q: lgb.LGBMRegressor(**PARAMS, monotone_constraints=mono)
            .fit(X[train_mask], df.loc[train_mask, q], sample_weight=w[train_mask]) for q in Q}


def predict(models, Xrows):
    return pd.DataFrame({q: m.predict(Xrows[X.columns]) for q, m in models.items()}, index=Xrows.index)


# ---- choose training weights on July (train <= Jun), MAE + peak MAE
weights = {"all=1": pd.Series(1.0, index=df.index),
           "copies=0.3": pd.Series(np.where(df.is_copy_day, 0.3, 1.0), index=df.index)}
START = df.ts >= "2021-01-08"
sel = []
for wname, w in weights.items():
    for fold, a, b in (("Jul", "2021-07-01", "2021-08-01"), ("AugSep", "2021-08-01", "2021-09-15")):
        tr = START & (df.ts < a); te = (df.ts >= a) & (df.ts < b)
        p = predict(fit(tr, w), X[te])
        sel.append({"weights": wname, "fold": fold, **score(df.loc[te, Q], p)})
sel = pd.DataFrame(sel).round(3)
print("Response-model accuracy (day-ahead, plan + calendar + temp):\n",
      sel[["weights", "fold", "MAE", "peak_MAE", "bias"]].to_string(index=False))
jul = sel[sel.fold == "Jul"].set_index("weights")
wbest = (jul.MAE.rank() + jul.peak_MAE.rank()).idxmin()
print("selected on July:", wbest)

tr = START & (df.ts < "2021-08-01")
models = fit(tr, weights[wbest])

# ---- response sanity: predicted change in hourly max per +10% production at that hour
te_days = df[(df.ts >= "2021-08-01") & (df.date.map(daily_state(df)) != IDLE)]
Xb = X.loc[te_days.index]
bump = te_days.production * 1.1
Xu = pd.concat([base_X.loc[te_days.index], plan_feats(bump, te_days.date)], axis=1)
dmax = predict(models, Xu).max(axis=1) - predict(models, Xb).max(axis=1)
prodpos = te_days.production > 0
print(f"\nResponse check: +10% production in every hour -> mean predicted hourly-max change "
      f"{dmax[prodpos].mean():+.2f} (production hours), {dmax[~prodpos].mean():+.2f} (no-production hours)")


# ---- search
def reschedule(day, budget_share, chunk_share=0.05):
    rows = df.index[df.date == day]
    prod = df.loc[rows, "production"].astype(float).copy()
    total = prod.sum()
    hrs = df.loc[rows, "hour"].to_numpy()
    active = np.flatnonzero(prod.to_numpy() > 0)
    lo, hi = hrs[active.min()], hrs[active.max()]
    cap = prod.max()
    recv_ok = np.array([(lo <= h <= hi) and h not in RISK for h in hrs])
    chunk = chunk_share * total
    moved = 0.0

    def pred(pr):
        Xd = pd.concat([base_X.loc[rows], plan_feats(pr, df.loc[rows, "date"])], axis=1)
        return predict(models, Xd).max(axis=1).to_numpy()

    cur = pred(prod)
    start_max = cur.max()
    while moved + chunk <= budget_share * total + 1e-9:
        donors = [i for i in np.argsort(-cur) if prod.iloc[i] >= chunk]
        if not donors:
            break
        d = donors[0]
        best = None
        for r in np.flatnonzero(recv_ok):
            if r == d or prod.iloc[r] + chunk > cap:
                continue
            trial = prod.copy(); trial.iloc[d] -= chunk; trial.iloc[r] += chunk
            m = pred(trial).max()
            if best is None or m < best[0]:
                best = (m, r, trial)
        if best is None or best[0] >= cur.max() - 1e-6:
            break
        prod = best[2]; cur = pred(prod); moved += chunk
    return {"date": day, "pred_max_before": start_max, "pred_max_after": cur.max(),
            "moved_share": moved / total if total else 0.0, "new_plan": prod}


days = sorted(te_days.loc[te_days.production > 0, "date"].unique())
actual_max = df.groupby("date").peak_q.max()
rows = []
plans = {}
for budget in (0.1, 0.2, 0.3):
    for day in days:
        r = reschedule(day, budget)
        plans[(budget, day)] = r.pop("new_plan")
        r.update({"budget": budget, "actual_max": actual_max[day]})
        rows.append(r)
    print("done budget", budget, flush=True)
res = pd.DataFrame(rows)
res["reduction"] = res.pred_max_before - res.pred_max_after
res.to_csv(OUT / "whatif_days.csv", index=False)
pd.to_pickle(plans, OUT / "plans.pkl")

print(f"\nOperating days with production, Aug-Sep: {len(days)}")
print("Response model daily-max error on these days (pred_before - actual): "
      f"MAE {np.abs(res[res.budget == 0.1].pred_max_before - res[res.budget == 0.1].actual_max).mean():.1f}, "
      f"bias {(res[res.budget == 0.1].pred_max_before - res[res.budget == 0.1].actual_max).mean():+.1f}")
summ = res.groupby("budget").agg(mean_reduction=("reduction", "mean"), median_reduction=("reduction", "median"),
                                 max_reduction=("reduction", "max"), days_reduced=("reduction", lambda x: (x > 0.5).sum()),
                                 mean_moved_share=("moved_share", "mean"),
                                 days_pred_ge_threshold_before=("pred_max_before", lambda x: (x >= 180).sum()),
                                 days_pred_ge_threshold_after=("pred_max_after", lambda x: (x >= 180).sum()))
print(summ.round(2))
# monthly max of predicted daily maxima (what a demand charge would see)
res["month"] = pd.to_datetime(res.date).dt.month
print("\nMonthly max of predicted daily peak (before -> after):")
print(res.groupby(["budget", "month"]).agg(before=("pred_max_before", "max"), after=("pred_max_after", "max")).round(1))
print("\nTop days by reduction (budget 20%):")
print(res[res.budget == 0.2].nlargest(8, "reduction")[["date", "actual_max", "pred_max_before", "pred_max_after", "reduction", "moved_share"]].round(1).to_string(index=False))
