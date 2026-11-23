"""Chapter 3: influence factors and error analysis.

1. SHAP drivers + interactions: next-hour forecaster (base, trained <= Jul) and
   the peak classifier (lead 0, trained <= Jul), explained on Aug-Sep.
2. Forecast-error conditions (Aug-Sep, repeated on Jul): calendar/production/
   weather groups, and a shallow "error-condition tree".
3. Peak-alert FN / FP conditions (two-tier alert, Aug-Sep lead 0).
4. Conditions accompanying actual peaks (Jun-Sep, where peaks exist).

Descriptive conditions use target-hour values (production, temperature, actual
load change); they describe when errors happen and are NOT forecast inputs.
Run: .venv/bin/python experiments/08_error_analysis.py   (after 03, 04)
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.tree import DecisionTreeRegressor, export_text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import Q, ROOT, load  # noqa: E402
from src.features import build  # noqa: E402
from src.peak import CLF_PARAMS, FC_PARAMS, context_features, episode_onset, peak_label  # noqa: E402
from src.state import IDLE, OPER, daily_state  # noqa: E402

OUT = ROOT / "outputs" / "exp08"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

df = load()
lbl, onset = peak_label(df), episode_onset(peak_label(df))
st = daily_state(df)
dst = df.date.map(st)
prev_st = df.date.map(st.shift(1))
df["daytype"] = np.select([dst == IDLE, (dst != IDLE) & (prev_st == IDLE), dst == OPER],
                          ["idle", "restart", "operating"], "partial")
TR = (df.ts >= "2021-01-08") & (df.ts < "2021-08-01")
TE = (df.ts >= "2021-08-01") & (df.ts < "2021-09-15")
base = build(df, use_weekly=False)


def section(t):
    print(f"\n{'=' * 8} {t} {'=' * 8}")


# ---------------------------------------------------------------- 1. SHAP
section("1a. forecaster drivers (mean |SHAP| averaged over q1-q4 models, Aug-Sep)")
sv_sum = None
fc_models = {}
for q in Q:
    m = lgb.LGBMRegressor(**FC_PARAMS).fit(base[TR], df.loc[TR, q])
    fc_models[q] = m
    sv = m.predict(base[TE], pred_contrib=True)[:, :-1]
    sv_sum = sv if sv_sum is None else sv_sum + sv
sv_fc = pd.DataFrame(sv_sum / 4, columns=base.columns, index=df.index[TE])
imp_fc = sv_fc.abs().mean().sort_values(ascending=False)
print(imp_fc.head(12).round(2))
imp_fc.to_csv(OUT / "shap_forecaster_importance.csv")
sv_fc.to_pickle(OUT / "shap_forecaster_values.pkl")

# on peak readings only: what drives the forecast when it matters
pk_rows = lbl[TE].astype(bool).to_numpy()
print("\n  drivers on actual peak hours only:")
print(sv_fc[pk_rows].abs().mean().sort_values(ascending=False).head(8).round(2))

section("1b. peak classifier drivers + interactions (lead 0)")
Xc = pd.concat([base, context_features(df), pd.DataFrame({"hour_next": df.hour}, index=df.index)], axis=1)
clf = lgb.LGBMClassifier(**CLF_PARAMS).fit(Xc[TR], lbl[TR])
expl = shap.TreeExplainer(clf)
sv_c = expl.shap_values(Xc[TE])
sv_c = sv_c[1] if isinstance(sv_c, list) else sv_c
sv_c = pd.DataFrame(sv_c, columns=Xc.columns, index=df.index[TE])
imp_c = sv_c.abs().mean().sort_values(ascending=False)
print(imp_c.head(12).round(3))
imp_c.to_csv(OUT / "shap_classifier_importance.csv")
sv_c.to_pickle(OUT / "shap_classifier_values.pkl")
# interactions on operating-hours sample
samp = df.index[TE & (df.hour.between(6, 20))]
rng = np.random.default_rng(0)
samp = rng.choice(samp, min(500, len(samp)), replace=False)
iv = expl.shap_interaction_values(Xc.loc[samp])
iv = iv[1] if isinstance(iv, list) else iv
iv = np.abs(iv).mean(0)
np.fill_diagonal(iv, 0)
pairs = [(Xc.columns[i], Xc.columns[j], iv[i, j] * 2) for i in range(len(Xc.columns)) for j in range(i + 1, len(Xc.columns))]
inter = pd.DataFrame(pairs, columns=["feat_a", "feat_b", "mean_abs_interaction"]).sort_values(
    "mean_abs_interaction", ascending=False)
print("\n  top interactions (classifier, log-odds units):")
print(inter.head(10).round(3).to_string(index=False))
inter.to_csv(OUT / "shap_classifier_interactions.csv", index=False)

# interaction shape: peak risk by hour group x previous-hour load
section("1c. interaction table: actual peak rate by clock x previous-hour mean (Jun-Sep)")
J = df.ts >= "2021-06-01"
hg = pd.cut(df.hour, [-1, 7, 11, 12, 14, 17, 23], labels=["00-07", "08-11", "12", "13-14", "15-17", "18-23"])
lb = pd.cut(df.mean_load.shift(1), [-1, 80, 100, 120, 140, 1e9], labels=["<80", "80-100", "100-120", "120-140", ">=140"])
tab = pd.crosstab(hg[J], lb[J], values=lbl[J], aggfunc="mean").round(2)
cnt = pd.crosstab(hg[J], lb[J])
print(tab); print("hours:\n", cnt)
tab.to_csv(OUT / "peak_rate_clock_x_prevload.csv")

# ---------------------------------------------------------------- 2. forecast error conditions
preds = pd.read_pickle(ROOT / "outputs" / "exp03" / "predictions.pkl")


def conditions(idx):
    c = pd.DataFrame(index=idx)
    c["hour"] = df.loc[idx, "hour"]
    c["daytype"] = df.loc[idx, "daytype"]
    p, pp = df.production, df.production.shift(1)
    cat = np.select([(p == 0) & (pp == 0), (pp == 0) & (p > 0), (pp > 0) & (p == 0),
                     p > 1.5 * pp, p < pp / 1.5], ["none", "start (0->+)", "stop (+->0)", "rise >50%", "fall >50%"], "steady")
    c["prod_change"] = pd.Series(cat, index=df.index)[idx]
    pos = df.production[df.production > 0]
    c["prod_level"] = pd.cut(df.production, [-1, 0, pos.quantile(1 / 3), pos.quantile(2 / 3), 1e9],
                             labels=["0", "low", "mid", "high"])[idx]
    c["temp"] = pd.cut(df.temp, [-50, 20, 25, 28, 50], labels=["<20", "20-25", "25-28", ">=28"])[idx]
    c["abs_load_change"] = (df.mean_load - df.mean_load.shift(1)).abs()[idx]
    c["peak_onset"] = onset[idx]
    c["zero_reading"] = (df[Q] == 0).any(axis=1)[idx]
    return c


err_rows = []
for fold in ("Jul", "AugSep"):
    p = preds[(fold, "base", "direct")]
    y = df.loc[p.index, Q]
    c = conditions(p.index)
    c["abs_err"] = (p - y).abs().mean(axis=1)
    c["bias"] = (p - y).mean(axis=1)
    c.to_csv(OUT / f"error_frame_{fold}.csv")
    for col in ("daytype", "prod_change", "prod_level", "temp", "peak_onset", "zero_reading"):
        g = c.groupby(col, observed=True).agg(hours=("abs_err", "size"), MAE=("abs_err", "mean"), bias=("bias", "mean"))
        g["share_of_total_error"] = c.groupby(col, observed=True).abs_err.sum() / c.abs_err.sum()
        for k, r in g.iterrows():
            err_rows.append({"fold": fold, "condition": col, "level": str(k), **r.to_dict()})
err = pd.DataFrame(err_rows)
err.to_csv(OUT / "error_by_condition.csv", index=False)
section("2a. forecast error by condition (MAE; bias = mean forecast - actual)")
print(err.pivot_table(index=["condition", "level"], columns="fold", values=["hours", "MAE", "bias"]).round(2))

section("2b. error-condition tree, Aug-Sep (production/calendar/weather only)")
c = pd.read_csv(OUT / "error_frame_AugSep.csv", index_col=0)
Xt = pd.get_dummies(c[["hour", "daytype", "prod_change", "prod_level", "temp"]], dtype=int)
tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=40, random_state=0).fit(Xt, c.abs_err)
print(export_text(tree, feature_names=list(Xt.columns), decimals=1))
section("2c. same tree adding actual |load change| (mechanism, outcome-derived)")
Xt2 = Xt.assign(abs_load_change=c.abs_load_change.fillna(0))
tree2 = DecisionTreeRegressor(max_depth=3, min_samples_leaf=40, random_state=0).fit(Xt2, c.abs_err)
print(export_text(tree2, feature_names=list(Xt2.columns), decimals=1))
top = c.nlargest(int(len(c) * 0.1), "abs_err")
print("top-10% error hours: share with |load change|>30:", round((top.abs_load_change > 30).mean(), 2),
      "vs all hours:", round((c.abs_load_change > 30).mean(), 2))

# ---------------------------------------------------------------- 3. alert FN / FP
section("3. two-tier alert: missed peaks (FN) and false reds (FP), Aug-Sep lead 0")
store = pd.read_pickle(ROOT / "outputs" / "exp04" / "scores.pkl")
r4 = pd.read_csv(ROOT / "outputs" / "exp04" / "results.csv")
cut = r4.query("fold == 'AugSep' and lead == 0 and detector == 'clf'").cutoff.iloc[0]
s = store[("AugSep", 0, "clf")]
rule = store[("AugSep", 0, "context_rule")] >= 0.5
idx = s.index
red = s >= cut
anyal = red | rule
y = lbl[idx].astype(bool)
a = pd.DataFrame({"ts": df.loc[idx, "ts"], "hour": df.loc[idx, "hour"], "daytype": df.loc[idx, "daytype"],
                  "prev_mean": df.mean_load.shift(1)[idx], "actual_max": df.loc[idx, "peak_q"],
                  "score": s, "onset": onset[idx].astype(bool), "production": df.loc[idx, "production"],
                  "prod_prev": df.production.shift(1)[idx], "temp": df.loc[idx, "temp"]})
a["outcome"] = np.select([y & red, y & ~red & rule, y & ~anyal, ~y & red], ["TP red", "TP yellow", "FN", "FP red"], "TN/other")
a.to_csv(OUT / "alert_outcomes_AugSep.csv")
print(a[a.outcome == "FN"].to_string())
fp = a[a.outcome == "FP red"]
print(f"\nFalse reds: {len(fp)}; near-misses (actual max 180-189): {int(fp.actual_max.between(180, 189).sum())}; "
      f"actual max >=170: {int((fp.actual_max >= 170).sum())}")
print("False reds by hour:", fp.hour.value_counts().sort_index().to_dict())
print("False reds by daytype:", fp.daytype.value_counts().to_dict())
print("Red TPs by daytype:", a[a.outcome == "TP red"].daytype.value_counts().to_dict())
missed_onsets = a[a.onset & ~anyal[idx]]
print("red-tier onsets missed but caught by yellow:", int((a.onset & ~red & rule).sum()))

# ---------------------------------------------------------------- 4. peak conditions
section("4. conditions accompanying actual peaks (Jun-Sep)")
J = (df.ts >= "2021-06-01")
cj = conditions(df.index[J]).assign(peak=lbl[J])
for col in ("daytype", "prod_level", "prod_change", "temp"):
    g = cj.groupby(col, observed=True).peak.agg(["size", "sum", "mean"]).rename(
        columns={"size": "hours", "sum": "peak_hours", "mean": "peak_rate"})
    print(g.round(3)); g.to_csv(OUT / f"peak_rate_by_{col}.csv")
daily = df[J].groupby("date").agg(peak_day=("peak_q", lambda x: (x >= 190).any()), prod=("production", "sum"),
                                 temp_max=("temp", "max"), dow=("dow", "first"), state=("mean_load", "mean"))
daily = daily[daily.state >= 70]
print("\noperating days Jun-Sep:", len(daily), "| peak days:", int(daily.peak_day.sum()))
print("daily production, peak vs non-peak days (median):", daily.groupby("peak_day")["prod"].median().round(0).to_dict())
print("daily max temp, peak vs non-peak (median):", daily.groupby("peak_day").temp_max.median().to_dict())
print("peak-day rate by weekday:", daily.groupby("dow").peak_day.mean().round(2).to_dict())
