"""Step 1 audit: data quality, copied-day map, 공장인원/인건비 checks,
weekly-lag contamination (tests 1-2) and production link on unique days.

Run: .venv/bin/python experiments/01_audit.py
Writes outputs/audit/*.csv and prints a summary.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import Q, ROOT, copy_map, load, load_raw, with_copy_status  # noqa: E402

OUT = ROOT / "outputs" / "audit"
OUT.mkdir(parents=True, exist_ok=True)


def section(t):
    print(f"\n=== {t} ===")


df = with_copy_status(load())
raw = load_raw()
cm = copy_map(df)
cm.to_csv(OUT / "copy_map.csv")

section("basic quality")
print("rows", len(df), "dates", df.date.nunique())
print("invalid hour labels", int(df.hour_label_invalid.sum()),
      "on", sorted(df.loc[df.hour_label_invalid, "date"].dt.strftime("%m-%d").unique()))
print("duplicate (날짜,시간) keys in raw", int(raw.duplicated(["날짜", "시간"]).sum()))
print("missing per column", df.isna().sum()[lambda s: s > 0].to_dict())
z = (df[Q] == 0)
print("zero readings", int(z.values.sum()), "in", int(z.any(axis=1).sum()), "hours;",
      "dates", sorted(df.loc[z.any(axis=1), "date"].dt.strftime("%m-%d").unique()))
print("평균 == round(mean of quarters)?",
      float((df.mean_load - df[Q].mean(axis=1)).abs().le(1).mean()))

section("공장인원 identity")
qsum = df[Q].sum(axis=1)
pred = np.where(qsum > 0, df.production / qsum.replace(0, np.nan), np.nan)
err = (df.staff - pred).abs()
print("공장인원 == 생산량/(q1+q2+q3+q4): max abs err", float(np.nanmax(err)),
      "share within 1e-6", float((err < 1e-6).mean()))

section("인건비 vs calendar")
# Korean public holidays 2021 in range (source: 관공서의 공휴일에 관한 규정)
holidays = pd.to_datetime([
    "2021-01-01", "2021-02-11", "2021-02-12", "2021-02-13", "2021-03-01",
    "2021-05-05", "2021-05-19", "2021-06-06", "2021-08-15", "2021-08-16"])
df["holiday"] = df.date.isin(holidays)
lr = df.groupby("date").agg(labor=("labor_rate", "first"), nuniq=("labor_rate", "nunique"),
                            dow=("dow", "first"), hol=("holiday", "first"),
                            prod=("production", "sum"), load=("mean_load", "mean"))
print("인건비 constant within each day?", bool((lr.nuniq == 1).all()))
print(pd.crosstab(lr.dow, lr.labor))
print("holiday dates' 인건비:", lr[lr.hol].labor.to_dict())
wk = lr[(lr.dow <= 5) & ~lr.hol]
print("weekday non-holiday: labor 1.0 count", int((wk.labor == 1.0).sum()),
      "of", len(wk))
print("mean daily load by 인건비 (weekdays):", wk.groupby("labor").load.mean().round(1).to_dict())
print("mean daily production by 인건비 (weekdays):", wk.groupby("labor")["prod"].mean().round(0).to_dict())
lr.to_csv(OUT / "labor_rate_by_date.csv")
print("인건비 by month:", df.groupby("month").labor_rate.mean().round(2).to_dict())

section("copied-day map")
print(cm.status.value_counts().to_dict(),
      "| groups", int((cm.group_size > 1).groupby(cm.group).first().sum()),
      "| groups with production also copied",
      int(cm[cm.group_size > 1].groupby("group").production_also_copied.first().sum()))
by_month = cm.assign(m=cm.index.month).groupby(["m", "status"]).size().unstack(fill_value=0)
print(by_month)
by_month.to_csv(OUT / "copy_status_by_month.csv")

section("test 1: target day == day-7 exactly")
cm["m"] = cm.index.month
t1 = cm.groupby("m").equals_minus7.agg(["sum", "count"])
t1["share"] = (t1["sum"] / t1["count"]).round(3)
print(t1.T)
print("Feb-Jun share", round(cm[cm.m.between(2, 6)].equals_minus7.mean(), 3),
      "| Aug-Sep share", round(cm[cm.m >= 8].equals_minus7.mean(), 3),
      "| day-1 exact copies overall", int(cm.equals_minus1.sum()))
t1.to_csv(OUT / "test1_minus7_copies.csv")

section("test 2: seasonal-naive baselines by day type")
g = df.set_index("ts")[Q]
res = []
for name, lag in [("same hour last week (lag 168h)", 168), ("same hour yesterday (lag 24h)", 24),
                  ("previous hour (lag 1h)", 1)]:
    pred = g.shift(lag, freq="h").reindex(g.index)
    ae = (pred - g).abs().mean(axis=1)
    d = df.set_index("ts").assign(ae=ae).dropna(subset=["ae"])
    d["period"] = np.where(d.month >= 8, "Aug-Sep", np.where(d.month >= 2, "Feb-Jul", "Jan"))
    for key, grp in d.groupby(["period", "equals_minus7"]):
        res.append({"baseline": name, "period": key[0], "day_is_exact_copy_of_minus7": key[1],
                    "hours": len(grp), "MAE": round(grp.ae.mean(), 2)})
    for key, grp in d.groupby(["period", "copy_status"]):
        res.append({"baseline": name, "period": key[0], "copy_status": key[1],
                    "hours": len(grp), "MAE": round(grp.ae.mean(), 2)})
t2 = pd.DataFrame(res)
t2.to_csv(OUT / "test2_baseline_by_daytype.csv", index=False)
with pd.option_context("display.width", 200, "display.max_rows", 100):
    print(t2)

section("production link: unique vs copied days")
d = df.copy()
d["logp"] = np.log1p(d.production)
d["period"] = np.where(d.is_copy_day, "copied/repeated", "unique")
rows = []
for per, grp in d.groupby("period"):
    wkg = grp[grp.dow <= 5]
    rows.append({"days": grp.date.nunique(), "set": per,
                 "hourly corr(mean_load, production)": round(grp.mean_load.corr(grp.production), 3),
                 "weekday corr": round(wkg.mean_load.corr(wkg.production), 3)})
    # within hour-of-day and weekday: does production still explain load?
    m = smf.ols("mean_load ~ logp + C(hour) + C(dow)", data=grp).fit(cov_type="cluster",
                                                                     cov_kwds={"groups": grp.date.dt.dayofyear})
    base = smf.ols("mean_load ~ C(hour) + C(dow)", data=grp).fit()
    rows[-1].update({"coef log1p(prod)": round(m.params.logp, 2),
                     "95% CI": f"[{m.conf_int().loc['logp', 0]:.2f}, {m.conf_int().loc['logp', 1]:.2f}]",
                     "R2 clock+dow": round(base.rsquared, 3), "R2 +prod": round(m.rsquared, 3)})
    daily = grp.groupby("date").agg(load=("mean_load", "sum"), prod=("production", "sum"), dow=("dow", "first"))
    rows[-1]["daily corr(energy, production)"] = round(daily.load.corr(daily["prod"]), 3)
t3 = pd.DataFrame(rows)
with pd.option_context("display.width", 250):
    print(t3.to_string(index=False))
t3.to_csv(OUT / "production_link.csv", index=False)

# within copy groups: same electricity, different production → production variance with zero load variance
grp_rows = []
for gidx, members in cm[cm.group_size > 1].groupby("group"):
    dates = members.index
    p = df[df.date.isin(dates)].groupby("date").production.sum()
    grp_rows.append({"group": gidx, "n": len(dates), "prod_min": p.min(), "prod_max": p.max()})
gr = pd.DataFrame(grp_rows)
print("copy groups where daily production differs by >2x while electricity identical:",
      int(((gr.prod_max > 2 * gr.prod_min.clip(lower=1)) & (gr.prod_max > 0)).sum()), "of", len(gr))
