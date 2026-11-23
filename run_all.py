"""End-to-end pipeline: data -> models -> test predictions -> result tables.

    .venv/bin/python run_all.py            (about 2-3 minutes on a laptop)

Everything is written to outputs/final/. Random seeds are fixed. Protocol:
  * next-hour forecasts are issued at the start of hour t with rows through t-1
  * validation = July 2021 (train 01-08..06-30); test = 2021-08-01..09-14
    (train 01-08..07-31, models frozen, inputs update hourly)
  * the test period was inspected during development; it is not untouched.
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import statsmodels.formula.api as smf
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src import dayahead  # noqa: E402
from src.data import Q, copy_map, load, load_raw, with_copy_status  # noqa: E402
from src.features import assert_no_future_leak, build  # noqa: E402
from src.metrics import PEAK, score  # noqa: E402
from src.models import (FC_PARAMS, PROD_FEATS, cap_day, fit_detector, fit_quarters, plan_feats,  # noqa: E402
                        predict_quarters, reschedule_day, risk_rule, tier_scores, two_tier, best_f1_cutoff)
from src.peak import context_features, detection_scores, episode_onset, peak_label  # noqa: E402
from src.state import IDLE, OPER, daily_state  # noqa: E402
from src.uncertainty import brier, deployed_interval_q, ece, fit_calibrator, log_loss  # noqa: E402
from src.weights import copy_weights  # noqa: E402

OUT = ROOT / "outputs" / "final"
OUT.mkdir(parents=True, exist_ok=True)
CHRONOS = ROOT / "data" / "reference" / "chronos2_zero_shot_predictions.csv"
START = pd.Timestamp("2021-01-08")
VAL = (pd.Timestamp("2021-07-01"), pd.Timestamp("2021-08-01"))
TEST = (pd.Timestamp("2021-08-01"), pd.Timestamp("2021-09-15"))
SUMMARY = {}
T0 = time.time()


def step(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def masks(df, period):
    a, b = period
    return (df.ts >= START) & (df.ts < a), (df.ts >= a) & (df.ts < b)


def r(x, k=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)


# =========================================================== 1. data
step("1/11 data audit")
raw = load_raw()
df = with_copy_status(load())
cm = copy_map(df)
st = daily_state(df)
df["state"] = df.date.map(st)
lbl, onset = peak_label(df), episode_onset(peak_label(df))
cm.to_csv(OUT / "copy_map.csv")
qsum = df[Q].sum(axis=1)
SUMMARY["data"] = {
    "rows": len(df), "dates": int(df.date.nunique()),
    "invalid_hour_labels_rebuilt": int(df.hour_label_invalid.sum()),
    "duplicate_keys_in_raw": int(raw.duplicated(["날짜", "시간"]).sum()),
    "missing_values": {k: int(v) for k, v in df.isna().sum().items() if v > 0 and k in
                       ("wind", "rain", "staff")},
    "zero_readings": int((df[Q] == 0).values.sum()),
    "staff_identity_max_error": float(np.nanmax((df.staff - df.production / qsum.replace(0, np.nan)).abs())),
    "labor_rate_is_clock_rule_days": int((df.groupby("date").apply(
        lambda g: ((g.labor_rate == 1.0) == g.hour.between(9, 17)).all())).sum()),
    "copy_status_days": cm.status.value_counts().to_dict(),
    "copy_groups_production_not_copied": int((~cm[cm.group_size > 1].groupby("group").production_also_copied.first()).sum()),
    "days_exact_copy_of_minus7_FebJun": int(cm[(cm.index.month >= 2) & (cm.index.month <= 6)].equals_minus7.sum()),
    "state_days": {"idle": int((st == 0).sum()), "partial": int((st == 1).sum()), "operating": int((st == 2).sum())},
}
for kw in ({"use_weekly": False}, {"use_weekly": True}):
    assert_no_future_leak(df, **kw)
dayahead.assert_strict_no_target_day(df)

# =========================================================== 2. next-hour forecasting
step("2/11 next-hour forecasters")
Xb = build(df, use_weekly=False)
Xw = build(df, use_weekly=True)
tr_v, te_v = masks(df, VAL)
tr_t, te_t = masks(df, TEST)
y_t = df.loc[te_t, Q]
preds = {}
# naive baselines
g = df.set_index("ts")[Q]
for name, lag in (("naive: previous hour", 1), ("naive: same hour yesterday", 24), ("naive: same hour last week", 168)):
    for tag, te in (("val", te_v), ("test", te_t)):
        p = g.shift(lag, freq="h").reindex(df.ts[te]).set_axis(df.index[te])
        preds[(name, tag)] = p
# ridge (linear baseline on the same features)
for tag, tr, te in (("val", tr_v, te_v), ("test", tr_t, te_t)):
    Xf = Xb.fillna(Xb[tr].median())
    m = tr & Xb[[c for c in Xb if c.startswith("q")]].notna().all(axis=1)
    preds[("Ridge (linear)", tag)] = pd.DataFrame(
        {q: make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xf[m], df.loc[m, q]).predict(Xf[te]) for q in Q},
        index=df.index[te])
    preds[("LightGBM (final)", tag)] = predict_quarters(fit_quarters(Xb, df, tr), Xb, te)
    preds[("LightGBM + weekly lags", tag)] = predict_quarters(fit_quarters(Xw, df, tr), Xw, te)
    # residual target (correction to the previous hour)
    res = {}
    for q in Q:
        yy = df[q] - Xb[f"{q}_l1"]; mm = tr & yy.notna()
        res[q] = lgb.LGBMRegressor(**FC_PARAMS).fit(Xb[mm], yy[mm]).predict(Xb[te]) + Xb.loc[te, f"{q}_l1"]
    preds[("LightGBM residual", tag)] = pd.DataFrame(res, index=df.index[te])
# Chronos-2 benchmark (zero-shot predictions saved from a separate run)
if CHRONOS.exists():
    ch = pd.read_csv(CHRONOS, parse_dates=["timestamp"])
    ch["q"] = ch.quarter_minute.map({15: "q1", 30: "q2", 45: "q3", 60: "q4"})
    chp = ch.pivot(index="timestamp", columns="q", values="prediction")[Q]
    cha = ch.pivot(index="timestamp", columns="q", values="actual")[Q]
    pos = pd.Series(df.index, index=df.ts)
    chp.index = cha.index = pos[chp.index].to_numpy()
    assert (cha - df.loc[cha.index, Q]).abs().to_numpy().max() == 0, "Chronos actuals misaligned"
    for tag, te in (("val", te_v), ("test", te_t)):
        preds[("Chronos-2 (zero-shot benchmark)", tag)] = chp.loc[df.index[te]]
        preds[("average(LightGBM, Chronos-2)", tag)] = (chp.loc[df.index[te]] + preds[("LightGBM (final)", tag)]) / 2

rows = []
names = sorted({k[0] for k in preds}, key=lambda s: (not s.startswith("naive"), s))
for name in names:
    rr = {"model": name}
    for tag, te in (("val", te_v), ("test", te_t)):
        p = preds[(name, tag)]
        ok = p.notna().all(axis=1)
        s = score(df.loc[p.index[ok], Q], p[ok])
        rr.update({f"{tag}_MAE": s["MAE"], f"{tag}_RMSE": s["RMSE"], f"{tag}_peak_MAE": s["peak_MAE"],
                   f"{tag}_peak_shortfall": s["peak_shortfall"]})
    # peak-hour detection from the forecast max, cutoff chosen on July
    pv, pt = preds[(name, "val")], preds[(name, "test")]
    okv = pv.notna().all(axis=1)
    cut = best_f1_cutoff(pv[okv].max(axis=1).to_numpy(), lbl[pv.index[okv]].to_numpy())
    d = detection_scores(pt.max(axis=1).to_numpy(), lbl[pt.index].to_numpy(), onset[pt.index].to_numpy(), cut)
    rr.update({"test_peakhour_F1": d["F1"], "test_peakhour_TP": d["TP"], "test_peakhour_FP": d["FP"],
               "test_onsets_caught": d["onsets_caught"]})
    rows.append(rr)
cmp = pd.DataFrame(rows)
cmp.round(3).to_csv(OUT / "ch2_next_hour_model_comparison.csv", index=False)
fin = cmp.set_index("model").loc["LightGBM (final)"]
SUMMARY["next_hour"] = {k: r(v) for k, v in fin.items()}
# block bootstrap vs Chronos
if CHRONOS.exists():
    a_ = (preds[("LightGBM (final)", "test")] - y_t).abs().mean(axis=1)
    b_ = (preds[("Chronos-2 (zero-shot benchmark)", "test")] - y_t).abs().mean(axis=1)
    dd = (a_ - b_).to_numpy()
    blk = ((df.loc[te_t, "ts"] - TEST[0]).dt.days // 7).to_numpy()
    rng = np.random.default_rng(0)
    ub = np.unique(blk)
    boots = [dd[np.concatenate([np.flatnonzero(blk == u) for u in rng.choice(ub, len(ub))])].mean() for _ in range(4000)]
    SUMMARY["next_hour"]["MAE_minus_chronos"] = r(dd.mean())
    SUMMARY["next_hour"]["MAE_minus_chronos_CI95"] = [r(x) for x in np.percentile(boots, [2.5, 97.5])]

# =========================================================== 3. peak detector + two-tier alert
step("3/11 peak detector, calibration, prediction intervals and alert levels")
Xc = pd.concat([Xb, context_features(df)], axis=1)
alerts = pd.DataFrame(index=df.index[te_t])
alert_rows, det_rows, cal_rows = [], [], []
opday = df.state != IDLE
# 90% conformal half-width of the deployed next-hour forecaster (rolling, past errors only).
# July: deployed model <= Jun, warm start = June errors of the model <= May.
jun = (df.ts >= "2021-06-01") & (df.ts < VAL[0])
cal_jun = predict_quarters(fit_quarters(Xb, df, (df.ts >= START) & (df.ts < "2021-06-01")), Xb, jun)
PI_Q = {"val": deployed_interval_q(df, cal_jun, preds[("LightGBM (final)", "val")]),
        "test": deployed_interval_q(df, preds[("LightGBM (final)", "val")], preds[("LightGBM (final)", "test")])}
cal_choice = {}
for lead in (0, 1):
    y = lbl.shift(-lead); on = onset.shift(-lead)
    X = Xc.assign(hour_next=(df.hour + lead) % 24)
    rule = risk_rule(df, lead)
    for tag, period in (("val", VAL), ("test", TEST)):
        a, b = period
        tr = (df.ts >= START) & (df.ts + pd.Timedelta(hours=lead) < a)
        te = (df.ts >= a) & (df.ts + pd.Timedelta(hours=lead) < b)
        w = copy_weights(df, tr)
        model, cut, oof = fit_detector(X, y, tr, w, dates=df.date, return_oof=True)
        s = pd.Series(model.predict_proba(X[te])[:, 1], index=df.index[te])
        d = detection_scores(s.to_numpy(), y[te].to_numpy(), on[te].to_numpy(), cut)
        rd = detection_scores(rule[te].to_numpy().astype(float), y[te].to_numpy(), on[te].to_numpy(), 0.5)
        det_rows.append({"lead": lead, "period": tag, "detector": "classifier", "cutoff": cut, **d})
        det_rows.append({"lead": lead, "period": tag, "detector": "risk-window rule", "cutoff": 0.5, **rd})
        tier = two_tier(s, cut, rule[te])
        ts_target = df.loc[te, "date"]
        alert_rows.append({"lead": lead, "period": tag, "variant": "two-tier",
                           **tier_scores(tier, y[te], on[te], ts_target, opday[te])})
        reason = pd.Series(np.where(tier == "YELLOW", "rule", ""), index=tier.index)
        # probability calibration: method chosen on July (lowest Brier), fitted on out-of-fold scores
        yo = y[oof.index].to_numpy().astype(int)
        if tag == "val":
            fits = {k: fit_calibrator(k, oof, yo) for k in ("none", "platt", "isotonic")}
            briers = {k: brier(f(s.to_numpy()), y[te].to_numpy()) for k, f in fits.items()}
            cal_choice[lead] = min(briers, key=briers.get)
            cal_rows += [{"lead": lead, "period": "val", "method": k, "brier": v} for k, v in briers.items()]
        calf = fit_calibrator(cal_choice[lead], oof, yo)
        p_cal = pd.Series(calf(s.to_numpy()), index=s.index)
        cal_rows.append({"lead": lead, "period": tag, "method": f"chosen: {cal_choice[lead]}", "brier": brier(p_cal, y[te]),
                         "logloss": log_loss(p_cal, y[te]), "ECE": ece(p_cal, y[te].to_numpy()),
                         "brier_raw": brier(s, y[te]), "logloss_raw": log_loss(s, y[te]), "ECE_raw": ece(s, y[te].to_numpy()),
                         "cutoff_raw": cut, "cutoff_calibrated": float(calf(np.array([cut]))[0])})
        if lead == 0 and tag == "test":
            tier0_rule_clf = tier.copy()     # rule + classifier only, before the interval condition
        if lead == 0:
            # interval-YELLOW: upper bound of the hourly max >= 190 and not already RED/YELLOW
            fc_ = preds[("LightGBM (final)", tag)]
            upper = fc_.max(axis=1) + PI_Q[tag]
            add = (tier == "GREEN") & (upper.reindex(tier.index) >= 190)
            tier = tier.where(~add, "YELLOW")
            reason[add] = "interval"
            alert_rows.append({"lead": lead, "period": tag, "variant": "two-tier + interval",
                               **tier_scores(tier, y[te], on[te], ts_target, opday[te])})
        if tag == "test":
            alerts[f"p_peak_lead{lead}"] = s.round(4)
            alerts[f"p_peak_lead{lead}_calibrated"] = p_cal.round(4)
            alerts[f"cutoff_lead{lead}_calibrated"] = round(float(calf(np.array([cut]))[0]), 4)
            alerts[f"tier_lead{lead}"] = tier
            alerts[f"yellow_reason_lead{lead}"] = reason
            if lead == 0:
                clf0, cut0 = model, cut
pd.DataFrame(cal_rows).round(4).to_csv(OUT / "ch2_calibration.csv", index=False)
pd.DataFrame(det_rows).round(3).to_csv(OUT / "ch2_peak_detector.csv", index=False)
tiers = pd.DataFrame(alert_rows)
tiers.round(3).to_csv(OUT / "ch4_two_tier_alert.csv", index=False)
SUMMARY["two_tier_alert_test"] = {f"lead{int(row['lead'])} {row['variant']}": {k: r(v) for k, v in row.items() if k not in ("lead", "period", "variant")}
                                  for _, row in tiers[tiers.period == "test"].iterrows()}
# prediction-interval coverage (Aug-Sep) and Chronos-2 q10-q90 for reference
yq = df.loc[te_t, Q].to_numpy()
fq = preds[("LightGBM (final)", "test")].to_numpy()
qq = PI_Q["test"].to_numpy()[:, None]
inside = (yq >= fq - qq) & (yq <= fq + qq)
SUMMARY["prediction_interval_90_test"] = {"coverage": r(inside.mean()), "coverage_at_peaks": r(inside[yq >= 190].mean()),
                                          "mean_width": r(2 * qq.mean())}
if CHRONOS.exists():
    chq = ch[(ch.timestamp >= TEST[0]) & (ch.timestamp < TEST[1])]
    ins = (chq.actual >= chq.q10) & (chq.actual <= chq.q90)
    SUMMARY["chronos_q10_q90_test"] = {"coverage": r(ins.mean()), "coverage_at_peaks": r(ins[chq.actual >= 190].mean()),
                                       "mean_width": r((chq.q90 - chq.q10).mean())}
SUMMARY["calibration"] = {f"lead{row['lead']} {row['period']} {row['method']}": {k: r(v) for k, v in row.items() if k not in ("lead", "period", "method") and pd.notna(v)}
                          for row in cal_rows}

# test prediction file (competition deliverable)
tp = df.loc[te_t, ["ts"] + Q].rename(columns=dict(zip(Q, ["actual_15분", "actual_30분", "actual_45분", "actual_60분"])))
fc = preds[("LightGBM (final)", "test")]
for q, c in zip(Q, ["15분", "30분", "45분", "60분"]):
    tp[f"forecast_{c}"] = fc[q].round(2)
for q, c in zip(Q, ["15분", "30분", "45분", "60분"]):
    tp[f"pi90_lower_{c}"] = (fc[q] - PI_Q["test"]).round(2)
    tp[f"pi90_upper_{c}"] = (fc[q] + PI_Q["test"]).round(2)
tp = tp.join(alerts)
tp.insert(tp.columns.get_loc("p_peak_lead1"), "lead1_target_ts", (tp.ts + pd.Timedelta(hours=1)).astype(str))

# =========================================================== 3b. weekly retraining (deployment test)
step("4/11 weekly retraining vs frozen")
# Retrain at each 7-day origin on all data before it; use for the next 7 days.
# Chosen after Aug-Sep had been seen, so reported as a deployment result, not the headline.
origins = pd.date_range(TEST[0], TEST[1] - pd.Timedelta(days=1), freq="7D")
wk_fc, wk_det = [], {0: [], 1: []}
for k, origin in enumerate(origins):
    end = origins[k + 1] if k + 1 < len(origins) else TEST[1]
    blk = (df.ts >= origin) & (df.ts < end)
    trw = (df.ts >= START) & (df.ts < origin)
    wk_fc.append(predict_quarters(fit_quarters(Xb, df, trw), Xb, blk))
    for lead in (0, 1):
        y = lbl.shift(-lead)
        X = Xc.assign(hour_next=(df.hour + lead) % 24)
        trl = (df.ts >= START) & (df.ts + pd.Timedelta(hours=lead) < origin)
        model, cut = fit_detector(X, y, trl, copy_weights(df, trl), dates=df.date)
        bl = blk & (df.ts + pd.Timedelta(hours=lead) < TEST[1])
        wk_det[lead].append(pd.DataFrame({"p": model.predict_proba(X[bl])[:, 1], "cut": cut, "origin": origin},
                                         index=df.index[bl]))
wk_fc = pd.concat(wk_fc)
fr_fc = preds[("LightGBM (final)", "test")]
wk_rows = []
for mode, p in (("frozen", fr_fc), ("weekly", wk_fc)):
    s = score(y_t, p)
    rr = {"mode": mode, "MAE": s["MAE"], "RMSE": s["RMSE"], "peak_MAE": s["peak_MAE"], "peak_shortfall": s["peak_shortfall"]}
    for lead in (0, 1):
        yl, ol, rule = lbl.shift(-lead), onset.shift(-lead), risk_rule(df, lead)
        if mode == "frozen":
            d = alerts[[f"p_peak_lead{lead}"]].dropna().rename(columns={f"p_peak_lead{lead}": "p"})
            d["cut"] = next(x["cutoff"] for x in det_rows if x["lead"] == lead and x["period"] == "test"
                            and x["detector"] == "classifier")
        else:
            d = pd.concat(wk_det[lead])
        flag = (d.p >= d.cut).to_numpy(); yy = yl[d.index].to_numpy() == 1
        tpn, fpn, fnn = (flag & yy).sum(), (flag & ~yy).sum(), (~flag & yy).sum()
        rr[f"clf_lead{lead}_F1"] = 2 * tpn / (2 * tpn + fpn + fnn)
        tier = two_tier(d.p, d.cut, rule[d.index])
        t_ = tier_scores(tier, yl[d.index], ol[d.index], df.loc[d.index, "date"], opday[d.index])
        rr[f"alert_lead{lead}_caught"] = t_["any_peaks"]
        rr[f"alert_lead{lead}_onsets_caught"] = t_["any_onsets"]
        rr[f"alert_lead{lead}_false_red"] = int(((tier == "RED") & ~yy).sum())
    wk_rows.append(rr)
wkres = pd.DataFrame(wk_rows)
wkres.round(3).to_csv(OUT / "ch4_weekly_retraining.csv", index=False)
ae_f, ae_w = (fr_fc - y_t).abs().mean(axis=1), (wk_fc - y_t).abs().mean(axis=1)
wk_id = pd.Series(np.searchsorted(origins.values, df.loc[te_t, "ts"].values, side="right") - 1, index=y_t.index)
byw = pd.DataFrame({"week_from": [o.date() for o in origins],
                    "MAE_frozen": ae_f.groupby(wk_id).mean().values, "MAE_weekly": ae_w.groupby(wk_id).mean().values})
byw["change"] = byw.MAE_weekly - byw.MAE_frozen
byw.round(3).to_csv(OUT / "ch4_weekly_retraining_by_week.csv", index=False)
dvec, bvec = (ae_w - ae_f).to_numpy(), wk_id.to_numpy()
rng = np.random.default_rng(0)
boots = [dvec[np.concatenate([np.flatnonzero(bvec == u) for u in rng.choice(len(origins), len(origins))])].mean()
         for _ in range(4000)]
SUMMARY["weekly_retraining"] = {"MAE_frozen": r(ae_f.mean()), "MAE_weekly": r(ae_w.mean()),
                                "MAE_diff_CI95": [r(x) for x in np.percentile(boots, [2.5, 97.5])],
                                "table": {row["mode"]: {k: r(v) for k, v in row.items() if k != "mode"}
                                          for _, row in wkres.iterrows()}}
for q, c in zip(Q, ["15분", "30분", "45분", "60분"]):
    tp.insert(tp.columns.get_loc("p_peak_lead0"), f"forecast_weekly_{c}", wk_fc[q].round(2))

# =========================================================== 4. day-ahead ladder
step("5/11 day-ahead scenarios")
scen = {"strict (no target-day data)": ("strict", {}),
        "+ hourly production plan": ("plan", {}),
        "+ shutdown calendar (idle vs running)": ("opcal2", {}),
        "+ shutdown calendar + plan": ("opcal2+plan", {}),
        "+ plan + weather (recorded, optimistic)": ("plan+weather", {}),
        "+ 3-level state (upper bound)": ("opcal", {})}
da_rows, da_test = [], {}
for name, (key, _) in scen.items():
    X = dayahead.build(df, key)
    rr = {"scenario": name}
    for tag, period in (("val", VAL), ("test", TEST)):
        tr, te = masks(df, period)
        p = predict_quarters(fit_quarters(X, df, tr), X, te)
        s = score(df.loc[te, Q], p)
        dm = (p.max(axis=1).groupby(df.loc[te, "date"]).max() - df.loc[te, Q].max(axis=1).groupby(df.loc[te, "date"]).max())
        rr.update({f"{tag}_MAE": s["MAE"], f"{tag}_peak_MAE": s["peak_MAE"], f"{tag}_daily_max_MAE": dm.abs().mean()})
        if tag == "test":
            ae = (p - df.loc[te, Q]).abs().mean(axis=1)
            for k, gg in ae.groupby(df.loc[te, "state"].map({0: "idle", 1: "partial", 2: "operating"})):
                rr[f"test_MAE[{k}]"] = gg.mean()
            da_test[name] = p
    da_rows.append(rr)
da = pd.DataFrame(da_rows)
da.round(3).to_csv(OUT / "ch4_day_ahead_value_of_information.csv", index=False)
SUMMARY["day_ahead_test_MAE"] = {row.scenario: r(row.test_MAE) for row in da.itertuples()}
for name, col in (("strict (no target-day data)", "dayahead_strict"),
                  ("+ shutdown calendar (idle vs running)", "dayahead_calendar")):
    for q, c in zip(Q, ["15분", "30분", "45분", "60분"]):
        tp[f"{col}_{c}"] = da_test[name][q].round(2)
tp.to_csv(OUT / "test_predictions.csv", index=False, encoding="utf-8-sig")

# =========================================================== 5. Chapter 3 analysis
step("6/11 influence and error analysis")
fc_models = fit_quarters(Xb, df, tr_t)
sv = sum(m.predict(Xb[te_t], pred_contrib=True)[:, :-1] for m in fc_models.values()) / 4
sv = pd.DataFrame(sv, columns=Xb.columns)
imp = pd.DataFrame({"all_hours": sv.abs().mean(), "peak_hours": sv[lbl[te_t].to_numpy() == 1].abs().mean()})
imp.sort_values("all_hours", ascending=False).round(3).to_csv(OUT / "ch3_shap_forecaster.csv")
Xc0 = Xc.assign(hour_next=df.hour)
expl = shap.TreeExplainer(clf0)
svc = expl.shap_values(Xc0[te_t])
svc = svc[1] if isinstance(svc, list) else svc
impc = pd.Series(np.abs(svc).mean(0), index=Xc0.columns).sort_values(ascending=False)
impc.round(4).to_csv(OUT / "ch3_shap_classifier.csv", header=["mean_abs_shap"])
samp = df.index[te_t & df.hour.between(6, 20)]
samp = np.random.default_rng(0).choice(samp, min(500, len(samp)), replace=False)
iv = expl.shap_interaction_values(Xc0.loc[samp])
iv = np.abs(iv[1] if isinstance(iv, list) else iv).mean(0)
np.fill_diagonal(iv, 0)
cols = Xc0.columns
inter = pd.DataFrame([(cols[i], cols[j], 2 * iv[i, j]) for i in range(len(cols)) for j in range(i + 1, len(cols))],
                     columns=["feature_a", "feature_b", "mean_abs_interaction"]).nlargest(15, "mean_abs_interaction")
inter.round(4).to_csv(OUT / "ch3_shap_classifier_interactions.csv", index=False)
J = df.ts >= "2021-06-01"
hg = pd.cut(df.hour, [-1, 7, 11, 12, 14, 17, 23], labels=["00-07", "08-11", "12", "13-14", "15-17", "18-23"])
lb = pd.cut(df.mean_load.shift(1), [-1, 80, 100, 120, 140, 1e9], labels=["<80", "80-100", "100-120", "120-140", ">=140"])
pd.crosstab(hg[J], lb[J], values=lbl[J], aggfunc="mean").round(3).to_csv(OUT / "ch3_peak_rate_clock_x_prevload.csv")
pd.crosstab(hg[J], lb[J]).to_csv(OUT / "ch3_hours_clock_x_prevload.csv")
# forecast error by condition (descriptive, target-hour values)
p = preds[("LightGBM (final)", "test")]
e = pd.DataFrame(index=p.index)
e["abs_err"] = (p - y_t).abs().mean(axis=1)
e["bias"] = (p - y_t).mean(axis=1)
prev_st = df.date.map(st.shift(1))
e["day type"] = np.select([df.state == IDLE, (df.state != IDLE) & (prev_st == IDLE), df.state == OPER],
                          ["idle", "restart after idle", "operating"], "partial")[te_t.to_numpy()]
pr, pp = df.production, df.production.shift(1)
e["production change"] = pd.Series(np.select(
    [(pr == 0) & (pp == 0), (pp == 0) & (pr > 0), (pp > 0) & (pr == 0), pr > 1.5 * pp, pr < pp / 1.5],
    ["none", "start (0->+)", "stop (+->0)", "rise >50%", "fall >50%"], "steady"), index=df.index)[te_t]
e["peak onset hour"] = onset[te_t].map({0: "no", 1: "yes"})
e["zero reading"] = (df[Q] == 0).any(axis=1)[te_t].map({False: "no", True: "yes"})
e["temperature"] = pd.cut(df.temp, [-50, 20, 25, 28, 50], labels=["<20", "20-25", "25-28", ">=28"])[te_t].astype(str)
er = []
for c in ("day type", "production change", "peak onset hour", "zero reading", "temperature"):
    for k, gg in e.groupby(c):
        er.append({"condition": c, "level": k, "hours": len(gg), "MAE": gg.abs_err.mean(), "bias": gg.bias.mean(),
                   "share_of_total_error": gg.abs_err.sum() / e.abs_err.sum()})
pd.DataFrame(er).round(3).to_csv(OUT / "ch3_forecast_error_by_condition.csv", index=False)
# alert FN / FP
t0 = tp.set_index(p.index)
y0 = lbl[te_t].astype(bool)
fnm = y0 & (t0.tier_lead0 == "GREEN")                                   # final alert
fnm_rc = y0 & (tier0_rule_clf.reindex(t0.index) == "GREEN")             # rule + classifier only
fpm = ~y0 & (t0.tier_lead0 == "RED")
fn_idx = fnm_rc[fnm_rc].index
fn_tab = df.loc[fn_idx, ["ts", "hour", "production", "temp", "peak_q"]].assign(
    prev_mean=df.mean_load.shift(1)[fn_idx], onset=onset[fn_idx],
    caught_by_final_alert=(t0.loc[fn_idx, "tier_lead0"] != "GREEN").to_numpy())
fn_tab.to_csv(OUT / "ch3_alert_missed_peaks.csv", index=False)
fp_max = df.loc[fpm[fpm].index, "peak_q"]
SUMMARY["alert_errors_test_lead0"] = {"missed_peak_hours_rule_classifier": int(fnm_rc.sum()),
                                      "missed_peak_hours_final": int(fnm.sum()), "false_red": int(fpm.sum()),
                                      "false_red_actual_max_180_189": int(fp_max.between(180, 189).sum()),
                                      "false_red_actual_max_ge_170": int((fp_max >= 170).sum())}
# peaks vs temperature within risk hours on running days (adjusted logit)
d = df[(df.ts >= "2021-06-01") & (df.state > 0) & df.hour.isin([8, 9, 10, 11, 13, 14])].assign(
    peak=lbl, prev=df.mean_load.shift(1), lp=np.log1p(df.production))
lm = smf.logit("peak ~ temp + lp + prev + C(hour)", data=d).fit(disp=0, cov_type="cluster",
                                                                   cov_kwds={"groups": d.date.dt.dayofyear})
ci = lm.conf_int().loc["temp"]
SUMMARY["peak_temperature_logit"] = {"hours": len(d), "odds_ratio_per_C": r(np.exp(lm.params.temp)),
                                     "OR_CI95": [r(np.exp(ci[0])), r(np.exp(ci[1]))]}
pd.DataFrame({
    "peak_rate_by_temp": d.groupby(pd.cut(d.temp, [-50, 25, 28, 50], labels=["<25", "25-28", ">=28"]), observed=True).peak.mean(),
}).round(3).to_csv(OUT / "ch3_peak_rate_by_temperature_riskhours.csv")

# =========================================================== 6. production what-if (null check)
step("7/11 production rescheduling what-if")
base_X = pd.concat([dayahead.strict(df), dayahead.opcal(df, levels=2), df[["temp"]]], axis=1)
Xr = pd.concat([base_X, plan_feats(df.production, df.date)], axis=1)
mono = [1 if c in PROD_FEATS else 0 for c in Xr.columns]
wr = copy_weights(df, tr_t)
resp = fit_quarters(Xr, df, tr_t, weight=wr, monotone_constraints=mono)
rs = score(y_t, predict_quarters(resp, Xr, te_t))
days = [d_ for d_ in sorted(df.loc[te_t & (df.state != IDLE) & (df.production > 0), "date"].unique())]
tdays = te_t & (df.state != IDLE)
bump = pd.concat([base_X.loc[tdays], plan_feats(df.production[tdays] * 1.1, df.date[tdays])], axis=1)[Xr.columns]
d_max = (np.column_stack([m.predict(bump) for m in resp.values()]).max(1)
         - np.column_stack([m.predict(Xr[tdays]) for m in resp.values()]).max(1))
wi = []
for budget in (0.1, 0.2, 0.3):
    for day in days:
        rows_ = df.index[df.date == day]
        b0, b1, mv = reschedule_day(rows_, df.loc[rows_, "production"], base_X, resp, Xr.columns, budget)
        wi.append({"budget": budget, "date": day, "pred_max_before": b0, "pred_max_after": b1, "moved_share": mv})
wi = pd.DataFrame(wi)
wi["reduction"] = wi.pred_max_before - wi.pred_max_after
wi.round(3).to_csv(OUT / "ch4_production_whatif_days.csv", index=False)
SUMMARY["production_whatif"] = {
    "response_model_test_MAE": r(rs["MAE"]), "response_model_test_peak_MAE": r(rs["peak_MAE"]),
    "plus10pct_production_hourly_max_change": r(float(np.mean(d_max[df.production[tdays].to_numpy() > 0]))),
    "mean_reduction_by_budget": {str(k): r(v) for k, v in wi.groupby("budget").reduction.mean().items()},
    "max_reduction_by_budget": {str(k): r(v) for k, v in wi.groupby("budget").reduction.max().items()}}

# =========================================================== 7. peak-cap simulation
step("8/11 peak-cap simulation")
flags = {}
for lead in (0, 1):
    tcol = tp[f"tier_lead{lead}"].copy()
    tcol.index = tcol.index + lead           # alert issued at t targets hour t+lead
    flags[lead] = tcol
cap_rows, mon = [], []
for pname, period in (("Jun-Jul", ("2021-06-01", "2021-08-01")), ("Aug-Sep", ("2021-08-01", "2021-09-15"))):
    P = df[(df.ts >= period[0]) & (df.ts < period[1])]
    for cap in (200, 190, 180, 170):
        ds, exc_h = [], pd.Series(0.0, index=P.index)
        for day, gg in P.groupby("date"):
            raw_ = gg[Q].to_numpy().ravel().astype(float)
            out_, exc, left, delay = cap_day(raw_, cap)
            exc_h[gg.index] = np.clip(gg[Q].to_numpy() - cap, 0, None).sum(axis=1) / 4
            ds.append({"excess": exc, "left": left, "delay": delay, "energy": raw_.sum() / 4})
            mon.append({"period": pname, "cap": cap, "month": day.month, "raw_max": raw_.max(), "new_max": out_.max()})
        ds = pd.DataFrame(ds); hit = ds[ds.excess > 0]
        rr = {"period": pname, "cap": cap, "days": len(ds), "days_needing_deferral": len(hit),
              "mean_deferred_per_day": hit.excess.mean() if len(hit) else 0.0,
              "max_deferred_per_day": hit.excess.max() if len(hit) else 0.0,
              "deferred_share_of_day_energy_pct": 100 * (hit.excess / hit.energy).mean() if len(hit) else 0.0,
              "longest_delay_min": ds.delay.max(), "days_backlog_not_cleared": int((ds.left > 1e-9).sum())}
        if pname == "Aug-Sep":
            tot = exc_h.sum()
            for lead in (0, 1):
                f = flags[lead].reindex(exc_h.index)
                rr[f"excess_in_red_lead{lead}_pct"] = 100 * exc_h[f == "RED"].sum() / tot
                rr[f"excess_in_any_alert_lead{lead}_pct"] = 100 * exc_h[f.isin(["RED", "YELLOW"])].sum() / tot
        cap_rows.append(rr)
cap = pd.DataFrame(cap_rows)
cap.round(3).to_csv(OUT / "ch4_peak_cap_simulation.csv", index=False)
pd.DataFrame(mon).groupby(["period", "cap", "month"]).max().round(1).to_csv(OUT / "ch4_monthly_max_capped.csv")
c190 = cap[(cap.period == "Aug-Sep") & (cap.cap == 190)].iloc[0]
SUMMARY["peak_cap_190_AugSep"] = {k: r(v) for k, v in c190.items() if k not in ("period",)}
SUMMARY["annual_max_reading"] = int(df[Q].max().max())

# =========================================================== 8. write summary
step("9/11 summary")
(OUT / "summary.json").write_text(json.dumps(SUMMARY, indent=2, ensure_ascii=False, default=str))
with pd.option_context("display.width", 250, "display.max_columns", 30):
    print("\nNext-hour comparison (test = Aug-Sep):")
    print(cmp[["model", "val_MAE", "val_peak_MAE", "test_MAE", "test_RMSE", "test_peak_MAE",
               "test_peakhour_F1", "test_onsets_caught"]].round(3).to_string(index=False))
    print("\nPeak detector:\n", pd.DataFrame(det_rows)[["lead", "period", "detector", "TP", "FP", "FN", "F1",
                                                        "onsets_caught", "PR_AUC"]].round(3).to_string(index=False))
    print("\nTwo-tier alert:\n", tiers[["lead", "period", "peak_hours", "red_hours", "red_peaks", "any_hours", "any_peaks",
                                       "any_onsets", "onsets", "missed_peaks", "red_F1", "red_per_operating_day",
                                       "yellow_per_operating_day"]].round(3).to_string(index=False))
    print("\nWeekly retraining vs frozen:\n", wkres.round(3).to_string(index=False))
    print(byw.round(2).to_string(index=False))
    print("\nDay-ahead:\n", da[["scenario", "val_MAE", "test_MAE", "test_peak_MAE", "test_daily_max_MAE"]].round(2).to_string(index=False))
    print("\nPeak cap (Aug-Sep):\n", cap[cap.period == "Aug-Sep"].round(2).T)
print(json.dumps({k: SUMMARY[k] for k in ("production_whatif", "alert_errors_test_lead0", "peak_temperature_logit")},
                 indent=1, default=str))
# =========================================================== 10. agent replay (rule-based, no API key)
step("10/11 agent cache and 45-day replay")
from agent import build_cache as agent_cache  # noqa: E402
from agent import replay as agent_replay  # noqa: E402
agent_cache.main()
agent_replay.main()
# =========================================================== 11. uncertainty and copy-leakage experiments
step("11/11 experiments 15-18 (calibration, intervals, peak-error CI, copy leakage)")
import runpy  # noqa: E402
for name in ("15_calibration.py", "16_conformal.py", "17_peak_error_ci.py", "18_copy_leakage.py"):
    runpy.run_path(str(ROOT / "experiments" / name), run_name="__main__")
step(f"done -> {OUT}, outputs/agent/ and outputs/exp15-18/")
