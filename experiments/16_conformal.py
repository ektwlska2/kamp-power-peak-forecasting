"""Conformal prediction intervals for the next-hour forecast (rubric: 불확실성 추정)
and the interval upper bound as an extra YELLOW condition.

Interval for each 15-min value: forecast +/- q (target 90%).
  global : q from last month's errors of the previous model (calibration set)
  band   : q per hour band (night 0-6, start-up 7-8, day 9-16, evening 17-23)
  rolling: q from the deployed model's own errors on hours completed in the
           previous 28 days (at least 7 days), warm start = global q
Evaluated on July (model <= Jun, calibration = June errors of model <= May)
and on Aug-Sep (model <= Jul, calibration = July errors of model <= Jun).

Interval-YELLOW: upper bound of the hourly max (max of forecasts + q) >= 190
and not already RED. The rule was proposed after reviewing the four Aug-Sep
misses, so July is its clean check.

Run: .venv/bin/python experiments/16_conformal.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import Q, ROOT, load  # noqa: E402
from src.features import build  # noqa: E402
from src.models import fit_detector, fit_quarters, predict_quarters, risk_rule, two_tier  # noqa: E402
from src.peak import context_features, episode_onset, peak_label  # noqa: E402
from src.state import IDLE, daily_state  # noqa: E402
from src.uncertainty import HOUR_BANDS, band_of, conformal_q, rolling_q  # noqa: E402
from src.weights import copy_weights  # noqa: E402

OUT = ROOT / "outputs" / "exp16"
OUT.mkdir(parents=True, exist_ok=True)
START, CAP, LEVEL = pd.Timestamp("2021-01-08"), 190, 0.9
CHRONOS = ROOT / "data" / "reference" / "chronos2_zero_shot_predictions.csv"

df = load()
df["state"] = df.date.map(daily_state(df))
lbl, onset = peak_label(df), episode_onset(peak_label(df))
Xb = build(df, use_weekly=False)
Xc = pd.concat([Xb, context_features(df)], axis=1).assign(hour_next=df.hour)
M = {m: pd.Timestamp(f"2021-{m:02d}-01") for m in (5, 6, 7, 8)}
END = pd.Timestamp("2021-09-15")


def forecast(train_end, a, b):
    tr = (df.ts >= START) & (df.ts < train_end)
    te = (df.ts >= a) & (df.ts < b)
    return predict_quarters(fit_quarters(Xb, df, tr), Xb, te)


def evaluate(name, cal_pred, eval_pred, eval_start, eval_end):
    """cal_pred: previous model on the previous month (calibration set)."""
    cal_err = (cal_pred - df.loc[cal_pred.index, Q]).abs()
    ev_err = (eval_pred - df.loc[eval_pred.index, Q]).abs()
    q_glob = conformal_q(cal_err.to_numpy().ravel(), LEVEL)
    cal_band = cal_err.groupby(df.loc[cal_err.index, "hour"].map(band_of))
    q_band = {k: conformal_q(v.to_numpy().ravel(), LEVEL) for k, v in cal_band}
    # rolling: history = calibration errors + deployed model's errors, hour-level (max over quarters)
    hist_ts = pd.concat([df.loc[cal_err.index, "ts"], df.loc[ev_err.index, "ts"]])
    hist_err_q = pd.concat([cal_err, ev_err])
    # quarter errors flattened with the hour's timestamp (an hour is complete one hour after its start)
    flat_ts = np.repeat(hist_ts.to_numpy() + np.timedelta64(1, "h"), 4)
    flat_err = hist_err_q.to_numpy().ravel()
    issue = df.loc[eval_pred.index, "ts"]
    q_roll = rolling_q(issue, flat_ts, flat_err, LEVEL, warm=q_glob)
    y = df.loc[eval_pred.index, Q].to_numpy()
    f = eval_pred.to_numpy()
    hrs = df.loc[eval_pred.index, "hour"].to_numpy()
    qs = {"global": np.full(len(f), q_glob), "band": np.array([q_band[band_of(h)] for h in hrs]), "rolling": q_roll}
    rows = []
    pk = y >= CAP
    for k, q in qs.items():
        inside = (y >= f - q[:, None]) & (y <= f + q[:, None])
        rows.append({"period": name, "method": k, "coverage": inside.mean(), "coverage_at_peaks": inside[pk].mean(),
                     "mean_width": float(2 * q.mean())})
    return pd.DataFrame(rows), qs, q_band, q_glob


def alerts(eval_pred, q, train_end, a, b):
    """Two-tier alert (as in run_all) plus the interval-YELLOW condition."""
    tr = (df.ts >= START) & (df.ts < train_end)
    te = (df.ts >= a) & (df.ts < b)
    model, cut = fit_detector(Xc, lbl, tr, copy_weights(df, tr), dates=df.date)
    s = pd.Series(model.predict_proba(Xc[te])[:, 1], index=df.index[te])
    rule = risk_rule(df)[te]
    base = two_tier(s, cut, rule)
    upper = pd.Series(eval_pred.max(axis=1).to_numpy() + q, index=eval_pred.index)
    ext = base.copy()
    ext[(base == "GREEN") & (upper >= CAP)] = "YELLOW"
    y, on = lbl[te].astype(bool), onset[te].astype(bool)
    opdays = df.loc[te & (df.state != IDLE), "date"].nunique()
    out = []
    for name, t in (("two-tier", base), ("two-tier + interval-YELLOW", ext)):
        anyal = t != "GREEN"
        out.append({"alert": name, "peak_hours": int(y.sum()), "caught": int((anyal & y).sum()), "onsets": int(on.sum()),
                    "onsets_caught": int((anyal & on).sum()), "alert_hours": int(anyal.sum()),
                    "false_alarms": int((anyal & ~y).sum()), "false_per_operating_day": float((anyal & ~y).sum() / opdays),
                    "yellow_hours": int((t == "YELLOW").sum()), "yellow_false": int(((t == "YELLOW") & ~y).sum())})
    added = (ext != base)
    near = df.loc[added[added].index, Q].max(axis=1)
    return pd.DataFrame(out), {"added_yellow_hours": int(added.sum()), "added_that_were_peaks": int((added & y).sum()),
                                "added_non_peak_actual_max_ge_170": int((near[~y[added[added].index]] >= 170).sum())}


# ---- July (clean check): model <= Jun, calibration = June errors of model <= May
cal_jul = forecast(pd.Timestamp("2021-06-01"), pd.Timestamp("2021-06-01"), pd.Timestamp("2021-07-01"))  # model <= May on June
ev_jul = forecast(pd.Timestamp("2021-07-01"), pd.Timestamp("2021-07-01"), pd.Timestamp("2021-08-01"))   # model <= Jun on July
cov_jul, qs_jul, _, qg_jul = evaluate("Jul", cal_jul, ev_jul, M[7], M[8])
al_jul, add_jul = alerts(ev_jul, qs_jul["rolling"], M[7], M[7], M[8])

# ---- Aug-Sep: model <= Jul, calibration = July errors of model <= Jun
ev_test = forecast(M[8], M[8], END)
cov_test, qs_test, qband_test, qg_test = evaluate("AugSep", ev_jul, ev_test, M[8], END)
al_test, add_test = alerts(ev_test, qs_test["rolling"], M[8], M[8], END)

# ---- Chronos-2 q10-q90 coverage on Aug-Sep (reference)
ch_cov = None
if CHRONOS.exists():
    ch = pd.read_csv(CHRONOS, parse_dates=["timestamp"])
    ch = ch[(ch.timestamp >= M[8]) & (ch.timestamp < END)]
    inside = (ch.actual >= ch.q10) & (ch.actual <= ch.q90)
    ch_cov = {"coverage": float(inside.mean()), "coverage_at_peaks": float(inside[ch.actual >= CAP].mean()),
              "mean_width": float((ch.q90 - ch.q10).mean())}

cov = pd.concat([cov_jul, cov_test])
cov.to_csv(OUT / "coverage.csv", index=False)
al = pd.concat([al_jul.assign(period="Jul"), al_test.assign(period="AugSep")])
al.to_csv(OUT / "interval_yellow.csv", index=False)
pd.DataFrame({"ts": df.loc[ev_test.index, "ts"].to_numpy(), "q_rolling": qs_test["rolling"],
              "upper_hour_max": ev_test.max(axis=1).to_numpy() + qs_test["rolling"]}).to_csv(OUT / "test_intervals.csv", index=False)
res = {"q_global_test": qg_test, "q_band_test": qband_test, "chronos_q10_q90": ch_cov,
       "added_by_interval_yellow": {"Jul": add_jul, "AugSep": add_test}}
(OUT / "summary.json").write_text(json.dumps(res, indent=2, default=float))
with pd.option_context("display.width", 200):
    print(cov.round(3).to_string(index=False))
    print("\nChronos-2 q10-q90 (Aug-Sep):", {k: round(v, 3) for k, v in (ch_cov or {}).items()})
    print("\n", al.round(2).to_string(index=False))
    print("\nadded yellow:", res["added_by_interval_yellow"])
