"""Precompute everything the agent needs for the Aug 1 - Sep 14 replay.

Run after run_all.py:
    .venv/bin/python -m agent.build_cache

Uses only model outputs from outputs/final/test_predictions.csv plus
statistics computed from data BEFORE the test period (Jun-Jul 2021), so the
agent's default numbers are not tuned on the test period.
Writes agent/cache/cache.pkl.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import dayahead  # noqa: E402
from src.data import Q, copy_map, load, with_copy_status  # noqa: E402
from src.features import build  # noqa: E402
from src.models import fit_detector, fit_quarters, predict_quarters  # noqa: E402
from src.peak import context_features, peak_label  # noqa: E402
from src.state import IDLE, daily_state  # noqa: E402
from src.weights import copy_weights  # noqa: E402

OUT = ROOT / "agent" / "cache"
OUT.mkdir(parents=True, exist_ok=True)
START, TEST0, TEST1 = pd.Timestamp("2021-01-08"), pd.Timestamp("2021-08-01"), pd.Timestamp("2021-09-15")
PRE0, PRE1 = pd.Timestamp("2021-06-01"), pd.Timestamp("2021-08-01")   # months with real peaks, before test
CAP = 190


def main():
    df = with_copy_status(load())
    st = daily_state(df)
    df["state"] = df.date.map(st)
    cm = copy_map(df)
    df["group"] = df.date.map(cm.group)
    lbl = peak_label(df)
    tp = pd.read_csv(ROOT / "outputs" / "final" / "test_predictions.csv", parse_dates=["ts"])
    pos = pd.Series(df.index, index=df.ts)
    tp.index = pos[tp.ts].to_numpy()

    # ------------------------------------------------ per-hour model outputs (test period)
    ko = ["15분", "30분", "45분", "60분"]
    hours = pd.DataFrame(index=tp.index)
    hours["ts"] = tp.ts
    for q, c in zip(Q, ko):
        hours[f"fc_{q}"] = tp[f"forecast_weekly_{c}"]          # deployment: forecaster retrained weekly
        hours[f"da_{q}"] = tp[f"dayahead_calendar_{c}"]        # day-ahead with shutdown calendar
    for lead in (0, 1):
        hours[f"p{lead}"] = tp[f"p_peak_lead{lead}"]
        hours[f"tier{lead}"] = tp[f"tier_lead{lead}"]
        hours[f"yellow_reason{lead}"] = tp[f"yellow_reason_lead{lead}"].fillna("")
    hours["p0_cal"] = tp["p_peak_lead0_calibrated"]
    hours["p1_cal"] = tp["p_peak_lead1_calibrated"]
    hours["pi_upper_max"] = tp[[f"pi90_upper_{c}" for c in ko]].max(axis=1)
    hours["pi_q"] = tp["pi90_upper_15분"] - tp["forecast_15분"]

    # classifier explanations (lead 0): same training as run_all, SHAP top factors per hour
    Xc = pd.concat([build(df, use_weekly=False), context_features(df)], axis=1).assign(hour_next=df.hour)
    tr = (df.ts >= START) & (df.ts < TEST0)
    model, cut0 = fit_detector(Xc, lbl, tr, copy_weights(df, tr), dates=df.date)
    te = df.index[(df.ts >= TEST0) & (df.ts < TEST1)]
    p_check = model.predict_proba(Xc.loc[te])[:, 1]
    assert np.allclose(p_check.round(4), hours.loc[te, "p0"].to_numpy(), atol=1e-4), "classifier mismatch with run_all"
    contrib = model.predict_proba(Xc.loc[te], pred_contrib=True)[:, :-1]
    top = []
    for i, idx in enumerate(te):
        order = np.argsort(-contrib[i])[:3]            # factors pushing risk UP most
        top.append([(Xc.columns[j], float(Xc.loc[idx, Xc.columns[j]]), float(contrib[i, j])) for j in order])
    hours["top_factors"] = pd.Series(top, index=te)

    # ------------------------------------------------ pre-test statistics (Jun-Jul)
    pre = (df.ts >= PRE0) & (df.ts < PRE1)
    exc = pd.Series(np.clip(df[Q].to_numpy() - CAP, 0, None).sum(1) / 4, index=df.index)
    runs = (lbl != lbl.shift()).cumsum()
    rl = lbl[pre].groupby(runs[pre]).agg(["first", "size"])
    rl = rl[rl["first"] == 1]["size"]
    # July margins: next-hour forecast shortfall on peak hours; day-ahead hourly-max error on running days
    Xb = build(df, use_weekly=False)
    trj = (df.ts >= START) & (df.ts < PRE0 + pd.DateOffset(months=1))
    tej = (df.ts >= pd.Timestamp("2021-07-01")) & (df.ts < PRE1)
    pj = predict_quarters(fit_quarters(Xb, df, trj), Xb, tej)
    shortfall = (df.loc[tej, Q].max(1) - pj.max(1))[lbl[tej] == 1]
    Xd = dayahead.build(df, "opcal2")
    pdj = predict_quarters(fit_quarters(Xd, df, trj), Xd, tej)
    run_j = tej & (df.state != IDLE)
    da_err = (df.loc[run_j, Q].max(1) - pdj.loc[df.index[run_j]].max(1))
    # peak rates for explanations (Jun-Jul only)
    hg = pd.cut(df.hour, [-1, 7, 11, 12, 14, 17, 23], labels=["00-07", "08-11", "12", "13-14", "15-17", "18-23"])
    lb = pd.cut(df.mean_load.shift(1), [-1, 80, 100, 120, 140, 1e9], labels=["<80", "80-100", "100-120", "120-140", ">=140"])
    rate = pd.DataFrame({"hg": hg, "lb": lb, "peak": lbl})[pre].groupby(["hg", "lb"], observed=True).peak.agg(["mean", "size"])
    daymax_temp = df.groupby("date").temp.max()
    prev_state = df.date.map(st.shift(1))
    restart = (df.state != IDLE) & (prev_state == IDLE)
    run_pre = pre & (df.state != IDLE)
    # data-quality / unusual-load thresholds (pre-test data only)
    qs_pre = df.loc[df.ts < PRE1, Q].to_numpy().ravel()
    jump_p999 = float(np.quantile(np.abs(np.diff(qs_pre)), 0.999))
    resid_jul = (df.loc[tej, Q].mean(1) - pj.mean(1)).abs()
    stats_monitor = {
        "jump_p999": round(jump_p999, 1),
        "resid_q99": round(float(resid_jul.quantile(0.99)), 1),
        "zeros_before_test": int((qs_pre == 0).sum()),
        "stuck_min_quarters": 8, "stuck_min_value": 35,
    }
    stats = {
        **stats_monitor,
        "cap": CAP,
        "excess_per_peak_hour_median": float(exc[pre & (lbl == 1)].median()),
        "excess_per_peak_hour_p80": float(exc[pre & (lbl == 1)].quantile(0.8)),
        "peak_run_hours_median": float(rl.median()), "peak_run_hours_p80": float(rl.quantile(0.8)),
        "n_peak_runs": int(len(rl)),
        "margin_peak": float(round(shortfall.median(), 1)),         # added to the next-hour forecast in RED hours
        "margin_dayahead": float(round(da_err.abs().median(), 1)),  # added to day-ahead forecast of later hours
        "peak_rate_restart_day": float(lbl[run_pre & restart].mean()),
        "peak_rate_running_day": float(lbl[run_pre].mean()),
        "peak_rate_hot": float(lbl[run_pre & (df.date.map(daymax_temp) >= 28)].mean()),
        "peak_rate_not_hot": float(lbl[run_pre & (df.date.map(daymax_temp) < 28)].mean()),
        "classifier_cutoff": float(cut0),
        "classifier_cutoff_calibrated": float(tp["cutoff_lead0_calibrated"].iloc[0]),
    }
    # July |actual mean - forecast mean| (forecaster trained <= Jun): warm start for the rolling unusual-load threshold
    gap_july = pd.Series((df.loc[tej, Q].mean(1) - pj.mean(1)).abs().to_numpy(), index=df.loc[tej, "ts"].to_numpy())
    cache = {"hours": hours, "stats": stats, "rate_table": rate, "gap_july": gap_july,
             "built_from": "outputs/final/test_predictions.csv"}
    pd.to_pickle(cache, OUT / "cache.pkl")
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()})
    print("hours cached:", len(hours))


if __name__ == "__main__":
    main()
