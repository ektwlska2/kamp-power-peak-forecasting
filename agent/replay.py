"""Replay the rule agent over every test day (Aug 1 - Sep 14) and measure it.

    .venv/bin/python -m agent.replay

Writes outputs/agent/replay_messages.csv and replay_metrics.json.
The replay measures the agent's messages against what actually happened; it
does not (and cannot) measure whether operators would have reduced the peak.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.agent import RuleAgent  # noqa: E402
from agent.llm import ScriptedClient, llm_answer  # noqa: E402
from src.data import Q  # noqa: E402

OUT = ROOT / "outputs" / "agent"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    agent = RuleAgent.create()
    df = agent.t.df
    cap = agent.t.cap
    days = pd.date_range("2021-08-01", "2021-09-14", freq="D")
    msgs = []
    for d in days:
        msgs += agent.run_day(d)
    m = pd.DataFrame([vars(x) for x in msgs])
    m["issued"] = pd.to_datetime(m.issued)
    m.drop(columns=["evidence"]).to_csv(OUT / "replay_messages.csv", index=False, encoding="utf-8-sig")

    test = df[(df.ts >= days[0]) & (df.ts < days[-1] + pd.Timedelta(days=1))].copy()
    test["peak"] = test[Q].max(axis=1) >= cap
    test["excess"] = np.clip(test[Q].to_numpy() - cap, 0, None).sum(axis=1) / 4
    test["need_units"] = np.clip(test[Q].max(axis=1) - cap, 0, None)
    running_days = int(test.groupby("date").state.first().ne(0).sum())

    act = m[m.kind == "action"].copy()
    act["hour_ts"] = act.issued
    act = act.merge(test[["ts", "peak", "excess", "need_units"]], left_on="hour_ts", right_on="ts", how="left")
    act["hold_units"] = act.check.map(lambda c: c.get("hold_units", 0.0) if isinstance(c, dict) else 0.0)
    act["passed"] = act.check.map(lambda c: c.get("passed") if isinstance(c, dict) else None)
    watch_ts = set(m[m.kind == "watch"].issued)
    pre_targets = set(m[m.kind == "pre_alert"].issued + pd.Timedelta(hours=1))
    red_ts = set(act.hour_ts)
    pk = test[test.peak]
    onset_ts = test.ts[test.peak & ~test.peak.shift(1, fill_value=False)].to_numpy()
    red_peak = act[act.peak]
    metrics = {
        "days": len(days), "running_days": running_days,
        "messages_by_kind": m.kind.value_counts().to_dict(),
        "operator_messages_per_running_day": round(len(m[m.kind.isin(["pre_alert", "watch", "action"])]) / running_days, 2),
        "grounding_pass_rate": round(float(m.grounded.mean()), 4),
        "ungrounded_messages": int((~m.grounded).sum()),
        "red_actions": len(act),
        "red_actions_defer": int((act.hold_units > 0).sum()),
        "red_actions_precaution_only": int((act.hold_units == 0).sum()),
        "verification_passed": int((act.passed == True).sum()),   # noqa: E712
        "verification_failed_fallback": int((act.passed == False).sum()),  # noqa: E712
        "red_actions_on_actual_peak_hours": int(act.peak.sum()),
        "peak_hours": int(pk.shape[0]),
        "peak_hours_with_red_action": int(pk.ts.isin(red_ts).sum()),
        "peak_hours_with_red_or_watch": int(pk.ts.isin(red_ts | watch_ts).sum()),
        "peak_hours_with_pre_alert_1h_ahead": int(pk.ts.isin(pre_targets).sum()),
        "peak_run_onsets": int(onset_ts.size),
        "onsets_warned_1h_ahead_or_already_red": int(pd.Series(onset_ts).isin(pre_targets | {t + pd.Timedelta(hours=1) for t in red_ts}).sum()),
        "onsets_with_red_or_watch_at_the_hour": int(pd.Series(onset_ts).isin(red_ts | watch_ts).sum()),
        "proposed_hold_enough_on_peak_hours": f"{int((red_peak.hold_units >= red_peak.need_units).sum())} / {len(red_peak)}",
        "excess_energy_total": round(float(test.excess.sum()), 2),
        "excess_energy_in_red_action_hours_pct": round(100 * float(test.excess[test.ts.isin(red_ts)].sum() / test.excess.sum()), 1),
        "excess_energy_in_red_or_watch_hours_pct": round(100 * float(test.excess[test.ts.isin(red_ts | watch_ts)].sum() / test.excess.sum()), 1),
        "mean_hold_units_red": round(float(act.hold_units.mean()), 1),
        "mean_need_units_on_peak_hours": round(float(pk.need_units.mean()), 1),
    }
    # data-quality and unusual-load monitor
    t = agent.t
    pre_hours = df.ts[(df.ts >= "2021-01-09") & (df.ts < days[0])]
    pre_flags = pd.Series([tuple(t.data_check(x)["flags"]) for x in pre_hours])
    test_hours = test.ts[test.ts > days[0]]
    tf = pd.DataFrame({"ts": test_hours.to_numpy(),
                       "dq": [tuple(t.data_check(x)["flags"]) for x in test_hours],
                       "affected": [t.inputs_affected(x) for x in test_hours]})
    tf["checked_hour"] = tf.ts - pd.Timedelta(hours=1)
    zero_hours = set(test.ts[(test[Q] == 0).any(axis=1)])
    fc = t.h.set_index(t.h.ts)[[f"fc_{q}" for q in Q]]
    act_q = test.set_index("ts")[Q]
    err = (fc.loc[tf.ts].to_numpy() - act_q.loc[tf.ts].to_numpy())
    tf["abs_err"] = np.abs(err).mean(axis=1)
    un = m[m.kind == "unusual"].copy()
    un["hour"] = un.issued - pd.Timedelta(hours=1)
    ctx = test.set_index("ts")
    onset_set = set(pd.to_datetime(onset_ts))
    prev_state = df.set_index("date").state.groupby(level=0).first().shift(1)
    metrics["monitor"] = {
        "thresholds": {"jump_p999": t.stats["jump_p999"], "resid_q99_july_warm_start": t.stats["resid_q99"], "rolling": "q99 of |gap|, previous 28 days",
                       "stuck": f">= {t.stats['stuck_min_quarters']} quarters at >= {t.stats['stuck_min_value']}"},
        "pretest_hours_checked": int(len(pre_flags)),
        "pretest_hours_flagged": int((pre_flags.map(len) > 0).sum()),
        "pretest_flag_types": pre_flags[pre_flags.map(len) > 0].map(lambda f: "+".join(f)).value_counts().to_dict(),
        "test_hours_flagged": int((tf.dq.map(len) > 0).sum()),
        "test_flag_types": tf.dq[tf.dq.map(len) > 0].map(lambda f: "+".join(f)).value_counts().to_dict(),
        "zero_reading_hours": len(zero_hours),
        "zero_reading_hours_flagged": int(tf.checked_hour[tf.dq.map(len) > 0].isin(zero_hours).sum()),
        "data_fault_events_reported": int(((m.kind == "data") & (m.check.map(lambda c: c.get("event")) == "start")).sum()),
        "forecast_MAE_inputs_affected": round(float(tf.abs_err[tf.affected].mean()), 2),
        "forecast_MAE_inputs_clean": round(float(tf.abs_err[~tf.affected].mean()), 2),
        "hours_inputs_affected": int(tf.affected.sum()),
        "unusual_load_messages": int(len(un)),
        "unusual_load_detail": [
            {"hour": str(r.hour), "gap": r.check.get("gap"), "direction": r.check.get("direction"),
             "peak_onset_hour": r.hour in onset_set,
             "restart_day": bool(prev_state.get(r.hour.normalize(), -1) == 0 and ctx.loc[r.hour, "state"] != 0),
             "day_state": int(ctx.loc[r.hour, "state"])}
            for r in un.itertuples()],
    }
    # checker sensitivity: same RED proposals, stricter maximum delay (fresh ledger per day)
    sens = {}
    for delay in (1, 2, 4):
        passed = total = 0
        for d in days:
            ledger = {}
            for hr in range(24):
                now = d + pd.Timedelta(hours=hr)
                if now not in set(act.hour_ts):
                    continue
                p = agent.t.propose(now)
                if p["type"] != "defer":
                    continue
                v = agent.t.verify(now, p, ledger, max_delay_hours=delay)
                for k_, u in v.get("allocation", {}).items():
                    ledger[k_] = ledger.get(k_, 0.0) + u
                passed += v["passed"]; total += 1
        sens[f"max_delay_{delay}h"] = f"{passed} / {total} passed"
    metrics["checker_sensitivity"] = sens
    # LLM safeguard demonstration with a scripted client (no API key):
    now = "2021-08-09 13:00"
    faithful = agent.answer("왜 경고야?", now).text
    hallucinated = "13:00 경고입니다. 피크 확률 99%이며 지금 부하를 50 줄이면 전기요금 120만 원이 절감됩니다."
    ok_msg = llm_answer(agent, "왜 경고야?", now, ScriptedClient([faithful]))
    bad_msg = llm_answer(agent, "왜 경고야?", now, ScriptedClient([hallucinated]))
    metrics["llm_safeguard_demo"] = {
        "faithful_text_accepted": ok_msg.source == "llm",
        "hallucinated_text_rejected": bad_msg.source.startswith("template"),
        "rejected_numbers": bad_msg.check.get("unsupported_numbers"),
    }
    (OUT / "replay_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
