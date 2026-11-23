"""Agent tests (plain asserts, no pytest needed).

    .venv/bin/python -m tests.test_agent
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.agent import RuleAgent  # noqa: E402
from agent.grounding import check  # noqa: E402
from agent.llm import ScriptedClient, llm_answer  # noqa: E402
from agent.tools import PeakTools  # noqa: E402
from src.data import Q  # noqa: E402

NOW = pd.Timestamp("2021-08-09 13:00")


def test_tools_use_no_future_data(agent):
    """Changing any recorded value at or after `now` must not change what the tools say at `now`."""
    df2 = agent.t.df.copy()
    fut = df2.ts >= NOW
    df2.loc[fut, Q + ["mean_load", "peak_q", "production", "temp"]] += 500
    t2 = PeakTools(df2, {"hours": agent.t.h, "stats": agent.t.stats, "rate_table": agent.t.rate,
                         "gap_july": agent.t.gap_hist[agent.t.gap_hist.index < pd.Timestamp("2021-08-01")]}, agent.sop)
    for name in ("status", "condition_rate", "analog_days", "propose", "data_check", "load_check"):
        a, b = getattr(agent.t, name)(NOW), getattr(t2, name)(NOW)
        assert a == b, f"{name} changed when future data changed"
    p = agent.t.propose(NOW)
    assert agent.t.verify(NOW, p) == t2.verify(NOW, p)


def test_day_briefing_uses_no_same_day_readings(agent):
    """The midnight briefing may use the shutdown calendar and forecasts, not the day's readings."""
    df2 = agent.t.df.copy()
    day = df2.date == NOW.normalize()
    df2.loc[day, Q + ["mean_load", "peak_q"]] += 500
    t2 = PeakTools(df2, {"hours": agent.t.h, "stats": agent.t.stats, "rate_table": agent.t.rate}, agent.sop)
    assert agent.t.day_briefing(NOW.normalize()) == t2.day_briefing(NOW.normalize())


def test_grounding(agent):
    ev = {"p": 0.78, "p_pct": 78, "hour": 13, "forecast_max": 188.0}
    assert check("13:00 피크 확률 78%, 예측 최대 188.0", ev, agent.sop)[0]
    ok, bad = check("13:00 피크 확률 97%, 절감 120만 원", ev, agent.sop)
    assert not ok and "97" in bad and "120" in bad
    assert not check("17:20 재투입", ev, agent.sop)[0]          # unsupported time


def test_all_scheduled_messages_grounded(agent):
    for d in pd.date_range("2021-08-01", "2021-09-14"):
        for m in agent.run_day(d):
            assert m.grounded, (m.issued, m.text, m.unsupported)


def test_all_question_types_grounded(agent):
    qs = ["왜 경고야?", "지금 뭘 해야 해?", "오늘 브리핑", "비슷한 과거 날은?", "다음 시간 상태는?", "안녕"]
    for d in ("2021-08-09", "2021-08-26", "2021-09-10"):
        for h in (2, 8, 13, 16):
            for q in qs:
                m = agent.answer(q, pd.Timestamp(d) + pd.Timedelta(hours=h))
                assert m.grounded, (d, h, q, m.text, m.unsupported)


def test_llm_safeguard(agent):
    good = agent.answer("왜 경고야?", NOW).text
    assert llm_answer(agent, "왜 경고야?", NOW, ScriptedClient([good])).source == "llm"
    bad = llm_answer(agent, "왜 경고야?", NOW, ScriptedClient(["피크 확률 99%, 50 줄이면 120만 원 절감"]))
    assert bad.source.startswith("template") and set(bad.check["unsupported_numbers"]) == {"99", "50", "120"}


def test_ledger_and_release_rules(agent):
    agent.run_day(NOW.normalize())
    hours = {int(agent.t.df.loc[r, "hour"]) for r, _ in agent.ledger}
    assert not hours & {8, 9, 10, 11, 13, 14}, "released into a risk-window hour"
    for (r, qi), units in agent.ledger.items():
        room = agent.t.cap - agent.t.h.loc[r, f"da_{Q[qi]}"] - agent.t.stats["margin_dayahead"]
        assert units <= room + 1e-6, "ledger over-fills a quarter"


def test_monitor_catches_outages_and_stays_quiet_before(agent):
    t = agent.t
    # every hour containing a zero reading is flagged when it completes
    zero_hours = agent.t.df.ts[(agent.t.df[Q] == 0).any(axis=1)]
    for zh in zero_hours:
        assert "zero" in t.data_check(zh + pd.Timedelta(hours=1))["flags"], zh
    # zero and stuck rules never fire before the test period
    for x in pd.date_range("2021-01-09", "2021-07-31 23:00", freq="h")[::7]:
        f = t.data_check(x)["flags"]
        assert "zero" not in f and "stuck" not in f, x


def test_unusual_load_uses_only_past_gaps(agent):
    """The rolling threshold for hour t-1 must not use the gap of t-1 itself or later hours."""
    now = pd.Timestamp("2021-09-04 08:00")
    th = agent.t.gap_threshold(now - pd.Timedelta(hours=1))
    hist = agent.t.gap_hist
    t2 = agent.t.gap_hist.copy()
    t2[t2.index >= now - pd.Timedelta(hours=1)] = 1e6
    agent.t.gap_hist = t2
    try:
        assert agent.t.gap_threshold(now - pd.Timedelta(hours=1)) == th
    finally:
        agent.t.gap_hist = hist


def test_monitor_messages_grounded(agent):
    for d in ("2021-08-28", "2021-08-29", "2021-09-04", "2021-09-05", "2021-09-08", "2021-09-11"):
        for m in agent.run_day(d):
            if m.kind in ("data", "unusual"):
                assert m.grounded, (m.text, m.unsupported)
    for ts in ("2021-08-28 18:00", "2021-09-08 16:00", "2021-08-09 13:00"):
        m = agent.answer("측정 이상 있어?", ts)
        assert m.grounded, (m.text, m.unsupported)


if __name__ == "__main__":
    agent = RuleAgent.create()
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t(agent)
        print("pass", t.__name__)
    print(f"{len(tests)} tests passed")
