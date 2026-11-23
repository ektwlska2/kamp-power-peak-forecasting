"""Peak-agent replay demo (no API key needed).

    .venv/bin/streamlit run agent/app.py        then open http://localhost:8501

Pick a test day and an hour: the page shows what the agent knew and said up
to that hour, and answers operator questions with the rule-based router.
"""
import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.agent import RuleAgent  # noqa: E402
from agent.llm import ScriptedClient, llm_answer  # noqa: E402
from agent.ui_format import format_body  # noqa: E402
from src.data import Q  # noqa: E402

st.set_page_config(page_title="피크 관리 에이전트", page_icon=":material/bolt:", layout="wide")

TIER = {"RED": ("경고", "#c0392b", "#fdecea"), "YELLOW": ("주의", "#a86b00", "#fff5dc"),
        "GREEN": ("정상", "#1e7b34", "#e8f5ec")}
KIND = {"briefing": ("사전 브리핑", ":material/event_note:"), "pre_alert": ("사전 경보", ":material/schedule:"),
        "watch": ("주의", ":material/visibility:"), "action": ("경고 · 조치", ":material/bolt:"),
        "outcome": ("결과 기록", ":material/fact_check:"), "answer": ("답변", ":material/chat:"),
        "data": ("데이터 점검", ":material/sensors:"), "unusual": ("이상 부하", ":material/troubleshoot:")}
EXAMPLES = ["왜 경고야?", "지금 뭘 해야 해?", "비슷한 과거 날은?", "다음 시간 상태는?", "측정 이상 있어?", "오늘 브리핑"]

st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px;}
  h1 {font-size: 1.7rem !important; margin-bottom: 0 !important;}
  .sub {color:#5d636a; font-size:.92rem; margin: .2rem 0 1.2rem;}
  .chip {display:inline-block; padding:.35rem .9rem; border-radius:999px; font-weight:700; font-size:1.05rem;}
  .card {background:#fff; border:1px solid #e3e6e1; border-radius:10px; padding:.8rem 1rem; height:100%;}
  .card .lab {color:#6b7178; font-size:.8rem; margin-bottom:.25rem;}
  .card .val {font-size:1.45rem; font-weight:700; line-height:1.2;}
  .card .note {color:#6b7178; font-size:.8rem;}
  .msg {border:1px solid #e3e6e1; border-left-width:5px; border-radius:10px;
        padding:.75rem 1rem .7rem; margin:.6rem 0; font-size:.95rem; line-height:1.65; color:#1a1d21;}
  .msg .head {display:flex; align-items:center; gap:.5rem; flex-wrap:wrap; margin-bottom:.35rem;}
  .msg .time {font-weight:700; font-size:.95rem; font-variant-numeric:tabular-nums;}
  .msg .kind {font-size:.78rem; font-weight:600; padding:.1rem .55rem; border-radius:999px; background:#eef0ec; color:#4b5157;}
  .msg .tier {font-size:.78rem; font-weight:700; padding:.1rem .55rem; border-radius:999px; background:#ffffffcc;}
  .msg .badge {margin-left:auto; font-size:.82rem;}
  .msg .title {font-weight:700; font-size:1rem; margin:.1rem 0 .3rem;}
  .msg ul {margin:.1rem 0 0 1.1rem; padding:0;}
  .msg li {margin:.15rem 0;}
  .msg li ul {margin-top:.1rem;}
  .msg .lbl {font-weight:700;}
  .ok {color:#1e7b34; font-weight:600;} .ng {color:#c0392b; font-weight:600;}
  .legend {color:#5d636a; font-size:.8rem; margin-top:-.4rem;}
  .legend span {margin-right:1rem; white-space:nowrap;}
  .legend i {display:inline-block; width:14px; height:3px; vertical-align:middle; margin-right:.35rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_agent():
    return RuleAgent.create()


@st.cache_data
def day_messages(day):
    return [vars(m) for m in get_agent().run_day(pd.Timestamp(day))]


agent = get_agent()
df, h = agent.t.df, agent.t.h
days = list(pd.date_range("2021-08-01", "2021-09-14", freq="D"))
if "hour" not in st.session_state:
    st.session_state.hour = 13


def step_hour(delta):
    st.session_state.hour = min(23, max(0, st.session_state.hour + delta))


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 재생")
    day = st.selectbox("날짜", days, index=days.index(pd.Timestamp("2021-08-09")),
                       format_func=lambda d: d.strftime("%m월 %d일 (%a)"), key="day")
    st.slider("시각", 0, 23, key="hour", format="%d:00")
    b1, b2 = st.columns(2)
    b1.button("◀ 이전", on_click=step_hour, args=(-1,), width="stretch")
    b2.button("다음 ▶", on_click=step_hour, args=(1,), width="stretch")
    show_outcomes = st.toggle("결과 기록 보기", value=False, key="show_outcomes")
    st.divider()
    st.caption("테스트 기간 2021-08-01 ~ 09-14를 시간 단위로 재생합니다. 에이전트는 현재 시각 이전의 데이터와 "
               "그 시각에 발행된 예측만 사용합니다. 값의 단위는 데이터의 기록 단위입니다.")

hour = st.session_state.hour
now = pd.Timestamp(day) + pd.Timedelta(hours=hour)
rows = df.index[df.date == pd.Timestamp(day)]
stt = agent.t.status(now)
running = agent.t.day_briefing(pd.Timestamp(day))["running"]

st.title("전력 피크 관리 에이전트")
st.markdown("<div class='sub'>예측 → 2단계 경보 → SOP 조치 제안 → 검증 → 작업자 승인 → 결과 기록 · 규칙 기반 (API 키 불필요)</div>",
            unsafe_allow_html=True)

tab_live, tab_sop, tab_safe, tab_eval = st.tabs([":material/play_circle: 실시간 재생", ":material/menu_book: SOP",
                                                 ":material/shield: 안전장치", ":material/analytics: 재생 검증"])

# ================================================================== live
with tab_live:
    name, fg, bg = TIER[stt["tier"]]
    c0, c1, c2, c3 = st.columns([1.2, 1, 1, 1])
    c0.markdown(f"<div class='card'><div class='lab'>{now:%m월 %d일} {now:%H}:00 현재 경보</div>"
                f"<span class='chip' style='color:{fg};background:{bg}'>● {name}</span>"
                f"<div class='note' style='margin-top:.4rem'>{'가동일' if running else '휴지일'}</div></div>",
                unsafe_allow_html=True)
    c1.markdown(f"<div class='card'><div class='lab'>피크 확률 (이번 시간)</div><div class='val'>{stt['p_peak'] * 100:.0f}%</div>"
                f"<div class='note'>적색 기준 {agent.t.stats.get('classifier_cutoff_calibrated', agent.t.stats['classifier_cutoff']) * 100:.0f}% (보정 확률)</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='card'><div class='lab'>예측 최대 (이번 시간)</div><div class='val'>{stt['forecast_max']:.1f}</div>"
                f"<div class='note'>상한 {agent.t.cap}</div></div>", unsafe_allow_html=True)
    if "next_tier" in stt:
        nn, nfg, nbg = TIER[stt["next_tier"]]
        c3.markdown(f"<div class='card'><div class='lab'>다음 시간 {stt['next_hour']:02d}:00</div>"
                    f"<span class='chip' style='color:{nfg};background:{nbg};font-size:.95rem'>● {nn}</span>"
                    f"<div class='note' style='margin-top:.4rem'>피크 확률 {stt['next_p_peak'] * 100:.0f}%</div></div>",
                    unsafe_allow_html=True)
    else:
        c3.markdown("<div class='card'><div class='lab'>다음 시간</div><div class='val'>–</div></div>", unsafe_allow_html=True)

    # chart ------------------------------------------------------------
    dd = pd.DataFrame({"hour": df.loc[rows, "hour"].to_numpy(),
                       "실제": df.loc[rows, Q].max(axis=1).to_numpy(),
                       "다음 시간 예측": h.loc[rows, [f"fc_{q}" for q in Q]].max(axis=1).round(1).to_numpy(),
                       "하루 전 예측": h.loc[rows, [f"da_{q}" for q in Q]].max(axis=1).round(1).to_numpy(),
                       "tier": h.loc[rows, "tier0"].to_numpy()})
    x = alt.X("hour:Q", scale=alt.Scale(domain=[-0.5, 23.5], nice=False),
              axis=alt.Axis(values=list(range(0, 24, 2)), labelExpr="datum.value + ':00'", title=None, grid=False))
    band_df = dd[(dd.hour <= hour) & (dd.tier != "GREEN")].assign(x0=lambda d: d.hour - 0.5, x1=lambda d: d.hour + 0.5)
    # fixed colours (no colour encoding) so the bands do not share a colour scale with the lines
    bx = alt.X("x0:Q", scale=alt.Scale(domain=[-0.5, 23.5], nice=False))
    bands = (alt.Chart(band_df[band_df.tier == "RED"]).mark_rect(opacity=0.14, color="#c0392b").encode(x=bx, x2="x1:Q")
             + alt.Chart(band_df[band_df.tier == "YELLOW"]).mark_rect(opacity=0.18, color="#e0a100").encode(x=bx, x2="x1:Q"))
    y = alt.Y("value:Q", title="시간 최대 15분값", scale=alt.Scale(domain=[0, 240]))
    long = dd.melt(id_vars=["hour", "tier"], value_vars=["실제", "다음 시간 예측", "하루 전 예측"], var_name="series")
    long = long[(long.series == "하루 전 예측") | ((long.series == "실제") & (long.hour < hour)) |
                ((long.series == "다음 시간 예측") & (long.hour <= hour))]
    colors = alt.Scale(domain=["실제", "다음 시간 예측", "하루 전 예측"], range=["#1a1d21", "#1d5fae", "#a3a8ad"])
    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=x, y=y, color=alt.Color("series:N", scale=colors, legend=None),
        strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=["실제", "다음 시간 예측", "하루 전 예측"],
                                                              range=[[1, 0], [1, 0], [4, 4]]), legend=None))
    pts = alt.Chart(long[long.series != "하루 전 예측"]).mark_circle(size=34).encode(
        x=x, y=y, color=alt.Color("series:N", scale=colors, legend=None),
        tooltip=[alt.Tooltip("hour:Q", title="시"), alt.Tooltip("series:N", title="구분"), alt.Tooltip("value:Q", title="값", format=".1f")])
    cap = alt.Chart(pd.DataFrame({"y": [agent.t.cap]})).mark_rule(color="#c0392b", strokeDash=[3, 3]).encode(y="y:Q")
    cap_t = alt.Chart(pd.DataFrame({"y": [agent.t.cap], "x": [23.3], "t": [f"상한 {agent.t.cap}"]})).mark_text(
        align="right", dy=-7, color="#c0392b", fontSize=11).encode(x=alt.X("x:Q", scale=alt.Scale(domain=[-0.5, 23.5])), y="y:Q", text="t:N")
    now_r = alt.Chart(pd.DataFrame({"x": [hour]})).mark_rule(color="#6b7178", strokeWidth=1).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[-0.5, 23.5])))
    chart = (bands + lines + pts + cap + cap_t + now_r).properties(height=280).configure_view(strokeWidth=0) \
        .configure_axis(labelColor="#6b7178", titleColor="#6b7178", gridColor="#eceee9", domainColor="#d5d8d2")
    st.altair_chart(chart, theme=None, width="stretch")
    st.markdown("<div class='legend'><span><i style='background:#1a1d21'></i>실제 (지난 시간)</span>"
                "<span><i style='background:#1d5fae'></i>다음 시간 예측 (매시 발행)</span>"
                "<span><i style='background:repeating-linear-gradient(90deg,#a3a8ad 0 4px,transparent 4px 8px)'></i>하루 전 예측</span>"
                "<span><i style='background:rgba(192,57,43,.35)'></i>경고 시간</span>"
                "<span><i style='background:rgba(224,161,0,.35)'></i>주의 시간</span></div>", unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.markdown("#### 에이전트 타임라인")
        msgs = [m for m in day_messages(day) if pd.Timestamp(m["issued"]) <= now
                and (show_outcomes or m["kind"] != "outcome")]
        if not running:
            st.info("휴지일입니다. 사전 브리핑만 발행됩니다.", icon=":material/bedtime:")
        box = st.container(height=520, border=False)
        with box:
            for m in reversed(msgs):
                kname, _ = KIND[m["kind"]]
                tname, fg, bg = TIER[m["tier"]]
                badge = ""
                if m["kind"] == "action" and "passed" in m["check"]:
                    badge = ("<span class='badge ok'>✓ 검증 통과</span>" if m["check"]["passed"]
                             else "<span class='badge ng'>✕ 검증 실패 → 대체 조치</span>")
                elif m["kind"] == "outcome":
                    v = m["check"].get("verdict", "")
                    cls = "ok" if v == "적중" else "ng" if v in ("미탐지", "오경보") else ""
                    badge = f"<span class='badge {cls}'>{v}</span>"
                tint = "#ffffff" if m["kind"] == "outcome" else bg
                st.markdown(f"<div class='msg' style='border-left-color:{fg};background:{tint}'><div class='head'>"
                            f"<span class='time'>{pd.Timestamp(m['issued']):%H:%M}</span>"
                            f"<span class='kind'>{kname}</span>"
                            f"<span class='tier' style='color:{fg};border:1px solid {fg}55'>{tname}</span>"
                            f"{badge}</div>{format_body(m['text'])}</div>", unsafe_allow_html=True)

    with right:
        st.markdown("#### 작업자 질문")
        pick = st.pills("예시 질문", EXAMPLES, default=EXAMPLES[0], key="pick", label_visibility="collapsed")
        typed = st.text_input("직접 입력", placeholder="예: 왜 경고야?", key="q", label_visibility="collapsed")
        q = typed or pick or EXAMPLES[0]
        ans = agent.answer(q, now)
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant", avatar=":material/bolt:"):
            st.markdown(ans.text.replace("\n", "  \n"))
            if ans.grounded:
                st.caption(":material/verified: 모든 숫자가 도구 결과와 일치합니다.")
            else:
                st.caption(f":material/error: 근거 없는 숫자: {ans.unsupported}")

# ================================================================== SOP
with tab_sop:
    sop = agent.sop
    st.markdown("#### 경보 단계")
    st.dataframe(pd.DataFrame([{"단계": v["name_ko"], "의미": v["meaning_ko"], "담당": v.get("owner_ko", "–"),
                                "대응 시간": f"{v['response_minutes']}분" if "response_minutes" in v else "–",
                                "조치": " / ".join(v["actions_ko"])} for v in sop["tiers"].values()]),
                 hide_index=True, width="stretch")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### 전일 브리핑 조건")
        for k, v in sop["horizons"]["day_ahead"]["actions_ko"].items():
            st.markdown(f"- {v}")
        st.markdown("#### 이연 규칙")
        st.markdown(sop["deferral"]["note_ko"])
    with c2:
        st.markdown("#### 검증 실패 시 대체 조치")
        for v in sop["fallback_ko"][1:]:
            st.markdown(f"- {v}")
        st.markdown("#### 부하 우선순위 (도입 시 작성)")
        st.dataframe(pd.DataFrame([{"등급": r["class"], "설명": r["desc_ko"], "이연": "가능" if r["deferrable"] else "불가",
                                    "최대 이연": f"{r.get('max_delay_min', 0)}분" if r["deferrable"] else "–"}
                                   for r in sop["load_priority_template"]]), hide_index=True, width="stretch")
    st.caption("전체 절차서: docs/SOP.md")

# ================================================================== safeguard
with tab_safe:
    st.markdown("#### LLM을 연결해도 숫자를 지어낼 수 없습니다")
    st.markdown("LLM은 도구 결과로만 답을 쓰고, 문장 속 모든 숫자를 도구 결과와 대조합니다. 근거 없는 숫자가 하나라도 있으면 "
                "답변을 거부하고 규칙 기반 템플릿 답변을 보여 줍니다. 아래는 API 키 없이 스크립트로 재현한 시험입니다 (8월 9일 13:00).")
    t_now = pd.Timestamp("2021-08-09 13:00")
    fake = "13:00 경고입니다. 피크 확률 99%이며 지금 부하를 50 줄이면 전기요금 120만 원이 절감됩니다."
    good = agent.answer("왜 경고야?", t_now).text
    s1, s2 = st.columns(2, gap="large")
    with s1:
        res = llm_answer(agent, "왜 경고야?", t_now, ScriptedClient([good]))
        st.markdown("**근거가 있는 LLM 답변**")
        st.success(good.replace("\n", "  \n"), icon=":material/check_circle:")
        st.caption(f"결과: {'채택' if res.source == 'llm' else '거부'}")
    with s2:
        res2 = llm_answer(agent, "왜 경고야?", t_now, ScriptedClient([fake]))
        st.markdown("**숫자를 지어낸 LLM 답변**")
        st.error(fake, icon=":material/block:")
        st.caption(f"결과: 거부 · 근거 없는 숫자 {', '.join(res2.check['unsupported_numbers'])} → 템플릿 답변으로 대체")

# ================================================================== replay evaluation
with tab_eval:
    p = ROOT / "outputs" / "agent" / "replay_metrics.json"
    if not p.exists():
        st.info("run_all.py 또는 python -m agent.replay 를 먼저 실행하십시오.")
    else:
        mt = json.loads(p.read_text())
        st.markdown(f"#### 테스트 {mt['days']}일 (가동일 {mt['running_days']}일) 재생 결과")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("경보가 있던 피크 시간", f"{mt['peak_hours_with_red_or_watch']} / {mt['peak_hours']}")
        k2.metric("1시간 전 경고·이미 적색인 피크 첫 시간", f"{mt['onsets_warned_1h_ahead_or_already_red']} / {mt['peak_run_onsets']}")
        k3.metric("경보 시간의 초과 에너지", f"{mt['excess_energy_in_red_or_watch_hours_pct']}%")
        k4.metric("근거 확인 통과", f"{mt['grounding_pass_rate'] * 100:.0f}%")
        st.dataframe(pd.DataFrame([
            ("적색 조치 수", mt["red_actions"]),
            ("검증 통과 / 실패", f"{mt['verification_passed']} / {mt['verification_failed_fallback']}"),
            ("제안 감축량 ≥ 실제 필요량 (피크 시간)", mt["proposed_hold_enough_on_peak_hours"]),
            ("평균 제안 감축량 / 평균 실제 필요량", f"{mt['mean_hold_units_red']} / {mt['mean_need_units_on_peak_hours']}"),
            ("작업자 대상 메시지 (가동일당)", mt["operator_messages_per_running_day"]),
            ("최대 이연 1시간일 때 검증 통과", mt["checker_sensitivity"]["max_delay_1h"].replace(" passed", "")),
            ("최대 이연 2시간일 때 검증 통과", mt["checker_sensitivity"]["max_delay_2h"].replace(" passed", "")),
            ("최대 이연 4시간일 때 검증 통과", mt["checker_sensitivity"]["max_delay_4h"].replace(" passed", "")),
        ], columns=["항목", "결과"]).astype(str), hide_index=True, width="stretch")
        mo = mt.get("monitor")
        if mo:
            st.markdown("#### 데이터 품질·이상 부하 감시")
            m1, m2, m3 = st.columns(3)
            m1.metric("0값 발생 시간 탐지", f"{mo['zero_reading_hours_flagged']} / {mo['zero_reading_hours']}")
            m2.metric("입력 이상 시 예측 오차", f"{mo['forecast_MAE_inputs_affected']}",
                      f"정상 입력 {mo['forecast_MAE_inputs_clean']}", delta_color="off")
            m3.metric("이상 부하 알림 (45일)", f"{mo['unusual_load_messages']}건")
            st.dataframe(pd.DataFrame([
                ("테스트 이전(1–7월) 점검 시간 중 경고", f"{mo['pretest_hours_flagged']} / {mo['pretest_hours_checked']} (모두 급변 규칙)"),
                ("보고된 측정 이상 사건", f"{mo['data_fault_events_reported']}건 (발생·복귀 시에만 알림)"),
                ("급변 기준 (1–7월 15분 변화 99.9%)", mo["thresholds"]["jump_p999"]),
                ("멈춤 기준", "같은 값이 8구간(2시간) 이상 연속, 값 35 이상"),
                ("이상 부하 기준", "최근 28일 예측 차이의 상위 1% (7월 값으로 시작)"),
            ], columns=["항목", "결과"]).astype(str), hide_index=True, width="stretch")
        st.caption("재생 검증은 경보와 제안이 실제 상황과 얼마나 맞았는지를 측정합니다. 작업자가 조치했을 때 피크가 실제로 줄었는지는 "
                   "이 데이터로 확인할 수 없습니다.")
