"""Report figures (Korean labels). Run after run_all.py:

    .venv/bin/python make_figures.py        -> outputs/figures/*.png

Every number is read from saved results (outputs/final, outputs/agent,
outputs/exp03) or recomputed from the raw data; nothing is typed in by hand.
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.data import Q, copy_map, load  # noqa: E402
from src.peak import episode_onset, peak_label  # noqa: E402
from src.state import IDLE, daily_state  # noqa: E402

OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
F = ROOT / "outputs" / "final"

# ---------------------------------------------------------------- style
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, LIGHT = "#9a9d97", "#d9dbd6"
RED, YELLOW, GREEN = "#d03b3b", "#e0a100", "#0ca30c"
INK, INK2, GRID = "#1a1a1a", "#52514e", "#e6e6e1"
plt.rcParams.update({
    "font.family": "NanumGothic", "axes.unicode_minus": False, "font.size": 10.5,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110, "savefig.dpi": 220,
    "legend.frameon": False, "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def bar_labels(ax, bars, fmt="{:.1f}", dx=0.0, inside=False, color=INK, horizontal=False, pad=3):
    for b in bars:
        v = b.get_width() if horizontal else b.get_height()
        if horizontal:
            ax.annotate(fmt.format(v), (b.get_x() + v, b.get_y() + b.get_height() / 2), xytext=(pad, 0),
                        textcoords="offset points", va="center", ha="left", fontsize=9.5, color=color)
        else:
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_y() + v), xytext=(0, pad),
                        textcoords="offset points", ha="center", va="bottom", fontsize=9.5, color=color)


df = load()
st = daily_state(df)
df["state"] = df.date.map(st)
lbl = peak_label(df)
onset = episode_onset(lbl)

# ================================================================ Ch1
# fig01: copied days per month
cm = copy_map(df)
cnt = cm.assign(m=cm.index.month).groupby(["m", "status"]).size().unstack(fill_value=0).reindex(columns=["unique", "first", "copy"], fill_value=0)
fig, ax = plt.subplots(figsize=(8, 3.6))
x = np.arange(len(cnt))
bottom = np.zeros(len(cnt))
for col, color, name in (("unique", BLUE, "고유일 (1회 등장)"), ("first", AQUA, "반복 프로파일의 최초일"), ("copy", ORANGE, "복제일")):
    ax.bar(x, cnt[col], bottom=bottom, color=color, width=0.62, label=name, edgecolor="white", linewidth=1.2)
    bottom += cnt[col].to_numpy()
ax.set_xticks(x, [f"{m}월" for m in cnt.index])
ax.set_ylabel("일 수")
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=9.5)
ax.grid(axis="x", visible=False)
save(fig, "fig01_copied_days.png")

# fig02: hourly profile by day state
prof = df.groupby(["state", "hour"]).mean_load.mean().unstack(0)
fig, ax = plt.subplots(figsize=(8, 3.6))
for a, b in ((7.5, 11.5), (12.5, 14.5)):
    ax.axvspan(a, b, color="#e8eef8", zorder=0)
ax.text(9.5, 186, "위험 시간대", ha="center", fontsize=9, color=INK2)
ax.text(13.5, 186, "위험", ha="center", fontsize=9, color=INK2)
for s, color, name in ((2, BLUE, "가동일"), (1, AQUA, "부분 가동일"), (0, ORANGE, "휴지일")):
    ax.plot(prof.index, prof[s], color=color, lw=2.2, label=f"{name} ({int((st == s).sum())}일)")
ax.set_xticks(range(0, 24, 2), [f"{h:02d}시" for h in range(0, 24, 2)])
ax.set_ylabel("시간 평균 15분값")
ax.set_ylim(0, 195)
ax.legend(loc="upper left", fontsize=9.5)
save(fig, "fig02_day_state_profile.png")

# fig03: weekly-lag effect by test week (exp03 predictions, base vs +weekly, direct)
p3 = ROOT / "outputs" / "exp03" / "predictions.pkl"
if p3.exists():
    k = pd.read_pickle(p3)
    b, w = k[("AugSep", "base", "direct")], k[("AugSep", "+weekly", "direct")]
    y = df.loc[b.index, Q]
    diff = (w - y).abs().mean(axis=1) - (b - y).abs().mean(axis=1)
    wk = (df.loc[b.index, "date"] - pd.to_timedelta(df.loc[b.index, "dow"] - 1, unit="D"))
    eff = diff.groupby(wk).mean()
    eff = eff[eff.index >= pd.Timestamp("2021-08-02")]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    colors = [RED if v > 0 else BLUE for v in eff]
    bars = ax.bar(range(len(eff)), eff, color=colors, width=0.6)
    for i, v in enumerate(eff):
        ax.annotate(f"{v:+.2f}", (i, v), xytext=(0, 3 if v > 0 else -13), textcoords="offset points", ha="center", fontsize=9.5)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_xticks(range(len(eff)), [f"{d.month}/{d.day} 주" for d in eff.index])
    ax.set_ylabel("MAE 변화 (포함 - 미포함)")
    ax.set_ylim(eff.min() - 0.6, eff.max() + 1.2)
    for j in (0, 1):
        ax.annotate("휴지 주간(8/2–6)과 재가동 주간" if j else "", (j, eff.iloc[j] + 0.35), xytext=(0.5, eff.max() + 0.75), ha="center",
                    fontsize=9.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.legend(handles=[Patch(color=RED, label="악화"), Patch(color=BLUE, label="개선")], loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, "fig03_weekly_lag_effect.png")

# ================================================================ Ch2
cmp = pd.read_csv(F / "ch2_next_hour_model_comparison.csv")
name_ko = {"naive: previous hour": "기준: 직전 시간", "naive: same hour yesterday": "기준: 어제 같은 시각",
           "naive: same hour last week": "기준: 지난주 같은 시각", "Ridge (linear)": "Ridge (선형)",
           "LightGBM (final)": "LightGBM (최종)", "LightGBM + weekly lags": "LightGBM + 주간 시차",
           "LightGBM residual": "LightGBM 잔차형", "Chronos-2 (zero-shot benchmark)": "Chronos-2 (사전학습, 비교)",
           "average(LightGBM, Chronos-2)": "평균 앙상블 (참고)"}
cmp["ko"] = cmp.model.map(name_ko)


def color_of(m):
    return BLUE if m == "LightGBM (final)" else ORANGE if m.startswith("Chronos") else LIGHT


fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
order = cmp.sort_values("test_MAE", ascending=False)
for ax, col, title, lim in ((axes[0], "test_MAE", "전체 오차 (MAE)", 36), (axes[1], "test_peak_MAE", "피크 구간 오차 (190 이상)", 100)):
    d = order
    bars = ax.barh(d.ko, d[col], color=[color_of(m) for m in d.model], height=0.62)
    bar_labels(ax, bars, horizontal=True)
    ax.set_title(title)
    ax.set_xlim(0, lim)
    ax.grid(axis="y", visible=False)
    for t in ax.get_yticklabels():
        if "최종" in t.get_text():
            t.set_fontweight("bold")
axes[1].set_yticklabels([])
fig.text(0.01, -0.02, "테스트 기간 2021-08-01 ~ 09-14 (1,080시간). 낮을수록 좋음. 모든 모델은 t-1시까지의 데이터만 사용.",
         fontsize=9, color=INK2)
save(fig, "fig04_model_comparison.png")

# fig05: two-level alert coverage of the 68 peak hours + detector comparison
tiers = pd.read_csv(F / "ch4_two_tier_alert.csv")
t0 = tiers[(tiers.lead == 0) & (tiers.period == "test") & (tiers.variant == "two-tier + interval")].iloc[0]
det = pd.read_csv(F / "ch2_peak_detector.csv")
fig, axes = plt.subplots(2, 1, figsize=(8.5, 3.6), gridspec_kw={"height_ratios": [1, 1]})
for ax, total, parts, title in (
        (axes[0], int(t0.peak_hours), [("적색 경보로 탐지", int(t0.red_peaks), RED), ("황색 경보로 탐지", int(t0.any_peaks - t0.red_peaks), YELLOW),
                                       ("미탐지", int(t0.missed_peaks), LIGHT)], f"피크 시간 {int(t0.peak_hours)}개"),
        (axes[1], int(t0.onsets), [("적색", int(t0.red_onsets), RED), ("황색", int(t0.any_onsets - t0.red_onsets), YELLOW),
                                   ("미탐지", int(t0.onsets - t0.any_onsets), LIGHT)], f"피크 첫 시간 {int(t0.onsets)}개")):
    left = 0
    for name, v, c in parts:
        ax.barh(0, v, left=left, color=c, height=0.55, edgecolor="white", linewidth=2)
        if v:
            ax.text(left + v / 2, 0, f"{name}\n{v}", ha="center", va="center", fontsize=9.5,
                    color="white" if c in (RED,) else INK)
        left += v
    ax.set_xlim(0, total)
    ax.set_yticks([0], [title])
    ax.set_xticks([])
    ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
fig.text(0.01, -0.03, f"경보율: 피크 시간 {t0.recall_any * 100:.0f}%, 피크 첫 시간 {t0.any_onsets / t0.onsets * 100:.0f}%. "
         f"가동일당 적색 {t0.red_per_operating_day:.1f}시간, 황색 {t0.yellow_per_operating_day:.1f}시간.", fontsize=9, color=INK2)
save(fig, "fig05_alert_coverage.png")

# fig06: Aug 9 replay
tp = pd.read_csv(F / "test_predictions.csv", parse_dates=["ts"])
d9 = tp[tp.ts.dt.date == pd.Timestamp("2021-08-09").date()]
A = ["actual_15분", "actual_30분", "actual_45분", "actual_60분"]
Fc = ["forecast_15분", "forecast_30분", "forecast_45분", "forecast_60분"]
hrs = d9.ts.dt.hour.to_numpy()
act, fc = d9[A].max(axis=1).to_numpy(), d9[Fc].max(axis=1).to_numpy()
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.5, 4.2), gridspec_kw={"height_ratios": [5, 0.6]}, sharex=True)
ax.plot(hrs, act, color=INK, lw=2.2, marker="o", ms=3.5, label="실제 시간 최대값")
ax.plot(hrs, fc, color=BLUE, lw=2.2, marker="o", ms=3.5, label="다음 시간 예측 최대값")
ax.axhline(190, color=RED, ls="--", lw=1.2)
ax.text(23.4, 193, "피크 기준 190", ha="right", fontsize=9, color=RED)
miss = [(h, a) for h, a, t in zip(hrs, act, d9.tier_lead0) if a >= 190 and t == "GREEN"]
for h, a in miss:
    ax.annotate("미탐지", (h, a), xytext=(h + 0.8, a + 10), fontsize=9, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set_ylabel("15분값")
ax.set_ylim(0, 235)
ax.legend(loc="upper left", fontsize=9.5)
cmap = {"RED": RED, "YELLOW": YELLOW, "GREEN": "#e9ebe6"}
for h, t in zip(hrs, d9.tier_lead0):
    ax2.add_patch(plt.Rectangle((h - 0.45, 0), 0.9, 1, color=cmap[t]))
ax2.set_ylim(0, 1)
ax2.set_yticks([0.5], ["경보"])
ax2.grid(False)
ax.legend(handles=ax.get_legend_handles_labels()[0] + [Patch(color=RED, label="적색 경보"), Patch(color=YELLOW, label="황색 경보")],
          loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9.5, ncol=4)
ax2.set_xticks(range(0, 24, 2), [f"{h:02d}시" for h in range(0, 24, 2)])
for s in ("left", "bottom"):
    ax2.spines[s].set_visible(False)
save(fig, "fig06_aug9_replay.png")

# fig07: calibration of the peak classifier (raw vs Platt-calibrated), from experiment 15
rel = pd.read_csv(ROOT / "outputs" / "exp15" / "reliability.csv")
t15 = json.loads((ROOT / "outputs" / "exp15" / "test.json").read_text())
fig, ax = plt.subplots(figsize=(6.4, 5.2))
ax.plot([0, 1], [0, 1], color=LIGHT, lw=1.2, ls="--", label="완전 보정")
for kind, color, name in (("raw", GREY, "보정 전"), ("calibrated", BLUE, "Platt 보정 후")):
    g = rel[rel.kind == kind]
    ax.plot(g.mean_p, g.rate, color=color, marker="o", lw=2, label=name)
ax.set_xlabel("예측 피크 확률 (구간 평균)")
ax.set_ylabel("실제 피크 비율")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(loc="upper left", fontsize=9.5)
ax.text(0.98, 0.04, f"log loss {t15['raw']['logloss']:.3f} → {t15['calibrated']['logloss']:.3f}\n"
        f"Brier {t15['raw']['brier']:.4f} → {t15['calibrated']['brier']:.4f}\n경보 시간은 동일", ha="right", fontsize=9,
        color=INK2, transform=ax.transAxes)
save(fig, "fig07_calibration.png")

# fig08: weekly retraining by week
byw = pd.read_csv(F / "ch4_weekly_retraining_by_week.csv", parse_dates=["week_from"])
fig, ax = plt.subplots(figsize=(8, 3.4))
x = np.arange(len(byw))
b1 = ax.bar(x - 0.19, byw.MAE_frozen, width=0.36, color=LIGHT, label="8월 1일 고정 모델")
b2 = ax.bar(x + 0.19, byw.MAE_weekly, width=0.36, color=BLUE, label="매주 재학습")
for i, r in byw.iterrows():
    ax.annotate("동일" if i == 0 else f"{r.change:+.2f}", (i + 0.19, r.MAE_weekly), xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=9, color=INK2)
ax.set_xticks(x, [f"{d.month}/{d.day}~" for d in byw.week_from])
ax.set_ylabel("다음 시간 예측 MAE")
ax.legend(loc="upper left", fontsize=9.5)
ax.grid(axis="x", visible=False)
s = json.loads((F / "summary.json").read_text())["weekly_retraining"]
fig.text(0.01, -0.03, f"전체 MAE {s['MAE_frozen']:.2f} → {s['MAE_weekly']:.2f} (차이 95% 구간 {s['MAE_diff_CI95'][0]:.2f} ~ {s['MAE_diff_CI95'][1]:.2f})",
         fontsize=9, color=INK2)
save(fig, "fig08_weekly_retraining.png")

# ================================================================ Ch3
# fig09: peak rate heatmap clock x previous-hour load (Jun-Sep)
rate = pd.read_csv(F / "ch3_peak_rate_clock_x_prevload.csv", index_col=0)
n = pd.read_csv(F / "ch3_hours_clock_x_prevload.csv", index_col=0)
fig, ax = plt.subplots(figsize=(7.6, 3.9))
vals = rate.to_numpy(float)
masked = np.ma.masked_where((n.to_numpy() == 0) | np.isnan(vals), vals)
im = ax.imshow(masked * 100, cmap="Blues", vmin=0, vmax=70, aspect="auto")
for i in range(vals.shape[0]):
    for j in range(vals.shape[1]):
        if n.iloc[i, j] == 0:
            ax.text(j, i, "–", ha="center", va="center", color=GREY, fontsize=9)
        else:
            v = vals[i, j] * 100
            ax.text(j, i, f"{v:.0f}%\n({n.iloc[i, j]}h)", ha="center", va="center", fontsize=8.5,
                    color="white" if v > 40 else INK)
ax.set_xticks(range(len(rate.columns)), ["80 미만" if c == "<80" else c.replace(">=", "≥") for c in rate.columns])
ax.set_yticks(range(len(rate.index)), [f"{r}시" for r in rate.index])
ax.set_xlabel("직전 시간 평균 부하")
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label("피크 발생률 (%)")
save(fig, "fig09_peak_rate_heatmap.png")

# fig10: peak rate by temperature + forecast error by condition
temp = pd.read_csv(F / "ch3_peak_rate_by_temperature_riskhours.csv")
err = pd.read_csv(F / "ch3_forecast_error_by_condition.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.5]})
ax = axes[0]
labels = {"<25": "25 °C 미만", "25-28": "25–28 °C", ">=28": "28 °C 이상"}
bars = ax.bar([labels[t] for t in temp.temp], temp.peak_rate_by_temp * 100, color=ORANGE, width=0.55)
bar_labels(ax, bars, fmt="{:.0f}%")
ax.set_title("기온별 피크 발생률 (위험 시간대, 가동일)")
ax.set_ylabel("%")
ax.set_ylim(0, 70)
ax.grid(axis="x", visible=False)
lg = json.loads((F / "summary.json").read_text())["peak_temperature_logit"]
ax.text(0.02, 0.96, f"시각·생산량·직전 부하 보정 후\n1 °C당 오즈비 {lg['odds_ratio_per_C']:.2f} ({lg['OR_CI95'][0]:.2f}–{lg['OR_CI95'][1]:.2f})",
        transform=ax.transAxes, va="top", fontsize=8.8, color=INK2)
ax = axes[1]
sel = [("zero reading", "yes", "0값 포함 시간"), ("peak onset hour", "yes", "피크 첫 시간"), ("production change", "start (0->+)", "생산 시작 시간"),
       ("day type", "restart after idle", "휴지 후 재가동일"), ("day type", "operating", "가동일"), ("production change", "none", "생산 없음"),
       ("day type", "idle", "휴지일")]
rows = [(lab, float(err[(err.condition == c) & (err.level == l)].MAE.iloc[0])) for c, l, lab in sel]
rows = rows[::-1]
bars = ax.barh([r[0] for r in rows], [r[1] for r in rows], color=[ORANGE if r[1] > 10 else LIGHT for r in rows], height=0.6)
bar_labels(ax, bars, horizontal=True)
ax.axvline(float(cmp[cmp.model == "LightGBM (final)"].test_MAE.iloc[0]), color=BLUE, ls="--", lw=1)
ax.text(float(cmp[cmp.model == "LightGBM (final)"].test_MAE.iloc[0]) + 0.3, -0.6, "전체 평균", color=BLUE, fontsize=8.5)
ax.set_title("조건별 다음 시간 예측 오차 (MAE)")
ax.set_xlim(0, 27)
ax.grid(axis="y", visible=False)
save(fig, "fig10_temperature_and_error.png")

# fig11: SHAP drivers
sf = pd.read_csv(F / "ch3_shap_forecaster.csv", index_col=0)
sc = pd.read_csv(F / "ch3_shap_classifier.csv", index_col=0).iloc[:, 0]
feat_ko = {"q4_l1": "직전 60분값", "q3_l1": "직전 45분값", "hour": "시각", "max_l1": "직전 시간 최대값",
           "mean_l24": "어제 같은 시각 평균", "q4_l3": "3시간 전 60분값", "d_l1": "직전 1시간 변화",
           "intra_slope_l1": "직전 시간 내 상승폭", "risk_window": "위험 시간대 규칙", "hours_since_peak": "마지막 피크 후 경과",
           "q1_l1": "직전 15분값", "roll24": "최근 24시간 평균", "d_l24": "어제 같은 시각 변화", "q3_l2": "2시간 전 45분값",
           "max24": "최근 24시간 최대", "q2_l3": "3시간 전 30분값"}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
top = sf.sort_values("peak_hours", ascending=False).head(7)[::-1]
ax = axes[0]
yy_ = np.arange(len(top))
ax.barh(yy_ - 0.2, top.all_hours, height=0.38, color=LIGHT, label="전체 시간")
ax.barh(yy_ + 0.2, top.peak_hours, height=0.38, color=BLUE, label="실제 피크 시간")
ax.set_yticks(yy_, [feat_ko.get(i, i) for i in top.index])
ax.set_title("예측 모델 영향 요인 (평균 |SHAP|)")
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim(0, 9)
for yi, (a_, p_) in enumerate(zip(top.all_hours, top.peak_hours)):
    if max(a_, p_) > 9:
        ax.text(8.9, yi, f"{a_:.1f} / {p_:.1f} ›", ha="right", va="center", fontsize=8.8, color=INK, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
ax.grid(axis="y", visible=False)
ax = axes[1]
tc = sc.head(8)[::-1]
ax.barh([feat_ko.get(i, i) for i in tc.index], tc, color=[RED if i == "risk_window" else BLUE for i in tc.index], height=0.6)
ax.set_title("피크 분류기 영향 요인 (평균 |SHAP|)")
ax.grid(axis="y", visible=False)
save(fig, "fig11_shap_drivers.png")

# ================================================================ Ch4
# fig12: day-ahead value of information
da = pd.read_csv(F / "ch4_day_ahead_value_of_information.csv")
ko = {"strict (no target-day data)": "사전 정보 없음 (엄격)", "+ hourly production plan": "+ 시간별 생산계획",
      "+ shutdown calendar (idle vs running)": "+ 가동 캘린더 (가동/휴지)", "+ shutdown calendar + plan": "+ 가동 캘린더 + 생산계획",
      "+ plan + weather (recorded, optimistic)": "+ 생산계획 + 기상 (실측, 낙관적)", "+ 3-level state (upper bound)": "+ 3단계 가동상태 (상한선)"}
d = da.assign(ko=da.scenario.map(ko)).sort_values("test_MAE", ascending=False)
base_mae = float(da[da.scenario.str.startswith("strict")].test_MAE.iloc[0])
fig, ax = plt.subplots(figsize=(8.5, 3.6))
cols = [GREY if s.startswith("strict") else "#b9d3f2" if "upper" in s else BLUE for s in d.scenario]
bars = ax.barh(d.ko, d.test_MAE, color=cols, height=0.6)
for b, v, s in zip(bars, d.test_MAE, d.scenario):
    tag = "" if s.startswith("strict") else f" ({(v / base_mae - 1) * 100:+.0f}%)"
    ax.annotate(f"{v:.1f}{tag}", (v, b.get_y() + b.get_height() / 2), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9.5)
ax.set_xlabel("하루 전 예측 MAE (8–9월)")
ax.set_xlim(0, 32)
ax.grid(axis="y", visible=False)
save(fig, "fig12_dayahead_value.png")

# fig13: peak-cap simulation
cap = pd.read_csv(F / "ch4_peak_cap_simulation.csv")
c = cap[cap.period == "Aug-Sep"].sort_values("cap", ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
ax = axes[0]
bars = ax.bar([f"{v}\n({int(d)}일)" for v, d in zip(c.cap, c.days_needing_deferral)], c.deferred_share_of_day_energy_pct, color=BLUE, width=0.55)
bar_labels(ax, bars, fmt="{:.2f}%")
ax.set_title("상한 유지에 필요한 이연 에너지 (하루 에너지 대비)")
ax.set_xlabel("15분 수요 상한 (이연이 필요한 날 수)")
ax.set_ylim(0, c.deferred_share_of_day_energy_pct.max() * 1.3)
ax.grid(axis="x", visible=False)
ax = axes[1]
x = np.arange(len(c))
ax.bar(x - 0.19, c.excess_in_any_alert_lead0_pct, width=0.36, color=BLUE, label="이번 시간 경보 (적색+황색)")
ax.bar(x + 0.19, c.excess_in_any_alert_lead1_pct, width=0.36, color="#86b6ef", label="1시간 전 경보")
ax.set_xticks(x, [str(v) for v in c.cap])
ax.set_ylim(0, 110)
ax.set_title("상한 초과 에너지 중 경보 시간에 발생한 비율")
ax.set_xlabel("15분 수요 상한")
ax.set_ylabel("%")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9)
ax.grid(axis="x", visible=False)
for i, v in enumerate(c.excess_in_any_alert_lead0_pct):
    ax.annotate(f"{v:.0f}%", (i - 0.19, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.8)
save(fig, "fig13_peak_cap.png")

# fig14: system architecture
fig, ax = plt.subplots(figsize=(10, 3.9))
ax.set_xlim(0, 100)
ax.set_ylim(0, 40)
ax.axis("off")


def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc="white", ec=color, lw=1.8))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=10.5, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 7.2, body, ha="center", va="top", fontsize=8.6, color=INK2, linespacing=1.45)


def arrow(x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.3))


box(1, 21, 17, 17, "데이터", "15분 수요전력\n생산량·기온\n가동 캘린더", GREY)
box(22, 21, 19, 17, "예측", "다음 시간 예측\n(LightGBM, 주간 재학습)\n하루 전 예측", BLUE)
box(45, 21, 19, 17, "2단계 경보", "적색: 피크 분류기 (보정 확률)\n황색: 시계열 규칙 또는\n90% 예측 구간 상한 ≥ 190", RED)
box(68, 21, 31, 17, "에이전트 (규칙 기반, API 불필요)", "전일 브리핑 · 원인 설명 · 유사 과거일\nSOP 조치 제안 → 검증(상한 시뮬레이션)\n숫자 근거 확인 · 데이터 품질 감시", AQUA)
box(22, 1, 19, 15, "결과 기록", "적중·오경보·미탐지\n→ 주간 재학습", GREY)
box(45, 1, 19, 15, "작업자", "조치 승인·실행\n(시스템은 제안만)", GREY)
box(68, 1, 31, 15, "선택형 LLM 연결부", "로컬 LLM 또는 API 연결 가능\n근거 없는 숫자 → 거부 후 템플릿 대체", "#9085e9")
arrow(18.5, 29.5, 21.5, 29.5)
arrow(41.5, 29.5, 44.5, 29.5)
arrow(64.5, 29.5, 67.5, 29.5)
arrow(76, 20.5, 57, 16.5)
arrow(44.5, 8.5, 41.5, 8.5)
arrow(31.5, 16.5, 31.5, 20.5)
arrow(83.5, 16.5, 83.5, 20.5)
save(fig, "fig14_system_architecture.png")

# fig15: data-quality monitor on the Aug 28-29 outage
w = df[(df.ts >= "2021-08-28 06:00") & (df.ts < "2021-08-29 18:00")]
qs = w[Q].to_numpy().ravel()
tq = np.repeat(w.ts.to_numpy(), 4) + np.tile(pd.to_timedelta([0, 15, 30, 45], unit="min").to_numpy(), len(w))
msgs = pd.read_csv(ROOT / "outputs" / "agent" / "replay_messages.csv", parse_dates=["issued"])
dm = msgs[(msgs.kind == "data") & (msgs.issued >= "2021-08-28") & (msgs.issued < "2021-08-30")]
fig, ax = plt.subplots(figsize=(9, 3.3))
zero = qs == 0
ax.plot(tq, qs, color=INK, lw=1.4)
ax.fill_between(tq, 0, qs.max() * 1.05, where=zero, color="#fdecea", step="mid", label="0값 구간")
for _, r in dm.iterrows():
    col = RED if "의심" in r.text else GREEN
    ax.axvline(r.issued, color=col, lw=1.4, ls="--")
    ax.text(r.issued, qs.max() * 1.02, " 이상 알림" if "의심" in r.text else " 정상 복귀", color=col, fontsize=9, va="top")
ax.set_ylabel("15분값")
ax.legend(loc="center right", fontsize=9)
import matplotlib.dates as mdates  # noqa: E402
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H시"))
mo = json.loads((ROOT / "outputs" / "agent" / "replay_metrics.json").read_text())["monitor"]
fig.text(0.01, -0.04, f"8/28 17시 ~ 8/29 11시 측정 중단. 입력에 이상이 있던 시간의 예측 오차 {mo['forecast_MAE_inputs_affected']} "
         f"(정상 입력 {mo['forecast_MAE_inputs_clean']}).", fontsize=9, color=INK2)
save(fig, "fig15_monitor_outage.png")
# fig16: 90% prediction-interval coverage vs Chronos-2 q10-q90 (Aug-Sep)
cov = pd.read_csv(ROOT / "outputs" / "exp16" / "coverage.csv")
c16 = json.loads((ROOT / "outputs" / "exp16" / "summary.json").read_text())["chronos_q10_q90"]
c = cov[cov.period == "AugSep"].set_index("method")
names = [("rolling", "롤링 (최종)"), ("band", "시간대별"), ("global", "고정")]
labels = [n for _, n in names] + ["Chronos-2 q10–q90"]
allv = [c.loc[k, "coverage"] * 100 for k, _ in names] + [c16["coverage"] * 100]
pkv = [c.loc[k, "coverage_at_peaks"] * 100 for k, _ in names] + [c16["coverage_at_peaks"] * 100]
fig, ax = plt.subplots(figsize=(8, 3.6))
x = np.arange(len(labels))
b1 = ax.bar(x - 0.19, allv, width=0.36, color=[BLUE, BLUE, BLUE, ORANGE], label="전체 15분값")
b2 = ax.bar(x + 0.19, pkv, width=0.36, color=["#86b6ef", "#86b6ef", "#86b6ef", "#f4b293"], label="피크 값 (190 이상)")
bar_labels(ax, b1, fmt="{:.0f}%")
bar_labels(ax, b2, fmt="{:.0f}%")
ax.axhline(90, color=INK2, ls="--", lw=1)
ax.text(len(labels) - 0.5, 91.5, "목표 90%", ha="right", fontsize=9, color=INK2)
ax.set_xticks(x, labels)
ax.set_ylim(0, 110)
ax.set_ylabel("실제 포함률 (%)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9)
ax.grid(axis="x", visible=False)
save(fig, "fig16_interval_coverage.png")

# fig17: copy leakage - validation estimates vs what actually happened
cv = pd.read_csv(ROOT / "outputs" / "exp18" / "cv_vs_actual.csv")
ko = {"random-hour": "무작위 시간 분할", "random-day": "무작위 일 분할", "copy-group": "복제 그룹 분할",
      "time-ordered (July)": "시간순 (7월)", "ACTUAL Aug-Sep test": "실제 테스트 (8–9월)"}
cv["ko"] = cv.scheme.map(ko)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
cols = [ORANGE, ORANGE, BLUE, BLUE, INK]
for ax, col, title in ((axes[0], "PR_AUC", "피크 분류 PR-AUC 추정 (높을수록 좋음)"), (axes[1], "MAE", "다음 시간 예측 MAE 추정 (낮을수록 좋음)")):
    bars = ax.barh(cv.ko[::-1], cv[col][::-1], color=cols[::-1], height=0.6)
    bar_labels(ax, bars, fmt="{:.2f}", horizontal=True)
    ax.axvline(float(cv[col].iloc[-1]), color=INK, ls="--", lw=1)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, cv[col].max() * 1.25)
    for t in ax.texts:
        t.set_bbox(dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
axes[1].set_yticklabels([])
fig.text(0.01, -0.04, "주황: 복제일이 학습·검증에 함께 들어감 (검증일의 71%가 학습에 복제본 보유). 파랑: 복제 오염 없음. 점선: 실제 테스트 결과.",
         fontsize=9, color=INK2)
save(fig, "fig17_copy_leakage.png")
print("done ->", OUT)
