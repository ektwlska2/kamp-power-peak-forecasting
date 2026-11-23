# 보정 확률 기반 2단계 경보 피크 예측 및 기동 분산형 수요관리 AI 시스템

제6회 K-인공지능 제조데이터 분석 경진대회 · 과제 ⑤ 자원 최적화 AI 데이터셋 (`okm_augumented_2021.csv`)

**English summary.** Entry to the 6th K-AI Manufacturing Data Analysis Competition (KAMP), task 5: factory power demand and peak forecasting. A next-hour LightGBM forecaster and a calibrated peak-risk classifier (Platt scaling, conformal prediction intervals) drive a two-tier peak alert; a day-ahead forecast is reported with and without the production plan and shutdown calendar. The data audit found that 115 of 257 days are copied load profiles, so validation is time-split and copies are down-weighted (random-day cross-validation overstates PR-AUC by 41%). Peak-load error is 29% lower than the Chronos foundation model, and 94% of peak hours are alerted. A rule-based operations agent with a Streamlit dashboard turns the alerts into actions. Report: `report/결과보고서_최종.pdf`, slides: `slides/발표자료.pdf`.

## 저장소 구성

| 항목 | 위치 |
|---|---|
| 소스코드 | `run_all.py`(전체 파이프라인), `src/`, `agent/`, `experiments/`, `tests/`, `make_figures.py` |
| 패키지 목록 | `requirements.txt` |
| 결과보고서 · 발표자료 | `report/결과보고서_최종.pdf`, `slides/발표자료.pdf` |
| 그림 | `outputs/figures/` |

**데이터는 저장소에 포함하지 않는다.** 대회 데이터 `okm_augumented_2021.csv`는 KAMP(한국인공지능제조플랫폼)에서 받아 `data/raw/`에 넣는다. `outputs/`(그림 제외)는 `run_all.py`를 실행하면 생성된다.

## 실행 방법

macOS / Linux
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt      # macOS는 LightGBM용 libomp 필요: brew install libomp
.venv/bin/python run_all.py                    # 약 4분
```

Windows (PowerShell)
```powershell
py -3.14 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python run_all.py
```

- 개발 환경: Python 3.14.7 (macOS arm64). 패키지 버전은 `requirements.txt`에 고정했다.
- 모든 난수 시드 고정(`src/models.py`의 `SEED = 0`). 같은 환경에서 두 번 실행한 결과 모든 출력 파일이 바이트 단위로 동일했다.
- `run_all.py` 한 번으로 데이터 진단 → 학습 → 추론 → 테스트 예측 파일 → 보고서용 표 → 에이전트 재생 검증 → 실험 15–18까지 자동 실행된다.

### `run_all.py` 단계

| 단계 | 내용 | 주요 출력 |
|---|---|---|
| 1 | 데이터 진단: 시간 재구성, 결측·중복·0값, 복제일 탐지, 일 가동상태 | `outputs/final/copy_map.csv` |
| 2 | 다음 시간 예측: 베이스라인 3종, Ridge, LightGBM(최종), Chronos-2 비교 | `ch2_next_hour_model_comparison.csv` |
| 3 | 피크 분류기, 확률 보정, 90% 예측 구간, 2단계 경보 | `ch2_peak_detector.csv`, `ch2_calibration.csv`, `ch4_two_tier_alert.csv` |
| 4 | 주간 재학습 대 고정 모델 | `ch4_weekly_retraining*.csv` |
| 5 | 하루 전 예측 (엄격 / +생산계획 / +가동 캘린더), 테스트 예측 결과 저장 | `ch4_day_ahead_value_of_information.csv`, **`test_predictions.csv`** |
| 6 | 영향요인(SHAP)과 오류 분석 | `ch3_*.csv` |
| 7 | 생산량 재배치 what-if | `ch4_production_whatif_days.csv` |
| 8 | 피크 상한(부하 이연) 시뮬레이션 | `ch4_peak_cap_simulation.csv`, `ch4_monthly_max_capped.csv` |
| 9 | 핵심 수치 요약 | `summary.json` |
| 10 | 에이전트 캐시 생성과 테스트 45일 재생 검증 | `outputs/agent/` |
| 11 | 실험 15–18 (확률 보정, 예측 구간, 피크 오차 신뢰구간, 복제 누설) | `outputs/exp15–18/` |

## 폴더 구성

```
README.md
requirements.txt
run_all.py                전체 파이프라인 (단일 실행 진입점)
make_figures.py           보고서 그림 생성 → outputs/figures/ (선택, 한글 글꼴 NanumGothic 사용)
src/
  data.py                 데이터 로드, 시간 재구성(행 순서 기준), 복제일 탐지
  features.py             다음 시간 예측 특징 (과거값만 사용, 누설 검사 포함)
  dayahead.py             하루 전 예측 특징 (엄격 / 생산계획 / 가동 캘린더 / 기상)
  state.py                일 가동상태 (휴지 / 부분 / 가동)
  peak.py                 피크 라벨, 피크 맥락 특징, 탐지 지표
  models.py               예측기, 피크 분류기, 2단계 경보, what-if, 피크 상한 시뮬레이션
  uncertainty.py          확률 보정, conformal 예측 구간, 블록 부트스트랩 신뢰구간
  weights.py              복제일 가중치 1/n
  metrics.py              MAE / RMSE / 피크 MAE / F1
agent/                    피크 관리 운영 에이전트 (아래 참조), sop.json = 기계가 읽는 SOP 규칙
tests/test_agent.py       에이전트 시험 10종
experiments/01~18         개발 과정의 모든 실험 (채택하지 않은 시도 포함, 아래 참조)
docs/
  SOP.md                  전력 피크 관리 표준작업절차서
data/                     (저장소 미포함)
  raw/okm_augumented_2021.csv               학습용 원본 데이터
  reference/chronos2_zero_shot_predictions.csv  비교 기준 Chronos-2 예측값
report/결과보고서_최종.pdf   결과보고서
slides/발표자료.pdf          발표자료
outputs/                  (figures/ 외에는 run_all.py 실행 시 생성)
  final/                  제출용 결과 (아래 참조)
  agent/                  에이전트 45일 재생 검증 결과
  exp15/ ~ exp18/         실험 15–18 결과
  figures/                보고서 그림 17종
.streamlit/config.toml    시연 화면 설정
```

## 주요 산출물 (`outputs/final/`)

| 파일 | 내용 |
|---|---|
| `test_predictions.csv` | **테스트 예측 결과 파일.** 실제값, 다음 시간 예측 4개(고정 모델, 주간 재학습 모델)와 90% 예측 구간(`pi90_lower/upper_*`), 피크 확률(원값·Platt 보정값)과 보정 기준값, 경보 단계와 황색 사유(lead 0, lead 1), 하루 전 예측(엄격 / 가동 캘린더) |
| `ch2_next_hour_model_comparison.csv` | 베이스라인 3종, Ridge, LightGBM(최종, +주간 시차, 잔차), Chronos-2, 평균 앙상블 비교 |
| `ch2_peak_detector.csv`, `ch2_calibration.csv` | 피크 분류기와 규칙 기반 탐지 비교, 확률 보정 |
| `ch3_*.csv` | SHAP 영향요인과 상호작용, 조건별 예측 오차, 경보 미탐지 목록, 피크 발생 조건 |
| `ch4_*.csv` | 2단계 경보, 주간 재학습, 하루 전 정보 가치, 생산량 재배치 what-if, 피크 상한 시뮬레이션 |
| `copy_map.csv` | 날짜별 복제 여부 (unique / first / copy) |
| `summary.json` | 핵심 수치 요약 |

## 예측 규약

| 항목 | 내용 |
|---|---|
| 다음 시간 예측 (주 결과) | t시 시작 시점에 t-1시까지 완료된 데이터만으로 t시의 15/30/45/60분 값 4개를 예측 |
| 하루 전 예측 (시나리오) | D-1일 24시 기준으로 D일 24시간 예측. 엄격 / +생산계획 / +가동 캘린더 |
| 검증 | 2021-07 (학습 01-08 ~ 06-30) |
| 테스트 | 2021-08-01 ~ 09-14 (학습 01-08 ~ 07-31, 모델 고정, 입력은 매시 갱신) |
| 배포 시험 | 같은 테스트 기간에 7일마다 그 시점까지의 데이터로 재학습 (주간 재학습) |
| 피크 | 15분 값 중 하나라도 190 이상인 시간 (분석 기준값, 계약전력 아님) |

테스트 기간은 개발 과정에서 조회되었으므로 완전히 미사용된 평가 자료는 아니다. 이 사실은 보고서에도 명시했다.

## 데이터 처리 원칙 (최소 정제)

- 2021-07-13과 07-15의 잘못된 `시간` 값 48개는 행 순서로 0~23시를 재구성했다. 두 날짜 모두 행이 24개씩이다.
- 0값 74개는 원인을 알 수 없으므로 그대로 유지하고, 운영 단계에서 데이터 품질 감시로 탐지한다.
- 제외한 변수:
  - `평균`: 네 개 15분 값의 평균(반올림)이므로 목표 누설
  - `공장인원`: `생산량 / (15분+30분+45분+60분)`으로 계산된 값이므로 목표 파생
  - `인건비`: 09~17시는 1.0, 그 외는 1.5인 시각 규칙이라 시각 정보와 중복
- 257일 중 160일은 다른 날짜와 96개 전력값이 완전히 같은 복제 프로파일이며, 대부분 생산량은 복제되지 않았다. 피크 분류기와 부하반응 모델에서는 이런 날에 가중치 1/n을 적용했다.

## 피크 관리 운영 에이전트 (보고서 4장)

`run_all.py`를 먼저 실행해야 한다. 10단계에서 에이전트 캐시(`agent/cache/`)가 생성된다.

```bash
.venv/bin/python -m tests.test_agent           # 에이전트 시험 10종 (미래 데이터 미사용, 근거 확인, 안전장치, 데이터 감시 등)
.venv/bin/streamlit run agent/app.py           # 재생 시연 (기본: 8월 9일 재가동일)
```

- API 키가 필요 없다. 기본 모드는 규칙 기반(템플릿)이다. LLM 연결부(`agent/llm.py`)는 선택 사항이며, 근거 없는 숫자가 있는 답변은 거부하고 템플릿 답변으로 대체한다.
- 절차서: `docs/SOP.md`, 기계가 읽는 규칙: `agent/sop.json`

| 파일 | 내용 |
|---|---|
| `agent/build_cache.py` | 테스트 예측 결과로 에이전트용 캐시 생성 (`run_all.py` 10단계) |
| `agent/tools.py` | 상태, 원인 설명(SHAP), 유사 과거일, 전일 브리핑, 조치 제안, 조치 검증, 결과 기록, 데이터 품질·이상 부하 감시 |
| `agent/agent.py` | 일정에 따른 메시지 발행과 질문 응답 (규칙 기반) |
| `agent/grounding.py` | 메시지 속 모든 숫자가 도구 결과에서 왔는지 확인 |
| `agent/llm.py` | 선택형 LLM 연결부와 시험용 스크립트 클라이언트 |
| `agent/replay.py` | 테스트 45일 재생 검증 |
| `agent/app.py`, `agent/ui_format.py` | Streamlit 시연 화면 |

## 실험 (`experiments/`)

실험 15–18은 `run_all.py`가 자동 실행한다. 실험 01–14는 개발 과정 기록으로, 각각 개별 실행하며 결과는 `outputs/expNN/`에 저장된다. 05·07·08·11·12는 03·04의 결과를 사용하므로 **번호 순서대로** 실행한다.

```bash
.venv/bin/python experiments/03_state_model.py    # 예: 개별 실행
```

| 실험 | 내용 |
|---|---|
| 01 | 데이터 진단: 품질, 복제일 지도, 공장인원·인건비 검사 |
| 02, 02b–02d | 주간 시차(지난주 같은 시각) 특징의 오염 검사와 진단 → 제외 |
| 03 | 가동상태 확률 특징 대 주간 시차 |
| 04 | 피크 분류기 (가동상태 특징 포함 여부) |
| 05 | 2단계 경보 (04의 점수 사용) |
| 06 | 하루 전 예측: 엄격 대 생산계획 시나리오 |
| 07 | Chronos-2와 동일 시간 비교 |
| 08 | 영향요인과 오류 분석 (보고서 3장) |
| 09 | 과거 기온의 피크 분류기 기여 → 제외 |
| 10 | 생산량 재배치 what-if |
| 11 | 피크 상한(부하 이연) 시뮬레이션 |
| 12 | 복제일의 검증 오염 검사 |
| 13 | 복제일 가중치 선택 (사전 선언 절차) |
| 14 | 주간 재학습 대 고정 모델 |
| 15 | 피크 분류기 확률 보정 (없음·Platt·등위 회귀, 7월 Brier로 선택 → Platt) |
| 16 | 90% conformal 예측 구간(고정·시간대별·롤링)과 구간 상한 황색 경보, 7월 별도 확인 |
| 17 | LightGBM 대 Chronos-2 피크 오차 차이의 7일 블록 부트스트랩 95% 구간 |
| 18 | 복제일을 무시한 교차검증의 성능 과대평가 (중복 표본 누설) |

## Chronos-2 비교 기준

`data/reference/chronos2_zero_shot_predictions.csv`는 사전학습 모델 Chronos-2를 zero-shot으로 실행한 예측값이다. 168시간 이력을 사용했고, t-1까지의 데이터만 쓰는 동일한 규약을 따른다. 최종 모델에는 사용하지 않으며 비교 기준으로만 쓴다. 파일이 없으면 해당 비교만 생략되고 나머지 파이프라인은 그대로 실행된다.
