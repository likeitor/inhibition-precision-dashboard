# 억제 통제 개인 측정 대시보드 (EPIC 데이터셋)

데이터: Lee, Smith, Hauenstein, Dworetsky, Kraus, Dorn, Nee & Gratton (2025).
Precise individual measures of inhibitory control. *Nature Human Behaviour*. DOI 10.1038/s41562-025-02198-2
OSF: https://osf.io/jk9nb  (원자료는 재배포하지 않고 요약 결과만 사용)

## 데이터 개요
- 참가자 9명, 1인당 36세션(EPIC10은 38), 과제 4종 + 단순 반응시간
- Flanker / Go-No-Go / Prime-Probe / Stroop: 1인당 약 6,900~7,800시행
- 설문 데이터는 비공개(저자 문의)

## 실행 순서
```bash
cd ~/Downloads/inhibition-dashboard
python -c "import scipy, pandas, numpy"      # 오류 나면: pip install -r requirements.txt

python scripts/01_build_trials.py            # .mat → data/processed/trials.csv.gz (+ simple_rt.csv.gz)
python scripts/02_summarize_sessions.py      # → conditions.csv.gz, effects.csv.gz
```

## 산출물 (data/processed)
| 파일 | 내용 |
|---|---|
| trials.csv.gz | 모든 시행 (participant, task, session, trial, condition, stim, response, correct, rt, extra, timestamp) |
| simple_rt.csv.gz | 단순 반응시간 시행 |
| conditions.csv.gz | 참가자×과제×세션×조건: n, 정확도, 평균/중앙값/SD RT |
| effects.csv.gz | 참가자×과제×세션: 간섭 효과 등 |

## 분석 규칙 (대시보드에 그대로 표기)
- RT는 정답 시행만 사용, 0.15초 미만·3.0초 초과는 RT 계산에서 제외 (`02_summarize_sessions.py` 상단 상수)
- run 번호 = 세션 번호(시간 순서)
- 9명의 사례 연구이므로 모집단 일반화는 하지 않음. 시행은 서로 독립이 아니므로 시행 수로 p값을 내지 않음

## 한계
- `simple_rt.csv.gz`의 `aux` 열 의미는 README에 없어 해석하지 않음
