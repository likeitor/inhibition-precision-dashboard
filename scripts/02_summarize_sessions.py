"""시행 표(trials.csv.gz) → 참가자×과제×세션 요약 (conditions.csv.gz, effects.csv.gz)

사용:
  python scripts/02_summarize_sessions.py

분석 규칙 (아래 상수를 바꾸면 전체가 다시 계산됨. 대시보드 하단에 그대로 표기할 것)
  - RT 계산은 '정답' 시행만 사용
  - RT가 MIN_RT 미만(예측 반응)이거나 MAX_RT 초과(주의 이탈)인 시행은 RT 계산에서만 제외
  - 정확도 계산에는 RT 제외 규칙을 적용하지 않음

출력
  conditions: participant, task, session, condition, n, n_rt_valid, acc, mean_rt, median_rt, sd_rt
  effects   : participant, task, session, effect, value
     effect 종류
       rt_interference    불일치 − 일치 평균 RT (flanker, primeprobe, stroop)  [초]
       acc_interference   일치 정확도 − 불일치 정확도 (같은 과제들)           [비율]
       stroop_vs_neutral  Stroop 불일치 − 중립 평균 RT                         [초]
       stroop_facilitation Stroop 중립 − 일치 평균 RT                          [초]
       false_alarm_rate   Go/No-Go: No-Go 시행에서 반응한 비율               [비율]
       miss_rate          Go/No-Go: Go 시행에서 반응하지 못한 비율            [비율]
       go_rt              Go/No-Go: Go 정답 시행 평균 RT                      [초]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MIN_RT = 0.15  # 초
MAX_RT = 3.0   # 초

KEYS = ["participant", "task", "session"]


def summarize_conditions(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t["rt_ok"] = (t["correct"] == 1) & t["rt"].between(MIN_RT, MAX_RT)
    t["rt_valid"] = t["rt"].where(t["rt_ok"])
    g = t.groupby(KEYS + ["condition"], dropna=False)
    out = g.agg(
        n=("trial", "size"),
        n_rt_valid=("rt_ok", "sum"),
        acc=("correct", "mean"),
        mean_rt=("rt_valid", "mean"),
        median_rt=("rt_valid", "median"),
        sd_rt=("rt_valid", "std"),
    ).reset_index()
    return out


def derive_effects(c: pd.DataFrame) -> pd.DataFrame:
    w = c.pivot_table(index=KEYS, columns="condition", values=["mean_rt", "acc"], aggfunc="first")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.reset_index()
    rows = []

    def add(mask, effect, series):
        sub = w.loc[mask, KEYS].copy()
        sub["effect"] = effect
        sub["value"] = series[mask].to_numpy()
        rows.append(sub)

    for task in ("flanker", "primeprobe", "stroop"):
        m = w["task"] == task
        if m.any():
            add(m, "rt_interference", w["mean_rt_incongruent"] - w["mean_rt_congruent"])
            add(m, "acc_interference", w["acc_congruent"] - w["acc_incongruent"])
    m = w["task"] == "stroop"
    if m.any() and "mean_rt_neutral" in w:
        add(m, "stroop_vs_neutral", w["mean_rt_incongruent"] - w["mean_rt_neutral"])
        add(m, "stroop_facilitation", w["mean_rt_neutral"] - w["mean_rt_congruent"])
    m = w["task"] == "gonogo"
    if m.any():
        add(m, "false_alarm_rate", 1 - w["acc_nogo"])
        add(m, "miss_rate", 1 - w["acc_go"])
        add(m, "go_rt", w["mean_rt_go"])
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", default="data/processed/trials.csv.gz")
    ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()

    p = Path(a.trials)
    if not p.exists():
        sys.exit(f"{p} 가 없습니다. 먼저 scripts/01_build_trials.py 를 실행하세요.")
    t = pd.read_csv(p, usecols=["participant", "task", "session", "trial", "condition", "correct", "rt"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cond = summarize_conditions(t)
    eff = derive_effects(cond)
    cond.to_csv(out / "conditions.csv.gz", index=False)
    eff.to_csv(out / "effects.csv.gz", index=False)

    n_all = t["rt"].notna().sum()
    n_drop = ((t["correct"] == 1) & t["rt"].notna() & ~t["rt"].between(MIN_RT, MAX_RT)).sum()
    print(f"RT 제외 규칙: {MIN_RT}s 미만 또는 {MAX_RT}s 초과 → 정답 시행 중 {n_drop:,}개 제외 (RT 있는 시행 {n_all:,}개)")
    print("\n== 과제별 평균 효과 (참가자·세션 평균) ==")
    print(eff.groupby(["task", "effect"])["value"].agg(["count", "mean", "std"]).round(4).to_string())
    print("\n저장:", out / "conditions.csv.gz", out / "effects.csv.gz")


if __name__ == "__main__":
    main()
