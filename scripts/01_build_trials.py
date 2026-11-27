"""EPIC 데이터셋(.mat) → 시행 단위 통합 표(trials.csv.gz) + 단순반응시간(simple_rt.csv.gz)

사용 (프로젝트 폴더에서):
  python scripts/01_build_trials.py
  python scripts/01_build_trials.py --raw data/raw/jk9nb-osfstorage-Data-archive --out data/processed

출력 열(trials): participant, task, session, trial, condition, stim, response, correct, rt, extra, timestamp
  - rt 단위: 초 (No-Go 시행은 NaN)
  - correct: 1/0
  - session: 파일명 runN 의 N (= 세션 번호, 시간 순서)
필요 패키지: numpy, pandas, scipy
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

STROOP_COND = {0: "neutral", 1: "incongruent", 2: "congruent"}

# 열 위치는 README(Detailed Data Information)와 샘플 파일로 확인한 값
TASKS = {
    "flanker": dict(dir="Flanker", cols=dict(cond=0, stim=1, resp=2, acc=3, rt=4, ts=5)),
    "gonogo": dict(dir="Gonogo", cols=dict(cond=0, stim=1, resp=2, acc=3, rt=4, ts=5)),
    "primeprobe": dict(dir="Primeprobe", cols=dict(cond=0, stim=1, resp=2, acc=3, rt=4, extra=5, ts=6)),
    "stroop": dict(dir="Stroop", cols=dict(word=0, ink=1, cond=2, resp=3, rt=4, acc=5, wordcolor=6, ts=7)),
}


def unwrap(x):
    """MATLAB 셀/배열 껍질을 벗겨 파이썬 값 하나로."""
    while isinstance(x, np.ndarray):
        if x.size == 0:
            return None
        x = x.flatten()[0]
    if isinstance(x, np.generic):
        x = x.item()
    return x


def to_float(v):
    try:
        return float(v) if v is not None else np.nan
    except (TypeError, ValueError):
        return np.nan


def to_acc(v):
    if v is None:
        return np.nan
    if isinstance(v, str):
        s = v.strip().upper()
        return 1.0 if s == "TRUE" else 0.0 if s == "FALSE" else np.nan
    return 1.0 if bool(v) else 0.0


def parse_run(path: Path, task: str):
    spec = TASKS[task]
    c = spec["cols"]
    m = loadmat(str(path), squeeze_me=False)
    if "allData" not in m:
        raise ValueError("allData 변수가 없음")
    a = m["allData"]
    if a.ndim != 2 or a.shape[1] <= max(c.values()):
        raise ValueError(f"예상과 다른 모양 {a.shape}")
    rows = []
    for i in range(a.shape[0]):
        r = a[i]
        g = lambda k: unwrap(r[c[k]]) if k in c else None  # noqa: E731
        if task == "stroop":
            cond = STROOP_COND.get(int(g("cond")), "unknown") if g("cond") is not None else None
            stim = f"{g('word')}|ink{g('ink')}"
            extra = None
        else:
            cond = str(g("cond")).lower() if g("cond") is not None else None
            stim = g("stim")
            extra = g("extra")
        rows.append(dict(
            trial=i + 1, condition=cond, stim=None if stim is None else str(stim),
            response=None if g("resp") is None else str(g("resp")),
            correct=to_acc(g("acc")), rt=to_float(g("rt")),
            extra=None if extra is None else str(extra), timestamp=to_float(g("ts")),
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/jk9nb-osfstorage-Data-archive")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()

    root = Path(args.raw) / "rawData"
    if not root.exists():
        sys.exit(f"rawData 폴더를 찾을 수 없습니다: {root}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frames, rt_frames, problems = [], [], []
    for pdir in sorted(root.glob("EPIC*"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0)):
        pid = int(re.sub(r"\D", "", pdir.name))
        for task, spec in TASKS.items():
            for f in sorted((pdir / spec["dir"]).glob("run*.mat")):
                sess = int(re.search(r"run(\d+)", f.name).group(1))
                try:
                    rows = parse_run(f, task)
                except Exception as e:  # 파일 하나가 깨져도 전체는 계속
                    problems.append((str(f), repr(e)))
                    continue
                df = pd.DataFrame(rows)
                df.insert(0, "session", sess)
                df.insert(0, "task", task)
                df.insert(0, "participant", pid)
                frames.append(df)
        for f in sorted((pdir / "RTtest").glob("RT*.mat")):
            sess = int(re.search(r"RT(\d+)", f.name).group(1))
            try:
                arr = loadmat(str(f))["allRT"]
                d = pd.DataFrame({"participant": pid, "session": sess, "trial": np.arange(1, len(arr) + 1),
                                  "rt": arr[:, 0], "aux": arr[:, 1] if arr.shape[1] > 1 else np.nan})
                rt_frames.append(d)
            except Exception as e:
                problems.append((str(f), repr(e)))
        print(f"EPIC{pid} 완료", flush=True)

    if not frames:
        sys.exit("읽은 파일이 없습니다. --raw 경로를 확인하세요.")
    trials = pd.concat(frames, ignore_index=True)
    trials.to_csv(out / "trials.csv.gz", index=False)
    if rt_frames:
        pd.concat(rt_frames, ignore_index=True).to_csv(out / "simple_rt.csv.gz", index=False)

    print("\n== 요약 ==")
    print(trials.groupby(["task"]).agg(trials=("trial", "size"), participants=("participant", "nunique"),
                                       sessions=("session", "nunique")).to_string())
    print("저장:", out / "trials.csv.gz")
    if problems:
        print(f"\n읽기 실패 {len(problems)}개 (처음 5개):")
        for p, e in problems[:5]:
            print(" ", p, e)
        if any("v7.3" in e or "HDF" in e or "NotImplemented" in e for _, e in problems):
            print("  → v7.3(HDF5) 형식일 수 있습니다. pip install h5py 후 알려 주세요.")


if __name__ == "__main__":
    main()
