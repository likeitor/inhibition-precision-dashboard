"""시행 수에 따른 개인 간섭효과의 정밀도·신뢰도 계산 (numpy/pandas만 사용).
출력: data/processed/precision.csv, splithalf.csv, icc.csv
RT는 정답 시행 중 [MIN_RT, MAX_RT] 구간만 사용 (02와 동일 규칙).
"""
import numpy as np, pandas as pd
MIN_RT, MAX_RT = 0.15, 3.0
TASKS = ["flanker", "primeprobe", "stroop"]
KS = [5, 10, 20, 40, 80, 160, 320, 640, 1000, 1280]
N_REP = 300
rng = np.random.default_rng(42)

t = pd.read_csv("data/processed/trials.csv.gz",
                usecols=["participant", "task", "session", "condition", "correct", "rt"])
t = t[(t.correct == 1) & t.rt.between(MIN_RT, MAX_RT) & t.condition.isin(["congruent", "incongruent"])]

pool = {}  # (task, participant) -> (cong rts, incong rts)
for (task, p), g in t.groupby(["task", "participant"]):
    if task in TASKS:
        pool[(task, p)] = (g[g.condition == "congruent"].rt.values * 1000,
                           g[g.condition == "incongruent"].rt.values * 1000)

prec, split = [], []
for task in TASKS:
    people = sorted(p for (tk, p) in pool if tk == task)
    for k in KS:
        ok = [p for p in people if min(len(pool[(task, p)][0]), len(pool[(task, p)][1])) >= 2 * k]
        if len(ok) < 3:
            continue
        a_all, b_all = {p: [] for p in ok}, {p: [] for p in ok}
        for p in ok:
            c, i = pool[(task, p)]
            for _ in range(N_REP):
                ci = rng.permutation(len(c))[:2 * k]; ii = rng.permutation(len(i))[:2 * k]
                e1 = i[ii[:k]].mean() - c[ci[:k]].mean()
                e2 = i[ii[k:]].mean() - c[ci[k:]].mean()
                a_all[p].append(e1); b_all[p].append(e2)
            both = np.array(a_all[p] + b_all[p])
            prec.append(dict(task=task, participant=p, k=k, mean=both.mean(), sd=both.std(ddof=1),
                             lo=np.percentile(both, 2.5), hi=np.percentile(both, 97.5)))
        A = np.array([a_all[p] for p in ok]); B = np.array([b_all[p] for p in ok])
        rs = np.array([np.corrcoef(A[:, r], B[:, r])[0, 1] for r in range(N_REP)])
        sb = 2 * rs / (1 + rs)  # Spearman-Brown: 전체 2k 시행 기준 신뢰도
        split.append(dict(task=task, k=k, n_persons=len(ok), r_half=np.nanmean(rs),
                          r_sb=np.nanmean(sb), sb_lo=np.nanpercentile(sb, 2.5), sb_hi=np.nanpercentile(sb, 97.5)))
pd.DataFrame(prec).to_csv("data/processed/precision.csv", index=False)
pd.DataFrame(split).to_csv("data/processed/splithalf.csv", index=False)

# 세션 단위 ICC(1): 세션별 간섭효과의 사람 간 vs 사람 내 분산
e = pd.read_csv("data/processed/effects.csv.gz")
print("effects columns:", list(e.columns))
col_task = "task"; col_eff = [c for c in e.columns if c in ("effect", "measure", "metric")][0]
val = [c for c in e.columns if c in ("value", "estimate")][0]
def icc1(df):
    g = [x.values for _, x in df.groupby("participant")[val]]
    g = [x[~np.isnan(x)] for x in g]; g = [x for x in g if len(x) > 1]
    N = sum(len(x) for x in g); kk = len(g); gm = np.concatenate(g).mean()
    msb = sum(len(x) * (x.mean() - gm) ** 2 for x in g) / (kk - 1)
    msw = sum(((x - x.mean()) ** 2).sum() for x in g) / (N - kk)
    n0 = (N - sum(len(x) ** 2 for x in g) / N) / (kk - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw), msb, msw, n0
rows = []
for (task, eff), d in e.groupby([col_task, col_eff]):
    if eff not in ("rt_interference", "acc_interference", "false_alarm_rate"):
        continue
    est, msb, msw, n0 = icc1(d)
    ids = d.participant.unique(); bs = []
    for _ in range(500):
        samp = rng.choice(ids, len(ids)); parts = []
        for j, p in enumerate(samp):
            x = d[d.participant == p].copy(); x["participant"] = j; parts.append(x)
        try: bs.append(icc1(pd.concat(parts))[0])
        except Exception: pass
    rows.append(dict(task=task, effect=eff, icc=est, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                     between_sd=np.sqrt(max(msb - msw, 0) / n0), within_sd=np.sqrt(msw)))
pd.DataFrame(rows).to_csv("data/processed/icc.csv", index=False)
print(pd.DataFrame(split).round(3).to_string()); print(pd.DataFrame(rows).round(3).to_string())

# ---- 연습 효과 보정: 사람별 선형 추세(세션 번호)를 제거한 뒤 ICC 재계산 ----
def detrend(d):
    out = []
    for p, x in d.groupby("participant"):
        x = x.dropna(subset=[val]).copy()
        if len(x) > 2:
            b = np.polyfit(x.session.values, x[val].values, 1)
            x[val] = x[val] - np.polyval(b, x.session.values) + x[val].mean()
        out.append(x)
    return pd.concat(out)
rows2 = []
for (task, eff), d in e.groupby([col_task, col_eff]):
    if eff not in ("rt_interference", "acc_interference", "false_alarm_rate"):
        continue
    dd = detrend(d); est = icc1(dd)[0]; ids = d.participant.unique(); bs = []
    for _ in range(500):
        samp = rng.choice(ids, len(ids)); parts = []
        for j, p in enumerate(samp):
            x = dd[dd.participant == p].copy(); x["participant"] = j; parts.append(x)
        try: bs.append(icc1(pd.concat(parts))[0])
        except Exception: pass
    rows2.append(dict(task=task, effect=eff, icc_detr=est, lo_detr=np.percentile(bs, 2.5), hi_detr=np.percentile(bs, 97.5)))
m = pd.read_csv("data/processed/icc.csv").merge(pd.DataFrame(rows2), on=["task", "effect"])
m.to_csv("data/processed/icc.csv", index=False)
print(m[["task", "effect", "icc", "icc_detr", "lo_detr", "hi_detr"]].round(3).to_string())
