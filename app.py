import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st

st.set_page_config(page_title="억제 통제 측정의 정밀도", layout="wide")
P = "data/processed/"
TASK_KO = {"flanker": "Flanker", "primeprobe": "Prime-Probe", "stroop": "Stroop", "gonogo": "Go/No-Go"}

import os
@st.cache_data
def load(stamp=None):
    return (pd.read_csv(P + "effects.csv.gz"), pd.read_csv(P + "precision.csv"),
            pd.read_csv(P + "splithalf.csv"), pd.read_csv(P + "icc.csv"))
eff, prec, split, icc = load(max(os.path.getmtime(P + f) for f in ["effects.csv.gz", "precision.csv", "splithalf.csv", "icc.csv"]))
eff["participant"] = eff["participant"].astype(str); prec["participant"] = prec["participant"].astype(str)

st.title("개인의 '억제 통제' 점수는 몇 번 측정해야 믿을 수 있을까?")
st.caption("EPIC 데이터셋(참가자 9명, 1인당 약 36회 세션) · 9명의 사례 연구이며 모집단 일반화는 불가합니다.")
tabs = st.tabs(["① 질문과 요약", "② 시행 수와 신뢰도", "③ 사람 간 vs 세션 간", "④ Go/No-Go", "⑤ 연습 효과", "⑥ 과제 간 관계(참고)", "⑦ 방법·한계·출처"])

with tabs[0]:
    st.markdown("""
**질문.** 간섭 효과(불일치 RT − 일치 RT)를 *한 사람*에 대해 안정적으로 재려면 시행이 몇 개 필요한가?
그리고 점수의 차이는 *사람 간 차이*인가, 같은 사람의 *날마다의 변동*인가?""")
    thr = st.slider("목표 신뢰도", 0.6, 0.95, 0.8, 0.05)
    def k_at(task, thr=0.8, col="r_sb"):
        """신뢰도가 thr에 처음 닿는 조건당 총 시행 수(두 절반 합 = 2k). 격자 사이는 선형 보간."""
        s = split[split.task == task].sort_values("k")
        k, r = s.k.values, s[col].values
        for i in range(1, len(k)):
            if r[i] >= thr > r[i - 1]:
                return int(round(2 * (k[i - 1] + (thr - r[i - 1]) / (r[i] - r[i - 1]) * (k[i] - k[i - 1]))))
        return None
    c = st.columns(3)
    for col, tk in zip(c, ["flanker", "primeprobe", "stroop"]):
        v, vl = k_at(tk, thr), k_at(tk, thr, "sb_lo")
        col.metric(f"{TASK_KO[tk]}: 필요한 조건당 총 시행", f"약 {v}개" if v else "—")
        col.caption(f"불확실성 하한까지 고려하면 약 {vl}개" if vl else "하한 기준은 이 격자에서 도달하지 않음")
    st.caption(f"신뢰도 {thr:.2f}에 도달하는 조건당 총 시행 수(두 절반 합계). 위 숫자는 300회 반복의 평균 기준이고, 작은 글씨는 2.5% 하한 기준입니다.")
    st.info("신뢰도는 사람 간 순위를 얼마나 일관되게 재현하느냐(분할-절반, Spearman-Brown 보정)이며, 시행 수가 적을수록 급격히 낮아집니다.")

with tabs[1]:
    task = st.selectbox("과제", ["flanker", "primeprobe", "stroop"], format_func=TASK_KO.get, key="t2")
    s = split[split.task == task]
    f = go.Figure()
    f.add_trace(go.Scatter(x=s.k, y=s.sb_hi, line=dict(width=0), showlegend=False))
    f.add_trace(go.Scatter(x=s.k, y=s.sb_lo, fill="tonexty", line=dict(width=0), name="2.5–97.5% 반복 범위"))
    f.add_trace(go.Scatter(x=s.k, y=s.r_sb, mode="lines+markers", name="평균 신뢰도"))
    f.add_hline(y=0.8, line_dash="dot")
    f.update_xaxes(type="log", title="조건당 시행 수 (절반당 k, 전체는 2k)"); f.update_yaxes(range=[-1.5, 1], title="신뢰도 (Spearman-Brown, 전체 2k 기준)")
    st.plotly_chart(f, use_container_width=True)
    st.caption("주의: 큰 k에서는 같은 시행 풀에서 반복 추출하므로 반복 간 독립성이 약해져 범위가 좁게 보일 수 있습니다.")
    p = prec[prec.task == task]
    fig = px.line(p, x="k", y="mean", color="participant", markers=True, log_x=True,
                  labels={"k": "조건당 시행 수", "mean": "간섭 효과 추정치 (ms)"})
    st.plotly_chart(fig, use_container_width=True)
    pk = st.select_slider("개인별 추정 불확실성 (조건당 시행 수)", sorted(p.k.unique()), value=40)
    q = p[p.k == pk].sort_values("participant")
    fig = go.Figure(go.Scatter(x=q.participant, y=q["mean"], mode="markers",
                               error_y=dict(type="data", symmetric=False, array=q.hi - q["mean"], arrayminus=q["mean"] - q.lo)))
    fig.update_layout(xaxis_title="참가자", yaxis_title="간섭 효과 (ms), 95% 범위")
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.markdown("세션별 간섭 효과의 **ICC(1)**: 1에 가까울수록 점수 차이가 '사람'에서, 0에 가까울수록 '날마다 변동'에서 옵니다. 막대는 참가자 부트스트랩 95% 구간.")
    d = icc.copy(); d["label"] = d.task.map(TASK_KO) + " · " + d.effect
    fig = go.Figure()
    fig.add_bar(x=d.label, y=d.icc, name="보정 전 (추세 포함)",
                error_y=dict(type="data", symmetric=False, array=d.hi - d.icc, arrayminus=d.icc - d.lo))
    fig.add_bar(x=d.label, y=d.icc_detr, name="보정 후 (연습 추세 제거)",
                error_y=dict(type="data", symmetric=False, array=d.hi_detr - d.icc_detr, arrayminus=d.icc_detr - d.lo_detr))
    fig.update_layout(barmode="group"); fig.update_yaxes(range=[0, 1], title="ICC(1)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**연습 효과를 고려했는가?** 사람별로 세션 번호에 대한 선형 추세를 뺀 뒤 ICC를 다시 구했습니다. "
                "보정 후 값이 오르면 세션 간 변동의 일부가 '연습에 따른 점수 하락'이었다는 뜻입니다. "
                "(선형 추세만 제거했으므로 비선형 학습 곡선은 남을 수 있습니다.)")
    task = st.selectbox("과제", ["flanker", "primeprobe", "stroop", "gonogo"], format_func=TASK_KO.get, key="t3")
    ef = "false_alarm_rate" if task == "gonogo" else "rt_interference"
    e = eff[(eff.task == task) & (eff.effect == ef)].copy()
    e["v"] = e.value * (100 if ef == "false_alarm_rate" else 1000)
    st.plotly_chart(px.strip(e, x="participant", y="v", color="participant",
                             labels={"v": "false alarm (%)" if ef == "false_alarm_rate" else "간섭 효과 (ms)"}), use_container_width=True)
    st.caption("점 하나 = 한 세션. 사람 안의 퍼짐(세션 간)과 사람 사이의 차이를 한눈에 비교합니다.")

with tabs[3]:
    st.markdown("**Go/No-Go: 같은 과제인데 사람마다 '멈추는 능력'이 왜 이렇게 다를까?** "
                "No-Go 자극에서 잘못 누른 비율(false alarm)은 사람마다 14~59%로 크게 다릅니다.")
    g = eff[eff.task == "gonogo"].pivot_table(index=["participant", "session"], columns="effect", values="value").reset_index()
    pm = g.groupby("participant")[["false_alarm_rate", "go_rt", "miss_rate"]].mean().reset_index()
    pm["fa"] = pm.false_alarm_rate * 100; pm["rt"] = pm.go_rt * 1000
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(pm.sort_values("fa"), x="participant", y="fa", labels={"fa": "false alarm (%)", "participant": "참가자"},
                           title="참가자별 평균 false alarm").update_xaxes(type="category"), use_container_width=True)
    c2.plotly_chart(px.scatter(pm, x="rt", y="fa", text="participant", labels={"rt": "Go 평균 RT (ms)", "fa": "false alarm (%)"},
                               title="빠르게 누르는 사람일수록 더 못 멈출까? (사람 간)").update_traces(mode="markers+text", textposition="top center"), use_container_width=True)
    r_between = pm.fa.corr(pm.rt)
    st.markdown(f"사람 간 상관 r = **{r_between:.2f}** (n=9, 방향만 참고). 빠르게 반응하는 사람이 No-Go에서 더 자주 실패하는 경향 = 속도–정확도 교환 가능성.")
    rows = []
    for pid, x in g.groupby("participant"):
        rows.append(dict(참가자=pid, **{"세션 내 r(FA, Go RT)": round(x.false_alarm_rate.corr(x.go_rt), 2)}))
    rr = pd.DataFrame(rows)
    st.markdown("**같은 사람 안에서도 성립하는가?** 세션마다 Go RT가 빠른 날 false alarm이 늘었는지(사람별 상관):")
    st.plotly_chart(px.bar(rr, x="참가자", y="세션 내 r(FA, Go RT)", range_y=[-1, 1]).update_xaxes(type="category"), use_container_width=True)
    st.warning("사람 간 패턴은 음의 상관이지만, 같은 사람 안에서는 부호가 제각각입니다. "
               "즉 '사람마다 전략이 다르다'는 설명은 가능해도, '날마다 속도를 올리면 실수가 는다'는 설명은 이 자료로 뒷받침되지 않습니다. "
               "9명의 관찰이라 인과나 일반화는 주장하지 않습니다.")

with tabs[4]:
    task = st.selectbox("과제", ["flanker", "primeprobe", "stroop"], format_func=TASK_KO.get, key="t4")
    e = eff[(eff.task == task) & (eff.effect == "rt_interference")].copy(); e["ms"] = e.value * 1000
    st.plotly_chart(px.line(e, x="session", y="ms", color="participant", labels={"ms": "간섭 효과 (ms)"}), use_container_width=True)
    g = e.groupby("session").ms.mean().reset_index()
    st.plotly_chart(px.line(g, x="session", y="ms", title="참가자 평균 (9명)"), use_container_width=True)

with tabs[5]:
    st.warning("n=9이므로 상관은 매우 불안정합니다. 참고용이며 결론에 사용하지 않습니다.")
    m = eff[eff.effect.isin(["rt_interference"]) | ((eff.task == "gonogo") & (eff.effect == "false_alarm_rate"))]
    w = m.groupby(["participant", "task"]).value.mean().unstack()
    st.plotly_chart(px.imshow(w.corr(), zmin=-1, zmax=1, text_auto=".2f", color_continuous_scale="RdBu_r"), use_container_width=True)

with tabs[6]:
    st.markdown("""
### 데이터
EPIC "Precise Individual Measures of Inhibitory Control" (OSF: jk9nb). 참가자 9명, 1인당 약 36회 세션, 과제는 Flanker·Go/No-Go·Prime-Probe·Stroop·단순 RT. 설문 자료는 비공개.

### 전처리
- RT는 **정답 시행**만, **0.15–3.0초** 범위만 사용. 정확도는 RT 구간과 무관하게 계산.
- 간섭 효과 = 불일치 평균 RT − 일치 평균 RT. Stroop은 중립 조건 비교도 계산.
- Go/No-Go: false alarm = No-Go에서 반응한 비율, Go RT = Go 정답 시행의 평균 RT.

### 분석 방법
- **신뢰도(②탭):** 사람별로 세션을 합친 시행에서 조건당 2k개를 무작위 추출해 둘로 나눠 간섭 효과를 구하고, 9명 간 상관을 300회 반복 평균(Spearman-Brown 보정). 그래프의 가로축 k는 절반당 시행 수이며, 보정된 신뢰도는 두 절반을 합친 2k 기준입니다. ①탭의 '필요한 조건당 총 시행'은 이 2k 값입니다. 0.8은 개인 수준 해석에 흔히 쓰는 관례적 기준이며, 슬라이더로 바꿀 수 있습니다.
- **ICC(③탭):** 세션 수준 효과로 비균형 ICC(1)을 구하고 참가자 부트스트랩(500회)으로 95% 구간 추정.
- **연습 효과 보정(③탭):** 사람별로 세션 번호에 대한 선형 추세를 제거한 뒤 ICC 재계산. 비선형 학습은 남을 수 있음.
- **Go/No-Go(④탭):** 사람 간 상관(평균값 간)과 사람 내 상관(세션 간)을 따로 계산.

### 한계
- **9명의 사례 연구이며 모집단 일반화는 불가합니다.** 신뢰구간이 넓은 값은 방향만 참고하세요.
- 한 사람 안의 시행은 독립이 아니므로 시행 수로 p값을 계산하지 않았습니다.
- 큰 시행 수(k≥640)에서는 같은 시행 풀에서 반복 추출해 반복 간 독립성이 약합니다.
- 사람 간 상관은 사람 내 과정을 뜻하지 않습니다(④탭).
- 과제 간 상관(⑥탭)은 n=9라 참고용입니다.

### 출처
EPIC dataset, OSF (https://osf.io/jk9nb). 데이터 라이선스와 인용 방식은 원 저장소의 안내를 따르세요.
""")
