# Lunara — Feature Lifecycle Intelligence Dashboard
# UMBC Data Science Capstone | Author: Venkata Sai Chandravadan Sobila

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import json, os

st.set_page_config(page_title="Lunara", page_icon="🌙", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: "Inter", sans-serif; }
.stApp { background: #f8fafc; }
.block-container { padding-top: 4rem; padding-left: 1.8rem; padding-right: 1.8rem; max-width: 1680px; }
section[data-testid="stSidebar"] { background: #fff !important; border-right: 1px solid #e2e8f0 !important; }
section[data-testid="stSidebar"] * { color: #334155 !important; }
.sh { font-size:.98rem; font-weight:700; color:#0f172a; border-left:3px solid #6366f1; padding-left:9px; margin:20px 0 10px; }
.kpi { background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:14px 10px; text-align:center; height:112px; display:flex; flex-direction:column; justify-content:center; }
.kpi-val { font-size:1.45rem; font-weight:800; line-height:1.1; margin:3px 0; }
.kpi-lbl { font-size:.65rem; letter-spacing:.08em; text-transform:uppercase; color:#64748b; }
.kpi-sub { font-size:.68rem; margin-top:2px; opacity:.7; }
.fc { background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:14px 8px; text-align:center; min-height:148px; }
.badge { display:inline-block; padding:3px 9px; border-radius:999px; font-size:.63rem; font-weight:800; text-transform:uppercase; }
.invest   { background:#dcfce7; color:#047857; border:1px solid #86efac; }
.maintain { background:#dbeafe; color:#1d4ed8; border:1px solid #93c5fd; }
.refactor { background:#fef3c7; color:#92400e; border:1px solid #fbbf24; }
.sunset   { background:#fee2e2; color:#b91c1c; border:1px solid #fca5a5; }
.ev { background:#fff; border:1px solid #e2e8f0; border-radius:9px; padding:11px 13px; margin-bottom:8px; }
.risk-HIGH   { border-left:3px solid #ef4444; padding-left:8px; margin-bottom:6px; }
.risk-MEDIUM { border-left:3px solid #f59e0b; padding-left:8px; margin-bottom:6px; }
.risk-LOW    { border-left:3px solid #10b981; padding-left:8px; margin-bottom:6px; }
.step { background:#fff; border-left:3px solid #6366f1; border-radius:6px; padding:7px 11px; margin-bottom:5px; font-size:.8rem; color:#475569; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────
FEATURES = ["Search","Recommendations","Wishlist","Reviews","Notifications","Checkout"]
LC_COLORS = {"Invest":"#10b981","Maintain":"#3b82f6","Refactor":"#f59e0b","Sunset":"#ef4444"}
LC_ICONS  = {"Invest":"🚀","Maintain":"🔧","Refactor":"🔄","Sunset":"🌅"}
FT_ICONS  = {"Search":"🔍","Recommendations":"✨","Wishlist":"💝","Reviews":"⭐","Notifications":"🔔","Checkout":"🛒"}
PALETTE   = ["#6366f1","#10b981","#ec4899","#f59e0b","#ef4444","#0ea5e9"]
WEIGHTS   = {"adoption_trend":0.30,"engagement_intensity":0.25,"issue_burden":-0.25,"defect_risk":-0.20}
THRESHOLDS = {"Invest":(75,101),"Maintain":(50,75),"Refactor":(25,50),"Sunset":(0,25)}

# plotly config — show modebar with download (camera) button only
CFG = {"displayModeBar": True, "modeBarButtonsToRemove": [
    "zoom2d","pan2d","select2d","lasso2d","zoomIn2d","zoomOut2d","autoScale2d","resetScale2d",
    "hoverClosestCartesian","hoverCompareCartesian","toggleSpikelines",
], "modeBarButtonsToAdd": ["toImage"], "responsive": True}

def assign_lc(score):
    for lbl, (lo, hi) in THRESHOLDS.items():
        if lo <= score < hi: return lbl
    return "Maintain"

def kpi(lbl, val, clr, ico="", sub=""):
    return (f'<div class="kpi"><div style="font-size:1.1rem">{ico}</div>'
            f'<div class="kpi-val" style="color:{clr}">{val}</div>'
            f'<div class="kpi-lbl">{lbl}</div><div class="kpi-sub" style="color:{clr}99">{sub}</div></div>')

def sh(t): st.markdown(f'<div class="sh">{t}</div>', unsafe_allow_html=True)

def style_fig(fig, h=420, ml=70, mb=75):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff",
        height=h, margin=dict(l=ml,r=20,t=35,b=mb),
        font=dict(family="Inter",color="#0f172a",size=12),
        hoverlabel=dict(bgcolor="#fff",font_size=12,font_family="Inter",bordercolor="#cbd5e1"),
        legend=dict(font=dict(color="#334155",size=11),bgcolor="rgba(0,0,0,0)",
                    orientation="h",y=-0.22,x=0),
    )
    for ax in ["xaxis","yaxis"]:
        getattr(fig.layout, ax).update(
            gridcolor="rgba(100,116,139,0.2)", linecolor="#cbd5e1",
            tickfont=dict(color="#334155",size=11), zeroline=False, showline=True,
        )
    return fig

# ── Data ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading…")
def load():
    bases = ["/content", os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()]
    base  = bases[0] if os.path.exists(os.path.join(bases[0],"data","processed")) else bases[1]
    p     = lambda n: os.path.join(base,"data","processed",n)
    der   = pd.read_csv(p("weekly_feature_data.csv"), parse_dates=["week"])
    sc    = pd.read_csv(p("feature_value_scores.csv"))
    pr    = pd.read_csv(p("forecast_predictions.csv"), parse_dates=["week"])
    rep   = json.load(open(p("decision_reports.json")))
    return der, sc, pr, rep

try:
    derived, scores, fpreds, reports = load()
    OK = True
except Exception as e:
    OK = False; ERR = str(e)

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="padding:10px 4px 2px"><div style="font-size:1.4rem;font-weight:800;color:#6366f1">🌙 Lunara</div>'
                '<div style="font-size:.74rem;color:#64748b">Feature Lifecycle Intelligence</div></div><hr>', unsafe_allow_html=True)
    page = st.radio("nav", ["🏠 Portfolio Overview","📈 Adoption Forecasting",
                             "🎯 Feature Deep Dive","📋 Decision Reports"],
                    label_visibility="collapsed")
    st.markdown("<hr>", unsafe_allow_html=True)
    if OK:
        dr = st.date_input("Date window", value=(derived["week"].min().date(), derived["week"].max().date()),
                           min_value=derived["week"].min().date(), max_value=derived["week"].max().date(),
                           label_visibility="collapsed")
        sel = st.multiselect("Features", FEATURES, default=FEATURES, label_visibility="collapsed")
        if len(dr) == 1: dr = (dr[0], derived["week"].max().date())
        if not sel: sel = FEATURES
    else:
        dr = None; sel = FEATURES
    st.markdown('<hr><div style="font-size:.68rem;color:#64748b;line-height:1.7">'
                '🔵 GA Sample · sessions<br>🟣 GitHub Issues · ops<br>🟠 NASA KC1 · defects<hr>'
                'UMBC DS Capstone<br>Dr. Chaojie (Jay) Wang<br>Venkata Sai Chandravadan Sobila</div>', unsafe_allow_html=True)

if not OK:
    st.error(f"Data not found — run Lunara_Analysis.ipynb first.\n\n`{ERR}`"); st.stop()

df = derived[(derived["week"]>=pd.Timestamp(dr[0])) & (derived["week"]<=pd.Timestamp(dr[1])) & derived["feature"].isin(sel)].copy()
sc = scores[scores["feature"].isin(sel)].copy()

# ══════════════════════════════════════════════════════════════
# PAGE 1 — Portfolio Overview
# ══════════════════════════════════════════════════════════════
if "Portfolio" in page:
    st.markdown('<div style="font-size:1.75rem;font-weight:800;color:#0f172a">🌙 Feature Portfolio Dashboard</div>'
                '<div style="font-size:.88rem;color:#64748b;margin-bottom:18px">Lifecycle intelligence · Lunara e-commerce</div>', unsafe_allow_html=True)

    cols = st.columns(6)
    for col, ico, lbl, val, clr, sub in zip(cols, ["👥","💰","🐛","⭐","🚀","↩️"],
        ["Weekly Users","Total Revenue","Total Issues","Avg Score","Invest-Ready","Bounce Rate"],
        [f"{int(df['weekly_active_users'].mean()*max(len(sel),1)):,}",
         f"${df['total_revenue'].sum():,.0f}", f"{int(df['issue_count'].sum()):,}",
         f"{sc['value_score'].mean():.1f}", f"{(sc['lifecycle_label']=='Invest').sum()}/{len(sel)}",
         f"{df['bounce_rate'].mean()*100:.1f}%"],
        ["#6366f1","#10b981","#ef4444","#f59e0b","#8b5cf6","#0ea5e9"],
        ["avg×features","all time","tracked","out of 100","features","avg sessions"]):
        with col: st.markdown(kpi(lbl,val,clr,ico,sub), unsafe_allow_html=True)

    sh("Feature Lifecycle Status")
    fc = st.columns(len(sel))
    for i, feat in enumerate(sel):
        row = scores[scores["feature"]==feat]
        if row.empty: continue
        row = row.iloc[0]; lc = row["lifecycle_label"]; clr = LC_COLORS[lc]
        with fc[i]:
            st.markdown(f'<div class="fc" style="border-color:{clr}50">'
                f'<div style="font-size:1.6rem">{FT_ICONS.get(feat,"🔹")}</div>'
                f'<div style="font-size:.82rem;font-weight:700">{feat}</div>'
                f'<div style="font-size:1.35rem;font-weight:800;color:{clr}">{row["value_score"]:.0f}</div>'
                f'<div style="font-size:.62rem;color:#64748b;text-transform:uppercase;margin:2px 0 5px">score</div>'
                f'<div class="badge {lc.lower()}">{LC_ICONS[lc]} {lc}</div></div>', unsafe_allow_html=True)

    L, R = st.columns([2.1, 1])
    with L:
        sh("Weekly Active Users")
        fig = style_fig(px.line(df, x="week", y="weekly_active_users", color="feature",
                                color_discrete_sequence=PALETTE))
        fig.update_traces(line_width=2.2)
        fig.update_layout(hovermode="x unified", legend_title_text="")
        st.plotly_chart(fig, use_container_width=True, config=CFG)
    with R:
        sh("Portfolio Composition")
        lcc = sc["lifecycle_label"].value_counts().reset_index(); lcc.columns = ["lc","n"]
        fig2 = px.pie(lcc, names="lc", values="n", color="lc", color_discrete_map=LC_COLORS, hole=0.55)
        fig2.update_traces(textfont_size=11, textfont_color="#fff", marker=dict(line=dict(color="#fff",width=2)))
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=420, margin=dict(l=10,r=10,t=15,b=50),
                           legend=dict(font=dict(color="#334155",size=11),bgcolor="rgba(0,0,0,0)",orientation="h",y=-0.08),
                           annotations=[dict(text="Portfolio",x=0.5,y=0.5,font=dict(size=12,color="#334155"),showarrow=False)])
        st.plotly_chart(fig2, use_container_width=True, config=CFG)

    sh("Engagement Heatmap")

    heatmap_cols = {
        "Pageviews": "avg_pageviews",
        "Session Dur.": "avg_session_duration",
        "Bounce": "bounce_rate",
        "Revenue": "total_revenue",
        "Issues": "issue_count",
    }

    available_cols = {label: col for label, col in heatmap_cols.items() if col in df.columns}

    hm = (
        df.groupby("feature")[list(available_cols.values())]
        .mean()
        .round(2)
    )

    hm.columns = list(available_cols.keys())

    hn = (hm - hm.min()) / (hm.max() - hm.min() + 1e-9)

    fig3 = go.Figure(
        go.Heatmap(
            z=hn.values,
            x=hm.columns.tolist(),
            y=hm.index.tolist(),
            colorscale=[
                [0, "#eef2ff"],
                [0.5, "#818cf8"],
                [1, "#312e81"],
            ],
            text=hm.values.round(1),
            texttemplate="%{text}",
            textfont=dict(size=11),
        )
    )

    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=310,
        margin=dict(l=90, r=20, t=20, b=50),
        xaxis=dict(tickfont=dict(size=11, color="#334155"), linecolor="#cbd5e1", showline=True),
        yaxis=dict(tickfont=dict(size=11, color="#334155"), linecolor="#cbd5e1", showline=True),
    )

    st.plotly_chart(fig3, use_container_width=True, config=CFG)

    C1, C2 = st.columns(2)
    with C1:
        sh("Issue Burden Over Time")
        iw = df.groupby(["week","feature"])["issue_count"].sum().reset_index()
        fig4 = style_fig(px.area(iw, x="week", y="issue_count", color="feature", color_discrete_sequence=PALETTE), h=350, mb=80)
        fig4.update_layout(hovermode="x unified", legend_title_text="")
        st.plotly_chart(fig4, use_container_width=True, config=CFG)
    with C2:
        sh("Monthly Revenue")
        tmp = df.copy(); tmp["month"] = tmp["week"].dt.to_period("M").astype(str)
        mr  = tmp.groupby(["month","feature"])["total_revenue"].sum().reset_index()
        fig5 = style_fig(px.bar(mr, x="month", y="total_revenue", color="feature",
                                barmode="stack", color_discrete_sequence=PALETTE), h=350, mb=80)
        fig5.update_layout(hovermode="x unified", legend_title_text="", xaxis_tickangle=-40)
        st.plotly_chart(fig5, use_container_width=True, config=CFG)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — Adoption Forecasting
# ══════════════════════════════════════════════════════════════
elif "Forecasting" in page:
    st.markdown('<div style="font-size:1.75rem;font-weight:800">📈 Adoption Forecasting</div>'
                '<div style="font-size:.88rem;color:#64748b;margin-bottom:16px">Gradient boosting · actual vs predicted WAU</div>', unsafe_allow_html=True)
    feat = st.selectbox("Feature", sel)
    fa   = derived[derived["feature"]==feat].sort_values("week")
    fp   = fpreds[fpreds["feature"]==feat]

    sh(f"{FT_ICONS.get(feat,'')} {feat} — Forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fa["week"], y=fa["weekly_active_users"], mode="lines", name="Actual",
                             line=dict(color="#6366f1",width=2.5), fill="tozeroy", fillcolor="rgba(99,102,241,0.07)"))
    if not fp.empty:
        std = fp["predicted_wau"].std() * 0.25
        fig.add_trace(go.Scatter(x=list(fp["week"])+list(fp["week"])[::-1],
                                 y=list(fp["predicted_wau"]+std)+list(fp["predicted_wau"]-std)[::-1],
                                 fill="toself", fillcolor="rgba(16,185,129,0.1)",
                                 line=dict(color="rgba(0,0,0,0)"), name="±1σ", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fp["week"], y=fp["predicted_wau"], mode="lines+markers", name="Forecast",
                                 line=dict(color="#10b981",width=2.5,dash="dash"),
                                 marker=dict(size=6,color="#10b981",line=dict(color="#fff",width=1.5))))
    fig.update_layout(hovermode="x unified", legend_title_text="")
    style_fig(fig, h=440)
    st.plotly_chart(fig, use_container_width=True, config=CFG)

    if not fp.empty:
        aa = fa[fa["week"].isin(fp["week"])]["weekly_active_users"]
        pa = fp[fp["week"].isin(fa["week"])]["predicted_wau"]
        if len(aa) and len(pa):
            n = min(len(aa),len(pa))
            rmse = np.sqrt(mean_squared_error(aa.values[:n], pa.values[:n]))
            mape = mean_absolute_percentage_error(aa.values[:n], pa.values[:n])
            trend = "↗ Growing" if fa["weekly_active_users"].iloc[-1] > fa["weekly_active_users"].iloc[-8] else "↘ Declining"
            for col, l, v, c, s in zip(st.columns(3),
                ["RMSE","MAPE","Trend"], [f"{rmse:.1f}",f"{mape*100:.1f}%",trend],
                ["#6366f1","#10b981","#10b981" if "Growing" in trend else "#ef4444"],
                ["forecast error","pct error","last 8 weeks"]):
                with col: st.markdown(kpi(l,v,c,sub=s), unsafe_allow_html=True)

    sh("All Features — Forecast Grid")
    fig2 = make_subplots(rows=2, cols=3, subplot_titles=[f"{FT_ICONS.get(f,'')} {f}" for f in FEATURES],
                         vertical_spacing=0.16, horizontal_spacing=0.08)
    for i, feat in enumerate(FEATURES):
        r, c = i//3+1, i%3+1
        fa2  = derived[derived["feature"]==feat].sort_values("week")
        fp2  = fpreds[fpreds["feature"]==feat]
        fig2.add_trace(go.Scatter(x=fa2["week"], y=fa2["weekly_active_users"], mode="lines",
                                  line=dict(color=PALETTE[i],width=1.7), showlegend=False), row=r, col=c)
        if not fp2.empty:
            fig2.add_trace(go.Scatter(x=fp2["week"], y=fp2["predicted_wau"], mode="lines",
                                      line=dict(color=PALETTE[i],dash="dash",width=1.7), showlegend=False), row=r, col=c)
    fig2.update_layout(height=580, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff",
                       margin=dict(l=45,r=15,t=55,b=35), font=dict(family="Inter",color="#0f172a"))
    for k in fig2.layout:
        if k.startswith("xaxis") or k.startswith("yaxis"):
            fig2.layout[k].update(gridcolor="rgba(100,116,139,0.2)",color="#334155",
                                  zeroline=False,tickfont=dict(size=9),linecolor="#cbd5e1",showline=True)
    st.plotly_chart(fig2, use_container_width=True, config=CFG)

# ══════════════════════════════════════════════════════════════
# PAGE 3 — Feature Deep Dive
# ══════════════════════════════════════════════════════════════
elif "Deep Dive" in page:
    st.markdown('<div style="font-size:1.75rem;font-weight:800">🎯 Feature Deep Dive</div>'
                '<div style="font-size:.88rem;color:#64748b;margin-bottom:16px">Granular metrics, health radar, trend analysis</div>', unsafe_allow_html=True)
    feat = st.selectbox("Feature", sel)
    fd   = derived[derived["feature"]==feat].sort_values("week")
    fs   = scores[scores["feature"]==feat].iloc[0]
    lc   = fs["lifecycle_label"]; clr = LC_COLORS[lc]
    r_, g_, b_ = int(clr[1:3],16), int(clr[3:5],16), int(clr[5:7],16)

    st.markdown(f'<div style="background:#fff;border:1px solid {clr}45;border-radius:14px;padding:16px 20px;'
                f'margin-bottom:16px;display:flex;align-items:center;gap:14px">'
                f'<div style="font-size:2.8rem">{FT_ICONS.get(feat,"🔹")}</div>'
                f'<div><div style="font-size:1.5rem;font-weight:800;color:#1e293b">{feat}</div>'
                f'<span class="badge {lc.lower()}">{LC_ICONS[lc]} {lc}</span>'
                f'<span style="color:{clr};font-weight:800;font-size:1.2rem;margin-left:12px">{fs["value_score"]:.0f}'
                f'<span style="font-size:.82rem;font-weight:400;color:#64748b"> / 100</span></span></div></div>', unsafe_allow_html=True)

    for col, l, v, c in zip(st.columns(4),
        ["Adoption Trend","Engagement","Issue Burden","Defect Risk"],
        [fs["adoption_trend"],fs["engagement_intensity"],fs["issue_burden"],fs["defect_risk"]],
        ["#6366f1","#10b981","#ef4444","#f59e0b"]):
        with col: st.markdown(kpi(l,f"{v:.0f}",c,sub="/ 100"), unsafe_allow_html=True)

    L, R = st.columns([2, 1])
    with L:
        metric_map = {"Weekly Active Users":"weekly_active_users","Total Revenue ($)":"total_revenue",
                      "Issue Count":"issue_count","Bounce Rate":"bounce_rate",
                      "Avg Pageviews":"avg_pageviews","Avg Session Duration (s)":"avg_session_duration"}
        ml = st.selectbox("Metric", list(metric_map.keys()))
        sh(f"{feat} — {ml}")
        fig = px.line(fd, x="week", y=metric_map[ml], color_discrete_sequence=[clr])
        fig.update_traces(line_width=2.4, fill="tozeroy", fillcolor=f"rgba({r_},{g_},{b_},0.07)")
        fig.add_trace(go.Scatter(x=fd["week"], y=fd[metric_map[ml]].rolling(4,min_periods=1).mean(),
                                 mode="lines", name="4w avg", line=dict(color="rgba(100,116,139,0.65)",width=1.5,dash="dot")))
        fig.update_layout(hovermode="x unified", legend_title_text="")
        style_fig(fig, h=400)
        st.plotly_chart(fig, use_container_width=True, config=CFG)
    with R:
        sh("Health Radar")
        cats = ["Adoption","Engagement","Low Issues","Low Defects","Adoption"]
        rv   = [fs["adoption_trend"],fs["engagement_intensity"],100-fs["issue_burden"],100-fs["defect_risk"],fs["adoption_trend"]]
        avg  = scores.mean(numeric_only=True)
        bm   = [avg["adoption_trend"],avg["engagement_intensity"],100-avg["issue_burden"],100-avg["defect_risk"],avg["adoption_trend"]]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatterpolar(r=bm, theta=cats, fill="toself", name="Avg",
                                       fillcolor="rgba(100,116,139,0.09)", line=dict(color="#94a3b8",width=1.3,dash="dot")))
        fig2.add_trace(go.Scatterpolar(r=rv, theta=cats, fill="toself", name=feat,
                                       fillcolor=f"rgba({r_},{g_},{b_},0.12)", line=dict(color=clr,width=2.2)))
        fig2.update_layout(polar=dict(bgcolor="#fff",
            radialaxis=dict(visible=True,range=[0,100],gridcolor="rgba(100,116,139,0.2)",tickfont=dict(size=8,color="#334155")),
            angularaxis=dict(color="#334155")), paper_bgcolor="rgba(0,0,0,0)", height=400,
            margin=dict(l=20,r=20,t=15,b=50),
            legend=dict(font=dict(color="#334155",size=10),bgcolor="rgba(0,0,0,0)",orientation="h",y=-0.1))
        st.plotly_chart(fig2, use_container_width=True, config=CFG)

    sh("WAU Distribution — All Features")
    fig3 = px.box(derived[derived["feature"].isin(sel)], x="feature", y="weekly_active_users",
                  color="feature", color_discrete_sequence=PALETTE, points="outliers")
    fig3.update_traces(marker_size=3.5)
    fig3.update_layout(showlegend=False)
    style_fig(fig3, h=340, mb=70)
    st.plotly_chart(fig3, use_container_width=True, config=CFG)

# ══════════════════════════════════════════════════════════════
# PAGE 4 — Decision Reports
# ══════════════════════════════════════════════════════════════
elif "Reports" in page:
    st.markdown('<div style="font-size:1.75rem;font-weight:800">📋 RAG Decision Reports</div>'
                '<div style="font-size:.88rem;color:#64748b;margin-bottom:16px">Evidence-grounded lifecycle recommendations</div>', unsafe_allow_html=True)
    feat   = st.selectbox("Feature", sel)
    report = next((r for r in reports if r["feature"]==feat), None)
    if report:
        lc = report["lifecycle"]; clr = LC_COLORS[lc]; m = report["metrics"]
        dim = "".join(f'<div style="text-align:center;background:#f8fafc;border-radius:7px;padding:10px;border:1px solid #e2e8f0">'
                      f'<div style="font-size:.61rem;color:#64748b;text-transform:uppercase">{k}</div>'
                      f'<div style="font-size:1.1rem;font-weight:800;color:{c};margin-top:2px">{m[mk]}</div></div>'
                      for k,mk,c in [("Adoption","adoption_trend","#6366f1"),("Engagement","engagement_intensity","#10b981"),
                                     ("Issues","issue_burden","#ef4444"),("Defects","defect_risk","#f59e0b")])
        components.html(
            f'<div style="font-family:Inter,sans-serif;background:#fff;border:1px solid {clr}40;border-radius:14px;padding:18px 22px;'
            f'box-shadow:0 3px 10px rgba(15,23,42,0.06)">'
            f'<style>.badge{{display:inline-block;padding:3px 9px;border-radius:999px;font-size:.63rem;font-weight:800;text-transform:uppercase;}}'
            f'.invest{{background:#dcfce7;color:#047857;border:1px solid #86efac;}}'
            f'.maintain{{background:#dbeafe;color:#1d4ed8;border:1px solid #93c5fd;}}'
            f'.refactor{{background:#fef3c7;color:#92400e;border:1px solid #fbbf24;}}'
            f'.sunset{{background:#fee2e2;color:#b91c1c;border:1px solid #fca5a5;}}</style>'
            f'<div style="display:flex;align-items:center;gap:14px;margin-bottom:12px">'
            f'<span style="font-size:2.4rem">{LC_ICONS[lc]}</span>'
            f'<div><div style="font-size:1.35rem;font-weight:800;color:#1e293b">{feat}</div>'
            f'<span class="badge {lc.lower()}">{lc}</span>'
            f'<span style="color:{clr};font-weight:800;font-size:1.1rem;margin-left:11px">{m["value_score"]}/100</span></div></div>'
            f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:9px">{dim}</div></div>', height=195)

        C1, C2 = st.columns(2)
        with C1:
            sh("📚 Evidence Cards")
            for card in report["evidence_cards"][:4]:
                st.markdown(f'<div class="ev"><div style="font-weight:700;color:#4338ca;font-size:.83rem">{card["title"]}</div>'
                            f'<div style="font-size:.69rem;color:#64748b;margin-bottom:4px">{card["source"]} · {card["relevance_score"]:.3f}</div>'
                            f'<div style="font-size:.78rem;color:#475569;line-height:1.5">{card["excerpt"][:200]}…</div></div>', unsafe_allow_html=True)
        with C2:
            sh("⚠️ Risks")
            ri = {"HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢"}
            for r in report["risks"]:
                st.markdown(f'<div class="risk-{r["level"]}"><div style="font-size:.72rem;font-weight:800;color:#334155">{ri[r["level"]]} {r["level"]}</div>'
                            f'<div style="font-size:.79rem;color:#475569">{r["description"]}</div></div>', unsafe_allow_html=True)
            sh("✅ Next Steps")
            for i, s in enumerate(report["next_steps"],1):
                st.markdown(f'<div class="step"><span style="color:#6366f1;font-weight:700">{i}.</span> {s}</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    sh("All Features Summary")
    rows = [{"Feature":f"{LC_ICONS[r['lifecycle']]} {r['feature']}","Lifecycle":r["lifecycle"],
             "Score":r["metrics"]["value_score"],"Adoption":r["metrics"]["adoption_trend"],
             "Engagement":r["metrics"]["engagement_intensity"],"Issues":r["metrics"]["issue_burden"],
             "Defects":r["metrics"]["defect_risk"],"Avg WAU":f"{r['metrics']['avg_weekly_users']:,}",
             "Revenue":f"${r['metrics']['recent_revenue']:,.0f}"}
            for r in reports if r["feature"] in sel]
    if rows: st.dataframe(pd.DataFrame(rows).set_index("Feature"), use_container_width=True, height=270)

