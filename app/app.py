"""
╔══════════════════════════════════════════════════════════════╗
║   🌙 LUNARA — Feature Lifecycle Intelligence Dashboard       ║
║   UMBC Data Science Capstone — Dr. Chaojie (Jay) Wang        ║
║   Author: Venkata Sai Chandravadan Sobila                    ║
╠══════════════════════════════════════════════════════════════╣
║   Run:  streamlit run app.py                                 ║
║   URL:  http://localhost:8501                                ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import json
import os

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lunara · Feature Lifecycle Intelligence",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: "Inter", sans-serif; }
  .stApp {
    background: linear-gradient(135deg, #0d0b1e 0%, #1a1542 40%, #0f1629 100%);
    min-height: 100vh;
  }

  section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.97) !important;
    border-right: 1px solid rgba(129,140,248,0.15) !important;
  }
  section[data-testid="stSidebar"] * { color: #c4c9e8 !important; }

  .block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }

  /* KPI Card */
  .kpi {
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 18px; padding: 22px 20px 18px;
    text-align: center;
    transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    height: 140px; display: flex; flex-direction: column; justify-content: center;
  }
  .kpi:hover { transform: translateY(-4px); box-shadow: 0 16px 40px rgba(0,0,0,0.5); border-color: rgba(129,140,248,0.35); }
  .kpi-icon  { font-size: 1.4rem; margin-bottom: 3px; }
  .kpi-val   { font-size: 1.85rem; font-weight: 800; line-height: 1.1; margin: 2px 0; }
  .kpi-label { font-size: .7rem; letter-spacing: .08em; text-transform: uppercase; color: #8892b0; margin-top: 4px; }
  .kpi-sub   { font-size: .73rem; margin-top: 3px; opacity: .65; }

  /* Feature Card */
  .feat-card {
    background: rgba(255,255,255,0.045); border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px; padding: 18px 14px; text-align: center;
    transition: all .2s; backdrop-filter: blur(10px);
  }
  .feat-card:hover { transform: translateY(-3px); box-shadow: 0 10px 30px rgba(0,0,0,0.4); }
  .feat-icon  { font-size: 2rem; margin-bottom: 6px; }
  .feat-name  { font-size: .88rem; font-weight: 600; color: #e2e8f0; margin-bottom: 4px; }
  .feat-score { font-size: 1.6rem; font-weight: 800; }
  .feat-sub   { font-size: .68rem; color: #64748b; text-transform: uppercase; letter-spacing: .06em; margin: 2px 0 6px; }

  /* Lifecycle Badges */
  .badge { display: inline-block; padding: 4px 14px; border-radius: 999px; font-size: .7rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }
  .invest   { background: #052e16; color: #6ee7b7; border: 1px solid #065f46; }
  .maintain { background: #0c1a3a; color: #93c5fd; border: 1px solid #1e3a5f; }
  .refactor { background: #2d1b00; color: #fcd34d; border: 1px solid #78350f; }
  .sunset   { background: #2d0a0a; color: #fca5a5; border: 1px solid #7f1d1d; }

  /* Section Heading */
  .sh { font-size: 1.05rem; font-weight: 600; color: #c4c9e8; border-left: 3px solid #818cf8; padding-left: 10px; margin: 28px 0 14px; }

  /* Evidence Card */
  .ev-card { background: rgba(99,102,241,0.07); border: 1px solid rgba(99,102,241,0.2); border-radius: 12px; padding: 14px 16px; margin-bottom: 10px; }
  .ev-title  { font-weight: 600; color: #a5b4fc; font-size: .86rem; margin-bottom: 2px; }
  .ev-source { font-size: .72rem; color: #4b5563; margin-bottom: 6px; }
  .ev-text   { font-size: .80rem; color: #94a3b8; line-height: 1.55; }

  /* Risk rows */
  .risk-HIGH   { border-left: 3px solid #ef4444; padding-left: 10px; margin-bottom: 8px; }
  .risk-MEDIUM { border-left: 3px solid #f59e0b; padding-left: 10px; margin-bottom: 8px; }
  .risk-LOW    { border-left: 3px solid #10b981; padding-left: 10px; margin-bottom: 8px; }
  .risk-label  { font-size: .75rem; font-weight: 700; color: #94a3b8; }
  .risk-desc   { font-size: .82rem; color: #cbd5e1; margin-top: 2px; }

  /* Next-step pill */
  .step { background: rgba(255,255,255,0.04); border-radius: 8px; padding: 9px 14px; margin-bottom: 7px; border-left: 3px solid #818cf8; font-size: .83rem; color: #cbd5e1; }

  /* Page title */
  .page-title { font-size: 1.9rem; font-weight: 800; color: #e2e8f0; margin: 0 0 4px; }
  .page-sub   { font-size: .88rem; color: #4b5563; margin-bottom: 24px; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
  ::-webkit-scrollbar-thumb { background: rgba(129,140,248,0.3); border-radius: 3px; }

  /* Widget labels */
  .stSelectbox label, .stMultiSelect label, .stSlider label, .stDateInput label { color: #8892b0 !important; font-size: .83rem !important; }

  hr { border-color: rgba(255,255,255,0.07); margin: 20px 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
FEATURES = ["Search", "Recommendations", "Wishlist", "Reviews", "Notifications", "Checkout"]

LIFECYCLE_COLORS = {"Invest": "#10b981", "Maintain": "#3b82f6", "Refactor": "#f59e0b", "Sunset": "#ef4444"}
LIFECYCLE_ICONS  = {"Invest": "🚀", "Maintain": "🔧", "Refactor": "🔄", "Sunset": "🌅"}
FEATURE_ICONS    = {"Search": "🔍", "Recommendations": "✨", "Wishlist": "💝",
                    "Reviews": "⭐", "Notifications": "🔔", "Checkout": "🛒"}
PALETTE = ["#818cf8", "#34d399", "#f472b6", "#fbbf24", "#f87171", "#60a5fa"]

SCORE_WEIGHTS = {
    "adoption_trend": 0.30, "engagement_intensity": 0.25,
    "issue_burden": -0.25,  "defect_risk": -0.20,
}
LIFECYCLE_THRESHOLDS = {
    "Invest": (75, 101), "Maintain": (50, 75), "Refactor": (25, 50), "Sunset": (0, 25),
}


def assign_lifecycle(score):
    for label, (lo, hi) in LIFECYCLE_THRESHOLDS.items():
        if lo <= score < hi:
            return label
    return "Maintain"


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading Lunara data…")
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))

    def p(name):
        return os.path.join(base, "data", "processed", name)

    derived = pd.read_csv(p("weekly_feature_data.csv"),  parse_dates=["week"])
    scores  = pd.read_csv(p("feature_value_scores.csv"))
    preds   = pd.read_csv(p("forecast_predictions.csv"), parse_dates=["week"])
    with open(p("decision_reports.json")) as f:
        reports = json.load(f)
    clf_path  = p("lifecycle_predictions.csv")
    clf_preds = pd.read_csv(clf_path) if os.path.exists(clf_path) else None
    return derived, scores, preds, reports, clf_preds


try:
    derived, scores, forecast_preds, reports, clf_preds = load_data()
    DATA_OK = True
except Exception as e:
    DATA_OK  = False
    DATA_ERR = str(e)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def dark_layout(fig, height=340, margin=None, legend=True):
    m = margin or dict(l=4, r=4, t=10, b=4)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=height, margin=m,
        font=dict(family="Inter", color="#94a3b8"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#64748b", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#64748b", zeroline=False),
        legend=dict(font=dict(color="#94a3b8", size=11), bgcolor="rgba(0,0,0,0)",
                    orientation="h", y=1.06) if legend else dict(visible=False),
    )
    return fig


def kpi(label, value, color, icon="", sub=""):
    return (f'<div class="kpi"><div class="kpi-icon">{icon}</div>'
            f'<div class="kpi-val" style="color:{color}">{value}</div>'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-sub" style="color:{color}99">{sub}</div></div>')


def sh(text):
    st.markdown(f'<div class="sh">{text}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 4px 8px">
      <div style="font-size:1.6rem;font-weight:800;color:#818cf8">🌙 Lunara</div>
      <div style="font-size:.78rem;color:#4b5563;margin-top:2px">Feature Lifecycle Intelligence</div>
    </div><hr>""", unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "🏠  Portfolio Overview",
        "📈  Adoption Forecasting",
        "🎯  Feature Deep Dive",
        "📋  Decision Reports",
        "🧪  Portfolio Simulator",
    ], label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:.78rem;color:#4b5563;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px">Filters</div>', unsafe_allow_html=True)

    if DATA_OK:
        date_min   = derived["week"].min().date()
        date_max   = derived["week"].max().date()
        date_range = st.date_input("Date window", value=[date_min, date_max],
                                   min_value=date_min, max_value=date_max,
                                   label_visibility="collapsed")
        selected_features = st.multiselect("Features", options=FEATURES, default=FEATURES,
                                           label_visibility="collapsed")
        if len(date_range) == 1:
            date_range = [date_range[0], date_max]

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:.78rem;color:#4b5563;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px">Data Sources</div>
    <div style="font-size:.8rem;margin-bottom:4px">🔵 GA Sample — User sessions</div>
    <div style="font-size:.8rem;margin-bottom:4px">🟣 GitHub Issues — Ops burden</div>
    <div style="font-size:.8rem;margin-bottom:12px">🟠 NASA KC1 — Defect risk</div>
    <hr>
    <div style="font-size:.72rem;color:#374151;line-height:1.6">
      UMBC Data Science Capstone<br>Dr. Chaojie (Jay) Wang<br>Venkata Sai Chandravadan Sobila
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# GUARD
# ─────────────────────────────────────────────────────────────
if not DATA_OK:
    st.markdown("""
    <div style="text-align:center;padding:80px 20px">
      <div style="font-size:4rem">🌙</div>
      <div style="font-size:1.6rem;font-weight:700;color:#e2e8f0;margin:16px 0 8px">Data files not found</div>
      <div style="color:#64748b;font-size:.95rem">
        Run the Jupyter notebook first to generate <code>data/processed/</code> files,<br>
        then launch this dashboard from the same project folder.
      </div>
      <div style="margin-top:20px;background:rgba(255,255,255,0.05);border-radius:12px;padding:16px 24px;display:inline-block;text-align:left">
        <code style="color:#a5b4fc;font-size:.85rem">streamlit run app.py</code>
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────
# FILTERED VIEWS
# ─────────────────────────────────────────────────────────────
derived_f = derived[
    (derived["week"] >= pd.Timestamp(date_range[0])) &
    (derived["week"] <= pd.Timestamp(date_range[1])) &
    (derived["feature"].isin(selected_features))
].copy()
scores_f = scores[scores["feature"].isin(selected_features)].copy()


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — PORTFOLIO OVERVIEW
# ═══════════════════════════════════════════════════════════════
if "Portfolio Overview" in page:

    st.markdown('<div class="page-title">🌙 Feature Portfolio Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Real-time lifecycle intelligence · Lunara e-commerce platform</div>', unsafe_allow_html=True)

    # KPI Row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    total_users  = int(derived_f["weekly_active_users"].mean() * max(len(selected_features), 1))
    total_rev    = derived_f["total_revenue"].sum()
    total_issues = int(derived_f["issue_count"].sum())
    avg_score    = scores_f["value_score"].mean() if not scores_f.empty else 0
    invest_n     = (scores_f["lifecycle_label"] == "Invest").sum()
    avg_bounce   = derived_f["bounce_rate"].mean() * 100

    for col, ico, lbl, val, clr, sub in [
        (c1, "👥", "Weekly Users",  f"{total_users:,}",       "#818cf8", "avg × features"),
        (c2, "💰", "Total Revenue", f"${total_rev:,.0f}",     "#34d399", "all time"),
        (c3, "🐛", "Total Issues",  f"{total_issues:,}",       "#f87171", "tracked period"),
        (c4, "⭐", "Avg Score",     f"{avg_score:.1f}",        "#fbbf24", "out of 100"),
        (c5, "🚀", "Invest-Ready",  f"{invest_n}/{len(selected_features)}", "#a78bfa", "features"),
        (c6, "↩️", "Bounce Rate",  f"{avg_bounce:.1f}%",      "#60a5fa", "avg sessions"),
    ]:
        with col:
            st.markdown(kpi(lbl, val, clr, ico, sub), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature Status Grid
    sh("Feature Lifecycle Status")
    cols = st.columns(len(selected_features))
    for i, feat in enumerate(selected_features):
        row = scores[scores["feature"] == feat]
        if row.empty:
            continue
        row = row.iloc[0]
        lc, sc, clr = row["lifecycle_label"], row["value_score"], LIFECYCLE_COLORS[row["lifecycle_label"]]
        with cols[i]:
            st.markdown(f"""
            <div class="feat-card" style="border-color:{clr}30">
              <div class="feat-icon">{FEATURE_ICONS.get(feat, "🔹")}</div>
              <div class="feat-name">{feat}</div>
              <div class="feat-score" style="color:{clr}">{sc:.0f}</div>
              <div class="feat-sub">value score</div>
              <div class="badge {lc.lower()}">{LIFECYCLE_ICONS[lc]} {lc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # WAU Trend + Donut
    left, right = st.columns([3, 2])
    with left:
        sh("Weekly Active Users — All Features")
        fig = px.line(derived_f, x="week", y="weekly_active_users", color="feature",
                      color_discrete_sequence=PALETTE)
        fig.update_traces(line_width=2)
        dark_layout(fig, height=330)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        sh("Portfolio Composition")
        lcc = scores_f["lifecycle_label"].value_counts().reset_index()
        lcc.columns = ["lifecycle", "count"]
        fig2 = px.pie(lcc, names="lifecycle", values="count", color="lifecycle",
                      color_discrete_map=LIFECYCLE_COLORS, hole=0.52)
        fig2.update_traces(textfont_size=12, textfont_color="#e2e8f0")
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", height=330,
            margin=dict(l=4, r=4, t=10, b=4),
            legend=dict(font=dict(color="#94a3b8", size=11), bgcolor="rgba(0,0,0,0)"),
            annotations=[dict(text="Portfolio", x=0.5, y=0.5,
                               font=dict(size=13, color="#94a3b8", family="Inter"), showarrow=False)],
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Engagement Heatmap
    sh("Feature Engagement Matrix (Normalized 0–1)")
    hm = derived_f.groupby("feature").agg({
        "avg_pageviews": "mean", "avg_session_duration": "mean",
        "bounce_rate": "mean", "total_transactions": "sum", "total_revenue": "sum",
    }).round(2)
    hm_norm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-9)
    fig3 = go.Figure(go.Heatmap(
        z=hm_norm.values,
        x=["Pageviews", "Session Dur.", "Bounce Rate", "Transactions", "Revenue"],
        y=hm_norm.index.tolist(),
        colorscale=[[0, "#1e1b4b"], [0.4, "#4f46e5"], [0.75, "#818cf8"], [1, "#c7d2fe"]],
        text=hm.values.round(2), texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
        colorbar=dict(tickfont=dict(color="#64748b"), len=0.8),
    ))
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=270, margin=dict(l=4, r=4, t=10, b=4),
        xaxis=dict(color="#64748b"), yaxis=dict(color="#64748b"),
        font=dict(family="Inter"),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Issue Burden + Revenue
    cl, cr = st.columns(2)
    with cl:
        sh("Operational Issue Burden Over Time")
        iw = derived_f.groupby(["week", "feature"])["issue_count"].sum().reset_index()
        fig4 = px.area(iw, x="week", y="issue_count", color="feature",
                       color_discrete_sequence=PALETTE)
        fig4.update_traces(line_width=1.5)
        dark_layout(fig4, height=290)
        st.plotly_chart(fig4, use_container_width=True)

    with cr:
        sh("Monthly Revenue by Feature (Stacked)")
        df_rev = derived_f.copy()
        df_rev["month"] = df_rev["week"].dt.to_period("M").astype(str)
        mr = df_rev.groupby(["month", "feature"])["total_revenue"].sum().reset_index()
        fig5 = px.bar(mr, x="month", y="total_revenue", color="feature",
                      barmode="stack", color_discrete_sequence=PALETTE)
        fig5.update_layout(xaxis_tickangle=-45)
        dark_layout(fig5, height=290)
        st.plotly_chart(fig5, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — ADOPTION FORECASTING
# ═══════════════════════════════════════════════════════════════
elif "Adoption Forecasting" in page:

    st.markdown('<div class="page-title">📈 Adoption Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">LightGBM model — actual weekly active users vs predicted</div>', unsafe_allow_html=True)

    sel = st.selectbox("Feature", selected_features, key="fc_feat")
    fa  = derived[derived["feature"] == sel].sort_values("week")
    fp  = forecast_preds[forecast_preds["feature"] == sel]

    sh(f"{FEATURE_ICONS.get(sel, '')} {sel} — Forecast Detail")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fa["week"], y=fa["weekly_active_users"],
        mode="lines", name="Actual",
        line=dict(color="#818cf8", width=2.5),
        fill="tozeroy", fillcolor="rgba(129,140,248,0.07)",
    ))
    if not fp.empty:
        std = fp["predicted_wau"].std() * 0.25
        fig.add_trace(go.Scatter(
            x=list(fp["week"]) + list(fp["week"])[::-1],
            y=list(fp["predicted_wau"] + std) + list(fp["predicted_wau"] - std)[::-1],
            fill="toself", fillcolor="rgba(52,211,153,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="±1σ band", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=fp["week"], y=fp["predicted_wau"],
            mode="lines+markers", name="Forecast",
            line=dict(color="#34d399", width=2.5, dash="dash"),
            marker=dict(size=7, color="#34d399", line=dict(color="#0d1117", width=1.5)),
        ))
        cutoff = fp["week"].min()
        fig.add_vline(x=cutoff, line_dash="dot", line_color="rgba(255,255,255,0.2)",
                      annotation_text="forecast start", annotation_font_color="#64748b",
                      annotation_font_size=11)

    dark_layout(fig, height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Metric row
    if not fp.empty:
        actual_aligned = fa[fa["week"].isin(fp["week"])]["weekly_active_users"]
        pred_aligned   = fp[fp["week"].isin(fa["week"])]["predicted_wau"]
        if len(actual_aligned) > 0 and len(pred_aligned) > 0:
            min_len = min(len(actual_aligned), len(pred_aligned))
            rmse = np.sqrt(mean_squared_error(actual_aligned.values[:min_len], pred_aligned.values[:min_len]))
            mape = mean_absolute_percentage_error(actual_aligned.values[:min_len], pred_aligned.values[:min_len])
            c1, c2, c3 = st.columns(3)
            trend = "↗ Growing" if fa["weekly_active_users"].iloc[-1] > fa["weekly_active_users"].iloc[-8] else "↘ Declining"
            tc = "#34d399" if "Growing" in trend else "#f87171"
            with c1: st.markdown(kpi("RMSE", f"{rmse:.1f}", "#818cf8", "📉", "forecast error"), unsafe_allow_html=True)
            with c2: st.markdown(kpi("MAPE", f"{mape*100:.1f}%", "#34d399", "📊", "pct error"), unsafe_allow_html=True)
            with c3: st.markdown(kpi("Trend", trend, tc, "📈", "last 8 weeks"), unsafe_allow_html=True)

    # All-features grid
    st.markdown("<br>", unsafe_allow_html=True)
    sh("All Features — Forecast Grid")
    fig2 = make_subplots(rows=2, cols=3,
                          subplot_titles=[f"{FEATURE_ICONS.get(f,'')} {f}" for f in FEATURES],
                          vertical_spacing=0.14, horizontal_spacing=0.06)
    for idx, feat in enumerate(FEATURES):
        r, c = idx // 3 + 1, idx % 3 + 1
        clr = PALETTE[idx]
        fa2 = derived[derived["feature"] == feat].sort_values("week")
        fp2 = forecast_preds[forecast_preds["feature"] == feat]
        fig2.add_trace(go.Scatter(x=fa2["week"], y=fa2["weekly_active_users"],
                                   mode="lines", line=dict(color=clr, width=1.8), showlegend=False), row=r, col=c)
        if not fp2.empty:
            fig2.add_trace(go.Scatter(x=fp2["week"], y=fp2["predicted_wau"],
                                       mode="lines", line=dict(color=clr, dash="dash", width=1.8), showlegend=False), row=r, col=c)
    fig2.update_layout(height=520, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=4, r=4, t=32, b=4),
                        font=dict(family="Inter", color="#94a3b8"))
    for key in fig2.layout:
        if key.startswith("xaxis") or key.startswith("yaxis"):
            fig2.layout[key].update(gridcolor="rgba(255,255,255,0.04)", color="#64748b", zeroline=False)
    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — FEATURE DEEP DIVE
# ═══════════════════════════════════════════════════════════════
elif "Feature Deep Dive" in page:

    st.markdown('<div class="page-title">🎯 Feature Deep Dive</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Granular metrics, health radar, and trend analysis per feature</div>', unsafe_allow_html=True)

    feat = st.selectbox("Select Feature", selected_features, key="dd_feat")
    fd   = derived[derived["feature"] == feat].sort_values("week")
    fs   = scores[scores["feature"] == feat].iloc[0]
    lc   = fs["lifecycle_label"]
    clr  = LIFECYCLE_COLORS[lc]

    # Feature banner
    st.markdown(f"""
    <div style="background:linear-gradient(120deg,{clr}18 0%,{clr}06 100%);
                border:1px solid {clr}30; border-radius:20px; padding:24px 30px; margin-bottom:20px;">
      <div style="display:flex; align-items:center; gap:20px;">
        <div style="font-size:3.5rem">{FEATURE_ICONS.get(feat, "🔹")}</div>
        <div>
          <div style="font-size:1.8rem;font-weight:800;color:#e2e8f0;margin-bottom:6px">{feat}</div>
          <span class="badge {lc.lower()}">{LIFECYCLE_ICONS[lc]} {lc}</span>
          <span style="color:{clr};font-size:1.4rem;font-weight:700;margin-left:16px">
            {fs['value_score']:.0f}<span style="font-size:.9rem;font-weight:400;color:#64748b"> / 100</span>
          </span>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # 4 score pills
    d1, d2, d3, d4 = st.columns(4)
    for col, lbl, val, ic, c in [
        (d1, "Adoption Trend",  fs["adoption_trend"],      "📈", "#818cf8"),
        (d2, "Engagement",      fs["engagement_intensity"], "💡", "#34d399"),
        (d3, "Issue Burden",    fs["issue_burden"],         "🐛", "#f87171"),
        (d4, "Defect Risk",     fs["defect_risk"],          "⚠️", "#fbbf24"),
    ]:
        with col:
            st.markdown(kpi(lbl, f"{val:.0f}", c, ic, "/ 100"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    cl, cr = st.columns([2, 1])
    with cl:
        metric_map = {
            "Weekly Active Users": "weekly_active_users",
            "Total Revenue ($)":   "total_revenue",
            "Issue Count":         "issue_count",
            "Bounce Rate":         "bounce_rate",
            "Avg Pageviews":       "avg_pageviews",
            "Avg Session Duration (s)": "avg_session_duration",
        }
        metric_label = st.selectbox("Metric to plot", list(metric_map.keys()), key="dd_metric")
        metric_col   = metric_map[metric_label]
        sh(f"{feat} — {metric_label} Over Time")

        fig = px.line(fd, x="week", y=metric_col, color_discrete_sequence=[clr])
        fig.update_traces(line_width=2.5, fill="tozeroy",
                          fillcolor=f"rgba({int(clr[1:3],16)},{int(clr[3:5],16)},{int(clr[5:7],16)},0.08)")
        fig.add_trace(go.Scatter(
            x=fd["week"], y=fd[metric_col].rolling(4, min_periods=1).mean(),
            mode="lines", name="4w avg",
            line=dict(color="rgba(255,255,255,0.22)", width=1.5, dash="dot"),
        ))
        dark_layout(fig, height=330)
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        sh("Health Radar vs Portfolio Avg")
        cats = ["Adoption", "Engagement", "Low Issues", "Low Defects", "Adoption"]
        rv   = [fs["adoption_trend"], fs["engagement_intensity"],
                100 - fs["issue_burden"], 100 - fs["defect_risk"], fs["adoption_trend"]]
        avg_row = scores.mean(numeric_only=True)
        bm = [avg_row["adoption_trend"], avg_row["engagement_intensity"],
              100 - avg_row["issue_burden"], 100 - avg_row["defect_risk"], avg_row["adoption_trend"]]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatterpolar(r=bm, theta=cats, fill="toself", name="Portfolio avg",
                                        fillcolor="rgba(100,116,139,0.12)",
                                        line=dict(color="#475569", width=1.5, dash="dot")))
        fig2.add_trace(go.Scatterpolar(r=rv, theta=cats, fill="toself", name=feat,
                                        fillcolor=f"rgba({int(clr[1:3],16)},{int(clr[3:5],16)},{int(clr[5:7],16)},0.15)",
                                        line=dict(color=clr, width=2.5)))
        fig2.update_layout(
            polar=dict(bgcolor="rgba(0,0,0,0)",
                       radialaxis=dict(visible=True, range=[0, 100],
                                       gridcolor="rgba(255,255,255,0.07)",
                                       color="#64748b", tickfont=dict(size=9)),
                       angularaxis=dict(color="#64748b", gridcolor="rgba(255,255,255,0.07)")),
            paper_bgcolor="rgba(0,0,0,0)", height=330,
            margin=dict(l=24, r=24, t=20, b=10),
            legend=dict(font=dict(color="#94a3b8", size=10), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Distribution box
    sh("Weekly Active Users Distribution — All Features")
    fig3 = px.box(derived[derived["feature"].isin(selected_features)],
                  x="feature", y="weekly_active_users", color="feature",
                  color_discrete_sequence=PALETTE, points="outliers")
    fig3.update_traces(marker_size=4)
    dark_layout(fig3, height=280, legend=False)
    st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 4 — DECISION REPORTS (RAG)
# ═══════════════════════════════════════════════════════════════
elif "Decision Reports" in page:

    st.markdown('<div class="page-title">📋 RAG Decision Reports</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Evidence-grounded lifecycle recommendations powered by semantic retrieval</div>', unsafe_allow_html=True)

    feat   = st.selectbox("Select Feature", selected_features, key="rep_feat")
    report = next((r for r in reports if r["feature"] == feat), None)

    if report:
        lc, clr, m = report["lifecycle"], LIFECYCLE_COLORS[report["lifecycle"]], report["metrics"]

        # Header
        dim_html = "".join(
            f'<div style="text-align:center;background:rgba(0,0,0,0.2);border-radius:10px;padding:10px">'
            f'<div style="font-size:.65rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em">{k}</div>'
            f'<div style="font-size:1.25rem;font-weight:700;color:{c};margin-top:3px">{m[mk]}</div></div>'
            for k, mk, c in [
                ("Adoption",     "adoption_trend",       "#818cf8"),
                ("Engagement",   "engagement_intensity",  "#34d399"),
                ("Issue Burden", "issue_burden",           "#f87171"),
                ("Defect Risk",  "defect_risk",            "#fbbf24"),
            ]
        )
        st.markdown(f"""
        <div style="background:linear-gradient(120deg,{clr}18 0%,{clr}05 100%);
                    border:1px solid {clr}25; border-radius:20px; padding:24px 28px; margin-bottom:20px">
          <div style="display:flex;align-items:center;gap:18px;margin-bottom:16px">
            <span style="font-size:3rem">{LIFECYCLE_ICONS[lc]}</span>
            <div>
              <div style="font-size:1.5rem;font-weight:700;color:#e2e8f0">{feat}</div>
              <span class="badge {lc.lower()}">{lc}</span>
              <span style="color:{clr};font-weight:700;font-size:1.2rem;margin-left:14px">{m['value_score']}/100</span>
            </div>
          </div>
          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px">{dim_html}</div>
        </div>""", unsafe_allow_html=True)

        col_l, col_r = st.columns(2)
        with col_l:
            sh("📚 Evidence Cards")
            for card in report["evidence_cards"][:4]:
                st.markdown(f"""
                <div class="ev-card">
                  <div class="ev-title">{card['title']}</div>
                  <div class="ev-source">{card['source']} &nbsp;·&nbsp; relevance {card['relevance_score']:.3f}</div>
                  <div class="ev-text">{card['excerpt'][:200]}…</div>
                </div>""", unsafe_allow_html=True)

        with col_r:
            sh("⚠️ Risk Assessment")
            risk_icons = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
            for r in report["risks"]:
                st.markdown(f"""
                <div class="risk-{r['level']}">
                  <div class="risk-label">{risk_icons[r['level']]} {r['level']}</div>
                  <div class="risk-desc">{r['description']}</div>
                </div>""", unsafe_allow_html=True)

            sh("✅ Next Steps")
            for i, step in enumerate(report["next_steps"], 1):
                st.markdown(f'<div class="step"><span style="color:#818cf8;font-weight:700">{i}. </span>{step}</div>',
                             unsafe_allow_html=True)

    st.markdown("<br><hr>", unsafe_allow_html=True)
    sh("📊 All Features Summary Table")
    rows = []
    for r in reports:
        if r["feature"] not in selected_features:
            continue
        rows.append({
            "Feature":    f"{LIFECYCLE_ICONS[r['lifecycle']]} {r['feature']}",
            "Lifecycle":  r["lifecycle"],
            "Score":      r["metrics"]["value_score"],
            "Adoption":   r["metrics"]["adoption_trend"],
            "Engagement": r["metrics"]["engagement_intensity"],
            "Issues":     r["metrics"]["issue_burden"],
            "Defects":    r["metrics"]["defect_risk"],
            "Avg WAU":    f"{r['metrics']['avg_weekly_users']:,}",
            "Revenue":    f"${r['metrics']['recent_revenue']:,.0f}",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Feature"), use_container_width=True, height=260)


# ═══════════════════════════════════════════════════════════════
# PAGE 5 — PORTFOLIO SIMULATOR
# ═══════════════════════════════════════════════════════════════
elif "Portfolio Simulator" in page:

    st.markdown('<div class="page-title">🧪 Portfolio Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">What-if analysis — adjust engineering levers and preview lifecycle changes</div>', unsafe_allow_html=True)

    feat = st.selectbox("Select Feature to Simulate", selected_features, key="sim_feat")
    fs   = scores[scores["feature"] == feat].iloc[0]

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        sh("📈 Boost Levers")
        ab = st.slider("Adoption Boost (%)",   0, 100, 0, key="ab")
        eb = st.slider("Engagement Boost (%)", 0, 100, 0, key="eb")
    with col_r:
        sh("📉 Reduction Levers")
        ir = st.slider("Issue Reduction (%)",  0, 100, 0, key="ir")
        dr = st.slider("Defect Reduction (%)", 0, 100, 0, key="dr")

    # Compute
    sa  = min(100, fs["adoption_trend"]       * (1 + ab / 100))
    se  = min(100, fs["engagement_intensity"] * (1 + eb / 100))
    si  = max(0,   fs["issue_burden"]         * (1 - ir / 100))
    sd  = max(0,   fs["defect_risk"]          * (1 - dr / 100))
    sv  = min(100, max(0,
        SCORE_WEIGHTS["adoption_trend"] * sa
        + SCORE_WEIGHTS["engagement_intensity"] * se
        + abs(SCORE_WEIGHTS["issue_burden"]) * (100 - si)
        + abs(SCORE_WEIGHTS["defect_risk"])  * (100 - sd)
    ))
    slc   = assign_lifecycle(sv)
    olc   = fs["lifecycle_label"]
    oc, nc = LIFECYCLE_COLORS[olc], LIFECYCLE_COLORS[slc]
    delta  = sv - fs["value_score"]

    st.markdown("<br>", unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""
        <div class="kpi" style="border-color:{oc}40;height:160px">
          <div class="kpi-label">Current State</div>
          <div style="font-size:2.2rem;margin:4px 0">{LIFECYCLE_ICONS[olc]}</div>
          <div class="kpi-val" style="color:{oc}">{fs['value_score']:.0f}</div>
          <div class="badge {olc.lower()}" style="margin-top:6px">{olc}</div>
        </div>""", unsafe_allow_html=True)
    with r2:
        arr = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        dc  = "#34d399" if delta > 0 else ("#f87171" if delta < 0 else "#94a3b8")
        msg = "Lifecycle changed!" if olc != slc else "Same lifecycle"
        st.markdown(f"""
        <div class="kpi" style="height:160px">
          <div class="kpi-label">Score Delta</div>
          <div style="font-size:2.8rem;color:{dc};margin:2px 0">{arr}</div>
          <div class="kpi-val" style="color:{dc}">{delta:+.1f}</div>
          <div style="font-size:.72rem;color:{dc}99;margin-top:4px">{msg}</div>
        </div>""", unsafe_allow_html=True)
    with r3:
        st.markdown(f"""
        <div class="kpi" style="border-color:{nc}40;height:160px">
          <div class="kpi-label">Simulated State</div>
          <div style="font-size:2.2rem;margin:4px 0">{LIFECYCLE_ICONS[slc]}</div>
          <div class="kpi-val" style="color:{nc}">{sv:.0f}</div>
          <div class="badge {slc.lower()}" style="margin-top:6px">{slc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Radar comparison
    sh("Radar Comparison — Current vs Simulated")
    cats   = ["Adoption", "Engagement", "Low Issues", "Low Defects", "Adoption"]
    orig_r = [fs["adoption_trend"], fs["engagement_intensity"],
              100 - fs["issue_burden"], 100 - fs["defect_risk"], fs["adoption_trend"]]
    sim_r  = [sa, se, 100 - si, 100 - sd, sa]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=orig_r, theta=cats, fill="toself", name=f"Current ({olc})",
                                   fillcolor=f"rgba({int(oc[1:3],16)},{int(oc[3:5],16)},{int(oc[5:7],16)},0.15)",
                                   line=dict(color=oc, width=2.5)))
    fig.add_trace(go.Scatterpolar(r=sim_r,  theta=cats, fill="toself", name=f"Simulated ({slc})",
                                   fillcolor=f"rgba({int(nc[1:3],16)},{int(nc[3:5],16)},{int(nc[5:7],16)},0.15)",
                                   line=dict(color=nc, width=2.5)))
    fig.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(visible=True, range=[0, 100],
                                   gridcolor="rgba(255,255,255,0.07)", color="#64748b", tickfont=dict(size=9)),
                   angularaxis=dict(color="#64748b", gridcolor="rgba(255,255,255,0.07)")),
        paper_bgcolor="rgba(0,0,0,0)", height=440,
        margin=dict(l=40, r=40, t=20, b=20),
        legend=dict(font=dict(color="#94a3b8", size=12), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Bar comparison
    sh("Dimension-by-Dimension Comparison")
    dims   = ["Adoption", "Engagement", "Low Issues", "Low Defects"]
    orig_v = [fs["adoption_trend"], fs["engagement_intensity"],
              100 - fs["issue_burden"], 100 - fs["defect_risk"]]
    sim_v  = [sa, se, 100 - si, 100 - sd]
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Current",   x=dims, y=orig_v, marker_color=oc, opacity=0.75))
    fig2.add_trace(go.Bar(name="Simulated", x=dims, y=sim_v,  marker_color=nc, opacity=0.9))
    fig2.update_layout(
        barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=300, margin=dict(l=4, r=4, t=10, b=4),
        font=dict(family="Inter", color="#94a3b8"),
        xaxis=dict(color="#64748b", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(color="#64748b", gridcolor="rgba(255,255,255,0.05)", range=[0, 105]),
        legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig2, use_container_width=True)
