"""
app.py — Revenue Activation OS
================================
Main entry point. Run with:  streamlit run app.py

Architecture
------------
  app.py                     ← this file: layout, sidebar, page routing
  utils/data_loader.py       ← data ingestion, cleaning, business logic
  utils/charts.py            ← all Plotly visualizations
  agents/__init__.py         ← AI agent registry (stubs for v2)
  data/pipeline_data.csv     ← source dataset
"""

import streamlit as st
import pandas as pd

from utils.data_loader import (
    load_data, filter_df, get_kpis, get_executive_summary,
    ACTIVE_STAGES, STAGE_ORDER, COLORS, RISK_COLORS
)
from utils.charts import (
    pipeline_by_stage, arr_by_industry, win_rate_by_rep,
    health_score_distribution, adoption_score_distribution,
    churn_risk_breakdown, avg_days_in_stage,
    risk_industry_heatmap, rep_performance_scatter, objection_breakdown
)
from agents import AGENT_REGISTRY
from pages.pipeline_analyst_page import render_pipeline_analyst_page

# ── Page config — must be first Streamlit call ────────────────────────────
st.set_page_config(
    page_title="Revenue Activation OS",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Base & typography ─────────────────────────────── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
    background-color: #0B1120;
    color: #E2E8F0;
  }

  /* ── Sidebar ───────────────────────────────────────── */
  [data-testid="stSidebar"] {
    background-color: #060D1A;
    border-right: 1px solid #1E2A3B;
  }
  [data-testid="stSidebar"] * { color: #94A3B8 !important; }
  [data-testid="stSidebar"] .sidebar-logo {
    font-size: 18px; font-weight: 700;
    color: #E2E8F0 !important;
    letter-spacing: -0.3px;
  }

  /* ── Main content ──────────────────────────────────── */
  .main .block-container {
    padding: 1.5rem 2rem 3rem 2rem;
    max-width: 1400px;
  }

  /* ── KPI cards ──────────────────────────────────────── */
  .kpi-card {
    background: #111827;
    border: 1px solid #1E2A3B;
    border-radius: 8px;
    padding: 20px 22px 16px 22px;
    position: relative;
    overflow: hidden;
  }
  .kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
  }
  .kpi-card.blue::before   { background: #2E86FF; }
  .kpi-card.green::before  { background: #00C48C; }
  .kpi-card.amber::before  { background: #F5A623; }
  .kpi-card.red::before    { background: #E8384F; }
  .kpi-card.purple::before { background: #7B61FF; }

  .kpi-label {
    font-size: 11px; font-weight: 600; letter-spacing: 0.8px;
    text-transform: uppercase; color: #64748B; margin-bottom: 8px;
  }
  .kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px; font-weight: 500;
    color: #E2E8F0; line-height: 1.1; margin-bottom: 6px;
  }
  .kpi-sub {
    font-size: 11px; color: #475569;
  }

  /* ── Section headers ───────────────────────────────── */
  .section-header {
    display: flex; align-items: center; gap: 10px;
    margin: 2rem 0 1rem 0; padding-bottom: 10px;
    border-bottom: 1px solid #1E2A3B;
  }
  .section-header .section-icon {
    font-size: 14px; color: #2E86FF;
  }
  .section-header h2 {
    font-size: 13px; font-weight: 600; letter-spacing: 0.6px;
    text-transform: uppercase; color: #94A3B8; margin: 0;
  }

  /* ── Executive summary card ────────────────────────── */
  .exec-card {
    background: #111827;
    border: 1px solid #1E2A3B;
    border-radius: 8px;
    padding: 20px 24px;
  }
  .exec-row {
    display: flex; justify-content: space-between;
    align-items: flex-start; gap: 12px;
    margin-bottom: 16px;
  }
  .exec-metric { flex: 1; }
  .exec-metric .label {
    font-size: 10px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.7px; color: #475569; margin-bottom: 4px;
  }
  .exec-metric .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 18px; font-weight: 500; color: #E2E8F0;
  }
  .exec-metric .value.danger { color: #E8384F; }
  .exec-metric .value.warning { color: #F5A623; }
  .exec-metric .value.success { color: #00C48C; }

  .insight-box {
    background: #0B1120;
    border-left: 3px solid #2E86FF;
    border-radius: 0 6px 6px 0;
    padding: 12px 16px;
    margin-top: 8px;
    font-size: 13px; color: #94A3B8; line-height: 1.6;
  }
  .insight-box strong { color: #E2E8F0; }

  /* ── Chart panels ───────────────────────────────────── */
  .chart-panel {
    background: #111827;
    border: 1px solid #1E2A3B;
    border-radius: 8px;
    padding: 16px;
  }

  /* ── Risk badge pills ───────────────────────────────── */
  .badge {
    display: inline-block;
    font-size: 10px; font-weight: 600;
    padding: 3px 8px; border-radius: 4px;
    text-transform: uppercase; letter-spacing: 0.5px;
  }
  .badge-red    { background: rgba(232,56,79,0.15);  color: #E8384F; }
  .badge-amber  { background: rgba(245,166,35,0.15); color: #F5A623; }
  .badge-green  { background: rgba(0,196,140,0.15);  color: #00C48C; }
  .badge-blue   { background: rgba(46,134,255,0.15); color: #2E86FF; }

  /* ── Agent cards ────────────────────────────────────── */
  .agent-card {
    background: #111827;
    border: 1px solid #1E2A3B;
    border-radius: 8px;
    padding: 16px 18px;
    opacity: 0.7;
    position: relative;
  }
  .agent-card .agent-badge {
    position: absolute; top: 14px; right: 14px;
    font-size: 9px; font-weight: 700; letter-spacing: 0.6px;
    text-transform: uppercase; color: #475569;
    background: #1E2A3B; padding: 3px 7px; border-radius: 3px;
  }
  .agent-card .agent-icon { font-size: 18px; margin-bottom: 8px; }
  .agent-card .agent-name {
    font-size: 13px; font-weight: 600; color: #94A3B8; margin-bottom: 4px;
  }
  .agent-card .agent-desc {
    font-size: 11px; color: #475569; line-height: 1.5;
  }

  /* ── Data table styling ─────────────────────────────── */
  [data-testid="stDataFrame"] {
    border: 1px solid #1E2A3B !important;
    border-radius: 8px;
  }

  /* ── Filters header ─────────────────────────────────── */
  .filter-heading {
    font-size: 10px; font-weight: 700; letter-spacing: 0.8px;
    text-transform: uppercase; color: #475569 !important;
    margin-bottom: 4px;
  }

  /* ── Pulse bar ──────────────────────────────────────── */
  .pulse-bar {
    background: #060D1A;
    border: 1px solid #1E2A3B;
    border-radius: 8px;
    padding: 12px 20px;
    display: flex; gap: 32px; align-items: center;
    margin-bottom: 1.5rem;
  }
  .pulse-item .p-label {
    font-size: 10px; font-weight: 600; letter-spacing: 0.6px;
    text-transform: uppercase; color: #475569;
  }
  .pulse-item .p-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 15px; font-weight: 500; color: #E2E8F0;
  }

  /* ── Streamlit element overrides ───────────────────── */
  div[data-testid="stSelectbox"] label,
  div[data-testid="stMultiSelect"] label {
    font-size: 10px !important; font-weight: 600 !important;
    letter-spacing: 0.7px !important; text-transform: uppercase !important;
    color: #475569 !important;
  }
  .stSelectbox > div, .stMultiSelect > div {
    background-color: #060D1A !important;
    border-color: #1E2A3B !important;
  }
  .stPlotlyChart { border-radius: 8px; overflow: hidden; }

  /* hide default streamlit header decoration */
  header[data-testid="stHeader"] { display: none; }
  .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Data loading (cached) ──────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def get_data():
    return load_data()


# ── Sidebar ────────────────────────────────────────────────────────────────
def render_sidebar(df: pd.DataFrame) -> dict:
    with st.sidebar:
        # Logo / branding
        st.markdown("""
        <div style="padding: 8px 0 24px 0;">
          <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;
                      color:#2E86FF;font-weight:700;margin-bottom:4px;">⬡ REVENUE</div>
          <div class="sidebar-logo">Activation OS</div>
          <div style="font-size:10px;color:#334155;margin-top:2px;">v1.0 · MVP Dashboard</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Page navigation
        st.markdown('<p class="filter-heading">Navigation</p>', unsafe_allow_html=True)
        page = st.selectbox(
            "page",
            ["Executive Overview", "Pipeline Intelligence", "Customer Health",
             "Sales Performance", "Account Explorer", "── AI Agents ──",
             "◈ Pipeline Analyst Agent"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Global filters
        st.markdown('<p class="filter-heading">Global Filters</p>', unsafe_allow_html=True)

        stages = st.multiselect(
            "Pipeline Stage",
            options=STAGE_ORDER,
            default=[],
            placeholder="All stages",
        )
        industries = st.multiselect(
            "Industry",
            options=sorted(df["Industry"].unique()),
            default=[],
            placeholder="All industries",
        )
        reps = st.multiselect(
            "Sales Rep",
            options=sorted(df["Sales Rep"].unique()),
            default=[],
            placeholder="All reps",
        )
        risk_tiers = st.multiselect(
            "Risk Tier",
            options=["High Risk", "Medium Risk", "Low Risk"],
            default=[],
            placeholder="All risk tiers",
        )
        deal_sizes = st.multiselect(
            "Deal Size",
            options=["SMB", "Mid-Market", "Enterprise"],
            default=[],
            placeholder="All deal sizes",
        )

        filters = {
            "stages":     stages,
            "industries": industries,
            "reps":       reps,
            "risk_tiers": risk_tiers,
            "deal_sizes": deal_sizes,
        }

        st.markdown("---")

        # Agent status panel
        st.markdown('<p class="filter-heading">AI Agents</p>', unsafe_allow_html=True)
        for key, agent in AGENT_REGISTRY.items():
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;
                        padding:6px 0;border-bottom:1px solid #1E2A3B;">
              <span style="font-size:14px;opacity:0.5">{agent['icon']}</span>
              <div>
                <div style="font-size:11px;font-weight:500;color:#475569">
                  {agent['name']}</div>
                <div style="font-size:9px;color:#2E86FF;letter-spacing:0.4px;">
                  Coming in v2</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        return page, filters


# ── KPI card renderer ──────────────────────────────────────────────────────
def kpi_card(label, value, sub, color="blue"):
    return f"""
    <div class="kpi-card {color}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>
    """


def section_header(icon, title):
    st.markdown(f"""
    <div class="section-header">
      <span class="section-icon">{icon}</span>
      <h2>{title}</h2>
    </div>
    """, unsafe_allow_html=True)


def chart_panel(fig, key=None):
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, key=key)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Revenue Pulse Bar ──────────────────────────────────────────────────────
def render_pulse_bar(kpis: dict):
    """Signature element: a scannable health strip at the top of every page."""
    wr_color = (
        "#00C48C" if kpis["win_rate"] >= 40
        else "#F5A623" if kpis["win_rate"] >= 25
        else "#E8384F"
    )
    health_color = (
        "#00C48C" if kpis["avg_health"] >= 65
        else "#F5A623" if kpis["avg_health"] >= 45
        else "#E8384F"
    )
    adoption_color = (
        "#00C48C" if kpis["avg_adoption"] >= 60
        else "#F5A623" if kpis["avg_adoption"] >= 40
        else "#E8384F"
    )

    st.markdown(f"""
    <div class="pulse-bar">
      <div style="font-size:10px;font-weight:700;letter-spacing:1.5px;
                  text-transform:uppercase;color:#2E86FF;white-space:nowrap;">
        REVENUE PULSE
      </div>
      <div style="width:1px;height:32px;background:#1E2A3B;flex-shrink:0;"></div>
      <div class="pulse-item">
        <div class="p-label">Active Pipeline</div>
        <div class="p-value">${kpis['active_pipeline']/1_000_000:.1f}M</div>
      </div>
      <div class="pulse-item">
        <div class="p-label">Win Rate</div>
        <div class="p-value" style="color:{wr_color}">{kpis['win_rate']:.1f}%</div>
      </div>
      <div class="pulse-item">
        <div class="p-label">Avg Health Score</div>
        <div class="p-value" style="color:{health_color}">{kpis['avg_health']:.0f}</div>
      </div>
      <div class="pulse-item">
        <div class="p-label">Avg Adoption</div>
        <div class="p-value" style="color:{adoption_color}">{kpis['avg_adoption']:.0f}</div>
      </div>
      <div class="pulse-item">
        <div class="p-label">High Risk Accounts</div>
        <div class="p-value" style="color:#E8384F">{kpis['high_risk_count']}</div>
      </div>
      <div class="pulse-item">
        <div class="p-label">Stale Deals</div>
        <div class="p-value" style="color:#F5A623">{kpis['stale_deals']}</div>
      </div>
      <div style="margin-left:auto;font-size:10px;color:#334155;">
        {kpis['total_opps']} opportunities · Live
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Pages ──────────────────────────────────────────────────────────────────

def page_executive_overview(df: pd.DataFrame, kpis: dict):
    summary = get_executive_summary(df)

    # ── KPI row ──────────────────────────────────────────
    section_header("◈", "Portfolio KPIs")
    cols = st.columns(5)
    cards = [
        ("Total ARR",           f"${kpis['total_arr']/1_000_000:.1f}M",
         f"{kpis['total_opps']} opportunities",          "blue"),
        ("Win Rate",            f"{kpis['win_rate']:.1f}%",
         "Closed Won / (Won + Lost)",                    "green"),
        ("Avg Adoption Score",  f"{kpis['avg_adoption']:.0f}/100",
         "Product engagement index",                     "purple"),
        ("Avg Health Score",    f"{kpis['avg_health']:.0f}/100",
         "Customer health index",                        "amber"),
        ("Avg Renewal Prob.",   f"{kpis['avg_renewal']:.0f}%",
         f"{kpis['high_risk_count']} high-risk accounts","red"),
    ]
    for col, (label, value, sub, color) in zip(cols, cards):
        col.markdown(kpi_card(label, value, sub, color), unsafe_allow_html=True)

    # ── Executive summary ─────────────────────────────────
    section_header("★", "Executive Summary")

    arr_risk_fmt  = f"${summary['arr_at_risk']/1_000_000:.1f}M"
    stale_arr_fmt = f"${summary['stale_arr']/1_000_000:.1f}M"
    pipeline_fmt  = f"${summary['total_pipeline_arr']/1_000_000:.1f}M"

    st.markdown(f"""
    <div class="exec-card">
      <div class="exec-row">
        <div class="exec-metric">
          <div class="label">Active Pipeline ARR</div>
          <div class="value">{pipeline_fmt}</div>
        </div>
        <div class="exec-metric">
          <div class="label">Win Rate</div>
          <div class="value {'warning' if summary['win_rate'] < 35 else 'success'}">{summary['win_rate']:.1f}%</div>
        </div>
        <div class="exec-metric">
          <div class="label">Pipeline Bottleneck</div>
          <div class="value warning">{summary['bottleneck_stage']}</div>
        </div>
        <div class="exec-metric">
          <div class="label">Avg Days at Bottleneck</div>
          <div class="value danger">{summary['bottleneck_days']:.0f} days</div>
        </div>
        <div class="exec-metric">
          <div class="label">Highest Risk Industry</div>
          <div class="value danger">{summary['riskiest_industry']}</div>
        </div>
      </div>
      <div class="exec-row" style="margin-bottom:0">
        <div class="exec-metric">
          <div class="label">ARR at Churn Risk</div>
          <div class="value danger">{arr_risk_fmt}</div>
        </div>
        <div class="exec-metric">
          <div class="label">High Risk Accounts</div>
          <div class="value danger">{summary['high_risk_count']}</div>
        </div>
        <div class="exec-metric">
          <div class="label">Stalled Deals</div>
          <div class="value warning">{summary['stale_deals']}</div>
        </div>
        <div class="exec-metric">
          <div class="label">Stalled ARR</div>
          <div class="value warning">{stale_arr_fmt}</div>
        </div>
        <div class="exec-metric">
          <div class="label">Expansion Industry</div>
          <div class="value success">{summary['expansion_industry']}</div>
        </div>
      </div>
      <div class="insight-box">
        <strong>Key Revenue Insight &nbsp;·&nbsp;</strong>
        {summary['key_insight']}
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row 1 ─────────────────────────────────────
    section_header("◈", "Pipeline & Risk Overview")
    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        chart_panel(pipeline_by_stage(df), key="exec_stage")
    with c2:
        chart_panel(churn_risk_breakdown(df), key="exec_churn")

    # ── Charts row 2 ─────────────────────────────────────
    c3, c4 = st.columns(2)
    with c3:
        chart_panel(avg_days_in_stage(df), key="exec_days")
    with c4:
        chart_panel(arr_by_industry(df), key="exec_industry")

    # ── Agent roadmap cards ───────────────────────────────
    section_header("⬡", "AI Agent Suite — Coming in v2")
    agent_cols = st.columns(5)
    for col, (key, agent) in zip(agent_cols, AGENT_REGISTRY.items()):
        with col:
            st.markdown(f"""
            <div class="agent-card">
              <div class="agent-badge">v2</div>
              <div class="agent-icon">{agent['icon']}</div>
              <div class="agent-name">{agent['name']}</div>
              <div class="agent-desc">{agent['description']}</div>
            </div>
            """, unsafe_allow_html=True)


def page_pipeline_intelligence(df: pd.DataFrame, kpis: dict):
    section_header("◈", "Pipeline Intelligence")

    # Headline metrics
    active = df[df["Stage"].isin(ACTIVE_STAGES)]
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Active Pipeline ARR",
        f"${kpis['active_pipeline']/1_000_000:.1f}M",
        f"{len(active)} open deals", "blue"), unsafe_allow_html=True)
    c2.markdown(kpi_card("Win Rate",
        f"{kpis['win_rate']:.1f}%",
        "of all closed deals", "green"), unsafe_allow_html=True)
    c3.markdown(kpi_card("Stale Deals",
        str(kpis['stale_deals']),
        "past 1.5× stage median", "amber"), unsafe_allow_html=True)
    stale_arr = df[df["Is Stale"]]["ARR"].sum()
    c4.markdown(kpi_card("Stalled ARR",
        f"${stale_arr/1_000_000:.1f}M",
        "at risk of slipping", "red"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    c1, c2 = st.columns([1.05, 0.95])
    with c1:
        section_header("", "Deals & ARR by Stage")
        chart_panel(pipeline_by_stage(df), key="pip_stage")
    with c2:
        section_header("", "Pipeline Velocity — Avg Days per Stage")
        chart_panel(avg_days_in_stage(df), key="pip_days")

    c3, c4 = st.columns(2)
    with c3:
        section_header("", "ARR Concentration by Industry")
        chart_panel(arr_by_industry(df), key="pip_industry")
    with c4:
        section_header("", "Objection Landscape")
        chart_panel(objection_breakdown(df), key="pip_obj")

    # Stale deal table
    section_header("⚠", "Stale Deal Alert Queue")
    stale_df = (
        df[df["Is Stale"]][[
            "Opportunity ID", "Company Name", "Sales Rep",
            "Stage", "ARR", "Days in Stage", "Objection Category", "Risk Tier"
        ]]
        .sort_values("ARR", ascending=False)
        .reset_index(drop=True)
    )
    stale_df["ARR"] = stale_df["ARR"].apply(lambda x: f"${x:,.0f}")
    stale_df["Days in Stage"] = stale_df["Days in Stage"].apply(lambda x: f"{x:.0f}d")
    st.dataframe(stale_df, use_container_width=True, height=280)


def page_customer_health(df: pd.DataFrame, kpis: dict):
    section_header("◉", "Customer Health & Retention")

    # Risk KPIs
    risk_counts = df["Risk Tier"].value_counts()
    high  = int(risk_counts.get("High Risk", 0))
    med   = int(risk_counts.get("Medium Risk", 0))
    low   = int(risk_counts.get("Low Risk", 0))
    arr_risk = df[df["Risk Tier"] == "High Risk"]["ARR"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("High Risk Accounts", str(high),
        "Renewal Probability < 50%", "red"), unsafe_allow_html=True)
    c2.markdown(kpi_card("Medium Risk Accounts", str(med),
        "Renewal Probability 50–70%", "amber"), unsafe_allow_html=True)
    c3.markdown(kpi_card("Low Risk Accounts", str(low),
        "Renewal Probability > 70%", "green"), unsafe_allow_html=True)
    c4.markdown(kpi_card("ARR at Churn Risk",
        f"${arr_risk/1_000_000:.1f}M",
        "High Risk tier total", "red"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row 1
    c1, c2 = st.columns(2)
    with c1:
        section_header("", "Churn Risk by ARR")
        chart_panel(churn_risk_breakdown(df), key="ch_donut")
    with c2:
        section_header("", "Risk Concentration by Industry")
        chart_panel(risk_industry_heatmap(df), key="ch_heatmap")

    # Charts row 2
    c3, c4 = st.columns(2)
    with c3:
        section_header("", "Customer Health Score Distribution")
        chart_panel(health_score_distribution(df), key="ch_health")
    with c4:
        section_header("", "Product Adoption Score Distribution")
        chart_panel(adoption_score_distribution(df), key="ch_adopt")

    # High-risk account table
    section_header("⚠", "High Risk Account Register")
    hr_df = (
        df[df["Risk Tier"] == "High Risk"][[
            "Company Name", "Industry", "Sales Rep",
            "ARR", "Customer Health Score",
            "Product Adoption Score", "Renewal Probability",
            "Stage", "Risk Tier"
        ]]
        .sort_values("ARR", ascending=False)
        .reset_index(drop=True)
    )
    hr_df["ARR"] = hr_df["ARR"].apply(lambda x: f"${x:,.0f}")
    hr_df["Renewal Probability"] = hr_df["Renewal Probability"].apply(lambda x: f"{x:.0f}%")
    st.dataframe(hr_df, use_container_width=True, height=300)


def page_sales_performance(df: pd.DataFrame, kpis: dict):
    section_header("◆", "Sales Team Performance")

    # Per-rep summary stats
    closed = df[df["Win/Loss"].isin(["Won", "Lost"])]
    rep_summary = (
        closed.groupby("Sales Rep")
        .apply(lambda g: pd.Series({
            "Deals Closed": len(g),
            "Win Rate (%)": round((g["Win/Loss"] == "Won").sum() / len(g) * 100, 1),
            "Won ARR": g[g["Win/Loss"] == "Won"]["ARR"].sum(),
        }), include_groups=False)
        .reset_index()
        .sort_values("Win Rate (%)", ascending=False)
        .reset_index(drop=True)
    )
    top_rep = rep_summary.iloc[0]["Sales Rep"]
    top_wr  = rep_summary.iloc[0]["Win Rate (%)"]

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Team Win Rate",
        f"{kpis['win_rate']:.1f}%", "All closed deals", "blue"),
        unsafe_allow_html=True)
    c2.markdown(kpi_card("Top Performer",
        top_rep.split()[-1], f"{top_wr:.0f}% win rate", "green"),
        unsafe_allow_html=True)
    c3.markdown(kpi_card("Avg Adoption Score",
        f"{kpis['avg_adoption']:.0f}/100",
        "Across all accounts", "purple"), unsafe_allow_html=True)
    c4.markdown(kpi_card("Stale Deals",
        str(kpis['stale_deals']),
        "Coaching opportunity signal", "amber"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row 1
    c1, c2 = st.columns([0.95, 1.05])
    with c1:
        section_header("", "Win Rate by Sales Rep")
        chart_panel(win_rate_by_rep(df), key="sp_winrate")
    with c2:
        section_header("", "Rep Performance Matrix")
        chart_panel(rep_performance_scatter(df), key="sp_scatter")

    # Charts row 2
    c3, c4 = st.columns(2)
    with c3:
        section_header("", "Objection Landscape")
        chart_panel(objection_breakdown(df), key="sp_obj")
    with c4:
        section_header("", "Pipeline Velocity")
        chart_panel(avg_days_in_stage(df), key="sp_days")

    # Rep leaderboard table
    section_header("◆", "Rep Leaderboard")
    rep_summary["Won ARR"] = rep_summary["Won ARR"].apply(lambda x: f"${x:,.0f}")
    st.dataframe(rep_summary, use_container_width=True, height=280)


def page_account_explorer(df: pd.DataFrame):
    section_header("⬡", "Account Explorer")
    st.markdown("""
    <div style="font-size:12px;color:#475569;margin-bottom:1rem;">
    Search, filter, and drill into individual accounts.
    Use the Global Filters in the sidebar to narrow the view.
    </div>
    """, unsafe_allow_html=True)

    # Search box
    search = st.text_input(
        "Search by company name or opportunity ID",
        placeholder="e.g. Apex Dynamics or OPP-0042",
        label_visibility="collapsed",
    )

    display_df = df.copy()
    if search:
        mask = (
            display_df["Company Name"].str.contains(search, case=False, na=False)
            | display_df["Opportunity ID"].str.contains(search, case=False, na=False)
        )
        display_df = display_df[mask]

    # Sort control
    col_sort, col_dir, _ = st.columns([2, 1, 4])
    sort_col = col_sort.selectbox(
        "Sort by",
        ["ARR", "Days in Stage", "Customer Health Score",
         "Product Adoption Score", "Renewal Probability"],
        label_visibility="collapsed",
    )
    sort_dir = col_dir.selectbox(
        "Order", ["Descending", "Ascending"],
        label_visibility="collapsed",
    )
    display_df = display_df.sort_values(
        sort_col, ascending=(sort_dir == "Ascending")
    )

    # Display columns
    show_cols = [
        "Opportunity ID", "Company Name", "Industry", "ARR",
        "Stage", "Win/Loss", "Sales Rep", "Deal Size",
        "Days in Stage", "Objection Category",
        "Product Adoption Score", "Customer Health Score",
        "Renewal Probability", "Risk Tier", "Composite Health"
    ]
    out_df = display_df[show_cols].reset_index(drop=True)
    out_df["ARR"] = out_df["ARR"].apply(lambda x: f"${x:,.0f}")

    st.markdown(f"""
    <div style="font-size:11px;color:#475569;margin-bottom:6px;">
      Showing <strong style="color:#94A3B8">{len(out_df)}</strong> of
      <strong style="color:#94A3B8">{len(df)}</strong> opportunities
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(out_df, use_container_width=True, height=500)

    # Quick stat summary of filtered set
    if len(display_df) > 0:
        section_header("", "Filtered Set Summary")
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Opportunities", len(display_df))
        s2.metric("Total ARR", f"${display_df['ARR'].sum():,.0f}")
        s3.metric("Avg Health", f"{display_df['Customer Health Score'].mean():.0f}")
        s4.metric("Avg Adoption", f"{display_df['Product Adoption Score'].mean():.0f}")
        s5.metric("High Risk", int((display_df["Risk Tier"] == "High Risk").sum()))


# ── Main render ────────────────────────────────────────────────────────────
def main():
    # Load base data
    with st.spinner("Loading pipeline data…"):
        df_raw = get_data()

    # Sidebar — returns page selection and active filters
    page, filters = render_sidebar(df_raw)

    # Apply filters
    df = filter_df(df_raw, filters)

    if len(df) == 0:
        st.warning("No data matches your current filters. Try adjusting the sidebar filters.")
        return

    # Compute KPIs on filtered data
    kpis = get_kpis(df)

    # Revenue Pulse bar — always visible
    render_pulse_bar(kpis)

    # Route to page
    if page == "Executive Overview":
        page_executive_overview(df, kpis)
    elif page == "Pipeline Intelligence":
        page_pipeline_intelligence(df, kpis)
    elif page == "Customer Health":
        page_customer_health(df, kpis)
    elif page == "Sales Performance":
        page_sales_performance(df, kpis)
    elif page == "Account Explorer":
        page_account_explorer(df)
    elif page == "◈ Pipeline Analyst Agent":
        render_pipeline_analyst_page(df)
    elif page == "── AI Agents ──":
        st.info("Select an agent from the navigation menu above.")


if __name__ == "__main__":
    main()
