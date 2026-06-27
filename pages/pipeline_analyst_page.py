"""
pages/pipeline_analyst_page.py
================================
Streamlit rendering layer for PipelineAnalystAgent output.
Import this module and call render_pipeline_analyst_page(df) from app.py.

This page is intentionally thin — all business logic lives in the agent.
The page's only job is to translate agent JSON into Streamlit widgets.
"""

from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Any

from agents.pipeline_analyst import PipelineAnalystAgent
from utils.data_loader import COLORS, STAGE_COLORS

# ── Colour helpers (matches existing dashboard palette) ───────────────────
PRIORITY_COLORS = {
    "P1": COLORS["danger"],
    "P2": COLORS["warning"],
    "P3": COLORS["neutral"],
}
TIER_COLORS = {
    "Top":        COLORS["success"],
    "Mid":        COLORS["primary"],
    "Developing": COLORS["warning"],
    "At Risk":    COLORS["danger"],
}
SIGNAL_COLORS = {
    "Expand":      COLORS["success"],
    "Protect":     COLORS["primary"],
    "Monitor":     COLORS["warning"],
    "Investigate": COLORS["danger"],
}
SEVERITY_COLORS = {
    "Critical": COLORS["danger"],
    "High":     COLORS["warning"],
    "Moderate": COLORS["neutral"],
}


# ── Cached agent execution ────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def _run_agent(_df: pd.DataFrame) -> dict[str, Any]:
    """Run PipelineAnalystAgent and cache the result for 5 minutes."""
    agent = PipelineAnalystAgent(_df)
    return agent.run()


# ── Shared layout primitives (mirrors app.py style) ───────────────────────
def _section(icon: str, title: str) -> None:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;
                margin:2rem 0 1rem 0;padding-bottom:10px;
                border-bottom:1px solid {COLORS['border']};">
      <span style="font-size:14px;color:{COLORS['primary']}">{icon}</span>
      <span style="font-size:13px;font-weight:600;letter-spacing:0.6px;
                   text-transform:uppercase;color:#94A3B8;">{title}</span>
    </div>
    """, unsafe_allow_html=True)


def _kpi(label: str, value: str, sub: str, color: str = COLORS["primary"]) -> str:
    return f"""
    <div style="background:{COLORS['surface']};border:1px solid {COLORS['border']};
                border-top:2px solid {color};border-radius:8px;
                padding:18px 20px 14px 20px;">
      <div style="font-size:10px;font-weight:600;letter-spacing:0.8px;
                  text-transform:uppercase;color:#64748B;margin-bottom:7px;">{label}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:26px;
                  font-weight:500;color:#E2E8F0;margin-bottom:5px;">{value}</div>
      <div style="font-size:11px;color:#475569;">{sub}</div>
    </div>"""


def _badge(text: str, color: str) -> str:
    return (
        f'<span style="display:inline-block;font-size:10px;font-weight:700;'
        f'padding:2px 7px;border-radius:4px;text-transform:uppercase;'
        f'letter-spacing:0.4px;background:{color}22;color:{color};">{text}</span>'
    )


def _panel(content_fn, *args, **kwargs) -> None:
    st.markdown(
        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
        f'border-radius:8px;padding:16px;">',
        unsafe_allow_html=True,
    )
    content_fn(*args, **kwargs)
    st.markdown("</div>", unsafe_allow_html=True)


# ── Section renderers ─────────────────────────────────────────────────────

def _render_health_score(ph: dict) -> None:
    score = ph["composite_score"]
    label = ph["health_label"]
    color = (
        COLORS["success"] if score >= 75
        else COLORS["warning"] if score >= 55
        else COLORS["danger"]
    )

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 36, "color": COLORS["text"],
                         "family": "JetBrains Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color": COLORS["text_dim"], "size": 10}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": COLORS["border"],
            "steps": [
                {"range": [0, 35],  "color": "#E8384F22"},
                {"range": [35, 55], "color": "#E8384F11"},
                {"range": [55, 75], "color": "#F5A62311"},
                {"range": [75, 100],"color": "#00C48C11"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": score,
            },
        },
        title={"text": f"Pipeline Health — {label}",
               "font": {"size": 13, "color": COLORS["text_dim"]}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
        font={"color": COLORS["text"]},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Score breakdown
    cols = st.columns(4)
    breakdown = [
        ("Win Rate Score",    ph["win_rate_score"],    "30% weight"),
        ("Velocity Score",    ph["velocity_score"],    "25% weight"),
        ("Risk Score",        ph["risk_score"],        "25% weight"),
        ("Coverage Score",    ph["coverage_score"],    "20% weight"),
    ]
    for col, (lbl, val, sub) in zip(cols, breakdown):
        sub_color = (
            COLORS["success"] if val >= 70
            else COLORS["warning"] if val >= 45
            else COLORS["danger"]
        )
        col.markdown(
            f'<div style="text-align:center;padding:8px;">'
            f'<div style="font-size:10px;color:#475569;letter-spacing:0.5px;margin-bottom:4px;">{lbl}</div>'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:20px;color:{sub_color};">{val:.0f}</div>'
            f'<div style="font-size:9px;color:#334155;">{sub}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_win_rate(wr: dict) -> None:
    # By-rep bar chart
    if wr["by_rep"]:
        reps_df = pd.DataFrame(wr["by_rep"]).sort_values("win_rate", ascending=True)
        team_avg = wr["team_avg_win_rate"]
        bar_colors = [
            COLORS["success"] if r >= team_avg
            else COLORS["warning"] if r >= team_avg * 0.75
            else COLORS["danger"]
            for r in reps_df["win_rate"]
        ]
        fig = go.Figure()
        fig.add_vline(
            x=team_avg, line_dash="dash", line_color=COLORS["neutral"],
            line_width=1.5,
            annotation_text=f"Team avg {team_avg:.0f}%",
            annotation_font=dict(color=COLORS["text_dim"], size=10),
        )
        fig.add_trace(go.Bar(
            x=reps_df["win_rate"], y=reps_df["sales_rep"],
            orientation="h",
            marker_color=bar_colors,
            marker_line_width=0,
            customdata=reps_df[["deal_count", "won_arr"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>Win Rate: <b>%{x:.1f}%</b><br>"
                "Closed: <b>%{customdata[0]}</b><br>"
                "Won ARR: <b>$%{customdata[1]:,.0f}</b><extra></extra>"
            ),
            text=reps_df["win_rate"].apply(lambda v: f"{v:.0f}%"),
            textposition="outside",
            textfont=dict(size=11, color=COLORS["text_dim"]),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=320, showlegend=False,
            margin=dict(l=10, r=40, t=10, b=10),
            xaxis=dict(showgrid=True, gridcolor=COLORS["border"],
                       tickfont=dict(color=COLORS["text_dim"], size=10),
                       range=[0, 100]),
            yaxis=dict(showgrid=False, tickfont=dict(color=COLORS["text_dim"], size=11)),
            font=dict(color=COLORS["text"]),
        )
        st.plotly_chart(fig, use_container_width=True)

    # By-industry table
    if wr["by_industry"]:
        st.markdown(
            '<div style="font-size:11px;font-weight:600;color:#64748B;'
            'letter-spacing:0.5px;text-transform:uppercase;margin-bottom:8px;">'
            'Win Rate by Industry</div>',
            unsafe_allow_html=True,
        )
        for item in wr["by_industry"]:
            wr_val = item["win_rate"]
            bar_w = int(wr_val)
            bar_col = (
                COLORS["success"] if wr_val >= wr["team_avg_win_rate"]
                else COLORS["warning"] if wr_val >= wr["team_avg_win_rate"] * 0.75
                else COLORS["danger"]
            )
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;'
                f'margin-bottom:5px;font-size:12px;">'
                f'<span style="width:130px;color:#94A3B8;white-space:nowrap;'
                f'overflow:hidden;text-overflow:ellipsis;">{item["industry"]}</span>'
                f'<div style="flex:1;background:{COLORS["border"]};border-radius:2px;height:6px;">'
                f'<div style="width:{bar_w}%;background:{bar_col};height:6px;border-radius:2px;"></div>'
                f'</div>'
                f'<span style="width:40px;text-align:right;font-family:JetBrains Mono,monospace;'
                f'font-size:11px;color:{bar_col};">{wr_val:.0f}%</span>'
                f'<span style="width:48px;text-align:right;font-size:10px;color:#475569;">'
                f'{item["deal_count"]}d</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


def _render_bottlenecks(bottlenecks: list[dict]) -> None:
    if not bottlenecks:
        st.info("No significant bottlenecks detected — pipeline velocity is within benchmarks.")
        return

    for bn in bottlenecks:
        sev_color = SEVERITY_COLORS.get(bn["severity"], COLORS["neutral"])
        pct = min(int(bn["pct_above_benchmark"]), 200)
        st.markdown(
            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
            f'border-left:4px solid {sev_color};border-radius:0 8px 8px 0;'
            f'padding:14px 18px;margin-bottom:10px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-start;">'
            f'<div>'
            f'{_badge(bn["severity"], sev_color)}&nbsp;&nbsp;'
            f'<span style="font-size:15px;font-weight:600;color:#E2E8F0;">{bn["stage"]}</span>'
            f'</div>'
            f'<div style="text-align:right;">'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:18px;color:{sev_color};">'
            f'{bn["avg_days"]:.0f}d avg</div>'
            f'<div style="font-size:10px;color:#475569;">benchmark: {bn["benchmark_days"]:.0f}d</div>'
            f'</div>'
            f'</div>'
            f'<div style="margin-top:10px;display:flex;gap:24px;">'
            f'<span style="font-size:11px;color:#94A3B8;">Deals: <b style="color:#E2E8F0">{bn["deal_count"]}</b></span>'
            f'<span style="font-size:11px;color:#94A3B8;">ARR at Risk: <b style="color:{sev_color}">{bn["arr_at_risk_fmt"]}</b></span>'
            f'<span style="font-size:11px;color:#94A3B8;">Excess: <b style="color:{sev_color}">+{bn["excess_days"]:.0f}d ({bn["pct_above_benchmark"]:.0f}% above)</b></span>'
            f'</div>'
            f'<div style="margin-top:8px;background:{COLORS["border"]};border-radius:2px;height:4px;">'
            f'<div style="width:{pct}%;max-width:100%;background:{sev_color};height:4px;border-radius:2px;"></div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_high_risk(risks: list[dict]) -> None:
    p1 = [r for r in risks if r["priority"] == "P1"]
    p2 = [r for r in risks if r["priority"] == "P2"]

    for priority_label, group in [("P1 — Immediate Action", p1), ("P2 — This Week", p2)]:
        if not group:
            continue
        color = COLORS["danger"] if "P1" in priority_label else COLORS["warning"]
        st.markdown(
            f'<div style="font-size:11px;font-weight:700;color:{color};'
            f'letter-spacing:0.6px;text-transform:uppercase;'
            f'margin:12px 0 8px 0;">{priority_label} ({len(group)} accounts)</div>',
            unsafe_allow_html=True,
        )
        for r in group[:10]:   # cap display at 10 per tier
            flags_html = " &nbsp;·&nbsp; ".join(
                f'<span style="color:#94A3B8">{f}</span>' for f in r["risk_flags"]
            )
            st.markdown(
                f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
                f'border-radius:8px;padding:12px 16px;margin-bottom:8px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
                f'<div>'
                f'<span style="font-size:13px;font-weight:600;color:#E2E8F0;">{r["company_name"]}</span>'
                f'&nbsp;<span style="font-size:11px;color:#475569;">{r["opportunity_id"]}</span>'
                f'</div>'
                f'<div style="display:flex;gap:12px;align-items:center;">'
                f'<span style="font-family:JetBrains Mono,monospace;font-size:14px;color:{color};">{r["arr_fmt"]}</span>'
                f'{_badge(r["stage"], STAGE_COLORS.get(r["stage"], COLORS["neutral"]))}'
                f'</div>'
                f'</div>'
                f'<div style="display:flex;gap:18px;margin-bottom:6px;">'
                f'<span style="font-size:11px;color:#64748B;">Rep: <b style="color:#94A3B8">{r["sales_rep"]}</b></span>'
                f'<span style="font-size:11px;color:#64748B;">Industry: <b style="color:#94A3B8">{r["industry"]}</b></span>'
                f'<span style="font-size:11px;color:#64748B;">Risk Score: <b style="color:{color}">{r["risk_score"]:.0f}/100</b></span>'
                f'<span style="font-size:11px;color:#64748B;">Renewal: <b style="color:{color}">{r["renewal_probability"]:.0f}%</b></span>'
                f'</div>'
                f'<div style="font-size:11px;color:#475569;line-height:1.5;">{flags_html}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def _render_rep_performance(reps: list[dict]) -> None:
    if not reps:
        return

    # Radar-style table
    for rep in reps:
        tier_color = TIER_COLORS.get(rep["performance_tier"], COLORS["neutral"])
        flags_html = (
            "<br>".join(
                f'<span style="color:#64748B;">▸ {f}</span>'
                for f in rep["coaching_flags"]
            )
            if rep["coaching_flags"]
            else '<span style="color:#00C48C;">No coaching flags — performing within expectations</span>'
        )
        st.markdown(
            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
            f'border-radius:8px;padding:14px 18px;margin-bottom:10px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">'
            f'<div>'
            f'<span style="font-size:14px;font-weight:600;color:#E2E8F0;">{rep["sales_rep"]}</span>'
            f'&nbsp;&nbsp;{_badge(rep["performance_tier"], tier_color)}'
            f'</div>'
            f'<div style="display:flex;gap:20px;text-align:right;">'
            f'<div><div style="font-size:9px;color:#475569;">WIN RATE</div>'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:18px;color:{tier_color};">'
            f'{rep["win_rate"]:.0f}%</div></div>'
            f'<div><div style="font-size:9px;color:#475569;">WON ARR</div>'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:18px;color:#E2E8F0;">'
            f'{rep["won_arr_fmt"]}</div></div>'
            f'<div><div style="font-size:9px;color:#475569;">OPEN PIPELINE</div>'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:18px;color:{COLORS["primary"]};">'
            f'{rep["open_arr_fmt"]}</div></div>'
            f'</div>'
            f'</div>'
            f'<div style="display:flex;gap:16px;margin-bottom:8px;">'
            f'<span style="font-size:11px;color:#64748B;">W: <b style="color:{COLORS["success"]}">{rep["deals_won"]}</b></span>'
            f'<span style="font-size:11px;color:#64748B;">L: <b style="color:{COLORS["danger"]}">{rep["deals_lost"]}</b></span>'
            f'<span style="font-size:11px;color:#64748B;">Open: <b style="color:#94A3B8">{rep["deals_open"]}</b></span>'
            f'<span style="font-size:11px;color:#64748B;">Avg Health: <b style="color:#94A3B8">{rep["avg_health_score"]:.0f}</b></span>'
            f'<span style="font-size:11px;color:#64748B;">Avg Adoption: <b style="color:#94A3B8">{rep["avg_adoption_score"]:.0f}</b></span>'
            f'</div>'
            f'<div style="font-size:11px;line-height:1.7;">{flags_html}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_industry(industries: list[dict]) -> None:
    # Bubble chart: x=win_rate, y=high_risk_pct, size=total_arr, color=signal
    if not industries:
        return

    df = pd.DataFrame(industries)
    df["color"] = df["signal"].map(SIGNAL_COLORS).fillna(COLORS["neutral"])
    df["size"]  = (df["total_arr"] / df["total_arr"].max() * 40 + 10).clip(10, 50)

    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["win_rate"]], y=[row["high_risk_pct"]],
            mode="markers+text",
            marker=dict(
                size=row["size"], color=row["color"],
                opacity=0.85, line=dict(width=1, color=COLORS["border"]),
            ),
            text=[row["industry"].split()[0]],
            textposition="top center",
            textfont=dict(size=9, color=COLORS["text_dim"]),
            hovertemplate=(
                f"<b>{row['industry']}</b><br>"
                f"Win Rate: <b>{row['win_rate']:.0f}%</b><br>"
                f"High Risk: <b>{row['high_risk_pct']:.0f}%</b><br>"
                f"ARR: <b>{row['total_arr_fmt']}</b><br>"
                f"Signal: <b>{row['signal']}</b>"
                "<extra></extra>"
            ),
            name=row["industry"],
            showlegend=False,
        ))

    # Quadrant lines
    avg_wr    = df["win_rate"].mean()
    avg_risk  = df["high_risk_pct"].mean()
    fig.add_hline(y=avg_risk, line_dash="dot", line_color=COLORS["border"], line_width=1)
    fig.add_vline(x=avg_wr,   line_dash="dot", line_color=COLORS["border"], line_width=1)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=340, showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Win Rate (%)", showgrid=True, gridcolor=COLORS["border"],
                   tickfont=dict(color=COLORS["text_dim"], size=10),
                   title_font=dict(color=COLORS["text_dim"])),
        yaxis=dict(title="% High Risk Accounts", showgrid=True, gridcolor=COLORS["border"],
                   tickfont=dict(color=COLORS["text_dim"], size=10),
                   title_font=dict(color=COLORS["text_dim"])),
        font=dict(color=COLORS["text"]),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Signal legend
    cols = st.columns(4)
    for col, (sig, color) in zip(cols, SIGNAL_COLORS.items()):
        count = int((df["signal"] == sig).sum())
        col.markdown(
            f'<div style="text-align:center;padding:8px;">'
            f'{_badge(sig, color)}'
            f'<div style="font-size:12px;font-family:JetBrains Mono,monospace;'
            f'color:{color};margin-top:4px;">{count}</div>'
            f'<div style="font-size:10px;color:#475569;">industries</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_opportunities(opps: list[dict]) -> None:
    type_colors = {
        "Quick Win":   COLORS["success"],
        "Expansion":   COLORS["primary"],
        "Acceleration": COLORS["warning"],
    }
    for opp in opps[:15]:   # Show top 15
        color = type_colors.get(opp["opportunity_type"], COLORS["neutral"])
        st.markdown(
            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
            f'border-left:3px solid {color};border-radius:0 8px 8px 0;'
            f'padding:12px 16px;margin-bottom:8px;display:flex;'
            f'justify-content:space-between;align-items:flex-start;">'
            f'<div style="flex:1;">'
            f'{_badge(opp["opportunity_type"], color)}'
            f'&nbsp;&nbsp;<span style="font-size:13px;font-weight:600;color:#E2E8F0;">'
            f'{opp["company_name"]}</span>'
            f'&nbsp;<span style="font-size:11px;color:#475569;">{opp["opportunity_id"]}</span>'
            f'<div style="margin-top:5px;font-size:11px;color:#64748B;">'
            f'{opp["rationale"]}</div>'
            f'<div style="margin-top:4px;font-size:11px;color:#475569;font-style:italic;">'
            f'→ {opp["recommended_action"]}</div>'
            f'</div>'
            f'<div style="text-align:right;margin-left:16px;flex-shrink:0;">'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:16px;color:{color};">'
            f'{opp["arr_fmt"]}</div>'
            f'<div style="font-size:10px;color:#475569;">{opp["stage"]}</div>'
            f'<div style="font-size:10px;color:#475569;">{opp["sales_rep"].split()[-1]}</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_recommendations(recs: list[dict]) -> None:
    priority_colors = {
        "P1 — Immediate":   COLORS["danger"],
        "P2 — This Week":   COLORS["warning"],
        "P3 — This Month":  COLORS["neutral"],
    }
    effort_colors = {
        "Low":    COLORS["success"],
        "Medium": COLORS["warning"],
        "High":   COLORS["danger"],
    }
    for rec in recs:
        p_color = priority_colors.get(rec["priority"], COLORS["neutral"])
        e_color = effort_colors.get(rec["effort"], COLORS["neutral"])
        st.markdown(
            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
            f'border-top:2px solid {p_color};border-radius:8px;'
            f'padding:16px 20px;margin-bottom:12px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;">'
            f'<div style="flex:1;">'
            f'<div style="margin-bottom:6px;">'
            f'{_badge(rec["id"], COLORS["primary"])}&nbsp;'
            f'{_badge(rec["priority"], p_color)}&nbsp;'
            f'{_badge(rec["category"], COLORS["neutral"])}'
            f'</div>'
            f'<div style="font-size:14px;font-weight:600;color:#E2E8F0;line-height:1.3;">'
            f'{rec["headline"]}</div>'
            f'</div>'
            f'<div style="text-align:right;margin-left:16px;flex-shrink:0;">'
            f'<div style="font-size:9px;color:#475569;margin-bottom:2px;">EST. ARR IMPACT</div>'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:18px;color:{p_color};">'
            f'{rec["estimated_arr_impact_fmt"]}</div>'
            f'</div>'
            f'</div>'
            f'<div style="font-size:12px;color:#94A3B8;line-height:1.6;margin-bottom:10px;">'
            f'{rec["detail"]}</div>'
            f'<div style="display:flex;gap:16px;border-top:1px solid {COLORS["border"]};'
            f'padding-top:8px;">'
            f'<span style="font-size:10px;color:#475569;">Trigger: '
            f'<b style="color:#64748B">{rec["metric_trigger"]}</b></span>'
            f'&nbsp;&nbsp;'
            f'<span style="font-size:10px;color:#475569;">Owner: '
            f'<b style="color:#94A3B8">{rec["owner"]}</b></span>'
            f'&nbsp;&nbsp;'
            f'{_badge("Effort: " + rec["effort"], e_color)}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Main page renderer ────────────────────────────────────────────────────

def render_pipeline_analyst_page(df: pd.DataFrame) -> None:
    """
    Main entry point called from app.py.
    Runs the agent (cached), then renders all sections.
    """
    # ── Agent execution ───────────────────────────────────
    with st.spinner("Running Pipeline Analyst Agent…"):
        report = _run_agent(df)

    ph    = report["pipeline_health"]
    wr    = report["win_rate_analysis"]
    bns   = report["pipeline_bottlenecks"]
    risks = report["high_risk_opportunities"]
    reps  = report["sales_rep_analysis"]
    inds  = report["industry_analysis"]
    opps  = report["revenue_opportunities"]
    recs  = report["recommendations"]
    es    = report["executive_summary"]

    # ── Agent header ──────────────────────────────────────
    st.markdown(f"""
    <div style="background:{COLORS['surface']};border:1px solid {COLORS['border']};
                border-radius:8px;padding:16px 20px;margin-bottom:1.5rem;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div>
          <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;
                      color:{COLORS['primary']};font-weight:700;margin-bottom:4px;">
            ◈ PIPELINE ANALYST AGENT · v1.0.0
          </div>
          <div style="font-size:15px;font-weight:600;color:#E2E8F0;line-height:1.4;max-width:820px;">
            {es['headline']}
          </div>
        </div>
        <div style="text-align:right;flex-shrink:0;margin-left:20px;">
          <div style="font-size:9px;color:#334155;margin-bottom:2px;">LAST RUN</div>
          <div style="font-size:11px;font-family:JetBrains Mono,monospace;color:#475569;">
            {report['meta']['run_timestamp'][:19].replace('T',' ')} UTC
          </div>
          <div style="font-size:10px;color:#334155;margin-top:4px;">
            {report['meta']['total_opportunities']} opportunities analysed
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Top KPIs ──────────────────────────────────────────
    cols = st.columns(5)
    cards = [
        ("Pipeline ARR",        ph["total_pipeline_arr_fmt"],  f"{len(df[df['Stage'].isin(['Prospecting','Discovery','Qualification','Proposal','Negotiation'])])} active deals", COLORS["primary"]),
        ("Overall Win Rate",    f"{wr['overall_win_rate']:.0f}%", f"{wr['total_won_deals']} won / {wr['total_closed_deals']} closed", COLORS["success"]),
        ("P1 Risk Accounts",    str(es["p1_accounts"]),         es["p1_arr_fmt"] + " ARR at risk",  COLORS["danger"]),
        ("Pipeline Bottlenecks",str(es["total_bottlenecks"]),   f"{es['critical_bottlenecks']} Critical",   COLORS["warning"]),
        ("Stale Deals",         str(es["stale_deals"]),          ph["stale_arr_fmt"] + " stalled ARR", COLORS["warning"]),
    ]
    for col, (label, value, sub, color) in zip(cols, cards):
        col.markdown(_kpi(label, value, sub, color), unsafe_allow_html=True)

    # ── Pipeline Health Score ─────────────────────────────
    _section("◈", "Pipeline Health Score")
    _render_health_score(ph)

    # ── Win Rate Analysis ─────────────────────────────────
    _section("◈", "Win Rate Analysis")
    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        st.markdown('<div style="font-size:11px;font-weight:600;color:#64748B;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:8px;">Win Rate by Sales Rep</div>', unsafe_allow_html=True)
        _render_win_rate(wr)
    with c2:
        st.markdown('<div style="font-size:11px;font-weight:600;color:#64748B;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:8px;">Loss Drivers — Top Objections</div>', unsafe_allow_html=True)
        if wr.get("top_loss_objections"):
            obj_df = pd.DataFrame(
                list(wr["top_loss_objections"].items()),
                columns=["Objection", "Count"]
            ).sort_values("Count", ascending=True)
            fig = go.Figure(go.Bar(
                x=obj_df["Count"], y=obj_df["Objection"],
                orientation="h",
                marker_color=COLORS["danger"],
                marker_line_width=0,
                opacity=0.8,
                text=obj_df["Count"],
                textposition="outside",
                textfont=dict(size=11, color=COLORS["text_dim"]),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=280, showlegend=False,
                margin=dict(l=10, r=40, t=10, b=10),
                xaxis=dict(showgrid=True, gridcolor=COLORS["border"],
                           tickfont=dict(color=COLORS["text_dim"], size=10)),
                yaxis=dict(showgrid=False, tickfont=dict(color=COLORS["text_dim"], size=11)),
                font=dict(color=COLORS["text"]),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Bottlenecks ───────────────────────────────────────
    _section("⚠", "Pipeline Bottleneck Analysis")
    _render_bottlenecks(bns)

    # ── High-Risk Opportunities ───────────────────────────
    _section("🔴", "High-Risk Opportunity Register")
    _render_high_risk(risks)

    # ── Rep Performance ───────────────────────────────────
    _section("◆", "Sales Rep Performance & Coaching Signals")
    _render_rep_performance(reps)

    # ── Industry Analysis ─────────────────────────────────
    _section("◉", "Industry Intelligence Matrix")
    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        _render_industry(inds)
    with c2:
        st.markdown('<div style="font-size:11px;font-weight:600;color:#64748B;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:8px;">Industry Signal Summary</div>', unsafe_allow_html=True)
        for ind in sorted(inds, key=lambda i: i["total_arr"], reverse=True)[:8]:
            sig_color = SIGNAL_COLORS.get(ind["signal"], COLORS["neutral"])
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:center;padding:7px 0;border-bottom:1px solid {COLORS["border"]};">'
                f'<span style="font-size:12px;color:#94A3B8;">{ind["industry"]}</span>'
                f'<div style="display:flex;gap:12px;align-items:center;">'
                f'<span style="font-family:JetBrains Mono,monospace;font-size:11px;color:#E2E8F0;">'
                f'{ind["total_arr_fmt"]}</span>'
                f'<span style="font-size:11px;color:#475569;">WR: {ind["win_rate"]:.0f}%</span>'
                f'{_badge(ind["signal"], sig_color)}'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    # ── Revenue Opportunities ─────────────────────────────
    _section("◇", "Revenue Opportunity Queue")
    tab_qw, tab_exp, tab_accel = st.tabs(["Quick Wins", "Expansion", "Acceleration"])
    with tab_qw:
        qw = [o for o in opps if o["opportunity_type"] == "Quick Win"]
        st.markdown(f'<div style="font-size:11px;color:#475569;margin-bottom:8px;">{len(qw)} quick-win opportunities identified</div>', unsafe_allow_html=True)
        _render_opportunities(qw)
    with tab_exp:
        exp = [o for o in opps if o["opportunity_type"] == "Expansion"]
        st.markdown(f'<div style="font-size:11px;color:#475569;margin-bottom:8px;">{len(exp)} expansion opportunities identified</div>', unsafe_allow_html=True)
        _render_opportunities(exp)
    with tab_accel:
        acc = [o for o in opps if o["opportunity_type"] == "Acceleration"]
        st.markdown(f'<div style="font-size:11px;color:#475569;margin-bottom:8px;">{len(acc)} pipeline acceleration opportunities identified</div>', unsafe_allow_html=True)
        _render_opportunities(acc)

    # ── Recommendations ───────────────────────────────────
    _section("★", f"Agent Recommendations ({len(recs)} prioritised actions)")
    _render_recommendations(recs)

    # ── Raw JSON expander ────────────────────────────────
    with st.expander("View raw agent output (JSON)", expanded=False):
        import json
        st.code(json.dumps(report, indent=2, default=str), language="json")
