"""
utils/charts.py
---------------
All Plotly chart definitions for Revenue Activation OS.
Each function answers a specific business question — documented inline.

Design principles:
  - Dark theme throughout (#0B1120 background family)
  - Consistent color encoding: green=healthy, amber=warning, red=critical
  - Minimal chrome — no gridlines unless they carry meaning
  - Every chart has a meaningful axis label, not a raw column name

Future: Agent outputs will be overlaid on these charts as annotation layers.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from utils.data_loader import (
    STAGE_ORDER, ACTIVE_STAGES, STAGE_COLORS, RISK_COLORS, COLORS
)

# ── Shared layout base ─────────────────────────────────────────────────────
LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'Inter', 'SF Pro Display', system-ui, sans-serif",
              color=COLORS["text"], size=12),
    title_font=dict(size=14, color=COLORS["text"]),
    margin=dict(l=16, r=16, t=40, b=16),
    legend=dict(
        bgcolor="rgba(17,24,39,0.8)",
        bordercolor=COLORS["border"],
        borderwidth=1,
        font=dict(size=11),
    ),
    hoverlabel=dict(
        bgcolor=COLORS["surface"],
        bordercolor=COLORS["border"],
        font_color=COLORS["text"],
        font_size=12,
    ),
)

AXIS_STYLE = dict(
    showgrid=True,
    gridcolor=COLORS["border"],
    gridwidth=1,
    zeroline=False,
    tickfont=dict(color=COLORS["text_dim"], size=11),
    title_font=dict(color=COLORS["text_dim"], size=12),
)

AXIS_STYLE_NOGRID = dict(
    showgrid=False,
    zeroline=False,
    tickfont=dict(color=COLORS["text_dim"], size=11),
    title_font=dict(color=COLORS["text_dim"], size=12),
)


def _apply_layout(fig, title="", height=380):
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, x=0, xanchor="left",
                   font=dict(size=13, color=COLORS["text"])),
        height=height,
    )
    return fig


# ── 1. Pipeline Distribution by Stage ─────────────────────────────────────
# Business question: Where is ARR concentrating in the funnel,
# and how many deals are stuck at each stage?
def pipeline_by_stage(df: pd.DataFrame) -> go.Figure:
    stage_df = (
        df.groupby("Stage", observed=True)
        .agg(
            Deal_Count=("Opportunity ID", "count"),
            Total_ARR=("ARR", "sum"),
        )
        .reset_index()
    )
    stage_df["Stage"] = pd.Categorical(
        stage_df["Stage"], categories=STAGE_ORDER, ordered=True
    )
    stage_df = stage_df.sort_values("Stage")
    stage_df["Color"] = stage_df["Stage"].map(STAGE_COLORS)
    stage_df["ARR_Label"] = stage_df["Total_ARR"].apply(
        lambda x: f"${x/1_000_000:.1f}M" if x >= 1_000_000 else f"${x/1_000:.0f}K"
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=stage_df["Stage"],
        y=stage_df["Deal_Count"],
        marker_color=stage_df["Color"].tolist(),
        marker_line_width=0,
        customdata=np.stack([
            stage_df["Total_ARR"],
            stage_df["ARR_Label"],
        ], axis=-1),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Deals: <b>%{y}</b><br>"
            "ARR: <b>%{customdata[1]}</b>"
            "<extra></extra>"
        ),
        text=stage_df["ARR_Label"],
        textposition="outside",
        textfont=dict(size=11, color=COLORS["text_dim"]),
    ))

    _apply_layout(fig, height=360)
    fig.update_xaxes(**AXIS_STYLE_NOGRID, title="Pipeline Stage")
    fig.update_yaxes(**AXIS_STYLE, title="Number of Deals")
    fig.update_layout(showlegend=False)
    return fig


# ── 2. ARR by Industry ─────────────────────────────────────────────────────
# Business question: Which industries represent the highest ARR concentration
# and potential revenue exposure?
def arr_by_industry(df: pd.DataFrame) -> go.Figure:
    ind_df = (
        df.groupby("Industry")
        .agg(Total_ARR=("ARR", "sum"), Deal_Count=("Opportunity ID", "count"))
        .reset_index()
        .sort_values("Total_ARR", ascending=True)
    )
    ind_df["ARR_M"] = ind_df["Total_ARR"] / 1_000_000
    ind_df["Label"] = ind_df["Total_ARR"].apply(
        lambda x: f"${x/1_000_000:.1f}M"
    )

    # Color gradient: top quartile gets accent blue, rest get muted
    q75 = ind_df["Total_ARR"].quantile(0.75)
    ind_df["Bar_Color"] = ind_df["Total_ARR"].apply(
        lambda x: COLORS["primary"] if x >= q75 else "#1E3A5F"
    )

    fig = go.Figure(go.Bar(
        x=ind_df["ARR_M"],
        y=ind_df["Industry"],
        orientation="h",
        marker_color=ind_df["Bar_Color"].tolist(),
        marker_line_width=0,
        customdata=np.stack([ind_df["Deal_Count"], ind_df["Label"]], axis=-1),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "ARR: <b>%{customdata[1]}</b><br>"
            "Deals: <b>%{customdata[0]}</b>"
            "<extra></extra>"
        ),
        text=ind_df["Label"],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["text_dim"]),
    ))

    _apply_layout(fig, height=420)
    fig.update_xaxes(**AXIS_STYLE, title="Total ARR (Millions)")
    fig.update_yaxes(**AXIS_STYLE_NOGRID, title="")
    fig.update_layout(showlegend=False)
    return fig


# ── 3. Win Rate by Sales Rep ───────────────────────────────────────────────
# Business question: Which reps are converting and which are underperforming?
# Surfaces the coaching signal for Sales Managers.
def win_rate_by_rep(df: pd.DataFrame) -> go.Figure:
    closed = df[df["Win/Loss"].isin(["Won", "Lost"])]
    rep_df = (
        closed.groupby("Sales Rep")
        .apply(lambda g: pd.Series({
            "Win Rate": (g["Win/Loss"] == "Won").sum() / len(g) * 100,
            "Deals Closed": len(g),
            "Avg ARR": g[g["Win/Loss"] == "Won"]["ARR"].mean() if (g["Win/Loss"] == "Won").any() else 0,
        }), include_groups=False)
        .reset_index()
        .sort_values("Win Rate", ascending=True)
    )

    team_avg = rep_df["Win Rate"].mean()

    rep_df["Bar_Color"] = rep_df["Win Rate"].apply(
        lambda x: COLORS["success"] if x >= team_avg
                  else (COLORS["warning"] if x >= team_avg * 0.7
                        else COLORS["danger"])
    )

    fig = go.Figure()
    # Team average reference line
    fig.add_vline(
        x=team_avg,
        line_dash="dash",
        line_color=COLORS["neutral"],
        line_width=1.5,
        annotation_text=f"Team avg {team_avg:.0f}%",
        annotation_position="top right",
        annotation_font=dict(color=COLORS["text_dim"], size=10),
    )

    fig.add_trace(go.Bar(
        x=rep_df["Win Rate"],
        y=rep_df["Sales Rep"],
        orientation="h",
        marker_color=rep_df["Bar_Color"].tolist(),
        marker_line_width=0,
        customdata=np.stack([
            rep_df["Deals Closed"],
            rep_df["Avg ARR"].fillna(0),
        ], axis=-1),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Win Rate: <b>%{x:.1f}%</b><br>"
            "Deals Closed: <b>%{customdata[0]}</b><br>"
            "Avg Won ARR: <b>$%{customdata[1]:,.0f}</b>"
            "<extra></extra>"
        ),
        text=rep_df["Win Rate"].apply(lambda x: f"{x:.0f}%"),
        textposition="outside",
        textfont=dict(size=11, color=COLORS["text_dim"]),
    ))

    _apply_layout(fig, height=360)
    fig.update_xaxes(**AXIS_STYLE, title="Win Rate (%)", range=[0, 100])
    fig.update_yaxes(**AXIS_STYLE_NOGRID, title="")
    fig.update_layout(showlegend=False)
    return fig


# ── 4. Customer Health Score Distribution ─────────────────────────────────
# Business question: What is the shape of customer health across the portfolio?
# A left-skewed distribution is a churn early-warning signal.
def health_score_distribution(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=df["Customer Health Score"],
        nbinsx=20,
        marker_color=COLORS["primary"],
        marker_line_width=0,
        opacity=0.85,
        name="Health Score",
        hovertemplate="Score: <b>%{x}</b><br>Count: <b>%{y}</b><extra></extra>",
    ))

    # Risk zone annotations
    fig.add_vrect(x0=0,  x1=40, fillcolor=COLORS["danger"],  opacity=0.08, line_width=0)
    fig.add_vrect(x0=40, x1=65, fillcolor=COLORS["warning"], opacity=0.06, line_width=0)
    fig.add_vrect(x0=65, x1=100,fillcolor=COLORS["success"], opacity=0.06, line_width=0)

    mean_val = df["Customer Health Score"].mean()
    fig.add_vline(
        x=mean_val, line_dash="dash", line_color=COLORS["warning"], line_width=1.5,
        annotation_text=f"Avg {mean_val:.0f}",
        annotation_font=dict(color=COLORS["warning"], size=10),
    )

    _apply_layout(fig, height=300)
    fig.update_xaxes(**AXIS_STYLE, title="Customer Health Score", range=[0, 100])
    fig.update_yaxes(**AXIS_STYLE, title="Accounts")
    fig.update_layout(showlegend=False, bargap=0.05)
    return fig


# ── 5. Product Adoption Score Distribution ────────────────────────────────
# Business question: Are customers actually using the product?
# Low adoption = churn risk and missed expansion opportunity.
def adoption_score_distribution(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df["Product Adoption Score"],
        nbinsx=20,
        marker_color=COLORS["success"],
        marker_line_width=0,
        opacity=0.85,
        name="Adoption Score",
        hovertemplate="Score: <b>%{x}</b><br>Count: <b>%{y}</b><extra></extra>",
    ))

    fig.add_vrect(x0=0,  x1=30, fillcolor=COLORS["danger"],  opacity=0.08, line_width=0)
    fig.add_vrect(x0=30, x1=60, fillcolor=COLORS["warning"], opacity=0.06, line_width=0)
    fig.add_vrect(x0=60, x1=100,fillcolor=COLORS["success"], opacity=0.06, line_width=0)

    mean_val = df["Product Adoption Score"].mean()
    fig.add_vline(
        x=mean_val, line_dash="dash", line_color=COLORS["warning"], line_width=1.5,
        annotation_text=f"Avg {mean_val:.0f}",
        annotation_font=dict(color=COLORS["warning"], size=10),
    )

    _apply_layout(fig, height=300)
    fig.update_xaxes(**AXIS_STYLE, title="Product Adoption Score", range=[0, 100])
    fig.update_yaxes(**AXIS_STYLE, title="Accounts")
    fig.update_layout(showlegend=False, bargap=0.05)
    return fig


# ── 6. Churn Risk Breakdown ────────────────────────────────────────────────
# Business question: How much ARR sits in each risk tier?
# Shows CS leaders where to prioritize retention effort.
def churn_risk_breakdown(df: pd.DataFrame) -> go.Figure:
    risk_df = (
        df.groupby("Risk Tier", observed=True)
        .agg(
            ARR=("ARR", "sum"),
            Count=("Opportunity ID", "count"),
        )
        .reset_index()
    )
    risk_df["ARR_M"] = risk_df["ARR"] / 1_000_000
    risk_df["Label"] = risk_df["ARR"].apply(
        lambda x: f"${x/1_000_000:.1f}M"
    )
    risk_df["Color"] = risk_df["Risk Tier"].map(RISK_COLORS)

    fig = go.Figure(go.Pie(
        labels=risk_df["Risk Tier"],
        values=risk_df["ARR"],
        marker=dict(
            colors=risk_df["Color"].tolist(),
            line=dict(color=COLORS["background"], width=3),
        ),
        hole=0.62,
        textinfo="percent+label",
        textfont=dict(size=12, color=COLORS["text"]),
        hovertemplate=(
            "<b>%{label}</b><br>"
            "ARR: <b>$%{value:,.0f}</b><br>"
            "Share: <b>%{percent}</b>"
            "<extra></extra>"
        ),
        direction="clockwise",
        sort=False,
    ))

    # Center annotation — total ARR
    total_arr = risk_df["ARR"].sum()
    fig.add_annotation(
        text=f"${total_arr/1_000_000:.1f}M<br><span style='font-size:10px'>Total ARR</span>",
        x=0.5, y=0.5,
        font=dict(size=16, color=COLORS["text"]),
        showarrow=False,
    )

    _apply_layout(fig, height=320)
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v", x=1.0, y=0.5,
            font=dict(size=11),
        ),
    )
    return fig


# ── 7. Average Days in Stage ───────────────────────────────────────────────
# Business question: Where is the pipeline velocity degrading?
# The stage with the longest avg dwell time is the bottleneck.
def avg_days_in_stage(df: pd.DataFrame) -> go.Figure:
    active = df[df["Stage"].isin(ACTIVE_STAGES)]
    days_df = (
        active.groupby("Stage", observed=True)["Days in Stage"]
        .agg(["mean", "median", "std"])
        .reset_index()
        .rename(columns={"mean": "Mean", "median": "Median", "std": "StdDev"})
    )
    days_df["Stage"] = pd.Categorical(
        days_df["Stage"], categories=ACTIVE_STAGES, ordered=True
    )
    days_df = days_df.sort_values("Stage")
    days_df["Color"] = days_df["Stage"].map(STAGE_COLORS)

    # Highlight the bottleneck (max mean days)
    max_idx = days_df["Mean"].idxmax()
    days_df["Is_Bottleneck"] = days_df.index == max_idx

    bar_colors = [
        COLORS["danger"] if is_bn else color
        for is_bn, color in zip(days_df["Is_Bottleneck"], days_df["Color"])
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=days_df["Stage"],
        y=days_df["Mean"],
        marker_color=bar_colors,
        marker_line_width=0,
        name="Avg Days",
        error_y=dict(
            type="data",
            array=days_df["StdDev"].fillna(0).tolist(),
            color=COLORS["text_dim"],
            thickness=1.5,
            width=4,
        ),
        customdata=np.stack([days_df["Median"], days_df["StdDev"].fillna(0)], axis=-1),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Avg Days: <b>%{y:.1f}</b><br>"
            "Median: <b>%{customdata[0]:.1f}</b><br>"
            "Std Dev: <b>±%{customdata[1]:.1f}</b>"
            "<extra></extra>"
        ),
        text=days_df["Mean"].apply(lambda x: f"{x:.0f}d"),
        textposition="outside",
        textfont=dict(size=11, color=COLORS["text_dim"]),
    ))

    # Bottleneck label
    bottleneck_stage = days_df.loc[max_idx, "Stage"]
    bottleneck_days  = days_df.loc[max_idx, "Mean"]
    fig.add_annotation(
        x=bottleneck_stage,
        y=bottleneck_days,
        text="⚠ Bottleneck",
        showarrow=False,
        yshift=26,
        font=dict(color=COLORS["danger"], size=10),
    )

    _apply_layout(fig, height=340)
    fig.update_xaxes(**AXIS_STYLE_NOGRID, title="Pipeline Stage")
    fig.update_yaxes(**AXIS_STYLE, title="Average Days in Stage")
    fig.update_layout(showlegend=False)
    return fig


# ── 8. Risk Tier × Industry Heatmap ───────────────────────────────────────
# Business question: Which industry-risk combinations need immediate attention?
def risk_industry_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = (
        df.groupby(["Industry", "Risk Tier"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["High Risk", "Medium Risk", "Low Risk"], fill_value=0)
    )
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot_sorted = pivot_pct.sort_values("High Risk", ascending=False)

    z_vals   = pivot_sorted.values
    y_labels = pivot_sorted.index.tolist()
    x_labels = pivot_sorted.columns.tolist()

    # Custom colorscale: green → amber → red
    colorscale = [
        [0.0,  "#00C48C"],
        [0.4,  "#1E3A5F"],
        [0.7,  "#F5A623"],
        [1.0,  "#E8384F"],
    ]

    fig = go.Figure(go.Heatmap(
        z=z_vals,
        x=x_labels,
        y=y_labels,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(
            title="% of Accounts",
            title_font=dict(color=COLORS["text_dim"], size=11),
            tickfont=dict(color=COLORS["text_dim"], size=10),
            thickness=12,
            len=0.8,
        ),
        text=[[f"{v:.0f}%" for v in row] for row in z_vals],
        texttemplate="%{text}",
        textfont=dict(size=10, color="white"),
        hovertemplate=(
            "Industry: <b>%{y}</b><br>"
            "Risk: <b>%{x}</b><br>"
            "Share: <b>%{z:.1f}%</b>"
            "<extra></extra>"
        ),
    ))

    _apply_layout(fig, height=400)
    fig.update_xaxes(**AXIS_STYLE_NOGRID, title="")
    fig.update_yaxes(**AXIS_STYLE_NOGRID, title="")
    return fig


# ── 9. Rep Performance Scatter ────────────────────────────────────────────
# Business question: Across all reps, who has the right combination of
# pipeline volume AND win rate? Surfaces coaching priority.
def rep_performance_scatter(df: pd.DataFrame) -> go.Figure:
    closed = df[df["Win/Loss"].isin(["Won", "Lost"])]
    rep_df = (
        closed.groupby("Sales Rep")
        .apply(lambda g: pd.Series({
            "Win Rate": (g["Win/Loss"] == "Won").sum() / len(g) * 100,
            "Deals Closed": len(g),
        }), include_groups=False)
        .reset_index()
    )
    active = df[df["Stage"].isin(ACTIVE_STAGES)]
    open_arr = active.groupby("Sales Rep")["ARR"].sum().reset_index()
    open_arr.columns = ["Sales Rep", "Open ARR"]
    rep_df = rep_df.merge(open_arr, on="Sales Rep", how="left").fillna(0)

    avg_wr  = rep_df["Win Rate"].mean()
    avg_arr = rep_df["Open ARR"].mean()

    fig = go.Figure()

    # Quadrant lines
    fig.add_hline(y=avg_wr,  line_dash="dot", line_color=COLORS["border"], line_width=1)
    fig.add_vline(x=avg_arr, line_dash="dot", line_color=COLORS["border"], line_width=1)

    fig.add_trace(go.Scatter(
        x=rep_df["Open ARR"],
        y=rep_df["Win Rate"],
        mode="markers+text",
        marker=dict(
            size=rep_df["Deals Closed"] * 4 + 12,
            color=rep_df["Win Rate"],
            colorscale=[[0, COLORS["danger"]], [0.5, COLORS["warning"]], [1, COLORS["success"]]],
            cmin=0, cmax=100,
            showscale=False,
            line=dict(width=1, color=COLORS["border"]),
        ),
        text=rep_df["Sales Rep"].str.split().str[-1],  # Last name only
        textposition="top center",
        textfont=dict(size=9, color=COLORS["text_dim"]),
        customdata=np.stack([rep_df["Deals Closed"], rep_df["Sales Rep"]], axis=-1),
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Win Rate: <b>%{y:.1f}%</b><br>"
            "Open Pipeline ARR: <b>$%{x:,.0f}</b><br>"
            "Deals Closed: <b>%{customdata[0]}</b>"
            "<extra></extra>"
        ),
    ))

    # Quadrant labels
    for text, x, y, anchor in [
        ("High Volume · High Win Rate",  avg_arr * 1.05, avg_wr + 2,  "left"),
        ("Low Volume · High Win Rate",   avg_arr * 0.95, avg_wr + 2,  "right"),
        ("High Volume · Low Win Rate",   avg_arr * 1.05, avg_wr - 2,  "left"),
        ("Low Volume · Low Win Rate",    avg_arr * 0.95, avg_wr - 2,  "right"),
    ]:
        fig.add_annotation(
            x=x, y=y, text=text,
            showarrow=False,
            font=dict(size=8, color=COLORS["text_dim"]),
            xanchor=anchor,
        )

    _apply_layout(fig, height=360)
    fig.update_xaxes(**AXIS_STYLE, title="Open Pipeline ARR ($)")
    fig.update_yaxes(**AXIS_STYLE, title="Win Rate (%)")
    return fig


# ── 10. Objection Category Breakdown ─────────────────────────────────────
# Business question: What are reps losing or stalling on?
# Input for Sales Coaching Agent enablement library.
def objection_breakdown(df: pd.DataFrame) -> go.Figure:
    obj_df = (
        df.groupby("Objection Category")
        .agg(Count=("Opportunity ID", "count"), ARR=("ARR", "sum"))
        .reset_index()
        .sort_values("Count", ascending=True)
    )
    obj_df["ARR_Label"] = obj_df["ARR"].apply(
        lambda x: f"${x/1_000_000:.1f}M"
    )

    fig = go.Figure(go.Bar(
        x=obj_df["Count"],
        y=obj_df["Objection Category"],
        orientation="h",
        marker_color=COLORS["primary"],
        marker_line_width=0,
        opacity=0.85,
        customdata=np.stack([obj_df["ARR_Label"]], axis=-1),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Deals: <b>%{x}</b><br>"
            "ARR Exposure: <b>%{customdata[0]}</b>"
            "<extra></extra>"
        ),
        text=obj_df["Count"],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["text_dim"]),
    ))

    _apply_layout(fig, height=340)
    fig.update_xaxes(**AXIS_STYLE, title="Number of Deals")
    fig.update_yaxes(**AXIS_STYLE_NOGRID, title="")
    fig.update_layout(showlegend=False)
    return fig
