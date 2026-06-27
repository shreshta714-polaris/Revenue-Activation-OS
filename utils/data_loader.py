"""
utils/data_loader.py
--------------------
Central data loading, cleaning, and business logic layer.
All transformations live here so pages stay thin and readable.

Future AI agents will extend this module by adding their own
computed columns or calling into agent modules in /agents/.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent.parent / "data" / "pipeline_data.csv"

STAGE_ORDER = [
    "Prospecting",
    "Discovery",
    "Qualification",
    "Proposal",
    "Negotiation",
    "Closed Won",
    "Closed Lost",
]

ACTIVE_STAGES = [
    "Prospecting", "Discovery", "Qualification", "Proposal", "Negotiation"
]

# Risk thresholds — single source of truth for all pages
RISK_HIGH_THRESHOLD   = 50   # Renewal Probability < 50 → High Risk
RISK_MEDIUM_THRESHOLD = 70   # 50–70 → Medium Risk; >70 → Low Risk

# Color palette — mirrors Streamlit theme tokens
COLORS = {
    "primary":    "#2E86FF",
    "success":    "#00C48C",
    "warning":    "#F5A623",
    "danger":     "#E8384F",
    "neutral":    "#8A94A6",
    "background": "#0B1120",
    "surface":    "#111827",
    "border":     "#1E2A3B",
    "text":       "#E2E8F0",
    "text_dim":   "#64748B",
}

STAGE_COLORS = {
    "Prospecting":   "#4A90D9",
    "Discovery":     "#2E86FF",
    "Qualification": "#7B61FF",
    "Proposal":      "#F5A623",
    "Negotiation":   "#FF7A45",
    "Closed Won":    "#00C48C",
    "Closed Lost":   "#E8384F",
}

RISK_COLORS = {
    "High Risk":    COLORS["danger"],
    "Medium Risk":  COLORS["warning"],
    "Low Risk":     COLORS["success"],
}


# ── Core loader ────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """Load and enrich the pipeline dataset. Returns a clean DataFrame."""
    df = pd.read_csv(DATA_PATH)
    df = _clean(df)
    df = _enrich(df)
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column types and handle edge cases."""
    df["ARR"] = pd.to_numeric(df["ARR"], errors="coerce").fillna(0)
    df["Days in Stage"] = pd.to_numeric(df["Days in Stage"], errors="coerce").fillna(0)
    df["Product Adoption Score"] = pd.to_numeric(df["Product Adoption Score"], errors="coerce").fillna(0)
    df["Customer Health Score"] = pd.to_numeric(df["Customer Health Score"], errors="coerce").fillna(0)
    df["Renewal Probability"] = pd.to_numeric(df["Renewal Probability"], errors="coerce").fillna(0)

    # Enforce stage ordering for charts
    df["Stage"] = pd.Categorical(df["Stage"], categories=STAGE_ORDER, ordered=True)

    return df


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add computed business logic columns used across all pages."""

    # ── Churn / renewal risk classification ──────────────────────────────
    df["Risk Tier"] = pd.cut(
        df["Renewal Probability"],
        bins=[-1, RISK_HIGH_THRESHOLD, RISK_MEDIUM_THRESHOLD, 101],
        labels=["High Risk", "Medium Risk", "Low Risk"],
    )

    # ── Composite Account Health Score ────────────────────────────────────
    # Blended metric surfaced in the Executive Summary
    # 40% renewal probability + 35% customer health + 25% product adoption
    df["Composite Health"] = (
        df["Renewal Probability"] * 0.40
        + df["Customer Health Score"] * 0.35
        + df["Product Adoption Score"] * 0.25
    ).round(1)

    # ── Deal urgency flags ────────────────────────────────────────────────
    # Stale = active deal with days-in-stage exceeding 1.5x median for that stage
    active = df["Stage"].isin(ACTIVE_STAGES)
    stage_medians = df[active].groupby("Stage", observed=True)["Days in Stage"].median()
    df["Stage Median Days"] = df["Stage"].map(stage_medians)
    df["Is Stale"] = active & (df["Days in Stage"] > df["Stage Median Days"] * 1.5)

    # ── ARR risk buckets ─────────────────────────────────────────────────
    df["ARR Band"] = pd.cut(
        df["ARR"],
        bins=[0, 50_000, 200_000, 500_000, float("inf")],
        labels=["<$50K", "$50K–$200K", "$200K–$500K", ">$500K"],
    )

    return df


# ── KPI helpers ────────────────────────────────────────────────────────────
def get_kpis(df: pd.DataFrame) -> dict:
    """Return top-line KPIs for the KPI card row."""
    closed = df[df["Win/Loss"].isin(["Won", "Lost"])]
    win_rate = (
        len(df[df["Win/Loss"] == "Won"]) / len(closed) * 100
        if len(closed) > 0 else 0
    )
    active = df[df["Stage"].isin(ACTIVE_STAGES)]
    return {
        "total_arr":        df["ARR"].sum(),
        "active_pipeline":  active["ARR"].sum(),
        "win_rate":         win_rate,
        "avg_adoption":     df["Product Adoption Score"].mean(),
        "avg_health":       df["Customer Health Score"].mean(),
        "avg_renewal":      df["Renewal Probability"].mean(),
        "high_risk_count":  int((df["Risk Tier"] == "High Risk").sum()),
        "stale_deals":      int(df["Is Stale"].sum()),
        "total_opps":       len(df),
    }


# ── Executive summary helpers ──────────────────────────────────────────────
def get_executive_summary(df: pd.DataFrame) -> dict:
    """
    Derive the insights powering the Executive Summary section.
    Returns a dict of named findings ready for display.

    Future: Agent 04 (Revenue Strategy Consultant) will replace this
    function with AI-generated narrative and cross-agent synthesis.
    """
    kpis = get_kpis(df)

    # Pipeline bottleneck — stage with highest avg days (active only)
    active = df[df["Stage"].isin(ACTIVE_STAGES)]
    stage_days = active.groupby("Stage", observed=True)["Days in Stage"].mean()
    bottleneck_stage = stage_days.idxmax()
    bottleneck_days  = stage_days.max()

    # Highest risk industry — by % of accounts that are High Risk
    industry_risk = (
        df.groupby("Industry")
        .apply(lambda g: (g["Risk Tier"] == "High Risk").mean() * 100, include_groups=False)
        .round(1)
    )
    riskiest_industry      = industry_risk.idxmax()
    riskiest_industry_pct  = industry_risk.max()

    # Rep with lowest win rate (min 5 closed deals)
    closed = df[df["Win/Loss"].isin(["Won", "Lost"])]
    rep_wins = closed.groupby("Sales Rep").apply(
        lambda g: (g["Win/Loss"] == "Won").sum() / len(g) * 100 if len(g) >= 5 else np.nan,
        include_groups=False
    ).dropna()
    lowest_rep      = rep_wins.idxmin() if not rep_wins.empty else "N/A"
    lowest_rep_wr   = rep_wins.min()    if not rep_wins.empty else 0

    # Top expansion candidate industry — high adoption + high health
    exp_signal = (
        df[(df["Product Adoption Score"] > 70) & (df["Customer Health Score"] > 70)]
        .groupby("Industry")
        .size()
    )
    expansion_industry = exp_signal.idxmax() if not exp_signal.empty else "N/A"

    # ARR at churn risk
    arr_at_risk = df[df["Risk Tier"] == "High Risk"]["ARR"].sum()

    # Key revenue insight — highest conviction single finding
    stale_arr = df[df["Is Stale"]]["ARR"].sum()

    return {
        "total_pipeline_arr":   kpis["active_pipeline"],
        "win_rate":             kpis["win_rate"],
        "bottleneck_stage":     bottleneck_stage,
        "bottleneck_days":      bottleneck_days,
        "riskiest_industry":    riskiest_industry,
        "riskiest_industry_pct": riskiest_industry_pct,
        "high_risk_count":      kpis["high_risk_count"],
        "arr_at_risk":          arr_at_risk,
        "lowest_win_rate_rep":  lowest_rep,
        "lowest_win_rate_pct":  lowest_rep_wr,
        "expansion_industry":   expansion_industry,
        "stale_deals":          kpis["stale_deals"],
        "stale_arr":            stale_arr,
        "key_insight": (
            f"{kpis['stale_deals']} active deals are stalled past their stage benchmark, "
            f"putting ${stale_arr:,.0f} ARR at risk of slipping. "
            f"{riskiest_industry} has the highest churn concentration at "
            f"{riskiest_industry_pct:.0f}% of accounts in the High Risk tier."
        ),
    }


# ── Filtering helpers ──────────────────────────────────────────────────────
def filter_df(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply sidebar filter dict to dataframe. Used by all pages."""
    if filters.get("stages"):
        df = df[df["Stage"].isin(filters["stages"])]
    if filters.get("industries"):
        df = df[df["Industry"].isin(filters["industries"])]
    if filters.get("reps"):
        df = df[df["Sales Rep"].isin(filters["reps"])]
    if filters.get("risk_tiers"):
        df = df[df["Risk Tier"].isin(filters["risk_tiers"])]
    if filters.get("deal_sizes"):
        df = df[df["Deal Size"].isin(filters["deal_sizes"])]
    return df
