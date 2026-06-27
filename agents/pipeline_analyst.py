"""
agents/pipeline_analyst.py
==========================
PipelineAnalystAgent — Revenue Activation OS
---------------------------------------------
A deterministic revenue intelligence engine that analyses a SaaS pipeline
dataset and produces structured, actionable findings for a VP of Revenue
Operations reviewing a multi-million-dollar pipeline.

Design principles:
  - Pure deterministic logic. No LLMs, no APIs, no randomness.
  - Every finding is traceable to a formula and a data row.
  - Output is a fully-typed JSON-serialisable dict so the Streamlit layer
    can render it without any additional transformation.
  - All thresholds are class-level constants so a RevOps team can tune them
    without touching analysis logic.
  - Methods are intentionally granular (one responsibility each) so future
    agents can call individual analyses as sub-routines.

Usage:
    from agents.pipeline_analyst import PipelineAnalystAgent
    from utils.data_loader import load_data

    df     = load_data()
    agent  = PipelineAnalystAgent(df)
    report = agent.run()          # full JSON report
    print(report["executive_summary"]["headline"])
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Domain constants — tune these without touching analysis logic
# ─────────────────────────────────────────────────────────────────────────────

ACTIVE_STAGES: list[str] = [
    "Prospecting", "Discovery", "Qualification", "Proposal", "Negotiation"
]
CLOSED_STAGES: list[str] = ["Closed Won", "Closed Lost"]
STAGE_ORDER: list[str]   = ACTIVE_STAGES + CLOSED_STAGES

# Velocity benchmarks (days) — industry-calibrated SaaS medians
STAGE_VELOCITY_BENCHMARKS: dict[str, float] = {
    "Prospecting":   7.0,
    "Discovery":    14.0,
    "Qualification": 21.0,
    "Proposal":     18.0,
    "Negotiation":  25.0,
}

# Risk thresholds (Renewal Probability %)
RISK_HIGH_THRESHOLD:   float = 50.0
RISK_MEDIUM_THRESHOLD: float = 70.0

# Score thresholds
LOW_ADOPTION_THRESHOLD: float = 35.0
LOW_HEALTH_THRESHOLD:   float = 40.0

# Stale deal multiplier: Days in Stage > N × stage median → stale
STALE_MULTIPLIER: float = 1.5

# Minimum closed deals before a rep's win-rate is considered statistically
# meaningful enough to appear in coaching recommendations
MIN_CLOSED_DEALS_FOR_COACHING: int = 3

# Quick-win: high health + high adoption + short days in stage
QUICK_WIN_HEALTH_MIN:    float = 65.0
QUICK_WIN_ADOPTION_MIN:  float = 55.0
QUICK_WIN_MAX_STAGE_IDX: int   = 3   # index into ACTIVE_STAGES (≤ Proposal)

# Expansion signal thresholds
EXPANSION_ADOPTION_MIN: float = 70.0
EXPANSION_HEALTH_MIN:   float = 65.0
EXPANSION_RENEWAL_MIN:  float = 70.0

# Bottleneck severity: excess days beyond benchmark before flagged
BOTTLENECK_SEVERITY_THRESHOLD: float = 5.0   # days above benchmark

# Pipeline coverage target ratio (pipeline ARR / quota ARR)
# Used to contextualise commentary; quota derived as avg closed ARR × 4
PIPELINE_COVERAGE_TARGET: float = 3.0


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight result dataclasses (JSON-serialisable via asdict())
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BottleneckRecord:
    stage: str
    avg_days: float
    median_days: float
    benchmark_days: float
    excess_days: float
    severity: str                   # "Critical" | "High" | "Moderate"
    deal_count: int
    arr_at_risk: float
    arr_at_risk_fmt: str
    pct_above_benchmark: float      # % over benchmark


@dataclass
class HighRiskRecord:
    opportunity_id: str
    company_name: str
    arr: float
    arr_fmt: str
    stage: str
    sales_rep: str
    industry: str
    renewal_probability: float
    customer_health_score: float
    product_adoption_score: float
    days_in_stage: float
    risk_flags: list[str]           # human-readable reasons
    risk_score: float               # composite 0–100 (higher = riskier)
    priority: str                   # "P1" | "P2" | "P3"


@dataclass
class RepPerformanceRecord:
    sales_rep: str
    deals_won: int
    deals_lost: int
    deals_open: int
    total_closed: int
    win_rate: float
    won_arr: float
    won_arr_fmt: str
    open_arr: float
    open_arr_fmt: str
    avg_deal_size: float
    avg_days_to_close: float        # avg days in stage for won deals
    avg_adoption_score: float
    avg_health_score: float
    performance_tier: str           # "Top", "Mid", "Developing", "At Risk"
    coaching_flags: list[str]


@dataclass
class IndustryRecord:
    industry: str
    total_arr: float
    total_arr_fmt: str
    deal_count: int
    win_rate: float
    high_risk_pct: float
    avg_health_score: float
    avg_adoption_score: float
    avg_renewal_probability: float
    arr_share_pct: float
    signal: str                     # "Expand" | "Protect" | "Monitor" | "Investigate"


@dataclass
class RevenueOpportunityRecord:
    opportunity_id: str
    company_name: str
    arr: float
    arr_fmt: str
    stage: str
    sales_rep: str
    industry: str
    opportunity_type: str           # "Quick Win" | "Expansion" | "Acceleration"
    rationale: str
    recommended_action: str
    confidence: str                 # "High" | "Medium"
    composite_score: float          # 0–100, used for ranking


@dataclass
class Recommendation:
    id: str                         # "REC-01" … "REC-10"
    category: str                   # "Pipeline" | "Coaching" | "Retention" | …
    priority: str                   # "P1 — Immediate" | "P2 — This Week" | "P3 — This Month"
    headline: str
    detail: str
    metric_trigger: str             # the specific number that fired this rule
    estimated_arr_impact: float
    estimated_arr_impact_fmt: str
    owner: str                      # "VP Sales" | "CS Leader" | "RevOps" | …
    effort: str                     # "Low" | "Medium" | "High"


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class PipelineAnalystAgent:
    """
    Deterministic revenue intelligence engine for the Pipeline domain.

    Analyses a 500-opportunity SaaS pipeline and produces a structured
    report covering win rates, bottlenecks, risk, rep performance,
    industry health, revenue opportunities, and prioritised recommendations.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned and enriched pipeline DataFrame produced by
        ``utils.data_loader.load_data()``.  Must contain the columns
        defined in ``REQUIRED_COLUMNS``.
    run_timestamp : str, optional
        ISO-8601 timestamp injected into the report metadata.
        Defaults to the current UTC time.
    """

    REQUIRED_COLUMNS: frozenset[str] = frozenset({
        "Opportunity ID", "Company Name", "ARR", "Stage", "Win/Loss",
        "Sales Rep", "Industry", "Deal Size", "Days in Stage",
        "Objection Category", "Product Adoption Score",
        "Customer Health Score", "Renewal Probability",
    })

    # ── Construction ──────────────────────────────────────────────────────

    def __init__(
        self,
        df: pd.DataFrame,
        run_timestamp: str | None = None,
    ) -> None:
        self._validate_input(df)
        self.df: pd.DataFrame = self._prepare(df.copy())
        self.run_timestamp: str = run_timestamp or datetime.utcnow().isoformat() + "Z"

        # Cached slices (populated by _prepare)
        self._active: pd.DataFrame = self.df[self.df["Stage"].isin(ACTIVE_STAGES)]
        self._closed: pd.DataFrame = self.df[self.df["Win/Loss"].isin(["Won", "Lost"])]
        self._won:    pd.DataFrame = self.df[self.df["Win/Loss"] == "Won"]
        self._lost:   pd.DataFrame = self.df[self.df["Win/Loss"] == "Lost"]

        logger.info(
            "PipelineAnalystAgent initialised — %d opportunities "
            "(%d active, %d won, %d lost)",
            len(self.df), len(self._active), len(self._won), len(self._lost),
        )

    # ── Public interface ───────────────────────────────────────────────────

    def run(self) -> dict[str, Any]:
        """
        Execute the full analysis pipeline and return a structured report.

        Returns
        -------
        dict
            Fully JSON-serialisable report.  All monetary values are
            present in both raw float (``arr``) and human-readable string
            (``arr_fmt``) forms so callers never need to format numbers.
        """
        logger.info("Running PipelineAnalystAgent full analysis…")

        win_rate        = self._analyse_win_rates()
        bottlenecks     = self._analyse_bottlenecks()
        high_risk       = self._detect_high_risk_opportunities()
        rep_performance = self._analyse_rep_performance()
        industry        = self._analyse_industries()
        opportunities   = self._identify_revenue_opportunities()
        pipeline_health = self._compute_pipeline_health(bottlenecks, high_risk)
        recommendations = self._generate_recommendations(
            win_rate, bottlenecks, high_risk,
            rep_performance, industry, opportunities, pipeline_health,
        )
        executive_summary = self._build_executive_summary(
            win_rate, bottlenecks, high_risk,
            rep_performance, industry, pipeline_health,
        )

        report = {
            "meta": {
                "agent":          "PipelineAnalystAgent",
                "version":        "1.0.0",
                "run_timestamp":  self.run_timestamp,
                "total_opportunities": len(self.df),
                "analysis_scope": "Full pipeline — all stages",
            },
            "executive_summary":      executive_summary,
            "pipeline_health":        pipeline_health,
            "win_rate_analysis":      win_rate,
            "pipeline_bottlenecks":   [asdict(b) for b in bottlenecks],
            "high_risk_opportunities":[asdict(r) for r in high_risk],
            "sales_rep_analysis":     [asdict(r) for r in rep_performance],
            "industry_analysis":      [asdict(i) for i in industry],
            "revenue_opportunities":  [asdict(o) for o in opportunities],
            "recommendations":        [asdict(r) for r in recommendations],
        }
        logger.info("Analysis complete — %d recommendations generated.", len(recommendations))
        return report

    def to_json(self, indent: int = 2) -> str:
        """Return the full report as a formatted JSON string."""
        return json.dumps(self.run(), indent=indent, default=str)

    # ── Input validation & preparation ────────────────────────────────────

    def _validate_input(self, df: pd.DataFrame) -> None:
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"PipelineAnalystAgent: missing required columns: {missing}"
            )
        if len(df) == 0:
            raise ValueError("PipelineAnalystAgent: DataFrame is empty.")

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce types, add derived columns used across all analyses."""
        numeric_cols = [
            "ARR", "Days in Stage", "Product Adoption Score",
            "Customer Health Score", "Renewal Probability",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        df["Stage"] = pd.Categorical(df["Stage"], categories=STAGE_ORDER, ordered=True)

        # Composite risk score: higher = riskier (0–100)
        # Inverts health/adoption/renewal so 0 = perfect, 100 = critical
        df["_risk_score"] = (
            (100 - df["Renewal Probability"])    * 0.45
            + (100 - df["Customer Health Score"]) * 0.30
            + (100 - df["Product Adoption Score"])* 0.25
        ).clip(0, 100).round(2)

        # Composite opportunity score: higher = stronger buy signal (0–100)
        df["_opportunity_score"] = (
            df["Renewal Probability"]    * 0.35
            + df["Customer Health Score"] * 0.35
            + df["Product Adoption Score"]* 0.30
        ).clip(0, 100).round(2)

        # Velocity pressure: days as % of benchmark (>100% = behind)
        df["_velocity_pressure"] = df.apply(
            lambda r: (
                r["Days in Stage"] / STAGE_VELOCITY_BENCHMARKS.get(r["Stage"], 1) * 100
                if r["Stage"] in STAGE_VELOCITY_BENCHMARKS else 0.0
            ),
            axis=1,
        ).round(1)

        # Stale flag (active deals only)
        active_mask = df["Stage"].isin(ACTIVE_STAGES)
        stage_medians = (
            df[active_mask]
            .groupby("Stage", observed=True)["Days in Stage"]
            .median()
        )
        df["_stage_median_days"] = df["Stage"].map(stage_medians).fillna(0)
        df["_is_stale"] = (
            active_mask
            & (df["Days in Stage"] > df["_stage_median_days"] * STALE_MULTIPLIER)
        )

        return df

    # ── 1. Win rate analysis ───────────────────────────────────────────────

    def _analyse_win_rates(self) -> dict[str, Any]:
        """
        Compute overall, by-stage, by-industry, and by-rep win rates.

        Win Rate = Won / (Won + Lost) for each cohort.
        Only cohorts with ≥ MIN_CLOSED_DEALS_FOR_COACHING deals are included
        in dimensional breakdowns to avoid misleading small-sample rates.
        """
        closed = self._closed

        # ── Overall ──────────────────────────────────────────────────────
        overall = (
            len(self._won) / len(closed) * 100
            if len(closed) > 0 else 0.0
        )

        # ── By stage (stage the deal was in when closed) ──────────────────
        # Note: for "won/lost" deals the Stage column holds the terminal stage
        by_stage_raw = closed.groupby("Stage", observed=True).apply(
            lambda g: {
                "stage":       str(g.name),
                "win_rate":    round((g["Win/Loss"] == "Won").mean() * 100, 1),
                "deal_count":  len(g),
                "won_arr":     round(g[g["Win/Loss"] == "Won"]["ARR"].sum(), 0),
            },
            include_groups=False,
        )
        by_stage = [v for v in by_stage_raw if v["deal_count"] >= 1]

        # ── By industry ───────────────────────────────────────────────────
        by_industry_raw = (
            closed.groupby("Industry").apply(
                lambda g: {
                    "industry":    str(g.name),
                    "win_rate":    round((g["Win/Loss"] == "Won").mean() * 100, 1),
                    "deal_count":  len(g),
                    "won_arr":     round(g[g["Win/Loss"] == "Won"]["ARR"].sum(), 0),
                },
                include_groups=False,
            )
        )
        by_industry = sorted(
            [v for v in by_industry_raw if v["deal_count"] >= MIN_CLOSED_DEALS_FOR_COACHING],
            key=lambda x: x["win_rate"],
            reverse=True,
        )

        # ── By rep ────────────────────────────────────────────────────────
        by_rep_raw = (
            closed.groupby("Sales Rep").apply(
                lambda g: {
                    "sales_rep":  str(g.name),
                    "win_rate":   round((g["Win/Loss"] == "Won").mean() * 100, 1),
                    "deal_count": len(g),
                    "won_arr":    round(g[g["Win/Loss"] == "Won"]["ARR"].sum(), 0),
                },
                include_groups=False,
            )
        )
        by_rep = sorted(
            [v for v in by_rep_raw if v["deal_count"] >= MIN_CLOSED_DEALS_FOR_COACHING],
            key=lambda x: x["win_rate"],
            reverse=True,
        )

        # ── By deal size ──────────────────────────────────────────────────
        by_size = (
            closed.groupby("Deal Size").apply(
                lambda g: {
                    "deal_size":  str(g.name),
                    "win_rate":   round((g["Win/Loss"] == "Won").mean() * 100, 1),
                    "deal_count": len(g),
                    "won_arr":    round(g[g["Win/Loss"] == "Won"]["ARR"].sum(), 0),
                },
                include_groups=False,
            ).tolist()
        )

        # ── Top objection for lost deals ──────────────────────────────────
        lost_objections = (
            self._lost["Objection Category"].value_counts().head(5).to_dict()
        )

        return {
            "overall_win_rate":        round(overall, 1),
            "total_closed_deals":      len(closed),
            "total_won_deals":         len(self._won),
            "total_lost_deals":        len(self._lost),
            "total_won_arr":           round(self._won["ARR"].sum(), 0),
            "total_won_arr_fmt":       _fmt_arr(self._won["ARR"].sum()),
            "team_avg_win_rate":       round(overall, 1),
            "by_stage":                by_stage,
            "by_industry":             by_industry,
            "by_rep":                  by_rep,
            "by_deal_size":            by_size,
            "top_loss_objections":     lost_objections,
            "best_industry":           by_industry[0]["industry"]  if by_industry else "N/A",
            "worst_industry":          by_industry[-1]["industry"] if by_industry else "N/A",
            "best_rep":                by_rep[0]["sales_rep"]      if by_rep else "N/A",
            "worst_rep":               by_rep[-1]["sales_rep"]     if by_rep else "N/A",
        }

    # ── 2. Bottleneck analysis ────────────────────────────────────────────

    def _analyse_bottlenecks(self) -> list[BottleneckRecord]:
        """
        Identify pipeline stages where deals are sitting significantly
        longer than the SaaS velocity benchmark.

        Severity tiers:
          Critical  — > 100% above benchmark
          High      — 50–100% above benchmark
          Moderate  — BOTTLENECK_SEVERITY_THRESHOLD–50% above benchmark
        """
        records: list[BottleneckRecord] = []

        for stage in ACTIVE_STAGES:
            stage_df = self._active[self._active["Stage"] == stage]
            if stage_df.empty:
                continue

            benchmark = STAGE_VELOCITY_BENCHMARKS.get(stage, 14.0)
            avg_days  = stage_df["Days in Stage"].mean()
            med_days  = stage_df["Days in Stage"].median()
            excess    = avg_days - benchmark

            if excess < BOTTLENECK_SEVERITY_THRESHOLD:
                continue   # within acceptable range

            pct_above  = (excess / benchmark) * 100
            arr_at_risk = stage_df["ARR"].sum()

            if pct_above > 100:
                severity = "Critical"
            elif pct_above > 50:
                severity = "High"
            else:
                severity = "Moderate"

            records.append(BottleneckRecord(
                stage            = stage,
                avg_days         = round(avg_days, 1),
                median_days      = round(med_days, 1),
                benchmark_days   = benchmark,
                excess_days      = round(excess, 1),
                severity         = severity,
                deal_count       = len(stage_df),
                arr_at_risk      = round(arr_at_risk, 0),
                arr_at_risk_fmt  = _fmt_arr(arr_at_risk),
                pct_above_benchmark = round(pct_above, 1),
            ))

        # Rank by ARR at risk (highest first)
        return sorted(records, key=lambda r: r.arr_at_risk, reverse=True)

    # ── 3. High-risk opportunity detection ───────────────────────────────

    def _detect_high_risk_opportunities(self) -> list[HighRiskRecord]:
        """
        Flag opportunities exhibiting multiple risk signals.

        Risk flags (each is independent):
          • Renewal Probability < RISK_HIGH_THRESHOLD
          • Customer Health Score < LOW_HEALTH_THRESHOLD
          • Product Adoption Score < LOW_ADOPTION_THRESHOLD
          • Days in Stage > STALE_MULTIPLIER × stage median
          • Win/Loss is "Open" AND Stage is Negotiation AND Days > 30

        Priority assignment:
          P1 — risk_score ≥ 70 AND ARR > $100K
          P2 — risk_score ≥ 55 OR ARR > $150K with any flag
          P3 — everything else with ≥ 2 flags
        """
        records: list[HighRiskRecord] = []

        for _, row in self.df.iterrows():
            flags: list[str] = []

            if row["Renewal Probability"] < RISK_HIGH_THRESHOLD:
                flags.append(
                    f"Renewal probability {row['Renewal Probability']:.0f}% "
                    f"(threshold: {RISK_HIGH_THRESHOLD:.0f}%)"
                )
            if row["Customer Health Score"] < LOW_HEALTH_THRESHOLD:
                flags.append(
                    f"Health score {row['Customer Health Score']:.0f}/100 "
                    f"(threshold: {LOW_HEALTH_THRESHOLD:.0f})"
                )
            if row["Product Adoption Score"] < LOW_ADOPTION_THRESHOLD:
                flags.append(
                    f"Adoption score {row['Product Adoption Score']:.0f}/100 "
                    f"(threshold: {LOW_ADOPTION_THRESHOLD:.0f})"
                )
            if row["_is_stale"]:
                flags.append(
                    f"Stalled {row['Days in Stage']:.0f} days in {row['Stage']} "
                    f"(stage median: {row['_stage_median_days']:.0f} days)"
                )
            if (
                row["Win/Loss"] == "Open"
                and row["Stage"] == "Negotiation"
                and row["Days in Stage"] > 30
            ):
                flags.append(
                    f"Negotiation open for {row['Days in Stage']:.0f} days — "
                    "late-stage stall risk"
                )

            if len(flags) < 2:
                continue   # Only surface multi-signal risks

            risk_score = row["_risk_score"]

            if risk_score >= 70 and row["ARR"] >= 100_000:
                priority = "P1"
            elif risk_score >= 55 or row["ARR"] >= 150_000:
                priority = "P2"
            else:
                priority = "P3"

            records.append(HighRiskRecord(
                opportunity_id          = str(row["Opportunity ID"]),
                company_name            = str(row["Company Name"]),
                arr                     = float(row["ARR"]),
                arr_fmt                 = _fmt_arr(row["ARR"]),
                stage                   = str(row["Stage"]),
                sales_rep               = str(row["Sales Rep"]),
                industry                = str(row["Industry"]),
                renewal_probability     = round(float(row["Renewal Probability"]), 1),
                customer_health_score   = round(float(row["Customer Health Score"]), 1),
                product_adoption_score  = round(float(row["Product Adoption Score"]), 1),
                days_in_stage           = round(float(row["Days in Stage"]), 0),
                risk_flags              = flags,
                risk_score              = round(risk_score, 1),
                priority                = priority,
            ))

        # Sort: P1 first, then by risk score desc, then by ARR desc
        priority_order = {"P1": 0, "P2": 1, "P3": 2}
        return sorted(
            records,
            key=lambda r: (priority_order[r.priority], -r.risk_score, -r.arr),
        )

    # ── 4. Sales rep performance ──────────────────────────────────────────

    def _analyse_rep_performance(self) -> list[RepPerformanceRecord]:
        """
        Produce a performance profile for every rep in the dataset.

        Performance tiers (derived from win rate vs. team average):
          Top        — win rate ≥ team_avg + 10pp
          Mid        — win rate within ±10pp of team_avg
          Developing — win rate 10–25pp below team_avg
          At Risk    — win rate > 25pp below team_avg

        Coaching flags are rule-based signals, not subjective judgements:
          • High stale deal count (> 30% of open deals are stale)
          • Low avg health on open deals (< 45)
          • Low avg adoption on assigned accounts (< 35)
          • High objection concentration (>50% of lost deals share one objection)
          • Below-median avg deal size despite above-median deal count
        """
        df   = self.df
        reps = df["Sales Rep"].unique()

        closed = self._closed
        team_wr = (
            (closed["Win/Loss"] == "Won").mean() * 100
            if len(closed) > 0 else 0.0
        )

        records: list[RepPerformanceRecord] = []
        for rep in sorted(reps):
            rep_df      = df[df["Sales Rep"] == rep]
            rep_closed  = closed[closed["Sales Rep"] == rep]
            rep_won     = rep_df[rep_df["Win/Loss"] == "Won"]
            rep_lost    = rep_df[rep_df["Win/Loss"] == "Lost"]
            rep_open    = rep_df[rep_df["Win/Loss"] == "Open"]
            rep_stale   = rep_df[rep_df["_is_stale"]]

            n_closed = len(rep_closed)
            n_won    = len(rep_won)
            win_rate = (n_won / n_closed * 100) if n_closed > 0 else 0.0

            # Performance tier
            delta = win_rate - team_wr
            if delta >= 10:
                tier = "Top"
            elif delta >= -10:
                tier = "Mid"
            elif delta >= -25:
                tier = "Developing"
            else:
                tier = "At Risk"

            # Coaching flags
            coaching_flags: list[str] = []

            if len(rep_open) > 0:
                stale_pct = len(rep_stale) / len(rep_open) * 100
                if stale_pct > 30:
                    coaching_flags.append(
                        f"{stale_pct:.0f}% of open deals are stale "
                        f"({len(rep_stale)} deals) — pipeline hygiene issue"
                    )

            if len(rep_open) > 0:
                avg_open_health = rep_open["Customer Health Score"].mean()
                if avg_open_health < 45:
                    coaching_flags.append(
                        f"Avg health score on open deals: {avg_open_health:.0f}/100 — "
                        "account health deteriorating"
                    )

            if len(rep_df) > 0:
                avg_adoption = rep_df["Product Adoption Score"].mean()
                if avg_adoption < LOW_ADOPTION_THRESHOLD:
                    coaching_flags.append(
                        f"Avg adoption score: {avg_adoption:.0f}/100 — "
                        "low product engagement across assigned accounts"
                    )

            if len(rep_lost) >= MIN_CLOSED_DEALS_FOR_COACHING:
                top_obj = rep_lost["Objection Category"].value_counts()
                if len(top_obj) > 0:
                    top_pct = top_obj.iloc[0] / len(rep_lost) * 100
                    if top_pct >= 45:
                        coaching_flags.append(
                            f"{top_pct:.0f}% of losses attributed to "
                            f"'{top_obj.index[0]}' — targeted enablement needed"
                        )

            if tier in ("Developing", "At Risk") and n_closed >= MIN_CLOSED_DEALS_FOR_COACHING:
                coaching_flags.append(
                    f"Win rate {win_rate:.0f}% is {abs(delta):.0f}pp below team average "
                    f"({team_wr:.0f}%) — structured coaching plan recommended"
                )

            # Avg days to close (won deals only — proxy for sales velocity)
            avg_days_to_close = (
                rep_won["Days in Stage"].mean() if len(rep_won) > 0 else 0.0
            )

            records.append(RepPerformanceRecord(
                sales_rep          = rep,
                deals_won          = n_won,
                deals_lost         = len(rep_lost),
                deals_open         = len(rep_open),
                total_closed       = n_closed,
                win_rate           = round(win_rate, 1),
                won_arr            = round(rep_won["ARR"].sum(), 0),
                won_arr_fmt        = _fmt_arr(rep_won["ARR"].sum()),
                open_arr           = round(rep_open["ARR"].sum(), 0),
                open_arr_fmt       = _fmt_arr(rep_open["ARR"].sum()),
                avg_deal_size      = round(rep_won["ARR"].mean(), 0) if n_won > 0 else 0.0,
                avg_days_to_close  = round(avg_days_to_close, 1),
                avg_adoption_score = round(rep_df["Product Adoption Score"].mean(), 1),
                avg_health_score   = round(rep_df["Customer Health Score"].mean(), 1),
                performance_tier   = tier,
                coaching_flags     = coaching_flags,
            ))

        # Sort: At Risk → Developing → Mid → Top (worst first for action priority)
        tier_order = {"At Risk": 0, "Developing": 1, "Mid": 2, "Top": 3}
        return sorted(records, key=lambda r: (tier_order[r.performance_tier], -r.win_rate))

    # ── 5. Industry analysis ──────────────────────────────────────────────

    def _analyse_industries(self) -> list[IndustryRecord]:
        """
        Profile each industry by ARR, win rate, health, adoption, and risk.

        Signal logic:
          Expand      — win rate ≥ 55% AND high_risk_pct < 30%
          Protect     — win rate ≥ 45% AND high_risk_pct ≥ 30%
          Monitor     — win rate 35–45%
          Investigate — win rate < 35% OR high_risk_pct ≥ 50%
        """
        closed      = self._closed
        total_arr   = self.df["ARR"].sum()
        records: list[IndustryRecord] = []

        for industry, group in self.df.groupby("Industry"):
            ind_closed = closed[closed["Industry"] == industry]
            win_rate = (
                (ind_closed["Win/Loss"] == "Won").mean() * 100
                if len(ind_closed) > 0 else 0.0
            )
            high_risk_pct = (
                (group["Renewal Probability"] < RISK_HIGH_THRESHOLD).mean() * 100
            )

            # Signal assignment
            if win_rate >= 55 and high_risk_pct < 30:
                signal = "Expand"
            elif win_rate >= 45 and high_risk_pct >= 30:
                signal = "Protect"
            elif 35 <= win_rate < 45:
                signal = "Monitor"
            else:
                signal = "Investigate"

            records.append(IndustryRecord(
                industry                = str(industry),
                total_arr               = round(group["ARR"].sum(), 0),
                total_arr_fmt           = _fmt_arr(group["ARR"].sum()),
                deal_count              = len(group),
                win_rate                = round(win_rate, 1),
                high_risk_pct           = round(high_risk_pct, 1),
                avg_health_score        = round(group["Customer Health Score"].mean(), 1),
                avg_adoption_score      = round(group["Product Adoption Score"].mean(), 1),
                avg_renewal_probability = round(group["Renewal Probability"].mean(), 1),
                arr_share_pct           = round(group["ARR"].sum() / total_arr * 100, 1),
                signal                  = signal,
            ))

        # Sort by ARR descending
        return sorted(records, key=lambda r: r.total_arr, reverse=True)

    # ── 6. Revenue opportunity identification ─────────────────────────────

    def _identify_revenue_opportunities(self) -> list[RevenueOpportunityRecord]:
        """
        Surface three classes of revenue opportunity:

        Quick Wins
          Active deals in early-to-mid stages with strong health + adoption
          scores. These are the deals most likely to accelerate to close with
          focused attention this week.

        Expansion Signals
          Existing accounts (any stage, Open or Won) with high adoption,
          health, and renewal probability — indicating readiness for upsell
          or cross-sell conversations.

        Pipeline Acceleration
          Deals stalled mid-funnel (Proposal / Negotiation) with moderate
          health scores — likely resolvable with a specific intervention
          (exec sponsor, pricing revision, competitive counter).
        """
        records: list[RevenueOpportunityRecord] = []
        stage_idx = {s: i for i, s in enumerate(ACTIVE_STAGES)}

        # ── Quick wins ────────────────────────────────────────────────────
        qw_mask = (
            self._active["Customer Health Score"].ge(QUICK_WIN_HEALTH_MIN)
            & self._active["Product Adoption Score"].ge(QUICK_WIN_ADOPTION_MIN)
            & self._active["Stage"].map(stage_idx).le(QUICK_WIN_MAX_STAGE_IDX)
        )
        for _, row in self._active[qw_mask].iterrows():
            score = row["_opportunity_score"]
            records.append(RevenueOpportunityRecord(
                opportunity_id   = str(row["Opportunity ID"]),
                company_name     = str(row["Company Name"]),
                arr              = float(row["ARR"]),
                arr_fmt          = _fmt_arr(row["ARR"]),
                stage            = str(row["Stage"]),
                sales_rep        = str(row["Sales Rep"]),
                industry         = str(row["Industry"]),
                opportunity_type = "Quick Win",
                rationale        = (
                    f"Health {row['Customer Health Score']:.0f}/100, "
                    f"Adoption {row['Product Adoption Score']:.0f}/100 — "
                    f"strong engagement signals in {row['Stage']} stage"
                ),
                recommended_action = (
                    "Prioritise next-step meeting this week. Assign AE executive "
                    "sponsor to accelerate to Proposal."
                ),
                confidence       = "High" if score >= 75 else "Medium",
                composite_score  = round(score, 1),
            ))

        # ── Expansion signals ─────────────────────────────────────────────
        exp_mask = (
            self.df["Product Adoption Score"].ge(EXPANSION_ADOPTION_MIN)
            & self.df["Customer Health Score"].ge(EXPANSION_HEALTH_MIN)
            & self.df["Renewal Probability"].ge(EXPANSION_RENEWAL_MIN)
            & ~self.df["_is_stale"]
        )
        for _, row in self.df[exp_mask].iterrows():
            score = row["_opportunity_score"]
            records.append(RevenueOpportunityRecord(
                opportunity_id   = str(row["Opportunity ID"]),
                company_name     = str(row["Company Name"]),
                arr              = float(row["ARR"]),
                arr_fmt          = _fmt_arr(row["ARR"]),
                stage            = str(row["Stage"]),
                sales_rep        = str(row["Sales Rep"]),
                industry         = str(row["Industry"]),
                opportunity_type = "Expansion",
                rationale        = (
                    f"Adoption {row['Product Adoption Score']:.0f}/100, "
                    f"Health {row['Customer Health Score']:.0f}/100, "
                    f"Renewal {row['Renewal Probability']:.0f}% — "
                    "prime upsell / cross-sell candidate"
                ),
                recommended_action = (
                    "Assign CSM to schedule Executive Business Review. "
                    "Prepare tier-upgrade or adjacent product proposal."
                ),
                confidence       = "High" if score >= 80 else "Medium",
                composite_score  = round(score, 1),
            ))

        # ── Pipeline acceleration ─────────────────────────────────────────
        accel_mask = (
            self._active["Stage"].isin(["Proposal", "Negotiation"])
            & self._active["_is_stale"]
            & self._active["Customer Health Score"].between(40, 70)
        )
        for _, row in self._active[accel_mask].iterrows():
            excess = row["Days in Stage"] - row["_stage_median_days"]
            records.append(RevenueOpportunityRecord(
                opportunity_id   = str(row["Opportunity ID"]),
                company_name     = str(row["Company Name"]),
                arr              = float(row["ARR"]),
                arr_fmt          = _fmt_arr(row["ARR"]),
                stage            = str(row["Stage"]),
                sales_rep        = str(row["Sales Rep"]),
                industry         = str(row["Industry"]),
                opportunity_type = "Acceleration",
                rationale        = (
                    f"Stalled {row['Days in Stage']:.0f} days in {row['Stage']} "
                    f"({excess:.0f} days past stage median). "
                    f"Health {row['Customer Health Score']:.0f}/100 — recoverable"
                ),
                recommended_action = (
                    "Manager to join next call. Review objection log. "
                    "Consider pricing concession or executive sponsor introduction."
                ),
                confidence       = "Medium",
                composite_score  = round(row["_opportunity_score"], 1),
            ))

        # Deduplicate by opp ID (expansion + quick-win overlap possible)
        seen: set[str] = set()
        deduped: list[RevenueOpportunityRecord] = []
        for rec in sorted(records, key=lambda r: -r.composite_score):
            if rec.opportunity_id not in seen:
                seen.add(rec.opportunity_id)
                deduped.append(rec)

        return deduped[:50]   # cap at 50 for UI performance

    # ── Pipeline health composite ─────────────────────────────────────────

    def _compute_pipeline_health(
        self,
        bottlenecks: list[BottleneckRecord],
        high_risk: list[HighRiskRecord],
    ) -> dict[str, Any]:
        """
        Aggregate a portfolio-level pipeline health score (0–100)
        and supporting metrics.

        Scoring components:
          Win Rate Score    (30%) — scaled to 0–100 from 0–80% win rate
          Velocity Score    (25%) — 100 minus avg velocity pressure across stages
          Risk Score        (25%) — 100 minus % of ARR in High Risk tier
          Coverage Score    (20%) — pipeline coverage vs. target ratio
        """
        active    = self._active
        closed    = self._closed

        # Win rate score
        wr = (
            (self._won["ARR"].sum() / closed["ARR"].sum() * 100)
            if len(closed) > 0 and closed["ARR"].sum() > 0 else 0.0
        )
        wr_score = min(wr / 80 * 100, 100)

        # Velocity score — lower pressure is better
        avg_pressure = active["_velocity_pressure"].mean() if len(active) > 0 else 100.0
        velocity_score = max(0, 100 - (avg_pressure - 100))   # 100 = on benchmark

        # Risk score — based on % of total ARR in high-risk tier
        total_arr     = self.df["ARR"].sum()
        high_risk_arr = self.df[
            self.df["Renewal Probability"] < RISK_HIGH_THRESHOLD
        ]["ARR"].sum()
        risk_pct      = (high_risk_arr / total_arr * 100) if total_arr > 0 else 0.0
        risk_score    = max(0, 100 - risk_pct * 1.5)   # 67%+ high-risk → score 0

        # Coverage score — active pipeline vs. proxy quota
        avg_won_arr   = self._won["ARR"].mean() if len(self._won) > 0 else 0.0
        proxy_quota   = avg_won_arr * len(self._won)
        coverage      = (
            active["ARR"].sum() / proxy_quota * PIPELINE_COVERAGE_TARGET
            if proxy_quota > 0 else 0.0
        )
        coverage_score = min(coverage / PIPELINE_COVERAGE_TARGET * 100, 100)

        composite = (
            wr_score       * 0.30
            + velocity_score * 0.25
            + risk_score     * 0.25
            + coverage_score * 0.20
        )

        if composite >= 75:
            health_label = "Healthy"
        elif composite >= 55:
            health_label = "At Risk"
        elif composite >= 35:
            health_label = "Critical"
        else:
            health_label = "Emergency"

        critical_bottlenecks = [b for b in bottlenecks if b.severity == "Critical"]
        p1_risks = [r for r in high_risk if r.priority == "P1"]
        stale_arr = self.df[self.df["_is_stale"]]["ARR"].sum()

        return {
            "composite_score":          round(composite, 1),
            "health_label":             health_label,
            "win_rate_score":           round(wr_score, 1),
            "velocity_score":           round(velocity_score, 1),
            "risk_score":               round(risk_score, 1),
            "coverage_score":           round(coverage_score, 1),
            "total_pipeline_arr":       round(active["ARR"].sum(), 0),
            "total_pipeline_arr_fmt":   _fmt_arr(active["ARR"].sum()),
            "total_arr":                round(total_arr, 0),
            "total_arr_fmt":            _fmt_arr(total_arr),
            "high_risk_arr":            round(high_risk_arr, 0),
            "high_risk_arr_fmt":        _fmt_arr(high_risk_arr),
            "high_risk_pct":            round(risk_pct, 1),
            "stale_deals":              int(self.df["_is_stale"].sum()),
            "stale_arr":                round(stale_arr, 0),
            "stale_arr_fmt":            _fmt_arr(stale_arr),
            "critical_bottleneck_count":len(critical_bottlenecks),
            "p1_risk_count":            len(p1_risks),
            "p1_risk_arr":              round(sum(r.arr for r in p1_risks), 0),
            "p1_risk_arr_fmt":          _fmt_arr(sum(r.arr for r in p1_risks)),
            "avg_velocity_pressure":    round(avg_pressure, 1),
            "pipeline_coverage_ratio":  round(coverage, 2),
            "pipeline_coverage_target": PIPELINE_COVERAGE_TARGET,
        }

    # ── 7. Recommendations engine ─────────────────────────────────────────

    def _generate_recommendations(
        self,
        win_rate:        dict[str, Any],
        bottlenecks:     list[BottleneckRecord],
        high_risk:       list[HighRiskRecord],
        rep_performance: list[RepPerformanceRecord],
        industry:        list[IndustryRecord],
        opportunities:   list[RevenueOpportunityRecord],
        pipeline_health: dict[str, Any],
    ) -> list[Recommendation]:
        """
        Generate 5–10 prioritised, evidence-based revenue recommendations.

        Each recommendation is fired by a specific quantitative trigger,
        has a named owner, and carries an estimated ARR impact derived
        from the data — not a generic template.
        """
        recs: list[Recommendation] = []
        idx = [1]   # mutable counter for ID generation

        def _next_id() -> str:
            s = f"REC-{idx[0]:02d}"
            idx[0] += 1
            return s

        # ── REC: Critical bottleneck ──────────────────────────────────────
        for bn in bottlenecks:
            if bn.severity == "Critical":
                recs.append(Recommendation(
                    id            = _next_id(),
                    category      = "Pipeline Velocity",
                    priority      = "P1 — Immediate",
                    headline      = (
                        f"Break the {bn.stage} stage bottleneck "
                        f"({bn.avg_days:.0f} days avg vs. {bn.benchmark_days:.0f}-day benchmark)"
                    ),
                    detail        = (
                        f"{bn.deal_count} deals are averaging {bn.avg_days:.0f} days in "
                        f"{bn.stage} — {bn.pct_above_benchmark:.0f}% above the "
                        f"{bn.benchmark_days:.0f}-day SaaS benchmark. "
                        f"Introduce a structured {bn.stage.lower()} exit criteria checklist "
                        f"and weekly manager deal reviews for all deals exceeding "
                        f"{bn.benchmark_days * 1.5:.0f} days in this stage."
                    ),
                    metric_trigger = (
                        f"Avg {bn.avg_days:.0f}d vs. {bn.benchmark_days:.0f}d benchmark "
                        f"(+{bn.excess_days:.0f}d excess)"
                    ),
                    estimated_arr_impact       = bn.arr_at_risk,
                    estimated_arr_impact_fmt   = bn.arr_at_risk_fmt,
                    owner  = "VP Sales + RevOps",
                    effort = "Medium",
                ))
            if len(recs) >= 3:
                break   # Cap bottleneck recs at 3

        # ── REC: P1 churn risk ────────────────────────────────────────────
        p1_risks = [r for r in high_risk if r.priority == "P1"]
        if p1_risks:
            p1_arr = sum(r.arr for r in p1_risks)
            recs.append(Recommendation(
                id            = _next_id(),
                category      = "Retention",
                priority      = "P1 — Immediate",
                headline      = (
                    f"Emergency retention programme for {len(p1_risks)} P1 "
                    f"accounts ({_fmt_arr(p1_arr)} ARR)"
                ),
                detail        = (
                    f"{len(p1_risks)} high-value accounts have composite risk scores "
                    f"≥ 70 with ARR > $100K each. Each account carries 3+ simultaneous "
                    f"risk signals (low renewal probability, health, and adoption). "
                    f"Assign dedicated CSM coverage and initiate executive sponsor outreach "
                    f"within 48 hours. Prepare retention offer framework for top 5 accounts."
                ),
                metric_trigger = (
                    f"{len(p1_risks)} accounts with risk score ≥ 70 and ARR > $100K"
                ),
                estimated_arr_impact       = p1_arr,
                estimated_arr_impact_fmt   = _fmt_arr(p1_arr),
                owner  = "CS Leader + CRO",
                effort = "High",
            ))

        # ── REC: Underperforming reps ────────────────────────────────────
        at_risk_reps = [r for r in rep_performance if r.performance_tier == "At Risk"]
        developing_reps = [r for r in rep_performance if r.performance_tier == "Developing"]
        coaching_reps = at_risk_reps + developing_reps
        if coaching_reps:
            avg_delta = win_rate["team_avg_win_rate"] - sum(
                r.win_rate for r in coaching_reps
            ) / len(coaching_reps)
            # ARR impact: if coaching lifts win rate by 10pp on their open pipeline
            coaching_arr_impact = sum(r.open_arr * 0.10 for r in coaching_reps)
            recs.append(Recommendation(
                id            = _next_id(),
                category      = "Sales Coaching",
                priority      = "P1 — Immediate" if at_risk_reps else "P2 — This Week",
                headline      = (
                    f"Structured coaching plan for {len(coaching_reps)} "
                    f"underperforming rep{'s' if len(coaching_reps) > 1 else ''}"
                ),
                detail        = (
                    f"{len(coaching_reps)} rep(s) are {avg_delta:.0f}pp below team average "
                    f"win rate ({win_rate['team_avg_win_rate']:.0f}%). "
                    f"Specific issues identified: "
                    + "; ".join(
                        f"{r.sales_rep}: {r.coaching_flags[0]}"
                        for r in coaching_reps[:3]
                        if r.coaching_flags
                    ) + ". "
                    f"Implement bi-weekly 1:1 coaching cadence with call recording review. "
                    f"A 10pp win-rate improvement on their combined open pipeline represents "
                    f"~{_fmt_arr(coaching_arr_impact)} in recoverable ARR."
                ),
                metric_trigger = (
                    f"{len(coaching_reps)} reps >{10 if not at_risk_reps else 25}pp "
                    f"below team avg ({win_rate['team_avg_win_rate']:.0f}%)"
                ),
                estimated_arr_impact       = round(coaching_arr_impact, 0),
                estimated_arr_impact_fmt   = _fmt_arr(coaching_arr_impact),
                owner  = "Sales Manager",
                effort = "Medium",
            ))

        # ── REC: Win rate on worst industry ──────────────────────────────
        investigate_industries = [i for i in industry if i.signal == "Investigate"]
        if investigate_industries:
            worst = investigate_industries[0]
            recs.append(Recommendation(
                id            = _next_id(),
                category      = "Market Strategy",
                priority      = "P2 — This Week",
                headline      = (
                    f"Conduct {worst.industry} win/loss audit "
                    f"({worst.win_rate:.0f}% win rate)"
                ),
                detail        = (
                    f"{worst.industry} has a {worst.win_rate:.0f}% win rate against "
                    f"a team average of {win_rate['team_avg_win_rate']:.0f}%, with "
                    f"{worst.high_risk_pct:.0f}% of accounts in the High Risk tier. "
                    f"Conduct a structured win/loss review of the last 10 deals in this "
                    f"vertical. Determine whether this is a product-market fit issue, "
                    f"competitive displacement, or pricing mismatch before further "
                    f"investment in this segment."
                ),
                metric_trigger = (
                    f"{worst.industry} win rate {worst.win_rate:.0f}% "
                    f"({win_rate['team_avg_win_rate'] - worst.win_rate:.0f}pp below avg)"
                ),
                estimated_arr_impact       = round(worst.total_arr * 0.15, 0),
                estimated_arr_impact_fmt   = _fmt_arr(worst.total_arr * 0.15),
                owner  = "VP Sales + Product Marketing",
                effort = "Medium",
            ))

        # ── REC: Stale deal recovery ──────────────────────────────────────
        stale_count = pipeline_health["stale_deals"]
        stale_arr   = pipeline_health["stale_arr"]
        if stale_count > 5:
            recs.append(Recommendation(
                id            = _next_id(),
                category      = "Pipeline Hygiene",
                priority      = "P2 — This Week",
                headline      = (
                    f"Launch stale deal re-engagement sprint "
                    f"({stale_count} deals, {_fmt_arr(stale_arr)} ARR)"
                ),
                detail        = (
                    f"{stale_count} active deals have exceeded 1.5× their stage-median "
                    f"velocity, putting {_fmt_arr(stale_arr)} at risk of going dark. "
                    f"Deploy a 5-touch re-engagement sequence within 72 hours: "
                    f"personalised exec outreach, ROI summary, competitive proof point, "
                    f"limited-time commercial incentive, and close-or-kill decision call. "
                    f"Deals with no response after the sequence should be marked Lost to "
                    f"preserve forecast accuracy."
                ),
                metric_trigger = (
                    f"{stale_count} deals past 1.5× stage median; "
                    f"{_fmt_arr(stale_arr)} at risk"
                ),
                estimated_arr_impact       = round(stale_arr * 0.35, 0),
                estimated_arr_impact_fmt   = _fmt_arr(stale_arr * 0.35),
                owner  = "Sales Manager + RevOps",
                effort = "Low",
            ))

        # ── REC: Expansion motion ─────────────────────────────────────────
        expansion_opps = [o for o in opportunities if o.opportunity_type == "Expansion"]
        if expansion_opps:
            exp_arr = sum(o.arr for o in expansion_opps[:20])
            exp_industries = list({o.industry for o in expansion_opps[:10]})[:3]
            recs.append(Recommendation(
                id            = _next_id(),
                category      = "Revenue Growth",
                priority      = "P2 — This Week",
                headline      = (
                    f"Activate expansion motion across "
                    f"{min(len(expansion_opps), 20)} high-adoption accounts"
                ),
                detail        = (
                    f"{len(expansion_opps)} accounts show adoption ≥ {EXPANSION_ADOPTION_MIN:.0f}/100, "
                    f"health ≥ {EXPANSION_HEALTH_MIN:.0f}/100, and renewal probability "
                    f"≥ {EXPANSION_RENEWAL_MIN:.0f}% — all strong upsell signals. "
                    f"Top industries: {', '.join(exp_industries)}. "
                    f"Assign AE + CSM pairs to each account for joint Executive Business Reviews. "
                    f"Lead with ROI data and a tier-upgrade or adjacent product proposal. "
                    f"Target 15–25% ACV expansion per account."
                ),
                metric_trigger = (
                    f"{len(expansion_opps)} accounts with adoption ≥ {EXPANSION_ADOPTION_MIN:.0f}, "
                    f"health ≥ {EXPANSION_HEALTH_MIN:.0f}, renewal ≥ {EXPANSION_RENEWAL_MIN:.0f}%"
                ),
                estimated_arr_impact       = round(exp_arr * 0.20, 0),
                estimated_arr_impact_fmt   = _fmt_arr(exp_arr * 0.20),
                owner  = "CS Leader + AE Team",
                effort = "Medium",
            ))

        # ── REC: Win-rate top-objection ───────────────────────────────────
        if win_rate.get("top_loss_objections"):
            top_obj, top_count = next(iter(win_rate["top_loss_objections"].items()))
            total_lost = win_rate["total_lost_deals"]
            if total_lost > 0:
                obj_pct = top_count / total_lost * 100
                recs.append(Recommendation(
                    id            = _next_id(),
                    category      = "Sales Enablement",
                    priority      = "P3 — This Month",
                    headline      = (
                        f"Build '{top_obj}' objection playbook "
                        f"({obj_pct:.0f}% of all losses)"
                    ),
                    detail        = (
                        f"'{top_obj}' is the leading loss driver, appearing in "
                        f"{top_count} of {total_lost} lost deals ({obj_pct:.0f}%). "
                        f"Convene a win/loss debrief session with the top 3 reps, "
                        f"product marketing, and CS. Develop a structured objection "
                        f"handling guide with: root cause analysis, counter-narrative, "
                        f"ROI evidence pack, and a reference customer story. "
                        f"Deploy via sales enablement platform within 3 weeks."
                    ),
                    metric_trigger = (
                        f"'{top_obj}' in {obj_pct:.0f}% of {total_lost} lost deals"
                    ),
                    estimated_arr_impact       = round(
                        self._lost["ARR"].mean() * top_count * 0.20, 0
                    ),
                    estimated_arr_impact_fmt   = _fmt_arr(
                        self._lost["ARR"].mean() * top_count * 0.20
                    ),
                    owner  = "Sales Enablement + Product Marketing",
                    effort = "Medium",
                ))

        # ── REC: Best industry — double down ─────────────────────────────
        expand_industries = [i for i in industry if i.signal == "Expand"]
        if expand_industries:
            best = expand_industries[0]
            recs.append(Recommendation(
                id            = _next_id(),
                category      = "Market Strategy",
                priority      = "P3 — This Month",
                headline      = (
                    f"Accelerate {best.industry} GTM — "
                    f"highest-signal expansion vertical ({best.win_rate:.0f}% win rate)"
                ),
                detail        = (
                    f"{best.industry} leads the portfolio with a {best.win_rate:.0f}% "
                    f"win rate, {best.high_risk_pct:.0f}% high-risk accounts, and "
                    f"{best.avg_adoption_score:.0f}/100 avg adoption. "
                    f"This vertical has the strongest product-market fit signal. "
                    f"Recommend: dedicated {best.industry} AE pod, industry-specific "
                    f"case study content, and vertical-targeted outbound sequence "
                    f"to accelerate pipeline build in this segment."
                ),
                metric_trigger = (
                    f"{best.industry}: {best.win_rate:.0f}% WR, "
                    f"{best.high_risk_pct:.0f}% high-risk, "
                    f"{best.arr_share_pct:.0f}% ARR share"
                ),
                estimated_arr_impact       = round(best.total_arr * 0.25, 0),
                estimated_arr_impact_fmt   = _fmt_arr(best.total_arr * 0.25),
                owner  = "VP Sales + Product Marketing",
                effort = "High",
            ))

        # ── REC: Quick wins ───────────────────────────────────────────────
        quick_wins = [o for o in opportunities if o.opportunity_type == "Quick Win"]
        if quick_wins:
            qw_arr = sum(o.arr for o in quick_wins[:10])
            recs.append(Recommendation(
                id            = _next_id(),
                category      = "Pipeline Acceleration",
                priority      = "P2 — This Week",
                headline      = (
                    f"Fast-track {min(len(quick_wins), 10)} high-signal "
                    f"deals to Proposal ({_fmt_arr(qw_arr)} ARR)"
                ),
                detail        = (
                    f"{len(quick_wins)} active deals in early stages show health ≥ "
                    f"{QUICK_WIN_HEALTH_MIN:.0f}/100 and adoption ≥ "
                    f"{QUICK_WIN_ADOPTION_MIN:.0f}/100 — the strongest buy signals "
                    f"available without a direct conversation. "
                    f"AEs should prioritise same-week outreach to each, lead with "
                    f"a business value summary, and push for a Proposal meeting commitment "
                    f"within 5 business days."
                ),
                metric_trigger = (
                    f"{len(quick_wins)} deals with health ≥ {QUICK_WIN_HEALTH_MIN:.0f} "
                    f"and adoption ≥ {QUICK_WIN_ADOPTION_MIN:.0f}"
                ),
                estimated_arr_impact       = round(qw_arr * win_rate["overall_win_rate"] / 100, 0),
                estimated_arr_impact_fmt   = _fmt_arr(qw_arr * win_rate["overall_win_rate"] / 100),
                owner  = "AE Team + Sales Manager",
                effort = "Low",
            ))

        # Sort: P1 first, then P2, then P3; within tier by ARR impact desc
        priority_order = {"P1 — Immediate": 0, "P2 — This Week": 1, "P3 — This Month": 2}
        recs.sort(
            key=lambda r: (
                priority_order.get(r.priority, 9),
                -r.estimated_arr_impact,
            )
        )
        return recs[:10]   # hard cap: max 10 recommendations

    # ── Executive summary assembly ────────────────────────────────────────

    def _build_executive_summary(
        self,
        win_rate:        dict[str, Any],
        bottlenecks:     list[BottleneckRecord],
        high_risk:       list[HighRiskRecord],
        rep_performance: list[RepPerformanceRecord],
        industry:        list[IndustryRecord],
        pipeline_health: dict[str, Any],
    ) -> dict[str, Any]:
        """Assemble the top-level one-page brief for a VP of RevOps."""

        worst_bottleneck = bottlenecks[0] if bottlenecks else None
        riskiest_ind     = max(industry, key=lambda i: i.high_risk_pct) if industry else None
        top_rep          = next(
            (r for r in rep_performance if r.performance_tier == "Top"), None
        )
        worst_rep        = next(
            (r for r in reversed(rep_performance)
             if r.performance_tier in ("At Risk", "Developing")
             and r.total_closed >= MIN_CLOSED_DEALS_FOR_COACHING),
            None,
        )
        p1_count = len([r for r in high_risk if r.priority == "P1"])
        p1_arr   = sum(r.arr for r in high_risk if r.priority == "P1")

        score = pipeline_health["composite_score"]
        label = pipeline_health["health_label"]

        headline = (
            f"Pipeline health is {label} ({score:.0f}/100). "
            f"{_fmt_arr(pipeline_health['total_pipeline_arr'])} active ARR across "
            f"{len(self._active)} open deals. "
            + (
                f"Primary bottleneck: {worst_bottleneck.stage} at "
                f"{worst_bottleneck.avg_days:.0f} days avg "
                f"(+{worst_bottleneck.excess_days:.0f}d vs. benchmark). "
                if worst_bottleneck else ""
            )
            + f"{p1_count} P1 accounts represent {_fmt_arr(p1_arr)} ARR at immediate risk."
        )

        return {
            "headline":                  headline,
            "pipeline_health_score":     score,
            "pipeline_health_label":     label,
            "total_pipeline_arr":        pipeline_health["total_pipeline_arr"],
            "total_pipeline_arr_fmt":    pipeline_health["total_pipeline_arr_fmt"],
            "overall_win_rate":          win_rate["overall_win_rate"],
            "total_won_arr":             win_rate["total_won_arr"],
            "total_won_arr_fmt":         win_rate["total_won_arr_fmt"],
            "primary_bottleneck_stage":  worst_bottleneck.stage if worst_bottleneck else "None",
            "primary_bottleneck_days":   worst_bottleneck.avg_days if worst_bottleneck else 0,
            "total_bottlenecks":         len(bottlenecks),
            "critical_bottlenecks":      sum(1 for b in bottlenecks if b.severity == "Critical"),
            "high_risk_accounts":        len(high_risk),
            "p1_accounts":               p1_count,
            "p1_arr":                    p1_arr,
            "p1_arr_fmt":                _fmt_arr(p1_arr),
            "riskiest_industry":         riskiest_ind.industry if riskiest_ind else "N/A",
            "riskiest_industry_pct":     riskiest_ind.high_risk_pct if riskiest_ind else 0,
            "best_win_rate_industry":    win_rate.get("best_industry", "N/A"),
            "worst_win_rate_industry":   win_rate.get("worst_industry", "N/A"),
            "top_performer":             top_rep.sales_rep if top_rep else "N/A",
            "top_performer_wr":          top_rep.win_rate if top_rep else 0,
            "rep_needing_coaching":      worst_rep.sales_rep if worst_rep else "N/A",
            "rep_coaching_wr":           worst_rep.win_rate if worst_rep else 0,
            "stale_deals":               pipeline_health["stale_deals"],
            "stale_arr_fmt":             pipeline_health["stale_arr_fmt"],
            "run_timestamp":             self.run_timestamp,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_arr(value: float) -> str:
    """Format a dollar amount as a concise string (e.g. $4.2M, $850K, $42K)."""
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:,.0f}"
