#!/usr/bin/env python3
"""
example_usage.py
================
Standalone demonstration of PipelineAnalystAgent.

Run from the project root:
    python example_usage.py

Or with a custom CSV:
    python example_usage.py --csv path/to/your_pipeline.csv

Outputs a human-readable report to the terminal and writes the full
JSON report to  output/pipeline_analyst_report.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

# ── Allow running from any directory ──────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from agents.pipeline_analyst import PipelineAnalystAgent
from utils.data_loader import load_data


# ── ANSI colour helpers (terminal output only) ────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"


def _c(text: str, *codes: str) -> str:
    return "".join(codes) + str(text) + RESET


def _divider(char: str = "─", width: int = 72, color: str = DIM) -> str:
    return _c(char * width, color)


def _header(title: str) -> None:
    print()
    print(_divider("═"))
    print(_c(f"  {title}", BOLD, WHITE))
    print(_divider("═"))


def _section(title: str) -> None:
    print()
    print(_c(f"  ── {title}", BOLD, CYAN))
    print(_divider())


def _fmt_pct(v: float, invert: bool = False) -> str:
    """Color a percentage green/amber/red based on direction."""
    good  = v >= 60 if not invert else v <= 30
    warn  = v >= 40 if not invert else v <= 50
    color = GREEN if good else (YELLOW if warn else RED)
    return _c(f"{v:.1f}%", color)


def _fmt_score(v: float) -> str:
    color = GREEN if v >= 70 else (YELLOW if v >= 45 else RED)
    return _c(f"{v:.1f}/100", color, BOLD)


def _priority_color(p: str) -> str:
    return {
        "P1 — Immediate":  RED,
        "P2 — This Week":  YELLOW,
        "P3 — This Month": DIM,
        "P1": RED,
        "P2": YELLOW,
        "P3": DIM,
    }.get(p, RESET)


# ── Report sections ───────────────────────────────────────────────────────

def print_executive_summary(report: dict) -> None:
    es = report["executive_summary"]
    ph = report["pipeline_health"]

    _header("PIPELINE ANALYST AGENT  ·  Revenue Activation OS")

    print(f"\n  {_c('Run timestamp:', DIM)} {report['meta']['run_timestamp']}")
    print(f"  {_c('Opportunities analysed:', DIM)} {report['meta']['total_opportunities']}")

    _section("Executive Summary")
    print(f"\n  {textwrap.fill(es['headline'], width=70, subsequent_indent='  ')}\n")

    # Health score
    score = ph["composite_score"]
    label = ph["health_label"]
    score_color = GREEN if score >= 75 else (YELLOW if score >= 55 else RED)
    bar_filled = int(score / 5)
    bar = _c("█" * bar_filled, score_color) + _c("░" * (20 - bar_filled), DIM)
    print(f"  Pipeline Health  {bar}  {_c(f'{score:.0f}/100  {label}', score_color, BOLD)}\n")

    cols = [
        ("Win Rate Score",  ph["win_rate_score"],  "30%"),
        ("Velocity Score",  ph["velocity_score"],  "25%"),
        ("Risk Score",      ph["risk_score"],       "25%"),
        ("Coverage Score",  ph["coverage_score"],  "20%"),
    ]
    for name, val, weight in cols:
        v_color = GREEN if val >= 70 else (YELLOW if val >= 45 else RED)
        print(f"    {name:<18} {_c(f'{val:.0f}', v_color, BOLD):<20}  {_c(weight, DIM)}")

    # Key numbers
    _section("Portfolio Snapshot")
    rows = [
        ("Active Pipeline ARR",    ph["total_pipeline_arr_fmt"],          BLUE),
        ("Total Won ARR",          report["win_rate_analysis"]["total_won_arr_fmt"], GREEN),
        ("Overall Win Rate",       f"{report['win_rate_analysis']['overall_win_rate']:.0f}%", GREEN),
        ("P1 Risk Accounts",       f"{es['p1_accounts']}  ({es['p1_arr_fmt']} ARR)",    RED),
        ("High-Risk Accounts",     str(es["high_risk_accounts"]),          RED),
        ("Pipeline Bottlenecks",   f"{es['total_bottlenecks']} detected ({es['critical_bottlenecks']} Critical)", YELLOW),
        ("Stale Deals",            f"{es['stale_deals']}  ({ph['stale_arr_fmt']} ARR)",  YELLOW),
        ("Top Performer",          f"{es['top_performer']} ({es['top_performer_wr']:.0f}% WR)", GREEN),
        ("Needs Coaching",         f"{es['rep_needing_coaching']} ({es['rep_coaching_wr']:.0f}% WR)", YELLOW),
        ("Riskiest Industry",      f"{es['riskiest_industry']} ({es['riskiest_industry_pct']:.0f}% high-risk)", RED),
    ]
    for label, value, color in rows:
        print(f"    {_c(label + ':', DIM):<36} {_c(value, color, BOLD)}")


def print_win_rates(report: dict) -> None:
    wr = report["win_rate_analysis"]
    _section("Win Rate Analysis")

    print(f"\n  {'Sales Rep':<22} {'Win Rate':>9}  {'Closed':>7}  {'Won ARR':>12}")
    print("  " + "─" * 54)
    for rep in wr["by_rep"]:
        wr_str = f"{rep['win_rate']:>5.0f}%"
        wr_color = (GREEN if rep["win_rate"] >= wr["team_avg_win_rate"]
                    else YELLOW if rep["win_rate"] >= wr["team_avg_win_rate"] * 0.75
                    else RED)
        print(
            f"  {rep['sales_rep']:<22} "
            f"{_c(wr_str, wr_color, BOLD):<28} "
            f"{rep['deal_count']:>7}  "
            f"{_c('${:>11,.0f}'.format(rep['won_arr']), BLUE)}"
        )

    print(f"\n  {'Deal Size':<22} {'Win Rate':>9}  {'Deals':>7}")
    print("  " + "─" * 40)
    for ds in wr.get("by_deal_size", []):
        print(f"  {ds['deal_size']:<22} {ds['win_rate']:>8.0f}%  {ds['deal_count']:>7}")

    print(f"\n  {_c('Top loss driver:', DIM)} ", end="")
    obj_items = list(wr.get("top_loss_objections", {}).items())
    if obj_items:
        top_obj, top_cnt = obj_items[0]
        print(_c(f"'{top_obj}' ({top_cnt} deals)", RED))


def print_bottlenecks(report: dict) -> None:
    bns = report["pipeline_bottlenecks"]
    if not bns:
        return
    _section("Pipeline Bottleneck Analysis")

    sev_colors = {"Critical": RED, "High": YELLOW, "Moderate": DIM}
    print(f"\n  {'Stage':<16} {'Avg Days':>9}  {'Benchmark':>10}  {'Excess':>7}  {'Severity':>10}  {'ARR at Risk':>12}")
    print("  " + "─" * 70)
    for b in bns:
        color      = sev_colors.get(b["severity"], DIM)
        avg_str    = "{:>7.0f}d".format(b["avg_days"])
        excess_str = "+{:.0f}d".format(b["excess_days"])
        sev_str    = "{:>10}".format(b["severity"])
        arr_str    = "{:>12}".format(b["arr_at_risk_fmt"])
        bench_str  = "{:>8.0f}d".format(b["benchmark_days"])
        print(
            f"  {b['stage']:<16} "
            + _c(avg_str, color, BOLD)
            + f"  {bench_str}  "
            + _c(excess_str, color)
            + "  "
            + _c(sev_str, color, BOLD)
            + "  "
            + _c(arr_str, color)
        )


def print_high_risk(report: dict) -> None:
    risks = report["high_risk_opportunities"]
    p1 = [r for r in risks if r["priority"] == "P1"]
    _section(f"High-Risk Opportunities  ({len(risks)} total — showing top P1 accounts)")

    if not p1:
        print("  No P1 accounts detected.")
        return

    for r in p1[:8]:
        print(f"\n  {_c(r['company_name'], RED, BOLD):<30} {_c(r['arr_fmt'], RED):<12} "
              f"{r['stage']:<15} {r['sales_rep']}")
        risk_score_str    = "{:.0f}/100".format(r["risk_score"])
        renewal_str       = "{:.0f}%".format(r["renewal_probability"])
        health_str        = "{:.0f}".format(r["customer_health_score"])
        adoption_str      = "{:.0f}".format(r["product_adoption_score"])
        print(
            "    Risk Score: " + _c(risk_score_str, RED)
            + "  Renewal: " + _c(renewal_str, RED)
            + "  Health: " + health_str
            + "  Adoption: " + adoption_str
        )
        for flag in r["risk_flags"]:
            print(f"    {_c('▸', RED)} {_c(flag, DIM)}")


def print_reps(report: dict) -> None:
    reps = report["sales_rep_analysis"]
    _section("Sales Rep Performance")
    tier_colors = {"Top": GREEN, "Mid": BLUE, "Developing": YELLOW, "At Risk": RED}

    print(f"\n  {'Rep':<20} {'Tier':<12} {'WR':>5}  {'W':>3} {'L':>3} {'Open':>5}  {'Won ARR':>10}  {'Open ARR':>11}")
    print("  " + "─" * 78)
    for r in reps:
        color    = tier_colors.get(r["performance_tier"], DIM)
        tier_str = "{:<11}".format(r["performance_tier"])
        wr_str   = "{:.0f}%".format(r["win_rate"])
        print(
            f"  {r['sales_rep']:<20} "
            + _c(tier_str, color, BOLD) + "  "
            + _c("{:>4}".format(wr_str), color) + "  "
            + "{:>3}  {:>3}  {:>5}  {:>10}  {:>11}".format(
                r["deals_won"], r["deals_lost"], r["deals_open"],
                r["won_arr_fmt"], r["open_arr_fmt"]
            )
        )
        if r["coaching_flags"]:
            for flag in r["coaching_flags"][:2]:
                short = flag[:65] + "…" if len(flag) > 65 else flag
                print(f"    {_c('↳', YELLOW)} {_c(short, DIM)}")


def print_industry(report: dict) -> None:
    inds = report["industry_analysis"]
    _section("Industry Intelligence")
    sig_colors = {"Expand": GREEN, "Protect": BLUE, "Monitor": YELLOW, "Investigate": RED}

    print(f"\n  {'Industry':<22} {'ARR':>10}  {'WR':>5}  {'Hi-Risk':>8}  {'Signal':>12}")
    print("  " + "─" * 62)
    for i in inds[:12]:
        color   = sig_colors.get(i["signal"], DIM)
        sig_str = "{:>12}".format(i["signal"])
        print(
            "  {:<22} {:>10}  {:>4.0f}%  {:>7.0f}%  {}".format(
                i["industry"], i["total_arr_fmt"],
                i["win_rate"], i["high_risk_pct"],
                _c(sig_str, color, BOLD)
            )
        )


def print_opportunities(report: dict) -> None:
    opps = report["revenue_opportunities"]
    qw   = [o for o in opps if o["opportunity_type"] == "Quick Win"]
    exp  = [o for o in opps if o["opportunity_type"] == "Expansion"]
    acc  = [o for o in opps if o["opportunity_type"] == "Acceleration"]
    _section(f"Revenue Opportunities  ({len(opps)} identified)")

    for label, group, color in [
        ("Quick Wins", qw, GREEN),
        ("Expansion",  exp, BLUE),
        ("Acceleration", acc, YELLOW),
    ]:
        print(f"\n  {_c(label + f'  ({len(group)})', color, BOLD)}")
        for o in group[:5]:
            print(f"    {o['company_name']:<28} {o['arr_fmt']:>8}  {o['stage']:<14} {_c(o['rationale'][:50], DIM)}")


def print_recommendations(report: dict) -> None:
    recs = report["recommendations"]
    _header(f"AGENT RECOMMENDATIONS  ({len(recs)} prioritised actions)")

    for r in recs:
        p_color = _priority_color(r["priority"])
        print(f"\n  {_c(r['id'], BOLD, WHITE)}  {_c(r['priority'], p_color, BOLD)}  "
              f"{_c('·', DIM)}  {_c(r['category'], CYAN)}")
        print(f"  {_c(r['headline'], BOLD, WHITE)}")
        print()
        wrapped = textwrap.fill(r["detail"], width=68, initial_indent="    ",
                                subsequent_indent="    ")
        print(_c(wrapped, DIM))
        print()
        print(f"    {_c('Trigger:', DIM)}  {r['metric_trigger']}")
        print(f"    {_c('Owner:', DIM)}    {r['owner']}")
        print(f"    {_c('Effort:', DIM)}   {r['effort']}")
        print(f"    {_c('Est. ARR Impact:', DIM)}  {_c(r['estimated_arr_impact_fmt'], GREEN, BOLD)}")
        print("  " + "─" * 68)


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run PipelineAnalystAgent")
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to pipeline CSV (default: data/pipeline_data.csv)",
    )
    parser.add_argument(
        "--json-only", action="store_true",
        help="Only write JSON output, suppress terminal report",
    )
    parser.add_argument(
        "--output", type=str, default="output/pipeline_analyst_report.json",
        help="Path to write JSON output",
    )
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────
    if args.csv:
        df = pd.read_csv(args.csv)
        print(f"Loaded {len(df)} rows from {args.csv}")
    else:
        df = load_data()

    # ── Run agent ──────────────────────────────────────────────────────────
    agent  = PipelineAnalystAgent(df)
    report = agent.run()

    # ── Write JSON ─────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    if not args.json_only:
        # ── Terminal report ────────────────────────────────────────────────
        print_executive_summary(report)
        print_win_rates(report)
        print_bottlenecks(report)
        print_high_risk(report)
        print_reps(report)
        print_industry(report)
        print_opportunities(report)
        print_recommendations(report)
        print()
        print(_divider("═"))
        print(_c(f"  JSON report written → {out_path}", GREEN))
        print(_divider("═"))
        print()


if __name__ == "__main__":
    main()
