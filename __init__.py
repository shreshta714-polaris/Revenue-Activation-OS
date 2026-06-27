"""
agents/__init__.py
------------------
Agent registry for Revenue Activation OS.

This package is the future home of all AI agent modules.
Each agent will be a self-contained class with a standard interface:

    class BaseAgent:
        def run(self, df: pd.DataFrame, context: dict) -> AgentOutput
        def get_insights(self) -> list[Insight]
        def get_recommendations(self) -> list[Recommendation]

Current status: STUB — agents not yet implemented.
The dashboard renders placeholder cards where agent outputs will appear.

Planned agents (v2):
    - PipelineAnalystAgent      → agents/pipeline_analyst.py
    - AdoptionAnalystAgent      → agents/adoption_analyst.py
    - CoachingAnalystAgent      → agents/coaching_analyst.py
    - StrategyConsultantAgent   → agents/strategy_consultant.py
    - ExecutiveBriefingAgent    → agents/executive_briefing.py
"""

AGENT_REGISTRY = {
    "pipeline_analyst": {
        "name":        "Pipeline Analyst",
        "icon":        "◈",
        "status":      "coming_soon",
        "description": "Detects revenue bottlenecks, stalled deals, and conversion anomalies.",
        "module":      "agents.pipeline_analyst",
    },
    "adoption_analyst": {
        "name":        "Customer Adoption Analyst",
        "icon":        "◉",
        "status":      "coming_soon",
        "description": "Tracks adoption signals to surface churn risk and expansion opportunities.",
        "module":      "agents.adoption_analyst",
    },
    "coaching_analyst": {
        "name":        "Sales Coaching Analyst",
        "icon":        "◆",
        "status":      "coming_soon",
        "description": "Identifies rep-level skill gaps from call patterns and deal outcomes.",
        "module":      "agents.coaching_analyst",
    },
    "strategy_consultant": {
        "name":        "Revenue Strategy Consultant",
        "icon":        "◇",
        "status":      "coming_soon",
        "description": "Synthesizes all agent outputs into cross-functional revenue strategy.",
        "module":      "agents.strategy_consultant",
    },
    "executive_briefing": {
        "name":        "Executive Briefing Agent",
        "icon":        "★",
        "status":      "coming_soon",
        "description": "Delivers decision-quality briefings to C-suite before every standup.",
        "module":      "agents.executive_briefing",
    },
}


def get_agent_status(agent_key: str) -> str:
    return AGENT_REGISTRY.get(agent_key, {}).get("status", "unknown")


def is_agent_available(agent_key: str) -> bool:
    return get_agent_status(agent_key) == "active"
