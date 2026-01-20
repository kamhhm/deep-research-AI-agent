"""
Configuration & Constants

All magic numbers, thresholds, and settings live here.
Single source of truth for the entire pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import os


# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "crunchbase_data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Ensure directories exist
for dir_path in [OUTPUT_DIR, LOG_DIR, CHECKPOINT_DIR]:
    dir_path.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# API KEYS (loaded from environment)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class APIKeys:
    """API credentials loaded from environment variables."""
    tavily: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    openai: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    perplexity: str = field(default_factory=lambda: os.getenv("PERPLEXITY_API_KEY", ""))

    def validate(self) -> list[str]:
        """Return list of missing API keys."""
        missing = []
        if not self.tavily:
            missing.append("TAVILY_API_KEY")
        if not self.openai:
            missing.append("OPENAI_API_KEY")
        if not self.perplexity:
            missing.append("PERPLEXITY_API_KEY")
        return missing


# ─────────────────────────────────────────────────────────────────────────────
# COST TRACKING
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CostEstimates:
    """
    Per-query cost estimates for budget tracking.
    Values in USD.
    """
    # Stage 1: Tavily search + GPT-4o-mini interpretation
    tavily_search: float = 0.001
    gpt4o_mini_call: float = 0.001
    stage_1_total: float = 0.002

    # Stage 2A: Perplexity Sonar Base
    sonar_base: float = 0.02

    # Stage 2B: Perplexity Sonar Pro
    sonar_pro_min: float = 0.05
    sonar_pro_max: float = 0.10

    # Stage 3: Perplexity Deep Research
    deep_research_min: float = 0.41
    deep_research_max: float = 1.19

    # Budget constraints
    target_avg_per_company: float = 0.10
    hard_budget_limit_total: float = 5000.0  # $5k ceiling for 44k companies


# ─────────────────────────────────────────────────────────────────────────────
# ESCALATION THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EscalationThresholds:
    """
    Decision thresholds for routing companies through the pipeline.

    Based on the 2x2 matrix:
        - Online Presence Score (0-100)
        - AI Mentions Found (boolean)
    """
    # Presence score boundaries
    low_presence_max: int = 30      # 0-30 = low presence
    medium_presence_max: int = 70   # 31-70 = medium presence
    # 71-100 = high presence

    # Confidence thresholds for findings
    high_confidence_min: float = 0.8
    medium_confidence_min: float = 0.5

    def get_presence_tier(self, score: int) -> Literal["low", "medium", "high"]:
        """Classify presence score into tier."""
        if score <= self.low_presence_max:
            return "low"
        elif score <= self.medium_presence_max:
            return "medium"
        else:
            return "high"


# ─────────────────────────────────────────────────────────────────────────────
# PROCESSING SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProcessingConfig:
    """Settings for batch processing and rate limiting."""
    # Concurrency
    max_concurrent_requests: int = 10
    batch_size: int = 100

    # Timeouts (seconds)
    http_timeout: float = 10.0
    api_timeout: float = 30.0

    # Rate limiting (requests per minute)
    tavily_rpm: int = 60
    openai_rpm: int = 500
    perplexity_rpm: int = 60

    # Checkpointing
    checkpoint_every: int = 100  # Save progress every N companies

    # Retry policy
    max_retries: int = 3
    retry_delay_base: float = 1.0  # Exponential backoff base


# ─────────────────────────────────────────────────────────────────────────────
# FINDING TYPES
# ─────────────────────────────────────────────────────────────────────────────

FINDING_TYPES = [
    # Customer-facing
    "chatgpt_customer_support",
    "ai_chatbot_general",

    # Engineering & Development
    "copilot_engineering",
    "ai_code_generation",
    "ai_testing_qa",

    # Marketing & Content
    "ai_content_generation",
    "ai_marketing_automation",
    "ai_copywriting",

    # Operations
    "ai_document_processing",
    "ai_data_analysis",
    "ai_workflow_automation",

    # HR & Recruiting
    "ai_recruiting_screening",
    "ai_hr_operations",

    # Sales
    "ai_sales_outreach",
    "ai_lead_scoring",

    # Other
    "ai_internal_tools_general",
    "ai_adoption_unspecified",

    # Negative
    "no_adoption_found",
    "insufficient_information",
]


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT INSTANCES
# ─────────────────────────────────────────────────────────────────────────────

# Create singleton instances for easy import
COSTS = CostEstimates()
THRESHOLDS = EscalationThresholds()
PROCESSING = ProcessingConfig()


def load_api_keys() -> APIKeys:
    """Load and validate API keys from environment."""
    keys = APIKeys()
    missing = keys.validate()
    if missing:
        print(f"⚠️  Missing API keys: {', '.join(missing)}")
        print("   Set them in .env or export as environment variables.")
    return keys
