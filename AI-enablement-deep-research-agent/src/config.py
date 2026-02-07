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
CREDENTIALS_DIR = PROJECT_ROOT / "credentials"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# Stage-specific output directories
STAGE1_OUTPUT_DIR = OUTPUT_DIR / "stage1"
STAGE1_TAVILY_DIR = STAGE1_OUTPUT_DIR / "tavily"
STAGE1_GPT_DIR = STAGE1_OUTPUT_DIR / "gpt"
STAGE2_OUTPUT_DIR = OUTPUT_DIR / "stage2"
STAGE3_OUTPUT_DIR = OUTPUT_DIR / "stage3"

# Ensure directories exist
for dir_path in [OUTPUT_DIR, LOG_DIR, CHECKPOINT_DIR,
                 STAGE1_OUTPUT_DIR, STAGE1_TAVILY_DIR, STAGE1_GPT_DIR,
                 STAGE2_OUTPUT_DIR, STAGE3_OUTPUT_DIR]:
    dir_path.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# CREDENTIAL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _load_credential(filename: str) -> str:
    """
    Load an API key from a credential file.
    
    Falls back to environment variable if file doesn't exist.
    
    Args:
        filename: Name of the credential file (e.g., 'tavily_api_key.txt')
    
    Returns:
        The API key string, or empty string if not found.
    """
    # Try loading from file first
    cred_path = CREDENTIALS_DIR / filename
    if cred_path.exists():
        content = cred_path.read_text().strip()
        # Skip comment lines and empty lines
        lines = [line.strip() for line in content.split('\n') 
                 if line.strip() and not line.strip().startswith('#')]
        if lines:
            return lines[0]  # Return first non-comment line
    
    # Fall back to environment variable
    env_var = filename.replace('_api_key.txt', '').upper() + '_API_KEY'
    return os.getenv(env_var, "")


# ─────────────────────────────────────────────────────────────────────────────
# API KEYS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class APIKeys:
    """
    API credentials loaded from credentials folder or environment variables.
    
    Priority:
    1. credentials/<service>_api_key.txt file
    2. Environment variable (e.g., TAVILY_API_KEY)
    """
    tavily: str = ""
    openai: str = ""
    perplexity: str = ""
    
    def __post_init__(self):
        """Load credentials if not already set."""
        if not self.tavily:
            object.__setattr__(self, 'tavily', _load_credential('tavily_api_key.txt'))
        if not self.openai:
            object.__setattr__(self, 'openai', _load_credential('openai_api_key.txt'))
        if not self.perplexity:
            object.__setattr__(self, 'perplexity', _load_credential('perplexity_api_key.txt'))

    def validate(self) -> list[str]:
        """Return list of missing API keys."""
        missing = []
        if not self.tavily:
            missing.append("tavily (credentials/tavily_api_key.txt)")
        if not self.openai:
            missing.append("openai (credentials/openai_api_key.txt)")
        if not self.perplexity:
            missing.append("perplexity (credentials/perplexity_api_key.txt)")
        return missing
    
    def status(self) -> dict[str, bool]:
        """Return status of each API key."""
        return {
            "tavily": bool(self.tavily),
            "openai": bool(self.openai),
            "perplexity": bool(self.perplexity),
        }


# ─────────────────────────────────────────────────────────────────────────────
# COST TRACKING
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CostEstimates:
    """
    Per-query cost estimates for budget tracking.
    Values in USD.
    
    Note: These are conservative estimates. Actual costs depend on:
    - API pricing tier (volume discounts available)
    - Token usage (varies by company description length)
    - Search result complexity
    
    Last updated: January 2026
    """
    # Stage 1: Tavily search + GPT-4o-mini interpretation
    # Tavily: ~$0.01/search on Researcher plan, less on Scale
    # GPT-4o-mini: ~$0.15/1M input tokens, ~$0.60/1M output tokens
    tavily_search: float = 0.01       # Conservative estimate
    gpt4o_mini_call: float = 0.001    # ~500 tokens in, 200 out
    stage_1_total: float = 0.011      # Tavily dominates the cost

    # Stage 2A: Perplexity Sonar Base (llama-3.1-sonar-small)
    # $0.20/1M input, $0.20/1M output + $5/1000 searches
    sonar_base: float = 0.02

    # Stage 2B: Perplexity Sonar Pro (llama-3.1-sonar-large)  
    # $1/1M input, $1/1M output + $5/1000 searches
    sonar_pro_min: float = 0.05
    sonar_pro_max: float = 0.10

    # Stage 3: Perplexity Deep Research
    # Significantly more expensive - multiple searches + reasoning
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
