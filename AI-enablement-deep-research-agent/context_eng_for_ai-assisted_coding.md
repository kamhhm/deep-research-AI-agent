# AI-Enablement Deep Research Agent - Project Context

## Project Overview

I am a research assistant for finance professor Jan Bena at UBC. We are working with a dataset of approximately 44,000 US startups from Crunchbase. The goal is to find evidence of **GenAI adoption in internal business processes** (not AI-native products, but how companies use AI tools like ChatGPT, Claude, Copilot internally for operations).

This requires building a **deep research agent** that can:
- Find information on startups with minimal online presence
- Be cost-optimized (target: **<$0.10 average per company**)
- Produce structured, citation-backed findings

---

## Key Constraints & Decisions

| Constraint | Decision |
|------------|----------|
| Budget | < $0.10 average per company across 44k startups |
| Research Scope | **100% of companies must be researched** - no industry filtering (to avoid sampling bias in academic research) |
| DeepSeek | Cannot use DeepSeek models (banned in Vancouver) |
| Output | Highly structured CSV with source citations |

---

## Final Research Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STAGE 1: PRESENCE FILTER                            │
│                    (Deterministic, Non-Agentic, ~$0.002/company)            │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. HEAD request to company homepage (check if alive)                       │
│  2. ONE Tavily/Serper search (controlled, predictable cost)                 │
│  3. GPT-4o-mini interprets results → outputs:                               │
│     - online_presence_score (0-100)                                         │
│     - ai_mentions_found (boolean)                                           │
│     - reasoning (brief)                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
          [Low Presence OR                [Sufficient Presence AND
           No AI Mentions]                    AI Mentions Found]
                    │                               │
                    ▼                               ▼
┌─────────────────────────────────────┐  ┌─────────────────────────────────────┐
│      STAGE 2A: QUICK CHECK          │  │     STAGE 2B: SONAR PRO             │
│   Perplexity Sonar Base (~$0.02)    │  │        (~$0.05-0.10/query)          │
│   Brief search for any GenAI usage  │  │   Deeper search with more sources   │
└─────────────────────────────────────┘  └─────────────────────────────────────┘
                    │                               │
                    ▼                               ▼
          [Ambiguous Results?]           [Strong signals but needs
           AI mentions but                  more depth?]
           unclear details                          │
                    │                               │
                    └───────────┬───────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 3: DEEP RESEARCH (Selective)                       │
│              Perplexity Sonar Deep Research (~$0.41-1.19/query)             │
│                                                                             │
│  Only triggered when:                                                       │
│  - High online presence + AI mentions found but details unclear             │
│  - Sonar Pro found signals but couldn't confirm specifics                   │
│  - Company is high-priority (tech industry, large, well-funded)             │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FINAL OUTPUT: STRUCTURED CSV                        │
│                                                                             │
│  Columns per finding row:                                                   │
│  - company_id, company_name                                                 │
│  - finding_type (e.g., "chatgpt_customer_support", "copilot_engineering")   │
│  - description (specific details of GenAI adoption)                         │
│  - source_url (citation)                                                    │
│  - confidence_score                                                         │
│  - research_stage_reached (1, 2A, 2B, or 3)                                 │
│                                                                             │
│  If no adoption found: one row with finding_type = "no_adoption_found"      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Escalation Criteria (Simplified)

The escalation decision is based on **two simple indicators**:

1. **Online Presence Score** (0-100): Does the company have enough web presence to research?
2. **AI Mentions Found** (boolean): Did the initial search find any mention of AI/GenAI usage?

| Presence Score | AI Mentions | Action |
|----------------|-------------|--------|
| Low (<30) | No | → Stage 2A (quick check), then likely mark "no adoption found" |
| Low (<30) | Yes | → Stage 2B (Sonar Pro) to investigate the AI mention |
| Medium (30-70) | No | → Stage 2A (quick check) |
| Medium (30-70) | Yes | → Stage 2B (Sonar Pro) |
| High (>70) | No | → Stage 2B (Sonar Pro) - worth deeper look |
| High (>70) | Yes | → Stage 2B (Sonar Pro) → possibly Stage 3 if ambiguous |

**Only escalate to Stage 3 (Deep Research)** when there's high likelihood of finding specific adoption details that justify the $0.41-1.19 cost.

---

## Cost Estimates

| Stage | Model | Cost/Query | Expected % of Companies |
|-------|-------|------------|-------------------------|
| 1 | Tavily + GPT-4o-mini | ~$0.002 | 100% |
| 2A | Perplexity Sonar Base | ~$0.02 | ~40% (low presence) |
| 2B | Perplexity Sonar Pro | ~$0.05-0.10 | ~55% |
| 3 | Perplexity Deep Research | ~$0.41-1.19 | ~5% (selective) |

**Estimated weighted average: ~$0.05-0.08 per company** (under $0.10 target)

---

## Important Design Decisions

### Why Deterministic Pre-Filter (Not Agentic)?
- Agentic web search (GPT-4o + web tool) has **unpredictable costs** because the model decides how many searches to run
- Deterministic approach: exactly 1 Tavily search + 1 LLM call = predictable, low cost

### Why No Industry Filtering?
- Originally considered skipping "unlikely" industries (restaurants, real estate, etc.)
- **Rejected**: This would introduce sampling bias in academic research
- Instead: Non-tech industries get a "research interest boost" - we're especially interested if they adopt GenAI

### Why Perplexity Over Tavily/Serper Alone?
- Tested narrow search queries (Tavily-style) on companies with minimal presence
- Result: "Practically nothing" found for companies like Tread Partners, Fabius, MarCore Group
- Perplexity's **open-ended agentic search** can explore broadly and find information that narrow keyword searches miss

---

## Input Data

Previous classification results are in:
`/gen-ai-business-function-adoption/insights/genai_classifications_combined.csv`

This contains ~44k companies with columns:
- `company_id`, `company_name`
- `genai_strict_label`, `genai_moderate_label`, `genai_lenient_label`
- `reasoning`
- Company metadata (descriptions, categories, etc.)

**Note**: These previous classifications have many false negatives because they were batch-processed without web access - the classifier only saw Crunchbase descriptions.

---

## Next Steps (When Resuming)

1. **Implement Stage 1 Presence Filter**
   - Write the deterministic filter function (HEAD request + Tavily + GPT-4o-mini)
   - Test on 10-20 sample companies
   - Verify cost is ~$0.002/company

2. **Build Perplexity Integration**
   - Set up Sonar Base and Sonar Pro API calls
   - Design prompts for internal GenAI adoption detection
   - Handle structured output parsing

3. **Design CSV Output Schema**
   - Finalize column structure
   - Build aggregation logic (multiple findings per company)

4. **Pilot Run**
   - Run 50-100 companies through full pipeline
   - Validate cost estimates
   - Tune escalation thresholds based on results

5. **Scale to Full Dataset**
   - Implement batch processing with rate limiting
   - Add checkpointing/resume capability
   - Monitor costs in real-time

---

## Tools & APIs to Use

| Purpose | Tool |
|---------|------|
| Web Search (Stage 1) | Tavily or Serper API |
| LLM Interpretation | GPT-4o-mini (cheap, fast) |
| Research (Stage 2-3) | Perplexity Sonar API (Base, Pro, Deep Research) |
| Agent Framework | LangChain DeepAgents (optional, for Stage 3) |
| Orchestration | Python with async/batch processing |

---

