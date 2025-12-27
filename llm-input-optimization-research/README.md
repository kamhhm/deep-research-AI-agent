# LLM Input Optimization Research

## Research Question

**Are short descriptions alone sufficient for accurate AI-native classification, or do long descriptions provide significant additional value?**

## Overview

This research project investigates whether reducing LLM input context (using only short descriptions instead of both short and long descriptions) maintains classification accuracy while reducing computational costs. The study compares classification results from two approaches on a dataset of 158,347 startups.

## Experimental Design

| Aspect | Baseline | Optimized |
|--------|----------|-----------|
| **Dataset** | 269,322 startups | 158,347 startups (with both descriptions) |
| **Input to LLM** | Short + Long descriptions | Short description only |
| **Model** | GPT-5-mini | GPT-5-mini (same) |
| **Tokens per request** | ~3,600 | ~3,400 |
| **Token savings** | - | 5.5% reduction |

### Hypothesis

Short descriptions contain sufficient information for accurate classification, enabling:
- 5-10% token cost reduction
- Faster processing (smaller payloads)
- Minimal accuracy loss (<5%)

### Success Criteria

- **Primary:** Agreement rate ≥95% between short-only and both descriptions
- **Secondary:** AI-native classification rate difference <2%

## Methodology

1. **Data Preparation:** Filtered dataset to include only startups with both short and long descriptions (158,347 startups) to ensure fair comparison

2. **Classification:** Re-classified all startups using only short descriptions, maintaining identical system prompts and model parameters

3. **Comparison:** Analyzed agreement rates, confidence distributions, and classification differences between baseline and optimized approaches

### Controlled Variables

- Same system prompt
- Same model (GPT-5-mini)
- Same startups (158,347 with both descriptions)
- Same classification criteria

### Key Metrics Analyzed

- Overall agreement rate between classifications
- Agreement by confidence level
- AI-native classification rate differences
- Confidence level changes
- Disagreement patterns

## Key Findings

### Overall Agreement Rate

- **≥95%:** Short descriptions sufficient for production use
- **90-95%:** Very good agreement with minor gaps
- **85-90%:** Good agreement but some context lost
- **<85%:** Long descriptions provide significant value

### Cost-Benefit Analysis

| Approach | Tokens per Request | Total Tokens (158,347) |
|----------|-------------------|------------------------|
| Both descriptions | 3,600 | 569.8M |
| Short only | 3,400 | 538.4M |
| **Savings** | **200 (5.5%)** | **31.4M (5.5%)** |

### Decision Matrix

| Agreement Rate | Cost Savings | Accuracy Loss | Recommendation |
|---------------|--------------|---------------|----------------|
| ≥95% | 5.5% | <5% | Use short-only |
| 90-95% | 5.5% | 5-10% | Case-by-case basis |
| 85-90% | 5.5% | 10-15% | Use both for accuracy |
| <85% | 5.5% | >15% | Use both descriptions |

## Expected Outcomes

### High Agreement (≥95%)
- **Finding:** Short descriptions are sufficient for accurate classification
- **Implication:** Can adopt short-only approach for production
- **Benefit:** 5.5% token cost reduction with <5% accuracy loss

### Good Agreement (90-95%)
- **Finding:** Short descriptions work well for most cases
- **Implication:** Use short-only for initial classification, both for low-confidence cases
- **Benefit:** Faster first-pass classification with targeted refinement

### Moderate Agreement (85-90%)
- **Finding:** Long descriptions add value in some cases
- **Implication:** Use both descriptions for accuracy-critical applications
- **Trade-off:** Cost savings not worth accuracy reduction

### Low Agreement (<85%)
- **Finding:** Long descriptions are critical for accurate classification
- **Implication:** Continue using both descriptions
- **Conclusion:** 5.5% cost savings not justified

## Research Impact

This experiment addresses key questions in LLM optimization:

1. **Cost Optimization:** Can we reduce token usage without sacrificing accuracy?
2. **Data Requirements:** What's the minimum information needed for accurate classification?
3. **Model Behavior:** How does the LLM perform with limited context?
4. **Confidence Calibration:** Does the model appropriately adjust confidence with less information?
5. **Edge Case Identification:** Which cases require detailed descriptions?

## Technical Details

- **Model:** GPT-5-mini via OpenAI Batch API
- **Dataset Size:** 158,347 startups
- **Processing Method:** Batch API for scalable processing
- **Analysis:** Statistical comparison of classification agreement and confidence distributions

## Project Structure

```
llm-input-optimization-research/
├── prepare_dataset.py              # Data preparation script
├── analysis/
│   └── compare_classifications.py  # Comparison analysis
├── GPT-5-mini/
│   └── processing/
│       ├── MTA_multi_batch_short_only.py  # Classification script
│       └── helper_scripts/         # Utility scripts
└── system_prompt.txt               # LLM system prompt
```

## Results & Analysis

The analysis produces:

1. **Agreement Metrics:** Overall agreement rate and agreement by confidence level
2. **Comparison Datasets:** Side-by-side comparison of baseline vs. optimized classifications
3. **Disagreement Analysis:** Detailed breakdown of cases where classifications differ

---
