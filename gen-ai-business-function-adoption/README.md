# GenAI Adoption Classification Research

## Research Question

**Can we accurately detect Generative AI (GenAI) adoption in startups using only their Crunchbase profile descriptions, without requiring web searches or external data?**

## Overview

This research project develops a scalable method to classify startup GenAI adoption using LLM-based analysis of publicly available Crunchbase data. The system analyzes company descriptions, categories, and metadata to identify evidence of GenAI usage across different business functions.

## Methodology

### Data Source
- **Dataset:** 44,000+ Crunchbase startups
- **Input Fields:** Company name, short description, long description, category list
- **Model:** GPT-5-mini via OpenAI Batch API

### Classification Approach

The system uses a three-tier assessment framework to classify GenAI adoption:

1. **Strict Mode** - Only counts explicit GenAI mentions (e.g., "GPT-based", "LLM-powered", "generative AI")
2. **Moderate Mode** - Includes explicit mentions plus strong implications from described functionality
3. **Lenient Mode** - Includes explicit mentions, strong implications, and suggestive generative-like capabilities

### Business Function Detection

When GenAI adoption is detected, the system identifies specific business functions:
- Customer support, Sales, Marketing, Product management
- Software development, Data analytics, Operations
- Supply chain/logistics, HR/Recruiting, Training/Onboarding
- Compliance/Legal, Finance/Accounting, Founder workflow

## Key Features

- **Scalable Processing:** Handles large datasets via OpenAI Batch API
- **Multi-Mode Assessment:** Provides three confidence levels for each classification
- **Function-Level Analysis:** Identifies specific business areas where GenAI is used
- **Automated Dashboard:** Generates interactive visualizations of results

## Project Structure

```
genai-adoption-classification/
├── data/
│   └── 44k_crunchbase_startups.csv      # Input dataset
├── prompts/
│   └── Jan_draft_prompt.txt             # LLM system prompt
├── scripts/
│   ├── batch_processor_w14.py           # Main classification script
│   ├── convert_results.py               # Result conversion utilities
│   └── generate_dashboard.py            # Dashboard generation
├── results/                              # Classification outputs
└── insights/                             # Generated visualizations
```

## Technical Details

- **Processing Method:** OpenAI Batch API for cost-effective large-scale processing
- **Output Format:** JSON with structured classification results
- **Analysis:** Statistical comparison across assessment modes
- **Visualization:** Interactive HTML dashboard with adoption metrics

## Research Impact

This project addresses key questions in startup technology adoption research:

1. **Data Efficiency:** Can we classify technology adoption without expensive web scraping?
2. **LLM Accuracy:** How reliable are LLMs for structured classification tasks?
3. **Adoption Patterns:** What business functions show highest GenAI adoption rates?
4. **Confidence Calibration:** How do different assessment strictness levels affect results?

## Results

The classification system produces:
- Adoption rates across strict, moderate, and lenient modes
- Business function distribution of GenAI usage
- Confidence level distributions
- Comparative analysis of assessment approaches

---

**For implementation details, see the scripts in the `scripts/` directory.**

