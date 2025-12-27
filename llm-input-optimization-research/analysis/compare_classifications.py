"""
Week 10 Analysis - Compare Short-Only vs Both Descriptions
===========================================================
Purpose: Compare classification results from Week 10 (short description only)
         with Week 9 (both short and long descriptions)

Metrics:
- Overall agreement rate
- Agreement rate by confidence level
- Disagreement patterns (where short-only differs from both)
- Confidence level comparison
- AI-native classification rate comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

print("="*70)
print("WEEK 10 ANALYSIS: SHORT vs BOTH DESCRIPTIONS")
print("="*70)
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# SECTION 1: LOAD DATA
# ============================================================================

print("[1] Loading classification results...")

# Week 9: Both descriptions (filter to startups with both descriptions)
WEEK9_FULL = "../../Week 9/GPT-5-mini/processing/classified_startups_gpt5_mini.csv"
WEEK9_DATASET = "../../Week 9/company_us_short_long_desc_.csv"

# Week 10: Short description only
WEEK10_SHORT = "../GPT-5-mini/processing/classified_startups_short_only.csv"
WEEK10_DATASET = "../company_us_both_descriptions.csv"

# Load datasets
df_week9_full = pd.read_csv(WEEK9_FULL)
df_week9_dataset = pd.read_csv(WEEK9_DATASET)
df_week10_short = pd.read_csv(WEEK10_SHORT)
df_week10_dataset = pd.read_csv(WEEK10_DATASET)

print(f"    Week 9 (full): {len(df_week9_full):,} classifications")
print(f"    Week 10 (short only): {len(df_week10_short):,} classifications")

# ============================================================================
# SECTION 2: FILTER WEEK 9 TO MATCHING STARTUPS
# ============================================================================

print("\n[2] Filtering Week 9 to startups with both descriptions...")

# Get org_uuids from Week 10 dataset (these have both descriptions)
both_desc_uuids = set(df_week10_dataset['org_uuid'].values)
print(f"    Startups with both descriptions: {len(both_desc_uuids):,}")

# Filter Week 9 classifications to only these startups
df_week9_both = df_week9_full[df_week9_full['CompanyID'].isin(both_desc_uuids)].copy()
print(f"    Week 9 filtered: {len(df_week9_both):,} classifications")

# ============================================================================
# SECTION 3: MERGE AND ALIGN DATA
# ============================================================================

print("\n[3] Merging datasets for comparison...")

# Merge on CompanyID
df_merged = pd.merge(
    df_week9_both,
    df_week10_short,
    on='CompanyID',
    suffixes=('_week9_both', '_week10_short')
)

print(f"    Matched startups: {len(df_merged):,}")

# Convert classification columns to integers
df_merged['AI_native_week9_both'] = pd.to_numeric(df_merged['AI_native_week9_both'], errors='coerce')
df_merged['AI_native_week10_short'] = pd.to_numeric(df_merged['AI_native_week10_short'], errors='coerce')
df_merged['Confidence_1to5_week9_both'] = pd.to_numeric(df_merged['Confidence_1to5_week9_both'], errors='coerce')
df_merged['Confidence_1to5_week10_short'] = pd.to_numeric(df_merged['Confidence_1to5_week10_short'], errors='coerce')

# ============================================================================
# SECTION 4: CALCULATE AGREEMENT METRICS
# ============================================================================

print("\n[4] Calculating agreement metrics...")

# Overall agreement
agreement = (df_merged['AI_native_week9_both'] == df_merged['AI_native_week10_short'])
agreement_rate = agreement.sum() / len(df_merged) * 100

print(f"\n{'='*70}")
print("OVERALL AGREEMENT")
print(f"{'='*70}")
print(f"Agreement Rate: {agreement_rate:.2f}%")
print(f"Agreements: {agreement.sum():,} / {len(df_merged):,}")
print(f"Disagreements: {(~agreement).sum():,}")

# Agreement by confidence level (Week 9)
print(f"\n{'='*70}")
print("AGREEMENT BY CONFIDENCE LEVEL (Week 9 with Both Descriptions)")
print(f"{'='*70}")

for conf in sorted(df_merged['Confidence_1to5_week9_both'].dropna().unique()):
    mask = df_merged['Confidence_1to5_week9_both'] == conf
    if mask.sum() > 0:
        conf_agreement = (df_merged[mask]['AI_native_week9_both'] == 
                         df_merged[mask]['AI_native_week10_short']).sum()
        conf_total = mask.sum()
        conf_rate = conf_agreement / conf_total * 100
        print(f"Confidence {int(conf)}: {conf_rate:.2f}% ({conf_agreement:,}/{conf_total:,})")

# Agreement by confidence level (Week 10)
print(f"\n{'='*70}")
print("AGREEMENT BY CONFIDENCE LEVEL (Week 10 with Short Only)")
print(f"{'='*70}")

for conf in sorted(df_merged['Confidence_1to5_week10_short'].dropna().unique()):
    mask = df_merged['Confidence_1to5_week10_short'] == conf
    if mask.sum() > 0:
        conf_agreement = (df_merged[mask]['AI_native_week9_both'] == 
                         df_merged[mask]['AI_native_week10_short']).sum()
        conf_total = mask.sum()
        conf_rate = conf_agreement / conf_total * 100
        print(f"Confidence {int(conf)}: {conf_rate:.2f}% ({conf_agreement:,}/{conf_total:,})")

# ============================================================================
# SECTION 5: CLASSIFICATION RATE COMPARISON
# ============================================================================

print(f"\n{'='*70}")
print("CLASSIFICATION RATE COMPARISON")
print(f"{'='*70}")

week9_ai_rate = (df_merged['AI_native_week9_both'] == 1).sum() / len(df_merged) * 100
week10_ai_rate = (df_merged['AI_native_week10_short'] == 1).sum() / len(df_merged) * 100

print(f"Week 9 (Both Desc) AI-Native Rate:  {week9_ai_rate:.2f}% ({(df_merged['AI_native_week9_both'] == 1).sum():,}/{len(df_merged):,})")
print(f"Week 10 (Short Only) AI-Native Rate: {week10_ai_rate:.2f}% ({(df_merged['AI_native_week10_short'] == 1).sum():,}/{len(df_merged):,})")
print(f"Difference: {week10_ai_rate - week9_ai_rate:+.2f}%")

# ============================================================================
# SECTION 6: CONFIDENCE DISTRIBUTION COMPARISON
# ============================================================================

print(f"\n{'='*70}")
print("CONFIDENCE DISTRIBUTION COMPARISON")
print(f"{'='*70}")

week9_conf_dist = df_merged['Confidence_1to5_week9_both'].value_counts().sort_index()
week10_conf_dist = df_merged['Confidence_1to5_week10_short'].value_counts().sort_index()

print("\nWeek 9 (Both Descriptions):")
for conf, count in week9_conf_dist.items():
    print(f"  Confidence {int(conf)}: {count:,} ({count/len(df_merged)*100:.1f}%)")

print("\nWeek 10 (Short Only):")
for conf, count in week10_conf_dist.items():
    print(f"  Confidence {int(conf)}: {count:,} ({count/len(df_merged)*100:.1f}%)")

week9_avg_conf = df_merged['Confidence_1to5_week9_both'].mean()
week10_avg_conf = df_merged['Confidence_1to5_week10_short'].mean()

print(f"\nAverage Confidence:")
print(f"  Week 9 (Both):  {week9_avg_conf:.2f}")
print(f"  Week 10 (Short): {week10_avg_conf:.2f}")
print(f"  Difference: {week10_avg_conf - week9_avg_conf:+.2f}")

# ============================================================================
# SECTION 7: DISAGREEMENT ANALYSIS
# ============================================================================

print(f"\n{'='*70}")
print("DISAGREEMENT ANALYSIS")
print(f"{'='*70}")

# Create disagreement dataset
df_disagree = df_merged[~agreement].copy()

print(f"Total disagreements: {len(df_disagree):,}")

# Pattern 1: Both→Short changed from Not AI to AI
both_not_short_ai = (df_disagree['AI_native_week9_both'] == 0) & (df_disagree['AI_native_week10_short'] == 1)
print(f"\nNot AI (Both) → AI (Short): {both_not_short_ai.sum():,} ({both_not_short_ai.sum()/len(df_disagree)*100:.1f}%)")

# Pattern 2: Both→Short changed from AI to Not AI
both_ai_short_not = (df_disagree['AI_native_week9_both'] == 1) & (df_disagree['AI_native_week10_short'] == 0)
print(f"AI (Both) → Not AI (Short): {both_ai_short_not.sum():,} ({both_ai_short_not.sum()/len(df_disagree)*100:.1f}%)")

# Save disagreement cases for manual review
print(f"\n[5] Saving disagreement dataset...")
df_disagree_export = df_disagree[[
    'CompanyID',
    'CompanyName_week9_both',
    'AI_native_week9_both',
    'Confidence_1to5_week9_both',
    'AI_native_week10_short',
    'Confidence_1to5_week10_short',
    'Reasons_3_points_week9_both',
    'Reasons_3_points_week10_short'
]]

df_disagree_export.columns = [
    'CompanyID',
    'CompanyName',
    'AI_native_Both',
    'Confidence_Both',
    'AI_native_Short',
    'Confidence_Short',
    'Reasons_Both',
    'Reasons_Short'
]

df_disagree_export.to_csv("disagreements_both_vs_short.csv", index=False)
print(f"    Saved to: disagreements_both_vs_short.csv")

# Save full comparison dataset
df_merged_export = df_merged[[
    'CompanyID',
    'CompanyName_week9_both',
    'AI_native_week9_both',
    'Confidence_1to5_week9_both',
    'AI_native_week10_short',
    'Confidence_1to5_week10_short',
    'Reasons_3_points_week9_both',
    'Reasons_3_points_week10_short'
]]

df_merged_export.columns = [
    'CompanyID',
    'CompanyName',
    'AI_native_Both',
    'Confidence_Both',
    'AI_native_Short',
    'Confidence_Short',
    'Reasons_Both',
    'Reasons_Short'
]

df_merged_export.to_csv("full_comparison_both_vs_short.csv", index=False)
print(f"    Saved to: full_comparison_both_vs_short.csv")

# ============================================================================
# SECTION 8: SUMMARY AND INTERPRETATION
# ============================================================================

print(f"\n{'='*70}")
print("SUMMARY AND INTERPRETATION")
print(f"{'='*70}")

print(f"\n1. Overall Agreement: {agreement_rate:.2f}%")
if agreement_rate >= 95:
    print("   → Excellent! Short descriptions provide nearly identical classifications.")
elif agreement_rate >= 90:
    print("   → Very good! Short descriptions are highly sufficient for classification.")
elif agreement_rate >= 85:
    print("   → Good! Short descriptions are generally sufficient with minor differences.")
elif agreement_rate >= 80:
    print("   → Moderate! Short descriptions may miss some context in edge cases.")
else:
    print("   → Low! Long descriptions provide significant additional value.")

print(f"\n2. AI-Native Classification Rate:")
print(f"   Week 9 (Both):  {week9_ai_rate:.2f}%")
print(f"   Week 10 (Short): {week10_ai_rate:.2f}%")
if abs(week10_ai_rate - week9_ai_rate) < 1:
    print("   → Nearly identical rates - short descriptions capture AI-native status well.")
elif week10_ai_rate > week9_ai_rate:
    print("   → Short-only classifies MORE as AI-native (may be less conservative).")
else:
    print("   → Short-only classifies FEWER as AI-native (may be more conservative).")

print(f"\n3. Average Confidence:")
print(f"   Week 9 (Both):  {week9_avg_conf:.2f}")
print(f"   Week 10 (Short): {week10_avg_conf:.2f}")
if week10_avg_conf < week9_avg_conf:
    print("   → Lower confidence with short-only (expected due to less information).")
elif week10_avg_conf > week9_avg_conf:
    print("   → Higher confidence with short-only (unexpected, investigate further).")
else:
    print("   → Similar confidence levels.")

print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}")
print(f"The analysis shows {agreement_rate:.2f}% agreement between classifications")
print(f"using short descriptions only vs. both descriptions.")
print(f"\nRecommendation:")
if agreement_rate >= 95:
    print("  ✓ SHORT DESCRIPTIONS ARE SUFFICIENT for accurate AI-native classification")
    print("  ✓ Can save costs and reduce token usage by using short descriptions only")
elif agreement_rate >= 90:
    print("  ✓ SHORT DESCRIPTIONS ARE HIGHLY SUFFICIENT for most use cases")
    print("  ✓ Consider using short-only for initial classification, both for edge cases")
elif agreement_rate >= 85:
    print("  → SHORT DESCRIPTIONS ARE GENERALLY SUFFICIENT but with some gaps")
    print("  → Long descriptions add value in ambiguous cases")
else:
    print("  ✗ LONG DESCRIPTIONS PROVIDE SIGNIFICANT VALUE")
    print("  ✗ Recommend continuing to use both descriptions for accuracy")

print(f"\n{'='*70}")
print("Analysis complete!")
print(f"{'='*70}")

