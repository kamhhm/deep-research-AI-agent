"""
Week 10 - Data Preparation Script
==================================
Purpose: Filter dataset to only include startups with BOTH short and long descriptions
         This creates the experimental dataset for testing short-description-only classification

Output: company_us_both_descriptions.csv (filtered dataset)
"""

import pandas as pd
import os

print("="*70)
print("WEEK 10 - DATA PREPARATION")
print("="*70)
print("Filtering dataset to startups with BOTH descriptions\n")

# Input: Original Week 9 dataset
INPUT_CSV = "../Week 9/company_us_short_long_desc_.csv"
OUTPUT_CSV = "company_us_both_descriptions.csv"

# Load the full dataset
print(f"[1] Loading original dataset...")
df = pd.read_csv(INPUT_CSV)
print(f"    Total startups: {len(df):,}")

# Count initial description status
print(f"\n[2] Analyzing description availability...")
has_short = (~df['short_description'].isna()) & (df['short_description'].str.strip() != '')
has_long = (~df['Long description'].isna()) & (df['Long description'].str.strip() != '')

print(f"    Startups with short description: {has_short.sum():,}")
print(f"    Startups with long description: {has_long.sum():,}")
print(f"    Startups with BOTH descriptions: {(has_short & has_long).sum():,}")

# Filter to only startups with BOTH descriptions
print(f"\n[3] Filtering to startups with BOTH descriptions...")
df_both = df[has_short & has_long].copy()
print(f"    Filtered dataset size: {len(df_both):,} startups")

# Save filtered dataset
print(f"\n[4] Saving filtered dataset...")
df_both.to_csv(OUTPUT_CSV, index=False)
print(f"    ✓ Saved to: {OUTPUT_CSV}")

# Summary statistics
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Original dataset:     {len(df):,} startups")
print(f"Filtered dataset:     {len(df_both):,} startups")
print(f"Percentage retained:  {len(df_both)/len(df)*100:.1f}%")
print(f"{'='*70}")
print("\n✓ Data preparation complete!")
print(f"\nNext steps:")
print(f"  1. Review {OUTPUT_CSV}")
print(f"  2. Run MTA_multi_batch_short_only.py to classify using SHORT descriptions only")
print(f"  3. Compare results with Week 9 (which used BOTH descriptions)")

