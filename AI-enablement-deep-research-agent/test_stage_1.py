"""
Test Stage 1 pipeline on 50 random companies from the Crunchbase dataset.

Runs the full pipeline:
1. Website health check
2. Tavily search (advanced, 5 results)
3. GPT-4o-mini classification

Outputs:
- outputs/stage1/stage1_run_<timestamp>.jsonl  (full logs per company)
- outputs/stage1/stage1_run_<timestamp>.csv    (summary CSV)

Usage:
    python test_stage_1.py
"""

import asyncio
import csv
import random
import time
from datetime import datetime
from pathlib import Path

# Add src to path so imports work
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.config import STAGE1_OUTPUT_DIR, STAGE1_TAVILY_DIR, STAGE1_GPT_DIR
from src.stage_1_filter import run_stage_1

# Paths
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "crunchbase_data" / "44k_crunchbase_startups.csv"


def load_random_companies(n: int = 50) -> list[dict]:
    """Load n random companies from the Crunchbase dataset."""
    with open(DATA_FILE, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return random.sample(rows, n)


def load_companies_by_ids(company_ids: list[int]) -> list[dict]:
    """Load specific companies by their rcid from the Crunchbase dataset."""
    id_set = set(str(cid) for cid in company_ids)
    with open(DATA_FILE, 'r') as f:
        reader = csv.DictReader(f)
        matches = [row for row in reader if row.get('rcid', '') in id_set]
    # Preserve the original order
    id_to_row = {row['rcid']: row for row in matches}
    return [id_to_row[str(cid)] for cid in company_ids if str(cid) in id_to_row]


async def main():
    # Load API keys
    from src.config import APIKeys
    keys = APIKeys()
    
    tavily_key = keys.tavily
    openai_key = keys.openai
    
    print(f"Tavily key: {'✓' if tavily_key else '✗'}")
    print(f"OpenAI key: {'✓' if openai_key else '✗'}")
    
    if not tavily_key or not openai_key:
        print("ERROR: Missing API keys. Cannot proceed.")
        return
    
    # Load companies — reuse from a previous run if JSONL path provided, else random
    REUSE_RUN = STAGE1_GPT_DIR / "gpt_20260207_150002.jsonl"  # Set to None for random
    
    if REUSE_RUN and REUSE_RUN.exists():
        import json as json_mod
        with open(REUSE_RUN) as f:
            prev_ids = [json_mod.loads(line)['company_id'] for line in f]
        companies = load_companies_by_ids(prev_ids)
        N = len(companies)
        print(f"\nReusing {N} companies from previous run: {REUSE_RUN.name}")
    else:
        N = 100
        companies = load_random_companies(N)
        print(f"\nLoaded {N} random companies.")
    
    print(f"Estimated cost: ~{N} x $0.021 = ${N * 0.021:.2f} (Tavily advanced + GPT-4o-mini)")
    
    # Output paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tavily_log_path = STAGE1_TAVILY_DIR / f"tavily_{timestamp}.jsonl"
    gpt_log_path = STAGE1_GPT_DIR / f"gpt_{timestamp}.jsonl"
    csv_path = STAGE1_OUTPUT_DIR / f"stage1_run_{timestamp}.csv"
    
    print(f"\nTavily log: {tavily_log_path}")
    print(f"GPT log:    {gpt_log_path}")
    print(f"Summary:    {csv_path}")
    
    input("\nPress Enter to start (or Ctrl+C to cancel)...")
    
    start_time = time.time()
    
    # Open log files and run pipeline
    results = []
    with open(tavily_log_path, 'w') as tavily_log, open(gpt_log_path, 'w') as gpt_log:
        for i, company in enumerate(companies, 1):
            name = company['name']
            homepage = company.get('homepage_url', '')
            desc = company.get('short_description', '')
            rcid = company.get('rcid', i)
            
            print(f"\n[{i}/{N}] {name}...", end=" ", flush=True)
            
            try:
                result = await run_stage_1(
                    company_id=int(rcid) if str(rcid).isdigit() else i,
                    company_name=name,
                    homepage_url=homepage or None,
                    company_description=desc or None,
                    tavily_api_key=tavily_key,
                    openai_api_key=openai_key,
                    tavily_log_file=tavily_log,
                    gpt_log_file=gpt_log
                )
                
                priority = result.research_priority
                score = result.presence_score
                reasoning = result.assessment.reasoning if result.assessment else ""
                
                priority_display = {
                    "high": f"HIGH ({score})",
                    "medium": f"MEDIUM ({score})",
                    "low": f"LOW ({score})",
                    "skip": f"SKIP ({score})",
                }.get(priority, priority)
                
                print(f"{priority_display} — {reasoning[:60]}")
                
                results.append({
                    'name': name,
                    'homepage': homepage,
                    'description': desc[:80],
                    'priority': priority,
                    'score': score,
                    'reasoning': reasoning,
                    'website_alive': result.website_status.is_alive if result.website_status else False,
                    'search_results': result.search_result.result_count if result.search_result else 0,
                    'error': result.assessment.error if result.assessment else None,
                })
                
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    'name': name,
                    'homepage': homepage,
                    'description': desc[:80],
                    'priority': 'error',
                    'score': 0,
                    'reasoning': str(e),
                    'website_alive': False,
                    'search_results': 0,
                    'error': str(e),
                })
            
            # Small delay to avoid rate limits
            await asyncio.sleep(0.5)
    
    elapsed = time.time() - start_time
    
    # Print summary
    print("\n" + "="*80)
    print(f"  STAGE 1 TEST RESULTS — {len(results)} companies in {elapsed:.1f}s")
    print("="*80)
    
    # Distribution
    from collections import Counter
    priority_counts = Counter(r['priority'] for r in results)
    
    print(f"\nPriority Distribution:")
    for priority in ['high', 'medium', 'low', 'skip', 'error']:
        count = priority_counts.get(priority, 0)
        pct = count / len(results) * 100
        bar = '█' * int(pct / 2)
        print(f"  {priority:8} {count:3} ({pct:5.1f}%) {bar}")
    
    # Score distribution
    scores = [r['score'] for r in results if r['priority'] != 'error']
    if scores:
        print(f"\nPresence Scores:")
        print(f"  Min: {min(scores)}, Max: {max(scores)}, Avg: {sum(scores)/len(scores):.0f}")
        
        # Histogram to check for clustering
        print(f"\n  Score histogram:")
        buckets = {}
        for s in scores:
            bucket = (s // 10) * 10
            buckets[bucket] = buckets.get(bucket, 0) + 1
        for bucket in sorted(buckets.keys()):
            count = buckets[bucket]
            print(f"    {bucket:3}-{bucket+9:3}: {'█' * count} ({count})")
    
    # Website status
    alive = sum(1 for r in results if r['website_alive'])
    print(f"\nWebsite Status: {alive}/{len(results)} alive")
    
    # Search results
    search_counts = [r['search_results'] for r in results]
    zero_results = sum(1 for c in search_counts if c == 0)
    print(f"Search Results: {zero_results}/{len(results)} with zero results")
    
    # Save summary CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'name', 'homepage', 'description', 'priority', 'score',
            'reasoning', 'website_alive', 'search_results', 'error'
        ])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nTavily log: {tavily_log_path}")
    print(f"GPT log:    {gpt_log_path}")
    print(f"Summary:    {csv_path}")
    
    # Show high priority companies
    high = [r for r in results if r['priority'] == 'high']
    if high:
        print(f"\n--- HIGH PRIORITY ({len(high)}) ---")
        for r in high:
            print(f"  {r['name']} (score={r['score']}): {r['reasoning'][:70]}")
    
    # Show skipped companies
    skipped = [r for r in results if r['priority'] == 'skip']
    if skipped:
        print(f"\n--- SKIPPED ({len(skipped)}) ---")
        for r in skipped:
            print(f"  {r['name']} (score={r['score']}): {r['reasoning'][:70]}")


if __name__ == "__main__":
    asyncio.run(main())
