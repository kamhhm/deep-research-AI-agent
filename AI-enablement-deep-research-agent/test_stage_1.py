"""
Test Stage 1 pipeline on 50 random companies from the Crunchbase dataset.

Runs the full pipeline:
1. Website health check
2. Tavily search (advanced, 5 results)
3. GPT-5-nano classification

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
from src.stage_1 import run_stage_1

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
    
    print(f"Estimated cost: ~{N} x $0.020 = ${N * 0.020:.2f} (Tavily advanced + GPT-5-nano)")
    
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
            long_desc = company.get('description', '')
            rcid = company.get('rcid', i)
            
            print(f"\n[{i}/{N}] {name}...", end=" ", flush=True)
            
            try:
                result = await run_stage_1(
                    company_id=int(rcid) if str(rcid).isdigit() else i,
                    company_name=name,
                    homepage_url=homepage or None,
                    company_description=desc or None,
                    long_description=long_desc or None,
                    tavily_api_key=tavily_key,
                    openai_api_key=openai_key,
                    tavily_log_file=tavily_log,
                    gpt_log_file=gpt_log
                )
                
                priority_score = result.research_priority_score
                presence = result.presence_score
                reasoning = result.assessment.reasoning if result.assessment else ""
                
                print(f"P{priority_score} (presence={presence}) — {reasoning[:60]}")
                
                results.append({
                    'name': name,
                    'homepage': homepage,
                    'description': desc[:80],
                    'priority_score': priority_score,
                    'presence_score': presence,
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
                    'priority_score': -1,
                    'presence_score': 0,
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
    
    # Research Priority Score distribution (0-5)
    from collections import Counter
    valid_results = [r for r in results if r['priority_score'] >= 0]
    priority_counts = Counter(r['priority_score'] for r in valid_results)
    error_count = sum(1 for r in results if r['priority_score'] < 0)
    
    labels = {
        5: "5 (definitely yields)",
        4: "4 (potentially yields)",
        3: "3 (worth a shot)",
        2: "2 (not worth it)",
        1: "1 (mostly unrelated)",
        0: "0 (not researchable)",
    }
    
    print(f"\nResearch Priority Score Distribution (0-5):")
    for score in [5, 4, 3, 2, 1, 0]:
        count = priority_counts.get(score, 0)
        pct = count / len(results) * 100
        bar = '█' * int(pct / 2)
        print(f"  {labels[score]:25} {count:3} ({pct:5.1f}%) {bar}")
    if error_count:
        pct = error_count / len(results) * 100
        print(f"  {'error':25} {error_count:3} ({pct:5.1f}%)")
    
    # Deep research candidates (score >= 3)
    deep_research = sum(1 for r in valid_results if r['priority_score'] >= 3)
    print(f"\n  Deep research candidates (score >= 3): {deep_research}/{len(results)} ({deep_research/len(results)*100:.0f}%)")
    
    # Presence score distribution
    presence_scores = [r['presence_score'] for r in valid_results]
    if presence_scores:
        print(f"\nPresence Scores (1-10):")
        print(f"  Min: {min(presence_scores)}, Max: {max(presence_scores)}, Avg: {sum(presence_scores)/len(presence_scores):.1f}")
    
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
            'name', 'homepage', 'description', 'priority_score', 'presence_score',
            'reasoning', 'website_alive', 'search_results', 'error'
        ])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nTavily log: {tavily_log_path}")
    print(f"GPT log:    {gpt_log_path}")
    print(f"Summary:    {csv_path}")
    
    # Show deep research candidates (score >= 3)
    candidates = [r for r in valid_results if r['priority_score'] >= 3]
    if candidates:
        candidates.sort(key=lambda r: r['priority_score'], reverse=True)
        print(f"\n--- DEEP RESEARCH CANDIDATES (score >= 3): {len(candidates)} ---")
        for r in candidates:
            print(f"  P{r['priority_score']} | {r['name']} (presence={r['presence_score']}): {r['reasoning'][:65]}")
    
    # Show not researchable (score 0)
    not_researchable = [r for r in valid_results if r['priority_score'] == 0]
    if not_researchable:
        print(f"\n--- NOT RESEARCHABLE (score 0): {len(not_researchable)} ---")
        for r in not_researchable:
            print(f"  {r['name']} (presence={r['presence_score']}): {r['reasoning'][:70]}")


if __name__ == "__main__":
    asyncio.run(main())
