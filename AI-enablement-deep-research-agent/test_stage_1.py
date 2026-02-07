"""
Test Stage 1 pipeline on 50 random companies from the Crunchbase dataset.

Runs the full pipeline:
1. Website health check
2. Tavily search (advanced, 5 results)
3. GPT-4o-mini classification

Usage:
    python test_stage_1.py
"""

import asyncio
import csv
import random
import time
from pathlib import Path

# Add src to path so imports work
import sys
sys.path.insert(0, str(Path(__file__).parent))

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


async def main():
    # Load API keys
    keys = None
    try:
        from config import APIKeys
        keys = APIKeys()
    except ImportError:
        pass
    
    if not keys:
        # Manual load
        creds_dir = BASE_DIR / "credentials"
        tavily_key = ""
        openai_key = ""
        
        tavily_file = creds_dir / "tavily_api_key.txt"
        if tavily_file.exists():
            lines = [l.strip() for l in tavily_file.read_text().split('\n') 
                     if l.strip() and not l.strip().startswith('#')]
            if lines:
                tavily_key = lines[0]
        
        openai_file = creds_dir / "openai_api_key.txt"
        if openai_file.exists():
            lines = [l.strip() for l in openai_file.read_text().split('\n') 
                     if l.strip() and not l.strip().startswith('#')]
            if lines:
                openai_key = lines[0]
    else:
        tavily_key = keys.tavily
        openai_key = keys.openai
    
    print(f"Tavily key: {'✓' if tavily_key else '✗'}")
    print(f"OpenAI key: {'✓' if openai_key else '✗'}")
    
    if not tavily_key or not openai_key:
        print("ERROR: Missing API keys. Cannot proceed.")
        return
    
    # Load companies
    companies = load_random_companies(50)
    print(f"\nLoaded {len(companies)} random companies.")
    print(f"Estimated cost: ~50 x $0.021 = $1.05 (Tavily advanced + GPT-4o-mini)")
    
    input("\nPress Enter to start (or Ctrl+C to cancel)...")
    
    start_time = time.time()
    
    # Run Stage 1 for each company
    results = []
    for i, company in enumerate(companies, 1):
        name = company['name']
        homepage = company.get('homepage_url', '')
        desc = company.get('short_description', '')
        rcid = company.get('rcid', i)
        
        print(f"\n[{i}/50] {name}...", end=" ", flush=True)
        
        try:
            result = await run_stage_1(
                company_id=int(rcid) if str(rcid).isdigit() else i,
                company_name=name,
                homepage_url=homepage or None,
                company_description=desc or None,
                tavily_api_key=tavily_key,
                openai_api_key=openai_key
            )
            
            priority = result.research_priority
            score = result.presence_score
            reasoning = result.assessment.reasoning if result.assessment else ""
            
            # Color-coded priority
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
    
    # Website status
    alive = sum(1 for r in results if r['website_alive'])
    print(f"\nWebsite Status: {alive}/{len(results)} alive")
    
    # Search results
    search_counts = [r['search_results'] for r in results]
    zero_results = sum(1 for c in search_counts if c == 0)
    print(f"Search Results: {zero_results}/{len(results)} with zero results")
    
    # Save results to CSV
    output_file = BASE_DIR / "outputs" / "stage_1_test_results.csv"
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'name', 'homepage', 'description', 'priority', 'score',
            'reasoning', 'website_alive', 'search_results', 'error'
        ])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to: {output_file}")
    
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
