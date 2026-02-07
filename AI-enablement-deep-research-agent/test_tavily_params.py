"""
Test script to compare Tavily search_depth: basic vs advanced.

Uses 5 random startups from the Crunchbase dataset.

- basic (1 credit): Returns NLP summary per URL
- advanced (2 credits): Returns reranked chunks per URL

Usage:
    python test_tavily_params.py

Requires: TAVILY_API_KEY in credentials/tavily_api_key.txt
"""

import asyncio
import csv
import random
from pathlib import Path
from urllib.parse import urlparse

import httpx

# Paths
BASE_DIR = Path(__file__).parent
CREDENTIALS_DIR = BASE_DIR / "credentials"
API_KEY_FILE = CREDENTIALS_DIR / "tavily_api_key.txt"
DATA_FILE = BASE_DIR / "crunchbase_data" / "44k_crunchbase_startups.csv"


def load_api_key() -> str:
    if API_KEY_FILE.exists():
        content = API_KEY_FILE.read_text().strip()
        lines = [line.strip() for line in content.split('\n') 
                 if line.strip() and not line.strip().startswith('#')]
        if lines:
            return lines[0]
    raise FileNotFoundError(f"Add your Tavily API key to {API_KEY_FILE}")


def load_random_companies(n: int = 5) -> list[dict]:
    """Load n random companies from the Crunchbase dataset."""
    with open(DATA_FILE, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return random.sample(rows, n)


def build_query(company: dict) -> str:
    """Build a search query for a company."""
    name = company['name']
    desc = company.get('short_description', '')
    url = company.get('homepage_url', '')
    
    parts = [f'"{name}"']
    
    # Add first ~60 chars of description
    if desc:
        excerpt = desc[:60].split('.')[0].strip()
        if len(excerpt) > 10:
            parts.append(f'"{excerpt}"')
    
    # Add domain constraint
    if url:
        try:
            domain = urlparse(url).netloc.replace("www.", "")
            if domain:
                parts.append(f"site:{domain}")
        except Exception:
            pass
    
    return " ".join(parts)


async def search_tavily(query: str, api_key: str, search_depth: str = "basic") -> dict:
    """Run a Tavily search with specified depth."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": search_depth,
            "max_results": 5,
            "include_answer": False,
            "include_raw_content": False,
        }
        
        response = await client.post("https://api.tavily.com/search", json=payload)
        response.raise_for_status()
        return response.json()


def print_results(results: dict, depth: str):
    """Print search results summary."""
    num_results = len(results.get('results', []))
    response_time = results.get('response_time', 'N/A')
    
    total_content = sum(len(r.get('content', '')) for r in results.get('results', []))
    
    print(f"  {depth.upper():8} | {num_results} results | {response_time:.2f}s | {total_content:,} chars total")
    
    # Show first result preview
    if results.get('results'):
        first = results['results'][0]
        content_preview = first.get('content', '')[:150].replace('\n', ' ')
        print(f"           First result: {first.get('title', 'N/A')[:50]}")
        print(f"           Preview: {content_preview}...")


async def test_company(company: dict, api_key: str):
    """Test basic vs advanced for a single company."""
    query = build_query(company)
    
    print(f"\n{'='*80}")
    print(f"Company: {company['name']}")
    print(f"URL: {company.get('homepage_url', 'N/A')}")
    print(f"Description: {company.get('short_description', 'N/A')[:100]}...")
    print(f"Query: {query}")
    print("-" * 80)
    
    # Test both depths
    try:
        basic = await search_tavily(query, api_key, "basic")
        print_results(basic, "basic")
    except Exception as e:
        print(f"  BASIC    | Error: {e}")
        basic = None
    
    await asyncio.sleep(0.5)
    
    try:
        advanced = await search_tavily(query, api_key, "advanced")
        print_results(advanced, "advanced")
    except Exception as e:
        print(f"  ADVANCED | Error: {e}")
        advanced = None
    
    return {
        'company': company['name'],
        'basic': basic,
        'advanced': advanced
    }


async def main():
    api_key = load_api_key()
    print(f"Loaded API key: {api_key[:10]}...")
    
    # Load random companies
    companies = load_random_companies(5)
    
    print(f"\nSelected {len(companies)} random companies:")
    for c in companies:
        print(f"  - {c['name']}")
    
    print(f"\nThis will use ~15 API credits (5 basic + 10 advanced).")
    input("\nPress Enter to continue (or Ctrl+C to cancel)...")
    
    # Test each company
    results = []
    for company in companies:
        result = await test_company(company, api_key)
        results.append(result)
        await asyncio.sleep(1)
    
    # Summary
    print("\n" + "="*80)
    print("  SUMMARY")
    print("="*80)
    
    basic_total_results = 0
    advanced_total_results = 0
    basic_total_chars = 0
    advanced_total_chars = 0
    
    for r in results:
        if r['basic']:
            basic_total_results += len(r['basic'].get('results', []))
            basic_total_chars += sum(len(x.get('content', '')) for x in r['basic'].get('results', []))
        if r['advanced']:
            advanced_total_results += len(r['advanced'].get('results', []))
            advanced_total_chars += sum(len(x.get('content', '')) for x in r['advanced'].get('results', []))
    
    print(f"""
Across {len(companies)} companies:

BASIC (5 credits total):
  - Total results: {basic_total_results}
  - Total content: {basic_total_chars:,} chars
  - Avg per company: {basic_total_chars // len(companies):,} chars

ADVANCED (10 credits total):
  - Total results: {advanced_total_results}
  - Total content: {advanced_total_chars:,} chars
  - Avg per company: {advanced_total_chars // len(companies):,} chars

Content ratio (advanced/basic): {advanced_total_chars / basic_total_chars:.1f}x
""")


if __name__ == "__main__":
    asyncio.run(main())
