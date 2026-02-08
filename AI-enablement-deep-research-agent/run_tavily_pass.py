"""
Tavily Pass — Production runner for Stage 1 web search.

Reads the Crunchbase CSV, runs website checks + Tavily searches,
and writes results incrementally to a JSONL file.

Checkpointing is implicit: the JSONL file IS the checkpoint.
On resume, existing rcids in the JSONL are skipped automatically.

Output format (one JSON object per line in tavily_results.jsonl):
{
    "rcid": 12345,
    "name": "Acme Corp",
    "short_description": "...",
    "description": "...",
    "homepage_url": "https://...",
    "website_check": { "url": "...", "is_alive": true, ... },
    "tavily": { "query": "...", "result_count": 5, "raw_response": {...} },
    "timestamp": "2026-02-05T..."
}

Usage:
    python run_tavily_pass.py                    # Process all 44K companies
    python run_tavily_pass.py --limit 100        # First 100 only (testing)
    python run_tavily_pass.py --resume            # Auto-resume from existing JSONL
"""

import argparse
import asyncio
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    PROCESSING, STAGE1_OUTPUT_DIR,
    APIKeys, DATA_DIR,
)
from src.rate_limiter import AsyncRateLimiter
from src.stage_1_filter import (
    check_website,
    search_tavily,
    WebsiteStatus,
    TavilySearchResult,
)

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FILE
# ─────────────────────────────────────────────────────────────────────────────

TAVILY_JSONL = STAGE1_OUTPUT_DIR / "tavily_results.jsonl"
DATA_FILE = DATA_DIR / "44k_crunchbase_startups.csv"


def load_existing_rcids(jsonl_path: Path) -> set[int]:
    """Scan existing JSONL and return the set of rcids already processed."""
    seen = set()
    if not jsonl_path.exists():
        return seen
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                seen.add(int(obj["rcid"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return seen


def load_csv_companies(csv_path: Path, limit: int | None = None) -> list[dict]:
    """Load companies from the Crunchbase CSV."""
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if limit is not None:
        rows = rows[:limit]
    return rows


def build_tavily_record(
    company: dict,
    website_status: WebsiteStatus,
    search_result: TavilySearchResult,
) -> dict:
    """Build the JSONL record for one company."""
    ws = website_status
    sr = search_result
    return {
        "rcid": int(company["rcid"]),
        "name": company["name"],
        "short_description": company.get("short_description") or None,
        "description": company.get("description") or None,
        "homepage_url": company.get("homepage_url") or None,
        "website_check": {
            "url": ws.url,
            "is_alive": ws.is_alive,
            "status_code": ws.status_code,
            "final_url": ws.final_url,
            "error": ws.error,
        },
        "tavily": {
            "query": sr.query,
            "result_count": sr.result_count,
            "raw_response": sr.raw_response,
            "error": sr.error,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def process_company(
    company: dict,
    tavily_api_key: str,
    http_client,
    tavily_client,
) -> dict:
    """Run website check + Tavily search for a single company, return JSONL record."""
    name = company["name"]
    homepage = company.get("homepage_url") or ""

    # Website check + Tavily search (sequential for now — Commit 7 parallelizes)
    website_status = await check_website(homepage, client=http_client)
    search_result = await search_tavily(
        company_name=name,
        homepage_url=homepage or None,
        company_description=company.get("short_description") or None,
        api_key=tavily_api_key,
        client=tavily_client,
    )

    return build_tavily_record(company, website_status, search_result)


async def main():
    parser = argparse.ArgumentParser(description="Run Tavily pass on Crunchbase dataset")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N companies")
    parser.add_argument("--output", type=str, default=None, help="Custom output JSONL path")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else TAVILY_JSONL

    # Load API keys
    keys = APIKeys()
    if not keys.tavily:
        print("ERROR: Tavily API key not found. Add to credentials/tavily_api_key.txt")
        sys.exit(1)

    # Load dataset
    print(f"Loading dataset from {DATA_FILE}...")
    companies = load_csv_companies(DATA_FILE, limit=args.limit)
    total = len(companies)
    print(f"  Total companies in scope: {total}")

    # Check existing progress (implicit checkpoint)
    existing = load_existing_rcids(output_path)
    if existing:
        print(f"  Already processed: {len(existing)} (resuming)")
    remaining = [c for c in companies if int(c["rcid"]) not in existing]
    print(f"  Remaining to process: {len(remaining)}")

    if not remaining:
        print("\nAll companies already processed. Nothing to do.")
        return

    # Cost estimate
    cost = len(remaining) * 0.02  # ~$0.02/company (Tavily advanced)
    print(f"\nEstimated Tavily cost: ${cost:.2f} ({len(remaining)} searches x $0.02)")
    print(f"Output: {output_path}")

    input("\nPress Enter to start (or Ctrl+C to cancel)...")

    # Create shared HTTP clients
    import httpx
    http_client = httpx.AsyncClient(
        timeout=PROCESSING.http_timeout,
        follow_redirects=True,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0; +https://ubc.ca)",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    tavily_client = httpx.AsyncClient(timeout=PROCESSING.tavily_timeout)

    # Rate limiter + concurrency semaphore
    tavily_limiter = AsyncRateLimiter(rpm=PROCESSING.tavily_rpm, name="tavily")
    semaphore = asyncio.Semaphore(PROCESSING.max_concurrent_requests)

    start_time = time.time()
    processed = 0
    errors = 0
    write_lock = asyncio.Lock()

    # Open JSONL in append mode for the duration of the run
    jsonl_file = open(output_path, "a")

    async def process_one(company: dict, idx: int) -> None:
        nonlocal processed, errors
        name = company["name"]

        async with semaphore:
            # Acquire rate limit slot before the Tavily call
            await tavily_limiter.acquire()

            try:
                record = await process_company(
                    company, keys.tavily, http_client, tavily_client,
                )

                # Write under lock for concurrency safety
                async with write_lock:
                    jsonl_file.write(json.dumps(record) + "\n")
                    jsonl_file.flush()
                processed += 1

                result_count = record["tavily"]["result_count"]
                alive = "alive" if record["website_check"]["is_alive"] else "dead"
                print(f"  [{idx}/{len(remaining)}] {name}: {alive}, {result_count} results")

            except Exception as e:
                errors += 1
                print(f"  [{idx}/{len(remaining)}] {name}: ERROR — {e}")

    try:
        # Process in batches for memory safety and progress reporting
        batch_size = 200
        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start:batch_start + batch_size]
            tasks = [
                process_one(company, batch_start + j + 1)
                for j, company in enumerate(batch)
            ]
            await asyncio.gather(*tasks)

            # Progress report after each batch
            elapsed = time.time() - start_time
            done = batch_start + len(batch)
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - done) / rate if rate > 0 else 0
            print(f"\n  --- Progress: {done}/{len(remaining)} | "
                  f"{rate:.1f} companies/sec | "
                  f"ETA: {eta/60:.0f}min | "
                  f"Errors: {errors} | "
                  f"Rate limiter: {tavily_limiter.stats} ---\n")

    finally:
        jsonl_file.close()
        await http_client.aclose()
        await tavily_client.aclose()

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"  Tavily pass complete")
    print(f"  Processed: {processed}/{len(remaining)} ({errors} errors)")
    print(f"  Time: {elapsed:.1f}s ({processed/elapsed:.1f} companies/sec)")
    print(f"  Output: {output_path}")
    total_in_file = len(load_existing_rcids(output_path))
    print(f"  Total in JSONL: {total_in_file}/{total}")
    print(f"  Rate limiter stats: {tavily_limiter.stats}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
