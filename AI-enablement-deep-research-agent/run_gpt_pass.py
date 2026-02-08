"""
GPT Pass — Classify companies using GPT-5-nano against Tavily results.

Reads the Tavily JSONL (output of run_tavily_pass.py), reconstructs the
data structures, and runs GPT classification. Results are written to a
separate JSONL file.

This is intentionally decoupled from the Tavily pass so you can:
- Re-run GPT with different prompts without re-running Tavily
- Compare prompt versions side by side
- Iterate on the classifier without incurring Tavily costs

Output format (one JSON object per line):
{
    "rcid": 12345,
    "name": "Acme Corp",
    "online_presence_score": 7,
    "research_priority_score": 4,
    "reasoning": "...",
    "error": null
}

Usage:
    python run_gpt_pass.py                           # Use default tavily_results.jsonl
    python run_gpt_pass.py --input custom.jsonl      # Custom Tavily JSONL
    python run_gpt_pass.py --tag v2                  # Tag output as gpt_v2.jsonl
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    PROCESSING, STAGE1_OUTPUT_DIR, STAGE1_GPT_DIR,
    APIKeys,
)
from src.jsonl_writer import AsyncJSONLWriter
from src.rate_limiter import AsyncRateLimiter
from src.stage_1_filter import (
    WebsiteStatus,
    TavilySearchResult,
    SearchSnippet,
    classify_company,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

TAVILY_JSONL = STAGE1_OUTPUT_DIR / "tavily_results.jsonl"


def load_tavily_records(jsonl_path: Path) -> list[dict]:
    """Load all records from the Tavily JSONL file."""
    records = []
    with open(jsonl_path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Warning: skipping malformed line {line_no}: {e}")
    return records


def load_existing_rcids(jsonl_path: Path) -> set[int]:
    """Scan existing GPT JSONL for already-classified rcids."""
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


def reconstruct_website_status(record: dict) -> WebsiteStatus:
    """Rebuild WebsiteStatus from a Tavily JSONL record."""
    wc = record.get("website_check", {})
    return WebsiteStatus(
        url=wc.get("url", ""),
        is_alive=wc.get("is_alive", False),
        status_code=wc.get("status_code"),
        final_url=wc.get("final_url"),
        error=wc.get("error"),
    )


def reconstruct_tavily_result(record: dict) -> TavilySearchResult:
    """Rebuild TavilySearchResult from a Tavily JSONL record."""
    tv = record.get("tavily", {})
    raw = tv.get("raw_response", {})

    snippets = []
    for r in raw.get("results", []):
        snippets.append(SearchSnippet(
            title=r.get("title", ""),
            url=r.get("url", ""),
            content=r.get("content", ""),
            score=r.get("score", 0.0),
        ))

    return TavilySearchResult(
        company_name=record.get("name", ""),
        query=tv.get("query", ""),
        snippets=snippets,
        result_count=tv.get("result_count", 0),
        raw_response=raw or None,
        error=tv.get("error"),
    )


def build_gpt_record(
    rcid: int,
    name: str,
    assessment,
) -> dict:
    """Build the GPT JSONL record."""
    return {
        "rcid": rcid,
        "name": name,
        "online_presence_score": assessment.online_presence_score,
        "research_priority_score": assessment.research_priority_score,
        "reasoning": assessment.reasoning,
        "error": assessment.error,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Run GPT classification pass on Tavily results")
    parser.add_argument("--input", type=str, default=None, help="Path to Tavily JSONL")
    parser.add_argument("--tag", type=str, default=None,
                        help="Version tag for output file (e.g. 'v2' -> gpt_v2.jsonl)")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N records")
    args = parser.parse_args()

    tavily_path = Path(args.input) if args.input else TAVILY_JSONL
    if not tavily_path.exists():
        print(f"ERROR: Tavily JSONL not found: {tavily_path}")
        print("  Run run_tavily_pass.py first to generate Tavily results.")
        sys.exit(1)

    # Output path
    if args.tag:
        output_name = f"gpt_{args.tag}.jsonl"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"gpt_{timestamp}.jsonl"
    output_path = STAGE1_GPT_DIR / output_name

    # Load API keys
    keys = APIKeys()
    if not keys.openai:
        print("ERROR: OpenAI API key not found. Add to credentials/openai_api_key.txt")
        sys.exit(1)

    # Load Tavily records
    print(f"Loading Tavily results from {tavily_path}...")
    records = load_tavily_records(tavily_path)
    print(f"  Total Tavily records: {len(records)}")

    if args.limit:
        records = records[:args.limit]
        print(f"  Limited to first {args.limit}")

    # Check existing progress
    existing = load_existing_rcids(output_path)
    if existing:
        print(f"  Already classified: {len(existing)} (resuming)")
    remaining = [r for r in records if int(r["rcid"]) not in existing]
    print(f"  Remaining to classify: {len(remaining)}")

    if not remaining:
        print("\nAll records already classified. Nothing to do.")
        return

    # Cost estimate (GPT-5-nano is very cheap)
    cost = len(remaining) * 0.0002
    print(f"\nEstimated GPT cost: ${cost:.3f} ({len(remaining)} calls x $0.0002)")
    print(f"Output: {output_path}")

    input("\nPress Enter to start (or Ctrl+C to cancel)...")

    # Create shared OpenAI client
    import httpx
    openai_client = httpx.AsyncClient(timeout=PROCESSING.openai_timeout)

    # Rate limiter + concurrency semaphore
    openai_limiter = AsyncRateLimiter(rpm=PROCESSING.openai_rpm, name="openai")
    semaphore = asyncio.Semaphore(PROCESSING.max_concurrent_requests)

    start_time = time.time()
    processed = 0
    errors = 0

    # Score distribution tracking
    score_dist = {i: 0 for i in range(6)}

    async with AsyncJSONLWriter(output_path) as writer:

        async def classify_one(record: dict, idx: int) -> None:
            nonlocal processed, errors
            rcid = int(record["rcid"])
            name = record["name"]

            # Reconstruct data from Tavily JSONL
            website_status = reconstruct_website_status(record)
            search_result = reconstruct_tavily_result(record)

            async with semaphore:
                await openai_limiter.acquire()

                try:
                    assessment = await classify_company(
                        company_name=name,
                        company_description=record.get("short_description"),
                        website_status=website_status,
                        search_result=search_result,
                        api_key=keys.openai,
                        homepage_url=record.get("homepage_url"),
                        long_description=record.get("description"),
                        client=openai_client,
                    )

                    gpt_record = build_gpt_record(rcid, name, assessment)
                    await writer.write(gpt_record)
                    processed += 1

                    score = assessment.research_priority_score
                    if 0 <= score <= 5:
                        score_dist[score] += 1
                    print(f"  [{idx}/{len(remaining)}] {name}: "
                          f"P{score} (presence={assessment.online_presence_score}) "
                          f"— {assessment.reasoning[:50]}")

                except Exception as e:
                    errors += 1
                    error_record = {
                        "rcid": rcid, "name": name,
                        "online_presence_score": 0, "research_priority_score": 0,
                        "reasoning": "", "error": f"{type(e).__name__}: {str(e)[:200]}",
                    }
                    await writer.write(error_record)
                    print(f"  [{idx}/{len(remaining)}] {name}: ERROR — {e}")

        try:
            # Process in batches for progress reporting
            batch_size = 200
            for batch_start in range(0, len(remaining), batch_size):
                batch = remaining[batch_start:batch_start + batch_size]
                tasks = [
                    classify_one(record, batch_start + j + 1)
                    for j, record in enumerate(batch)
                ]
                await asyncio.gather(*tasks)

                # Progress report
                elapsed = time.time() - start_time
                done = batch_start + len(batch)
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (len(remaining) - done) / rate if rate > 0 else 0
                print(f"\n  --- Progress: {done}/{len(remaining)} | "
                      f"{rate:.1f}/sec | ETA: {eta/60:.0f}min | "
                      f"Errors: {errors} | "
                      f"Rate limiter: {openai_limiter.stats} ---\n")

        finally:
            await openai_client.aclose()

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'='*60}")
    print(f"  GPT pass complete")
    print(f"  Processed: {processed}/{len(remaining)} ({errors} errors)")
    print(f"  Time: {elapsed:.1f}s ({processed/elapsed:.1f}/sec)")
    print(f"  Output: {output_path}")

    print(f"\n  Score distribution:")
    labels = {
        5: "5 (definitely yields)",
        4: "4 (potentially yields)",
        3: "3 (worth a shot)",
        2: "2 (not worth it)",
        1: "1 (mostly unrelated)",
        0: "0 (not researchable)",
    }
    for score in [5, 4, 3, 2, 1, 0]:
        count = score_dist[score]
        pct = count / processed * 100 if processed else 0
        bar = "█" * int(pct / 2)
        print(f"    {labels[score]:25} {count:4} ({pct:5.1f}%) {bar}")

    deep = sum(score_dist[s] for s in [3, 4, 5])
    print(f"\n  Deep research candidates (>= 3): {deep}/{processed} "
          f"({deep/processed*100:.0f}%)" if processed else "")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
