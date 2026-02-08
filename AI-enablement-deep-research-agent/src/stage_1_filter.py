"""
Stage 1: Presence Filter

The first stage of the pipeline. Gathers general company intelligence
and predicts research priority for subsequent deep research stages.

Components:
    1. Website health check (HEAD request)
    2. Tavily search for general company information
    3. GPT-5-nano analysis to predict research priority

Cost: ~$0.011 per company ($0.01 Tavily + $0.001 GPT-5-nano)
"""

import asyncio
import json as json_module
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import IO, Optional
from urllib.parse import urlparse

import httpx

from .config import PROCESSING, APIKeys, PROMPTS_DIR


# ─────────────────────────────────────────────────────────────────────────────
# WEBSITE HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WebsiteStatus:
    """
    Result of checking a company's website.
    
    Attributes:
        url: The URL that was checked.
        is_alive: Whether the site responded successfully.
        status_code: HTTP status code (if any response received).
        final_url: URL after redirects (if different from original).
        response_time_ms: How long the request took.
        error: Error message if the check failed.
    """
    url: str
    is_alive: bool
    status_code: Optional[int] = None
    final_url: Optional[str] = None
    response_time_ms: Optional[float] = None
    error: Optional[str] = None
    
    @property
    def is_redirect(self) -> bool:
        """Did the URL redirect to a different domain?"""
        if not self.final_url or not self.url:
            return False
        
        original_domain = urlparse(self.url).netloc.lower()
        final_domain = urlparse(self.final_url).netloc.lower()
        
        # Strip www. for comparison
        original_domain = original_domain.replace("www.", "")
        final_domain = final_domain.replace("www.", "")
        
        return original_domain != final_domain


async def check_website(
    url: str,
    timeout: float = PROCESSING.http_timeout,
    follow_redirects: bool = True,
    client: Optional[httpx.AsyncClient] = None,
) -> WebsiteStatus:
    """
    Check if a website is alive using a HEAD request.
    
    Uses HEAD instead of GET because:
    - Faster (no response body)
    - Lower bandwidth
    - Sufficient to check if site is up
    
    Args:
        url: The URL to check.
        timeout: Request timeout in seconds.
        follow_redirects: Whether to follow redirects.
        client: Optional shared httpx.AsyncClient for connection pooling.
    
    Returns:
        WebsiteStatus with check results.
    """
    # Normalize URL
    if not url:
        return WebsiteStatus(
            url=url or "",
            is_alive=False,
            error="No URL provided"
        )
    
    # Ensure URL has scheme
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    
    async def _do_check(c: httpx.AsyncClient) -> WebsiteStatus:
        try:
            # Try HEAD first (faster)
            start_time = asyncio.get_event_loop().time()
            response = await c.head(url)
            elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Some servers don't support HEAD, fall back to GET
            if response.status_code == 405:  # Method Not Allowed
                start_time = asyncio.get_event_loop().time()
                response = await c.get(url)
                elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Consider 2xx and 3xx as "alive"
            is_alive = 200 <= response.status_code < 400
            
            return WebsiteStatus(
                url=url,
                is_alive=is_alive,
                status_code=response.status_code,
                final_url=str(response.url) if response.url != url else None,
                response_time_ms=round(elapsed_ms, 1),
            )
            
        except httpx.TimeoutException:
            return WebsiteStatus(
                url=url,
                is_alive=False,
                error="Timeout"
            )
        except httpx.ConnectError as e:
            # DNS failure, connection refused, etc.
            error_msg = str(e)
            if "Name or service not known" in error_msg or "getaddrinfo failed" in error_msg:
                error_msg = "DNS lookup failed"
            elif "Connection refused" in error_msg:
                error_msg = "Connection refused"
            else:
                error_msg = "Connection failed"
            
            return WebsiteStatus(
                url=url,
                is_alive=False,
                error=error_msg
            )
        except httpx.TooManyRedirects:
            return WebsiteStatus(
                url=url,
                is_alive=False,
                error="Too many redirects"
            )
        except Exception as e:
            # SSL errors, other network issues
            error_type = type(e).__name__
            return WebsiteStatus(
                url=url,
                is_alive=False,
                error=f"{error_type}: {str(e)[:100]}"
            )
    
    if client is not None:
        return await _do_check(client)
    
    # Fallback: create a one-off client (backward compat for simple scripts)
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=follow_redirects,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0; +https://ubc.ca)",
            "Accept": "text/html,application/xhtml+xml",
        }
    ) as new_client:
        return await _do_check(new_client)


async def check_websites_batch(
    urls: list[str],
    max_concurrent: int = PROCESSING.max_concurrent_requests
) -> list[WebsiteStatus]:
    """
    Check multiple websites concurrently.
    
    Uses semaphore to limit concurrency and avoid overwhelming
    network/getting rate limited.
    
    Args:
        urls: List of URLs to check.
        max_concurrent: Maximum simultaneous requests.
    
    Returns:
        List of WebsiteStatus in same order as input URLs.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def check_with_semaphore(url: str) -> WebsiteStatus:
        async with semaphore:
            return await check_website(url)
    
    tasks = [check_with_semaphore(url) for url in urls]
    return await asyncio.gather(*tasks)


# ─────────────────────────────────────────────────────────────────────────────
# SYNCHRONOUS WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────

def check_website_sync(url: str) -> WebsiteStatus:
    """
    Synchronous wrapper for check_website.
    
    Useful for simple scripts and testing.
    """
    return asyncio.run(check_website(url))


def check_websites_batch_sync(urls: list[str]) -> list[WebsiteStatus]:
    """
    Synchronous wrapper for check_websites_batch.
    """
    return asyncio.run(check_websites_batch(urls))


# ─────────────────────────────────────────────────────────────────────────────
# TAVILY SEARCH
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchSnippet:
    """A single search result snippet."""
    title: str
    url: str
    content: str
    score: float = 0.0


@dataclass
class TavilySearchResult:
    """
    Result from a Tavily search for general company information.
    
    Attributes:
        company_name: The company we searched for.
        query: The exact query used.
        snippets: List of relevant search result snippets.
        result_count: Number of results returned (indicator of online presence).
        error: Error message if search failed.
    """
    company_name: str
    query: str
    snippets: list[SearchSnippet]
    result_count: int
    raw_response: Optional[dict] = None
    error: Optional[str] = None


def build_search_query(
    company_name: str,
    homepage_url: Optional[str] = None,
    company_description: Optional[str] = None
) -> str:
    """
    Build a natural language search query for company information.
    
    Uses a conversational format that works well with Tavily's search API
    and avoids overly restrictive constraints (like site:) that can cause
    zero results for lesser-known companies.
    
    Example output:
    - "provide me information on the company Blue Dot"
    
    Args:
        company_name: Name of the company.
        homepage_url: Not used (kept for API compatibility).
        company_description: Not used (kept for API compatibility).
    
    Returns:
        Natural language search query string.
    """
    return f"provide me information on the company {company_name}"


async def search_tavily(
    company_name: str,
    homepage_url: Optional[str] = None,
    company_description: Optional[str] = None,
    api_key: Optional[str] = None,
    max_results: int = 5,
    client: Optional[httpx.AsyncClient] = None,
) -> TavilySearchResult:
    """
    Search Tavily for general company information.
    
    This is a controlled, single search per company to ensure
    predictable costs (~$0.02 per search with advanced depth).
    
    Args:
        company_name: Name of the company to search for.
        homepage_url: Company's website (for domain constraint).
        company_description: Crunchbase description (for disambiguation).
        api_key: Tavily API key. Uses credentials folder or env var if not provided.
        max_results: Maximum number of results to return.
        client: Optional shared httpx.AsyncClient for connection pooling.
    
    Returns:
        TavilySearchResult with general company information snippets.
    """
    if not api_key:
        keys = APIKeys()
        api_key = keys.tavily
    
    if not api_key:
        return TavilySearchResult(
            company_name=company_name,
            query="",
            snippets=[],
            result_count=0,
            error="Tavily API key not set. Add to credentials/tavily_api_key.txt"
        )
    
    query = build_search_query(company_name, homepage_url, company_description)
    
    async def _do_search(c: httpx.AsyncClient) -> TavilySearchResult:
        try:
            response = await c.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "advanced",  # Highest relevance, returns reranked chunks
                    "max_results": max_results,
                    "chunks_per_source": 3,  # Multiple snippets per source for richer context
                    "include_answer": False,  # GPT does its own assessment
                    "include_raw_content": False,  # Not needed for Stage 1 filtering
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # Parse results into snippets
            snippets = []
            for result in data.get("results", []):
                snippet = SearchSnippet(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    content=result.get("content", ""),
                    score=result.get("score", 0.0),
                )
                snippets.append(snippet)
            
            return TavilySearchResult(
                company_name=company_name,
                query=query,
                snippets=snippets,
                result_count=len(snippets),
                raw_response=data,
            )
            
        except httpx.HTTPStatusError as e:
            return TavilySearchResult(
                company_name=company_name,
                query=query,
                snippets=[],
                result_count=0,
                error=f"HTTP {e.response.status_code}: {e.response.text[:100]}"
            )
        except Exception as e:
            return TavilySearchResult(
                company_name=company_name,
                query=query,
                snippets=[],
                result_count=0,
                error=f"{type(e).__name__}: {str(e)[:100]}"
            )
    
    if client is not None:
        return await _do_search(client)
    
    # Fallback: create a one-off client (backward compat for simple scripts)
    async with httpx.AsyncClient(timeout=PROCESSING.tavily_timeout) as new_client:
        return await _do_search(new_client)


async def search_tavily_batch(
    companies: list[tuple[str, Optional[str], Optional[str]]],  # (name, homepage_url, description)
    api_key: Optional[str] = None,
    max_concurrent: int = 5  # Tavily rate limits are stricter
) -> list[TavilySearchResult]:
    """
    Search Tavily for multiple companies concurrently.
    
    Args:
        companies: List of (company_name, homepage_url, description) tuples.
        api_key: Tavily API key.
        max_concurrent: Maximum simultaneous requests (default 5 for Tavily).
    
    Returns:
        List of TavilySearchResult in same order as input.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def search_with_semaphore(name: str, url: Optional[str], desc: Optional[str]) -> TavilySearchResult:
        async with semaphore:
            return await search_tavily(name, url, desc, api_key)
    
    tasks = [search_with_semaphore(name, url, desc) for name, url, desc in companies]
    return await asyncio.gather(*tasks)


# ─────────────────────────────────────────────────────────────────────────────
# TAVILY SYNC WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────

def search_tavily_sync(
    company_name: str,
    homepage_url: Optional[str] = None,
    company_description: Optional[str] = None,
    api_key: Optional[str] = None
) -> TavilySearchResult:
    """Synchronous wrapper for search_tavily."""
    return asyncio.run(search_tavily(company_name, homepage_url, company_description, api_key))


def search_tavily_batch_sync(
    companies: list[tuple[str, Optional[str], Optional[str]]],
    api_key: Optional[str] = None
) -> list[TavilySearchResult]:
    """Synchronous wrapper for search_tavily_batch."""
    return asyncio.run(search_tavily_batch(companies, api_key))


# ─────────────────────────────────────────────────────────────────────────────
# GPT-5-nano CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PresenceAssessment:
    """
    GPT-5-nano's assessment of a company's profile and research priority.
    
    This is the output of Stage 1 and determines routing to subsequent stages.
    Includes raw API data for logging/auditing.
    """
    company_name: str
    online_presence_score: int  # 1-10
    research_priority_score: int  # 0-5 (0=not researchable, 5=definitely yields GenAI findings)
    reasoning: str  # Brief explanation for consistency assessment and priority judgment
    error: Optional[str] = None
    
    # Raw data for logging — not used for routing, but preserved for auditing
    user_prompt: Optional[str] = None  # The exact prompt sent to GPT
    raw_gpt_response: Optional[dict] = None  # Full OpenAI API response


def _load_system_prompt() -> str:
    """
    Load the classifier system prompt from the prompts file.
    
    Returns:
        The system prompt string.
    """
    prompt_file = PROMPTS_DIR / "stage_1_classifier.txt"
    
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    content = prompt_file.read_text()
    
    # Extract only the SYSTEM PROMPT section (between "## SYSTEM PROMPT" and "---")
    lines = content.split('\n')
    in_system_prompt = False
    system_prompt_lines = []
    
    for line in lines:
        if line.strip() == "## SYSTEM PROMPT":
            in_system_prompt = True
            continue
        if in_system_prompt and line.strip() == "---":
            break
        if in_system_prompt:
            system_prompt_lines.append(line)
    
    return '\n'.join(system_prompt_lines).strip()


# Load system prompt from file (cached at module load time)
CLASSIFIER_SYSTEM_PROMPT = _load_system_prompt()


def _build_classifier_prompt(
    company_name: str,
    company_description: Optional[str],
    website_status: WebsiteStatus,
    search_result: TavilySearchResult,
    homepage_url: Optional[str] = None,
    long_description: Optional[str] = None,
) -> str:
    """
    Build the user prompt for the classifier.
    
    Separates the Crunchbase profile (ground truth) from the search tool
    output so GPT can assess consistency between them.
    """
    # --- CRUNCHBASE PROFILE section ---
    profile_lines = [
        "CRUNCHBASE PROFILE:",
        f"Company: {company_name}",
        f"Description: {company_description or 'None'}",
    ]
    if long_description and long_description.strip():
        profile_lines.append(f"About: {long_description.strip()}")
    profile_lines.append(f"Website: {homepage_url or 'None'}")
    
    # --- SEARCH TOOL OUTPUT section ---
    if website_status.is_alive:
        website_info = f"ACTIVE ({website_status.status_code})"
    else:
        website_info = f"DOWN ({website_status.error})"
    
    if search_result.error:
        search_info = f"Search error: {search_result.error}"
    elif search_result.result_count == 0:
        search_info = "No search results found."
    else:
        # Include ALL snippets with FULL content
        snippets = []
        for i, snippet in enumerate(search_result.snippets, 1):
            snippets.append(f"{i}. {snippet.title}\n   URL: {snippet.url}\n   {snippet.content}")
        search_info = "\n\n".join(snippets)
    
    search_lines = [
        "SEARCH TOOL OUTPUT:",
        f'The following results were retrieved by searching only the company name "{company_name}".',
        f"Website check: {website_info}",
        f"Results ({search_result.result_count} found):",
        "",
        search_info,
    ]
    
    prompt = "\n".join(profile_lines) + "\n\n" + "\n".join(search_lines)
    return prompt


async def classify_company(
    company_name: str,
    company_description: Optional[str],
    website_status: WebsiteStatus,
    search_result: TavilySearchResult,
    api_key: Optional[str] = None,
    homepage_url: Optional[str] = None,
    long_description: Optional[str] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> PresenceAssessment:
    """
    Use GPT-5-nano to assess a company's profile and predict research priority.
    
    Args:
        company_name: Name of the company.
        company_description: Crunchbase short description (if available).
        website_status: Result from check_website().
        search_result: Result from search_tavily().
        api_key: OpenAI API key. Uses env var if not provided.
        homepage_url: Company homepage URL from Crunchbase (for profile context).
        long_description: Crunchbase long description (if available).
        client: Optional shared httpx.AsyncClient for connection pooling.
    
    Returns:
        PresenceAssessment with online_presence_score and research_priority_score.
    """
    if not api_key:
        keys = APIKeys()
        api_key = keys.openai
    
    if not api_key:
        return PresenceAssessment(
            company_name=company_name,
            online_presence_score=0,
            research_priority_score=0,
            reasoning="",
            error="OpenAI API key not set. Add to credentials/openai_api_key.txt"
        )
    
    user_prompt = _build_classifier_prompt(
        company_name, company_description, website_status, search_result,
        homepage_url=homepage_url, long_description=long_description,
    )
    
    async def _do_classify(c: httpx.AsyncClient) -> PresenceAssessment:
        try:
            response = await c.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-5-nano",
                    "messages": [
                        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "response_format": {"type": "json_object"},
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # Parse the response
            content = data["choices"][0]["message"]["content"]
            result = json_module.loads(content)
            
            return PresenceAssessment(
                company_name=company_name,
                online_presence_score=int(result.get("online_presence_score", 0)),
                research_priority_score=int(result.get("research_priority_score", 0)),
                reasoning=result.get("reasoning", ""),
                user_prompt=user_prompt,
                raw_gpt_response=data,
            )
            
        except httpx.HTTPStatusError as e:
            return PresenceAssessment(
                company_name=company_name,
                online_presence_score=0,
                research_priority_score=0,
                reasoning="",
                error=f"HTTP {e.response.status_code}: {e.response.text[:100]}",
                user_prompt=user_prompt,
            )
        except json_module.JSONDecodeError as e:
            return PresenceAssessment(
                company_name=company_name,
                online_presence_score=0,
                research_priority_score=0,
                reasoning="",
                error=f"Failed to parse GPT response: {str(e)}",
                user_prompt=user_prompt,
            )
        except Exception as e:
            return PresenceAssessment(
                company_name=company_name,
                online_presence_score=0,
                research_priority_score=0,
                reasoning="",
                error=f"{type(e).__name__}: {str(e)[:100]}",
                user_prompt=user_prompt,
            )
    
    if client is not None:
        return await _do_classify(client)
    
    # Fallback: create a one-off client (backward compat for simple scripts)
    async with httpx.AsyncClient(timeout=PROCESSING.openai_timeout) as new_client:
        return await _do_classify(new_client)


def classify_company_sync(
    company_name: str,
    company_description: Optional[str],
    website_status: WebsiteStatus,
    search_result: TavilySearchResult,
    api_key: Optional[str] = None,
    homepage_url: Optional[str] = None,
    long_description: Optional[str] = None,
) -> PresenceAssessment:
    """Synchronous wrapper for classify_company (creates its own client)."""
    return asyncio.run(classify_company(
        company_name, company_description, website_status, search_result, api_key,
        homepage_url=homepage_url, long_description=long_description,
        # No shared client — sync wrappers are for simple scripts
    ))


# ─────────────────────────────────────────────────────────────────────────────
# COMPLETE STAGE 1 PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Stage1Result:
    """
    Complete result from Stage 1 presence filter.
    
    Combines website check, search, and classification into one result.
    """
    company_name: str
    company_id: int
    
    # Component results
    website_status: WebsiteStatus
    search_result: TavilySearchResult
    assessment: PresenceAssessment
    
    # Summary for easy access
    presence_score: int
    research_priority_score: int  # 0-5 (0=not researchable, 3+=worth deep research)
    
    # Cost tracking
    estimated_cost: float = 0.011  # $0.01 Tavily + $0.001 GPT-5-nano
    
    @property
    def should_deep_research(self) -> bool:
        """Should this company proceed to deep research? (score >= 3)"""
        return self.research_priority_score >= 3
    
    @property
    def is_high_priority(self) -> bool:
        """Is this a high priority company for deep research? (score >= 4)"""
        return self.research_priority_score >= 4


def _build_tavily_log_entry(
    result: Stage1Result,
    company_description: Optional[str] = None,
    homepage_url: Optional[str] = None,
) -> dict:
    """
    Build a JSONL entry for the Tavily log file.
    
    Contains: company metadata, website check, and full Tavily raw response.
    """
    ws = result.website_status
    sr = result.search_result
    
    return {
        "company_id": result.company_id,
        "company_name": result.company_name,
        "company_description": company_description,
        "homepage_url": homepage_url,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "website_check": {
            "url": ws.url,
            "is_alive": ws.is_alive,
            "status_code": ws.status_code,
            "final_url": ws.final_url,
            "is_redirect": ws.is_redirect,
            "error": ws.error,
        },
        "tavily": {
            "query": sr.query,
            "result_count": sr.result_count,
            "raw_response": sr.raw_response,
        },
    }


def _build_gpt_log_entry(
    result: Stage1Result,
) -> dict:
    """
    Build a JSONL entry for the GPT log file.
    
    Contains only: company ID/name and GPT output (parsed result).
    Input data (Tavily results, prompts) lives in the Tavily log file.
    """
    assess = result.assessment
    
    return {
        "company_id": result.company_id,
        "company_name": result.company_name,
        "online_presence_score": assess.online_presence_score,
        "research_priority_score": assess.research_priority_score,
        "reasoning": assess.reasoning,
        "error": assess.error,
    }


async def run_stage_1(
    company_id: int,
    company_name: str,
    homepage_url: Optional[str],
    company_description: Optional[str],
    long_description: Optional[str] = None,
    tavily_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    tavily_log_file: Optional[IO] = None,
    gpt_log_file: Optional[IO] = None,
    http_client: Optional[httpx.AsyncClient] = None,
    tavily_client: Optional[httpx.AsyncClient] = None,
    openai_client: Optional[httpx.AsyncClient] = None,
) -> Stage1Result:
    """
    Run the complete Stage 1 presence filter for a single company.
    
    This orchestrates:
    1. Website health check
    2. Tavily search for general company information (name-only query)
    3. GPT-5-nano analysis comparing search results against Crunchbase profile
    
    Args:
        company_id: Unique identifier (rcid from Crunchbase).
        company_name: Company name.
        homepage_url: Company website URL.
        company_description: Crunchbase short description.
        long_description: Crunchbase long description (if available).
        tavily_api_key: Tavily API key.
        openai_api_key: OpenAI API key.
        tavily_log_file: Optional file handle for Tavily JSONL log.
        gpt_log_file: Optional file handle for GPT JSONL log.
        http_client: Optional shared httpx client for website checks.
        tavily_client: Optional shared httpx client for Tavily API.
        openai_client: Optional shared httpx client for OpenAI API.
    
    Returns:
        Stage1Result with all component results and research priority score.
    """
    # Step 1: Check website
    website_status = await check_website(homepage_url or "", client=http_client)
    
    # Step 2: Search for general company information (name-only query)
    search_result = await search_tavily(
        company_name, 
        homepage_url,
        company_description,
        api_key=tavily_api_key,
        client=tavily_client,
    )
    
    # Step 3: Analyze and assign research priority (GPT compares search vs Crunchbase profile)
    assessment = await classify_company(
        company_name,
        company_description,
        website_status,
        search_result,
        api_key=openai_api_key,
        homepage_url=homepage_url,
        long_description=long_description,
        client=openai_client,
    )
    
    result = Stage1Result(
        company_name=company_name,
        company_id=company_id,
        website_status=website_status,
        search_result=search_result,
        assessment=assessment,
        presence_score=assessment.online_presence_score,
        research_priority_score=assessment.research_priority_score,
    )
    
    # Write JSONL log entries (separate files for Tavily and GPT)
    if tavily_log_file is not None:
        entry = _build_tavily_log_entry(result, company_description, homepage_url)
        tavily_log_file.write(json_module.dumps(entry) + "\n")
        tavily_log_file.flush()
    
    if gpt_log_file is not None:
        entry = _build_gpt_log_entry(result)
        gpt_log_file.write(json_module.dumps(entry) + "\n")
        gpt_log_file.flush()
    
    return result


def run_stage_1_sync(
    company_id: int,
    company_name: str,
    homepage_url: Optional[str],
    company_description: Optional[str],
    long_description: Optional[str] = None,
    tavily_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    tavily_log_file: Optional[IO] = None,
    gpt_log_file: Optional[IO] = None
) -> Stage1Result:
    """Synchronous wrapper for run_stage_1."""
    return asyncio.run(run_stage_1(
        company_id, company_name, homepage_url, company_description,
        long_description, tavily_api_key, openai_api_key,
        tavily_log_file, gpt_log_file
    ))
