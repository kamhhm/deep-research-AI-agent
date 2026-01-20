"""
Stage 1: Presence Filter

The first stage of the pipeline. Determines online presence and
routes companies to appropriate subsequent stages.

Components:
    1. Website health check (HEAD request)
    2. Tavily search for AI mentions
    3. GPT-4o-mini interpretation and routing decision

Cost: ~$0.002 per company
"""

import asyncio
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import httpx

from .config import PROCESSING


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
    follow_redirects: bool = True
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
    
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=follow_redirects,
        # Common headers to avoid bot detection
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0; +https://ubc.ca)",
            "Accept": "text/html,application/xhtml+xml",
        }
    ) as client:
        try:
            # Try HEAD first (faster)
            start_time = asyncio.get_event_loop().time()
            response = await client.head(url)
            elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Some servers don't support HEAD, fall back to GET
            if response.status_code == 405:  # Method Not Allowed
                start_time = asyncio.get_event_loop().time()
                response = await client.get(url)
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
    Result from a Tavily search for AI mentions.
    
    Attributes:
        company_name: The company we searched for.
        query: The exact query used.
        snippets: List of relevant search result snippets.
        ai_keywords_found: AI-related keywords found in snippets.
        has_ai_mentions: Quick boolean - did we find any AI mentions?
        error: Error message if search failed.
    """
    company_name: str
    query: str
    snippets: list[SearchSnippet]
    ai_keywords_found: list[str]
    has_ai_mentions: bool
    raw_response: Optional[dict] = None
    error: Optional[str] = None


# AI-related keywords to look for in search results
AI_KEYWORDS = [
    # GenAI tools
    "chatgpt", "gpt-4", "gpt-5", "openai", "claude", "anthropic",
    "copilot", "github copilot", "gemini", "bard",
    # General AI terms
    "generative ai", "genai", "gen ai", "large language model", "llm",
    "ai assistant", "ai-powered", "ai powered", "artificial intelligence",
    "machine learning", "ml model",
    # AI adoption signals
    "uses ai", "using ai", "adopted ai", "implementing ai",
    "ai tools", "ai platform", "ai solution",
    # Specific use cases
    "ai automation", "ai chatbot", "ai customer service",
    "ai content", "ai writing", "ai coding",
]


def build_search_query(company_name: str, homepage_url: Optional[str] = None) -> str:
    """
    Build an optimized search query for finding GenAI adoption evidence.
    
    The query is designed to find:
    - News about the company using AI tools
    - Blog posts or announcements about AI adoption
    - Job postings mentioning AI tools
    
    Args:
        company_name: Name of the company.
        homepage_url: Company's website URL (used to extract domain).
    
    Returns:
        Search query string.
    """
    # Extract domain if available (helps with disambiguation)
    domain_hint = ""
    if homepage_url:
        try:
            domain = urlparse(homepage_url).netloc.replace("www.", "")
            if domain:
                domain_hint = f" site:{domain} OR"
        except Exception:
            pass
    
    # Build query: company name + AI-related terms
    # We use OR to cast a wide net while keeping it focused
    query = (
        f'"{company_name}"{domain_hint} '
        f'(ChatGPT OR "AI tools" OR "generative AI" OR Copilot OR '
        f'"artificial intelligence" OR "machine learning" OR "AI-powered")'
    )
    
    return query


def _extract_ai_keywords(text: str) -> list[str]:
    """
    Extract AI-related keywords found in text.
    
    Args:
        text: Text to search for keywords.
    
    Returns:
        List of found keywords (deduplicated, lowercase).
    """
    text_lower = text.lower()
    found = set()
    
    for keyword in AI_KEYWORDS:
        if keyword.lower() in text_lower:
            found.add(keyword.lower())
    
    return sorted(found)


async def search_tavily(
    company_name: str,
    homepage_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_results: int = 5
) -> TavilySearchResult:
    """
    Search Tavily for AI adoption mentions about a company.
    
    This is a controlled, single search per company to ensure
    predictable costs (~$0.001 per search).
    
    Args:
        company_name: Name of the company to search for.
        homepage_url: Company's website (helps with disambiguation).
        api_key: Tavily API key. Uses env var if not provided.
        max_results: Maximum number of results to return.
    
    Returns:
        TavilySearchResult with snippets and AI mention detection.
    """
    import os
    
    api_key = api_key or os.getenv("TAVILY_API_KEY")
    if not api_key:
        return TavilySearchResult(
            company_name=company_name,
            query="",
            snippets=[],
            ai_keywords_found=[],
            has_ai_mentions=False,
            error="TAVILY_API_KEY not set"
        )
    
    query = build_search_query(company_name, homepage_url)
    
    async with httpx.AsyncClient(timeout=PROCESSING.api_timeout) as client:
        try:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "basic",  # "basic" is cheaper than "advanced"
                    "max_results": max_results,
                    "include_answer": False,  # We don't need summarization
                    "include_raw_content": False,  # Keep response small
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # Parse results into snippets
            snippets = []
            all_text = ""
            
            for result in data.get("results", []):
                snippet = SearchSnippet(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    content=result.get("content", ""),
                    score=result.get("score", 0.0),
                )
                snippets.append(snippet)
                all_text += f" {snippet.title} {snippet.content}"
            
            # Extract AI keywords from all text
            ai_keywords = _extract_ai_keywords(all_text)
            
            return TavilySearchResult(
                company_name=company_name,
                query=query,
                snippets=snippets,
                ai_keywords_found=ai_keywords,
                has_ai_mentions=len(ai_keywords) > 0,
                raw_response=data,
            )
            
        except httpx.HTTPStatusError as e:
            return TavilySearchResult(
                company_name=company_name,
                query=query,
                snippets=[],
                ai_keywords_found=[],
                has_ai_mentions=False,
                error=f"HTTP {e.response.status_code}: {e.response.text[:100]}"
            )
        except Exception as e:
            return TavilySearchResult(
                company_name=company_name,
                query=query,
                snippets=[],
                ai_keywords_found=[],
                has_ai_mentions=False,
                error=f"{type(e).__name__}: {str(e)[:100]}"
            )


async def search_tavily_batch(
    companies: list[tuple[str, Optional[str]]],  # (name, homepage_url)
    api_key: Optional[str] = None,
    max_concurrent: int = 5  # Tavily rate limits are stricter
) -> list[TavilySearchResult]:
    """
    Search Tavily for multiple companies concurrently.
    
    Args:
        companies: List of (company_name, homepage_url) tuples.
        api_key: Tavily API key.
        max_concurrent: Maximum simultaneous requests (default 5 for Tavily).
    
    Returns:
        List of TavilySearchResult in same order as input.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def search_with_semaphore(name: str, url: Optional[str]) -> TavilySearchResult:
        async with semaphore:
            return await search_tavily(name, url, api_key)
    
    tasks = [search_with_semaphore(name, url) for name, url in companies]
    return await asyncio.gather(*tasks)


# ─────────────────────────────────────────────────────────────────────────────
# TAVILY SYNC WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────

def search_tavily_sync(
    company_name: str,
    homepage_url: Optional[str] = None,
    api_key: Optional[str] = None
) -> TavilySearchResult:
    """Synchronous wrapper for search_tavily."""
    return asyncio.run(search_tavily(company_name, homepage_url, api_key))


def search_tavily_batch_sync(
    companies: list[tuple[str, Optional[str]]],
    api_key: Optional[str] = None
) -> list[TavilySearchResult]:
    """Synchronous wrapper for search_tavily_batch."""
    return asyncio.run(search_tavily_batch(companies, api_key))


# ─────────────────────────────────────────────────────────────────────────────
# GPT-4o-mini CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PresenceAssessment:
    """
    GPT-4o-mini's assessment of a company's online presence and AI signals.
    
    This is the output of Stage 1 and determines routing to subsequent stages.
    """
    company_name: str
    
    # Presence assessment
    online_presence_score: int  # 0-100
    presence_reasoning: str
    
    # AI signal detection
    ai_mentions_found: bool
    ai_signals: list[str]  # Specific signals found (e.g., "mentions ChatGPT in blog")
    
    # Routing decision
    recommended_stage: str  # "2A", "2B", "3", or "stop"
    routing_reasoning: str
    
    # Metadata
    error: Optional[str] = None


# System prompt for the classifier
CLASSIFIER_SYSTEM_PROMPT = """You are an AI research assistant helping identify evidence of GenAI adoption in business operations.

Your task: Analyze the provided information about a company and assess:
1. Their online presence (how much information is available about them)
2. Whether there are any signals they might be using GenAI tools internally

IMPORTANT DISTINCTIONS:
- We care about INTERNAL GenAI usage (using ChatGPT for customer support, Copilot for coding, AI for content generation)
- We do NOT care about companies that BUILD AI products - we want companies that USE AI tools
- A company selling "AI-powered analytics" is NOT what we're looking for
- A company using "ChatGPT to respond to customer emails" IS what we're looking for

Respond with a JSON object containing your assessment."""


def _build_classifier_prompt(
    company_name: str,
    company_description: Optional[str],
    website_status: WebsiteStatus,
    search_result: TavilySearchResult
) -> str:
    """
    Build the user prompt for the classifier.
    
    Combines all available information into a structured prompt.
    """
    # Website status section
    if website_status.is_alive:
        website_info = f"Website ({website_status.url}): ACTIVE (responded with {website_status.status_code})"
        if website_status.is_redirect:
            website_info += f"\n  → Redirected to: {website_status.final_url}"
    else:
        website_info = f"Website: INACTIVE or UNREACHABLE ({website_status.error})"
    
    # Search results section
    if search_result.error:
        search_info = f"Search failed: {search_result.error}"
    elif not search_result.snippets:
        search_info = "Search returned no results."
    else:
        search_info = f"Search found {len(search_result.snippets)} results:\n"
        for i, snippet in enumerate(search_result.snippets[:5], 1):
            search_info += f"\n{i}. {snippet.title}\n"
            search_info += f"   URL: {snippet.url}\n"
            search_info += f"   Content: {snippet.content[:300]}...\n"
    
    # AI keywords found
    if search_result.ai_keywords_found:
        ai_keywords_info = f"AI-related keywords detected: {', '.join(search_result.ai_keywords_found)}"
    else:
        ai_keywords_info = "No AI-related keywords detected in search results."
    
    prompt = f"""## Company: {company_name}

### Company Description (from Crunchbase)
{company_description or "No description available."}

### Website Status
{website_info}

### Web Search Results
Query: {search_result.query}
{search_info}

### AI Keyword Analysis
{ai_keywords_info}

---

Based on this information, provide your assessment as JSON:

```json
{{
  "online_presence_score": <0-100>,
  "presence_reasoning": "<brief explanation of score>",
  "ai_mentions_found": <true/false>,
  "ai_signals": ["<specific signal 1>", "<specific signal 2>", ...],
  "recommended_stage": "<2A|2B|stop>",
  "routing_reasoning": "<why this routing decision>"
}}
```

ROUTING GUIDELINES:
- "stop": No web presence AND no AI signals → mark as "insufficient information"
- "2A": Low presence OR no AI signals → quick Sonar Base check
- "2B": Good presence AND AI signals found → deeper Sonar Pro investigation

Score guidelines:
- 0-20: Virtually no online presence (dead website, no search results)
- 21-40: Minimal presence (few search results, basic info only)
- 41-60: Moderate presence (active website, some news/mentions)
- 61-80: Good presence (multiple sources, recent activity)
- 81-100: Strong presence (well-documented company, many sources)"""

    return prompt


async def classify_company(
    company_name: str,
    company_description: Optional[str],
    website_status: WebsiteStatus,
    search_result: TavilySearchResult,
    api_key: Optional[str] = None
) -> PresenceAssessment:
    """
    Use GPT-4o-mini to classify a company's online presence and AI signals.
    
    This is the "brain" of Stage 1 - it interprets the raw data and makes
    the routing decision for subsequent stages.
    
    Args:
        company_name: Name of the company.
        company_description: Crunchbase description (if available).
        website_status: Result from check_website().
        search_result: Result from search_tavily().
        api_key: OpenAI API key. Uses env var if not provided.
    
    Returns:
        PresenceAssessment with scores, signals, and routing decision.
    """
    import os
    import json
    
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return PresenceAssessment(
            company_name=company_name,
            online_presence_score=0,
            presence_reasoning="",
            ai_mentions_found=False,
            ai_signals=[],
            recommended_stage="stop",
            routing_reasoning="",
            error="OPENAI_API_KEY not set"
        )
    
    user_prompt = _build_classifier_prompt(
        company_name, company_description, website_status, search_result
    )
    
    async with httpx.AsyncClient(timeout=PROCESSING.api_timeout) as client:
        try:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.1,  # Low temperature for consistent classification
                    "max_tokens": 500,
                    "response_format": {"type": "json_object"},
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # Parse the response
            content = data["choices"][0]["message"]["content"]
            result = json.loads(content)
            
            return PresenceAssessment(
                company_name=company_name,
                online_presence_score=int(result.get("online_presence_score", 0)),
                presence_reasoning=result.get("presence_reasoning", ""),
                ai_mentions_found=bool(result.get("ai_mentions_found", False)),
                ai_signals=result.get("ai_signals", []),
                recommended_stage=result.get("recommended_stage", "2A"),
                routing_reasoning=result.get("routing_reasoning", ""),
            )
            
        except httpx.HTTPStatusError as e:
            return PresenceAssessment(
                company_name=company_name,
                online_presence_score=0,
                presence_reasoning="",
                ai_mentions_found=False,
                ai_signals=[],
                recommended_stage="stop",
                routing_reasoning="",
                error=f"HTTP {e.response.status_code}: {e.response.text[:100]}"
            )
        except json.JSONDecodeError as e:
            return PresenceAssessment(
                company_name=company_name,
                online_presence_score=0,
                presence_reasoning="",
                ai_mentions_found=False,
                ai_signals=[],
                recommended_stage="stop",
                routing_reasoning="",
                error=f"Failed to parse GPT response: {str(e)}"
            )
        except Exception as e:
            return PresenceAssessment(
                company_name=company_name,
                online_presence_score=0,
                presence_reasoning="",
                ai_mentions_found=False,
                ai_signals=[],
                recommended_stage="stop",
                routing_reasoning="",
                error=f"{type(e).__name__}: {str(e)[:100]}"
            )


def classify_company_sync(
    company_name: str,
    company_description: Optional[str],
    website_status: WebsiteStatus,
    search_result: TavilySearchResult,
    api_key: Optional[str] = None
) -> PresenceAssessment:
    """Synchronous wrapper for classify_company."""
    return asyncio.run(classify_company(
        company_name, company_description, website_status, search_result, api_key
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
    ai_mentions_found: bool
    next_stage: str
    
    # Cost tracking
    estimated_cost: float = 0.011  # $0.01 Tavily + $0.001 GPT-4o-mini
    
    @property
    def should_continue(self) -> bool:
        """Should this company proceed to Stage 2?"""
        return self.next_stage in ("2A", "2B", "3")


async def run_stage_1(
    company_id: int,
    company_name: str,
    homepage_url: Optional[str],
    company_description: Optional[str],
    tavily_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None
) -> Stage1Result:
    """
    Run the complete Stage 1 presence filter for a single company.
    
    This orchestrates:
    1. Website health check
    2. Tavily search for AI mentions
    3. GPT-4o-mini classification and routing
    
    Args:
        company_id: Unique identifier (rcid from Crunchbase).
        company_name: Company name.
        homepage_url: Company website URL.
        company_description: Crunchbase description.
        tavily_api_key: Tavily API key.
        openai_api_key: OpenAI API key.
    
    Returns:
        Stage1Result with all component results and routing decision.
    """
    # Step 1: Check website
    website_status = await check_website(homepage_url or "")
    
    # Step 2: Search for AI mentions
    search_result = await search_tavily(
        company_name, 
        homepage_url,
        api_key=tavily_api_key
    )
    
    # Step 3: Classify and route
    assessment = await classify_company(
        company_name,
        company_description,
        website_status,
        search_result,
        api_key=openai_api_key
    )
    
    return Stage1Result(
        company_name=company_name,
        company_id=company_id,
        website_status=website_status,
        search_result=search_result,
        assessment=assessment,
        presence_score=assessment.online_presence_score,
        ai_mentions_found=assessment.ai_mentions_found,
        next_stage=assessment.recommended_stage,
    )


def run_stage_1_sync(
    company_id: int,
    company_name: str,
    homepage_url: Optional[str],
    company_description: Optional[str],
    tavily_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None
) -> Stage1Result:
    """Synchronous wrapper for run_stage_1."""
    return asyncio.run(run_stage_1(
        company_id, company_name, homepage_url, company_description,
        tavily_api_key, openai_api_key
    ))
