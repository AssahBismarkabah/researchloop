from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlparse

from models import Source
from retry import call_with_retries
from source_policy import SourcePolicy
from storage import utc_now


class SearchError(RuntimeError):
    def __init__(self, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class SearchBackend(ABC):
    name: str

    @abstractmethod
    def search(self, query: str, max_results: int) -> list[Source]:
        raise NotImplementedError


class NoSearch(SearchBackend):
    name = "none"

    def search(self, query: str, max_results: int) -> list[Source]:
        return []


class TavilySearch(SearchBackend):
    name = "tavily"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int = 60,
        source_policy: SourcePolicy | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.timeout = timeout
        self.source_policy = source_policy or SourcePolicy.default()
        if not self.api_key:
            raise SearchError("Missing TAVILY_API_KEY for Tavily search.")

    def search(self, query: str, max_results: int) -> list[Source]:
        direct_url = _direct_url(query)
        if direct_url:
            return self._fetch_url(direct_url, query)
        body = {
            "query": query,
            "search_depth": self.source_policy.search_depth,
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": True,
        }
        if self.source_policy.start_date or self.source_policy.end_date or self.source_policy.time_range:
            body["topic"] = "news"
        if self.source_policy.include_domains:
            body["include_domains"] = self.source_policy.include_domains
        if self.source_policy.exclude_domains:
            body["exclude_domains"] = self.source_policy.exclude_domains
        if self.source_policy.start_date:
            body["start_date"] = self.source_policy.start_date
        if self.source_policy.end_date:
            body["end_date"] = self.source_policy.end_date
        if self.source_policy.time_range and not (self.source_policy.start_date or self.source_policy.end_date):
            body["time_range"] = self.source_policy.time_range
        payload = self._post_json("/search", body)

        sources = []
        for result in payload.get("results") or []:
            result_date = _result_date(result)
            if not self._result_date_allowed(result_date):
                continue
            url = str(result.get("url") or "")
            if not self.source_policy.allows_url(url):
                continue
            content = result.get("raw_content") or result.get("content") or ""
            sources.append(
                Source(
                    id="",
                    title=str(result.get("title") or result.get("url") or "Untitled source"),
                    url=url,
                    content=str(content),
                    retrieved_at=utc_now(),
                    source_type="web",
                    query=query,
                    metadata={
                        "score": result.get("score"),
                        "published_date": result.get("published_date"),
                        "result_date": result_date.isoformat() if result_date else None,
                        "source_policy_version": self.source_policy.version,
                        "source_policy_time_range": self.source_policy.time_range,
                        "source_policy_start_date": self.source_policy.start_date,
                        "source_policy_end_date": self.source_policy.end_date,
                    },
                )
            )
        if self.source_policy.extract_after_search:
            self._extract_sources(sources)
        return sources

    def _fetch_url(self, url: str, query: str) -> list[Source]:
        source = Source(
            id="",
            title=url,
            url=url,
            content="",
            retrieved_at=utc_now(),
            source_type="web",
            query=query,
            metadata={
                "explicit_fetch": True,
                "source_policy_version": self.source_policy.version,
                "source_policy_time_range": self.source_policy.time_range,
                "source_policy_start_date": self.source_policy.start_date,
                "source_policy_end_date": self.source_policy.end_date,
            },
        )
        self._extract_sources([source])
        if source.metadata.get("extract_failed") and not source.content:
            return []
        return [source]

    def _extract_sources(self, sources: list[Source]) -> None:
        url_sources = [source for source in sources if source.url]
        for chunk in _chunks(url_sources, 20):
            payload = self._post_json(
                "/extract",
                {
                    "urls": [source.url for source in chunk],
                    "extract_depth": self.source_policy.extract_depth,
                    "format": self.source_policy.extract_format,
                },
            )
            extracted_by_url = {
                str(result.get("url") or ""): str(result.get("raw_content") or "")
                for result in payload.get("results") or []
            }
            failed_urls = [str(result.get("url") or "") for result in payload.get("failed_results") or []]
            for source in chunk:
                source.metadata["extract_attempted"] = True
                extracted = extracted_by_url.get(source.url, "")
                if extracted:
                    source.metadata["extract_content_length"] = len(extracted)
                if extracted and len(extracted) > len(source.content):
                    source.content = extracted
                    source.metadata["extracted"] = True
                    source.metadata["extract_depth"] = self.source_policy.extract_depth
                    source.metadata["extract_format"] = self.source_policy.extract_format
                elif source.url in failed_urls:
                    source.metadata["extract_failed"] = True

    def _result_date_allowed(self, result_date: date | None) -> bool:
        if result_date is None:
            return not (self.source_policy.start_date or self.source_policy.end_date)
        start = _parse_result_date(self.source_policy.start_date)
        end = _parse_result_date(self.source_policy.end_date)
        if start and result_date < start:
            return False
        if end and result_date >= end:
            return False
        if not (start or end):
            floor = _time_range_floor(self.source_policy.time_range)
            if floor and result_date < floor:
                return False
        return True

    def _post_json(self, path: str, body: dict[str, object]) -> dict[str, object]:
        def once() -> dict[str, object]:
            request = urllib.request.Request(
                f"https://api.tavily.com{path}",
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code in {429, 500, 502, 503, 504}
                raise SearchError(f"Tavily returned HTTP {exc.code}: {detail}", retryable=retryable) from exc
            except urllib.error.URLError as exc:
                raise SearchError(f"Could not reach Tavily: {exc}") from exc
            except TimeoutError as exc:
                raise SearchError(f"Tavily timed out after {self.timeout} seconds.", retryable=True) from exc

        return call_with_retries(once, lambda exc: bool(getattr(exc, "retryable", False)))


def build_search_backend(name: str, source_policy: SourcePolicy | None = None) -> SearchBackend:
    if name == "none":
        return NoSearch()
    if name == "tavily":
        return TavilySearch(source_policy=source_policy)
    raise ValueError(f"Unknown search backend: {name}")


def _chunks(values: list[Source], size: int) -> list[list[Source]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _direct_url(query: str) -> str | None:
    match = re.search(r"https?://[^\s`\"')]+", query.strip())
    if not match:
        return None
    url = match.group(0).rstrip(".,;:")
    parsed = urlparse(url)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return url
    return None


def _result_date(result: object) -> date | None:
    if not isinstance(result, dict):
        return None
    published = _parse_result_date(result.get("published_date"))
    if published is not None:
        return published
    twitter_date = _twitter_status_date(result.get("url"))
    if twitter_date is not None:
        return twitter_date
    for field in ("url", "title"):
        inferred = _find_date_in_text(result.get(field))
        if inferred is not None:
            return inferred
    return None


def _parse_result_date(value: object) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        pass
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _find_date_in_text(value: object) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for pattern in (
        r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b",
        r"\b(\d{1,2})[-/](\d{1,2})[-/](20\d{2})\b",
    ):
        match = re.search(pattern, text)
        if not match:
            continue
        try:
            if len(match.group(1)) == 4:
                return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            return date(int(match.group(3)), int(match.group(2)), int(match.group(1)))
        except ValueError:
            continue
    return _parse_result_date(text)


def _twitter_status_date(value: object) -> date | None:
    if value is None:
        return None
    parsed = urlparse(str(value))
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if host not in {"x.com", "twitter.com", "mobile.twitter.com"}:
        return None
    match = re.search(r"/(?:i/)?status(?:es)?/(\d+)", parsed.path)
    if not match:
        return None
    try:
        snowflake = int(match.group(1))
    except ValueError:
        return None
    timestamp_ms = (snowflake >> 22) + 1288834974657
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).date()


def _time_range_floor(time_range: str | None) -> date | None:
    if time_range == "day":
        return date.today() - timedelta(days=1)
    if time_range == "week":
        return date.today() - timedelta(days=7)
    if time_range == "month":
        return date.today() - timedelta(days=31)
    if time_range == "year":
        return date.today() - timedelta(days=366)
    return None
