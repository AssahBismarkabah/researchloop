from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

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
        body = {
            "query": query,
            "search_depth": self.source_policy.search_depth,
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": True,
        }
        if self.source_policy.include_domains:
            body["include_domains"] = self.source_policy.include_domains
        if self.source_policy.exclude_domains:
            body["exclude_domains"] = self.source_policy.exclude_domains
        if self.source_policy.time_range:
            body["time_range"] = self.source_policy.time_range
        payload = self._post_json("/search", body)

        sources = []
        for result in payload.get("results") or []:
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
                        "source_policy_version": self.source_policy.version,
                    },
                )
            )
        if self.source_policy.extract_after_search:
            self._extract_sources(sources)
        return sources

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
