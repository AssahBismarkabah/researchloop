from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

from models import Source
from source_policy import SourcePolicy
from storage import utc_now


class SearchError(RuntimeError):
    pass


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
        request = urllib.request.Request(
            "https://api.tavily.com/search",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise SearchError(f"Tavily returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise SearchError(f"Could not reach Tavily: {exc}") from exc

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
        return sources


def build_search_backend(name: str, source_policy: SourcePolicy | None = None) -> SearchBackend:
    if name == "none":
        return NoSearch()
    if name == "tavily":
        return TavilySearch(source_policy=source_policy)
    raise ValueError(f"Unknown search backend: {name}")
