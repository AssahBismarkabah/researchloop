from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


POLICY_FILENAME = "source_policy.json"
VALID_SEARCH_DEPTHS = {"basic", "advanced"}


DEFAULT_EXCLUDE_DOMAINS = [
    "facebook.com",
    "instagram.com",
    "medium.com",
    "quora.com",
    "reddit.com",
    "youtube.com",
]


@dataclass
class SourcePolicy:
    version: int = 1
    search_depth: str = "advanced"
    include_domains: list[str] = field(default_factory=list)
    exclude_domains: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE_DOMAINS))
    preferred_source_types: list[str] = field(
        default_factory=lambda: [
            "primary sources",
            "official documentation",
            "standards",
            "academic papers",
            "filings",
            "original data",
            "direct product documentation",
        ]
    )
    notes: list[str] = field(
        default_factory=lambda: [
            "Use include_domains for narrow research where the source universe is known.",
            "Use exclude_domains for noisy social, video, forum, or repost domains.",
            "Manual sources can still be ingested when a blocked domain is intentionally needed.",
        ]
    )

    def __post_init__(self) -> None:
        self.search_depth = str(self.search_depth).strip().lower()
        if self.search_depth not in VALID_SEARCH_DEPTHS:
            raise ValueError(
                f"source policy search_depth must be one of {sorted(VALID_SEARCH_DEPTHS)}, got {self.search_depth!r}"
            )
        self.include_domains = _clean_domains(_as_list(self.include_domains))
        self.exclude_domains = _clean_domains(_as_list(self.exclude_domains))
        self.preferred_source_types = _clean_strings(_as_list(self.preferred_source_types))
        self.notes = _clean_strings(_as_list(self.notes))

    @classmethod
    def default(cls) -> "SourcePolicy":
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourcePolicy":
        return cls(
            version=int(data.get("version") or 1),
            search_depth=str(data.get("search_depth") or "advanced"),
            include_domains=_as_list(data.get("include_domains") or []),
            exclude_domains=_as_list(data.get("exclude_domains") or []),
            preferred_source_types=_as_list(data.get("preferred_source_types") or []),
            notes=_as_list(data.get("notes") or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "search_depth": self.search_depth,
            "include_domains": self.include_domains,
            "exclude_domains": self.exclude_domains,
            "preferred_source_types": self.preferred_source_types,
            "notes": self.notes,
        }

    def allows_url(self, url: str) -> bool:
        host = _host_from_url(url)
        if not host:
            return not self.include_domains
        if _matches_any(host, self.exclude_domains):
            return False
        if self.include_domains and not _matches_any(host, self.include_domains):
            return False
        return True


def load_source_policy(path: Path | None = None) -> SourcePolicy:
    if path is None:
        return SourcePolicy.default()
    if not path.exists():
        raise ValueError(f"source policy file does not exist: {path}")
    return SourcePolicy.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_policy_for_workspace(workspace: Path, explicit_path: Path | None = None) -> SourcePolicy:
    if explicit_path is not None:
        return load_source_policy(explicit_path)
    workspace_policy = workspace / POLICY_FILENAME
    if workspace_policy.exists():
        return load_source_policy(workspace_policy)
    root_policy = Path(POLICY_FILENAME)
    if root_policy.exists():
        return load_source_policy(root_policy)
    return SourcePolicy.default()


def load_default_policy(explicit_path: Path | None = None) -> SourcePolicy:
    if explicit_path is not None:
        return load_source_policy(explicit_path)
    root_policy = Path(POLICY_FILENAME)
    if root_policy.exists():
        return load_source_policy(root_policy)
    return SourcePolicy.default()


def write_source_policy(path: Path, policy: SourcePolicy) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _clean_domains(values: list[Any]) -> list[str]:
    domains: list[str] = []
    for value in values:
        domain = str(value).strip().lower()
        domain = domain.removeprefix("http://").removeprefix("https://")
        domain = domain.split("/", 1)[0]
        domain = domain.removeprefix("www.")
        if domain and domain not in domains:
            domains.append(domain)
    return domains


def _clean_strings(values: list[Any]) -> list[str]:
    strings: list[str] = []
    for value in values:
        item = str(value).strip()
        if item:
            strings.append(item)
    return strings


def _host_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return host.removeprefix("www.")


def _matches_any(host: str, domains: list[str]) -> bool:
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)
