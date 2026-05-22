from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


POLICY_FILENAME = "source_policy.json"
VALID_SEARCH_DEPTHS = {"basic", "advanced"}
VALID_TIME_RANGES = {"day", "week", "month", "year"}
VALID_EXTRACT_FORMATS = {"markdown", "text"}


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
    time_range: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    extract_after_search: bool = True
    extract_depth: str = "basic"
    extract_format: str = "markdown"
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
        self.time_range = str(self.time_range).strip().lower() if self.time_range is not None else None
        if self.time_range == "":
            self.time_range = None
        if self.time_range is not None and self.time_range not in VALID_TIME_RANGES:
            raise ValueError(
                f"source policy time_range must be one of {sorted(VALID_TIME_RANGES)}, got {self.time_range!r}"
            )
        self.start_date = _clean_date(self.start_date, "start_date")
        self.end_date = _clean_date(self.end_date, "end_date")
        if self.start_date and self.end_date and self.start_date >= self.end_date:
            raise ValueError("source policy start_date must be before end_date")
        self.extract_after_search = _as_bool(self.extract_after_search)
        self.extract_depth = str(self.extract_depth).strip().lower()
        if self.extract_depth not in VALID_SEARCH_DEPTHS:
            raise ValueError(
                f"source policy extract_depth must be one of {sorted(VALID_SEARCH_DEPTHS)}, got {self.extract_depth!r}"
            )
        self.extract_format = str(self.extract_format).strip().lower()
        if self.extract_format not in VALID_EXTRACT_FORMATS:
            raise ValueError(
                f"source policy extract_format must be one of {sorted(VALID_EXTRACT_FORMATS)}, got {self.extract_format!r}"
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
            time_range=data.get("time_range"),
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
            extract_after_search=data.get("extract_after_search", True),
            extract_depth=str(data.get("extract_depth") or "basic"),
            extract_format=str(data.get("extract_format") or "markdown"),
            include_domains=_as_list(data.get("include_domains") or []),
            exclude_domains=_as_list(data.get("exclude_domains") or []),
            preferred_source_types=_as_list(data.get("preferred_source_types") or []),
            notes=_as_list(data.get("notes") or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "search_depth": self.search_depth,
            "time_range": self.time_range,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "extract_after_search": self.extract_after_search,
            "extract_depth": self.extract_depth,
            "extract_format": self.extract_format,
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


def policy_for_question(
    policy: SourcePolicy,
    question: str,
    today: date | None = None,
) -> SourcePolicy:
    if policy.start_date or policy.end_date or policy.time_range:
        return policy
    text = question.lower()
    anchor = today or date.today()
    explicit_date = _extract_explicit_date(text)
    if explicit_date is not None:
        day = explicit_date.isoformat()
        return _copy_policy(
            policy,
            start_date=day,
            end_date=_next_day_iso(explicit_date),
            notes=policy.notes
            + [
                f"Explicit topic date detected; search is constrained to {day}.",
            ],
        )
    if _has_strict_today_window(text):
        return _copy_policy(
            policy,
            start_date=anchor.isoformat(),
            end_date=_next_day_iso(anchor),
            notes=policy.notes
            + [
                "Date-sensitive topic detected; search is constrained to the current date.",
            ],
        )
    if _has_last_24_hour_window(text):
        return _copy_policy(
            policy,
            time_range="day",
            notes=policy.notes
            + [
                "Last-24-hours topic detected; search uses Tavily time_range=day.",
            ],
        )
    if _has_recent_window(text):
        return _copy_policy(
            policy,
            time_range="day",
            notes=policy.notes
            + [
                "Recent/current topic detected; search uses Tavily time_range=day.",
            ],
        )
    return policy


def _copy_policy(policy: SourcePolicy, **overrides: Any) -> SourcePolicy:
    data = policy.to_dict()
    data.update(overrides)
    return SourcePolicy.from_dict(data)


def _next_day_iso(day: date) -> str:
    return (day + timedelta(days=1)).isoformat()


def _has_strict_today_window(text: str) -> bool:
    patterns = [
        "today",
        "daily briefing",
        "daily tech briefing",
        "last 12 hours",
        "past 12 hours",
    ]
    return any(pattern in text for pattern in patterns)


def _has_last_24_hour_window(text: str) -> bool:
    return any(pattern in text for pattern in ["last 24 hours", "past 24 hours", "older than 24 hours"])


def _has_recent_window(text: str) -> bool:
    patterns = [
        "latest",
        "current",
        "recent",
        "this week",
    ]
    return any(pattern in text for pattern in patterns)


def _extract_explicit_date(text: str) -> date | None:
    import re

    iso_match = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", text)
    if iso_match:
        try:
            return date.fromisoformat(iso_match.group(0))
        except ValueError:
            return None

    month_names = (
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    )
    month_re = "|".join(month_names)
    patterns = [
        rf"\b(\d{{1,2}})\s+({month_re})\s+(20\d{{2}})\b",
        rf"\b({month_re})\s+(\d{{1,2}}),?\s+(20\d{{2}})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        try:
            if match.group(1).isdigit():
                day = int(match.group(1))
                month = month_names.index(match.group(2)) + 1
                year = int(match.group(3))
            else:
                month = month_names.index(match.group(1)) + 1
                day = int(match.group(2))
                year = int(match.group(3))
            return date(year, month, day)
        except (ValueError, IndexError):
            return None
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(value)


def _clean_date(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ValueError(f"source policy {field_name} must use YYYY-MM-DD, got {text!r}") from exc


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
