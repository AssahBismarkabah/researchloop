from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Source:
    id: str
    title: str
    url: str
    content: str
    retrieved_at: str
    source_type: str = "manual"
    query: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Source":
        return cls(
            id=str(data["id"]),
            title=str(data.get("title") or data["id"]),
            url=str(data.get("url") or ""),
            content=str(data.get("content") or ""),
            retrieved_at=str(data.get("retrieved_at") or ""),
            source_type=str(data.get("source_type") or "manual"),
            query=str(data.get("query") or ""),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class Claim:
    id: str
    text: str
    source_ids: list[str]
    confidence: str = "medium"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Claim":
        raw_source_ids = data.get("source_ids") or []
        return cls(
            id=str(data["id"]),
            text=str(data.get("text") or ""),
            source_ids=[str(item) for item in raw_source_ids],
            confidence=str(data.get("confidence") or "medium"),
            notes=str(data.get("notes") or ""),
        )


@dataclass
class ResearchResult:
    report_markdown: str
    claims: list[Claim]
    gaps: list[str]
    summary: str = ""


@dataclass
class Evaluation:
    score: float
    source_count: int
    claim_count: int
    cited_source_count: int
    unsupported_claim_count: int
    gap_count: int
    citation_count: int
    structure_score: float
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
