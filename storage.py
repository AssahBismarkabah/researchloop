from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from models import Claim, Source


RESULTS_HEADER = (
    "iteration\ttimestamp\tscore\tstatus\tsources\tclaims\tunsupported\tgaps\t"
    "backend\tsearch_backend\tdescription\n"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "research-topic"


def workspace_path(root: Path, name: str) -> Path:
    return root / slugify(name)


def read_text(path: Path, default: str = "") -> str:
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(body, encoding="utf-8")


def load_sources(workspace: Path) -> list[Source]:
    return [Source.from_dict(row) for row in read_jsonl(workspace / "sources.jsonl")]


def save_sources(workspace: Path, sources: list[Source]) -> None:
    write_jsonl(workspace / "sources.jsonl", [source.to_dict() for source in sources])


def load_claims(workspace: Path) -> list[Claim]:
    return [Claim.from_dict(row) for row in read_jsonl(workspace / "claims.jsonl")]


def save_claims(workspace: Path, claims: list[Claim]) -> None:
    write_jsonl(workspace / "claims.jsonl", [claim.to_dict() for claim in claims])


def next_source_id(sources: list[Source]) -> str:
    max_seen = 0
    for source in sources:
        if source.id.startswith("S") and source.id[1:].isdigit():
            max_seen = max(max_seen, int(source.id[1:]))
    return f"S{max_seen + 1}"


def ensure_results_file(workspace: Path) -> None:
    path = workspace / "results.tsv"
    if not path.exists():
        write_text(path, RESULTS_HEADER)


def append_result_row(
    workspace: Path,
    iteration: str,
    timestamp: str,
    score: float,
    status: str,
    sources: int,
    claims: int,
    unsupported: int,
    gaps: int,
    backend: str,
    search_backend: str,
    description: str,
) -> None:
    ensure_results_file(workspace)
    clean_description = description.replace("\t", " ").replace("\n", " ").strip()
    row = (
        f"{iteration}\t{timestamp}\t{score:.2f}\t{status}\t{sources}\t{claims}\t"
        f"{unsupported}\t{gaps}\t{backend}\t{search_backend}\t{clean_description}\n"
    )
    with (workspace / "results.tsv").open("a", encoding="utf-8") as handle:
        handle.write(row)
