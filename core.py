from __future__ import annotations

from pathlib import Path

from llm import ResearchLLM
from models import Claim, Evaluation, Source
from scoring import evaluate_report, format_evaluation
from search import SearchBackend
from storage import (
    append_result_row,
    ensure_results_file,
    load_claims,
    load_sources,
    next_source_id,
    read_json,
    read_text,
    safe_stamp,
    save_claims,
    save_sources,
    utc_now,
    workspace_path,
    write_json,
    write_jsonl,
    write_text,
)


def init_workspace(root: Path, name: str, question: str) -> Path:
    workspace = workspace_path(root, name)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "iterations").mkdir(exist_ok=True)
    write_text(
        workspace / "topic.md",
        f"""# Research Topic

## Question

{question.strip()}

## Source Policy

- Prefer primary sources, official documentation, academic papers, standards,
  filings, original data, and direct product documentation.
- Use secondary sources only when they add context that primary sources do not.
- Every substantive report claim must cite source IDs like [S1].
- Record uncertainty and open gaps instead of forcing completeness.

## Evaluation Goal

Produce a source-backed report that improves coverage, citation quality,
contradiction handling, and updateability over a one-shot answer.
""",
    )
    write_text(workspace / "report.md", "# Research Report\n\nNo kept report yet.\n")
    write_jsonl(workspace / "sources.jsonl", [])
    write_jsonl(workspace / "claims.jsonl", [])
    write_json(workspace / "state.json", {"best_score": 0.0, "best_iteration": None})
    ensure_results_file(workspace)
    return workspace


def add_manual_source(
    workspace: Path,
    title: str,
    content: str,
    url: str = "",
    source_type: str = "manual",
) -> Source:
    sources = load_sources(workspace)
    source = Source(
        id=next_source_id(sources),
        title=title,
        url=url,
        content=content,
        retrieved_at=utc_now(),
        source_type=source_type,
    )
    sources.append(source)
    save_sources(workspace, sources)
    return source


def run_iteration(
    workspace: Path,
    llm: ResearchLLM,
    search_backend: SearchBackend,
    max_results: int = 5,
    min_delta: float = 0.1,
) -> dict[str, object]:
    timestamp = utc_now()
    iteration_id = _next_iteration_id(workspace)
    iteration_dir = workspace / "iterations" / iteration_id
    iteration_dir.mkdir(parents=True, exist_ok=False)

    topic = read_text(workspace / "topic.md")
    previous_report = read_text(workspace / "report.md")
    previous_claims = load_claims(workspace)
    sources = load_sources(workspace)
    previous_eval = read_json(workspace / "state.json", {"best_score": 0.0})

    gaps = _extract_prior_gaps(workspace)
    queries = llm.plan_queries(topic, previous_report, gaps)
    added_sources = _collect_sources(search_backend, sources, queries, max_results)
    if added_sources:
        sources.extend(added_sources)
        save_sources(workspace, sources)

    candidate = llm.synthesize(topic, sources, previous_report, previous_claims)
    evaluation = evaluate_report(candidate.report_markdown, candidate.claims, sources, candidate.gaps)
    best_score = float(previous_eval.get("best_score") or 0.0)
    has_report = bool(candidate.report_markdown.strip())
    keep = has_report and evaluation.score > 0 and (best_score == 0.0 or evaluation.score >= best_score + min_delta)
    status = "keep" if keep else "discard"

    write_text(iteration_dir / "candidate_report.md", candidate.report_markdown + "\n")
    write_jsonl(iteration_dir / "candidate_claims.jsonl", [claim.to_dict() for claim in candidate.claims])
    write_jsonl(iteration_dir / "added_sources.jsonl", [source.to_dict() for source in added_sources])
    write_json(iteration_dir / "queries.json", {"queries": queries})
    write_json(iteration_dir / "evaluation.json", evaluation.to_dict())
    write_text(iteration_dir / "summary.md", _format_iteration_summary(candidate.summary, evaluation, status))

    if keep:
        write_text(workspace / "report.md", candidate.report_markdown + "\n")
        save_claims(workspace, _renumber_claims(candidate.claims))
        write_text(workspace / "eval.md", format_evaluation(evaluation))
        write_json(
            workspace / "state.json",
            {
                "best_score": evaluation.score,
                "best_iteration": iteration_id,
                "updated_at": timestamp,
            },
        )

    append_result_row(
        workspace=workspace,
        iteration=iteration_id,
        timestamp=timestamp,
        score=evaluation.score,
        status=status,
        sources=evaluation.source_count,
        claims=evaluation.claim_count,
        unsupported=evaluation.unsupported_claim_count,
        gaps=evaluation.gap_count,
        backend=llm.name,
        search_backend=search_backend.name,
        description=candidate.summary or "research iteration",
    )

    return {
        "iteration": iteration_id,
        "status": status,
        "score": evaluation.score,
        "best_score": max(best_score, evaluation.score) if keep else best_score,
        "evaluation": evaluation,
        "added_sources": len(added_sources),
    }


def evaluate_workspace(workspace: Path) -> Evaluation:
    sources = load_sources(workspace)
    claims = load_claims(workspace)
    report = read_text(workspace / "report.md")
    gaps = _extract_report_gaps(report)
    evaluation = evaluate_report(report, claims, sources, gaps)
    write_text(workspace / "eval.md", format_evaluation(evaluation))
    return evaluation


def _collect_sources(
    search_backend: SearchBackend,
    existing_sources: list[Source],
    queries: list[str],
    max_results: int,
) -> list[Source]:
    if search_backend.name == "none":
        return []
    seen_urls = {source.url for source in existing_sources if source.url}
    collected: list[Source] = []
    sources_so_far = list(existing_sources)
    for query in queries:
        for candidate in search_backend.search(query, max_results=max_results):
            if candidate.url and candidate.url in seen_urls:
                continue
            candidate.id = next_source_id(sources_so_far + collected)
            collected.append(candidate)
            if candidate.url:
                seen_urls.add(candidate.url)
    return collected


def _next_iteration_id(workspace: Path) -> str:
    existing = sorted((workspace / "iterations").glob("*"))
    index = len([path for path in existing if path.is_dir()]) + 1
    return f"{safe_stamp()}-{index:03d}"


def _renumber_claims(claims: list[Claim]) -> list[Claim]:
    return [
        Claim(
            id=f"C{index}",
            text=claim.text,
            source_ids=claim.source_ids,
            confidence=claim.confidence,
            notes=claim.notes,
        )
        for index, claim in enumerate(claims, start=1)
    ]


def _extract_prior_gaps(workspace: Path) -> list[str]:
    eval_text = read_text(workspace / "eval.md")
    gaps = []
    for line in eval_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ") and "gap" in stripped.lower():
            gaps.append(stripped[2:])
    return gaps


def _extract_report_gaps(markdown: str) -> list[str]:
    gaps: list[str] = []
    in_gaps = False
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.lower() == "## open gaps":
            in_gaps = True
            continue
        if in_gaps and stripped.startswith("## "):
            break
        if in_gaps and stripped.startswith("- "):
            gaps.append(stripped[2:].strip())
    return gaps


def _format_iteration_summary(summary: str, evaluation: Evaluation, status: str) -> str:
    return f"""# Iteration Summary

Status: {status}
Score: {evaluation.score:.2f}/100

## Change Summary

{summary or "No summary returned."}

## Evaluator Notes

{chr(10).join(f"- {note}" for note in evaluation.notes) or "- No evaluator notes."}
"""
