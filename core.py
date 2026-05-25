from __future__ import annotations

from pathlib import Path
from typing import Callable

from llm import ResearchLLM
from models import Claim, Evaluation, Source
from run_config import RUN_CONFIG_FILENAME, RunConfig, write_run_config
from scoring import evaluate_report, format_evaluation
from search import SearchBackend
from source_policy import POLICY_FILENAME, SourcePolicy, policy_for_question, write_source_policy
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
from task_requirements import missing_requirements


ProgressCallback = Callable[[str, str, dict[str, object]], None]


def init_workspace(
    root: Path,
    name: str,
    question: str,
    source_policy: SourcePolicy | None = None,
    run_config: RunConfig | None = None,
) -> Path:
    workspace = workspace_path(root, name)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "iterations").mkdir(exist_ok=True)
    policy = policy_for_question(source_policy or SourcePolicy.default(), question)
    write_text(
        workspace / "topic.md",
        f"""# Research Topic

## Question

{question.strip()}

## Source Policy

Operational source-selection rules are stored in `{POLICY_FILENAME}`.
Run behavior is stored in `{RUN_CONFIG_FILENAME}`.

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
    write_source_policy(workspace / POLICY_FILENAME, policy)
    write_run_config(workspace / RUN_CONFIG_FILENAME, run_config or RunConfig.default())
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
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    timestamp = utc_now()
    iteration_id = _next_iteration_id(workspace)
    iteration_dir = workspace / "iterations" / iteration_id
    iteration_dir.mkdir(parents=True, exist_ok=False)

    _emit_progress(progress, "workspace", "Preparing iteration", f"Iteration {iteration_id}")
    topic = read_text(workspace / "topic.md")
    previous_report = read_text(workspace / "report.md")
    previous_claims = load_claims(workspace)
    sources = load_sources(workspace)
    previous_eval = read_json(workspace / "state.json", {"best_score": 0.0})

    gaps = _extract_prior_gaps(workspace)
    if search_backend.name == "none":
        queries = []
        _emit_progress(progress, "search", "Using saved sources", f"{len(sources)} sources available", source_count=len(sources))
    else:
        _emit_progress(
            progress,
            "planning",
            "Planning search queries",
            f"{len(sources)} saved sources; {len(gaps)} prior gaps",
            source_count=len(sources),
            gap_count=len(gaps),
        )
        queries = llm.plan_queries(topic, previous_report, gaps)
        _emit_progress(progress, "planning", "Search plan ready", f"{len(queries)} queries", query_count=len(queries))
    added_sources = _collect_sources(search_backend, sources, queries, max_results, progress=progress)
    if added_sources:
        sources.extend(added_sources)
        save_sources(workspace, sources)
    write_json(iteration_dir / "queries.json", {"queries": queries})
    write_jsonl(iteration_dir / "added_sources.jsonl", [source.to_dict() for source in added_sources])

    try:
        _emit_progress(
            progress,
            "synthesis",
            "Writing candidate report",
            f"{len(sources)} sources available",
            source_count=len(sources),
        )
        candidate = llm.synthesize(topic, sources, previous_report, previous_claims)
    except Exception as exc:
        _emit_progress(progress, "error", "Synthesis failed", str(exc), state="error", source_count=len(sources))
        write_json(
            iteration_dir / "error.json",
            {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "stage": "synthesis",
                "sources_available": len(sources),
            },
        )
        append_result_row(
            workspace=workspace,
            iteration=iteration_id,
            timestamp=timestamp,
            score=0.0,
            status="error",
            sources=len(sources),
            claims=len(previous_claims),
            unsupported=0,
            gaps=len(gaps),
            backend=llm.name,
            search_backend=search_backend.name,
            description=f"synthesis failed: {exc}",
        )
        raise
    _emit_progress(
        progress,
        "evaluation",
        "Checking citations and score",
        f"{len(candidate.claims)} claims returned",
        claim_count=len(candidate.claims),
    )
    evaluation = evaluate_report(candidate.report_markdown, candidate.claims, sources, candidate.gaps)
    _apply_task_compliance(topic, candidate.report_markdown, evaluation)
    best_score = float(previous_eval.get("best_score") or 0.0)
    has_report = bool(candidate.report_markdown.strip())
    keep = has_report and evaluation.score > 0 and (best_score == 0.0 or evaluation.score >= best_score + min_delta)
    status = "keep" if keep else "discard"
    _emit_progress(
        progress,
        "saving",
        "Saving result",
        f"{status} candidate with score {evaluation.score:.2f}",
        score=evaluation.score,
        decision=status,
    )

    write_text(iteration_dir / "candidate_report.md", candidate.report_markdown + "\n")
    write_jsonl(iteration_dir / "candidate_claims.jsonl", [claim.to_dict() for claim in candidate.claims])
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

    _emit_progress(
        progress,
        "done",
        "Done",
        f"{status} with score {evaluation.score:.2f}",
        state="done",
        score=evaluation.score,
        decision=status,
        source_count=evaluation.source_count,
        claim_count=evaluation.claim_count,
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
    progress: ProgressCallback | None = None,
) -> list[Source]:
    if search_backend.name == "none":
        return []
    seen_urls = {source.url for source in existing_sources if source.url}
    collected: list[Source] = []
    sources_so_far = list(existing_sources)
    total_queries = len(queries)
    for index, query in enumerate(queries, start=1):
        _emit_progress(
            progress,
            "search",
            f"Searching sources {index} of {total_queries}",
            query,
            query_index=index,
            query_count=total_queries,
            source_count=len(sources_so_far) + len(collected),
        )
        before_count = len(collected)
        for candidate in search_backend.search(query, max_results=max_results):
            if candidate.url and candidate.url in seen_urls:
                continue
            candidate.id = next_source_id(sources_so_far + collected)
            collected.append(candidate)
            if candidate.url:
                seen_urls.add(candidate.url)
        added_count = len(collected) - before_count
        _emit_progress(
            progress,
            "search",
            f"Collected {added_count} new sources",
            query,
            query_index=index,
            query_count=total_queries,
            source_count=len(sources_so_far) + len(collected),
            added_sources=added_count,
        )
    return collected


def _emit_progress(
    progress: ProgressCallback | None,
    step: str,
    title: str,
    detail: str = "",
    **metadata: object,
) -> None:
    if progress is None:
        return
    payload = dict(metadata)
    payload["detail"] = detail
    progress(step, title, payload)


def _apply_task_compliance(topic: str, report_markdown: str, evaluation: Evaluation) -> None:
    missing = missing_requirements(topic, report_markdown)
    if not missing:
        return
    evaluation.score = 0.0
    evaluation.notes.append("Missing requested deliverable section(s): " + ", ".join(missing))


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
