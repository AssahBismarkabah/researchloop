from __future__ import annotations

import re

from models import Claim, Evaluation, Source

_CITATION_RE = re.compile(r"\[(S\d+)\]")


def citation_ids(markdown: str) -> list[str]:
    return _CITATION_RE.findall(markdown)


def evaluate_report(
    report_markdown: str,
    claims: list[Claim],
    sources: list[Source],
    gaps: list[str],
) -> Evaluation:
    known_source_ids = {source.id for source in sources}
    cited_ids = set(citation_ids(report_markdown))
    supported_claims = [
        claim for claim in claims if claim.source_ids and all(sid in known_source_ids for sid in claim.source_ids)
    ]
    unsupported_claim_count = len(claims) - len(supported_claims)
    claim_count = len(claims)
    citation_count = len(citation_ids(report_markdown))
    citation_coverage = len(supported_claims) / claim_count if claim_count else 0.0
    cited_source_count = len(cited_ids & known_source_ids)
    source_diversity = min(cited_source_count / 4, 1.0)
    claim_volume = min(claim_count / 8, 1.0)
    report_citation_density = min(citation_count / max(claim_count, 1), 1.0)
    structure_score = _structure_score(report_markdown)

    score = (
        30.0 * citation_coverage
        + 20.0 * source_diversity
        + 20.0 * claim_volume
        + 20.0 * report_citation_density
        + 10.0 * structure_score
    )
    score -= 8.0 * unsupported_claim_count
    score -= 2.0 * max(0, len(gaps) - 4)
    score = max(0.0, min(100.0, score))

    notes: list[str] = []
    if not sources:
        notes.append("No sources are available; this run cannot be treated as evidence-backed research.")
    if unsupported_claim_count:
        notes.append(f"{unsupported_claim_count} claim(s) have missing or invalid source IDs.")
    if cited_source_count < min(2, len(sources)):
        notes.append("The report cites too few distinct sources.")
    if structure_score < 1.0:
        notes.append("The report is missing one or more expected sections.")
    if not gaps:
        notes.append("No open gaps were recorded; verify that this is not false completeness.")

    return Evaluation(
        score=round(score, 2),
        source_count=len(sources),
        claim_count=claim_count,
        cited_source_count=cited_source_count,
        unsupported_claim_count=unsupported_claim_count,
        gap_count=len(gaps),
        citation_count=citation_count,
        structure_score=round(structure_score, 2),
        notes=notes,
    )


def format_evaluation(evaluation: Evaluation) -> str:
    notes = "\n".join(f"- {note}" for note in evaluation.notes) or "- No evaluator notes."
    return f"""# Evaluation

Score: {evaluation.score:.2f}/100

## Metrics

- Sources: {evaluation.source_count}
- Claims: {evaluation.claim_count}
- Cited sources: {evaluation.cited_source_count}
- Citations in report: {evaluation.citation_count}
- Unsupported claims: {evaluation.unsupported_claim_count}
- Open gaps: {evaluation.gap_count}
- Structure score: {evaluation.structure_score:.2f}

## Notes

{notes}
"""


def _structure_score(markdown: str) -> float:
    expected = ["## Current Answer", "## Evidence", "## Open Gaps", "## Sources"]
    present = sum(1 for heading in expected if heading.lower() in markdown.lower())
    return present / len(expected)
