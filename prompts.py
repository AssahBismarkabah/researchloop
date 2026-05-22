from __future__ import annotations

from datetime import datetime, timezone

from models import Claim, Source


SYSTEM_PROMPT = """You are a careful research agent.
Use only the supplied source records as evidence.
Every substantive claim must cite source IDs in square brackets, such as [S1].
If the source set is weak, say so directly and record open gaps.
Return valid JSON only when asked for JSON.
"""


def query_plan_prompt(topic: str, previous_report: str, gaps: list[str]) -> str:
    gap_text = "\n".join(f"- {gap}" for gap in gaps) or "- No prior gaps."
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"""Research topic:
{topic}

Current date (UTC): {today}

Current report excerpt:
{previous_report[:3000] if previous_report else "No report yet."}

Known gaps:
{gap_text}

Return JSON with this shape:
{{
  "queries": ["specific search query", "..."],
  "rationale": "short reason"
}}

Create 3 to 5 targeted queries. Prefer primary sources, official documentation,
academic papers, standards, filings, and original data where appropriate.
If the topic asks for "today", "latest", "current", or a recent time window,
make each query explicitly current-date-aware. Do not add stale years unless the
topic asks for historical research.
If the topic includes explicit source instructions, required URLs, or named
domains, preserve them as targeted search queries where possible.
"""


def synthesis_prompt(
    topic: str,
    sources: list[Source],
    previous_report: str,
    previous_claims: list[Claim],
) -> str:
    source_blocks = []
    for source in sources[:25]:
        content = source.content.strip().replace("\x00", "")
        source_blocks.append(
            f"[{source.id}] {source.title}\nURL: {source.url or 'n/a'}\n"
            f"Type: {source.source_type}\nRetrieved: {source.retrieved_at}\n"
            f"Query: {source.query or 'n/a'}\nContent:\n{content[:220]}"
        )
    source_text = "\n\n---\n\n".join(source_blocks) or "No sources supplied."
    claim_text = "\n".join(
        f"- {claim.id}: {claim.text} ({', '.join(claim.source_ids)})"
        for claim in previous_claims[:20]
    ) or "No prior claims."

    return f"""Research topic:
{topic}

Previous report:
{previous_report[:4000] if previous_report else "No prior report."}

Previous claims:
{claim_text}

Source records:
{source_text}

Return JSON with this shape:
{{
  "summary": "short summary of what changed in this iteration",
  "current_answer": ["short cited paragraph or bullet", "..."],
  "evidence": [
    {{
      "claim": "evidence-backed claim with citation markup like [S1]",
      "source_ids": ["S1"]
    }}
  ],
  "claims": [
    {{
      "text": "atomic claim with no citation markup inside the text",
      "source_ids": ["S1"],
      "confidence": "high|medium|low",
      "notes": "optional caveat"
    }}
  ],
  "gaps": ["open question or missing source"]
}}

Report requirements:
- Cite every substantive sentence with source IDs like [S1].
- Do not cite sources that were not supplied.
- Do not hide uncertainty. If evidence is thin, make that visible.
- Do not use emoji or decorative symbols in headings.
- Preserve explicit formatting requirements from the research topic. If the
  topic asks for named sections such as Problems, Pain Points, Investment
  Opportunities, categories, tables, or action items, include them in the report.
- Keep current_answer to 4-6 concise items.
- Keep evidence to 6-8 high-value items.
- Return no more than 8 high-value claims.
- Prefer synthesis over repeating source descriptions.
"""
