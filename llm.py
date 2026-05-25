from __future__ import annotations

import http.client
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from models import Claim, ResearchResult, Source
from prompts import SYSTEM_PROMPT, query_plan_prompt, synthesis_prompt
from retry import call_with_retries


class LLMError(RuntimeError):
    def __init__(self, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


_RETRYABLE_TRANSPORT_ERRORS = (
    TimeoutError,
    ConnectionResetError,
    ConnectionAbortedError,
    BrokenPipeError,
    http.client.RemoteDisconnected,
)


class ResearchLLM(ABC):
    name: str

    @abstractmethod
    def plan_queries(self, topic: str, previous_report: str, gaps: list[str]) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def synthesize(
        self,
        topic: str,
        sources: list[Source],
        previous_report: str,
        previous_claims: list[Claim],
    ) -> ResearchResult:
        raise NotImplementedError


class OpenAICompatibleLLM(ResearchLLM):
    """LLM adapter for OpenAI-compatible chat completions endpoints."""

    name = "openai-compatible"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        timeout: int = 90,
        synthesis_mode: str | None = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.getenv("OPENAI_COMPAT_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_COMPAT_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("RESEARCH_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        self.temperature = temperature
        self.timeout = int(os.getenv("RESEARCH_LLM_TIMEOUT", str(timeout)))
        self.attempts = max(1, int(os.getenv("RESEARCH_LLM_RETRY_ATTEMPTS", "2")))
        self.synthesis_mode = (synthesis_mode or os.getenv("RESEARCH_SYNTHESIS_MODE", "json")).strip().lower()
        if self.synthesis_mode not in {"json", "markdown"}:
            raise LLMError("synthesis_mode must be 'json' or 'markdown'.")
        if not self.api_key:
            raise LLMError(
                "Missing API key. Set OPENAI_COMPAT_API_KEY or OPENAI_API_KEY for the "
                "openai-compatible backend."
            )

    def plan_queries(self, topic: str, previous_report: str, gaps: list[str]) -> list[str]:
        try:
            payload = self._complete_json(query_plan_prompt(topic, previous_report, gaps))
        except (LLMError, json.JSONDecodeError):
            return [_extract_question(topic)]
        queries = payload.get("queries") or []
        return [str(query).strip() for query in queries if str(query).strip()][:5]

    def synthesize(
        self,
        topic: str,
        sources: list[Source],
        previous_report: str,
        previous_claims: list[Claim],
    ) -> ResearchResult:
        if self.synthesis_mode == "markdown":
            return self._synthesize_markdown(topic, sources, previous_report, previous_claims)
        try:
            payload = self._complete_json(synthesis_prompt(topic, sources, previous_report, previous_claims))
        except (LLMError, json.JSONDecodeError):
            return self._synthesize_markdown(topic, sources, previous_report, previous_claims)
        claims = []
        for index, raw_claim in enumerate(payload.get("claims") or [], start=1):
            raw_source_ids = raw_claim.get("source_ids") or []
            claims.append(
                Claim(
                    id=f"C{index}",
                    text=str(raw_claim.get("text") or "").strip(),
                    source_ids=[str(item) for item in raw_source_ids],
                    confidence=str(raw_claim.get("confidence") or "medium"),
                    notes=str(raw_claim.get("notes") or ""),
                )
            )
        report_markdown = str(payload.get("report_markdown") or "").strip()
        if not report_markdown:
            report_markdown = _build_report_markdown(
                topic=topic,
                sources=sources,
                current_answer=payload.get("current_answer") or [],
                evidence=payload.get("evidence") or [],
                gaps=payload.get("gaps") or [],
            )
        return ResearchResult(
            report_markdown=report_markdown,
            claims=claims,
            gaps=[str(item) for item in payload.get("gaps") or []],
            summary=str(payload.get("summary") or ""),
        )

    def _synthesize_markdown(
        self,
        topic: str,
        sources: list[Source],
        previous_report: str,
        previous_claims: list[Claim],
    ) -> ResearchResult:
        try:
            report_markdown = self._complete_text(
                markdown_synthesis_prompt(topic, sources, previous_report, previous_claims),
                attempts=1,
            )
        except LLMError as exc:
            if not exc.retryable:
                raise
            report_markdown = self._complete_text(
                markdown_synthesis_prompt(
                    topic,
                    sources,
                    previous_report,
                    previous_claims,
                    source_limit=8,
                    source_chars=120,
                    topic_chars=1800,
                    word_limit=700,
                    compact=True,
                ),
                max_tokens=int(os.getenv("RESEARCH_COMPACT_TEXT_MAX_TOKENS", "1800")),
                attempts=1,
            )
        return ResearchResult(
            report_markdown=report_markdown.strip(),
            claims=_claims_from_report(report_markdown),
            gaps=_gaps_from_report(report_markdown),
            summary="Generated as Markdown and extracted claims from the report.",
        )

    def _complete_json(self, user_prompt: str) -> dict[str, Any]:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": int(os.getenv("RESEARCH_MAX_TOKENS", "1800")),
            "response_format": {"type": "json_object"},
        }
        try:
            data = self._post("/chat/completions", body)
        except LLMError as exc:
            if "response_format" not in str(exc):
                raise
            body.pop("response_format", None)
            data = self._post("/chat/completions", body)
        content = _extract_chat_content(data)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError as exc:
                    raise LLMError("The LLM response contained malformed JSON.") from exc
            raise LLMError("The LLM response was not valid JSON.")

    def _complete_text(self, user_prompt: str, max_tokens: int | None = None, attempts: int | None = None) -> str:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens
            if max_tokens is not None
            else int(os.getenv("RESEARCH_TEXT_MAX_TOKENS") or os.getenv("RESEARCH_MAX_TOKENS", "2800")),
        }
        data = self._post("/chat/completions", body, attempts=attempts)
        return _extract_chat_content(data)

    def _post(self, path: str, body: dict[str, Any], attempts: int | None = None) -> dict[str, Any]:
        def once() -> dict[str, Any]:
            request = urllib.request.Request(
                f"{self.base_url}{path}",
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
                raise LLMError(f"LLM endpoint returned HTTP {exc.code}: {detail}", retryable=retryable) from exc
            except urllib.error.URLError as exc:
                raise LLMError(
                    f"Could not reach LLM endpoint: {exc}",
                    retryable=_is_retryable_url_error(exc),
                ) from exc
            except _RETRYABLE_TRANSPORT_ERRORS as exc:
                raise LLMError(f"LLM endpoint transport error: {exc}", retryable=True) from exc

        total_attempts = attempts if attempts is not None else self.attempts
        return call_with_retries(once, lambda exc: bool(getattr(exc, "retryable", False)), attempts=total_attempts)


class GeminiLLM(ResearchLLM):
    """Native Gemini generateContent adapter."""

    name = "gemini"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        timeout: int = 90,
        synthesis_mode: str | None = None,
        thinking_budget: int | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL") or os.getenv("RESEARCH_MODEL") or "gemini-2.5-flash"
        self.temperature = temperature
        self.timeout = int(os.getenv("GEMINI_TIMEOUT") or os.getenv("RESEARCH_LLM_TIMEOUT", str(timeout)))
        self.attempts = max(1, int(os.getenv("GEMINI_RETRY_ATTEMPTS") or os.getenv("RESEARCH_LLM_RETRY_ATTEMPTS", "2")))
        self.synthesis_mode = (synthesis_mode or os.getenv("RESEARCH_SYNTHESIS_MODE", "markdown")).strip().lower()
        if self.synthesis_mode not in {"json", "markdown"}:
            raise LLMError("synthesis_mode must be 'json' or 'markdown'.")
        if thinking_budget is None:
            budget = os.getenv("GEMINI_THINKING_BUDGET", "0").strip()
            self.thinking_budget = int(budget) if budget else None
        else:
            self.thinking_budget = thinking_budget
        if not self.api_key:
            raise LLMError("Missing API key. Set GEMINI_API_KEY for the gemini backend.")

    def plan_queries(self, topic: str, previous_report: str, gaps: list[str]) -> list[str]:
        try:
            payload = self._complete_json(query_plan_prompt(topic, previous_report, gaps))
        except (LLMError, json.JSONDecodeError):
            return [_extract_question(topic)]
        queries = payload.get("queries") or []
        return [str(query).strip() for query in queries if str(query).strip()][:5]

    def synthesize(
        self,
        topic: str,
        sources: list[Source],
        previous_report: str,
        previous_claims: list[Claim],
    ) -> ResearchResult:
        if self.synthesis_mode == "markdown":
            return self._synthesize_markdown(topic, sources, previous_report, previous_claims)
        try:
            payload = self._complete_json(synthesis_prompt(topic, sources, previous_report, previous_claims))
        except (LLMError, json.JSONDecodeError):
            return self._synthesize_markdown(topic, sources, previous_report, previous_claims)
        report_markdown = str(payload.get("report_markdown") or "").strip()
        claims = _claims_from_report(report_markdown)
        if not report_markdown:
            report_markdown = _build_report_markdown(
                topic=topic,
                sources=sources,
                current_answer=payload.get("current_answer") or [],
                evidence=payload.get("evidence") or [],
                gaps=payload.get("gaps") or [],
            )
            claims = _claims_from_report(report_markdown)
        return ResearchResult(
            report_markdown=report_markdown,
            claims=claims,
            gaps=[str(item) for item in payload.get("gaps") or _gaps_from_report(report_markdown)],
            summary=str(payload.get("summary") or ""),
        )

    def _synthesize_markdown(
        self,
        topic: str,
        sources: list[Source],
        previous_report: str,
        previous_claims: list[Claim],
    ) -> ResearchResult:
        try:
            report_markdown = self._complete_text(
                markdown_synthesis_prompt(topic, sources, previous_report, previous_claims),
                attempts=1,
            )
        except LLMError as exc:
            if not exc.retryable:
                raise
            report_markdown = self._complete_text(
                markdown_synthesis_prompt(
                    topic,
                    sources,
                    previous_report,
                    previous_claims,
                    source_limit=8,
                    source_chars=120,
                    topic_chars=1800,
                    word_limit=700,
                    compact=True,
                ),
                max_tokens=int(os.getenv("RESEARCH_COMPACT_TEXT_MAX_TOKENS", "1800")),
                attempts=1,
            )
        return ResearchResult(
            report_markdown=report_markdown.strip(),
            claims=_claims_from_report(report_markdown),
            gaps=_gaps_from_report(report_markdown),
            summary="Generated as Markdown and extracted claims from the report.",
        )

    def _complete_json(self, user_prompt: str) -> dict[str, Any]:
        content = self._complete_text(
            user_prompt + "\n\nReturn only valid JSON.",
            max_tokens=int(os.getenv("RESEARCH_MAX_TOKENS", "1800")),
        )
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError as exc:
                    raise LLMError("The Gemini response contained malformed JSON.") from exc
            raise LLMError("The Gemini response was not valid JSON.")

    def _complete_text(self, user_prompt: str, max_tokens: int | None = None, attempts: int | None = None) -> str:
        generation_config: dict[str, Any] = {
            "temperature": self.temperature,
            "maxOutputTokens": max_tokens
            if max_tokens is not None
            else int(os.getenv("RESEARCH_TEXT_MAX_TOKENS") or os.getenv("RESEARCH_MAX_TOKENS", "2800")),
        }
        if self.thinking_budget is not None:
            generation_config["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}
        body = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": generation_config,
        }
        data = self._post_generate_content(body, attempts=attempts)
        return _extract_gemini_content(data)

    def _post_generate_content(self, body: dict[str, Any], attempts: int | None = None) -> dict[str, Any]:
        model_path = urllib.parse.quote(self.model.removeprefix("models/"), safe="")

        def once() -> dict[str, Any]:
            request = urllib.request.Request(
                f"{self.base_url}/models/{model_path}:generateContent",
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "x-goog-api-key": self.api_key,
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
                raise LLMError(f"Gemini endpoint returned HTTP {exc.code}: {detail}", retryable=retryable) from exc
            except urllib.error.URLError as exc:
                raise LLMError(
                    f"Could not reach Gemini endpoint: {exc}",
                    retryable=_is_retryable_url_error(exc),
                ) from exc
            except _RETRYABLE_TRANSPORT_ERRORS as exc:
                raise LLMError(f"Gemini endpoint transport error: {exc}", retryable=True) from exc

        total_attempts = attempts if attempts is not None else self.attempts
        return call_with_retries(once, lambda exc: bool(getattr(exc, "retryable", False)), attempts=total_attempts)


def build_llm(name: str, model: str | None = None, synthesis_mode: str | None = None) -> ResearchLLM:
    if name in {"openai-compatible", "openai", "chat-completions"}:
        return OpenAICompatibleLLM(model=model, synthesis_mode=synthesis_mode)
    if name in {"gemini", "google-gemini"}:
        return GeminiLLM(model=model, synthesis_mode=synthesis_mode)
    raise ValueError(f"Unknown LLM backend: {name}")


def _extract_chat_content(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise LLMError("LLM endpoint returned no choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "".join(parts)
    raise LLMError("LLM endpoint returned an unsupported message content shape.")


def _extract_gemini_content(data: dict[str, Any]) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        feedback = data.get("promptFeedback") or {}
        raise LLMError(f"Gemini endpoint returned no candidates: {feedback}")
    candidate = candidates[0]
    content = candidate.get("content") or {}
    parts = content.get("parts") or []
    text = "".join(str(part.get("text") or "") for part in parts if isinstance(part, dict) and not part.get("thought"))
    if text:
        return text
    finish_reason = candidate.get("finishReason") or "unknown"
    raise LLMError(f"Gemini endpoint returned no text content; finishReason={finish_reason}.")


def _is_retryable_url_error(exc: urllib.error.URLError) -> bool:
    return isinstance(getattr(exc, "reason", None), _RETRYABLE_TRANSPORT_ERRORS)


def _build_report_markdown(
    topic: str,
    sources: list[Source],
    current_answer: list[Any],
    evidence: list[Any],
    gaps: list[Any],
) -> str:
    answer_lines = [str(item).strip() for item in current_answer if str(item).strip()]
    evidence_lines = []
    for item in evidence:
        if isinstance(item, dict):
            claim = str(item.get("claim") or "").strip()
            source_ids = item.get("source_ids") or []
            citations = " ".join(f"[{source_id}]" for source_id in source_ids if f"[{source_id}]" not in claim)
            text = f"{claim} {citations}".strip()
        else:
            text = str(item).strip()
        if text:
            evidence_lines.append(text)
    gap_lines = [str(item).strip() for item in gaps if str(item).strip()]
    source_lines = [
        f"- [{source.id}] {source.title} ({source.url or 'local/manual'})"
        for source in sources
    ]

    return f"""# Research Report

## Question

{_extract_question(topic)}

## Current Answer

{_format_bullets(answer_lines)}

## Evidence

{_format_bullets(evidence_lines)}

## Open Gaps

{_format_bullets(gap_lines)}

## Sources

{chr(10).join(source_lines) or "- No sources recorded."}
"""


def markdown_synthesis_prompt(
    topic: str,
    sources: list[Source],
    previous_report: str,
    previous_claims: list[Claim],
    source_limit: int | None = None,
    source_chars: int | None = None,
    topic_chars: int | None = None,
    word_limit: int = 900,
    compact: bool = False,
) -> str:
    source_blocks = []
    max_sources = source_limit if source_limit is not None else int(os.getenv("RESEARCH_SYNTHESIS_SOURCE_LIMIT", "14"))
    max_source_chars = source_chars if source_chars is not None else int(os.getenv("RESEARCH_SYNTHESIS_SOURCE_CHARS", "180"))
    max_topic_chars = topic_chars if topic_chars is not None else int(os.getenv("RESEARCH_SYNTHESIS_TOPIC_CHARS", "3200"))
    user_request = _limit_text(_extract_user_request(topic), max_topic_chars)
    for source in sources[:max_sources]:
        content = source.content.strip().replace("\x00", "")
        source_blocks.append(
            f"[{source.id}] {source.title}\nURL: {source.url or 'n/a'}\n"
            f"Type: {source.source_type}\nRetrieved: {source.retrieved_at}\n"
            f"Query: {source.query or 'n/a'}\nContent:\n{_limit_text(content, max_source_chars)}"
        )
    source_text = "\n\n---\n\n".join(source_blocks) or "No sources supplied."
    mode_note = "This is a compact retry after the provider timed out. " if compact else ""
    return f"""User request:
{user_request}

Previous report:
{previous_report[:800] if previous_report else "No prior report."}

Previous claim count: {len(previous_claims)}

Source records:
{source_text}

Write Markdown only. Use these exact sections:
# Tech Briefing
## Question
## Current Answer
## Evidence
## Open Gaps
## Sources

Rules:
- {mode_note}Prefer the strongest supported findings over exhaustive coverage.
- Produce the requested deliverable. Do not explain "Tech Briefing Generation"
  as a concept unless the user explicitly asks for that explanation.
- Under ## Question, restate the requested briefing in one sentence.
- Under ## Current Answer, write the actual briefing. If the request names
  sections, categories, Problems & Pain Points, or Investment Opportunities,
  preserve those sections as subheadings inside ## Current Answer.
- If the request asks for today's news or a last-12/24-hour window, only use
  supplied sources that are plausibly from that window. Treat stale or undated
  search results as unusable unless they are live pages that were explicitly
  fetched by URL.
- Include headline, 1-2 sentence summary, and source URL when the user asks for
  them.
- If the user requests an external delivery destination that the runner cannot
  access, record that as an open gap. Do not claim it was delivered.
- Every substantive bullet or sentence must cite supplied source IDs like [S1].
- Do not cite source IDs that were not supplied.
- Do not use emoji or decorative symbols in headings.
- Keep the report under {word_limit} words.
- Prefer synthesis over source-by-source summary.
- Include 6-8 evidence bullets.
"""


def _claims_from_report(markdown: str) -> list[Claim]:
    claims: list[Claim] = []
    for line in markdown.splitlines():
        text = line.strip().lstrip("- ").strip()
        if not text or not text.startswith(tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")):
            continue
        source_ids = re.findall(r"\[(S\d+)\]", text)
        if not source_ids:
            continue
        clean_text = re.sub(r"\s*\[S\d+\]", "", text).strip()
        claims.append(
            Claim(
                id=f"C{len(claims) + 1}",
                text=clean_text,
                source_ids=source_ids,
                confidence="medium",
                notes="Extracted from Markdown fallback report.",
            )
        )
        if len(claims) >= 8:
            break
    return claims


def _gaps_from_report(markdown: str) -> list[str]:
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


def _format_bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) or "- Not enough evidence yet."


def _limit_text(value: str, limit: int) -> str:
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n[truncated]"


def _extract_user_request(topic: str) -> str:
    lines = topic.splitlines()
    request_lines: list[str] = []
    in_question = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower() == "## question":
            in_question = True
            continue
        if in_question and stripped.lower() == "## source policy":
            break
        if in_question:
            request_lines.append(line)
    text = "\n".join(request_lines).strip()
    return text or topic.strip()


def _extract_question(topic: str) -> str:
    lines = topic.splitlines()
    for index, line in enumerate(lines):
        if line.strip().lower() == "## question":
            for candidate in lines[index + 1:]:
                if candidate.strip() and not candidate.strip().startswith("#"):
                    return candidate.strip()
    for line in lines:
        if line.strip() and not line.strip().startswith("#"):
            return line.strip()
    return "Research question not specified."
