from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from models import Claim, ResearchResult, Source
from prompts import SYSTEM_PROMPT, query_plan_prompt, synthesis_prompt


class LLMError(RuntimeError):
    pass


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
        timeout: int = 120,
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
        self.timeout = timeout
        if not self.api_key:
            raise LLMError(
                "Missing API key. Set OPENAI_COMPAT_API_KEY or OPENAI_API_KEY for the "
                "openai-compatible backend."
            )

    def plan_queries(self, topic: str, previous_report: str, gaps: list[str]) -> list[str]:
        payload = self._complete_json(query_plan_prompt(topic, previous_report, gaps))
        queries = payload.get("queries") or []
        return [str(query).strip() for query in queries if str(query).strip()][:5]

    def synthesize(
        self,
        topic: str,
        sources: list[Source],
        previous_report: str,
        previous_claims: list[Claim],
    ) -> ResearchResult:
        payload = self._complete_json(synthesis_prompt(topic, sources, previous_report, previous_claims))
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
        return ResearchResult(
            report_markdown=str(payload.get("report_markdown") or "").strip(),
            claims=claims,
            gaps=[str(item) for item in payload.get("gaps") or []],
            summary=str(payload.get("summary") or ""),
        )

    def _complete_json(self, user_prompt: str) -> dict[str, Any]:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
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
                return json.loads(match.group(0))
            raise LLMError("The LLM response was not valid JSON.")

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
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
            raise LLMError(f"LLM endpoint returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"Could not reach LLM endpoint: {exc}") from exc


def build_llm(name: str, model: str | None = None) -> ResearchLLM:
    if name in {"openai-compatible", "openai", "chat-completions"}:
        return OpenAICompatibleLLM(model=model)
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
