from __future__ import annotations

import unittest

from llm import LLMError, OpenAICompatibleLLM, markdown_synthesis_prompt
from models import Source


def make_source(index: int, content: str = "content") -> Source:
    return Source(
        id=f"S{index}",
        title=f"Source {index}",
        url=f"https://example.com/{index}",
        content=content,
        retrieved_at="2026-05-22T00:00:00Z",
        source_type="web",
        query="test query",
    )


class TimeoutThenSuccessLLM(OpenAICompatibleLLM):
    def __init__(self) -> None:
        super().__init__(api_key="test-key", model="test-model", synthesis_mode="markdown")
        self.prompts: list[str] = []

    def _complete_text(self, user_prompt: str, max_tokens: int | None = None, attempts: int | None = None) -> str:
        self.prompts.append(user_prompt)
        if len(self.prompts) == 1:
            raise LLMError("synthetic timeout", retryable=True)
        return "# Research Report\n\n## Current Answer\n\nRecovered [S1].\n"


class LLMTests(unittest.TestCase):
    def test_markdown_prompt_caps_source_count_and_content(self) -> None:
        prompt = markdown_synthesis_prompt(
            topic="A" * 80,
            sources=[make_source(index, content="x" * 100) for index in range(1, 6)],
            previous_report="",
            previous_claims=[],
            source_limit=2,
            source_chars=12,
            topic_chars=20,
        )

        self.assertIn("[S1]", prompt)
        self.assertIn("[S2]", prompt)
        self.assertNotIn("[S3]", prompt)
        self.assertIn("[truncated]", prompt)

    def test_markdown_prompt_preserves_full_user_deliverable(self) -> None:
        prompt = markdown_synthesis_prompt(
            topic="""# Research Topic

## Question

Tech Briefing Generation

## Core Task: Generate Tech Briefing

## Problems & Pain Points

## Investment Opportunities

## Source Policy

Internal policy.""",
            sources=[make_source(1)],
            previous_report="",
            previous_claims=[],
        )

        self.assertIn("## Core Task: Generate Tech Briefing", prompt)
        self.assertIn("## Problems & Pain Points", prompt)
        self.assertIn("## Investment Opportunities", prompt)
        self.assertNotIn("Internal policy", prompt)
        self.assertIn("Do not explain \"Tech Briefing Generation\"", prompt)

    def test_markdown_synthesis_retries_with_compact_prompt_after_timeout(self) -> None:
        llm = TimeoutThenSuccessLLM()

        result = llm.synthesize(
            topic="Broad daily research request " * 100,
            sources=[make_source(index, content="source content " * 20) for index in range(1, 12)],
            previous_report="",
            previous_claims=[],
        )

        self.assertIn("Recovered", result.report_markdown)
        self.assertEqual(len(llm.prompts), 2)
        self.assertIn("compact retry", llm.prompts[1])
        self.assertNotIn("[S9]", llm.prompts[1])


if __name__ == "__main__":
    unittest.main()
