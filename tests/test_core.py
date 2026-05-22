from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core import add_manual_source, init_workspace, run_iteration
from llm import ResearchLLM
from models import Claim, ResearchResult, Source
from run_config import RunConfig, load_run_config_for_workspace, write_run_config
from scoring import evaluate_report
from search import NoSearch, SearchBackend, TavilySearch
from source_policy import SourcePolicy, load_policy_for_workspace
from storage import load_claims, load_sources, read_text


class FixedLLM(ResearchLLM):
    name = "fixed-test-double"

    def plan_queries(self, topic: str, previous_report: str, gaps: list[str]) -> list[str]:
        return ["test query"]

    def synthesize(
        self,
        topic: str,
        sources: list[Source],
        previous_report: str,
        previous_claims: list[Claim],
    ) -> ResearchResult:
        source = sources[0]
        return ResearchResult(
            summary="Created a cited candidate report.",
            report_markdown=f"""# Research Report

## Question

What is being tested?

## Current Answer

The prototype can keep a cited report when the claim references a known source [{source.id}].

## Evidence

| Claim | Sources |
| --- | --- |
| The source supports the candidate report. | [{source.id}] |

## Open Gaps

- Add more independent sources.

## Sources

- [{source.id}] {source.title}
""",
            claims=[
                Claim(
                    id="C99",
                    text="The source supports the candidate report.",
                    source_ids=[source.id],
                    confidence="high",
                )
            ],
            gaps=["Add more independent sources."],
        )


class FailingSynthesisLLM(ResearchLLM):
    name = "failing-synthesis-test-double"

    def plan_queries(self, topic: str, previous_report: str, gaps: list[str]) -> list[str]:
        return ["test query"]

    def synthesize(
        self,
        topic: str,
        sources: list[Source],
        previous_report: str,
        previous_claims: list[Claim],
    ) -> ResearchResult:
        raise RuntimeError("synthetic synthesis failure")


class OneSourceSearch(SearchBackend):
    name = "one-source"

    def search(self, query: str, max_results: int) -> list[Source]:
        return [
            Source(
                id="",
                title="Collected source",
                url="https://example.com/source",
                content="Collected content.",
                retrieved_at="2026-01-01T00:00:00Z",
                source_type="web",
                query=query,
            )
        ]


class CoreTests(unittest.TestCase):
    def test_workspace_lifecycle_keeps_improved_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = init_workspace(root, "agent research", "Can the loop work?")
            add_manual_source(
                workspace,
                title="Seed source",
                url="local://seed",
                content="The seed source supports a grounded candidate report.",
            )

            result = run_iteration(workspace, FixedLLM(), NoSearch())

            self.assertEqual(result["status"], "keep")
            self.assertGreater(result["score"], 0)
            self.assertIn("[S1]", read_text(workspace / "report.md"))
            self.assertEqual(load_claims(workspace)[0].id, "C1")
            self.assertIn("keep", read_text(workspace / "results.tsv"))

    def test_init_writes_reviewable_source_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = init_workspace(
                Path(tmp),
                "policy test",
                "What should search prefer?",
                source_policy=SourcePolicy(
                    time_range="DAY",
                    include_domains=["Example.com", "https://docs.example.com/path"],
                    exclude_domains=["forum.example.com"],
                ),
            )

            policy = json.loads((workspace / "source_policy.json").read_text(encoding="utf-8"))

            self.assertEqual(policy["time_range"], "day")
            self.assertTrue(policy["extract_after_search"])
            self.assertEqual(policy["extract_depth"], "basic")
            self.assertEqual(policy["include_domains"], ["example.com", "docs.example.com"])
            self.assertEqual(policy["exclude_domains"], ["forum.example.com"])
            self.assertIn("source_policy.json", read_text(workspace / "topic.md"))

    def test_init_writes_reviewable_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = init_workspace(
                Path(tmp),
                "run config test",
                "How should the runner behave?",
                run_config=RunConfig(search_backend="none", synthesis_mode="json", max_results=3),
            )

            config = json.loads((workspace / "run_config.json").read_text(encoding="utf-8"))

            self.assertEqual(config["search_backend"], "none")
            self.assertEqual(config["synthesis_mode"], "json")
            self.assertEqual(config["max_results"], 3)
            self.assertIn("run_config.json", read_text(workspace / "topic.md"))

    def test_run_config_resolution_prefers_workspace_file_and_supports_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = init_workspace(Path(tmp), "config resolution", "Which run config is used?")
            write_run_config(
                workspace / "run_config.json",
                RunConfig(search_backend="none", synthesis_mode="json", max_results=2),
            )

            config = load_run_config_for_workspace(workspace)
            overridden = config.with_overrides(max_results=8, search_backend=None)

            self.assertEqual(config.search_backend, "none")
            self.assertEqual(config.synthesis_mode, "json")
            self.assertEqual(overridden.search_backend, "none")
            self.assertEqual(overridden.max_results, 8)

    def test_source_policy_filters_domains(self) -> None:
        policy = SourcePolicy(
            include_domains=["example.com"],
            exclude_domains=["forum.example.com"],
        )

        self.assertTrue(policy.allows_url("https://docs.example.com/reference"))
        self.assertFalse(policy.allows_url("https://forum.example.com/thread"))
        self.assertFalse(policy.allows_url("https://unrelated.test/article"))

    def test_workspace_policy_resolution_prefers_workspace_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = init_workspace(
                Path(tmp),
                "policy resolution",
                "Which policy is used?",
                source_policy=SourcePolicy(include_domains=["workspace.example"]),
            )

            policy = load_policy_for_workspace(workspace)

            self.assertEqual(policy.include_domains, ["workspace.example"])

    def test_tavily_extract_enriches_short_search_content(self) -> None:
        search = TavilySearch(
            api_key="test-key",
            source_policy=SourcePolicy(extract_after_search=True),
        )

        def fake_post(path: str, body: dict[str, object]) -> dict[str, object]:
            if path == "/search":
                return {
                    "results": [
                        {
                            "title": "Short result",
                            "url": "https://example.com/article",
                            "content": "short",
                            "score": 0.9,
                        }
                    ]
                }
            if path == "/extract":
                self.assertEqual(body["urls"], ["https://example.com/article"])
                return {
                    "results": [
                        {
                            "url": "https://example.com/article",
                            "raw_content": "long extracted markdown content",
                        }
                    ],
                    "failed_results": [],
                }
            raise AssertionError(f"unexpected path: {path}")

        search._post_json = fake_post  # type: ignore[method-assign]

        sources = search.search("test query", max_results=1)

        self.assertEqual(sources[0].content, "long extracted markdown content")
        self.assertTrue(sources[0].metadata["extract_attempted"])
        self.assertTrue(sources[0].metadata["extracted"])

    def test_synthesis_failure_preserves_collected_sources_and_logs_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = init_workspace(Path(tmp), "resume test", "Can sources survive failure?")

            with self.assertRaises(RuntimeError):
                run_iteration(workspace, FailingSynthesisLLM(), OneSourceSearch())

            self.assertEqual(len(load_sources(workspace)), 1)
            self.assertIn("error", read_text(workspace / "results.tsv"))
            error_files = list((workspace / "iterations").glob("*/error.json"))
            self.assertEqual(len(error_files), 1)
            self.assertIn("synthetic synthesis failure", read_text(error_files[0]))

    def test_evaluator_penalizes_unsupported_claims(self) -> None:
        sources = [
            Source(
                id="S1",
                title="Known source",
                url="local://known",
                content="Known content.",
                retrieved_at="2026-01-01T00:00:00Z",
            )
        ]
        claims = [
            Claim(id="C1", text="Supported", source_ids=["S1"]),
            Claim(id="C2", text="Unsupported", source_ids=["S404"]),
        ]
        evaluation = evaluate_report(
            "# Research Report\n\n## Current Answer\n\nSupported [S1].\n\n## Evidence\n\nx\n\n## Open Gaps\n\n- gap\n\n## Sources\n\n- [S1]",
            claims,
            sources,
            gaps=["gap"],
        )

        self.assertEqual(evaluation.unsupported_claim_count, 1)
        self.assertLess(evaluation.score, 100)

    def test_evaluator_flags_weak_textual_support(self) -> None:
        sources = [
            Source(
                id="S1",
                title="Known source",
                url="local://known",
                content="This source discusses database migrations and release notes.",
                retrieved_at="2026-01-01T00:00:00Z",
            )
        ]
        claims = [
            Claim(
                id="C1",
                text="The company launched a quantum computing chip for medical imaging.",
                source_ids=["S1"],
            )
        ]

        evaluation = evaluate_report(
            "# Research Report\n\n## Current Answer\n\nClaim [S1].\n\n## Evidence\n\nx\n\n## Open Gaps\n\n- gap\n\n## Sources\n\n- [S1]",
            claims,
            sources,
            gaps=["gap"],
        )

        self.assertEqual(evaluation.weak_claim_count, 1)
        self.assertIn("weak textual support", "\n".join(evaluation.notes))


if __name__ == "__main__":
    unittest.main()
