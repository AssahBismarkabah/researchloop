from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core import add_manual_source, init_workspace, run_iteration
from llm import ResearchLLM
from models import Claim, ResearchResult, Source
from scoring import evaluate_report
from search import NoSearch
from source_policy import SourcePolicy, load_policy_for_workspace
from storage import load_claims, read_text


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
                    include_domains=["Example.com", "https://docs.example.com/path"],
                    exclude_domains=["forum.example.com"],
                ),
            )

            policy = json.loads((workspace / "source_policy.json").read_text(encoding="utf-8"))

            self.assertEqual(policy["include_domains"], ["example.com", "docs.example.com"])
            self.assertEqual(policy["exclude_domains"], ["forum.example.com"])
            self.assertIn("source_policy.json", read_text(workspace / "topic.md"))

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


if __name__ == "__main__":
    unittest.main()
