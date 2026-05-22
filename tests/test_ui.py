from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core import add_manual_source, init_workspace
from storage import write_json, write_text
from ui import (
    INDEX_HTML,
    ResearchUI,
    list_research_summaries,
    question_from_topic,
    workspace_name_from_question,
    workspace_result_payload,
)


class UITests(unittest.TestCase):
    def test_workspace_name_from_question_is_short_and_stable(self) -> None:
        name = workspace_name_from_question(
            "What are the important software updates for May 2026?",
            "abcdef1234567890",
        )

        self.assertTrue(name.startswith("what-are-the-important-software-updates-for-may"))
        self.assertTrue(name.endswith("-abcdef12"))
        self.assertLessEqual(len(name), 57)

    def test_create_job_can_prepare_without_starting_worker(self) -> None:
        app = ResearchUI()

        job = app.create_job("What changed in developer tools?", start=False)

        self.assertEqual(job.status, "queued")
        self.assertEqual(app.get_job(job.id), job)

    def test_create_job_rejects_empty_question(self) -> None:
        app = ResearchUI()

        with self.assertRaises(ValueError):
            app.create_job("   ", start=False)

    def test_question_from_topic_reads_question_section(self) -> None:
        question = question_from_topic(
            "# Research Topic\n\n## Question\n\nWhat changed?\n\n## Source Policy\n\nRules."
        )

        self.assertEqual(question, "What changed?")

    def test_list_research_summaries_reads_existing_workspaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = init_workspace(root, "past research", "What happened before?")
            add_manual_source(
                workspace,
                title="Local source",
                url="local://source",
                content="Source content.",
            )
            write_json(workspace / "state.json", {"best_score": 91.0, "best_iteration": "iter-1"})

            summaries = list_research_summaries(root)

            self.assertEqual(summaries[0]["name"], "past-research")
            self.assertEqual(summaries[0]["question"], "What happened before?")
            self.assertEqual(summaries[0]["score"], 91.0)
            self.assertEqual(summaries[0]["source_count"], 1)

    def test_list_researches_includes_queued_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = ResearchUI(root=Path(tmp))
            job = app.create_job("What is queued?", start=False)

            researches = app.list_researches()

            self.assertEqual(researches[0]["job_id"], job.id)
            self.assertEqual(researches[0]["status"], "queued")
            self.assertEqual(researches[0]["question"], "What is queued?")

    def test_delete_research_removes_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = ResearchUI(root=root)
            workspace = init_workspace(root, "old research", "What can be removed?")

            result = app.delete_research(workspace.name)

            self.assertEqual(result["deleted"], workspace.name)
            self.assertFalse(workspace.exists())
            self.assertEqual(app.list_researches(), [])

    def test_delete_research_rejects_running_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = ResearchUI(root=root)
            job = app.create_job("What is still running?", start=False)
            job.status = "running"
            workspace = init_workspace(root, job.workspace_name, job.question)

            with self.assertRaises(RuntimeError):
                app.delete_research(workspace.name)

            self.assertTrue(workspace.exists())

    def test_workspace_result_payload_returns_report_quality_and_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = init_workspace(Path(tmp), "ui payload", "What should be shown?")
            add_manual_source(
                workspace,
                title="Local source",
                url="local://source",
                content="Source content for the local UI.",
            )
            write_text(workspace / "report.md", "# Research Report\n\nAnswer [S1].\n")
            write_text(workspace / "eval.md", "# Evaluation\n\nScore: 88.00/100\n")
            write_json(workspace / "state.json", {"best_score": 88.0, "best_iteration": "iter-1"})

            payload = workspace_result_payload(workspace)

            self.assertEqual(payload["workspace_name"], workspace.name)
            self.assertEqual(payload["score"], 88.0)
            self.assertEqual(payload["best_iteration"], "iter-1")
            self.assertIn("Answer [S1]", payload["report"])
            self.assertEqual(payload["sources"][0]["id"], "S1")
            self.assertEqual(payload["sources"][0]["content_length"], 32)

    def test_ui_renderer_links_report_urls_and_source_citations(self) -> None:
        self.assertIn("safeHref", INDEX_HTML)
        self.assertIn("citation-link", INDEX_HTML)
        self.assertIn('href="#source-${sourceId}"', INDEX_HTML)
        self.assertIn("item.id = `source-${source.id}`", INDEX_HTML)

    def test_ui_has_research_delete_action(self) -> None:
        self.assertIn('method: "DELETE"', INDEX_HTML)
        self.assertIn("research-delete", INDEX_HTML)
        self.assertIn("window.confirm", INDEX_HTML)


if __name__ == "__main__":
    unittest.main()
