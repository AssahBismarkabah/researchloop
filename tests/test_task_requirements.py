from __future__ import annotations

import unittest

from task_requirements import extract_requirements, missing_requirements


def _topic_for(question: str) -> str:
    return f"""# Research Topic

## Question

{question}

## Source Policy

Operational source-selection rules are stored separately.
"""


class TaskRequirementTests(unittest.TestCase):
    def test_extracts_requested_markdown_sections_without_operational_headings(self) -> None:
        topic = _topic_for(
            """Tech Briefing Generation

## Core Task: Generate Tech Briefing

## Execution Instructions

## Date Filter

## Problems & Pain Points

## Investment Opportunities

## Delivery
"""
        )

        labels = [requirement.label for requirement in extract_requirements(topic)]

        self.assertIn("Tech Briefing", labels)
        self.assertIn("Problems & Pain Points", labels)
        self.assertIn("Investment Opportunities", labels)
        self.assertNotIn("Execution Instructions", labels)
        self.assertNotIn("Date Filter", labels)
        self.assertNotIn("Delivery", labels)

    def test_extracts_topic_categories_from_topic_bullets(self) -> None:
        topic = _topic_for(
            """## Topics (Priority Order):

* **PRIMARY FOCUS:** Developer tools, frameworks, libraries, SDKs.
* Open source highlights (trending repositories, new releases, innovative projects).
* Software security (vulnerabilities, patches, threat intelligence).
"""
        )

        labels = [requirement.label for requirement in extract_requirements(topic)]

        self.assertIn("Developer tools, frameworks, libraries, SDKs", labels)
        self.assertIn("Open source highlights", labels)
        self.assertIn("Software security", labels)

    def test_missing_requirements_reports_absent_requested_deliverables(self) -> None:
        topic = _topic_for(
            "Generate a tech briefing with Problems & Pain Points and Investment Opportunities."
        )
        report = """# Research Report

## Current Answer

This is a generic report.
"""

        self.assertEqual(
            missing_requirements(topic, report),
            ["Problems & Pain Points", "Investment Opportunities"],
        )

    def test_simple_questions_do_not_create_requirements(self) -> None:
        topic = _topic_for("Can the loop work?")

        self.assertEqual(extract_requirements(topic), [])


if __name__ == "__main__":
    unittest.main()
