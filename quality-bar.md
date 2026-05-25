# ResearchLoop Quality Bar

This file exists to stop the project from turning into an endless pile of
features. It defines the level I am trying to reach before adding more tools.

## Current Target

The current target is a serious local research runner for source-backed reports.
It is not a full Deep Research replacement, browser agent, academic discovery
engine, truth oracle, hosted product, or publishing system.

At this level, the system is good enough when it can:

- run with `researchloop run <workspace>`;
- start a thin local UI with `researchloop ui`;
- keep run behavior in `run_config.json`;
- keep source-selection rules in `source_policy.json`;
- search the web through Tavily when configured;
- store extracted source snapshots with stable IDs;
- produce a cited report;
- record claims, evaluator notes, gaps, and iteration history;
- preserve failed attempts and collected sources for inspection;
- make it clear why a run passed, failed, or needs human review.

The tools already chosen are enough for this level:

- native Gemini for the current default model backend;
- OpenAI-compatible chat completions for provider portability;
- Tavily Search and Extract for normal web source collection;
- plain files for auditability and repeatable runs;
- a local verifier for citation and evidence hygiene;
- a localhost UI as a small control surface over the same runner.

## Bench Gate

Before adding a new major tool, run at least three real research prompts through
the current system and inspect the workspace artifacts.

For each bench run, inspect:

- `report.md`;
- `sources.jsonl`;
- `claims.jsonl`;
- `eval.md`;
- `results.tsv`;
- `iterations/`.

A run passes the current bar if:

- the answer directly addresses the research question;
- important claims cite source IDs like `[S1]`;
- cited source IDs exist in `sources.jsonl`;
- source snapshots contain enough content to audit the answer;
- weak or missing evidence is visible in `eval.md` or open gaps;
- rerunning the workspace does not require reconstructing a long command.

## Bench Log

| Date | Workspace | Prompt Type | Result | Main Failure Pattern | Decision |
| --- | --- | --- | --- | --- | --- |
| 2026-05-22 | `workspaces/daily-tech-brief-2026-05-22` | daily technology briefing | pass | none blocking | keep current stack |
| TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | TBD | TBD | TBD | TBD | TBD |

## Tool Admission Rule

Do not add a major tool because it sounds useful. Add one only when the bench
log shows a repeated failure pattern that the current stack cannot reasonably
handle.

Examples:

- Add browser/runtime page inspection only if repeated runs fail because sources
  are JS-rendered, login-gated, or unreadable through Search and Extract.
- Add official API connectors only if repeated runs need structured data that
  web pages expose poorly.
- Add model fallback or budget policy only if repeated runs fail because of
  provider reliability, latency, or cost.
- Add scheduling only after recurring reports are already useful when run
  manually.
- Add publishing only after the generated report is worth distributing.
- Add hosted infrastructure only after local runs need to be shared or continue
  after the local machine is closed.

## Not Yet Needed

These are intentionally outside the current level:

- Kernel or another browser runtime;
- Notion storage or publishing;
- a hosted product UI;
- a job queue;
- automatic truth verification;
- broad connector support.

If the bench runs prove one of these is necessary, the decision should be added
to the bench log with the concrete failure that forced it.
