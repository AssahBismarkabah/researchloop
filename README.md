# ResearchLoop

`ResearchLoop` is a local, file-based research runner. The goal is not to make
another chat box that answers a question once. The goal is to make research run
like a small, inspectable system.

If you need a polished one-off report, use ChatGPT Deep Research, Perplexity,
NotebookLM, Elicit, or whatever tool is best for that job. That is not the thing
I am trying to replace here.

What I want here is different: I want to give an agent a research question, a
run config, a source policy, a model endpoint, and a workspace, then let it work
through the topic while leaving the evidence behind. It should search, snapshot
sources, write a candidate report, check whether the claims are cited, keep the
report only if it improves the previous one, and leave me enough files to
understand what happened.

If the answer is bad, I do not want to guess why. I want to see the sources, the
claims, the prompt, the score, the gaps, and the discarded attempts.

The core loop is deliberately simple:

```text
plan queries -> collect source snapshots -> write candidate report
             -> verify claims and citations -> keep/discard -> repeat
```

The point is to own the research operation: source policy, source snapshots,
claim records, evaluator notes, iteration history, repeatable reruns, and
publishing hooks for recurring workflows.

## What This Is For

This starts to matter when the same kind of research has to happen again and
again, and the process matters as much as the final answer:

- daily tech briefings;
- security and vulnerability watchlists;
- market or industry monitoring;
- research over internal notes plus web sources;
- reports that need an audit trail, not just a final paragraph.

If the output needs to become a daily briefing, a watchlist, a report archive, a
Notion page, or an internal workflow, then I want the research process to be
programmable and inspectable instead of hidden inside a chat session.

## How It Works

The repo is intentionally small. A research topic becomes a directory of plain
files. The human programs the topic, run config, and source policy. The agent
produces source snapshots, candidate reports, claim records, evaluator notes,
and an iteration log.

The important files are the interface:

```text
quality-bar.md        # target level, bench gate, and tool admission rules
program.md            # operating instructions for bounded research runs
run_config.json       # default run behavior copied into each workspace
source_policy.json    # source-selection rules copied into each workspace
workspaces/<name>/topic.md          # the research question and constraints
workspaces/<name>/run_config.json   # how this workspace runs
workspaces/<name>/source_policy.json # source rules for this workspace
workspaces/<name>/sources.jsonl     # source snapshots with stable IDs like S1
workspaces/<name>/claims.jsonl      # kept claim records with source IDs
workspaces/<name>/report.md         # current best report
workspaces/<name>/eval.md           # verifier summary for the current best report
workspaces/<name>/results.tsv       # iteration log
workspaces/<name>/state.json        # current best score and iteration
workspaces/<name>/iterations/       # candidate artifacts for every run
```

By design, `report.md` is not overwritten just because the model wrote
something new. A candidate has to beat the current score. If it loses, the
candidate is discarded as the current report but preserved under `iterations/`
so the failure can still be inspected.

The metric is intentionally practical. It is not a truth oracle. It rewards
cited claims, source coverage, expected structure, and visible open gaps. It
penalizes unsupported claims and thin evidence. The metric exists so the loop has
a repeatable signal, not so humans can stop reviewing the result.

## Project Structure

```text
researchloop.py       # module entrypoint
cli.py                # command-line interface
core.py               # workspace lifecycle and keep/discard loop
llm.py                # OpenAI-compatible chat-completions adapter
search.py             # search backend adapter
run_config.py         # run configuration loading and validation
source_policy.py      # source policy loading and URL filtering
scoring.py            # transparent verifier score
prompts.py            # planning and synthesis prompts
models.py             # source, claim, report, evaluation records
storage.py            # plain-file persistence helpers
```

`program.md` is the human-facing operating document: it tells an agent how to
run bounded research work. `source_policy.json` is where source rules live. The
Python files are the runner; the workspace files are the research record.

## Verification

The verifier is intentionally transparent. It does not prove truth. It checks
whether the report is operationally usable:

- every substantive claim should cite source IDs like `[S1]`;
- claim records must point to known sources;
- cited sources should contain enough meaningful claim terms to count as
  textually supportive;
- the report should use multiple cited sources where possible;
- the report should include the expected sections;
- open gaps should be recorded instead of hidden;
- unsupported claims and excessive gaps lower the score.

This gives a repeatable signal for citation discipline and auditability. Human
review is still required for legal, medical, financial, policy, security, or
other high-stakes research.

## Endpoint Model

The LLM adapter uses OpenAI-compatible chat completions, not OpenAI-specific
Responses APIs.

Set these in `.env` or your shell:

```bash
OPENAI_COMPAT_BASE_URL="https://your-compatible-host/v1"
OPENAI_COMPAT_API_KEY="..."
RESEARCH_MODEL="your-model-name"
```

Aliases are also accepted:

```bash
OPENAI_BASE_URL="https://api.openai.com/v1"
OPENAI_API_KEY="..."
OPENAI_MODEL="gpt-4o-mini"
```

The request path is:

```text
POST {base_url}/chat/completions
```

JSON synthesis uses `response_format: {"type": "json_object"}` when the
endpoint supports it. The default run config uses Markdown synthesis because it
works across more OpenAI-compatible endpoints.

## Search Policy

Manual source ingestion works without a search API. Automated web search uses
Tavily when `TAVILY_API_KEY` is set and the workspace `run_config.json` says
`"search_backend": "tavily"`.

Source-selection rules belong in `source_policy.json`, and `researchloop init`
copies that policy into every workspace so runs remain auditable.

```json
{
  "search_depth": "advanced",
  "time_range": null,
  "extract_after_search": true,
  "extract_depth": "basic",
  "extract_format": "markdown",
  "include_domains": [],
  "exclude_domains": [
    "facebook.com",
    "instagram.com",
    "medium.com",
    "quora.com",
    "reddit.com",
    "youtube.com"
  ]
}
```

Use `include_domains` when a topic should be constrained to known primary
sources. Use `exclude_domains` to remove low-signal domains. Use
`"time_range": "day"` for current-day research. By default, Tavily search
results are enriched through Tavily Extract so the stored source snapshots have
cleaner page content than search snippets alone.

## Run Research

After installing the CLI and configuring an OpenAI-compatible endpoint, start a
research topic with the question you want answered:

```bash
python -m researchloop init software-news \
  "What are the most important software industry updates this month?"
python -m researchloop run workspaces/software-news
```

For a local browser UI, run:

```bash
python -m researchloop ui
```

Then open `http://127.0.0.1:8787`. The UI creates normal workspaces under
`workspaces/`; it does not use a database or hide artifacts behind the browser.
Use `New` to start a run and `Researches` to reopen previous local workspaces.

`init` copies `run_config.json` and `source_policy.json` into the workspace.
That is intentional. I want the behavior of a research run to live with the
research record, not in a long command that disappears from history.

The checked-in default is source-backed web research:

```json
{
  "backend": "openai-compatible",
  "search_backend": "tavily",
  "synthesis_mode": "markdown",
  "max_results": 5,
  "iterations": 1,
  "min_delta": 0.1
}
```

The kept answer is written to `workspaces/software-news/report.md`. The same
workspace also keeps `sources.jsonl`, `eval.md`, `results.tsv`, and every
candidate iteration for audit.

If you do not want web search, ingest trusted material first, set
`"search_backend": "none"` in the workspace `run_config.json`, and run the
workspace normally:

```bash
python -m researchloop ingest workspaces/software-news \
  --title "Internal notes" \
  --text "Your source text here."
python -m researchloop run workspaces/software-news
```

## Design Choices

- **Plain files over hidden state.** Research artifacts should be readable,
  diffable, commit-friendly, and easy to move.
- **OpenAI-compatible endpoint.** The runner should work with any compatible
  `/chat/completions` provider, not a single vendor API.
- **Workspace config over command flags.** The normal path is
  `researchloop run <workspace>`. The run settings are plain files beside the
  research artifacts.
- **Local UI as a thin shell.** `researchloop ui` starts a small localhost app
  over the same runner. The browser is a control surface, not a second product
  path.
- **Source policy is code-like config.** Search rules belong in
  `source_policy.json`, not buried inside prompts or environment variables.
- **Keep/discard is the control loop.** The current report changes only when a
  candidate improves the score; bad runs remain inspectable.
- **The verifier is humble.** It checks evidence hygiene. It does not certify
  truth, investment advice, medical advice, legal advice, or anything else that
  needs human judgment.

## Current Limits

- Source quality is better with extraction, but social feeds and front pages can
  still return truncated or noisy records without a browser or official API.
- The verifier checks citation discipline, structure, and lightweight textual
  support, not factual truth.
- Transient provider failures are retried, but there is no budget policy,
  model-fallback policy, or job queue yet.
- Workflows are still prompt/file driven; there is no dedicated product UI.

## Notable Links

- [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch)
