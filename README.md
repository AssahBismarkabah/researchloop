# ResearchLoop

`ResearchLoop` is a local, file-based research runner for auditable deep
research. It runs a repeatable improve-or-discard loop where source-backed
reports are treated as versioned research artifacts.

The core idea is simple:

```text
plan queries -> collect source snapshots -> write candidate report
             -> verify claims and citations -> keep/discard -> repeat
```

This is not a clone of ChatGPT Deep Research, Perplexity, Elicit, NotebookLM, or
STORM. Those tools are already strong at producing reports. ResearchLoop focuses
on the part that is usually hidden: durable run history, source snapshots,
claim-level evidence, evaluator notes, and repeatable reruns.

## How It Works

Each research topic gets a workspace. A run reads the topic, source policy,
existing sources, previous claims, and current best report. The model plans
search queries, optional Tavily search adds source snapshots, the model writes a
candidate report with structured claims, and the verifier scores the candidate.

If the candidate improves the current score by the configured delta, it becomes
the kept report. Otherwise it is discarded, but its full artifacts remain under
`iterations/` for inspection.

The important files are:

```text
program.md            # operating instructions for bounded research runs
source_policy.json    # default source-selection policy copied into workspaces
topic.md              # workspace question, source policy note, evaluation goal
sources.jsonl         # source snapshots with stable IDs like S1
claims.jsonl          # kept claim records with source IDs
report.md             # current best report
eval.md               # verifier summary for the current best report
results.tsv           # iteration log
state.json            # current best score and iteration
iterations/           # candidate artifacts for every run
```

`program.md` is the human-facing operating document: it tells an agent how to
run bounded research work. The source and report artifacts stay as normal files
so humans can review, edit, diff, and rerun the process without trusting hidden
state.

## Verification

The verifier is intentionally transparent. It does not prove truth. It checks
whether the report is operationally usable:

- every substantive claim should cite source IDs like `[S1]`;
- claim records must point to known sources;
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

with `response_format: {"type": "json_object"}`.

## Search Policy

Manual source ingestion works without a search API. Automated web search uses
Tavily when `TAVILY_API_KEY` is set and `--search tavily` is passed.

Credentials belong in `.env`. Source-selection rules belong in
`source_policy.json`, and `researchloop init` copies that policy into every
workspace so runs remain auditable.

```json
{
  "search_depth": "advanced",
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
sources. Use `exclude_domains` to remove low-signal domains. A run can override
the workspace policy explicitly:

```bash
python -m researchloop run workspaces/ai-research-agents \
  --search tavily \
  --source-policy source_policy.json
```

## Quick Start

Create and install the local environment:

```bash
uv venv .venv
uv pip install -e .
```

Create a workspace:

```bash
python -m researchloop init ai-research-agents \
  "What are the practical gaps in current AI deep research tools?"
```

Add a manual source:

```bash
python -m researchloop ingest workspaces/ai-research-agents \
  --title "Internal notes" \
  --url "local://notes" \
  --text "Existing deep research tools produce good reports but rarely expose repeatable claim-level run history."
```

Run one iteration with only manual sources:

```bash
python -m researchloop run workspaces/ai-research-agents \
  --backend openai-compatible \
  --search none \
  --model "$RESEARCH_MODEL"
```

Run with Tavily source discovery:

```bash
python -m researchloop run workspaces/ai-research-agents \
  --search tavily \
  --max-results 5
```

Re-score the kept report:

```bash
python -m researchloop evaluate workspaces/ai-research-agents
```

## Design Choices

- **File-based state.** Workspaces are plain files, so every run can be
  inspected, committed, archived, or diffed.
- **OpenAI-compatible endpoint.** The model backend is not tied to OpenAI.
- **Explicit source policy.** Search rules are reviewable config, not hidden
  environment defaults.
- **Keep/discard loop.** The runner keeps only candidate reports that improve
  the current score, while preserving discarded iteration artifacts.
- **Transparent verifier.** The score is a visible engineering signal for
  evidence quality, not a claim that the answer is true.

## Status

ResearchLoop has moved past the initial sketch into an early working tool.
The core loop, OpenAI-compatible model calls, manual ingestion, Tavily search,
source policy, run artifacts, scoring, and CLI tests are in place.

It is still not a production research platform. The next maturity work is better
source extraction, richer verifier checks, budget controls, provider retries,
and stronger topic-specific source policy templates.

## Notable Links

- [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch)
