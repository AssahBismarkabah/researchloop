# ResearchLoop

`ResearchLoop` is a small prototype for an auditable deep-research loop inspired by
Karpathy's `autoresearch`, but aimed at topic research instead of GPU model
training.

The experiment we want to test:

> Can an agent produce a better, more auditable research report by iterating with
> source collection, claim extraction, verification, and diffs than by producing
> a one-shot report?

This is not a clone of ChatGPT Deep Research, Perplexity, Elicit, or NotebookLM.
Those tools already produce strong reports. This prototype focuses on what they
do not expose as directly: durable run history, source snapshots, claim-level
evidence, evaluator notes, and repeatable reruns.

## Design

Each research workspace is just files:

```text
topic.md              # question, source policy, evaluation goal
sources.jsonl         # source snapshots with stable IDs like S1
claims.jsonl          # kept claim records with source IDs
report.md             # current best report
eval.md               # current evaluator summary
results.tsv           # iteration log
state.json            # current best score and iteration
iterations/           # candidate artifacts for every run
```

The loop is:

```text
topic -> plan queries -> search/ingest sources -> synthesize report
      -> extract claims -> evaluate -> keep/discard -> log
```

The score is intentionally transparent rather than magical. It rewards citation
coverage, cited source diversity, claim volume, report citation density, and
expected report structure. It penalizes unsupported claims and excessive open
gaps. It is a prototype metric, not truth.

## Endpoint Model

The LLM adapter uses OpenAI-compatible chat completions, not OpenAI-specific
Responses APIs.

Set:

```bash
export OPENAI_COMPAT_BASE_URL="https://your-compatible-host/v1"
export OPENAI_COMPAT_API_KEY="..."
export RESEARCH_MODEL="your-model-name"
```

Aliases are also accepted:

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-4o-mini"
```

The request path is:

```text
POST {base_url}/chat/completions
```

with `response_format: {"type": "json_object"}`.

## Optional Search

Manual source ingestion works without a search API. For automated web search,
set:

```bash
export TAVILY_API_KEY="..."
```

and run with `--search tavily`.

## Quick Start

Create a workspace:

```bash
python -m researchloop init ai-research-agents "What are the practical gaps in current AI deep research tools?"
```

Add a manual source:

```bash
python -m researchloop ingest workspaces/ai-research-agents \
  --title "Internal notes" \
  --url "local://notes" \
  --text "Existing deep research tools produce good reports but rarely expose repeatable claim-level run history."
```

Run one iteration with an OpenAI-compatible endpoint:

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

## What Would Make This Worth Continuing?

Continue if the prototype consistently gives us:

- better citation discipline than one-shot research;
- useful `iterations/` artifacts that show what changed;
- source snapshots that make reports reproducible;
- verifier notes that catch real gaps;
- low enough operational cost to rerun topics periodically.

Stop or redesign if it mostly creates longer reports, rubber-stamps weak claims,
or costs more attention than direct use of existing tools.
