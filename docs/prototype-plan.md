# Prototype Plan

## Intent

Build a small, professional experiment to test whether the `autoresearch`
control pattern transfers from ML training to topic research.

Karpathy's loop has:

```text
edit train.py -> train for fixed time -> score val_bpb -> keep/discard
```

This prototype uses:

```text
collect sources -> write report -> score evidence quality -> keep/discard
```

## Existing Tool Landscape

Existing tools already cover one-shot or productized research:

- ChatGPT Deep Research and similar tools produce cited reports.
- Perplexity is strong for fast web-grounded research.
- Gemini Deep Research and NotebookLM are strong in Google-centered workflows.
- Elicit, Consensus, Semantic Scholar, and Scite are stronger for academic
  evidence discovery.
- LangChain Open Deep Research, GPT Researcher, and STORM show open-source
  research-agent patterns.
- Tavily, Exa, Firecrawl, and similar APIs provide search and extraction
  infrastructure.

The gap for us is not "generate a report." The gap is a local, auditable,
repeatable research pipeline with explicit source snapshots, claim records,
run history, and a visible evaluator.

## MVP Scope

In scope:

- file-based workspaces;
- OpenAI-compatible chat-completions endpoint;
- manual source ingestion;
- optional Tavily search;
- reviewable source policy in `source_policy.json`;
- candidate iterations stored under `iterations/`;
- keep/discard scoring;
- `results.tsv` run log;
- tests for storage, scoring, and loop behavior.

Out of scope:

- browser UI;
- multi-agent orchestration;
- background daemon;
- PDF parsing;
- vector database;
- provider-specific APIs;
- automated truth guarantees.

## Success Criteria

The experiment is promising if:

- reports improve across iterations;
- every major claim can be traced to source IDs;
- discarded iterations are still inspectable;
- source snapshots make reruns auditable;
- the evaluator catches weak citation coverage and missing source support.

## Known Caveats

- The score is a proxy. It measures evidence discipline, not truth.
- The LLM can still write unsupported prose; the evaluator only catches
  detectable citation problems.
- Search quality depends on the configured search backend.
- Domain constraints are policy choices, so they belong in `source_policy.json`
  and should be reviewed per research topic.
- Human review remains mandatory for legal, medical, financial, policy, or
  other high-stakes topics.
