# ResearchLoop agent program

This file describes how to run a bounded autonomous research experiment.

## Setup

1. Agree on a workspace name and research question.
2. Create the workspace:

   ```bash
   python -m researchloop init <name> "<question>"
   ```

3. Configure an OpenAI-compatible chat-completions endpoint:

   ```bash
   export OPENAI_COMPAT_BASE_URL="https://your-compatible-host/v1"
   export OPENAI_COMPAT_API_KEY="..."
   export RESEARCH_MODEL="your-model-name"
   ```

4. Optional: configure search.

   ```bash
   export TAVILY_API_KEY="..."
   ```

5. Add any seed sources with `python -m researchloop ingest`.

## Research Loop

Run one iteration:

```bash
python -m researchloop run workspaces/<name> --search tavily --max-results 5
```

If search is disabled, use:

```bash
python -m researchloop run workspaces/<name> --search none
```

Each iteration:

1. Reads `topic.md`, `sources.jsonl`, `claims.jsonl`, and `report.md`.
2. Asks the LLM for targeted search queries.
3. Adds new source snapshots if search is enabled.
4. Asks the LLM to write a candidate report and claim set.
5. Scores the candidate.
6. Keeps it if it improves the current score by `--min-delta`.
7. Logs the result to `results.tsv`.
8. Stores all candidate artifacts under `iterations/`.

## Rules

- Do not claim completeness when the source set is thin.
- Every substantive report claim must cite source IDs like `[S1]`.
- Prefer primary sources and official documentation over summaries.
- Keep source snapshots, even when an iteration is discarded.
- Treat the score as a guide, not as truth.
- Stop after the agreed iteration or budget cap.
