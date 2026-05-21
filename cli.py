from __future__ import annotations

import argparse
from pathlib import Path

from core import add_manual_source, evaluate_workspace, init_workspace, run_iteration
from llm import LLMError, build_llm
from search import SearchError, build_search_backend
from storage import read_text


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (LLMError, SearchError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="researchloop",
        description="Run an auditable, file-based deep research loop.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    init_parser = subcommands.add_parser("init", help="Create a research workspace.")
    init_parser.add_argument("name", help="Workspace name.")
    init_parser.add_argument("question", help="Research question.")
    init_parser.add_argument("--root", type=Path, default=Path("workspaces"), help="Workspace root directory.")
    init_parser.set_defaults(func=cmd_init)

    ingest_parser = subcommands.add_parser("ingest", help="Add a manual source to a workspace.")
    ingest_parser.add_argument("workspace", type=Path)
    ingest_parser.add_argument("--title", required=True)
    ingest_parser.add_argument("--url", default="")
    group = ingest_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=Path)
    group.add_argument("--text")
    ingest_parser.set_defaults(func=cmd_ingest)

    run_parser = subcommands.add_parser("run", help="Run one or more research iterations.")
    run_parser.add_argument("workspace", type=Path)
    run_parser.add_argument("--iterations", type=int, default=1)
    run_parser.add_argument(
        "--backend",
        default="openai-compatible",
        choices=["openai-compatible", "openai", "chat-completions"],
        help="LLM backend. All choices use OpenAI-compatible chat completions.",
    )
    run_parser.add_argument("--model", default=None, help="Model name for the compatible endpoint.")
    run_parser.add_argument("--search", default="none", choices=["none", "tavily"])
    run_parser.add_argument("--max-results", type=int, default=5)
    run_parser.add_argument("--min-delta", type=float, default=0.1)
    run_parser.set_defaults(func=cmd_run)

    eval_parser = subcommands.add_parser("evaluate", help="Re-score the current kept report.")
    eval_parser.add_argument("workspace", type=Path)
    eval_parser.set_defaults(func=cmd_evaluate)

    show_parser = subcommands.add_parser("show", help="Print workspace status.")
    show_parser.add_argument("workspace", type=Path)
    show_parser.set_defaults(func=cmd_show)

    return parser


def cmd_init(args: argparse.Namespace) -> int:
    workspace = init_workspace(args.root, args.name, args.question)
    print(workspace)
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    content = args.text if args.text is not None else read_text(args.file)
    source = add_manual_source(args.workspace, title=args.title, url=args.url, content=content)
    print(f"added {source.id}: {source.title}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    llm = build_llm(args.backend, model=args.model)
    search_backend = build_search_backend(args.search)
    for _ in range(args.iterations):
        result = run_iteration(
            workspace=args.workspace,
            llm=llm,
            search_backend=search_backend,
            max_results=args.max_results,
            min_delta=args.min_delta,
        )
        print(
            f"{result['iteration']}: {result['status']} "
            f"score={result['score']:.2f} added_sources={result['added_sources']}"
        )
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    evaluation = evaluate_workspace(args.workspace)
    print(f"score={evaluation.score:.2f} unsupported={evaluation.unsupported_claim_count}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    state = read_text(args.workspace / "state.json", "{}").strip()
    print(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
