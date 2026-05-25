from __future__ import annotations

import argparse
import os
from pathlib import Path

from core import add_manual_source, evaluate_workspace, init_workspace, run_iteration
from llm import LLMError, build_llm
from run_config import load_default_run_config, load_run_config_for_workspace
from search import SearchError, build_search_backend
from source_policy import load_default_policy, load_policy_for_workspace, policy_for_question
from storage import read_text


def main(argv: list[str] | None = None) -> int:
    load_dotenv(Path(".env"))
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
    init_parser.add_argument(
        "--source-policy",
        type=Path,
        default=None,
        help="Optional source_policy.json to copy into the workspace.",
    )
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
    run_parser.add_argument("--iterations", type=int, default=None, help=argparse.SUPPRESS)
    run_parser.add_argument(
        "--backend",
        default=None,
        choices=["openai-compatible", "openai", "chat-completions", "gemini", "google-gemini"],
        help=argparse.SUPPRESS,
    )
    run_parser.add_argument("--model", default=None, help=argparse.SUPPRESS)
    run_parser.add_argument(
        "--synthesis-mode",
        choices=["json", "markdown"],
        default=None,
        help=argparse.SUPPRESS,
    )
    run_parser.add_argument("--search", default=None, choices=["none", "tavily"], help=argparse.SUPPRESS)
    run_parser.add_argument(
        "--source-policy",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    run_parser.add_argument("--max-results", type=int, default=None, help=argparse.SUPPRESS)
    run_parser.add_argument("--min-delta", type=float, default=None, help=argparse.SUPPRESS)
    run_parser.set_defaults(func=cmd_run)

    eval_parser = subcommands.add_parser("evaluate", help="Re-score the current kept report.")
    eval_parser.add_argument("workspace", type=Path)
    eval_parser.set_defaults(func=cmd_evaluate)

    show_parser = subcommands.add_parser("show", help="Print workspace status.")
    show_parser.add_argument("workspace", type=Path)
    show_parser.set_defaults(func=cmd_show)

    ui_parser = subcommands.add_parser("ui", help="Start the local browser UI.")
    ui_parser.add_argument("--host", default=os.getenv("RESEARCH_UI_HOST", "127.0.0.1"), help=argparse.SUPPRESS)
    ui_parser.add_argument("--port", type=int, default=int(os.getenv("RESEARCH_UI_PORT", "8787")), help=argparse.SUPPRESS)
    ui_parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.getenv("RESEARCH_WORKSPACE_ROOT", "workspaces")),
        help=argparse.SUPPRESS,
    )
    ui_parser.set_defaults(func=cmd_ui)

    return parser


def cmd_init(args: argparse.Namespace) -> int:
    policy = load_default_policy(args.source_policy)
    run_config = load_default_run_config()
    workspace = init_workspace(args.root, args.name, args.question, source_policy=policy, run_config=run_config)
    print(workspace)
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    content = args.text if args.text is not None else read_text(args.file)
    source = add_manual_source(args.workspace, title=args.title, url=args.url, content=content)
    print(f"added {source.id}: {source.title}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = load_run_config_for_workspace(args.workspace).with_overrides(
        backend=args.backend,
        model=args.model,
        synthesis_mode=args.synthesis_mode,
        search_backend=args.search,
        max_results=args.max_results,
        min_delta=args.min_delta,
        iterations=args.iterations,
    )
    llm = build_llm(config.backend, model=config.model, synthesis_mode=config.synthesis_mode)
    source_policy = policy_for_question(
        load_policy_for_workspace(args.workspace, args.source_policy),
        read_text(args.workspace / "topic.md"),
    )
    search_backend = build_search_backend(config.search_backend, source_policy=source_policy)
    for _ in range(config.iterations):
        result = run_iteration(
            workspace=args.workspace,
            llm=llm,
            search_backend=search_backend,
            max_results=config.max_results,
            min_delta=config.min_delta,
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


def cmd_ui(args: argparse.Namespace) -> int:
    from ui import run_ui

    run_ui(host=args.host, port=args.port, root=args.root)
    return 0


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


if __name__ == "__main__":
    raise SystemExit(main())
