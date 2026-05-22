from __future__ import annotations

import json
import os
import shutil
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from core import init_workspace, run_iteration
from llm import build_llm
from run_config import load_default_run_config, load_run_config_for_workspace
from search import build_search_backend
from source_policy import load_default_policy, load_policy_for_workspace
from storage import load_sources, read_json, read_text, slugify, utc_now


DEFAULT_UI_HOST = "127.0.0.1"
DEFAULT_UI_PORT = 8787


@dataclass
class ResearchJob:
    id: str
    question: str
    workspace_name: str
    status: str = "queued"
    stage: str = "Queued"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    workspace: str = ""
    result: dict[str, Any] | None = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "workspace_name": self.workspace_name,
            "status": self.status,
            "stage": self.stage,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "workspace": self.workspace,
            "result": self.result,
            "error": self.error,
        }


class ResearchUI:
    def __init__(self, root: Path = Path("workspaces")) -> None:
        self.root = root
        self.jobs: dict[str, ResearchJob] = {}
        self._lock = threading.Lock()

    def create_job(self, question: str, start: bool = True) -> ResearchJob:
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Research question is required.")
        job_id = uuid.uuid4().hex
        job = ResearchJob(
            id=job_id,
            question=clean_question,
            workspace_name=workspace_name_from_question(clean_question, job_id),
        )
        with self._lock:
            self.jobs[job.id] = job
        if start:
            thread = threading.Thread(target=self._run_job, args=(job.id,), daemon=True)
            thread.start()
        return job

    def get_job(self, job_id: str) -> ResearchJob | None:
        with self._lock:
            return self.jobs.get(job_id)

    def list_researches(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self.jobs.values())
        return list_research_summaries(self.root, jobs)

    def get_research(self, workspace_name: str) -> dict[str, Any]:
        workspace = workspace_path_from_name(self.root, workspace_name)
        return workspace_result_payload(workspace)

    def delete_research(self, workspace_name: str) -> dict[str, str]:
        workspace = workspace_path_from_name(self.root, workspace_name)
        with self._lock:
            active_job = next(
                (
                    job
                    for job in self.jobs.values()
                    if job.workspace_name == workspace.name and job.status in {"queued", "running"}
                ),
                None,
            )
            if active_job is not None:
                raise RuntimeError("Research is still running.")
        shutil.rmtree(workspace)
        with self._lock:
            for job_id, job in list(self.jobs.items()):
                if job.workspace_name == workspace.name:
                    del self.jobs[job_id]
        return {"deleted": workspace.name}

    def _set_job(self, job_id: str, **updates: Any) -> ResearchJob:
        with self._lock:
            job = self.jobs[job_id]
            for key, value in updates.items():
                setattr(job, key, value)
            job.updated_at = utc_now()
            return job

    def _run_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if job is None:
            return
        try:
            self._set_job(job_id, status="running", stage="Creating workspace")
            workspace = init_workspace(
                self.root,
                job.workspace_name,
                job.question,
                source_policy=load_default_policy(),
                run_config=load_default_run_config(),
            )
            self._set_job(job_id, workspace=str(workspace), stage="Researching")

            config = load_run_config_for_workspace(workspace)
            llm = build_llm(config.backend, model=config.model, synthesis_mode=config.synthesis_mode)
            source_policy = load_policy_for_workspace(workspace)
            search_backend = build_search_backend(config.search_backend, source_policy=source_policy)

            latest_result: dict[str, object] | None = None
            for index in range(config.iterations):
                self._set_job(job_id, stage=f"Running iteration {index + 1} of {config.iterations}")
                latest_result = run_iteration(
                    workspace=workspace,
                    llm=llm,
                    search_backend=search_backend,
                    max_results=config.max_results,
                    min_delta=config.min_delta,
                )

            self._set_job(
                job_id,
                status="done",
                stage="Done",
                result=workspace_result_payload(workspace, latest_result),
            )
        except Exception as exc:
            self._set_job(
                job_id,
                status="error",
                stage="Error",
                error=str(exc),
                result={"traceback": traceback.format_exc()},
            )


def workspace_name_from_question(question: str, job_id: str) -> str:
    stem = slugify(question)[:48].strip("-") or "research"
    return f"{stem}-{job_id[:8]}"


def list_research_summaries(root: Path, jobs: list[ResearchJob] | None = None) -> list[dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    if root.exists():
        for workspace in root.iterdir():
            if workspace.is_dir() and (workspace / "topic.md").exists():
                summaries[workspace.name] = workspace_summary(workspace)

    for job in jobs or []:
        summary = summaries.get(job.workspace_name) or {
            "name": job.workspace_name,
            "workspace": job.workspace or str(root / job.workspace_name),
            "question": job.question,
            "score": 0.0,
            "source_count": 0,
            "best_iteration": None,
        }
        summary.update(
            {
                "job_id": job.id,
                "status": job.status,
                "stage": job.stage,
                "updated_at": job.updated_at,
            }
        )
        summaries[job.workspace_name] = summary

    return sorted(summaries.values(), key=lambda item: str(item.get("updated_at") or ""), reverse=True)


def workspace_summary(workspace: Path) -> dict[str, Any]:
    state = read_json(workspace / "state.json", {"best_score": 0.0, "best_iteration": None})
    sources = load_sources(workspace)
    last_result = _last_result_row(workspace)
    status = last_result.get("status") or ("done" if state.get("best_iteration") else "ready")
    return {
        "name": workspace.name,
        "workspace": str(workspace),
        "question": question_from_topic(read_text(workspace / "topic.md")),
        "status": status,
        "stage": status.title(),
        "score": float(state.get("best_score") or 0.0),
        "source_count": len(sources),
        "best_iteration": state.get("best_iteration"),
        "updated_at": str(state.get("updated_at") or _mtime_utc(workspace)),
    }


def workspace_path_from_name(root: Path, workspace_name: str) -> Path:
    clean_name = workspace_name.strip()
    if clean_name != slugify(clean_name):
        raise ValueError("Invalid workspace name.")
    workspace = root / clean_name
    if not (workspace / "topic.md").exists():
        raise FileNotFoundError("Research workspace not found.")
    return workspace


def question_from_topic(topic_markdown: str) -> str:
    lines: list[str] = []
    in_question = False
    for line in topic_markdown.splitlines():
        stripped = line.strip()
        if stripped.lower() == "## question":
            in_question = True
            continue
        if in_question and stripped.startswith("## "):
            break
        if in_question and stripped:
            lines.append(stripped)
    return " ".join(lines).strip() or "Untitled research"


def workspace_result_payload(workspace: Path, latest_result: dict[str, object] | None = None) -> dict[str, Any]:
    state = read_json(workspace / "state.json", {"best_score": 0.0, "best_iteration": None})
    sources = load_sources(workspace)
    report = read_text(workspace / "report.md")
    eval_text = read_text(workspace / "eval.md")
    return {
        "workspace_name": workspace.name,
        "workspace": str(workspace),
        "report": report,
        "score": float(state.get("best_score") or 0.0),
        "best_iteration": state.get("best_iteration"),
        "latest_iteration": latest_result or {},
        "eval": eval_text,
        "sources": [
            {
                "id": source.id,
                "title": source.title,
                "url": source.url,
                "source_type": source.source_type,
                "query": source.query,
                "content_length": len(source.content),
                "extracted": bool(source.metadata.get("extracted")),
            }
            for source in sources
        ],
    }


def _last_result_row(workspace: Path) -> dict[str, str]:
    rows = [line for line in read_text(workspace / "results.tsv").splitlines() if line.strip()]
    if len(rows) < 2:
        return {}
    headers = rows[0].split("\t")
    values = rows[-1].split("\t")
    return dict(zip(headers, values))


def _mtime_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_ui(host: str | None = None, port: int | None = None, root: Path | None = None) -> None:
    app = ResearchUI(root=root or Path(os.getenv("RESEARCH_WORKSPACE_ROOT", "workspaces")))
    bind_host = host or os.getenv("RESEARCH_UI_HOST", DEFAULT_UI_HOST)
    bind_port = port or int(os.getenv("RESEARCH_UI_PORT", str(DEFAULT_UI_PORT)))
    handler = make_handler(app)
    try:
        server = ThreadingHTTPServer((bind_host, bind_port), handler)
    except OSError as exc:
        raise ValueError(f"Could not start UI on {bind_host}:{bind_port}: {exc}") from exc
    url = f"http://{server.server_address[0]}:{server.server_address[1]}"
    print(f"ResearchLoop UI running at {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nResearchLoop UI stopped.")
    finally:
        server.server_close()


def make_handler(app: ResearchUI) -> type[BaseHTTPRequestHandler]:
    class ResearchRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(INDEX_HTML)
                return
            if parsed.path == "/api/health":
                self._send_json({"status": "ok"})
                return
            if parsed.path == "/api/researches":
                self._send_json({"researches": app.list_researches()})
                return
            if parsed.path.startswith("/api/researches/"):
                workspace_name = unquote(parsed.path.removeprefix("/api/researches/").strip("/"))
                try:
                    self._send_json(app.get_research(workspace_name))
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, status=400)
                except FileNotFoundError as exc:
                    self._send_json({"error": str(exc)}, status=404)
                return
            if parsed.path.startswith("/api/jobs/"):
                job_id = parsed.path.removeprefix("/api/jobs/").strip("/")
                job = app.get_job(job_id)
                if job is None:
                    self._send_json({"error": "Job not found."}, status=404)
                    return
                self._send_json(job.to_dict())
                return
            self._send_json({"error": "Not found."}, status=404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/research":
                self._send_json({"error": "Not found."}, status=404)
                return
            try:
                payload = self._read_json()
                job = app.create_job(str(payload.get("question") or ""))
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(job.to_dict(), status=202)

        def do_DELETE(self) -> None:
            parsed = urlparse(self.path)
            if not parsed.path.startswith("/api/researches/"):
                self._send_json({"error": "Not found."}, status=404)
                return
            workspace_name = unquote(parsed.path.removeprefix("/api/researches/").strip("/"))
            try:
                self._send_json(app.delete_research(workspace_name))
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
            except FileNotFoundError as exc:
                self._send_json({"error": str(exc)}, status=404)
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, status=409)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length).decode("utf-8")
            if not raw.strip():
                return {}
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError("Request body must be valid JSON.") from exc
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object.")
            return payload

        def _send_html(self, body: str, status: int = 200) -> None:
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return ResearchRequestHandler


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ResearchLoop</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #ffffff;
      --surface: #ffffff;
      --muted-surface: #f7f7f7;
      --border: #e5e5e5;
      --text: #0a0a0a;
      --muted: #737373;
      --success: #147a2e;
      --danger: #b42318;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      line-height: 1.5;
    }

    button,
    textarea {
      font: inherit;
    }

    .shell {
      width: min(920px, calc(100% - 32px));
      margin: 0 auto;
      padding: 24px 0 48px;
    }

    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      min-height: 40px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 40px;
    }

    .header-left {
      display: flex;
      align-items: center;
      gap: 18px;
      min-width: 0;
    }

    .brand {
      font-weight: 600;
    }

    .tabs {
      display: flex;
      align-items: center;
      gap: 4px;
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 3px;
    }

    .tab {
      min-height: 28px;
      padding: 0 12px;
      border-radius: 999px;
      border: 0;
      background: transparent;
      color: var(--muted);
      font-size: 12px;
    }

    .tab.is-active {
      background: #000000;
      color: #ffffff;
    }

    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      border: 1px solid var(--border);
      border-radius: 999px;
      color: var(--muted);
      padding: 3px 10px;
      font-size: 12px;
    }

    .status-pill::before {
      content: "";
      width: 7px;
      height: 7px;
      border-radius: 999px;
      background: var(--muted);
    }

    .status-pill[data-state="connected"]::before,
    .status-pill[data-state="done"]::before {
      background: var(--success);
    }

    .status-pill[data-state="running"]::before {
      background: #0a0a0a;
    }

    .status-pill[data-state="error"]::before,
    .status-pill[data-state="disconnected"]::before {
      background: var(--danger);
    }

    main {
      display: grid;
      gap: 28px;
    }

    .view {
      display: none;
    }

    .view.is-visible {
      display: block;
    }

    .compose {
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      background: var(--surface);
    }

    label {
      display: block;
      margin-bottom: 10px;
      font-weight: 500;
    }

    textarea {
      display: block;
      width: 100%;
      min-height: 150px;
      resize: vertical;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      color: var(--text);
      background: var(--surface);
      outline: none;
    }

    textarea:focus {
      border-color: #0a0a0a;
      box-shadow: 0 0 0 2px #f2f2f2;
    }

    .actions {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-top: 12px;
    }

    button {
      border: 0;
      border-radius: 8px;
      background: #000000;
      color: #ffffff;
      min-height: 38px;
      padding: 0 18px;
      cursor: pointer;
      font-weight: 500;
    }

    button:disabled {
      cursor: not-allowed;
      opacity: 0.55;
    }

    .hint {
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }

    .progress,
    .result,
    .error {
      display: none;
      border-top: 1px solid var(--border);
      padding-top: 20px;
    }

    .progress.is-visible,
    .result.is-visible,
    .error.is-visible {
      display: block;
    }

    .result-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }

    .result.is-collapsed .result-body {
      display: none;
    }

    .steps {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin-top: 12px;
    }

    .step {
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
      min-height: 52px;
      background: var(--muted-surface);
      color: var(--muted);
      font-size: 13px;
    }

    .step.is-active {
      background: var(--surface);
      color: var(--text);
      border-color: #0a0a0a;
    }

    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 10px 0 18px;
    }

    .metric {
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 4px 10px;
      color: var(--muted);
      font-size: 12px;
    }

    .report,
    .eval {
      width: 100%;
      margin: 0;
      overflow-wrap: anywhere;
      line-height: 1.65;
    }

    .report > *:first-child,
    .eval > *:first-child {
      margin-top: 0;
    }

    .report h1,
    .report h2,
    .report h3,
    .eval h1,
    .eval h2,
    .eval h3 {
      margin: 28px 0 10px;
      line-height: 1.2;
    }

    .report h1,
    .eval h1 {
      font-size: 22px;
    }

    .report h2,
    .eval h2 {
      font-size: 18px;
    }

    .report h3,
    .eval h3 {
      font-size: 15px;
    }

    .report p,
    .eval p {
      margin: 0 0 12px;
    }

    .report ul,
    .report ol,
    .eval ul,
    .eval ol {
      margin: 0 0 14px;
      padding-left: 22px;
    }

    .report li,
    .eval li {
      margin: 6px 0;
    }

    .report table,
    .eval table {
      width: 100%;
      border-collapse: collapse;
      margin: 16px 0;
      font-size: 13px;
    }

    .report th,
    .report td,
    .eval th,
    .eval td {
      border: 1px solid var(--border);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }

    .report th,
    .eval th {
      background: var(--muted-surface);
      font-weight: 500;
    }

    .report code,
    .eval code {
      border: 1px solid var(--border);
      border-radius: 6px;
      background: var(--muted-surface);
      padding: 1px 5px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.92em;
    }

    .report pre,
    .eval pre {
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--muted-surface);
      padding: 12px;
      overflow-x: auto;
    }

    .report pre code,
    .eval pre code {
      border: 0;
      background: transparent;
      padding: 0;
    }

    .report blockquote,
    .eval blockquote {
      margin: 14px 0;
      padding-left: 12px;
      border-left: 2px solid var(--border);
      color: var(--muted);
    }

    .report a,
    .eval a {
      color: var(--text);
      text-decoration: underline;
      text-decoration-color: var(--border);
      text-underline-offset: 2px;
    }

    .report a:hover,
    .eval a:hover {
      text-decoration-color: var(--text);
    }

    .citation-link {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.92em;
    }

    .report hr,
    .eval hr {
      border: 0;
      border-top: 1px solid var(--border);
      margin: 22px 0;
    }

    h1 {
      margin: 0 0 12px;
      font-size: 22px;
      line-height: 1.2;
    }

    h2 {
      margin: 28px 0 12px;
      font-size: 15px;
      line-height: 1.2;
    }

    .sources {
      display: grid;
      gap: 8px;
      margin: 0;
      padding: 0;
      list-style: none;
    }

    .source {
      display: grid;
      grid-template-columns: 54px minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      border-top: 1px solid var(--border);
      padding: 10px 0;
    }

    .source-id {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      color: var(--muted);
      font-size: 12px;
    }

    .source-title {
      overflow-wrap: anywhere;
    }

    .source-link {
      color: var(--muted);
      text-decoration: none;
      font-size: 12px;
    }

    .source-link:hover {
      color: var(--text);
      text-decoration: underline;
    }

    .source:target {
      border-radius: 8px;
      outline: 1px solid #0a0a0a;
      outline-offset: 2px;
      background: var(--muted-surface);
    }

    details {
      border-top: 1px solid var(--border);
      margin-top: 24px;
      padding-top: 12px;
    }

    summary {
      cursor: pointer;
      color: var(--muted);
      font-size: 13px;
    }

    .eval {
      margin-top: 12px;
    }

    .error {
      color: var(--danger);
    }

    .list-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }

    .ghost-button {
      min-height: 30px;
      padding: 0 12px;
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--text);
      font-size: 12px;
    }

    .researches {
      display: grid;
      gap: 8px;
      margin: 0;
      padding: 0;
      list-style: none;
    }

    .research-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 0;
      align-items: center;
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--surface);
      overflow: hidden;
    }

    .research-row:hover {
      border-color: #0a0a0a;
    }

    .research-open {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: center;
      width: 100%;
      min-height: auto;
      border: 0;
      border-radius: 0;
      background: transparent;
      color: var(--text);
      padding: 12px;
      text-align: left;
    }

    .research-open:hover {
      background: var(--muted-surface);
    }

    .research-delete {
      align-self: stretch;
      min-height: auto;
      border-left: 1px solid var(--border);
      border-radius: 0;
      background: var(--surface);
      color: var(--danger);
      padding: 0 14px;
      font-size: 12px;
    }

    .research-delete:hover {
      background: #fff7f7;
    }

    .research-title {
      overflow-wrap: anywhere;
      font-weight: 500;
    }

    .research-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 5px;
      color: var(--muted);
      font-size: 12px;
    }

    .empty {
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      color: var(--muted);
      background: var(--muted-surface);
    }

    @media (max-width: 680px) {
      .shell {
        width: min(100% - 24px, 920px);
        padding-top: 16px;
      }

      header {
        align-items: flex-start;
        gap: 12px;
        flex-direction: column;
        margin-bottom: 24px;
      }

      .header-left {
        align-items: flex-start;
        flex-direction: column;
        gap: 10px;
        width: 100%;
      }

      .steps {
        grid-template-columns: 1fr;
      }

      .actions {
        align-items: stretch;
        flex-direction: column;
      }

      button {
        width: 100%;
      }

      .tabs,
      .tab {
        width: 100%;
      }

      .research-row {
        grid-template-columns: 1fr;
        gap: 0;
      }

      .research-open {
        grid-template-columns: 1fr;
      }

      .research-delete {
        min-height: 36px;
        border-left: 0;
        border-top: 1px solid var(--border);
      }

      .source {
        grid-template-columns: 44px minmax(0, 1fr);
      }

      .source-link {
        grid-column: 2;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div class="header-left">
        <div class="brand">ResearchLoop</div>
        <nav class="tabs" aria-label="Views">
          <button class="tab is-active" type="button" data-view="newView">New</button>
          <button class="tab" type="button" data-view="researchesView">Researches</button>
        </nav>
      </div>
      <div class="status-pill" id="topStatus" data-state="checking">Checking local</div>
    </header>

    <main>
      <section class="view is-visible" id="newView">
        <form class="compose" id="researchForm">
          <label for="question">What do you want to research?</label>
          <textarea id="question" name="question" placeholder="Research question"></textarea>
          <div class="actions">
            <div class="hint" id="workspaceHint">workspaces/</div>
            <button type="submit" id="runButton">Run</button>
          </div>
        </form>

        <section class="progress" id="progress">
          <h1>Researching</h1>
          <div class="hint" id="stageText">Queued</div>
          <div class="steps">
            <div class="step is-active" data-step="Creating workspace">Workspace</div>
            <div class="step" data-step="Researching">Sources</div>
            <div class="step" data-step="Running iteration">Report</div>
            <div class="step" data-step="Done">Check</div>
          </div>
        </section>
      </section>

      <section class="view" id="researchesView">
        <div class="list-header">
          <h1>Researches</h1>
          <button class="ghost-button" type="button" id="refreshResearches">Refresh</button>
        </div>
        <ul class="researches" id="researchesList"></ul>
      </section>

      <section class="error" id="errorBox"></section>

      <section class="result" id="result">
        <div class="result-header">
          <h1>Report</h1>
          <button class="ghost-button" type="button" id="toggleResult">Hide report</button>
        </div>
        <div class="result-body">
          <div class="meta" id="metrics"></div>
          <article class="report" id="reportText"></article>

          <h2>Sources Used</h2>
          <ul class="sources" id="sourcesList"></ul>

          <details>
            <summary>Inspect run</summary>
            <article class="eval" id="evalText"></article>
          </details>
        </div>
      </section>
    </main>
  </div>

  <script>
    const form = document.getElementById("researchForm");
    const question = document.getElementById("question");
    const runButton = document.getElementById("runButton");
    const progress = document.getElementById("progress");
    const result = document.getElementById("result");
    const errorBox = document.getElementById("errorBox");
    const stageText = document.getElementById("stageText");
    const topStatus = document.getElementById("topStatus");
    const workspaceHint = document.getElementById("workspaceHint");
    const metrics = document.getElementById("metrics");
    const reportText = document.getElementById("reportText");
    const sourcesList = document.getElementById("sourcesList");
    const evalText = document.getElementById("evalText");
    const toggleResult = document.getElementById("toggleResult");
    const steps = Array.from(document.querySelectorAll(".step"));
    const tabs = Array.from(document.querySelectorAll(".tab"));
    const views = Array.from(document.querySelectorAll(".view"));
    const researchesList = document.getElementById("researchesList");
    const refreshResearches = document.getElementById("refreshResearches");
    let pollTimer = null;
    let activeJobId = null;
    let currentResearchName = "";
    let statusMode = "idle";

    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        setActiveView(tab.dataset.view);
      });
    });

    refreshResearches.addEventListener("click", () => loadResearches());
    toggleResult.addEventListener("click", () => {
      const collapsed = result.classList.toggle("is-collapsed");
      toggleResult.textContent = collapsed ? "Show report" : "Hide report";
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      stopPolling();
      setActiveView("newView", false);
      setRunning(true);
      hideResult();
      try {
        const response = await fetch("/api/research", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: question.value })
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || "Research could not start.");
        }
        workspaceHint.textContent = payload.workspace_name;
        startPolling(payload.id);
      } catch (error) {
        showError(error.message);
        setRunning(false);
      }
    });

    function startPolling(jobId) {
      activeJobId = jobId;
      stopPolling();
      poll(jobId);
      pollTimer = setInterval(() => poll(jobId), 1400);
    }

    function stopPolling() {
      if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
    }

    async function poll(jobId) {
      try {
        const response = await fetch(`/api/jobs/${jobId}`, { cache: "no-store" });
        const job = await response.json();
        if (jobId !== activeJobId) return;
        if (!response.ok) {
          showError(job.error || "Job not found.");
          activeJobId = null;
          stopPolling();
          setRunning(false);
          return;
        }
        updateProgress(job);
        if (job.status === "done") {
          activeJobId = null;
          stopPolling();
          setRunning(false);
          renderResult(job.result);
          loadResearches();
        }
        if (job.status === "error") {
          activeJobId = null;
          stopPolling();
          setRunning(false);
          showError(job.error || "Research failed.");
          loadResearches();
        }
      } catch (_error) {
        if (jobId !== activeJobId) return;
        stageText.textContent = "Checking local run";
        setStatus("Checking run", "running");
      }
    }

    function setRunning(running) {
      runButton.disabled = running;
      question.disabled = running;
      progress.classList.toggle("is-visible", running);
      errorBox.classList.remove("is-visible");
      if (running) {
        statusMode = "running";
        setStatus("Running locally", "running");
      } else if (statusMode === "running") {
        statusMode = "idle";
        checkConnection();
      }
    }

    function updateProgress(job) {
      stageText.textContent = job.stage || "Running";
      workspaceHint.textContent = job.workspace || job.workspace_name || "workspaces/";
      steps.forEach((step) => {
        const key = step.getAttribute("data-step");
        step.classList.toggle("is-active", (job.stage || "").startsWith(key));
      });
    }

    function renderResult(data) {
      currentResearchName = data.workspace_name || "";
      result.classList.add("is-visible");
      result.classList.remove("is-collapsed");
      toggleResult.textContent = "Hide report";
      statusMode = "done";
      setStatus("Done locally", "done");
      reportText.innerHTML = renderMarkdown(data.report || "");
      evalText.innerHTML = renderMarkdown(`Workspace: ${data.workspace}\\n\\n${data.eval || ""}`);
      metrics.innerHTML = "";
      addMetric(`Score ${Number(data.score || 0).toFixed(2)}`);
      addMetric(`${(data.sources || []).length} sources`);
      if (data.best_iteration) addMetric(`Best ${data.best_iteration}`);
      sourcesList.innerHTML = "";
      (data.sources || []).forEach((source) => {
        const item = document.createElement("li");
        item.className = "source";
        item.id = `source-${source.id}`;
        const id = document.createElement("div");
        id.className = "source-id";
        id.textContent = source.id;
        const title = document.createElement("div");
        title.className = "source-title";
        title.textContent = source.title || "Untitled source";
        item.append(id, title);
        if (source.url) {
          const link = document.createElement("a");
          link.className = "source-link";
          link.href = source.url;
          link.target = "_blank";
          link.rel = "noreferrer";
          link.textContent = "Open";
          item.append(link);
        }
        sourcesList.append(item);
      });
    }

    async function loadResearches(options = {}) {
      const quiet = Boolean(options.quiet);
      if (!quiet) {
        researchesList.innerHTML = `<li class="empty">Loading researches</li>`;
      }
      try {
        const response = await fetch("/api/researches", { cache: "no-store" });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || "Could not load researches.");
        }
        renderResearches(payload.researches || []);
        reconcileActiveResearch(payload.researches || []);
      } catch (error) {
        if (quiet) return;
        researchesList.innerHTML = "";
        const item = document.createElement("li");
        item.className = "empty";
        item.textContent = error.message;
        researchesList.append(item);
      }
    }

    function renderResearches(items) {
      researchesList.innerHTML = "";
      if (!items.length) {
        const item = document.createElement("li");
        item.className = "empty";
        item.textContent = "No researches yet.";
        researchesList.append(item);
        return;
      }
      items.forEach((research) => {
        const item = document.createElement("li");
        const row = document.createElement("div");
        row.className = "research-row";
        const button = document.createElement("button");
        button.className = "research-open";
        button.type = "button";
        button.addEventListener("click", () => openResearch(research.name));
        const body = document.createElement("div");
        const title = document.createElement("div");
        title.className = "research-title";
        title.textContent = research.question || research.name;
        const meta = document.createElement("div");
        meta.className = "research-meta";
        meta.textContent = `${research.status || "ready"} · score ${Number(research.score || 0).toFixed(2)} · ${research.source_count || 0} sources`;
        body.append(title, meta);
        const updated = document.createElement("div");
        updated.className = "hint";
        updated.textContent = shortDate(research.updated_at);
        button.append(body, updated);
        const deleteButton = document.createElement("button");
        deleteButton.className = "research-delete";
        deleteButton.type = "button";
        deleteButton.textContent = "Delete";
        deleteButton.disabled = ["queued", "running"].includes(research.status);
        deleteButton.addEventListener("click", () => deleteResearch(research));
        row.append(button, deleteButton);
        item.append(row);
        researchesList.append(item);
      });
    }

    function reconcileActiveResearch(items) {
      if (activeJobId) {
        const active = items.find((item) => item.job_id === activeJobId);
        if (active && active.status === "done") {
          activeJobId = null;
          stopPolling();
          setRunning(false);
          openResearch(active.name);
          return;
        }
        if (active && active.status === "error") {
          activeJobId = null;
          stopPolling();
          setRunning(false);
          showError(active.stage || "Research failed.");
          return;
        }
      }

      const running = items.find((item) => item.job_id && ["queued", "running"].includes(item.status));
      if (!activeJobId && running) {
        setRunning(true);
        updateProgress({
          stage: running.stage || running.status,
          workspace: running.workspace,
          workspace_name: running.name
        });
        startPolling(running.job_id);
      }
    }

    async function openResearch(name) {
      try {
        const response = await fetch(`/api/researches/${encodeURIComponent(name)}`);
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || "Could not open research.");
        }
        renderResult(payload);
      } catch (error) {
        showError(error.message);
      }
    }

    async function deleteResearch(research) {
      const label = research.question || research.name;
      const confirmed = window.confirm(`Delete "${label}"? This removes the local workspace files.`);
      if (!confirmed) return;
      try {
        const response = await fetch(`/api/researches/${encodeURIComponent(research.name)}`, {
          method: "DELETE"
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || "Could not delete research.");
        }
        if (currentResearchName === research.name) {
          hideResult();
        }
        statusMode = "idle";
        setStatus("Connected locally", "connected");
        loadResearches();
      } catch (error) {
        showError(error.message);
      }
    }

    function setActiveView(viewId, refresh = true) {
      tabs.forEach((tab) => {
        tab.classList.toggle("is-active", tab.dataset.view === viewId);
      });
      views.forEach((view) => {
        view.classList.toggle("is-visible", view.id === viewId);
      });
      if (viewId === "researchesView" && refresh) {
        loadResearches();
      }
    }

    function addMetric(text) {
      const item = document.createElement("div");
      item.className = "metric";
      item.textContent = text;
      metrics.append(item);
    }

    function showError(message) {
      errorBox.textContent = message;
      errorBox.classList.add("is-visible");
      statusMode = "error";
      setStatus("Local error", "error");
    }

    function hideResult() {
      currentResearchName = "";
      result.classList.remove("is-visible");
      result.classList.remove("is-collapsed");
      toggleResult.textContent = "Hide report";
      reportText.textContent = "";
      sourcesList.innerHTML = "";
      metrics.innerHTML = "";
      evalText.textContent = "";
    }

    function renderMarkdown(markdown) {
      const lines = String(markdown || "").replace(/\\r\\n/g, "\\n").split("\\n");
      const html = [];
      let paragraph = [];
      let listType = null;
      let inCode = false;
      let codeLines = [];

      const flushParagraph = () => {
        if (!paragraph.length) return;
        html.push(`<p>${renderInline(paragraph.join(" "))}</p>`);
        paragraph = [];
      };

      const closeList = () => {
        if (!listType) return;
        html.push(`</${listType}>`);
        listType = null;
      };

      const flushCode = () => {
        html.push(`<pre><code>${escapeHtml(codeLines.join("\\n"))}</code></pre>`);
        codeLines = [];
      };

      for (let index = 0; index < lines.length; index += 1) {
        const line = lines[index];
        const trimmed = line.trim();

        if (trimmed.startsWith("```")) {
          if (inCode) {
            flushCode();
            inCode = false;
          } else {
            flushParagraph();
            closeList();
            inCode = true;
          }
          continue;
        }

        if (inCode) {
          codeLines.push(line);
          continue;
        }

        if (!trimmed) {
          flushParagraph();
          closeList();
          continue;
        }

        if (isTableStart(lines, index)) {
          flushParagraph();
          closeList();
          const table = collectTable(lines, index);
          html.push(renderTable(table.rows));
          index = table.endIndex;
          continue;
        }

        if (/^---+$/.test(trimmed)) {
          flushParagraph();
          closeList();
          html.push("<hr>");
          continue;
        }

        const heading = /^(#{1,3})\\s+(.+)$/.exec(trimmed);
        if (heading) {
          flushParagraph();
          closeList();
          const level = heading[1].length;
          html.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
          continue;
        }

        const quote = /^>\\s+(.+)$/.exec(trimmed);
        if (quote) {
          flushParagraph();
          closeList();
          html.push(`<blockquote>${renderInline(quote[1])}</blockquote>`);
          continue;
        }

        const unordered = /^[-*]\\s+(.+)$/.exec(trimmed);
        if (unordered) {
          flushParagraph();
          if (listType !== "ul") {
            closeList();
            listType = "ul";
            html.push("<ul>");
          }
          html.push(`<li>${renderInline(unordered[1])}</li>`);
          continue;
        }

        const ordered = /^\\d+\\.\\s+(.+)$/.exec(trimmed);
        if (ordered) {
          flushParagraph();
          if (listType !== "ol") {
            closeList();
            listType = "ol";
            html.push("<ol>");
          }
          html.push(`<li>${renderInline(ordered[1])}</li>`);
          continue;
        }

        paragraph.push(trimmed);
      }

      if (inCode) flushCode();
      flushParagraph();
      closeList();
      return html.join("\\n");
    }

    function isTableStart(lines, index) {
      const current = lines[index]?.trim() || "";
      const next = lines[index + 1]?.trim() || "";
      return current.startsWith("|") && current.endsWith("|") && /^\\|?\\s*:?-{3,}:?\\s*(\\|\\s*:?-{3,}:?\\s*)+\\|?$/.test(next);
    }

    function collectTable(lines, startIndex) {
      const rows = [splitTableRow(lines[startIndex])];
      let index = startIndex + 2;
      while (index < lines.length) {
        const line = lines[index].trim();
        if (!line.startsWith("|") || !line.endsWith("|")) break;
        rows.push(splitTableRow(line));
        index += 1;
      }
      return { rows, endIndex: index - 1 };
    }

    function splitTableRow(line) {
      return line.replace(/^\\|/, "").replace(/\\|$/, "").split("|").map((cell) => cell.trim());
    }

    function renderTable(rows) {
      const [head, ...body] = rows;
      const headHtml = head.map((cell) => `<th>${renderInline(cell)}</th>`).join("");
      const bodyHtml = body
        .map((row) => `<tr>${row.map((cell) => `<td>${renderInline(cell)}</td>`).join("")}</tr>`)
        .join("");
      return `<table><thead><tr>${headHtml}</tr></thead><tbody>${bodyHtml}</tbody></table>`;
    }

    function renderInline(value) {
      const placeholders = [];
      const hold = (html) => {
        const token = `@@INLINE_${placeholders.length}@@`;
        placeholders.push(html);
        return token;
      };

      let text = escapeHtml(value);
      text = text.replace(/`([^`]+)`/g, (_match, code) => hold(`<code>${code}</code>`));
      text = text.replace(/\\[([^\\]]+)\\]\\((https?:\\/\\/[^\\s)]+)\\)/g, (match, label, url) => {
        const href = safeHref(url);
        if (!href) return match;
        return hold(`<a href="${href}" target="_blank" rel="noreferrer">${label}</a>`);
      });
      text = text.replace(/(^|[\\s(])((?:https?:\\/\\/)[^\\s<]+)/g, (_match, prefix, rawUrl) => {
        const split = splitTrailingPunctuation(rawUrl);
        const href = safeHref(split.url);
        if (!href) return `${prefix}${rawUrl}`;
        return `${prefix}${hold(`<a href="${href}" target="_blank" rel="noreferrer">${split.url}</a>`)}${split.trailing}`;
      });
      text = text.replace(/\\[(S\\d+)\\]/g, (_match, sourceId) =>
        hold(`<a class="citation-link" href="#source-${sourceId}">[${sourceId}]</a>`)
      );
      text = text
        .replace(/\\*\\*([^*]+)\\*\\*/g, "<strong>$1</strong>")
        .replace(/\\*([^*]+)\\*/g, "<em>$1</em>");

      return text.replace(/@@INLINE_(\\d+)@@/g, (_match, index) => placeholders[Number(index)] || "");
    }

    function splitTrailingPunctuation(value) {
      let url = String(value);
      let trailing = "";
      while (/[.,;:!?)\\]]$/.test(url)) {
        trailing = url.slice(-1) + trailing;
        url = url.slice(0, -1);
      }
      return { url, trailing };
    }

    function safeHref(value) {
      const unescaped = String(value).replace(/&amp;/g, "&");
      if (!/^https?:\\/\\//i.test(unescaped)) return "";
      return escapeAttribute(unescaped);
    }

    function escapeAttribute(value) {
      return String(value)
        .replace(/&/g, "&amp;")
        .replace(/"/g, "&quot;");
    }

    function escapeHtml(value) {
      return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }

    function shortDate(value) {
      if (!value) return "";
      return String(value).replace("T", " ").replace("Z", "");
    }

    function setStatus(text, state) {
      topStatus.textContent = text;
      topStatus.dataset.state = state;
    }

    async function checkConnection() {
      if (statusMode !== "idle") return;
      try {
        const response = await fetch("/api/health", { cache: "no-store" });
        if (!response.ok) throw new Error("health check failed");
        setStatus("Connected locally", "connected");
      } catch (_error) {
        setStatus("Disconnected", "disconnected");
      }
    }

    checkConnection();
    setInterval(checkConnection, 5000);
    setInterval(() => {
      if (activeJobId) loadResearches({ quiet: true });
    }, 5000);
    loadResearches();
  </script>
</body>
</html>
"""
