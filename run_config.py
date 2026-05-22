from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RUN_CONFIG_FILENAME = "run_config.json"
VALID_BACKENDS = {"openai-compatible", "openai", "chat-completions"}
VALID_SEARCH_BACKENDS = {"none", "tavily"}
VALID_SYNTHESIS_MODES = {"json", "markdown"}


@dataclass(frozen=True)
class RunConfig:
    version: int = 1
    backend: str = "openai-compatible"
    model: str | None = None
    synthesis_mode: str = "markdown"
    search_backend: str = "tavily"
    max_results: int = 5
    min_delta: float = 0.1
    iterations: int = 1

    def __post_init__(self) -> None:
        backend = _clean_choice("backend", self.backend, VALID_BACKENDS)
        synthesis_mode = _clean_choice("synthesis_mode", self.synthesis_mode, VALID_SYNTHESIS_MODES)
        search_backend = _clean_choice("search_backend", self.search_backend, VALID_SEARCH_BACKENDS)
        model = self.model.strip() if isinstance(self.model, str) else self.model
        if model == "":
            model = None
        max_results = int(self.max_results)
        iterations = int(self.iterations)
        if max_results < 1:
            raise ValueError("run config max_results must be at least 1.")
        if iterations < 1:
            raise ValueError("run config iterations must be at least 1.")
        object.__setattr__(self, "version", int(self.version))
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "synthesis_mode", synthesis_mode)
        object.__setattr__(self, "search_backend", search_backend)
        object.__setattr__(self, "max_results", max_results)
        object.__setattr__(self, "min_delta", float(self.min_delta))
        object.__setattr__(self, "iterations", iterations)

    @classmethod
    def default(cls) -> "RunConfig":
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        model = data.get("model")
        if model is not None:
            model = str(model)
        return cls(
            version=int(data.get("version") or 1),
            backend=str(data.get("backend") or "openai-compatible"),
            model=model,
            synthesis_mode=str(data.get("synthesis_mode") or "markdown"),
            search_backend=str(data.get("search_backend") or data.get("search") or "tavily"),
            max_results=int(data["max_results"]) if data.get("max_results") is not None else 5,
            min_delta=float(data["min_delta"]) if data.get("min_delta") is not None else 0.1,
            iterations=int(data["iterations"]) if data.get("iterations") is not None else 1,
        )

    def with_overrides(self, **overrides: Any) -> "RunConfig":
        data = self.to_dict()
        for key, value in overrides.items():
            if value is not None:
                data[key] = value
        return RunConfig.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "backend": self.backend,
            "model": self.model,
            "synthesis_mode": self.synthesis_mode,
            "search_backend": self.search_backend,
            "max_results": self.max_results,
            "min_delta": self.min_delta,
            "iterations": self.iterations,
        }


def load_run_config(path: Path | None = None) -> RunConfig:
    if path is None:
        return RunConfig.default()
    if not path.exists():
        raise ValueError(f"run config file does not exist: {path}")
    return RunConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_run_config_for_workspace(workspace: Path) -> RunConfig:
    workspace_config = workspace / RUN_CONFIG_FILENAME
    if workspace_config.exists():
        return load_run_config(workspace_config)
    root_config = Path(RUN_CONFIG_FILENAME)
    if root_config.exists():
        return load_run_config(root_config)
    return RunConfig.default()


def load_default_run_config() -> RunConfig:
    root_config = Path(RUN_CONFIG_FILENAME)
    if root_config.exists():
        return load_run_config(root_config)
    return RunConfig.default()


def write_run_config(path: Path, config: RunConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean_choice(name: str, value: str, valid: set[str]) -> str:
    cleaned = str(value).strip().lower()
    if cleaned not in valid:
        raise ValueError(f"run config {name} must be one of {sorted(valid)}, got {cleaned!r}")
    return cleaned
