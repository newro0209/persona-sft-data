"""Pipeline configuration — the only place paths, models and ratios appear.

Every path in the pipeline is derived from a single ``data_root``. Stages do not
know where their output goes; the config tells them. That is what keeps
``Path("data/runs/exaone35-research/generated.jsonl")`` from reappearing as a
default fifteen files deep, which is how the previous pipeline became
impossible to re-run.

Experiments are distinguished by config, not by one-off directories: point
``data_root`` somewhere else and set smaller limits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a config file is missing something a stage needs."""


@dataclass(frozen=True)
class TeacherConfig:
    """One teacher, addressed over an OpenAI-compatible endpoint.

    ``model`` is sent with every request. vLLM rejects an id it is not serving
    with a 404, so pointing at a server running the other model fails loudly
    instead of silently generating with the wrong teacher.
    """

    name: str
    model: str
    base_url: str
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 512
    concurrency: int = 200
    timeout: float = 300.0

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> TeacherConfig:
        missing = [k for k in ("model", "base_url") if k not in raw]
        if missing:
            raise ConfigError(f"teacher {name!r} is missing {', '.join(missing)}")
        known = {f for f in cls.__dataclass_fields__ if f != "name"}
        unknown = set(raw) - known
        if unknown:
            raise ConfigError(f"teacher {name!r} has unknown keys: {sorted(unknown)}")
        return cls(name=name, **raw)


@dataclass(frozen=True)
class PipelineConfig:
    """Everything a stage needs to know, loaded from one JSON file."""

    path: Path
    root: Path
    data_root: Path
    seed: int
    persona_doc: Path
    teachers: dict[str, TeacherConfig]
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Where export writes trainer-ready datasets. Separate from data_root
    # because the two have different lives: data/ is the pipeline's working
    # state, datasets/ is what gets handed to a trainer or published. Defaults
    # to <root>/datasets in __post_init__, since a field default cannot see root.
    datasets_root: Path | None = None

    def __post_init__(self) -> None:
        if self.datasets_root is None:
            object.__setattr__(self, "datasets_root", (self.root / "datasets").resolve())

    # ---- paths -----------------------------------------------------------
    # Three directories, always. raw/ is what a generator produced, filtered/
    # is what survived the gates, final/ is the mixed and split corpus.

    def raw(self, name: str) -> Path:
        return self.data_root / "raw" / f"{name}.jsonl"

    def filtered(self, name: str) -> Path:
        return self.data_root / "filtered" / f"{name}.jsonl"

    def final(self, name: str) -> Path:
        return self.data_root / "final" / f"{name}.jsonl"

    @staticmethod
    def stats_path(output: Path) -> Path:
        """Every stage writes one of these next to its output."""
        return output.with_suffix(output.suffix + ".stats.json")

    @staticmethod
    def rejected_path(output: Path) -> Path:
        """Rejects are written, not dropped: unless you can count what was
        thrown away you cannot say anything about quality."""
        return output.with_suffix(output.suffix + ".rejected.jsonl")

    @staticmethod
    def sample_path(output: Path) -> Path:
        """A small random sample of survivors, for a human to actually read."""
        return output.with_suffix(output.suffix + ".sample.jsonl")

    # ---- lookups ---------------------------------------------------------

    def stage(self, name: str) -> dict[str, Any]:
        if name not in self.stages:
            raise ConfigError(f"config {self.path} has no stage {name!r}")
        return self.stages[name]

    def teacher_for(self, stage_name: str) -> TeacherConfig:
        """Resolve the teacher a stage asked for, by name."""
        wanted = self.stage(stage_name).get("teacher")
        if wanted is None:
            raise ConfigError(f"stage {stage_name!r} declares no teacher")
        if wanted not in self.teachers:
            raise ConfigError(
                f"stage {stage_name!r} wants teacher {wanted!r}; "
                f"config defines {sorted(self.teachers)}"
            )
        return self.teachers[wanted]

    def stage_seed(self, stage_name: str) -> int:
        """A per-stage seed derived from the global one, so re-running a single
        stage does not reshuffle the others."""
        return self.seed + sum(ord(c) for c in stage_name)

    # ---- loading ---------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> PipelineConfig:
        # Resolve before storing: the manifest records this path relative to
        # root, and a config given as "configs/mongle.json" is not relative to an
        # absolute root.
        path = Path(path).resolve()
        if not path.exists():
            raise ConfigError(f"config not found: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))

        root = path.parent.parent
        for required in ("data_root", "seed", "persona_doc", "teachers", "stages"):
            if required not in raw:
                raise ConfigError(f"config {path} is missing {required!r}")

        teachers = {
            name: TeacherConfig.from_dict(name, spec)
            for name, spec in raw["teachers"].items()
        }
        if not teachers:
            raise ConfigError(f"config {path} defines no teachers")

        return cls(
            path=path,
            root=root,
            data_root=(root / raw["data_root"]).resolve(),
            seed=int(raw["seed"]),
            persona_doc=(root / raw["persona_doc"]).resolve(),
            teachers=teachers,
            datasets_root=(root / raw.get("datasets_root", "datasets")).resolve(),
            stages=raw["stages"],
        )

    def ensure_dirs(self) -> None:
        for sub in ("raw", "filtered", "final"):
            (self.data_root / sub).mkdir(parents=True, exist_ok=True)
