"""설정 — 경로·모델·비율·한도가 나타나는 유일한 곳.

모든 경로는 ``data_root`` 하나에서 파생된다. 단계는 자기 출력 경로를 모르고 설정이
알려 준다. 단계별 설정은 dict가 아니라 단계 플러그인이 선언한 dataclass로 만들어
로드 시점에 검증한다 — 모르는 키가 조용히 무시되는 일이 없다.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from persona_sft_data.core.registry import (
    EXTRACTORS,
    FORMATS,
    PROFILES,
    RECIPES,
    STAGES,
    TEACHERS,
    TRANSLATORS,
    PluginError,
    load_plugins,
)

_LANGUAGE = re.compile(r"^[a-z]{2}$")
CHAT_TEMPLATES = ("chatml",)


class ConfigError(ValueError):
    """설정 파일에 단계가 필요로 하는 것이 없거나 잘못됐다."""


def build_settings(settings_type: type, raw: dict[str, Any], where: str) -> Any:
    """dict를 dataclass로. 모르는 키와 빠진 필수 키는 ``where``와 함께 알린다."""
    fields = {f.name: f for f in dataclasses.fields(settings_type)}
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        raise ConfigError(f"{where}: 모르는 키 {unknown} (허용: {sorted(fields)})")
    missing = [
        n for n, f in fields.items()
        if n not in raw and f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    ]
    if missing:
        raise ConfigError(f"{where}: 필수 키가 없다 {missing}")
    return settings_type(**raw)


def _validate_recipe(stage_name: str, raw: dict[str, Any] | None) -> None:
    """단계 설정의 ``recipe``를 로드 시점에 검증한다.

    실행 시점에만 보면 ``run``이 교사 단계를 다 돌린 뒤(실제 교사면 수천 호출)
    내보내기에서 터지고, ``check``도 설정 오류(2)가 아니라 단계 실패(1)로 끝난다.
    """
    where = f"stages.{stage_name}.recipe"
    if raw is not None and not isinstance(raw, Mapping):
        raise ConfigError(f"{where}는 kind를 담은 객체여야 한다: {raw!r}")
    settings = dict(raw or {})
    kind = settings.pop("kind", None)
    if not kind:
        raise ConfigError(f"{where}.kind가 없다")
    try:
        recipe = RECIPES.get(str(kind))
    except PluginError as exc:
        raise ConfigError(f"{where}: {exc}") from None
    build_settings(recipe.settings_type, settings, where)


@dataclass(frozen=True)
class TeacherConfig:
    """교사 하나. ``kind``가 백엔드 플러그인을 고른다."""

    name: str
    model: str
    base_url: str
    kind: str = "openai"
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 256
    concurrency: int = 64
    timeout: float = 300.0
    api_key: str | None = None

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> "TeacherConfig":
        return build_settings(cls, {"name": name, **raw}, f"teacher {name!r}")


@dataclass(frozen=True)
class SourceConfig:
    """외부 텍스트 소스 하나."""

    name: str
    format: str
    language: str
    license: str
    fields: tuple[str, ...]
    url: str | None = None
    path: Path | None = None
    extract_kind: str = "field"
    extract: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any], root: Path) -> "SourceConfig":
        where = f"source {name!r}"
        known = {"format", "language", "license", "fields", "url", "path", "extract"}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ConfigError(f"{where}: 모르는 키 {unknown} (허용: {sorted(known)})")
        for key in ("format", "language", "license", "fields"):
            if key not in raw:
                raise ConfigError(f"{where}: 필수 키 {key!r}가 없다")
        if bool(raw.get("url")) == bool(raw.get("path")):
            raise ConfigError(f"{where}: url과 path 중 정확히 하나만 적는다")
        fields_value = tuple(str(f) for f in raw["fields"])
        if not fields_value:
            raise ConfigError(f"{where}: fields가 비어 있다")
        language = str(raw["language"]).lower()
        if not _LANGUAGE.match(language):
            raise ConfigError(f"{where}: language는 ISO 639-1 두 글자여야 한다: {language!r}")
        extract = dict(raw.get("extract") or {})
        kind = str(extract.pop("kind", "field"))
        return cls(
            name=name, format=str(raw["format"]), language=language, license=str(raw["license"]),
            fields=fields_value, url=raw.get("url"),
            path=(root / raw["path"]).resolve() if raw.get("path") else None,
            extract_kind=kind, extract=extract,
        )


@dataclass(frozen=True)
class StudentConfig:
    """학생(학습 대상) 모델. 내보내기와 레시피가 읽는다."""

    model: str
    trust_remote_code: bool = True
    chat_template: str = "chatml"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StudentConfig":
        student = build_settings(cls, raw, "student")
        if not student.model:
            raise ConfigError("student.model이 비어 있다")
        if student.chat_template not in CHAT_TEMPLATES:
            raise ConfigError(
                f"student.chat_template {student.chat_template!r}은(는) 지원하지 않는다 (허용: {CHAT_TEMPLATES})"
            )
        return student


@dataclass(frozen=True)
class PipelineConfig:
    """한 JSON 파일에서 온, 단계가 알아야 할 전부."""

    path: Path
    root: Path
    profile: str
    language: str
    data_root: Path
    datasets_root: Path
    seed: int
    persona_doc: Path
    plugins: tuple[str, ...]
    student: StudentConfig
    teachers: dict[str, TeacherConfig]
    sources: dict[str, SourceConfig]
    stages: dict[str, Any]

    # -- 경로 ------------------------------------------------------------------

    def raw(self, name: str) -> Path:
        return self.data_root / "raw" / f"{name}.jsonl"

    def filtered(self, name: str) -> Path:
        return self.data_root / "filtered" / f"{name}.jsonl"

    def final(self, name: str) -> Path:
        return self.data_root / "final" / f"{name}.jsonl"

    @staticmethod
    def stats_path(output: Path) -> Path:
        return output.with_suffix(output.suffix + ".stats.json")

    @staticmethod
    def rejected_path(output: Path) -> Path:
        return output.with_suffix(output.suffix + ".rejected.jsonl")

    @staticmethod
    def sample_path(output: Path) -> Path:
        return output.with_suffix(output.suffix + ".sample.jsonl")

    def ensure_dirs(self) -> None:
        for sub in ("raw", "filtered", "final", "cache"):
            (self.data_root / sub).mkdir(parents=True, exist_ok=True)

    # -- 조회 ------------------------------------------------------------------

    def has_stage(self, name: str) -> bool:
        return name in self.stages

    def stage_settings(self, name: str) -> Any:
        if name not in self.stages:
            raise ConfigError(f"설정 {self.path}에 stage {name!r}이(가) 없다")
        return self.stages[name]

    def teacher_for(self, stage_name: str) -> TeacherConfig:
        wanted = getattr(self.stage_settings(stage_name), "teacher", None)
        if wanted is None:
            raise ConfigError(f"stage {stage_name!r}은(는) teacher를 선언하지 않았다")
        return self.teachers[wanted]

    def source(self, name: str) -> SourceConfig:
        if name not in self.sources:
            raise ConfigError(f"source {name!r}이(가) 설정에 없다 (있는 것: {sorted(self.sources)})")
        return self.sources[name]

    def stage_seed(self, stage_name: str) -> int:
        """전역 시드에서 단계별로 파생. 한 단계만 다시 돌려도 다른 단계가 안 바뀐다."""
        return self.seed + sum(ord(c) for c in stage_name)

    def session_stages(self) -> tuple[str, ...]:
        """세션을 raw에 내는, 설정된 단계들."""
        out = []
        for name in self.stages:
            plugin = STAGES.get(name)
            if getattr(plugin, "record_kind", None) == "session" and getattr(plugin, "produces", None) == "raw":
                out.append(name)
        return tuple(out)

    def validate_pipeline(self) -> None:
        """단계 사이의 관계. ``run``과 ``check``가 부른다."""
        if "assemble" not in self.stages:
            raise ConfigError("stages.assemble은 필수다")
        if "respond" in self.stages and "ingest" not in self.stages:
            raise ConfigError("stages.respond는 stages.ingest를 필요로 한다")
        assemble = self.stage_settings("assemble")
        ratios = dict(getattr(assemble, "ratios", {}))
        extra = sorted(set(ratios) - set(self.session_stages()))
        if extra:
            raise ConfigError(
                f"stages.assemble.ratios의 키 {extra}은(는) 세션 생성 단계가 아니다 (있는 것: {self.session_stages()})"
            )
        if abs(sum(ratios.values()) - 1.0) > 1e-6:
            raise ConfigError(f"stages.assemble.ratios의 합이 1이 아니다: {sum(ratios.values())}")
        split = dict(getattr(assemble, "split", {}))
        if set(split) != {"train", "val", "test"} or abs(sum(split.values()) - 1.0) > 1e-6:
            raise ConfigError(f"stages.assemble.split은 train·val·test 세 키의 합이 1이어야 한다: {split}")

    # -- 로드 ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        path = Path(path).resolve()
        if not path.exists():
            raise ConfigError(f"설정 파일이 없다: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        # 상대 경로는 설정 파일의 부모의 부모(저장소 루트)를 기준으로 푼다.
        root = path.parent.parent
        for key in ("profile", "language", "data_root", "seed", "persona_doc", "student", "teachers", "stages"):
            if raw.get(key) is None:
                raise ConfigError(f"설정 {path}에 {key!r}가 없다")

        try:
            # 로컬 플러그인은 저장소 루트 기준이다 — 콘솔 스크립트의 sys.path에는 없다.
            plugins = tuple(load_plugins(raw.get("plugins") or [], search_path=root))
            if raw["profile"] not in PROFILES.names():
                raise ConfigError(f"profile {raw['profile']!r}은(는) 등록되지 않았다 (있는 것: {PROFILES.names()})")
        except PluginError as exc:
            raise ConfigError(str(exc)) from None

        language = str(raw["language"]).lower()
        if not _LANGUAGE.match(language):
            raise ConfigError(f"language는 ISO 639-1 두 글자여야 한다: {language!r}")

        teachers = {n: TeacherConfig.from_dict(n, spec) for n, spec in raw["teachers"].items()}
        for t in teachers.values():
            if t.kind not in TEACHERS.names():
                raise ConfigError(
                    f"teacher {t.name!r}의 kind {t.kind!r}은(는) 등록되지 않았다 (있는 것: {TEACHERS.names()})"
                )
        sources = {n: SourceConfig.from_dict(n, spec, root) for n, spec in (raw.get("sources") or {}).items()}
        for s in sources.values():
            if s.format not in FORMATS.names():
                raise ConfigError(
                    f"source {s.name!r}의 format {s.format!r}은(는) 등록되지 않았다 (있는 것: {FORMATS.names()})"
                )
            if s.extract_kind not in EXTRACTORS.names():
                raise ConfigError(
                    f"source {s.name!r}의 extract.kind {s.extract_kind!r}은(는) 등록되지 않았다 "
                    f"(있는 것: {EXTRACTORS.names()})"
                )
            # 추출 설정도 여기서 검증한다. 읽기 시점에만 보면 모르는 키가 단계 실패(1)로
            # 나타나 설정 오류(2)와 구분되지 않는다.
            build_settings(
                EXTRACTORS.get(s.extract_kind).settings_type, dict(s.extract), f"source {s.name!r} extract"
            )

        stages: dict[str, Any] = {}
        for name, spec in raw["stages"].items():
            try:
                plugin = STAGES.get(name)
            except PluginError as exc:
                raise ConfigError(f"stage {name!r}: {exc}") from None
            settings = build_settings(plugin.settings_type, dict(spec or {}), f"stages.{name}")
            teacher = getattr(settings, "teacher", None)
            if teacher is not None and teacher not in teachers:
                raise ConfigError(
                    f"stages.{name}: teacher {teacher!r}은(는) 정의되지 않았다 (있는 것: {sorted(teachers)})"
                )
            translator = getattr(settings, "translator", None)
            if translator is not None and translator not in TRANSLATORS.names():
                raise ConfigError(
                    f"stages.{name}: translator {translator!r}은(는) 등록되지 않았다 (있는 것: {TRANSLATORS.names()})"
                )
            for source_name in getattr(settings, "sources", ()) or ():
                if source_name not in sources:
                    raise ConfigError(
                        f"stages.{name}: source {source_name!r}은(는) 정의되지 않았다 (있는 것: {sorted(sources)})"
                    )
            if hasattr(settings, "recipe"):
                _validate_recipe(name, settings.recipe)
            stages[name] = settings

        return cls(
            path=path, root=root, profile=str(raw["profile"]), language=language,
            data_root=(root / raw["data_root"]).resolve(),
            datasets_root=(root / raw.get("datasets_root", "datasets")).resolve(),
            seed=int(raw["seed"]), persona_doc=(root / raw["persona_doc"]).resolve(),
            plugins=plugins, student=StudentConfig.from_dict(dict(raw["student"])),
            teachers=teachers, sources=sources, stages=stages,
        )
