"""여덟 확장점의 인터페이스.

플러그인은 상속하지 않고 모양만 맞추면 된다(``typing.Protocol``). 여기 적힌
시그니처가 정본이고, 내장 구현도 이 모양을 따른다.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from persona_sft_data.core.config import PipelineConfig, TeacherConfig
    from persona_sft_data.core.gates import GateSettings, Verdict
    from persona_sft_data.core.persona import Persona
    from persona_sft_data.core.runner import StageContext, StageStats
    from persona_sft_data.recipes.base import ExportInfo
    from persona_sft_data.teacher.base import Request, Result


# -- 단계 ---------------------------------------------------------------------

@runtime_checkable
class Stage(Protocol):
    """레코드를 내는 단계(``mode="records"``)와 파일을 직접 쓰는 단계(``"artifact"``)."""

    name: str
    config_name: str                       # 설정 ``stages``에서 이 단계를 찾는 키
    mode: Literal["records", "artifact"]
    record_kind: Literal["session", "utterance"] | None
    produces: Literal["raw", "filtered", "final"] | None
    settings_type: type

    def requires(self, config: PipelineConfig) -> tuple[str, ...]: ...
    def instances(self, config: PipelineConfig) -> list[Stage]: ...
    def preflight(self, ctx: StageContext) -> None: ...
    def run(self, ctx: StageContext) -> Iterator[dict[str, Any]] | StageStats: ...


# -- 소스 ---------------------------------------------------------------------

class Format(Protocol):
    name: str
    extensions: tuple[str, ...]            # 캐시 파일 이름에 쓸 확장자, 첫 것을 쓴다

    def rows(self, data: bytes, fields: Sequence[str]) -> Iterator[dict[str, Any]]: ...


class Extractor(Protocol):
    name: str
    settings_type: type

    def extract(self, row: Mapping[str, Any], fields: Sequence[str], settings: Any) -> Iterator[str]: ...


# -- 교사 ---------------------------------------------------------------------

class Teacher(Protocol):
    name: str

    def check(self) -> None: ...
    def generate(self, requests: Sequence[Request]) -> list[Result]: ...


class TeacherFactory(Protocol):
    name: str

    def build(self, cfg: TeacherConfig) -> Teacher: ...


# -- 번역기 -------------------------------------------------------------------

class Translator(Protocol):
    name: str

    def translate(self, texts: Sequence[str], source_language: str) -> list[str | None]: ...


class TranslatorFactory(Protocol):
    name: str

    def build(self, ctx: StageContext, teacher: Teacher) -> Translator: ...


# -- 레시피 -------------------------------------------------------------------

class Recipe(Protocol):
    name: str
    settings_type: type

    def write(self, out_dir: Path, info: ExportInfo, settings: Any) -> list[Path]: ...


# -- 프로필 -------------------------------------------------------------------

class Profile(Protocol):
    name: str
    assistant_label: str
    user_label: str
    writer_framing: str
    required_sections: tuple[str, ...]
    default_flows: tuple[str, ...]
    default_turns: tuple[int, ...]
    extra_rules: tuple[str, ...]

    def document_template(self, persona_name: str) -> str: ...


# -- 규칙 ---------------------------------------------------------------------

class Rule(Protocol):
    name: str

    def check(self, turns: Sequence[Mapping[str, str]], verdict: Verdict) -> None: ...


class RuleFactory(Protocol):
    name: str
    constraint_key: str                    # 페르소나 문서 ``## 제약`` 표의 규칙 키

    def build(self, persona: Persona, value: str, settings: GateSettings) -> Rule: ...
