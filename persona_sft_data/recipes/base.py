"""레시피가 받는 것: 내보낸 데이터셋에 대한 사실."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from persona_sft_data.core.config import StudentConfig


@dataclass(frozen=True)
class LengthReport:
    """렌더링된 학습 텍스트의 길이 분포. ``method``가 토큰인지 글자인지 말한다."""

    method: str
    count: int
    p50: int
    p95: int
    p99: int
    max: int
    cutoff_len: int

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class ExportInfo:
    """export 단계가 레시피에 건네는 전부. 레시피는 이것만 보고 파일을 쓴다."""

    name: str
    out_dir: Path
    root: Path
    files: dict[str, dict[str, Any]]
    student: StudentConfig
    system_prompt: str
    chat_template_name: str
    length_report: LengthReport
    persona_name: str
    profile: str
    seed: int
