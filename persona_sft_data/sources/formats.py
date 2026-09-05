"""포맷 어댑터. 각각 바이트를 받아 선택한 열만 가진 행을 낸다.

선택한 열만 실체화하는 것이 중요하다. 한 공개 데이터셋의 답변 열은 AI 어시스턴트
문장("저는 인공지능 챗봇이기 때문에...")이고, 그것이 코퍼스에 들어가면 페르소나가
금지한 바로 그 말을 가르친다. 설정의 ``fields``가 그 경계다.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterator, Sequence
from typing import Any

from persona_sft_data.core.registry import FORMATS


def _project(row: dict[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    """행에서 ``fields``에 있는 열만 남긴다."""
    return {f: row.get(f) for f in fields if f in row}


class _Delimited:
    """헤더가 있는 구분자 파일. 구분자만 하위 클래스가 정한다."""

    delimiter = "\t"

    def rows(self, data: bytes, fields: Sequence[str]) -> Iterator[dict[str, Any]]:
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig", errors="replace")), delimiter=self.delimiter)
        for row in reader:
            yield _project(row, fields)


@FORMATS.register("tsv", origin="builtin")
class TsvFormat(_Delimited):
    name = "tsv"
    extensions = (".tsv",)
    delimiter = "\t"


@FORMATS.register("csv", origin="builtin")
class CsvFormat(_Delimited):
    name = "csv"
    extensions = (".csv",)
    delimiter = ","


@FORMATS.register("jsonl", origin="builtin")
class JsonlFormat:
    """한 줄이 객체 하나. 객체가 아닌 줄은 버린다."""

    name = "jsonl"
    extensions = (".jsonl",)

    def rows(self, data: bytes, fields: Sequence[str]) -> Iterator[dict[str, Any]]:
        for line in data.decode("utf-8", errors="replace").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield _project(value, fields)


@FORMATS.register("json", origin="builtin")
class JsonFormat:
    """배열, 또는 배열을 값으로 가진 객체(첫 배열 값)."""

    name = "json"
    extensions = (".json",)

    def rows(self, data: bytes, fields: Sequence[str]) -> Iterator[dict[str, Any]]:
        value = json.loads(data.decode("utf-8", errors="replace"))
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, list)), [])
        for item in value or []:
            if isinstance(item, dict):
                yield _project(item, fields)


@FORMATS.register("parquet", origin="builtin")
class ParquetFormat:
    """pyarrow는 여기서만, 지연 import. 없으면 이 소스는 "읽을 수 없음"이다."""

    name = "parquet"
    extensions = (".parquet",)

    def rows(self, data: bytes, fields: Sequence[str]) -> Iterator[dict[str, Any]]:
        import pyarrow.parquet as pq  # noqa: PLC0415 - 선택 의존성

        yield from pq.read_table(io.BytesIO(data), columns=list(fields)).to_pylist()


@FORMATS.register("text", origin="builtin")
class TextFormat:
    """줄 하나가 행 하나. 열 이름은 ``fields[0]``."""

    name = "text"
    extensions = (".txt",)

    def rows(self, data: bytes, fields: Sequence[str]) -> Iterator[dict[str, Any]]:
        key = fields[0]
        for line in data.decode("utf-8-sig", errors="replace").splitlines():
            if line.strip():
                yield {key: line.strip()}
