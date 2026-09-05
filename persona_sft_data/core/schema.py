"""레코드 계약.

파이프라인에는 레코드 종류가 둘이다. **세션**은 ``user``/``assistant``가 번갈아
말하는 대화 하나이고, **발화**는 외부 소스에서 가져온 사람의 문장 하나다. 단계는
자기가 내는 종류를 선언하고, 러너는 그 종류의 ``normalize``와 ``fingerprint``로
검증·중복 제거를 한다. 정규화는 출처 필드를 버리지 않는다 — 어떤 소스를 나중에
빼고 싶을 때 필터 한 줄이면 되는 것은 레코드마다 출처가 붙어 다니기 때문이다.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, Protocol

ROLES = ("user", "assistant")
_LANGUAGE = re.compile(r"^[a-z]{2}$")


class SchemaError(ValueError):
    """레코드가 계약에 맞지 않는다."""


class RecordKind(Protocol):
    """레코드 종류 하나가 아는 것: 이름, 게이트 적용 여부, 정규화, 중복 제거 키."""

    name: str
    gated: bool

    def normalize(self, record: Mapping[str, Any]) -> dict[str, Any]: ...
    def fingerprint(self, record: Mapping[str, Any]) -> str: ...


def normalize_text(value: Any) -> str:
    """NFC 정규화, 폭 없는 공백 제거, 공백 축약."""
    if not isinstance(value, str):
        raise SchemaError("텍스트는 문자열이어야 한다")
    value = unicodedata.normalize("NFC", value).replace("​", "")
    return " ".join(value.split()).strip()


def _required(record: Mapping[str, Any], key: str) -> str:
    value = str(record.get(key, "") or "").strip()
    if not value:
        raise SchemaError(f"{key}이(가) 비어 있다")
    return value


class SessionKind:
    """대화 세션. 게이트 대상이다."""

    name = "session"
    gated = True

    def normalize(self, record: Mapping[str, Any]) -> dict[str, Any]:
        session_id = _required(record, "id")
        source = _required(record, "source")
        scenario = str(record.get("scenario", "unknown") or "unknown").strip() or "unknown"
        turns_value = record.get("turns")
        if not isinstance(turns_value, list) or len(turns_value) < 2 or len(turns_value) % 2:
            raise SchemaError("turns는 2개 이상 짝수 개여야 한다")
        turns: list[dict[str, str]] = []
        for index, turn in enumerate(turns_value):
            if not isinstance(turn, Mapping):
                raise SchemaError(f"turn {index}은(는) 객체여야 한다")
            expected = ROLES[index % 2]
            if str(turn.get("role", "")) != expected:
                raise SchemaError(f"turn {index}의 role은 {expected!r}여야 한다")
            text = normalize_text(turn.get("text", ""))
            if not text:
                raise SchemaError(f"turn {index}이(가) 비어 있다")
            turns.append({"role": expected, "text": text})
        out = dict(record)
        out.update(id=session_id, source=source, scenario=scenario, turns=turns)
        return out

    def fingerprint(self, record: Mapping[str, Any]) -> str:
        turns = self.normalize(record)["turns"]
        canonical = "\n".join(f"{t['role']}:{t['text']}" for t in turns).casefold()
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class UtteranceKind:
    """외부 소스의 발화 하나. 게이트 대상이 아니다 — 사람이 쓴 문장은 페르소나 규칙에 묶이지 않는다."""

    name = "utterance"
    gated = False

    def normalize(self, record: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(record)
        out["id"] = _required(record, "id")
        out["source"] = _required(record, "source")
        out["license"] = _required(record, "license")
        text = normalize_text(record.get("text", ""))
        if not text:
            raise SchemaError("text이(가) 비어 있다")
        out["text"] = text
        language = _required(record, "language").lower()
        if not _LANGUAGE.match(language):
            raise SchemaError(f"language는 ISO 639-1 두 글자여야 한다: {language!r}")
        out["language"] = language
        return out

    def fingerprint(self, record: Mapping[str, Any]) -> str:
        canonical = normalize_text(record.get("text", "")).casefold()
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


RECORD_KINDS: dict[str, RecordKind] = {"session": SessionKind(), "utterance": UtteranceKind()}


# -- JSONL --------------------------------------------------------------------

def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """JSONL을 한 줄씩 읽는다. 빈 줄은 건너뛰고, 깨진 줄은 ``경로:줄번호``로 알린다."""
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaError(f"{path}:{number}: JSON이 아니다") from exc
            if not isinstance(value, dict):
                raise SchemaError(f"{path}:{number}: 레코드는 객체여야 한다")
            yield value


def append_jsonl(handle: Any, record: Mapping[str, Any]) -> None:
    """열려 있는 텍스트 핸들에 레코드 한 줄을 덧붙인다. 한글은 이스케이프하지 않는다."""
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    """레코드들을 JSONL로 쓰고 쓴 개수를 돌려준다. 부모 디렉터리는 만들어 준다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            append_jsonl(handle, record)
            count += 1
    return count
