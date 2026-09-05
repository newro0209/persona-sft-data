"""Canonical JSONL schema shared by all P1 data sources."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

USER_TAG = "<|u|>"
PET_TAG = "<|p|>"
ROLE_TAGS = {"user": USER_TAG, "pet": PET_TAG}


class SessionError(ValueError):
    """Raised when a generated record does not match the corpus contract."""


class RecordContract(Protocol):
    """Small interface implemented by records exchanged between P1 stages."""

    name: str

    def normalize(self, record: Mapping[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class FunctionRecordContract:
    """Adapt a normalizer function to the :class:`RecordContract` interface."""

    name: str
    normalizer: Callable[[Mapping[str, Any]], dict[str, Any]]

    def normalize(self, record: Mapping[str, Any]) -> dict[str, Any]:
        return self.normalizer(record)


def normalize_text(value: str) -> str:
    """Apply the corpus-wide NFC and whitespace contract to one utterance."""

    if not isinstance(value, str):
        raise SessionError("turn text must be a string")
    value = unicodedata.normalize("NFC", value)
    value = " ".join(value.replace("\u200b", "").split())
    return value.strip()


def normalize_session(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a session without discarding provenance fields."""

    session_id = str(record.get("id", "")).strip()
    source = str(record.get("source", "")).strip()
    scenario = str(record.get("scenario", "unknown")).strip() or "unknown"
    turns_value = record.get("turns")
    if not session_id:
        raise SessionError("session id is required")
    if not source:
        raise SessionError("session source is required")
    if not isinstance(turns_value, list):
        raise SessionError("turns must be a list")
    if len(turns_value) < 2 or len(turns_value) % 2:
        raise SessionError("a session needs an even number of turns")

    turns: list[dict[str, str]] = []
    for index, turn in enumerate(turns_value):
        if not isinstance(turn, Mapping):
            raise SessionError(f"turn {index} must be an object")
        expected_role = "user" if index % 2 == 0 else "pet"
        role = str(turn.get("role", ""))
        if role != expected_role:
            raise SessionError(f"turn {index} must have role {expected_role!r}")
        text = normalize_text(turn.get("text", ""))
        if not text:
            raise SessionError(f"turn {index} is empty")
        if USER_TAG in text or PET_TAG in text:
            raise SessionError(f"turn {index} contains a reserved role tag")
        turns.append({"role": role, "text": text})

    normalized = dict(record)
    normalized.update(id=session_id, source=source, scenario=scenario, turns=turns)
    return normalized


PAIR_TEXT_FIELDS = ("user", "pet", "reference_user", "reference_pet")


def normalize_exchange_pair(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the stable exchange-pair fields shared by generators and gates.

    Scenario, stage, reference index, model and review provenance remain extensible
    metadata.  Downstream stages that require one of those fields validate it at
    their own boundary.
    """

    pair_id_value = record.get("id")
    if not isinstance(pair_id_value, str) or not pair_id_value.strip():
        raise SessionError("exchange pair id must be a nonempty string")
    pair_id = pair_id_value.strip()
    missing = [field for field in PAIR_TEXT_FIELDS if field not in record]
    if missing:
        raise SessionError(f"exchange pair is missing fields: {missing}")
    normalized = dict(record)
    normalized["id"] = pair_id
    for field in PAIR_TEXT_FIELDS:
        text = normalize_text(record[field])
        if not text:
            raise SessionError(f"exchange pair field {field!r} is empty")
        if USER_TAG in text or PET_TAG in text:
            raise SessionError(f"exchange pair field {field!r} contains a reserved role tag")
        normalized[field] = text
    return normalized


def serialize_turns(turns: Iterable[Mapping[str, str]]) -> str:
    """Serialize a dialogue as the exact role-tagged training stream."""

    parts: list[str] = []
    for turn in turns:
        role = str(turn["role"])
        try:
            tag = ROLE_TAGS[role]
        except KeyError as error:
            raise SessionError(f"unknown role: {role!r}") from error
        parts.extend((tag, normalize_text(turn["text"])))
    return "".join(parts)


def session_text(record: Mapping[str, Any]) -> str:
    return serialize_turns(normalize_session(record)["turns"])


def content_fingerprint(record: Mapping[str, Any]) -> str:
    canonical = session_text(record).casefold().encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def read_jsonl(
    path: Path, *, contract: RecordContract | None = None
) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise SessionError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, dict):
                raise SessionError(f"{path}:{line_number}: record must be an object")
            if contract is not None:
                try:
                    value = contract.normalize(value)
                except (KeyError, TypeError, ValueError) as error:
                    raise SessionError(
                        f"{path}:{line_number}: {contract.name} contract: {error}"
                    ) from error
            yield value


def append_jsonl(
    handle: Any,
    record: Mapping[str, Any],
    *,
    contract: RecordContract | None = None,
) -> None:
    if contract is not None:
        record = contract.normalize(record)
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_jsonl(
    path: Path,
    records: Iterable[Mapping[str, Any]],
    *,
    contract: RecordContract | None = None,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            append_jsonl(handle, record, contract=contract)
            count += 1
    return count


@dataclass(frozen=True)
class JsonlStageIO:
    """Typed JSONL boundary for one pipeline record kind.

    Existing ``read_jsonl``/``write_jsonl`` callers remain valid; new and
    refactored stages can bind a contract once instead of validating ad hoc.
    """

    contract: RecordContract | None = None

    def read(self, path: Path) -> Iterator[dict[str, Any]]:
        return read_jsonl(path, contract=self.contract)

    def read_all(self, path: Path) -> list[dict[str, Any]]:
        return list(self.read(path))

    def append(self, handle: Any, record: Mapping[str, Any]) -> None:
        append_jsonl(handle, record, contract=self.contract)

    def write(self, path: Path, records: Iterable[Mapping[str, Any]]) -> int:
        return write_jsonl(path, records, contract=self.contract)


SESSION_CONTRACT = FunctionRecordContract("session/v1", normalize_session)
EXCHANGE_PAIR_CONTRACT = FunctionRecordContract(
    "exchange-pair/v1", normalize_exchange_pair
)
RAW_JSONL = JsonlStageIO()
SESSION_JSONL = JsonlStageIO(SESSION_CONTRACT)
EXCHANGE_PAIR_JSONL = JsonlStageIO(EXCHANGE_PAIR_CONTRACT)
