import unicodedata

import pytest

from persona_sft_data.schema import (
    EXCHANGE_PAIR_JSONL,
    SESSION_JSONL,
    SessionError,
    normalize_session,
    serialize_turns,
)


def test_normalize_session_enforces_nfc_and_alternating_roles():
    decomposed = unicodedata.normalize("NFD", "안녕")
    session = normalize_session({
        "id": "x",
        "source": "test",
        "turns": [
            {"role": "user", "text": f"  {decomposed}  "},
            {"role": "pet", "text": "응, 안녕!"},
        ],
    })
    assert session["turns"][0]["text"] == "안녕"
    assert serialize_turns(session["turns"]) == "<|u|>안녕<|p|>응, 안녕!"


def test_normalize_session_rejects_role_order_errors():
    with pytest.raises(SessionError, match="role 'user'"):
        normalize_session({
            "id": "x",
            "source": "test",
            "turns": [
                {"role": "pet", "text": "안녕"},
                {"role": "user", "text": "안녕"},
            ],
        })


def test_jsonl_stage_io_enforces_and_normalizes_its_record_contract(tmp_path):
    pair_path = tmp_path / "pairs.jsonl"
    count = EXCHANGE_PAIR_JSONL.write(
        pair_path,
        [
            {
                "id": " pair-1 ",
                "user": "  같이  놀자 ",
                "pet": "응, 좋아!",
                "reference_user": "놀자",
                "reference_pet": "그래!",
                "generator": "test",
            }
        ],
    )

    assert count == 1
    record = EXCHANGE_PAIR_JSONL.read_all(pair_path)[0]
    assert record["id"] == "pair-1"
    assert record["user"] == "같이 놀자"
    assert record["generator"] == "test"

    invalid_path = tmp_path / "invalid.jsonl"
    invalid_path.write_text('{"id":"bad"}\n', encoding="utf-8")
    with pytest.raises(SessionError, match="exchange-pair/v1 contract"):
        EXCHANGE_PAIR_JSONL.read_all(invalid_path)


def test_session_stage_io_preserves_provenance(tmp_path):
    path = tmp_path / "sessions.jsonl"
    SESSION_JSONL.write(
        path,
        [
            {
                "id": "session-1",
                "source": "test",
                "provenance": {"model": "mock"},
                "turns": [
                    {"role": "user", "text": "안녕"},
                    {"role": "pet", "text": "응, 안녕!"},
                ],
            }
        ],
    )
    assert SESSION_JSONL.read_all(path)[0]["provenance"] == {"model": "mock"}
