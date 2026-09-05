"""레코드 계약: 세션은 user/assistant 교대, 발화는 출처 필드 필수."""
import unicodedata

import pytest

from persona_sft_data.core.schema import (
    RECORD_KINDS, SchemaError, normalize_text, read_jsonl, write_jsonl,
)

SESSION = RECORD_KINDS["session"]
UTTERANCE = RECORD_KINDS["utterance"]


def _session(turns, **extra):
    return {"id": "s1", "source": "dialogue", "turns": turns, **extra}


def test_session_normalizes_nfc_and_whitespace_and_keeps_provenance():
    decomposed = unicodedata.normalize("NFD", "안녕")
    out = SESSION.normalize(_session(
        [{"role": "user", "text": f"  {decomposed}​  "}, {"role": "assistant", "text": "응,  안녕!"}],
        license="synthetic", generator=["m"],
    ))
    assert out["turns"] == [{"role": "user", "text": "안녕"}, {"role": "assistant", "text": "응, 안녕!"}]
    assert out["license"] == "synthetic" and out["generator"] == ["m"]
    assert out["scenario"] == "unknown"


@pytest.mark.parametrize("turns, message", [
    ([], "짝수"),
    ([{"role": "user", "text": "야"}], "짝수"),
    ([{"role": "assistant", "text": "응"}, {"role": "user", "text": "야"}], "user"),
    ([{"role": "user", "text": "야"}, {"role": "user", "text": "야"}], "assistant"),
    ([{"role": "user", "text": "  "}, {"role": "assistant", "text": "응"}], "비어"),
    ([{"role": "user", "text": "야"}, {"role": "pet", "text": "응"}], "assistant"),
])
def test_session_rejects_bad_turns(turns, message):
    with pytest.raises(SchemaError, match=message):
        SESSION.normalize(_session(turns))


@pytest.mark.parametrize("missing", ["id", "source"])
def test_session_requires_id_and_source(missing):
    record = _session([{"role": "user", "text": "야"}, {"role": "assistant", "text": "응"}])
    record[missing] = ""
    with pytest.raises(SchemaError, match=missing):
        SESSION.normalize(record)


def test_session_fingerprint_ignores_spacing_and_case_but_not_words():
    a = _session([{"role": "user", "text": "같이 놀자"}, {"role": "assistant", "text": "응 좋아"}])
    b = _session([{"role": "user", "text": "같이   놀자"}, {"role": "assistant", "text": "응 좋아"}])
    c = _session([{"role": "user", "text": "같이 놀자"}, {"role": "assistant", "text": "응 싫어"}])
    assert SESSION.fingerprint(a) == SESSION.fingerprint(b) != SESSION.fingerprint(c)
    assert SESSION.gated is True


def test_utterance_requires_text_source_language_license():
    out = UTTERANCE.normalize({"id": "u1", "text": " 밥  먹었어? ", "source": "s", "language": "KO", "license": "mit", "url": "x"})
    assert out["text"] == "밥 먹었어?" and out["language"] == "ko" and out["url"] == "x"
    assert UTTERANCE.gated is False
    for key in ("text", "source", "language", "license"):
        broken = {"id": "u1", "text": "야", "source": "s", "language": "ko", "license": "mit"}
        broken[key] = ""
        with pytest.raises(SchemaError, match=key):
            UTTERANCE.normalize(broken)
    with pytest.raises(SchemaError, match="language"):
        UTTERANCE.normalize({"id": "u1", "text": "야", "source": "s", "language": "korean", "license": "mit"})


def test_utterance_fingerprint_is_the_normalized_text():
    a = {"id": "1", "text": "밥 먹었어?", "source": "s", "language": "ko", "license": "mit"}
    b = {**a, "id": "2", "text": "밥  먹었어?"}
    assert UTTERANCE.fingerprint(a) == UTTERANCE.fingerprint(b)


def test_jsonl_roundtrip_and_line_numbered_errors(tmp_path):
    path = tmp_path / "x.jsonl"
    assert write_jsonl(path, [{"a": 1}, {"b": "한글"}]) == 2
    assert list(read_jsonl(path)) == [{"a": 1}, {"b": "한글"}]
    path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(SchemaError, match=r"x\.jsonl:2"):
        list(read_jsonl(path))


def test_normalize_text_rejects_non_strings():
    with pytest.raises(SchemaError):
        normalize_text(3)
