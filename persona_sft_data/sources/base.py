"""소스 하나를 바이트로 가져와 발화 문자열로 바꾸는 경로.

가져오기(url은 캐시, path는 그대로) → 포맷 어댑터가 행으로 → 추출기가 발화로. 어느
단계가 실패해도 예외 대신 로그와 ``None``이다: 소스 하나가 죽어서 실행 전체가 죽지
않는다. 무엇이 얼마나 기여했는지는 어차피 통계에 남는다.
"""

from __future__ import annotations

import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from persona_sft_data.core.config import SourceConfig, build_settings
from persona_sft_data.core.registry import EXTRACTORS, FORMATS


@dataclass(frozen=True)
class Utterance:
    """사람이 쓴 한 줄과 그 레코드가 실어야 할 출처."""

    text: str
    source: str
    language: str
    license: str
    url: str | None = None
    original_text: str | None = None
    original_language: str | None = None


def _fetch(url: str, timeout: float) -> bytes:
    """이 모듈의 유일한 네트워크 호출. 테스트가 바꿔 끼운다."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def fetch_source(cfg: SourceConfig, cache_dir: Path, *, timeout: float,
                 log: Callable[[str], None]) -> bytes | None:
    """path면 읽고, url이면 ``cache_dir/<이름><확장자>``에 한 번 받아 둔다."""
    if cfg.path is not None:
        if not cfg.path.exists():
            log(f"[source {cfg.name}] 파일이 없다: {cfg.path}. 이 소스는 건너뛴다.")
            return None
        return cfg.path.read_bytes()
    ext = FORMATS.get(cfg.format).extensions[0]
    cache = cache_dir / f"{cfg.name}{ext}"
    if cache.exists():
        return cache.read_bytes()
    try:
        data = _fetch(cfg.url or "", timeout)
    except Exception as exc:  # noqa: BLE001 - 어떤 실패든 같은 처리
        log(f"[source {cfg.name}] 다운로드 실패 {cfg.url}: {type(exc).__name__}: {exc}. "
            f"이 소스는 건너뛴다. 파일을 {cache}에 두면 쓴다.")
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(data)
    log(f"[source {cfg.name}] {len(data):,} bytes -> {cache}")
    return data


def read_utterances(cfg: SourceConfig, data: bytes) -> Iterator[str]:
    """포맷 어댑터 → 행, 추출기 → 발화. 추출 설정은 여기서 검증한다."""
    fmt = FORMATS.get(cfg.format)
    extractor = EXTRACTORS.get(cfg.extract_kind)
    settings = build_settings(extractor.settings_type, dict(cfg.extract), f"source {cfg.name!r} extract")
    for row in fmt.rows(data, cfg.fields):
        yield from extractor.extract(row, cfg.fields, settings)
