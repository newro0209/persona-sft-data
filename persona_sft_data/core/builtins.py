"""내장 플러그인 모듈 목록.

레지스트리가 처음 조회될 때 여기 적힌 모듈을 import 한다. 각 모듈은 import 되면서
``@<REGISTRY>.register("...", origin="builtin")`` 데코레이터로 자신을 등록한다.
새 내장 플러그인을 추가하면 이 목록에도 적는다. 목록이 없으면 pip로 설치하지 않은
소스 트리에서 내장이 발견되지 않는다 — entry point는 설치된 패키지에만 있다.
"""

from __future__ import annotations

import importlib

BUILTIN_MODULES: tuple[str, ...] = (
    "persona_sft_data.rules",
    "persona_sft_data.teacher.openai_compat",
    "persona_sft_data.teacher.fake",
    "persona_sft_data.profiles",
    "persona_sft_data.sources.formats",
    "persona_sft_data.sources.extractors",
    "persona_sft_data.sources.translate",
)

_loaded = False


def load() -> None:
    """내장 모듈을 한 번만 import 한다."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    for module in BUILTIN_MODULES:
        importlib.import_module(module)
