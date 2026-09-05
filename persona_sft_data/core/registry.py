"""플러그인 레지스트리.

여덟 개 확장점(stages·formats·extractors·teachers·translators·recipes·profiles·rules)이
전부 ``Registry`` 인스턴스 하나씩이다. 등록 경로는 셋이고, 같은 이름이면 우선순위가
높은 쪽이 남는다.

1. ``plugins`` — 설정의 ``plugins`` 목록으로 import 된 로컬 모듈의 데코레이터
2. ``entry_point`` — 설치된 패키지의 ``persona_sft_data.<그룹>`` entry point
3. ``builtin`` — 이 패키지 자신 (``core/builtins.py``가 지연 import)

내장도 entry point로 선언돼 있지만, 같은 객체가 두 경로로 오면 먼저 온 출처를
유지한다. 그래서 ``plugins`` 명령의 표에서 내장은 ``builtin``으로 보인다.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Generic, TypeVar

T = TypeVar("T")

# 낮을수록 우선한다.
ORIGIN_RANK = {"plugins": 0, "entry_point": 1, "builtin": 2}


class PluginError(RuntimeError):
    """플러그인을 찾지 못했거나 불러오지 못했다."""


@dataclass(frozen=True)
class Registration(Generic[T]):
    """등록된 객체 하나와 그것이 어디서 왔는지."""

    name: str
    obj: T
    origin: str
    path: str


class Registry(Generic[T]):
    """이름 → 객체. 조회 시점에 내장과 entry point를 지연 발견한다."""

    def __init__(self, group: str) -> None:
        self.group = group
        self._items: dict[str, Registration[T]] = {}
        self._discovered = False

    # -- 등록 ---------------------------------------------------------------

    def register(self, name: str, *, origin: str = "plugins") -> Callable[[T], T]:
        """클래스에 붙이는 데코레이터. 클래스는 인자 없이 인스턴스화해 등록한다.

        레지스트리는 바로 쓸 수 있는 인스턴스를 든다 — 플러그인은 상태 없는 객체다.
        생성 인자가 필요한 것은 ``add()``에 인스턴스를 직접 넘긴다.
        """

        def decorate(obj: T) -> T:
            self.add(name, obj, origin=origin)
            return obj

        return decorate

    def add(self, name: str, obj: Any, *, origin: str) -> None:
        if origin not in ORIGIN_RANK:
            raise PluginError(
                f"{self.group}: 알 수 없는 출처 {origin!r} (허용: {sorted(ORIGIN_RANK)})"
            )
        current = self._items.get(name)
        if current is not None:
            same = current.obj is obj or (isinstance(obj, type) and type(current.obj) is obj)
            if same:
                return  # 같은 플러그인이 두 경로로 왔다. 먼저 온 출처를 유지한다.
            if ORIGIN_RANK[origin] > ORIGIN_RANK[current.origin]:
                return  # 우선순위가 낮은 쪽은 덮어쓰지 못한다.
        instance = obj() if isinstance(obj, type) else obj
        cls = obj if isinstance(obj, type) else type(obj)
        # 경로는 ``모듈:클래스`` 꼴이다. ``__qualname__``은 함수 안에서 정의한 클래스에
        # ``<locals>``를 끼워 넣으므로 ``__name__``을 쓴다.
        self._items[name] = Registration(name, instance, origin, f"{cls.__module__}:{cls.__name__}")

    # -- 발견 ---------------------------------------------------------------

    def _discover(self) -> None:
        if self._discovered:
            return
        self._discovered = True
        from persona_sft_data.core import builtins  # 순환 import를 피하려고 여기서

        builtins.load()
        for ep in metadata.entry_points(group=self.group):
            try:
                obj = ep.load()
            except Exception as exc:  # noqa: BLE001 - 어떤 실패든 같은 안내
                raise PluginError(
                    f"{self.group}: entry point {ep.name!r} ({ep.value}) 로드 실패: {exc}"
                ) from exc
            self.add(ep.name, obj, origin="entry_point")

    # -- 조회 ---------------------------------------------------------------

    def get(self, name: str) -> T:
        self._discover()
        try:
            return self._items[name].obj
        except KeyError:
            raise PluginError(
                f"{self.group}: {name!r}은(는) 등록되지 않았다. 등록된 이름: {self.names()}"
            ) from None

    def names(self) -> list[str]:
        self._discover()
        return sorted(self._items)

    def items(self) -> dict[str, T]:
        self._discover()
        return {name: r.obj for name, r in sorted(self._items.items())}

    def describe(self) -> list[Registration[T]]:
        self._discover()
        return [self._items[name] for name in sorted(self._items)]


def load_plugins(modules: Iterable[str]) -> list[str]:
    """설정의 ``plugins`` 목록을 import 한다. 모듈은 import 되면서 자신을 등록한다."""
    loaded: list[str] = []
    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError as exc:
            raise PluginError(f"plugins 모듈 {module!r}을(를) import 하지 못했다: {exc}") from exc
        loaded.append(module)
    return loaded


STAGES: Registry = Registry("persona_sft_data.stages")
FORMATS: Registry = Registry("persona_sft_data.formats")
EXTRACTORS: Registry = Registry("persona_sft_data.extractors")
TEACHERS: Registry = Registry("persona_sft_data.teachers")
TRANSLATORS: Registry = Registry("persona_sft_data.translators")
RECIPES: Registry = Registry("persona_sft_data.recipes")
PROFILES: Registry = Registry("persona_sft_data.profiles")
RULES: Registry = Registry("persona_sft_data.rules")

GROUPS: dict[str, Registry] = {
    "stages": STAGES,
    "formats": FORMATS,
    "extractors": EXTRACTORS,
    "teachers": TEACHERS,
    "translators": TRANSLATORS,
    "recipes": RECIPES,
    "profiles": PROFILES,
    "rules": RULES,
}
