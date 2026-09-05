"""프로필 = 용도별 기본값 묶음. 코드는 프로필을 구분하지 않고 속성만 읽는다."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    assistant_label: str
    user_label: str
    writer_framing: str
    required_sections: tuple[str, ...]
    default_flows: tuple[str, ...]
    default_turns: tuple[int, ...]
    extra_rules: tuple[str, ...]
    default_constraints: tuple[tuple[str, str], ...]
    identity_hint: str
    relationship_hint: str
    register_hint: str
    background_hint: str
    situations_hint: tuple[str, ...]

    def document_template(self, persona_name: str) -> str:
        """``init``이 쓰는 문서 골격. 파서를 통과하는 완전한 문서다."""
        constraints = "".join(f"| {k} | {v} |\n" for k, v in self.default_constraints)
        situations = "".join(f"{i}. {s}\n" for i, s in enumerate(self.situations_hint, 1))
        flows = "".join(f"- {f}\n" for f in self.default_flows)
        background = f"\n## 배경\n\n{self.background_hint}\n" if "배경" in self.required_sections else ""
        return f"""# {persona_name} 페르소나 정의

- 프로필: {self.name}

## 핵심 정의

| 항목 | 규칙 |
| --- | --- |
| 이름 | {persona_name} |
| 정체성 | {self.identity_hint} |
| 사용자와의 관계 | {self.relationship_hint} |
| 말투 | {self.register_hint} |
| 성격 | 여기에 성격을 두세 문장으로 적는다. |
| 응답 길이 | {dict(self.default_constraints).get('발화 길이', '1~3문장')} |
| 지식 범위 | 캐릭터가 아는 것과 모르는 것을 적는다. |

## 제약

| 규칙 | 값 |
| --- | --- |
{constraints}
## 발화 원칙

1. 먼저 지금의 감정이나 의도를 말하고, 필요하면 짧은 제안을 덧붙인다.
2. 모르는 것은 꾸며내지 않고 모른다고 말한 뒤 캐릭터의 화제로 돌아온다.
3. 문법적으로 완결된 문장을 쓴다.
{background}
## 하지 않는 말과 행동

- 자신을 AI, 인공지능, 언어모델, 챗봇, 프로그램 또는 컴퓨터라고 말하지 않는다.
- 캐릭터가 알 수 없는 사실을 아는 척하지 않는다.

## 다룰 상황

{situations}
## 대화 흐름

{flows}
## 예시 대화

```text
U: 안녕, 누구야?
A: 나는 {persona_name}. 반가워.
```
"""
