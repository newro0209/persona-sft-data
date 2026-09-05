"""채팅 템플릿. base 모델에는 템플릿이 없으므로 이 프로젝트가 정한다.

jinja 텍스트는 트레이너용이고, 파이썬 렌더러는 표본 파일과 길이 측정용이다. 둘은
같은 바이트를 내야 하며 테스트가 그것을 확인한다. jinja의 generation 마커는 TRL의
assistant_only_loss가 마스크를 만들 때 쓴다.

**개행은 블록 태그 뒤가 아니라 앞에 둔다.** 트레이너가 어떤 jinja 환경으로 이
텍스트를 컴파일할지 우리가 정하지 못한다. ``trim_blocks=True``면 블록 태그 바로
뒤의 개행이 사라지므로 ``<|im_end|>{% endgeneration %}\\n``은 환경에 따라 다른
바이트를 낸다 — assistant 턴 뒤 개행이 없어져 ``<|im_end|>``와 다음
``<|im_start|>``가 붙는다. 개행을 ``{% endgeneration %}`` 앞의 리터럴로 옮기고 뒤에
남는 공백은 ``{%- else`` 로 걷어 내 기본 환경과 ``trim_blocks``/``lstrip_blocks``
환경이 같은 바이트를 내게 한다. 테스트가 두 환경 모두를 대조한다.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

CHATML_JINJA = (
    "{%- for message in messages -%}\n"
    "{%- if message['role'] == 'assistant' -%}\n"
    "<|im_start|>assistant\n"
    "{% generation %}{{ message['content'] }}<|im_end|>\n{% endgeneration %}\n"
    "{%- else -%}\n"
    "<|im_start|>{{ message['role'] }}\n"
    "{{ message['content'] }}<|im_end|>\n"
    "{% endif -%}\n"
    "{%- endfor -%}\n"
    "{%- if add_generation_prompt -%}\n"
    "<|im_start|>assistant\n"
    "{% endif -%}\n"
)


def render_chatml(messages: Sequence[dict[str, str]], *, add_generation_prompt: bool = False) -> str:
    """``CHATML_JINJA``와 같은 바이트를 내는 파이썬 렌더러."""
    text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    if add_generation_prompt:
        text += "<|im_start|>assistant\n"
    return text


# 템플릿 이름 → (jinja 텍스트, 파이썬 렌더러)
CHAT_TEMPLATES: dict[str, tuple[str, Callable[..., str]]] = {"chatml": (CHATML_JINJA, render_chatml)}


def jinja_for(name: str) -> str:
    return CHAT_TEMPLATES[name][0]


def renderer_for(name: str) -> Callable[..., str]:
    return CHAT_TEMPLATES[name][1]
