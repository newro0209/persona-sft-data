"""채팅 템플릿. base 모델에는 템플릿이 없으므로 이 프로젝트가 정한다.

jinja 텍스트는 트레이너용이고, 파이썬 렌더러는 표본 파일과 길이 측정용이다. 둘은
같은 바이트를 내야 하며 테스트가 그것을 확인한다. jinja의 generation 마커는 TRL의
assistant_only_loss가 마스크를 만들 때 쓴다.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

CHATML_JINJA = (
    "{%- for message in messages -%}\n"
    "{%- if message['role'] == 'assistant' -%}\n"
    "<|im_start|>assistant\n"
    "{% generation %}{{ message['content'] }}<|im_end|>{% endgeneration %}\n"
    "{% else -%}\n"
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
