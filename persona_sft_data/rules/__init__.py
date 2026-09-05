"""내장 규칙 플러그인. import 되면서 각자 RULES에 등록한다."""

from persona_sft_data.rules import (  # noqa: F401
    ai_claim, ellipsis, emoji, length, markdown, name_suffix, register, repeat,
    role_label, script, third_person,
)
