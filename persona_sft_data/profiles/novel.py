"""소설 등장인물 프로필. 인물의 목소리로 독자와 대화한다."""

from persona_sft_data.core.registry import PROFILES
from persona_sft_data.profiles.base import ProfileSpec

PROFILE = ProfileSpec(
    name="novel",
    assistant_label="화자",
    user_label="독자",
    writer_framing="너는 소설가다. 아래 등장인물의 목소리로 독자와 나누는 대화를 쓴다.",
    required_sections=("배경",),
    default_flows=("독자가 과거를 묻는 흐름", "독자가 갈등에 대해 묻는 흐름", "독자가 다른 인물에 대해 묻는 흐름",
                   "독자가 일상을 묻는 흐름", "인물이 고백하듯 말하는 흐름"),
    default_turns=(2, 3),
    extra_rules=("인물의 시점과 시대를 벗어나지 않는다.",),
    default_constraints=(("말투", "자유"), ("발화 길이", "1~4문장"), ("문자", "한글"), ("이모지", "금지"),
                         ("마크다운", "금지"), ("역할 표기", "금지"), ("AI 자칭", "금지"), ("반복", "금지")),
    identity_hint="소설 속 인물 (이름, 나이, 처지)",
    relationship_hint="독자와 대화하는 화자",
    register_hint="인물의 성격에 맞는 말투",
    background_hint="작품의 배경, 인물의 과거, 관계, 갈등을 적는다.",
    situations_hint=("회상, 후회, 다짐", "갈등, 대립, 화해", "일상, 취향, 습관", "다른 인물에 대한 생각", "독자의 질문에 답하기"),
)
PROFILES.add("novel", PROFILE, origin="builtin", path=f"{__name__}:PROFILE")
