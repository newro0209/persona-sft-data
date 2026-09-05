"""게임 NPC 프로필. 세계관(배경)이 필수이고, 상대는 플레이어다."""

from persona_sft_data.core.registry import PROFILES
from persona_sft_data.profiles.base import ProfileSpec

PROFILE = ProfileSpec(
    name="npc",
    assistant_label="NPC",
    user_label="플레이어",
    writer_framing="너는 게임 시나리오 작가다. 아래 세계관 속 NPC와 플레이어가 주고받는 대화를 쓴다.",
    required_sections=("배경",),
    default_flows=("플레이어가 처음 말을 거는 흐름", "플레이어가 퀘스트를 묻는 흐름", "플레이어가 거래를 시도하는 흐름",
                   "플레이어가 세계에 대해 묻는 흐름", "플레이어가 적대적으로 구는 흐름", "플레이어가 다시 찾아온 흐름"),
    default_turns=(2, 3, 4),
    extra_rules=("배경에 적힌 세계 밖의 지식이나 현실 세계를 언급하지 않는다.",),
    default_constraints=(("말투", "존댓말"), ("발화 길이", "1~3문장"), ("문자", "한글"), ("이모지", "금지"),
                         ("마크다운", "금지"), ("역할 표기", "금지"), ("AI 자칭", "금지"), ("반복", "금지")),
    identity_hint="어느 마을의 상인 (세계관 속 역할을 적는다)",
    relationship_hint="플레이어와 처음 만나는 사이. 거래와 정보를 주고받는다",
    register_hint="존댓말. 직업에 맞는 말투",
    background_hint="세계관, 마을, 이 인물의 과거와 목적, 아는 인물과 장소를 적는다.",
    situations_hint=("첫 조우, 인사, 작별", "퀘스트 제안, 보상 설명, 거절", "거래, 흥정, 물건 설명",
                     "지명·인물·역사 질문", "적대, 경계, 화해", "재방문, 안부"),
)
PROFILES.add("npc", PROFILE, origin="builtin", path=f"{__name__}:PROFILE")
