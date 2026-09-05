# PEFT 페르소나 데이터 도구 — 재설계 스펙

- 상태: 사용자 승인 완료 (2026-09-05)
- 대상: `persona_sft_data/` 전면 재구성, 문서·설정·테스트 재작성, 구 산출물 삭제
- 선행 문서: 구 설계 `docs/pipeline-design.md` (이 스펙이 대체하며 삭제한다)
- 2026-09-06 구현 후 실제 코드에 맞게 정정한 절: §4.2(`describe()` 반환형,
  `TranslatorFactory.build` 시그니처) · §5(필수 단계는 `assemble`뿐) ·
  §6.2(`Persona.length_rule()` 삭제, 값 해석은 규칙 플러그인) · §6.3(`## 대화 흐름`
  이관) · §8(`respond`의 `translator`) · §9(`reject_record` 센티널, `finalize` 훅) ·
  §10.6(split 파일을 `finalize`에서) · §11.3(언어 이름 표는 `teacher/prompts.py`) ·
  §12(`register`의 존댓말 쪽은 허용 목록) · §13(개행과 `trim_blocks`, 카드의
  `language`·번역 모델, split 파일의 출처). 그 밖의 절은 승인 당시 그대로다.

## 1. 목적

이 프로젝트는 원래 ESP32용 소형 언어모델을 **처음부터 학습**시킬 코퍼스를 만드는
파이프라인이었다. 그래서 2,400만 토큰 예산, 슬롯 채우기 대화 25만 개, 시드를 3배로
부풀리는 단계, `<|u|>`/`<|p|>` 학습 스트림 태그, 캐시 워밍업용 프리앰블 같은 것이
코드에 들어 있다.

목적을 바꾼다. **페르소나 문서 하나, 교사 모델, 임의 포맷·임의 언어의 외부 텍스트
데이터셋에서 사전학습된 LLM을 PEFT(LoRA)로 미세조정할 데이터셋과 학습 레시피를
만드는 도구**가 된다. 페르소나는 반려 펫만이 아니라 AI 반려로봇, 게임 NPC, 소설
등장인물, TRPG 진행자, 세계관 안내자처럼 종류가 다양하다.

산출물은 다음이다.

- OpenAI `messages` 포맷 JSONL 분할 3개 (`train` `val` `test`)
- 학생 모델이 학습·추론에 같이 쓸 채팅 템플릿 (`chat_template.jinja`)
- LLaMA-Factory 레시피 (`dataset_info.json`, LoRA SFT YAML, 실행 안내)
- manifest와 Hugging Face 데이터 카드

## 2. 결정 사항

사용자와 합의한 것을 표로 고정한다. 이후 문서는 이 결정을 전제로 한다.

| 항목 | 결정 |
| --- | --- |
| 범위 | 데이터 파이프라인 + PEFT 레시피 산출. 학습 실행은 외부 도구(LLaMA-Factory)가 한다 |
| 학생 모델 (기본값) | `kakaocorp/kanana-2-1.3b-base`. 설정에만 있고 코드에는 없다 |
| 채팅 템플릿 | ChatML. base 모델에 템플릿이 없고 `<\|im_start\|>`/`<\|im_end\|>` 토큰은 어휘에 있으므로 프로젝트가 정한다 |
| 생성 단계 | `seed`→`dialogue`로 개명 유지, `real`→`ingest`+`respond`로 분리. `expand`·`template` 삭제 |
| 외부 소스 | 포맷 무관(tsv·csv·jsonl·json·parquet·text), 언어 무관(교사가 번역) |
| 구조 | 플러그인 프레임워크. entry point + 설정의 로컬 모듈 목록 + 내장, 모두 같은 레지스트리 |
| 페르소나 종류 | 프로필 플러그인 `companion` `npc` `novel` `trpg` `lore` |
| 게이트 | 문서의 `## 제약` 표 행이 있을 때만 해당 규칙이 켜진다. 규칙은 플러그인 |
| 구 산출물 | `data/`·`datasets/` 삭제. `personas/mongle.md`는 TinyML 문단 제거 후 새 스키마로 이관 |
| 작업 방식 | 단계마다 한국어 커밋·푸시. 주석·독스트링은 한국어 |

## 3. 범위 밖

- 학습·평가 실행. 레시피를 내놓는 데서 끝난다.
- vLLM 서버 기동·모델 교체. 사람이 띄우고 파이프라인은 붙는다 (`setup/`이 돕는다).
- 재개·체크포인트. 단계는 처음부터 다시 돈다. `limit`으로 작게 잘라 돌리는 것이 싸다.
- 다중 GPU, 분산 생성, 교사 응답 캐시.
- 한국어 이외 언어의 **페르소나**. 소스는 어떤 언어든 번역해 쓰지만, 프롬프트·게이트·시스템 프롬프트는 한국어 페르소나를 전제한다. `language` 설정은 번역 목표 언어와 데이터 카드 표기에만 쓰인다. 문자 규칙은 페르소나 문서의 `제약` 표가 정한다.

## 4. 아키텍처

### 4.1 패키지

```
persona_sft_data/
  __init__.py
  __main__.py
  cli.py                  명령: check · run · export · sources · plugins · init · status
  core/
    config.py             PipelineConfig · TeacherConfig · SourceConfig · StudentConfig, 단계 설정 검증
    persona.py            페르소나 문서 파서, Persona, system_prompt()
    schema.py             RecordKind(세션·발화) 정규화·지문, JSONL 입출력
    runner.py             단계 실행 계약: 검증 → 중복 제거 → 게이트 → 통계 → 파일
    gates.py              Gate = 구조 규칙 + 제약 표에서 생성된 규칙 체인
    registry.py           Registry 클래스, 8개 그룹 인스턴스, 로딩 순서
    plugin.py             플러그인 인터페이스(Protocol) 정의
  teacher/
    base.py               Teacher · Request · Result, batched()
    openai_compat.py      OpenAI 호환 chat-completions 백엔드 (vLLM)
    fake.py               FakeTeacher
    prompts.py            프롬프트 조립 (페르소나 + 프로필 + 제약)
  sources/
    base.py               Utterance, Source(fetch·cache), 행 읽기 계약
    formats.py            tsv · csv · jsonl · json · parquet · text 어댑터
    extractors.py         field · regex · conversation · list 추출기
    translate.py          Translator 인터페이스, TeacherTranslator
    safety.py             금칙어 기본 목록과 검사
    topic.py              페르소나 바이그램 주제 신호
  stages/
    ingest.py             소스 → 발화 (번역·주제·안전 필터)
    dialogue.py           교사가 상황·흐름별 대화 작성
    respond.py            발화에 교사가 페르소나로 응답
    filter.py             세션 게이트 + 교차 레코드 과다 반복 제거
    assemble.py           개수 비율 혼합 · 세션 단위 분할 · manifest
    export.py             messages JSONL · 템플릿 · 레시피 · 카드
  profiles/
    base.py               Profile 인터페이스
    companion.py npc.py novel.py trpg.py lore.py
  rules/
    base.py               Rule 인터페이스, Verdict
    structure.py          항상 켜지는 구조 규칙
    register.py length.py script.py emoji.py markdown.py role_label.py
    ai_claim.py repeat.py third_person.py name_suffix.py ellipsis.py
  recipes/
    base.py               Recipe 인터페이스
    chat_template.py      ChatML 렌더러 + jinja 텍스트
    llamafactory.py       dataset_info.json + lora_sft.yaml + README
```

표준 라이브러리만으로 동작한다는 원칙은 유지한다. 선택 의존성은 셋이다:
`parquet`(pyarrow), `student`(tokenizers, huggingface_hub — 학생 토크나이저로 길이
측정), `dev`(pytest).

### 4.2 플러그인 체계

`core/registry.py`의 `Registry[T]` 하나가 모든 확장점을 관리한다.

```python
class Registry(Generic[T]):
    def __init__(self, group: str) -> None: ...        # "persona_sft_data.stages"
    def register(self, name: str) -> Callable[[T], T]: ...   # 데코레이터
    def get(self, name: str) -> T: ...                 # 없으면 PluginError: 등록된 이름 목록 포함
    def names(self) -> list[str]: ...
    def describe(self) -> list[Registration[T]]: ...   # 이름 순. Registration(name, obj, origin, path)

STAGES      = Registry("persona_sft_data.stages")
FORMATS     = Registry("persona_sft_data.formats")
EXTRACTORS  = Registry("persona_sft_data.extractors")
TEACHERS    = Registry("persona_sft_data.teachers")
TRANSLATORS = Registry("persona_sft_data.translators")
RECIPES     = Registry("persona_sft_data.recipes")
PROFILES    = Registry("persona_sft_data.profiles")
RULES       = Registry("persona_sft_data.rules")
```

등록 경로는 셋이고, 같은 이름이면 앞의 것이 이긴다.

1. **설정의 `plugins`** — `["my_pkg.my_stage"]`처럼 모듈 경로를 적으면 `run`·`check`가
   시작할 때 import 한다. 모듈 안의 `@STAGES.register("name")` 데코레이터가 등록한다.
   설치하지 않은 로컬 스크립트를 붙이는 길이다.
2. **entry point** — 설치된 패키지의 `[project.entry-points."persona_sft_data.<group>"]`.
   `importlib.metadata.entry_points(group=...)`로 발견하고, `get()`이 처음 요청할 때
   지연 로드한다.
3. **내장** — 이 패키지 자신도 `pyproject.toml`에 entry point로 선언한다. 그래서
   `persona-sft-data plugins`가 내장과 외부를 같은 표로 보여 주고, 내장을 외부
   플러그인으로 덮어쓸 수도 있다.

각 그룹의 인터페이스는 `core/plugin.py`에 `Protocol`로 둔다. 플러그인은 상속 없이
모양만 맞추면 된다.

두 곳이 구현에서 이 스펙의 첫 안과 달라졌다. `describe()`는 이름 없는 tuple 대신
`Registration(name, obj, origin, path)` 데이터클래스를 돌려준다 — `plugins` 명령이
필드 이름으로 표를 만들고, 레지스트리 내부도 같은 객체를 담고 있어 변환이 없다.
번역기 팩토리는 `build(ctx, teacher)`다: 번역기는 자기 설정 dataclass가 없고,
`ingest`가 이미 만들어 둔 교사(`ingest.teacher`)를 그대로 받는다. 같은 서버에 두 번
붙을 이유가 없고, `ctx`에 설정·로거·언어가 다 들어 있다.

| 그룹 | 인터페이스 요약 |
| --- | --- |
| stages | `name`, `mode: "records"\|"artifact"`, `record_kind: "session"\|"utterance"\|None`, `produces: "raw"\|"filtered"\|"final"\|None`, `settings_type: type`, `requires(config) -> tuple[str, ...]`, `preflight(ctx)`, `run(ctx)` |
| formats | `name`, `extensions`, `rows(data: bytes, fields: Sequence[str]) -> Iterator[dict]` |
| extractors | `name`, `settings_type`, `extract(row: dict, fields: Sequence[str], settings) -> Iterator[str]` |
| teachers | `name`, `build(cfg: TeacherConfig) -> Teacher`; `Teacher`는 `check()`, `generate(requests) -> list[Result]` |
| translators | `name`, `build(ctx, teacher) -> Translator`; `Translator`는 `translate(texts, source_language) -> list[str \| None]` |
| recipes | `name`, `settings_type`, `write(out_dir, export_info, settings) -> list[Path]` |
| profiles | `name`, `assistant_label`, `user_label`, `writer_framing`, `required_sections`, `default_flows`, `default_turns`, `extra_rules`, `document_template()` |
| rules | `name`, `constraint_key`, `build(persona, value, settings) -> Rule`; `Rule.check(turns, verdict)` |

### 4.3 디자인 패턴과 그 자리

| 패턴 | 어디에 | 무엇을 막는가 |
| --- | --- | --- |
| Registry | 8개 그룹 | `_import_stage`의 `dir()` 탐색과 하드코딩 튜플 |
| Adapter | `sources/formats.py` | 소스마다 파서 함수를 새로 쓰는 일 |
| Strategy | 추출기, 교사, 번역기, 레시피, 프로필 | `if profile == "npc"` 분기 |
| Template Method | `core/runner.py` `execute()` | 단계마다 파일·통계·거절 처리를 다시 쓰는 일 |
| Parameter Object | `StageContext` | 단계가 설정·페르소나·경로를 직접 조립하는 일 |
| Factory | `TEACHERS.get(kind).build(cfg)`, `RULES.get(key).build(...)` | 생성 로직이 사용처에 흩어지는 일 |
| Chain of Responsibility | `Gate`의 규칙 체인 | 하나의 거대한 `check_persona` |
| Command | `cli.py`의 서브커맨드 클래스 | `main()` 안의 `if args.command == ...` 사슬 |

## 5. 설정

JSON 하나가 모델 id·URL·경로·비율·한도가 나타나는 유일한 곳이라는 원칙은 유지한다.
테스트가 강제하는 금지 리터럴에 **데이터셋 URL**을 추가한다.

```json
{
  "profile": "companion",
  "language": "ko",
  "data_root": "data",
  "datasets_root": "datasets",
  "seed": 20260905,
  "persona_doc": "personas/mongle.md",
  "plugins": [],

  "student": {
    "model": "kakaocorp/kanana-2-1.3b-base",
    "trust_remote_code": true,
    "chat_template": "chatml"
  },

  "teachers": {
    "reasoner": {
      "kind": "openai",
      "model": "NotoriousH2/kanana-2-30b-a3b-instruct-2601-awq-w4a16",
      "base_url": "http://localhost:8000",
      "temperature": 1.0, "top_p": 0.95, "max_tokens": 320, "concurrency": 200
    },
    "bulk": {
      "kind": "openai",
      "model": "kakaocorp/kanana-2-3b-instruct",
      "base_url": "http://localhost:8000",
      "temperature": 0.9, "top_p": 0.95, "max_tokens": 128, "concurrency": 256
    }
  },

  "sources": {
    "smilestyle": {
      "format": "tsv",
      "url": "https://raw.githubusercontent.com/smilegate-ai/korean_smile_style_dataset/main/smilestyle_dataset.tsv",
      "fields": ["informal", "chat"],
      "language": "ko", "license": "smilestyle"
    },
    "korean_safe_conversation": {
      "format": "parquet",
      "url": "https://huggingface.co/datasets/jojo0217/korean_safe_conversation/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
      "fields": ["instruction"],
      "language": "ko", "license": "apache-2.0"
    },
    "open_korean_instructions": {
      "format": "parquet",
      "url": "https://huggingface.co/datasets/heegyu/open-korean-instructions/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
      "fields": ["text"],
      "extract": {"kind": "regex", "pattern": "<usr>\\s*(.*?)\\s*(?=<bot>|<usr>|<sys>|$)"},
      "language": "ko", "license": "mit"
    },
    "korean_role_playing": {
      "format": "parquet",
      "url": "https://huggingface.co/datasets/huggingface-KREW/korean-role-playing/resolve/refs%2Fconvert%2Fparquet/general-roleplay-data/train/0000.parquet",
      "fields": ["text"],
      "extract": {"kind": "conversation", "role_key": "role", "content_key": "content",
                  "exclude_roles": ["assistant", "bot", "character", "system"]},
      "language": "ko", "license": "apache-2.0"
    },
    "soda": {
      "format": "parquet",
      "url": "https://huggingface.co/datasets/allenai/soda/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet",
      "fields": ["dialogue"],
      "extract": {"kind": "list", "keep": "even"},
      "language": "en", "license": "cc-by-4.0"
    }
  },

  "stages": {
    "ingest": {
      "teacher": "bulk", "translator": "teacher",
      "sources": ["smilestyle", "korean_safe_conversation", "open_korean_instructions",
                  "korean_role_playing", "soda"],
      "limit_per_source": 3000, "min_chars": 2, "max_chars": 60,
      "topic_min_hits": 1, "download_timeout": 60
    },
    "dialogue": {"teacher": "reasoner", "per_situation": 40, "turns": [2, 3, 4]},
    "respond":  {"teacher": "bulk", "limit": 4000},
    "filter":   {"max_identical_assistant_turns": 20, "min_turns": 2, "max_turns": 16},
    "assemble": {"ratios": {"dialogue": 0.7, "respond": 0.3}, "max_sessions": 8000,
                 "split": {"train": 0.9, "val": 0.05, "test": 0.05}},
    "export": {
      "name": "mongle-peft-v1",
      "recipe": {"kind": "llamafactory", "lora_rank": 16, "lora_alpha": 32,
                 "learning_rate": 2e-4, "epochs": 3, "cutoff_len": "auto",
                 "batch_size": 8, "gradient_accumulation": 2}
    }
  }
}
```

의미와 검증 규칙:

- `profile`은 `PROFILES`에 있어야 한다. `language`는 ISO 639-1 두 글자.
- `student.model`은 필수. `chat_template`은 지금 `chatml` 하나만 허용하며, 다른 값은
  `ConfigError`.
- `teachers.<이름>.kind`는 `TEACHERS`에 있어야 하고 기본값은 `openai`. 나머지 키는
  `TeacherConfig` 필드와 같아야 한다(모르는 키는 오류).
- `sources.<이름>`은 `url`과 `path` 중 정확히 하나. `format`은 `FORMATS`에, `extract.kind`는
  `EXTRACTORS`에 있어야 하며 기본은 `field`. `language`·`license` 필수. `fields`는 비어
  있지 않은 목록.
- `stages.<이름>`은 `STAGES`에 있어야 하고, 값은 그 단계의 `settings_type` dataclass로
  생성된다. 모르는 키·빠진 필수 키는 `ConfigError`가 단계 이름과 함께 알린다.
  `teacher`·`translator`가 있으면 존재하는 이름이어야 한다.
- `ingest`·`respond`·`filter`·`export`는 설정에서 뺄 수 있다. **`assemble`만 필수**이고,
  `respond`가 있으면 `ingest`도 있어야 한다. `dialogue`는 사실상 거의 항상 필요하지만
  검증이 강제하지 않는다 — 외부 데이터셋만으로 코퍼스를 만드는 설정(`ingest`+`respond`
  +`assemble`)이 정당하고, `assemble.ratios`의 키가 "설정된 세션 생성 단계의 부분집합"
  이어야 한다는 규칙이 이미 세션 생성 단계가 하나도 없는 설정을 막는다.
- `assemble.ratios`의 키는 설정된 세션 생성 단계 이름의 부분집합이며 합은 1.
- 모든 상대 경로는 설정 파일의 부모의 부모(저장소 루트)를 기준으로 푼다. 지금과 같다.
- `stage_seed(name)`은 전역 시드에서 단계별로 파생한다. 지금과 같다.

## 6. 페르소나 문서 스키마

문서는 여전히 페르소나의 단일 출처이고, 파서는 여전히 엄격하다. 절 구성이 바뀐다.

| 절 | 필수 | 형태 | 쓰임 |
| --- | --- | --- | --- |
| `## 핵심 정의` | 필수 | 2열 표. 행: 이름·정체성·사용자와의 관계·말투·성격·응답 길이·지식 범위 | 시스템 프롬프트, 교사 프롬프트 |
| `## 제약` | 필수 | 2열 표 `\| 규칙 \| 값 \|` | **게이트 규칙 생성.** 행이 없는 규칙은 꺼진다 |
| `## 발화 원칙` | 필수 | 번호 목록 | 프롬프트 |
| `## 다룰 상황` | 필수 | 번호 목록. 한 줄에 쉼표로 여러 순간 | `dialogue`의 커버리지 단위(beat), 주제 필터 신호 |
| `## 배경` | 프로필이 요구하면 필수 | 자유 마크다운 | 시스템 프롬프트와 교사 프롬프트에 그대로 삽입 |
| `## 하지 않는 말과 행동` | 선택 | 불릿 | 프롬프트 |
| `## 어휘와 표현` | 선택 | 2열 표 `\| 감정·상태 \| 예시 \|` | 프롬프트 어휘 표본, 주제 필터 신호 |
| `## 대화 흐름` | 선택 | 불릿 | `dialogue`가 흐름을 고르는 목록. 없으면 프로필 기본값 |
| `## 예시 대화` | 선택 | ```` ```text ```` 블록, `U:`/`A:` 줄 | 교사 프롬프트 few-shot |

### 6.1 제약 표 문법

| 규칙 키 | 허용 값 | 만드는 규칙 플러그인 |
| --- | --- | --- |
| 말투 | `반말` `존댓말` `서술체` `자유` | `register` (`자유`·`서술체`는 규칙 없음) |
| 발화 길이 | `N~M글자` 또는 `N~M문장` | `length` |
| 문자 | `한글` `영문` `혼용` | `script` (`혼용`은 규칙 없음) |
| 이모지 | `금지` `허용` | `emoji` |
| 마크다운 | `금지` `허용` | `markdown` |
| 역할 표기 | `금지` | `role_label` |
| AI 자칭 | `금지` | `ai_claim` |
| 반복 | `금지` | `repeat` |
| 3인칭 자칭 | `금지` | `third_person_self` |
| 이름 어미 | `금지` | `name_suffix` |
| 말줄임표 | `최대 N개` | `ellipsis` |

모르는 키는 `PersonaError`. 값이 문법에 맞지 않아도 `PersonaError`. 규칙은 문서의
값에서만 만들어지고 코드에 기본값이 없다 — 표에 없는 규칙은 꺼진 것이다.

### 6.2 `Persona` 데이터클래스

```python
@dataclass(frozen=True)
class Persona:
    name: str
    core: dict[str, str]
    constraints: dict[str, str]
    principles: tuple[str, ...]
    situations: tuple[str, ...]
    background: str | None
    prohibitions: tuple[str, ...]
    vocabulary: dict[str, tuple[str, ...]]
    flows: tuple[str, ...]
    examples: tuple[tuple[dict[str, str], ...], ...]   # 예시 대화들, 각각 turns
    source: Path

    beats -> tuple[str, ...]                # 다룰 상황을 쉼표로 쪼갠 것
    system_prompt() -> str                  # 핵심 정의 + 배경 + 발화 원칙 + 하지 않는 것
```

`Persona`는 제약 표를 **해석하지 않는다.** `constraints`에 문자열 그대로 담아 두고,
`N~M글자`/`N~M문장` 같은 값의 문법을 아는 것은 그 값을 쓰는 규칙 플러그인뿐이다 —
발화 길이는 `rules/length.py`의 `LengthFactory.build()`가 파싱해 `LengthRule(lo, hi,
unit)`을 만든다. 규칙마다 값 문법이 다르므로 해석을 규칙 쪽에 두면 `Persona`가
규칙 목록을 알 필요가 없고, 새 규칙을 붙일 때 `Persona`를 건드리지 않는다.

`system_prompt()`는 문서의 표·목록을 평문으로 옮긴 것이고 프롬프트용 문장을 따로
쓰지 않는다. 지금과 같은 원칙이다. `제약` 표는 시스템 프롬프트에 넣지 않는다 —
그것은 검열 규칙이지 캐릭터 설명이 아니다. 대신 교사 프롬프트의 `[반드시 지킬 것]`
블록이 제약 표에서 렌더링된다.

### 6.3 `personas/mongle.md` 이관

- 머리말의 P1/P2·SentencePiece·토큰 수·캐시 워밍업 문단, `## 승인 뒤 동결할 항목`을 지운다.
- `## 제약` 표를 추가한다: 말투=반말, 발화 길이=4~35글자, 문자=한글, 이모지=금지,
  마크다운=금지, 역할 표기=금지, AI 자칭=금지, 반복=금지, 3인칭 자칭=금지,
  이름 어미=금지, 말줄임표=최대 1개. 전부 지금 게이트가 하는 검사와 같다.
- `## 감정 표현과 어휘` → `## 어휘와 표현`. `## 고정 프리앰블 대화` → `## 예시 대화`,
  `<|u|>`/`<|p|>` 태그를 `U:`/`A:` 줄로 바꾼다.
- `## 대화 흐름` 절을 추가한다(불릿). `dialogue`가 흐름을 고를 때 `persona.flows`를
  `profile.default_flows`보다 먼저 쓰므로, 이 절이 있으면 프로필 기본값 대신 문서의
  흐름이 쓰인다.
- 그 외 페르소나의 의미(이름·말투·성격·원칙·상황·금지)는 바꾸지 않는다.

## 7. 프로필

프로필은 용도별 기본값 묶음이다. 코드 어디에도 `if profile == ...`가 없어야 한다.

```python
class Profile(Protocol):
    name: str
    assistant_label: str          # 프롬프트에서 assistant를 부르는 말
    user_label: str               # 프롬프트에서 user를 부르는 말
    writer_framing: str           # 교사 시스템 프롬프트 첫 문단
    required_sections: tuple[str, ...]
    default_flows: tuple[str, ...]
    default_turns: tuple[int, ...]
    extra_rules: tuple[str, ...]  # [반드시 지킬 것]에 추가되는 문장
    def document_template(self, name: str) -> str: ...   # init이 쓰는 문서 골격
```

| 프로필 | assistant / user | 프레이밍 | 필수 절 추가 | 기본 흐름 예 | 추가 규칙 |
| --- | --- | --- | --- | --- | --- |
| `companion` | 펫 / 사용자 | 캐릭터가 사용자와 주고받는 짧은 일상 대화를 쓰는 작가 | — | 다정하게 · 무심하게 · 걱정하며 · 장난스럽게 · **사용자가 존댓말로** · 캐릭터가 먼저 원함 · 거절 | 대사만, 행동·표정 묘사 없음 |
| `npc` | NPC / 플레이어 | 게임 속 NPC와 플레이어의 대화를 쓰는 작가 | 배경 | 첫 조우 · 퀘스트 제안 · 거래 · 정보 요청 · 적대 · 재방문 | 세계관 밖 지식 언급 없음 |
| `novel` | 화자 / 독자 | 소설 등장인물의 목소리로 답하는 작가 | 배경 | 회상 · 갈등 · 고백 · 일상 · 독자의 질문 | 인물 시점 유지 |
| `trpg` | 진행자 / 플레이어 | TRPG 세션 로그를 쓰는 작가 | 배경 | 탐색 · 전투 선언 · 협상 · 판정 요청 · 휴식 | 판정 결과를 지어내지 않고 플레이어에게 묻기 |
| `lore` | 안내자 / 질문자 | 세계관 설정을 설명하는 안내자의 문답을 쓰는 작가 | 배경 | 지명 · 인물 · 역사 · 규칙 · 모르는 것 | 배경에 없는 사실은 모른다고 답하기 |

`companion`의 흐름과 규칙은 지금 코퍼스로 검증된 프롬프트에서 온다. 나머지 넷은
FakeTeacher로 형식 계약만 검증하며, 실제 교사 수율은 **미측정**이다(§19).

## 8. 레코드 스키마

레코드 종류가 둘이다. `core/schema.py`의 `RecordKind`가 각각을 안다.

**세션** — `dialogue`·`respond`·`filter`·`assemble`의 레코드.

```json
{"id": "dialogue-000123", "source": "dialogue", "scenario": "배고픔",
 "license": "synthetic", "generator": ["<교사 모델 id>"],
 "turns": [{"role": "user", "text": "배고파?"}, {"role": "assistant", "text": "응, 꼬르륵. 밥 줘."}]}
```

`turns`는 `user`로 시작해 `assistant`로 끝나며 번갈아 온다. 짝수 개. 텍스트는 NFC,
공백 정규화, 비어 있지 않음. `pet` 역할과 `<|u|>`/`<|p|>` 태그는 없다. `respond`의
레코드는 `utterance_id`·`source_dataset`·`source_url`·`original_language`·`translator`
를 추가로 갖는다 — 발화의 출처 필드를 하나도 잃지 않고 옮긴다. 데이터 카드가 소스별
번역 모델을 적을 수 있는 것은 `translator`가 여기까지 따라오기 때문이다(§13).
번역되지 않은 발화에서 온 레코드는 `original_language`·`translator`가 `null`이다.
`assemble`은 `split`을 붙인다.

**발화** — `ingest`의 레코드.

```json
{"id": "soda-000042", "text": "오늘 저녁 뭐 먹을래?", "source": "soda",
 "language": "ko", "license": "cc-by-4.0", "url": "https://...",
 "original_text": "What do you want for dinner tonight?", "original_language": "en",
 "translator": "<교사 모델 id>"}
```

번역되지 않은 발화는 `original_*`·`translator`가 없다.

```python
class RecordKind(Protocol):
    name: str                                  # "session" | "utterance"
    gated: bool                                # 게이트 적용 여부
    def normalize(self, record: Mapping) -> dict: ...   # SchemaError를 던진다
    def fingerprint(self, record: Mapping) -> str: ...  # 중복 제거 키
```

세션 지문은 `role:text`를 casefold 해 이은 문자열의 sha256. 발화 지문은 정규화한
`text`의 sha256.

## 9. 러너 계약

`core/runner.py`의 `execute(stage, config, *, log)`는 `mode == "records"`인 단계를
다음 순서로 돌린다. 지금 러너와 같고, 레코드 종류 선택이 더해졌다.

1. `config.stage_settings(stage.name)`(dataclass), 페르소나, 프로필, 출력 경로, 단계
   시드로 `StageContext`를 만든다. `ctx.gate`는 종류가 세션일 때만 만든다. 게이트의
   턴 수 범위는 `filter` 단계 설정에서 오고, `filter`가 설정에 없으면 기본값
   (`min_turns` 2, `max_turns` 16)을 쓴다. 생성 단계도 낼 때부터 같은 게이트를
   통과하므로 `raw/`에는 이미 통과분만 있다. 지금과 같다.
2. `stage.run(ctx)`가 내는 레코드마다: `metric(**kwargs)` 센티널(`{"_metric": True, …}`)
   이면 통계에 합산. `reject_record(record, reasons)` 센티널(`{"_reject": True, …}`)이면
   정규화·게이트를 다시 걸지 않고 그 사유로 거절한다. 그 밖이면 `kind.normalize` →
   실패는 `schema:<이유>`로 거절. 지문 중복은 `duplicate`로 거절. `gated`면
   `gate.check` → 실패 사유로 거절. 통과분은 출력에 쓰고 표본 200개를 저수지 표집한다.
3. 출력·거절 파일은 `.tmp`에 쓰고 단계가 끝까지 성공한 뒤에 `os.replace`로 제자리에
   옮긴다. 그다음 `<출력>.rejected.jsonl`(거절 전부 + `_reject_reasons`),
   `<출력>.sample.jsonl`, `<출력>.stats.json`을 쓴다. 통계 필드는 지금과 같다
   (`produced` `rejected` `duplicates` `source_filtered` `reject_reasons`
   `teacher_model` `teacher_calls` `teacher_failures` `completion_tokens` `seconds`
   `environment`).
4. 단계가 `finalize(ctx, stats)`를 선언해 두면 마지막에 그것을 부른다(선택). 3의
   교체·통계 쓰기가 끝난 뒤이므로 `finalize`는 `ctx.output`을 다시 읽어 파생 파일을
   만들 수 있다. `assemble`이 이것으로 split 파일과 `final/manifest.json`을 쓴다.

**센티널이 둘인 이유.** 남길 레코드가 있는 거절(단계가 파일 전체를 보고 거른 것)은
`reject_record()`로 넘겨 `.rejected.jsonl`에 남기고, 남길 레코드가 아예 없는 거절
(교사 호출 실패 `teacher_error`, 파싱 실패 `unparseable`, 빈 응답 `empty_reply`)은
`metric(rejected=…, reject_reasons=…)`으로 개수만 센다. 같은 거절을 둘 다로 보고하면
이중 계수가 된다.

**`finalize`가 필요한 이유.** 파생 파일을 `run` 안에서 쓰면 러너의 정규화·중복
제거·게이트를 우회해, 러너가 거절한 레코드가 파생 파일에 남는다. `assemble`이
`run`에서 split 파일을 쓰면 `export`가 거절된 세션을 데이터셋에 싣게 된다.

`mode == "artifact"`인 단계(`export`)는 러너를 거치지 않는다. `run(ctx)`가 직접
파일을 쓰고 `StageStats`를 돌려준다.

```python
@dataclass
class StageContext:
    name: str
    config: PipelineConfig
    persona: Persona
    profile: Profile
    settings: Any                 # 단계의 settings_type 인스턴스
    rng: random.Random
    output: Path | None
    gate: Gate | None
    log: Callable[[str], None]
    def read(self, stage_name: str, *, area: str = "raw") -> Iterator[dict]: ...
```

## 10. 단계

### 10.1 DAG와 실행 순서

```
sources ─ ingest ─ respond ─┐
persona ─ dialogue ─────────┼─ filter ─ assemble ─ export
```

`run`은 설정에 있는 단계만 골라 `requires(config)`로 위상 정렬한다. 하드코딩된
순서 튜플은 없다. `run --stage X`는 X만 돌리고, 입력 파일이 없으면 어느 단계를 먼저
돌리라고 알려 준다. `check`는 설정에 있는 단계마다 `preflight(ctx)`를 부른다.

| 단계 | mode | record_kind | produces | requires |
| --- | --- | --- | --- | --- |
| ingest | records | utterance | raw | () |
| dialogue | records | session | raw | () |
| respond | records | session | raw | (ingest,) |
| filter | records | session | filtered | 설정된 세션 생성 단계 전부 |
| assemble | records | session | final | (filter,) |
| export | artifact | — | — | (assemble,) |

### 10.2 ingest

설정 `IngestSettings(teacher, translator, sources, limit_per_source, min_chars,
max_chars, topic_min_hits, blocked_stems, download_timeout)`.

소스마다:

1. `Source.fetch()` — `url`이면 `data_root/cache/<이름>.<확장자>`에 한 번 받고, `path`면
   그대로 읽는다. 실패는 로그로 남기고 그 소스만 건너뛴다.
2. `FORMATS.get(format).rows(data, fields)` → 행. pyarrow가 없으면 parquet 소스는
   "읽을 수 없음"으로 건너뛴다.
3. `EXTRACTORS.get(extract.kind).extract(row, fields, settings)` → 원문 발화들.
4. 싼 필터를 먼저: NFC·공백 정규화, 빈 것 제거, 원문 글자 수 `min_chars~max_chars`
   (번역이 필요한 소스는 원문 기준 `max_chars * 2`까지 허용하고, 번역 뒤 다시
   `max_chars`로 잰다), 소스 안 중복 제거.
5. `limit_per_source` 개를 단계 시드로 무작위 표집. **번역 전에** 잘라서 비용을 묶는다.
6. `language != config.language`면 `TRANSLATORS.get(translator).build(...)`로 배치
   번역. 실패(`None`)는 `translation_failed`로 거절 집계.
7. 주제 필터 `topic.in_scope(text, signal, min_hits)` — 페르소나의 다룰 상황·어휘
   표·배경에서 만든 바이그램 신호. 탈락은 `source_filtered`(거절이 아님)로 센다.
8. 금칙어 `safety.is_unsafe(text, stems)` — 토큰 앞부분 일치. 탈락은 `unsafe_source`.
9. 발화 레코드를 낸다. 러너가 소스 간 중복을 지문으로 한 번 더 거른다.

통계 `extra`에 소스별 `{raw, distinct, sampled, translated, translation_failed,
in_scope, unsafe}`를 남긴다.

### 10.3 dialogue

설정 `DialogueSettings(teacher, per_situation, turns=None)`. 지금 `seed`와 같다:
`persona.beats`를 빠짐없이 순회하고 beat마다 `per_situation`개, 턴 수는 `turns`
(없으면 `profile.default_turns`)에서 추첨, 흐름은 `persona.flows or
profile.default_flows`에서 추첨, 배치는 교사
`concurrency` 크기. 응답은 `prompts.parse_dialogue`(`U:`/`A:`)와 `repair_dialogue`로
정리하고, 형식 불량은 `unparseable`, 호출 실패는 `teacher_error`로 센다.
`preflight`는 `teacher.check()`.

### 10.4 respond

설정 `RespondSettings(teacher, limit)`. `ctx.read("ingest")`의 발화를 단계 시드로
섞어 `limit`개까지, 발화마다 한 요청. 응답은 `reply_text`(첫 줄, 역할 표기·따옴표
제거)로 정리하고 빈 답은 `empty_reply`. 레코드는 `user`=발화, `assistant`=답, 그리고
발화의 출처 필드를 옮겨 싣는다.

### 10.5 filter

설정 `FilterSettings(max_identical_assistant_turns, min_turns, max_turns)`. 세션을
내는 raw 파일마다 한 번씩 돈다(지금처럼 `stages_for(config)`). 러너의 게이트가
레코드별 검사를 하고, 이 단계는 파일 전체를 봐야 하는 것만 한다: 같은 assistant
발화가 `max_identical_assistant_turns`를 넘으면 `assistant_line_overused`로 거절.

### 10.6 assemble

설정 `AssembleSettings(ratios, max_sessions, split)`. `filtered/`의 세션을 `source`
필드(=생성 단계 이름)로 버킷에 모으고, 버킷마다 `max_sessions * ratio`개를 섞어서
뽑는다. 모자라면 `SHORTFALL`로 로그와 manifest에 적고 비율을 조용히 바꾸지 않는다.
전체를 섞은 뒤 **세션 단위**로 `split`에 따라 `train`/`val`/`test`를 붙여 러너에
yield 한다. `final/{train,val,test}.jsonl`과 `final/manifest.json`(설정 전체, 페르소나
해시, 버킷별 개수, 부족분, 단계별 stats 요약)은 `run`이 아니라 `finalize(ctx, stats)`
에서 쓴다 — 러너 통과분만 실려야 하므로(§9). 토큰 예산·토크나이저·글자당 토큰
추정은 없다.

### 10.7 export

§13.

## 11. 소스

### 11.1 포맷 어댑터

| 이름 | 입력 | 행 |
| --- | --- | --- |
| `tsv` `csv` | 헤더 있는 구분자 파일 (`utf-8-sig`) | `DictReader`의 행. `fields`에 없는 열은 버린다 |
| `jsonl` | 한 줄 하나의 객체 | 그 객체 |
| `json` | 객체 배열, 또는 `{"data": [...]}`처럼 배열을 담은 객체(첫 배열 값) | 배열 원소 |
| `parquet` | pyarrow로 `fields` 열만 투영 | `to_pylist()` 행 |
| `text` | 줄 단위 | `{"text": 줄}` — `fields`는 `["text"]` |

`fields` 열만 실체화하는 것은 parquet에서 특히 중요하다. 지금 `real.py`가 답변 열을
절대 읽지 않는 이유(AI 어시스턴트 문장 유입)는 그대로 유효하며, 설정이 `fields`로
그것을 표현한다.

### 11.2 추출기

| 이름 | 설정 | 동작 |
| --- | --- | --- |
| `field` | — | 선택한 열 각각의 문자열이 발화 하나 |
| `regex` | `pattern`, `group`(기본 1) | 열의 문자열에서 패턴의 그룹을 전부 |
| `conversation` | `role_key`, `content_key`, `include_roles` 또는 `exclude_roles` | 열이 `[{role, content}]` 목록. 역할 조건에 맞는 `content` |
| `list` | `keep: even\|odd\|all` | 열이 문자열 목록(교대 화자). 짝수·홀수 인덱스만 |

### 11.3 번역기

`TeacherTranslator`는 `ingest.teacher`로 지정된 교사에 배치로 보낸다. 프롬프트는
`prompts.translate_system(source_language, target_language)`: "다음 {원어} 문장을
자연스러운 {목표어} 구어체로 옮겨라. 뜻만 옮기고 설명·따옴표·역할 표기 없이 한 줄."
언어 코드 → 이름 표(`LANGUAGE_NAMES`)와 `language_name()`은 `teacher/prompts.py`에
있다 — 프롬프트 문자열이 전부 거기 모여 있고 `sources/translate.py`는 배치 나누기와
실패 자리 표시만 맡는다. 표에 없는 코드는 코드 그대로 쓴다. 응답은 `reply_text`로 정리한다. `check`는 `ingest`의 소스 중 번역이
필요한 것이 있을 때만 교사를 확인한다.

### 11.4 금칙어·주제

`safety.DEFAULT_STEMS`는 지금 `UNSAFE_STEMS`(실측으로 고른 14개)이고 `blocked_stems`
설정이 통째로 대체한다. `topic.signal(persona)`는 다룰 상황·어휘 표·배경의 한글
바이그램 집합이다. 배경이 길면 신호가 넓어져 필터가 느슨해지는데, 그것은 배경이
넓은 페르소나의 올바른 동작이다.

## 12. 게이트 규칙

`Gate`는 규칙 목록이다. `Gate.from_persona(persona, profile, settings)`가 만든다.

- **구조 규칙**(항상): 비어 있지 않음, `min_turns~max_turns`, user 시작, assistant
  종료, 역할 교대, 빈 발화 없음. 사유: `empty` `too_few_turns` `too_many_turns`
  `does_not_start_with_user` `does_not_end_with_assistant` `roles_not_alternating`
  `utterance_empty`.
- **제약 규칙**: `persona.constraints`의 행마다 `RULES.get(key).build(persona, value,
  settings)`. 표에 없는 키의 규칙은 만들지 않는다.

| 규칙 | 검사 (assistant 발화만) | 사유 |
| --- | --- | --- |
| `register` | 반말이면 존댓말 표현(`~요` `습니다` `세요` …)이 있으면 거절. 존댓말이면 **존댓말 종결이 아니면** 거절 (반말 종결 목록을 열거하지 않는다) | `honorific` / `informal_ending` |
| `length` | 글자 수 또는 문장 수(`.?!…` 기준)가 범위 밖 | `assistant_too_short` / `assistant_too_long` |
| `script` | 한글이면 한자·가나·라틴 단어 거절, 한글 없음 거절. 영문이면 한글·한자·가나 거절 | `cjk_characters` `kana_characters` `latin_words` `no_hangul` `hangul_characters` |
| `emoji` | 기호·이모지 문자 | `emoji` |
| `markdown` | 제목·강조·목록·코드·링크 | `markdown` |
| `role_label` | 발화 어디든 `U:` `A:` `사용자:` `user:` 등 | `role_label_in_text` |
| `ai_claim` | AI·인공지능·언어모델·챗봇·프로그램·컴퓨터·시스템 프롬프트·학습 데이터·토큰 | `claims_to_be_ai` |
| `repeat` | 연속 단어 반복, 연속 구절 반복 | `repeated_phrase` |
| `third_person_self` | `<이름>(이|가|은|는|도|이도)`로 시작 | `third_person_self` |
| `name_suffix` | 이름의 음절을 어미로 단 토큰 | `name_suffix_babytalk` |
| `ellipsis` | `…` 개수 > N | `multiple_ellipsis` |

`register`의 존댓말 쪽은 **허용 목록**이다: 반말 종결을 열거해 거절하는 것이 아니라,
문장 끝이 존댓말 종결이 아니면 `informal_ending`으로 거절한다. 반말 종결은 수가
많고 계속 늘어나므로 열거하는 쪽이 늘 뚫린다. 두 정규식(존댓말 표현·존댓말 종결)의
실제 목록은 `rules/register.py`의 `HONORIFIC`·`HONORIFIC_END`가 단일 출처다 — 이
표는 목록을 복사하지 않는다.

지금 `test_foundation.py`의 "실제 교사 출력에서 나온 위반" 케이스는 전부 그대로
통과해야 한다(`mongle.md`의 제약 표가 같은 규칙을 켜므로).

## 13. 내보내기와 레시피

`export`는 `final/{train,val,test}.jsonl`을 읽어 `datasets/<name>/`을 만든다
(`ctx.config.final(split)`, split은 `train`·`val`·`test`. 하나라도 없으면
"먼저 assemble 단계를 돌려라"로 멈춘다).

그 세 파일은 `assemble`의 `finalize(ctx, stats)`가 쓴다. `assemble`은 러너에 세션을
yield 할 뿐이고(러너 출력은 `final/assemble.jsonl`), 러너가 정규화·지문 중복 제거·
게이트를 끝내고 출력을 제자리에 옮긴 뒤에 `finalize`가 그 출력을 다시 읽어 `split`
필드별로 나눈다. 그래서 `export`가 데이터셋에 싣는 것은 **러너 통과분뿐**이다 —
거절된 세션은 `final/assemble.jsonl.rejected.jsonl`에만 있다.

```
datasets/<name>/
  train.jsonl  val.jsonl  test.jsonl     OpenAI messages, 한 줄 한 대화
  system_prompt.txt                       persona.system_prompt()
  chat_template.jinja                     ChatML (TRL용 generation 마커 포함)
  rendered_sample.txt                     train 첫 3개를 템플릿으로 렌더링한 학습 텍스트
  manifest.json
  README.md                               데이터 카드
  recipe/llamafactory/
    dataset_info.json
    lora_sft.yaml
    README.md                             실행 명령과 주의점
```

**messages 레코드.** `messages[0]`은 항상 system(`system_prompt.txt`와 같음), 뒤로
`user`/`assistant` 교대. 출처 필드(`id` `source` `scenario` `license` `generator`
`source_dataset` `source_url` `original_language`)는 그대로 실린다.

**채팅 템플릿.** `recipes/chat_template.py`가 jinja 텍스트와 같은 형식을 내는 파이썬
렌더러를 둘 다 갖는다. 렌더 결과는 다음이고, LLaMA-Factory의 `chatml` 템플릿과
바이트 단위로 같다.

```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>
```

jinja 쪽은 assistant 본문을 `{% generation %}…{% endgeneration %}`로 감싸 TRL의
`assistant_only_loss`가 마스크를 만들 수 있게 한다. 테스트가 파이썬 렌더러의 출력을
손으로 쓴 기대 문자열과 대조한다.

**jinja 텍스트의 개행은 블록 태그 앞에 둔다.** 트레이너가 어떤 jinja 환경으로 이
텍스트를 컴파일할지 우리가 정하지 못하고, `trim_blocks=True`면 블록 태그 바로 뒤의
개행이 사라진다. `<|im_end|>{% endgeneration %}\n`처럼 쓰면 그 환경에서 assistant 턴
뒤 개행이 없어져 `<|im_end|>`와 다음 `<|im_start|>`가 붙어 위 형식과 다른 바이트가
나온다. 그래서 개행을 `{% endgeneration %}` **앞**의 리터럴로 두고 뒤에 남는 공백은
`{%- else`로 걷어 낸다. 테스트가 기본 환경과 `trim_blocks`/`lstrip_blocks` 환경에서
각각 렌더해 파이썬 렌더러와 바이트가 같은지 확인한다(`{% generation %}`은 표준
jinja2가 모르는 태그이므로 테스트가 최소 확장으로 파싱시킨다 — 태그를 지워 렌더하면
블록 구조가 사라져 이 결함을 잡지 못한다).

**레시피 `llamafactory`.** 설정 `LlamaFactorySettings(lora_rank, lora_alpha,
lora_dropout=0.05, learning_rate, epochs, cutoff_len, batch_size,
gradient_accumulation, warmup_ratio=0.05)`.

- `dataset_info.json`: `{"<name>": {"file_name": "../../train.jsonl", "formatting":
  "sharegpt", "columns": {"messages": "messages"}, "tags": {"role_tag": "role",
  "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant",
  "system_tag": "system"}}, "<name>_val": {... "val.jsonl"}}`.
- `lora_sft.yaml`: `model_name_or_path`=`student.model`, `trust_remote_code`,
  `stage: sft`, `do_train: true`, `finetuning_type: lora`, `lora_rank/alpha/dropout`,
  `lora_target: all`, `dataset: <name>`, `eval_dataset: <name>_val`,
  `dataset_dir`(이 디렉터리의 절대 경로), `template: chatml`, `cutoff_len`,
  `train_on_prompt: false`, `output_dir: saves/<name>/lora`, `per_device_train_batch_size`,
  `gradient_accumulation_steps`, `learning_rate`, `num_train_epochs`,
  `lr_scheduler_type: cosine`, `warmup_ratio`, `bf16: true`, `eval_strategy: steps`,
  `eval_steps: 100`, `save_steps: 200`, `logging_steps: 10`, `report_to: none`,
  `overwrite_output_dir: true`.
- `README.md`: `llamafactory-cli train recipe/llamafactory/lora_sft.yaml` 한 줄과, 이
  모델이 `custom_code`라 `trust_remote_code`가 필요하다는 것, `cutoff_len`이 어떻게
  정해졌는지.

**길이 보고.** `student` extra가 있으면 `huggingface_hub.hf_hub_download(model,
"tokenizer.json")`으로 받은 토크나이저로 렌더링된 학습 텍스트의 토큰 수 p50·p95·p99·
최대를 잰다. 없으면 글자 수를 재고 `method: "characters"`로 적는다.
`cutoff_len: "auto"`는 p99를 64의 배수로 올림(최소 256)하고, 글자 수 기준일 때는
글자 수 p99를 그대로 쓴다(한국어는 글자당 1토큰 미만이므로 넉넉한 쪽이다).
manifest에 `length_report`로 방법과 분위수를 적는다.

**데이터 카드.** 지금 카드에 더해 소스별 원어와 번역 모델, 프로필, 학생 모델,
채팅 템플릿 이름, 길이 보고를 적는다. 번역된 소스가 있으면 원본 라이선스를 그대로
표시한다. 번역 모델은 발화 레코드의 `translator`가 `respond` 세션 레코드로 옮겨진
것을 `manifest.source_datasets[<이름>].translator`로 모은 값이다(§8).

YAML 머리말의 `language` 목록과 언어 태그(`korean`은 `ko`일 때만)는 설정의
`language`에서 만들고, 그 값은 `manifest.language`로 실려 온다. 카드 생성 함수
(`dataset_card(manifest, system_prompt)`)는 manifest만 보므로 코드에 언어 리터럴이
박히지 않는다(§3).

## 14. CLI

`persona-sft-data <명령> [--config 경로]`. 도움말은 한국어. `python -m persona_sft_data`도 같다.

| 명령 | 하는 일 |
| --- | --- |
| `check` | 설정 로드, 페르소나 파싱 결과 요약, 프로필, 설정된 단계마다 `preflight`(교사 접속·모델 일치, 소스 몇 행 읽기·추출 표본, 학생 토크나이저 가용 여부) |
| `run [--stage X]` | 설정된 단계를 위상 순으로. `export`도 포함 |
| `export [--name N]` | assemble 결과에서 내보내기만 |
| `sources [--sample N] [--translate]` | 소스별로 N개 발화를 보여 준다. `--translate`면 번역 결과도. 교사 없이 소스 설정을 점검하는 용도 |
| `plugins` | 그룹별 등록 목록: 이름, 출처(내장·entry point·plugins), 객체 경로 |
| `init <이름> [--profile P]` | `personas/<이름>.md`(프로필 골격)와 `configs/<이름>.json`(기본값, 소스 없음)을 만든다. 이미 있으면 거부 |
| `status [--watch]` | 단계별 산출 개수·목표·수율 한 화면. `progress.py`를 대체 |

서브커맨드는 `cli.py`의 `Command` 클래스들이고 `main()`은 파서 구성과 디스패치만 한다.

## 15. 오류 처리

- 설정·페르소나·플러그인 오류는 `ConfigError`·`PersonaError`·`PluginError`로 종료 코드
  2, 원인을 한 줄로.
- 교사 접속 실패·모델 불일치는 `TeacherError`로 생성 전에 멈춘다. 개별 호출 실패는
  `Result.error`로 돌아와 통계에 센다.
- 소스 하나의 다운로드·파싱 실패는 그 소스만 건너뛰고 로그와 통계에 남긴다. 소스
  전부가 실패하면 `ingest`는 빈 파일과 통계를 쓰고 `respond`는 입력이 비었다고 알린다.
- 번역 실패는 발화 단위로 `translation_failed`.
- 러너의 거절은 예외가 아니다. 예외는 계약 위반(단계가 잘못된 형태를 낸 것)일 때만
  난다.

## 16. 테스트

GPU·네트워크 없이 전부 돈다.

- **불변식**(`test_foundation.py`): 페르소나 이름·모델 id·`data/` 경로·**데이터셋 URL**이
  `persona_sft_data/`의 실행 문자열에 없다. 페르소나 파서는 필수 절 누락·제약 문법
  오류에 예외. 실제 교사 출력에서 나온 위반 7종을 게이트가 잡는다. 설정은 모든
  경로를 `data_root`에서 파생한다.
- **레지스트리**: 데코레이터·entry point·`plugins` 모듈 세 경로, 우선순위, 없는 이름의
  오류 메시지.
- **포맷·추출기**: 각 포맷의 작은 픽스처, 각 추출기의 경계(빈 열, 역할 없음, 홀짝).
- **번역기**: FakeTeacher로 배치·실패·역할 표기 제거.
- **단계**: FakeTeacher와 `tests/fixtures/`의 로컬 소스로 다섯 단계 각각. 통계 필드,
  거절 사유, 시드 재현성, 입력 없음 메시지.
- **게이트**: 제약 표 행별 켜짐·꺼짐, 존댓말 페르소나에서 반말 거절, 문장 수 길이.
- **프로필**: 다섯 프로필의 문서 골격이 파서를 통과하고, 프롬프트가 프로필의 라벨과
  프레이밍을 포함한다.
- **내보내기·레시피**: 파일 목록, messages 형태, 템플릿 렌더 기대 문자열,
  `dataset_info.json` 키, YAML 필수 키, 길이 보고 두 방식.
- **CLI 스모크**: `configs/smoke.json`(`kind: fake` 교사, 로컬 픽스처 소스, 영어 소스
  하나 포함)로 `check` → `run` → `export`가 임시 디렉터리에서 끝까지 돈다.

## 17. 삭제·이관 목록

| 대상 | 처리 |
| --- | --- |
| `stages/template.py`, `stages/expand.py`, `tests/test_template.py`, `tests/test_expand.py` | 삭제 |
| `stages/seed.py` → `stages/dialogue.py`, `stages/real.py` → `stages/ingest.py` + `stages/respond.py` + `sources/*` | 이관 |
| `progress.py` | 삭제, `status` 명령으로 대체 |
| `schema.py`의 `USER_TAG` `PET_TAG` `serialize_turns` `session_text` `normalize_exchange_pair` `EXCHANGE_PAIR_*` `JsonlStageIO` `RecordContract` | 삭제, `RecordKind`로 대체 |
| `assemble.py`의 `TokenCounter`, `target_tokens`, `chars_per_token_estimate`, `tokenizer` | 삭제 |
| `backend.py` → `teacher/`, `gates.py` → `core/gates.py` + `rules/` | 이관 |
| `docs/pipeline-design.md` | 삭제 (이 스펙이 대체) |
| `docs/wsl-vllm.md`, `setup/wsl_vllm_setup.sh`, `setup/wsl_net_mode.ps1` | 유지 |
| `setup/overnight.sh` | 새 단계 이름·순서로 재작성 (reasoner: dialogue, bulk: ingest·respond) |
| `configs/mongle.json`, `configs/smoke.json` | 새 스키마로 재작성 |
| `personas/mongle.md` | §6.3 |
| `README.md` | 재작성 |
| `data/`, `datasets/` | 삭제 (gitignore 대상, 사용자 승인) |
| `pyproject.toml` | extras `parquet`·`student`·`dev`, entry point 8그룹, `persona-sft-data` 스크립트 |

## 18. 구현 순서

단계마다 테스트가 통과한 상태로 한국어 커밋·푸시한다. 3~7은 병렬 에이전트로 나눌 수 있다.

1. 이 스펙 커밋.
2. 삭제와 문서 이관: §17의 삭제, `personas/mongle.md` 새 스키마, `data/`·`datasets/` 제거.
3. `core/`: registry · plugin · config · persona · schema · runner · gates. `rules/` 전부.
4. `teacher/`: base · openai_compat · fake · prompts. `profiles/` 다섯.
5. `sources/`: base · formats · extractors · translate · safety · topic.
6. `stages/`: ingest · dialogue · respond · filter · assemble.
7. `recipes/`와 `stages/export.py`.
8. `cli.py`, `configs/`, `tests/fixtures/`, 스모크 엔드투엔드.
9. `README.md`, `setup/overnight.sh`, `pyproject.toml` entry point 마무리.

## 19. 한계와 미측정

측정하지 않은 것을 측정한 것처럼 쓰지 않는다.

| 항목 | 상태 |
| --- | --- |
| `companion` 프롬프트의 실제 교사 수율 | 구 파이프라인에서 seed 89%, real 28%(길이 규칙 초과가 원인) 측정. 새 코드로 재측정 필요 |
| `npc` `novel` `trpg` `lore` 프롬프트 | **미측정.** 형식 계약만 테스트 |
| 교사 번역 품질 (3B, 영→한) | **미측정.** 짧은 구어 문장에 한정해 쓰고 `sources --translate`로 눈으로 본다 |
| kanana-2-1.3b-base + ChatML LoRA 학습 | **미실행.** 레시피는 LLaMA-Factory 문서의 필드로 구성했고, 실제 학습은 사용자가 돌린다 |
| `cutoff_len` 자동값 | 길이 분포에서 계산. 학습 메모리와의 관계는 미측정 |
