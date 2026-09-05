# 데이터 생성 파이프라인 재설계

- 상태: 사용자 승인 완료
- 대상: `persona_sft_data/` 전면 재작성, `data/` 레이아웃 재구성
- 작성일: 2026-09-04
- 선행 문서: [한국어 Tiny LM 설계](2026-09-04-korean-tiny-lm-design.md), [페르소나 정의](../../persona.md)

## 1. 목적

P1(데이터)·P2(토크나이저·학습)는 이미 동작하는 결과를 냈다. 코퍼스 1,200만
토큰으로 학습한 체크포인트가 val loss 0.6626을 기록했고, `persona_sft_dataerate.py`가
한국어 대화를 만들어 낸다. **이 재작성의 목적은 결과를 개선하는 것이 아니라
결과를 다시 만들 수 있게 하는 것이다.**

현재 파이프라인은 한 번 돌려서 한 번 나온 코퍼스를 만들었다. 같은 코퍼스를 다시
만들 수도, 비율 하나를 바꿔 다시 돌릴 수도, 교사 모델을 바꿔 끼울 수도 없다.
그 세 가지를 가능하게 하는 것이 목표다.

## 2. 무엇이 문제인가 — 측정치

`persona_sft_data/`은 29개 파일 7,090줄이다. 실험을 반복한 흔적이 지워지지 않고 그대로
남아 있다.

**중복 생성기 9종.** `teacher_seed` · `teacher_expand` · `diverse_dialogues` ·
`generate_pairs` · `pivot_pairs` · `adapt_foreign_pairs` · `hf_foreign_pairs` ·
`qwen38_pairs` · `qwen38_foreign_pairs`. 검증기는 4종
(`validate_pairs` · `validate_foreign_pairs` · `validate_paraphrase_bank` ·
`pair_quality`), 번역기는 3종이다. 각각은 앞선 것의 복사본에서 갈라져 나왔고,
어느 것이 현재 코퍼스를 만들었는지는 코드만 봐서는 알 수 없다.

**하드코딩 3종.**

| 종류 | 실측 | 예 |
| --- | --- | --- |
| 모델 id | 8곳 | `generate_pairs.py:27` `"hf.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-GGUF:Q5_K_M"`, `assemble.py` `"teacher_qwen38"` |
| 경로 리터럴 | 약 15곳 | `generate_pairs.py:415` `default=Path("data/runs/exaone35-research/generated.jsonl")` |
| 페르소나 문자열 | **10개 파일** | `persona.py`에 32회, `templates.py`에 17회, `adapt_foreign_pairs.py`에 6회 |

모델을 바꾸려면 8곳을 고쳐야 하고, 프롬프트 분기가 `if prompt_profile ==
"exaone35"` 형태로 3곳(`:99`, `:161`, `:255`)에 박혀 있다. 페르소나 한 줄을
고치면 10개 파일이 서로 어긋난다. **`personas/mongle.md`가 "동결"이라고 선언한
내용이 코드에는 열 벌 복사되어 있다.**

**`data/` 195개 파일.** `bootstrap/ filtered/ generated/ packed/ pilot/ raw/
review/ runs/ smoke/` 9개 최상위 디렉터리에 일회성 실험 디렉터리가 섞여 있다
(`pilot/safe_pairs_35b_256_review`, `review/soda_foreign_crossvalidated`,
`runs/p1-exaone35-eval-v1`). 어느 파일이 `data/corpus.jsonl`에 기여했는지 알 수
없다.

**외국어 번역 체인 약 1,350줄.** `foreign_sources` → `translate_foreign_pairs`
→ `adapt_foreign_pairs` → `validate_foreign_pairs`는 영어 SODA 대화를 한국어로
옮겨 오는 경로다. `foreign_sources.py`는 `PET_RELEVANT` ·
`UNSUITABLE_SOURCE` · `UNSUITABLE_DIALOGUE` 세 개의 거대한 정규식으로 영어
대화를 걸러 낸다. **이 체인이 존재한 유일한 이유는 로컬에 쓸 만한 한국어 교사가
없었기 때문이다.** 그 전제가 이번에 사라진다.

## 3. 교사 모델과 서빙

### 3.1 선택

| 역할 | 모델 | 크기 | 근거 |
| --- | --- | --- | --- |
| 시드·추론 | `kanana-2-30b-a3b-instruct-2601` **AWQ w4a16** | **16.6GiB** | 30,670.8M 파라미터의 `DeepseekV3ForCausalLM` MoE. 라우팅 전문가 128개 중 **토큰당 6개만 활성**(+공유 2개)이라 실질 3B 속도가 난다. |
| 대량 대화 | `kakaocorp/kanana-2-3b-instruct` | bf16 약 7GB | 3,509M, qwen3 아키텍처. 물량 담당. |

가중치는 `NotoriousH2/kanana-2-30b-a3b-instruct-2601-awq-w4a16`을 쓴다
(4샤드 16.6GiB, llm-compressor `recipe.yaml` 포함, 이 모델의 양자화 중 다운로드
최다). 대안은 `lee5j/kanana-2-30b-a3b-instruct-2601-awq-w4a16`과 동 `-gptq-w4a16`이다.
kakao 공식 양자화는 없다.

**두 교사가 모두 kanana 계열이라 채팅 템플릿과 한국어 문체가 일관된다.**
`chat_template.jinja`가 두 저장소에서 동일하다.

#### 왜 4비트인가 — 8비트는 들어가지 않는다

MoE는 활성 파라미터가 3B라도 **전문가 전체를 VRAM에 상주시켜야 한다.** 30.7B
전부가 올라간다.

GPU 실측은 **32,607MiB = 31.84GiB**다(`nvidia-smi`, compute capability 12.0).

| 양자화 | 실측 파일 크기 | 31.84GiB에 |
| --- | --- | --- |
| Q8_0 GGUF | **30.4GiB** | 남는 자리 1.4GiB. KV 캐시·활성값·CUDA 컨텍스트가 못 들어간다 |
| Q6_K GGUF | 23.5GiB | 크기는 맞으나 아래의 GGUF 문제가 남는다 |
| Q5_K_M GGUF | 20.3GiB | 위와 같다 |
| **AWQ w4a16** | **16.6GiB** | KV 캐시에 약 15GiB가 남는다 |

Q8은 **크기에서 끝난다.** 30.4GiB를 올리면 1.4GiB가 남고, 그 안에 KV 캐시가
들어가지 못한다. 배치를 키워 처리량을 얻는다는 전제 자체가 성립하지 않는다.

GGUF 경로는 크기와 별개로 두 가지가 걸린다. vLLM 문서의 하드웨어 지원표는
**GGUF를 "Volta through Hopper"까지로 적고 있어 Blackwell(sm_120)이 목록에
없다.** 그리고 GGUF가 안 되면 llama.cpp/ollama로 내려가는데, **그쪽은 연속
배칭이 없어 대량 생성 처리량이 vLLM보다 크게 낮다** — 속도를 높이려는 목적과
어긋난다.

같은 표에서 Marlin은 "Turing onwards"라 Blackwell을 포함한다. 이 저장소의
가중치는 `compressed-tensors` W4A16(`pack-quantized`, group 128, 대칭 int4,
`lm_head` 제외)이고 vLLM이 이를 Marlin 커널로 처리하므로 지원 범위 안이다.
**다만 표는 근거이지 측정이 아니므로, 실제 기동은 스파이크로 확인한다(§11).**

4비트를 고르는 이유는 타협이 아니라 처리량이다. 활성 3B에 **MLA 어텐션**
(`kv_lora_rank: 512`, `qk_rope_head_dim: 64`)이 겹쳐 토큰당 KV 캐시가 작으므로,
남은 14GiB로 배치를 크게 잡을 수 있다. 1,200만 토큰 규모에서는 Q4와 Q6의 품질
차이보다 처리량 차이가 지배적이다.

**EXAONE-4.0-32B는 채택하지 않는다.** 32B 밀집 모델이라 같은 4비트에서도 토큰당
연산이 A3B의 열 배다. 같은 VRAM으로 더 느리다.

**HyperCLOVAX-SEED도 채택하지 않는다.** 공개된 것은 0.5B와 1.5B뿐으로 kanana 3B의
1/7~1/2 크기이고, 1.5B는 gated이며, 커스텀 아키텍처라 vLLM 지원이 불확실하다.
교사 품질이 학생 품질의 천장이므로 물량 담당을 더 작은 모델로 내리지 않는다.
kanana-3B가 실패할 때의 대안으로만 남긴다.

### 3.2 서빙

vLLM은 리눅스 전용이다. 이 기계의 WSL2 Ubuntu 24.04를 쓴다 (Python 3.12.3,
`/` 여유 944GB, `nvidia-smi`로 RTX 5090 32GB 보임, torch·vLLM·uv **미설치**).

```
WSL2 Ubuntu 24.04                     Windows
┌──────────────────────────┐          ┌─────────────────────────┐
│ vllm serve <model>       │          │ .venv\python     │
│   --port 8000            │◄── HTTP ─┤   -m persona_sft_data run      │
│ OpenAI 호환 엔드포인트    │ localhost │   --config configs/...  │
└──────────────────────────┘          └─────────────────────────┘
```

**경계는 HTTP 하나다.** WSL/Windows 파이썬 인터롭을 코드에 들이지 않는다.
파이썬 쪽은 `openai` 클라이언트로 `http://localhost:8000/v1`에 붙을 뿐이고,
상대가 vLLM인지 ollama인지 진짜 OpenAI인지 알 필요가 없다. WSL의
`localhost:8000`은 Windows에서 그대로 보인다(WSL2 localhost forwarding).

**두 모델은 순차로 올린다.** 16.6GiB + 7GB = 약 24GB로 동시 적재는 KV 캐시까지
고려하면 빠듯하고, 단계가 어차피 순차이므로 얻을 것이 없다. `seed` 단계에서
30B-A3B, `expand`·`real` 단계에서 3B를 띄운다. 모델 교체는 서버 재시작이고,
이는 파이프라인 밖의 수동 조작으로 둔다(§9).

## 4. 아키텍처

### 4.1 모듈

29개 파일을 12개 모듈(+ `__init__.py` 2개)로 줄인다.

```
persona_sft_data/
  __init__.py
  config.py      PipelineConfig / TeacherConfig / StageConfig
  persona.py     personas/mongle.md 파서 — 코드에 페르소나 문자열 0개
  schema.py      Session / Turn 레코드 + 경계 검증
  backend.py     Teacher 인터페이스 + vLLM OpenAI 클라이언트
  prompts.py     프롬프트 조립 (페르소나 + 상황 + 지시)
  cli.py         python -m persona_sft_data run --config <path> [--stage <name>]
  stages/
    __init__.py  STAGES 레지스트리
    seed.py      30B-A3B: 상황별 대화 골격
    expand.py    3B: 골격 → 대량 변형
    real.py      실제 한국어 코퍼스 수집 + 페르소나 응답 부착
    template.py  슬롯 템플릿
    filter.py    품질 게이트
    assemble.py  비율 혼합 + 분할 + manifest
```

`prompts.py`가 12번째 모듈이다. 프롬프트를 스테이지마다 흩뿌리지 않고 한곳에
모으는 것이 `if prompt_profile == "exaone35"` 분기가 되살아나지 않게 하는
방법이다.

### 4.2 스테이지 계약

**모든 스테이지가 같은 모양이다.**

```python
def run(cfg: PipelineConfig, stage: StageConfig) -> StageResult:
    """입력 jsonl 읽기 → 출력 jsonl + stats.json 쓰기."""
```

- 입력은 `stage.inputs`(경로 목록, 없을 수 있음), 출력은 `stage.output` 하나.
- 모든 레코드는 쓰기 직전 `schema.validate_session()`을 통과해야 한다.
  통과하지 못한 레코드는 버리지 않고 `<output>.rejected.jsonl`에 사유와 함께
  적는다. **버려진 것을 셀 수 없으면 품질을 말할 수 없다.**
- 모든 스테이지가 `<output>.stats.json`을 쓴다: 입력 수, 출력 수, 거절 수,
  거절 사유별 집계, 소요 시간, 사용한 모델 id.

교사를 새로 붙이는 일이 파일 추가가 아니라 **config 한 줄**이 되는 것이 이
계약의 목적이다.

### 4.3 DAG

```
        ┌─ seed (30B-A3B) ─→ expand (3B) ────┐
        │                                     │
config ─┼─ real (3B + SmileStyle) ───────────┼─→ filter ─→ assemble
        │                                     │
        └─ template (규칙, 모델 없음) ─────────┘
```

`real` · `template`은 `seed`와 독립이므로 병렬로 돌 수 있다. 다만 `real`과
`expand`가 같은 3B 서버를 쓰므로 실행은 순차로 둔다 — 동시성 이득보다
서버 하나를 두 프로세스가 두드릴 때의 진단 난이도가 크다.

### 4.4 외국어 번역 체인 제거

`foreign_sources` · `translate_foreign_pairs` · `adapt_foreign_pairs` ·
`validate_foreign_pairs` · `pivot_pairs` 약 1,350줄을 **전부 삭제한다.**
영어 SODA를 걸러 번역하던 이유는 로컬 한국어 교사의 부재였고, kanana가 그
자리를 대신한다. 세 개의 거대 정규식 블록리스트도 같이 사라진다.

`real` 스테이지는 **진짜 한국어**만 남긴다. 원본은 Smilegate AI의
`korean_smile_style_dataset`(raw.githubusercontent 직접 다운로드, 인증 불필요,
사람이 쓴 한국어 구어)이다. 사용자 발화를 여기서 가져오고, 몽글의 응답은
현재처럼 하드코딩된 `NEUTRAL_REPLIES`/`QUESTION_REPLIES` 튜플에서 뽑는 대신
3B 교사가 생성한다. 이것이 `real_corpus.py`에 남은 마지막 하드코딩을 없앤다.

명세의 "실제 한국어 대화 10~20%"에서 **실제**는 합성이 아니라는 뜻이므로,
사용자 발화가 사람이 쓴 것이면 요건을 만족한다.

#### real 슬라이스의 실제 한계 — 측정치

SmileStyle을 실측했다(2026-09-04, 2.36MB 다운로드 성공).

| 항목 | 값 |
| --- | --- |
| 전체 행 | 3,705 |
| 유효 행 | 3,470 |
| 문체 열 | 17 (`formal` `informal` `chat` `choding` `joongding` `sosim` 등) |
| 최대 발화 수 | 3,470 × 17 = 58,990 |

현재 manifest의 real은 **51,112 세션**이다. 즉 **원본을 이미 거의 다 소진했다.**
게다가 17개 열은 같은 의미를 문체만 바꿔 쓴 것이므로 **서로 다른 의미는
3,470개뿐이다.** 이것으로 12M 토큰의 15%인 1.8M 토큰을 채우면 의미 반복이 심하다.
증량 여지가 없다는 사실을 설계에 명시해 둔다.

보강 후보로 `jojo0217/korean_safe_conversation`(27,000행, **apache-2.0**,
사람 검수, 국립국어원 모두의 말뭉치 + AIHub 감성대화 기반)을 조사했다. 쓸 수는
있으나 **함정이 있다.**

**`output` 열을 절대 쓰지 않는다.** 이 데이터셋의 답변은 AI 어시스턴트 발화다 —
실제 값이 `저는 인공지능 챗봇이기 때문에 여행을 떠나지는 못했습니다.` 같은
문장이다. 이는 `personas/mongle.md`가 **명시적으로 금지한 바로 그 문장**이다
("자신을 AI, 인공지능, 언어모델, 챗봇... 이라고 말하지 않는다"). instruction/output을
자연스럽게 쌍으로 쓰면 페르소나가 금지한 문장이 코퍼스에 그대로 주입된다.
**`instruction` 열만 쓰고 `output`은 버린다.**

`instruction` 열도 그대로는 못 쓴다. 대부분 존댓말이고 주제가 여행·방송처럼
펫의 생활 범위 밖이다. 따라서 `real` 스테이지는 이 열에 두 단계를 적용한다.

1. 주제 필터 — 페르소나의 상황 15종에 닿는 발화만 남긴다.
2. 3B 교사가 반말로 문체를 옮긴다. **의미는 사람이 쓴 것이 유지되므로 "실제"의
   성격을 잃지 않는다.**

두 소스를 합쳐도 서로 다른 의미는 3만 개 규모다. **real 슬라이스의 의미 다양성이
teacher 슬라이스보다 낮다는 것은 이 프로젝트의 구조적 제약이며, 감출 것이 아니라
`stats.json`에 고유 의미 수로 보고한다.**

## 5. 설정

`configs/mongle.json` **하나가 모델 id·비율·한도·경로가 등장하는 유일한 곳이다.**

```json
{
  "data_root": "data",
  "seed": 20260904,
  "persona_doc": "personas/mongle.md",
  "teachers": {
    "reasoner": {
      "model": "NotoriousH2/kanana-2-30b-a3b-instruct-2601-awq-w4a16",
      "base_url": "http://localhost:8000/v1",
      "temperature": 0.8,
      "max_tokens": 512,
      "concurrency": 200
    },
    "bulk": {
      "model": "kakaocorp/kanana-2-3b-instruct",
      "base_url": "http://localhost:8000/v1",
      "temperature": 1.0,
      "max_tokens": 256,
      "concurrency": 256
    }
  },
  "stages": {
    "seed":     { "teacher": "reasoner", "per_situation": 60 },
    "expand":   { "teacher": "bulk", "variants_per_seed": 24 },
    "real":     { "teacher": "bulk", "limit": 60000 },
    "template": { "limit": 40000 },
    "filter":   { "max_utterance_chars": 35, "min_utterance_chars": 2 },
    "assemble": {
      "target_tokens": 12000000,
      "ratios": { "teacher": 0.70, "real": 0.15, "template": 0.15 },
      "split":  { "train": 0.98, "val": 0.01, "test": 0.01 }
    }
  }
}
```

**경로 리터럴은 코드에 0개다.** 모든 경로는 `data_root`에서 파생된다:
`cfg.raw("seed")` → `data/raw/seed.jsonl`. 스테이지가 자기 출력 경로를 아는
것이 아니라 config가 알려 준다.

두 교사의 `base_url`이 같은 것은 오타가 아니다. 순차 적재이므로 같은 포트에
시점만 다르게 뜬다. **어느 쪽이 떠 있는지는 요청의 `model` 필드가 판정한다** —
vLLM은 자기가 서빙하지 않는 모델 id를 받으면 404로 거절하므로, 잘못된 서버에
붙었을 때 조용히 다른 모델로 생성되는 일이 구조적으로 불가능하다. `backend.py`는
이 404를 "서버에 다른 모델이 떠 있다"는 메시지로 바꿔 사람에게 보여 준다.

`filter`의 `max_utterance_chars: 35`는 `personas/mongle.md`의 "한 발화는 대체로
4~35글자"에서 온 값이다. 페르소나 문서가 근거이고 config가 그 값을 집행한다.

## 6. 페르소나 단일 출처

`personas/mongle.md`가 유일한 출처다. `persona_sft_data/persona.py`가 이 문서를 파싱해
아래를 뽑아낸다.

| 추출 대상 | 출처 절 |
| --- | --- |
| 이름, 말투, 응답 길이 규칙 | `## 핵심 정의` 표 |
| 발화 원칙 6개 | `## 발화 원칙` 번호 목록 |
| 감정별 선호 어휘 | `## 감정 표현과 어휘` 표 |
| 금지 표현 | `## 하지 않는 말과 행동` 목록 |
| 상황 15종 | `## 다룰 상황` 번호 목록 |
| 고정 프리앰블 | `## 고정 프리앰블 대화`의 ```text 블록 |

`prompts.py`가 이것을 교사 프롬프트로 조립한다. **파이썬 소스에 `몽글`이라는
문자열이 나타나면 안 된다.** 테스트로 강제한다: `persona_sft_data/**/*.py`를 읽어
페르소나 이름이 등장하면 실패시킨다. 현재 32회 등장하는 `persona.py`가 0회가
되는 것이 이 테스트의 목표다.

페르소나 문서가 "동결"이므로 파서는 문서가 바뀌면 소리를 내야 한다. 필수 절이
없거나 표 모양이 다르면 예외를 던진다 — 조용히 빈 값을 반환하지 않는다.

## 7. 데이터 레이아웃과 스키마

### 7.1 레이아웃

9개 디렉터리를 3개로 줄인다.

```
data/
  raw/        seed.jsonl  teacher.jsonl  real.jsonl  template.jsonl
              (+ 각각의 .stats.json, .rejected.jsonl)
  filtered/   같은 이름들, 게이트 통과분
  final/      corpus.jsonl  train.jsonl  val.jsonl  test.jsonl  manifest.json
```

구 파이프라인의 산출물 1.5GB는 `data/archive-p1/`으로 격리했다(2026-09-04).
**`data/`는 gitignore이므로 그것을 지우면 되돌릴 수 없다.** 새 코퍼스로 학습한
모델이 val loss 0.6515를 넘는 것이 확인되면 비교 대상으로 남길 이유가 사라진다.
그때까지는 위 세 디렉터리 옆에 임시로 공존한다. 내용과 삭제 방법은
`data/archive-p1/README.md`에 있다.

일회성 실험 디렉터리는 만들지 않는다. 실험이 필요하면 `data_root`를 다른 곳으로
가리키는 config를 쓴다 — `configs/smoke.json`이 `data_root: "data/smoke"`와
작은 `limit`을 갖는 식이다. **디렉터리가 아니라 config가 실험을 구분한다.**

### 7.2 레코드 스키마

현재 코퍼스의 스키마를 유지한다. 학습기와 토크나이저가 이미 이 모양을 읽고
있으므로 바꿀 이유가 없다.

```json
{
  "id": "teacher-expand-231809",
  "source": "teacher_expand",
  "scenario": "심심함과 놀이",
  "generator": ["kakaocorp/kanana-2-3b-instruct"],
  "license": "synthetic",
  "turns": [
    {"role": "user", "text": "심심하지?"},
    {"role": "pet",  "text": "응, 좀 심심했어."}
  ],
  "split": "train"
}
```

`seed_id` · `context_index` · `real_source` · `source_url` 같은 스테이지별
필드는 선택으로 허용한다. `schema.validate_session()`이 강제하는 것은 필수
필드의 존재, `turns`가 `user`로 시작해 `pet`으로 끝나며 번갈아 나오는 것,
그리고 텍스트가 비어 있지 않은 것이다.

### 7.3 재현성

`assemble`이 쓰는 `final/manifest.json`은 현재 형식을 유지하되 두 가지를
더한다.

- `config`: 사용한 config 전체를 그대로 박아 넣는다.
- `stages`: 각 스테이지의 `stats.json` 요약(입출력 수, 모델 id, 소요 시간).

manifest 하나로 "이 코퍼스가 어떤 모델·비율·시드에서 나왔는가"에 답할 수 있어야
한다. 지금은 답할 수 없다.

## 8. 품질 게이트

`filter` 스테이지 하나로 모은다. 현재 `filter.py` · `quality.py` ·
`pair_quality.py` · `policy.py` · `text_rules.py` · `validate_*` 4종에 흩어진
것을 합친다.

게이트는 두 종류로만 나눈다.

1. **구조 게이트** — 스키마, 턴 교대, 길이 범위, 중복 세션. 규칙 기반이고
   설정값은 config에서 온다.
2. **페르소나 게이트** — 존댓말 종결(`~요` `~습니다` `~세요`), 자기를 AI라고
   말하기, 이모지·마크다운, 같은 문장 연속 반복. **규칙은
   `personas/mongle.md`의 `## 하지 않는 말과 행동`에서 파생되며 코드에 새로
   쓰지 않는다.**

`paraphrase_bank.py` · `curate_paraphrase_bank.py` ·
`validate_paraphrase_bank.py` 698줄은 **삭제한다.** 미리 만든 패러프레이즈
은행에서 조합하던 것은 약한 교사를 우회하는 방법이었고, kanana에 직접 물으면
된다.

`human_quality_gate.py` · `apply_manual_review.py` 448줄도 **삭제한다.**
대신 `filter`가 통과분에서 무작위 표본 200개를 `filtered/<name>.sample.jsonl`로
떨궈 사람이 눈으로 볼 수 있게 한다. 검토 결과를 코퍼스에 되먹이는 자동 경로는
만들지 않는다 — 한 번도 반복해서 쓰이지 않은 기능이다.

## 9. 만들지 않는 것

- **vLLM 서버 자동 기동·모델 교체.** 파이프라인이 WSL 프로세스를 관리하지
  않는다. 사람이 서버를 띄우고, 파이프라인은 붙을 뿐이다. 실패하면 붙지 못했다고
  명확히 말한다.
- **재시도·체크포인트 재개.** 스테이지는 처음부터 다시 돈다. `limit`으로
  작게 잘라 돌리는 것이 재개 로직보다 싸다.
- **다중 GPU·분산 생성.** GPU 한 장이다.
- **교사 응답 캐시.** 같은 프롬프트를 두 번 보내지 않는 구조이므로 캐시가
  맞을 일이 없다.
- **검토 결과 되먹임 자동화** (§8).

## 10. 구현 순서와 병렬화

기반이 먼저 서야 스테이지들이 그것을 import할 수 있다.

**1차 (순차, 기반).** `config.py` → `schema.py` → `persona.py` →
`backend.py` → `prompts.py`. 서로 의존하므로 나눌 수 없다.

**2차 (병렬, 서브에이전트).** 6개 스테이지는 기반 위에서 서로 독립이다.
`seed` · `expand` · `real` · `template` · `filter` · `assemble`을 병렬로
구현한다. 각자 자기 `tests/test_gen_<stage>.py`를 함께 낸다.

**3차 (순차).** `cli.py`, 구 파일 삭제, `configs/smoke.json`으로 소량
엔드투엔드 실행, 문서 갱신.

교사 호출이 필요한 테스트는 `backend.py`의 인터페이스를 가짜로 채워 돌린다.
**단위 테스트는 GPU도 WSL도 요구하지 않는다.**

## 11. 위험과 미확인 사항

**측정하지 않은 것을 측정한 것처럼 쓰지 않기 위해 명시한다.**

| 항목 | 상태 | 영향 |
| --- | --- | --- |
| WSL2에 vLLM 설치 | ✅ **vLLM 0.28.0 / torch 2.13.0+cu130** | 절차는 [WSL vLLM 문서](../../setup/wsl-vllm.md)에 있다 |
| RTX 5090(Blackwell, sm_120) 지원 | ✅ `arch_list`에 `sm_120` 포함 | — |
| AWQ 가중치 다운로드 | ✅ 16.6GiB | — |
| vLLM의 `deepseek_v3` MoE + W4A16 | ✅ **적재 15.95GiB**, `MARLIN` WNA16 MoE + `TRITON_MLA` 선택 | 예측 16.6보다 낫다. KV 캐시에 약 16GiB가 남는다 |
| 커뮤니티 AWQ 가중치의 건전성 | ✅ **생성 품질로 확인** | 아래 품질 측정 참조 |
| 30B-A3B 처리량 | ✅ **20건 동시 0.77초, 201.5 tok/s** | 짧은 응답 기준. 긴 생성에서는 달라진다 |
| kanana의 한국어 반말 품질 | ⚠️ **좋으나 페르소나 위반이 남는다** | 아래 참조 |
| **vLLM HTTP 서버 접속** | ✅ **동작.** NAT 모드 + `HF_HUB_OFFLINE=1` | 동시 400건에서 **2,517 tok/s** |
| kanana **3B**의 품질 | ⏳ 미측정 | 물량 담당이므로 별도 측정이 필요하다 |
| 생성 소요 시간 | ✅ **한두 시간 규모** | 2,517 tok/s 기준 1,200만 토큰 약 80분. 균질한 요청 기준이라 자릿수 감각으로만 쓴다 |
| real 슬라이스 의미 다양성 | ⚠️ 두 소스 합쳐 약 3만 의미 | 1.8M 토큰을 채우면 반복이 남는다. 비율 조정은 결과를 보고 판단 |

### 11.0 스테이지 실측 (2026-09-04 스모크)

전체 체인을 한 번 돌려 얻은 수율이다. **단위 테스트로는 잡히지 않는 결함이
여기서 나왔다** — 가짜 교사는 언제나 올바른 형식을 돌려주기 때문이다.

| 스테이지 | 수율 | 비고 |
| --- | ---: | --- |
| seed | 89.8% | 30B-A3B |
| expand | **97.2%** | 처음 측정은 **1.8%**였다 |
| real | 85.0% | 소스 필터링은 거절과 분리해 센다 |
| template | 100% | 교사를 쓰지 않으므로 당연하다 |

`expand`의 1.8%는 전부 형식 오류였고 원인은 프롬프트였다. `seed_system`에만
하드 룰(첫 줄 `U:`, 길이, 한글 전용, 묘사 금지)을 넣고 `expand_system`과
`real_system`에는 옮기지 않았다. 세 단계를 거쳐 고쳤다.

1. 하드 룰을 세 프롬프트 모두에 → **66.3%**
2. `repair_dialogue` — 앞뒤 잘라내기만 → 52.8%로 **오히려 나빠졌다.** 결함의
   다수가 프레이밍이 아니라 **한 발화를 두 줄로 쪼갠 것**이었다. 같은 역할이
   연속되면 합치도록 바꿔 → **69.8%**
3. `expand_user`가 `위 대화를 다르게 표현해라` 한 줄뿐이라 모델이 펫 발화만
   고치고 사용자 쪽을 통째로 빠뜨렸다. 양쪽을 모두 쓰라고 명시 → **97.2%**

**교사 생성을 버리는 것과 고치는 것은 다르다.** 내용이 멀쩡한데 첫 줄이 틀렸다고
버리면 비싼 쪽을 버리는 것이다. 다만 `repair_dialogue`는 만들어 내지 않는다 —
자르고 합칠 뿐이고, 그래도 교대가 안 맞으면 게이트로 넘긴다.

#### 소스 필터링은 거절이 아니다

`real`이 처음에 수율 0.2%를 보고했다. 페르소나 범위 밖 소스 21,197건을 게이트
거절과 같이 세고 있었기 때문이다. **실제로 생성한 것의 85%는 통과했다.**
품질을 말하려고 만든 통계가 품질을 왜곡하면 없느니만 못하므로,
`source_filtered`를 `rejected`와 분리했다.

### 11.0.1 본 실행 (2026-09-05, 6시간 23분)

`setup/overnight.sh`가 생성부터 학습까지 무인으로 완주했다.

| 스테이지 | 결과 | 수율 | 시간 |
| --- | ---: | ---: | ---: |
| seed (30B) | 105,503 | 89.4% | 84분 |
| expand (3B) | 228,390 | 72.2% | 193분 |
| real (3B) | 54,590 | 28.0% | 92분 |
| template | 250,000 | 100% | 29초 |
| **코퍼스** | **384,020 세션 / 21,526,659 토큰** | | |

`real`이 3,257에서 54,590 세션으로 **17배**가 됐다. 소스를 넷으로 늘린
효과다. 그래도 목표 15%에 못 미쳐 8.9%이고, `assemble`이 169만 토큰 부족을
보고한다 — 3B 답변의 72%가 35자 제한을 넘겨 버려진다.

토크나이저는 vocab 2048에 1.6392 chars/token, unknown 0.

### 11.0.2 하나의 버그가 만든 세 개의 오진

이 코퍼스에서 내린 진단 셋이 전부 틀렸고, **원인은 모두 같았다.**

손실 마스킹을 학습 루프에만 넣고 `evaluate_loss`에는 넣지 않았다. train은
펫 발화만, val은 전체 토큰을 쟀다. 서로 다른 것을 나란히 놓고 비교한 것이다.

| 관찰한 증상 | 내린 진단 | 실제 |
| --- | --- | --- |
| 파라미터 43%↑, 수용영역 50%↑에도 손실 불변 | 데이터가 병목 | 지표가 학습 대상과 다른 축을 봄 |
| train 1.94 vs val 3.59 | 과적합 | 두 숫자가 다른 것을 잼. 실제 격차는 0.06 |
| 22~27에폭에서 멈춤 | 손실이 1.9에서 정체 | 조기 종료가 틀린 신호로 일찍 끊음 |

지표를 맞추자 학습이 **200에폭까지** 이어지고 손실이 1.90 → **1.76**으로
내려갔다. 데이터를 4.1배로 늘려도 손실이 1%밖에 안 움직였던 것도 같은 이유다.

**교훈은 P0의 것과 같다.** 그때는 "구조적 검증이 수치적 검증이 아니다"였고,
이번은 "**학습이 최적화하는 것과 평가가 재는 것이 같아야 한다**"이다. 둘 다
측정 도구 자체가 틀렸는데 결과가 그럴듯해서 오래 갔다.

### 11.0.3 폭 대 깊이 — 답

지표를 고친 뒤 공정하게 비교한 결과다. 모두 약 102만 파라미터, window 128.

| 후보 | 층 | FFN | 수용영역 | val loss | 에폭 |
| --- | ---: | ---: | ---: | ---: | ---: |
| l5_ffn288 | 5 | 288 | 636 | **1.7643** | 197 |
| l6_ffn224 | 6 | 224 | 763 | 1.7662 | 196 |
| l4_ffn384 | 4 | 384 | 509 | 1.7693 | 207 |

**0.3% 안에 있다.** 8배 더 학습한 뒤에도 수용영역 509 → 763은 측정 가능한
이득을 주지 않는다. 차이가 잡음이라면 실기 지연으로 정하는 것이 맞고, 그러면
층이 가장 적은 `l4_ffn384`다.

윈도우 자체는 데이터가 정했다. 세션이 중앙값 40토큰·최대 252토큰이라 window
128이 99.1%를 통째로 담는 반면 256은 무관한 세션 5.5개를 한 학습 시퀀스에
넣는다.

### 11.1 품질 측정 — 자동 검사가 절반을 놓쳤다

30B-A3B AWQ에 한 줄짜리 시스템 프롬프트를 주고 20개 발화를 생성했다. 자동
검사(존댓말·"나는 AI야")는 **위반 0/20**을 보고했다. **출력을 직접 읽으니
최소 5건이 위반이었다.**

| 출력 | 위반 |
| --- | --- |
| `몽글이도 잘래 🐾` | 이모지 금지 |
| `몽글몽글 기달몽!`, `깜짝 놀랐몽!` | "조사나 종결어미를 일부러 틀리거나 유아어를 남발하지 않는다" |
| `몽글도 이제 졸려`, `몽글이도 잘 잤어!` | 자기를 3인칭으로 부름 — 친구 관계 설정과 어긋난다 |

**이것이 §8의 설계 근거다.** 페르소나 게이트 규칙을 즉석에서 쓰면 절반을
빠뜨린다. 규칙은 `personas/mongle.md`의 금지 목록에서 기계적으로 파생해야 하고,
사람이 고른 몇 개를 코드에 적어서는 안 된다. 프롬프트도 마찬가지로
`prompts.py`가 페르소나 문서 전체에서 조립한다(§6).

기본 문체·길이는 좋다. 응답 길이 5~15자로 목표 4~35 안에 있고 반말이 자연스럽다.
남은 위반은 프롬프트 개선과 `filter` 게이트로 잡을 성격이다.

### 11.2 HTTP 경계 — 해결됨

§3.2가 가정한 **Windows 파이썬 → WSL vLLM** 경로가 실제로 동작한다. 조건은 두 가지다.

1. **NAT 네트워킹으로 서빙한다.** mirrored 모드에서는 vLLM만 루프백으로 닿지
   않는다(같은 순간 파이썬 `http.server`는 200). 두 모드가 반대로 고장 나 있어
   설치·다운로드는 mirrored, 서빙은 NAT으로 나눈다. `setup/wsl_net_mode.ps1`
   이 전환한다.
2. **`HF_HUB_OFFLINE=1`.** vLLM은 캐시된 모델이어도 리비전을 Hub에 확인하려 하고,
   NAT은 그 SYN을 거절이 아니라 **버려서** 기동이 몇 분씩 매달린다. 실측
   `SYN-SENT ... -> 3.168.178.31:443`. `~/vllm-teacher-env.sh`에 들어 있다.

자세한 근거는 [WSL vLLM 문서](../../setup/wsl-vllm.md).

#### 처리량 — 배치를 키우면 HTTP 비용이 사라진다

처음엔 20개 프롬프트로만 재고 "HTTP가 오프라인보다 2.9배 느리다"고 적었다.
**그 결론은 틀렸다.** 응답이 5~15자뿐이라 요청당 오버헤드가 최대로 부각되는
조건이었을 뿐이다. 배치를 키워 다시 쟀다.

| 요청 수 | 동시 | `max_tokens` | 소요 | 생성 토큰 | 처리량 | 요청당 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 20 | 64 | 3.59초 | 988 | 275 tok/s | 180ms |
| 100 | 50 | 256 | 6.59초 | 5,736 | 871 tok/s | 66ms |
| 200 | 100 | 256 | 7.49초 | 11,206 | 1,497 tok/s | 37ms |
| **400** | **200** | **256** | **9.01초** | **22,690** | **2,517 tok/s** | **23ms** |

요청당 비용이 180ms에서 23ms로 떨어지고, 400건에서도 **곡선이 아직 포화되지
않았다.** HTTP 왕복 비용은 고정비라 배치가 커질수록 묻힌다. 따라서 §3.2의 HTTP
경계를 바꿀 이유가 없고, 파이프라인을 WSL 안으로 옮길 필요도 없다.

**교사 호출은 크게 묶어서 보낸다.** `TeacherConfig.concurrency`의 기본값을
16/64로 잡았던 것은 이 측정 전이며, 200 이상이 맞다(§5).

##### 생성 시간 추정

2,517 tok/s를 기준으로 하면 teacher 슬라이스 840만 토큰이 **약 56분**,
1,200만 토큰 전체가 약 80분이다. 3B 교사는 더 빠르므로 `expand`는 이보다 짧다.
**다만 이는 `max_tokens=256`인 균질한 요청의 값이고, 실제 파이프라인은 프롬프트가
길고 거절·재시도가 섞이므로 그대로 받지 않는다.** 자릿수 감각으로만 쓴다 —
수 시간이 아니라 한두 시간 규모다.

#### 프롬프트가 위반을 절반 이하로 줄인다

§11.1의 한 줄짜리 프롬프트는 위반 5/20을 냈다. 이모지 금지와 3인칭 금지를 명시한
프롬프트로 바꾸니 **1~2/20**이 됐다. 다만 응답이 퉁명스러워져 20건 중 `싫어.`가
3회 나왔다. 금지를 나열하면 모델이 거절로 수렴한다는 뜻이므로, `prompts.py`는
금지 목록만이 아니라 §6의 선호 어휘와 발화 원칙을 함께 넣어야 한다.
