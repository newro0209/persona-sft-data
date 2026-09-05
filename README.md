# persona-sft-data

페르소나 문서 하나와 교사 모델 하나로 **LLM 페르소나 미세조정 데이터셋**을 만든다.
페르소나가 무엇을 말하고 무엇을 말하지 않는지는 `personas/<이름>.md` 한 곳에만
있다. 코드에는 페르소나 문자열이 없고, 테스트가 그것을 강제한다.

지금 들어 있는 페르소나는 `몽글` — 작은 반려 펫, 항상 반말, 한 발화 4~35자.
그 데이터셋은 **384,020개 대화 / 1,901,432턴**이고 `datasets/mongle-v1/`에
OpenAI `messages` 포맷으로 내보내진다. 원래는 [my-llm](../my-llm)의 ESP32용
소형 모델을 학습시키려고 만든 파이프라인이며, 그쪽은 여기서 나온 코퍼스를
받아 자기 토크나이저로 패킹한다.

## 빠르게

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -e ".[dev,real]"
.venv\Scripts\python.exe -m pytest                                        # 90개, 1초

.venv\Scripts\python.exe -m persona_sft_data check  --config configs/mongle.json   # 설정·페르소나·교사 점검
.venv\Scripts\python.exe -m persona_sft_data run    --config configs/mongle.json   # 전부 (교사 서버 필요, ~6시간)
.venv\Scripts\python.exe -m persona_sft_data export --config configs/mongle.json   # 조립된 코퍼스 → 데이터셋 (10초)
```

`run --stage seed|expand|real|template|filter|assemble|export`로 한 단계만 돌릴 수
있다. 교사 없이 파이프라인 전체를 돌려 보려면 `configs/smoke.json`
(`FakeTeacher`, 몇 분).

## 파이프라인

```
personas/mongle.md ─┐
                    ├─ seed ──── 추론 교사(30B)가 상황(59 beat)마다 대화를 새로 씀
                    ├─ expand ── 대량 교사(3B)가 seed 하나당 변주 3개
                    ├─ real ──── 공개 한국어 대화 4종의 사용자 발화에 교사가 페르소나로 답함
                    └─ template  페르소나 어휘표만으로 조합 (교사 없음)
                          │
                       filter ── 페르소나 문서에서 파생된 게이트: 존댓말, 길이, AI 자칭, 역할표기, 반복 …
                          │
                       assemble  비율 혼합(teacher 0.7 / real 0.15 / template 0.15) · 세션 단위 분할 · manifest
                          │
                       export ── messages JSONL + system_prompt.txt + manifest.json + 데이터 카드
```

- **세션**이 내부 단위다: `{id, source, scenario, license, generator, turns:[{role:user|pet, text}]}`.
  출처와 라이선스가 레코드마다 붙어 다니므로, 나중에 어떤 소스를 빼고 싶으면
  필터 한 줄이다.
- **모든 경로는 `data_root` 하나에서 파생된다.** `data/raw/`는 생성기가 낸 것,
  `data/filtered/`는 게이트를 통과한 것, `data/final/`은 혼합·분할된 코퍼스.
  단계마다 `.stats.json`(수율·거부 사유), `.rejected.jsonl`(버린 것 전부),
  `.sample.jsonl`(사람이 읽을 200개)이 같이 나온다.
- **게이트는 문서에서 나온다.** 금지 표현, 길이 범위(`4~35글자`), 상황 목록은
  `personas/mongle.md`를 파싱해서 쓴다. 문서가 바뀌면 게이트도 바뀌고, 문서
  형태가 깨지면 빈 규칙으로 조용히 넘어가지 않고 파서가 실패한다.
- **교사는 설정에만 있다.** 모델 id가 코드에 나타나면 테스트가 실패한다.
  vLLM의 OpenAI 호환 엔드포인트면 무엇이든 되고, `FakeTeacher`로 교사 없이도
  전체 흐름이 돈다.

## 내보내기 포맷

`datasets/<name>/{train,val,test}.jsonl`, 한 줄에 한 대화:

```json
{"id": "...", "source": "teacher_seed", "scenario": "배고픔", "license": "synthetic",
 "generator": ["..."],
 "messages": [
   {"role": "system",    "content": "이름: 몽글\n정체성: ...\n\n발화 원칙:\n1. ...\n\n하지 않는 말과 행동:\n- ..."},
   {"role": "user",      "content": "배고파?"},
   {"role": "assistant", "content": "응, 꼬르륵. 밥 줘."}
 ]}
```

시스템 프롬프트는 페르소나 문서의 핵심 정의 표·발화 원칙·금지 목록을 그대로
평문으로 옮긴 것이다(`Persona.system_prompt()`). 프롬프트용으로 따로 쓴 문장이
없으므로, 모델이 학습하는 정의와 코퍼스를 생성·검열한 정의가 같은 것이다.

Axolotl · LLaMA-Factory · TRL이 그대로 읽는다. 함께 나오는 `manifest.json`에
분할별 개수·sha256, 소스별 라이선스, 생성 모델, 페르소나 문서 해시가 있고,
`README.md`는 그것을 Hugging Face 데이터 카드로 렌더링한 것이다.

## 새 페르소나

1. `personas/<이름>.md`를 `mongle.md`와 같은 절 구성으로 쓴다 — `## 핵심 정의`
   (표), `## 발화 원칙`(번호 목록), `## 감정 표현과 어휘`(표), `## 하지 않는 말과
   행동`(불릿), `## 다룰 상황`(번호 목록), `## 고정 프리앰블 대화`(코드 블록).
   빠진 절은 파서가 거부한다.
2. `configs/<이름>.json`을 `mongle.json`에서 복사해 `persona_doc`과
   `stages.export.name`을 바꾼다.
3. `check` → `run` → `export`.

## 교사 서버

WSL2의 vLLM으로 `kakaocorp/kanana-2-3b-instruct`(대량)와 30B MoE의 AWQ w4a16
(추론)을 번갈아 띄웠다. 재현 가능한 한 줄 설치가 `setup/wsl_vllm_setup.sh`,
겪은 문제와 해법의 표가 [docs/wsl-vllm.md](docs/wsl-vllm.md)에 있다 — 새로
시작하는 사람이 같은 함정을 다시 밟지 않도록 쓴 문서다. 무인 야간 실행은
`setup/overnight.sh`.

## 저장소 구조

```
persona_sft_data/     파이프라인 (표준 라이브러리만)
  config.py           PipelineConfig — 모든 경로·모델·비율의 유일한 출처
  persona.py          페르소나 문서 파서 + system_prompt()
  backend.py          OpenAI 호환 교사 클라이언트, FakeTeacher
  prompts.py          교사 프롬프트 (규칙은 페르소나에서 렌더링)
  gates.py            품질 게이트
  runner.py           단계 계약: 정규화 → 중복 제거 → 게이트 → 통계
  stages/             seed · expand · real · template · filter · assemble · export
personas/             페르소나 문서 (단일 진실)
configs/              mongle.json (본 실행), smoke.json (교사 없이)
setup/, docs/         vLLM 교사 서버 구축, 트러블슈팅, 파이프라인 설계 스펙
tests/                90개. 페르소나 문자열·모델 id·data/ 리터럴이 코드에 있으면 실패
data/                 (gitignore) raw · filtered · final · cache · smoke
datasets/             (gitignore) export 결과
```

## 이 코퍼스의 수치

| | |
| --- | --- |
| 세션 / 턴 | 384,020 / 1,901,432 |
| 분할 | train 360,980 · val 11,520 · test 11,520 (세션 단위) |
| 소스 | teacher_expand 173,605 · teacher_seed 79,991 · template 75,834 · real 54,590 |
| 교사 호출 | seed 118,000 (수율 89%) · expand 316,509 (72%) · real 194,737 (28%) |
| 생성 시간 | 약 6시간 (RTX 5090 하나, vLLM) |

`real`의 수율 28%는 3B 교사의 답 72%가 35자 규칙을 넘겨서다. 그래서 `real`은
목표 15%에 못 미치는 약 9%다 — 게이트를 풀지 않고 남겨 둔 미해결 문제다.
