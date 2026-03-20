# CLAUDE.md -- WhisperLiveKit

@progrem.md

## Always Reference

- **andrej-karpathy-skills:karpathy-guidelines** skill: 코드 작성, 리뷰, 리팩토링 시 항상 이 스킬을 참조하여 과도한 복잡성을 피하고, 수술적 변경을 하며, 가정을 명시하고, 검증 가능한 성공 기준을 정의할 것.
- **notebooklm MCP**: 지식 관리, 리서치, 문서 정리 작업 시 notebooklm MCP 서버를 활용할 것. (서버 미연결 시 WebSearch/WebFetch로 대체)
- **외부 리서치**: 논문, 기법, 라이브러리 문서 등 외부 자료가 필요할 때 **WebSearch**로 검색하고 **WebFetch**로 상세 내용을 확인할 것. 최신 정보가 필요한 경우 항상 외부 검색을 우선 수행.

## Build & Test

Install for development:

```sh
pip install -e ".[test]"
```

Test with real audio using `TestHarness` (requires models + audio files):

```python
import asyncio
from whisperlivekit import TestHarness

async def main():
    async with TestHarness(model_size="base", lan="en", diarization=True) as h:
        await h.feed("audio.wav", speed=1.0)     # feed at real-time
        await h.drain(2.0)                         # let ASR catch up
        h.print_state()                            # see current output

        await h.silence(7.0, speed=1.0)            # 7s silence
        await h.wait_for_silence()                 # verify detection

        result = await h.finish()
        print(f"WER: {result.wer('expected text'):.2%}")
        print(f"Speakers: {result.speakers}")
        print(f"Text at 3s: {result.text_at(3.0)}")

asyncio.run(main())
```

## Architecture

WhisperLiveKit is a real-time speech transcription system using WebSockets.

- **TranscriptionEngine** (singleton) loads models once at startup and is shared across all sessions.
- **AudioProcessor** is created per WebSocket session. It runs an async producer-consumer pipeline: FFmpeg decodes audio, Silero VAD detects speech, the ASR backend transcribes, and results stream back to the client.
- Two streaming policies:
  - **LocalAgreement** (HypothesisBuffer) -- confirms tokens only when consecutive inferences agree.
  - **SimulStreaming** (AlignAtt attention-based) -- emits tokens as soon as alignment attention is confident.
- 6 ASR backends: WhisperASR, FasterWhisperASR, MLXWhisper, VoxtralMLX, VoxtralHF, Qwen3.
- **SessionASRProxy** wraps the shared ASR with a per-session language override, using a lock to safely swap `original_language` during `transcribe()`.
- **DiffTracker** implements a snapshot-then-diff protocol for bandwidth-efficient incremental WebSocket updates (opt-in via `?mode=diff`).

## Key Files

| File | Purpose |
|---|---|
| `config.py` | `WhisperLiveKitConfig` dataclass -- single source of truth for configuration |
| `core.py` | `TranscriptionEngine` singleton, `online_factory()`, diarization/translation factories |
| `audio_processor.py` | Per-session async pipeline (FFmpeg -> VAD -> ASR -> output) |
| `basic_server.py` | FastAPI server: WebSocket `/asr`, REST `/v1/audio/transcriptions`, CLI `wlk` |
| `timed_objects.py` | `ASRToken`, `Segment`, `FrontData` data structures |
| `diff_protocol.py` | `DiffTracker` -- snapshot-then-diff WebSocket protocol |
| `session_asr_proxy.py` | `SessionASRProxy` -- thread-safe per-session language wrapper |
| `parse_args.py` | CLI argument parser, returns `WhisperLiveKitConfig` |
| `test_client.py` | Headless WebSocket test client (`wlk-test`) |
| `test_harness.py` | In-process testing harness (`TestHarness`) for real E2E testing |
| `local_agreement/online_asr.py` | `OnlineASRProcessor` for LocalAgreement policy |
| `simul_whisper/` | SimulStreaming policy implementation (AlignAtt) |

## Key Patterns

- **TranscriptionEngine** uses double-checked locking for thread-safe singleton initialization. Never create a second instance in production. Use `TranscriptionEngine.reset()` in tests only to switch backends.
- **WhisperLiveKitConfig** dataclass is the single source of truth. Use `from_namespace()` (from argparse) or `from_kwargs()` (programmatic). `parse_args()` returns a `WhisperLiveKitConfig`, not a raw Namespace.
- **online_factory()** in `core.py` routes to the correct online processor class based on backend and policy.
- **FrontData.to_dict()** is the canonical output format for WebSocket messages.
- **SessionASRProxy** uses `__getattr__` delegation -- it forwards everything except `transcribe()` to the wrapped ASR.
- The server exposes `self.args` as a `Namespace` on `TranscriptionEngine` for backward compatibility with `AudioProcessor`.

## Adding a New ASR Backend

1. Create `whisperlivekit/my_backend.py` with a class implementing:
   - `transcribe(audio, init_prompt="")` -- run inference on audio array
   - `ts_words(result)` -- extract timestamped words from result
   - `segments_end_ts(result)` -- extract segment end timestamps
   - `use_vad()` -- whether this backend needs external VAD
2. Set required attributes on the class: `sep`, `original_language`, `backend_choice`, `SAMPLING_RATE`, `confidence_validation`, `tokenizer`, `buffer_trimming`, `buffer_trimming_sec`.
3. Register in `core.py`:
   - Add an `elif` branch in `TranscriptionEngine._do_init()` to instantiate the backend.
   - Add a routing case in `online_factory()` to return the appropriate online processor.
4. Add the backend choice to CLI args in `parse_args.py`.

## Testing with TestHarness

`TestHarness` wraps AudioProcessor in-process for full pipeline testing without a server.

Key methods:
- `feed(path, speed=1.0)` -- feed audio at controlled speed (0 = instant)
- `silence(duration, speed=1.0)` -- inject silence (>5s triggers silence detection)
- `drain(seconds)` -- wait for ASR to catch up without feeding audio
- `finish(timeout)` -- signal end-of-audio, wait for pipeline to drain
- `state` -- current `TestState` with lines, buffers, speakers, timestamps
- `wait_for(predicate)` / `wait_for_text()` / `wait_for_silence()` / `wait_for_speakers(n)`
- `snapshot_at(audio_time)` -- historical state at a given audio position
- `on_update(callback)` -- register callback for each state update

`TestState` provides:
- `text`, `committed_text` -- full or committed-only transcription
- `speakers`, `n_speakers`, `has_silence` -- speaker/silence info
- `line_at(time_s)`, `speaker_at(time_s)`, `text_at(time_s)` -- query by timestamp
- `lines_between(start, end)`, `text_between(start, end)` -- query by time range
- `wer(reference)`, `wer_detailed(reference)` -- evaluation against ground truth
- `speech_lines`, `silence_segments` -- filtered line lists

## OpenAI-Compatible REST API

The server exposes an OpenAI-compatible batch transcription endpoint:

```bash
# Transcribe a file (drop-in replacement for OpenAI)
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.mp3 \
  -F response_format=verbose_json

# Works with the OpenAI Python client
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
result = client.audio.transcriptions.create(model="whisper-1", file=open("audio.mp3", "rb"))
print(result.text)
```

Supported `response_format` values: `json`, `verbose_json`, `text`, `srt`, `vtt`.
The `model` parameter is accepted but ignored (uses the server's configured backend).

## Agent Team Guidelines

### 팀 구성 원칙

- 작업 범위에 맞게 **최소한의 에이전트**로 구성한다. 2~3명이면 충분한 일에 5명을 투입하지 않는다.
- 각 에이전트는 **명확한 역할 경계**를 가진다. 역할이 겹치면 충돌이 발생한다.
- **Read-only 에이전트**(Explore, Plan)에게 구현 작업을 할당하지 않는다.
- 모든 팀원은 이 CLAUDE.md와 `@progrem.md`를 숙지한 상태로 작업한다.

### 권장 팀 구성

#### 1. 기능 개발 (Feature Development)

| 역할 | 이름 예시 | subagent_type | 담당 |
|---|---|---|---|
| lead | `lead` | (main) | 설계 결정, 태스크 분배, 코드 리뷰 |
| researcher | `researcher` | Explore | 코드베이스 탐색, 외부 자료 조사 (WebSearch/WebFetch/notebooklm) |
| coder | `coder` | (general) | 구현, 파일 수정 |
| tester | `tester` | (general) | TestHarness로 E2E 검증, 벤치마크 |

#### 2. 새 ASR 백엔드 추가

| 역할 | 담당 |
|---|---|
| researcher | 백엔드 API 문서 조사, 기존 백엔드 패턴 분석 |
| coder | `my_backend.py` 작성, `core.py`/`parse_args.py` 등록 |
| tester | TestHarness로 WER/RTF 벤치마크, 기존 백엔드와 비교 |

#### 3. 버그 수정 (Bug Fix)

| 역할 | 담당 |
|---|---|
| explorer | 로그/코드 분석으로 근본 원인 추적 |
| fixer | 수술적 수정 (karpathy-guidelines 준수) |

2명이면 충분하다. tester를 추가하는 것은 회귀 테스트가 필요할 때만.

#### 4. Autoresearch (progrem.md)

단독 에이전트가 자율적으로 실험 루프를 수행한다. 팀 구성이 불필요하다. `progrem.md`의 LOOP FOREVER 프로토콜을 따른다.

#### 5. 풀스택 기능 (Server + Frontend)

| 역할 | 담당 |
|---|---|
| backend | FastAPI/WebSocket 서버 수정 (`basic_server.py`, `audio_processor.py`) |
| frontend | `live_transcription.html` UI 수정 (frontend-design 스킬 활용) |
| tester | wlk-test 클라이언트 + 브라우저 동작 검증 |

### 팀 재편성 지침

#### 규모 축소 (Scale Down)

- 에이전트가 **대기 상태로 반복 전환**되면 해당 역할을 병합하거나 shutdown한다.
- 남은 작업이 **단일 도메인**에 집중되면 불필요한 역할을 정리한다.
- 목표: 유휴 에이전트 0. idle 상태가 지속되면 즉시 재편성.

#### 규모 확대 (Scale Up)

- 한 에이전트에 **블로킹 태스크가 누적**되면 역할을 분리한다.
- **독립적인 병렬 작업**이 3개 이상 식별되면 에이전트를 추가한다.
- 예: coder 1명이 backend + frontend를 모두 담당하고 있으면 분리.

#### 역할 전환 (Role Pivot)

- researcher의 조사가 끝나면 **tester로 전환**하거나 shutdown한다. 조사만 하고 앉아있지 않는다.
- 버그 수정 완료 후 fixer를 **coder로 재할당**하여 후속 개선 작업에 투입할 수 있다.

#### 재편성 트리거

다음 상황이 발생하면 lead는 팀 구조를 재검토한다:

1. **태스크 완료율 정체** — 30분 이상 진전 없음
2. **블로킹 의존성** — 에이전트 간 대기가 연쇄적으로 발생
3. **스코프 변경** — 사용자가 방향을 수정하거나 새로운 요구사항 추가
4. **에이전트 실패** — 동일 작업에서 3회 이상 실패 시 접근 방식 변경

#### 재편성 절차

1. 현재 TaskList 상태를 확인한다.
2. 완료/불필요한 에이전트에게 `shutdown_request`를 보낸다.
3. 필요시 새 에이전트를 spawn하고 미완료 태스크를 재할당한다.
4. 사용자에게 재편성 사유를 간단히 보고한다.

### 작업 단위 (Work Units)

모든 작업은 **원자적 단위(atomic unit)**로 분해하여 진행한다.

#### 작업 단위 정의

- 하나의 작업 단위 = **하나의 논리적 변경** = **하나의 커밋**
- 좋은 예: "SessionASRProxy에 timeout 파라미터 추가", "VAD threshold 설정 가능하게 변경"
- 나쁜 예: "여러 파일 리팩토링 + 새 기능 + 버그 수정" (3개 단위로 분리해야 함)

#### 작업 흐름

1. **태스크를 작업 단위로 분해** — TaskCreate로 각 단위를 등록
2. **단위별 구현** — 하나의 작업 단위를 완료할 때마다 커밋
3. **단위별 검증** — 커밋 전 해당 변경이 기존 기능을 깨뜨리지 않는지 확인
4. **다음 단위로 이동** — TaskUpdate로 완료 표시 후 다음 작업

### 커밋 워크플로우

#### 커밋 타이밍

- **작업 단위 1개 완료 시** 즉시 커밋한다. 여러 단위를 묶어서 커밋하지 않는다.
- 커밋 전 `git diff`로 의도하지 않은 변경이 포함되지 않았는지 반드시 확인한다.
- `.env`, credentials, 대용량 바이너리는 절대 커밋하지 않는다.

#### 커밋 메시지 규칙

```
<type>: <간결한 설명>

<필요시 상세 내용>

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
```

type: `feat`, `fix`, `refactor`, `test`, `docs`, `bench`, `chore`

#### 커밋 권한

- **팀 모드**: lead가 커밋을 총괄한다. coder/tester는 구현 후 lead에게 커밋 요청을 보낸다.
- **단독 모드**: 사용자의 명시적 승인 후 커밋한다.
- **autoresearch 모드**: `progrem.md` 프로토콜에 따라 자율 커밋 허용.

### 푸시 & PR 워크플로우

#### 브랜치 전략

- 기능 개발: `feat/<feature-name>` 브랜치에서 작업 후 main으로 PR
- 버그 수정: `fix/<issue-description>` 브랜치
- 실험: `autoresearch/<tag>` 브랜치 (progrem.md 참조)
- main에 직접 push하지 않는다. 항상 PR을 통해 머지한다.

#### PR 생성

1. 작업 단위들이 모여 **하나의 논리적 기능/수정**을 이루면 PR을 생성한다.
2. `gh pr create`로 생성하며, 제목은 70자 이내로 간결하게.
3. PR body에 포함할 내용:
   - **Summary**: 변경 사항 요약 (1~3 bullet)
   - **Changes**: 작업 단위별 커밋 목록
   - **Test plan**: 검증 방법 체크리스트
   - **Related**: 관련 이슈/문서 링크

#### PR 셀프 리뷰

PR 생성 후 lead(또는 담당 에이전트)가 **자동으로 셀프 리뷰**를 수행한다:

1. `gh pr diff <PR번호>`로 전체 diff 확인
2. 다음 체크리스트를 검증:
   - [ ] 불필요한 변경이 섞여 있지 않은가 (scope creep)
   - [ ] karpathy-guidelines 위반 — 과도한 복잡성, 불필요한 추상화가 없는가
   - [ ] TranscriptionEngine 싱글톤 패턴 위반이 없는가
   - [ ] original_language 직접 수정이 없는가 (SessionASRProxy 사용 여부)
   - [ ] 하드코딩된 값, 누락된 설정이 없는가
   - [ ] 보안 취약점 (injection, XSS 등)이 없는가
3. 문제 발견 시 `gh pr comment`로 인라인 코멘트를 남기고 수정한다.
4. 리뷰 통과 시 사용자에게 PR URL과 리뷰 요약을 보고한다.

#### 외부 PR 리뷰

사용자가 외부 PR 리뷰를 요청하면:

1. `gh pr view <PR번호>` + `gh pr diff <PR번호>`로 내용 파악
2. `gh api repos/{owner}/{repo}/pulls/{PR번호}/comments`로 기존 코멘트 확인
3. 위 셀프 리뷰 체크리스트 + 프로젝트별 규칙(Do NOT 섹션) 기준으로 리뷰
4. `gh pr review <PR번호> --comment --body "리뷰 내용"` 또는 개별 라인 코멘트

### 팀 간 규칙

- **TranscriptionEngine은 싱글톤이다.** 여러 에이전트가 동시에 `core.py`를 수정하지 않는다.
- **파일 충돌 방지**: 같은 파일을 2명 이상이 동시 수정하지 않는다. 태스크 단위로 파일 소유권을 명확히 한다.
- **커밋은 lead가 총괄한다.** 개별 에이전트가 독자적으로 git commit하지 않는다 (사용자가 명시적으로 허용하거나 autoresearch 모드인 경우 제외).
- **외부 리서치 결과 공유**: researcher가 찾은 자료는 SendMessage로 관련 에이전트에게 즉시 전달한다. 태스크 메모에만 남기지 않는다.
- **PR은 반드시 셀프 리뷰를 거친 후** 사용자에게 보고한다. 리뷰 없이 머지 요청하지 않는다.

## Do NOT

- Do not create a second `TranscriptionEngine` instance. It is a singleton; the constructor returns the existing instance after the first call.
- Do not modify `original_language` on the shared ASR directly. Use `SessionASRProxy` for per-session language overrides.
- Do not assume the frontend handles diff protocol messages. Diff mode is opt-in (`?mode=diff`) and ignored by default.
- Do not write mock-based unit tests. Use `TestHarness` with real audio for pipeline testing.
