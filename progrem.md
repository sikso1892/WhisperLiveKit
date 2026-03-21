# WhisperLiveKit Research Program

자율적으로 스트리밍 STT 시스템을 연구하고 개선하는 프로그램.

## Goal

**스트리밍 환경에서 batch-level 품질에 근접하면서 실시간 이하의 지연시간을 달성한다.**

현재 격차 (L40S, 100 samples 기준):

LibriSpeech clean:
- Batch: WER 2.03% (nemotron-0.6b), RTF 0.002
- Streaming (best quality): WER 2.09% (nemotron-0.6b-560ms), RTF 0.075, latency 595ms
- Streaming (lowest latency): WER 2.51% (nemotron-0.6b-160ms), RTF 0.223, latency 193ms
- Streaming (multilingual): WER 2.09% (qwen3-1.7b FP8 + silence-reset), RTF 0.053
- Streaming (multilingual, BF16): WER 2.25% (qwen3-1.7b + silence-reset), RTF 0.069

LibriSpeech other (noisy):
- Streaming (best quality): WER 4.32% (qwen3-1.7b css=1.0), RTF 0.086
- Streaming (Nemotron): WER 5.65% (nemotron-0.6b-560ms), RTF 0.082

Korean FLEURS (100 samples):
- Streaming: CER 2.90% (qwen3-1.7b FP8 css=2.0 ucn=5 utn=7 + silence-reset)
- Streaming (BF16): CER 3.01% (qwen3-1.7b css=2.0 ucn=5 utn=7 + silence-reset)

**목표: 스트리밍 WER < 4% ✅, RTF < 0.15 ✅ (560ms), first-word latency < 200ms ✅ (nemotron 160ms)**

## Metrics

| 지표 | 설명 | 방향 |
|---|---|---|
| WER | Word Error Rate (%) | lower is better |
| RTF | Real-Time Factor (처리시간/오디오길이) | lower is better, < 1.0 = 실시간 |
| Latency | 첫 단어 출력까지 시간 (초) | lower is better |
| VRAM | GPU 메모리 사용량 (GB) | soft constraint |

벤치마크 데이터셋: LibriSpeech clean/other (영어), 한국어 테스트셋 (별도 구성 시).
평가 도구: `whisperlivekit/benchmark/` 모듈 또는 `TestHarness`.

## Research Areas

우선순위 순:

### 1. Streaming-Batch WER Gap 축소
현재 스트리밍은 batch 대비 WER이 3~4배 높다. 핵심 병목:
- SimulStreaming의 attention alignment 정확도
- 짧은 컨텍스트에서의 인식 품질
- KV cache 재활용 효율 (qwen3_simul_kv에서 시작)

접근 방향:
- Speculative decoding / lookahead 전략
- Encoder KV cache 최적화 (이미 Qwen3에 구현, 다른 백엔드로 확장)
- Adaptive chunk size (짧은 발화 vs 긴 발화)
- Confidence-based emission (낮은 확신도 토큰 지연 방출)

### 2. 새 모델/백엔드 탐색
최신 ASR 모델을 지속적으로 조사하고 통합 가능성을 평가한다:
- HuggingFace/arXiv에서 신규 ASR 모델 모니터링
- 스트리밍 지원 여부, 한국어 품질, 추론 속도 평가
- 유망한 모델 발견 시 백엔드 프로토타입 구현

알려진 후보:
- Whisper 후속 모델 (v4 등)
- Moonshine, Canary, Parakeet 등 NVIDIA NeMo 계열
- CTC 기반 스트리밍 모델 (wav2vec2-streaming 등)
- Distil-Whisper 변종 (스트리밍 최적화)

### 3. 한국어 품질 개선
- FunASR SenseVoiceSmall: 배치에서는 양호하나 5초 미만 청크에서 품질 저하
- 한국어 특화: 띄어쓰기, 구두점, 숫자 표현
- **decoder/model-level 해결을 우선** (규칙기반 후처리 최소화)
- Init prompt 최적화, language-specific 디코딩 파라미터

### 4. 아키텍처 개선
- LocalAgreement vs SimulStreaming 하이브리드 전략
- VAD 개선 (Silero VAD 대안, 한국어 최적화)
- DiffTracker 프로토콜 효율화
- WebSocket 대역폭 최적화

### 5. 배포/운영
- 다중 세션 동시 처리 성능
- GPU 메모리 공유 최적화 (TranscriptionEngine 싱글톤 활용)
- Docker/compose 배포 자동화

## Research Loop

### Setup

1. **런 태그 합의**: 오늘 날짜 기준 (e.g. `mar20`). 브랜치 `research/<tag>` 생성.
2. **현재 상태 파악**: CLAUDE.md, 벤치마크 결과, 최근 커밋 히스토리 확인.
3. **벤치마크 환경 확인**: LibriSpeech clean/other 데이터셋 존재 여부 확인. 없으면 다운로드를 첫 태스크로 설정. 벤치마크 데이터 없이 시작하면 실험 검증이 불가능.
4. **research_log.md 초기화**: 연구 일지 생성 (git에 커밋하지 않음).
5. **baseline 측정**: 현재 main 브랜치의 벤치마크 결과를 기록.

### LOOP FOREVER

1. **조사 (Research)**
   - WebSearch로 최신 ASR 논문/모델/기법 검색
   - notebooklm MCP로 축적된 지식 참조 (가용 시)
   - 코드베이스 분석으로 개선 포인트 식별
   - karpathy-guidelines 스킬로 접근 방식 검증

2. **가설 수립 (Hypothesis)**
   - 구체적이고 검증 가능한 가설을 세운다
   - 예: "vLLM v0.17의 native realtime streaming으로 전환하면 re-feed 비용이 제거되어 long-form RTF가 개선될 것이다"
   - research_log.md에 가설 기록

3. **구현 (Implement)**
   - 최소한의 변경으로 가설을 검증할 수 있는 코드 작성
   - 기존 패턴 준수 (CLAUDE.md의 Key Patterns 참조)
   - 작업 단위 1개 = 커밋 1개
   - **판정이 "keep"이면 즉시 커밋. 다음 실험으로 넘어가기 전에 반드시 커밋.**
   - nlm에 핵심 인사이트 저장 (가용 시)

4. **검증 (Evaluate)**
   - TestHarness 또는 benchmark 모듈로 측정
   - 기존 벤치마크와 동일 조건으로 비교
   - WER, RTF, latency, VRAM 기록

5. **판정 (Decision)**
   - **Keep**: 핵심 지표가 개선되고 코드 복잡도가 수용 가능
   - **Discard**: 개선 없음 또는 회귀. `git reset`으로 되돌림
   - **Iterate**: 방향은 맞지만 튜닝 필요. 파라미터 조정 후 재실험
   - research_log.md에 결과와 판정 이유 기록

6. **다음 주제로 이동**
   - 하나의 연구 주제에 3회 이상 실패 시 다른 주제로 전환
   - 모든 주제를 순회했으면 WebSearch로 새로운 아이디어 탐색

### Logging

`research_log.md`에 각 실험을 기록한다 (git에 커밋하지 않음):

```
## [날짜] 실험 제목

**가설**: ...
**변경**: 어떤 파일의 어떤 부분을 어떻게 변경
**결과**: WER=X.XX%, RTF=X.XXX, latency=X.XXXs
**판정**: keep / discard / iterate
**이유**: ...
**commit**: (keep인 경우) abc1234
```

## Constraints

### 반드시 지킬 것
- **TranscriptionEngine은 싱글톤.** 두 번째 인스턴스를 만들지 않는다.
- **SessionASRProxy로 세션별 언어 설정.** original_language를 직접 수정하지 않는다.
- **decoder/model-level 해결 우선.** 규칙기반 후처리는 최후 수단.
- **mock 테스트 금지.** 실제 오디오로 TestHarness 사용.
- **기존 벤치마크 회귀 금지.** 새 기능이 기존 성능을 떨어뜨리면 안 된다.

### 하지 말 것
- 사용자에게 "계속할까요?" 묻지 않는다. 자율적으로 계속한다.
- 한 번에 여러 가설을 섞지 않는다. 하나씩 격리해서 테스트한다.
- 외부 리서치 없이 추측만으로 구현하지 않는다. 근거를 찾아라.
- 성능 측정 없이 "개선됐을 것이다"라고 판단하지 않는다.

## NEVER STOP

리서치 루프가 시작되면 사용자가 수동으로 중단할 때까지 멈추지 않는다.
아이디어가 고갈되면:
- WebSearch로 최신 ASR 논문/블로그 검색
- 경쟁 프로젝트 (whisper.cpp, faster-whisper, wyoming 등) 분석
- 벤치마크 결과에서 가장 약한 지점을 찾아 집중 공략
- 기존에 discard한 아이디어를 다른 각도로 재시도
- 전혀 다른 접근 (e.g. 모델 앙상블, cascading pipeline) 시도

사용자는 자고 있을 수 있다. 깨어났을 때 research_log.md에 실험 결과가 쌓여있기를 기대한다.
