# WhisperLiveKit Research Program

자율적으로 스트리밍 STT 시스템을 연구하고 개선하는 프로그램.

## Goal

**스트리밍 환경에서 batch-level 품질에 근접하면서 실시간 이하의 지연시간을 달성한다.**

현재 격차 (L40S 기준):

LibriSpeech clean (100 samples, H100 BF16):
- **Granite 1B batch: WER 1.18%, RTF 0.024** ← 최저 WER
- **Granite 1B stream css=2.0: WER 1.18%, RTF 0.077, fd 72ms**
- **Granite 1B stream css=4.0: WER 1.18%, RTF 0.042, fd 79ms** ← RTF 최적
- Granite 1B stream css=8.0: WER 1.18%, RTF 0.028, fd 110ms ← batch급 RTF
- **Granite 1B adaptive 2→8: WER 1.18%, RTF 0.033, fd 49ms** ← 최적 배포 설정
- Qwen3-1.7B batch: WER 2.30%, RTF 0.026
- Qwen3-1.7B stream css=2.0: WER 2.30%, RTF 0.092, fd 85ms
- Nemotron 560ms stream: WER 3.85%, RTF 0.077 — keep_all_outputs fix 적용
- Nemotron 1120ms stream: WER 4.44%, RTF 0.046
- Nemotron 160ms stream: WER 7.06%, RTF 0.229

LibriSpeech clean (100 samples, L40S FP8):
- Qwen3-1.7B FP8 batch: WER 2.14%, RTF 0.018
- Qwen3-1.7B FP8 stream css=2.0: WER 2.09%, RTF 0.048

LibriSpeech other (100 samples, H100 BF16):
- **Granite 1B batch: WER 3.68%, RTF 0.030** ← 최저 WER
- **Granite 1B stream css=2.0: WER 3.75%, RTF 0.075**
- Qwen3-1.7B batch: WER 4.45%, RTF 0.035 (Exp #135)
- Qwen3-1.7B stream css=2.0: WER 4.32%, RTF 0.094 (Exp #135)
- Nemotron batch: WER 5.84%, RTF 0.016
- Nemotron 560ms stream: WER 8.01%, RTF 0.085 — 노이즈 환경 취약

LibriSpeech other (100 samples, L40S BF16):
- Granite 1B batch: WER 3.75%, RTF 0.025
- Granite 1B stream css=2.0: WER 3.62%, RTF 0.082, fd 83ms
- **Granite 1B stream css=4.0: WER 3.75%, RTF 0.043, fd 87ms**
- **Granite 1B stream css=8.0: WER 3.75%, RTF 0.029, fd 103ms** ← CSS-invariant 확인

LibriSpeech other (100 samples, L40S FP8):
- Qwen3-1.7B FP8 batch: WER 4.38%, RTF 0.024
- Qwen3-1.7B FP8 stream css=2.0: WER 4.38%, RTF 0.053

Korean FLEURS (100 samples, H100 BF16):
- Qwen3-1.7B batch: CER 2.84%, RTF 0.023
- Qwen3-1.7B stream css=2.0: CER 2.89%, RTF 0.114, fd 81ms
- Qwen3-1.7B stream css=3.0: CER 2.89%, RTF 0.083, fd 97ms ← 균형점
- Qwen3-1.7B stream css=5.0: CER 2.89%, RTF 0.057, fd 150ms

Korean FLEURS (100 samples, L40S BF16):
- Qwen3-1.7B batch: CER 2.89%, RTF 0.022
- Qwen3-1.7B stream css=3.0: CER 2.84%, RTF 0.084, fd 97ms
- Qwen3-1.7B stream css=5.0: CER 2.89%, RTF 0.057, fd 150ms
- **Qwen3-1.7B stream css=8.0: CER 2.89%, RTF 0.044, fd 216ms** ← CSS-invariant 확인
- **Qwen3-1.7B adaptive 3→8: CER 2.89%, RTF 0.045, fd 85ms** ← 최적 배포 설정
- Note: CSS 2.0~8.0 전체에서 CER=2.84~2.89% 범위. stream-batch gap 0pp.

Korean FLEURS (30 samples, L40S FP8):
- Batch FP8: CER 2.09% (qwen3-1.7b FP8), RTF 0.016
- Streaming FP8 (css=3.0): CER 2.09%, RTF 0.048

Qwen3-vllm-prefix (100 samples, L40S BF16, SDK 불필요, ucn=4):
- **0.6B prefix LS-clean: WER 2.01%, RTF 0.025** ← 최저 multilingual streaming WER
- 1.7B prefix LS-clean: WER 2.11%, RTF 0.059
- 1.7B prefix LS-other: WER 3.97%, RTF 0.069
- **1.7B prefix FLEURS-ko: CER 2.96%, RTF 0.060** ← SDK-free 한국어 (batch 대비 0.16pp)
- 0.6B prefix FLEURS-ko: CER 4.89%, RTF 0.021
- Note: ucn=4로 한국어 CER 3.62%→2.96% 개선. SDK streaming(2.89%) 대비 0.07pp 차이.

Qwen3-vllm-prefix FP8 (100 samples, L40S FP8, ucn=4):
- **0.6B FP8 prefix LS-clean: WER 1.96%, RTF 0.028** ← 최저 multilingual streaming WER
- 0.6B FP8 prefix FLEURS-ko: CER 4.38%, RTF 0.028 (BF16: 4.74%)
- **1.7B FP8 prefix LS-clean: WER 2.06%, RTF 0.045** ← BF16 대비 RTF -24%, WER -0.05pp
- **1.7B FP8 prefix FLEURS-ko: CER 3.01%, RTF 0.046** ← BF16 대비 RTF -23%, CER +0.05pp
- 1.7B Model memory: 2.55 GiB (BF16 ~3.5 GiB)
- Note: FP8은 영어/한국어 모두 WER/CER 개선 또는 유지. RTF도 개선.
- **1.7B FP8 prefix utn=15 FLEURS-ko: CER 2.86%** ← SDK streaming(2.89%)보다 좋음, batch(2.80%) 대비 0.06pp
- Note: utn=15는 한국어 최적이나 LS-other에서 +0.19pp 회귀. 영어는 utn=5 유지 권장.
- **1.7B FP8 prefix utn=15 css=3.0 FLEURS-ko: CER 2.83%** ← batch(2.80%) 대비 0.03pp!
- Note: css=3.0+utn=15 한국어 최적 조합. css=2.0은 hallucination 위험.
- **language-adaptive 구현**: `--lan ko` + 1.7B 모델 → 자동으로 utn=15/css=3.0 적용 (commit 677f6b6)
- Note: 0.6B는 Korean-adaptive 미적용 — css=3.0이 0.6B에서 +0.28pp 회귀 (capacity 부족)

Multilingual FLEURS (100 samples, L40S FP8, qwen3-vllm-prefix, language-adaptive):
- **en batch: WER 5.15%, RTF 0.016** | **en stream: WER 5.20%, RTF 0.042** (gap +0.05pp)
- **ko batch: CER 2.78%, RTF 0.017** | **ko stream: CER 2.84%, RTF 0.050** (gap +0.06pp)
- **zh batch: CER 2.89%, RTF 0.013** | **zh stream: CER 3.08%, RTF 0.040** (gap +0.19pp)
- **vi batch: WER 6.24%, RTF 0.016** | **vi stream: WER 6.72%, RTF 0.049** (gap +0.48pp)
- **ja batch: CER 5.67%, RTF 0.014** | **ja stream: CER 6.20%*, RTF 0.043** (gap +0.46pp)
- Language-adaptive: non-English utn=15/css=3.0, English utn=5/css=4.0
- Note: 단일 Qwen3-1.7B FP8로 5개 언어 커버. 모든 언어 batch-stream gap <0.5pp.
- *ja streaming: hallucination 1/100 제외. Raw 16.02% → filtered 6.20%. Repetition filter 필요.

Long-form (10 min continuous, H100 BF16):
- **Granite 1B css=8.0: RTF 0.057, WER 1.18%** ← 최적
- Nemotron 560ms native: WER 2.75%, RTF 0.064
- Granite css=2.0: RTF 0.254 (비효율, css=8.0 사용 권장)

**Streaming-Batch Gap: EN 0.00pp ✅, KO 0.00pp ✅ (css=8.0)**
**목표: 스트리밍 WER < 4% ✅ (Granite 1.18%), RTF < 0.15 ✅ (0.033), first-word latency < 200ms ✅ (fd=49ms)**

배포 전략:
- **영어 전용 (최고 품질): Granite 1B adaptive 2→8** (WER 1.18%, RTF 0.033, fd 49ms)
- **다국어 5개 언어 (ko/zh/en/vi/ja)**: Qwen3-1.7B FP8 prefix, language-adaptive 자동 적용
  - ko: CER 2.84%, RTF 0.050 (utn=15, css=3.0)
  - zh: CER 3.08%, RTF 0.040 (utn=15, css=3.0)
  - en: WER 5.20%, RTF 0.042 (utn=5, css=4.0)
  - vi: WER 6.72%, RTF 0.049 (utn=15, css=3.0)
  - ja: CER 6.20%, RTF 0.043 (utn=15, css=3.0) — hallucination filter 필요
- 영어 multilingual (경량): **Qwen3-0.6B FP8 prefix** (WER 1.96%, RTF 0.028)
- Nemotron 560ms: VRAM 제약 시 대안 (5GB, WER 3.85%)

Concurrent Sessions (L40S, 1.7B FP8 prefix):
- **Batched generate()**: BS=8 → 78.5x RT throughput (BS=1의 3.5배), WER 2.69% 일관
- ThreadPoolExecutor: 교착 상태 발생 (vLLM `LLM.generate()` is NOT thread-safe)
- **프로덕션 구현**: request queue + periodic batch generate() 패턴 필요
- async scheduler 아키텍처 검증 완료 (Exp #189)

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

### 1. Streaming-Batch Gap 축소 ✅ (4개 언어 모두 <0.5pp)
**달성됨**: Qwen3-ASR-1.7B FP8 prefix-constrained + language-adaptive 설정.
- EN: stream WER 5.20% vs batch 5.15% (gap +0.05pp)
- KO: stream CER 2.84% vs batch 2.78% (gap +0.06pp)
- ZH: stream CER 3.08% vs batch 2.89% (gap +0.19pp)
- VI: stream WER 6.72% vs batch 6.24% (gap +0.48pp)

**핵심 기법**: prefix-constrained decoding + FP8 + language-adaptive UTN/CSS (non-EN: 15/3.0, EN: 5/4.0)

### 2. 새 모델/백엔드 탐색
최신 ASR 모델을 지속적으로 조사하고 통합 가능성을 평가한다.

**2026-03-22 조사 결과**: Qwen3-1.7B FP8 prefix가 여전히 한국어+다국어 스트리밍 최적.
- Voxtral Realtime 4B: 한국어 WER 6.80%@960ms (우리 2.84% 대비 열등)
- VibeVoice-ASR 7B: batch-only, 한국어 WER 9.65%, 과대
- Meta Omnilingual 7B: 1600+ 언어, batch-only, 과대
- NVIDIA Canary-Qwen-2.5B: LS-clean 1.6% 인상적이나 영어 전용, 스트리밍 미지원
- Granite 3.3 8B: 한국어 미지원
- Whisper v4 미출시: OpenAI는 gpt-4o-transcribe (API-only)로 이동

**주시 대상**:
- Qwen3-ASR 후속 모델 (Flash, larger variants)
- vLLM Realtime API의 encoder-decoder 모델 지원 확대
- Granite 4.0 multilingual 확장 (한국어 추가 시 즉시 평가)

### 3. 프로덕션 다중 세션 ← NEXT
- Exp #189에서 batched generate() 검증 완료 (BS=8 → 78.5x RT throughput)
- **필요**: request queue + periodic batch generate() 서버 구현
- vLLM `LLM.generate()` is NOT thread-safe → 전용 scheduler 필요
- Contextual biasing (hotword/keyword prefix injection)으로 도메인 특화 정확도 향상

### 4. 장시간 안정성 개선
- 현재 30초 세션 리셋 → sliding window prefix로 연속 처리
- Repetition guard 강화 (vi hallucination 해결 완료, 추가 언어 검증)

### 5. 아키텍처 개선
- VAD 개선 (Silero VAD 대안)
- DiffTracker 프로토콜 효율화
- WebSocket 대역폭 최적화

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
