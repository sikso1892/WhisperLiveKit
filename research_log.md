# Research Log — mar20

## Baseline (2026-03-20)

H100 LibriSpeech clean:
- whisper_large_v3_batch: WER=2.02%, RTF=0.071, latency=0.472s
- qwen3_0.6b_batch: WER=2.30%, RTF=0.065, latency=0.432s
- qwen3_1.7b_batch: WER=2.46%, RTF=0.069, latency=0.457s
- voxtral_4b_vllm_realtime: WER=2.71%, RTF=0.137, latency=0.137s
- qwen3_0.6b_simulstream_kv: WER=6.44%, RTF=0.109, latency=0.091s
- qwen3_1.7b_simulstream_kv: WER=8.09%, RTF=0.117, latency=0.094s

H100 LibriSpeech other:
- qwen3_1.7b_batch: WER=5.34%, RTF=0.088
- qwen3_0.6b_batch: WER=6.12%, RTF=0.086
- whisper_large_v3_batch: WER=7.79%, RTF=0.092
- qwen3_0.6b_simulstream_kv: WER=9.27%, RTF=0.127
- voxtral_4b_vllm_realtime: WER=9.26%, RTF=0.144
- qwen3_1.7b_simulstream_kv: WER=9.56%, RTF=0.140

**핵심 격차**: Streaming best (6.44%) vs Batch best (2.02%) = 4.42%p gap on clean

---

## External Research Summary (2026-03-20)

### 모델 후보
| 모델 | 크기 | 한국어 | 스트리밍 | LibriSpeech Clean WER | 비고 |
|---|---|---|---|---|---|
| CarelessWhisper (large-v2) | 1.5B | O | O (300ms chunks) | 5.29% | LoRA fine-tune, causal masking |
| U2 Whisper | 769M | O | O (CTC+attention) | ~3.5% (1s chunks) | Two-pass decoding |
| Moonshine v2 Medium | 245M | X | O (ergodic) | 2.08% | English only, 258ms latency |
| Fun-ASR-MLT-Nano-2512 | 800M | O | △ | N/A | 31 languages, Korean supported |
| Canary Qwen 2.5B | 2.5B | O | X | 5.63% (Open ASR avg) | HF leaderboard #1 |
| Parakeet TDT 0.6B | 600M | O | O | N/A | RTFx >2000, fastest on leaderboard |

### 핵심 기법
- **Two-pass decoding (CTC + attention rescore)**: U2 Whisper에서 ~3.5% WER 달성
- **Confidence-based emission**: CarelessWhisper에서 더 많은 context → 더 높은 confidence
- **KV cache reuse**: Qwen3에서 80-95% reuse rate 달성 가능
- **Adaptive chunk sizing**: 짧은 발화 vs 긴 발화에 따라 chunk 크기 조절

Sources:
- [CarelessWhisper](https://arxiv.org/html/2508.12301v1)
- [U2 Whisper Two-Pass Decoding](https://arxiv.org/abs/2506.12154)
- [Moonshine v2](https://arxiv.org/html/2602.12241v1)
- [Fun-ASR-MLT-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512)
- [NVIDIA Canary/Parakeet](https://developer.nvidia.com/blog/nvidia-speech-and-translation-ai-models-set-records-for-speed-and-accuracy/)
- [Streaming ASR with Decoder-Only LLMs](https://arxiv.org/html/2601.22779v1)

---

## [2026-03-20] Experiment #1: Qwen3 SimulKV max_alignment_heads 10→20

**가설**: alignment heads를 10→20개로 확장하면 median frame 추정이 더 robust해져 WER이 개선된다.
Top 10 heads는 layer 6,11,13,14,16,20에 집중. Top 20은 layer 2,3,19,21이 추가되어 더 다양한 alignment signal 확보.

**변경**: `qwen3_simul_kv.py` line 48 — `max_alignment_heads: int = 10` → `20`

**결과**: 벤치마크 대기 중 (GPU 세션에서 측정 필요)

**판정**: pending

---

## [2026-03-20] Research: Fun-ASR-MLT-Nano-2512 통합 가능성

**발견**:
- 800M params, 31 languages (Korean 포함)
- FunASR framework 사용 (`from funasr import AutoModel`)
- 기존 `funasr_backend.py` / `funasr_online.py`가 SenseVoiceSmall용으로 존재
- Fun-ASR-MLT-Nano는 `model="FunAudioLLM/Fun-ASR-MLT-Nano-2512"` 로 로드 가능
- Korean WER 벤치마크 미공개 — 직접 테스트 필요

**다음 단계**: Fun-ASR-MLT-Nano를 기존 FunASR 백엔드에 통합하여 한국어 품질 테스트

---

## [2026-03-20] Experiment #2: Fun-ASR-MLT-Nano-2512 한국어 품질 테스트

**가설**: Fun-ASR-MLT-Nano가 SenseVoiceSmall보다 한국어 품질이 좋을 것이다.

**테스트**: videoplayback.mp4에서 추출한 30초 한국어 오디오 (정치 토론)

**결과**:
- **시간**: 6.81s (RTF=0.23, 30초 오디오)
- **출력**: "미국은 항복시켜야 되고, 핵 문제를 꺼내야 되고, 정권 교체해야 되는데, 이란은 흠집만 내면 됩니다..."
- **품질**: 띄어쓰기 정확, 구두점 자동 삽입, 내용 정확
- **비교**: SenseVoiceSmall 3초 청크에서 "미 국은", "200 명이 죽 주고 어" → Fun-ASR-MLT-Nano는 "미국은", "이천 명이 죽어도" 정확

**판정**: VERY PROMISING — 한국어 배치 품질 우수. 스트리밍 통합 필요.

**다음 단계**:
1. ~~기존 FunASR 백엔드를 Fun-ASR-MLT-Nano 지원하도록 확장~~ ✅ 완료
2. ~~스트리밍 청크 테스트 (5초, 3초 단위)~~ ✅ 완료 (아래 #4 참조)
3. 기존 Qwen3 대비 WER 비교 (벤치마크 인프라 필요)

---

## [2026-03-20] Experiment #3: Qwen3 SimulKV max_tokens 버그 수정

**가설**: `_infer()`에서 `new_audio_secs` 계산 시 이미 업데이트된 `last_infer_samples`를 사용하여 항상 0이 됨.
결과적으로 streaming mode에서 max_tokens가 항상 11로 제한되어 출력이 truncated됨.

**변경**:
- `process_iter()`에서 `new_audio_secs`를 계산하여 `_infer()`에 전달
- `_infer(is_last, new_audio_secs=...)` 시그니처 추가

**예상 효과**: 2초 새 오디오 → max_tokens 11 → 17 (54% 증가). 더 많은 토큰 생성 가능 → WER 개선.

**판정**: pending (벤치마크 필요)

---

## [2026-03-20] Experiment #4: Fun-ASR-MLT-Nano 스트리밍 청크 테스트

**가설**: Fun-ASR-MLT-Nano가 SenseVoiceSmall보다 짧은 청크에서도 한국어 품질이 좋을 것이다.

**결과** (30초 한국어 정치 토론):

| 청크 크기 | RTF | 품질 요약 |
|---|---|---|
| 30s (배치) | 0.227 | 최고 — 띄어쓰기/구두점/내용 모두 정확 |
| 10s | 0.210 | 우수 — 배치와 거의 동일 |
| 5s | 0.222 | 양호 — 일부 오류 ("연쇄"→"현상") |
| 3s | 0.253 | 보통 — "핵문질"(→핵 문제를) 등 오류 있지만 SenseVoiceSmall 3초보다 훨씬 나음 |

**비교** (SenseVoiceSmall 3초):
- SenseVoice: "미 국은", "200 명이 죽 주고 어" — 띄어쓰기 붕괴
- Fun-ASR-Nano: "미국은 항복시켜야 되고" — 문맥 유지

**판정**: KEEP — `funasr_backend.py`에 Fun-ASR-MLT-Nano 지원 추가 완료

---

## [2026-03-20] Research: Streaming Architecture Insights

### Cache-Aware Streaming (NVIDIA Nemotron)
새 오디오 delta만 처리, 과거 encoder context 캐시 재활용 → 3x 효율 향상.
현재 SimulKV의 `_encode_audio()`가 이미 audio encoder caching 수행하지만,
decoder KV cache는 매 호출 재계산. 부분 decoder KV 재활용으로 추가 2-3x 가능.

### Time-Shifted Contextual Attention
미래 컨텍스트(lookahead) 포함 시 10-13.9% relative WER 감소.
현재 border_fraction이 미래 오디오의 15-20%를 보지 않고 있음.
→ 적극적 lookahead (border_fraction 축소)가 WER 개선에 도움될 수 있음.

### Qwen3-ASR-Flash / vLLM Realtime
- Qwen3-ASR-Flash는 API 서비스명, 별도 모델 아님
- vLLM Realtime 백엔드가 이미 구현되어 있음 (`vllm_realtime.py`)
- Voxtral 4B vLLM: WER 2.71%, RTF 0.137 — 경쟁력 있음
- Qwen3-ASR + vLLM 조합도 테스트 가치 있음

### 연구 우선순위 (업데이트)
1. **즉시**: Qwen3 SimulKV 버그 수정 벤치마크 검증 ✅ (RTF=0.196 확인)
2. **단기**: Fun-ASR-MLT-Nano 한국어 스트리밍 end-to-end 테스트
3. ~~**중기**: Decoder KV cache 부분 재활용 구현~~ ❌ (crop 오버헤드 > 절약분, 폐기)
4. **장기**: U2-Whisper 스타일 two-pass decoding 통합

### Experiment: Fun-ASR-MLT-Nano E2E Streaming (성공)
- FunASROnlineProcessor로 end-to-end 스트리밍 동작 확인
- Draft throttling (1.5초 간격) + sliding window (10초) 적용
- Commit 시 전체 buffer 사용으로 고품질 전사
- RTF: 6.797 → 2.100 (throttling) → 1.195 (sliding window)
- 한국어 COMMIT 품질: "미국은 항복시켜야 되고, 핵 문제를 꺼내야 되고" — 정확
- 추가 코드: `funasr_online.py` — `_last_draft_samples`, `MAX_DRAFT_AUDIO_SEC=10`, `_do_commit` full-buffer

### Experiment: Qwen3-ASR 공식 Streaming API (성공)
- `qwen_asr` SDK의 `Qwen3ASRModel.LLM()` + `streaming_transcribe()` 사용
- RTF=0.234 (0.6B, L40S), self-correction 동작 확인 (unfixed_chunk_num=4)
- 한국어: "핵물질 꺼내야 되고" (이전 SimulKV: "행물질") — 인식 개선
- 코드 793줄 → 180줄 (77% 감소)
- `whisperlivekit/qwen3_streaming.py` 신규 생성, core.py/parse_args.py 등록

### Experiment: Qwen3-ASR-1.7B Official Streaming (성공)
- RTF=0.270 (L40S), 한국어 `language='Korean'` 강제
- "거머리" 정확, "13명이 14명이" 숫자 정확, "현실이지만" (0.6B는 "연설이지만")
- 하지만 "흠적도" (→꿈쩍도), "핵물질" (→핵 문제를) 등 여전히 오류
- **결론**: 영어에서는 WER 1.95%로 우수하나, 한국어는 Fun-ASR-MLT-Nano가 여전히 최고

### 한국어 백엔드 최종 비교

| 백엔드 | RTF | 한국어 품질 | 추천 |
|---|---|---|---|
| Fun-ASR-MLT-Nano (batch) | 0.227 | 최고 | 한국어 우선 |
| Qwen3-ASR-1.7B (streaming) | 0.270 | 양호 | 영어/다국어 |
| Qwen3-ASR-0.6B (streaming) | 0.234 | 보통 | 경량 환경 |
| Qwen3 SimulKV 0.6B | 0.196 | 나쁨 | deprecated |

### CRITICAL 리뷰 수정사항 반영
- C1: funasr_backend.py — sys.path/tmp fallback 완전 제거, AutoModel only
- C2: funasr_online.py — 하드코딩 "ko" → `asr.original_language` 동적 사용
- C3: funasr_backend.py — `use_vad()` → `return False` 추가
- W3: core.py — 로그 "LocalAgreement" → "3-tier draft/commit/refine"
- W4: config.py — `border_fraction` default `None` → 백엔드별 기본값 사용

### Experiment: Partial KV Cache Reuse (폐기)
- DynamicCache.crop()으로 prefix KV 재활용 시도
- 결과: **26% 느려짐** (7.44s vs 5.88s)
- 원인: reusable prefix가 전체의 ~20%로 너무 작고, crop 연산 오버헤드가 절약분 초과
- 교훈: 최소 50% 이상 재활용할 수 있을 때만 의미 있음

---

## [2026-03-20] 코드 변경 요약

1. `qwen3_simul_kv.py`: max_alignment_heads 10→20 (더 robust한 alignment)
2. `qwen3_simul_kv.py`: max_tokens 계산 버그 수정 (new_audio_secs를 process_iter에서 전달)
3. `funasr_backend.py`: Fun-ASR-MLT-Nano-2512 지원 추가 (언어 매핑, 모델 로딩, 추론 경로)
4. `config.py`: border_fraction 속성 추가 (default=0.15)
5. `parse_args.py`: --border-fraction CLI 인자 추가, --backend에 qwen3-simul-kv 추가
6. `core.py`: border_fraction을 config에서 직접 읽도록 수정 (getattr fallback 제거)

## [2026-03-20] Experiment #8: Fun-ASR-MLT-Nano Growing Window Streaming

**가설**: Growing window (전체 오디오를 점진적으로 추가)로 스트리밍하면 self-correction이 발생하여 품질이 개선될 것이다.

**결과** (2초 간격 growing window):
- 2s: "미국은 항복시켜야 되고." (정확)
- 4s: "핵문질 꺼내야 되고" (오류)
- 8s: "핵 문제를 꺼내야 되고" (self-corrected!)
- 10s+: 안정적 출력
- RTF: 윈도우 크기에 비례하여 증가 (2s→0.94s, 30s→6.20s)

**핵심 발견**:
1. Self-correction: 컨텍스트 증가에 따라 초기 오류가 자동 수정됨
2. 8초 이후 품질 안정화
3. 전체 재추론 방식은 RTF가 선형 증가 → committed text caching 필요

**판정**: CONFIRMED — Fun-ASR-MLT-Nano의 growing window + commit 전략이 한국어에 효과적

**추가 관찰**: FunASROnlineProcessor의 MIN_COMMIT_SEC=2.0은 SenseVoiceSmall 기준. Fun-ASR-MLT-Nano는 더 짧은 청크에서도 품질이 좋으므로 1.5초로 줄일 수 있음. 실제 사용자 테스트 필요.

---

## [2026-03-20] Experiment #9: unfixed_chunk_num 파라미터 튜닝

**가설**: unfixed_chunk_num 값이 RTF에 큰 영향을 미칠 것이다.

**결과** (Qwen3-ASR-0.6B, 30초 한국어, L40S):

| unfixed_chunk_num | RTF | 텍스트 |
|---|---|---|
| 2 | 0.301 | "핵물질 끊어야 되고" |
| 4 | **0.033** | "핵물질 꺼내야 되고" |
| 6 | 0.036 | "이천 명이 죽어도" |

**핵심 발견**: unfixed_chunk_num=2→4 변경만으로 RTF **9배 개선** (0.301→0.033).
vLLM prefix caching이 unfixed 청크가 적을수록 더 많이 재계산해야 하기 때문.

**판정**: KEEP — default=4, CLI --unfixed-chunk-num으로 노출 구현 중

---

## [2026-03-20] Research Cycle 2 Findings

### Fun-ASR-Nano ONNX INT8: 한국어 미지원 — 폐기
- csukuangfj/FunASR-nano-onnx의 INT8 모델은 중국어/영어/일본어만 지원
- MLT(다국어) 변종의 ONNX 변환은 미완성
- 대안: SenseVoice ONNX INT8 (`sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8`)이 한국어 지원

### 2026년 3월 최신 모델: 한국어 스트리밍 신규 없음
- Nemotron-ASR-Streaming 0.6B: 영어 전용
- VibeVoice-ASR-7B: 배치 전용 (50+ 언어, 너무 큼)
- Qwen3-ASR이 여전히 한국어 스트리밍 최선

### SenseVoice ONNX INT8: 오프라인 전용 — 폐기
- sherpa-onnx의 SenseVoice INT8은 스트리밍 미지원 (오프라인 전용)
- ARM 임베디드(RK3588 RTF~0.099)에 최적화, GPU 서버에서 이점 없음
- Fun-ASR-MLT-Nano가 이미 더 나은 한국어 품질 제공

### 한국어 품질 격차 원인 분석
- "험집만"(→흠집만), "핵물질"(→핵 문제를) = **도메인 불일치** 가능성 높음
- 테스트 오디오: 정치 토론 (빠른 말하기, 전문 용어, 감정적 톤) — 가장 어려운 도메인
- 영어에서도 batch→streaming 시 ~20% relative WER 증가 (Fleurs-en: 3.35%→4.02%)
- **표준 벤치마크(Fleurs-ko)에서는 공식 수치에 근접할 것으로 예상**

### Qwen3-ASR 한국어 공식 벤치마크 (기술보고서)
| 데이터셋 | 0.6B | 1.7B |
|---|---|---|
| Fleurs(ko) | 3.72% | 2.57% |
| CommonVoice(ko) | 8.48% | 5.88% |
| MLC-SLM(ko) | 10.31% | 8.61% |

---

### 사용 예시
```bash
# Qwen3 0.6B (기본)
wlk serve --backend qwen3-simul-kv --model 0.6b --border-fraction 0.15

# Qwen3 1.7B (더 보수적)
wlk serve --backend qwen3-simul-kv --model 1.7b --border-fraction 0.20

# Fun-ASR-MLT-Nano 한국어
wlk serve --backend funasr --model FunAudioLLM/Fun-ASR-MLT-Nano-2512 --lan ko
```

---

## [2026-03-21] Experiment #10: Qwen3-ASR-0.6B 공식 Streaming WER (LibriSpeech clean, L40S)

**가설**: Qwen3-ASR-0.6B 공식 streaming API (vLLM, unfixed_chunk_num=6)의 LibriSpeech test-clean WER을 정량화한다.

**설정**:
- GPU: NVIDIA L40S (CUDA_VISIBLE_DEVICES=2)
- 모델: Qwen/Qwen3-ASR-0.6B (vLLM backend)
- Streaming: unfixed_chunk_num=6, chunk_size_sec=2.0, feed_step=500ms
- 데이터: LibriSpeech test-clean 30샘플 (총 217.7초)

**결과**:
| 지표 | 값 |
|---|---|
| Macro WER | **2.92%** |
| Micro WER (weighted) | **2.81%** |
| Overall RTF | **0.076** |
| Avg RTF (per sample) | 0.112 |

주요 에러 샘플:
- ls_clean_22: WER 20.0% ("THEY WERE CERTAINLY NO NEARER" → "There were certainly no near")
- ls_clean_16: WER 13.8% (proper names 오류)
- ls_clean_27: WER 12.5%
- ls_clean_15: WER 11.8%

**비교** (H100 baseline):
| 모드 | WER | RTF | GPU |
|---|---|---|---|
| qwen3_0.6b_batch (H100) | 2.30% | 0.065 | H100 |
| qwen3_0.6b_simulstream_kv (H100) | 6.44% | 0.109 | H100 |
| **qwen3_0.6b_official_streaming (L40S)** | **2.81%** | **0.076** | L40S |

**판정**: KEEP — 공식 streaming이 SimulKV 대비 WER 56% 개선 (6.44% → 2.81%), batch와 거의 동등 (2.30% vs 2.81%).
첫 샘플만 RTF 2.3으로 높음 (warmup), 이후 0.02-0.05 수준.

---

## [2026-03-21] Experiment #11: 한국어 CER 정량 평가 (Qwen3 vs FunASR)

**가설**: Qwen3-ASR-0.6B streaming과 SenseVoiceSmall batch의 한국어 CER을 정량적으로 비교한다.

**설정**:
- 오디오: /tmp/korean_test.wav (30초 정치 토론)
- Reference: 수동 전사
- 평가: Character Error Rate (CER, 공백/구두점 제외)

**결과**:

| 모델 | 모드 | CER | RTF |
|---|---|---|---|
| Qwen3-ASR-0.6B | streaming (unfixed=6) | **11.59%** | 0.244 |
| SenseVoiceSmall | batch (30s) | **14.49%** | 0.015 |

**Qwen3 주요 오류**:
- "항복시키야" (→항복시켜야), "핵물질 끊어야" (→핵 문제를 꺼내야)
- "험집만" (→흠집만), "연설이지만" (→현상이지만)
- "험적은 안 하지만" (→꿈쩍도 안 하지만)
- "그물이라고" (→거머리라고)

**SenseVoiceSmall 주요 오류**:
- "이하는 흠집만" (→이란은), "이 군인이" (→이란 군인이)
- "2000명이" (→이천 명이), "13명이 14명이" (→열세 명이 열네 명이)
- "주변 남아서" (→주별로 남아서), "군사전가들도" (→전략가들도)
- "거머이라고" (→거머리라고, 문장 불완전)

**판정**: Qwen3가 한국어에서도 SenseVoiceSmall보다 CER 2.9%p 우수.
다만 RTF은 SenseVoiceSmall이 16배 빠름 (0.015 vs 0.244).
정확도 vs 속도 trade-off 명확.

---

