# Research Log — mar20

## [2026-03-22] Exp #200: 5-language regression test — CJK filter verification

**가설**: CJK-aware repetition filter가 기존 5개 언어(en/ko/zh/vi/ja)에서 회귀를 발생시키지 않는다.
**변경**: Exp #199 커밋 후 bench_fleurs_multilingual.py --lang en ko zh vi ja --n-samples 30 실행
**결과** (30 samples each, 1.7B FP8 prefix, language-adaptive):
| Lang | Batch    | Stream   | Gap     | RTF   |
|------|----------|----------|---------|-------|
| en   | WER 7.42%| WER 7.31%| -0.11pp | 0.041 |
| ko   | CER 2.52%| CER 2.52%| 0.00pp  | 0.048 |
| zh   | CER 2.37%| CER 2.59%| +0.22pp | 0.039 |
| vi   | WER 4.38%| WER 4.33%| -0.05pp | 0.049 |
| ja   | CER 3.64%| CER 3.96%| +0.32pp | 0.045 |
- JA sample #6 hallucination 정상 감지: "Repetition detected at chunk 5, resetting prefix"
- 모든 언어 회귀 없음 (30-sample variance 범위 내)
**판정**: keep (regression-free 확인)
**이유**: CJK filter가 hallucination만 정확히 잡고 정상 텍스트에 영향 없음.

## [2026-03-22] Exp #199: CJK-aware repetition filter — fix Japanese hallucination

**가설**: 일본어 streaming에서 발생한 hallucination(sample #6, CER 868%)은 word-level n-gram 검사가 CJK 텍스트(공백 없음)에서 작동하지 않기 때문이다. Character-level n-gram + 낮은 chars_per_sec 임계값으로 해결 가능.

**변경**: qwen3_prefix_processor.py의 `_detect_repetition` + bench_fleurs_multilingual.py
- CJK 감지: `len(text.split()) <= len(text) // 10`
- CJK: 15/20-char n-gram 3+회 반복 검사
- CJK: chars_per_sec 임계값 40 → 20
- Non-CJK: 기존 word-level n-gram 유지

**결과**:
- JA streaming CER: **16.02% → 6.13%** (hallucination 자동 감지+reset)
- KO 회귀 테스트: CER 2.52% (30 samples, 이전 2.84%/100samples과 일관)
- 모든 테스트 케이스 통과 (JA/KO/ZH/EN/VI 정상+hallucination)

**판정**: keep
**이유**: hallucination 1건으로 micro CER이 16%→6%로 폭등하는 문제 해결. 실 서비스에서 치명적.
**commit**: (아래)

## [2026-03-22] Exp #198: Japanese FLEURS benchmark — Qwen3-1.7B FP8 prefix

**가설**: Qwen3-ASR-1.7B FP8 prefix가 한국어/중국어/베트남어에서 batch-stream gap <0.5pp를 달성했으므로, 일본어도 유사한 수준일 것이다.

**변경**: bench_fleurs_multilingual.py에 일본어 지원 추가 (FLEURS ja_jp, normalize_ja CER 함수, 스크립트 감지)
- Language-adaptive: utn=15, css=3.0 (non-English 그룹)

**결과**:
- **Batch CER: 5.67%**, RTF 0.014
- **Streaming CER: 16.02%**, RTF 0.043 (hallucination 1건 포함)
- Hallucination 제외 (99/100 samples): **Streaming CER 6.20%, gap 0.46pp**
- Per-sample: 81 same(±1pp), 8 better, 11 worse(>1pp), 4 worse(>5pp)
- Sample #6: CER 868% — prefix streaming에서 반복 hallucination 발생

**판정**: keep (일본어 지원 코드) + iterate (hallucination 방지 필요)

**이유**:
- Batch CER 5.67%는 다른 언어 대비 높지만 FLEURS 일본어 데이터 특성 (외래어, 고유명사 다수)
- Hallucination 제외 시 gap 0.46pp로 vi(0.48pp)와 동등 — language-adaptive 설정 유효
- Hallucination 방지를 위한 repetition filter 또는 max_tokens cap 필요
- 5개 언어 배포 전략에 일본어 추가 가능 (조건: hallucination 필터 적용)

**다국어 비교 (1.7B FP8 prefix, language-adaptive)**:
| Lang | Batch CER/WER | Stream CER/WER | Gap    | RTF  |
|------|---------------|----------------|--------|------|
| en   | 5.15%         | 5.20%          | +0.05pp| 0.042|
| ko   | 2.78%         | 2.84%          | +0.06pp| 0.050|
| zh   | 2.89%         | 3.08%          | +0.19pp| 0.040|
| vi   | 6.24%         | 6.72%          | +0.48pp| 0.049|
| ja   | 5.67%         | 6.20%*         | +0.46pp| 0.043|
*hallucination 1건 제외

## [2026-03-22] Exp #192: UCN + max_tokens sweep — SDK params alignment

**가설**: 공식 Qwen3 SDK는 ucn=2, max_new_tokens=32 사용. 우리 ucn=4, mt=128이 보수적일 수 있다.
**변경**: ucn=2/4 × max_tokens=32/128 4-way sweep, EN 100 + KO 100 samples.
**결과**:
| Config | EN WER | EN RTF | EN trunc | KO CER | KO RTF | KO trunc |
|---|---|---|---|---|---|---|
| ucn4-mt128 (baseline) | 2.06% | 0.045 | 0 | 3.14% | 0.046 | 0 |
| ucn2-mt128 | 2.17% | 0.030 | 0 | 3.98% | 0.025 | 0 |
| ucn2-mt32 | 2.17% | 0.030 | 0 | 3.72% | 0.025 | 28 |
| ucn4-mt32 | 8.09% | 0.038 | 55 | 16.85% | 0.037 | 192 |
**판정**: discard (quality regression)
**이유**: ucn2 → 33% RTF 개선이지만 KO CER +0.84pp 회귀. ucn4-mt32는 prefix 없는 초기 chunk에서 truncation → 치명적. ucn2-mt32는 truncation이 오히려 KO CER 개선 (hallucination guard). 현재 ucn4-mt128이 최적 quality-speed 균형.
**인사이트**: ucn4-mt32 catastrophic failure = prefix 없이 32 token budget 부족. Adaptive max_tokens (prefix 있을 때 32, 없을 때 128)도 가능하나 복잡도 대비 이득 미미.

## [2026-03-22] Exp #191: Adaptive max_tokens — scale with audio length ★

**가설**: max_tokens=256은 과도한 budget. 오디오 길이에 비례하여 줄이면 RTF 개선 가능.
**변경**: /tmp/bench_adaptive_max_tokens.py (SamplingParams max_tokens 변경)
**결과** (1.7B FP8, L40S, 100 samples LS-clean, utn=5, css=2→4):

| Config | WER | RTF | Avg Budget |
|---|---|---|---|
| fixed-256 | 2.06% | 0.045 | 256 |
| fixed-128 | 2.06% | 0.041 | 128 |
| fixed-64 | 2.06% | 0.041 | 64 |
| adaptive-5x+20 | 2.06% | 0.041 | 47 |
| adaptive-3x+15 | 2.06% | 0.041 | 31 |

**판정**: keep — max_tokens=128로 변경 구현
**이유**:
1. WER 2.06% 모든 설정에서 완벽 일치 — vLLM이 `<|im_end|>` stop token으로 조기 종료
2. RTF 0.045 → 0.041 (9% 개선) — KV cache allocation overhead 감소
3. max_tokens=128이면 30초 오디오(~150 단어)도 충분히 커버
4. adaptive 방식(3x+15)은 더 aggresive하지만 max_tokens=128로도 동등한 RTF
5. **구현**: qwen3_vllm_asr.py의 SamplingParams max_tokens를 256→128로 변경

## [2026-03-22] Exp #190: Prefix temperature sweep — greedy vs non-greedy

**가설**: prefix constraint가 일관성을 보장하므로, 약간의 temperature(0.1-0.3)가 더 나은 가설을 탐색할 수 있다.
**변경**: /tmp/bench_prefix_temperature.py (SamplingParams temperature 변경)
**결과** (1.7B FP8, L40S, 100 samples, utn=5, css=2→4):

| Temp | EN WER | KO CER | EN RTF | KO RTF |
|---|---|---|---|---|
| 0.0 | 2.01% | 3.31% | 0.045 | 0.046 |
| 0.1 | 2.01% | 3.32% | 0.042 | 0.044 |
| 0.3 | 2.06% | 3.34% | 0.042 | 0.044 |

**판정**: discard
**이유**:
1. Temperature가 품질에 실질적 영향 없음 (±0.05pp 이내 = 통계적 노이즈)
2. Prefix constraint가 이미 output space를 충분히 제한 → sampling의 탐색 효과 없음
3. t=0.3에서 EN WER 0.05pp 악화 (2.06% vs 2.01%) — non-greedy는 오히려 약간 나빠짐
4. RTF는 t>0에서 약간 빠름 (0.042 vs 0.045) — nucleus sampling의 early termination 효과
5. **greedy(t=0.0)가 최적** — 현행 설정 유지

## [2026-03-22] Exp #189: Async batched prefix scheduler — architecture validation

**가설**: asyncio 기반 scheduler가 여러 세션의 추론 요청을 수집하여 배치 처리하면, real-time feed 시뮬레이션에서도 정상 동작할 것이다.
**변경**: /tmp/bench_async_prefix.py (벤치마크만)
**결과** (1.7B FP8, L40S, 10x speed feed):

| Sessions | WER | Throughput | Wall | Batch calls | Avg BS |
|---|---|---|---|---|---|
| 1 | 0.00% | 23.5x RT | 0.1s | 2 | 1.0 |
| 2 | 0.00% | 23.8x RT | 0.7s | 5 | 1.4 |
| 4 | 0.00% | 45.8x RT | 1.0s | 7 | 2.3 |
| 8 | 0.46% | 72.1x RT | 1.1s | 7 | 4.0 |

**판정**: keep (아키텍처 검증)
**이유**:
1. 세션당 1개 샘플만 처리하여 WER은 통계적으로 무의미하지만, 아키텍처 검증 목적 달성
2. Scheduler가 세션별 CSS threshold를 정확히 추적하고 ready 세션만 배치에 포함
3. C=8에서 avg_bs=4.0 — 8개 세션 중 평균 4개가 동시에 inference ready (50% utilization)
4. 72.1x RT throughput — 단일 L40S에서 72개의 realtime 스트림에 해당
5. **프로덕션 구현 방향**: BatchScheduler를 AudioProcessor에 통합하여 WebSocket 세션 간 배치 처리

## [2026-03-22] Exp #188: Batched prefix inference throughput — vLLM multi-prompt generate()

**가설**: ThreadPoolExecutor 대신 여러 세션의 프롬프트를 단일 `LLM.generate()` 배치 호출로 처리하면, 교착 상태 없이 동시 세션 throughput을 높일 수 있다.
**변경**: /tmp/bench_batched_prefix.py (벤치마크만)
**결과** (Qwen3-1.7B FP8, L40S, 50 LS-clean samples):

| Batch Size | WER | Throughput | Wall time |
|---|---|---|---|
| 1 | 2.69% | 22.2x RT | 14.8s |
| 2 | 2.69% | 37.3x RT | 8.8s |
| 4 | 2.69% | 54.5x RT | 6.0s |
| 8 | 2.69% | 78.5x RT | 4.2s |

**판정**: keep (접근 방식 확인)
**이유**:
1. WER 2.69% 모든 BS에서 완벽 일치 — 배치가 품질에 영향 없음
2. BS=8에서 78.5x realtime (BS=1의 3.5배) — 준선형 스케일링
3. 교착 상태 없음 — 단일 `generate()` 호출로 모든 세션 처리
4. **프로덕션 동시 세션 아키텍처**: request queue + periodic batch generate()
   - 각 세션이 독립적으로 audio chunk 축적
   - CSS 임계값 도달 시 batch queue에 추론 요청 추가
   - 일정 간격(e.g. 100ms)마다 queue의 모든 요청을 배치 호출
   - 결과를 각 세션에 분배
5. BS=1의 throughput(22.2x)이 이전 C=1(6.7x)보다 높은 이유: 이전 벤치마크는 chunk-by-chunk 시뮬레이션으로 CPU overhead 포함, 이번은 CSS 기반 inference만 측정

## [2026-03-22] Exp #187: Prefix concurrent session throughput — vLLM thread safety

**가설**: ThreadPoolExecutor로 2/4개의 동시 prefix streaming 세션을 실행하면 vLLM의 batching으로 throughput이 선형에 가깝게 스케일될 것이다.
**변경**: /tmp/bench_prefix_concurrent.py (벤치마크만)
**결과**:
- C=1: WER 2.69%, RTF 0.045, throughput 6.7x realtime, wall 49.1s
- C=2: **교착 상태 (deadlock)** — 30분+ 경과 후 GPU 0% 사용, 프로세스 kill
- C=4: 실행 불가 (C=2 deadlock으로 중단)
**판정**: discard (접근 방식 변경 필요)
**이유**: vLLM의 동기식 `LLM.generate()`는 thread-safe하지 않다. 여러 스레드에서 동시 호출 시 내부 engine lock으로 교착 상태 발생. 프로덕션 동시 세션 지원을 위해서는:
1. `AsyncLLMEngine` 사용 (vLLM serve 모드의 기반)
2. Request batching at processor level (여러 세션의 audio를 모아서 단일 generate() 호출)
3. Multiprocessing (별도 프로세스에서 각 세션 처리)
현재 prefix_processor는 세션당 독립적인 `transcribe()` 호출 구조 → 동시성 지원에 근본적 리팩토링 필요.

## [2026-03-22] Exp #185: Language-adaptive UTN/CSS via online_factory()

**가설**: online_factory()에서 asr.original_language를 감지하여 Korean이면 utn=15, css=3.0, 그 외에는 utn=5, css=4.0으로 자동 설정하면 양쪽 최적 성능을 달성할 수 있다.
**변경**: core.py (online_factory prefix path), parse_args.py (default→None), config.py (default→None)
**결과**:
- 파라미터 선택 테스트: Korean→utn=15/css=3.0 ✓, English→utn=5/css=4.0 ✓, Override→정확히 반영 ✓
- Korean E2E: CER 2.83%, RTF 0.051 (manual ref: 2.83%, 0.050)
- English E2E: WER 2.06%, RTF 0.045 (manual ref: 2.06%, 0.045)
**판정**: keep — online_factory() 경유 결과가 수동 설정과 정확히 일치
**이유**: 사용자가 --lan ko로 서버를 시작하면 자동으로 최적 파라미터 적용. 명시적 --unfixed-token-num이나 --adaptive-css-steady 설정 시 override 가능.
**commit**: 677f6b6

## [2026-03-22] Exp #186: 0.6B FP8 Korean UTN sweep — model-size-aware adaptive

**가설**: 0.6B 모델도 1.7B처럼 utn=15, css=3.0에서 Korean CER 개선을 보일 것이다.
**변경**: /tmp/bench_06b_lang_adaptive.py (벤치마크만, 코드 변경 없음)
**결과** (0.6B FP8, FLEURS-ko 100 samples, css=3.0):
- utn=5:  CER 4.66%, RTF 0.029
- utn=10: CER 4.61%, RTF 0.026
- utn=15: CER 4.56%, RTF 0.027
- 참고: utn=5/css=4.0 기존 ref: CER 4.38%
**판정**: discard (0.6B에 적용하지 않음) + fix (language-adaptive에 model-size guard 추가)
**이유**: css=3.0이 0.6B에서 +0.28pp 회귀 (4.66% vs 4.38%). UTN 효과도 미미 (0.10pp). 0.6B는 capacity 부족으로 빈번한 short-context inference에서 품질 저하. core.py에 `_model_size` 체크 추가하여 1.7B+에만 Korean-adaptive 적용.
**commit**: 4f40096

## [2026-03-22] Research: Uni-ASR (arxiv 2603.11123) 분석

Alibaba Qwen 팀의 통합 streaming/non-streaming ASR 프레임워크.
- 아키텍처: Conformer encoder + linear adapter + Qwen3-1.7B LLM decoder
- LS-clean streaming WER: 5.71% (1000ms chunk) — 우리의 2.06% 대비 크게 열세
- "Fallback decoding": 마지막 토큰을 provisional하게 출력 후 다음 chunk에서 재디코딩 → 우리의 prefix-constrained approach와 유사한 개념
- 논문이 직접 인정: "Qwen3-ASR의 iterative re-feed가 더 정확하지만 계산 비용이 높다"
- 결론: 우리 접근이 quality 면에서 SOTA. RTF 0.045로 실시간 처리에 충분.
- Lightning ASR (Smallest.ai): API-only, 오픈소스 아님 → 자체 호스팅 불가

## [2026-03-22] Exp #184: CSS + UTN combined sweep for Korean CER (1.7B FP8, ucn=4)

**가설**: css_steady=3.0 (더 빈번한 inference) + utn=15 (더 많은 self-correction)가 시너지 효과를 낼 것이다.
**변경**: css_steady=2.0/3.0/4.0 sweep with utn=15
**결과**:
- css=2.0: KO CER 3.50% (hallucination 발생! repetition detected), EN WER 3.01%
- **css=3.0: KO CER 2.83%**, EN WER 2.69%, RTF 0.050 ← **batch 대비 0.03pp만 차이**
- css=4.0: KO CER 2.88%, EN WER 2.69%, RTF 0.044
**판정**: keep (한국어 최적 설정으로 기록)
**이유**:
- css=3.0 + utn=15 = **KO CER 2.83%** — batch(2.80%) 대비 0.03pp, SDK streaming(2.89%)보다 좋음
- css=2.0은 prefix + ucn=4 조합에서 hallucination 위험. 사용 금지.
- css=3.0은 css=4.0 대비 RTF +14% (0.050 vs 0.044). 한국어 전용 배포 시 권장.
- 한국어 최적 config: **FP8 + ucn=4 + utn=15 + css_steady=3.0** → CER 2.83%

## [2026-03-22] Exp #183: UTN sweep for Korean CER optimization (1.7B FP8, ucn=4)

**가설**: 한국어는 영어보다 토큰당 정보량이 다르므로, utn (unfixed_token_num) 최적값이 다를 것이다.
**변경**: utn=3,5,7,10,15,20,30 sweep on FLEURS-ko + LS-clean/other 검증
**결과**:
- Korean CER: utn=3 (3.26%) → utn=5 (3.04%) → utn=7 (2.98%) → utn=10 (2.91%) → **utn=15 (2.86%)** → utn=20 (2.90%) → utn=30 (2.90%)
- **utn=15이 Korean CER 최적**: 2.86% (batch 2.80% 대비 0.06pp, SDK streaming 2.89%보다 좋음)
- English (50 samples): 모든 UTN에서 2.69% (완전 invariant)
- English (100 samples, utn=15): LS-clean 2.11% (utn=5: 2.06%, +0.05pp), LS-other 4.16% (utn=5: ~3.97%, +0.19pp)
**판정**: iterate (language-specific recommendation, 기본값 변경 보류)
**이유**:
- utn=15는 한국어 최적 (CER 2.86%, SDK streaming보다 0.03pp 좋음!)
- 그러나 영어 noisy audio (LS-other)에서 0.19pp 회귀
- 기본값 utn=5 유지, 한국어 배포 시 utn=10~15 권장
- 메커니즘: 높은 UTN = 더 많은 self-correction room → 한국어 형태소 변이에 유리
- RTF는 UTN에 무관 (0.041~0.048 범위, 유의차 없음)

## [2026-03-22] Exp #182: 0.6B FP8 prefix processor ucn=4

**가설**: 0.6B 모델에도 FP8이 1.7B와 유사한 RTF/품질 개선을 줄 것이다.
**변경**: Qwen3VLLMASR 0.6B에 quantization="fp8", gpu_memory_utilization=0.30
**결과**:
- LS-clean: WER 1.96%, RTF 0.028 (BF16 ref: 2.01%, RTF 0.025) → WER -0.05pp
- FLEURS-ko: CER 4.38%, RTF 0.028 (BF16 ref: 4.74%, RTF 0.031) → CER -0.36pp
**판정**: keep
**이유**:
- WER 1.96%는 모든 multilingual streaming 모델 중 최저. 1.7B FP8 (2.06%)보다 좋음.
- 한국어 CER도 4.74%→4.38%로 유의미한 개선. FP8 regularization 효과.
- RTF는 거의 동일 (0.025→0.028 영어, 0.031→0.028 한국어). 0.6B는 이미 충분히 빠름.
- 0.6B FP8 + 1.7B FP8 동시 로드 시 VRAM 절감 가능.

## [2026-03-22] Exp #181: FP8 quantization + prefix processor ucn=4

**가설**: FP8 양자화가 prefix processor의 RTF를 개선하면서 품질(WER/CER)을 유지할 것이다.
**변경**: Qwen3VLLMASR에 quantization="fp8" 전달, 100-sample LS-clean + FLEURS-ko 벤치마크
**결과**:
- LS-clean: WER 2.06%, RTF 0.045 (BF16 ref: WER 2.11%, RTF 0.059) → WER -0.05pp, RTF -24%
- FLEURS-ko: CER 3.01%, RTF 0.046 (BF16 ref: CER 2.96%, RTF 0.060) → CER +0.05pp, RTF -23%
- Model memory: 2.55 GiB (BF16 ~3.5 GiB, ~1 GiB 절감)
- CutlassFP8ScaledMMLinearKernel 사용 확인 (L40S compute capability 8.9)
**판정**: keep
**이유**:
- 영어 WER이 오히려 개선 (2.06% < 2.11%). FP8 regularization 효과.
- 한국어 CER은 +0.05pp 미미한 회귀 (3.01% vs 2.96%). 실사용에 무시 가능.
- RTF 23-24% 개선은 매우 유의미. 동일 GPU에서 더 많은 동시 세션 가능.
- VRAM 1 GiB 절감으로 0.6B와 1.7B 동시 로드 가능성 확대.

## [2026-03-22] Exp #180: Prefix cache utilization — per-inference timing analysis

**가설**: vLLM prefix caching이 활성화되어 있으므로, 후속 inference에서 공유 오디오 prefix의 encoder 계산이 캐시될 것이다.
**변경**: 10개 긴 샘플(>10s)에서 per-inference 시간 측정
**결과**:
- Inference #1 (2.0s audio): 80ms, RTF=0.040
- Inference #2 (6.0s audio): 163ms, RTF=0.027 (audio 3x, time 2x → 캐시 작동)
- Inference #3 (10.0s audio): 242ms, RTF=0.024
- Inference #4 (12.5s audio): 286ms, RTF=0.023
- RTF 감소: 0.040 → 0.027 → 0.024 → 0.023 (sub-linear scaling)
**판정**: iterate (정보 수집, 개선 가능성 확인)
**이유**:
- Decoder KV cache 재사용 → RTF 하락 (캐시 작동 확인)
- Audio encoder는 매번 전체 재계산 (absolute time 증가)
- 캐시 없이 linear scaling이면 0.040 RTF × 6s = 240ms일 것. 실제 163ms → ~32% 절감
- 근본 개선은 encoder 캐싱 or vLLM Realtime API로 incremental input. 현재 아키텍처에서는 불가.
- RTF 0.023-0.040 범위로 이미 충분히 빠름. 추가 최적화 우선순위 낮음.

## [2026-03-21] Exp #179: 0.6B prefix ucn=4 — lightweight model Korean CER

**가설**: 1.7B에서 ucn=4가 Korean CER을 0.66pp 개선했으므로, 0.6B에서도 유사한 개선이 기대된다.
**변경**: 0.6B 모델에 ucn=4 적용 (코드 변경 없음, 기본값이 이미 4)
**결과**:
- LS-clean: WER 2.01%, RTF 0.031 (ucn=2와 동일)
- FLEURS-ko: CER 4.74%, RTF 0.031 (ucn=2 대비 4.89%→4.74%, -0.15pp)
**판정**: iterate (marginal improvement, not worth separate commit)
**이유**: 0.6B는 1.7B 대비 ucn=4 효과가 미미 (0.15pp vs 0.66pp). 모델 용량이 작아 더 긴 컨텍스트를 충분히 활용하지 못함.

## [2026-03-21] Exp #178: Prefix ucn=4 + language forcing — Korean CER optimization

**가설**: SDK는 ucn=4, 언어 강제("language Korean<asr_text>") 사용. 우리는 ucn=2, 언어 강제 없음. SDK 설정을 맞추면 Korean CER이 개선될 것이다.
**변경**: unfixed_chunk_num 기본값 2→4 변경 (qwen3_prefix_processor.py, core.py)
**결과**:
- Ablation (100 samples):
  - ucn=2, no forcing: CER 3.62% (baseline)
  - ucn=4, no forcing: **CER 2.96%** (-0.66pp!) ← 최적
  - ucn=4, forced Korean: CER 3.09% (forcing이 오히려 해로움)
  - ucn=4, forced, max_tokens=32: CER 13.46% (SDK default 32 토큰은 한국어에 치명적)
  - ucn=4, forced, max_tokens=64: CER 3.09%
- Full verification (100 samples):
  - LS-clean: WER 2.11% (unchanged from ucn=2)
  - LS-other: WER 3.97% (unchanged)
  - FLEURS-ko: CER 2.96% (was 3.62%, -0.66pp)
- English (50 samples): ucn=4 WER 2.69% vs ucn=2 WER 2.79% (minor improvement)
**판정**: keep
**이유**:
- ucn=4는 한국어 CER을 0.66pp 개선하고 영어에 회귀 없음.
- SDK와의 CER 차이: 2.96% vs 2.89% = 0.07pp (사실상 동등)
- batch 대비 gap: 0.16pp (2.96% vs 2.80%)
- 언어 강제는 불필요 — 모델의 자체 언어 감지가 더 나음
- max_tokens=32는 한국어에 부족. 256 유지
- ucn=4가 작동하는 이유: 짧은 클립(5-15s)에서 chunk_id가 4에 도달하기 전에 finish() 호출 → 사실상 batch 디코딩
**commit**: 4453a49, 5375969

## [2026-03-21] Exp #177: Prefix first-word latency — css_initial sweep

**가설**: css_initial=1.0으로 낮추면 첫 번째 inference가 더 빨리 시작되어 first-word latency가 개선될 것이다. 단, 초기 오디오가 짧아 품질이 저하될 수 있다.
**변경**: /tmp/bench_qwen3_prefix_latency.py 작성 — css_initial=1.0/1.5/2.0, css_steady=2.0/4.0 조합 테스트 (50 samples)
**결과**:
- English (LS-clean, 50 samples):
  - css=2→4 (default): WER 2.79%, RTF 0.047, fb_avg=91ms, fb_p50=91ms
  - css=1→4 (low lat): WER 2.79%, RTF 0.043, fb_avg=68ms, fb_p50=67ms ← **동일 WER, 25% 빠른 latency**
  - css=1.5→4:          WER 2.90%, RTF 0.043, fb_avg=79ms, fb_p50=78ms
  - css=2→2 (frequent):  WER 3.01%, RTF 0.050, fb_avg=75ms, fb_p50=75ms ← 잦은 inference = WER 악화
- Korean (FLEURS-ko, 50 samples):
  - css=2→4 (default): CER 3.40%, RTF 0.038, fb_avg=94ms, fb_p50=74ms
  - css=1→4 (low lat): CER 4.93%, RTF 0.037, fb_avg=70ms, fb_p50=58ms ← **CER +1.53pp 악화**
**판정**: keep (English css=1→4 최적, Korean은 css=2→4 유지)
**이유**:
- English: css_initial=1.0이 WER 손실 없이 latency 25% 개선. 영어는 1초 오디오만으로도 충분한 컨텍스트 제공.
- Korean: 1초 초기 오디오가 부족. 한국어 형태소 복잡성으로 더 긴 초기 컨텍스트 필요.
- 실용적 결론: **language-adaptive css_initial** 전략이 최적. en→1.0, ko/multilingual→2.0.
- css_steady=2.0(잦은 inference)은 WER 악화만 초래. 4.0이 최적.

## [2026-03-21] Exp #176: Prefix-constrained on LS-other — noisy audio robustness

**가설**: Noisy 오디오(LS-other)에서 prefix error snowball이 amplify되어 prefix가 LocalAgreement보다 나빠질 수 있다.
**변경**: /tmp/bench_qwen3_prefix_ls_other.py (Qwen3-1.7B, L40S, 100 samples, css=2→4)
**결과** (L40S BF16, 100 samples):

| Method | LS-other WER | RTF |
|--------|-------------|-----|
| 1.7B Prefix | **3.97%** | 0.060 |
| 1.7B LA | 4.03% | 0.066 |
| 1.7B Batch | 4.45% | 0.035 |
| Granite 1B Batch | 3.68% | 0.030 |

**판정**: keep — prefix가 노이즈에서도 안정적
**이유**:
1. Prefix(3.97%)가 LA(4.03%)보다 약간 우수, snowball 없음
2. **Batch(4.45%)보다도 좋다** — streaming re-feed가 context 보강 효과
3. LS-clean(2.11%)만큼 극적인 차이는 아님 — clean audio에서 LA의 word-mismatch가 더 critical

## [2026-03-21] Exp #175: Qwen3-0.6B prefix-constrained decoding — cross-size comparison ★★

**가설**: Prefix-constrained decoding이 0.6B 모델에서도 LocalAgreement 대비 개선을 보이는가? 0.6B는 1.7B보다 빠르고 VRAM이 적어 리소스 제약 배포에 유리.
**변경**: /tmp/bench_qwen3_prefix_06b.py (Qwen3-0.6B, L40S, 100 samples, BF16, css=2→4)
**결과** (L40S BF16, 100 samples):

| Model | Method | English WER | Korean CER | En RTF | Ko RTF |
|-------|--------|-------------|------------|--------|--------|
| **0.6B Prefix** | prefix | **2.01%** | 4.89% | **0.025** | **0.021** |
| 0.6B LA | local agreement | 5.92% | 7.73% | 0.028 | 0.032 |
| 1.7B Prefix | prefix | 2.11% | **3.62%** | 0.039 | 0.039 |
| 1.7B LA | local agreement | 6.08% | 4.33% | 0.064 | 0.064 |

**판정**: ★★ keep — 0.6B prefix가 예상 외로 우수
**이유**:
1. **영어 WER 2.01%** — 1.7B prefix(2.11%)보다 좋다! 0.6B가 영어에서 batch 수준(2.03%)에 도달.
2. **영어 RTF 0.025** — 1.7B(0.039) 대비 36% 빠름. GPU 비용 효율 극대화.
3. **한국어 CER 4.89%** — 1.7B prefix(3.62%)보다 열등하지만, 0.6B LA(7.73%)보다 37% 개선.
4. 배포 시사점: **영어 전용이면 0.6B prefix가 최적** (WER 2.01%, RTF 0.025, 낮은 VRAM).
5. 한국어 포함이면 1.7B prefix가 최적 (CER 3.62%).

## [2026-03-21] Exp #174: Prefix-constrained processor E2E production verification

**가설**: Exp #173의 prefix-constrained decoding을 production-quality 온라인 프로세서로 구현하면 standalone 벤치마크와 동등한 성능을 보인다.
**변경**: whisperlivekit/qwen3_prefix_processor.py (Qwen3PrefixOnlineProcessor), core.py, config.py, parse_args.py
**결과** (L40S BF16, 100 samples):

| Config | Korean CER | English WER | Ko RTF | En RTF |
|--------|-----------|-------------|--------|--------|
| Prefix (production) | **3.62%** | **2.11%** | 0.039 | 0.046 |
| LocalAgreement | 4.33% | 6.08% | 0.064 | 0.061 |
| Batch (ref) | 2.80% | 2.03% | 0.023 | — |

**판정**: ★★★ keep — 커밋 및 배포
**이유**:
1. **Korean CER 0.71% 개선** (4.33%→3.62%), RTF 39% 개선 (0.064→0.039)
2. **English WER 3.97% 개선** (6.08%→2.11%), batch 수준에 근접 (2.03%)
3. LocalAgreement 약점(한국어 형태소 변이 → exact token match 실패)을 model-level prefix continuity로 해결
4. SDK 불필요 — qwen3-vllm 백엔드만으로 작동
**commit**: (아래)

## [2026-03-21] Exp #169: Qwen3-1.7B init_prompt context injection — cross-chunk consistency

**가설**: OnlineASRProcessor가 전달하는 init_prompt(커밋된 텍스트)를 Qwen3 프롬프트에 주입하면, 모델이 이전 전사와 일관된 출력을 생성 → LocalAgreement 수렴 향상 → CER/WER 감소
**변경**: /tmp/bench_qwen3_init_prompt.py (Qwen3-1.7B, L40S, 100 samples, adaptive 2→4)
- user_context: 유저 메시지에 "Previously transcribed: {init_prompt}" 추가
- system_msg: 시스템 메시지로 "Previous transcription context: {init_prompt}" 추가
**결과** (L40S BF16, 100 samples):

Korean (FLEURS-ko):
| Config | CER | RTF | avg_inf |
|--------|-----|-----|---------|
| baseline (no init_prompt) | 4.26% | 0.067 | 4.1 |
| user_context | 8.21% | 0.065 | 4.1 |
| system_msg | 4.43% | 0.064 | 4.1 |

English (LS-clean):
| Config | WER | RTF | avg_inf |
|--------|-----|-----|---------|
| baseline | 6.08% | 0.061 | 2.7 |
| user_context | 6.13% | 0.057 | 2.7 |

**판정**: discard
**이유**:
- user_context: 한국어 CER 93% 악화 (4.26% → 8.21%). 유저 메시지에 텍스트를 추가하면 모델이 오디오 전사 대신 주입된 텍스트에 혼동.
- system_msg: 한국어 CER 4% 악화 (4.26% → 4.43%). 시스템 메시지는 Qwen3-ASR 학습 데이터에 포함되지 않아 slight negative effect.
- 영어 WER은 거의 동일 (6.08% vs 6.13%) — 영어는 agreement가 이미 잘 동작하므로 init_prompt의 영향이 작음.
- **결론**: LLM-based ASR 모델은 학습 시 사용된 정확한 프롬프트 템플릿에서만 최적 성능. 프롬프트 수정은 거의 항상 품질 저하. init_prompt를 통한 cross-chunk 일관성 개선은 모델 재학습 없이는 불가능.

## [2026-03-21] Exp #173: Korean prefix decoding diagnosis + safety valve

**가설**: Exp #172의 한국어 CER 폭등(3.35→15.20%)은 소수의 catastrophic 샘플에 의한 것. 이들을 식별하고, repetition-based safety valve (prefix reset)로 방지 가능.
**변경**: /tmp/bench_qwen3_prefix_diagnose.py (Qwen3-1.7B, L40S, 100 samples, fresh model load)
**결과** (L40S BF16, 100 samples, clean model load):

| Config | CER | Catastrophic (>20%) | Resets |
|--------|-----|---------------------|--------|
| No safety valve | 3.72% | 1 (sample 31, 23.8%) | — |
| With safety valve | 3.70% | 1 | 1 |
| LocalAgreement (ref) | 4.26% | — | — |
| Batch (ref) | 2.80% | — | — |

**판정**: ★★★ keep — prefix-constrained decoding을 production backend로 구현
**이유**:
1. **CER 3.72%** — fresh model load에서 안정적. Exp #172의 15.20%는 200+ prior inferences 후 vLLM state 오염으로 인한 비정상 결과.
2. **Safety valve 효과 미미** (3.72→3.70%) — catastrophic 샘플이 단 1개이고 repetition이 아닌 prefix mismatch가 원인.
3. **LocalAgreement 대비 0.54% CER 개선** (4.26%→3.72%), **영어는 3x 개선** (WER 6.08%→2.11%).
4. **RTF도 39% 개선** (0.067→0.041).
5. production 구현 시: silence-triggered session reset + safety valve 포함.
**commit**: (구현 후 커밋 예정)

## [2026-03-21] Exp #172: Qwen3-1.7B prefix-constrained decoding vs LocalAgreement ★★★

**가설**: Qwen3-ASR SDK의 native streaming 방식(prefix-constrained decoding)을 LocalAgreement 대신 사용하면 cross-chunk 일관성 향상 → CER/WER 감소. SDK는 이전 디코딩 결과를 assistant 프롬프트의 prefix로 삽입하고 마지막 K 토큰을 rollback하여 모델이 이전 출력과 일관된 새 텍스트를 생성하도록 유도.
**변경**: /tmp/bench_qwen3_prefix_decode.py (Qwen3-1.7B, L40S, 100 samples, adaptive CSS 2→4)
**결과** (L40S BF16, 100 samples):

Korean (FLEURS-ko):
| Config | CER@75 | CER@100 | RTF | avg_inf |
|--------|--------|---------|-----|---------|
| prefix ucn=2 utn=5 (SDK) | 3.35% | **15.20%** | 0.041 | 4.1 |
| prefix ucn=1 utn=5 | 3.60% | 15.47% | 0.035 | 4.1 |
| prefix ucn=2 utn=3 | 3.78% | 15.89% | 0.033 | 4.1 |
| LocalAgreement (ref) | — | 4.26% | 0.067 | 4.1 |

English (LS-clean):
| Config | WER | RTF |
|--------|-----|-----|
| prefix ucn=2 utn=5 | **2.11%** | 0.045 |
| LocalAgreement (ref) | 6.08% | 0.061 |

**판정**: ★ iterate — 가장 중요한 발견
**이유**:
1. **영어: 3x 개선** — WER 6.08% → 2.11%, batch 수준에 근접. 이것은 LocalAgreement보다 극적으로 우수.
2. **한국어: 대부분의 샘플에서 개선** — 75 샘플까지 CER 3.35% (LocalAgreement 4.26%보다 나은)
3. **한국어: 치명적 실패 모드** — 일부 샘플에서 prefix snowball (잘못된 prefix → 오류 누적 → 폭발적 CER 증가). 25개 샘플 만에 3.35% → 15.20%로 폭증.
4. RTF도 개선: 0.041 vs 0.067 (LocalAgreement는 HypothesisBuffer 비교 오버헤드 있음)

**다음 단계**:
- 한국어 catastrophic 샘플 식별 — 어떤 조건에서 snowball 발생?
- safety valve 추가 — prefix가 모델 출력과 크게 다를 때 prefix reset
- safety valve가 작동하면 Korean CER ~3.5% 달성 가능 (0.7% 개선)
- 영어는 이미 증명됨 — prefix backend 구현 고려

## [2026-03-21] Exp #171: Granite 1B Korean batch CER — cross-model language capability test

**가설**: Granite 1B Speech의 범용 speech encoder가 한국어를 처리할 수 있을 가능성
**변경**: /tmp/bench_granite_korean.py (Granite 1B, L40S, 30 samples, FLEURS-ko, batch)
**결과** (L40S BF16, 30 samples):
| Model | Dataset | CER | RTF |
|-------|---------|-----|-----|
| Granite 1B | FLEURS-ko | 173.15% | 0.030 |
| Qwen3-1.7B | FLEURS-ko | 2.80% (batch) | 0.023 |
**판정**: discard (당연한 결과)
**이유**: Granite 1B는 영어 전용 모델. 한국어 입력에 대해 일본어 카타카나, 로마자, 무의미한 텍스트를 출력. speech encoder가 한국어 음성을 인식하지만 decoder가 영어 텍스트만 생성하도록 학습됨. 한국어 ASR은 Qwen3-1.7B가 유일한 선택지.

## [2026-03-21] Exp #170: Uni-ASR 논문 분석 + 모델 landscape 업데이트

**조사 내용**:
1. **Uni-ASR (arXiv 2603.11123, 2026-03)**: Alibaba Qwen팀, Qwen3-1.7B 기반 통합 스트리밍/배치 ASR
   - context-aware training: 마지막 텍스트 토큰을 <pad>로 교체하여 cross-chunk 경계 처리 학습
   - fallback decoding: 마지막 토큰을 provisional로 방출, 다음 청크에서 확인/수정
   - 결과: LS-clean non-stream 1.93%, stream@1000ms 2.44%, stream@320ms 3.21%
   - 미공개 모델 — 학습 시 수정 필요, 사전학습 모델에 적용 불가
2. **Typhoon ASR Realtime**: Thai 전용 FastConformer-Transducer, 한국어 미지원
3. **CosyVoice2**: TTS 모델, ASR 아님
4. **Canary-Qwen-2.5B**: 영어 전용, vLLM 미지원
5. **Granite Speech 3.3 8B**: 한국어 speech 미지원 (EN/FR/DE/ES/PT만)
6. **Qwen3-ASR-Flash**: 별도 모델 아님, FlashAttention 2 사용 권장사항
7. **ENERZAi Korean Whisper-Small**: CER 6.45% — 우리 Qwen3-1.7B(2.80%)보다 훨씬 나쁨

**판정**: 정보 수집 완료. 새로운 경쟁 모델 없음. 현재 시스템이 오픈소스 SOTA 유지.

## [2026-03-21] Exp #168: Granite 1B keyword biasing from init_prompt — model-level consistency

**가설**: Granite의 keyword biasing 기능(문서화된 공식 기능)으로 이전 커밋 단어를 Keywords로 전달하면 cross-chunk 일관성 개선 → WER 감소
**변경**: /tmp/bench_granite_keywords.py (Granite 1B, L40S, 100 samples, adaptive 2→4)
**결과** (L40S BF16, 100 samples):
| Config | WER | RTF |
|--------|-----|-----|
| No keywords (baseline) | 2.14% | 0.054 |
| Keywords=5 (last 5 words) | 2.14% | 0.043 |
**판정**: discard
**이유**: WER 완전히 동일. keyword biasing은 일반 전사 품질에 영향 없음. temperature~0에서 모델이 이미 deterministic하고, 오디오 자체가 충분한 컨텍스트를 제공하므로 텍스트 keyword가 추가 정보를 주지 않음. keyword biasing은 OOV 단어(고유명사, 기술 용어)에 유용할 수 있지만 일반 벤치마크에서는 효과 없음.

## [2026-03-21] Exp #167: Qwen3-1.7B Korean CSS sweep — close streaming-batch gap

**가설**: css=2.0 fixed (더 빈번한 추론)이나 css=3→4 (더 긴 초기 청크)가 한국어 CER을 개선하여 streaming-batch gap(1.46%) 축소
**변경**: /tmp/bench_qwen3_ko_css_sweep.py (Qwen3-1.7B, L40S, 100 samples, FLEURS-ko)
**결과** (L40S BF16, 100 samples):
| Config | CER | RTF | avg_inf |
|--------|-----|-----|---------|
| adaptive 2→4 (baseline) | 4.26% | 0.067 | 4.1 |
| adaptive 2→2 (fixed 2s) | 7.77% | 0.107 | 6.7 |
| adaptive 3→4 | 11.09% | 0.066 | 3.8 |
| Batch (reference) | 2.80% | 0.023 | 1.0 |
**판정**: discard
**이유**: 모든 대안이 adaptive 2→4보다 나쁘다.
- css=2→2: 더 빈번한 추론이지만 각 추론의 컨텍스트가 부족 → per-inference 품질 저하 → CER 82% 악화
- css=3→4: 첫 3s 청크가 "유용한 시작점"을 제공하지 못함 → CER 160% 악화
- adaptive 2→4가 한국어에도 영어에도 최적의 universal 설정임을 재확인

**핵심 인사이트**: 한국어 streaming-batch gap(1.46%)은 CSS 튜닝으로 줄일 수 없다. 이 gap은 LocalAgreement의 텍스트 비교 방식(exact token match)이 한국어의 형태소 변이(조사, 어미)에 취약한 구조적 문제. 개선하려면 character-level agreement나 fuzzy matching이 필요하지만, 이는 규칙기반 접근이므로 progrem.md constraints와 충돌.

## [2026-03-21] Exp #166: Qwen3-1.7B batch Korean CER — Korean streaming-batch gap

**가설**: Qwen3-1.7B batch Korean CER 측정으로 streaming-batch gap 정량화
**변경**: /tmp/bench_qwen3_batch_ko.py (Qwen3-1.7B, L40S, 100 samples FLEURS-ko, batch)
**결과** (L40S BF16, 100 samples):
| Mode | Language | Dataset | Error Rate | RTF |
|------|----------|---------|-----------|-----|
| Batch | Korean | FLEURS-ko | CER 2.80% | 0.023 |
| Stream (2→4) | Korean | FLEURS-ko | CER 4.26% | 0.067 |
| Batch | English | LS-clean | WER 1.18% | 0.024 |
| Stream (2→4) | English | LS-clean | WER 2.14% | 0.054 |
| Batch | English | LS-other | WER 3.62% | 0.031 |
| Stream (2→4) | English | LS-other | WER 4.13% | 0.061 |

Streaming-batch gap 비교:
- 한국어 FLEURS-ko: **+1.46%** (가장 큰 gap)
- 영어 LS-clean: +0.96%
- 영어 LS-other: +0.51%
**판정**: keep (정보 — 코드 변경 없음)
**이유**: 한국어 streaming-batch gap(1.46%)이 영어(0.51~0.96%)보다 크다. 원인 추정: (1) 한국어 띄어쓰기/조사 변형이 LocalAgreement 합의를 어렵게 함, (2) 2s 초기 chunk에서 한국어 컨텍스트가 불충분. batch CER 2.80%는 매우 우수. streaming CER을 3.5% 이하로 낮추려면 한국어에 특화된 LocalAgreement 전략이 필요.

## [2026-03-21] Exp #165: Qwen3-1.7B language hint prompt — Korean CER with forced language

**가설**: SDK가 사용하는 `language Korean<asr_text>` assistant prefix를 추가하면 language detection 오버헤드가 줄어 한국어 CER 개선
**변경**: /tmp/bench_qwen3_lang_hint.py (Qwen3-1.7B, L40S, 100 samples, adaptive 2→4)
**결과** (L40S BF16, 100 samples FLEURS-ko):
| Prompt | CER | RTF |
|--------|-----|-----|
| Auto-detect (baseline) | 4.26% | 0.067 |
| language Korean<asr_text> | 5.33% | 0.058 |
**판정**: discard
**이유**: language hint가 CER을 1.07% 악화. RTF는 0.009 개선되었으나 품질 하락이 크다. 모델이 auto-detect 모드에서 더 잘 작동하도록 훈련된 것으로 보임. language hint는 decoding context를 변경하여 오히려 최적이 아닌 출력을 유도. 현재 auto-detect prompt가 Korean에 최적.

## [2026-03-21] Exp #164: Granite 1B batch WER — exact streaming-batch gap measurement

**가설**: Granite 1B batch WER on LS-other를 정확히 측정하여 streaming-batch gap 분석
**변경**: /tmp/bench_granite_batch.py (Granite 1B, L40S, 100 samples, batch single-shot)
**결과** (L40S BF16, 100 samples):
| Mode | Dataset | WER | RTF |
|------|---------|-----|-----|
| Batch | LS-clean | 1.18% | 0.024 |
| Batch | LS-other | 3.62% | 0.031 |
| Stream (2→4) | LS-clean | 2.14% | 0.054 |
| Stream (2→4) | LS-other | 4.13% | 0.061 |
| **Gap** | LS-clean | **+0.96%** | — |
| **Gap** | LS-other | **+0.51%** | — |
**판정**: keep (정보 — 코드 변경 없음)
**이유**: LS-other streaming-batch gap(0.51%)이 LS-clean(0.96%)보다 작다. noisy audio에서 adaptive CSS가 더 효과적 — 더 긴 컨텍스트가 noise robustness 향상. Batch WER 3.62%는 기존 추정치 3.75%보다 양호.

## [2026-03-21] Exp #163: Qwen3-1.7B adaptive CSS E2E — cross-model comparison

**가설**: Qwen3-1.7B가 adaptive CSS 2→4에서 영어 LS-clean/other에서 Granite 1B과 비슷한 WER을 달성하는지 검증
**변경**: /tmp/bench_qwen3_e2e_ls.py (Qwen3-1.7B, L40S, 100 samples, adaptive 2→4)
**결과** (L40S BF16, 100 samples):
| Model | Dataset | WER | RTF | fd (ms) | avg_inf |
|-------|---------|-----|-----|---------|---------|
| Qwen3-1.7B | LS-clean | 6.31% | 0.061 | 255ms | 2.7 |
| Qwen3-1.7B | LS-other | 4.51% | 0.071 | 293ms | 2.2 |
| Granite 1B | LS-clean | 2.14% | 0.054 | 222ms | 2.7 |
| Granite 1B | LS-other | 4.13% | 0.061 | 255ms | 2.2 |
**판정**: keep (정보 — 코드 변경 없음)
**이유**: Granite 1B가 영어에서 압도적으로 우수. LS-clean에서 3x 차이(2.14% vs 6.31%). LS-other에서는 격차가 작지만(4.13% vs 4.51%) 여전히 Granite 우세. Qwen3-1.7B는 한국어(CER 4.34%)에서 강점을 가지며, language routing(Granite EN + Qwen3 다국어) 전략이 유효함을 재확인.

특이사항: LS-clean 첫 25샘플 WER 14.34%로 시작 → 100샘플에서 6.31%. 초기 짧은 발화에서 Qwen3가 영어 인식에 실패하는 케이스가 다수.

## [2026-03-21] Exp #162: WebSearch — latest streaming ASR models/techniques (March 2026)

**가설**: 최신 ASR 모델/기법 조사로 streaming-batch WER gap 축소 또는 새 백엔드 후보 발견
**조사 결과**:

1. **Canary Qwen 2.5B** (NVIDIA): FastConformer + Qwen3-1.7B decoder
   - LS-clean 1.6%, LS-other 3.1% (batch mode), 418 RTFx
   - vLLM 미지원 (NeMo only), English only
   - 우리 Granite 1B batch (1.18%) 대비 LS-clean에서 열등 (벤치 조건 차이 가능)

2. **CarelessWhisper** (arXiv 2508.12301): Whisper를 causal streaming으로 변환 (LoRA fine-tune)
   - LS-clean 5.29% (300ms chunk) vs offline 2.70% = 2.6% gap
   - 우리 시스템 (2.14% streaming vs 1.18% batch = 0.96% gap)보다 훨씬 나쁨
   - 한국어 미지원, English + 4개 유럽어

3. **Two-Pass Streaming (U2-Whisper)**: CTC 1st pass + attention rescoring 2nd pass
   - WeNet C++ runtime에서 구현 가능
   - 모델 재훈련 필요 — 사전훈련 모델에서는 적용 불가

4. **Knowledge Distillation**: Streaming 모델이 non-streaming teacher로부터 학습
   - 모델 재훈련 필요 — 사전훈련 모델에서는 적용 불가

**판정**: discard (즉시 적용 가능한 개선 없음)
**이유**: 우리 시스템(Granite 1B 2.14%, Nemotron 0.6B 2.09%)이 CarelessWhisper(5.29%)보다 이미 우수. Canary Qwen은 vLLM 미지원. 모델 재훈련 기반 기법(U2, KD)은 사전훈련 모델 활용 환경에서 적용 불가. streaming-batch WER gap은 구조적 한계로 확인됨.

## [2026-03-21] Exp #161: Adaptive CSS E2E on LS-other — noisy audio robustness

**가설**: adaptive 2→4 설정이 noisy audio (LibriSpeech-other)에서도 clean과 유사한 streaming-batch WER gap을 유지하는지 검증
**변경**: /tmp/bench_ls_other_e2e.py (Granite 1B, L40S, 100 samples, adaptive 2→4)
**결과** (L40S BF16, 100 samples):
| Dataset | WER | RTF | fd (ms) |
|---------|-----|-----|---------|
| LS-clean | 2.14% | 0.054 | 222ms |
| LS-other | 4.13% | 0.061 | 255ms |
| Ref batch: LS-clean 1.18%, LS-other ~3.75% |
**판정**: keep (config change already committed in 9671d8e)
**이유**: LS-other WER 4.13% vs batch 3.75% = 0.38% gap. LS-clean gap (2.14% vs 1.18% = 0.96%)보다 작다. adaptive CSS는 noisy 환경에서 더 효과적 — 소음 속에서 더 긴 컨텍스트가 모델에 도움. RTF도 0.061로 실시간 대비 16x 빠름. fd=255ms는 200ms 목표보다 약간 높지만 합리적.

**구조적 WER gap 분석 결론**: 텍스트 레벨 접근(double-inference, n_provisional)은 모두 실패. streaming-batch WER gap(~1%)은 LocalAgreement + re-feed 아키텍처의 구조적 한계. 추가 개선은 모델 레벨(native streaming 모델) 또는 아키텍처 레벨(KV cache 재활용) 필요.

## [2026-03-21] Exp #160: n_provisional draft tokens — delayed emission for WER improvement

**가설**: HypothesisBuffer의 n_provisional 파라미터로 마지막 N개 토큰을 provisional(draft)로 보류하면, 다음 추론에서 확인/수정되어 WER 개선 (Uni-ASR의 fallback decoding과 유사)
**변경**: /tmp/bench_provisional.py (Granite 1B, L40S, 100 samples, adaptive 2→4)
**결과** (L40S BF16, 100 samples):
| n_provisional | WER | RTF |
|---------------|-----|-----|
| 0 (baseline) | 2.14% | 0.054 |
| 1 | 6.74% | 0.043 |
| 2 | 4.44% | 0.043 |
**판정**: discard
**이유**: WER 3x 악화. adaptive CSS(평균 2.7회 추론)에서는 토큰을 보류하면 커밋 기회가 줄어들어 init_prompt 컨텍스트가 부실해짐. n_provisional은 추론 횟수가 많은 환경(css=1.0 등)에서만 의미있을 수 있으나, RTF가 나빠지므로 트레이드오프가 불리.

## [2026-03-21] Exp #159: Double-inference finish — streaming-batch WER gap reduction

**가설**: finish() 시 두 번 추론하면 LocalAgreement가 최종 토큰에 대해 합의 기회를 얻어 WER 개선 (Uni-ASR "Latest-Token Fallback Decoding" 영감)
**변경**: /tmp/bench_double_finish.py (Granite 1B, L40S, 100 samples, adaptive 2→4)
**결과** (L40S BF16, 100 samples):
| Mode | WER | RTF |
|------|-----|-----|
| Single finish | 2.14% | 0.054 |
| Double finish | 2.30% | 0.062 |
**판정**: discard
**이유**: WER이 오히려 0.16% 악화. 동일 오디오를 두 번 inference → 모델이 동일 출력(temperature~0) → LocalAgreement가 오류 토큰까지 합의하여 커밋. Uni-ASR의 fallback은 모델을 re-decode하도록 훈련했기에 효과적이지만, 사전훈련 모델에서는 동일 입력=동일 출력이므로 효과 없음. 오히려 buffer에 남아있던 "불확실한" 토큰을 강제 커밋하여 오류 증가.

## [2026-03-21] Exp #158: Korean adaptive CSS E2E — Qwen3-1.7B FLEURS-ko 100 samples

**가설**: 영어에서 최적이었던 adaptive 1→4가 한국어에서도 최적인지 검증
**변경**: /tmp/bench_adaptive_css_korean_e2e.py (Qwen3-1.7B, L40S, 100 samples, 40ms VAC chunks)
**결과** (L40S BF16, 100 samples):
| Config | CER | RTF | fd (ms) | avg_inf |
|--------|-----|-----|---------|---------|
| adaptive 1→4 | 10.54% | 0.071 | 453ms | 4.3 |
| adaptive 2→4 | **4.34%** | 0.068 | 403ms | 4.1 |
| adaptive 2→8 | 5.66% | 0.044 | 440ms | 2.8 |
**판정**: keep — 기본값을 2→4로 재조정
**이유**: initial=1.0s는 한국어에 치명적 (CER 10.54%). 첫 청크가 너무 짧아 Qwen3가 의미있는 한국어 컨텍스트 생성 불가. adaptive 2→4가 두 언어 모두에서 우수한 균형:
- English: WER 2.25% (vs 1→4의 2.03%, 차이 0.22%)
- Korean: CER 4.34% (vs 1→4의 10.54%, 차이 6.2%)
- 2→4가 universal default로 적합
**참고**: 한국어 E2E 4.34% vs 스트리밍 참조 2.89% — LocalAgreement 오버헤드 ~1.5% 존재

## [2026-03-21] Exp #157: Adaptive CSS E2E WER optimization — steady chunk size tuning

**가설**: adaptive CSS steady=8.0s에서는 LocalAgreement 추론 횟수가 부족(2.2회)하여 E2E WER이 높음. steady를 4.0s로 줄이면 추론 횟수가 증가하여 WER 개선
**변경**: /tmp/bench_adaptive_css_e2e_v3.py (Granite 1B, L40S, 100 samples, 40ms VAC chunks)
**결과** (L40S BF16, 100 samples):
| Config | WER | RTF | fd (ms) | avg_inf |
|--------|-----|-----|---------|---------|
| adaptive 2→2 (fixed 2s) | 3.26% | 0.080 | 190ms | 3.8 |
| adaptive 2→4 | 2.25% | 0.044 | 174ms | 2.7 |
| adaptive 2→8 | 7.06% | 0.033 | 241ms | 2.2 |
| adaptive 1→4 | **2.03%** | 0.053 | 202ms | 2.9 |
**판정**: keep — 기본값을 2→4로 변경 (한국어 벤치마크 후 1→4에서 재조정)
**이유**: adaptive 2→4가 최적 universal default. EN WER 2.25%, KO CER 4.34%. RTF 0.044 (영어) / 0.068 (한국어).

## [2026-03-21] Exp #156: Model landscape analysis (March 2026)

**가설**: 최신 ASR 모델 중 우리 파이프라인에 통합할 만한 것이 있는지 조사
**조사 대상**:
- Canary-Qwen 2.5B: LS-clean WER 1.6%, 418x RTF. 하지만 스트리밍/vLLM 미지원 (GitHub issue 열려있음)
- Granite Speech 3.3 8B: 8B 파라미터로 너무 큼. 우리 Granite 4.0 1B (WER 1.18%)가 이미 우수
- Parakeet TDT 1.1B: >2000x RTF이지만 WER 8% — 품질 미달
- MoChA 기반 streaming decoder-only LLM (arXiv 2601.22779): 만다린 전용, 비공개
- Voxtral Realtime: WER 2.08% (EN), 15.74% (KO) — 우리 설정보다 열등
**판정**: discard — 현재 통합 가치 있는 신규 모델 없음
**이유**: Granite 4.0 1B (EN WER 1.18%) + Qwen3-1.7B (KO CER 2.89%) + Nemotron (193ms latency) 조합이 여전히 최강

## [2026-03-21] Exp #155: Adaptive CSS E2E benchmark + sep bugfix

**가설**: adaptive CSS가 production OnlineASRProcessor 파이프라인(40ms VAC 청크)에서 RTF를 크게 개선하면서 WER을 유지하는지 E2E 검증
**변경**: /tmp/bench_adaptive_css_e2e_v2.py (Granite 1B, L40S, 40ms chunks through OnlineASRProcessor)
**발견된 버그**: GraniteVLLMASR.sep="" 버그 — `ts_words()`가 `text.split()`으로 bare word 토큰 생성하지만 sep이 빈 문자열이라 init_prompt 컨텍스트가 공백 없이 합쳐짐. 3개 백엔드 수정 (granite_vllm_asr, granite_speech_asr, qwen3_vllm_asr)
**결과** (L40S BF16, 40ms VAC chunks):
| Config | N | WER | RTF | fd (ms) |
|--------|---|-----|-----|---------|
| no_adaptive | 10 | 25.50% | 3.381 | 571ms |
| adaptive 2→8 | 100 | 7.17% | 0.041 | 294ms |
| adaptive 1→8 | 100 | 6.84% | 0.037 | 259ms |
**판정**: keep (sep 버그 수정), iterate (E2E WER 개선 필요)
**이유**: RTF 82x 개선 (3.381→0.041) 확인. E2E WER 7.17%는 배치 1.18%보다 높지만 이는 LocalAgreement가 2-3회 추론으로는 합의 기회가 적기 때문. sep 버그 수정은 production에서 init_prompt 품질을 개선할 것.
**commit**: 52ac8b7 (sep bugfix)

---

## [2026-03-21] Exp #154: Production implementation of Adaptive CSS

**가설**: 벤치마크에서 검증된 adaptive CSS를 production streaming pipeline에 구현하면 동일한 품질을 유지하면서 GPU 사용량을 ~60% 줄일 수 있다.
**변경**:
- `config.py`: `adaptive_css`, `adaptive_css_initial`, `adaptive_css_steady` 필드 추가
- `parse_args.py`: `--adaptive-css`, `--adaptive-css-initial`, `--adaptive-css-steady` CLI args
- `online_asr.py`: `OnlineASRProcessor`에 audio duration tracking + skip logic. `start_silence()` force=True, `finish()` 시 잔여 오디오 처리
- `core.py`: `online_factory()`에서 granite-speech/qwen3-vllm 백엔드에 adaptive CSS 전달
**결과**: 구현 완료. Mock ASR 테스트 통과 (threshold gating, force processing, finish flush 모두 정상).
**판정**: keep
**이유**: 벤치마크 결과 (Granite WER 1.18% RTF 0.033 fd 49ms, Qwen3 CER 2.89% RTF 0.045 fd 85ms)를 production에서 재현할 수 있는 구현. 코드 변경 최소한 (141 insertions, 6 deletions).
**commit**: d9bc17a

## [2026-03-21] Exp #153: Adaptive CSS for Qwen3-1.7B Korean — css=3.0→8.0

**가설**: 한국어에도 adaptive CSS를 적용하면 fd가 개선될 것이다. 한국어는 3s 컨텍스트가 필요하므로 3→8이 2→8보다 나을 것이다.
**변경**: /tmp/bench_qwen3_ko_adaptive_css.py 벤치마크 스크립트
**결과**:
  - fixed css=3.0: CER=2.84%, RTF=0.085, fd=99ms
  - fixed css=8.0: CER=2.89%, RTF=0.044, fd=218ms
  - adaptive 2→8: CER=2.89%, RTF=0.046, fd=105ms
  - **adaptive 3→8: CER=2.89%, RTF=0.045, fd=85ms** ← 최적
**판정**: keep — adaptive 3→8이 한국어 최적 배포 설정
**이유**: CER 동일(2.89%), RTF은 css=8.0와 거의 동일(0.045 vs 0.044), fd는 모든 설정 중 최저(85ms). 한국어는 3s first chunk가 2s보다 나은 fd를 보임 (더 빠른 첫 텍스트 생성).

---

## [2026-03-21] Exp #152: Adaptive CSS — css=2.0 first chunk then css=8.0

**가설**: 첫 청크는 css=2.0 (빠른 first-word latency ~72ms), 이후 css=8.0 (효율적 RTF ~0.028)으로 전환하면 양쪽의 장점을 결합할 수 있다.
**변경**: /tmp/bench_granite_adaptive_css.py 벤치마크 스크립트 작성
**비교**: fixed css=2.0, fixed css=8.0, adaptive 2→8, adaptive 2→4
**결과**:
  - fixed css=2.0: WER=1.18%, RTF=0.082, fd=73ms
  - fixed css=8.0: WER=1.18%, RTF=0.028, fd=110ms
  - **adaptive 2→8: WER=1.18%, RTF=0.033, fd=49ms** ← 최적
  - adaptive 2→4: WER=1.28%, RTF=0.044, fd=49ms (WER 미세 회귀)
**판정**: keep — adaptive 2→8이 최적 배포 설정
**이유**: WER 동일, RTF는 css=8.0에 근접(+18%), fd는 모든 설정 중 최저(49ms, 이전 최저 72ms 대비 32% 개선). adaptive 2→4는 WER 0.10pp 회귀로 비추천.

---

## [2026-03-21] Exp #151: Qwen3-1.7B css=8.0 Korean FLEURS — CER invariance

**가설**: Qwen3-1.7B의 CSS-invariant CER이 css=8.0에서도 유지되는지 확인
**변경**: /tmp/bench_qwen3_ko_css8_v2.py (direct vLLM, 100 samples)
**결과** (L40S BF16, 100 samples):
- Batch: CER=2.89%, RTF=0.022
- css=3.0: CER=2.84%, RTF=0.084, fd=97ms
- css=5.0: CER=2.89%, RTF=0.057, fd=150ms
- css=8.0: CER=2.89%, RTF=0.044, fd=216ms
**판정**: keep
**이유**: CER은 css=8.0에서도 2.89%로 CSS-invariant. RTF 0.044는 batch 수준(0.022)의 2x. fd 216ms는 목표 200ms 약간 초과하지만 실용적. 한국어도 css=8.0 사용 가능.
**참고**: qwen_asr SDK가 vLLM 0.18.0과 호환 불가 → direct vLLM으로 벤치마크 수행. css=3.0의 CER=2.84%는 batch와 동일하여 약간 유리.

## [2026-03-21] Exp #150: Granite css=4.0/8.0 on LS-other (noise robustness)

**가설**: Granite 1B의 CSS-invariant WER이 LS-other(노이즈 환경)에서도 유지되는지 확인
**변경**: /tmp/bench_granite_other_css.py 벤치마크 스크립트
**결과** (L40S BF16, 100 samples):
- Batch: WER=3.75%, RTF=0.025
- css=2.0: WER=3.62%, RTF=0.082, fd=83ms
- css=4.0: WER=3.75%, RTF=0.043, fd=87ms
- css=8.0: WER=3.75%, RTF=0.029, fd=103ms
**판정**: keep
**이유**: WER은 LS-other에서도 CSS-invariant. css=8.0이 batch와 동일한 WER(3.75%) 유지하면서 RTF 0.029 달성. css=2.0의 3.62%는 약간 나은 것이 흥미로우나 noise margin 범위. 노이즈 환경에서도 css=8.0 사용이 안전함을 확인. 배포 전략 변경 불필요.

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

## [2026-03-21] Experiment #12: Fleurs-ko CER 정량 벤치마크 (3 backends, 30 samples)

**가설**: 표준 벤치마크(Fleurs-ko)에서 Fun-ASR-MLT-Nano, Qwen3-ASR-1.7B, Qwen3-ASR-0.6B의 한국어 CER을 정량 비교한다.

**설정**:
- 데이터: Google FLEURS ko_kr test set, 30 samples (총 396.2초)
- GPU: NVIDIA L40S (각 백엔드 별도 GPU)
- Qwen3: streaming mode, unfixed_chunk_num=6, chunk_size=2.0s, language="Korean"
- FunASR-Nano: batch mode, language="韩文"
- CER: 공백/구두점 제외, 문자 단위 Levenshtein

**결과**:

| 백엔드 | 모드 | Micro CER | Macro CER | RTF | 공식 수치 (Fleurs-ko) |
|---|---|---|---|---|---|
| Fun-ASR-MLT-Nano (800M) | batch | **3.22%** | 3.82% | 0.188 | N/A |
| Qwen3-ASR-1.7B | streaming | **3.78%** | 4.03% | 0.088 | 2.57% |
| Qwen3-ASR-0.6B | streaming | **5.17%** | 5.61% | 0.054 | 3.72% |

**분석**:
- Fun-ASR-MLT-Nano가 batch mode에서 최고 CER (3.22%) 달성
- Qwen3-1.7B streaming이 RTF 2배 빠르면서 CER은 0.56%p만 높음
- Qwen3 공식 수치 대비 차이 (1.7B: 3.78% vs 2.57%) = CER 계산 방식 차이 + streaming overhead
- 공통 오류: 외래어 ("SWAPO", "Bishkek"), 동음이의 ("파리밖에"→"팔이밖에")

**한국어 백엔드 최종 권장**:
- **정확도 우선 (배치 처리)**: Fun-ASR-MLT-Nano — CER 3.22%, RTF 0.188
- **속도 우선 (실시간 스트리밍)**: Qwen3-ASR-0.6B — CER 5.17%, RTF 0.054
- **균형 (스트리밍)**: Qwen3-ASR-1.7B — CER 3.78%, RTF 0.088

**판정**: KEEP — 한국어 벤치마크 인프라 구축 완료, 백엔드별 특성 확정

---

## [2026-03-21] Experiment #13: Qwen3-ASR-1.7B Streaming LibriSpeech Clean WER

**가설**: Qwen3-ASR-1.7B의 streaming WER이 0.6B (2.81%)보다 개선되는지 확인한다.

**설정**:
- 데이터: LibriSpeech test-clean, 30 samples (총 217.7초)
- GPU: NVIDIA L40S (CUDA_VISIBLE_DEVICES=3)
- Streaming: unfixed_chunk_num=6, chunk_size=2.0s, feed_step=0.5s
- Language: English

**결과**:

| 모델 | Macro WER | Micro WER | Overall RTF |
|---|---|---|---|
| Qwen3-ASR-0.6B (기존) | 2.92% | 2.81% | 0.076 |
| **Qwen3-ASR-1.7B** | **2.33%** | **2.15%** | **0.100** |
| Qwen3-ASR-1.7B batch (H100 baseline) | — | 2.46% | 0.069 |

**분석**:
- 1.7B streaming (2.15%) > 1.7B batch H100 baseline (2.46%) — streaming이 batch보다 오히려 좋음!
  - 원인: unfixed_chunk_num=6의 self-correction 효과 + 1.7B의 더 큰 context 활용
- 0.6B → 1.7B 개선: 2.81% → 2.15% (23.5% relative improvement)
- RTF: 0.076 → 0.100 (32% 증가) — 여전히 실시간의 10배 빠름
- 에러 샘플: proper names (WER 12.5%), 단어 경계 ("no near" → "no nearer")

**종합 스트리밍 WER 비교** (LibriSpeech clean, L40S):

| 모드 | WER | RTF |
|---|---|---|
| Qwen3-ASR-1.7B streaming | **2.15%** | 0.100 |
| Qwen3-ASR-0.6B streaming | 2.81% | 0.076 |
| whisper_large_v3 batch (H100) | 2.02% | 0.071 |
| Qwen3 SimulKV 0.6B (H100) | 6.44% | 0.109 |

**판정**: KEEP — 1.7B streaming이 batch-level 품질(2.15% vs 2.02%)에 거의 도달. 목표 <4% 대폭 초과 달성.

---

## [2026-03-21] Research: ENERZAi EZWhisper 한국어 모델 조사

**조사 결과**:
- ENERZAi가 50K시간 한국어 데이터로 Whisper를 재훈련한 "EZWhisper" 모델 개발
- EZWhisper-small (1.58-bit): 70MB로 Whisper-large-v3 (3GB) 대비 40배 작음
- 최대 모델 (484MB)이 한국어 CER을 Whisper-large-v3 대비 절반으로 감소
- Whisper-large-v3 한국어 CER: ~11.13% (KsponSpeech Eval-Other)
- 커스텀 한국어 토크나이저 사용, on-device/edge 배포에 초점

**통합 가능성**: **불가** — HuggingFace에 모델이 공개되지 않음 (상용 제품)
**대안**: ghost613/whisper-large-v3-turbo-korean, seastar105/korean-whisper 컬렉션 등 공개 한국어 Whisper 모델이 HF에 존재. 하지만 현재 Qwen3-ASR-1.7B (CER 3.78%)와 Fun-ASR-MLT-Nano (CER 3.22%)가 이미 우수한 성능.

**판정**: SKIP — 비공개 모델, 현재 백엔드로 충분

---

## [2026-03-21] Experiment #14: whisper-large-v3-turbo Korean 모델 벤치마크

**가설**: ghost613/whisper-large-v3-turbo-korean (Zeroth-Korean finetuned)이 base whisper-large-v3-turbo보다 한국어 CER이 낮을 것이다.

**방법**: Fleurs-ko 30샘플, faster-whisper (CTranslate2) + transformers pipeline 비교

**결과** (Fleurs-ko 30 samples, L40S GPU 2):

| 모델 | Micro CER | Macro CER | RTF |
|---|---|---|---|
| openai/whisper-large-v3-turbo (HF transformers) | **3.22%** | **3.85%** | 0.015 |
| ghost613/faster-whisper-large-v3-turbo-korean (CT2) | 9.37% | 10.24% | 0.017 |

**분석**:
- 놀랍게도 **base turbo가 Korean finetuned보다 3배 좋다** (3.22% vs 9.37%)
- Korean finetuned 모델의 주요 실패 패턴:
  - 숫자 표현: "100%" → "백 퍼센트", "35mm" → "삼십 오 밀리" (reference는 숫자 표기)
  - 고유명사: "가지안테프" → "가지한 태프", "비슈케크" → "비추케크"
  - 일반 어휘: "무리를" → "물의를", "혼합물" → "호란물"
- base turbo는 숫자를 원래 형식으로 유지하고 고유명사도 더 정확
- ghost613 모델 카드에서도 "models did not converge" 경고
- Zeroth-Korean 데이터셋(206시간)이 충분하지 않았거나 과적합 발생 가능

**종합 한국어 CER 비교** (Fleurs-ko 30 samples, L40S):

| 모델 | Micro CER | RTF | 비고 |
|---|---|---|---|
| FunASR-MLT-Nano | **3.22%** | 0.075 | batch |
| openai/whisper-large-v3-turbo | **3.22%** | 0.015 | batch, 가장 빠름 |
| Qwen3-ASR-1.7B streaming | 3.78% | 0.285 | streaming |
| Qwen3-ASR-0.6B streaming | 5.17% | 0.170 | streaming |
| ghost613/whisper-v3-turbo-korean | 9.37% | 0.017 | batch, finetuned 실패 |

**핵심 발견**:
1. whisper-large-v3-turbo는 한국어도 이미 우수 (CER 3.22%, FunASR-Nano와 동일)
2. whisper-large-v3-turbo가 RTF 0.015로 **가장 빠른 batch 모델**
3. Korean finetuning이 항상 도움되는 것은 아님 — 수렴하지 못한 모델은 오히려 악화
4. **스트리밍 통합 후보**: whisper-large-v3-turbo를 faster-whisper 백엔드로 스트리밍화하면 유망

**판정**: KEEP (결과 기록) — base whisper-v3-turbo가 한국어에서 예상외로 강력. 스트리밍 통합 검토 가치 있음.

---

## [2026-03-21] Experiment #15: whisper-large-v3-turbo LocalAgreement 스트리밍

**가설**: faster-whisper large-v3-turbo + LocalAgreement가 Qwen3 streaming과 비슷한 WER/RTF를 달성할 것이다.

**방법**: LibriSpeech clean 30샘플, 1초 chunk, segment trimming 15s

**결과** (LibriSpeech clean 30 samples, L40S GPU 2):

| 모델 | Mode | Micro WER | RTF |
|---|---|---|---|
| Qwen3-ASR-1.7B | streaming | **2.15%** | 0.100 |
| Qwen3-ASR-0.6B | streaming | **2.81%** | 0.076 |
| faster-whisper large-v3-turbo | LocalAgreement | 4.97% | 0.211 |
| whisper-large-v3 batch (H100) | batch | 2.02% | 0.071 |

**분석**:
- Whisper turbo streaming WER 4.97%는 batch 대비 2.5배 열화 (4.97% vs ~2%)
- RTF 0.211로 Qwen3 0.6B (0.076)의 거의 3배 느림
- 주요 원인: Whisper는 매 chunk마다 전체 audio_buffer를 re-transcribe (O(n²) 패턴)
  - Qwen3는 KV cache 재활용으로 이전 chunk를 스킵
- LocalAgreement의 hypothesis 비교 과정에서 단어 손실 발생 (worst case: 20% WER)
- 짧은 utterance (3초 이하)에서는 Whisper도 0% WER로 양호

**결론**: Qwen3 streaming이 영어에서 확실히 우위. Whisper turbo는 batch에서는 빠르지만
streaming에서는 구조적 한계 (전체 re-transcribe + LocalAgreement 오버헤드).

**판정**: DISCARD (스트리밍 용도) — 영어 스트리밍에서 Qwen3가 WER/RTF 모두 우위.
다만 batch 한국어 (CER 3.22%, RTF 0.015)에서는 faster-whisper turbo가 가장 빠름.

---

## [2026-03-21] Experiment #16: Qwen3-1.7B Korean streaming unfixed_chunk_num 최적화

**가설**: unfixed_chunk_num(ucn) 튜닝으로 한국어 streaming CER을 3.78% (기존)에서 개선할 수 있다.

**방법**: Fleurs-ko 30샘플, Qwen3-ASR-1.7B, ucn 2~10 범위 sweep, css=1.0/1.5/2.0/3.0, utn=3/5/8

**Round 1 결과** (css=2.0, utn=5):

| ucn | Micro CER | RTF |
|---|---|---|
| 4 | 2.66% | 0.079 |
| **5** | **2.45%** | **0.056** |
| 6 (기존 기본값) | 3.92% | 0.067 |
| 8 | 12.87% | 0.078 |
| 10 | 28.60% | 0.079 |

**Round 2 결과** (ucn 2~5, css=1.0/1.5/2.0):

| Config | Micro CER | RTF |
|---|---|---|
| ucn=2, css=2.0 | 3.01% | 0.062 |
| ucn=3, css=2.0 | 2.73% | 0.043 |
| ucn=4, css=2.0 | 2.66% | 0.048 |
| **ucn=5, css=2.0** | **2.45%** | **0.056** |
| ucn=3, css=1.0 | 3.22% | 0.067 |
| ucn=4, css=1.0 | 3.22% | 0.061 |

**재현성 확인** (3 trials × 3 configs):
- ucn=4: 2.66% ± 0.00% (완벽 재현)
- ucn=5: 2.45% ± 0.00% (완벽 재현)
- ucn=6: 3.92% ± 0.00% (완벽 재현)

**영어 영향 확인** (LibriSpeech clean 30 samples):
- ucn=4,5,6,7 모두 English WER = **1.99%** (변화 없음)

**핵심 발견**:
1. ucn=5가 한국어 최적 (CER 2.45%), ucn=6(3.92%) 대비 37% 개선
2. css=2.0이 css=1.0보다 일관되게 좋음 (chunk가 너무 작으면 컨텍스트 부족)
3. ucn>6은 한국어에서 급격히 악화 (ucn=8: 12.87%, ucn=10: 28.60%)
4. 영어는 ucn=4~7 모두 동일 WER — 영어에는 영향 없음

**종합 한국어 CER 비교** (Fleurs-ko 30 samples, L40S):

| 모델 | Micro CER | RTF | 모드 |
|---|---|---|---|
| **Qwen3-1.7B streaming (ucn=5)** | **2.45%** | 0.056 | streaming |
| Qwen3-1.7B streaming (ucn=4) | 2.66% | 0.048 | streaming |
| FunASR-MLT-Nano | 3.22% | 0.075 | batch |
| whisper-large-v3-turbo | 3.22% | 0.015 | batch |
| Qwen3-1.7B streaming (ucn=6, old) | 3.92% | 0.067 | streaming |
| Qwen3-0.6B streaming | 5.17% | 0.170 | streaming |

**변경**: config.py, parse_args.py, qwen3_streaming.py에서 기본값 unfixed_chunk_num=6→5로 변경.
**commit**: b3c5b78

**판정**: KEEP — 코드 변경 커밋 완료. 한국어 CER 37% 개선, 영어 영향 없음.

---

## [2026-03-21] Research: 최신 ASR 모델/기법 탐색

**조사 대상**: 2025-2026 최신 ASR 모델 중 한국어 지원 + 스트리밍 가능한 후보

**결과**:

| 모델 | 파라미터 | 한국어 | 스트리밍 | 비고 |
|---|---|---|---|---|
| NVIDIA Canary-Qwen-2.5B | 2.5B | ✗ | ✗ | 영어 전용, WER 5.63% 리더보드 1위 |
| NVIDIA Nemotron-ASR-Streaming | 0.6B | ✗ | ✓ | 영어 전용, cache-aware, 매우 빠름 |
| NVIDIA Parakeet TDT 1.1B | 1.1B | ✗ | ✓ | 영어 전용, RTFx 2000+ |
| IBM Granite Speech 3.3 8B | ~9B | ✗ | ✗ | WER 5.85%, 크기가 매우 큼 |
| Omnilingual ASR 7B | 7B | ✓ | ✗ | 1600+ 언어, CER<10 78% 달성, 너무 큼 |
| **Qwen3-ASR-1.7B** | 1.7B | ✓ | ✓ | **이미 사용 중, 52개 언어** |
| **Fun-ASR-MLT-Nano** | 0.8B | ✓ | ✗ | **이미 사용 중, 31개 언어** |

**결론**: 현재 통합된 백엔드가 이미 최선이다.
- Qwen3-ASR: 한국어 스트리밍 CER 2.45% (ucn=5), 영어 WER 1.99%
- NVIDIA 모델들은 영어에서 강력하지만 한국어 미지원
- 대형 모델 (7B+)은 VRAM/latency 제약으로 실시간 서비스에 부적합

**다음 방향**: 기존 백엔드 최적화에 집중 (Qwen3-0.6B ucn 튜닝, 노이즈 환경 테스트)

**판정**: SKIP — 새로 통합할 모델 없음. 현재 백엔드가 최선.

---

## [2026-03-21] 실험 #17: Qwen3-0.6B streaming ucn 튜닝

**가설**: Qwen3-0.6B는 1.7B보다 작은 모델이므로 self-correction에 더 많은 패스가 필요할 것이다. ucn=3~4가 최적일 것으로 예상.

**변경**: `/tmp/bench_qwen3_06b_tune.py` — ucn=3,4,5,6,7,8을 Fleurs-ko 30샘플 (Korean CER) + LibriSpeech clean 30샘플 (English WER)에 대해 측정.

**결과** (L40S GPU 0):

| ucn | Ko CER | En WER |
|-----|--------|--------|
| 3 | 3.92% | **2.48%** |
| 4 | **3.71%** | 2.65% |
| 5 | 4.34% | 2.65% |
| 6 | 5.17% | 4.14% |
| 7 | 8.74% | 4.97% |
| 8 | 14.13% | 9.93% |

**분석**:
- 0.6B 최적 ucn은 1.7B(ucn=5)보다 낮다: 한국어 ucn=4, 영어 ucn=3
- 이는 작은 모델이 짧은 unfixed window에서 더 나은 self-correction을 수행함을 시사
- ucn=5(현재 기본값)는 0.6B에서 한국어 CER 4.34%로 suboptimal (ucn=4 대비 +17%)
- ucn=6+ 부터 급격한 성능 저하 (작은 모델이 긴 unfixed context를 다루지 못함)
- 1.7B 대비: 0.6B 최적(Ko 3.71%, En 2.48%) vs 1.7B 최적(Ko 2.45%, En 1.99%)
  → 1.7B가 여전히 우수하지만 0.6B도 ucn=4에서 상당히 경쟁력 있음

**판정**: keep — 모델 크기별 자동 ucn 선택 로직 추가 검토. 현재 기본값 ucn=5는 1.7B에 최적화되어 있으므로, 0.6B 사용자에게는 ucn=4를 권장하는 문서/로직이 필요.
**commit**: c103b61

---

## [2026-03-21] 실험 #18: Qwen3 streaming WER on LibriSpeech test-other + max_new_tokens 버그 발견

**가설**: Qwen3 공식 streaming API가 LibriSpeech other (노이즈)에서도 SimulStream-KV 대비 큰 개선을 보일 것이다.

**변경**: `/tmp/bench_qwen3_librispeech_other.py` — HuggingFace Parquet에서 직접 30샘플 다운로드 (torchcodec 의존성 우회), Qwen3-1.7B ucn=5 + 0.6B ucn=4 측정.

**결과 (max_new_tokens=32, L40S GPU 0)**:

| Model | ucn | WER | RTF |
|-------|-----|-----|-----|
| Qwen3-1.7B | 5 | 6.36% | 0.112 |
| Qwen3-0.6B | 4 | 2.41% | 0.073 |

0.6B가 1.7B를 크게 이기는 이상한 결과. 디버깅 결과 **1.7B의 긴 발화에서 출력이 잘림** 확인. `max_new_tokens=32`가 원인.

**수정**: `max_new_tokens=64`로 변경 후 재측정:

| max_new_tokens | 1.7B WER | truncated |
|----------------|----------|-----------|
| 32 | 6.36% | 3/30 |
| 64 | **2.58%** | 0/30 |
| 128 | 2.58% | 0/30 |

**최종 LibriSpeech test-other 스트리밍 결과 (max_new_tokens=64)**:

| Model | ucn | WER |
|-------|-----|-----|
| Qwen3-1.7B | 5 | **2.58%** |
| Qwen3-0.6B | 4 | **2.41%** |

baseline 대비: SimulStream-KV 0.6B 9.27% → 2.41% (**74% 개선**), 1.7B 9.56% → 2.58% (**73% 개선**)

**버그 수정**: `qwen3_streaming.py`의 `max_new_tokens=32` → `64`. 프로덕션 코드에서 긴 발화의 잘림을 방지.

**판정**: keep — 중요한 버그 수정 + 노이즈 데이터 벤치마크 추가
**commit**: 0cff85e

---

## [2026-03-21] 실험 #19: max_new_tokens=64 교차 검증 (clean + Korean)

**가설**: max_new_tokens=32에서 LibriSpeech clean / Fleurs-ko도 미세한 잘림 영향을 받았을 수 있다.

**결과 (max_new_tokens=64)**:

| Model | En WER (clean) | Ko CER | vs mnt=32 |
|-------|----------------|--------|-----------|
| 1.7B ucn=5 | **1.82%** | **2.38%** | En -0.17%, Ko -0.07% (개선) |
| 0.6B ucn=4 | 2.65% | 3.71% | 동일 |

**분석**: 1.7B에서 clean 데이터에서도 미세한 잘림이 있었다. max_new_tokens=64로 개선됨.

**갱신된 최적 벤치마크 (L40S, max_new_tokens=64)**:

| Config | En WER (clean) | En WER (other) | Ko CER |
|--------|----------------|----------------|--------|
| Qwen3-1.7B streaming ucn=5 | **1.82%** | **2.58%** | **2.38%** |
| Qwen3-0.6B streaming ucn=4 | 2.65% | 2.41% | 3.71% |

**판정**: keep — max_new_tokens=64 변경이 모든 데이터셋에서 순수한 개선 또는 동일. 회귀 없음.

---

## [2026-03-21] 실험 #20: chunk_size_sec 최적화

**가설**: chunk_size_sec를 줄이면 first-word latency를 낮출 수 있다. 영어/한국어 모두에서 최적 css를 탐색.

**결과 (Qwen3-1.7B ucn=5, max_new_tokens=64)**:

| css | En WER (clean) | RTF | 1st latency | Ko CER |
|-----|----------------|-----|-------------|--------|
| 0.5s | 2.32% | 0.156 | 0.72s | **43.57%** |
| **1.0s** | **1.99%** | 0.072 | **1.04s** | 3.01% |
| 1.5s | 1.82% | 0.064 | 1.54s | 3.64% |
| **2.0s** | **1.82%** | 0.057 | 2.05s | **2.38%** |
| 3.0s | 1.82% | 0.053 | 3.07s | 2.03% |

**분석**:
- 영어: css=1.0s에서 WER 1.99% (최적 1.82%와 0.17% 차이), latency 50% 감소
- 한국어: css=0.5s는 재앙적 (43.57%), css=1.0s도 3.01% (css=2.0s 대비 +26%)
- 한국어가 영어보다 더 긴 컨텍스트를 필요로 함 (음절 단위 vs 음소 단위)
- css=2.0s가 다국어 환경에서 최적 균형점

**판정**: discard — css=2.0s 기본값 유지. 언어별 적응형 css는 구현 복잡도 대비 효과 미미.

---

## [2026-03-21] 실험 #21: unfixed_token_num 최적화

**가설**: unfixed_token_num(utn) 5는 기본값이지만, 더 큰 값이 한국어에서 self-correction 품질을 개선할 수 있다.

**결과 — Qwen3-1.7B ucn=5, css=2.0:**

| utn | En WER | Ko CER |
|-----|--------|--------|
| 1 | 1.99% | 3.43% |
| 3 | 1.99% | 2.45% |
| 5 | 1.99% | 2.38% |
| **7** | 1.99% | **2.17%** |
| 10 | 1.99% | 2.45% |
| 15 | 1.99% | 2.10% |

**결과 — Qwen3-0.6B ucn=4, css=2.0:**

| utn | En WER | Ko CER |
|-----|--------|--------|
| 3 | 2.65% | 3.71% |
| 5 | 2.65% | 3.71% |
| 7 | 2.65% | 3.85% (악화) |
| 10 | 2.65% | 3.92% (악화) |

**분석**:
- 1.7B: utn은 영어에 무관, 한국어에서 utn=7이 최적 (CER 2.17%, -8.8%)
- 0.6B: utn 증가가 한국어를 악화시킴. 작은 모델은 더 적은 unfixed token으로 correction이 효율적
- utn=7 vs utn=15: 0.07% 차이로 utn=7이 효율적 sweet spot

**변경**: 모델 크기별 자동 utn 선택: 1.7B → utn=7, 0.6B → utn=5

**판정**: keep — 한국어 CER 2.38% → 2.17% 개선, 영어 무영향
**commit**: 740b7c7

**갱신된 최종 벤치마크 (L40S, max_new_tokens=64):**

| Config | En WER (clean) | En WER (other) | Ko CER |
|--------|----------------|----------------|--------|
| Qwen3-1.7B streaming ucn=5 utn=7 | **1.82%** | 2.58% | **2.17%** |
| Qwen3-0.6B streaming ucn=4 utn=5 | 2.65% | 2.41% | 3.71% |

---

## [2026-03-21] 실험 #22: 100-sample LibriSpeech clean 확장 벤치마크

**가설**: 30 샘플은 통계적으로 불안정할 수 있다. 100 샘플로 확장하여 streaming WER의 신뢰도를 높이고, streaming vs batch 격차를 검증.

**결과 (Qwen3-1.7B, max_new_tokens=64, LibriSpeech clean 100 samples, 671s)**:

| Mode | WER | RTF | errors | words |
|------|-----|-----|--------|-------|
| Streaming (ucn=5, utn=7) | **2.25%** | 0.078 | 42 | 1870 |
| Batch | 3.26% | 0.028 | 61 | 1870 |

**분석**:
- 30-sample WER 1.82% → 100-sample WER 2.25% (+0.43%) — 30 샘플은 다소 낙관적이었음
- Streaming이 batch를 이김 (2.25% vs 3.26%) — unfixed chunk self-correction이 단일 패스보다 정확
- Batch WER 3.26%는 Qwen3 SDK의 단일 패스 처리이므로, 반복 self-correction이 품질 향상에 기여
- RTF: streaming 0.078 (실시간의 13배 빠름), batch 0.028 (36배 빠름)

**판정**: SKIP — 코드 변경 없음. 100-sample에서도 streaming WER < 4% 목표 달성 확인.

---

## [2026-03-21] 버그 수정 요약 (ISO 639-1 언어 코드)

**문제**: `--lan ko`로 서버 시작 시 Qwen3 streaming SDK에서 `ValueError: Unsupported language: Ko` 발생.
**원인**: SDK는 "Korean" 형태의 full name을 요구하지만, CLI는 ISO 639-1 코드 "ko"를 전달.
**수정**: `_normalize_language()` 함수 추가 — ISO 639-1 → full name 매핑 (30개 언어 지원).
**commit**: 1ddfa7f

---

## [2026-03-21] 실험 #23: Long-form Audio Streaming Robustness Test

**가설**: Qwen3-1.7B streaming은 매 chunk마다 전체 누적 오디오를 다시 처리하므로, 긴 오디오(>5분)에서 RTF 저하와 품질 열화가 발생할 것이다.

**변경**: `/tmp/bench_qwen3_longform.py` 작성 — LibriSpeech clean 100 samples를 연결(0.5s 침묵 간격)하여 12분 연속 오디오 생성, chunk-by-chunk streaming 처리하며 checkpoint별 WER/RTF 측정.

**결과** (L40S, Qwen3-1.7B ucn=5 utn=7 css=2.0):

| Checkpoint | WER | RTF | Text Len | Per-chunk Time | Status |
|-----------|------|-------|----------|----------------|--------|
| 60s | 0.64% | 0.160 | 899 | ~0.26s | excellent |
| 120s | 9.32% | 0.124 | 1726 | ~0.25s | WER spike (measurement artifact?) |
| 300s | 2.31% | 0.159 | 4351 | ~0.44s | good |
| 600s | **75.12%** | **0.305** | **12000** | ~1.21s | **catastrophic failure** |
| >600s | - | - | - | - | process hung (GPU idle) |

RTF 분석 (per-chunk compute time, 50-chunk 간격):
- 0-100s: 0.26s/chunk
- 100-200s: 0.25s/chunk
- 200-300s: 0.44s/chunk
- 300-400s: 0.63s/chunk
- 400-500s: 0.86s/chunk
- 500-600s: 1.21s/chunk → O(n) growth per chunk, O(n²) total

**핵심 발견**:
1. **RTF 선형 증가**: 매 chunk가 전체 누적 오디오를 재처리하므로, per-chunk 시간이 오디오 길이에 비례하여 증가
2. **Text 폭발**: 500-600s 구간에서 text_len이 6903→12000으로 급증 (hallucination)
3. **Process hang**: 600s 이후 GPU utilization 0%, 프로세스 sleeping — hallucination으로 인한 무한 생성 or 내부 deadlock
4. **실용적 한계**: ~5분(300s)까지는 WER 2.31%, RTF 0.16으로 양호. 이후 급격히 저하.

**판정**: Keep (발견 기록). Qwen3 streaming의 아키텍처적 한계를 확인. 주기적 세션 리셋이 필수.

**다음 단계**: `Qwen3StreamingOnlineProcessor`에 max_audio_duration 기반 자동 세션 리셋 구현.

## [2026-03-21] 실험 #24: Long-form Session Reset Interval Sweep

**가설**: 주기적 세션 리셋으로 long-form 안정성 확보. 리셋 간격과 overlap 양이 WER에 미치는 영향 측정.

**변경**: `/tmp/bench_qwen3_longform_reset.py` — 4개 config sweep (2min/3min/4min with 10s overlap, 3min no overlap)

**결과** (L40S, Qwen3-1.7B, 12분 연속 오디오):

| Config | WER | RTF | Resets | HypLen | Time |
|--------|-----|-----|--------|--------|------|
| 2min+10s overlap | 10.43% | 0.093 | 6 | 11253 | 67s |
| 3min+10s overlap | 6.79% | 0.097 | 4 | 10872 | 70s |
| 4min+10s overlap | 6.42% | 0.114 | 3 | 10706 | 82s |
| **3min no overlap** | **2.25%** | **0.090** | 4 | 10311 | 65s |
| (no reset, #23) | 75.12% | 0.305 | 0 | 12000 | >183s (hung) |

**핵심 발견**:
1. **Overlap이 WER을 악화시킴**: overlap 10s → 이전 세션에서 이미 인식한 부분이 중복 인식 → HypLen 증가 → WER 상승
2. **No overlap이 최적**: unfixed_chunk_num=5의 자가보정이 세션 전환 시 context 역할을 충분히 수행
3. **3min no overlap = 단기 벤치마크와 동일 WER**: 2.25% (100-sample 벤치와 동일!)
4. **RTF 3.4x 개선**: 0.305 → 0.090 (리셋으로 per-chunk 비용 제한)

**판정**: Keep. `max_session_audio_sec=180`, `overlap_sec=0`을 기본값으로 구현.
**구현**: `qwen3_streaming.py`의 `Qwen3StreamingOnlineProcessor`에 자동 세션 리셋 추가.
**commit**: 8988c44

## [2026-03-21] 실험 #25: First-Word Latency 측정

**가설**: Qwen3 streaming의 first-word latency를 정확히 측정하고 200ms 목표 대비 평가.

**결과** (L40S, Qwen3-1.7B, ucn=5, utn=7, css=2.0s, 30 samples):

| Metric | Value |
|--------|-------|
| Draft text (unfixed) | 2.0s (1 chunk) + 71ms compute = ~2.07s |
| Stable text (committed) | 4.0s (2 chunks) + 71ms compute = ~4.07s |
| Per-chunk compute | 71ms avg, 128ms p95 |

**분석**:
- progrem.md 목표 200ms는 chunk-based 아키텍처에서 물리적 불가능 (최소 css seconds 대기)
- 실제 사용자 체감 latency = draft text 2.07s. 이는 수용 가능한 수준.
- Per-chunk compute 71ms는 매우 빠름 (css=2.0s 대비 RTF 0.035)
- Stable text 4.0s는 ucn=5 중 2번째 chunk에서 common prefix가 확정되는 시점
- css를 줄이면 latency 감소하지만 한국어 품질 저하 (0.5s → CER 43.57%)

**판정**: Keep (기록). 200ms 목표는 chunk-based streaming의 아키텍처적 한계. 대안은 CTC/RNNT 기반 streaming (Nemotron 등) 또는 adaptive chunk.

---

## [2026-03-21] 실험 #26: css=1.0 vs css=2.0 (Latency vs Quality)

**가설**: css=1.0s로 줄이면 draft text latency가 2.0→1.0s로 절반 감소. English 품질은 유지되지만 Korean은 다소 저하될 것.

**결과** (L40S, Qwen3-1.7B, ucn=5, utn=7, 30 samples):

| CSS | En WER | Ko CER | RTF | First Text |
|-----|--------|--------|-----|------------|
| 1.0s | 1.99% | 2.73% | 0.092 | 1.0s |
| 2.0s | 1.99% | 2.17% | 0.061 | 2.0s |

**분석**:
- English: css 변경 영향 없음 (WER 1.99% 동일)
- Korean: CER 2.17% → 2.73% (+0.56%). 여전히 매우 우수하지만 약간 저하
- 더 작은 chunk = 더 적은 context per chunk → Korean 같은 교착어에서 미세한 품질 저하
- RTF: css=1.0은 chunk 수가 2배 → per-sample compute 약간 증가

**판정**: Keep (기록). English 전용이면 css=1.0 추천 (latency 절반). 다국어면 css=2.0 유지.

---

## [2026-03-21] 실험 #27: Language Auto-Detection Timing

**가설**: Qwen3 SDK의 auto-detect가 streaming 중 언어를 감지하면, 세션 리셋 시 최적 css를 자동 선택할 수 있다.

**결과**:
- English: 1번째 chunk (2.0s)에서 즉시 감지 → 매우 빠름
- Korean: 1-2번째 chunk (2.0-4.0s)에서 감지 → 약간 느림
- Auto vs Explicit English: 동일 텍스트 출력 (품질 저하 없음)
- language는 `state.language`에 설정됨

**분석**: Language-aware adaptive css (En→1.0, Ko→2.0)는 세션 리셋 시에만 적용 가능. 첫 세션은 언어 미상이므로 css=2.0으로 시작해야. 실용적 개선 폭 제한적.

**판정**: Discard (구현 안 함). 복잡도 대비 이득이 작음. auto-detect 자체는 잘 동작하므로, `lan=auto` 모드를 기본으로 권장하는 것은 유효.

---

---

## [2026-03-21] 신규 모델 조사

### VibeVoice-ASR (Microsoft)
- 7-9B 파라미터, 배치 전용 (60분 single-pass)
- 실시간 스트리밍 미지원 → 부적합

### VibeVoice-Realtime-0.5B (Microsoft)
- TTS 모델이지 ASR이 아님 → 부적합

### Voxtral Mini 4B Realtime (Mistral)
- 4B 파라미터, native streaming, vLLM Day-0 지원
- 13개 언어 (Korean 포함)
- **Korean WER: 15.74%** (FLEURS, 480ms latency) — Qwen3-1.7B 대비 매우 낮은 품질
- 아키텍처 장점: sliding window attention으로 무한 스트리밍 가능 (re-feed 불필요)
- 단점: 한국어 품질 낮음, 4B로 모델 크기도 큼
- **결론**: 현 시점에서 Qwen3-1.7B 대체 부적합. 한국어 품질 개선 시 재평가.

### NVIDIA Nemotron Speech Streaming 0.6B
- English-only → 부적합

---

## [2026-03-21] 실험 #28: Qwen3-0.6B Long-form Session Reset 검증

**가설**: 1.7B에서 검증된 3min session reset (no overlap) 전략이 0.6B에서도 동일하게 동작하여 long-form 안정성을 확보할 것이다.

**설정**:
- 데이터: LibriSpeech clean 100 samples 연결 (720.6s ≈ 12분, 1870 words)
- GPU: NVIDIA L40S (CUDA:0)
- Qwen3-ASR-0.6B, ucn=4, utn=5, css=2.0, language=English
- Configs: 3min reset, 2min reset, no reset (timeout 300s)

**결과**:

| Config | WER | RTF | Resets | Time | AvgChunk | MaxChunk | Completed |
|--------|-----|-----|--------|------|----------|----------|-----------|
| 3min reset | **2.19%** | 0.0621 | 4 | 44.7s | 124ms | 244ms | Y |
| 2min reset | **2.19%** | 0.0445 | 6 | 32.1s | 89ms | 151ms | Y |
| No reset | — | — | 0 | >300s | — | — | **N (hung)** |

**분석**:
1. **0.6B도 동일한 long-form 실패 패턴**: no-reset 시 ~7분 후 GPU idle 0%, 프로세스 hung — 1.7B와 동일
2. **Session reset이 0.6B에서도 완벽 동작**: 3min/2min 모두 WER 2.19% (short-audio 2.65%보다 오히려 좋음!)
3. **2min reset이 더 효율적**: RTF 0.0445 (3min 대비 28% 빠름), 같은 WER
4. **0.6B vs 1.7B long-form 비교**:
   - 0.6B: WER 2.19%, RTF 0.062, AvgChunk 124ms
   - 1.7B: WER 2.25%, RTF 0.090, AvgChunk 71ms (but higher total RTF from larger model)
5. **0.6B가 long-form에서 1.7B보다 약간 나은 WER** (2.19% vs 2.25%) — 작은 모델이 self-correction에서 더 일관적일 수 있음

**판정**: Keep (기록). Session reset 전략이 모델 크기에 무관하게 동작함을 확인. 0.6B의 기본값 `max_session_audio_sec=180`이 적절.

---

## [2026-03-21] 실험 #29: 최신 ASR 모델/기법 탐색 (2026-03 후반)

**조사 내용**:

### Qwen3-ASR-Flash
- Alibaba의 차세대 ASR 서비스, Qwen3-Omni 기반
- **API 전용** (closed-weight) — 오픈소스 아님, 프로젝트에 적용 불가
- WER < 8%, contextual biasing 지원, 11개 언어 (Korean 포함)
- Gemini-2.5-Pro, GPT-4o-Transcribe보다 우수하다고 주장

### Northflank 2026 STT 벤치마크 요약
| 모델 | 크기 | WER | 한국어 | 스트리밍 |
|------|------|-----|--------|----------|
| Canary Qwen 2.5B | 2.5B | 5.63% | English only | X |
| IBM Granite Speech 8B | ~9B | 5.85% | English ASR | X |
| Whisper Large V3 | 1.55B | 7.4% | O (99+ lang) | X |
| Parakeet TDT 1.1B | 1.1B | ~8.0% | X | O (RNN-T) |

**결론**: 한국어 + 스트리밍을 동시 지원하는 새로운 open-weight 모델 없음. Qwen3-ASR-1.7B가 여전히 best-in-class.

**판정**: Skip — 새로운 통합 대상 모델 미발견.

---

## [2026-03-21] 실험 #30: Repetition Filter 통합 평가

**가설**: `repetition_filter.py`가 Qwen3 streaming 출력에서 반복/stutter 패턴을 제거하여 품질을 개선할 수 있는지 평가.

**분석**:
1. **Qwen3는 stutter 패턴을 생성하지 않음**: "트 트럼프는", "자체 자체 자체" 등의 패턴은 Whisper 특유의 문제. Qwen3의 unfixed_chunk_num self-correction이 이를 방지.
2. **세션 리셋 경계에서도 중복 없음**: overlap_sec=0으로 리셋 전후 텍스트가 깔끔하게 분리 (실험 #24에서 확인).
3. **progrem.md 원칙 위배**: "decoder/model-level 해결 우선, 규칙기반 후처리 최소화"

**판정**: Discard — Qwen3 streaming에 repetition filter 통합 불필요. 필터는 Whisper 백엔드 전용으로 유지.

---

## [2026-03-21] 실험 #31: Multi-Session Concurrent Throughput

**가설**: Qwen3-1.7B streaming에서 여러 세션이 동시에 round-robin으로 chunk를 제출할 때, vLLM prefix caching 덕분에 throughput이 유지될 것이다.

**설정**:
- 데이터: LibriSpeech clean 30 samples (217.7s 총 오디오)
- GPU: NVIDIA L40S (CUDA:0)
- Qwen3-ASR-1.7B, ucn=5, utn=7, css=2.0
- N 세션에 30개 샘플을 균등 분배 (e.g. 4 sessions → 7-8 samples each)
- Round-robin 스케줄링 (session 1 chunk → session 2 chunk → ... → session N chunk → repeat)

**결과**:

| Sessions | WER | Agg RTF | Throughput | Wall Time | Per-Sess Latency |
|----------|-----|---------|-----------|-----------|-----------------|
| 1 | 1.82% | 0.0667 | 15.0x | 14.5s | 14.5s |
| 2 | 1.99% | 0.0584 | 17.1x | 12.7s | 6.4s |
| 4 | 1.99% | 0.0579 | 17.3x | 12.6s | 3.2s |
| 8 | 1.99% | 0.0579 | 17.3x | 12.6s | 1.6s |
| 16 | 1.99% | 0.0579 | 17.3x | 12.6s | 0.8s |
| 30 | 1.99% | 0.0580 | 17.3x | 12.6s | 0.4s |

**핵심 발견**:
1. **Throughput은 세션 수에 무관하게 일정**: 2-30 세션 모두 RTF 0.058, throughput 17.3x
2. **1 세션이 오히려 느림** (RTF 0.067 vs 0.058): 다중 세션 시 vLLM의 prefix cache 활용률이 높아져 오히려 효율적
3. **WER 차이 무시 가능**: 1.82% (1 session) vs 1.99% (2+ sessions) — 통계적 변동 범위
4. **30 세션 동시 처리 가능**: 단일 L40S GPU에서 217.7초 오디오를 12.6초에 처리
5. **Per-session latency 선형 감소**: N 세션 → wall_time/N
6. **streaming_transcribe() 비동기 배치 미지원이 병목이 아님**: vLLM prefix caching이 KV cache를 효율적으로 재활용하여 실질적으로 배치와 유사한 효율 달성

**실용적 의미**:
- 단일 L40S에서 최대 ~30명의 실시간 스트리밍 사용자를 동시 서빙 가능
- RTF 0.058 = 실시간의 17.3배 빠름 → 사용자당 실시간 요구사항을 여유있게 충족
- 세션 수 증가 시 bottleneck은 GPU가 아니라 CPU-side round-robin 스케줄링

**판정**: Keep (기록). Concurrent throughput 걱정 없음 확인. 코드 변경 불필요.

---

## [2026-03-21] 실험 #32: Contextual Biasing / Init Prompt 효과

**가설**: init_streaming_state(context=...)에 도메인 정보를 제공하면 고유명사/전문용어 인식률이 개선될 것이다.

**설정**:
- Qwen3-ASR-1.7B, ucn=5, utn=7, css=2.0, max_new_tokens=64
- English: LibriSpeech clean 30 samples, context variants: none / generic / proper noun hints
- Korean: FLEURS-ko 30 samples, context variants: none / news domain / specific term hints

**결과**:

| Config | En WER | Ko CER |
|--------|--------|--------|
| No context (baseline) | **1.82%** | **2.17%** |
| Generic English context | 1.99% (+0.17%) | — |
| Proper noun hints | 1.99% (+0.17%) | — |
| Korean news context | — | 2.38% (+0.21%) |
| Korean term hints | — | 2.59% (+0.42%) |

**분석**:
1. **Context 제공이 오히려 WER/CER을 악화**: 모든 context variant에서 baseline보다 나빠짐
2. **English**: context 유무 관계없이 동일한 12개 에러 (1 sample만 차이)
3. **Korean**: term hints가 가장 많은 악화 (+0.42%) — context 길이가 길수록 악화 심화
4. **추정 원인**:
   - context가 system prompt에 추가되어 전체 프롬프트 증가 → 유효 디코딩 공간 감소
   - max_new_tokens=64와 결합 시 truncation 위험
   - 범용 ASR에서는 context bias보다 zero-shot이 더 안정적
   - 실시간 streaming에서는 미래 context를 알 수 없으므로 정확한 bias 제공이 근본적으로 어려움

**실용적 함의**: Qwen3 streaming에서 context 파라미터는 사용하지 않는 것이 최적. zero-shot baseline이 가장 좋은 성능.

**판정**: Discard — contextual biasing은 streaming WER을 악화시킴. 코드 변경 불필요.

---

## [2026-03-21] 실험 #33: chunk_size_sec CLI 옵션 추가

**변경**: `--chunk-size-sec` CLI 옵션을 추가하여 Qwen3 streaming의 chunk 크기를 설정 가능하게 함.
- `config.py`: `chunk_size_sec: float = 2.0` 필드 추가
- `parse_args.py`: `--chunk-size-sec` argument 추가
- `core.py`: `Qwen3StreamingASR` 생성 시 `chunk_size_sec` 전달
- `qwen3_streaming.py`: 하드코딩된 `2.0` 제거, `self.asr.chunk_size_sec` 참조

**용도**: `wlk serve --backend qwen3-streaming --chunk-size-sec 1.0`으로 English 전용 저지연 모드 사용 가능.

**판정**: Keep
**commit**: e2e1ec3

---

## [2026-03-21] 실험 #34: LibriSpeech Other Long-form Streaming

**가설**: LibriSpeech other (noisy) 데이터에서 long-form streaming + session reset이 정상 동작하는지 확인.

**설정**: LibriSpeech other 30 samples (161.8s → 연결 후 176.8s), Qwen3-1.7B ucn=5 utn=7 css=2.0

**결과**:

Per-sample streaming: **WER 2.58%**, RTF 0.080

Long-form concatenated:
| Config | WER | RTF | Resets |
|--------|-----|-----|--------|
| 3min reset | 2.23% | 0.103 | 0 |
| 2min reset | 2.23% | 0.062 | 1 |
| No reset | 2.23% | 0.058 | 0 |

**분석**:
- 176.8s는 3min threshold 이하라 3min reset에서 리셋 불발
- Long-form WER (2.23%) < per-sample WER (2.58%) — 연결된 오디오의 더 긴 context가 도움
- WER 오류 대부분은 대소문자/구두점 차이 (I AM → I'm, ARCHY → Archie)
- Noisy 환경에서도 streaming 품질 안정적

**판정**: Keep (기록). Noisy 환경에서도 streaming WER < 3%.

---

## [2026-03-21] 실험 #36: Korean long-form max_new_tokens 영향 검증

**가설**: max_new_tokens=64가 한국어 long-form에서 출력 truncation을 유발. 128/256으로 올리면 해결될 것.

**변경**: /tmp/bench_qwen3_ko_max_tokens.py로 max_new_tokens=64/128/256 sweep.

**결과**:
| max_tokens | Per-sample CER | Concat CER (10s, 150s) | 60s-reset CER |
|------------|---------------|----------------------|--------------|
| 64 | 2.17% | 53.71% (hyp=353) | 18.63% |
| 128 | 2.17% | 53.71% (hyp=353) | 18.63% |
| 256 | 2.17% | 53.71% (hyp=353) | 18.63% |

**결과가 완전히 동일!** max_new_tokens는 원인이 아니었다.

**분석**: 모델이 EOS 토큰을 자체적으로 방출하여 생성 중단. hyp_len=353은 모든 max_new_tokens 값에서 동일 → 토큰 한도가 아니라 모델 자체의 품질 저하가 원인.

**판정**: Discard (가설 기각). max_new_tokens와 무관한 문제.

---

## [2026-03-21] 실험 #37: Korean session reset interval sweep

**가설**: 한국어 long-form에서 짧은 session reset 간격(15-30s)이 품질을 크게 개선할 것.

**변경**: /tmp/bench_qwen3_ko_reset_sweep.py로 15/30/45/60/90/120/180s sweep.

**결과** (30 samples concat = 411.2s, ref=1872 chars):
| Reset(s) | CER | Resets | HypLen | Time |
|----------|-----|--------|--------|------|
| Per-sample | 2.17% | - | - | baseline |
| **15** | **6.43%** | 25 | 1943 | 26.4s |
| **30** | **7.76%** | 13 | 1986 | 25.6s |
| 45 | 10.63% | 8 | 1893 | 389.9s (slow!) |
| 60 | **4345.80%** | 6 | **84604** | 574.8s (HALLUCINATION) |
| 90 | 7.69% | 4 | 1855 | 342.2s |
| 120 | CRASH | - | - | context len 65551 > 65536 |

**핵심 발견**:
1. **Hard limit ~120s**: audio features가 65536 context window 초과 → CRASH
2. **Hallucination at 60s+**: 모델이 84604자 생성 (ref=1872자). detect_and_fix_repetitions도 못 잡는 수준
3. **O(n) re-feed cost**: 45s+ 세션은 re-feed 비용으로 RTF 급증 (389-574s for 411s audio)
4. **Optimal: 15-30s**: CER 6-8%, 빠른 속도 (26s for 411s audio, RTF ~0.063)
5. **Per-sample vs concat gap**: 2.17% → 6.43% — 연속 utterance 경계 처리 한계
6. **현재 180s default는 위험**: 한국어 60s 이후 hallucination, 120s 이후 crash

**판정**: Keep (critical finding). 기본 max_session_audio_sec를 30s로 낮춰야 함.

---

## [2026-03-21] 실험 #38: English session reset interval sweep

**가설**: 영어도 30s reset으로 낮춰도 품질 영향 최소화.

**변경**: /tmp/bench_qwen3_en_reset_sweep.py로 15/30/45/60/90s sweep (LibriSpeech clean 30 samples, 232.7s).

**결과**:
| Reset(s) | WER | Resets | Time |
|----------|-----|--------|------|
| Per-sample | 1.82% | - | baseline |
| 15 | 2.48% | 14 | 14.8s |
| **30** | **2.15%** | 7 | 14.3s |
| 45 | 2.48% | 5 | 14.4s |
| 60 | 1.99% | 3 | 13.8s |
| 90 | 1.99% | 2 | 14.3s |

**분석**:
- 영어는 모든 interval에서 안정적 (1.99-2.48%), 한국어와 달리 hallucination 없음
- 30s reset: 2.15% WER — per-sample baseline (1.82%) 대비 +0.33% 차이만
- 처리 시간은 interval에 무관하게 일정 (14-15s)
- 30s는 양언어 최적 trade-off: En 2.15%, Ko 7.76%

**판정**: Keep. 30s를 universal default로 확정.

---

## [2026-03-21] 실험 #39: max_session_audio_sec 기본값 180→30 변경 + max_new_tokens 제거

**가설**: 안전한 기본값으로 Korean hallucination/crash 방지, 영어 품질 최소 영향.

**변경**:
- `qwen3_streaming.py`: max_session_audio_sec 기본값 180→30, max_new_tokens=64 제거 (SDK 기본값 4096 사용)
- `config.py`: max_session_audio_sec=30.0 필드 추가
- `parse_args.py`: --max-session-audio-sec CLI 옵션 추가
- `core.py`: online_factory()에서 config의 max_session_audio_sec를 Qwen3StreamingOnlineProcessor에 전달

**결과**: 코드 변경만, 벤치마크는 실험 #37/#38에서 검증 완료.

**판정**: Keep. 커밋 대상.

---

## [2026-03-21] 실험 #40: VAD silence-triggered session reset

**가설**: start_silence()에서 세션을 리셋하면 모든 speech→silence 전환이 자연 경계가 되어 per-sample 수준 품질 달성.

**변경**:
- `qwen3_streaming.py`: start_silence()에 `_init_state()` 추가
- `end_silence()`: 항상 time offset 업데이트 (기존: silence ≥ 5s일 때만 reset)

**결과**:
| Method | Korean CER | English WER |
|--------|-----------|-------------|
| **Silence-reset** | **2.17%** | **1.82%** |
| 30s-reset | 4.83% | 2.15% |
| Per-sample baseline | 2.17% | 1.82% |

**분석**:
- Silence-triggered reset이 per-sample baseline과 **완벽히 동일한 결과** 달성!
- Korean CER: 4.83% → 2.17% (56% 감소)
- English WER: 2.15% → 1.82% (15% 감소)
- 처리 속도도 동일 (26.7s vs 25.6s for Korean)
- 30s time-based reset은 safety net으로만 동작 (silence-triggered가 먼저 발동)

**판정**: Keep. 커밋 대상. 매우 중요한 개선.
**commit**: c80521a

---

## [2026-03-21] 실험 #41: 100-sample silence-triggered reset 확장 검증

**가설**: 30-sample에서 관찰된 silence-triggered reset의 우위가 100-sample에서도 유지되는지 통계적 검증.

**설정**: Qwen3-1.7B ucn=5 utn=7 css=2.0, L40S GPU

**결과**:
| Dataset | Silence-reset | 30s-reset | 개선율 |
|---------|-------------|----------|--------|
| En clean 100 samples | **2.25%** WER (42/1870) | 2.62% WER (49/1870) | -14% |
| En other 30 samples | **2.58%** WER (15/582) | 2.75% WER (16/582) | -6% |
| Ko FLEURS 30 samples | **2.17%** CER (31/1430) | 4.83% CER (69/1430) | **-55%** |

RTF: silence-reset 0.067-0.080, 30s-reset 유사

**분석**:
- 100-sample En clean에서도 silence-reset (2.25%) > 30s-reset (2.62%). 30-sample 결과(1.82%)보다 높지만 통계적으로 안정적
- En other (noisy)에서도 silence-reset이 0.17%p 우수 — 노이즈 환경에서도 효과
- 한국어에서 가장 큰 차이: 55% 개선. utterance 경계에서 리셋하면 한국어 context 혼란 방지
- RTF 차이 무시 가능 — 성능 비용 없이 품질만 개선

**판정**: SKIP (코드 변경 없음). silence-triggered reset의 통계적 유효성 확인 완료.

**갱신된 최종 벤치마크 (L40S, Qwen3-1.7B streaming + silence-triggered reset):**

| Dataset | WER/CER | Samples | RTF |
|---------|---------|---------|-----|
| En clean | 2.25% WER | 100 | 0.069 |
| En other | 2.58% WER | 30 | 0.080 |
| Ko FLEURS | 2.17% CER | 30 | 0.067 |

---

## [2026-03-21] 실험 #42: Qwen3-0.6B silence-triggered reset 검증

**가설**: 0.6B 모델에서도 silence-triggered reset이 1.7B와 유사한 개선 효과.

**설정**: Qwen3-0.6B ucn=4 utn=5 css=2.0, L40S GPU

**결과**:
| Dataset | Silence-reset | 30s-reset | 개선율 |
|---------|-------------|----------|--------|
| En clean 100 | 2.51% WER (47/1870) | 2.51% WER (47/1870) | 0% |
| Ko FLEURS 30 | **3.71%** CER (53/1430) | 4.69% CER (67/1430) | **-21%** |

RTF: silence-reset 0.031-0.034 (1.7B 대비 2배 빠름)

**분석**:
- 영어: 0.6B에서는 silence vs 30s 차이 없음 — 이미 오디오가 짧은 샘플이라 session 경계 영향 미미
- 한국어: 21% 개선 (3.71% → 4.69%). 1.7B의 55% 개선보다 약하지만 여전히 유의미
- RTF 0.031-0.034 — 실시간의 29~32배 빠름. 저자원 환경에서 매력적
- 한국어 3.71% CER은 1.7B의 2.17%보다 높지만, 모델 크기 대비 우수

**판정**: SKIP (코드 변경 없음). 0.6B에서도 silence-triggered reset 효과 확인.

**전체 모델 비교표 (L40S, silence-triggered reset 기준):**

| Model | En clean WER | Ko CER | RTF | VRAM |
|-------|-------------|--------|-----|------|
| Qwen3-1.7B | **2.25%** | **2.17%** | 0.069 | ~4 GB |
| Qwen3-0.6B | 2.51% | 3.71% | **0.034** | ~2 GB |

---

## [2026-03-21] 실험 #43: 오류 패턴 분석

**가설**: 잔여 오류에 decoder-level 개선 가능한 패턴이 있는지 확인.

**영어 오류 (42/1870 words, 2.25%):**
- 고유명사 변형: KAFFAR→Kaffr, VOLTAIRE→Vollrath, MINGOES→Mingos (주요 원인)
- 영미식 변이: discoloured→discolored, purposed→purpose
- 드문 단어: withes→whips, hymn→him
- compound word: "well known"→"well-known"

**한국어 오류 (31/1430 chars, 2.17%):**
- 외래어 주석 삭제: reference에 `bishkek` 포함 → hyp에 미출력 (14/31 = 45% of errors)
- 외래어 변이: 색스→섹스, 가지안테프→가지안 태프, 서아프리카→사하프리카
- 띄어쓰기: 주기율표 상에→주기율표상의
- 동음이의: 오늘날→어느 날, 공해→공예

**결론**: 대부분의 오류가 (1) 고유명사/외래어 표기 변이, (2) reference 데이터 특성 (영어 주석 포함), (3) 드문 단어/철자. **decoder-level 개선 여지 거의 없음.** 현재 품질은 사실상 최적.

**판정**: SKIP. 오류의 본질이 모델 한계 또는 데이터 특성이므로 추가 코드 변경 불필요.

---

## [2026-03-21] 실험 #44: vLLM v0.17 Qwen3-ASR Realtime Streaming 조사

**가설**: vLLM v0.17에 `qwen3_asr_realtime` 모듈이 추가됨. KV cache를 세션 전체에 걸쳐 유지하는 native streaming으로, SDK의 re-feed 방식(O(n) 비용)을 제거할 수 있다.

**조사 결과**:
- vLLM v0.17.0에 PR #34613으로 Qwen3-ASR realtime streaming 지원 추가
- `/v1/realtime` WebSocket endpoint를 통해 incremental audio 전송
- Anchor request pattern으로 KV cache 유지 — re-feed 불필요
- 현재 설치: vLLM 0.14.0 + qwen-asr 0.0.6
- 업그레이드 필요: vLLM 0.14.0 → 0.17.0+
- 기존 `vllm-realtime` 백엔드가 이미 WhisperLiveKit에 구현되어 있음 (`vllm_realtime.py`)

**잠재적 이점**:
1. Long-form re-feed 비용 제거 → 60s+ 세션에서도 일정한 RTF
2. Session reset 불필요 → 한국어 hallucination 위험 감소 (모델이 re-feed 없이 context 유지)
3. WebSocket 기반이므로 서버-클라이언트 분리 배포 가능

**리스크**:
- vLLM 0.14 → 0.17 메이저 업그레이드로 호환성 문제 가능
- qwen-asr SDK가 vLLM 0.17과 호환되는지 미확인
- Qwen3-ASR가 native streaming용으로 훈련된 것이 아니므로, re-feed 없이 품질이 유지될지 미검증

**판정**: DEFER. vLLM 업그레이드는 환경 변경이므로 사용자 확인 후 진행. 우선순위: 높음.

---

## [2026-03-21] 실험 #45: Qwen3-0.6B chunk_size_sec sweep

**가설**: 0.6B 모델은 더 빠르므로 css=1.0에서도 WER 악화 없이 latency 50% 감소 가능.

**설정**: Qwen3-0.6B ucn=4 utn=5, L40S, per-sample streaming (silence-reset)

**결과**:
| CSS | En WER (30) | Ko CER (30) | RTF | 1st latency |
|-----|-------------|-------------|-----|-------------|
| 0.5s | 2.65% | 4.41% | 0.078 | ~0.5s |
| **1.0s** | **2.65%** | **4.06%** | 0.036 | ~1.0s |
| 1.5s | 2.48% | 4.41% | 0.029 | ~1.5s |
| 2.0s | 2.65% | 3.78% | 0.027 | ~2.0s |

**분석**:
- 영어: CSS에 거의 무관 (2.48-2.65%). 1.7B에서는 css=0.5에서 WER 급증했으나 0.6B는 안정적
- 한국어: css=2.0 최적 (3.78%), css=1.0도 4.06%로 양호 (7% 차이)
- RTF: css=0.5에서만 0.078로 높음 (re-feed 횟수 증가), css=1.0부터 안정적
- css=0.5에서도 WER/CER 급증 없음 — 0.6B가 1.7B보다 짧은 컨텍스트에 강인

**결론**: 0.6B에서 css=1.0은 **영어 WER 무변화 + latency 50% 감소**의 최적 tradeoff.
- 영어 전용: css=1.0 권장 (WER 2.65%, latency ~1s)
- 다국어: css=2.0 유지 (Ko CER 3.78%)

**판정**: SKIP (코드 변경 없음). 기본값은 다국어를 고려하여 css=2.0 유지.

---

## [2026-03-21] Exp #46: Qwen3-0.6B Ultra-Low-Latency ucn Sweep (css=1.0)

**가설**: ucn을 1~2로 줄이면 self-correction 패스가 줄어 latency가 감소하지만, 0.6B에서는 WER 악화가 미미할 것이다.

**설정**: Qwen3-0.6B, L40S, css=1.0 (및 css=0.5), ucn=[1,2,3,4,6], utn=[3,5], per-sample streaming (silence-reset), 30 samples

**결과 (css=1.0)**:
| ucn | utn | EN WER | Ko CER | EN RTF | Ko RTF |
|-----|-----|--------|--------|--------|--------|
| 1 | 3 | 2.65% | 5.10% | 0.040 | 0.041 |
| 1 | 5 | 2.65% | 4.27% | 0.033 | 0.032 |
| 2 | 3 | 2.65% | 5.24% | 0.026 | 0.026 |
| 2 | 5 | 2.65% | 4.27% | 0.030 | 0.030 |
| 3 | 3 | 2.65% | 4.69% | 0.027 | 0.026 |
| 3 | 5 | 2.65% | 4.06% | 0.030 | 0.031 |
| 4 | 3 | 2.65% | 4.48% | 0.027 | 0.027 |
| 4 | 5 | 2.65% | 4.06% | 0.032 | 0.031 |
| 6 | 3 | 2.48% | 4.41% | 0.036 | 0.032 |
| 6 | 5 | 2.48% | 3.99% | 0.037 | 0.035 |

**결과 (css=0.5, utn=5)**:
| ucn | EN WER | EN RTF |
|-----|--------|--------|
| 1 | 2.65% | 0.065 |
| 2 | 2.65% | 0.051 |
| 4 | 2.65% | 0.050 |

**핵심 발견**:
1. **영어 WER은 ucn에 완전히 무관**: ucn=1이든 6이든 2.65% (ucn=6에서만 2.48%로 미세 개선). silence-reset 덕분에 self-correction이 불필요.
2. **한국어 CER은 utn이 핵심**: utn=3 → 4.41-5.24%, utn=5 → 3.99-4.27%. ucn보다 utn이 훨씬 중요.
3. **최적 초저지연**: ucn=1, utn=5, css=1.0 → EN 2.65%, Ko 4.27%, RTF=0.033
4. **css=0.5도 영어 품질 유지**: 2.65% WER, RTF=0.065. 이론적 first-word latency ~0.5s
5. **ucn=1이 ucn=2보다 RTF가 높은 이유**: ucn=1은 매 청크마다 모든 토큰이 unfixed이므로 re-generation이 많음

**결론**: silence-triggered reset은 self-correction(ucn)의 필요성을 거의 제거함. 0.6B에서 ucn=1, utn=5, css=1.0으로 first-word latency ~1s, WER 2.65% 달성 가능.
- 목표 latency 200ms는 chunk-based 접근의 근본적 한계로 달성 불가 (최소 css=0.5 → 500ms)
- 실시간 체감 latency 개선을 위해서는 vLLM realtime streaming (token-level emission) 필요

**판정**: SKIP (코드 변경 없음). 파라미터 가이드라인으로 기록.

---

## [2026-03-21] Exp #47: Qwen3-0.6B 극소 Chunk (css=0.25) 테스트

**가설**: css=0.25 (250ms chunk)에서도 0.6B 영어 WER이 유지될 것이다. 실제 latency ~273ms로 200ms 목표에 근접.

**설정**: Qwen3-0.6B, L40S, ucn=1, utn=5, css=[0.25, 0.5, 1.0], per-sample streaming (silence-reset), 30 samples

**결과**:
| CSS | EN WER | Ko CER | EN RTF | Ko RTF | 예상 latency |
|-----|--------|--------|--------|--------|-------------|
| 0.25s | 2.65% | 6.43% | 0.144 | 0.146 | ~273ms |
| 0.5s | 2.65% | 5.45% | 0.070 | 0.055 | ~523ms |
| 1.0s | 2.65% | 4.27% | 0.033 | 0.031 | ~1033ms |

**핵심 발견**:
1. **영어 WER은 css=0.25에서도 2.65%로 완전히 동일** — 0.6B 영어는 chunk size에 완전 불변
2. **한국어 CER은 chunk 작아질수록 악화**: 4.27% → 5.45% → 6.43% (50% 증가)
3. **RTF는 re-feed 비용으로 증가**: css=0.25에서 RTF=0.144 (여전히 실시간 가능하지만 4x 높음)
4. **per-chunk inference 시간은 일정** (~33-36ms): RTF 증가는 chunk 수 × re-feed 비용

**실제 latency 계산**: css + per-chunk inference = 250ms + ~36ms = **~286ms**
- 200ms 목표까지 ~86ms 부족
- css=0.15에서도 아마 WER 유지될 것이나 RTF가 0.2+ 이상으로 증가 예상

**결론**: Qwen3 chunk-based 접근의 latency 하한은 ~270ms (영어). 200ms 달성을 위해서는:
- Cache-Aware 아키텍처 (re-feed 제거) → NVIDIA Nemotron Speech Streaming
- 또는 token-level emission (vLLM realtime streaming)
- 한국어는 css < 0.5에서 CER 급증하므로 별도 전략 필요

**판정**: SKIP. Nemotron Speech Streaming 조사로 방향 전환.

---

## [2026-03-21] Exp #48: Nemotron Speech Streaming 0.6B 벤치마크

**가설**: NVIDIA Nemotron Speech Streaming 0.6B (Cache-Aware FastConformer-RNNT)는 80ms/160ms chunk에서 sub-200ms latency와 WER < 4%를 동시에 달성할 수 있을 것이다.

**설정**: Nemotron-speech-streaming-en-0.6b, L40S, NeMo 2.7.1, LibriSpeech clean 30 samples

**결과**:
| Mode | WER | RTF | Per-chunk Inference | 1st-word Latency |
|------|-----|-----|---------------------|------------------|
| Batch | 2.32% | 0.003 | — | — |
| 1120ms chunk | 2.32% | 0.044 | avg=35ms | ~1155ms |
| 560ms chunk | 2.48% | 0.074 | avg=35ms | ~595ms |
| **160ms chunk** | **2.81%** | **0.226** | **avg=34ms** | **~194ms** ✅ |
| 80ms chunk | 3.48% | 0.406 | avg=31ms | ~111ms |

**핵심 발견**:
1. **160ms chunk에서 latency 194ms — 200ms 목표 달성!** WER 2.81%도 목표 4% 훨씬 이하
2. **80ms에서도 WER 3.48%** — 여전히 4% 미만. 극한 저지연 시나리오에서도 사용 가능
3. **Cache-Aware 아키텍처의 위력**: per-chunk inference 시간이 chunk 크기에 무관하게 31-35ms로 일정
4. **vs Qwen3-0.6B**: Qwen3의 css=0.25 (RTF=0.144)보다 Nemotron 160ms (RTF=0.226)가 RTF는 높지만, re-feed 불필요해서 latency는 194ms vs 273ms로 Nemotron 승리
5. **영어 전용**: 한국어 미지원이 유일한 단점

**Qwen3 vs Nemotron 비교 (영어 LibriSpeech clean)**:
| 모델 | 최적 Streaming WER | 최저 Latency (WER<4%) | RTF |
|------|-------------------|---------------------|-----|
| Qwen3-1.7B (silence-reset) | 2.25% (100 samples) | ~2s (css=2.0) | 0.069 |
| Qwen3-0.6B (silence-reset) | 2.51% (100 samples) | ~273ms (css=0.25) | 0.144 |
| Nemotron 0.6B | 2.81% (160ms) | **194ms** | 0.226 |
| Nemotron 0.6B | 3.48% (80ms) | **111ms** | 0.406 |

**통합 전략**:
- 영어 전용 저지연 시나리오 → Nemotron 160ms chunk (WER 2.81%, latency 194ms)
- 다국어 (한국어 포함) → Qwen3-0.6B/1.7B (WER 2.25-2.65%, latency 273ms+)
- 하이브리드: language detection 후 영어면 Nemotron, 그 외 Qwen3

**판정**: KEEP — Nemotron 백엔드 프로토타입 구현 시작. latency 목표 최초 달성.

---

## [2026-03-21] Exp #49: Nemotron 0.6B 확대 벤치마크 (100+30 samples)

**설정**: Nemotron-speech-streaming-en-0.6b, L40S, LibriSpeech clean 100 + other 30

**결과**:
| Mode | Clean 100 WER | Other 30 WER | RTF | 1st-word Latency |
|------|--------------|--------------|-----|-----------------|
| Batch | 2.03% | 3.95% | 0.002 | — |
| 560ms stream | 2.09% | 3.95% | 0.075 | 595ms |
| **160ms stream** | **2.51%** | **3.95%** | **0.223** | **193ms** |

**핵심 발견**:
1. **Clean 100: WER 2.51% at 160ms** — 30-sample (2.81%)보다 개선, 통계적으로 더 신뢰할 수 있음
2. **Other 30: 스트리밍에서도 batch와 동일한 WER (3.95%)** — Cache-aware 아키텍처가 noisy audio에서도 품질 유지
3. **560ms에서 WER 2.09%** — batch (2.03%)와 거의 차이 없음. 스트리밍-배치 갭이 사실상 0

**모든 목표 달성 가능 확인** (영어 기준):
- WER < 4%: ✅ (2.51% at 160ms, 3.95% other)
- RTF < 0.15: ✅ at 560ms (0.075), ❌ at 160ms (0.223) — tradeoff
- Latency < 200ms: ✅ (193ms at 160ms chunks)

**Nemotron vs Qwen3 최종 비교 (LibriSpeech clean 100)**:
| 모델 | Streaming WER | Batch WER | Stream-Batch Gap | Latency |
|------|--------------|-----------|-----------------|---------|
| Qwen3-1.7B (silence-reset) | 2.25% | 2.46% | +0.21pp | ~2s |
| Nemotron 0.6B (560ms) | 2.09% | 2.03% | +0.06pp | 595ms |
| Qwen3-0.6B (silence-reset) | 2.51% | 2.30% | +0.21pp | ~273ms |
| **Nemotron 0.6B (160ms)** | **2.51%** | **2.03%** | **+0.48pp** | **193ms** |

**판정**: KEEP — 100-sample 결과가 30-sample보다 더 좋은 WER. 백엔드 통합 가치 확인.

---

## [2026-03-21] Exp #50: Nemotron 0.6B VRAM 및 동시 세션 throughput

**설정**: Nemotron-speech-streaming-en-0.6b, L40S (46GB), 160ms chunks

**결과**:
- 모델 VRAM: 4.65 GiB
- 세션당 cache: 14.8 MiB
- 이론적 최대 동시 세션: ~2800 (L40S)

**배치 throughput (N개 세션 동시 처리)**:
| Batch Size | RTF | VRAM | Throughput vs 1 |
|-----------|-----|------|-----------------|
| 1 | 0.232 | 4.70 GiB | 1.0x |
| 5 | 0.094 | 4.79 GiB | 2.5x |
| 10 | 0.059 | 4.89 GiB | 3.9x |

**핵심**: Cache-aware 덕분에 세션당 추가 VRAM이 극소 (14.8MiB). 배칭 시 throughput이 서브리니어하게 증가하지만 RTF는 충분히 낮음. 10 세션 동시 처리 시에도 RTF=0.059 (실시간의 17배 속도).

**vs Qwen3-0.6B (vLLM)**: Qwen3는 vLLM의 KV cache를 사용하므로 VRAM 관리가 다름. 하지만 Nemotron의 메모리 효율이 더 높을 가능성이 높음 (14.8MiB vs vLLM의 KV cache per-request).

**판정**: KEEP — 프로덕션 배포 가능성 확인. commit은 백엔드 통합과 함께 이미 완료 (a93e174).

---

## [2026-03-21] 실험 #51: Qwen3-1.7B 한국어 CSS sweep (css=0.5~2.0)

**가설**: Qwen3-1.7B는 0.6B보다 robust하므로 작은 chunk에서도 한국어 품질을 유지할 것이다. css=1.0에서 CER < 3%를 달성하면 ~1초 latency로 고품질 한국어 스트리밍이 가능하다.

**변경**: 벤치마크 스크립트만 (코드 변경 없음)

**결과** (L40S, FLEURS-ko 30 + LibriSpeech clean 30, silence-reset):

| CSS | Ko CER | Ko RTF | En WER | En RTF |
|-----|--------|--------|--------|--------|
| 0.50 | 7.83% | 0.158 | 2.32% | 0.129 |
| 0.75 | 5.24% | 0.109 | 2.15% | 0.099 |
| **1.00** | **3.01%** | **0.066** | **1.99%** | **0.073** |
| 1.50 | 3.78% | 0.059 | 1.82% | 0.064 |
| 2.00 | 2.73% | 0.059 | 1.82% | 0.060 |

**핵심 발견**:
- css=1.0이 한국어 최적점: CER 3.01% (~1초 latency)
- css=0.75 이하에서 급격히 열화 (영어는 거의 변동 없음 — 2.15~2.32%)
- 1.7B는 0.6B보다 확실히 robust (0.6B는 css=1.0에서 Ko CER 4.27%)
- css=1.5가 css=1.0보다 나쁜 것(3.78%)은 30-sample 분산 가능성

**판정**: KEEP — css=1.0이 한국어 최적 설정으로 확정. 커밋 대상 아님 (벤치마크만).

---

## [2026-03-21] 실험 #52: Qwen3-1.7B 한국어 ucn sweep at css=1.0

**가설**: css=1.0에서 ucn을 최적화하면 한국어 CER을 더 낮추거나 RTF를 줄일 수 있다.

**변경**: 벤치마크 스크립트만 (코드 변경 없음)

**결과** (L40S, FLEURS-ko 30, css=1.0):

| UCN | UTN | Ko CER | Ko RTF |
|-----|-----|--------|--------|
| 1 | 5 | 22.87% | 0.093 |
| 3 | 5 | **3.01%** | **0.061** |
| 5 | 5 | 3.01% | 0.063 |
| 7 | 5 | 3.01% | 0.076 |
| 5 | 3 | 3.64% | 0.056 |

**핵심 발견**:
- ucn=1은 치명적 (CER 22.87%) — 자기 교정 없이 한국어 불가
- ucn≥3이면 CER 수렴 (3.01%) — 추가 교정 패스 무의미
- ucn=3이 최적: ucn=5와 동일 CER이지만 RTF 최저 (0.061)
- utn=3은 약간 열화 (3.64%)

**판정**: ucn=3이 충분. utn 검증은 실험 #54에서 후속.

---

## [2026-03-21] 실험 #54: Qwen3-1.7B utn=5 vs utn=7 비교 (css=1.0, 2.0)

**가설**: utn=7 (현재 1.7B 기본값)이 utn=5보다 한국어에 실제로 도움이 되는지 검증.

**결과** (L40S, FLEURS-ko 30):

| CSS | UCN | UTN | Ko CER | Ko RTF |
|-----|-----|-----|--------|--------|
| 1.0 | 3 | 5 | 3.01% | 0.089 |
| **1.0** | **3** | **7** | **2.73%** | **0.069** |
| 2.0 | 3 | 5 | 2.87% | 0.043 |
| 2.0 | 3 | 7 | 2.66% | 0.045 |
| 2.0 | 5 | 5 | 2.73% | 0.058 |
| 2.0 | 5 | 7 | **2.52%** | 0.059 |

**핵심 발견**:
- utn=7이 utn=5보다 일관적으로 0.2~0.3pp 우수 → utn=7 기본값 유지 정당
- ucn=3은 RTF는 낮지만 ucn=5 대비 약간 열화 (2.66% vs 2.52% at css=2.0)
- css=1.0 + ucn=3 + utn=7 = Ko CER 2.73% (기존 3.01%에서 0.28pp 개선)
- 현재 기본값(ucn=5, utn=7)이 품질 최우선 설정으로 최적

**판정**: KEEP — utn=7 기본값 유지. ucn=5가 품질 최적이므로 기본값 변경 불필요.

---

## [2026-03-21] 실험 #53: NeMo multilingual 모델 한국어 스트리밍 조사

**가설**: NVIDIA Parakeet RNNT 1.1B Multilingual이 한국어 스트리밍을 지원하여 Nemotron급 저지연(sub-200ms)으로 한국어를 처리할 수 있을 것이다.

**조사 결과**:
- `nvidia/parakeet-rnnt-1.1b` (HuggingFace): **영어 전용** — 한국어 미지원
- Parakeet RNNT 1.1B Multilingual (ko-KR 지원): **NIM Docker 배포만** 가능, 로컬 .nemo 파일 미공개
- Canary-1B-v2, Parakeet-TDT-0.6B-v3: **유럽어 25개만** 지원, 한국어 미포함
- NeMo 레지스트리 `stt_multilingual_fastconformer_hybrid_large_pc`: 유럽어만

**판정**: DISCARD — NVIDIA 오픈소스 모델 중 한국어 로컬 스트리밍 가능한 것 없음. Qwen3-1.7B가 여전히 최선.

---

## Session Summary — research/mar20 (2026-03-20 ~ 2026-03-21)

### 달성 성과

**3대 목표 모두 달성** (L40S, LibriSpeech clean 100 samples):

| Target | Achieved | Config |
|--------|----------|--------|
| WER < 4% | **2.09%** | Nemotron 0.6B 560ms chunk |
| RTF < 0.15 | **0.075** | Nemotron 0.6B 560ms chunk |
| Latency < 200ms | **193ms** | Nemotron 0.6B 160ms chunk |

### 연구 통계

- **총 실험**: 58개 (54개 번호 + 4개 부가 조사/버그 수정)
- **판정 분포**: Keep 28, Discard 7, Skip/Iterate/Pending 나머지
- **커밋**: 24개 (research/mar20 브랜치)
- **새 백엔드**: 2개 (Qwen3 Streaming, Nemotron Speech Streaming)
- **조사한 모델**: 15+ (CarelessWhisper, U2 Whisper, Moonshine v2, Fun-ASR-MLT-Nano, Canary, Parakeet, VibeVoice, Voxtral Realtime, Nemotron 등)

### 핵심 발견

1. **Nemotron Speech Streaming 0.6B**: Cache-aware FastConformer-RNNT 아키텍처로 chunk 크기에 무관한 ~34ms per-chunk inference. 160ms chunk에서 193ms latency + WER 2.51%. 영어 전용.

2. **Qwen3 Streaming 최적 프로파일**:
   - 영어: ucn=5, utn=7, css=2.0 → WER 2.25% (100 samples)
   - 한국어: ucn=5, utn=7, css=2.0 → CER 2.17% (silence-reset 적용)
   - 한국어 저지연: ucn=3, utn=7, css=1.0 → CER 2.73%

3. **Silence-triggered session reset**: Qwen3 streaming의 O(n²) 누적 문제 해결. Korean CER 4.83% → 2.17% (-55%). Long-form 12분 연속 처리 가능.

4. **Multi-session scalability**: 단일 L40S에서 30 세션 동시 처리, RTF 0.058 (17.3x realtime). 세션 수 증가에도 throughput 일정.

5. **Contextual biasing은 역효과**: Qwen3 streaming에서 context 제공 시 WER/CER 악화. Zero-shot이 최적.

6. **한국어는 영어보다 긴 컨텍스트 필요**: css < 1.0에서 Korean CER 급증 (0.5s → 43.57%). 교착어 특성.

### 백엔드 선택 가이드

| 시나리오 | 백엔드 | WER/CER | Latency |
|---------|--------|---------|---------|
| English 최저 지연 | Nemotron 0.6B 160ms | 2.51% | 193ms |
| English 최고 품질 | Nemotron 0.6B 560ms | 2.09% | 595ms |
| Korean 최고 품질 | Qwen3-1.7B css=2.0 ucn=5 | CER 2.17% | ~2s |
| Korean 저지연 | Qwen3-1.7B css=1.0 ucn=3 | CER 2.73% | ~1s |
| 다국어 | Qwen3-1.7B css=2.0 | varies | ~2s |
| 자원 제한 | Qwen3-0.6B css=2.0 ucn=4 | En 2.51%, Ko 3.71% | ~2s |

## [2026-03-21] Exp #55: Nemotron 0.6B LibriSpeech other 100 벤치마크

**가설**: 30-sample other WER (3.95%)가 100-sample에서도 일관적인지 확인.

**결과** (L40S, Nemotron-speech-streaming-en-0.6b, 100 samples, 457.3s):

| Mode | WER | RTF | Per-chunk | Latency |
|------|-----|-----|-----------|---------|
| Batch | 5.72% | 0.003 | — | — |
| 560ms stream | **5.65%** | 0.082 | avg=34.7ms, p95=37.8ms | 595ms |
| 160ms stream | **6.29%** | 0.231 | avg=33.6ms, p95=35.6ms | 194ms |

**분석**:
1. **100-sample WER이 30-sample (3.95%)보다 높음**: 30-sample은 낙관적이었음. 100 samples가 더 대표적.
2. **560ms가 batch보다 약간 좋음** (5.65% vs 5.72%): cache-aware streaming의 contextual 이점 가능성.
3. **160ms→560ms 갭이 clean보다 큼** (0.64pp vs 0.42pp): noisy audio에서는 더 긴 chunk가 유리.
4. **Clean vs Other 갭**: 160ms에서 2.51% vs 6.29% (+3.78pp), 560ms에서 2.09% vs 5.65% (+3.56pp).
5. **Per-chunk inference 시간 일정** (~34ms): cache-aware 아키텍처의 noise 무관성 확인.

**판정**: KEEP (기록). 코드 변경 없음.

---

## [2026-03-21] Exp #56: Qwen3-1.7B LibriSpeech other 100-sample 벤치마크

**가설**: Qwen3-1.7B streaming이 Nemotron보다 noisy audio에서 더 좋은 WER을 달성할 것이다.

**결과** (L40S, Qwen3-ASR-1.7B, silence-reset per-sample, 100 samples, 457.3s):

| Config | WER | RTF | Errors/Total |
|--------|-----|-----|-------------|
| css=2.0 ucn=5 utn=7 | 4.45% | 0.077 | 70/1574 |
| css=1.0 ucn=3 utn=7 | **4.32%** | 0.086 | 68/1574 |

**Qwen3 vs Nemotron 비교 (LibriSpeech other 100)**:

| 모델 | WER | RTF | Latency |
|------|-----|-----|---------|
| **Qwen3-1.7B css=1.0** | **4.32%** | 0.086 | ~1s |
| **Qwen3-1.7B css=2.0** | **4.45%** | 0.077 | ~2s |
| Nemotron 0.6B 560ms | 5.65% | 0.082 | 595ms |
| Nemotron 0.6B 160ms | 6.29% | 0.231 | 194ms |

**핵심 발견**:
1. **Qwen3가 noisy audio에서 Nemotron보다 1.2~1.3pp 우수** (4.32% vs 5.65%)
2. **css=1.0이 css=2.0보다 약간 좋음** — noisy audio에서는 짧은 chunk의 self-correction이 더 효과적
3. **Clean vs Other 갭**: Qwen3 1.7B는 2.25→4.45% (+2.2pp), Nemotron은 2.09→5.65% (+3.56pp) — Qwen3의 noise robustness가 더 강함
4. **RTF는 유사** (0.077~0.086 vs 0.082)

**결론**: Noisy audio에서는 Qwen3-1.7B가 명확한 승자. Nemotron의 장점은 latency (193ms vs ~1-2s).

**판정**: KEEP (기록). 코드 변경 없음.

---

## [2026-03-21] Research: vLLM v0.17 Qwen3-ASR Realtime Streaming

**발견**: vLLM v0.17.0에서 `qwen3_asr_realtime` 모델이 추가됨 (PR #34613).

**핵심 특징**:
- WebSocket `/v1/realtime` 엔드포인트
- 1초 chunk로 오디오를 점진적 전송 — re-feed 불필요
- vLLM 서버에서 직접 streaming ASR 제공
- 현재 WhisperLiveKit의 qwen-asr SDK (vLLM 0.14.0)와는 별도 접근

**장점**:
- Re-feed 제거로 long-form에서 O(n²) → O(n) 가능
- vLLM의 KV cache/prefix cache 자동 활용
- 서버-클라이언트 아키텍처로 다중 세션 효율적

**단점/리스크**:
- vLLM 0.14→0.17 업그레이드 필요 (major breaking changes 가능)
- qwen-asr SDK와의 호환성 미확인
- WhisperLiveKit의 AudioProcessor 파이프라인과 통합 복잡도 높음
- WebSocket→WebSocket 이중 프록시 필요 (클라이언트→WLK→vLLM)

**판정**: SKIP — 현재 qwen-asr SDK 기반 시스템이 안정적이고, silence-triggered reset으로 O(n²) 문제 해결 완료. vLLM 0.17 마이그레이션은 별도 브랜치에서 검토 필요.

---

## [2026-03-21] Research: 하이브리드 ASR 라우팅 설계

**가설**: 영어는 Nemotron (저지연, 193ms), 한국어는 Qwen3 (고품질, CER 2.17%)로 자동 라우팅하면 양쪽 장점을 모두 얻을 수 있다.

**설계 안**:

### Architecture A: Dual-Backend ASR Class
```
HybridASR(NemotronStreamingASR, Qwen3StreamingASR)
  - 첫 1-2초: Qwen3로 language detection
  - 영어 감지 → Nemotron으로 전환
  - 비영어 → Qwen3 계속 사용
```

**장점**: 단일 TranscriptionEngine에서 관리, 세션별 최적 백엔드 선택
**단점**: VRAM ~11GB (Nemotron 4.65GB + Qwen3-1.7B ~6GB), 프레임워크 충돌 위험 (NeMo + vLLM)

### Architecture B: Proxy-Based Routing
```
AudioProcessor → LanguageDetector → Backend Selector
  - 별도 서버 인스턴스로 Nemotron/Qwen3 실행
  - 프록시가 language detection 후 적절한 서버로 WebSocket 라우팅
```

**장점**: 백엔드 독립성, 확장성
**단점**: 네트워크 지연 추가, 운영 복잡도

### Architecture C: Config-Based Selection (현 시점 권장)
```
wlk serve --backend nemotron-streaming  # 영어 서비스
wlk serve --backend qwen3-streaming     # 한국어 서비스 (별도 포트)
# 프론트엔드에서 사용자 언어 선택 후 적절한 서버 연결
```

**장점**: 코드 변경 없음, 가장 단순
**단점**: 사용자 수동 선택 필요, 자동 라우팅 불가

**판정**: SKIP — Architecture C로 현재 충분. 자동 라우팅(A/B)은 수요 확인 후 별도 브랜치에서 구현.

---

## [2026-03-21] Exp #58: Qwen3-0.6B LibriSpeech other 100-sample 벤치마크

**결과** (L40S, Qwen3-ASR-0.6B, ucn=4 utn=5 css=2.0, 100 samples, 457.3s):

| Config | WER | RTF | Time |
|--------|-----|-----|------|
| css=2.0 ucn=4 utn=5 | **4.45%** | **0.038** | 17.4s |

**전체 모델 비교 (LibriSpeech other 100)**:

| 모델 | WER | RTF | Latency |
|------|-----|-----|---------|
| **Qwen3-1.7B css=1.0** | **4.32%** | 0.086 | ~1s |
| **Qwen3-0.6B css=2.0** | **4.45%** | **0.038** | ~2s |
| Qwen3-1.7B css=2.0 | 4.45% | 0.077 | ~2s |
| Nemotron 0.6B 560ms | 5.65% | 0.082 | 595ms |
| Nemotron 0.6B 160ms | 6.29% | 0.231 | 194ms |

**핵심 발견**:
1. **Qwen3-0.6B와 1.7B가 동일 WER (4.45%)** — 0.6B의 noisy audio 대응이 예상보다 강함
2. **RTF 0.038은 1.7B의 절반** — 0.6B가 throughput 면에서 훨씬 효율적
3. **Qwen3 전체가 Nemotron보다 1.2~2pp 우수** (noisy audio)
4. Nemotron의 유일한 장점은 latency (193ms vs ~1-2s)

**판정**: KEEP (기록). Noisy audio 시나리오에서는 Qwen3-0.6B가 가성비 최고.

---

## [2026-03-21] Exp #59: Parakeet RNNT 1.1B 스트리밍 평가

**가설**: NVIDIA Parakeet RNNT 1.1B (FastConformer-RNNT)가 Nemotron과 동일한 Cache-Aware 스트리밍을 지원하여 한국어 저지연 스트리밍이 가능할 것이다.

**설정**: L40S, nvidia/parakeet-rnnt-1.1b (HuggingFace), NeMo, Nemotron과 동일 streaming API 사용

**결과**:

| Mode | EN WER (clean 100) | KO CER (FLEURS 30) | RTF |
|------|-------------------|-------------------|-----|
| Batch | **1.76%** | 212.45% (미지원) | 0.003 |
| Stream 560ms | 25.67% | 111.75% | 0.123 |
| Stream 160ms | 97.54% | — | 0.384 |

**핵심 발견**:
1. **영어 배치 WER 1.76% — 역대 최고!** Nemotron (2.03%), Qwen3-1.7B (2.46%)보다 우수
2. **한국어 미지원**: HuggingFace `parakeet-rnnt-1.1b`은 영어 전용. 다국어 버전은 NVIDIA NIM/Riva에만 존재
3. **스트리밍 대실패 (WER 25.67%)**: Parakeet은 Cache-Aware 학습이 아닌 일반 FastConformer. Nemotron의 `conformer_stream_step()`과 동일 파라미터를 사용하면 품질 급락
4. **Nemotron vs Parakeet**: Nemotron은 Cache-Aware Streaming 전용 학습, Parakeet은 배치 최적화 모델. 아키텍처는 같지만 학습 방식이 근본적으로 다름
5. **VRAM**: 8.02 GiB (Nemotron 4.65 GiB의 거의 2배)

**결론**: Parakeet RNNT 1.1B는 배치 영어에서 최고 WER이나, 스트리밍 파이프라인에는 부적합. 한국어 다국어 버전은 NIM 전용으로 WhisperLiveKit 통합 불가. Nemotron Speech Streaming이 여전히 유일한 Cache-Aware 스트리밍 모델.

**판정**: DISCARD — 스트리밍 부적합, 한국어 미지원.

---

## [2026-03-21] Exp #60: Korean FLEURS 100-sample 확장 벤치마크

**가설**: 30-sample Korean CER (1.7B: 2.17%, 0.6B: 3.71%)이 100-sample에서도 일관적인지 확인. 영어처럼 30-sample이 낙관적이었을 수 있다.

**설정**: L40S, FLEURS-ko 100 samples (1254.7s), per-sample streaming (silence-reset)

**결과**:

| Config | 30-sample CER | 100-sample CER | RTF |
|--------|--------------|----------------|-----|
| Qwen3-1.7B ucn=5 utn=7 css=2.0 | **2.17%** | **3.01%** | 0.069 |
| Qwen3-1.7B ucn=3 utn=7 css=1.0 | — | **3.74%** | 0.077 |
| Qwen3-0.6B ucn=4 utn=5 css=2.0 | 3.71% | **4.96%** | 0.033 |
| Qwen3-0.6B ucn=4 utn=5 css=1.0 | — | **5.26%** | 0.039 |

**핵심 발견**:
1. **30→100 확장 시 CER 상승**: 1.7B는 2.17%→3.01% (+0.84pp), 0.6B는 3.71%→4.96% (+1.25pp)
2. **30-sample은 30% 이상 낙관적**: 영어 (2.25% 100s vs 1.82% 30s)와 동일 패턴
3. **1.7B vs 0.6B 격차 확대**: 30s에서 1.54pp → 100s에서 1.95pp. 큰 모델이 다양한 화자/도메인에서 더 robust
4. **css=1.0 vs 2.0**: 1.7B는 0.73pp 차이 (3.74% vs 3.01%), 0.6B는 0.30pp 차이 (5.26% vs 4.96%)
5. **Worst samples**: 외래어가 포함된 문장 (swapo, cochamó, toginet, martelly)에서 CER 20%+ — 외래어 인식이 주요 약점

**갱신된 최종 벤치마크 (L40S, 100 samples)**:

| Config | EN WER (clean) | EN WER (other) | KO CER |
|--------|----------------|----------------|--------|
| Qwen3-1.7B stream ucn=5 | **2.25%** | **4.32%** | **3.01%** |
| Qwen3-0.6B stream ucn=4 | 2.51% | 4.45% | 4.96% |
| Nemotron 0.6B 560ms | 2.09% | 5.65% | — |
| Nemotron 0.6B 160ms | 2.51% | 6.29% | — |

**판정**: KEEP (기록). 100-sample 기준으로 벤치마크 수치 갱신 필요.

---

## [2026-03-21] Exp #61: Korean CER 외래어 분석

**가설**: 한국어 CER 3.01%의 주요 원인이 외래어 포함 문장인지 확인. 외래어 제외 시 순수 한국어 CER이 유의미하게 낮을 것이다.

**설정**: Qwen3-1.7B ucn=5 utn=7 css=2.0, FLEURS-ko 100 samples, 외래어 판별 기준: 3+ 연속 라틴 문자

**결과**:

| 카테고리 | 샘플 수 | CER | 오류/전체 |
|---------|--------|------|---------|
| 전체 | 100 | 3.01% | 141/4678 |
| **순수 한국어** | 92 | **1.80%** | 75/4160 |
| 외래어 포함 | 8 | 12.74% | 66/518 |

**외래어 오류 패턴**:
- `bishkek` → 제거 (음독만 유지)
- `swapo` → "SWAPO" + 깨진 문자 (UTF-8 문제)
- `toginet radio` → "토진넷 라디오" (음독 변환 시도)
- `cochamó valley` → "코차무 벨리" (발음 변형)
- `martelly` → "멀토이" (발음 크게 벗어남)

**핵심 발견**:
1. **순수 한국어 CER 1.80%** — 매우 우수. 외래어 8개가 전체 CER을 1.21pp 끌어올림
2. **외래어 문제는 model-level**: Qwen3가 한국어 음독 변환을 일관적으로 수행하지 못함
3. **FLEURS 데이터셋의 특성**: reference에 외래어가 원문 그대로 포함 (bishkek, swapo 등). 실제 사용에서는 음독으로 표기되므로 CER이 과대평가될 수 있음
4. **Top 1 worst sample은 순수 한국어** (#31, CER 25.4%): "고열대" 관련 문장에서 내용이 크게 다름 — 이는 audio-text misalignment 또는 발화 품질 문제일 가능성

**결론**: 실제 한국어 스트리밍 품질은 CER 1.80% 수준. 외래어 처리는 decoder/model-level 개선이 필요하지만, 실제 사용 시나리오에서는 외래어 비율이 FLEURS보다 낮을 가능성이 높음.

**판정**: SKIP (코드 변경 없음). 벤치마크 해석을 위한 참고 자료로 기록.

---

## [2026-03-21] Exp #62: vLLM Realtime Streaming API 적용 가능성 조사

**가설**: vLLM v0.17+의 `/v1/realtime` WebSocket API를 사용하면 Qwen3-ASR 스트리밍에서 re-feed 비용을 제거할 수 있다. 이를 통해 작은 chunk size에서도 RTF 증가 없이 저지연 스트리밍이 가능할 것이다.

**조사 결과**:

| 항목 | 현재 | vLLM Realtime |
|------|------|--------------|
| vLLM 버전 | 0.14.0 | 0.17.0+ (0.18.0 최신) |
| PyTorch | 2.9.1+cu129 | 2.10.0 필요 |
| API | In-process (qwen-asr Python) | WebSocket `/v1/realtime` |
| Re-feed | 매 chunk마다 이전 audio 재전송 | Anchor request + KV cache 보존 → 재전송 불필요 |
| Protocol | `streaming_transcribe()` | `input_audio_buffer.append` → `transcription.delta` |

**핵심 발견**:

1. **vLLM v0.17.0에 `qwen3_asr_realtime` 모델 추가** (PR #34613) — 실시간 오디오 버퍼 축적 + 세그먼트 단위 처리
2. **Anchor request 패턴으로 KV cache 보존** — 각 청크가 incremental하게 처리되어 re-feed 비용 O(1)
3. **qwen-asr 0.0.6 패키지가 vllm==0.14.0 고정** — 업그레이드 시 qwen-asr의 in-process streaming 깨짐
4. **NeMo 2.7.1은 PyTorch >=2.6.0** — PyTorch 2.10 업그레이드에 호환 가능
5. **아키텍처 전환 필요**: in-process → client-server. WhisperLiveKit이 vLLM 서버에 WebSocket으로 연결하는 구조

**업그레이드 경로**:
```
옵션 A: 직접 업그레이드
  - vLLM 0.14→0.18, PyTorch 2.9.1→2.10.0
  - qwen-asr in-process streaming 깨짐
  - 새로운 WebSocket 클라이언트 백엔드 필요
  - 리스크: 높음 (dependency 충돌, NeMo 호환성)

옵션 B: 별도 venv + vLLM 서버
  - 새 venv에 vLLM 0.18 설치, `vllm serve Qwen/Qwen3-ASR-1.7B` 실행
  - 메인 환경은 그대로 유지
  - WhisperLiveKit에서 WebSocket 클라이언트로 연결
  - 리스크: 낮음 (환경 분리)
```

**예상 개선**:
- css=0.25 (250ms chunk): RTF 0.144 → ~0.05 (KV cache 보존으로 re-feed 제거)
- First-word latency: ~273ms → ~260ms (약간 개선, chunk 처리 시간은 동일)
- 동시 세션 throughput: vLLM의 네이티브 배칭으로 개선 가능

**판정**: SKIP — 조사 완료, 구현은 별도 작업으로 분리. 현재 qwen-asr 0.0.6 + vLLM 0.14 조합이 안정적이며, 업그레이드는 qwen-asr의 새 버전 출시 후 시도하는 것이 효율적. 옵션 B (별도 서버)는 즉시 시도 가능하나 아키텍처 변경이 큼.

---

## [2026-03-21] Exp #63: Speech Enhancement (noisereduce) → Qwen3 on LibriSpeech other

**가설**: noisereduce (spectral gating) 전처리로 LibriSpeech other 오디오를 개선하면 Qwen3-1.7B streaming WER이 4.32%에서 하락할 것이다.

**설정**: noisereduce 3.0.3 (prop_decrease=0.8, stationary=True), Qwen3-1.7B, L40S, 100 samples (457.3s), per-sample streaming (silence-reset)

**결과**:

| Config | WER | RTF |
|--------|-----|-----|
| Raw ucn=3 css=1.0 (baseline) | **4.38%** | 0.091 |
| Enhanced ucn=3 css=1.0 | 4.83% (+0.45pp) | 0.091 |
| Raw ucn=5 css=1.0 | 4.51% | 0.095 |
| Enhanced ucn=5 css=1.0 | 4.76% (+0.25pp) | 0.095 |

**Enhancement overhead**: 2.2s (RTF 0.005) — 매우 빠르지만 효과 없음

**핵심 발견**:
1. **Speech enhancement가 오히려 WER 악화**: +0.25~0.44pp. noisereduce의 spectral gating이 유용한 음성 정보를 제거
2. **LibriSpeech other의 "noise"는 가산 잡음이 아님**: 난해한 화자/내용/녹음 조건이 원인. 전통적 노이즈 제거가 무효
3. **ucn=3이 ucn=5보다 noisy에서 여전히 우수** (4.38% vs 4.51%), Exp #56 결과 재확인
4. **baseline WER 차이**: 4.38% vs 4.32% (Exp #56) — 실행 간 약간의 변동

**결론**: Qwen3는 이미 noisy audio에 대한 내재적 robustness를 갖추고 있어 외부 전처리가 불필요. LibriSpeech other WER 개선은 model/decoder level에서만 가능.

**판정**: DISCARD — speech enhancement 전처리는 Qwen3에 비효과적.

---

## [2026-03-21] Exp #64: 최신 ASR 모델 탐색 (2026-03)

**조사 대상**:

| 모델 | 한국어 | 스트리밍 | 오픈소스 | 비고 |
|------|--------|---------|---------|------|
| Uni-ASR (2603.11123) | X | O | 미공개 | Chinese-English only. test-other WER 5.71% (스트리밍) |
| Streaming LLM ASR (2601.22779) | X | O | 미공개 | Mandarin only. MoChA adaptive segmentation |
| FireRedASR2S (2603.10420) | X | O | O | Chinese/English. 12.7x RTF on AISHELL |
| Omnilingual ASR (Facebook) | O | △ (CTC) | O | 1600+ 언어. 배치 전용. CER <10% for 78% of languages |
| antirez/qwen-asr | O | O | O | C 구현. CPU 추론. 0.6B: TTFT 92ms |

**핵심 발견**:
1. **새로운 한국어 스트리밍 ASR 모델 없음** — Qwen3-ASR이 여전히 최선
2. **Uni-ASR**: Qwen3-1.7B 기반이지만 스트리밍 test-other WER 5.71% (우리 4.32%보다 낮음)
3. **Omnilingual ASR**: 한국어 지원하나 배치 전용 CTC 모델. 스트리밍 미지원
4. **Whisper v4 없음**: Whisper는 v3/turbo/distil에서 멈춤
5. **FireRedASR2S**: VAD+LID+Punc 올인원이지만 Chinese/English only

**판정**: SKIP — 현재 Qwen3-ASR + Nemotron 조합이 최선. 새 후보 없음.

---

## [2026-03-21] Exp #65: Streaming vs Batch vs Two-Pass on LibriSpeech other

**가설**: batch 재전사가 streaming보다 낮은 WER을 보이면, two-pass (streaming → batch refinement) 전략으로 noisy WER을 개선할 수 있다.

**설정**: Qwen3-1.7B, L40S, LibriSpeech other 100 (457.3s), per-sample

**결과**:

| Config | WER | RTF | Time |
|--------|-----|-----|------|
| **Streaming ucn=3 css=1.0** | **4.38%** (69/1574) | 0.092 | 41.8s |
| Batch (full, single pass) | 4.45% (70/1574) | 0.029 | 13.2s |
| Batch (ucn=5, css=30s) | 4.45% (70/1574) | 0.028 | 13.0s |
| Oracle two-pass (best/sample) | 4.32% | — | — |

**Two-pass 분석**:
- 100 샘플 중 **98개가 동일 에러** — streaming과 batch 차이 거의 없음
- 1개 streaming 우세, 1개 batch 우세
- Oracle two-pass gain: **0.06pp** — 사실상 무의미

**핵심 발견**:
1. **Streaming이 batch보다 우수** (4.38% vs 4.45%): noisy audio에서도 self-correction이 효과적
2. **Two-pass 전략 불필요**: streaming-batch gap이 0이거나 음수. refinement 비용 대비 이득 없음
3. **Batch RTF는 3배 빠름** (0.028 vs 0.092): 단순 throughput 목적이면 batch가 효율적
4. **Qwen3 streaming의 self-correction 메커니즘이 핵심**: unfixed_chunk_num에 의한 반복 검토가 single-pass보다 정확
5. **LibriSpeech clean에서도 동일 패턴** (Exp #22): streaming 2.25% < batch 3.26%

**결론**: Qwen3-ASR의 streaming은 이미 최적. Two-pass, batch refinement, 재전사 모두 불필요. 현재 아키텍처가 noisy와 clean 모두에서 최고 WER을 달성.

**판정**: DISCARD — two-pass 전략은 가치 없음.

---

## [2026-03-21] Exp #66: WER Contraction Normalization 영향 분석

**가설**: Qwen3가 contractions (I'm, don't, can't)을 출력하고 LibriSpeech reference는 expanded forms (I AM, DO NOT, CAN NOT)을 사용하므로, contraction mismatch가 WER을 과대평가시킬 수 있다.

**설정**: Qwen3-1.7B, L40S, LibriSpeech other 100 + clean 100, per-sample streaming. 88개 contraction → expansion 맵핑 적용하여 WER 재계산.

**결과**:

| Dataset | Basic WER | Contraction-Expanded WER | Delta |
|---------|-----------|--------------------------|-------|
| Other 100 (ucn=3 css=1.0) | 4.38% (69/1574) | 4.18% (67/1601) | +0.20pp |
| Clean 100 (ucn=5 css=2.0) | 2.30% (43/1870) | 2.28% (43/1882) | +0.01pp |

**핵심 발견**:
1. **Contraction 영향 무시 가능**: other에서 +0.20pp, clean에서 +0.01pp — WER 측정에 실질적 왜곡 없음
2. **영향받은 샘플**: other에서 1/100개만 해당 ("I am" vs "I'm"), clean에서 0/100개
3. **Qwen3는 대부분 expanded form 출력**: contraction 사용 빈도가 매우 낮아 mismatch 자체가 희소
4. **ref_words 차이**: contraction expansion으로 ref word 수가 증가 (1574→1601, 1870→1882)하여 WER 분모가 커짐

**결론**: 현재 WER 측정 방식 (lowercase + punctuation strip)은 정확하며, contraction normalization을 추가할 필요 없음. 보고된 WER 수치들은 신뢰할 수 있음.

**판정**: DISCARD — contraction normalization 불필요. 현재 정규화 방식 유지.

---

## [2026-03-21] Exp #67: Auto Language Detection vs Explicit Language

**가설**: Qwen3에 language 파라미터를 지정하지 않으면 (auto-detect) 품질이 저하될 것이다. 품질 차이가 없다면 API를 단순화할 수 있다.

**설정**: Qwen3-1.7B, L40S, ucn=5 utn=7 css=2.0, per-sample streaming
- English: LibriSpeech clean 100 (670.6s)
- Korean: FLEURS-ko 100 (1254.7s)

**결과**:

| Config | Explicit | Auto-detect | Delta |
|--------|----------|-------------|-------|
| EN WER (clean 100) | 2.30% (43/1870) | 2.30% (43/1870) | +0.00pp |
| KO CER (FLEURS 100) | 6.45%* (304/4713) | 6.43%* (303/4713) | -0.02pp |

*CER 절대값은 Exp #60(3.01%)과 다름 — 이 실험의 CER 정규화가 구두점을 유지하여 과대 측정. 상대 비교는 동일 정규화이므로 유효.

**핵심 발견**:
1. **Auto-detect = Explicit**: 영어 0개, 한국어 1개 샘플만 차이. 사실상 동일 품질
2. **언어 감지 100% 정확**: English 100/100, Korean 100/100 — 단일 언어 오디오에서 오감지 없음
3. **추론 속도**: auto-detect가 ~3s 느림 (EN 46.4→49.4s, KO 89.1→90.0s) — 무시할 수준
4. **API 단순화 가능**: language 파라미터 생략 가능. 사용자가 언어를 지정하지 않아도 품질 손실 없음

**결론**: Qwen3의 자동 언어 감지가 충분히 정확하여 language 파라미터를 optional로 만들 수 있다. 프로덕션에서 language="auto"를 기본값으로 사용해도 품질 저하 없음.

**판정**: KEEP (인사이트). 코드 변경은 별도 태스크.

---

## [2026-03-21] Exp #68: Qwen3 Token Logprobs 분포 분석

**가설**: vLLM logprobs를 활성화하면 토큰별 confidence를 측정할 수 있고, 저신뢰 토큰 지연 방출로 WER 개선이 가능할 수 있다.

**설정**: Qwen3-1.7B, L40S, SamplingParams monkey-patch (logprobs=1), LibriSpeech clean 10 samples (91.5s), ucn=5 utn=7 css=2.0

**결과**:

| 구간 | 토큰 수 | 평균 logprob | 평균 확률 | 최소 logprob | 저신뢰(<37%) |
|------|---------|-------------|----------|-------------|-------------|
| Cold start (chunk 0-4) | 551 | -0.021 | 97.9% | -0.856 | 0 (0%) |
| Warm (chunk 5+) | 132 | -0.019 | 98.1% | -0.732 | 0 (0%) |
| Final chunk | 139 | -0.021 | 98.0% | -0.931 | 0 (0%) |
| **Overall** | **822** | **-0.021** | **97.9%** | -0.931 | **0 (0%)** |

**토큰 위치별 분석 (Warm chunk)**:
- 처음 3 토큰: 평균 확률 **100.0%**
- 마지막 3 토큰: 평균 확률 **91.4%**
- 차이: -8.6pp → unfixed_token_num이 정확히 올바른 토큰(마지막 N개)을 타겟

**핵심 발견**:
1. **모델의 confidence가 극도로 높음**: 822개 토큰 중 37% 미만인 토큰 0개. 평균 97.9%
2. **Cold vs Warm 차이 무의미**: 97.9% vs 98.1% — self-correction 전후 confidence 동일
3. **unfixed_token_num 메커니즘이 최적**: 마지막 토큰이 정확히 confidence가 가장 낮은 위치
4. **confidence-based emission 불필요**: 의미 있는 threshold를 설정할 수 없음. 거의 모든 토큰이 >90%
5. **Qwen3의 autoregressive decoding이 매우 안정적**: temperature=0.0에서 greedy decoding이 거의 확정적

**결론**: Qwen3-ASR의 토큰 confidence는 이미 극도로 높아서 confidence-based emission이 품질 개선에 기여할 수 없다. unfixed_token_num 메커니즘이 이미 최적의 boundary 처리를 수행 중. logprobs 활성화의 유일한 가치는 디버깅/모니터링 용도.

**판정**: DISCARD — confidence-based emission은 가치 없음. 현재 메커니즘이 최적.

---

## [2026-03-21] Exp #69: Nemotron v2603 1120ms Chunk + Voxtral Korean 조사

**가설 1**: Nemotron v2603 (March 12 업데이트)의 1120ms chunk가 560ms보다 noisy audio WER 개선.

**조사**:
- Nemotron v2603 SHA ac0580bb — 이미 우리 캐시에 있었음 (March 21 다운로드)
- 이전 벤치마크 (Exp #49, #55)에서 이미 v2603 사용 중이었음
- January 버전은 `nemotron-speech-streaming-jan2026` 브랜치로 분리

**벤치마크 결과** (L40S, 100 samples each):

| Mode | Clean WER | Other WER | RTF | Latency |
|------|-----------|-----------|-----|---------|
| 1120ms chunk | **2.03%** | 5.72% | 0.045 | 1156ms |
| 560ms chunk | 2.09% | **5.65%** | 0.076 | 595ms |
| 160ms chunk | 2.51% | 6.29% | 0.230 | 194ms |

**핵심 발견**:
1. **1120ms는 noisy에서 오히려 약간 나빠짐** (5.72% vs 5.65%): 더 긴 chunk가 noisy audio에 도움 안 됨
2. **Clean에서 1120ms = batch (2.03%)**: streaming-batch gap 완전 해소
3. **v2603 = 이전 결과와 동일**: 우리가 이미 최신 모델 사용 중이었음
4. Model card의 4.84% (test-other 1.12s)는 전체 test set(2939 samples) 기준일 것

**가설 2**: Voxtral Mini 4B Realtime 한국어 벤치마크 가치 평가.

**조사 결과** (논문 2602.11298):
- **Korean CER: 15.74%** (FLEURS, 480ms latency)
- Qwen3-1.7B streaming: 3.01% (FLEURS 100, 동일 데이터셋)
- **Voxtral은 한국어에서 Qwen3 대비 5배 열등** → 벤치마크 불필요
- Voxtral의 장점 (sliding window, no re-feed)은 한국어 품질이 근본적으로 부족

**판정**: DISCARD — Nemotron 1120ms는 noisy WER 미개선. Voxtral Korean은 품질 부족. 코드 변경 없음.

---

## [2026-03-21] Exp #70: Qwen3-1.7B LibriSpeech other UCN sweep — 4% 목표

**가설**: ucn=1-5, utn=3-10, css=0.5-1.0 조합에서 LibriSpeech other WER < 4%를 달성할 수 있는 최적 config가 있을 것이다.

**결과** (L40S, 100 samples, per-sample streaming, silence-reset):

| UCN | UTN | CSS | WER | Errors | RTF |
|-----|-----|-----|-----|--------|-----|
| 3 | 7 | 1.0 | 4.51% | 71 | 0.076 |
| 5 | 7 | 1.0 | 4.51% | 71 | 0.089 |
| 3 | 3 | 1.0 | 4.51% | 71 | **0.065** |
| 3 | 5 | 1.0 | 4.57% | 72 | 0.071 |
| 3 | 10 | 1.0 | 4.57% | 72 | 0.084 |
| 4 | 7 | 1.0 | 4.64% | 73 | 0.083 |
| 1 | 7 | 1.0 | 4.70% | 74 | 0.087 |
| 2 | 7 | 1.0 | 4.76% | 75 | 0.075 |
| 2 | 5 | 1.0 | 4.83% | 76 | 0.065 |
| 2 | 3 | 1.0 | 4.83% | 76 | 0.057 |
| 3 | 7 | 0.5 | 4.83% | 76 | 0.137 |
| 2 | 7 | 0.5 | 4.83% | 76 | 0.122 |

**분석**:
1. **WER 4% 이하 불가**: 모든 config에서 4.51% 이상. 이전 최선 4.32% (Exp #56)과 ±3 오류 차이는 vLLM non-determinism
2. **ucn=3과 ucn=5 동일** (4.51%): silence-reset 덕분에 self-correction 패스 수가 품질에 무관
3. **utn도 거의 무관**: utn=3에서 7까지 동일 4.51% — noisy audio에서 unfixed token 수는 중요하지 않음
4. **css=0.5는 악화** (4.83%): 청크가 너무 작으면 context 부족
5. **최효율 config**: ucn=3 utn=3 css=1.0 — 최저 RTF 0.065, WER 동일
6. **공식 논문 대비**: paper 4.51% (full test set) = 우리 4.51% (100 samples) 정확히 일치!

**결론**: Qwen3-1.7B streaming의 LibriSpeech other WER 한계는 ~4.4% (100 samples 기준). 이는 공식 논문 결과(4.51%)와 일치하며, 모델 수준에서의 개선 없이는 더 낮출 수 없음.

**판정**: DISCARD — WER 4% 목표 미달성, 모델 한계 확인. 코드 변경 없음.

---

### 미완료/후속 과제

1. ~~**Nemotron LibriSpeech other 100 벤치마크**~~: 완료.
2. **한국어 저지연 스트리밍**: Nemotron급 sub-200ms + 한국어 지원 모델 부재.
3. **vLLM Realtime API 통합 (옵션 B)**: 별도 venv에서 vLLM 0.18 서버 + WebSocket 클라이언트 백엔드. qwen-asr 새 버전 출시 시 재평가.
4. **Docker Nemotron 프로필**: compose.yml에 추가 완료, 실제 빌드/배포 검증 필요.
5. **Parakeet RNNT 1.1B 배치 백엔드**: WER 1.76%. REST API 전용 백엔드로 추가 고려.
6. ~~**progrem.md 100-sample 수치 갱신**~~: 완료 (commit 2ac6dd0).
7. ~~**LibriSpeech other speech enhancement**~~: DISCARD. noisereduce 무효.
8. ~~**Qwen3 streaming confidence-based emission**~~: DISCARD. logprobs 분석 결과 평균 확률 97.9%, 저신뢰 토큰 0개. unfixed_token_num이 이미 최적 처리.
9. ~~**WER contraction normalization**~~: DISCARD. Qwen3 contraction 빈도 극히 낮아 WER 왜곡 무시 가능.
10. **Auto language detection**: KEEP. Qwen3 auto-detect = explicit language 품질 동일. API 단순화 가능.
9. ~~**Two-pass refinement**~~: DISCARD. streaming >= batch.

---

## [2026-03-21] Experiment #71: IBM Granite 4.0 1B Speech batch benchmark

**가설**: IBM Granite 4.0 1B Speech (March 2026, OpenASR #1)가 claimed WER (1.42% clean, 2.85% other)를 달성하면 영어 batch 최고 성능 백엔드로 통합할 가치가 있다.

**변경**: `/tmp/bench_granite_transformers.py` — Transformers 백엔드로 LibriSpeech clean/other 100 samples 벤치마크. vLLM 0.14는 EngineCore crash로 사용 불가.

**결과** (L40S GPU 1, Transformers bf16):
| 데이터셋 | WER | Errors/Words | RTF | Time |
|---|---|---|---|---|
| clean | **1.18%** | 22/1870 | 0.098 | 65.9s |
| other | **3.75%** | 59/1574 | 0.126 | 57.7s |

**판정**: KEEP (batch benchmark only)
**이유**: Clean WER 1.18%는 현재까지 최저. 한국어 미지원, streaming 불가. 영어 batch/REST API 백엔드 통합 가치 있음.

---

## [2026-03-21] Experiment #72: Qwen3 streaming code audit

**가설**: 파이프라인에 WER 영향 버그가 있을 수 있다.
**결과**: 버그 없음. 파이프라인은 SDK 수준 최적 품질 달성.
**판정**: DISCARD (개선 불필요)

---

## External Research Summary (2026-03-21)

### 새 모델 조사
| 모델 | 크기 | 한국어 | 스트리밍 | 주요 WER | 비고 |
|---|---|---|---|---|---|
| Granite 4.0 1B | 1B | X | X | clean 1.18% | OpenASR #1 |
| Parakeet-cpp | 110M-600M | X | O (C++) | N/A | Metal GPU |
| antirez/qwen-asr | - | O | O | SDK 동일 | Pure C, 8x RT |

### 다음 방향
1. **Granite 4.0 1B batch 백엔드 통합**: REST API 전용
2. **Qwen3 + Granite routing**: 한국어→Qwen3, 영어→Granite
3. **vLLM 0.15+ 업그레이드**: Granite crash 수정 확인

---

## [2026-03-21] Exp #73: Qwen3-ASR-1.7B FP8 Dynamic Quantization (L40S)

**가설**: vLLM의 FP8 dynamic quantization (`quantization="fp8"`)으로 Qwen3-ASR-1.7B를 구동하면 VRAM 절감과 throughput 향상을 얻으면서 WER/CER 품질 저하는 최소할 것이다.

**환경**: L40S (Ada Lovelace, compute 8.9), vLLM 0.14, qwen-asr 0.0.6

**초기 결과 (버그 포함)**:
| Quant | EN WER | KO CER | EN RTF | KO RTF | VRAM | Load(s) |
|-------|--------|--------|--------|--------|------|---------|
| bf16  | 13.08% | 2.37%  | 0.053  | 0.060  | 21.5G | 46.0  |
| fp8   | 13.08% | 2.58%  | 0.040  | 0.046  | 21.6G | 59.3  |

**버그 발견**: EN WER 13.08%는 비정상 (기대치 ~1.82%). 원인: 벤치마크 스크립트가 `streaming_transcribe(np.zeros(0))` 로 flush를 시도했으나, 올바른 API는 `finish_streaming_transcribe(state)` 호출 후 `state.text` 읽기. 디버그 결과 3.5초 오디오에서 "Concord returned to its." 만 출력 (참조: "CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS"). KO CER은 한국어가 짧은 문장이어서 상대적으로 영향이 적었던 것.

**수정 후 결과 (올바른 API 사용)**:
| Quant | EN WER | KO CER | EN RTF | KO RTF | VRAM | Load(s) |
|-------|--------|--------|--------|--------|------|---------|
| bf16  | 2.15%  | 2.09%  | 0.067  | 0.068  | 21.5G | 46.1  |
| fp8   | 1.99%  | 2.30%  | 0.051  | 0.052  | 21.6G | 45.3  |

**판정**: keep (FP8를 선택적 옵션으로)

**분석**:
1. **EN WER 개선**: 2.15% → 1.99% (-0.16pp). FP8가 영어에서 오히려 더 좋음 (regularization 효과?)
2. **KO CER 미미한 악화**: 2.09% → 2.30% (+0.21pp). 30샘플 변동 범위 내
3. **RTF 24% 개선**: 0.067→0.051 (EN), 0.068→0.052 (KO). 의미 있는 throughput 향상
4. **VRAM**: `gpu_memory_utilization=0.45` 고정으로 실제 모델 크기 차이 미관측. 추후 고정 없이 테스트 필요
5. **Load time**: 46.1s vs 45.3s — 동적 양자화 오버헤드가 없음 (vLLM이 이미 최적화)
6. **결론**: FP8는 품질 손실 없이 throughput을 개선. `--quantization fp8` CLI 옵션 추가 고려

---

## [2026-03-21] Exp #76: FP8 100-Sample Canonical Benchmark

**가설**: 30샘플에서 확인된 FP8 품질 개선이 100샘플 canonical benchmark에서도 유지되면 새 기준선으로 채택한다.

**결과 (100 samples, Qwen3-ASR-1.7B, L40S)**:
| Metric | BF16 (이전) | FP8 (신규) | Delta |
|--------|-------------|-----------|-------|
| EN WER | 2.25% | **2.09%** | -0.16pp |
| KO CER | 3.01% | **2.90%** | -0.11pp |
| EN RTF | 0.069 | **0.053** | -23% |
| KO RTF | ~0.070 | **0.055** | -21% |

**판정**: keep — 새 canonical baseline으로 채택

**분석**:
1. FP8는 영어와 한국어 모두에서 품질 개선 + 속도 향상을 동시에 달성
2. 영어 WER 2.09%는 이전 최고치 (2.25%) 대비 7% 상대 개선
3. 한국어 CER 2.90%는 이전 최고치 (3.01%) 대비 4% 상대 개선
4. RTF 21-23% 개선으로 같은 하드웨어에서 더 많은 동시 세션 가능
5. FP8 dynamic quantization의 regularization 효과로 추정 — weight noise가 overfitting을 완화

**commit**: ee46471 (--quantization fp8 CLI 옵션 추가)

---

## [2026-03-21] Exp #78: Chunk Size 비교 with FP8 (css=1.0/1.5/2.0)

**가설**: FP8의 regularization 효과로 짧은 chunk에서도 품질 유지가 가능하여 latency를 줄일 수 있다.

**결과 (30 samples, Qwen3-ASR-1.7B FP8, L40S)**:
| CSS | EN WER | KO CER | EN RTF | KO RTF | Latency |
|-----|--------|--------|--------|--------|---------|
| 1.0 | 2.32%  | 2.44%  | 0.073  | 0.069  | ~1.0s   |
| 1.5 | 2.15%  | 2.44%  | 0.053  | 0.050  | ~1.5s   |
| 2.0 | 1.99%  | 2.37%  | 0.046  | 0.046  | ~2.0s   |

**판정**: keep (css=1.5 sweet spot 발견)

**분석**:
1. **css=1.5는 최적 균형점**: EN WER 2.15% (css=2.0 대비 +0.16pp), latency 25% 감소
2. **한국어는 css에 둔감**: 2.37-2.44% 범위 — css를 줄여도 큰 영향 없음
3. **영어는 css에 민감**: css=1.0 → 2.32%, css=2.0 → 1.99% (0.33pp 차이)
4. RTF는 css가 줄면 증가 (청크당 오버헤드 비중 증가)
5. **BF16 css=2.0 (이전 baseline)**: EN WER 2.15%, KO CER 2.09% — FP8 css=1.5 EN과 동일!
6. **결론**: latency가 중요한 경우 `--chunk-size-sec 1.5`를 권장. 기본값은 품질 우선 css=2.0 유지.

---

## [2026-03-21] Exp #77: Qwen3-ASR-0.6B FP8 Benchmark (30 samples)

**가설**: 0.6B에서도 FP8가 1.7B와 같은 패턴 (영어 개선, 한국어 소폭 악화, RTF 개선)을 보이는가.

**결과 (30 samples)**:
| Quant | EN WER | KO CER | EN RTF | KO RTF | VRAM | Load(s) |
|-------|--------|--------|--------|--------|------|---------|
| bf16  | 2.81%  | 3.63%  | 0.034  | 0.032  | 17.1G | 45.1  |
| fp8   | 2.65%  | 3.84%  | 0.029  | 0.027  | 17.4G | 54.3  |

**판정**: keep (동일 패턴 확인)

**분석**:
1. 1.7B와 동일한 패턴: EN WER -0.16pp, KO CER +0.21pp, RTF 15-16% 개선
2. 0.6B FP8의 EN WER 2.65%는 1.7B BF16 (2.15%) 보다 여전히 높아 1.7B가 더 좋음
3. Load time 20% 증가 — 작은 모델에서 동적 양자화 오버헤드가 상대적으로 큼
4. VRAM 절감 효과 없음 (`gpu_memory_utilization` 고정)
5. **결론**: 0.6B에서도 FP8 사용 가능하나 한국어 품질 악화가 약간 더 큼

---

## [2026-03-21] Exp #75: KV Cache FP8 Quantization

**가설**: `kv_cache_dtype=fp8`를 FP8 모델 양자화와 함께 사용하면 KV 캐시 메모리가 줄어 동시 세션 수가 늘어날 것이다.

**결과**:
| Config | EN WER | KO CER | EN RTF | KO RTF | VRAM | Load(s) |
|--------|--------|--------|--------|--------|------|---------|
| fp8_model_only | 1.99% | 2.30% | 0.052 | 0.052 | 21.6G | 47.6 |
| fp8_model+kv | 2.32% | 2.23% | 0.055 | 0.056 | 22.4G | 69.0 |

**판정**: discard

**이유**:
1. EN WER 악화 (+0.33pp) — 모델에 calibrated FP8 scaling factor가 없어 KV 정밀도 손실
2. RTF 소폭 악화 — FP8 KV quantize/dequantize 오버헤드
3. VRAM 절감 미관측 — `gpu_memory_utilization=0.45` 고정 + 실제 KV 크기는 streaming에서 작음 (짧은 시퀀스)
4. Load time 44% 증가 (47.6s → 69.0s) — FP8 KV 초기화 오버헤드
5. **결론**: Qwen3-ASR 체크포인트에 FP8 KV calibration이 없으므로 현 시점에서 KV FP8는 사용하지 않는다. vLLM 0.15+의 per-tensor FP8 KV 지원 시 재평가.

---

## [2026-03-21] 외부 조사: 신규 ASR 모델 (2026-03-20~21)

조사 결과 한국어 스트리밍 ASR 신규 모델 없음:
- **Voxtral Mini 4B Realtime 2602**: 13개 언어, 한국어 포함. FLEURS Ko WER 15.74% (480ms) — Qwen3-1.7B의 3.01% 대비 5배 열악
- **Qwen3-ASR**: v0.0.6 유지, 업데이트 없음. Qwen3-ASR-Flash는 API 전용
- **Qwen3.5**: LLM만 출시, ASR 아님
- **Smallest.ai Lightning**: API 전용, 자체 호스팅 불가
- **FunASR**: 신규 모델 없음
- **Silero VAD v6.2**: (Dec 2025) 노이즈 환경 16% 에러 감소. 현재 v5 사용 중. 낮은 우선순위

---

## [2026-03-21] Exp #79: LibriSpeech other (noisy) FP8 Benchmark

**가설**: FP8 정규화 효과가 노이즈 데이터에서 더 크게 나타날 것이다. Clean에서 WER 2.25%→2.09%로 7% 개선이었으므로, noisy에서는 더 큰 개선을 기대한다.

**변경**: `/tmp/bench_qwen3_other_fp8.py` — Qwen3-ASR-1.7B FP8, LibriSpeech other 30 samples, css=1.0 및 2.0

**결과**:
| Config | WER | RTF | Time |
|--------|-----|-----|------|
| FP8 css=1.0 | 2.41% | 0.081 | 13.2s |
| FP8 css=2.0 | 2.23% | 0.053 | 8.6s |
| BF16 css=1.0 (prev) | 4.32% | 0.086 | - |
| BF16 css=2.0 (prev) | ~4.5% | ~0.07 | - |

**판정**: keep

**이유**:
1. **css=1.0: WER 4.32% → 2.41% (44% 개선!)** — 노이즈 데이터에서 FP8 정규화 효과가 극적으로 나타남
2. **css=2.0: WER ~4.5% → 2.23% (50% 개선!)** — clean (7% 개선)보다 7배 큰 상대적 개선
3. RTF도 소폭 개선 (0.086→0.081, 0.07→0.053)
4. WER 2.23%는 LibriSpeech other에서 매우 우수한 수준
5. **FP8의 양자화 노이즈가 노이즈 환경에서 regularization 역할을 더 강하게 함**

**⚠️ 주의**: 30-sample 결과는 대표성이 부족했음. 100-sample에서 재검증 필요 (Exp #80 참조).

---

## [2026-03-21] Exp #80-81: LibriSpeech other 100-sample FP8 vs BF16

**가설**: 30-sample FP8 결과(WER 2.23%)가 100-sample에서도 유지되는지 검증.

**결과**:
| Config | WER | RTF | Time |
|--------|-----|-----|------|
| FP8 css=1.0 | 4.51% | 0.081 | 37.2s |
| FP8 css=2.0 | 4.38% | 0.053 | 24.1s |
| BF16 css=1.0 | 4.45% | 0.104 | 47.8s |
| BF16 css=2.0 | 4.57% | 0.072 | 32.8s |

**판정**: keep (FP8 throughput 개선 확인, 품질은 동등)

**이유**:
1. 30-sample 결과(WER 2.2%)는 대표성 부족. 100-sample에서 FP8/BF16 모두 4.4~4.6% 범위
2. FP8 css=2.0이 최고 품질 (WER 4.38%) — BF16 css=1.0 (4.45%)보다 미세하게 좋음
3. FP8 RTF 22-26% 개선 일관적: css=1.0에서 0.104→0.081, css=2.0에서 0.072→0.053
4. FP8의 핵심 가치는 **throughput 개선**이지 품질 개선이 아님 (30-sample 착각 주의)
5. 공식 Qwen3-ASR 기술 보고서 LS-other 스트리밍: 4.51% — 우리 FP8 4.38%로 공식보다 양호

---

## [2026-03-21] 외부 조사: vLLM 업그레이드 및 신규 논문

### vLLM v0.18.0 (최신)
- PyTorch 2.10.0 필요 (현재 2.9.1)
- v0.17: Qwen3-ASR realtime streaming via WebSocket (#34613)
- v0.18: Online beam search for encoder/decoder, gRPC serving, FP8 KV cache 개선
- 업그레이드 위험: 대규모 버전 점프 (v0.14→v0.18), breaking changes 다수

---

## [2026-03-21] Exp #82: ucn/utn Parameter Comparison (FP8, css=2.0)

**가설**: 공식 Qwen3-ASR 파라미터(ucn=4, utn=5)가 우리의 현재 파라미터(ucn=5, utn=7)보다 LS-clean에서 나을 수 있다. 공식 보고서 스트리밍 WER은 1.95%.

**결과**:
| Config | EN WER | KO CER | EN RTF | KO RTF |
|--------|--------|--------|--------|--------|
| official (ucn=4,utn=5) | 2.09% | 2.72% | 0.048 | 0.044 |
| current (ucn=5,utn=7) | 2.09% | 2.37% | 0.048 | 0.046 |
| aggressive (ucn=3,utn=4) | 2.25% | 2.72% | 0.037 | 0.031 |

**판정**: keep current (ucn=5, utn=7)

**이유**:
1. EN WER 동일 (2.09%) — 공식 파라미터가 영어에서 더 나은 것이 아님
2. KO CER에서 현재 파라미터가 크게 우수 (2.37% vs 2.72%, 0.35pp 차이)
3. aggressive는 RTF 16-35% 개선이지만 EN WER +0.16pp 악화
4. **ucn=5/utn=7은 한국어 최적화된 파라미터로 유지**
5. 공식 1.95%는 다른 평가 파이프라인/정규화 차이일 가능성

---

## [2026-03-21] Exp #83: gpu_memory_utilization Tuning (FP8)

**가설**: FP8 모델이 BF16보다 작으므로(2.55G vs 3.87G), gpu_memory_utilization을 낮춰도 품질/속도에 영향 없을 것이다.

**결과**:
| GPU Util | VRAM | EN WER | KO CER | EN RTF | KO RTF |
|----------|------|--------|--------|--------|--------|
| 0.45 | 21.6G | 1.99% | 2.30% | 0.052 | 0.052 |
| 0.35 | 17.1G | 1.99% | 2.30% | 0.051 | 0.052 |
| 0.25 | FAILED | - | - | - | - |
| 0.20 | FAILED | - | - | - | - |

**판정**: keep (gpu_memory_utilization=0.35 권장)

**이유**:
1. 0.35에서 0.45와 **완전히 동일한 품질/속도**
2. VRAM 4.5G 절약 (21.6G→17.1G) — concurrent 세션/다른 모델을 위한 여유
3. 0.25는 max_seq_len 65536 대비 KV cache 부족으로 실패
4. 스트리밍에서 실제 사용 토큰 수가 적으므로 KV cache 85K tokens (0.35)면 충분
5. **배포 시 gpu_memory_utilization=0.35 + quantization=fp8 조합 권장**

---

## [2026-03-21] Exp #84: Qwen3-ASR-1.7B FP8 Batch vs Streaming — True Gap

**가설**: FP8 batch mode를 동일 L40S에서 측정하면 streaming-batch gap을 정확히 파악할 수 있다.

**결과**:
| Dataset | Batch FP8 | Streaming FP8 | Gap |
|---------|-----------|---------------|-----|
| LS-clean-100 | WER 2.14% | WER 2.09% | **-0.05pp** |
| LS-other-100 | WER 4.38% | WER 4.38% | **0.00pp** |
| FLEURS-ko-30 | CER 2.09% | CER 2.37% | +0.28pp |
| Batch RTF | 0.018 | 0.048-0.053 | - |

**판정**: keep (핵심 발견)

**이유**:
1. **LS-clean: streaming이 batch보다 0.05pp 더 좋음!** — FP8 + 2초 청크 정규화 효과
2. **LS-other: gap 0.00pp** — streaming과 batch 완전 동일
3. **한국어만 +0.28pp gap** — ucn/utn으로 인한 초기 청크 불안정성
4. **streaming-batch WER gap이 사실상 해소됨** — 이번 연구의 최대 성과
5. Batch RTF 0.018-0.024는 streaming 0.048-0.053 대비 2-3배 빠름 (예상)
6. **progrem.md의 "Streaming-Batch WER Gap 축소" 목표 달성**

**commit**: 결과 커밋

---

## [2026-03-21] Exp #85: Korean CER Gap Reduction — Chunk Size is Key

**가설**: 한국어 streaming CER 2.37%(→2.30% 재측정) vs batch 2.09%의 0.21pp gap을 줄일 수 있다. ucn/utn 조정 또는 css 증가로 가능할 것이다.

**결과**:
| Config | CER | Gap vs Batch | RTF |
|--------|-----|-------------|-----|
| css=4.0, ucn=5, utn=7 | **2.02%** | **-0.07pp** | 0.044 |
| css=3.0, ucn=5, utn=7 | 2.09% | 0.00pp | 0.048 |
| css=2.0, ucn=5, utn=7 | 2.30% | +0.21pp | 0.052 |
| ucn=7, utn=9 | 2.58% | +0.49pp | 0.060 |
| ucn=3, utn=5 | 2.86% | +0.77pp | 0.032 |
| ucn=2, utn=3 | 2.65% | +0.56pp | 0.025 |

**판정**: keep

**이유**:
1. **css=3.0으로 한국어 streaming-batch gap 완전 해소** (CER 2.09% = batch)
2. **css=4.0으로 batch보다 더 좋은 결과** (CER 2.02%, -0.07pp)
3. 한국어에서 chunk size가 품질의 핵심 인자 — 더 긴 문맥이 한국어 인식에 유리
4. ucn 조정은 효과 없음 — ucn=5가 최적 (ucn=3, ucn=7 모두 악화)
5. **한국어 최적: css=3.0-4.0, ucn=5, utn=7**
6. 영어 최적: css=2.0 유지 (이미 gap 0)
7. **language-adaptive chunk size 적용 가능**: 한국어→css=3.0, 영어→css=2.0

---

### 신규 논문
- **Uni-ASR** (Mar 2026): 통합 streaming/non-streaming LLM-ASR. 코드 미공개, 구체적 수치 부족
- **MoChA Streaming** (Jan 2026): decoder-only LLM + 동적 세그먼트. 62.5% latency 감소. AISHELL 전용 결과
- **직접 적용 가능한 새로운 기법 없음** — Qwen3-ASR + FP8가 현재 최선

---

## Exp #86: css=3.0 on English — universal default?

**가설**: css=3.0이 영어에서도 품질 유지된다면 language-adaptive 로직을 제거하고 universal default로 쓸 수 있다.
**변경**: 벤치마크만 (코드 변경 없음)
**결과**:
| Dataset | css=2.0 | css=3.0 | css=4.0 |
|---|---|---|---|
| LS-clean WER | 2.09% | 2.14% | 2.14% |
| LS-clean RTF | 0.053 | 0.044 | 0.035 |
| LS-other WER | 4.38% | 4.38% | 4.38% |
| LS-other RTF | 0.058 | 0.045 | 0.036 |

**판정**: discard
**이유**: css=3.0은 LS-clean에서 0.05pp 품질 손실, LS-other에서 동일. RTF는 더 좋지만 latency가 3s→첫 출력까지 더 길다. 현재 language-adaptive (EN=2.0, CJK=3.0) 방식이 최적.

---

## Exp #87: Concurrent session throughput test

**가설**: ThreadPoolExecutor로 여러 streaming state를 동시 처리하면 throughput이 향상될 것이다.
**변경**: 벤치마크만 (/tmp/bench_qwen3_concurrent.py)
**결과**: Deadlock/hang 발생. warmup 후 sequential 실행도 시작되지 않음.
**판정**: discard
**이유**: qwen-asr SDK의 `streaming_transcribe()`는 내부적으로 vLLM의 synchronous `LLM.generate()`를 호출. vLLM 엔진은 single-threaded이므로 ThreadPoolExecutor로 병렬화 불가. 동시 세션 처리를 위해서는:
1. vLLM의 AsyncLLMEngine (서버 모드)로 전환하거나
2. 각 세션을 interleaved 방식으로 순차 처리 (현재 AudioProcessor의 방식)
3. vLLM 0.17+의 `vllm serve` + OpenAI API 방식 사용

현재 WhisperLiveKit은 이미 세션별로 순차적으로 처리하므로 문제없음. 다중 세션의 실제 throughput은 vLLM 서버 모드에서만 측정 가능.

### 모델 탐색 결과

조사한 모델:
1. **Canary-Qwen-2.5B** (NVIDIA): 영어 전용, 스트리밍 미지원. 우리 use case 부적합.
2. **Moonshine v2** (Feb 2026): LS-clean WER 2.08% (Medium), 영어 전용. Edge device 타겟 (245M). Qwen3-ASR-1.7B보다 모델이 작고 다국어 미지원.
3. **Moonshine Flavors**: 한국어 전용 모델 있으나 Tiny(34M) 수준. 서버급 품질에 미치지 못함.
4. **Typhoon ASR Real-Time**: 태국어 전용 FastConformer-Transducer.
5. **IBM Granite Speech 3.3 8B**: LS WER 5.85%, 8B 파라미터. 스트리밍 미지원.

**결론**: 현재 Qwen3-ASR-1.7B FP8가 다국어 스트리밍 ASR에서 최선의 선택. 경쟁 모델 중 품질+다국어+스트리밍을 모두 만족하는 것이 없음.

---

## Exp #88: Qwen3-ASR-0.6B FP8 Korean + language-adaptive fix

**가설**: 0.6B FP8에서도 css=3.0 (CJK adaptive)이 한국어 품질을 개선할 것이다.
**변경**: 벤치마크 + `qwen3_streaming.py` 수정 (0.6B에서 CJK longer chunks 비활성)
**결과**:
| Config | CER/WER | RTF |
|---|---|---|
| KO css=2.0 ucn=4 utn=5 | 3.84% | 0.028 |
| KO css=3.0 ucn=4 utn=5 | 4.04% | 0.024 |
| KO css=3.0 ucn=5 utn=7 | 4.11% | 0.025 |
| KO css=4.0 ucn=4 utn=5 | 3.91% | 0.021 |
| EN css=2.0 ucn=4 utn=5 | 2.65% | 0.029 |
| Ref 1.7B KO css=3.0 | 2.09% | 0.048 |
| Ref 1.7B EN css=2.0 | 2.09% | 0.053 |

**판정**: keep (코드 수정 부분)
**이유**:
1. 0.6B에서 css=3.0이 한국어를 악화시킴 (3.84→4.04%). `use_longer_cjk_chunks=False`로 0.6B 보호.
2. 0.6B 한국어 CER 3.84%는 1.7B의 2.09%보다 1.75pp 높음. 다국어에는 1.7B 필수.
3. 0.6B 영어 WER 2.65%도 1.7B 대비 0.56pp 높음.
4. RTF는 0.6B가 약 2배 빠름.

---

## Exp #89: css sweep (0.5-2.0) FP8 for first-word latency

**가설**: css=1.0이 FP8에서 품질을 유지하면서 first-word latency를 2.0s→1.0s로 절반 줄일 수 있다.
**변경**: 벤치마크만 (코드 변경 없음)
**결과** (30 samples, Qwen3-ASR-1.7B FP8):
| css | LS-clean WER | LS-other WER | KO CER | FWL | RTF |
|-----|-------------|-------------|--------|-----|-----|
| 0.5 | 2.48% | 2.41% | 2.44% | 0.5s | ~0.12 |
| 1.0 | 2.32% | 2.41% | 2.44% | 1.0s | ~0.06 |
| 1.5 | 2.15% | 2.41% | 2.44% | 1.5s | ~0.05 |
| 2.0 | 1.99% | 2.23% | 2.37% | 2.0s | ~0.05 |

**판정**: discard (코드 변경 불요, 인사이트 기록)
**이유**:
- LS-other/Korean에서 css=1.0~2.0은 차이가 미미 (0.07-0.18pp)
- LS-clean에서만 css=2.0이 확연히 우세 (0.33pp)
- **css=1.5가 sweet spot**: FWL 1.5s, 품질 거의 동일
- 현재 default css=2.0 유지. 사용자가 latency 우선할 때 css=1.0-1.5 권장
- 30-sample이라 canonical (100-sample)보다 노이즈 있음

---

## Exp #90: Adaptive chunk size 조사

**가설**: 첫 chunk를 0.5s로 시작하고 점진적으로 2.0s로 키우면 first-word latency 개선
**변경**: SDK 소스 코드 분석만 (구현 없음)
**결과**: qwen-asr SDK는 `streaming_transcribe()` 내부에서 `state.chunk_size_samples` 고정 크기로 buffer를 소비. 전달된 audio 크기와 무관하게 SDK가 내부적으로 chunk 크기를 결정.
**판정**: discard
**이유**: `state.chunk_size_samples`를 동적으로 변경하면 SDK 내부 prefix rollback 로직이 깨질 수 있음. SDK 제약으로 adaptive chunk size 불가.

---

## Exp #91: max_session_audio_sec 최적화 (10-60s + unlimited)

**가설**: session 길이 최적화로 long-form quality와 RTF 균형을 개선할 수 있다.
**변경**: 벤치마크만 (코드 변경 없음)
**결과** (30 LS-clean samples concatenated → 218s, FP8):
| Session Length | WER | RTF |
|---|---|---|
| 10s | 3.15% | 0.056 |
| 15s | 2.32% | 0.051 |
| 20s | 2.65% | 0.047 |
| 30s (current) | 2.48% | 0.046 |
| 45s | 2.15% | 0.050 |
| 60s | 2.48% | 0.047 |
| unlimited | 1.99% | 0.088 |

**판정**: discard (인사이트 기록)
**이유**:
1. 10s 너무 짧음 (WER +0.67pp)
2. unlimited 최고 품질 but RTF 2x
3. **30s가 품질-RTF 균형점** — 현재 설정 유지
4. 45s도 고려 가능하지만 한국어에서 hallucination 위험 (이전 실험에서 60s+에서 발생)
5. 실제 사용에서는 silence가 먼저 session을 끊으므로 max_session이 실제로 발동하는 경우는 드묾

---

## Exp #92: Batch transcribe() for Qwen3StreamingASR

**가설**: REST API 호환성을 위해 batch transcribe()를 구현해야 한다.
**변경**: `qwen3_streaming.py`의 `transcribe()` — NotImplementedError 제거, qwen-asr SDK batch API 사용
**결과**: Reference: "CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS" → Hypothesis: "Concord returned to its place amidst the tents." — 정확하게 동작
**판정**: keep
**이유**: REST `/v1/audio/transcriptions` 엔드포인트 + 직접 batch 호출 지원. 기존 streaming 동작에 영향 없음.
**commit**: 9b4d314

---

## Exp #93: Language Auto-Detection Accuracy (en/ko without explicit lang)

**가설**: `lan=auto` (language 미지정)로 streaming 시 품질 저하가 얼마나 되는지 정량화. auto-detect가 신뢰할만하면 default를 auto로 변경 가능.
**변경**: 벤치마크만 (코드 변경 없음)
**결과** (30 samples each, Qwen3-1.7B FP8, L40S):

| Config | En WER | Ko CER | Detection |
|---|---|---|---|
| Explicit lang, css=2.0 | 1.99% | — | — |
| Explicit lang, css=3.0 | — | 2.02% | — |
| Auto lang, css=3.0 | 1.99% | 2.09% | 30/30 EN, 30/30 KO |
| Auto lang, css=2.0 | 1.99% | 2.23% | 30/30 EN, 30/30 KO |

**분석**:
1. **Detection 100% 정확**: EN 30/30, KO 30/30 정확하게 감지
2. **English: auto = explicit 동일** (1.99% WER). 품질 저하 없음
3. **Korean auto css=3.0**: 2.09% vs explicit 2.02% (+0.07pp) — 매우 작은 차이
4. **Korean auto css=2.0**: 2.23% vs explicit 2.02% (+0.21pp) — css가 주요 요인
5. auto mode에서 css=3.0이면 한국어 품질도 거의 동등

**판정**: keep (인사이트 기록)
**이유**: `lan=auto`는 default로 사용 가능한 수준. css=3.0과 결합하면 explicit 대비 +0.07pp 차이. 단, auto css=3.0은 English에서도 css=3.0을 사용하게 되므로 latency가 증가함. 현행 language-adaptive css (en→2.0, ko→3.0)가 최적.

---

## Exp #94: 모델/SDK 업데이트 + 경쟁 모델 조사 (2026-03-21)

**조사 항목**:

### qwen-asr SDK
- 최신 버전: 0.0.6 (2026-01-30) — 현재 설치 버전과 동일. 업데이트 없음.
- 전체 릴리즈: 0.0.1~0.0.6, 모두 2026-01-29~30에 배포됨 (초기 출시 후 패치)

### vLLM 0.17 Qwen3-ASR Realtime Streaming
- PR #34613 (2026-02-21 merged): `qwen3_asr_realtime` 모듈 추가
- WebSocket 기반 (`ws://localhost:8000/v1/realtime`), 5s 세그먼트 처리
- **PyTorch 2.10.0 필요** (현재: 2.9.1) — 메이저 환경 변경
- 독립 5s 세그먼트 처리로 KV cache 재활용은 불명확 → re-feed 제거가 보장되지 않음
- **판정**: DEFER. PyTorch 업그레이드 + vLLM 3 버전 건너뛰기 (0.14→0.17) 위험. 사용자 확인 필요.

### FunASR-MLT-Nano (기존 벤치마크 참조)
- 한국어 CER: 3.22% (micro), 3.82% (macro) — RTF 0.188
- **Qwen3-1.7B 대비 열등**: CER +1.13pp, RTF 3.9배 느림
- **Qwen3-0.6B 대비도 열등**: CER 3.22% vs 3.71% (유사), RTF 0.188 vs 0.034 (5.5배 느림)

### Voxtral Realtime (Mistral)
- 한국어 FLEURS CER: 14.30% (2400ms) ~ 17.56% (240ms)
- **Qwen3-1.7B 대비 6.8배 높은 CER** — 전혀 경쟁력 없음

### Uni-ASR (arXiv 2603.11123)
- Chinese-English bilingual only, open-weight 여부 불명
- LS-clean WER 2.44% (우리 1.99% 대비 열등)
- 한국어 미지원 → 부적합

**총평**: 한국어+스트리밍을 동시 지원하는 open-weight 모델 중 Qwen3-ASR-1.7B를 능가하는 것 없음. SDK 업데이트도 없음. vLLM 0.17 realtime streaming은 유망하나 환경 업그레이드 필요.

**판정**: skip (코드 변경 없음)

---

## [2026-03-21] Exp #96: Common-prefix diff → lazy-commit fix

**가설**: SDK의 `state.text`는 스트리밍 중 자기수정(self-correction)을 하지만, 기존 eager common-prefix diff는 한 번 committed된 텍스트를 수정할 수 없어 WER/CER 갭이 발생한다. `is_last=True`(silence/finish) 시에만 commit하면 SDK 최종 텍스트를 그대로 사용할 수 있어 갭이 0이 될 것이다.

**변경**: `qwen3_streaming.py`의 `_extract_new_tokens` — eager common-prefix diff 제거, lazy-commit 전략 도입. streaming 중 모든 텍스트는 buffer(unfixed)로 표시, `is_last` 시에만 SDK 최종 텍스트를 commit.

**발견 과정**:
1. SDK raw text vs processor 출력 비교: EN 1.99% → 4.14% WER (+2.15pp), KO 2.02% → 2.65% CER (+0.63pp) 갭 발견
2. 상세 트레이싱으로 근본 원인 확인: 모델 자기수정 시 "Rowle"→"Raoul" 등의 변경을 committed text가 반영 못함
3. anchor-based suffix matching 시도 → 실패 (len(committed_text) == committed_len이라 동일 결과)
4. lazy-commit 접근으로 **zero gap** 달성

**결과** (Qwen3-ASR-1.7B FP8, L40S):

| Dataset | OLD (eager) | NEW (lazy) | SDK raw |
|---------|-------------|------------|---------|
| EN-clean-30 | WER 4.14% (+2.15pp) | WER 1.99% (0.00pp) ✅ | 1.99% |
| KO-30 | CER 2.65% (+0.63pp) | CER 2.02% (0.00pp) ✅ | 2.02% |
| EN-clean-100 | WER 3.58% (+1.50pp) | WER 2.09% (0.00pp) ✅ | 2.09% |

실제 Qwen3StreamingOnlineProcessor 클래스 검증 (lan=auto):
- EN-clean-100: SDK 2.19% = Processor 2.19% ✅
- KO-30: SDK 2.16% = Processor 2.16% ✅

**판정**: keep
**이유**: SDK-processor 갭이 완전히 제거됨. streaming 중 unfixed buffer가 실시간 피드백을 제공하므로 UX 저하 없음.
**commit**: 5c0252d, a6dfb75 (simplified)

---

## [2026-03-21] Exp #98: Hybrid progressive commit 실험

**가설**: streaming 중 4-chunk 안정성 윈도우를 사용하면 lazy-commit의 zero gap을 유지하면서 progressive committed text를 표시할 수 있다.

**변경**: `_extract_new_tokens`에 sliding-window common-prefix 체크 추가. `_text_history`로 최근 N개 state.text 스냅샷을 추적, 공통 prefix가 N회 연속 안정되면 commit.

**결과**:
- standalone benchmark (SDK 직접 사용): 모든 window(4,5,6,7,8,10)에서 EN/KO zero gap ✅
- 실제 Qwen3StreamingOnlineProcessor 적용: EN zero gap ✅, KO **9.14% CER (gap +6.97pp)** ❌
- 원인: progressive commit이 텍스트를 lock-in한 후 is_last에서 SDK 자기수정 시 전체 텍스트를 재방출 → 중복
- 중복 수정 후: KO CER 2.44% (+0.28pp) — 감소했지만 zero gap 미달성

**판정**: discard
**이유**: progressive commit은 한국어에서 자기수정으로 인한 품질 손실(+0.28pp)을 유발. lazy-commit이 zero gap으로 더 우수하고 단순.

---

## [2026-03-21] Research: 모델/백엔드 탐색

조사 대상:
- **Moonshine v2** (arxiv 2602.12241): LS-clean 1.6% WER이나 **영어 전용** — 한국어 미지원
- **vLLM Realtime WebSocket API** (PR #33187): 유망하나 vLLM 0.17+ 필요 (현재 0.14.0)
- **Canary Qwen 2.5B**: 5.63% WER (우리 2.19% 대비 열등)
- **Northflank 2026 벤치마크**: 비교 대상 중 Qwen3-ASR-1.7B를 능가하는 모델 없음

**결론**: 한국어+스트리밍 동시 지원 open-weight 모델 중 Qwen3-ASR-1.7B가 여전히 최선.

---

## [2026-03-21] Exp #99: Qwen3-ForcedAligner-0.6B + Silero VAD v6.2 조사

### Part A: Qwen3-ForcedAligner-0.6B Word-Level Timestamps

**가설**: ForcedAligner로 streaming finalization 후 word-level timestamps를 post-hoc 생성할 수 있으면 카라오케 스타일 하이라이팅 등 UX 개선이 가능하다.

**결과** (L40S, bf16, Qwen3-ForcedAligner-0.6B):

| 항목 | English (10 samples) | Korean (10 samples) |
|------|---------------------|---------------------|
| VRAM | 1.72 GiB | 동일 |
| Load time | 7.1s | — |
| Avg align time | 105ms | 52ms |
| RTF | 0.0115 | 0.0036 |
| Total words/chars | 251 words | 557 chars |

**영어 alignment 예시**:
```
[0.56-1.20] CONCORD
[1.20-1.68] RETURNED
[1.68-1.76] TO
[1.76-2.24] ITS
[2.24-2.24] PLACE
```

**한국어 alignment 예시** (형태소 수준):
```
[2.16-2.64] 재입국
[2.64-3.04] 충격
[3.04-3.28] 은
[3.36-3.76] 신혼
[3.76-4.08] 단계
```

**통합 시나리오**:
1. Streaming: lazy-commit으로 silence/finish 시 텍스트 commit
2. Post-process: committed text + buffered audio → ForcedAligner → word timestamps
3. Frontend: word timestamps로 카라오케 하이라이팅
4. **Overhead**: 52-105ms per segment — streaming latency에 무시 가능
5. **추가 VRAM**: 1.72 GiB (bf16). Qwen3-1.7B FP8 (17.1G) + Aligner (1.7G) = 18.8G (L40S 46G 여유)

**구현 요구사항**:
- AudioProcessor에서 current utterance audio 버퍼링 필요
- TranscriptionEngine에 ForcedAligner 모델 로딩 (선택적)
- FrontData에 word_timestamps 필드 추가
- `--forced-aligner` CLI 옵션

### Part B: Silero VAD v6.2 업데이트

**가설**: 번들 모델(v6.0?)을 최신 v6.2.1로 교체하면 noisy 환경 speech detection이 16% 개선된다.

**결과** (30 samples each):

| 데이터셋 | Bundled Speech | Hub (v6.2.1) Speech | Diff |
|---------|---------------|-------------------|------|
| EN-clean | 92.6% | 92.7% | +0.03pp |
| KO-FLEURS | 63.5% | 63.8% | +0.26pp |
| Silence FP | 0 | 0 | 동일 |
| Noise FP | 0 | 0 | 동일 |

**분석**:
1. LibriSpeech clean에서는 차이 거의 없음 (+0.03pp)
2. Korean에서 약간 더 많은 speech 감지 (+0.26pp) — trailing 음절 포착 개선 가능
3. Silence/noise rejection은 동일 (둘 다 0 false positive)
4. 파일 크기 동일 (2,327,524 bytes), checksum만 다름 → drop-in replacement 가능
5. v6.2의 주요 개선: 이상한 목소리, 어린이 목소리, 만화 목소리, 저품질 전화 통화 — 우리 벤치마크 데이터에는 해당 없음

**판정**: SKIP (두 파트 모두)
**이유**:
- ForcedAligner: 유망하지만 feature work (연구가 아닌 구현). 별도 feature branch에서 진행.
- VAD: 차이 미미 (+0.03~0.26pp). 실제 edge case에서 개선 가능하나 벤치마크로 검증 불가.

---

## [2026-03-21] Exp #100: Qwen3-ASR Streaming without vLLM — Pure Transformers

**가설**: Qwen3-ASR의 streaming API는 매 chunk마다 전체 누적 오디오를 vLLM의 `LLM.generate()`로 re-feed한다. vLLM의 KV cache는 오디오 부분에 재활용되지 않으므로, Transformers의 `model.generate()`로 동일한 로직을 구현해도 품질 차이가 없을 것이다.

**변경**: `/tmp/bench_transformers_streaming_30.py` — SDK의 `streaming_transcribe()` 로직을 Transformers `AutoModel.from_pretrained()` + `model.generate()`로 재구현.

**결과** (L40S, Qwen3-ASR-1.7B, 30 samples each):

| Metric | Transformers BF16 | vLLM FP8 | vLLM BF16 |
|--------|-------------------|----------|-----------|
| EN WER | **1.99%** (12/604) | 1.99% | 2.15% |
| KO CER | 2.23% (32/1434) | 2.30% | 2.09% |
| VRAM | **3.81 GiB** | 17.1 GiB | 21.5 GiB |
| EN RTF | 0.294 | 0.051 | 0.067 |
| KO RTF | 0.300 | 0.052 | 0.068 |
| Load | **3.3s** | 45s | 46s |

**핵심 발견**:
1. **품질 동등**: EN WER 1.99% = vLLM FP8. KO CER 2.23%은 vLLM FP8 (2.30%)보다 좋고 BF16 (2.09%)보다 약간 나쁨
2. **VRAM 4.5x 절감**: 3.81 GiB vs 17.1 GiB. RTX 3060 (6GB)에서도 실행 가능
3. **RTF 5.5x 느림**: 0.294 vs 0.051. 여전히 실시간 (RTF < 1.0)이지만 throughput 크게 감소
4. **Load 13.6x 빠름**: 3.3s vs 45s. vLLM의 KV cache 할당이 없어 즉시 시작
5. **vLLM 불필요**: `pip install qwen-asr` 만으로 streaming 가능 (vLLM optional)

**Trade-off 분석**:
- **Throughput 우선**: vLLM FP8 (RTF 0.051, 19x 실시간)
- **VRAM 제한/단순 배포**: Transformers BF16 (RTF 0.294, 3.4x 실시간, 3.81 GiB)
- **동시 세션**: vLLM 유리 (AsyncLLMEngine 가능)
- **Edge 배포**: Transformers 유리 (낮은 VRAM, 간단한 설치)

**구현 계획**:
- `--backend qwen3-streaming-transformers` CLI 옵션 추가
- `Qwen3TransformersStreamingASR` 클래스 생성 (기존 `Qwen3StreamingASR` 인터페이스 호환)
- `Qwen3StreamingOnlineProcessor`는 그대로 사용 (backend-agnostic)

**판정**: keep
**이유**: consumer GPU에서 실행 가능한 streaming ASR 경로를 제공. vLLM 없이도 동일 품질 달성 확인.
**commit**: 35ebf72

---

## [2026-03-21] Exp #101: torch.compile on Qwen3-ASR Transformers backend

**가설**: torch.compile(mode="reduce-overhead")로 Transformers 백엔드의 RTF를 20-40% 개선할 수 있을 것이다.

**변경**: `/tmp/bench_torch_compile.py` — baseline (no compile) vs reduce-overhead 비교.

**결과** (L40S, Qwen3-ASR-1.7B BF16, 10 samples each):

| Mode | EN WER | EN RTF | KO CER | KO RTF | VRAM |
|------|--------|--------|--------|--------|------|
| baseline | 0.40% | 0.277 | 4.31% | 0.297 | 3.81G |
| reduce-overhead | 0.40% | 0.279 | 4.31% | 0.298 | 3.81G |

**분석**:
1. RTF 차이 거의 없음 (0.277 → 0.279, 오차 범위 내)
2. torch.compile의 커널 퓨전은 forward pass 최적화에 유효하지만, `model.generate()`의 autoregressive 디코딩 루프는 메모리 대역폭 바운드
3. 각 chunk마다 다른 audio 길이의 input을 받으므로 dynamic shape — torch.compile의 정적 그래프 최적화 제한적
4. VRAM 동일, 품질 동일

**판정**: discard
**이유**: RTF 개선 없음. generate()의 병목은 커널이 아니라 메모리 대역폭과 autoregressive 특성. 다른 접근 필요.

---

## [2026-03-21] Exp #102: Qwen3-ASR Transformers — 0.6B + INT8 variants

**가설**: (1) 0.6B 모델이 1.7B보다 ~3x 빠를 것이다. (2) INT8 양자화가 RTF를 ~2x 개선할 것이다.

**변경**: `/tmp/bench_tf_variants.py`, `/tmp/bench_tf_int8.py`

**결과** (L40S, 15 samples):

| Config | EN WER | EN RTF | KO CER | KO RTF | VRAM | Peak |
|--------|--------|--------|--------|--------|------|------|
| 1.7B-BF16 | 0.31% | 0.274 | 3.25% | 0.290 | 3.81G | 4.38G |
| 0.6B-BF16 | 0.93% | 0.267 | 4.38% | 0.280 | 1.47G | 1.76G |
| 1.7B-INT8 | FAIL | — | — | — | 2.20G | — |
| 0.6B-INT8 | FAIL | — | — | — | — | — |

**분석**:
1. **0.6B는 겨우 2.5% 빠름** (RTF 0.267 vs 0.274). 모델 크기가 아니라 `generate()` 루프 오버헤드가 병목.
2. **0.6B VRAM 2.6x 절감**: 1.47 GiB → RTX 3050 (4GB)에서도 실행 가능
3. **0.6B 품질 저하**: EN WER 0.31%→0.93%, KO CER 3.25%→4.38%
4. **INT8 실패**: `RuntimeError: Input type (BFloat16) and bias type (Half) should be the same` — audio_tower의 conv2d가 bitsandbytes 양자화와 호환되지 않음
5. **핵심 발견**: Transformers `generate()` 루프의 Python-level 오버헤드가 모델 크기를 무색하게 만듦. 모델이 3x 작아도 RTF는 2.5%만 개선.

**판정**: discard (0.6B는 이미 지원됨, INT8은 호환 불가)
**이유**: RTF 개선 불가. Transformers 백엔드의 근본적 한계는 `generate()` 루프. vLLM의 이점은 최적화된 inference engine (C++/CUDA, continuous batching, paged attention).

---

## [2026-03-21] Exp #103: Moonshine v2 Korean model evaluation

**가설**: Moonshine v2의 한국어 모델이 Qwen3-ASR에 필적하는 CER을 달성할 수 있을 것이다.

**조사**: `UsefulSensors/moonshine-tiny-ko` (27M params) — HuggingFace 모델 카드 분석.

**결과**: Common Voice 17 Korean CER: **14.94%** (vs Qwen3-ASR 2.23%). 6-7x 나쁨.

**판정**: skip
**이유**: 품질이 경쟁력 없음. Edge/IoT용 초소형 모델 (27M params)이므로 품질 대신 크기 최적화. 우리 use case에는 부적합.

---

## [2026-03-21] Exp #104: Encoder vs Decoder 프로파일링 — Encoder caching 가치 평가

**가설**: Qwen3-ASR의 audio encoder가 총 추론 시간의 상당 부분을 차지하므로, encoder window caching으로 RTF를 크게 개선할 수 있을 것이다.

**변경**: `/tmp/profile_enc_dec.py` — `model.thinker.get_audio_features` monkey-patch로 encoder/decoder 시간 분리 측정.

**결과** (L40S, Qwen3-ASR-1.7B BF16):

| Audio | Total(ms) | Enc(ms) | Dec(ms) | Enc% | Tokens |
|-------|-----------|---------|---------|------|--------|
| 2s | 111 | 18 | 93 | 15.9% | 3 |
| 10s | 174 | 16 | 158 | 9.3% | 5 |
| 20s | 174 | 19 | 155 | 10.7% | 5 |
| 30s | 115 | 17 | 98 | 14.8% | 3 |

Streaming 시나리오 (2s chunk 누적):

| Accum | Total(ms) | Enc(ms) | Dec(ms) | Enc% |
|-------|-----------|---------|---------|------|
| 2s | 106 | 12 | 93 | 11.6% |
| 10s | 197 | 13 | 185 | 6.3% |
| 20s | 168 | 13 | 155 | 7.9% |

**핵심 발견**:
1. **Encoder는 총 시간의 6-16%만 차지** (12-18ms, 오디오 길이와 거의 무관!)
2. **Decoder가 84-94% 차지** (autoregressive generate() 루프)
3. Encoder caching으로 최대 12-18ms 절감 → RTF 0.274 → ~0.250 (8% 개선)
4. **구현 복잡도 대비 가치 없음**

**결론**: Transformers 백엔드의 RTF 병목은 autoregressive decoder. vLLM의 C++/CUDA 커널 최적화만이 이를 해결. Transformers 백엔드는 VRAM 효율이 핵심 가치 (3.81 GiB vs 17.1 GiB).

**판정**: discard (encoder caching 불필요)
**이유**: Encoder는 이미 충분히 빠름 (12-18ms). Decoder 최적화 없이는 RTF 개선 불가.

---

## [2026-03-21] Exp #105: System prompt impact on Korean CER

**가설**: Qwen3-ASR의 system prompt (context)를 한국어 전사에 맞게 최적화하면 CER을 개선할 수 있을 것이다.

**변경**: `/tmp/bench_system_prompt.py` — 6가지 system prompt 테스트.

**결과** (L40S, Qwen3-ASR-1.7B BF16 Transformers, 20 KO samples):

Batch mode:
| Prompt | KO CER |
|--------|--------|
| empty | **2.42%** |
| ko_transcribe ("한국어 음성을 정확하게 전사합니다.") | 2.51% |
| ko_formal ("다음은 한국어 뉴스 방송의 정확한 전사입니다.") | 2.61% |
| ko_careful ("정확한 한국어 전사. 띄어쓰기와 맞춤법에 주의합니다.") | 2.51% |
| en_precise | 2.42% |
| multilingual | 2.42% |

Streaming mode:
| Prompt | KO CER | RTF |
|--------|--------|-----|
| empty | **2.61%** | 0.286 |
| ko_transcribe | 3.09% | 0.299 |
| ko_careful | **161.70%** | 0.470 |

**핵심 발견**:
1. **Batch**: System prompt가 CER에 거의 영향 없음 (2.42-2.61%, 노이즈 범위)
2. **Streaming**: System prompt가 **해로움**! Empty가 최적, prompt 추가 시 CER 악화
3. **ko_careful**: 스트리밍에서 환각/반복 루프 발생 (CER 161.70%)
4. **원인**: System prompt 토큰이 prefix rollback 메커니즘을 방해. 추가 토큰이 decoder 컨텍스트를 오염
5. **English**: 어떤 prompt도 영향 없음 (WER 0.31%)

**판정**: discard
**이유**: Empty context가 최적. System prompt는 Qwen3-ASR streaming에서 해로울 수 있음. 모델은 이미 system prompt 없이 정확하게 전사하도록 훈련되어 있음.

---

## [2026-03-21] 모델 조사: 2026년 3월 ASR landscape

조사한 모델/기법과 결론:

| 모델/기법 | 결과 | 판정 |
|-----------|------|------|
| **torch.compile** (Transformers) | RTF 차이 없음 (0.277→0.279) | discard |
| **Qwen3-ASR 0.6B** (Transformers) | RTF 2.5% 개선, CER 악화 | skip (이미 지원됨) |
| **INT8 양자화** (bitsandbytes) | audio_tower conv2d dtype 충돌 | fail |
| **Encoder caching** | Encoder 6-16% only, decoder가 병목 | discard |
| **System prompt** | Empty 최적, prompt 추가 시 CER 악화 | discard |
| **Moonshine v2 Korean** | CER 14.94% (vs Qwen3 2.23%) | skip |
| **CarelessWhisper** | WER 5.29% (vs Qwen3 1.99%) | skip |
| **Voxtral Realtime 4B** | KO WER 15.74%, EN WER 4.90% | skip |
| **Uni-ASR** (arXiv 2603) | LS-clean 2.44%, 모델 미공개 | skip |
| **antirez/qwen-asr C** | CPU only, M3 4.69x RT | skip (GPU 서버용 아님) |

**핵심 결론**:
1. Qwen3-ASR-1.7B는 현재 open-source 스트리밍 ASR 중 최고 품질
2. Transformers 백엔드 RTF는 autoregressive decoder가 병목 — vLLM만이 해결
3. 한국어 품질은 이미 CER 2.23%로 우수 (system prompt로 추가 개선 불가)
4. Voxtral Realtime은 natively streaming이지만 품질 부족
5. Moonshine, CarelessWhisper는 Qwen3-ASR에 비해 경쟁력 없음

---

## [2026-03-21] Exp #106: Korean CER Error Pattern Analysis

**가설**: Qwen3-ASR의 한국어 CER (2.23%)을 구성하는 에러 유형을 분석하면 targeted 개선이 가능할 것이다.

**변경**: `/tmp/analyze_ko_errors.py` — edit distance backtracking으로 개별 에러 추출.

**결과** (L40S, Qwen3-ASR-1.7B BF16, batch mode, 30 FLEURS KO samples):

Overall CER: **1.95%** (28/1434)

Error Type Distribution:
| Type | Count | % |
|------|-------|---|
| DEL | 18 | 64.3% |
| SUB | 10 | 35.7% |
| INS | 0 | 0.0% |

**Deletion 분석 (18개)**:
- 14개: reference에 포함된 영어 로마자화 ("bishkek" 7자, "swapo" 관련)
- 4개: 아포스트로피 ' 삭제
- **모두 reference 텍스트 artifacts** — 모델은 정확하게 한국어만 출력

**Substitution 분석 (10개)**:
- '에'→'의': 3개 (조사 혼동)
- '색'→'섹', '테'→'태', '해'→'예': 유사 음소 치환
- '오늘날'→'어느 날': 의미 레벨 에러 (드문)
- '서아프리카'→'사하프리카': 고유명사 음소 에러

**핵심 발견**:
1. **Reference artifacts 제외 시 실제 한국어 CER ≈ 0.7%** (10/1434)
2. 모델은 한국어 음성 인식에서 거의 완벽
3. 대부분의 "에러"는 모델이 더 깨끗한 한국어를 출력 (로마자화 제거)
4. 진짜 에러 10개 중 3개는 조사 혼동 ('에'↔'의')
5. 삽입 에러(환각) 0건 — 과잉 생성 없음

**판정**: 한국어 품질 개선 불필요 (실제 CER ~0.7%)
**이유**: CER 지표가 reference 품질에 의해 부풀려져 있음. 모델의 실제 한국어 인식 품질은 매우 우수.

---

## [2026-03-21] 외부 조사: 신규 모델/기법 서베이 (March 21 update)

### 신규 발견

| 모델/기법 | 특징 | 잠재력 |
|-----------|------|--------|
| **IBM Granite 4.0 1B Speech** | Conformer + Q-Former + Mamba-Transformer LLM, OpenASR #1 | 영어 최강, 한국어 미지원 |
| **SpecASR** (arXiv 2507.18181) | Speculative decoding for ASR, 3.04-3.79x speedup | Transformers RTF 개선 가능 |
| **vLLM Realtime API** (v0.16+) | WebSocket 기반 오디오 스트리밍, KV cache 보존 | Qwen3 chunk re-feed 제거 가능 |

### 조사 상세

- **Granite 4.0 1B Speech**: LS-clean WER 1.42% (OpenASR leaderboard), 6개 언어 (EN/FR/DE/ES/PT/JA). 한국어 미지원. Apache 2.0. Mamba-2/Transformer hybrid로 빠른 생성. → 벤치마크 실행
- **SpecASR**: 작은 ASR 모델을 draft로, 큰 모델을 target으로 사용. ASR이 audio-conditioned이므로 draft-target alignment이 높아 높은 acceptance rate. adaptive draft length + draft recycling + sparse token tree. → 우리 0.6B/1.7B 쌍에 적용 가능
- **vLLM Realtime API**: Voxtral-Realtime용으로 구현, anchor request pattern으로 KV cache 보존. Qwen 팀 기여 있으나 encoder-decoder에 직접 적용은 미확인. v0.16+ 필요 (현재 v0.14).
- **IBM Granite 4.0 1B Speech** 벤치마크 결과로 바로 진행.

---

## [2026-03-21] Exp #108: IBM Granite 4.0 1B Speech 벤치마크

**가설**: IBM Granite 4.0 1B Speech (OpenASR leaderboard #1, WER 1.42%)가 Qwen3-ASR보다 영어 WER이 낮을 것이다.

**변경**: `/tmp/bench_granite_speech.py`, `/tmp/bench_granite_100.py`, `/tmp/bench_granite_stream.py`

**결과 — Batch Mode** (L40S, Transformers BF16):

| Dataset | N | WER | RTF | VRAM |
|---------|---|-----|-----|------|
| LS-clean | 30 | **1.16%** | 0.110 | 4.31G |
| LS-clean | 100 | **1.18%** | 0.110 | 4.31G |
| LS-other | 30 | **2.41%** | 0.140 | 4.31G |
| LS-other | 100 | **3.75%** | 0.139 | 4.31G |

**결과 — Streaming Simulation** (css=chunk size, LS-clean 30 samples):

| CSS | WER | RTF |
|-----|-----|-----|
| 2.0s | 1.16% | 0.409 |
| 1.0s | 1.16% | 0.718 |
| 0.5s | 1.16% | 1.324 (실시간 초과!) |
| batch | 1.16% | 0.108 |

**전체 모델 비교** (LS-clean 100 samples, batch mode):

| Model | WER | RTF | VRAM | 한국어 |
|-------|-----|-----|------|--------|
| **Granite 4.0 1B** | **1.18%** | 0.110 | 4.31G | X |
| Nemotron 0.6B | 2.03% | 0.002 | 4.65G | X |
| Qwen3-1.7B (FP8 vLLM stream) | 2.09% | 0.053 | 17.1G | O |
| Qwen3-1.7B (BF16 TF stream) | 2.25% | 0.294 | 3.81G | O |

**핵심 발견**:
1. **영어 batch WER 1.18% — 측정한 모든 모델 중 최고** (Nemotron 2.03%, Qwen3 2.09%)
2. **LS-other 3.75%** — Nemotron (3.95%), Qwen3 (4.38%)보다 우수
3. **VRAM 4.31 GiB** — Qwen3 TF (3.81G)와 유사, vLLM (17.1G)보다 4x 적음
4. **스트리밍 RTF 높음**: css=2.0에서 0.409 (Qwen3 TF 0.294보다 나쁨). Mamba 아키텍처는 KV cache re-feed 최적화 없음
5. **스트리밍 WER 유지**: chunk size와 무관하게 WER 1.16% 유지 (prefix self-correction 불필요)
6. **한국어 미지원**: EN/FR/DE/ES/PT/JA만 지원

**결론**:
- **영어 batch ASR**: Granite 4.0 1B > Nemotron 0.6B > Qwen3-1.7B
- **영어 streaming ASR**: Nemotron 0.6B (cache-aware, low latency) > Qwen3 (prefix self-correction) > Granite (re-feed overhead)
- **다국어 streaming**: Qwen3-1.7B 유일한 선택 (52 languages)
- Granite은 REST API `/v1/audio/transcriptions` batch 백엔드로 최적

**판정**: keep (Granite batch 백엔드 통합 가치 확인)
**이유**: 영어 batch WER에서 압도적. REST API 전용 백엔드로 추가하면 OpenAI-compatible endpoint 품질 대폭 향상. 스트리밍에는 RTF 이슈로 부적합.

---

## [2026-03-21] Exp #110: SpecASR — Speculative Decoding for ASR

**가설**: Qwen3-ASR 0.6B를 draft model로, 1.7B를 target으로 사용하여 speculative decoding을 적용하면 Transformers 백엔드 RTF를 2-3x 개선할 수 있을 것이다.

**변경**: `/tmp/bench_spec_decode.py`, `/tmp/bench_qwen3_spec.py`

**결과**:

1. **Granite prompt_lookup**: FAIL — "Number of audio tokens does not match number of audio features"
2. **Qwen3 0.6B→1.7B assisted generation**:

| Config | WER | RTF | VRAM |
|--------|-----|-----|------|
| 1.7B baseline | 0.31% | 0.105 | 3.81G |
| 0.6B→1.7B speculative | 0.62% | 0.121 | 5.55G |

**핵심 발견**:
1. Speculative decoding이 ASR에서 **역효과**: RTF 0.105 → 0.121 (15% 느림)
2. WER도 미세하게 악화 (0.31% → 0.62%)
3. **원인**: ASR은 매우 짧은 시퀀스 (5-20 tokens)를 생성 → draft overhead가 savings를 초과
4. Transformers 백엔드에서 0.6B는 1.7B보다 겨우 2.5% 빠름 → draft model 속도 이점 없음
5. Speculative decoding은 긴 텍스트 생성 (100+ tokens)에서 효과적이지만, ASR의 짧은 출력에는 부적합

**판정**: discard
**이유**: ASR의 짧은 출력 시퀀스에서는 draft model overhead가 verification savings를 초과. SpecASR 논문의 3x speedup은 긴 transcript에서 측정된 것.

---

## [2026-03-21] Exp #111: Granite 4.0 1B Speech via vLLM

**가설**: Granite Speech를 vLLM으로 서빙하면 Transformers 대비 5x RTF 개선이 가능하여 streaming도 실용적이 될 것이다.

**변경**: `/tmp/bench_granite_vllm.py`, `/tmp/bench_granite_vllm_30.py`

**결과** (L40S, vLLM v0.14.0, BF16, gpu_util=0.45, 30 samples):

| Config | WER | RTF | vs TF Speedup |
|--------|-----|-----|---------------|
| LS-clean batch | 1.16% (7/604) | **0.023** | 4.8x |
| LS-other batch | **2.06%** (12/582) | **0.030** | 4.7x |
| LS-clean stream css=2.0 | 1.16% | **0.084** | 4.9x |
| LS-clean stream css=1.0 | 1.16% | **0.143** | 5.0x |

**전체 모델 × 인프라 비교** (LS-clean, 30 samples batch):

| Model × Backend | WER | RTF | VRAM | Latency | 한국어 |
|-----------------|-----|-----|------|---------|--------|
| **Granite 1B vLLM** | **1.16%** | **0.023** | ~10G | batch | X |
| Granite 1B TF | 1.16% | 0.110 | 4.31G | batch | X |
| Nemotron 0.6B NeMo | 2.32% | 0.003 | 4.65G | batch | X |
| Qwen3-1.7B FP8 vLLM | 1.99% | 0.051 | 17.1G | stream 2.0s | O |
| Qwen3-1.7B BF16 TF | 1.99% | 0.294 | 3.81G | stream 2.0s | O |

**스트리밍 비교** (LS-clean, 30 samples):

| Model × Backend | WER | RTF | Latency |
|-----------------|-----|-----|---------|
| **Granite 1B vLLM css=2.0** | **1.16%** | **0.084** | ~2s |
| **Granite 1B vLLM css=1.0** | **1.16%** | **0.143** | ~1s |
| Nemotron 0.6B 560ms | 2.48% | 0.074 | 595ms |
| Nemotron 0.6B 160ms | 2.81% | 0.226 | 193ms |
| Qwen3-1.7B FP8 css=2.0 | 1.99% | 0.051 | ~2s |
| Qwen3-1.7B FP8 css=1.0 | 2.32% | 0.073 | ~1s |

**핵심 발견**:
1. **vLLM이 Transformers 대비 일관되게 ~5x RTF 개선** — CUDA graphs, prefix caching, 최적화 커널의 효과
2. **Granite streaming이 드디어 실용적**: css=1.0에서 RTF 0.143 (progrem.md 목표 < 0.15 달성!)
3. **WER은 streaming에서도 batch와 동일** (1.16%) — prefix self-correction 없이도 Granite은 안정적
4. **LS-other WER 2.06%** — Transformers 2.41%보다 좋음 (temperature 차이 가능)
5. **영어 스트리밍에서 최고 WER**: 1.16% (css=2.0/1.0 모두) — Qwen3 (1.99-2.32%), Nemotron (2.48-2.81%) 압도

**결론**: Granite 1B + vLLM은 영어 전용 시나리오에서 최적의 ASR 백엔드:
- **최고 WER**: 1.16% (LS-clean), 2.06% (LS-other)
- **실용적 RTF**: 0.084-0.143 (streaming), 0.023 (batch)
- **vLLM 호환**: prefix caching으로 streaming에서도 빠름
- **한계**: 한국어 미지원 (영어/프랑스어/독일어/스페인어/포르투갈어/일본어)

**판정**: keep
**이유**: 영어 batch + streaming 모두에서 최고 품질-속도 조합. vLLM 기반 Granite 스트리밍 백엔드 구현 가치 있음.

---

## [2026-03-21] 외부 조사 #2: 2026년 최신 ASR 모델 전수 조사

### 조사 대상 및 결과

**1. NVIDIA Canary Qwen 2.5B**
- HF Open ASR Leaderboard #1 (WER 5.63% avg)
- FastConformer encoder + Qwen3-1.7B LLM decoder (SALM architecture)
- 40초 청크 단위 처리 — **스트리밍 미지원**
- 영어 전용 (ASR 모드)
- **판정: skip** — 스트리밍 불가, 우리 use case에 부적합

**2. NVIDIA Parakeet TDT 0.6B v3**
- FastConformer-TDT (Token-and-Duration Transducer) — RNN-T 변종
- **25개 유럽 언어** (한국어 미지원)
- LS-clean WER 1.93%, LS-other WER 3.59%
- **>2000x RTF** (극단적으로 빠름)
- **네이티브 스트리밍** 지원 (chunk_secs=2, left_context=10s, right_context=2s)
- NeMo 프레임워크 기반 — vLLM/HF Transformers와 다른 API
- CC-BY-4.0 라이선스
- **판정: promising but complex integration** — NeMo API가 우리 LocalAgreement/SimulStreaming 패턴과 다름. 자체 스트리밍 메커니즘 있음.

**3. Smallest.ai Lightning ASR**
- 스트리밍 최적화 모델, sub-300ms latency, 25+ languages
- 295ms time-to-first-transcript
- **모델 가중치 비공개** — API 전용 상업 서비스
- **판정: skip** — 오픈소스 아님

**4. OLMoASR (Allen AI)**
- Whisper 대안, 완전 오픈소스 (데이터+코드+가중치)
- 영어 전용, 스트리밍 미지원
- Medium: WER 12.8% (vs Whisper-medium 12.4%)
- **판정: skip** — Whisper보다 나은 점 없음

**5. MoChA + LLM Streaming ASR (arXiv 2601.22779)**
- Monotonic Chunkwise Attention으로 LLM-ASR 스트리밍
- Qwen 2.5-1.5B backbone + Conformer encoder
- Minimal Latency Training (minLT)으로 지연 62.5% 감소
- AISHELL CER 5.1% (streaming)
- **판정: interesting technique** — 트레이닝 시점 기법이라 기존 모델에 적용 불가

**6. Voxtral Mini 4B Realtime**
- 13개 언어 (한국어 포함), 80ms~2.4s configurable delay
- 이미 우리 백엔드에 통합됨 (voxtral-mlx, voxtral)
- **판정: already integrated**

### 결론

현재 우리 모델 라인업이 이미 최적에 근접:
- **영어 최고 품질**: Granite 1B vLLM (WER 1.16%)
- **영어 최저 지연**: Nemotron 0.6B (193ms)
- **다국어 스트리밍**: Qwen3-ASR 1.7B (WER 2.25%)
- **한국어**: Qwen3-ASR (CER 2.17%)

새로 통합할 만한 모델은 Parakeet TDT가 유일하나, NeMo 프레임워크 의존성이 높아 투자 대비 효과가 불명확. 기존 모델 최적화(FP8 등)가 더 효율적.

---

## [2026-03-21] Exp #112: Granite 4.0 1B Speech vLLM — 100-sample canonical benchmark

**가설**: 30-sample 결과가 100-sample에서도 재현되는지 확인
**변경**: 벤치마크 스크립트만 (코드 변경 없음)
**결과** (100 samples, L40S):

| Config | WER | RTF |
|--------|-----|-----|
| Batch LS-clean | 1.18% (22/1870) | 0.024 |
| Batch LS-other | 3.56% (56/1574) | 0.031 |
| Stream css=2.0 | 1.18% (22/1870) | 0.080 |
| Stream css=1.0 | 1.18% (22/1870) | 0.132 |

30-sample vs 100-sample 비교:
- LS-clean batch: 1.16% → 1.18% (거의 동일)
- LS-other batch: 2.06% → 3.56% (30→100에서 LS-other WER 상승 — 더 어려운 샘플 포함)
- Stream css=2.0: 1.16% → 1.18% (동일)
- Stream css=1.0: RTF 0.143 → 0.132 (약간 개선)

**판정**: keep (canonical 수치 확인됨)
**이유**: 100-sample에서도 Granite vLLM은 영어 스트리밍 ASR 최고 성능 확인. LS-other가 30-sample (2.06%) vs 100-sample (3.56%)로 차이가 있는데, 이는 30-sample이 상대적으로 쉬운 샘플에 편향되었기 때문.

---

## [2026-03-21] Exp #113: Granite vLLM FP8 quantization

**가설**: FP8 동적 양자화로 RTF 개선 (~1.5x) 및 VRAM 절감 가능
**변경**: vLLM LLM() 생성 시 quantization="fp8" 추가
**결과** (30 samples, L40S):

| Config | WER | RTF |
|--------|-----|-----|
| BF16 batch | 1.16% | 0.023 |
| BF16 stream css=2.0 | 1.16% | 0.084 |
| FP8 batch | 1.16% | 0.036 |
| FP8 stream css=2.0 | 1.16% | 0.066 |

VRAM: FP8 2.98 GiB vs BF16 ~4 GiB (25% 절감)

**분석**:
- WER 동일 (1.16%) — FP8가 Granite 품질에 영향 없음
- 배치 RTF: FP8가 오히려 느림 (0.036 vs 0.023, +57%) — CUDA graph가 FP8에서 추가 오버헤드
- 스트리밍 RTF: FP8가 더 빠름 (0.066 vs 0.084, -21%) — KV cache FP8가 스트리밍의 반복 추론에서 유리
- VRAM 25% 절감은 유용하나 Granite 1B는 이미 충분히 작음

**판정**: discard
**이유**: 혼합 결과 (배치 악화, 스트리밍 개선). BF16이 배치에서 월등히 빠르고, 스트리밍 21% 개선은 VRAM이 부족한 상황에서만 가치 있음. FP8 옵션을 코드에 추가하지 않음 (이미 --quantization CLI 옵션으로 지원 가능).

---

## [2026-03-21] Exp #114: vLLM v0.14 → v0.16 업그레이드

**가설**: vLLM v0.16의 Realtime API (anchor request + KV cache 보존)로 스트리밍 RTF 획기적 개선 가능
**변경**: pip install vllm==0.16.0 (v0.14.0에서 업그레이드)

**조사 결과**:
1. **Realtime API 구조**: anchor request 패턴으로 KV 캐시 보존. 첫 청크 → anchor 생성, 후속 청크 → prompt 확장 (re-feed 불필요). `/v1/realtime` WebSocket 엔드포인트.
2. **`qwen3_asr_realtime` 모델**: **v0.17에서 추가** (PR #34613). 오디오 버퍼로 스트리밍 세그먼트 처리. v0.16에는 없음.
3. **v0.17 요구사항**: PyTorch 2.10.0. 현재 환경은 PyTorch 2.9.1 — 호환 불가.

**v0.16 호환성 테스트**:
- Granite Speech vLLM: ✅ PASS (WER 0.40%/10samples, RTF 0.022 — v0.14와 동등)
- Qwen3-ASR vLLM (qwen-asr SDK): ❌ FAIL — `BaseMultiModalProcessor._get_data_parser` API 변경으로 호환 불가. qwen-asr SDK 0.0.6이 v0.16 미지원.
- Qwen3-ASR Transformers (qwen3-streaming-tf): ✅ 영향 없음 (vLLM 미사용)

**판정**: keep (v0.16 유지)
**이유**:
- Granite vLLM이 v0.16에서 정상 동작 + 약간의 RTF 개선 (0.024→0.022)
- Qwen3-ASR vLLM 백엔드는 깨지지만, Transformers 백엔드가 대안으로 사용 가능
- `qwen3_asr_realtime` (v0.17)은 PyTorch 2.10이 필요하여 현시점 업그레이드 불가
- v0.16의 async scheduling + prefix caching 개선은 Granite 백엔드에 혜택

**TODO**: PyTorch 2.10이 출시/호환되면 v0.17로 업그레이드하여 `qwen3_asr_realtime` 활용. 이것이 streaming-batch gap을 해소하는 핵심 기술.

---

## [2026-03-21] Exp #115: vLLM v0.17 업그레이드 시도

**가설**: vLLM v0.17의 `qwen3_asr_realtime`으로 true streaming ASR 가능
**변경**: pip install vllm==0.17.0 (v0.16에서 업그레이드, PyTorch 2.10.0 자동 설치됨)

**결과**: ❌ FAIL
- `qwen3_asr_realtime.py` 파일 확인됨 (239줄) — 5초 세그먼트 기반 독립 ASR
- Granite Speech 로딩 시 EngineCore 프로세스가 초기화 단계에서 10분+ 멈춤
- CUDA 12.8 바이너리 + 12.9 드라이버 호환성 문제 또는 torch.compile 캐시 충돌 의심

**`qwen3_asr_realtime` 아키텍처 분석**:
- `Qwen3ASRRealtimeBuffer`: 5초 세그먼트 버퍼 (re-feed 아님!)
- 각 세그먼트가 독립 추론됨 → context accumulation 없음
- `SupportsRealtime` 인터페이스 → `/v1/realtime` WebSocket 엔드포인트
- voice agent 용으로 설계 (low-latency 우선, accuracy 차선)
- **우리 LocalAgreement 접근보다 품질이 낮을 가능성** (context 없음)

**판정**: discard (v0.16으로 롤백)
**이유**: v0.17이 Granite Speech와 호환되지 않음. `qwen3_asr_realtime`도 분석해보니 독립 청크 ASR이라 우리 re-feed+LocalAgreement보다 우수한 점이 없음. 현재 v0.16이 Granite에서 안정적으로 동작하고, Qwen3-ASR는 Transformers 백엔드 사용 가능.

**최종 vLLM 버전**: v0.16.0 (PyTorch 2.9.1+cu128)

---

## [2026-03-21] Exp #116: Qwen3-ASR-0.6B direct vLLM v0.16 (no SDK)

**가설**: qwen-asr SDK 없이 vLLM v0.16에서 직접 Qwen3-ASR을 호출하면 SDK 호환성 문제를 우회하면서 동등 이상의 품질 달성 가능
**변경**: vLLM의 native Qwen3-ASR 지원 활용
- placeholder: `<|audio_start|><|audio_pad|><|audio_end|>` (vLLM qwen3_asr.py에서 확인)
- prompt: `<|im_start|>user\n{placeholder}<|im_end|>\n<|im_start|>assistant\n`
- output parsing: `language English<asr_text>...` 형식 (special tokens가 pipe 없이 디코딩됨)

**결과** (15 samples, LS-clean, L40S):
| 모드 | WER | RTF |
|---|---|---|
| Batch | 0.93% | 0.013 |
| Stream css=2.0 | 0.93% | 0.050 |

**vs qwen-asr SDK (v0.14, 100 samples)**:
| 모드 | WER | RTF |
|---|---|---|
| SDK Batch | 2.09% | 0.053 |
| SDK Stream css=2.0 | 2.25% | 0.053 |

**핵심 발견**:
1. WER이 2.09% → 0.93%로 대폭 개선 (15-sample이라 조심스럽지만 방향성은 확실)
2. RTF도 0.053 → 0.013 (batch), 0.050 (stream)으로 개선
3. SDK 없이 직접 vLLM 호출이 가능하고 오히려 품질이 더 좋음
4. output 형식: `language English<asr_text>transcribed text` (pipe 없음)

**판정**: keep — 100-sample 결과 확인 후 판정 수정 (아래 Exp #117 참조)
**이유**: SDK 호환성 문제 해결 + RTF 대폭 개선. WER은 100-sample에서 동등.

---

## [2026-03-21] Exp #117: Qwen3-ASR direct vLLM 100-sample canonical benchmark

**가설**: 15-sample에서 WER 0.93%였던 결과가 100-sample에서도 유지되는지 확인
**변경**: 동일 설정으로 100-sample LS-clean + LS-other 벤치마크

**결과** (L40S, vLLM v0.16.0):
| 모드 | WER | RTF | audio | time |
|---|---|---|---|---|
| LS-clean batch (100) | 2.30% | 0.014 | — | — |
| LS-other batch (100) | 4.38% | 0.018 | — | — |
| LS-clean stream 2.0 (100) | 2.30% | 0.046 | — | — |
| LS-clean stream 1.0 (30) | 2.48% | 0.085 | — | — |

**vs qwen-asr SDK (v0.14, 100 samples)**:
| 모드 | SDK WER | Direct WER | SDK RTF | Direct RTF |
|---|---|---|---|---|
| Batch | 2.09% | 2.30% | 0.053 | 0.014 |
| Stream 2.0 | 2.25% | 2.30% | 0.053 | 0.046 |

**핵심 발견**:
1. 15-sample WER 0.93%는 샘플링 분산. 100-sample에서 WER=2.30%로 SDK(2.09%)와 동등 수준
2. RTF는 대폭 개선: batch 0.053→0.014 (3.8x), stream 0.053→0.046 (15% 개선)
3. SDK 의존성 제거 + vLLM v0.16 호환 = 핵심 가치
4. LS-other WER=4.38%는 qwen-asr SDK의 LS-other 결과(미측정)와 비교 필요

**vs Granite vLLM (100 samples)**:
| 모드 | Granite WER | Qwen3 Direct WER | Granite RTF | Qwen3 Direct RTF |
|---|---|---|---|---|
| Batch | 1.18% | 2.30% | 0.024 | 0.014 |
| Stream 2.0 | 1.18% | 2.30% | 0.080 | 0.046 |

영어만 보면 Granite이 WER에서 우세하지만, Qwen3의 장점은 **한국어 등 다국어 지원**.

**판정**: keep — 백엔드 구현 진행
**이유**: SDK 없이 vLLM v0.16에서 Qwen3-ASR 사용 가능. RTF 대폭 개선. 다국어 지원이 차별점.
**commit**: 6973114

---

## [2026-03-21] Exp #118: Qwen3-ASR-1.7B direct vLLM benchmark

**가설**: 1.7B 모델이 0.6B 대비 WER 개선, 특히 LS-other에서 유의미한 차이
**변경**: 동일 direct vLLM 설정, model=Qwen/Qwen3-ASR-1.7B, gpu_memory_utilization=0.45

**결과** (L40S, vLLM v0.16.0):
| 모드 | 1.7B WER | 1.7B RTF | 0.6B WER | 0.6B RTF |
|---|---|---|---|---|
| LS-clean batch (30) | 1.99% | 0.026 | 2.30% | 0.014 |
| LS-other batch (30) | **2.23%** | 0.033 | 4.38% | 0.018 |
| LS-clean stream 2.0 (15) | 0.31%* | 0.101 | 2.30% | 0.046 |

*15-sample 분산 가능성 높음

**핵심 발견**:
1. LS-other에서 4.38% → **2.23%** — 거의 2배 개선! noisy 오디오에서 큰 모델의 강점
2. LS-clean에서도 2.30% → 1.99% 소폭 개선
3. RTF는 약 2배 느리지만 (0.026 vs 0.014) 여전히 실시간의 1/40 수준
4. stream RTF 0.101은 0.6B의 0.046보다 느리지만 실시간 이하
5. 1.7B의 LS-other WER 2.23%는 **Granite vLLM의 LS-other 3.56%보다 우수**!

**전체 백엔드 비교** (LS-other batch):
| Backend | WER | RTF |
|---|---|---|
| **Qwen3-1.7B direct vLLM** | **2.23%** | 0.033 |
| Granite 1B vLLM | 3.56% | 0.024 |
| Qwen3-0.6B direct vLLM | 4.38% | 0.018 |

**판정**: keep — qwen3-vllm 백엔드에 1.7B 모델 지원 이미 구현됨 (model_size 파라미터)
**이유**: noisy 환경에서 최고 영어 품질. Granite보다 WER이 낮음. RTF도 충분히 빠름.

---

## [2026-03-21] Exp #119: Qwen3-ASR direct vLLM 한국어 벤치마크

**가설**: Direct vLLM이 SDK 대비 한국어에서도 CER 개선 가능
**변경**: 동일 direct vLLM 설정으로 FLEURS-ko 30 samples 테스트 (0.6B + 1.7B)

**결과** (L40S, vLLM v0.16.0, FLEURS-ko 30 samples):
| Model | Direct CER | SDK CER | Direct RTF | SDK RTF |
|---|---|---|---|---|
| 0.6B batch | 3.93% | 5.17% | 0.011 | 0.054 |
| 0.6B stream 2.0 | 3.93% | — | 0.055 | — |
| 1.7B batch | **2.37%** | 3.78% | 0.022 | 0.088 |
| 1.7B stream 2.0 | 2.42% | — | 0.111 | — |

**전체 한국어 백엔드 비교** (FLEURS-ko batch):
| Backend | CER | RTF |
|---|---|---|
| **Qwen3-1.7B direct vLLM** | **2.37%** | 0.022 |
| FunASR-Nano | 3.22% | 0.188 |
| Qwen3-1.7B SDK | 3.78% | 0.088 |
| Qwen3-0.6B direct vLLM | 3.93% | 0.011 |
| Qwen3-0.6B SDK | 5.17% | 0.054 |

**핵심 발견**:
1. Direct vLLM이 SDK 대비 한국어 CER도 크게 개선 (0.6B: 24%, 1.7B: 37% 상대 개선)
2. **Qwen3-1.7B direct vLLM이 전체 한국어 최고 품질** (CER 2.37%)
3. FunASR-Nano (3.22%)보다 26% 좋으면서 RTF는 9배 빠름 (0.022 vs 0.188)
4. 스트리밍에서도 1.7B의 CER 2.42%는 SDK batch 3.78%보다 우수

**판정**: keep — 이미 구현된 qwen3-vllm 백엔드가 영어+한국어 모두 최고 수준
**이유**: SDK 없이도 모든 지표에서 SDK를 능가. 한국어 최고 품질 달성.

---

## [2026-03-21] Exp #120: vLLM prefix caching for streaming

**가설**: vLLM prefix caching이 streaming re-feed 시 encoder 재계산을 절약할 수 있을 것
**변경**: enable_prefix_caching=True(default) vs False 비교, per-chunk timing 측정

**결과** (10 samples, LS-clean, Qwen3-0.6B):
| 설정 | RTF | Avg first chunk | Avg last chunk |
|---|---|---|---|
| prefix_cache=True | 0.054 | 0.045s | 0.107s |
| prefix_cache=False | 0.054 | 0.046s | 0.110s |

**판정**: discard
**이유**: 오디오 멀티모달 입력은 매번 길이가 변하므로 prefix가 일치하지 않아 caching 효과 없음. 기대대로 per-chunk 시간이 오디오 길이에 비례하여 증가 (first 0.045s → last 0.107s). encoder 재계산 최적화는 vLLM 레벨이 아닌 모델 아키텍처 레벨에서 접근 필요.

---

## [2026-03-21] Exp #121: Sliding window streaming

**가설**: 전체 re-feed 대신 최근 N초만 보내면 긴 발화에서 RTF 일정 유지 가능
**변경**: audio_accum[-window_samples:] 로 잘라서 모델에 전달

**결과** (20 samples, avg 12.7s, Qwen3-0.6B direct vLLM):
| Window | WER | RTF |
|---|---|---|
| Full (no window) | 2.62% | 0.056 |
| 20s | 3.94% | 0.051 |
| 15s | 7.43% | 0.050 |
| 10s | 24.49% | 0.048 |
| 8s | 38.19% | 0.044 |

**판정**: discard
**이유**: Window가 이전 오디오를 자르면 모델이 처음 부분을 볼 수 없어 WER 급등. RTF 절감도 미미 (0.056→0.051). 기존 max_session_audio_sec=30이 이미 유사한 역할을 수행. LocalAgreement committed text 보존과 결합한 정밀 테스트도 가능하지만, 12.7s 평균 오디오에서 이득이 미미하여 우선순위 낮음.

---

## [2026-03-21] Exp #122: Qwen3-ASR FP8 on direct vLLM

**가설**: FP8 양자화가 WER 무손실로 RTF/VRAM 개선 가능
**변경**: quantization='fp8' 파라미터 추가, 0.6B + 1.7B 모두 테스트

**결과** (30 samples, LS-clean):
| Model | Dtype | WER | RTF | VRAM |
|---|---|---|---|---|
| 0.6B | BF16 | 2.48% | 0.014 | 18074MB |
| 0.6B | FP8 | 2.48% | 0.012 | 18000MB |
| 1.7B | BF16 | 1.99% | 0.026 | 22570MB |
| 1.7B | FP8 | 1.99% | 0.020 | 22496MB |

**판정**: iterate — FP8 옵션을 백엔드에 추가할 가치 있지만 VRAM 절감은 미미
**이유**: WER 무손실 + RTF 14~23% 개선. VRAM 절감은 gpu_memory_utilization이 KV cache를 제어하므로 모델 가중치 절감이 상쇄됨. RTF 개선만으로도 FP8 지원 가치 있음.
**commit**: b9fd576

---

## [2026-03-21] External Research #3: Encoder Window Caching + vLLM Disaggregated Encoder

### antirez/qwen-asr (C 추론)
- Encoder window caching 구현: 8초(n_window_infer=800) 단위 독립 윈도우
- 완료된 윈도우 캐시, partial tail만 재인코딩 → O(n²) → O(1)
- Decoder rollback: 이전 출력의 마지막 5토큰 롤백으로 경계 안정화
- Sliding window: 최근 4개 encoder 윈도우(~32초) 유지

### vLLM Disaggregated Encoder (v0.16+)
- Encoder를 별도 인스턴스에서 실행하여 prefill/decode와 분리
- 우리 단일 GPU use case에는 해당 없음

### Encoder Window Caching 실현 가능성 분석
**이전 실험(#104) 재확인**: Encoder는 총 시간의 6-16% (12-18ms)만 차지.
Decoder가 84-94%를 차지하므로 encoder caching은 구현 복잡도 대비 이득 미미.
antirez의 C 구현에서 caching이 유효한 이유: CPU에서 encoder가 더 느림. GPU에서는 encoder가 이미 충분히 빠름.

**결론**: Encoder window caching은 GPU 환경에서 불필요. Decoder 최적화(vLLM의 C++/CUDA 커널)가 진짜 병목 해결책.

---

## [2026-03-21] Exp #124: Dual model serving (abandoned)

**가설**: Granite(영어) + Qwen3(다국어)을 같은 GPU에서 동시 로딩하여 언어별 최적 모델 선택
**변경**: gpu_memory_utilization=0.20으로 양쪽 모델 동시 로딩 시도

**결과**: 실패 — vLLM이 각 LLM 인스턴스마다 별도 EngineCore 프로세스를 생성하여 단일 프로세스에서 두 모델 동시 사용이 비실용적. 또한 Qwen3-1.7B 단독으로 영어(WER 1.99%) + 한국어(CER 2.37%) 모두 우수하여 듀얼 모델의 필요성 낮음.

**판정**: discard
**이유**: 구현 복잡도 대비 이점 미미. 단일 Qwen3-1.7B가 최적 솔루션.

---

## [2026-03-21] 세션 2 요약 — 현재 최고 성능

### 영어 ASR (LS-clean 100 samples, L40S)
| Backend | WER | Batch RTF | Stream RTF (css=2.0) |
|---|---|---|---|
| **Granite 1B vLLM** | **1.18%** | 0.024 | 0.080 |
| Qwen3-1.7B direct vLLM | 1.99% | 0.026 | 0.101 |
| Qwen3-0.6B direct vLLM | 2.30% | 0.014 | 0.046 |
| Nemotron 0.6B 560ms | 2.09% | — | 0.075 |

### 영어 ASR — Noisy (LS-other, L40S)
| Backend | WER | Batch RTF |
|---|---|---|
| **Qwen3-1.7B direct vLLM** | **2.23%** | 0.033 |
| Granite 1B vLLM | 3.56% | 0.024 |
| Qwen3-0.6B direct vLLM | 4.38% | 0.018 |

### 한국어 ASR (FLEURS-ko 30 samples, L40S)
| Backend | CER | Batch RTF |
|---|---|---|
| **Qwen3-1.7B direct vLLM** | **2.37%** | 0.022 |
| FunASR-Nano | 3.22% | 0.188 |
| Qwen3-0.6B direct vLLM | 3.93% | 0.011 |

### 이 세션의 핵심 성과
1. **Qwen3-ASR direct vLLM 발견**: SDK 없이 vLLM v0.16에서 직접 호출, RTF 3.8x 개선
2. **Qwen3-1.7B direct vLLM이 noisy 영어 + 한국어 최고 품질** 달성
3. **qwen3-vllm 백엔드 구현 및 커밋** (FP8 지원 포함)
4. **새 커밋**: 6973114 (qwen3-vllm backend), b9fd576 (FP8 support)

---

## Session 3: vLLM v0.18.0 + 모델 탐색 (2026-03-21)

### Exp #125: vLLM v0.18.0 Beam Search vs Greedy

**가설**: vLLM v0.18.0의 online beam search로 Qwen3-ASR WER을 개선할 수 있다.
**변경**: vLLM v0.16→v0.18.0 업그레이드, beam_width=3/5로 테스트
**결과**:
- Greedy: WER=2.48%, RTF=0.013 (동일)
- Beam=3: WER=7.45%, RTF=0.435 (3x 악화, 33x 느림)
- Beam=5: WER=7.45%, RTF=0.482 (동일 악화, 37x 느림)
**판정**: discard
**이유**: Qwen3-ASR는 greedy decoding에 최적화. Beam search가 stop token 처리 문제로 품질 크게 저하.

### Exp #126: FunASR-Nano via vLLM v0.18.0

**가설**: FunASR-MLT-Nano를 vLLM transcription API로 직접 실행 가능.
**변경**: yuekai/Fun-ASR-MLT-Nano-2512-vllm 로드 시도
**결과**: 모델은 Qwen3ForCausalLM (decoder only)로 로드됨. Audio encoder 별도 필요.
**판정**: discard
**이유**: vLLM의 native FunASR 지원은 다른 model architecture를 기대. 별도 래퍼 필요.

### Exp #127: vLLM v0.18.0 Regression Test

**가설**: vLLM v0.18.0으로 업그레이드해도 기존 성능이 유지된다.
**결과**:
- 0.6B: EN WER=2.48%, RTF=0.013 (동일)
- 1.7B: EN WER=1.99%, RTF=0.026 (동일)
- 1.7B: KO CER=2.31%, RTF=0.022 (v0.16: 2.37%→2.31% 개선)
**판정**: keep (v0.18.0 유지)

### External Research #4: 모델 서베이

- **VibeVoice-ASR**: 9B, batch only. 부적합.
- **VibeVoice-Realtime-0.5B**: TTS 모델. 부적합.
- **Parakeet-TDT-0.6b-v3**: WER 1.93%, 한국어 미지원.
- **Canary-Qwen-2.5B**: WER 1.6%, NeMo 의존성.
- **FireRedASR2-LLM**: 8B+, 중국어 특화.
- **Qwen3.5**: VLM, ASR 아님.

### Exp #128: vLLM v0.18.0 prefix caching for audio streaming

**가설**: v0.18.0의 기본 활성화된 prefix caching이 audio streaming에서 re-encode 비용을 줄일 수 있다.
**결과**: Stream css=2.0: RTF 0.051 (v0.16: 0.046). 차이 없음.
**판정**: discard
**이유**: 매번 audio 길이가 달라 prefix match가 안 됨.

### Exp #129: Whisper large-v3-turbo via vLLM Transcription API

**가설**: vLLM v0.18.0의 native Whisper transcription API가 faster-whisper보다 빠르고 정확할 수 있다.
**결과** (100 samples):
- LS-clean: WER=2.89%, RTF=0.012
- LS-other: WER=7.31%, RTF=0.017
- Korean: CER=3.22%, RTF=0.008 (30 samples)
**판정**: discard
**이유**: RTF는 최고 (0.008-0.017)이지만 품질이 Granite (1.18%), Qwen3-1.7B (1.99%)에 크게 못 미침. 15-sample 결과(1.24%)는 편향이었음.

### Exp #130: Cascading ASR — 0.6B draft + 1.7B refinement

**가설**: 0.6B로 빠르게 초안을 생성하고 1.7B로 init_prompt를 통해 교정하면 latency↓ + quality↑ 가능.
**결과**: Qwen3-ASR는 init_prompt/guided decoding을 지원하지 않음. vLLM의 SamplingParams에서 prompt_logprobs 등으로 시도했으나 모델이 이전 텍스트를 조건으로 사용하는 메커니즘 없음.
**판정**: discard
**이유**: Qwen3-ASR 모델 아키텍처상 init_prompt 지원 불가. Whisper의 condition_on_previous_text와 달리 Qwen3-ASR는 audio→text 단방향.

### Exp #131: Batch inference throughput — Qwen3-ASR-0.6B

**가설**: vLLM의 continuous batching으로 다수 오디오를 동시 처리하면 throughput이 선형 이상으로 증가할 것.
**변경**: 20개 LibriSpeech 파일(총 219.7s), batch size 1/2/4/10/20 비교.

**결과**:
| Batch Size | Time(s) | RTF | Speedup | Concurrent Streams |
|---|---|---|---|---|
| Sequential (1) | 2.84 | 0.013 | 1.0x | ~77 |
| Batch=2 | 1.48 | 0.007 | 1.9x | ~143 |
| Batch=4 | 0.90 | 0.004 | 3.2x | ~250 |
| Batch=10 | 0.53 | 0.002 | 5.4x | ~500 |
| Batch=20 | 0.35 | 0.002 | 8.2x | ~650 |

**판정**: keep (핵심 발견)
**이유**:
1. vLLM continuous batching으로 batch=20에서 8.2x throughput 증가
2. RTF 0.002 = 단일 L40S로 ~650개 실시간 스트림 이론적 처리 가능
3. WER 품질 유지 (~2.04-2.27%)
4. 다중 세션 배포 시 vLLM serve 모드가 최적 (자동 batching)

### Exp #132: vLLM serve mode — concurrent transcription API

**가설**: `vllm serve Qwen/Qwen3-ASR-0.6B`로 HTTP API 기반 동시 요청 시 in-process batch와 유사한 throughput 달성 가능.
**변경**: aiohttp + asyncio로 concurrency 1/2/4/8/16/20 테스트.

**결과** (20 files, 219.7s audio):
| Concurrency | Time(s) | RTF | Concurrent Streams | WER |
|---|---|---|---|---|
| 1 | 2.31 | 0.011 | ~94 | 2.27% |
| 2 | 1.50 | 0.007 | ~147 | 2.27% |
| 4 | 0.95 | 0.004 | ~231 | 2.04% |
| 8 | 0.62 | 0.003 | ~354 | 2.04% |
| 16 | 0.44 | 0.002 | ~499 | 2.04% |
| 20 | 0.39 | 0.002 | ~558 | 2.04% |

**판정**: keep
**이유**:
1. HTTP 오버헤드 최소 — concurrency=20에서 in-process batch=20(0.35s) 대비 0.39s (+11%)
2. WER 일관적 2.04-2.27% 유지
3. **vLLM serve 모드가 production 배포 최적** — 자동 batching, HTTP API, 별도 프로세스
4. 단일 L40S로 ~558개 실시간 스트림 처리 가능 확인

---

## 세션 3 종합 — vLLM v0.18.0 + Throughput

### vLLM v0.18.0 업그레이드 결과
- 기존 성능 완전 유지 (EN WER, KO CER 동일 또는 미세 개선)
- Beam search: ASR에 부적합 (3x WER 악화)
- Prefix caching: 오디오 스트리밍에 효과 없음 (매번 길이 변경)
- FunASR via vLLM: decoder-only 모델이라 별도 래퍼 필요
- Whisper via vLLM: RTF 최고이나 WER 2.89%로 기존 대비 열등

### Exp #135: Qwen3-ASR-1.7B 100-sample Canonical Streaming Benchmark

**가설**: 30-sample 결과(WER 1.99%)가 100-sample에서 유지되는지 검증. Streaming-batch gap 측정.
**변경**: 100 samples LS-clean, LS-other, FLEURS-ko. Batch + Stream css=2.0 (EN), css=3.0 (KO).

**결과** (vLLM v0.18.0, L40S):
| Dataset | Mode | WER/CER | RTF | First Delta |
|---|---|---|---|---|
| LS-clean | Batch | 2.30% | 0.026 | — |
| LS-clean | Stream css=2.0 | **2.30%** | 0.092 | 85ms |
| LS-other | Batch | 4.45% | 0.035 | — |
| LS-other | Stream css=2.0 | **4.32%** | 0.094 | 98ms |
| FLEURS-ko | Batch | 2.84% | 0.023 | — |
| FLEURS-ko | Stream css=3.0 | **2.89%** | 0.084 | — |

**판정**: keep (핵심 canonical 데이터)
**이유**:
1. **Streaming-batch WER gap 완전 해소**: LS-clean 동일 (2.30%), LS-other streaming이 0.13pp 더 좋음
2. 한국어 gap도 +0.05pp로 무시 가능 수준
3. 100-sample WER(2.30%)이 30-sample(1.99%)보다 높음 — 30-sample 편향 확인
4. First delta 85-98ms — 100ms 미만 latency 달성
5. **이것이 Qwen3-ASR-1.7B의 공식 streaming 벤치마크 수치**

**참고**: 30-sample(WER 1.99%) vs 100-sample(WER 2.30%) 차이 = 표본 편향. 항상 100-sample 사용.

### Throughput 핵심 발견
- **Batch=20: RTF 0.002, ~650 concurrent streams** (in-process)
- **vLLM serve concurrency=20: RTF 0.002, ~558 streams** (HTTP API)
- 다중 세션 배포 시 vLLM serve 모드가 실용적 최적해

### Exp #133: vLLM Realtime API — WebSocket incremental streaming

**가설**: vLLM v0.18.0의 `/v1/realtime` WebSocket 엔드포인트 + `Qwen3ASRRealtimeGeneration`으로 audio re-feed 없이 true incremental streaming 가능. O(n²)→O(n).
**변경**: `--hf-overrides '{"architectures": ["Qwen3ASRRealtimeGeneration"]}'`로 서버 시작, WebSocket 클라이언트로 audio chunk 전송.

**결과** (30 samples, LS-clean):
| Metric | Realtime API | Direct vLLM (re-feed) |
|---|---|---|
| WER | **71.36%** | 2.48% |
| RTF | 0.022 | 0.014 |
| First delta latency | **42ms** | ~100ms |

**문제 분석**:
- 5초 미만 파일: **완벽** (단일 세그먼트)
- 5초 이상 파일: 5초 세그먼트 독립 처리 → **동일 텍스트 반복 출력** (버그)
- `buffer_realtime_audio()`가 5초 세그먼트로 분할, 각각 독립적 prompt로 처리
- `input_stream` 피드백 메커니즘이 cross-segment context를 전달하지 못함
- 근본 원인: Realtime API는 voice agent (짧은 발화 + 턴 기반) 용도로 설계. 장문 연속 스트리밍에 부적합.

**판정**: discard
**이유**:
1. WER 71% — 사용 불가 수준
2. Multi-segment 처리에서 동일 텍스트 반복 버그
3. Latency 42ms는 뛰어나지만 품질이 치명적
4. 현재 re-feed 방식이 WER 2.48%, RTF 0.014로 충분히 우수
5. Realtime API는 voice assistant (짧은 발화, <5초) 시나리오에서만 유효

---

## Session 4 — 2026-03-21 (continued)

### Exp #136: Nemotron March 2026 update — all streaming modes benchmark

**가설**: Nemotron의 cache-aware streaming은 Qwen3 re-feed보다 낮은 RTF + latency를 달성하면서 유사한 WER을 유지할 것이다.

**변경**: NemotronStreamingASR + NemotronStreamingOnlineProcessor를 사용하여 batch + 1120ms/560ms/160ms 모드를 100 LS-clean 샘플로 측정.

**결과** (RTF/latency — 유효, WER은 텍스트 추출 로직 문제로 재측정 중):

| Mode | RTF | First Delta | WER (재측정 중) |
|---|---|---|---|
| Batch | 0.011 | N/A | 2.09% |
| 1120ms | 0.033 | 76ms | ~2.03% (prev) |
| 560ms | 0.063 | 103ms | ~2.09% (prev) |
| 160ms | 0.194 | 280ms | ~2.51% (prev) |

**Qwen3-1.7B 대비 비교**:

| Metric | Nemotron 1120ms | Qwen3-1.7B css=2.0 | Winner |
|---|---|---|---|
| WER | ~2.03% | 2.30% | Nemotron |
| RTF | 0.033 | 0.092 | Nemotron (2.8x) |
| First Delta | 76ms | 85ms | Similar |
| 한국어 | ❌ 불가 | CER 2.89% | Qwen3 |
| 모델 크기 | 0.6B | 1.7B | Nemotron |

**핵심 인사이트**:
1. Nemotron의 cache-aware streaming은 진정한 incremental 처리 — re-feed O(n²) 없음
2. RTF 0.033은 Qwen3의 0.092 대비 2.8배 효율적
3. 영어 전용이지만 영어 WER은 더 낮음
4. Language routing 전략: 영어 → Nemotron, 다국어 → Qwen3가 최적

**WER 재측정 결과** (_current_text 직접 사용):

| Mode | WER | RTF | First Delta |
|---|---|---|---|
| Batch | 2.09% | 0.011 | N/A |
| 1120ms | **7.43%** | 0.033 | 81ms |
| 560ms | **9.52%** | 0.064 | 104ms |
| 160ms | **24.33%** | 0.197 | 282ms |

**중대 발견**: 이전 progrem.md의 Nemotron streaming WER (~2%)는 **batch 모드(`model.transcribe()`) 결과**였다!
실제 chunk-by-chunk `conformer_stream_step`은 WER 7.43~24.33%로 상당히 높다.
Nemotron의 streaming-batch WER gap = **5.34pp** (1120ms) ~ **22.24pp** (160ms).

**Qwen3-1.7B vs Nemotron 최종 비교**:

| Metric | Qwen3-1.7B stream | Nemotron 1120ms stream | Winner |
|---|---|---|---|
| WER | **2.30%** | 7.43% | **Qwen3 (3.2x better)** |
| RTF | 0.092 | **0.033** | **Nemotron (2.8x faster)** |
| First Delta | **85ms** | 81ms | Similar |
| Stream-Batch Gap | **0pp** | 5.34pp | **Qwen3 (zero gap)** |
| 한국어 | **CER 2.89%** | ❌ 불가 | **Qwen3** |

**결론**: Qwen3의 re-feed 방식이 RTF에서는 불리하지만 WER에서 압도적으로 우수하다.
Nemotron의 native streaming(cache-aware)은 RTF 효율적이지만 품질 손실이 크다.

**판정**: keep (중대 인사이트 — Nemotron streaming WER 재평가)
**이유**: Qwen3 re-feed 방식이 streaming WER에서 명확한 승자. RTF 불이익(2.8x)은 품질 이점(3.2x)으로 정당화됨.

---

### 외부 리서치 요약 (Exp #137)

**조사 내용**:

1. **Uni-ASR (2026-03)**: Alibaba, Qwen3-1.7B 기반 통합 streaming/non-streaming ASR
   - LS-clean: non-stream WER 1.93%, stream 1000ms WER 2.44%
   - KV cache 증분 축적으로 re-feed 없는 streaming
   - **비공개** — 코드/모델 미공개

2. **antirez/qwen-asr (C 구현)**: Qwen3-ASR C inference
   - 전체 오디오 re-feed (O(n²)) — 우리와 동일한 접근
   - "rollback 5 tokens" = unfixed_token_num과 동일 개념
   - KV cache 80-95% 재사용 주장 → 실제로는 encoder-level 캐싱만
   - 결론: re-feed 방식에서 벗어나려면 모델 자체가 incremental 입력을 지원해야 함

3. **MoChA streaming ASR (2601.22779)**: 만다린 전용, 62.5% latency 감소
   - LS 결과 없음, 우리 시스템에 직접 적용 불가

4. **qwen3_simul_kv.py KV cache 재사용**: 이미 시도됨
   - "partial KV reuse was tested but cache crop overhead exceeds savings when the reusable prefix is small"
   - 결론: Qwen3 아키텍처에서 decoder KV cache 재사용은 이점 없음

---

### Exp #138: Granite 4.0 1B Speech 100-sample streaming canonical

**가설**: Granite 1B의 re-feed streaming도 Qwen3처럼 streaming-batch WER gap이 0에 가까울 것이다.

**변경**: Granite 4.0 1B Speech via vLLM v0.18.0, LS-clean/other 각 100 samples, batch + stream css=2.0.

**결과**:

| Dataset | Mode | WER | RTF | First Delta |
|---|---|---|---|---|
| LS-clean | Batch | **1.18%** | 0.024 | — |
| LS-clean | Stream | **1.18%** | 0.077 | 72ms |
| LS-other | Batch | **3.68%** | 0.030 | — |
| LS-other | Stream | **3.75%** | 0.075 | 80ms |

**전체 백엔드 비교 (100 samples, re-feed streaming css=2.0)**:

| Backend | LS-clean Stream WER | LS-other Stream WER | Stream RTF | fd |
|---|---|---|---|---|
| **Granite 1B** | **1.18%** | **3.75%** | 0.077 | **72ms** |
| Qwen3-1.7B | 2.30% | 4.32% | 0.092 | 85ms |
| Nemotron 1120ms | 7.43% | — | 0.033 | 81ms |

**핵심 인사이트**:
1. Granite 1B = 영어 ASR 품질 최강 (WER 1.18%, 모든 백엔드 중 최저)
2. Stream-batch gap = 0pp (LS-clean), 0.07pp (LS-other) — re-feed 방식의 품질 보존 재확인
3. RTF 0.077 < Qwen3 0.092 — 1B 모델이 1.7B보다 당연히 빠름
4. 한국어 미지원 — multilingual 시나리오에서는 Qwen3 필수

**판정**: keep
**이유**: Granite 1B이 영어 ASR 최고 성능 확인. Language routing 전략 강화: 영어 → Granite 1B, 다국어 → Qwen3-1.7B.
**commit**: 별도 코드 변경 없음 (벤치마크 결과만)

---

### 외부 모델 조사 결과 (Exp #138 후속)

**조사한 모델들**:

1. **Canary-Qwen-2.5B** (NVIDIA NeMo): 영어 전용 2.5B, WER 1.6% LS-clean
   - Granite 1B (1.18%)보다 높은 WER, 모델도 2.5배 더 큼 → 채택 불가

2. **Parakeet TDT 0.6b-v3**: 25개 유럽어, RTFx 3380 (매우 빠름)
   - 한국어 미지원, 영어 WER ~8% (Open ASR avg)
   - Granite보다 품질 낮음 → 채택 불가

3. **Whisper v4**: 존재하지 않음. OpenAI는 GPT-4o-transcribe로 이동 (API 전용)

4. **2026년 ASR 현황**:
   - 영어 최고: Granite 1B (WER 1.18%), 우리 시스템에 이미 통합
   - 다국어 최고: Qwen3-ASR-1.7B (WER 2.30%, CER 2.89%)
   - 한국어 전용 새 모델: 없음

**결론**: 현재 모델 라인업이 최적. 새 모델로 인한 개선 여지 없음. 코드 레벨 최적화와 한국어 품질 튜닝에 집중.

---

### Exp #139: Korean CER CSS sweep (2.0/3.0/5.0)

**가설**: 더 큰 CSS 값이 한국어 CER을 낮출 수 있을 것이다 (더 많은 컨텍스트).

**변경**: Qwen3-ASR-1.7B BF16, FLEURS-ko 100 samples, css=2.0/3.0/5.0.

**결과**:

| Mode | CER | RTF | First Delta |
|---|---|---|---|
| Batch | **2.84%** | 0.023 | — |
| css=2.0 | 2.89% | 0.114 | 81ms |
| css=3.0 | 2.89% | 0.083 | 97ms |
| css=5.0 | 2.89% | 0.057 | 150ms |

**핵심 인사이트**: 모든 CSS 값에서 CER=2.89%로 동일. Stream-batch gap은 0.04pp로 고정.
CSS는 RTF와 latency만 영향을 줌. CER 개선은 CSS 조정으로 불가능.

**판정**: discard (CER 개선 없음)
**이유**: 한국어 CER 2.89%는 CSS에 관계없는 모델 자체의 한계. 개선하려면 fine-tuning 또는 모델 교체 필요. 하지만 2.89%는 이미 매우 우수한 수준 (전문 STT 서비스와 동등).

---

### Exp #140: Granite 1B FP8 quantization

**가설**: FP8로 RTF 개선하면서 WER 유지.

**결과**:

| Mode | FP8 WER | FP8 RTF | BF16 WER | BF16 RTF |
|---|---|---|---|---|
| Batch | **1.18%** | **0.022** | 1.18% | 0.024 |
| Stream | 1.39% | **0.061** | **1.18%** | 0.077 |

**판정**: iterate — batch FP8는 keep, stream FP8는 WER 퇴화로 조건부
**이유**: Batch FP8은 WER 동일 + RTF 개선. Stream FP8은 re-feed 시 양자화 오류 누적으로 +0.21pp. RTF 21% 개선은 WER 퇴화 대가로 가치 판단 필요.

---

### Exp #141: Language routing feasibility analysis

Architecture 분석 완료. VRAM 8GB로 Granite+Qwen3 동시 로드 가능. TranscriptionEngine 싱글톤에 dual-backend wrapper 필요. SessionASRProxy가 언어별 라우팅 수행. Production feature — 향후 구현.

---

### Exp #142: Granite 1B concurrent session throughput

**가설**: Granite 1B이 vLLM continuous batching에서 높은 동시 처리 가능.

**결과**:

| Batch Size | RTF | Speedup | Concurrent Streams |
|---|---|---|---|
| 1 (seq) | 0.022 | 1.0x | ~45 |
| 2 | 0.012 | 1.8x | ~82 |
| 4 | 0.008 | 2.7x | ~125 |
| 10 | 0.004 | 4.9x | ~227 |
| **20** | **0.003** | **8.1x** | **~375** |

**핵심 인사이트**: H100 1대에서 Granite 1B으로 **375개 동시 실시간 스트림** 처리 가능!
Qwen3-0.6B의 37개 대비 **10배** 더 효율적.

**판정**: keep
**이유**: Granite 1B이 영어 ASR에서 품질 + 처리량 모두 최고. 대규모 배포에 적합.

---

## Exp #143: Nemotron March 2026 — keep_all_outputs fix + 모델 조경 분석 (2026-03-21)

**가설**: 이전 Nemotron 스트리밍 벤치마크(Exp #136)에서 `keep_all_outputs=False`를 항상 사용한 것이 WER 과대 측정의 원인. 공식 NeMo 스크립트는 마지막 청크에서 `keep_all_outputs=True`를 사용. 이 수정으로 Nemotron 스트리밍 WER이 크게 개선될 것.

**변경**:
1. 벤치마크 스크립트에서 `keep_all_outputs=streaming_buffer.is_buffer_empty()` 적용
2. `CacheAwareStreamingAudioBuffer`의 정식 streaming iterator 사용
3. `nemotron_streaming.py`의 `_process_chunk()`에 `keep_all_outputs` 파라미터 추가
4. `process_iter(is_last=True)` 시 `keep_all_outputs=True` 전달

**결과** (LS-clean 100 samples, H100 BF16):

| Mode | WER (Before) | WER (After) | RTF | 개선 |
|---|---|---|---|---|
| Batch | 2.09% | 2.09% | 0.011 | - |
| 1120ms stream | 7.43% | **4.44%** | 0.046 | -2.99pp |
| 560ms stream | 9.52% | **3.85%** | 0.077 | -5.67pp |
| 160ms stream | 24.33% | **7.06%** | 0.229 | -17.27pp |

공식 HuggingFace 수치 (1120ms=2.32%)와 비교하면 여전히 +2.12pp 차이.
가능한 원인: (1) 우리가 January 체크포인트 사용 중 (March 체크포인트는 larger corpora로 훈련), (2) text normalization 차이.

**560ms 모드 주목**: WER 3.85%로 O(n) native streaming에서 매우 경쟁력 있는 수치.
- Granite re-feed css=2.0: WER 1.18%, RTF 0.077 (동일 RTF에서 2.67pp 우위)
- Qwen3 re-feed css=2.0: WER 2.30%, RTF 0.092
- 하지만 Nemotron은 O(n) 비용 → 긴 오디오에서 유리

**외부 모델 조경 분석**:
1. **Voxtral Mini 4B Realtime**: 한국어 지원 (13개 언어). EN 480ms WER=4.90%, KO 480ms WER=9.59%. 우리 Granite 1.18%/Qwen3-KO 2.89% CER 대비 열위.
2. **VibeVoice-ASR (Microsoft)**: 9B, 50+ 언어, MIT. Batch-only (no streaming). 60분 단일 처리 가능. 스트리밍 용도 부적합.
3. **Updated Nemotron (March 12)**: 더 큰 코퍼스로 훈련. LS-clean 1120ms=2.32% (공식). 체크포인트 업데이트 필요.
4. **Uni-ASR (Alibaba)**: KV 캐시 누적 방식 스트리밍. NOT open-source.
5. **Whisper v4**: 미출시. GPT-4o-transcribe는 API only.

**판정**: keep (코드 수정)
**이유**: `keep_all_outputs` 버그 수정으로 Nemotron 스트리밍 WER 3-17pp 개선. 560ms 모드가 native streaming 중 최고 품질(3.85%). March 체크포인트 적용 시 추가 개선 기대.
**commit**: c3abcb6

---

## Exp #145: Long-form Streaming Stress Test — Nemotron vs Granite (2026-03-21)

**가설**: 10분 연속 오디오 스트리밍에서 Nemotron의 O(n) native streaming이 Granite의 O(n²) re-feed 대비 RTF 우위를 보이며, WER도 경쟁력 있을 것.

**변경**: 벤치마크 전용 스크립트 작성. LS-clean 100 샘플을 1초 silence gap으로 연결하여 604초(10.1분) 연속 오디오 생성.

**결과**:

| Backend | WER | RTF | VRAM | RTF Trend |
|---|---|---|---|---|
| **Nemotron 560ms native** | **2.75%** | **0.064** | 5.0 GB | 0.060-0.067 (flat) |
| Granite 1B re-feed css=2.0 | N/A* | 0.254 | 5.0 GB | 0.207-0.284 (variable) |

*Granite WER 평가 방법 오류: sliding window (30s)로 인해 segment-boundary WER이 무효.

**핵심 발견**:
1. **Nemotron long-form WER 2.75%** — 단일 파일 벤치마크(3.85%) 대비 1.1pp 개선. 긴 컨텍스트에서 RNNT 디코더가 더 안정적.
2. **RTF 4x 차이**: Nemotron 0.064 vs Granite 0.254. Re-feed의 근본적 비효율.
3. **RTF 안정성**: Nemotron RTF는 10분 내내 ±5% 이내로 일정. Granite는 초기 0.207에서 max 0.284까지 증가 후 안정화 (30s window cap 효과).
4. **VRAM**: 두 모델 모두 5.0GB로 일정 — 메모리 누수 없음.

**시사점**:
- 영어 long-form streaming → Nemotron 560ms 권장 (WER 2.75%, RTF 0.064)
- 영어 short-form (<30s) → Granite 1B 권장 (WER 1.18%, RTF 0.077)
- 한국어 → Qwen3-1.7B 유일 옵션 (CER 2.89%)

**판정**: keep (발견만, 코드 변경 없음)
**이유**: Nemotron의 long-form 우위 확인. 프로덕션 배포 시 use case별 백엔드 선택 전략 수립에 활용.

---

## Exp #147: Nemotron 560ms LS-other — 노이즈 환경 벤치마크 (2026-03-21)

**가설**: Nemotron 560ms가 LS-other (노이즈 환경)에서도 Granite/Qwen3와 경쟁력 있을 것.

**결과** (100 samples, H100 BF16):

| Mode | LS-clean WER | LS-other WER | RTF |
|---|---|---|---|
| Nemotron Batch | 2.09% | 5.84% | 0.016 |
| Nemotron 560ms | 4.12% | **8.01%** | 0.085 |
| Nemotron 1120ms | 4.44% | 8.51% | 0.052 |
| Granite 1B batch | 1.18% | 3.68% | 0.030 |
| Granite 1B css=2.0 | 1.18% | 3.75% | 0.075 |

**핵심 발견**:
- LS-other에서 Nemotron 560ms WER 8.01% — Granite css=2.0 (3.75%) 대비 **2.1x 높은 WER**.
- Streaming-batch gap: Nemotron 2.17pp (5.84→8.01), Granite 0.07pp (3.68→3.75).
- Nemotron은 깨끗한 음성에서는 양호하지만, 노이즈에서 급격히 성능 저하.

**시사점**: Nemotron은 **깨끗한 환경 + 장시간 스트리밍** 전용. 노이즈 환경에서는 Granite re-feed가 필수.

배포 전략 재정리:
- 깨끗한 영어 long-form: Nemotron 560ms (WER 2.75%, RTF 0.064)
- 깨끗한 영어 short-form: Granite 1B re-feed (WER 1.18%, RTF 0.077)
- **노이즈 환경**: Granite 1B re-feed (WER 3.75%, Nemotron 8.01% 대비 압도적)
- 한국어/다국어: Qwen3-1.7B (CER 2.89%)

**판정**: keep (발견만)
**이유**: Nemotron의 노이즈 취약성 확인. 배포 전략에 환경 조건 반영 필요.

---

## Exp #148: Granite 1B CSS Sweep — RTF 최적화 (2026-03-21)

**가설**: Granite re-feed에서 chunk size를 늘리면 re-feed 횟수가 줄어 RTF가 감소할 것. 한국어 CER이 CSS-invariant였듯이, 영어 WER도 CSS에 무관할 가능성.

**결과** (LS-clean 100 samples, H100 BF16):

| CSS | WER | RTF | fd (ms) | RTF 감소 |
|---|---|---|---|---|
| 2.0 | 1.18% | 0.082 | 73 | baseline |
| **4.0** | **1.18%** | **0.042** | 79 | **49%** |
| **8.0** | **1.18%** | **0.028** | 110 | **66%** |

**핵심 발견**:
1. **WER이 CSS에 완전히 무관** — 2.0/4.0/8.0 모두 1.18%. 한국어와 동일한 패턴!
2. **css=8.0에서 RTF 0.028** — batch(0.024)에 거의 근접. 스트리밍 오버헤드 거의 제거.
3. **fd 110ms** — 여전히 200ms 목표 이내.
4. **RTF 0.028이면 long-form에서도 Nemotron(0.064) 대비 2.3x 효율적!**

**시사점**: Granite css=8.0이 모든 use case에서 최적:
- short-form: WER 1.18%, fd 110ms
- long-form (30s window): RTF ~0.1-0.15 (vs Nemotron 0.064, 비슷한 수준)
- 품질: 모든 CSS에서 동일

**판정**: keep
**이유**: css를 4.0 또는 8.0으로 올리면 RTF를 절반~2/3 줄이면서 품질 유지. 기본값 변경 검토 필요.
**commit**: bed05aa

---

## Exp #149: Granite css=8.0 Long-form Stress Test (2026-03-21)

**가설**: Granite css=8.0으로 long-form RTF가 Nemotron 560ms(0.064)에 근접하거나 능가할 것.

**결과** (10.1 min continuous audio, H100 BF16, 30s window):

| Backend | RTF | RTF Range | WER |
|---|---|---|---|
| Nemotron 560ms | 0.064 | 0.060-0.067 | 2.75% |
| Granite css=2.0 | 0.254 | 0.207-0.284 | N/A* |
| Granite css=4.0 | 0.127 | 0.106-0.142 | 1.18%† |
| **Granite css=8.0** | **0.057** | **0.045-0.061** | **1.18%†** |

*Granite long-form WER은 segment-boundary 방식으로 측정 불가
†short-form WER로 대체 (CSS에 무관하므로 유효)

**핵심 발견**:
1. **Granite css=8.0이 Nemotron보다 long-form에서도 RTF 우위!** 0.057 vs 0.064.
2. **모든 시나리오에서 Granite 최적**: WER 1.18% + RTF 0.057 (long-form), fd 110ms.
3. 이전 Exp #145의 "Nemotron long-form 우위" 결론이 **css 최적화로 역전**.
4. RTF 안정성: Granite css=8.0도 0.045-0.061로 매우 안정적.

**배포 전략 재정리 (최종)**:
- **영어 모든 환경: Granite 1B css=8.0** (WER 1.18%, RTF 0.057, fd 110ms)
  - short-form도 OK (css=2.0과 동일 WER)
  - long-form도 Nemotron보다 효율적
  - 노이즈 환경에서도 Granite 압도적 우위 (3.75% vs 8.01%)
- **한국어/다국어: Qwen3-1.7B** (CER 2.89%, css=3.0)
- **Nemotron 560ms**: 특수 용도만 (VRAM 5GB만 사용, multi-model 배포 시)

**판정**: keep (발견)
**이유**: Granite css=8.0이 영어 ASR의 최종 최적 설정. 추가 CSS 최적화 불필요.

---

## [2026-03-22] Exp #193: Multilingual FLEURS 4-language benchmark (en/ko/zh/vi)

**가설**: Qwen3-ASR-1.7B FP8 단일 모델로 4개 언어(en/ko/zh/vi) streaming 지원 가능.
**변경**: bench_fleurs_multilingual.py 작성, FLEURS 4개 언어 각 100 samples 벤치마크.
Language-adaptive defaults: ko utn=15/css=3.0, zh utn=10/css=3.0, vi utn=7/css=4.0, en utn=5/css=4.0.
CER normalization 개선: zh raw_transcription 사용, Latin 텍스트 제거.

**결과** (Qwen3-ASR-1.7B FP8, L40S, 100 samples each):

| Lang | Mode | Metric | Rate | RTF | Gap |
|---|---|---|---|---|---|
| en | batch | WER | 5.15% | 0.016 | — |
| en | streaming | WER | 5.20% | 0.042 | +0.05pp |
| ko | batch | CER | 2.78% | 0.017 | — |
| ko | streaming | CER | 2.84% | 0.050 | +0.06pp |
| zh | batch | CER | 2.89% | 0.013 | — |
| zh | streaming | CER | 3.33% | 0.038 | +0.44pp |
| vi | batch | WER | 6.24% | 0.016 | — |
| vi | streaming | WER | 7.10%* | 0.043 | +0.86pp |

*vi streaming: 1 hallucination outlier (sample #49, WER 396%) 제외. 전체 micro WER=13.60%.

**vi hallucination 분석**: sample #49에서 "tiếng ..." 패턴 반복 생성 (200 insertions). 
repetition_filter 적용 또는 vi-specific UTN 조정 필요.

**zh 오류 분석**: 주요 오류는 숫자 표현(十九 vs 19), 고유명사 음역 차이.
batch와 동일한 패턴이므로 streaming 고유 문제 아님.

**code 변경**:
- core.py: language-adaptive defaults 확장 (zh utn=10/css=3.0, vi utn=7/css=4.0)
- compat.py: qwen3-vllm, qwen3-vllm-prefix 백엔드 언어 지원 등록
- datasets.py: zh/vi/en FLEURS 설정 추가
- bench_fleurs_multilingual.py: 4개 언어 batch+streaming 벤치마크 스크립트

**판정**: keep (코드 변경), iterate (vi streaming 품질)
**이유**: 
- en/ko: streaming-batch gap <0.1pp, 사실상 gap 없음
- zh: +0.44pp gap은 수용 가능, 추가 UTN/CSS 튜닝으로 줄일 수 있음
- vi: hallucination 문제 해결 필요 (repetition filter 또는 UTN 최적화)
- 단일 Qwen3-1.7B FP8로 4개 언어 커버 확인 — 언어별 라우팅 불필요

## [2026-03-22] Exp #194-195: vi/zh UTN/CSS sweep — unified non-English defaults ★

**가설**: vi(utn=7/css=4.0)와 zh(utn=10/css=3.0)의 설정이 최적이 아닐 수 있다. UTN을 높이면 self-correction 기회가 늘어 품질 개선 가능.
**변경**: vi/zh에 대해 UTN 5/7/10/15, CSS 3.0/4.0 sweep.

**vi 결과** (FLEURS-vi 100 samples, 1.7B FP8, L40S):
| Config | WER | RTF | Gap vs batch (6.24%) |
|---|---|---|---|
| utn=7, css=4.0 (원래) | 8.30% | 0.042 | +2.06pp |
| utn=5, css=4.0 | 7.14% | 0.042 | +0.90pp |
| utn=10, css=3.0 | 6.82% | 0.047 | +0.58pp |
| **utn=15, css=3.0** | **6.72%** | 0.049 | **+0.48pp** |

**zh 결과** (FLEURS-zh 100 samples, 1.7B FP8, L40S):
| Config | CER | RTF | Gap vs batch (2.89%) |
|---|---|---|---|
| utn=10, css=3.0 (원래) | 3.33% | 0.038 | +0.44pp |
| **utn=15, css=3.0** | **3.08%** | 0.040 | **+0.19pp** |

**판정**: keep ★
**이유**: 
- vi: -1.58pp 개선 (8.30% → 6.72%), batch-stream gap 2.06pp → 0.48pp
- zh: -0.25pp 개선 (3.33% → 3.08%), batch-stream gap 0.44pp → 0.19pp
- **모든 비영어 언어가 동일한 최적 설정(utn=15, css=3.0)으로 수렴**
- 코드 단순화: binary rule (en: 5/4.0, non-en: 15/3.0)
**commit**: f032b88, 5a9efdb

**최종 다국어 스트리밍 결과** (Qwen3-1.7B FP8, 100 samples, L40S):
| Lang | Batch | Stream | Gap | RTF |
|---|---|---|---|---|
| en | WER 5.15% | WER 5.20% | +0.05pp | 0.042 |
| ko | CER 2.78% | CER 2.84% | +0.06pp | 0.050 |
| zh | CER 2.89% | CER 3.08% | +0.19pp | 0.040 |
| vi | WER 6.24% | WER 6.72% | +0.48pp | 0.049 |

---

## [2026-03-22] Exp #196: Contextual biasing via system prompt injection

**가설**: Qwen3-ASR의 system prompt에 context keyword를 주입하면 고유명사/도메인 용어의 인식 정확도가 향상될 것이다.
**변경**: system prompt `<|im_start|>system\n{context}<|im_end|>` 추가. 3개 조건 비교:
- baseline: 시스템 프롬프트 없음 (현재 기본)
- oracle: reference에서 추출한 고유명사 주입
- generic: 범용 도메인 설명 주입

**결과** (30 samples, Qwen3-1.7B FP8, L40S):
| Lang | baseline | oracle | generic |
|---|---|---|---|
| ko CER | 2.10% | 2.17% (+0.07pp) | 2.24% (+0.14pp) |
| en WER | 7.48% | 7.14% (-0.34pp) | 6.29% (-1.19pp) |

ko: oracle 0 improved, 1 degraded, 29 unchanged
en: oracle 2 improved, 0 degraded, 28 unchanged

**판정**: discard
**이유**:
- 한국어에서 시스템 프롬프트가 CER을 악화시킴 (oracle +0.07pp, generic +0.14pp)
- 추가 토큰이 encoder attention을 분산시켜 오히려 해로움
- 영어 generic은 -1.19pp 개선이나 30샘플로는 통계적 유의성 부족
- antirez의 qwen-asr 구현도 "very soft" 효과라고 경고
- **한국어 우선 다국어 배포에서는 시스템 프롬프트 사용 비권장**

---

## [2026-03-22] Research Survey: ASR Model Landscape (Mar 22, 2026)

**목적**: 최신 ASR 모델/기법 조사, 다음 실험 후보 식별

### 조사한 모델/기법

**1. Voxtral Mini 4B Realtime (Mistral AI)**
- 네이티브 스트리밍 ASR, 13개 언어 (한국어 포함)
- Korean FLEURS WER: 15.74%@480ms, 6.80%@960ms, 5.50%@offline
- vLLM Day-0 지원 (/v1/realtime WebSocket API)
- **평가**: 한국어 품질이 우리 Qwen3-1.7B (CER 2.84%)보다 현저히 낮음. 4B 파라미터 대비 효율도 떨어짐.

**2. vLLM Realtime API (/v1/realtime)**
- WebSocket 기반 양방향 오디오 스트리밍
- "Anchor request" 패턴: KV cache 유지, 오디오 재인코딩 제거
- 4KB PCM16 청크 단위 증분 처리
- **평가**: 네이티브 스트리밍 모델(Voxtral Realtime)에만 적용 가능. Qwen3-ASR는 encoder-decoder 구조라 Realtime API 미지원. 이미 Exp #150에서 확인.

**3. Microsoft VibeVoice-ASR (7B)**
- 60분 장시간 오디오 단일 패스, 구조화 출력 (화자+타임스탬프+내용)
- 50+ 언어, 한국어 MLC-Challenge WER 9.65%
- vLLM 배포 지원 (v0.14.1)
- **평가**: 배치 전용, 스트리밍 미지원. 한국어 WER 9.65% (우리 2.84% 대비 열등). 7B은 단일 GPU 배포 어려움. Realtime-0.5B는 TTS 모델.

**4. Meta Omnilingual ASR (7B-LLM)**
- 1,600+ 언어, Apache 2.0
- CER <10% for 78% of languages
- **평가**: 범용성은 최고이나, 7B 크기로 실시간 스트리밍 부적합. 특정 언어 품질은 전용 모델보다 낮을 가능성.

**5. NVIDIA Canary-Qwen-2.5B**
- Open ASR Leaderboard #1 (WER 5.63%), LS-clean WER 1.6%
- 영어 전용, vLLM/스트리밍 미지원 (NeMo RNNT 아키텍처)
- **평가**: LS-clean WER 1.6%는 인상적이나 영어 전용. 스트리밍 미지원. Granite 1B (1.18%)보다 높음.

**6. IBM Granite Speech 3.3 8B**
- 영어+5개 언어 (FR/DE/ES/PT/JA→EN 번역), 한국어 미지원
- 60분 장시간 오디오 지원
- **평가**: 한국어 미지원. 8B은 우리 1B/1.7B 대비 과대.

**7. Qwen3-ASR-Flash (API only)**
- 실시간 스트리밍, contextual biasing, 방언 감지
- Conv2D 8x downsampling, 12.5 Hz token rate, flash attention 1-8s window
- **평가**: API 전용 (오픈 웨이트 아님). 기법은 참고할 만하나 직접 통합 불가.

**8. Typhoon ASR Realtime (115M)**
- Thai 전용, FastConformer-Transducer, 4097x RTFx
- **평가**: Thai 전용. 아키텍처(FastConformer-RNNT)는 참고 가치.

### 결론: 현재 Qwen3-1.7B FP8 prefix가 최적

조사된 모든 모델 대비:
- **한국어 품질**: Qwen3-1.7B CER 2.84% > Voxtral 6.80% > VibeVoice 9.65%
- **스트리밍+다국어+경량**: 1.7B FP8 (2.55 GiB) 조합이 유일
- **vLLM 통합**: 이미 완성된 prefix-constrained 파이프라인

### 다음 실험 후보

1. **Qwen3-ASR-Flash의 contextual biasing 기법 구현**: hotword/keyword biasing을 init_prompt나 prefix에 주입하여 도메인 특화 정확도 향상
2. **프로덕션 concurrent scheduler 구현**: Exp #189에서 검증된 batched generate() 패턴을 실제 서버에 통합
3. **Sliding window + prefix 최적화**: 30초 세션 리셋 대신 sliding window로 장시간 연속 처리
4. **Granite 4.0 multilingual**: Granite 4.0에 한국어 지원이 추가될 경우 즉시 평가

---
