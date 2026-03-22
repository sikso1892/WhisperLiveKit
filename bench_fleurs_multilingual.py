#!/usr/bin/env python3
"""Multilingual FLEURS benchmark for Qwen3-ASR prefix-constrained streaming.

Benchmarks 4 languages: en, ko, zh, vi using FLEURS test sets.
Uses direct vLLM inference (qwen3-vllm-prefix backend pattern).

Usage:
    # All 4 languages on GPU 0
    CUDA_VISIBLE_DEVICES=0 python bench_fleurs_multilingual.py --model 1.7b --n-samples 100

    # Single language
    CUDA_VISIBLE_DEVICES=0 python bench_fleurs_multilingual.py --model 1.7b --lang ko --n-samples 100

    # FP8 quantization
    CUDA_VISIBLE_DEVICES=0 python bench_fleurs_multilingual.py --model 1.7b --quantization fp8 --n-samples 100

    # Batch mode (no streaming, just vLLM generate per sample)
    CUDA_VISIBLE_DEVICES=0 python bench_fleurs_multilingual.py --model 1.7b --mode batch
"""

import argparse
import json
import logging
import re
import sys
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# FLEURS language configs: (hf_config, language_code, metric_type)
FLEURS_CONFIGS = {
    "en": ("en_us", "en", "wer"),
    "ko": ("ko_kr", "ko", "cer"),
    "zh": ("cmn_hans_cn", "zh", "cer"),
    "vi": ("vi_vn", "vi", "wer"),
    "ja": ("ja_jp", "ja", "cer"),
}


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_fleurs(lang: str, n_samples: int = 100) -> List[dict]:
    """Download FLEURS test samples for a given language."""
    import io
    import tarfile

    import soundfile as sf
    from huggingface_hub import hf_hub_download

    config_name, lang_code, _ = FLEURS_CONFIGS[lang]
    cache_dir = Path.home() / ".cache" / "whisperlivekit" / f"fleurs_{lang}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / f"metadata_{n_samples}.json"

    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if len(meta) >= n_samples:
            valid = [m for m in meta[:n_samples] if Path(m["path"]).exists()]
            if len(valid) == n_samples:
                logger.info("Using cached %d FLEURS-%s samples", n_samples, lang)
                return valid

    logger.info("Downloading FLEURS-%s test set via HuggingFace Hub...", lang)

    # Download TSV metadata
    tsv_path = hf_hub_download(
        "google/fleurs", f"data/{config_name}/test.tsv", repo_type="dataset"
    )
    entries = []
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 6:
                entries.append({
                    "id": parts[0],
                    "filename": parts[1],
                    "raw_transcription": parts[2],
                    "transcription": parts[3],
                    "num_samples": int(parts[5]),
                    "gender": parts[6] if len(parts) > 6 else "",
                })

    # Download and extract audio tar.gz
    tar_path = hf_hub_download(
        "google/fleurs", f"data/{config_name}/audio/test.tar.gz", repo_type="dataset"
    )
    audio_dir = cache_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting audio files for %s...", lang)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(audio_dir)

    # Build filename -> extracted path mapping
    audio_files = {}
    for p in audio_dir.rglob("*.wav"):
        audio_files[p.name] = p

    samples = []
    for i, entry in enumerate(entries):
        if len(samples) >= n_samples:
            break

        audio_path = audio_files.get(entry["filename"])
        if not audio_path or not audio_path.exists():
            logger.warning("Audio not found: %s", entry["filename"])
            continue

        audio_array, sr = sf.read(str(audio_path), dtype="float32")
        audio_array = np.array(audio_array, dtype=np.float32)

        if sr != SAMPLE_RATE:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        # Save as standardized WAV
        wav_path = cache_dir / f"fleurs_{lang}_{i:03d}.wav"
        with wave.open(str(wav_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            audio_int16 = (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16)
            wf.writeframes(audio_int16.tobytes())

        # Use raw_transcription for CJK languages (avoids spaced-out chars
        # and embedded Latin in the normalized transcription field)
        if lang in ("zh",):
            ref = entry["raw_transcription"]
        else:
            ref = entry["transcription"]
        duration = len(audio_array) / sr

        samples.append({
            "path": str(wav_path),
            "reference": ref,
            "duration": round(duration, 2),
            "index": i,
            "language": lang,
        })

    meta_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2))
    logger.info("Downloaded %d FLEURS-%s samples, total %.1fs",
                len(samples), lang, sum(s["duration"] for s in samples))
    return samples


# ---------------------------------------------------------------------------
# Error rate computation
# ---------------------------------------------------------------------------

def compute_error_rate(reference: str, hypothesis: str, metric: str) -> dict:
    """Compute WER or CER depending on the metric type."""
    if metric == "cer":
        return _compute_cer(reference, hypothesis)
    else:
        return _compute_wer(reference, hypothesis)


def _compute_cer(reference: str, hypothesis: str) -> dict:
    """Compute Character Error Rate (CER).

    For Chinese: strips Latin text (foreign names in reference), keeps CJK + punctuation.
    For Korean: strips non-Korean/alphanumeric.
    """
    def normalize_zh(text: str) -> list:
        # Remove parenthesized Latin names like (Recep Tayyip Erdoğan)
        text = re.sub(r'[（(][A-Za-z\s·.\u00C0-\u024F]+[）)]', '', text)
        # Remove standalone Latin words (not mixed with CJK)
        text = re.sub(r'[A-Za-z\u00C0-\u024F]+', '', text)
        # Remove punctuation and whitespace
        text = re.sub(r'[^\u4e00-\u9fff\u3400-\u4dbf\uff00-\uffef0-9]', '', text)
        return list(text)

    def normalize_ko(text: str) -> list:
        text = re.sub(r'[^\w\uac00-\ud7af]', '', text)
        return list(text)

    def normalize_ja(text: str) -> list:
        # Remove parenthesized Latin names
        text = re.sub(r'[（(][A-Za-z\s·.\u00C0-\u024F]+[）)]', '', text)
        # Remove standalone Latin words
        text = re.sub(r'[A-Za-z\u00C0-\u024F]+', '', text)
        # Keep hiragana, katakana, kanji, numbers
        text = re.sub(r'[^\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\uff00-\uffef0-9]', '', text)
        return list(text)

    def normalize_generic(text: str) -> list:
        text = re.sub(r'\s+', '', text)
        text = re.sub(r'[^\w\u4e00-\u9fff\uac00-\ud7af\u3040-\u309f\u30a0-\u30ff]', '', text)
        return list(text)

    # Detect primary script
    cjk_count = len(re.findall(r'[\u4e00-\u9fff]', reference))
    ko_count = len(re.findall(r'[\uac00-\ud7af]', reference))
    ja_count = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', reference))

    if ja_count > 3 and ja_count >= ko_count:
        ref_chars = normalize_ja(reference)
        hyp_chars = normalize_ja(hypothesis)
    elif cjk_count > ko_count and cjk_count > 5 and ja_count < 3:
        ref_chars = normalize_zh(reference)
        hyp_chars = normalize_zh(hypothesis)
    elif ko_count > cjk_count and ko_count > 5:
        ref_chars = normalize_ko(reference)
        hyp_chars = normalize_ko(hypothesis)
    else:
        ref_chars = normalize_generic(reference)
        hyp_chars = normalize_generic(hypothesis)

    return _levenshtein_detail(ref_chars, hyp_chars, "cer")


def _compute_wer(reference: str, hypothesis: str) -> dict:
    """Compute Word Error Rate (WER)."""
    def normalize(text: str) -> list:
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    ref_words = normalize(reference)
    hyp_words = normalize(hypothesis)
    return _levenshtein_detail(ref_words, hyp_words, "wer")


def _levenshtein_detail(ref: list, hyp: list, metric_name: str) -> dict:
    """Compute Levenshtein distance with error type breakdown."""
    if not ref:
        return {metric_name: 0.0 if not hyp else 1.0,
                "ref_len": 0, "hyp_len": len(hyp),
                "substitutions": 0, "insertions": len(hyp), "deletions": 0}

    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    # Backtrace
    subs, ins, dels = 0, 0, 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            subs += 1; i -= 1; j -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            ins += 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            dels += 1; i -= 1
        else:
            break

    rate = (subs + ins + dels) / len(ref)
    return {
        metric_name: rate, "ref_len": len(ref), "hyp_len": len(hyp),
        "substitutions": subs, "insertions": ins, "deletions": dels,
    }


# ---------------------------------------------------------------------------
# vLLM model loading
# ---------------------------------------------------------------------------

def load_vllm_model(model_size: str, gpu_mem_util: float = 0.45,
                    quantization: Optional[str] = None):
    """Load Qwen3-ASR via vLLM."""
    from vllm import LLM, SamplingParams

    if "1.7" in model_size:
        model_id = "Qwen/Qwen3-ASR-1.7B"
    elif "0.6" in model_size:
        model_id = "Qwen/Qwen3-ASR-0.6B"
    else:
        model_id = model_size  # allow full model path

    quant_label = f" ({quantization})" if quantization else ""
    logger.info("Loading Qwen3-ASR via vLLM: %s%s", model_id, quant_label)

    llm_kwargs = dict(
        model=model_id,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=4096,
        trust_remote_code=True,
    )
    if quantization:
        llm_kwargs["quantization"] = quantization

    llm = LLM(**llm_kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=128, stop=["<|im_end|>"])

    audio_placeholder = "<|audio_start|><|audio_pad|><|audio_end|>"
    prompt_tpl = (
        f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # Warmup
    warmup = np.random.randn(SAMPLE_RATE * 2).astype(np.float32) * 0.01
    llm.generate(
        [{"prompt": prompt_tpl, "multi_modal_data": {"audio": warmup}}],
        sp,
    )
    logger.info("Model loaded and warmed up")

    return llm, sp, prompt_tpl, model_id


def _parse_qwen3_output(text: str) -> str:
    """Parse Qwen3-ASR output."""
    for marker in ("<|asr_text|>", "<asr_text>"):
        if marker in text:
            text = text.split(marker, 1)[1].strip()
            break
    text = re.sub(r"^language\s+\w+\s*", "", text).strip()
    return text


def _detect_repetition(text: str, audio_sec: float) -> bool:
    """Detect repetition/hallucination in generated text."""
    if len(text) <= 20:
        return False
    words = text.split()
    if len(words) <= 6:
        return False
    for n in (2, 3, 4):
        if len(words) < n * 3:
            continue
        last_ngram = " ".join(words[-n:])
        if text.count(last_ngram) >= 3:
            return True
    if audio_sec > 0 and len(text) / audio_sec > 40:
        return True
    return False


# ---------------------------------------------------------------------------
# Benchmark modes
# ---------------------------------------------------------------------------

def run_batch(llm, sp, prompt_tpl, samples: list, lang: str) -> list:
    """Run batch inference (one generate() per sample, no streaming prefix)."""
    metric_type = FLEURS_CONFIGS[lang][2]
    results = []

    for i, sample in enumerate(samples):
        audio = _load_audio(sample["path"])
        inp = {"prompt": prompt_tpl, "multi_modal_data": {"audio": audio}}

        t0 = time.perf_counter()
        outputs = llm.generate([inp], sp, use_tqdm=False)
        elapsed = time.perf_counter() - t0

        raw = outputs[0].outputs[0].text.strip()
        hyp = _parse_qwen3_output(raw)
        err = compute_error_rate(sample["reference"], hyp, metric_type)
        rtf = elapsed / sample["duration"]

        results.append({
            "index": sample["index"],
            "language": lang,
            "duration": sample["duration"],
            "error_rate": err[metric_type],
            "error_details": err,
            "rtf": round(rtf, 3),
            "elapsed": round(elapsed, 2),
            "hypothesis": hyp,
            "reference": sample["reference"],
            "metric_type": metric_type,
        })

        logger.info("[%s %d/%d] %s=%.1f%% RTF=%.3f | ref: %s | hyp: %s",
                    lang, i+1, len(samples), metric_type.upper(),
                    err[metric_type]*100, rtf,
                    sample["reference"][:40], hyp[:40])

    return results


def run_streaming(llm, sp, prompt_tpl, samples: list, lang: str,
                  ucn: int = 4, utn: int = 5,
                  css_initial: float = 2.0, css_steady: float = 4.0) -> list:
    """Run prefix-constrained streaming inference."""
    from transformers import AutoTokenizer

    metric_type = FLEURS_CONFIGS[lang][2]

    # Determine model from llm config
    model_id = llm.llm_engine.model_config.model
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    results = []

    for i, sample in enumerate(samples):
        audio = _load_audio(sample["path"])
        chunk_size = int(css_initial * SAMPLE_RATE)
        steady_chunk_size = int(css_steady * SAMPLE_RATE)

        raw_decoded = ""
        chunk_id = 0
        total_inferences = 0
        t0 = time.perf_counter()

        # Feed in chunks
        offset = 0
        while offset < len(audio):
            # Adaptive chunk size
            cs = chunk_size if total_inferences == 0 else steady_chunk_size
            chunk_end = min(offset + cs, len(audio))
            current_audio = audio[:chunk_end]

            # Build prefix
            prefix = ""
            if chunk_id >= ucn and raw_decoded:
                cur_ids = tokenizer.encode(raw_decoded)
                k = utn
                while True:
                    end_idx = max(0, len(cur_ids) - k)
                    p = tokenizer.decode(cur_ids[:end_idx]) if end_idx > 0 else ""
                    if '\ufffd' not in p:
                        prefix = p
                        break
                    if end_idx == 0:
                        prefix = ""
                        break
                    k += 1

            # Generate
            prompt = prompt_tpl + prefix
            inp = {"prompt": prompt, "multi_modal_data": {"audio": current_audio}}
            outputs = llm.generate([inp], sp, use_tqdm=False)
            gen_text = outputs[0].outputs[0].text

            raw_decoded = (prefix + gen_text) if prefix else gen_text
            chunk_id += 1
            total_inferences += 1

            # Repetition guard (mirrors qwen3_prefix_processor safety valve)
            parsed_check = _parse_qwen3_output(raw_decoded)
            if _detect_repetition(parsed_check, len(current_audio) / SAMPLE_RATE):
                logger.warning("[%s %d] Repetition detected at chunk %d, resetting prefix",
                               lang, sample["index"], chunk_id)
                raw_decoded = ""

            offset = chunk_end

        elapsed = time.perf_counter() - t0

        hyp = _parse_qwen3_output(raw_decoded)
        err = compute_error_rate(sample["reference"], hyp, metric_type)
        rtf = elapsed / sample["duration"]

        results.append({
            "index": sample["index"],
            "language": lang,
            "duration": sample["duration"],
            "error_rate": err[metric_type],
            "error_details": err,
            "rtf": round(rtf, 3),
            "elapsed": round(elapsed, 2),
            "hypothesis": hyp,
            "reference": sample["reference"],
            "metric_type": metric_type,
            "n_inferences": total_inferences,
        })

        logger.info("[%s %d/%d] %s=%.1f%% RTF=%.3f infs=%d | ref: %s | hyp: %s",
                    lang, i+1, len(samples), metric_type.upper(),
                    err[metric_type]*100, rtf, total_inferences,
                    sample["reference"][:40], hyp[:40])

    return results


def _load_audio(path: str) -> np.ndarray:
    """Load audio file as float32 numpy array."""
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32")
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(lang: str, mode: str, results: list):
    """Print summary statistics for one language."""
    metric_type = FLEURS_CONFIGS[lang][2]
    metric_key = metric_type

    total_ref = sum(r["error_details"]["ref_len"] for r in results)
    total_errors = sum(
        r["error_details"]["substitutions"] + r["error_details"]["insertions"] + r["error_details"]["deletions"]
        for r in results
    )
    micro_rate = total_errors / max(total_ref, 1)
    macro_rate = sum(r["error_rate"] for r in results) / len(results)
    total_audio = sum(r["duration"] for r in results)
    total_elapsed = sum(r["elapsed"] for r in results)
    overall_rtf = total_elapsed / total_audio

    print(f"\n{'='*60}")
    print(f"Language: {lang} | Mode: {mode} | Metric: {metric_type.upper()}")
    print(f"Samples: {len(results)}")
    print(f"Total audio: {total_audio:.1f}s")
    print(f"Micro {metric_type.upper()} (weighted): {micro_rate*100:.2f}%")
    print(f"Macro {metric_type.upper()} (average):  {macro_rate*100:.2f}%")
    print(f"Overall RTF: {overall_rtf:.3f}")
    print(f"{'='*60}")

    # Worst samples
    worst = sorted(results, key=lambda r: r["error_rate"], reverse=True)[:5]
    print(f"\nWorst 5 samples:")
    for r in worst:
        print(f"  [{r['index']}] {metric_type.upper()}={r['error_rate']*100:.1f}%")
        print(f"    ref: {r['reference'][:80]}")
        print(f"    hyp: {r['hypothesis'][:80]}")


# ---------------------------------------------------------------------------
# Language-adaptive defaults (from progrem.md)
# ---------------------------------------------------------------------------

def get_language_defaults(lang: str, model_size: str) -> dict:
    """Get language-adaptive UTN/CSS defaults."""
    is_large = "1.7" in model_size or "3b" in model_size.lower()

    # Non-English languages benefit from utn=15/css=3.0 on 1.7B+
    if is_large and lang in ("ko", "zh", "vi", "ja"):
        return {"utn": 15, "css_initial": 2.0, "css_steady": 3.0}
    # English / default
    return {"utn": 5, "css_initial": 2.0, "css_steady": 4.0}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multilingual FLEURS benchmark")
    parser.add_argument("--model", default="1.7b", help="Model size (0.6b, 1.7b, or full path)")
    parser.add_argument("--lang", nargs="*", default=None,
                        help="Languages to benchmark (default: all 4). E.g. --lang ko zh")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--mode", choices=["batch", "streaming", "both"], default="both",
                        help="Benchmark mode")
    parser.add_argument("--quantization", default=None, help="Quantization (e.g. fp8)")
    parser.add_argument("--gpu-mem-util", type=float, default=0.45)
    parser.add_argument("--ucn", type=int, default=4, help="Unfixed chunk num")
    parser.add_argument("--utn", type=int, default=None,
                        help="Unfixed token num (default: language-adaptive)")
    parser.add_argument("--css-initial", type=float, default=None,
                        help="CSS initial (default: language-adaptive)")
    parser.add_argument("--css-steady", type=float, default=None,
                        help="CSS steady (default: language-adaptive)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    languages = args.lang or list(FLEURS_CONFIGS.keys())
    for lang in languages:
        if lang not in FLEURS_CONFIGS:
            parser.error(f"Unknown language: {lang}. Choose from {list(FLEURS_CONFIGS.keys())}")

    # Download all needed datasets first
    all_samples = {}
    for lang in languages:
        all_samples[lang] = download_fleurs(lang, args.n_samples)

    # Load model once
    llm, sp, prompt_tpl, model_id = load_vllm_model(
        args.model, args.gpu_mem_util, args.quantization
    )

    all_results = {}
    modes = ["batch", "streaming"] if args.mode == "both" else [args.mode]

    for mode in modes:
        for lang in languages:
            samples = all_samples[lang]
            logger.info("Running %s benchmark for %s (%d samples)", mode, lang, len(samples))

            if mode == "batch":
                results = run_batch(llm, sp, prompt_tpl, samples, lang)
            else:
                defaults = get_language_defaults(lang, args.model)
                utn = args.utn if args.utn is not None else defaults["utn"]
                css_i = args.css_initial if args.css_initial is not None else defaults["css_initial"]
                css_s = args.css_steady if args.css_steady is not None else defaults["css_steady"]
                logger.info("Streaming params for %s: ucn=%d, utn=%d, css=%.1f→%.1f",
                            lang, args.ucn, utn, css_i, css_s)
                results = run_streaming(
                    llm, sp, prompt_tpl, samples, lang,
                    ucn=args.ucn, utn=utn,
                    css_initial=css_i, css_steady=css_s,
                )

            print_summary(lang, mode, results)
            all_results[f"{lang}_{mode}"] = results

    # Save results
    quant_tag = f"_{args.quantization}" if args.quantization else ""
    output_path = args.output or f"bench_fleurs_multilingual_{args.model}{quant_tag}.json"

    output = {
        "model": model_id,
        "model_size": args.model,
        "quantization": args.quantization,
        "n_samples": args.n_samples,
        "mode": args.mode,
        "languages": languages,
    }

    for key, results in all_results.items():
        lang = key.split("_")[0]
        mode = key.split("_", 1)[1]
        metric_type = FLEURS_CONFIGS[lang][2]

        total_ref = sum(r["error_details"]["ref_len"] for r in results)
        total_errors = sum(
            r["error_details"]["substitutions"] + r["error_details"]["insertions"] + r["error_details"]["deletions"]
            for r in results
        )
        micro_rate = total_errors / max(total_ref, 1)
        total_audio = sum(r["duration"] for r in results)
        total_elapsed = sum(r["elapsed"] for r in results)

        output[key] = {
            f"micro_{metric_type}": round(micro_rate * 100, 2),
            "overall_rtf": round(total_elapsed / total_audio, 3),
            "total_audio_s": round(total_audio, 1),
            "total_elapsed_s": round(total_elapsed, 1),
            "results": results,
        }

    Path(output_path).write_text(json.dumps(output, ensure_ascii=False, indent=2))
    logger.info("Results saved to %s", output_path)

    # Print final summary table
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY — {model_id} {quant_tag or 'BF16'}")
    print(f"{'='*70}")
    print(f"{'Lang':<6} {'Mode':<10} {'Metric':<6} {'Rate':>8} {'RTF':>8}")
    print(f"{'-'*70}")
    for key, results in all_results.items():
        lang = key.split("_")[0]
        mode = key.split("_", 1)[1]
        metric_type = FLEURS_CONFIGS[lang][2]
        total_ref = sum(r["error_details"]["ref_len"] for r in results)
        total_errors = sum(
            r["error_details"]["substitutions"] + r["error_details"]["insertions"] + r["error_details"]["deletions"]
            for r in results
        )
        micro_rate = total_errors / max(total_ref, 1)
        total_audio = sum(r["duration"] for r in results)
        total_elapsed = sum(r["elapsed"] for r in results)
        rtf = total_elapsed / total_audio
        print(f"{lang:<6} {mode:<10} {metric_type.upper():<6} {micro_rate*100:>7.2f}% {rtf:>7.3f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
