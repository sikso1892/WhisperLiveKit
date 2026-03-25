#!/usr/bin/env python3
"""Fleurs-ko CER benchmark for Korean ASR backends.

Usage:
    CUDA_VISIBLE_DEVICES=0 python bench_fleurs_ko.py --backend qwen3-0.6b
    CUDA_VISIBLE_DEVICES=1 python bench_fleurs_ko.py --backend qwen3-1.7b
    CUDA_VISIBLE_DEVICES=2 python bench_fleurs_ko.py --backend funasr-nano
    CUDA_VISIBLE_DEVICES=3 python bench_fleurs_ko.py --backend funasr-sensevoice
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import wave
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
N_SAMPLES = 30  # 30 Fleurs-ko test samples


def download_fleurs_ko(n_samples: int = N_SAMPLES) -> list:
    """Download Fleurs-ko test samples via HuggingFace Hub (no datasets scripts)."""
    import io
    import tarfile

    import soundfile as sf
    from huggingface_hub import hf_hub_download

    cache_dir = Path.home() / ".cache" / "whisperlivekit" / "fleurs_ko"
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "metadata.json"

    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if len(meta) >= n_samples:
            valid = [m for m in meta[:n_samples] if Path(m["path"]).exists()]
            if len(valid) == n_samples:
                logger.info("Using cached %d Fleurs-ko samples", n_samples)
                return valid

    logger.info("Downloading Fleurs-ko test set via HuggingFace Hub...")

    # Download TSV metadata
    tsv_path = hf_hub_download("google/fleurs", "data/ko_kr/test.tsv", repo_type="dataset")
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
    tar_path = hf_hub_download("google/fleurs", "data/ko_kr/audio/test.tar.gz", repo_type="dataset")
    audio_dir = cache_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Build filename -> extracted path mapping
    logger.info("Extracting audio files...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(audio_dir)

    # Find extracted audio files
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
        wav_path = cache_dir / f"fleurs_ko_{i:03d}.wav"
        with wave.open(str(wav_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            audio_int16 = (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16)
            wf.writeframes(audio_int16.tobytes())

        ref = entry["transcription"]
        duration = len(audio_array) / sr

        samples.append({
            "path": str(wav_path),
            "reference": ref,
            "duration": round(duration, 2),
            "index": i,
        })
        logger.info("  [%d] %.1fs: %s", i, duration, ref[:60])

    meta_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2))
    logger.info("Downloaded %d samples, total %.1fs",
                len(samples), sum(s["duration"] for s in samples))
    return samples


def compute_cer(reference: str, hypothesis: str) -> dict:
    """Compute Character Error Rate (excluding spaces and punctuation)."""
    def normalize(text: str) -> list:
        text = re.sub(r'[^\w가-힣a-zA-Z0-9]', '', text)
        return list(text)

    ref_chars = normalize(reference)
    hyp_chars = normalize(hypothesis)

    if not ref_chars:
        return {"cer": 0.0 if not hyp_chars else 1.0, "ref_chars": 0,
                "hyp_chars": len(hyp_chars), "substitutions": 0,
                "insertions": len(hyp_chars), "deletions": 0}

    # Levenshtein distance
    n, m = len(ref_chars), len(hyp_chars)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    # Backtrace for error types
    subs, ins, dels = 0, 0, 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_chars[i-1] == hyp_chars[j-1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            subs += 1; i -= 1; j -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            ins += 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            dels += 1; i -= 1
        else:
            break

    cer = (subs + ins + dels) / len(ref_chars)
    return {
        "cer": cer, "ref_chars": len(ref_chars), "hyp_chars": len(hyp_chars),
        "substitutions": subs, "insertions": ins, "deletions": dels,
    }


def load_audio(path: str) -> np.ndarray:
    """Load audio file as float32 numpy array."""
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32")
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


# ---- Backend runners ----

def run_qwen3_streaming(samples: list, model_id: str, unfixed_chunk_num: int = 6) -> list:
    """Run Qwen3-ASR streaming on each sample."""
    from qwen_asr import Qwen3ASRModel

    logger.info("Loading Qwen3-ASR streaming: %s", model_id)
    asr = Qwen3ASRModel.LLM(
        model=model_id,
        gpu_memory_utilization=0.8,
        max_new_tokens=32,
    )

    results = []
    for i, sample in enumerate(samples):
        audio = load_audio(sample["path"])
        chunk_size = int(2.0 * SAMPLE_RATE)

        state = asr.init_streaming_state(
            unfixed_chunk_num=unfixed_chunk_num,
            unfixed_token_num=5,
            chunk_size_sec=2.0,
            language="Korean",
        )

        t0 = time.perf_counter()

        # Feed in 2-second chunks
        offset = 0
        while offset < len(audio):
            chunk = audio[offset:offset + chunk_size]
            asr.streaming_transcribe(chunk, state)
            offset += chunk_size

        asr.finish_streaming_transcribe(state)
        elapsed = time.perf_counter() - t0

        hyp = state.text or ""
        cer_result = compute_cer(sample["reference"], hyp)
        rtf = elapsed / sample["duration"]

        results.append({
            "index": sample["index"],
            "duration": sample["duration"],
            "cer": cer_result["cer"],
            "cer_details": cer_result,
            "rtf": round(rtf, 3),
            "elapsed": round(elapsed, 2),
            "hypothesis": hyp,
            "reference": sample["reference"],
        })

        logger.info("[%d/%d] CER=%.1f%% RTF=%.3f | ref: %s | hyp: %s",
                    i+1, len(samples), cer_result["cer"]*100, rtf,
                    sample["reference"][:40], hyp[:40])

    return results


def run_funasr(samples: list, model_id: str, language: str = "ko") -> list:
    """Run FunASR (SenseVoice or MLT-Nano) on each sample in batch mode."""
    # Fix FunASRNano import path issue (ctc/tools modules in package subdir)
    import sys as _sys
    _nano_dir = os.path.join(
        os.path.dirname(__import__("funasr").__file__), "models", "fun_asr_nano"
    )
    if _nano_dir not in _sys.path:
        _sys.path.insert(0, _nano_dir)
    from funasr import AutoModel

    is_nano = "fun-asr" in model_id.lower() and "nano" in model_id.lower()

    logger.info("Loading FunASR model: %s (nano=%s)", model_id, is_nano)
    if is_nano:
        model = AutoModel(
            model=model_id,
            trust_remote_code=True,
            device="cuda:0",
            disable_update=True,
        )
    else:
        model = AutoModel(model=model_id, device="cuda:0")

    # Language mapping for Nano
    nano_lang_map = {
        "ko": "韩文", "zh": "中文", "en": "英文", "ja": "日文",
    }

    results = []
    for i, sample in enumerate(samples):
        t0 = time.perf_counter()

        if is_nano:
            result = model.generate(
                input=[sample["path"]],
                cache={},
                batch_size=1,
                language=nano_lang_map.get(language, "韩文"),
                itn=True,
            )
        else:
            kwargs = {"input": sample["path"], "cache": {}, "use_itn": True}
            if language:
                kwargs["language"] = language
            result = model.generate(**kwargs)
            # Post-process SenseVoice
            if "sensevoice" in model_id.lower():
                try:
                    from funasr.utils.postprocess_utils import rich_transcription_postprocess
                    for r in result:
                        if "text" in r:
                            r["text"] = rich_transcription_postprocess(r["text"])
                except ImportError:
                    pass

        elapsed = time.perf_counter() - t0

        hyp = ""
        if result:
            if isinstance(result[0], dict):
                hyp = result[0].get("text", "")
            elif isinstance(result[0], list) and result[0]:
                hyp = result[0][0].get("text", "")
        hyp = hyp.strip()

        cer_result = compute_cer(sample["reference"], hyp)
        rtf = elapsed / sample["duration"]

        results.append({
            "index": sample["index"],
            "duration": sample["duration"],
            "cer": cer_result["cer"],
            "cer_details": cer_result,
            "rtf": round(rtf, 3),
            "elapsed": round(elapsed, 2),
            "hypothesis": hyp,
            "reference": sample["reference"],
        })

        logger.info("[%d/%d] CER=%.1f%% RTF=%.3f | ref: %s | hyp: %s",
                    i+1, len(samples), cer_result["cer"]*100, rtf,
                    sample["reference"][:40], hyp[:40])

    return results


def print_summary(backend_name: str, results: list):
    """Print summary statistics."""
    total_ref_chars = sum(r["cer_details"]["ref_chars"] for r in results)
    total_errors = sum(
        r["cer_details"]["substitutions"] + r["cer_details"]["insertions"] + r["cer_details"]["deletions"]
        for r in results
    )
    micro_cer = total_errors / max(total_ref_chars, 1)
    macro_cer = sum(r["cer"] for r in results) / len(results)
    total_audio = sum(r["duration"] for r in results)
    total_elapsed = sum(r["elapsed"] for r in results)
    overall_rtf = total_elapsed / total_audio

    print(f"\n{'='*60}")
    print(f"Backend: {backend_name}")
    print(f"Samples: {len(results)}")
    print(f"Total audio: {total_audio:.1f}s")
    print(f"Micro CER (weighted): {micro_cer*100:.2f}%")
    print(f"Macro CER (average):  {macro_cer*100:.2f}%")
    print(f"Overall RTF: {overall_rtf:.3f}")
    print(f"Avg RTF: {sum(r['rtf'] for r in results) / len(results):.3f}")
    print(f"{'='*60}")

    # Worst samples
    worst = sorted(results, key=lambda r: r["cer"], reverse=True)[:5]
    print("\nWorst 5 samples:")
    for r in worst:
        print(f"  [{r['index']}] CER={r['cer']*100:.1f}%")
        print(f"    ref: {r['reference'][:80]}")
        print(f"    hyp: {r['hypothesis'][:80]}")


def main():
    parser = argparse.ArgumentParser(description="Fleurs-ko CER benchmark")
    parser.add_argument("--backend", required=True,
                        choices=["qwen3-0.6b", "qwen3-1.7b", "funasr-nano", "funasr-sensevoice"],
                        help="Backend to benchmark")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--unfixed-chunk-num", type=int, default=6)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    samples = download_fleurs_ko(args.n_samples)
    logger.info("Loaded %d samples", len(samples))

    backend_name = args.backend
    if backend_name == "qwen3-0.6b":
        results = run_qwen3_streaming(samples, "Qwen/Qwen3-ASR-0.6B", args.unfixed_chunk_num)
    elif backend_name == "qwen3-1.7b":
        results = run_qwen3_streaming(samples, "Qwen/Qwen3-ASR-1.7B", args.unfixed_chunk_num)
    elif backend_name == "funasr-nano":
        results = run_funasr(samples, "FunAudioLLM/Fun-ASR-MLT-Nano-2512", "ko")
    elif backend_name == "funasr-sensevoice":
        results = run_funasr(samples, "iic/SenseVoiceSmall", "ko")

    print_summary(backend_name, results)

    # Save results
    output_path = args.output or f"bench_fleurs_ko_{backend_name}.json"
    output = {
        "backend": backend_name,
        "n_samples": len(results),
        "total_audio_s": sum(r["duration"] for r in results),
        "micro_cer": sum(
            r["cer_details"]["substitutions"] + r["cer_details"]["insertions"] + r["cer_details"]["deletions"]
            for r in results
        ) / max(sum(r["cer_details"]["ref_chars"] for r in results), 1),
        "macro_cer": sum(r["cer"] for r in results) / len(results),
        "overall_rtf": sum(r["elapsed"] for r in results) / sum(r["duration"] for r in results),
        "results": results,
    }
    Path(output_path).write_text(json.dumps(output, ensure_ascii=False, indent=2))
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
