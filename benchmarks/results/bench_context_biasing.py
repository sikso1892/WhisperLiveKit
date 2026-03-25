#!/usr/bin/env python3
"""Exp #196: Contextual biasing via system prompt injection.

Tests whether adding a system prompt with context keywords improves
Qwen3-ASR batch accuracy on FLEURS-ko/en.

Conditions:
1. baseline: No system prompt (current default)
2. oracle: Extract proper nouns from reference, inject as context
3. generic: Use generic domain context (practical scenario)

Usage:
    CUDA_VISIBLE_DEVICES=0 python bench_context_biasing.py --n-samples 30
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Reuse FLEURS download and error rate from multilingual bench
from bench_fleurs_multilingual import (
    download_fleurs, compute_error_rate, _load_audio, _parse_qwen3_output,
    FLEURS_CONFIGS,
)


def extract_proper_nouns_ko(text: str) -> list:
    """Extract likely proper nouns from Korean text.

    Heuristic: words containing Latin characters, numbers,
    or standalone 2-3 character Korean words (often names/places).
    """
    keywords = []
    # Latin words (foreign names, terms)
    latin = re.findall(r'[A-Za-z\u00C0-\u024F]{2,}', text)
    keywords.extend(latin)
    # Numbers
    numbers = re.findall(r'\d+', text)
    keywords.extend(numbers)
    return keywords


def extract_proper_nouns_en(text: str) -> list:
    """Extract capitalized words (proper nouns) from English text."""
    words = text.split()
    # Skip first word (always capitalized), find others
    proper = [w.strip('.,;:!?') for w in words[1:] if w[0].isupper() and len(w) > 1]
    # Add numbers
    numbers = re.findall(r'\d+', text)
    proper.extend(numbers)
    return list(set(proper))


def build_prompt(audio_placeholder: str, system_msg: str = None) -> str:
    """Build prompt template with optional system message."""
    parts = []
    if system_msg:
        parts.append(f"<|im_start|>system\n{system_msg}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


GENERIC_CONTEXTS = {
    "ko": "이 오디오는 뉴스, 역사, 과학, 문화 등 다양한 주제의 한국어 음성입니다.",
    "en": "This audio contains English speech about news, history, science, culture, and various topics.",
}


def run_condition(llm, sp, samples, lang, condition, system_msg_fn=None):
    """Run one experimental condition on all samples."""
    metric_type = FLEURS_CONFIGS[lang][2]
    audio_placeholder = "<|audio_start|><|audio_pad|><|audio_end|>"
    results = []

    for i, sample in enumerate(samples):
        audio = _load_audio(sample["path"])

        # Build system message based on condition
        if condition == "baseline":
            sys_msg = None
        elif condition == "oracle":
            if lang == "ko":
                kw = extract_proper_nouns_ko(sample["reference"])
            else:
                kw = extract_proper_nouns_en(sample["reference"])
            sys_msg = f"Context keywords: {', '.join(kw)}" if kw else None
        elif condition == "generic":
            sys_msg = GENERIC_CONTEXTS.get(lang, "")
        else:
            sys_msg = None

        prompt = build_prompt(audio_placeholder, sys_msg)
        inp = {"prompt": prompt, "multi_modal_data": {"audio": audio}}

        t0 = time.perf_counter()
        outputs = llm.generate([inp], sp, use_tqdm=False)
        elapsed = time.perf_counter() - t0

        raw = outputs[0].outputs[0].text.strip()
        hyp = _parse_qwen3_output(raw)
        err = compute_error_rate(sample["reference"], hyp, metric_type)
        rtf = elapsed / sample["duration"]

        results.append({
            "index": sample["index"],
            "condition": condition,
            "system_msg": sys_msg,
            "error_rate": err[metric_type],
            "error_details": err,
            "rtf": round(rtf, 3),
            "hypothesis": hyp,
            "reference": sample["reference"],
        })

        if (i + 1) % 10 == 0 or i == 0:
            logger.info("[%s %s %d/%d] %s=%.1f%% RTF=%.3f",
                        lang, condition, i+1, len(samples), metric_type.upper(),
                        err[metric_type]*100, rtf)

    # Compute micro error rate
    total_ref = sum(r["error_details"]["ref_len"] for r in results)
    total_errors = sum(
        r["error_details"]["substitutions"] + r["error_details"]["insertions"] + r["error_details"]["deletions"]
        for r in results
    )
    micro = total_errors / max(total_ref, 1)
    avg_rtf = sum(r["rtf"] for r in results) / len(results)

    return {
        "condition": condition,
        "language": lang,
        f"micro_{metric_type}": round(micro * 100, 2),
        "avg_rtf": round(avg_rtf, 3),
        "n_samples": len(results),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Context biasing experiment")
    parser.add_argument("--model", default="1.7b")
    parser.add_argument("--lang", nargs="*", default=["ko", "en"])
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--quantization", default="fp8")
    parser.add_argument("--gpu-mem-util", type=float, default=0.45)
    args = parser.parse_args()

    from bench_fleurs_multilingual import load_vllm_model

    # Download data
    all_samples = {}
    for lang in args.lang:
        all_samples[lang] = download_fleurs(lang, args.n_samples)

    # Load model
    llm, sp, _, model_id = load_vllm_model(
        args.model, args.gpu_mem_util, args.quantization
    )

    conditions = ["baseline", "oracle", "generic"]
    all_results = {}

    for lang in args.lang:
        for cond in conditions:
            logger.info("Running %s/%s (%d samples)", lang, cond, len(all_samples[lang]))
            result = run_condition(llm, sp, all_samples[lang], lang, cond)
            all_results[f"{lang}_{cond}"] = result

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"CONTEXTUAL BIASING RESULTS — {model_id}")
    print(f"{'='*70}")
    print(f"{'Lang':<6} {'Condition':<12} {'Metric':<6} {'Rate':>8} {'RTF':>8}")
    print(f"{'-'*70}")
    for key, res in all_results.items():
        lang = key.split("_")[0]
        metric_type = FLEURS_CONFIGS[lang][2]
        rate = res[f"micro_{metric_type}"]
        print(f"{lang:<6} {res['condition']:<12} {metric_type.upper():<6} {rate:>7.2f}% {res['avg_rtf']:>7.3f}")
    print(f"{'='*70}")

    # Per-sample comparison
    for lang in args.lang:
        metric_type = FLEURS_CONFIGS[lang][2]
        baseline_results = all_results[f"{lang}_baseline"]["results"]
        oracle_results = all_results[f"{lang}_oracle"]["results"]

        improved = 0
        degraded = 0
        for b, o in zip(baseline_results, oracle_results):
            if o["error_rate"] < b["error_rate"] - 0.001:
                improved += 1
            elif o["error_rate"] > b["error_rate"] + 0.001:
                degraded += 1

        print(f"\n{lang} oracle vs baseline: {improved} improved, {degraded} degraded, "
              f"{len(baseline_results) - improved - degraded} unchanged")

    # Save
    output_path = f"bench_context_biasing_{args.model}.json"
    output = {
        "model": model_id,
        "quantization": args.quantization,
        "n_samples": args.n_samples,
    }
    for key, res in all_results.items():
        output[key] = {k: v for k, v in res.items() if k != "results"}
        output[key]["per_sample"] = res["results"]

    Path(output_path).write_text(json.dumps(output, ensure_ascii=False, indent=2))
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
