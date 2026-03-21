"""
Qwen3-ASR via vLLM — SDK-free multilingual ASR backend.

Supports 0.6B and 1.7B model sizes. No qwen-asr SDK dependency.
Direct vLLM inference with native Qwen3-ASR model support (v0.16+).

100-sample benchmarks (L40S, vLLM v0.16):
  0.6B LS-clean: WER 2.30% (batch RTF 0.014), WER 2.30% (stream css=2.0 RTF 0.046)
  0.6B LS-other: WER 4.38% (batch RTF 0.018)

Supported languages: 100+ languages (same as Qwen3-ASR).

Usage:
    wlk serve --backend qwen3-vllm
    wlk serve --backend qwen3-vllm --model Qwen/Qwen3-ASR-1.7B
"""

import logging
import re
import sys
from typing import List

import numpy as np

from whisperlivekit.local_agreement.backends import ASRBase
from whisperlivekit.timed_objects import ASRToken

logger = logging.getLogger(__name__)


def _parse_qwen3_output(text: str) -> str:
    """Parse Qwen3-ASR output: 'language English<asr_text>transcribed text'."""
    for marker in ("<|asr_text|>", "<asr_text>"):
        if marker in text:
            text = text.split(marker, 1)[1].strip()
            break
    text = re.sub(r"^language\s+\w+\s*", "", text).strip()
    return text


class Qwen3VLLMASR(ASRBase):
    """Qwen3-ASR via vLLM — fast multilingual ASR without SDK dependency."""

    sep = " "
    SAMPLING_RATE = 16000

    def __init__(self, lan="en", model_size=None, cache_dir=None,
                 model_dir=None, logfile=sys.stderr,
                 gpu_memory_utilization=0.45, quantization=None, **kwargs):
        self.logfile = logfile
        self.transcribe_kargs = {}
        self.original_language = None if lan == "auto" else lan
        self._gpu_mem_util = gpu_memory_utilization
        self._model_size = model_size
        self._quantization = quantization
        self.model = self._load_model(model_dir)

    def _load_model(self, model_dir=None):
        from vllm import LLM, SamplingParams

        if model_dir:
            model_id = model_dir
        elif self._model_size and "1.7" in str(self._model_size):
            model_id = "Qwen/Qwen3-ASR-1.7B"
        else:
            model_id = "Qwen/Qwen3-ASR-0.6B"

        quant_label = f" ({self._quantization})" if self._quantization else ""
        logger.info("Loading Qwen3-ASR via vLLM: %s%s", model_id, quant_label)
        llm_kwargs = dict(
            model=model_id,
            dtype="bfloat16",
            gpu_memory_utilization=self._gpu_mem_util,
            max_model_len=4096,
            trust_remote_code=True,
        )
        if self._quantization:
            llm_kwargs["quantization"] = self._quantization
        self._llm = LLM(**llm_kwargs)
        self._sp = SamplingParams(temperature=0.0, max_tokens=256, stop=["<|im_end|>"])

        audio_placeholder = "<|audio_start|><|audio_pad|><|audio_end|>"
        self._prompt_tpl = (
            f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # Warmup
        warmup = np.random.randn(self.SAMPLING_RATE * 2).astype(np.float32) * 0.01
        self._llm.generate(
            [{"prompt": self._prompt_tpl, "multi_modal_data": {"audio": warmup}}],
            self._sp,
        )
        logger.info("Qwen3-ASR vLLM loaded and warmed up")
        return self._llm

    def transcribe(self, audio: np.ndarray, init_prompt: str = ""):
        outputs = self._llm.generate(
            [{"prompt": self._prompt_tpl, "multi_modal_data": {"audio": audio.astype(np.float32)}}],
            self._sp,
        )
        raw = outputs[0].outputs[0].text.strip()
        text = _parse_qwen3_output(raw)
        audio_duration = len(audio) / self.SAMPLING_RATE
        return {"text": text, "duration": audio_duration}

    def ts_words(self, result) -> List[ASRToken]:
        text = result.get("text", "").strip()
        if not text:
            return []
        duration = result.get("duration", 1.0)
        words = text.split()
        if not words:
            return []
        time_per_word = duration / len(words)
        return [
            ASRToken(
                start=round(i * time_per_word, 2),
                end=round((i + 1) * time_per_word, 2),
                text=word,
            )
            for i, word in enumerate(words)
        ]

    def segments_end_ts(self, result) -> List[float]:
        return [result.get("duration", 1.0)]

    def use_vad(self):
        return True
