"""
IBM Granite 4.0 1B Speech ASR backend via vLLM.

5x faster than Transformers backend. Best English ASR quality measured:
WER 1.16% LS-clean, 2.06% LS-other (30 samples), RTF 0.023 (batch).

Streaming via LocalAgreement: WER 1.16% at css=2.0 (RTF 0.084).

Supported languages: en, fr, de, es, pt, ja.

Usage:
    wlk serve --backend granite-speech-vllm
"""

import logging
import sys
from typing import List

import numpy as np

from whisperlivekit.local_agreement.backends import ASRBase
from whisperlivekit.timed_objects import ASRToken

logger = logging.getLogger(__name__)


class GraniteVLLMASR(ASRBase):
    """Granite 4.0 1B Speech via vLLM — fast batch/streaming ASR."""

    sep = ""
    SAMPLING_RATE = 16000

    def __init__(self, lan="en", model_size=None, cache_dir=None,
                 model_dir=None, logfile=sys.stderr,
                 gpu_memory_utilization=0.45, **kwargs):
        self.logfile = logfile
        self.transcribe_kargs = {}
        self.original_language = None if lan == "auto" else lan
        self._gpu_mem_util = gpu_memory_utilization
        self.model = self._load_model(model_dir)

    def _load_model(self, model_dir=None):
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor

        model_id = model_dir or "ibm-granite/granite-4.0-1b-speech"

        logger.info("Loading Granite Speech via vLLM: %s", model_id)
        self._llm = LLM(
            model=model_id,
            dtype="bfloat16",
            gpu_memory_utilization=self._gpu_mem_util,
            max_model_len=2048,
        )
        self._sp = SamplingParams(temperature=0.01, max_tokens=256)

        processor = AutoProcessor.from_pretrained(model_id)
        self._tokenizer = processor.tokenizer
        user_prompt = "<|audio|>can you transcribe the speech into a written format?"
        chat = [{"role": "user", "content": user_prompt}]
        self._prompt_tpl = self._tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        # Warmup
        warmup = np.random.randn(self.SAMPLING_RATE * 2).astype(np.float32) * 0.01
        self._llm.generate(
            [{"prompt": self._prompt_tpl, "multi_modal_data": {"audio": warmup}}],
            self._sp,
        )
        logger.info("Granite Speech vLLM loaded and warmed up")
        return self._llm

    def transcribe(self, audio: np.ndarray, init_prompt: str = ""):
        outputs = self._llm.generate(
            [{"prompt": self._prompt_tpl, "multi_modal_data": {"audio": audio.astype(np.float32)}}],
            self._sp,
        )
        text = outputs[0].outputs[0].text.strip()
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
