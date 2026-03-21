"""
Qwen3-ASR streaming backend via pure Transformers (no vLLM dependency).

Replicates the SDK's streaming logic using AutoModel.from_pretrained()
instead of LLM(). Achieves identical quality with 4.5x less VRAM.

Comparison (Qwen3-ASR-1.7B, L40S):
  - VRAM: 3.81 GiB (vs 17.1 GiB with vLLM FP8)
  - Quality: EN WER 1.99%, KO CER 2.23% (comparable to vLLM)
  - RTF: 0.294 (vs 0.053 with vLLM — 5.5x slower but still real-time)

Install:
    pip install qwen-asr

Usage:
    wlk serve --backend qwen3-streaming-tf --model Qwen/Qwen3-ASR-1.7B
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from whisperlivekit.qwen3_streaming import (
    QWEN3_MODEL_MAPPING,
    SAMPLE_RATE,
    _LONGER_CHUNK_LANGUAGES,
    _normalize_language,
)

logger = logging.getLogger(__name__)


@dataclass
class TransformersStreamingState:
    """Streaming state mimicking qwen_asr.ASRStreamingState."""

    unfixed_chunk_num: int
    unfixed_token_num: int
    chunk_size_sec: float
    chunk_size_samples: int

    chunk_id: int
    buffer: np.ndarray
    audio_accum: np.ndarray

    prompt_raw: str
    force_language: Optional[str]

    language: str
    text: str
    _raw_decoded: str


class TransformersStreamingEngine:
    """Mimics qwen_asr.Qwen3ASRModel API using pure Transformers."""

    def __init__(self, model_id: str, dtype=torch.bfloat16, device="cuda:0"):
        try:
            from qwen_asr.core.transformers_backend import (
                Qwen3ASRConfig,
                Qwen3ASRForConditionalGeneration,
                Qwen3ASRProcessor,
            )
            from qwen_asr.inference.utils import parse_asr_output, normalize_language_name, validate_language
            from transformers import AutoConfig, AutoModel, AutoProcessor
        except ImportError:
            raise ImportError(
                "qwen-asr is required. Install: pip install qwen-asr"
            )

        # Register custom model
        AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
        AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
        AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)

        self._parse_asr_output = parse_asr_output
        self._normalize_language_name = normalize_language_name
        self._validate_language = validate_language

        logger.info("Loading Qwen3-ASR (Transformers, %s): %s", dtype, model_id)
        self.model = AutoModel.from_pretrained(
            model_id, dtype=dtype, device_map=device,
            attn_implementation="sdpa",
        )
        self.processor = AutoProcessor.from_pretrained(model_id, fix_mistral_regex=True)
        self.dtype = dtype
        self.device = device

        vram = torch.cuda.memory_allocated() / 1024**3
        logger.info("Qwen3-ASR Transformers model loaded (VRAM: %.2f GiB)", vram)

    def _build_prompt(self, context="", force_language=None):
        msgs = [
            {"role": "system", "content": context or ""},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ]
        base = self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
        if force_language:
            base += f"language {force_language}<asr_text>"
        return base

    def init_streaming_state(
        self,
        context: str = "",
        language: Optional[str] = None,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        chunk_size_sec: float = 2.0,
    ) -> TransformersStreamingState:
        force_language = None
        if language is not None and str(language).strip():
            ln = self._normalize_language_name(str(language))
            self._validate_language(ln)
            force_language = ln

        chunk_size_samples = max(1, int(round(float(chunk_size_sec) * SAMPLE_RATE)))
        prompt_raw = self._build_prompt(context=context, force_language=force_language)

        return TransformersStreamingState(
            unfixed_chunk_num=int(unfixed_chunk_num),
            unfixed_token_num=int(unfixed_token_num),
            chunk_size_sec=float(chunk_size_sec),
            chunk_size_samples=chunk_size_samples,
            chunk_id=0,
            buffer=np.zeros((0,), dtype=np.float32),
            audio_accum=np.zeros((0,), dtype=np.float32),
            prompt_raw=prompt_raw,
            force_language=force_language,
            language="",
            text="",
            _raw_decoded="",
        )

    @torch.no_grad()
    def streaming_transcribe(
        self, pcm16k: np.ndarray, state: TransformersStreamingState
    ) -> TransformersStreamingState:
        x = np.asarray(pcm16k).reshape(-1)
        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        else:
            x = x.astype(np.float32, copy=False)

        if x.shape[0] > 0:
            state.buffer = np.concatenate([state.buffer, x])

        while state.buffer.shape[0] >= state.chunk_size_samples:
            chunk = state.buffer[:state.chunk_size_samples]
            state.buffer = state.buffer[state.chunk_size_samples:]

            if state.audio_accum.shape[0] == 0:
                state.audio_accum = chunk
            else:
                state.audio_accum = np.concatenate([state.audio_accum, chunk])

            prefix = self._build_prefix(state)
            self._run_inference(state, prefix)

        return state

    def finish_streaming_transcribe(
        self, state: TransformersStreamingState
    ) -> TransformersStreamingState:
        if state.buffer is None or state.buffer.shape[0] == 0:
            return state

        tail = state.buffer
        state.buffer = np.zeros((0,), dtype=np.float32)

        if state.audio_accum.shape[0] == 0:
            state.audio_accum = tail
        else:
            state.audio_accum = np.concatenate([state.audio_accum, tail])

        prefix = self._build_prefix(state)
        self._run_inference(state, prefix)
        state.chunk_id += 1

        return state

    def _build_prefix(self, state: TransformersStreamingState) -> str:
        if state.chunk_id < state.unfixed_chunk_num or not state._raw_decoded:
            return ""

        cur_ids = self.processor.tokenizer.encode(state._raw_decoded)
        k = int(state.unfixed_token_num)
        while True:
            end_idx = max(0, len(cur_ids) - k)
            prefix = (
                self.processor.tokenizer.decode(cur_ids[:end_idx])
                if end_idx > 0
                else ""
            )
            if "\ufffd" not in prefix:
                break
            if end_idx == 0:
                prefix = ""
                break
            k += 1
        return prefix

    def _run_inference(self, state: TransformersStreamingState, prefix: str):
        prompt = state.prompt_raw + prefix

        inputs = self.processor(
            text=[prompt],
            audio=[state.audio_accum],
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(self.model.device).to(self.dtype)

        output_ids = self.model.generate(**inputs, max_new_tokens=512)
        gen_text = self.processor.batch_decode(
            output_ids.sequences[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        state._raw_decoded = (prefix + gen_text) if prefix else gen_text
        lang, txt = self._parse_asr_output(
            state._raw_decoded, user_language=state.force_language
        )
        state.language = lang
        state.text = txt
        state.chunk_id += 1

    def transcribe(self, audio_tuple, language=None):
        """Batch transcribe for REST API compatibility."""
        audio, sr = audio_tuple
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        prompt = self._build_prompt(force_language=language)
        inputs = self.processor(
            text=[prompt],
            audio=[audio.astype(np.float32)],
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(self.model.device).to(self.dtype)

        output_ids = self.model.generate(**inputs, max_new_tokens=512)
        gen_text = self.processor.batch_decode(
            output_ids.sequences[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        _, text = self._parse_asr_output(gen_text, user_language=language)
        return text


class Qwen3TransformersStreamingASR:
    """Shared ASR backend using Qwen3-ASR Transformers (no vLLM)."""

    sep = ""

    def __init__(
        self,
        model_size: str = None,
        model_dir: str = None,
        lan: str = "auto",
        model_cache_dir: Optional[str] = None,
        model_path: Optional[str] = None,
        unfixed_chunk_num: int = 5,
        **kwargs,
    ):
        self.transcribe_kargs = {}
        self.original_language = _normalize_language(lan)

        if model_dir:
            model_id = model_dir
        elif model_path:
            model_id = model_path
        elif model_size:
            model_id = QWEN3_MODEL_MAPPING.get(model_size.lower(), model_size)
        else:
            model_id = "Qwen/Qwen3-ASR-1.7B"

        self.model_id = model_id

        is_06b = "0.6b" in model_id.lower()
        if unfixed_chunk_num == 5:
            self.unfixed_chunk_num = 4 if is_06b else 5
            if is_06b:
                logger.info("Auto-selected unfixed_chunk_num=4 for 0.6B model")
        else:
            self.unfixed_chunk_num = unfixed_chunk_num
        self.unfixed_token_num = 5 if is_06b else 7
        self.chunk_size_sec = kwargs.get("chunk_size_sec", 2.0)
        self.use_longer_cjk_chunks = not is_06b

        dtype_str = kwargs.get("dtype", "bfloat16")
        dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
        self.asr = TransformersStreamingEngine(model_id, dtype=dtype)

    def transcribe(self, audio, init_prompt=""):
        lang = self.original_language
        return self.asr.transcribe((audio, SAMPLE_RATE), language=lang)
