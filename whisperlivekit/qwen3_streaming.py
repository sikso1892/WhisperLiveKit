"""
Qwen3-ASR official streaming backend via qwen_asr SDK + vLLM.

Uses the official Qwen3ASRModel.LLM streaming API, which provides:
- Chunked audio processing with self-correction (unfixed chunks)
- vLLM-optimized inference with KV cache management
- Official streaming WER: 1.95% (1.7B) / 4.51% (0.6B) on LibriSpeech clean

Install:
    pip install qwen-asr[vllm]

Usage:
    wlk serve --backend qwen3-streaming --model Qwen/Qwen3-ASR-1.7B
"""

import logging
import sys
from typing import List, Optional, Tuple

import numpy as np

from whisperlivekit.timed_objects import ASRToken, ChangeSpeaker, Transcript

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

QWEN3_MODEL_MAPPING = {
    "0.6b": "Qwen/Qwen3-ASR-0.6B",
    "1.7b": "Qwen/Qwen3-ASR-1.7B",
}


class Qwen3StreamingASR:
    """Shared ASR backend using Qwen3-ASR official streaming API (vLLM)."""

    sep = ""

    def __init__(
        self,
        model_size: str = None,
        model_dir: str = None,
        lan: str = "auto",
        gpu_memory_utilization: float = 0.8,
        model_cache_dir: Optional[str] = None,
        model_path: Optional[str] = None,
        **kwargs,
    ):
        self.transcribe_kargs = {}
        self.original_language = None if lan == "auto" else lan

        if model_dir:
            model_id = model_dir
        elif model_path:
            model_id = model_path
        elif model_size:
            model_id = QWEN3_MODEL_MAPPING.get(model_size.lower(), model_size)
        else:
            model_id = "Qwen/Qwen3-ASR-1.7B"

        self.model_id = model_id
        self.gpu_memory_utilization = gpu_memory_utilization
        self._load_model()

    def _load_model(self):
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError:
            raise ImportError(
                "qwen-asr[vllm] is required for Qwen3 streaming. "
                "Install: pip install qwen-asr[vllm]"
            )

        logger.info("Loading Qwen3-ASR streaming (vLLM): %s", self.model_id)
        self.asr = Qwen3ASRModel.LLM(
            model=self.model_id,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_new_tokens=32,
        )
        logger.info("Qwen3-ASR streaming model loaded")

    def transcribe(self, audio):
        raise NotImplementedError(
            "Qwen3StreamingASR uses streaming_transcribe(), not batch transcribe()"
        )


class Qwen3StreamingOnlineProcessor:
    """
    Per-session online processor using Qwen3-ASR official streaming API.

    The SDK accumulates audio internally and re-processes the full buffer
    each chunk. vLLM's KV cache makes this efficient.

    Fixed/unfixed text is determined by diffing consecutive state.text
    values: the common prefix is fixed (committed), the rest is unfixed.
    """

    SAMPLING_RATE = 16000
    MIN_DURATION_REAL_SILENCE = 5

    def __init__(self, asr: Qwen3StreamingASR, logfile=sys.stderr):
        self.asr = asr
        self.logfile = logfile
        self.end = 0.0
        self.buffer: List[ASRToken] = []
        self._audio_queue: List[np.ndarray] = []
        self._prev_text = ""
        self._committed_len = 0  # characters of state.text already emitted
        self._state = None
        self._speaker = -1
        self._global_time_offset = 0.0
        self._init_state()

    def _init_state(self):
        """Initialize or reset streaming state."""
        kwargs = {
            "unfixed_chunk_num": 4,
            "unfixed_token_num": 5,
            "chunk_size_sec": 2.0,
        }
        if self.asr.original_language:
            kwargs["language"] = self.asr.original_language
        self._state = self.asr.asr.init_streaming_state(**kwargs)
        self._prev_text = ""
        self._committed_len = 0
        self._audio_queue = []

    @property
    def speaker(self):
        return self._speaker

    @speaker.setter
    def speaker(self, value):
        self._speaker = value

    @property
    def global_time_offset(self):
        return self._global_time_offset

    @global_time_offset.setter
    def global_time_offset(self, value):
        self._global_time_offset = value

    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time: float):
        self.end = audio_stream_end_time
        self._audio_queue.append(audio.astype(np.float32))

    def process_iter(self, is_last=False) -> Tuple[List[ASRToken], float]:
        if not self._audio_queue and not is_last:
            return [], self.end

        # Feed queued audio to streaming state
        if self._audio_queue:
            chunk = np.concatenate(self._audio_queue)
            self._audio_queue = []
            self.asr.asr.streaming_transcribe(chunk, self._state)

        if is_last:
            self.asr.asr.finish_streaming_transcribe(self._state)

        current_text = self._state.text or ""
        new_tokens = self._extract_new_tokens(current_text, is_last)

        self.buffer = []
        return new_tokens, self.end

    def _extract_new_tokens(self, current_text: str, is_last: bool) -> List[ASRToken]:
        """Extract newly fixed tokens via common-prefix diff."""
        prev = self._prev_text
        self._prev_text = current_text

        if not current_text:
            return []

        # Find common prefix length between previous and current text.
        # Common prefix = text that hasn't changed = fixed/committed.
        common = 0
        limit = min(len(prev), len(current_text))
        for i in range(limit):
            if prev[i] == current_text[i]:
                common = i + 1
            else:
                break

        # On is_last, commit everything
        if is_last:
            common = len(current_text)

        # Emit text from _committed_len to common
        if common > self._committed_len:
            new_text = current_text[self._committed_len:common]
            self._committed_len = common

            if not new_text.strip():
                return []

            chunk_sec = self._state.chunk_size_sec if hasattr(self._state, 'chunk_size_sec') else 2.0
            token = ASRToken(
                start=round(max(self.end - chunk_sec, 0.0), 2),
                end=round(self.end, 2),
                text=new_text,
                speaker=self._speaker,
                detected_language=getattr(self._state, 'language', None),
            ).with_offset(self._global_time_offset)
            return [token]

        return []

    def get_buffer(self) -> Transcript:
        """Return current unfixed text as draft."""
        current = (self._state.text or "") if self._state else ""
        unfixed = current[self._committed_len:]
        return Transcript(None, None, unfixed)

    def start_silence(self) -> Tuple[List[ASRToken], float]:
        return self.process_iter(is_last=True)

    def end_silence(self, silence_duration: float, offset: float):
        if silence_duration >= self.MIN_DURATION_REAL_SILENCE:
            self._init_state()
            self._global_time_offset = silence_duration + offset

    def new_speaker(self, change_speaker: ChangeSpeaker):
        self.process_iter(is_last=True)
        self._init_state()
        self._speaker = change_speaker.speaker
        self._global_time_offset = change_speaker.start

    def warmup(self, audio: np.ndarray, init_prompt: str = ""):
        try:
            state = self.asr.asr.init_streaming_state(
                unfixed_chunk_num=1, chunk_size_sec=1.0,
            )
            self.asr.asr.streaming_transcribe(audio[:SAMPLE_RATE * 2], state)
            self.asr.asr.finish_streaming_transcribe(state)
            logger.info("Warmup complete")
        except Exception as e:
            logger.warning("Warmup failed: %s", e)

    def finish(self) -> Tuple[List[ASRToken], float]:
        return self.process_iter(is_last=True)
