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

# Map ISO 639-1 codes to Qwen3 SDK full language names
_LANG_CODE_TO_NAME = {
    "ar": "Arabic", "zh": "Chinese", "yue": "Cantonese", "cs": "Czech",
    "da": "Danish", "nl": "Dutch", "en": "English", "fil": "Filipino",
    "fi": "Finnish", "fr": "French", "de": "German", "el": "Greek",
    "hi": "Hindi", "hu": "Hungarian", "id": "Indonesian", "it": "Italian",
    "ja": "Japanese", "ko": "Korean", "mk": "Macedonian", "ms": "Malay",
    "fa": "Persian", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian",
    "ru": "Russian", "es": "Spanish", "sv": "Swedish", "th": "Thai",
    "tr": "Turkish", "vi": "Vietnamese",
}


def _normalize_language(lan: str) -> Optional[str]:
    """Convert language code/name to Qwen3 SDK format."""
    if not lan or lan == "auto":
        return None
    # Already a full name (e.g. "Korean")
    if lan.capitalize() in _LANG_CODE_TO_NAME.values():
        return lan.capitalize()
    # ISO code (e.g. "ko")
    return _LANG_CODE_TO_NAME.get(lan.lower())


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
        self.gpu_memory_utilization = gpu_memory_utilization

        # Auto-select optimal parameters based on model size
        is_06b = "0.6b" in model_id.lower()
        if unfixed_chunk_num == 5:  # default = not explicitly set
            self.unfixed_chunk_num = 4 if is_06b else 5
            if is_06b:
                logger.info("Auto-selected unfixed_chunk_num=4 for 0.6B model")
        else:
            self.unfixed_chunk_num = unfixed_chunk_num
        # 1.7B benefits from utn=7 for Korean; 0.6B stays at 5
        self.unfixed_token_num = 5 if is_06b else 7
        self.chunk_size_sec = kwargs.get("chunk_size_sec", 2.0)

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

    Long-form stability: the SDK re-feeds all accumulated audio every chunk,
    so per-chunk cost grows O(n) with audio length. To prevent RTF degradation,
    hallucination (Korean 60s+), and context window overflow (120s+), the
    processor automatically resets the streaming state when accumulated audio
    exceeds max_session_audio_sec (default 30s). Overlap is disabled by
    default (overlap_sec=0) because the unfixed_chunk_num self-correction
    mechanism handles context transitions well without re-feeding previous
    audio.
    """

    SAMPLING_RATE = 16000
    MIN_DURATION_REAL_SILENCE = 5

    def __init__(self, asr: Qwen3StreamingASR, logfile=sys.stderr,
                 max_session_audio_sec: float = 30.0,
                 overlap_sec: float = 0.0):
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

        # Long-form session management
        self._max_session_audio_sec = max_session_audio_sec
        self._overlap_sec = overlap_sec
        self._session_audio_fed = 0  # samples fed to current session
        self._recent_audio: List[np.ndarray] = []  # rolling buffer for overlap

        self._init_state()

    def _init_state(self):
        """Initialize or reset streaming state."""
        kwargs = {
            "unfixed_chunk_num": self.asr.unfixed_chunk_num,
            "unfixed_token_num": self.asr.unfixed_token_num,
            "chunk_size_sec": self.asr.chunk_size_sec,
        }
        if self.asr.original_language:
            kwargs["language"] = self.asr.original_language
        self._state = self.asr.asr.init_streaming_state(**kwargs)
        self._prev_text = ""
        self._committed_len = 0
        self._audio_queue = []
        self._session_audio_fed = 0
        self._recent_audio = []

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
        audio_f32 = audio.astype(np.float32)
        self._audio_queue.append(audio_f32)
        # Keep rolling buffer for overlap on session reset (only if overlap enabled)
        if self._overlap_sec > 0:
            self._recent_audio.append(audio_f32)
            overlap_samples = int(self._overlap_sec * self.SAMPLING_RATE)
            total = sum(len(a) for a in self._recent_audio)
            while total > overlap_samples and len(self._recent_audio) > 1:
                total -= len(self._recent_audio[0])
                self._recent_audio.pop(0)

    def _maybe_reset_session(self) -> List[ASRToken]:
        """Reset session if accumulated audio exceeds threshold. Returns emitted tokens."""
        max_samples = int(self._max_session_audio_sec * self.SAMPLING_RATE)
        if self._max_session_audio_sec <= 0 or self._session_audio_fed < max_samples:
            return []

        # Finalize current session
        self.asr.asr.finish_streaming_transcribe(self._state)
        current_text = self._state.text or ""
        tokens = self._extract_new_tokens(current_text, is_last=True)

        logger.info(
            "Session reset at %.1fs audio (text=%d chars)",
            self._session_audio_fed / self.SAMPLING_RATE, len(current_text)
        )

        # Save overlap audio before resetting
        overlap_audio = np.concatenate(self._recent_audio) if self._recent_audio else None

        # Reset state
        self._init_state()

        # Re-feed overlap audio to new session
        if overlap_audio is not None and len(overlap_audio) > 0:
            chunk_samples = int(self.asr.chunk_size_sec * self.SAMPLING_RATE)
            offset = 0
            while offset < len(overlap_audio):
                ov_chunk = overlap_audio[offset:offset + chunk_samples]
                self.asr.asr.streaming_transcribe(ov_chunk, self._state)
                offset += chunk_samples
            self._session_audio_fed = len(overlap_audio)
            self._recent_audio = [overlap_audio]

        return tokens

    def process_iter(self, is_last=False) -> Tuple[List[ASRToken], float]:
        if not self._audio_queue and not is_last:
            return [], self.end

        # Feed queued audio to streaming state
        if self._audio_queue:
            chunk = np.concatenate(self._audio_queue)
            self._audio_queue = []
            self.asr.asr.streaming_transcribe(chunk, self._state)
            self._session_audio_fed += len(chunk)

        # Check for automatic session reset (long-form stability)
        reset_tokens = self._maybe_reset_session()

        if is_last:
            self.asr.asr.finish_streaming_transcribe(self._state)

        current_text = self._state.text or ""
        new_tokens = self._extract_new_tokens(current_text, is_last)

        self.buffer = []
        return reset_tokens + new_tokens, self.end

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
        result = self.process_iter(is_last=True)
        # Reset session on every speech→silence transition.
        # This prevents audio accumulation across utterances, which causes
        # O(n) re-feed cost growth, Korean hallucination, and quality loss.
        self._init_state()
        return result

    def end_silence(self, silence_duration: float, offset: float):
        # Session already reset in start_silence(); just update time offset.
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
