"""
NVIDIA Nemotron Speech Streaming backend via NeMo.

Cache-Aware FastConformer-RNNT architecture provides:
- True streaming with zero redundant computation (no re-feed)
- Configurable chunk sizes: 80ms, 160ms, 560ms, 1120ms
- Sub-200ms first-word latency at 160ms chunks
- English only

Install:
    pip install nemo_toolkit[asr]

Usage:
    wlk serve --backend nemotron-streaming
"""

import logging
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch

from whisperlivekit.timed_objects import ASRToken, ChangeSpeaker, Transcript

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Map chunk label to att_context_size [left_chunks, right_context]
CHUNK_CONFIGS = {
    "80ms": [70, 0],    # 1 frame = 80ms
    "160ms": [70, 1],   # 2 frames = 160ms
    "560ms": [70, 6],   # 7 frames = 560ms
    "1120ms": [70, 13], # 14 frames = 1120ms
}


class NemotronStreamingASR:
    """Shared ASR backend using Nemotron Speech Streaming (NeMo)."""

    sep = " "

    def __init__(
        self,
        model_size: str = None,
        model_dir: str = None,
        lan: str = "en",
        chunk_mode: str = "560ms",
        model_cache_dir: Optional[str] = None,
        model_path: Optional[str] = None,
        **kwargs,
    ):
        self.original_language = "en"
        self.chunk_mode = chunk_mode
        self.att_context_size = CHUNK_CONFIGS.get(chunk_mode, CHUNK_CONFIGS["160ms"])

        model_name = model_path or model_dir or "nvidia/nemotron-speech-streaming-en-0.6b"
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError:
            raise ImportError(
                "nemo_toolkit[asr] is required for Nemotron streaming. "
                "Install: pip install nemo_toolkit[asr]"
            )

        logger.info("Loading Nemotron Speech Streaming: %s (chunk=%s)", self.model_name, self.chunk_mode)
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name)
        self.model = self.model.cuda()
        self.model.eval()

        # Configure encoder for streaming
        right_ctx = self.att_context_size[1]
        self.model.encoder.setup_streaming_params(
            chunk_size=right_ctx + 1,
            left_chunks=self.att_context_size[0],
            shift_size=right_ctx + 1,
        )
        self.chunk_ms = (right_ctx + 1) * 80
        logger.info("Nemotron model loaded, chunk=%dms", self.chunk_ms)

    def transcribe(self, audio):
        raise NotImplementedError(
            "NemotronStreamingASR uses streaming processor, not batch transcribe()"
        )


class NemotronStreamingOnlineProcessor:
    """
    Per-session online processor using Nemotron cache-aware streaming.

    Unlike Qwen3's re-feed approach, Nemotron processes each chunk exactly once
    and maintains encoder caches across chunks. This eliminates O(n) cost growth
    and enables sub-200ms latency.
    """

    SAMPLING_RATE = 16000
    MIN_DURATION_REAL_SILENCE = 5

    def __init__(self, asr: NemotronStreamingASR, logfile=sys.stderr):
        self.asr = asr
        self.logfile = logfile
        self.end = 0.0
        self.buffer: List[ASRToken] = []
        self._audio_queue: List[np.ndarray] = []
        self._prev_text = ""
        self._committed_len = 0
        self._speaker = -1
        self._global_time_offset = 0.0

        # Streaming state
        self._cache_last_channel = None
        self._cache_last_time = None
        self._cache_last_channel_len = None
        self._previous_hypotheses = None
        self._pred_out_stream = None
        self._current_text = ""
        self._step_num = 0

        # Preprocessing buffer
        self._streaming_buffer = None

        self._init_state()

    def _init_state(self):
        """Initialize or reset streaming state."""
        from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

        self._streaming_buffer = CacheAwareStreamingAudioBuffer(
            model=self.asr.model,
            online_normalization=False,
            pad_and_drop_preencoded=False,
        )

        # Initialize caches
        self._cache_last_channel, self._cache_last_time, self._cache_last_channel_len = (
            self.asr.model.encoder.get_initial_cache_state(batch_size=1)
        )
        self._previous_hypotheses = None
        self._pred_out_stream = None
        self._current_text = ""
        self._prev_text = ""
        self._committed_len = 0
        self._audio_queue = []
        self._step_num = 0

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

        # Feed queued audio and run streaming inference
        if self._audio_queue:
            chunk = np.concatenate(self._audio_queue)
            self._audio_queue = []
            self._process_chunk(chunk, keep_all_outputs=is_last)

        new_tokens = self._extract_new_tokens(is_last)
        self.buffer = []
        return new_tokens, self.end

    def _process_chunk(self, audio: np.ndarray, keep_all_outputs: bool = False):
        """Process audio chunk through cache-aware encoder + RNNT decoder."""
        # Preprocess audio through NeMo pipeline
        processed_signal, processed_signal_length = self._streaming_buffer.preprocess_audio(
            audio, device=self.asr.model.device
        )

        if processed_signal is None or processed_signal.shape[-1] == 0:
            return

        drop_extra = (
            self.asr.model.encoder.streaming_cfg.drop_extra_pre_encoded
            if self._step_num != 0 else 0
        )

        with torch.inference_mode():
            (
                self._pred_out_stream,
                transcribed_texts,
                self._cache_last_channel,
                self._cache_last_time,
                self._cache_last_channel_len,
                self._previous_hypotheses,
            ) = self.asr.model.conformer_stream_step(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
                cache_last_channel=self._cache_last_channel,
                cache_last_time=self._cache_last_time,
                cache_last_channel_len=self._cache_last_channel_len,
                keep_all_outputs=keep_all_outputs,
                previous_hypotheses=self._previous_hypotheses,
                previous_pred_out=self._pred_out_stream,
                drop_extra_pre_encoded=drop_extra,
                return_transcription=True,
            )

        self._step_num += 1

        if transcribed_texts:
            hyp = transcribed_texts[0]
            self._current_text = hyp.text if hasattr(hyp, 'text') else str(hyp)

    def _extract_new_tokens(self, is_last: bool) -> List[ASRToken]:
        """Extract newly committed tokens via text diff."""
        current = self._current_text
        prev = self._prev_text
        self._prev_text = current

        if not current:
            return []

        # Find common prefix (stable text)
        common = 0
        limit = min(len(prev), len(current))
        for i in range(limit):
            if prev[i] == current[i]:
                common = i + 1
            else:
                break

        if is_last:
            common = len(current)

        if common > self._committed_len:
            new_text = current[self._committed_len:common]
            self._committed_len = common

            if not new_text.strip():
                return []

            chunk_sec = self.asr.chunk_ms / 1000.0
            token = ASRToken(
                start=round(max(self.end - chunk_sec, 0.0), 2),
                end=round(self.end, 2),
                text=new_text,
                speaker=self._speaker,
                detected_language="en",
            ).with_offset(self._global_time_offset)
            return [token]

        return []

    def get_buffer(self) -> Transcript:
        """Return current unfixed text as draft."""
        unfixed = self._current_text[self._committed_len:]
        return Transcript(None, None, unfixed)

    def start_silence(self) -> Tuple[List[ASRToken], float]:
        result = self.process_iter(is_last=True)
        self._init_state()
        return result

    def end_silence(self, silence_duration: float, offset: float):
        self._global_time_offset = silence_duration + offset

    def new_speaker(self, change_speaker: ChangeSpeaker):
        self.process_iter(is_last=True)
        self._init_state()
        self._speaker = change_speaker.speaker
        self._global_time_offset = change_speaker.start

    def warmup(self, audio: np.ndarray, init_prompt: str = ""):
        try:
            # Quick warmup: process a short chunk
            processed_signal, processed_signal_length = self._streaming_buffer.preprocess_audio(
                audio[:SAMPLE_RATE * 2], device=self.asr.model.device
            )
            if processed_signal is not None and processed_signal.shape[-1] > 0:
                with torch.inference_mode():
                    self.asr.model.conformer_stream_step(
                        processed_signal=processed_signal,
                        processed_signal_length=processed_signal_length,
                        cache_last_channel=self._cache_last_channel,
                        cache_last_time=self._cache_last_time,
                        cache_last_channel_len=self._cache_last_channel_len,
                        return_transcription=True,
                    )
            # Reset after warmup
            self._init_state()
            logger.info("Warmup complete")
        except Exception as e:
            logger.warning("Warmup failed: %s", e)

    def finish(self) -> Tuple[List[ASRToken], float]:
        return self.process_iter(is_last=True)
