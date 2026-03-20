"""FunASR online processor -- 3-tier draft/commit/refine.

Tier 1 (draft):   SenseVoiceSmall re-transcribes every ~0.5s → grey text
Tier 2 (commit):  On meaningful pause (≥1s silence) → commit draft as sentence
Tier 3 (refine):  On commit with ≥5s audio → Whisper re-transcribes for quality

Short micro-pauses (breathing, <1s) are ignored — the buffer keeps growing,
giving the model more context for better quality. Commits only happen at
natural speech boundaries detected by silence duration.
"""

import logging
import sys
from typing import List, Optional, Tuple

import numpy as np

from whisperlivekit.timed_objects import ASRToken, Transcript

logger = logging.getLogger(__name__)

MIN_DURATION_REAL_SILENCE = 5
WINDOW_MAX_SEC = 30.0       # hard max before forced commit
MIN_COMMIT_SEC = 2.0        # minimum buffer for any commit
MEANINGFUL_PAUSE_SEC = 0.7  # silence duration that signals utterance boundary
LONG_BUFFER_SEC = 8.0       # commit on any pause if buffer exceeds this
MIN_AUDIO_SEC = 0.5         # minimum audio for draft transcription
MIN_REFINE_AUDIO_SEC = 3.0  # minimum audio for Whisper refinement


class FunASROnlineProcessor:
    """3-tier processor: SenseVoice draft → pause-based commit → Whisper refine."""

    SAMPLING_RATE = 16000

    def __init__(self, asr, logfile=sys.stderr):
        self.asr = asr
        self.logfile = logfile
        self._refine_model = None
        self._refine_loaded = False
        self._pending_commit: List[ASRToken] = []
        self.init()

    # ── Whisper refinement model ──────────────────────────────────────

    def _get_refine_model(self):
        if not self._refine_loaded:
            self._refine_loaded = True
            try:
                from faster_whisper import WhisperModel
                self._refine_model = WhisperModel(
                    "large-v3-turbo",
                    device="cuda",
                    compute_type="float16",
                )
                logger.info("Refinement model loaded: Whisper large-v3-turbo")
            except Exception as e:
                logger.warning(f"Refinement model unavailable: {e}")
        return self._refine_model

    def _refine(self, audio: np.ndarray, draft_text: str) -> Optional[str]:
        """Tier 3: Whisper re-transcription with draft as decoder context."""
        audio_dur = len(audio) / self.SAMPLING_RATE
        if audio_dur < MIN_REFINE_AUDIO_SEC:
            return None
        if not draft_text or len(draft_text) < 3:
            return None

        model = self._get_refine_model()
        if model is None:
            return None

        try:
            refine_lang = getattr(self.asr, 'original_language', None)
            segments, info = model.transcribe(
                audio,
                language=refine_lang,
                initial_prompt=draft_text,
                beam_size=5,
                vad_filter=True,
                no_speech_threshold=0.6,
            )
            text_parts = []
            for seg in segments:
                if seg.no_speech_prob < 0.5:
                    text_parts.append(seg.text.strip())
            refined = " ".join(text_parts).strip()
            if not refined:
                return None
            if len(refined) > len(draft_text) * 3:
                logger.warning(f"Hallucination blocked: draft={len(draft_text)} refined={len(refined)}")
                return None
            logger.info(f"Refined ({audio_dur:.1f}s): {draft_text[:40]}… → {refined[:40]}…")
            return refined
        except Exception as e:
            logger.warning(f"Refinement failed: {e}")
        return None

    # ── State management ──────────────────────────────────────────────

    def init(self, offset: Optional[float] = None):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_time_offset = offset if offset is not None else 0.0
        self._draft_text: str = ""
        self._pending_commit: List[ASRToken] = []
        self.end = 0.0
        self._last_draft_samples: int = 0

    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time=None):
        self.audio_buffer = np.append(self.audio_buffer, audio)
        if audio_stream_end_time is not None:
            self.end = audio_stream_end_time

    def get_audio_buffer_end_time(self) -> float:
        return self.buffer_time_offset + len(self.audio_buffer) / self.SAMPLING_RATE

    # ── Tier 1: Draft ─────────────────────────────────────────────────

    # Maximum audio for draft transcription (seconds). Caps RTF for large models.
    # 5s keeps RTF < 1.0 for Fun-ASR-MLT-Nano (800M params) on L40S GPU.
    MAX_DRAFT_AUDIO_SEC = 5.0

    def _transcribe_draft(self) -> str:
        buf_len = len(self.audio_buffer)
        buf_dur = buf_len / self.SAMPLING_RATE
        if buf_dur < MIN_AUDIO_SEC:
            return ""
        # Throttle: skip draft if less than 1.5s of new audio since last draft
        new_samples = buf_len - self._last_draft_samples
        if new_samples < int(1.5 * self.SAMPLING_RATE) and self._draft_text:
            return self._draft_text
        self._last_draft_samples = buf_len
        # Cap audio length to keep RTF bounded for large models
        max_samples = int(self.MAX_DRAFT_AUDIO_SEC * self.SAMPLING_RATE)
        audio = self.audio_buffer[-max_samples:] if buf_len > max_samples else self.audio_buffer
        res = self.asr.transcribe(audio, init_prompt="")
        tokens = self.asr.ts_words(res)
        if not tokens:
            return ""
        return self.asr.sep.join(t.text for t in tokens).strip()

    def get_buffer(self):
        return Transcript(None, None, self._draft_text)

    # ── Tier 2+3: Commit + Refine ────────────────────────────────────

    def _do_commit(self) -> Tuple[List[ASRToken], float]:
        """Execute commit: full-buffer transcription → optional Whisper refine → token."""
        end_time = self.get_audio_buffer_end_time()
        # Commit uses the FULL buffer (no sliding window) for best quality
        res = self.asr.transcribe(self.audio_buffer, init_prompt="")
        tokens = self.asr.ts_words(res)
        draft = self.asr.sep.join(t.text for t in tokens).strip() if tokens else ""
        draft = draft or self._draft_text
        if not draft:
            self.audio_buffer = np.array([], dtype=np.float32)
            self.buffer_time_offset = end_time
            self._draft_text = ""
            return [], end_time

        # Tier 3: Whisper refinement on the full buffer
        final_text = draft
        refined = self._refine(self.audio_buffer, draft)
        if refined:
            final_text = refined

        final_text = final_text.rstrip()
        if final_text and final_text[-1] not in '.!?。':
            final_text += '.'

        token = ASRToken(
            start=round(self.buffer_time_offset, 2),
            end=round(end_time, 2),
            text=final_text,
        )

        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_time_offset = end_time
        self._draft_text = ""

        return [token], end_time

    # ── VAD silence handling ──────────────────────────────────────────

    def start_silence(self):
        """VAD silence START — don't commit yet. Wait for end_silence
        to know the pause duration before deciding.
        """
        # Just refresh draft with latest transcription
        if len(self.audio_buffer) / self.SAMPLING_RATE >= MIN_AUDIO_SEC:
            self._draft_text = self._transcribe_draft()
        return [], self.get_audio_buffer_end_time()

    def end_silence(self, silence_duration, offset):
        """VAD silence END — now we know the pause duration.

        Decision matrix:
        - Micro-pause (<0.7s) AND short buffer (<8s): ignore, keep accumulating
        - Meaningful pause (≥0.7s) AND buffer ≥ 2s: COMMIT + REFINE
        - Any pause AND buffer ≥ 8s: COMMIT + REFINE (buffer safety)
        - Long silence (≥5s): full reset
        """
        if not silence_duration or silence_duration <= 0:
            return

        buffer_dur = len(self.audio_buffer) / self.SAMPLING_RATE
        is_meaningful_pause = silence_duration >= MEANINGFUL_PAUSE_SEC
        buffer_long_enough = buffer_dur >= MIN_COMMIT_SEC
        buffer_too_long = buffer_dur >= LONG_BUFFER_SEC

        should_commit = (
            (is_meaningful_pause and buffer_long_enough)
            or buffer_too_long
        )

        if should_commit:
            tokens, end_time = self._do_commit()
            self._pending_commit = tokens
            logger.info(
                f"Commit: pause={silence_duration:.2f}s buf={buffer_dur:.1f}s "
                f"→ {len(tokens)} tokens"
            )

        # Handle the silence gap itself
        if silence_duration >= MIN_DURATION_REAL_SILENCE:
            self.init(offset=silence_duration + offset)
        else:
            gap = np.zeros(
                int(self.SAMPLING_RATE * silence_duration), dtype=np.float32
            )
            self.insert_audio_chunk(gap)

    def new_speaker(self, change_speaker):
        self.process_iter()
        self.init(offset=change_speaker.start)

    # ── Main loop ─────────────────────────────────────────────────────

    def process_iter(self, is_last=False) -> Tuple[List[ASRToken], float]:
        """Called for each audio chunk.

        Returns pending commit tokens (from end_silence) or triggers
        safety commits for long buffers / end-of-stream.
        """
        end_time = self.get_audio_buffer_end_time()
        buffer_dur = len(self.audio_buffer) / self.SAMPLING_RATE

        # Deliver pending commit from end_silence
        if self._pending_commit:
            tokens = self._pending_commit
            self._pending_commit = []
            return tokens, end_time

        if buffer_dur < MIN_AUDIO_SEC:
            return [], end_time

        # Tier 1: draft transcription
        draft = self._transcribe_draft()
        self._draft_text = draft

        # Safety / end-of-stream commit
        if buffer_dur >= WINDOW_MAX_SEC or is_last:
            if draft:
                return self._do_commit()

        return [], end_time

    def warmup(self, audio, init_prompt=""):
        pass
