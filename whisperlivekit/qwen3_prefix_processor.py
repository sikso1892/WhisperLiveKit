"""
Qwen3-ASR prefix-constrained streaming processor.

Uses the Qwen3-ASR SDK's streaming approach without requiring the SDK:
instead of LocalAgreement (comparing consecutive hypotheses), it appends
previous decoded text as a prefix in the assistant response, rolling back
last K tokens for self-correction.

This ensures cross-chunk consistency at the model level rather than relying
on text-level agreement, which is especially beneficial for agglutinative
languages like Korean where morphological variation defeats exact token match.

Benchmark results (L40S, 100 samples, BF16, ucn=4):
  FLEURS-ko: CER 2.96% (vs LocalAgreement 4.26%, batch 2.80%)
  LS-clean:  WER 2.69% (50 samples, vs LocalAgreement 6.08%)
  RTF: ~0.060 (vs LocalAgreement 0.067)

Usage:
    wlk serve --backend qwen3-vllm-prefix --model 1.7b --lan ko
"""

import logging
import re
import sys
from typing import List, Optional, Tuple

import numpy as np

from whisperlivekit.timed_objects import ASRToken, ChangeSpeaker, Transcript

logger = logging.getLogger(__name__)


def _parse_qwen3_output(text: str) -> str:
    """Parse Qwen3-ASR output: 'language English<asr_text>transcribed text'."""
    for marker in ("<|asr_text|>", "<asr_text>"):
        if marker in text:
            text = text.split(marker, 1)[1].strip()
            break
    text = re.sub(r"^language\s+\w+\s*", "", text).strip()
    return text


class Qwen3PrefixOnlineProcessor:
    """Per-session online processor using prefix-constrained decoding.

    Instead of LocalAgreement's hypothesis comparison, this processor:
    1. Accumulates all audio seen so far
    2. Appends previous decoded text (minus last K tokens) as assistant prefix
    3. Model generates continuation after the prefix
    4. This ensures cross-chunk consistency at the model level

    Commit strategy — lazy commit:
    During streaming, no text is committed — all shown as buffer.
    On silence/finish, the final text is committed unconditionally.
    This avoids locking in wrong text when the model self-corrects.

    Long-form stability: automatic session reset when accumulated audio
    exceeds max_session_audio_sec (default 30s).
    """

    SAMPLING_RATE = 16000
    MIN_DURATION_REAL_SILENCE = 5

    def __init__(
        self,
        asr,
        unfixed_chunk_num: int = 4,
        unfixed_token_num: int = 5,
        css_initial: float = 2.0,
        css_steady: float = 4.0,
        max_session_audio_sec: float = 30.0,
    ):
        self.asr = asr
        self._ucn = unfixed_chunk_num
        self._utn = unfixed_token_num
        self._css_initial = css_initial
        self._css_steady = css_steady
        self._max_session_audio_sec = max_session_audio_sec

        # Load tokenizer for prefix token rollback
        from transformers import AutoTokenizer

        model_size = getattr(asr, '_model_size', None)
        if model_size and "1.7" in str(model_size):
            model_id = "Qwen/Qwen3-ASR-1.7B"
        else:
            model_id = "Qwen/Qwen3-ASR-0.6B"
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )

        logger.info(
            "Prefix-constrained processor: ucn=%d, utn=%d, css=%.1f→%.1f, max_session=%.0fs",
            unfixed_chunk_num, unfixed_token_num, css_initial, css_steady, max_session_audio_sec,
        )

        self._init_state()

    def _init_state(self):
        """Initialize or reset streaming state."""
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self._raw_decoded = ""
        self._committed_text = ""
        self._chunk_id = 0
        self._inferences = 0
        self._total_inserted_sec = 0.0
        self._last_process_sec = 0.0
        self._speaker = -1
        self._global_time_offset = 0.0
        self.end = 0.0
        self.buffer: List[ASRToken] = []

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
        self.audio_buffer = np.append(self.audio_buffer, audio.astype(np.float32))
        self._total_inserted_sec += len(audio) / self.SAMPLING_RATE

    def _build_prefix(self) -> str:
        """Build prefix from previous decoded text with token rollback.

        Skips prefix for the first `unfixed_chunk_num` inferences.
        For subsequent inferences, encodes the accumulated text, removes
        the last `unfixed_token_num` tokens, and decodes back to text.
        """
        if self._chunk_id < self._ucn or not self._raw_decoded:
            return ""

        cur_ids = self._tokenizer.encode(self._raw_decoded)
        k = self._utn
        while True:
            end_idx = max(0, len(cur_ids) - k)
            prefix = self._tokenizer.decode(cur_ids[:end_idx]) if end_idx > 0 else ""
            if '\ufffd' not in prefix:
                break
            if end_idx == 0:
                prefix = ""
                break
            k += 1
        return prefix

    def _run_inference(self):
        """Run a single inference with prefix-constrained decoding."""
        prefix = self._build_prefix()
        prompt = self.asr._prompt_tpl + prefix

        inp = {"prompt": prompt, "multi_modal_data": {"audio": self.audio_buffer.copy()}}
        outputs = self.asr._llm.generate([inp], self.asr._sp, use_tqdm=False)
        gen_text = outputs[0].outputs[0].text

        self._raw_decoded = (prefix + gen_text) if prefix else gen_text
        self._chunk_id += 1
        self._inferences += 1
        self._last_process_sec = self._total_inserted_sec

        # Safety valve: detect repetition (hallucination protection)
        parsed = _parse_qwen3_output(self._raw_decoded)
        if len(parsed) > 20:
            words = parsed.split()
            if len(words) > 4:
                last4 = " ".join(words[-4:])
                if parsed.count(last4) >= 3:
                    logger.warning("Repetition detected in prefix decode, resetting prefix")
                    self._raw_decoded = ""

    def _maybe_reset_session(self) -> List[ASRToken]:
        """Reset session if accumulated audio exceeds threshold."""
        max_samples = int(self._max_session_audio_sec * self.SAMPLING_RATE)
        if self._max_session_audio_sec <= 0 or len(self.audio_buffer) < max_samples:
            return []

        # Finalize current session
        current = _parse_qwen3_output(self._raw_decoded)
        tokens = self._extract_committed_tokens(current, is_last=True)

        logger.info(
            "Prefix processor session reset at %.1fs audio (text=%d chars)",
            len(self.audio_buffer) / self.SAMPLING_RATE,
            len(current),
        )

        self._init_state()
        return tokens

    def process_iter(self, is_last=False) -> Tuple[List[ASRToken], float]:
        """Process current audio buffer.

        Returns (committed_tokens, end_time). During streaming, returns
        empty tokens (all text shown as buffer). On is_last (silence/finish),
        commits the final text.
        """
        # Adaptive CSS: skip if not enough new audio
        if not is_last:
            new_sec = self._total_inserted_sec - self._last_process_sec
            threshold = self._css_initial if self._inferences == 0 else self._css_steady
            if new_sec < threshold:
                return [], self.end

        if self.audio_buffer.size == 0:
            return [], self.end

        # Check for session reset (long-form stability)
        reset_tokens = self._maybe_reset_session()

        # Run inference
        self._run_inference()

        if is_last:
            current = _parse_qwen3_output(self._raw_decoded)
            tokens = self._extract_committed_tokens(current, is_last=True)
            return reset_tokens + tokens, self.end

        return reset_tokens, self.end

    def _extract_committed_tokens(
        self, current_text: str, is_last: bool
    ) -> List[ASRToken]:
        """Extract committed tokens using lazy-commit strategy.

        Only commits on is_last (silence/finish). Returns the new text
        since last commit as a single ASRToken.
        """
        if not current_text or not is_last:
            return []

        committed = self._committed_text
        if current_text.startswith(committed):
            new_text = current_text[len(committed):]
        else:
            new_text = current_text

        if not new_text or not new_text.strip():
            return []

        if current_text.startswith(committed):
            self._committed_text = committed + new_text
        else:
            self._committed_text = current_text

        token = ASRToken(
            start=round(max(self.end - 2.0, 0.0), 2),
            end=round(self.end, 2),
            text=new_text.strip(),
            speaker=self._speaker,
        ).with_offset(self._global_time_offset)
        return [token]

    def get_buffer(self) -> Transcript:
        """Return current unfixed text as draft."""
        current = _parse_qwen3_output(self._raw_decoded) if self._raw_decoded else ""
        committed = self._committed_text
        if committed and current.startswith(committed):
            unfixed = current[len(committed):]
        elif committed:
            unfixed = current[min(len(committed), len(current)):]
        else:
            unfixed = current
        return Transcript(None, None, unfixed)

    def start_silence(self) -> Tuple[List[ASRToken], float]:
        """Finalize on silence start, reset session."""
        result = self.process_iter(is_last=True)
        self._init_state()
        return result

    def end_silence(self, silence_duration: float, offset: float):
        """Update time offset after silence ends."""
        self._global_time_offset = silence_duration + offset

    def new_speaker(self, change_speaker: ChangeSpeaker):
        """Handle speaker change: finalize and reset."""
        self.process_iter(is_last=True)
        self._init_state()
        self._speaker = change_speaker.speaker
        self._global_time_offset = change_speaker.start

    def finish(self) -> Tuple[List[ASRToken], float]:
        """Flush remaining text when processing ends."""
        return self.process_iter(is_last=True)
