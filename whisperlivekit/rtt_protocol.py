"""RTT (Real-Time Translation) WebSocket protocol adapter.

Transforms WhisperLiveKit's FrontData updates into the RTT speech-session
protocol: transcript (partial), transcript_end (final), and finish
(translation) events.

Uses UtteranceTracker internally for utterance boundary detection (EPD).
When an utterance finalizes, transcript_end is emitted and translation
is dispatched asynchronously.

Protocol spec:
  - transcript:      intermediate STT result (non_final_text updates)
  - transcript_end:  final STT result for a complete utterance
  - finish:          translation result, matched by transcript_id
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from whisperlivekit.utterance_tracker import UtteranceTracker


@dataclass
class RTTConfig:
    """Per-session RTT configuration from the connect event."""
    connection_id: str = ""
    hint_lang_code_list: List[str] = field(default_factory=list)
    tgt_lang_code_list: List[str] = field(default_factory=list)
    epd_threshold: float = 0.5
    max_utterance_duration: float = 15.0


class RTTProtocol:
    """Converts FrontData updates into RTT protocol messages.

    Parameters
    ----------
    config:
        Per-session RTT configuration parsed from the connect event.
    sentence_splitter:
        Optional callable for natural sentence boundary detection.
    """

    def __init__(
        self,
        config: RTTConfig,
        sentence_splitter: Optional[Callable] = None,
    ):
        self._config = config
        self._connection_id = config.connection_id or str(uuid.uuid4())

        # Internal utterance tracker for EPD and boundary detection
        self._tracker = UtteranceTracker(
            epd_threshold=config.epd_threshold,
            max_utterance_duration=config.max_utterance_duration,
            sentence_splitter=sentence_splitter,
        )

        # Current partial state
        self._current_transcript_id = str(uuid.uuid4())
        self._current_text = ""
        self._detected_language = ""

    def process_update(
        self,
        front_data_dict: Dict[str, Any],
        is_silence: bool = False,
    ) -> List[Dict[str, Any]]:
        """Process a FrontData update and return RTT protocol messages.

        Returns a list of transcript and/or transcript_end events.
        Translation (finish) events are NOT included here — they are
        handled asynchronously by the caller after transcript_end.
        """
        messages: List[Dict[str, Any]] = []

        # Delegate to UtteranceTracker for boundary detection
        utterance_msgs = self._tracker.process_update(front_data_dict, is_silence)

        for msg in utterance_msgs:
            if msg.get("final"):
                # Utterance finalized → emit transcript_end
                messages.append(self._make_transcript_end(msg))
                # Rotate transcript_id for next utterance
                self._current_transcript_id = str(uuid.uuid4())
                self._current_text = ""
            else:
                # Partial → emit transcript
                messages.append(self._make_transcript(msg))

        return messages

    def force_finalize(self) -> List[Dict[str, Any]]:
        """Force-finalize on disconnect/stop. Returns transcript_end if pending."""
        messages: List[Dict[str, Any]] = []
        for msg in self._tracker.force_finalize():
            messages.append(self._make_transcript_end(msg))
        return messages

    @staticmethod
    def make_ready() -> Dict[str, Any]:
        """Create the ready_for_transcript event."""
        return {"event": "ready_for_transcript"}

    @staticmethod
    def make_finish(
        transcript_id: str,
        src_text: str,
        src_lang_code: str,
        translations: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Create a finish event with translation results.

        Parameters
        ----------
        transcript_id:
            Must match the transcript_end it corresponds to.
        src_text:
            Source transcription text.
        src_lang_code:
            Detected or hinted source language code.
        translations:
            List of {"lang_code": "en", "text": "..."} dicts.
        """
        return {
            "event": "finish",
            "payload": {
                "chat_type": "message",
                "transcript_id": transcript_id,
                "src_text": src_text,
                "src_lang_code": src_lang_code,
                "translation_list": translations,
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_text_and_language(self, msg: Dict[str, Any]) -> tuple:
        """Extract text and language from an UtteranceTracker message."""
        text = ""
        alts = msg.get("alternatives", [])
        if alts:
            text = alts[0].get("text", "")
        return text

    def _make_transcript(self, utterance_msg: Dict[str, Any]) -> Dict[str, Any]:
        """Convert UtteranceTracker partial to RTT transcript event."""
        text = self._extract_text_and_language(utterance_msg)
        self._current_text = text

        # Determine language from hints or detected
        lang_code = self._detected_language or (
            self._config.hint_lang_code_list[0]
            if self._config.hint_lang_code_list
            else ""
        )

        return {
            "event": "transcript",
            "data": {
                "connection_id": self._connection_id,
                "transcript_id": self._current_transcript_id,
                "text": text,
                "final_text": "",
                "non_final_text": text,
                "duration": 0,
                "language_code": lang_code,
                "speaker": "1",
            },
        }

    def _make_transcript_end(self, utterance_msg: Dict[str, Any]) -> Dict[str, Any]:
        """Convert UtteranceTracker final to RTT transcript_end event."""
        text = self._extract_text_and_language(utterance_msg)
        if not text:
            text = self._current_text

        offset_ms = utterance_msg.get("start_at", 0)
        duration_ms = utterance_msg.get("duration", 0)

        lang_code = self._detected_language or (
            self._config.hint_lang_code_list[0]
            if self._config.hint_lang_code_list
            else ""
        )

        return {
            "event": "transcript_end",
            "data": {
                "connection_id": self._connection_id,
                "transcript_id": self._current_transcript_id,
                "text": text,
                "offset_ms": offset_ms,
                "duration": duration_ms,
                "language_code": lang_code,
                "speaker": "1",
            },
        }

    def set_detected_language(self, lang_code: str):
        """Update detected language (called when ASR detects language)."""
        self._detected_language = lang_code
