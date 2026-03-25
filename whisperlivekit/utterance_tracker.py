"""utterance-style streaming utterance protocol for WhisperLiveKit.

Transforms FrontData updates into utterance-based messages with sequential
``seq`` numbers, partial/final status, and millisecond timing.

Protocol
--------
Opt-in via query parameter: ``ws://host:port/asr?mode=utterance``

Each message:

.. code-block:: json

    {
        "seq": 0,
        "start_at": 1500,
        "duration": 0,
        "final": false,
        "alternatives": [{
            "text": "partial text...",
            "confidence": 0.0,
            "words": []
        }]
    }

Lifecycle:
1. While speech is ongoing, emit ``partial`` messages (``final=false``,
   ``duration=0``) with the current utterance text.
2. When an utterance boundary is detected -- either by silence (EPD) or by
   exceeding ``max_utterance_duration`` -- emit a ``final`` message
   (``final=true``, ``duration`` set to actual ms) and increment ``seq``.
3. The next speech resumes a new utterance at the next ``seq``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from whisperlivekit.repetition_filter import remove_repetitions
from whisperlivekit.timed_objects import FrontData, Segment


def _time_str_to_seconds(time_str: str) -> float:
    """Convert ``H:MM:SS.cc`` format string to seconds.

    The FrontData/Segment ``start``/``end`` fields use the ``format_time``
    helper which produces strings like ``0:00:03.50``.  We need numeric
    seconds for duration math.
    """
    m = re.match(r"(\d+):(\d+):(\d+)\.(\d+)", time_str)
    if not m:
        # Fallback: try interpreting as a plain float
        try:
            return float(time_str)
        except (TypeError, ValueError):
            return 0.0
    hours, minutes, seconds, centiseconds = (
        int(m.group(1)),
        int(m.group(2)),
        int(m.group(3)),
        int(m.group(4)),
    )
    return hours * 3600 + minutes * 60 + seconds + centiseconds / 100.0


def _seconds_to_ms(seconds: float) -> int:
    """Convert seconds to integer milliseconds."""
    return int(round(seconds * 1000))


@dataclass
class _WordEntry:
    """A single word with timing, used in ``final`` alternatives."""

    text: str
    start_at: int  # ms
    duration: int  # ms
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "start_at": self.start_at,
            "duration": self.duration,
            "confidence": self.confidence,
        }


@dataclass
class _UtteranceState:
    """Internal bookkeeping for the current in-progress utterance."""

    seq: int = 0
    start_s: float = 0.0  # seconds from stream start
    text: str = ""
    # Track line-level word data for final emission
    words: List[_WordEntry] = field(default_factory=list)
    # Track which committed lines have already been absorbed so we don't
    # double-count on subsequent updates.
    absorbed_line_count: int = 0
    last_end_s: float = 0.0  # latest end time seen (seconds)
    silence_start_s: Optional[float] = None  # when silence began


class UtteranceTracker:
    """Tracks utterance boundaries and formats utterance-based messages.

    Parameters
    ----------
    epd_threshold:
        Seconds of silence required to finalize an utterance (End-Point
        Detection).  Defaults to 0.5 s.
    max_utterance_duration:
        If continuous speech exceeds this many seconds the tracker will
        attempt to split the utterance at a natural sentence boundary.
        Defaults to 15.0 s.
    sentence_splitter:
        An optional callable that takes a string and returns a list of
        sentence strings.  Used when ``max_utterance_duration`` forces a
        split so that the break happens at a natural boundary.  When
        ``None``, punctuation-based heuristics are used instead.
    """

    def __init__(
        self,
        epd_threshold: float = 0.5,
        max_utterance_duration: float = 15.0,
        sentence_splitter: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        self._epd_threshold = epd_threshold
        self._max_utterance_duration = max_utterance_duration
        self._sentence_splitter = sentence_splitter

        self._next_seq: int = 0
        self._current: _UtteranceState = self._new_utterance(start_s=0.0)
        self._stream_started: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_update(
        self,
        front_data_dict: Dict[str, Any],
        is_silence: bool = False,
    ) -> List[Dict[str, Any]]:
        """Process a serialized FrontData update and return utterance messages.

        Parameters
        ----------
        front_data_dict:
            The result of ``FrontData.to_dict()``.
        is_silence:
            ``True`` when VAD indicates silence in the current frame.

        Returns
        -------
        A list of zero or more utterance-formatted message dicts.  Typically
        one partial *or* one final+partial pair.
        """
        messages: List[Dict[str, Any]] = []

        lines: List[Dict[str, Any]] = front_data_dict.get("lines", [])
        buffer_text: str = (front_data_dict.get("buffer_transcription") or "").strip()

        # ---- Gather text from committed lines + buffer ----
        full_text, last_end_s = self._gather_text(lines, buffer_text)

        if last_end_s > 0:
            self._current.last_end_s = max(self._current.last_end_s, last_end_s)

        if not self._stream_started and full_text:
            self._stream_started = True

        # ---- Silence tracking for EPD ----
        if is_silence:
            if self._current.silence_start_s is None:
                self._current.silence_start_s = self._current.last_end_s
            silence_dur = self._current.last_end_s - self._current.silence_start_s
            if (
                silence_dur >= self._epd_threshold
                and self._current.text
            ):
                # Finalize the current utterance via EPD
                messages.append(self._finalize(lines))
                # Reset silence tracking on the fresh utterance
                self._current.silence_start_s = None
                return messages
        else:
            self._current.silence_start_s = None

        # ---- Update current utterance text ----
        self._current.text = full_text

        # ---- Max-duration forced split ----
        utterance_dur_s = self._current.last_end_s - self._current.start_s
        if (
            utterance_dur_s > self._max_utterance_duration
            and self._current.text
        ):
            split_msgs = self._split_by_duration(lines)
            messages.extend(split_msgs)
            # After splitting, emit a partial for the remainder if any
            if self._current.text:
                messages.append(self._make_partial())
            return messages

        # ---- Emit partial ----
        if self._current.text:
            messages.append(self._make_partial())

        return messages

    def force_finalize(self) -> List[Dict[str, Any]]:
        """Force-finalize the current utterance (e.g. on disconnect).

        Returns a list containing a single final message, or an empty list
        if there is no active utterance.
        """
        if not self._current.text:
            return []
        return [self._finalize([])]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new_utterance(self, start_s: float) -> _UtteranceState:
        """Create a fresh utterance state and advance the sequence counter."""
        state = _UtteranceState(
            seq=self._next_seq,
            start_s=start_s,
            last_end_s=start_s,
        )
        self._next_seq += 1
        return state

    def _gather_text(
        self,
        lines: List[Dict[str, Any]],
        buffer_text: str,
    ) -> tuple[str, float]:
        """Combine committed lines + buffer into utterance text.

        Returns ``(combined_text, latest_end_seconds)``.
        """
        parts: List[str] = []
        latest_end_s: float = self._current.last_end_s

        for line in lines:
            text = (line.get("text") or "").strip()
            if not text:
                continue
            parts.append(text)
            end_val = line.get("end")
            if end_val is not None:
                end_s = (
                    _time_str_to_seconds(end_val)
                    if isinstance(end_val, str)
                    else float(end_val)
                )
                latest_end_s = max(latest_end_s, end_s)

        if buffer_text:
            parts.append(buffer_text)

        combined = remove_repetitions(" ".join(parts).strip())
        return combined, latest_end_s

    def _build_words(self, lines: List[Dict[str, Any]]) -> List[_WordEntry]:
        """Build word-level entries from committed lines for final messages."""
        words: List[_WordEntry] = []
        for line in lines:
            text = (line.get("text") or "").strip()
            if not text:
                continue
            start_val = line.get("start")
            end_val = line.get("end")
            start_s = (
                _time_str_to_seconds(start_val) if isinstance(start_val, str) else float(start_val or 0)
            )
            end_s = (
                _time_str_to_seconds(end_val) if isinstance(end_val, str) else float(end_val or 0)
            )
            line_dur_ms = _seconds_to_ms(end_s - start_s)
            # Split the line text into individual words and distribute
            # timing proportionally.
            line_words = text.split()
            n = len(line_words)
            if n == 0:
                continue
            word_dur = line_dur_ms // n if n else 0
            for i, w in enumerate(line_words):
                w_start_ms = _seconds_to_ms(start_s) + i * word_dur
                w_dur = word_dur if i < n - 1 else (line_dur_ms - i * word_dur)
                words.append(
                    _WordEntry(text=w, start_at=w_start_ms, duration=max(0, w_dur))
                )
        return words

    def _make_partial(self) -> Dict[str, Any]:
        """Build a partial (non-final) utterance message for the current state."""
        return {
            "seq": self._current.seq,
            "start_at": _seconds_to_ms(self._current.start_s),
            "duration": 0,
            "final": False,
            "alternatives": [
                {
                    "text": self._current.text,
                    "confidence": 0.0,
                    "words": [],
                }
            ],
        }

    def _make_final(
        self,
        text: str,
        start_s: float,
        end_s: float,
        words: List[_WordEntry],
    ) -> Dict[str, Any]:
        """Build a final utterance message."""
        duration_ms = max(0, _seconds_to_ms(end_s - start_s))
        avg_confidence = 0.0
        if words:
            avg_confidence = sum(w.confidence for w in words) / len(words)
        return {
            "seq": self._current.seq,
            "start_at": _seconds_to_ms(start_s),
            "duration": duration_ms,
            "final": True,
            "alternatives": [
                {
                    "text": text.strip(),
                    "confidence": avg_confidence,
                    "words": [w.to_dict() for w in words],
                }
            ],
        }

    def _finalize(self, lines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Finalize the current utterance and prepare a new one."""
        words = self._build_words(lines)
        msg = self._make_final(
            text=self._current.text,
            start_s=self._current.start_s,
            end_s=self._current.last_end_s,
            words=words,
        )
        # Start the next utterance right after the current one ends
        self._current = self._new_utterance(start_s=self._current.last_end_s)
        return msg

    def _split_by_duration(
        self, lines: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Force-split a long utterance, preferring a natural sentence break.

        Returns a list of final messages (usually one), and updates
        ``self._current`` to hold only the remainder text.
        """
        text = self._current.text
        messages: List[Dict[str, Any]] = []

        split_point = self._find_split_point(text)

        if split_point is not None and 0 < split_point < len(text):
            finalized_text = text[:split_point].strip()
            remainder_text = text[split_point:].strip()
        else:
            # No good split point: finalize everything
            finalized_text = text
            remainder_text = ""

        # Estimate timing for the split.  We distribute duration
        # proportionally by character count.
        total_dur_s = self._current.last_end_s - self._current.start_s
        if text:
            ratio = len(finalized_text) / len(text)
        else:
            ratio = 1.0
        split_end_s = self._current.start_s + total_dur_s * ratio

        # Build word entries only from the finalized portion
        words = self._build_words(lines)
        # Trim words to roughly match finalized text word count
        finalized_word_count = len(finalized_text.split())
        trimmed_words = words[:finalized_word_count]

        msg = self._make_final(
            text=finalized_text,
            start_s=self._current.start_s,
            end_s=split_end_s,
            words=trimmed_words,
        )
        messages.append(msg)

        # Start the next utterance with the remainder
        self._current = self._new_utterance(start_s=split_end_s)
        self._current.text = remainder_text
        self._current.last_end_s = self._current.start_s + total_dur_s * (1.0 - ratio)

        return messages

    def _find_split_point(self, text: str) -> Optional[int]:
        """Find the best character index to split ``text``.

        Tries the sentence splitter first, then falls back to
        punctuation-based heuristics.

        Returns the character index *after* the split (i.e. the start of the
        remainder), or ``None`` if no good split was found.
        """
        if not text:
            return None

        # Strategy 1: use the provided sentence splitter
        if self._sentence_splitter is not None:
            try:
                sentences = self._sentence_splitter(text)
                if sentences and len(sentences) > 1:
                    # Split after the first sentence
                    first = sentences[0]
                    idx = text.find(first)
                    if idx >= 0:
                        return idx + len(first)
            except Exception:
                pass  # fall through to heuristic

        # Strategy 2: punctuation-based heuristic -- find the last
        # sentence-ending punctuation mark that is at least 40% into the
        # text (to avoid trivially short first segments).
        min_pos = len(text) * 2 // 5
        best: Optional[int] = None
        for i, ch in enumerate(text):
            if ch in ".!?\u3002\uff01\uff1f" and i >= min_pos:
                best = i + 1  # split after the punctuation
        if best is not None:
            return best

        # Strategy 3: split at the last space past the midpoint
        mid = len(text) // 2
        last_space = text.rfind(" ", mid)
        if last_space > 0:
            return last_space + 1

        return None

    def reset(self) -> None:
        """Reset all internal state."""
        self._next_seq = 0
        self._current = self._new_utterance(start_s=0.0)
        self._stream_started = False
