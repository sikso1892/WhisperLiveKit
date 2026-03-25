"""Korean-aware sentence splitter for streaming transcription.

Rule-based splitting on sentence-ending patterns common in Korean
speech. No external model dependency required.
"""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Korean sentence-ending suffixes (sorted longest-first for greedy match)
_ENDERS = sorted([
    # -습니다 forms
    '었습니다', '겠습니다', '있습니다', '없습니다', '됐습니다',
    '하겠습니다', '드리겠습니다',
    '합니다', '습니다', '됩니다', '입니다',
    # -어요/-아요 general forms
    '했어요', '됐어요', '갔어요', '왔어요', '봤어요',
    '있어요', '없어요', '싶어요', '같아요', '좋아요',
    '거예요', '아니에요', '그래요', '맞아요',
    '어요', '아요',  # general polite endings
    # -요 forms
    '거든요', '잖아요', '는데요', '다고요', '라고요',
    '건데요', '했거든요', '있거든요',
    '해요', '돼요', '네요', '군요', '나요', '가요',
    # -죠 forms
    '하죠', '되죠', '이죠', '거죠', '그렇죠', '맞죠',
    # -고요 forms
    '했고요', '하고요', '되고요', '이고요',
    # -는데 forms (mid-sentence connectors that often end utterances in speech)
    '했는데', '였는데', '하는데', '졌는데', '았는데', '었는데',
    '인데', '한데', '같은데',
    # other
    '니까',
    # -시다 (e.g., 합시다)
    '시다',
], key=len, reverse=True)


class SentenceSplitter:
    """Rule-based Korean sentence splitter for streaming use."""

    def __init__(self, model_name: str = None, lang_code: str = None):
        pass

    def split(self, text: str) -> List[str]:
        """Split text into sentences at Korean ending patterns."""
        if not text or not text.strip():
            return []

        sentences = []
        remaining = text.strip()

        while remaining:
            # Find the earliest sentence-ending position
            best_pos = -1
            best_len = 0

            for ender in _ENDERS:
                idx = remaining.find(ender)
                if idx >= 0:
                    end_pos = idx + len(ender)
                    # Must be followed by space or end-of-string
                    if end_pos >= len(remaining) or remaining[end_pos] in ' \t\n':
                        if best_pos < 0 or idx < best_pos or (idx == best_pos and len(ender) > best_len):
                            best_pos = idx
                            best_len = len(ender)

            # Also check explicit punctuation followed by space
            for punc in ['. ', '? ', '! ']:
                idx = remaining.find(punc)
                if idx >= 0 and (best_pos < 0 or idx < best_pos):
                    best_pos = idx
                    best_len = 1  # just the punctuation char

            if best_pos >= 0:
                end = best_pos + best_len
                sentence = remaining[:end].strip()
                if sentence:
                    sentences.append(sentence)
                remaining = remaining[end:].strip()
            else:
                # No more split points
                if remaining.strip():
                    sentences.append(remaining.strip())
                break

        return sentences

    def ends_with_ender(self, text: str) -> bool:
        """Return True if text ends with a Korean sentence-ending pattern.

        Tolerates trailing punctuation (., !, ?) and/or whitespace.
        """
        if not text:
            return False
        # Strip trailing whitespace and punctuation to expose the ender
        stripped = text.rstrip()
        stripped = stripped.rstrip('.!?')
        for ender in _ENDERS:
            if stripped.endswith(ender):
                return True
        return False

    def split_incremental(self, text: str) -> Tuple[List[str], str]:
        """Split for streaming: (complete_sentences, remaining_partial).

        - If the whole text ends with an ender, return ([text], "").
        - If enders appear mid-text, split there: (complete, remaining).
        - If no ender is found, return ([], text).
        """
        if not text or not text.strip():
            return [], ""

        stripped = text.strip()

        # If the entire text ends with an ender, it is one complete sentence
        if self.ends_with_ender(stripped):
            sentences = self.split(stripped)
            # All pieces are complete (the last one also ends with ender)
            return sentences, ""

        # Otherwise, split and treat the last piece as partial
        sentences = self.split(stripped)

        if len(sentences) <= 1:
            # No ender found anywhere → everything is partial
            return [], stripped

        return sentences[:-1], sentences[-1]
