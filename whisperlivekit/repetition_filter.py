"""Post-processing filter to remove stuttered/repeated phrases from transcriptions.

Targets patterns like:
- "안보 안보리" → "안보리"
- "가 가 가슴이" → "가슴이"
- "트 트럼프는" → "트럼프는"
- "자체 자체 자체" → "자체"
- "형해 형해와" → "형해와"
"""

import re


def remove_repetitions(text: str) -> str:
    """Remove stuttered word repetitions from transcription text.

    Handles both exact repetitions ("자체 자체 자체") and prefix
    stutters ("트 트럼프는", "안보 안보리") common in Whisper output.
    """
    if not text:
        return text

    words = text.split()
    if len(words) <= 1:
        return text

    result = []
    i = 0
    while i < len(words):
        word = words[i]

        # Look ahead for exact repetitions: "자체 자체 자체" → keep last
        j = i + 1
        while j < len(words) and words[j] == word:
            j += 1

        if j > i + 1:
            # Had exact repeats. Check if next word starts with this word (stutter prefix)
            if j < len(words) and words[j].startswith(word) and len(words[j]) > len(word):
                # "안보 안보리" → skip to "안보리"
                i = j
                continue
            else:
                # "자체 자체 자체" → keep one
                result.append(word)
                i = j
                continue

        # Check for prefix stutter: "트 트럼프는" where word is prefix of next
        if i + 1 < len(words):
            next_word = words[i + 1]
            if (
                next_word.startswith(word)
                and len(next_word) > len(word)
                and len(word) <= 2  # single syllable stutters only
            ):
                # Skip the stutter prefix, next iteration picks up the full word
                i += 1
                continue

        result.append(word)
        i += 1

    return " ".join(result)


def clean_korean_text(text: str) -> str:
    """Clean up Korean transcription artifacts.

    Fixes:
    - Spurious mid-sentence periods: "되 고. 꺼" → "되고 꺼"
    - Fragmented syllables: "멈추 는게" → "멈추는게" (single jamo followed by particle)
    - Repeated sentence endings: "합니다. 합니다" → "합니다."
    - Excessive punctuation: "있지 않습니까?" stays, "는." → "는"
    """
    if not text:
        return text

    # Remove spurious periods between Korean text fragments,
    # but NOT after known sentence-ending patterns (다, 요, 죠, etc.)
    # e.g., "되고. 꺼" → "되고 꺼" but "합니다. 그래서" stays
    text = re.sub(r'(?<=[가-힣])(?<![다요죠까])\.\s+(?=[가-힣])', ' ', text)

    # Remove trailing single-character periods: "는." at end of fragment
    text = re.sub(r'\b([가-힣])\.\s', r'\1 ', text)

    # Merge fragmented Korean particles: "되 고" → "되고", "해야 되" → "해야 되"
    # Only merge single jamo/syllable fragments followed by common particles
    particles = r'(?:은|는|이|가|을|를|에|의|도|와|과|로|고|며|면|서|지|만|요|죠|니다|습니다|겠|든|게|거|지만|에서|으로|까지|부터|보다|처럼|같이|대로|만큼)'
    text = re.sub(rf'([가-힣])\s+({particles})(?=\s|$|[,.])', r'\1\2', text)

    # Remove repeated full phrases at sentence level
    # "합니다. 합니다" → "합니다."
    text = re.sub(r'(\S{2,}[.!?])\s+\1', r'\1', text)

    # Clean up multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)

    return text.strip()
