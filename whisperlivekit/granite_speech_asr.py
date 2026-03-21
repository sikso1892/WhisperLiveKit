"""
IBM Granite 4.0 1B Speech ASR backend (batch-only).

Uses HuggingFace Transformers for inference. Best English ASR quality:
WER 1.18% LibriSpeech clean, 3.75% other (100 samples).

Supported languages: en, fr, de, es, pt, ja.

Usage:
    wlk serve --backend granite-speech
"""

import logging
import sys
from typing import List, Optional

import numpy as np

from whisperlivekit.local_agreement.backends import ASRBase
from whisperlivekit.timed_objects import ASRToken

logger = logging.getLogger(__name__)

GRANITE_SUPPORTED_LANGUAGES = {"en", "fr", "de", "es", "pt", "ja"}


class GraniteSpeechASR(ASRBase):
    """Granite 4.0 1B Speech batch ASR via HuggingFace Transformers."""

    sep = ""
    SAMPLING_RATE = 16000

    def __init__(self, lan="en", model_size=None, cache_dir=None,
                 model_dir=None, logfile=sys.stderr, **kwargs):
        self.logfile = logfile
        self.transcribe_kargs = {}
        self.original_language = None if lan == "auto" else lan
        self.model = self.load_model(model_size, cache_dir, model_dir)

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        model_id = model_dir or "ibm-granite/granite-4.0-1b-speech"

        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16
        else:
            device = "cpu"
            dtype = torch.float32

        logger.info("Loading Granite Speech: %s (%s, %s)", model_id, dtype, device)
        self._processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, device_map=device, dtype=dtype
        )
        model.eval()
        self._device = device
        logger.info("Granite Speech loaded")
        return model

    def transcribe(self, audio: np.ndarray, init_prompt: str = ""):
        import torch

        wav_tensor = torch.from_numpy(audio).unsqueeze(0)

        user_prompt = "<|audio|>can you transcribe the speech into a written format?"
        chat = [{"role": "user", "content": user_prompt}]
        prompt = self._processor.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self._processor(
            prompt, wav_tensor, device=self._device, return_tensors="pt"
        ).to(self._device)

        with torch.inference_mode():
            model_outputs = self.model.generate(
                **model_inputs, max_new_tokens=200, do_sample=False, num_beams=1
            )

        num_input_tokens = model_inputs["input_ids"].shape[-1]
        new_tokens = model_outputs[0, num_input_tokens:].unsqueeze(0)
        text = self._processor.tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0].strip()

        audio_duration = len(audio) / self.SAMPLING_RATE
        return {"text": text, "duration": audio_duration}

    def ts_words(self, result) -> List[ASRToken]:
        text = result.get("text", "").strip()
        if not text:
            return []

        duration = result.get("duration", 1.0)
        words = text.split()
        if not words:
            return []

        # Granite doesn't provide word-level timestamps, distribute evenly
        time_per_word = duration / len(words)
        tokens = []
        for i, word in enumerate(words):
            start = round(i * time_per_word, 2)
            end = round((i + 1) * time_per_word, 2)
            tokens.append(ASRToken(start=start, end=end, text=word))
        return tokens

    def segments_end_ts(self, result) -> List[float]:
        duration = result.get("duration", 1.0)
        return [duration]

    def use_vad(self):
        return True
