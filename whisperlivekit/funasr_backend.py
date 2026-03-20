"""FunASR backend for WhisperLiveKit.

Supports two model families:
  - SenseVoiceSmall (iic/SenseVoiceSmall) — fast, multilingual, rich text
  - Fun-ASR-MLT-Nano (FunAudioLLM/Fun-ASR-MLT-Nano-2512) — 800M params,
    31 languages including Korean, superior Korean quality

Install:
    pip install funasr

Usage:
    wlk serve --backend funasr --model iic/SenseVoiceSmall --lan ko
    wlk serve --backend funasr --model FunAudioLLM/Fun-ASR-MLT-Nano-2512 --lan ko
"""

import logging
import sys
from typing import List, Optional

import numpy as np

from whisperlivekit.timed_objects import ASRToken

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "iic/SenseVoiceSmall"

# Fun-ASR-MLT-Nano uses Chinese language names
_LANG_TO_FUNASR_NANO = {
    "ko": "韩文", "zh": "中文", "en": "英文", "ja": "日文",
    "vi": "越南文", "id": "印尼文", "th": "泰文", "ms": "马来文",
    "tl": "菲律宾文", "ar": "阿拉伯文", "hi": "印地文",
    "de": "德文", "fr": "法文", "es": "西班牙文", "pt": "葡萄牙文",
    "it": "意大利文", "nl": "荷兰文", "pl": "波兰文", "sv": "瑞典文",
    "da": "丹麦文", "fi": "芬兰文", "el": "希腊文", "hu": "匈牙利文",
    "cs": "捷克文", "sk": "斯洛伐克文", "ro": "罗马尼亚文",
    "bg": "保加利亚文", "hr": "克罗地亚文", "sl": "斯洛文尼亚文",
    "et": "爱沙尼亚文", "lv": "拉脱维亚文", "lt": "立陶宛文",
}


def _is_nano_model(model_id: str) -> bool:
    lower = model_id.lower()
    return "fun-asr" in lower and "nano" in lower


class FunASR:
    """FunASR backend using SenseVoiceSmall or Fun-ASR-MLT-Nano."""

    sep = " "
    SAMPLING_RATE = 16000

    def __init__(
        self,
        lan="auto",
        model_size=None,
        cache_dir=None,
        model_dir=None,
        lora_path=None,
        logfile=sys.stderr,
        **kwargs,
    ):
        self.logfile = logfile
        self.transcribe_kargs = {}
        self.lora_path = lora_path
        self.original_language = None if lan == "auto" else lan
        self._lan = lan

        if model_dir:
            self._model_id = model_dir
        elif model_size and "/" in model_size:
            self._model_id = model_size
        else:
            self._model_id = _DEFAULT_MODEL

        self._is_nano = _is_nano_model(self._model_id)
        self.model = self._load_model(cache_dir)

    def _load_model(self, cache_dir: Optional[str] = None):
        if self._is_nano:
            return self._load_nano_model(cache_dir)
        return self._load_automodel(cache_dir)

    def _load_automodel(self, cache_dir: Optional[str] = None):
        try:
            from funasr import AutoModel
        except ImportError:
            raise ImportError(
                "funasr is required for the FunASR backend. "
                "Install it with: pip install funasr"
            )

        logger.info(f"Loading FunASR model: {self._model_id}")
        kwargs = {"model": self._model_id, "device": "cuda:0"}
        if cache_dir:
            kwargs["hub"] = "hf"
            kwargs["cache_dir"] = cache_dir

        model = AutoModel(**kwargs)
        logger.info(f"FunASR model loaded: {self._model_id}")
        return model

    def _load_nano_model(self, cache_dir: Optional[str] = None):
        """Load Fun-ASR-MLT-Nano via AutoModel with trust_remote_code."""
        try:
            from funasr import AutoModel
        except ImportError:
            raise ImportError(
                "funasr is required for Fun-ASR-Nano. "
                "Install it with: pip install funasr"
            )

        logger.info(f"Loading Fun-ASR-Nano model: {self._model_id}")
        model = AutoModel(
            model=self._model_id,
            trust_remote_code=True,
            device="cuda:0",
            disable_update=True,
        )
        logger.info("Fun-ASR-Nano loaded via AutoModel")
        return model

    def transcribe(self, audio: np.ndarray, init_prompt: str = ""):
        if len(audio) == 0:
            return []

        self._audio_duration = len(audio) / self.SAMPLING_RATE

        if self._is_nano:
            return self._transcribe_nano(audio)
        return self._transcribe_automodel(audio)

    def _transcribe_automodel(self, audio: np.ndarray):
        kwargs = {
            "input": audio,
            "cache": {},
            "use_itn": True,
            "batch_size_s": 60,
        }
        if self.original_language:
            kwargs["language"] = self.original_language

        try:
            result = self.model.generate(**kwargs)
        except Exception as e:
            logger.error(f"FunASR transcribe error: {e}")
            return []

        if result and "sensevoice" in self._model_id.lower():
            try:
                from funasr.utils.postprocess_utils import rich_transcription_postprocess
                for r in result:
                    if "text" in r:
                        r["text"] = rich_transcription_postprocess(r["text"])
            except ImportError:
                pass

        return result

    def _transcribe_nano(self, audio: np.ndarray):
        import soundfile as sf
        import tempfile
        import os

        # Fun-ASR-Nano needs file path input
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f, audio, self.SAMPLING_RATE)
            tmp_path = f.name

        try:
            nano_lang = _LANG_TO_FUNASR_NANO.get(
                self.original_language or "", "英文"
            )

            if hasattr(self.model, 'generate'):
                # AutoModel path
                result = self.model.generate(
                    input=[tmp_path],
                    cache={},
                    batch_size=1,
                    language=nano_lang,
                    itn=True,
                )
            else:
                # Direct FunASRNano path
                result = self.model.inference(
                    data_in=[tmp_path],
                    language=nano_lang,
                    **self._nano_kwargs,
                )
                # Normalize output format
                if result and isinstance(result[0], list):
                    result = result[0]
        except Exception as e:
            logger.error(f"Fun-ASR-Nano transcribe error: {e}")
            return []
        finally:
            os.unlink(tmp_path)

        return result

    def ts_words(self, result) -> List[ASRToken]:
        tokens = []
        if not result:
            return tokens

        audio_dur = getattr(self, '_audio_duration', 0.0)

        for item in result:
            text = item.get("text", "").strip()
            if not text:
                continue

            timestamp = item.get("timestamp", [])
            sentence_info = item.get("sentence_info", [])

            if timestamp:
                words = text.split()
                for i, word in enumerate(words):
                    if i < len(timestamp):
                        start_s = timestamp[i][0] / 1000.0
                        end_s = timestamp[i][1] / 1000.0
                    elif tokens:
                        start_s = tokens[-1].end
                        end_s = start_s + 0.1
                    else:
                        start_s, end_s = 0.0, 0.1
                    tokens.append(ASRToken(start_s, end_s, word))
            elif sentence_info:
                for sent in sentence_info:
                    sent_text = sent.get("text", "").strip()
                    sent_start = sent.get("start", 0) / 1000.0
                    sent_end = sent.get("end", 0) / 1000.0
                    sent_words = sent_text.split()
                    if not sent_words:
                        continue
                    dur = (sent_end - sent_start) / len(sent_words)
                    for j, w in enumerate(sent_words):
                        s = sent_start + j * dur
                        tokens.append(ASRToken(s, s + dur, w))
            else:
                words = text.split()
                if not words:
                    continue
                dur = max(audio_dur, 1.0) / len(words)
                for j, w in enumerate(words):
                    s = j * dur
                    tokens.append(ASRToken(round(s, 3), round(s + dur, 3), w))

        return tokens

    def segments_end_ts(self, result) -> List[float]:
        ends = []
        if not result:
            return ends

        for item in result:
            sentence_info = item.get("sentence_info", [])
            if sentence_info:
                for sent in sentence_info:
                    ends.append(sent.get("end", 0) / 1000.0)
            elif item.get("timestamp"):
                ends.append(item["timestamp"][-1][1] / 1000.0)
            elif getattr(self, '_audio_duration', 0) > 0:
                ends.append(self._audio_duration)

        return ends

    def use_vad(self):
        """External VAD is used; FunASR's internal VAD is disabled."""
        return False
