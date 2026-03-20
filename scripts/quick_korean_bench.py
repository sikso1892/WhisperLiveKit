"""Quick Korean quality comparison between ASR backends.

Usage:
    python scripts/quick_korean_bench.py /path/to/korean_audio.wav

Compares Fun-ASR-MLT-Nano vs SenseVoiceSmall (if available) on the same audio.
"""

import sys
import time

import numpy as np
import soundfile as sf


def test_funasr_nano(audio, sr):
    """Test Fun-ASR-MLT-Nano-2512."""
    try:
        sys.path.insert(0, "/tmp/Fun-ASR")
        from model import FunASRNano

        model_dir = "/home/moonsik/.cache/modelscope/hub/models/FunAudioLLM/Fun-ASR-MLT-Nano-2512"
        m, kwargs = FunASRNano.from_pretrained(model=model_dir, device="cuda:0")
        m.eval()

        sf.write("/tmp/_bench.wav", audio, sr)
        t0 = time.time()
        res = m.inference(data_in=["/tmp/_bench.wav"], language="韩文", **kwargs)
        elapsed = time.time() - t0
        text = res[0][0]["text"]
        return text, elapsed
    except Exception as e:
        return f"Error: {e}", 0


def test_sensevoice(audio, sr):
    """Test SenseVoiceSmall."""
    try:
        from funasr import AutoModel

        model = AutoModel(model="iic/SenseVoiceSmall", device="cuda:1", disable_update=True)
        t0 = time.time()
        res = model.generate(input=audio, cache={}, language="ko", use_itn=True)
        elapsed = time.time() - t0

        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        text = rich_transcription_postprocess(res[0]["text"])
        return text, elapsed
    except Exception as e:
        return f"Error: {e}", 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/quick_korean_bench.py <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]
    audio, sr = sf.read(audio_path)
    duration = len(audio) / sr
    print(f"Audio: {audio_path} ({duration:.1f}s, sr={sr})")
    print()

    # Test different chunk sizes
    for chunk_sec in [None, 5, 3]:
        label = "full" if chunk_sec is None else f"{chunk_sec}s"
        test_audio = audio if chunk_sec is None else audio[: int(chunk_sec * sr)]
        test_dur = len(test_audio) / sr

        print(f"=== {label} ({test_dur:.1f}s) ===")

        text_nano, t_nano = test_funasr_nano(test_audio, sr)
        rtf_nano = t_nano / test_dur if test_dur > 0 else 0
        print(f"  Fun-ASR-Nano ({t_nano:.2f}s, RTF={rtf_nano:.3f}): {text_nano[:200]}")

        text_sv, t_sv = test_sensevoice(test_audio, sr)
        rtf_sv = t_sv / test_dur if test_dur > 0 else 0
        print(f"  SenseVoice  ({t_sv:.2f}s, RTF={rtf_sv:.3f}): {text_sv[:200]}")
        print()


if __name__ == "__main__":
    import warnings
    import logging

    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    main()
