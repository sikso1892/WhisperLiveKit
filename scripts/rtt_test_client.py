#!/usr/bin/env python3
"""RTT speech-session WebSocket test client.

Usage:
    python scripts/rtt_test_client.py /tmp/korean_test.wav
    python scripts/rtt_test_client.py /tmp/korean_test.wav --url ws://localhost:8200/v1/realtime/speech-session
    python scripts/rtt_test_client.py /tmp/korean_test.wav --hint-lang ko --tgt-lang en ja
"""

import argparse
import asyncio
import json
import struct
import sys
import time

import numpy as np
import soundfile as sf

try:
    import websockets
except ImportError:
    print("pip install websockets")
    sys.exit(1)


async def run_session(args):
    url = f"{args.url}?token={args.token}"
    print(f"Connecting to {url}")

    async with websockets.connect(url) as ws:
        # 1. Send connect
        connect_msg = {
            "event": "connect",
            "data": {
                "hint_lang_code_list": args.hint_lang,
                "tgt_lang_code_list": args.tgt_lang,
            },
        }
        await ws.send(json.dumps(connect_msg))
        print(f"→ connect (hints={args.hint_lang}, targets={args.tgt_lang})")

        # 2. Wait for ready_for_transcript
        resp = json.loads(await ws.recv())
        assert resp["event"] == "ready_for_transcript", f"Expected ready, got {resp}"
        print(f"← ready_for_transcript")

        # 3. Send start
        await ws.send(json.dumps({"event": "start"}))
        print(f"→ start")

        # 4. Load and stream audio
        audio, sr = sf.read(args.audio, dtype="float32")
        if sr != 16000:
            # Simple resample
            ratio = sr / 16000
            new_len = int(len(audio) / ratio)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
            sr = 16000
        duration = len(audio) / sr
        print(f"Audio: {duration:.1f}s, sr={sr}")

        # Convert to PCM s16le
        pcm = (audio * 32767).astype(np.int16).tobytes()

        # Stream in chunks
        chunk_duration = args.chunk_ms / 1000.0
        chunk_bytes = int(chunk_duration * sr * 2)  # 2 bytes per sample (s16le)

        # Start receive task
        transcript_count = 0
        transcript_end_count = 0
        finish_count = 0

        async def receiver():
            nonlocal transcript_count, transcript_end_count, finish_count
            try:
                async for msg in ws:
                    data = json.loads(msg)
                    event = data.get("event", "")
                    if event == "transcript":
                        transcript_count += 1
                        text = data["data"].get("non_final_text", "")
                        lang = data["data"].get("language_code", "")
                        print(f"  ← transcript [{lang}]: {text[:80]}")
                    elif event == "transcript_end":
                        transcript_end_count += 1
                        text = data["data"].get("text", "")
                        tid = data["data"].get("transcript_id", "")[:8]
                        dur = data["data"].get("duration", 0)
                        print(f"  ← transcript_end [{tid}] ({dur}ms): {text[:100]}")
                    elif event == "finish":
                        finish_count += 1
                        payload = data.get("payload", {})
                        src = payload.get("src_text", "")[:50]
                        for tr in payload.get("translation_list", []):
                            print(f"  ← finish [{tr['lang_code']}]: {tr['text'][:80]}")
                    else:
                        print(f"  ← {event}: {json.dumps(data)[:100]}")
            except websockets.ConnectionClosed:
                pass

        recv_task = asyncio.create_task(receiver())

        # Send audio chunks at real-time speed
        t_start = time.time()
        offset = 0
        chunks_sent = 0
        while offset < len(pcm):
            chunk = pcm[offset:offset + chunk_bytes]
            await ws.send(chunk)
            offset += chunk_bytes
            chunks_sent += 1

            # Pace at real-time × speed factor
            expected_time = chunks_sent * chunk_duration / args.speed
            elapsed = time.time() - t_start
            if elapsed < expected_time:
                await asyncio.sleep(expected_time - elapsed)

        print(f"→ sent {chunks_sent} chunks ({duration:.1f}s audio)")

        # 5. Send stop
        await ws.send(json.dumps({"event": "stop"}))
        print(f"→ stop")

        # Wait for remaining events
        await asyncio.sleep(3.0)
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass

        print(f"\n--- {transcript_count} transcripts | {transcript_end_count} transcript_ends | {finish_count} finishes ---")


def main():
    parser = argparse.ArgumentParser(description="RTT speech-session test client")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--url", default="ws://localhost:8200/v1/realtime/speech-session")
    parser.add_argument("--token", default="test-token-123")
    parser.add_argument("--hint-lang", nargs="+", default=["ko"])
    parser.add_argument("--tgt-lang", nargs="+", default=["en"])
    parser.add_argument("--chunk-ms", type=int, default=500)
    parser.add_argument("--speed", type=float, default=2.0, help="Playback speed multiplier")
    args = parser.parse_args()

    asyncio.run(run_session(args))


if __name__ == "__main__":
    main()
