import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from whisperlivekit import AudioProcessor, TranscriptionEngine, get_inline_ui_html, parse_args

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger("whisperlivekit.qwen3_asr").setLevel(logging.DEBUG)

config = parse_args()
transcription_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global transcription_engine
    transcription_engine = TranscriptionEngine(config=config)
    yield

app = FastAPI(
    title="Flitto On-Prem STT Demo",
    description=(
        "Real-time speech transcription API with WebSocket streaming and OpenAI-compatible REST endpoints.\n\n"
        "## Endpoints\n\n"
        "| Protocol | Path | Description |\n"
        "|----------|------|-------------|\n"
        "| REST | `POST /v1/audio/transcriptions` | OpenAI-compatible batch transcription |\n"
        "| REST | `GET /v1/models` | List available models |\n"
        "| REST | `GET /health` | Health check |\n"
        "| WebSocket | `ws://.../asr` | Live streaming transcription |\n"
        "| WebSocket | `ws://.../v1/listen` | Deepgram-compatible live transcription |\n\n"
        "## WebSocket Modes\n\n"
        "| Mode | Query | Description |\n"
        "|------|-------|-------------|\n"
        "| Full | `?mode=full` | Default - complete FrontData JSON per update |\n"
        "| Diff | `?mode=diff` | Bandwidth-efficient incremental diffs |\n"
        "| Utterance | `?mode=utterance` | Utterance mode: seq-based with partial/final |\n\n"
        "## Multi-user Support\n\n"
        "The server supports concurrent sessions. Each WebSocket connection creates an independent "
        "`AudioProcessor` pipeline with per-session language override via `SessionASRProxy`. "
        "The `TranscriptionEngine` singleton is shared across all sessions for efficient GPU utilization.\n\n"
        "## WebSocket Protocol\n\n"
        "1. Connect to `ws://<host>:<port>/asr?language=en&mode=full`\n"
        "2. Receive config message: `{\"type\": \"config\", \"useAudioWorklet\": bool, \"mode\": str}`\n"
        "3. Send binary audio frames\n"
        "4. Receive JSON transcription results\n"
        "5. On completion, receive `{\"type\": \"ready_to_stop\"}`\n"
    ),
    version="0.2.20",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Transcription", "description": "Audio transcription endpoints (OpenAI-compatible)"},
        {"name": "Models", "description": "Model information"},
        {"name": "Health", "description": "Server health and status"},
        {"name": "WebSocket", "description": "Real-time streaming transcription via WebSocket"},
        {"name": "UI", "description": "Web-based transcription interface"},
    ],
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["UI"], summary="Web UI", include_in_schema=False)
async def get():
    return HTMLResponse(get_inline_ui_html())


@app.get("/health", tags=["Health"], summary="Health check")
async def health():
    """Returns server status, configured backend, and readiness state."""
    global transcription_engine
    backend = getattr(transcription_engine.config, "backend", "whisper") if transcription_engine else None
    return JSONResponse({
        "status": "ok",
        "backend": backend,
        "ready": transcription_engine is not None,
    })


async def handle_websocket_results(websocket, results_generator, diff_tracker=None, utterance_tracker=None):
    """Consumes results from the audio processor and sends them via WebSocket."""
    try:
        async for response in results_generator:
            if utterance_tracker is not None:
                d = response.to_dict()
                is_silence = any(
                    line.get("speaker") == -2 for line in d.get("lines", [])
                )
                for msg in utterance_tracker.process_update(d, is_silence=is_silence):
                    await websocket.send_json(msg)
            elif diff_tracker is not None:
                await websocket.send_json(diff_tracker.to_message(response))
            else:
                await websocket.send_json(response.to_dict())
        # Finalize any remaining utterance
        if utterance_tracker is not None:
            for msg in utterance_tracker.force_finalize():
                await websocket.send_json(msg)
        logger.info("Results generator finished. Sending 'ready_to_stop' to client.")
        await websocket.send_json({"type": "ready_to_stop"})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected while handling results (client likely closed connection).")
    except Exception as e:
        logger.exception(f"Error in WebSocket results handler: {e}")


@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    """Live streaming transcription via WebSocket.

    Query parameters:
    - `language`: ISO 639-1 language code for per-session override
    - `mode`: `full` (default), `diff` (incremental updates), or `utterance` (utterance-based seq/final protocol)

    Each connection creates an independent AudioProcessor pipeline.
    Multiple clients can connect simultaneously.
    """
    global transcription_engine

    # Read per-session options from query parameters
    session_language = websocket.query_params.get("language", None)
    mode = websocket.query_params.get("mode", "full")

    audio_processor = AudioProcessor(
        transcription_engine=transcription_engine,
        language=session_language,
    )
    await websocket.accept()
    logger.info(
        "WebSocket connection opened.%s",
        f" language={session_language}" if session_language else "",
    )
    diff_tracker = None
    utterance_tracker = None
    if mode == "diff":
        from whisperlivekit.diff_protocol import DiffTracker
        diff_tracker = DiffTracker()
        logger.info("Client requested diff mode")
    elif mode == "utterance":
        from whisperlivekit.utterance_tracker import UtteranceTracker
        sentence_split_fn = None
        try:
            from whisperlivekit.sentence_splitter import SentenceSplitter
            splitter = SentenceSplitter()
            sentence_split_fn = splitter.split
        except Exception:
            logger.info("wtpsplit not available, using punctuation heuristics for utterance splitting")
        utterance_tracker = UtteranceTracker(
            epd_threshold=0.5,
            max_utterance_duration=15.0,
            sentence_splitter=sentence_split_fn,
        )
        logger.info("Client requested utterance mode (utterance-based)")

    try:
        await websocket.send_json({"type": "config", "useAudioWorklet": bool(config.pcm_input), "mode": mode})
    except Exception as e:
        logger.warning(f"Failed to send config to client: {e}")

    results_generator = await audio_processor.create_tasks()
    websocket_task = asyncio.create_task(
        handle_websocket_results(websocket, results_generator, diff_tracker, utterance_tracker)
    )

    try:
        while True:
            message = await websocket.receive_bytes()
            await audio_processor.process_audio(message)
    except KeyError as e:
        if 'bytes' in str(e):
            logger.warning("Client has closed the connection.")
        else:
            logger.error(f"Unexpected KeyError in websocket_endpoint: {e}", exc_info=True)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client during message receiving loop.")
    except Exception as e:
        logger.error(f"Unexpected error in websocket_endpoint main loop: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up WebSocket endpoint...")
        if not websocket_task.done():
            websocket_task.cancel()
        try:
            await websocket_task
        except asyncio.CancelledError:
            logger.info("WebSocket results handler task was cancelled.")
        except Exception as e:
            logger.warning(f"Exception while awaiting websocket_task completion: {e}")

        await audio_processor.cleanup()
        logger.info("WebSocket endpoint cleaned up successfully.")


# ---------------------------------------------------------------------------
# Deepgram-compatible WebSocket API  (/v1/listen)
# ---------------------------------------------------------------------------

@app.websocket("/v1/listen")
async def deepgram_websocket_endpoint(websocket: WebSocket):
    """Deepgram-compatible live transcription WebSocket."""
    global transcription_engine
    from whisperlivekit.deepgram_compat import handle_deepgram_websocket
    await handle_deepgram_websocket(websocket, transcription_engine, config)


# ---------------------------------------------------------------------------
# OpenAI-compatible REST API  (/v1/audio/transcriptions)
# ---------------------------------------------------------------------------

async def _convert_to_pcm(audio_bytes: bytes) -> bytes:
    """Convert any audio format to PCM s16le mono 16kHz using ffmpeg."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-i", "pipe:0",
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        "-loglevel", "error",
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(input=audio_bytes)
    if proc.returncode != 0:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=f"Audio conversion failed: {stderr.decode().strip()}")
    return stdout


def _parse_time_str(time_str: str) -> float:
    """Parse 'H:MM:SS.cc' to seconds."""
    parts = time_str.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def _format_openai_response(front_data, response_format: str, language: Optional[str], duration: float) -> dict:
    """Convert FrontData to OpenAI-compatible response."""
    d = front_data.to_dict()
    lines = d.get("lines", [])

    # Combine all speech text (exclude silence segments)
    text_parts = [l["text"] for l in lines if l.get("text") and l.get("speaker", 0) != -2]
    full_text = " ".join(text_parts).strip()

    if response_format == "text":
        return full_text

    # Build segments and words for verbose_json
    segments = []
    words = []
    for i, line in enumerate(lines):
        if line.get("speaker") == -2 or not line.get("text"):
            continue
        start = _parse_time_str(line.get("start", "0:00:00"))
        end = _parse_time_str(line.get("end", "0:00:00"))
        segments.append({
            "id": len(segments),
            "start": round(start, 2),
            "end": round(end, 2),
            "text": line["text"],
        })
        # Split segment text into approximate words with estimated timestamps
        seg_words = line["text"].split()
        if seg_words:
            word_duration = (end - start) / max(len(seg_words), 1)
            for j, word in enumerate(seg_words):
                words.append({
                    "word": word,
                    "start": round(start + j * word_duration, 2),
                    "end": round(start + (j + 1) * word_duration, 2),
                })

    if response_format == "verbose_json":
        return {
            "task": "transcribe",
            "language": language or "unknown",
            "duration": round(duration, 2),
            "text": full_text,
            "words": words,
            "segments": segments,
        }

    if response_format in ("srt", "vtt"):
        lines_out = []
        if response_format == "vtt":
            lines_out.append("WEBVTT\n")
        for i, seg in enumerate(segments):
            start_ts = _srt_timestamp(seg["start"], response_format)
            end_ts = _srt_timestamp(seg["end"], response_format)
            if response_format == "srt":
                lines_out.append(f"{i + 1}")
            lines_out.append(f"{start_ts} --> {end_ts}")
            lines_out.append(seg["text"])
            lines_out.append("")
        return "\n".join(lines_out)

    # Default: json
    return {"text": full_text}


def _srt_timestamp(seconds: float, fmt: str) -> str:
    """Format seconds as SRT (HH:MM:SS,mmm) or VTT (HH:MM:SS.mmm) timestamp."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    sep = "," if fmt == "srt" else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


@app.post("/v1/audio/transcriptions", tags=["Transcription"], summary="Transcribe audio file")
async def create_transcription(
    file: UploadFile = File(..., description="Audio file to transcribe (mp3, wav, m4a, etc.)"),
    model: str = Form(default="", description="Model name (accepted but ignored; uses server backend)"),
    language: Optional[str] = Form(default=None, description="ISO 639-1 language code (e.g. 'en', 'ko')"),
    prompt: str = Form(default="", description="Optional prompt to guide transcription"),
    response_format: str = Form(default="json", description="Response format: json, verbose_json, text, srt, vtt"),
    timestamp_granularities: Optional[List[str]] = Form(default=None, description="Timestamp granularities: word, segment"),
):
    """OpenAI-compatible audio transcription endpoint.

    Drop-in replacement for OpenAI's `/v1/audio/transcriptions` API.
    Supports concurrent requests — each request creates an independent processing pipeline.

    **Example with curl:**
    ```bash
    curl http://localhost:8000/v1/audio/transcriptions \\
      -F file=@audio.mp3 \\
      -F response_format=verbose_json
    ```

    **Example with OpenAI Python client:**
    ```python
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
    result = client.audio.transcriptions.create(
        model="whisper-1", file=open("audio.mp3", "rb")
    )
    ```
    """
    global transcription_engine

    audio_bytes = await file.read()
    if not audio_bytes:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Empty audio file")

    # Convert to PCM for pipeline processing
    pcm_data = await _convert_to_pcm(audio_bytes)
    duration = len(pcm_data) / (16000 * 2)  # 16kHz, 16-bit

    # Process through the full pipeline
    processor = AudioProcessor(
        transcription_engine=transcription_engine,
        language=language,
    )
    # Force PCM input regardless of server config
    processor.is_pcm_input = True

    results_gen = await processor.create_tasks()

    # Collect results in background while feeding audio
    final_result = None

    async def collect():
        nonlocal final_result
        async for result in results_gen:
            final_result = result

    collect_task = asyncio.create_task(collect())

    # Feed audio in chunks (1 second each)
    chunk_size = 16000 * 2  # 1 second of PCM
    for i in range(0, len(pcm_data), chunk_size):
        await processor.process_audio(pcm_data[i:i + chunk_size])

    # Signal end of audio
    await processor.process_audio(b"")

    # Wait for pipeline to finish
    try:
        await asyncio.wait_for(collect_task, timeout=120.0)
    except asyncio.TimeoutError:
        logger.warning("Transcription timed out after 120s")
    finally:
        await processor.cleanup()

    if final_result is None:
        return JSONResponse({"text": ""})

    result = _format_openai_response(final_result, response_format, language, duration)

    if isinstance(result, str):
        return PlainTextResponse(result)
    return JSONResponse(result)


@app.get("/v1/models", tags=["Models"], summary="List models")
async def list_models():
    """OpenAI-compatible model listing. Returns the currently configured backend and model."""
    global transcription_engine
    backend = getattr(transcription_engine.config, "backend", "whisper") if transcription_engine else "whisper"
    model_size = getattr(transcription_engine.config, "model_size", "base") if transcription_engine else "base"
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": f"{backend}/{model_size}" if backend != "whisper" else f"whisper-{model_size}",
            "object": "model",
            "owned_by": "whisperlivekit",
        }],
    })


def main():
    """Entry point for the CLI command."""
    import uvicorn

    from whisperlivekit.cli import print_banner

    ssl = bool(config.ssl_certfile and config.ssl_keyfile)
    print_banner(config, config.host, config.port, ssl=ssl)

    uvicorn_kwargs = {
        "app": "whisperlivekit.basic_server:app",
        "host": config.host,
        "port": config.port,
        "reload": False,
        "log_level": "info",
        "lifespan": "on",
    }

    ssl_kwargs = {}
    if config.ssl_certfile or config.ssl_keyfile:
        if not (config.ssl_certfile and config.ssl_keyfile):
            raise ValueError("Both --ssl-certfile and --ssl-keyfile must be specified together.")
        ssl_kwargs = {
            "ssl_certfile": config.ssl_certfile,
            "ssl_keyfile": config.ssl_keyfile,
        }

    if ssl_kwargs:
        uvicorn_kwargs = {**uvicorn_kwargs, **ssl_kwargs}
    if config.forwarded_allow_ips:
        uvicorn_kwargs = {**uvicorn_kwargs, "forwarded_allow_ips": config.forwarded_allow_ips}

    uvicorn.run(**uvicorn_kwargs)

if __name__ == "__main__":
    main()
