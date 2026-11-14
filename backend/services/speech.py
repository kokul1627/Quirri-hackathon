from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import HTTPException, UploadFile

logger = logging.getLogger(__name__)

MAX_AUDIO_BYTES = 20 * 1024 * 1024
TRANSCRIBE_TIMEOUT_SECONDS = 180
QUEUE_TIMEOUT_SECONDS = 2
SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".webm", ".ogg", ".oga", ".flac",
    ".mp4", ".mpeg", ".mpga", ".aac", ".opus", ".wma",
}
SUPPORTED_CONTENT_TYPES = {
    "audio/wav", "audio/wave", "audio/x-wav", "audio/mpeg", "audio/mp3",
    "audio/mp4", "audio/x-m4a", "audio/webm", "video/webm", "audio/ogg",
    "application/ogg", "audio/flac", "audio/x-flac", "audio/aac",
    "audio/opus", "audio/x-ms-wma", "application/octet-stream",
}


class LocalWhisperService:
    """One CPU Whisper model per FastAPI process; inference serialized for low RAM."""

    def __init__(self) -> None:
        self.provider = os.getenv("SPEECH_PROVIDER", "local_whisper").strip().lower()
        self.model_name = os.getenv("WHISPER_MODEL", "base").strip().lower()
        self.device = os.getenv("WHISPER_DEVICE", "cpu").strip().lower()
        self.language = os.getenv("WHISPER_LANGUAGE", "auto").strip().lower()
        self.model: Any = None
        self.load_error: Optional[str] = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._queue = asyncio.Semaphore(1)

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        with self._load_lock:
            if self.model is not None:
                return
            if self.model_name not in ("base", "tiny"):
                self.load_error = "unsupported_model"
                raise ValueError("WHISPER_MODEL must be base or tiny")
            if self.provider != "local_whisper" or self.device != "cpu":
                self.load_error = "unsupported_configuration"
                raise ValueError("Speech provider must be local_whisper and device must be cpu")
            try:
                import whisper

                self.model = whisper.load_model(self.model_name, device="cpu")
                self.load_error = None
            except Exception as exc:
                self.model = None
                self.load_error = type(exc).__name__
                logger.error("Local Whisper model unavailable (%s)", self.load_error)
                raise

    async def transcribe_upload(self, upload: UploadFile) -> Dict[str, str]:
        if not self.ready:
            raise HTTPException(503, "Local speech model is unavailable. Check Whisper installation and model download.")

        filename = upload.filename or "audio"
        extension = Path(filename).suffix.lower()
        content_type = (upload.content_type or "application/octet-stream").split(";", 1)[0].strip().lower()
        if extension not in SUPPORTED_EXTENSIONS or content_type not in SUPPORTED_CONTENT_TYPES:
            raise HTTPException(415, "Unsupported audio type. Use WAV, MP3, M4A, WebM, OGG, FLAC or another FFmpeg audio format.")

        data = await upload.read(MAX_AUDIO_BYTES + 1)
        if not data:
            raise HTTPException(400, "Audio file is empty.")
        if len(data) > MAX_AUDIO_BYTES:
            raise HTTPException(413, "Audio file exceeds the 20 MB limit.")

        try:
            await asyncio.wait_for(self._queue.acquire(), timeout=QUEUE_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            raise HTTPException(503, "Speech service is busy. Try again shortly or use text input.")

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, self._transcribe_sync, data, extension)
        timed_out = False
        try:
            result = await asyncio.wait_for(asyncio.shield(future), timeout=TRANSCRIBE_TIMEOUT_SECONDS)
            text = str(result.get("text", "")).strip()
            language = str(result.get("language", "")).strip()
            if not text:
                raise HTTPException(422, "No speech was detected. Try again or use text input.")
            return {"text": text, "language": language or "unknown"}
        except asyncio.TimeoutError:
            timed_out = True
            future.add_done_callback(lambda _: loop.call_soon_threadsafe(self._queue.release))
            raise HTTPException(504, "Transcription took too long. Try a shorter recording or use text input.")
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("Local Whisper transcription failed (%s)", type(exc).__name__)
            raise HTTPException(502, "Audio could not be transcribed. Check the recording and try again.")
        finally:
            if not timed_out:
                self._queue.release()

    def _transcribe_sync(self, data: bytes, extension: str) -> Dict[str, Any]:
        temp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(prefix="kit-audio-", suffix=extension, delete=False) as temp:
                temp.write(data)
                temp_path = temp.name
            with self._inference_lock:
                language = None if self.language in ("", "auto") else self.language
                return self.model.transcribe(temp_path, fp16=False, language=language)
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    logger.warning("Temporary audio cleanup failed")


speech_service = LocalWhisperService()
