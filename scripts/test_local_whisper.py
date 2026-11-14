"""Windows smoke test for the configured local Whisper service and HTTP routes."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app import app


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="kit-whisper-test-") as temp_dir:
        audio_path = Path(temp_dir) / "attendance.wav"
        escaped_path = str(audio_path).replace("'", "''")
        powershell = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.SetOutputToWaveFile('{escaped_path}'); "
            "$s.Speak('I want to check my attendance'); $s.Dispose()"
        )
        subprocess.run(["powershell", "-NoProfile", "-Command", powershell], check=True, capture_output=True)
        audio = audio_path.read_bytes()

        with TestClient(app) as client:
            assert client.get("/").status_code == 200
            assert client.get("/docs").status_code == 200
            health = client.get("/speech/health")
            assert health.status_code == 200, health.text
            assert health.json()["status"] == "ready"
            assert health.json()["device"] == "cpu"

            response = client.post("/speech/transcribe", files={"file": ("attendance.wav", audio, "audio/wav")})
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["success"] is True
            assert result["text"].strip()
            assert result["language"]
            assert result["provider"] == "local_whisper"
            assert result["model"] in ("base", "tiny")

            unsupported = client.post("/speech/transcribe", files={"file": ("notes.txt", b"hello", "text/plain")})
            assert unsupported.status_code == 415
            empty = client.post("/speech/transcribe", files={"file": ("empty.wav", b"", "audio/wav")})
            assert empty.status_code == 400

            print("PASS: model load, root/docs, ready health, nonempty transcription, unsupported type, empty audio")
            print("Transcript received; language=" + result["language"])


if __name__ == "__main__":
    main()
