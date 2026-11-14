"""Test text chat and voice endpoints against KIT FastAPI application."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app import app


def test_typed_chat(client: TestClient) -> None:
    print("\n--- Testing Typed Chat (/api/chat) ---")
    queries = [
        "What courses are available?",
        "What is the policy for on-duty?",
        "Hello",
    ]
    for q in queries:
        resp = client.post("/api/chat", json={"message": q})
        assert resp.status_code == 200, f"Failed on query: {q}, response: {resp.text}"
        data = resp.json()
        print(f"Q: '{q}'")
        print(f"Intent: {data.get('intent')} | Source: {data.get('source')} | Gemini status: {data.get('gemini_status')}")
        print(f"A: {data.get('answer', '')[:120]}...\n")


def test_voice_endpoints(client: TestClient) -> None:
    print("\n--- Testing Voice Endpoints (/speech/transcribe & /api/voice) ---")
    with tempfile.TemporaryDirectory(prefix="kit-test-voice-") as temp_dir:
        audio_path = Path(temp_dir) / "test.wav"
        escaped_path = str(audio_path).replace("'", "''")
        ps = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.SetOutputToWaveFile('{escaped_path}'); "
            "$s.Speak('Tell me about KIT courses'); $s.Dispose()"
        )
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, capture_output=True)
        audio_bytes = audio_path.read_bytes()

        # 1. Test transcription
        transcribe_resp = client.post(
            "/speech/transcribe",
            files={"file": ("test.wav", audio_bytes, "audio/wav")}
        )
        assert transcribe_resp.status_code == 200, transcribe_resp.text
        t_data = transcribe_resp.json()
        print(f"Transcription: '{t_data.get('text')}' (language={t_data.get('language')})")

        # 2. Test /api/voice
        voice_resp = client.post(
            "/api/voice",
            files={"audio": ("test.wav", audio_bytes, "audio/wav")}
        )
        assert voice_resp.status_code == 200, voice_resp.text
        v_data = voice_resp.json()
        print(f"Voice chat transcript: '{v_data.get('transcript')}'")
        print(f"Voice chat answer: {v_data.get('answer', '')[:120]}...")
        print(f"Intent: {v_data.get('intent')} | Gemini status: {v_data.get('gemini_status')}")


def main() -> None:
    with TestClient(app) as client:
        test_typed_chat(client)
        test_voice_endpoints(client)
    print("\nALL TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()

