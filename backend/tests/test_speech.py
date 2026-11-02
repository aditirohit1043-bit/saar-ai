"""Mocked API tests for Gnani Prisma transcription and Timbre synthesis."""

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import app.gnani_speech as gnani_speech
from app.main import app


def api_request(method: str, path: str, **kwargs: object) -> httpx.Response:
    async def send_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send_request())


class FakeGnaniClient:
    def __init__(self, response: httpx.Response, calls: list[dict[str, Any]]) -> None:
        self.response = response
        self.calls = calls

    async def __aenter__(self) -> "FakeGnaniClient":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        self.calls.append({"url": url, **kwargs})
        return self.response


def mock_gnani(
    monkeypatch: pytest.MonkeyPatch,
    response: httpx.Response,
    calls: list[dict[str, Any]],
) -> None:
    def fake_client(*, timeout: float) -> FakeGnaniClient:
        assert timeout == gnani_speech.REQUEST_TIMEOUT_SECONDS
        return FakeGnaniClient(response, calls)

    monkeypatch.setattr(
        gnani_speech,
        "httpx",
        SimpleNamespace(
            AsyncClient=fake_client,
            TimeoutException=httpx.TimeoutException,
            HTTPError=httpx.HTTPError,
        ),
    )


def provider_response(
    url: str,
    *,
    status_code: int = 200,
    json_body: Any = None,
    content: bytes = b"",
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    response_arguments: dict[str, Any] = {"content": content}
    if json_body is not None:
        response_arguments = {"json": json_body}
    return httpx.Response(
        status_code,
        **response_arguments,
        headers=headers,
        request=httpx.Request("POST", url),
    )


def test_prisma_transcription_uses_documented_request_and_normalizes_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "mock-gnani-key-never-return-this"
    monkeypatch.setenv("GNANI_API_KEY", secret)
    calls: list[dict[str, Any]] = []
    mock_gnani(
        monkeypatch,
        provider_response(
            gnani_speech.GNANI_STT_URL,
            json_body={
                "success": True,
                "request_id": "request-123",
                "transcript": "કાલે અરજી કરો",
                "model": "prisma-v3.0",
                "detected_language_code": "gu-IN",
            },
        ),
        calls,
    )

    response = api_request(
        "POST",
        "/api/speech/transcribe",
        files={"audio_file": ("C:\\private\\question.wav", b"fake-audio-bytes", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "transcript": "કાલે અરજી કરો",
        "model": "prisma-v3.0",
        "request_id": "request-123",
        "detected_language_code": "gu-IN",
        "status": "transcribed",
        "message": None,
    }
    assert secret not in response.text
    assert len(calls) == 1
    assert calls[0]["url"] == "https://api.vachana.ai/stt/v3"
    assert calls[0]["headers"]["X-API-Key-ID"] == secret
    assert calls[0]["files"]["audio_file"] == ("question.wav", b"fake-audio-bytes", "audio/wav")
    assert calls[0]["data"] == {
        "model": "prisma-v3.0",
        "language_code": "indic-auto",
        "format": "transcribe",
    }


def test_transcription_reports_missing_key_as_unavailable_without_calling_gnani(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GNANI_API_KEY", raising=False)
    monkeypatch.setattr(gnani_speech, "_api_key", lambda: "")

    response = api_request(
        "POST",
        "/api/speech/transcribe",
        files={"audio_file": ("question.wav", b"fake-audio", "audio/wav")},
    )

    assert response.status_code == 503
    assert "GNANI_API_KEY is not configured" in response.json()["detail"]


def test_transcription_provider_auth_failure_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "mock-gnani-key-never-return-this"
    monkeypatch.setenv("GNANI_API_KEY", secret)
    calls: list[dict[str, Any]] = []
    mock_gnani(
        monkeypatch,
        provider_response(gnani_speech.GNANI_STT_URL, status_code=401),
        calls,
    )

    response = api_request(
        "POST",
        "/api/speech/transcribe",
        files={"audio_file": ("question.wav", b"fake-audio", "audio/wav")},
    )

    assert response.status_code == 503
    assert "backend API key" in response.json()["detail"]
    assert secret not in response.text
    assert secret not in caplog.text


def test_transcription_rejects_unsupported_audio_before_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GNANI_API_KEY", "mock-key")
    calls: list[dict[str, Any]] = []
    mock_gnani(
        monkeypatch,
        provider_response(gnani_speech.GNANI_STT_URL, json_body={"success": True, "transcript": ""}),
        calls,
    )

    response = api_request(
        "POST",
        "/api/speech/transcribe",
        files={"audio_file": ("notes.txt", b"not audio", "text/plain")},
    )

    assert response.status_code == 415
    assert "WAV, MP3, OGG, FLAC, AAC, or M4A" in response.json()["detail"]
    assert calls == []


def test_timbre_synthesis_uses_documented_request_and_returns_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "mock-gnani-key-never-return-this"
    monkeypatch.setenv("GNANI_API_KEY", secret)
    calls: list[dict[str, Any]] = []
    audio_bytes = b"RIFF-mocked-wav-data"
    mock_gnani(
        monkeypatch,
        provider_response(
            gnani_speech.GNANI_TTS_URL,
            content=audio_bytes,
            headers={"content-type": "audio/wav"},
        ),
        calls,
    )

    response = api_request(
        "POST",
        "/api/speech/synthesize",
        json={"text": "તમારી અરજી મળી ગઈ છે.", "response_language": "gu"},
    )

    assert response.status_code == 200
    assert response.content == audio_bytes
    assert response.headers["content-type"] == "audio/wav"
    assert secret.encode() not in response.content
    assert len(calls) == 1
    assert calls[0]["url"] == "https://api.vachana.ai/api/v1/tts/inference"
    assert calls[0]["headers"]["X-API-Key-ID"] == secret
    assert "Authorization" not in calls[0]["headers"]
    assert calls[0]["json"] == {
        "text": "તમારી અરજી મળી ગઈ છે.",
        "voice": "Falak",
        "model": "timbre-v2.5",
        "language": "gu-IN",
        "audio_config": {
            "sample_rate": 24_000,
            "num_channels": 1,
            "sample_width": 2,
            "encoding": "linear_pcm",
            "container": "wav",
        },
    }


def test_synthesis_reports_missing_key_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GNANI_API_KEY", raising=False)
    monkeypatch.setattr(gnani_speech, "_api_key", lambda: "")

    response = api_request(
        "POST",
        "/api/speech/synthesize",
        json={"text": "Hello there.", "response_language": "en"},
    )

    assert response.status_code == 503
    assert "GNANI_API_KEY is not configured" in response.json()["detail"]


def test_synthesis_provider_failure_returns_safe_service_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "mock-gnani-key-never-return-this"
    monkeypatch.setenv("GNANI_API_KEY", secret)
    calls: list[dict[str, Any]] = []
    mock_gnani(
        monkeypatch,
        provider_response(gnani_speech.GNANI_TTS_URL, status_code=503),
        calls,
    )

    response = api_request(
        "POST",
        "/api/speech/synthesize",
        json={"text": "Hello there.", "response_language": "en"},
    )

    assert response.status_code == 503
    assert "temporarily unavailable" in response.json()["detail"]
    assert secret not in response.text
    assert secret not in caplog.text
