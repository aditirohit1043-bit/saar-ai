"""Small HTTP clients for Gnani Prisma and Timbre REST APIs."""

import os
from pathlib import Path

import httpx


GNANI_STT_URL = "https://api.vachana.ai/stt/v3"
GNANI_TTS_URL = "https://api.vachana.ai/api/v1/tts/inference"
PRISMA_MODEL = "prisma-v3.0"
TIMBRE_MODEL = "timbre-v2.5"
REQUEST_TIMEOUT_SECONDS = 60.0

TTS_VOICES = {
    "en": ("en-IN", "Kaveri"),
    "hi": ("hi-IN", "Nalini"),
    "gu": ("gu-IN", "Falak"),
}


class GnaniUnavailable(Exception):
    """Gnani is not configured or is temporarily unreachable."""

    def __init__(self, message: str, status_code: int = 503) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _api_key() -> str:
    """Read the server-side key from the environment or the ignored root .env."""
    environment_key = os.getenv("GNANI_API_KEY", "").strip()
    if environment_key:
        return environment_key

    project_root = Path(__file__).resolve().parents[2]
    env_file = project_root / ".env"
    try:
        lines = env_file.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return ""

    for line in lines:
        key, separator, value = line.partition("=")
        if separator and key.strip() == "GNANI_API_KEY":
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            return value.strip()
    return ""


def _api_headers(api_key: str) -> dict[str, str]:
    # Gnani's documented REST APIs use this header name (not Bearer auth).
    return {"X-API-Key-ID": api_key}


def _raise_for_provider_status(status_code: int, *, action: str) -> None:
    if status_code < 400:
        return
    if status_code in {401, 403}:
        raise GnaniUnavailable(
            "Gnani rejected the backend API key or its service access. Check the key configuration.",
            503,
        )
    if status_code == 429:
        raise GnaniUnavailable(
            "Gnani is busy with another request. Please try again shortly.",
            503,
        )
    if status_code in {408, 504} or status_code >= 500:
        raise GnaniUnavailable(
            f"Gnani {action} is temporarily unavailable. Please try again shortly.",
            503,
        )
    if status_code in {400, 413, 415, 422}:
        if action == "speech transcription":
            raise GnaniUnavailable(
                "Gnani couldn't process this audio. Use a supported WAV, MP3, OGG, FLAC, AAC, or M4A clip up to 30 seconds.",
                422,
            )
        raise GnaniUnavailable(
            "Gnani couldn't synthesize this text. Please shorten the answer and try again.",
            422,
        )
    raise GnaniUnavailable(
        f"Gnani {action} failed. Please try again.",
        502,
    )


async def transcribe_with_prisma(
    *,
    filename: str,
    audio_bytes: bytes,
    content_type: str,
) -> dict[str, str | None]:
    """Submit a short audio clip to the documented Prisma v3 REST endpoint."""
    api_key = _api_key()
    if not api_key:
        raise GnaniUnavailable(
            "Speech transcription is unavailable because GNANI_API_KEY is not configured on the backend."
        )

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(
                GNANI_STT_URL,
                headers=_api_headers(api_key),
                files={"audio_file": (filename, audio_bytes, content_type)},
                data={
                    "model": PRISMA_MODEL,
                    "language_code": "indic-auto",
                    "format": "transcribe",
                },
            )
    except httpx.TimeoutException as error:
        raise GnaniUnavailable(
            "Gnani transcription took too long. Please try a shorter audio clip.",
            504,
        ) from error
    except httpx.HTTPError as error:
        raise GnaniUnavailable(
            "Saar couldn't reach Gnani transcription. Please try again shortly.",
            503,
        ) from error

    _raise_for_provider_status(response.status_code, action="speech transcription")
    try:
        body = response.json()
    except ValueError as error:
        raise GnaniUnavailable("Gnani returned an unreadable transcription response.", 502) from error

    if not isinstance(body, dict):
        raise GnaniUnavailable("Gnani returned an unreadable transcription response.", 502)
    transcript = body.get("transcript")
    if body.get("success") is not True or not isinstance(transcript, str):
        raise GnaniUnavailable("Gnani didn't return a transcript. Please try another audio clip.", 502)

    return {
        "transcript": transcript,
        "model": body.get("model") if isinstance(body.get("model"), str) else PRISMA_MODEL,
        "request_id": body.get("request_id") if isinstance(body.get("request_id"), str) else None,
        "detected_language_code": (
            body.get("detected_language_code")
            if isinstance(body.get("detected_language_code"), str)
            else None
        ),
    }


async def synthesize_with_timbre(*, text: str, response_language: str) -> bytes:
    """Return WAV bytes from the documented Timbre v2.5 REST endpoint."""
    api_key = _api_key()
    if not api_key:
        raise GnaniUnavailable(
            "Speech generation is unavailable because GNANI_API_KEY is not configured on the backend."
        )

    language_code, voice = TTS_VOICES[response_language]
    payload = {
        "text": text,
        "voice": voice,
        "model": TIMBRE_MODEL,
        "language": language_code,
        "audio_config": {
            "sample_rate": 24_000,
            "num_channels": 1,
            "sample_width": 2,
            "encoding": "linear_pcm",
            "container": "wav",
        },
    }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(
                GNANI_TTS_URL,
                headers={**_api_headers(api_key), "Content-Type": "application/json"},
                json=payload,
            )
    except httpx.TimeoutException as error:
        raise GnaniUnavailable(
            "Gnani speech generation took too long. Please try again.",
            504,
        ) from error
    except httpx.HTTPError as error:
        raise GnaniUnavailable(
            "Saar couldn't reach Gnani speech generation. Please try again shortly.",
            503,
        ) from error

    _raise_for_provider_status(response.status_code, action="speech generation")
    if not response.content:
        raise GnaniUnavailable("Gnani returned empty audio. Please try again.", 502)

    response_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
    if response_type not in {"audio/wav", "audio/x-wav", "application/octet-stream"}:
        raise GnaniUnavailable("Gnani returned an unexpected audio format.", 502)
    return response.content
