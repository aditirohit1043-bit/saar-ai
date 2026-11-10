"""FastAPI application for the Saar project."""

import mimetypes
import os
import shutil
import subprocess
from pathlib import Path

import pymupdf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from .gnani_speech import GnaniUnavailable, synthesize_with_timbre, transcribe_with_prisma
from .qa import MAX_SOURCE_CHARACTERS, answer_document_question
from .qa_types import DocumentAnswerResponse, DocumentQuestionRequest
from .speech_types import SpeechSynthesisRequest, TranscriptionResponse

app = FastAPI(
    title="Saar API",
    description="REST API for the Saar document assistant.",
    version="0.1.0",
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_AUDIO_UPLOAD_BYTES = 10 * 1024 * 1024
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a"}
PDF_HEADER = b"%PDF-"
OCR_DPI = 200
MIN_SELECTABLE_TEXT_FOR_OCR = 80
OCR_LANGUAGE_BY_RESPONSE_LANGUAGE = {"en": "eng", "gu": "guj", "hi": "hin"}
WINDOWS_TESSERACT_PATH = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
OCR_TIMEOUT_SECONDS = 60


class OCRUnavailableError(RuntimeError):
    """Raised when the local Tesseract executable cannot be found."""


class OCRLanguageDataError(RuntimeError):
    """Raised when Tesseract is missing one of the requested language packs."""


class OCRProcessingError(RuntimeError):
    """Raised when Tesseract could not process a rendered page."""


def _tesseract_executable() -> str:
    configured_path = os.environ.get("SAAR_TESSERACT_CMD")
    if configured_path:
        resolved_path = shutil.which(configured_path)
        if resolved_path:
            return resolved_path
        if Path(configured_path).is_file():
            return configured_path
        raise OCRUnavailableError("The configured Tesseract executable was not found.")

    path_from_path = shutil.which("tesseract")
    if path_from_path:
        return path_from_path
    if WINDOWS_TESSERACT_PATH.is_file():
        return str(WINDOWS_TESSERACT_PATH)
    raise OCRUnavailableError("Tesseract was not found on PATH or at its Windows default path.")


def _ocr_language_argument(response_language: str) -> str:
    selected_language = OCR_LANGUAGE_BY_RESPONSE_LANGUAGE.get(response_language, "eng")
    return selected_language if selected_language == "eng" else f"{selected_language}+eng"


def _ocr_page_text(page: pymupdf.Page, response_language: str) -> str:
    """Render one PDF page and recognize it with the installed local Tesseract binary."""
    executable = _tesseract_executable()
    try:
        image_bytes = page.get_pixmap(dpi=OCR_DPI, alpha=False).tobytes("png")
        result = subprocess.run(
            [
                executable,
                "stdin",
                "-",
                "-l",
                _ocr_language_argument(response_language),
                "--psm",
                "3",
            ],
            input=image_bytes,
            capture_output=True,
            timeout=OCR_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise OCRProcessingError("Tesseract timed out while reading the page.") from error
    except (OSError, RuntimeError, ValueError) as error:
        raise OCRProcessingError("Tesseract could not process the rendered page.") from error

    if result.returncode != 0:
        error_output = result.stderr.decode("utf-8", errors="replace").casefold()
        if "failed loading language" in error_output or "error opening data file" in error_output:
            raise OCRLanguageDataError("A requested Tesseract language pack is missing.")
        raise OCRProcessingError("Tesseract returned an OCR error.")

    return result.stdout.decode("utf-8", errors="replace").strip()


def _page_number_list(page_numbers: list[int]) -> str:
    return ", ".join(str(page_number) for page_number in sorted(set(page_numbers)))


@app.get("/api/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """Report that the API process is responding."""
    return {"status": "ok"}


@app.post("/api/documents/upload", tags=["documents"])
async def upload_document(
    file: UploadFile = File(...),
    response_language: str = Form("en"),
) -> dict[str, object]:
    """Extract selectable text and locally OCR scanned pages without storing the PDF."""
    try:
        contents = await file.read(MAX_UPLOAD_BYTES + 1)

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="This file is empty. Choose a PDF that contains a notice.",
            )

        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="This PDF is larger than the 10 MB upload limit.",
            )

        # PDF readers permit the header to appear within the first 1,024 bytes.
        if PDF_HEADER not in contents[:1024]:
            raise HTTPException(
                status_code=415,
                detail="That file doesn't look like a PDF. Please choose a PDF notice.",
            )

        try:
            document = pymupdf.open(stream=contents, filetype="pdf")
        except (pymupdf.FileDataError, pymupdf.EmptyFileError, RuntimeError, ValueError) as error:
            raise HTTPException(
                status_code=400,
                detail="Saar couldn't read this PDF. It may be damaged or malformed.",
            ) from error

        try:
            if document.needs_pass:
                raise HTTPException(
                    status_code=400,
                    detail="This PDF is password-protected. Remove its password and try again.",
                )

            pages: list[dict[str, object]] = []
            ocr_failed_pages: list[int] = []
            ocr_unreadable_pages: list[int] = []
            ocr_failure_kinds: set[str] = set()

            try:
                for page_number, page in enumerate(document, start=1):
                    page_text = page.get_text("text").strip()
                    page_ocr_used = False
                    if (
                        len(page_text) < MIN_SELECTABLE_TEXT_FOR_OCR
                        and page.get_images(full=True)
                    ):
                        try:
                            ocr_text = _ocr_page_text(page, response_language)
                        except OCRUnavailableError:
                            ocr_failed_pages.append(page_number)
                            ocr_failure_kinds.add("tesseract")
                        except OCRLanguageDataError:
                            ocr_failed_pages.append(page_number)
                            ocr_failure_kinds.add("language_data")
                        except Exception:
                            ocr_failed_pages.append(page_number)
                            ocr_failure_kinds.add("processing")
                        else:
                            if ocr_text:
                                page_text = ocr_text
                                page_ocr_used = True
                            else:
                                ocr_unreadable_pages.append(page_number)
                    pages.append({
                        "page_number": page_number,
                        "text": page_text,
                        "ocr_used": page_ocr_used,
                    })
            except (pymupdf.FileDataError, RuntimeError, ValueError) as error:
                raise HTTPException(
                    status_code=400,
                    detail="Saar couldn't read this PDF. It may be damaged or malformed.",
                ) from error

            page_count = len(pages)
            extracted_text = "\n\n".join(page["text"] for page in pages).strip()
            filename = file.filename or "notice.pdf"

            if page_count == 0:
                return {
                    "filename": filename,
                    "page_count": 0,
                    "extracted_text": "",
                    "pages": [],
                    "ocr_used": False,
                    "status": "empty",
                    "message": "This PDF doesn't contain any pages to read.",
                }

            ocr_problem_pages = _page_number_list(
                ocr_failed_pages + ocr_unreadable_pages
            )
            if ocr_problem_pages:
                if "tesseract" in ocr_failure_kinds:
                    setup_guidance = (
                        " Install Tesseract or set SAAR_TESSERACT_CMD to its executable path, "
                        "then restart Saar."
                    )
                    failure_status = "ocr_unavailable"
                elif "language_data" in ocr_failure_kinds:
                    setup_guidance = (
                        " Install the selected language data (eng, hin, and guj), then retry."
                    )
                    failure_status = "ocr_unavailable"
                elif ocr_failed_pages:
                    setup_guidance = " Check Tesseract and try a clearer scan."
                    failure_status = "ocr_failed"
                else:
                    setup_guidance = " Try a clearer scan."
                    failure_status = "ocr_failed"

                if extracted_text:
                    message = (
                        "Saar extracted some text, but couldn't read scanned page(s) "
                        f"{ocr_problem_pages}." + setup_guidance
                    )
                    status = "partial"
                elif ocr_failed_pages:
                    message = (
                        "Saar couldn't run local OCR on scanned page(s) "
                        f"{ocr_problem_pages}." + setup_guidance
                    )
                    status = failure_status
                else:
                    message = (
                        "Local OCR couldn't recognize readable text on scanned page(s) "
                        f"{ocr_problem_pages}. Try a clearer scan."
                    )
                    status = "ocr_failed"
            elif not extracted_text:
                message = (
                    "We couldn't find readable text or scanned images in this PDF. "
                    "It may be blank or contain unsupported vector artwork."
                )
                status = "no_text"
            else:
                message = None
                status = "success"

            return {
                "filename": filename,
                "page_count": page_count,
                "extracted_text": extracted_text,
                "pages": pages,
                "ocr_used": any(page["ocr_used"] for page in pages),
                "status": status,
                "message": message,
            }
        finally:
            document.close()
    finally:
        await file.close()


@app.post(
    "/api/documents/ask",
    response_model=DocumentAnswerResponse,
    tags=["documents"],
)
async def ask_document_question(
    request: DocumentQuestionRequest,
) -> DocumentAnswerResponse:
    """Answer from page text sent by the current frontend document session."""
    source_characters = sum(len(page.text) for page in request.pages)
    if source_characters > MAX_SOURCE_CHARACTERS:
        raise HTTPException(
            status_code=413,
            detail="This document has too much extracted text to search in one request.",
        )

    return await answer_document_question(request)


@app.post(
    "/api/speech/transcribe",
    response_model=TranscriptionResponse,
    tags=["speech"],
)
async def transcribe_audio(audio_file: UploadFile = File(...)) -> TranscriptionResponse:
    """Transcribe a short uploaded audio clip with Gnani Prisma v3.0."""
    try:
        supplied_filename = audio_file.filename or ""
        filename = supplied_filename.replace("\\", "/").rsplit("/", 1)[-1]
        filename = filename.replace("\r", "").replace("\n", "").replace("\x00", "")
        if not supplied_filename:
            filename = "recording.wav"
        extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if extension not in SUPPORTED_AUDIO_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail="Choose a WAV, MP3, OGG, FLAC, AAC, or M4A audio file.",
            )

        contents = await audio_file.read(MAX_AUDIO_UPLOAD_BYTES + 1)
        if not contents:
            raise HTTPException(status_code=400, detail="The audio file is empty.")
        if len(contents) > MAX_AUDIO_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="This audio file exceeds Saar's 10 MB upload limit.",
            )

        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        try:
            result = await transcribe_with_prisma(
                filename=filename,
                audio_bytes=contents,
                content_type=content_type,
            )
        except GnaniUnavailable as error:
            raise HTTPException(status_code=error.status_code, detail=error.message) from None

        transcript = result["transcript"] or ""
        return TranscriptionResponse(
            **result,
            status="transcribed" if transcript.strip() else "no_speech",
            message=None if transcript.strip() else "No speech was detected in this audio clip.",
        )
    finally:
        await audio_file.close()


@app.post("/api/speech/synthesize", tags=["speech"])
async def synthesize_speech(request: SpeechSynthesisRequest) -> Response:
    """Generate a WAV response for Saar's answer with Gnani Timbre v2.5."""
    try:
        audio_bytes = await synthesize_with_timbre(
            text=request.text,
            response_language=request.response_language,
        )
    except GnaniUnavailable as error:
        raise HTTPException(status_code=error.status_code, detail=error.message) from None

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="saar-answer.wav"'},
    )
