"""FastAPI application for the Saar project."""

import mimetypes

import pymupdf
from fastapi import FastAPI, File, HTTPException, UploadFile
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


@app.get("/api/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """Report that the API process is responding."""
    return {"status": "ok"}


@app.post("/api/documents/upload", tags=["documents"])
async def upload_document(file: UploadFile = File(...)) -> dict[str, object]:
    """Extract selectable text from an uploaded PDF without storing it."""
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
            image_only_pages: list[int] = []

            try:
                for page_number, page in enumerate(document, start=1):
                    page_text = page.get_text("text").strip()
                    pages.append({"page_number": page_number, "text": page_text})

                    if not page_text and page.get_images(full=True):
                        image_only_pages.append(page_number)
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
                    "status": "empty",
                    "message": "This PDF doesn't contain any pages to read.",
                }

            if not extracted_text:
                if image_only_pages:
                    message = (
                        "This PDF appears to be scanned or image-only. "
                        "OCR isn't supported yet, so Saar couldn't extract text."
                    )
                else:
                    message = (
                        "We couldn't find selectable text in this PDF. It may be blank or scanned. "
                        "OCR isn't supported yet."
                    )
                status = "no_text"
            elif image_only_pages:
                message = (
                    "Text was extracted, but some pages appear to be scanned or image-only. "
                    "OCR isn't supported yet for those pages."
                )
                status = "partial"
            else:
                message = None
                status = "success"

            return {
                "filename": filename,
                "page_count": page_count,
                "extracted_text": extracted_text,
                "pages": pages,
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
