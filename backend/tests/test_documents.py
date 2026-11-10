"""API tests for in-memory PDF upload and text extraction."""

import asyncio
import struct
import zlib

import httpx
import pymupdf
import pytest

import app.main as main
from app.main import MAX_UPLOAD_BYTES, app

def api_request(method: str, path: str, **kwargs: object) -> httpx.Response:
    """Send an HTTP request directly to the ASGI app without opening a server."""

    async def send_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send_request())


def make_pdf(page_texts: list[str]) -> bytes:
    """Create a small PDF with the supplied text on separate pages."""
    document = pymupdf.open()
    for text in page_texts:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    payload = chunk_type + data
    return (
        struct.pack(">I", len(data))
        + payload
        + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
    )


def make_tiny_png() -> bytes:
    """Build a 1×1 PNG using only the Python standard library."""
    header = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    pixel = zlib.compress(b"\x00\x77\x99\x77\xff")
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", pixel)
        + png_chunk(b"IEND", b"")
    )


def make_image_only_pdf() -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_image(pymupdf.Rect(40, 40, 160, 160), stream=make_tiny_png())
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def make_scanned_pdf(page_texts: list[str]) -> bytes:
    """Build image-only PDF pages by rasterizing locally generated notice text."""
    document = pymupdf.open()
    for text in page_texts:
        source = pymupdf.open()
        source_page = source.new_page()
        source_page.insert_text((48, 64), text, fontsize=24)
        image = source_page.get_pixmap(
            matrix=pymupdf.Matrix(1.5, 1.5),
            alpha=False,
        ).tobytes("png")
        source.close()

        page = document.new_page()
        page.insert_image(page.rect, stream=image)

    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def make_mixed_pdf() -> bytes:
    """Build a PDF with a selectable first page and image-only second page."""
    document = pymupdf.open()
    text_page = document.new_page()
    text_page.insert_text((72, 72), "Scholarship notice information is on this page.")

    source = pymupdf.open()
    source_page = source.new_page()
    source_page.insert_text((48, 64), "Application Deadline: 30 June 2026", fontsize=18)
    image = source_page.get_pixmap(
        matrix=pymupdf.Matrix(2, 2),
        alpha=False,
    ).tobytes("png")
    source.close()

    scanned_page = document.new_page()
    scanned_page.insert_image(scanned_page.rect, stream=image)
    scanned_page.insert_text((72, 760), "Header")
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def test_health_endpoint_still_works() -> None:
    response = api_request("GET", "/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_extracts_text_page_by_page() -> None:
    pdf_bytes = make_pdf(["Notice: application deadline is 30 June.", "Bring a photo ID."])

    response = api_request(
        "POST",
        "/api/documents/upload",
        files={"file": ("student-notice.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["filename"] == "student-notice.pdf"
    assert result["page_count"] == 2
    assert result["status"] == "success"
    assert result["ocr_used"] is False
    assert "application deadline is 30 June" in result["extracted_text"]
    assert result["pages"] == [
        {"page_number": 1, "text": "Notice: application deadline is 30 June.", "ocr_used": False},
        {"page_number": 2, "text": "Bring a photo ID.", "ocr_used": False},
    ]


def test_upload_rejects_a_non_pdf_file() -> None:
    response = api_request(
        "POST",
        "/api/documents/upload",
        files={"file": ("notice.txt", b"This is not a PDF.", "text/plain")},
    )

    assert response.status_code == 415
    assert "doesn't look like a PDF" in response.json()["detail"]


def test_upload_reports_an_empty_file() -> None:
    response = api_request(
        "POST",
        "/api/documents/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 400
    assert "file is empty" in response.json()["detail"]


def test_upload_ocr_extracts_text_from_a_scanned_pdf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ocr_calls: list[int] = []

    def recognize_page(page: pymupdf.Page, language_argument: str) -> str:
        assert not page.get_text("text").strip()
        assert page.get_images(full=True)
        assert language_argument == "en"
        assert main._ocr_language_argument(language_argument) == "eng"
        ocr_calls.append(page.number + 1)
        return "Application Deadline: 30 June 2026"

    monkeypatch.setattr(main, "_ocr_page_text", recognize_page)
    response = api_request(
        "POST",
        "/api/documents/upload",
        files={"file": ("scanned-notice.pdf", make_scanned_pdf(["Scanned notice"]), "application/pdf")},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "success"
    assert result["ocr_used"] is True
    assert result["extracted_text"] == "Application Deadline: 30 June 2026"
    assert result["pages"] == [{
        "page_number": 1,
        "text": "Application Deadline: 30 June 2026",
        "ocr_used": True,
    }]
    assert result["message"] is None
    assert ocr_calls == [1]


def test_ocr_language_uses_selected_language_with_english_fallback() -> None:
    assert main._ocr_language_argument("en") == "eng"
    assert main._ocr_language_argument("gu") == "guj+eng"
    assert main._ocr_language_argument("hi") == "hin+eng"
    assert main._ocr_language_argument("unsupported") == "eng"


def test_installed_tesseract_recognizes_a_real_scanned_pdf() -> None:
    try:
        main._tesseract_executable()
    except main.OCRUnavailableError:
        pytest.skip("Local Tesseract is not installed")

    document = pymupdf.open(
        stream=make_scanned_pdf(["Application Deadline: 30 June 2026"]),
        filetype="pdf",
    )
    page = document[0]
    assert not page.get_text("text").strip()
    assert page.get_images(full=True)

    recognized_text = main._ocr_page_text(page, "en")

    assert "30 June 2026" in recognized_text
    document.close()


def test_upload_ocr_handles_mixed_pages_and_preserves_qa_citation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ocr_calls: list[int] = []

    def recognize_page(page: pymupdf.Page, language_argument: str) -> str:
        assert page.get_text("text").strip() == "Header"
        assert language_argument == "gu"
        assert main._ocr_language_argument(language_argument) == "guj+eng"
        ocr_calls.append(page.number + 1)
        return "Header\nApplication Deadline\nApplications close on 30 June 2026."

    monkeypatch.setattr(main, "_ocr_page_text", recognize_page)
    upload = api_request(
        "POST",
        "/api/documents/upload",
        files={"file": ("mixed-notice.pdf", make_mixed_pdf(), "application/pdf")},
        data={"response_language": "gu"},
    )

    assert upload.status_code == 200
    document = upload.json()
    assert document["status"] == "success"
    assert document["ocr_used"] is True
    assert document["pages"] == [
        {
            "page_number": 1,
            "text": "Scholarship notice information is on this page.",
            "ocr_used": False,
        },
        {
            "page_number": 2,
            "text": "Header\nApplication Deadline\nApplications close on 30 June 2026.",
            "ocr_used": True,
        },
    ]
    assert ocr_calls == [2]

    answer = api_request(
        "POST",
        "/api/documents/ask",
        json={
            "question": "What is the application deadline?",
            "pages": document["pages"],
            "response_language": "en",
        },
    ).json()
    assert answer["status"] == "answered"
    assert "30 June 2026" in answer["answer"]
    assert answer["sources"] == [{
        "page_number": 2,
        "text": "Application Deadline\nApplications close on 30 June 2026.",
    }]


def test_upload_reports_local_ocr_failure_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_ocr(page: pymupdf.Page, language_argument: str) -> str:
        raise main.OCRUnavailableError("Tesseract unavailable")

    monkeypatch.setattr(main, "_ocr_page_text", fail_ocr)
    response = api_request(
        "POST",
        "/api/documents/upload",
        files={"file": ("scanned-notice.pdf", make_scanned_pdf(["Scanned notice"]), "application/pdf")},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "ocr_unavailable"
    assert result["ocr_used"] is False
    assert result["extracted_text"] == ""
    assert result["pages"] == [{"page_number": 1, "text": "", "ocr_used": False}]
    assert "couldn't run local OCR" in result["message"]
    assert "Install Tesseract" in result["message"]
    assert "SAAR_TESSERACT_CMD" in result["message"]


def test_upload_reports_missing_tesseract_language_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_ocr(page: pymupdf.Page, language_argument: str) -> str:
        raise main.OCRLanguageDataError("Gujarati language data is missing")

    monkeypatch.setattr(main, "_ocr_page_text", fail_ocr)
    response = api_request(
        "POST",
        "/api/documents/upload",
        files={"file": ("scanned-notice.pdf", make_scanned_pdf(["Scanned notice"]), "application/pdf")},
        data={"response_language": "gu"},
    )

    result = response.json()
    assert response.status_code == 200
    assert result["status"] == "ocr_unavailable"
    assert result["ocr_used"] is False
    assert "selected language data" in result["message"]
    assert "eng, hin, and guj" in result["message"]


def test_upload_reports_a_valid_blank_pdf() -> None:
    response = api_request(
        "POST",
        "/api/documents/upload",
        files={"file": ("blank-notice.pdf", make_pdf([""]), "application/pdf")},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "no_text"
    assert result["page_count"] == 1
    assert result["extracted_text"] == ""
    assert "may be blank" in result["message"]


def test_upload_rejects_a_malformed_pdf() -> None:
    malformed_pdf = b"%PDF-1.7\nThis is not a valid PDF structure.\n%%EOF\n"

    response = api_request(
        "POST",
        "/api/documents/upload",
        files={"file": ("damaged.pdf", malformed_pdf, "application/pdf")},
    )

    assert response.status_code == 400
    assert "damaged or malformed" in response.json()["detail"]


def test_upload_rejects_a_pdf_over_the_size_limit() -> None:
    oversized_pdf = b"%PDF-1.7\n" + b"x" * MAX_UPLOAD_BYTES

    response = api_request(
        "POST",
        "/api/documents/upload",
        files={"file": ("large.pdf", oversized_pdf, "application/pdf")},
    )

    assert response.status_code == 413
    assert "10 MB upload limit" in response.json()["detail"]


def test_upload_reports_a_password_protected_pdf() -> None:
    document = pymupdf.open()
    document.new_page()
    encrypted_pdf = document.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner-secret",
        user_pw="user-secret",
    )
    document.close()

    response = api_request(
        "POST",
        "/api/documents/upload",
        files={"file": ("protected.pdf", encrypted_pdf, "application/pdf")},
    )

    assert response.status_code == 400
    assert "password-protected" in response.json()["detail"]
