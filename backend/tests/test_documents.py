"""API tests for in-memory PDF upload and text extraction."""

import asyncio
import struct
import zlib

import httpx
import pymupdf

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
    assert "application deadline is 30 June" in result["extracted_text"]
    assert result["pages"] == [
        {"page_number": 1, "text": "Notice: application deadline is 30 June."},
        {"page_number": 2, "text": "Bring a photo ID."},
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


def test_upload_reports_image_only_pdf_without_claiming_text_was_extracted() -> None:
    response = api_request(
        "POST",
        "/api/documents/upload",
        files={"file": ("scanned-notice.pdf", make_image_only_pdf(), "application/pdf")},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "no_text"
    assert result["extracted_text"] == ""
    assert result["pages"] == [{"page_number": 1, "text": ""}]
    assert "scanned or image-only" in result["message"]
    assert "OCR isn't supported yet" in result["message"]


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
    assert "OCR isn't supported yet" in result["message"]


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
