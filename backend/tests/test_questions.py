"""API tests for grounded document questions and source handling."""

import asyncio

import httpx
import pytest

import app.qa as qa
from app.main import app


def api_request(method: str, path: str, **kwargs: object) -> httpx.Response:
    """Send an HTTP request directly to the FastAPI app."""

    async def send_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send_request())


@pytest.fixture(autouse=True)
def leave_evon_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SAAR_EVON_BASE_URL", raising=False)


def ask(question: str, pages: list[dict[str, object]]) -> httpx.Response:
    return api_request(
        "POST",
        "/api/documents/ask",
        json={"question": question, "pages": pages, "response_language": "en"},
    )


def test_question_is_answered_from_a_supporting_passage() -> None:
    response = ask(
        "What is the application deadline?",
        [{"page_number": 2, "text": "Applications close on 30 June 2026."}],
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "answered"
    assert result["mode"] == "extractive"
    assert "30 June 2026" in result["answer"]


def test_question_missing_from_the_notice_gets_an_abstaining_answer() -> None:
    response = ask(
        "Does this notice provide housing support?",
        [{"page_number": 1, "text": "Applications close on 30 June 2026."}],
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "not_found"
    assert result["answer"] == "The notice does not provide enough information to answer that question."
    assert result["sources"] == []


def test_answer_returns_source_passages_and_page_numbers_across_pages() -> None:
    response = ask(
        "What is the application deadline?",
        [
            {"page_number": 1, "text": "This notice is for student applications."},
            {"page_number": 4, "text": "Applications close on 30 June 2026."},
        ],
    )

    assert response.status_code == 200
    result = response.json()
    assert any(source["page_number"] == 4 for source in result["sources"])
    assert any("30 June 2026" in source["text"] for source in result["sources"])


def test_empty_document_is_handled_without_calling_a_model() -> None:
    response = ask("What is the deadline?", [{"page_number": 1, "text": "  "}])

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "empty_document"
    assert "couldn't find readable text" in result["answer"]
    assert result["sources"] == []


def test_model_api_failure_falls_back_to_a_labeled_source_passage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_to_reach_evon(*args: object, **kwargs: object) -> str:
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(qa, "generate_with_evon", fail_to_reach_evon)
    response = ask(
        "What is the application deadline?",
        [{"page_number": 3, "text": "Applications close on 30 June 2026."}],
    )

    assert response.status_code == 200
    result = response.json()
    assert result["mode"] == "extractive"
    assert "Evon couldn't be reached" in result["message"]
    assert "30 June 2026" in result["answer"]
    assert result["sources"][0]["page_number"] == 3


def test_retrieval_bounds_the_passage_context() -> None:
    pages = [
        qa.DocumentPageText(
            page_number=page_number,
            text=(f"Application deadline on page {page_number} is 30 June. " * 100),
        )
        for page_number in range(1, 8)
    ]

    passages = qa.retrieve_relevant_passages("What is the application deadline?", pages)

    assert len(passages) <= qa.MAX_SELECTED_PASSAGES
    assert sum(len(passage.text) for passage in passages) <= qa.MAX_CONTEXT_CHARACTERS
