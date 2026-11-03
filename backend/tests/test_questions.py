"""API tests for grounded document questions and source handling."""

import asyncio
from pathlib import Path

import httpx
import pymupdf
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


def ask(
    question: str,
    pages: list[dict[str, object]],
    response_language: str = "en",
) -> httpx.Response:
    return api_request(
        "POST",
        "/api/documents/ask",
        json={
            "question": question,
            "pages": pages,
            "response_language": response_language,
        },
    )


def sample_notice_pages() -> list[dict[str, object]]:
    sample_pdf = Path(__file__).resolve().parents[2] / "sample_fictional_scholarship_notice.pdf"
    with pymupdf.open(sample_pdf) as document:
        return [
            {"page_number": page_number, "text": page.get_text("text").strip()}
            for page_number, page in enumerate(document, start=1)
        ]


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


def test_scholarship_deadline_returns_one_concise_supported_sentence() -> None:
    # Synthetic notice text keeps the test independent of any real applicant data.
    notice = (
        "The scholarship application deadline is 15 July 2026. "
        "Applicants must be enrolled full-time and submit proof of income. "
        "The award amount is Rs. 25,000 per year."
    )
    response = ask(
        "What is the scholarship application deadline?",
        [{"page_number": 2, "text": notice}],
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "answered"
    assert "15 July 2026" in result["answer"]
    assert "proof of income" not in result["answer"]
    assert "Rs. 25,000" not in result["answer"]
    assert len(result["answer"]) < 160
    assert result["sources"] == [
        {"page_number": 2, "text": "The scholarship application deadline is 15 July 2026."}
    ]

    amount_response = ask(
        "How much is the scholarship award?",
        [{"page_number": 2, "text": notice}],
    )
    amount_result = amount_response.json()
    assert amount_result["status"] == "answered"
    assert "Rs. 25,000" in amount_result["answer"]
    assert amount_result["sources"] == [
        {"page_number": 2, "text": "The award amount is Rs. 25,000 per year."}
    ]


def test_scholarship_question_abstains_when_requested_detail_is_missing() -> None:
    response = ask(
        "Does this scholarship include a hostel allowance?",
        [{
            "page_number": 2,
            "text": "The scholarship application deadline is 15 July 2026. "
            "The award amount is Rs. 25,000 per year.",
        }],
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "not_found"
    assert result["answer"] == "The notice does not provide enough information to answer that question."
    assert result["sources"] == []


def test_required_documents_question_uses_its_named_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_calls: list[bool] = []

    async def return_how_to_apply_incorrectly(*args: object, **kwargs: object) -> str:
        model_calls.append(True)
        return (
            "How to Apply: Prepare the listed documents and submit the scholarship "
            "application through your institution's student support office."
        )

    monkeypatch.setattr(qa, "generate_with_evon", return_how_to_apply_incorrectly)
    notice = (
        "Student Scholarship Notice\n"
        "Documents Required\n"
        "- Student identity card\n"
        "- Latest academic marksheet\n"
        "- Proof of family income\n"
        "- Bank account details or passbook copy\n"
        "- Recent passport-size photograph\n"
        "How to Apply: Prepare the listed documents and submit the scholarship "
        "application through your institution's student support office."
    )
    response = ask(
        "What documents are required for the scholarship?",
        [{"page_number": 3, "text": notice}],
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "answered"
    assert result["answer"] == (
        "The required documents are: Student identity card; Latest academic marksheet; "
        "Proof of family income; Bank account details or passbook copy; "
        "Recent passport-size photograph."
    )
    assert result["sources"] == [{
        "page_number": 3,
        "text": "Documents Required\n"
        "- Student identity card\n"
        "- Latest academic marksheet\n"
        "- Proof of family income\n"
        "- Bank account details or passbook copy\n"
        "- Recent passport-size photograph",
    }]
    assert model_calls == []


def test_project_sample_pdf_extracts_and_answers_its_numbered_document_section() -> None:
    extracted_pages = sample_notice_pages()

    parsed_sections = qa._page_sections(
        qa.DocumentPageText(
            page_number=extracted_pages[0]["page_number"],
            text=extracted_pages[0]["text"],
        )
    )
    assert [section.heading for section in parsed_sections] == [
        "", "Application Deadline", "Eligibility Criteria", "Documents Required",
        "How to Apply", "Important Information",
    ]
    documents_section = next(
        section for section in parsed_sections if section.heading == "Documents Required"
    )
    assert documents_section.lines == [
        "• Student identity card",
        "• Latest academic marksheet",
        "• Proof of family income",
        "• Bank account details or passbook copy",
        "• A recent passport-size photograph",
    ]

    ranked_sections = qa._rank_relevant_passages(
        "What documents are required for the scholarship?",
        [
            qa.DocumentPageText(
                page_number=extracted_pages[0]["page_number"],
                text=extracted_pages[0]["text"],
            )
        ],
    )
    assert ranked_sections[0].heading == "Documents Required"

    response = ask("What documents are required for the scholarship?", extracted_pages)
    result = response.json()
    assert response.status_code == 200
    assert result["answer"] == (
        "The required documents are: Student identity card; Latest academic marksheet; "
        "Proof of family income; Bank account details or passbook copy; "
        "A recent passport-size photograph."
    )
    assert result["sources"] == [{
        "page_number": 1,
        "text": "Documents Required\n"
        "• Student identity card\n"
        "• Latest academic marksheet\n"
        "• Proof of family income\n"
        "• Bank account details or passbook copy\n"
        "• A recent passport-size photograph",
    }]


@pytest.mark.parametrize(
    ("question", "expected_intent"),
    [
        ("શિષ્યવૃત્તિ માટે કયા દસ્તાવેજો જરૂરી છે?", "documents"),
        ("શિષ્યવૃત્તિ માટે અરજી કરવાની છેલ્લી તારીખ શું છે?", "deadline"),
        ("શિષ્યવૃત્તિની રકમ કેટલી છે?", "amount"),
        ("શિષ્યવૃત્તિ માટે પાત્રતા શું છે?", "eligibility"),
        ("શિષ્યવૃત્તિ માટે અરજી કેવી રીતે કરવી?", "how_to_apply"),
    ],
)
def test_gujarati_question_mapping_covers_common_scholarship_intents(
    question: str,
    expected_intent: str,
) -> None:
    normalized, intent = qa.map_gujarati_question(question)

    assert intent == expected_intent
    assert all(ord(character) < 128 for character in normalized)


@pytest.mark.parametrize(
    "question",
    [
        "छात्रवृत्ति के लिए कौन-कौन से दस्तावेज़ आवश्यक हैं?",
        "आवश्यक दस्तावेज़ कौन से हैं?",
        "छात्रवृत्ति के लिए जरूरी कागजात कौन-कौन से हैं?",
    ],
)
def test_hindi_document_phrases_map_to_required_documents_intent(question: str) -> None:
    normalized, intent = qa.map_question_for_retrieval(question)

    assert normalized == "What documents are required for the scholarship?"
    assert intent == "documents"


@pytest.mark.parametrize(
    ("response_language", "expected_items"),
    [
        (
            "hi",
            [
                "छात्र पहचान पत्र",
                "नवीनतम शैक्षणिक अंकपत्र",
                "परिवार की आय का प्रमाण",
                "बैंक खाते का विवरण या पासबुक की प्रति",
                "हाल की पासपोर्ट आकार की तस्वीर",
            ],
        ),
        (
            "gu",
            [
                "વિદ્યાર્થી ઓળખપત્ર",
                "નવીનતમ શૈક્ષણિક ગુણપત્રક",
                "પરિવારની આવકનો પુરાવો",
                "બેંક ખાતાની વિગતો અથવા પાસબુકની નકલ",
                "તાજેતરનો પાસપોર્ટ કદનો ફોટો",
            ],
        ),
    ],
)
def test_document_list_answer_is_translated_but_sources_stay_in_english(
    response_language: str,
    expected_items: list[str],
) -> None:
    response = ask(
        "શિષ્યવૃત્તિ માટે કયા દસ્તાવેજો જરૂરી છે?",
        sample_notice_pages(),
        response_language=response_language,
    )

    result = response.json()
    assert response.status_code == 200
    assert result["status"] == "answered"
    assert result["answer"].startswith(
        qa.REQUIRED_DOCUMENT_INTRODUCTIONS[response_language]
    )
    for item in expected_items:
        assert item in result["answer"]
    assert all(
        english_item not in result["answer"]
        for english_item in ("Student identity card", "Latest academic marksheet", "Proof of family income")
    )
    assert result["sources"] == [{
        "page_number": 1,
        "text": "Documents Required\n"
        "• Student identity card\n"
        "• Latest academic marksheet\n"
        "• Proof of family income\n"
        "• Bank account details or passbook copy\n"
        "• A recent passport-size photograph",
    }]


def test_exact_hindi_document_question_answers_from_project_pdf_api() -> None:
    # sample_notice_pages extracts the real project PDF with PyMuPDF; ask() sends
    # those extracted pages through the FastAPI endpoint, not a helper-only path.
    extracted_pages = sample_notice_pages()
    response = ask(
        "छात्रवृत्ति के लिए कौन-कौन से दस्तावेज़ आवश्यक हैं?",
        extracted_pages,
        response_language="hi",
    )

    result = response.json()
    assert response.status_code == 200
    assert result["status"] == "answered"
    assert result["answer"].startswith(qa.REQUIRED_DOCUMENT_INTRODUCTIONS["hi"])
    for item in (
        "छात्र पहचान पत्र",
        "नवीनतम शैक्षणिक अंकपत्र",
        "परिवार की आय का प्रमाण",
        "बैंक खाते का विवरण या पासबुक की प्रति",
        "हाल की पासपोर्ट आकार की तस्वीर",
    ):
        assert item in result["answer"]
    assert all(
        english_item not in result["answer"]
        for english_item in (
            "Student identity card",
            "Latest academic marksheet",
            "Proof of family income",
            "Bank account details",
            "passport-size photograph",
        )
    )
    assert result["sources"] == [{
        "page_number": 1,
        "text": "Documents Required\n"
        "• Student identity card\n"
        "• Latest academic marksheet\n"
        "• Proof of family income\n"
        "• Bank account details or passbook copy\n"
        "• A recent passport-size photograph",
    }]


def test_hindi_missing_scholarship_amount_abstains_via_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_calls: list[bool] = []

    async def invent_amount(*args: object, **kwargs: object) -> str:
        model_calls.append(True)
        return "छात्रवृत्ति की राशि ₹50,000 है।"

    monkeypatch.setattr(qa, "generate_with_evon", invent_amount)
    response = ask(
        "छात्रवृत्ति की राशि कितनी है?",
        sample_notice_pages(),
        response_language="hi",
    )

    result = response.json()
    assert response.status_code == 200
    assert result["status"] == "not_found"
    assert result["answer"] == qa.NOT_ENOUGH_INFORMATION["hi"]
    assert result["sources"] == []
    assert model_calls == []


def test_gujarati_deadline_question_uses_the_application_deadline_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_calls: list[tuple[str, str]] = []

    async def decline_to_answer(
        question: str,
        passages: list[qa.SourcePassage],
        response_language: str,
    ) -> str:
        model_calls.append((question, response_language))
        return qa.INSUFFICIENT_INFORMATION_TOKEN

    monkeypatch.setattr(qa, "generate_with_evon", decline_to_answer)
    gujarati_question = "શિષ્યવૃત્તિ માટે અરજી કરવાની છેલ્લી તારીખ શું છે?"
    response = ask(gujarati_question, sample_notice_pages(), response_language="gu")

    result = response.json()
    assert response.status_code == 200
    assert result["status"] == "answered"
    assert "30 November 2026" in result["answer"]
    assert result["answer"].startswith(qa.EXTRACTIVE_INTRODUCTIONS["gu"])
    assert result["sources"][0]["page_number"] == 1
    assert result["sources"][0]["text"].startswith("Application Deadline\n")
    assert model_calls == [(gujarati_question, "gu")]


@pytest.mark.parametrize(
    ("question", "expected_heading", "answer_phrase"),
    [
        (
            "શિષ્યવૃત્તિ માટે પાત્રતા શું છે?",
            "Eligibility Criteria",
            "currently enrolled in an undergraduate program",
        ),
        (
            "શિષ્યવૃત્તિ માટે અરજી કેવી રીતે કરવી?",
            "How to Apply",
            "support office",
        ),
    ],
)
def test_gujarati_eligibility_and_application_questions_use_matching_sections(
    question: str,
    expected_heading: str,
    answer_phrase: str,
) -> None:
    response = ask(question, sample_notice_pages(), response_language="gu")

    result = response.json()
    assert response.status_code == 200
    assert result["status"] == "answered"
    assert result["answer"].startswith(qa.EXTRACTIVE_INTRODUCTIONS["gu"])
    assert answer_phrase in result["answer"]
    assert result["sources"][0]["page_number"] == 1
    assert result["sources"][0]["text"].startswith(f"{expected_heading}\n")


@pytest.mark.parametrize("response_language", ["hi", "gu"])
def test_missing_scholarship_amount_abstains_in_selected_language(
    monkeypatch: pytest.MonkeyPatch,
    response_language: str,
) -> None:
    model_calls: list[bool] = []

    async def invent_amount(*args: object, **kwargs: object) -> str:
        model_calls.append(True)
        return "The scholarship amount is ₹50,000."

    monkeypatch.setattr(qa, "generate_with_evon", invent_amount)
    response = ask(
        "શિષ્યવૃત્તિની રકમ કેટલી છે?",
        sample_notice_pages(),
        response_language=response_language,
    )

    result = response.json()
    assert response.status_code == 200
    assert result["status"] == "not_found"
    assert result["answer"] == qa.NOT_ENOUGH_INFORMATION[response_language]
    assert result["sources"] == []
    assert model_calls == []


def test_missing_document_details_abstain_instead_of_using_how_to_apply() -> None:
    response = ask(
        "What documents are required for the scholarship?",
        [{
            "page_number": 3,
            "text": "Documents Required\n"
            "How to Apply\n"
            "Prepare the listed documents and submit the scholarship application "
            "through your institution's student support office.",
        }],
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "not_found"
    assert result["answer"] == "The notice does not provide enough information to answer that question."
    assert result["sources"] == []


def test_application_deadline_is_retrieved_from_the_deadline_section() -> None:
    response = ask(
        "What is the application deadline?",
        [{
            "page_number": 5,
            "text": "Application Deadline\n"
            "Applications close on 30 June 2026.\n"
            "How to Apply\nSubmit the application through the student support office.",
        }],
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "answered"
    assert "30 June 2026" in result["answer"]
    assert result["sources"] == [
        {"page_number": 5, "text": "Applications close on 30 June 2026."}
    ]


def test_missing_scholarship_amount_is_not_guessed() -> None:
    response = ask(
        "How much is the scholarship amount?",
        [{
            "page_number": 2,
            "text": "Documents Required\n- Student identity card\n"
            "How to Apply\nSubmit the scholarship application through your institution.",
        }],
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "not_found"
    assert result["answer"] == "The notice does not provide enough information to answer that question."
    assert result["sources"] == []


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
