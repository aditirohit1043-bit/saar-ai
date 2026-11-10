"""API tests for grounded document questions and source handling."""

import asyncio
import unicodedata
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
    assert amount_result["answer"] == "The award amount is Rs. 25,000 per year."
    assert amount_result["sources"] == [
        {"page_number": 2, "text": "The award amount is Rs. 25,000 per year."}
    ]


@pytest.mark.parametrize(
    "question",
    [
        "What is the scholarship amount?",
        "How much scholarship can I get?",
        "How much financial assistance is offered?",
        "What is the award amount?",
        "How much money will I receive?",
    ],
)
def test_english_amount_phrasings_return_natural_cited_notice_amount(
    question: str,
) -> None:
    notice = [{
        "page_number": 3,
        "text": "Scholarship Amount\n"
        "The award amount is Rs. 25,000 per year for applicants who maintain "
        "satisfactory academic progress.\n"
        "This amount is subject to annual review.\n"
        "Eligibility Criteria\nApplicants must be enrolled at a recognized institution.",
    }]

    assert qa.map_question_for_retrieval(question) == (
        qa.CANONICAL_QUESTIONS_BY_INTENT["amount"],
        "amount",
    )
    response = ask(question, notice)
    result = response.json()

    assert response.status_code == 200
    assert result["status"] == "answered"
    assert result["answer"] == (
        "The award amount is Rs. 25,000 per year for applicants who maintain "
        "satisfactory academic progress. This amount is subject to annual review."
    )
    assert result["sources"] == [{
        "page_number": 3,
        "text": "Scholarship Amount\n"
        "The award amount is Rs. 25,000 per year for applicants who maintain "
        "satisfactory academic progress.\n"
        "This amount is subject to annual review.",
    }]


@pytest.mark.parametrize(
    "question",
    [
        "What is the scholarship amount?",
        "How much scholarship can I get?",
        "How much financial assistance is offered?",
        "What is the award amount?",
        "How much money will I receive?",
    ],
)
def test_english_amount_phrasings_abstain_when_project_notice_has_no_amount(
    question: str,
) -> None:
    assert qa.map_question_for_retrieval(question) == (
        qa.CANONICAL_QUESTIONS_BY_INTENT["amount"],
        "amount",
    )
    response = ask(question, sample_notice_pages())
    result = response.json()

    assert response.status_code == 200
    assert result["status"] == "not_found"
    assert result["answer"] == qa.NOT_ENOUGH_INFORMATION["en"]
    assert result["sources"] == []


def test_application_fee_is_not_mistaken_for_a_scholarship_amount() -> None:
    response = ask(
        "What is the scholarship amount?",
        [{
            "page_number": 2,
            "text": "Scholarship Application Fee\n"
            "The scholarship application fee amount is Rs. 500.",
        }],
    )
    result = response.json()

    assert response.status_code == 200
    assert result["status"] == "not_found"
    assert result["answer"] == qa.NOT_ENOUGH_INFORMATION["en"]
    assert result["sources"] == []


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

    monkeypatch.setattr(
        qa, "generate_with_evon", return_how_to_apply_incorrectly, raising=False
    )
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
    "question",
    [
        "What should I submit along with my scholarship application?",
        "Which documents should I submit?",
        "What paperwork is required for the application?",
        "What do I need to provide for this scholarship?",
    ],
)
def test_english_document_requirement_paraphrases_use_the_named_section(
    question: str,
) -> None:
    assert qa.map_english_question(question) == (
        qa.CANONICAL_QUESTIONS_BY_INTENT["documents"],
        "documents",
    )

    result = ask(question, sample_notice_pages()).json()

    assert result["status"] == "answered"
    assert result["answer"] == (
        "The required documents are: Student identity card; Latest academic marksheet; "
        "Proof of family income; Bank account details or passbook copy; "
        "A recent passport-size photograph."
    )
    assert result["sources"][0]["page_number"] == 1
    assert result["sources"][0]["text"].startswith("Documents Required\n")
    for item in (
        "Student identity card",
        "Latest academic marksheet",
        "Proof of family income",
        "Bank account details or passbook copy",
        "A recent passport-size photograph",
    ):
        assert item in result["answer"]
        assert item in result["sources"][0]["text"]


@pytest.mark.parametrize(
    "question",
    [
        "What should I submit along with my scholarship application?",
        "Which documents should I submit?",
        "What paperwork is required for the application?",
        "What do I need to provide for this scholarship?",
    ],
)
def test_english_document_requirement_paraphrases_abstain_without_list_evidence(
    question: str,
) -> None:
    result = ask(
        question,
        [{
            "page_number": 1,
            "text": "How to Apply\nSubmit the scholarship application through your institution.",
        }],
    ).json()

    assert result["status"] == "not_found"
    assert result["answer"] == qa.NOT_ENOUGH_INFORMATION["en"]
    assert result["sources"] == []


def test_last_date_to_apply_uses_deadline_from_actual_project_pdf() -> None:
    extracted_pages = sample_notice_pages()
    question = "What is the last date to apply?"

    assert any(
        "Application Deadline: 30 November 2026" in page["text"]
        for page in extracted_pages
    )
    assert qa.map_question_for_retrieval(question) == (
        qa.CANONICAL_QUESTIONS_BY_INTENT["deadline"],
        "deadline",
    )

    response = ask(question, extracted_pages)
    result = response.json()

    assert response.status_code == 200
    assert result["status"] == "answered"
    assert "30 November 2026" in result["answer"]
    assert result["sources"] == [{
        "page_number": 1,
        "text": "Application Deadline\n30 November 2026 (fictional date)",
    }]


@pytest.mark.parametrize(
    "question",
    [
        "What are the eligibility criteria?",
        "Who can apply for this scholarship?",
        "Am I eligible for this scholarship?",
        "What qualifications do applicants need?",
    ],
)
def test_english_eligibility_phrasings_return_natural_cited_criteria(
    question: str,
) -> None:
    extracted_pages = sample_notice_pages()
    assert qa.map_question_for_retrieval(question) == (
        qa.CANONICAL_QUESTIONS_BY_INTENT["eligibility"],
        "eligibility",
    )

    response = ask(question, extracted_pages)
    result = response.json()

    assert response.status_code == 200
    assert result["status"] == "answered"
    assert result["answer"] == (
        "Eligibility criteria stated in the notice:\n"
        "• Applicants must be currently enrolled in an undergraduate program at a recognized institution.\n"
        "• Applicants should demonstrate financial need and maintain satisfactory academic progress."
    )
    assert "you are eligible" not in result["answer"].casefold()
    assert result["sources"][0]["page_number"] == 1
    assert result["sources"][0]["text"] == (
        "Eligibility Criteria\n"
        "Applicants must be currently enrolled in an undergraduate program at a recognized institution.\n"
        "Applicants should demonstrate financial need and maintain satisfactory academic progress."
    )
    assert "currently enrolled in an undergraduate program" in result["sources"][0]["text"]
    assert "financial need" in result["sources"][0]["text"]
    assert "satisfactory academic progress" in result["sources"][0]["text"]
    assert "fictional" not in result["answer"].casefold()


def test_eligibility_answer_keeps_only_relevant_bullets_and_page_sources() -> None:
    pages = [
        {
            "page_number": 1,
            "text": "Scheme Overview\n"
            "This notice describes the education loan interest subsidy scheme.",
        },
        {
            "page_number": 2,
            "text": "Eligibility Criteria\n"
            "- Applicants must be permanent residents of Gujarat.\n"
            "- Students must be enrolled full-time in a recognized course.\n"
            "- The applicant's annual family income should not exceed Rs. 6 lakh.\n"
            "For more information, visit https://education.gov.in/loan-subsidy.\n"
            "Related Schemes\n"
            "Another scheme offers vocational training and hostel support to students.",
        },
        {
            "page_number": 3,
            "text": "Eligibility Criteria (continued)\n"
            "- Applicants should maintain satisfactory academic progress.\n"
            "Official website: https://education.gov.in/eligibility",
        },
    ]

    response = ask(
        "What is the eligibility criteria for this scheme?",
        pages,
    )
    result = response.json()

    assert response.status_code == 200
    assert result["status"] == "answered"
    assert result["answer"] == (
        "Eligibility criteria stated in the notice:\n"
        "• Applicants must be permanent residents of Gujarat.\n"
        "• Students must be enrolled full-time in a recognized course.\n"
        "• The applicant's annual family income should not exceed Rs. 6 lakh.\n"
        "• Applicants should maintain satisfactory academic progress."
    )
    assert result["sources"] == [
        {
            "page_number": 2,
            "text": "Eligibility Criteria\n"
            "- Applicants must be permanent residents of Gujarat.\n"
            "- Students must be enrolled full-time in a recognized course.\n"
            "- The applicant's annual family income should not exceed Rs. 6 lakh.",
        },
        {
            "page_number": 3,
            "text": "Eligibility Criteria (continued)\n"
            "- Applicants should maintain satisfactory academic progress.",
        },
    ]
    assert "https://" not in result["answer"]
    assert "Related Schemes" not in result["answer"]
    assert "vocational training" not in result["answer"]
    assert len(result["answer"]) < 400


def test_english_eligibility_question_abstains_without_eligibility_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_calls: list[bool] = []

    async def invent_eligibility(*args: object, **kwargs: object) -> str:
        model_calls.append(True)
        return "Applicants must have a 3.5 GPA."

    monkeypatch.setattr(qa, "generate_with_evon", invent_eligibility, raising=False)
    response = ask(
        "Who can apply for this scholarship?",
        [{
            "page_number": 1,
            "text": "Important Information\n"
            "Eligibility details are not included in this notice.",
        }],
    )
    result = response.json()

    assert response.status_code == 200
    assert result["status"] == "not_found"
    assert result["answer"] == qa.NOT_ENOUGH_INFORMATION["en"]
    assert result["sources"] == []
    assert model_calls == []


def test_eligibility_question_abstains_when_section_has_no_criteria(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_calls: list[bool] = []

    async def invent_eligibility(*args: object, **kwargs: object) -> str:
        model_calls.append(True)
        return "Applicants need a 3.5 GPA."

    monkeypatch.setattr(qa, "generate_with_evon", invent_eligibility, raising=False)
    response = ask(
        "Who is eligible for this scheme?",
        [{
            "page_number": 2,
            "text": "Eligibility Criteria\n"
            "See the list of related schemes at https://education.gov.in/schemes.",
        }],
    )
    result = response.json()

    assert response.status_code == 200
    assert result["status"] == "not_found"
    assert result["answer"] == qa.NOT_ENOUGH_INFORMATION["en"]
    assert result["sources"] == []
    assert model_calls == []


@pytest.mark.parametrize(
    "question",
    [
        "When should I submit my scholarship application?",
        "When do I need to submit the application?",
        "What is the application deadline?",
        "What is the application deadline for this scholarship?",
        "What is the deadline for scholarship applications?",
        "When do scholarship applications close?",
    ],
)
def test_application_deadline_phrasings_use_actual_project_pdf_section(
    question: str,
) -> None:
    extracted_pages = sample_notice_pages()

    assert qa.map_question_for_retrieval(question) == (
        qa.CANONICAL_QUESTIONS_BY_INTENT["deadline"],
        "deadline",
    )

    response = ask(question, extracted_pages)
    result = response.json()

    assert response.status_code == 200
    assert result["status"] == "answered"
    assert result["answer"] == (
        "You should submit your scholarship application by 30 November 2026. "
        "Please note that this date is marked as fictional in the uploaded notice."
    )
    assert result["sources"] == [{
        "page_number": 1,
        "text": "Application Deadline\n30 November 2026 (fictional date)",
    }]


def test_last_date_to_apply_abstains_when_only_notice_date_is_present() -> None:
    response = ask(
        "What is the last date to apply?",
        [{
            "page_number": 2,
            "text": "Notice Date: 2 November 2026\n"
            "How to Apply\nSubmit the application through the student support office.",
        }],
    )
    result = response.json()

    assert response.status_code == 200
    assert result["status"] == "not_found"
    assert result["answer"] == qa.NOT_ENOUGH_INFORMATION["en"]
    assert result["sources"] == []


def test_application_deadline_question_abstains_without_deadline_evidence() -> None:
    response = ask(
        "What is the application deadline for this scholarship?",
        [{
            "page_number": 1,
            "text": "The scholarship application is available through this sample notice.",
        }],
    )
    result = response.json()

    assert response.status_code == 200
    assert result["status"] == "not_found"
    assert result["answer"] == qa.NOT_ENOUGH_INFORMATION["en"]
    assert result["sources"] == []


def test_natural_deadline_answer_preserves_other_source_caveats() -> None:
    response = ask(
        "What is the application deadline?",
        [{
            "page_number": 2,
            "text": "Application Deadline\n"
            "Applications close on 12 July 2027 (subject to confirmation).",
        }],
    )
    result = response.json()

    assert response.status_code == 200
    assert result["answer"] == (
        "You should submit your scholarship application by 12 July 2027. "
        "The notice also notes: “(subject to confirmation)”."
    )
    assert result["sources"] == [{
        "page_number": 2,
        "text": "Application Deadline\n"
        "Applications close on 12 July 2027 (subject to confirmation).",
    }]


def test_last_date_to_apply_accepts_a_named_deadline_without_a_year() -> None:
    response = ask(
        "What is the last date to apply?",
        [{"page_number": 3, "text": "Application Deadline\n30 November"}],
    )
    result = response.json()

    assert response.status_code == 200
    assert result["status"] == "answered"
    assert "30 November" in result["answer"]
    assert result["sources"] == [
        {"page_number": 3, "text": "Application Deadline\n30 November"}
    ]


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
    ("question", "expected_intent"),
    [
        ("छात्रवृत्ति के आवेदन में कौन से कागज़ात जमा करने होंगे?", "documents"),
        ("आवेदन कब तक कर सकते हैं?", "deadline"),
        ("छात्रवृत्ति कितनी मिलेगी?", "amount"),
        ("कौन-कौन छात्र इस छात्रवृत्ति के लिए योग्य हैं?", "eligibility"),
        ("आवेदन कहाँ जमा करना है?", "how_to_apply"),
        ("શિષ્યવૃત્તિ માટે કયા કાગળો જરૂરી છે?", "documents"),
        ("અરજી ક્યારે બંધ થાય છે?", "deadline"),
        ("શિષ્યવૃત્તિની રકમ કેટલી છે?", "amount"),
        ("કયા વિદ્યાર્થીઓ શિષ્યવૃત્તિ માટે લાયક છે?", "eligibility"),
        ("અરજી ક્યાં જમા કરાવવી?", "how_to_apply"),
    ],
)
def test_hindi_and_gujarati_paraphrases_map_to_common_intents(
    question: str,
    expected_intent: str,
) -> None:
    normalized, intent = qa.map_question_for_retrieval(question)

    assert normalized == qa.CANONICAL_QUESTIONS_BY_INTENT[expected_intent]
    assert intent == expected_intent


@pytest.mark.parametrize(
    ("question", "expected_intent"),
    [
        ("छात्रवृत्ति की आखिरी तारीख क्या है?", "deadline"),
        ("आवेदन की अंतिम तिथि कब है?", "deadline"),
        ("कितनी छात्रवृत्ति मिलेगी?", "amount"),
        ("राशि छात्रवृत्ति की कितनी है?", "amount"),
        ("शिष्यवृत्ती માટે અરજી કરવાની છેલ્લી તારીખ શું છે?", "deadline"),
        ("શિષ્યવૃત્તિ માટે ક્યારે સુધી અરજી કરી શકું?", "deadline"),
        ("કેટલી સહાય મળશે?", "amount"),
        ("શિષ્યવૃત્તિની અરજી કરવાની રીત શું છે?", "how_to_apply"),
    ],
)
def test_intent_matching_handles_word_order_variants(
    question: str,
    expected_intent: str,
) -> None:
    normalized, intent = qa.map_question_for_retrieval(question)

    assert normalized == qa.CANONICAL_QUESTIONS_BY_INTENT[expected_intent]
    assert intent == expected_intent


@pytest.mark.parametrize(
    ("question", "expected_intent"),
    [
        ("छात्रवृत्ति के लिए कौन-कौन से दस्तावेज़ आवश्यक हैं?", "documents"),
        ("શિષ્યવૃત્તિ માટે કયા દસ્તાવેજો જરૂરી છે?", "documents"),
    ],
)
def test_intent_matching_normalizes_unicode_composition(
    question: str,
    expected_intent: str,
) -> None:
    decomposed_question = unicodedata.normalize("NFD", question)
    normalized, intent = qa.map_question_for_retrieval(decomposed_question)

    assert normalized == qa.CANONICAL_QUESTIONS_BY_INTENT[expected_intent]
    assert intent == expected_intent


def test_english_document_question_maps_to_document_intent() -> None:
    question = "What documents are required for the scholarship?"

    assert qa.map_question_for_retrieval(question) == (
        qa.CANONICAL_QUESTIONS_BY_INTENT["documents"],
        "documents",
    )


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


@pytest.mark.parametrize(
    ("question", "response_language", "intent", "answer_phrase"),
    [
        (
            "छात्रवृत्ति के आवेदन में कौन से कागज़ात जमा करने होंगे?",
            "hi", "documents", "छात्र पहचान पत्र",
        ),
        (
            "શિષ્યવૃત્તિ માટે કયા કાગળો જરૂરી છે?",
            "gu", "documents", "વિદ્યાર્થી ઓળખપત્ર",
        ),
        ("आवेदन की अंतिम तिथि कब है?", "hi", "deadline", "30 November 2026"),
        ("અરજી ક્યારે બંધ થાય છે?", "gu", "deadline", "30 November 2026"),
        (
            "कौन-कौन छात्र इस छात्रवृत्ति के लिए योग्य हैं?",
            "hi", "eligibility", "currently enrolled in an undergraduate program",
        ),
        (
            "કયા વિદ્યાર્થીઓ શિષ્યવૃત્તિ માટે લાયક છે?",
            "gu", "eligibility", "currently enrolled in an undergraduate program",
        ),
        (
            "आवेदन कहाँ जमा करना है?",
            "hi", "how_to_apply", "support office",
        ),
        (
            "અરજી ક્યાં જમા કરાવવી?",
            "gu", "how_to_apply", "support office",
        ),
    ],
)
def test_localized_paraphrases_return_matching_cited_project_pdf_section(
    question: str,
    response_language: str,
    intent: str,
    answer_phrase: str,
) -> None:
    response = ask(question, sample_notice_pages(), response_language=response_language)

    result = response.json()
    assert response.status_code == 200
    assert result["status"] == "answered"
    assert answer_phrase in result["answer"]
    assert result["sources"]
    assert result["sources"][0]["page_number"] == 1
    assert result["sources"][0]["text"].startswith(
        {
            "documents": "Documents Required\n",
            "deadline": "Application Deadline\n",
            "eligibility": "Eligibility Criteria\n",
            "how_to_apply": "How to Apply\n",
        }[intent]
    )


def test_gujarati_question_retrieves_unicode_normalized_gujarati_document_list() -> None:
    heading = unicodedata.normalize("NFD", "જરૂરી દસ્તાવેજો")
    notice = (
        f"{heading}\n"
        "- વિદ્યાર્થી ઓળખપત્ર\n"
        "- તાજેતરની ગુણપત્રક\n"
        "- કુટુંબની આવકનો પુરાવો"
    )

    response = ask(
        "શિષ્યવૃત્તિ માટે કયા દસ્તાવેજો જરૂરી છે?",
        [{"page_number": 2, "text": notice}],
        response_language="gu",
    )
    result = response.json()

    assert response.status_code == 200
    assert result["status"] == "answered"
    assert "વિદ્યાર્થી ઓળખપત્ર" in result["answer"]
    assert "તાજેતરની ગુણપત્રક" in result["answer"]
    assert "કુટુંબની આવકનો પુરાવો" in result["answer"]
    assert result["sources"] == [{
        "page_number": 2,
        "text": f"{unicodedata.normalize('NFC', heading)}\n"
        "- વિદ્યાર્થી ઓળખપત્ર\n"
        "- તાજેતરની ગુણપત્રક\n"
        "- કુટુંબની આવકનો પુરાવો",
    }]


def test_hindi_deadline_question_retrieves_gujarati_source_on_correct_page() -> None:
    pages = [
        {"page_number": 1, "text": "યોજનાની સામાન્ય માહિતી અને સંપર્ક વિગતો."},
        {
            "page_number": 2,
            "text": "અરજીની છેલ્લી તારીખ\nઅરજી કરવાની છેલ્લી તારીખ 30 November 2026 છે.",
        },
    ]

    response = ask(
        "आवेदन की आखिरी तारीख क्या है?",
        pages,
        response_language="hi",
    )
    result = response.json()

    assert response.status_code == 200
    assert result["status"] == "answered"
    assert "30 November 2026" in result["answer"]
    assert result["sources"] == [{
        "page_number": 2,
        "text": "અરજીની છેલ્લી તારીખ\nઅરજી કરવાની છેલ્લી તારીખ 30 November 2026 છે.",
    }]


def test_unsupported_question_abstains_even_if_a_model_would_guess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_calls: list[bool] = []

    async def invent_travel_benefit(*args: object, **kwargs: object) -> str:
        model_calls.append(True)
        return "The notice covers travel expenses."

    monkeypatch.setattr(
        qa, "generate_with_evon", invent_travel_benefit, raising=False
    )
    response = ask(
        "Does the notice cover travel expenses?",
        [{
            "page_number": 1,
            "text": "Education Loan Interest Subsidy\n"
            "The scheme supports interest on eligible education loans.",
        }],
    )
    result = response.json()

    assert response.status_code == 200
    assert result["status"] == "not_found"
    assert result["answer"] == qa.NOT_ENOUGH_INFORMATION["en"]
    assert result["sources"] == []
    assert model_calls == []


def test_hindi_question_uses_supported_section_without_unvalidated_model_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_calls: list[bool] = []

    async def return_unsupported_answer(*args: object, **kwargs: object) -> str:
        model_calls.append(True)
        return "Applicants with a 3.5 GPA qualify."

    monkeypatch.setattr(
        qa, "generate_with_evon", return_unsupported_answer, raising=False
    )
    original_question = "कौन-कौन छात्र इस छात्रवृत्ति के लिए योग्य हैं?"
    response = ask(original_question, sample_notice_pages(), response_language="hi")

    result = response.json()
    assert response.status_code == 200
    assert result["status"] == "answered"
    assert result["mode"] == "extractive"
    assert "currently enrolled in an undergraduate program" in result["answer"]
    assert "3.5 GPA" not in result["answer"]
    assert result["sources"][0]["page_number"] == 1
    assert model_calls == []


@pytest.mark.parametrize(
    ("question", "response_language"),
    [
        ("कितनी छात्रवृत्ति मिलेगी?", "hi"),
        ("કેટલી સહાય મળશે?", "gu"),
    ],
)
def test_localized_missing_amount_question_abstains_via_api(
    question: str,
    response_language: str,
) -> None:
    response = ask(question, sample_notice_pages(), response_language=response_language)

    result = response.json()
    assert response.status_code == 200
    assert result["status"] == "not_found"
    assert result["answer"] == qa.NOT_ENOUGH_INFORMATION[response_language]
    assert result["sources"] == []


@pytest.mark.parametrize(
    ("question", "response_language"),
    [
        ("छात्रवृत्ति की रकम कितनी है?", "hi"),
        ("કેટલી સહાય મળશે?", "gu"),
    ],
)
def test_localized_amount_question_returns_only_the_supported_section(
    question: str,
    response_language: str,
) -> None:
    response = ask(
        question,
        [{
            "page_number": 4,
            "text": "Scholarship Amount\nThe award amount is Rs. 25,000 per year.\n"
            "Eligibility Criteria\nApplicants must be enrolled at a recognized institution.",
        }],
        response_language=response_language,
    )

    result = response.json()
    assert response.status_code == 200
    assert result["status"] == "answered"
    assert "Rs. 25,000 per year" in result["answer"]
    assert result["sources"] == [
        {
            "page_number": 4,
            "text": "Scholarship Amount\nThe award amount is Rs. 25,000 per year.",
        }
    ]


def test_gujarati_subsidy_amount_question_uses_english_subsidy_section() -> None:
    source_text = (
        "Amount of Subsidy\n"
        "Subsidy is available on education Loan up to Rs.10lakh."
    )
    response = ask(
        "આ યોજનામાં કેટલી સબસિડી મળે છે?",
        [{"page_number": 1, "text": source_text}],
        response_language="gu",
    )

    result = response.json()
    assert response.status_code == 200
    assert result["status"] == "answered"
    assert result["mode"] == "extractive"
    assert result["answer"] == (
        "નોટિસમાં સબસિડીની ચોક્કસ રકમ જણાવેલી નથી; તેમાં લખ્યું છે કે "
        "રૂ. 10 લાખ સુધીની શિક્ષણ લોન પર સબસિડી ઉપલબ્ધ છે."
    )
    assert result["sources"] == [{"page_number": 1, "text": source_text}]


def test_gujarati_subsidy_amount_question_abstains_without_amount_evidence() -> None:
    response = ask(
        "આ યોજનામાં કેટલી સબસિડી મળે છે?",
        [{
            "page_number": 1,
            "text": "Scheme Overview\nThe scheme supports students pursuing higher education.",
        }],
        response_language="gu",
    )

    result = response.json()
    assert response.status_code == 200
    assert result["status"] == "not_found"
    assert result["answer"] == qa.NOT_ENOUGH_INFORMATION["gu"]
    assert result["sources"] == []


def test_hindi_subsidy_amount_question_uses_english_subsidy_section() -> None:
    source_text = (
        "Amount of Subsidy\n"
        "Subsidy is available on education Loan up to Rs.10lakh."
    )
    response = ask(
        "इस योजना में कितनी सब्सिडी मिलती है?",
        [{"page_number": 1, "text": source_text}],
        response_language="hi",
    )

    result = response.json()
    assert response.status_code == 200
    assert result["status"] == "answered"
    assert result["mode"] == "extractive"
    assert result["answer"] == (
        "नोटिस में सब्सिडी की निश्चित राशि नहीं बताई गई है; इसमें कहा गया है कि "
        "₹ 10 लाख तक के शिक्षा ऋण पर सब्सिडी उपलब्ध है।"
    )
    assert result["sources"] == [{"page_number": 1, "text": source_text}]


def test_hindi_subsidy_amount_question_abstains_without_amount_evidence() -> None:
    response = ask(
        "इस योजना में कितनी सब्सिडी मिलती है?",
        [{
            "page_number": 1,
            "text": "Scheme Overview\nThe scheme supports students pursuing higher education.",
        }],
        response_language="hi",
    )

    result = response.json()
    assert response.status_code == 200
    assert result["status"] == "not_found"
    assert result["answer"] == qa.NOT_ENOUGH_INFORMATION["hi"]
    assert result["sources"] == []


@pytest.mark.parametrize(
    ("question", "response_language"),
    [
        ("वजीफा कब मिलेगा?", "hi"),
        ("વજીફો ક્યારે મળશે?", "gu"),
        ("શિષ્યવૃત્તિ ક્યારે સુધી મળશે?", "gu"),
    ],
)
def test_stipend_disbursement_timing_question_abstains_without_timing(
    question: str,
    response_language: str,
) -> None:
    notice = [{
        "page_number": 2,
        "text": "Application Deadline\nApplications close on 30 November 2026.\n"
        "Scholarship Amount\nThe award amount is Rs. 25,000 per year.",
    }]

    assert qa.map_question_for_retrieval(question) == (question, None)
    response = ask(question, notice, response_language=response_language)

    result = response.json()
    assert response.status_code == 200
    assert result["status"] == "not_found"
    assert result["answer"] == qa.NOT_ENOUGH_INFORMATION[response_language]
    assert result["sources"] == []


@pytest.mark.parametrize(
    ("question", "response_language"),
    [
        ("When is the application deadline, and who is eligible?", "en"),
        ("What is the deadline, and who can apply?", "en"),
        ("आवेदन की आखिरी तारीख क्या है और कौन पात्र है?", "hi"),
        ("અરજીની છેલ્લી તારીખ શું છે અને કોણ પાત્ર છે?", "gu"),
    ],
)
def test_compound_deadline_and_eligibility_question_gets_clarification(
    question: str,
    response_language: str,
) -> None:
    notice = [{
        "page_number": 1,
        "text": "Application Deadline\nApplications close on 30 November 2026.\n"
        "Eligibility Criteria\nApplicants must be enrolled at a recognized institution.",
    }]
    response = ask(question, notice, response_language=response_language)

    result = response.json()
    assert response.status_code == 200
    assert result["status"] == "not_found"
    assert result["answer"] == qa.MULTIPLE_INTENTS_CLARIFICATION[response_language]
    assert result["sources"] == []


@pytest.mark.parametrize(
    ("question", "response_language"),
    [
        ("छात्रवृत्ति के बारे में बताइए।", "hi"),
        ("શિષ્યવૃત્તિ વિશે જણાવો.", "gu"),
        ("छात्रवृत्ति कब मिलेगी?", "hi"),
        ("શિષ્યવૃત્તિ ક્યારે મળશે?", "gu"),
        ("Tell me more.", "en"),
        ("When should I submit?", "en"),
        ("Will I receive the scholarship?", "en"),
    ],
)
def test_ambiguous_question_does_not_force_an_intent_or_answer(
    question: str,
    response_language: str,
) -> None:
    notice = [{"page_number": 1, "text": "General notice information."}]

    assert qa.map_question_for_retrieval(question) == (question, None)
    response = ask(question, notice, response_language=response_language)

    result = response.json()
    assert response.status_code == 200
    assert result["status"] == "not_found"
    assert result["answer"] == qa.NOT_ENOUGH_INFORMATION[response_language]
    assert result["sources"] == []


@pytest.mark.parametrize(
    ("question", "response_language"),
    [
        ("आवश्यक दस्तावेज़ कौन से हैं?", "hi"),
        ("શિષ્યવૃત્તિ માટે કયા દસ્તાવેજો જરૂરી છે?", "gu"),
    ],
)
def test_localized_document_question_abstains_when_list_is_absent(
    question: str,
    response_language: str,
) -> None:
    notice = [{
        "page_number": 3,
        "text": "Documents Required\n"
        "How to Apply\n"
        "Prepare the listed documents and submit the application through the student support office.",
    }]
    response = ask(question, notice, response_language=response_language)

    result = response.json()
    assert response.status_code == 200
    assert result["status"] == "not_found"
    assert result["answer"] == qa.NOT_ENOUGH_INFORMATION[response_language]
    assert result["sources"] == []


def test_hindi_missing_scholarship_amount_abstains_via_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_calls: list[bool] = []

    async def invent_amount(*args: object, **kwargs: object) -> str:
        model_calls.append(True)
        return "छात्रवृत्ति की राशि ₹50,000 है।"

    monkeypatch.setattr(qa, "generate_with_evon", invent_amount, raising=False)
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
        return "INSUFFICIENT_INFORMATION"

    monkeypatch.setattr(qa, "generate_with_evon", decline_to_answer, raising=False)
    gujarati_question = "શિષ્યવૃત્તિ માટે અરજી કરવાની છેલ્લી તારીખ શું છે?"
    response = ask(gujarati_question, sample_notice_pages(), response_language="gu")

    result = response.json()
    assert response.status_code == 200
    assert result["status"] == "answered"
    assert "30 November 2026" in result["answer"]
    assert result["answer"].startswith(
        "તમારે તમારી શિષ્યવૃત્તિની અરજી 30 November 2026 સુધીમાં"
    )
    assert "કાલ્પનિક" in result["answer"]
    assert result["sources"][0]["page_number"] == 1
    assert result["sources"][0]["text"].startswith("Application Deadline\n")
    assert model_calls == []


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
    expected_intro = (
        qa.ELIGIBILITY_ANSWER_INTRODUCTIONS["gu"]
        if expected_heading == "Eligibility Criteria"
        else qa.EXTRACTIVE_INTRODUCTIONS["gu"]
    )
    assert result["answer"].startswith(expected_intro)
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

    monkeypatch.setattr(qa, "generate_with_evon", invent_amount, raising=False)
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
        {
            "page_number": 5,
            "text": "Application Deadline\nApplications close on 30 June 2026.",
        }
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
    assert result["answer"] == (
        "You should submit your scholarship application by 30 June 2026."
    )
    assert any(source["page_number"] == 4 for source in result["sources"])
    assert any("30 June 2026" in source["text"] for source in result["sources"])


def test_empty_document_is_handled_without_calling_a_model() -> None:
    response = ask("What is the deadline?", [{"page_number": 1, "text": "  "}])

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "empty_document"
    assert "couldn't find readable text" in result["answer"]
    assert "local OCR completed successfully" in result["message"]
    assert "doesn't support yet" not in result["message"]
    assert result["sources"] == []


def test_document_qa_keeps_deterministic_answer_when_model_would_invent_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_calls: list[bool] = []

    async def invent_answer(*args: object, **kwargs: object) -> str:
        model_calls.append(True)
        return "You must submit the application by 1 January 2030."

    monkeypatch.setattr(qa, "generate_with_evon", invent_answer, raising=False)
    response = ask(
        "What is the application deadline?",
        [{"page_number": 3, "text": "Applications close on 30 June 2026."}],
    )

    assert response.status_code == 200
    result = response.json()
    assert result["mode"] == "extractive"
    assert result["message"] is None
    assert "30 June 2026" in result["answer"]
    assert "2030" not in result["answer"]
    assert result["sources"][0]["page_number"] == 3
    assert model_calls == []


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
