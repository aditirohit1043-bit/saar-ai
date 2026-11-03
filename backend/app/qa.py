"""Bounded passage retrieval and source-grounded document answers."""

import math
import re
import textwrap
from dataclasses import dataclass
from typing import Literal

from .evon import (
    INSUFFICIENT_INFORMATION_TOKEN,
    EvonNotConfigured,
    generate_with_evon,
)
from .qa_types import (
    DocumentAnswerResponse,
    DocumentQuestionRequest,
    DocumentPageText,
    SourcePassage,
)

MAX_SOURCE_CHARACTERS = 1_000_000
MAX_PASSAGE_CHARACTERS = 900
MAX_SELECTED_PASSAGES = 4
MAX_CONTEXT_CHARACTERS = 4_000
MIN_RELEVANCE_SCORE = 0.18
MAX_EXTRACTIVE_ANSWER_CHARACTERS = 360

NOT_ENOUGH_INFORMATION = {
    "en": "The notice does not provide enough information to answer that question.",
    "hi": "इस सवाल का जवाब देने के लिए नोटिस में पर्याप्त जानकारी नहीं है।",
    "gu": "આ પ્રશ્નનો જવાબ આપવા માટે નોટિસમાં પૂરતી માહિતી નથી.",
}
EMPTY_DOCUMENT_ANSWER = {
    "en": "I couldn't find readable text in this notice, so I can't answer questions from it yet.",
    "hi": "इस नोटिस में पढ़ने योग्य टेक्स्ट नहीं मिला, इसलिए मैं अभी इसके बारे में जवाब नहीं दे सकता।",
    "gu": "આ નોટિસમાં વાંચી શકાય એવો ટેક્સ્ટ મળ્યો નથી, તેથી હું હજી તેના વિશે જવાબ આપી શકતો નથી.",
}
EXTRACTIVE_INTRODUCTIONS = {
    "en": "The notice says:",
    "hi": "नोटिस में लिखा है:",
    "gu": "નોટિસમાં લખ્યું છે:",
}
REQUIRED_DOCUMENT_INTRODUCTIONS = {
    "en": "The required documents are:",
    "hi": "आवश्यक दस्तावेज़ हैं:",
    "gu": "જરૂરી દસ્તાવેજો છે:",
}

TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
SENTENCE_SPLIT_PATTERN = re.compile(
    r"(?<=[!?।])\s+|(?i:(?<!Rs)(?<=\.)\s+(?=[A-Z]))|\n+"
)

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "could", "did", "do", "does", "for", "from", "give", "has", "have", "how",
    "i", "in", "is", "it", "me", "of", "on", "or", "please", "tell", "that",
    "the", "their", "there", "this", "to", "was", "were", "what", "when", "where",
    "which", "who", "why", "will", "with", "would", "you", "your",
    "क्या", "कब", "कैसे", "क्यों", "है", "हैं", "और", "में", "का", "की", "को",
    "શું", "ક્યારે", "કેમ", "કેવી", "છે", "અને", "માં", "નો", "ની", "ને",
}

TERM_FAMILIES = {
    "apply": {"apply", "application", "applications", "register", "registration", "submit"},
    "application": {"apply", "application", "applications", "register", "registration", "submit"},
    "applications": {"apply", "application", "applications", "register", "registration", "submit"},
    "deadline": {"deadline", "deadlines", "due", "date", "close", "closes", "closing", "until"},
    "deadlines": {"deadline", "deadlines", "due", "date", "close", "closes", "closing", "until"},
    "due": {"deadline", "deadlines", "due", "date", "close", "closes", "closing", "until"},
    "date": {"deadline", "deadlines", "due", "date", "close", "closes", "closing", "until"},
    "document": {"document", "documents", "paper", "papers", "certificate", "proof", "id", "form"},
    "documents": {"document", "documents", "paper", "papers", "certificate", "proof", "id", "form"},
    "paper": {"document", "documents", "paper", "papers", "certificate", "proof", "id", "form"},
    "required": {"required", "require", "need", "needs", "must", "bring", "submit"},
    "require": {"required", "require", "need", "needs", "must", "bring", "submit"},
    "need": {"required", "require", "need", "needs", "must", "bring", "submit"},
    "eligibility": {"eligibility", "eligible", "qualify", "qualification", "criteria"},
    "eligible": {"eligibility", "eligible", "qualify", "qualification", "criteria"},
    "fee": {"fee", "fees", "cost", "charge", "payment", "pay"},
    "fees": {"fee", "fees", "cost", "charge", "payment", "pay"},
    "scholarship": {"scholarship", "scholarships", "award", "awards"},
    "scholarships": {"scholarship", "scholarships", "award", "awards"},
    "award": {"scholarship", "scholarships", "award", "awards"},
    "awards": {"scholarship", "scholarships", "award", "awards"},
    "amount": {"amount", "amounts", "much", "money", "value", "values", "worth", "stipend"},
    "amounts": {"amount", "amounts", "much", "money", "value", "values", "worth", "stipend"},
    "much": {"amount", "amounts", "much", "money", "value", "values", "worth", "stipend"},
    "money": {"amount", "amounts", "much", "money", "value", "values", "worth", "stipend"},
    "stipend": {"amount", "amounts", "much", "money", "value", "values", "worth", "stipend"},
}

SECTION_HEADING_NAMES = {
    "documents required", "required documents", "documents needed",
    "required papers", "papers required", "how to apply",
    "application process", "application procedure", "eligibility",
    "eligibility criteria", "application deadline", "important dates",
    "key dates", "important information", "scholarship amount", "award amount",
}
LIST_MARKER_PATTERN = re.compile(r"^\s*(?:(?:[-*•▪◦])\s*|\d+\s*[.)]\s*)")
GUJARATI_SCRIPT_PATTERN = re.compile(r"[\u0a80-\u0aff]")
HINDI_SCRIPT_PATTERN = re.compile(r"[\u0900-\u097f]")
GUJARATI_SCHOLARSHIP_TERMS = ("શિષ્યવૃત્તિ", "સ્કોલરશિપ", "સ્કોલરશીપ")
GUJARATI_DOCUMENT_TERMS = ("દસ્તાવેજ", "કાગળ")
GUJARATI_REQUIRED_TERMS = (
    "જરૂરી", "આવશ્યક", "જોઈએ", "જોઇએ", "પડશે", "આપવા", "જમા", "સબમિટ", "કયા",
)
GUJARATI_DEADLINE_TERMS = ("છેલ્લી તારીખ", "અંતિમ તારીખ", "સમયમર્યાદા", "ક્યારે અરજી")
GUJARATI_AMOUNT_TERMS = ("રકમ", "કેટલી", "કેટલા", "કેટલું", "સહાય", "સ્ટાઇપેન્ડ")
GUJARATI_ELIGIBILITY_TERMS = ("પાત્રતા", "લાયકાત", "પાત્ર", "લાયક")
GUJARATI_APPLICATION_TERMS = ("અરજી",)
GUJARATI_HOW_TO_APPLY_TERMS = ("કેવી રીતે", "કઈ રીતે", "ક્યાં", "પ્રક્રિયા")
HINDI_DOCUMENT_TERMS = ("दस्तावेज़", "दस्तावेज", "कागजात", "कागज़ात", "कागज", "कागज़")
HINDI_REQUIRED_DOCUMENT_TERMS = (
    "आवश्यक", "जरूरी", "ज़रूरी", "अनिवार्य", "चाहिए", "कौन-कौन से",
    "कौन कौन से", "कौन-कौन-से", "कौन से", "कौनसे",
)
AMOUNT_VALUE_PATTERN = re.compile(
    r"(?:₹\s*\d[\d,]*|\b(?:INR|Rs\.?|rupees?)\s*\d[\d,]*|"
    r"\b(?:amount|stipend|award|grant|worth)\b[^.\n]{0,60}\b\d[\d,]*)",
    re.IGNORECASE,
)
SECTION_NAMES_BY_INTENT = {
    "deadline": {"application deadline", "important dates", "key dates"},
    "amount": {"scholarship amount", "award amount"},
    "eligibility": {"eligibility", "eligibility criteria"},
    "how_to_apply": {"how to apply", "application process", "application procedure"},
}


@dataclass(frozen=True)
class _DocumentSection:
    heading: str
    lines: list[str]


@dataclass(frozen=True)
class _PassageCandidate:
    source: SourcePassage
    heading: str
    body: str


class _RankedPassage:
    def __init__(self, candidate: _PassageCandidate, score: float, order: int) -> None:
        self.source = candidate.source
        self.heading = candidate.heading
        self.body = candidate.body
        self.score = score
        self.order = order


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in TOKEN_PATTERN.findall(text.casefold())
        if token not in STOP_WORDS
    }


def map_gujarati_question(question: str) -> tuple[str, str | None]:
    """Map common Gujarati scholarship questions to English retrieval queries."""
    if not GUJARATI_SCRIPT_PATTERN.search(question):
        return question, None

    text = " ".join(question.casefold().split())
    has_scholarship = any(term in text for term in GUJARATI_SCHOLARSHIP_TERMS)
    has_application = any(term in text for term in GUJARATI_APPLICATION_TERMS)

    if (
        any(term in text for term in GUJARATI_DOCUMENT_TERMS)
        and any(term in text for term in GUJARATI_REQUIRED_TERMS)
    ):
        return "What documents are required for the scholarship?", "documents"
    if (
        any(term in text for term in GUJARATI_DEADLINE_TERMS)
        and (has_scholarship or has_application)
    ):
        return "What is the application deadline?", "deadline"
    if (
        any(term in text for term in GUJARATI_AMOUNT_TERMS)
        and has_scholarship
    ):
        return "What is the scholarship amount?", "amount"
    if any(term in text for term in GUJARATI_ELIGIBILITY_TERMS):
        return "What are the eligibility criteria?", "eligibility"
    if (
        has_application
        and any(term in text for term in GUJARATI_HOW_TO_APPLY_TERMS)
    ):
        return "How do I apply for the scholarship?", "how_to_apply"
    return question, None


def map_hindi_question(question: str) -> tuple[str, str | None]:
    """Map common Hindi document questions to English retrieval terms."""
    if not HINDI_SCRIPT_PATTERN.search(question):
        return question, None

    text = " ".join(question.casefold().split())
    has_document_term = any(term in text for term in HINDI_DOCUMENT_TERMS)
    has_required_term = any(term in text for term in HINDI_REQUIRED_DOCUMENT_TERMS)
    if has_document_term and has_required_term:
        return "What documents are required for the scholarship?", "documents"
    return question, None


def map_question_for_retrieval(question: str) -> tuple[str, str | None]:
    """Normalize supported Gujarati and Hindi intents for English document retrieval."""
    mapped_question, intent = map_gujarati_question(question)
    if intent is not None or mapped_question != question:
        return mapped_question, intent
    return map_hindi_question(question)


def _translate_document_item(item: str, response_language: str) -> str:
    """Translate common document names; retain unfamiliar source terms verbatim."""
    if response_language == "en":
        return item

    normalized = re.sub(r"\s+", " ", item.casefold()).strip(" .,:;-\u2022")
    normalized = re.sub(r"^(?:a|an|the)\s+", "", normalized)
    recent = "latest" in normalized or "recent" in normalized
    student = "student" in normalized

    if "identity card" in normalized or "id card" in normalized:
        if response_language == "hi":
            return "छात्र पहचान पत्र" if student else "पहचान पत्र"
        return "વિદ્યાર્થી ઓળખપત્ર" if student else "ઓળખપત્ર"

    if any(term in normalized for term in ("marksheet", "mark sheet", "transcript", "grade sheet")):
        if response_language == "hi":
            label = "शैक्षणिक अंकपत्र" if "academic" in normalized else "अंकपत्र"
            return f"नवीनतम {label}" if recent else label
        label = "શૈક્ષણિક ગુણપત્રક" if "academic" in normalized else "ગુણપત્રક"
        return f"નવીનતમ {label}" if recent else label

    if "family income" in normalized and any(
        term in normalized for term in ("proof", "certificate", "evidence")
    ):
        return "परिवार की आय का प्रमाण" if response_language == "hi" else "પરિવારની આવકનો પુરાવો"
    if "income certificate" in normalized:
        return "आय प्रमाणपत्र" if response_language == "hi" else "આવકનું પ્રમાણપત્ર"
    if "caste certificate" in normalized:
        return "जाति प्रमाणपत्र" if response_language == "hi" else "જાતિનું પ્રમાણપત્ર"
    if any(term in normalized for term in ("domicile certificate", "residence certificate")):
        return "निवास प्रमाणपत्र" if response_language == "hi" else "રહેઠાણનું પ્રમાણપત્ર"

    if "bank account details" in normalized and "passbook copy" in normalized:
        return (
            "बैंक खाते का विवरण या पासबुक की प्रति"
            if response_language == "hi"
            else "બેંક ખાતાની વિગતો અથવા પાસબુકની નકલ"
        )
    if "bank account details" in normalized:
        return "बैंक खाते का विवरण" if response_language == "hi" else "બેંક ખાતાની વિગતો"
    if "passbook copy" in normalized:
        return "पासबुक की प्रति" if response_language == "hi" else "પાસબુકની નકલ"

    if "passport" in normalized and any(term in normalized for term in ("photograph", "photo")):
        if response_language == "hi":
            prefix = "हाल की " if recent else ""
            return f"{prefix}पासपोर्ट आकार की तस्वीर"
        prefix = "તાજેતરનો " if recent else ""
        return f"{prefix}પાસપોર્ટ કદનો ફોટો"

    if "aadhaar" in normalized or "aadhar" in normalized:
        return "आधार कार्ड" if response_language == "hi" else "આધાર કાર્ડ"
    if "birth certificate" in normalized:
        return "जन्म प्रमाणपत्र" if response_language == "hi" else "જન્મ પ્રમાણપત્ર"
    if "fee receipt" in normalized:
        return "शुल्क रसीद" if response_language == "hi" else "ફીની રસીદ"

    return item


def _query_concepts(question: str) -> list[set[str]]:
    concepts: list[set[str]] = []
    for token in TOKEN_PATTERN.findall(question.casefold()):
        if token in STOP_WORDS:
            continue
        family = set(TERM_FAMILIES.get(token, {token}))
        overlapping = [index for index, concept in enumerate(concepts) if concept & family]
        if not overlapping:
            concepts.append(family)
            continue

        first_index = overlapping[0]
        concepts[first_index].update(family)
        for index in reversed(overlapping[1:]):
            concepts[first_index].update(concepts.pop(index))
    return concepts


def _split_long_segment(segment: str) -> list[str]:
    chunks = textwrap.wrap(
        segment,
        width=MAX_PASSAGE_CHARACTERS,
        break_long_words=True,
        break_on_hyphens=False,
    )
    return chunks or [segment]


def _heading_name(value: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(value.casefold()))


def _heading_line(line: str) -> tuple[str, str] | None:
    """Return a recognized section heading and any text following its colon."""
    marker = LIST_MARKER_PATTERN.match(line)
    candidate = LIST_MARKER_PATTERN.sub("", line, count=1).strip() if marker else line

    heading, separator, remainder = candidate.partition(":")
    normalized = _heading_name(heading.strip())
    if normalized not in SECTION_HEADING_NAMES:
        normalized = _heading_name(candidate.strip())
        if normalized not in SECTION_HEADING_NAMES:
            return None
        heading, remainder, separator = candidate, "", ""

    return heading.strip().rstrip(":"), remainder.strip() if separator else ""


def _page_sections(page: DocumentPageText) -> list[_DocumentSection]:
    """Keep each recognized heading attached to the lines beneath it."""
    sections: list[_DocumentSection] = []
    heading = ""
    lines: list[str] = []

    def save_section() -> None:
        if heading or lines:
            sections.append(_DocumentSection(heading=heading, lines=lines.copy()))

    for raw_line in page.text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        heading_line = _heading_line(line)
        if heading_line:
            save_section()
            heading, inline_text = heading_line
            lines = [inline_text] if inline_text else []
        else:
            lines.append(line)

    save_section()
    if not sections and page.text.strip():
        return [_DocumentSection(heading="", lines=[page.text.strip()])]
    return sections


def _page_passages(page: DocumentPageText) -> list[_PassageCandidate]:
    passages: list[_PassageCandidate] = []
    for section in _page_sections(page):
        chunks: list[str] = []
        current = ""

        def add_current() -> None:
            nonlocal current
            if current:
                chunks.append(current)
                current = ""

        for line in section.lines:
            segments = [
                part.strip()
                for part in SENTENCE_SPLIT_PATTERN.split(line)
                if part.strip()
            ]
            for segment in segments:
                if len(segment) > MAX_PASSAGE_CHARACTERS:
                    add_current()
                    chunks.extend(_split_long_segment(segment))
                elif not current:
                    current = segment
                elif len(current) + len(segment) + 1 <= MAX_PASSAGE_CHARACTERS:
                    current = f"{current}\n{segment}"
                else:
                    add_current()
                    current = segment

        add_current()
        for body in chunks:
            source_text = f"{section.heading}\n{body}" if section.heading else body
            passages.append(
                _PassageCandidate(
                    source=SourcePassage(page_number=page.page_number, text=source_text),
                    heading=section.heading,
                    body=body,
                )
            )
    return passages


def _all_passages(pages: list[DocumentPageText]) -> list[_PassageCandidate]:
    return [passage for page in pages for passage in _page_passages(page)]


def _rank_relevant_passages(
    question: str,
    pages: list[DocumentPageText],
) -> list[_RankedPassage]:
    """Rank section chunks by query match and give matching headings extra weight."""
    concepts = _query_concepts(question)
    candidates = _all_passages(pages)
    if not concepts or not candidates:
        return []

    body_tokens = [_tokens(candidate.body) for candidate in candidates]
    heading_tokens = [_tokens(candidate.heading) for candidate in candidates]
    document_frequencies = [
        sum(bool(concept & (body | heading)) for body, heading in zip(body_tokens, heading_tokens))
        for concept in concepts
    ]
    weights = [
        math.log((len(candidates) + 1) / (frequency + 1)) + 1
        for frequency in document_frequencies
    ]
    total_weight = sum(weights)
    ranked: list[_RankedPassage] = []

    for order, (candidate, body, heading) in enumerate(
        zip(candidates, body_tokens, heading_tokens)
    ):
        matched_weight = sum(
            weight * (3 if concept & heading else 1)
            for concept, weight in zip(concepts, weights)
            if concept & (body | heading)
        )
        score = matched_weight / (total_weight * 3) if total_weight else 0.0
        if score >= MIN_RELEVANCE_SCORE:
            ranked.append(_RankedPassage(candidate, score, order))

    ranked.sort(key=lambda item: (-item.score, item.order))
    selected: list[_RankedPassage] = []
    selected_text = 0
    seen: set[tuple[int, str]] = set()

    for item in ranked:
        key = (item.source.page_number, item.source.text)
        if key in seen:
            continue
        separator_size = 2 if selected else 0
        if selected_text + len(item.source.text) + separator_size > MAX_CONTEXT_CHARACTERS:
            continue
        selected.append(item)
        seen.add(key)
        selected_text += len(item.source.text) + separator_size
        if len(selected) == MAX_SELECTED_PASSAGES:
            break

    return selected


def retrieve_relevant_passages(
    question: str,
    pages: list[DocumentPageText],
) -> list[SourcePassage]:
    """Rank section-aware source passages by query match and heading relevance."""
    return [item.source for item in _rank_relevant_passages(question, pages)]


def _is_required_documents_question(question: str) -> bool:
    concepts = _query_concepts(question)
    has_document_term = any(concept & TERM_FAMILIES["document"] for concept in concepts)
    has_required_term = any(concept & TERM_FAMILIES["required"] for concept in concepts)
    return has_document_term and has_required_term


def _required_documents_response(
    question: str,
    passages: list[_RankedPassage],
    response_language: str,
) -> tuple[str, SourcePassage] | None:
    """Build a document-list answer only from a matching required-documents section."""
    if not _is_required_documents_question(question):
        return None

    for passage in passages:
        heading_tokens = _tokens(passage.heading)
        if not (
            heading_tokens & TERM_FAMILIES["document"]
            and heading_tokens & TERM_FAMILIES["required"]
        ):
            continue

        lines = [line.strip() for line in passage.body.splitlines() if line.strip()]
        items = [LIST_MARKER_PATTERN.sub("", line).strip() for line in lines]
        items = [item.rstrip(";,. ") for item in items if item]
        if len(items) == 1 and ";" in items[0]:
            items = [item.strip().rstrip(".,") for item in items[0].split(";") if item.strip()]
        if not items:
            continue

        translated_items = [
            _translate_document_item(item, response_language)
            for item in items
        ]
        item_list = "; ".join(translated_items)
        answer = f"{REQUIRED_DOCUMENT_INTRODUCTIONS[response_language]} {item_list}."
        return answer, passage.source
    return None


def _mapped_section_response(
    intent: str | None,
    passages: list[_RankedPassage],
    response_language: str,
) -> tuple[str, SourcePassage] | None:
    """Quote the text under the section selected by a mapped Gujarati question."""
    section_names = SECTION_NAMES_BY_INTENT.get(intent or "")
    if not section_names:
        return None

    matching = [
        passage
        for passage in passages
        if _heading_name(passage.heading) in section_names and passage.body.strip()
    ]
    if not matching:
        return None

    selected = matching[0]
    same_section = sorted(
        (
            passage for passage in matching
            if passage.source.page_number == selected.source.page_number
            and passage.heading == selected.heading
        ),
        key=lambda passage: passage.order,
    )
    body = "\n".join(passage.body.strip() for passage in same_section).strip()
    answer = f'{EXTRACTIVE_INTRODUCTIONS[response_language]} “{body}”'
    source = SourcePassage(
        page_number=selected.source.page_number,
        text=f"{selected.heading}\n{body}",
    )
    return answer, source


def _has_explicit_amount(passages: list[_RankedPassage]) -> bool:
    return any(AMOUNT_VALUE_PATTERN.search(passage.source.text) for passage in passages)


def _extractive_support(
    question: str,
    passages: list[_RankedPassage],
) -> list[SourcePassage]:
    """Return concise, sentence-level evidence only when the question matches it."""
    concepts = _query_concepts(question)
    if not concepts or not passages:
        return []

    # Require the key terms to co-occur in one sentence. Combining unrelated
    # words from several passages can make lexical retrieval look more certain
    # than the notice supports.
    minimum_matches = 1 if len(concepts) == 1 else max(2, math.ceil(len(concepts) * 0.6))
    ranked: list[tuple[float, int, SourcePassage]] = []
    order = 0
    for passage in passages:
        for sentence in SENTENCE_SPLIT_PATTERN.split(passage.body):
            text = sentence.strip()
            if not text:
                continue
            tokens = _tokens(text)
            matches = sum(bool(concept & tokens) for concept in concepts)
            if matches < minimum_matches or len(text) > MAX_EXTRACTIVE_ANSWER_CHARACTERS:
                order += 1
                continue
            coverage = matches / len(concepts)
            evidence = SourcePassage(
                page_number=passage.source.page_number,
                text=text,
            )
            ranked.append((coverage, order, evidence))
            order += 1

    if not ranked:
        return []

    # One short sentence is easier to review and avoids presenting a retrieved
    # paragraph as though it were a generated answer.
    ranked.sort(key=lambda item: (-item[0], len(item[2].text), item[1]))
    return [ranked[0][2]]


def _not_found_response(
    *,
    response_language: str,
    mode: Literal["evon", "extractive"] = "extractive",
    message: str | None = None,
) -> DocumentAnswerResponse:
    return DocumentAnswerResponse(
        answer=NOT_ENOUGH_INFORMATION[response_language],
        sources=[],
        status="not_found",
        mode=mode,
        message=message,
    )


async def answer_document_question(
    request: DocumentQuestionRequest,
) -> DocumentAnswerResponse:
    pages = [page for page in request.pages if page.text.strip()]
    if not pages:
        return DocumentAnswerResponse(
            answer=EMPTY_DOCUMENT_ANSWER[request.response_language],
            sources=[],
            status="empty_document",
            mode="extractive",
            message="This notice has no readable text. Scanned pages need OCR, which Saar doesn't support yet.",
        )

    retrieval_question, mapped_intent = map_question_for_retrieval(request.question)
    ranked_passages = _rank_relevant_passages(retrieval_question, pages)
    if not ranked_passages:
        return _not_found_response(response_language=request.response_language)
    passages = [passage.source for passage in ranked_passages]

    if _is_required_documents_question(retrieval_question):
        document_answer = _required_documents_response(
            retrieval_question,
            ranked_passages,
            request.response_language,
        )
        if not document_answer:
            return _not_found_response(response_language=request.response_language)
        answer, source = document_answer
        return DocumentAnswerResponse(
            answer=answer,
            sources=[source],
            status="answered",
            mode="extractive",
        )

    mapped_section_answer = _mapped_section_response(
        mapped_intent,
        ranked_passages,
        request.response_language,
    )
    if mapped_intent == "amount" and not _has_explicit_amount(ranked_passages):
        return _not_found_response(response_language=request.response_language)

    evon_error_message: str | None = None
    try:
        model_answer = await generate_with_evon(
            request.question,
            passages,
            request.response_language,
        )
    except EvonNotConfigured:
        model_answer = None
        evon_error_message = (
            "Evon isn't configured. Saar is answering with short quotes from matching notice text."
        )
    except Exception:
        model_answer = None
        evon_error_message = (
            "Evon couldn't be reached. Saar is answering with short quotes from matching notice text."
        )

    if model_answer is not None:
        if model_answer.strip() == INSUFFICIENT_INFORMATION_TOKEN:
            if mapped_section_answer:
                answer, source = mapped_section_answer
                return DocumentAnswerResponse(
                    answer=answer,
                    sources=[source],
                    status="answered",
                    mode="extractive",
                )
            return _not_found_response(
                response_language=request.response_language,
                mode="evon",
            )
        return DocumentAnswerResponse(
            answer=model_answer,
            sources=passages,
            status="answered",
            mode="evon",
        )

    if mapped_section_answer:
        answer, source = mapped_section_answer
        return DocumentAnswerResponse(
            answer=answer,
            sources=[source],
            status="answered",
            mode="extractive",
            message=evon_error_message,
        )

    evidence = _extractive_support(retrieval_question, ranked_passages)
    if not evidence:
        return _not_found_response(
            response_language=request.response_language,
            message=evon_error_message,
        )

    first_passage = evidence[0].text
    extractive_intro = EXTRACTIVE_INTRODUCTIONS[request.response_language]
    return DocumentAnswerResponse(
        answer=f'{extractive_intro} “{first_passage}”',
        sources=evidence,
        status="answered",
        mode="extractive",
        message=evon_error_message,
    )
