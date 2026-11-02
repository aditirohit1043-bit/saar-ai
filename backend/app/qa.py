"""Bounded passage retrieval and source-grounded document answers."""

import math
import re
import textwrap
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

TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?।])\s+|\n+")

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
}


class _RankedPassage:
    def __init__(self, source: SourcePassage, score: float, order: int) -> None:
        self.source = source
        self.score = score
        self.order = order


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in TOKEN_PATTERN.findall(text.casefold())
        if token not in STOP_WORDS
    }


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


def _page_passages(page: DocumentPageText) -> list[SourcePassage]:
    segments = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(page.text) if part.strip()]
    passages: list[SourcePassage] = []
    current = ""

    def add_current() -> None:
        nonlocal current
        if current:
            passages.append(SourcePassage(page_number=page.page_number, text=current))
            current = ""

    for segment in segments:
        if len(segment) > MAX_PASSAGE_CHARACTERS:
            add_current()
            passages.extend(
                SourcePassage(page_number=page.page_number, text=chunk)
                for chunk in _split_long_segment(segment)
            )
        elif not current:
            current = segment
        elif len(current) + len(segment) + 1 <= MAX_PASSAGE_CHARACTERS:
            current = f"{current} {segment}"
        else:
            add_current()
            current = segment

    add_current()
    return passages


def _all_passages(pages: list[DocumentPageText]) -> list[SourcePassage]:
    return [passage for page in pages for passage in _page_passages(page)]


def retrieve_relevant_passages(
    question: str,
    pages: list[DocumentPageText],
) -> list[SourcePassage]:
    """Rank document passages by weighted query-term coverage and cap context size."""
    concepts = _query_concepts(question)
    candidates = _all_passages(pages)
    if not concepts or not candidates:
        return []

    passage_tokens = [_tokens(passage.text) for passage in candidates]
    document_frequencies = [
        sum(bool(concept & tokens) for tokens in passage_tokens)
        for concept in concepts
    ]
    weights = [
        math.log((len(candidates) + 1) / (frequency + 1)) + 1
        for frequency in document_frequencies
    ]
    total_weight = sum(weights)
    ranked: list[_RankedPassage] = []

    for order, (passage, tokens) in enumerate(zip(candidates, passage_tokens)):
        matched_weight = sum(
            weight
            for concept, weight in zip(concepts, weights)
            if concept & tokens
        )
        score = matched_weight / total_weight if total_weight else 0.0
        if score >= MIN_RELEVANCE_SCORE:
            ranked.append(_RankedPassage(passage, score, order))

    ranked.sort(key=lambda item: (-item.score, item.order))
    selected: list[SourcePassage] = []
    selected_text = 0
    seen: set[tuple[int, str]] = set()

    for item in ranked:
        key = (item.source.page_number, item.source.text)
        if key in seen:
            continue
        separator_size = 2 if selected else 0
        if selected_text + len(item.source.text) + separator_size > MAX_CONTEXT_CHARACTERS:
            continue
        selected.append(item.source)
        seen.add(key)
        selected_text += len(item.source.text) + separator_size
        if len(selected) == MAX_SELECTED_PASSAGES:
            break

    return selected


def _covers_query(question: str, passages: list[SourcePassage]) -> bool:
    concepts = _query_concepts(question)
    if not concepts or not passages:
        return False
    context_tokens = _tokens(" ".join(passage.text for passage in passages))
    return all(bool(concept & context_tokens) for concept in concepts)


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

    passages = retrieve_relevant_passages(request.question, pages)
    if not passages:
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
            "Evon isn't configured. Saar is showing matching text from the notice without summarizing it."
        )
    except Exception:
        model_answer = None
        evon_error_message = (
            "Evon couldn't be reached. Saar is showing matching text from the notice without summarizing it."
        )

    if model_answer is not None:
        if model_answer.strip() == INSUFFICIENT_INFORMATION_TOKEN:
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

    if not _covers_query(request.question, passages):
        return _not_found_response(
            response_language=request.response_language,
            message=evon_error_message,
        )

    first_passage = passages[0].text
    extractive_intro = EXTRACTIVE_INTRODUCTIONS[request.response_language]
    return DocumentAnswerResponse(
        answer=f'{extractive_intro} “{first_passage}”',
        sources=passages,
        status="answered",
        mode="extractive",
        message=evon_error_message,
    )
