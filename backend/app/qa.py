"""Bounded passage retrieval and source-grounded document answers."""

import math
import re
import textwrap
import unicodedata
from dataclasses import dataclass
from typing import Literal

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
DEADLINE_ANSWER_TEMPLATES = {
    "en": {
        "answer": "You should submit your scholarship application by {date}.",
        "fictional_caveat": (
            " Please note that this date is marked as fictional in the uploaded notice."
        ),
        "other_caveat": " The notice also notes: “{caveat}”.",
    },
    "hi": {
        "answer": "आपको अपना छात्रवृत्ति आवेदन {date} तक जमा करना चाहिए।",
        "fictional_caveat": (
            " कृपया ध्यान दें कि अपलोड किए गए नोटिस में इस तारीख को काल्पनिक बताया गया है।"
        ),
        "other_caveat": " नोटिस में यह भी लिखा है: “{caveat}”.",
    },
    "gu": {
        "answer": "તમારે તમારી શિષ્યવૃત્તિની અરજી {date} સુધીમાં સબમિટ કરવી જોઈએ.",
        "fictional_caveat": (
            " કૃપા કરીને ધ્યાન આપો કે અપલોડ કરેલી નોટિસમાં આ તારીખને કાલ્પનિક દર્શાવવામાં આવી છે."
        ),
        "other_caveat": " નોટિસમાં એમ પણ લખ્યું છે: “{caveat}”.",
    },
}
REQUIRED_DOCUMENT_INTRODUCTIONS = {
    "en": "The required documents are:",
    "hi": "आवश्यक दस्तावेज़ हैं:",
    "gu": "જરૂરી દસ્તાવેજો છે:",
}
ELIGIBILITY_ANSWER_INTRODUCTIONS = {
    "en": "Eligibility criteria stated in the notice:",
    "hi": "नोटिस में दी गई पात्रता शर्तें:",
    "gu": "નોટિસમાં દર્શાવેલી પાત્રતાની શરતો:",
}
ELIGIBILITY_CRITERIA_TERMS = frozenset({
    "applicant", "applicants", "student", "students", "borrower", "borrowers",
    "candidate", "candidates", "beneficiary", "beneficiaries", "resident", "residents",
    "residency", "domicile", "domiciled", "enrolled", "enrollment", "admitted",
    "admission", "pursuing", "studying", "course", "courses", "program", "programme",
    "institution", "college", "income", "family", "household", "age", "marks", "grades",
    "gpa", "category", "caste", "merit", "academic", "disability", "financial", "need",
    "qualify", "qualified", "must", "shall", "should", "requires", "required", "minimum",
    "maximum", "below", "above", "exceed", "fulltime", "आवेदक", "छात्र", "विद्यार्थी",
    "उम्मीदवार", "निवासी", "निवास", "पात्र", "योग्य", "आय", "परिवार", "आयु", "अध्ययन",
    "प्रवेश", "मान्यता", "अंक", "श्रेणी", "जाति", "दिव्यांग", "आवश्यक", "होना", "चाहिए",
    "અરજદાર", "ઉમેદવાર", "વિદ્યાર્થી", "વતની", "રહેવાસી", "પાત્ર", "લાયક", "આવક", "કુટુંબ",
    "ઉંમર", "અભ્યાસ", "પ્રવેશ", "માન્ય", "અભ્યાસક્રમ", "કોર્સ", "ગુણ", "અંક", "વર્ગ", "જાતિ",
    "દિવ્યાંગ", "આવશ્યક", "જોઈએ", "હોવો", "હોવી",
})
ELIGIBILITY_BOILERPLATE_PATTERN = re.compile(
    r"(?:https?://|www\.|\b[\w-]+\.(?:gov|nic|com|org|net|edu)\b|"
    r"[\w.+-]+@[\w.-]+\.\w+|\b(?:visit (?:our |the |official )?website|"
    r"for more information|contact (?:us|details)|helpline|help line)\b)",
    re.IGNORECASE,
)
MULTIPLE_INTENTS_CLARIFICATION = {
    "en": "Please ask about the application deadline and eligibility separately so I can point to the right parts of the notice.",
    "hi": "कृपया आवेदन की अंतिम तिथि और पात्रता के बारे में अलग-अलग पूछें, ताकि मैं नोटिस के सही हिस्से दिखा सकूँ।",
    "gu": "કૃપા કરીને અરજીની અંતિમ તારીખ અને પાત્રતા વિશે અલગ અલગ પૂછો, જેથી હું નોટિસના યોગ્ય વિભાગો બતાવી શકું.",
}

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

# These aliases are used only for local passage matching. They do not create
# answer content, and they keep Gujarati/Hindi OCR text searchable after a
# localized question has been mapped to a canonical English retrieval query.
RETRIEVAL_TERM_ALIAS_GROUPS = (
    frozenset({
        "document", "documents", "paper", "papers", "paperwork", "certificate", "proof",
        "दस्तावेज़", "दस्तावेज", "कागजात", "कागज़ात", "कागज", "काग़ज़", "प्रमाणपत्र",
        "દસ્તાવેજ", "દસ્તાવેજો", "કાગળ", "કાગળો", "પ્રમાણપત્ર", "સર્ટિફિકેટ",
    }),
    frozenset({
        "required", "require", "need", "needs", "must", "bring", "submit",
        "आवश्यक", "जरूरी", "ज़रूरी", "अनिवार्य", "चाहिए", "मांगे", "जमा",
        "જરૂરી", "આવશ્યક", "જોઈએ", "જોઇએ", "જમા", "સબમિટ",
    }),
    frozenset({
        "application", "applications", "apply", "register", "registration",
        "आवेदन", "आवेदन पत्र", "फॉर्म", "અરજી", "અરજીઓ", "ફોર્મ",
    }),
    frozenset({
        "deadline", "deadlines", "due", "date", "close", "closes", "closing", "until",
        "last", "अंतिम", "आखिरी", "तिथि", "तारीख", "समय", "सीमा",
        "છેલ્લી", "અંતિમ", "તારીખ", "મુદત", "સમયમર્યાદા",
    }),
    frozenset({
        "eligibility", "eligible", "qualify", "qualification", "criteria",
        "पात्रता", "योग्यता", "पात्र", "योग्य", "લાયકાત", "પાત્રતા", "યોગ્યતા", "લાયક",
    }),
    frozenset({
        "scholarship", "scholarships", "award", "awards", "student aid",
        "छात्रवृत्ति", "छात्रवृति", "स्कॉलरशिप", "વૃત્તિ", "શિષ્યવૃત્તિ", "વિદ્યાવૃત્તિ",
    }),
    frozenset({
        "amount", "amounts", "much", "money", "value", "values", "worth", "stipend",
        "assistance", "grant", "राशि", "रकम", "धनराशि", "सहायता", "सब्सिडी", "ब्याज",
        "રકમ", "સહાય", "સબસિડી", "વ્યાજ", "ટકા", "ટકાવારી",
    }),
    frozenset({
        "loan", "loans", "education", "educational", "interest", "subsidy", "scheme",
        "ऋण", "कर्ज", "शिक्षा", "शैक्षिक", "ब्याज", "योजना", "अनुदान",
        "લોન", "શિક્ષણ", "શૈક્ષણિક", "વ્યાજ", "યોજના", "સબસિડી",
    }),
)

SECTION_HEADING_NAMES = {
    "documents required", "required documents", "documents needed",
    "required papers", "papers required", "how to apply",
    "application process", "application procedure", "eligibility",
    "eligibility criteria", "application deadline", "important dates",
    "key dates", "important information", "scholarship amount", "award amount",
    "eligibility criteria continued", "other schemes", "related schemes",
    "additional schemes", "more schemes",
}
SECTION_HEADING_ALIASES = {
    "eligibility criteria continued": "eligibility criteria",
    "જરૂરી દસ્તાવેજ": "documents required",
    "જરૂરી દસ્તાવેજો": "documents required",
    "આવશ્યક દસ્તાવેજો": "documents required",
    "દસ્તાવેજો જરૂરી": "documents required",
    "પાત્રતા": "eligibility",
    "પાત્રતા માપદંડ": "eligibility criteria",
    "લાયકાત": "eligibility",
    "લાયકાતના માપદંડ": "eligibility criteria",
    "અરજી પ્રક્રિયા": "application process",
    "અરજી કરવાની પ્રક્રિયા": "application procedure",
    "અરજી કરવાની રીત": "application procedure",
    "અરજીની છેલ્લી તારીખ": "application deadline",
    "અરજી કરવાની છેલ્લી તારીખ": "application deadline",
    "અરજીની અંતિમ તારીખ": "application deadline",
    "સહાયની રકમ": "scholarship amount",
    "વ્યાજ સહાયની રકમ": "scholarship amount",
    "amount of subsidy": "scholarship amount",
    "subsidy amount": "scholarship amount",
    "आवश्यक दस्तावेज़": "documents required",
    "आवश्यक दस्तावेज": "documents required",
    "जरूरी कागजात": "documents required",
    "पात्रता": "eligibility",
    "पात्रता मापदंड": "eligibility criteria",
    "आवेदन प्रक्रिया": "application process",
    "आवेदन की प्रक्रिया": "application procedure",
    "आवेदन की अंतिम तिथि": "application deadline",
    "आवेदन की आखिरी तारीख": "application deadline",
    "सहायता राशि": "scholarship amount",
    "ब्याज सहायता राशि": "scholarship amount",
}
LIST_MARKER_PATTERN = re.compile(r"^\s*(?:(?:[-*•▪◦])\s*|\d+\s*[.)]\s*)")
GUJARATI_SCRIPT_PATTERN = re.compile(r"[\u0a80-\u0aff]")
HINDI_SCRIPT_PATTERN = re.compile(r"[\u0900-\u097f]")
GUJARATI_INTENT_TERMS = {
    "scholarship": ("શિષ્યવૃત્તિ", "વિદ્યાવૃત્તિ", "સ્કોલરશિપ", "સ્કોલરશીપ"),
    "application": ("અરજી", "અરજી ફોર્મ", "ફોર્મ"),
    "documents": ("દસ્તાવેજ", "કાગળ", "પ્રમાણપત્ર", "સર્ટિફિકેટ"),
    "required_documents": (
        "જરૂરી", "આવશ્યક", "જોઈએ", "જોઇએ", "પડશે", "આપવા", "જમા કરાવવા",
        "સબમિટ", "કયા", "કઈ", "કયો", "કયા કયા", "કેટલા", "કેટલી", "શું જોઈએ",
    ),
    "deadline": (
        "છેલ્લી તારીખ", "અંતિમ તારીખ", "સમયમર્યાદા", "છેલ્લી મુદત", "અંતિમ મુદત",
        "અરજી ક્યારે બંધ", "અરજી કરવાની છેલ્લી તારીખ", "અરજીની અંતિમ તારીખ",
    ),
    "when": ("ક્યારે", "કયા દિવસે", "કઈ તારીખે", "ક્યારે સુધી"),
    "amount": ("રકમ", "રાશિ", "ધનરાશિ", "પૈસા", "રૂપિયા", "રૂપિયો", "amount"),
    "amount_topic": (
        "સહાય", "આર્થિક સહાય", "સબસિડી", "વ્યાજ સહાય", "સ્ટાઇપેન્ડ", "વજીફો", "પુરસ્કાર",
    ),
    "quantity": ("કેટલી", "કેટલું", "કેટલા", "કેટલો"),
    "eligibility": (
        "પાત્રતા", "લાયકાત", "પાત્ર", "લાયક", "યોગ્યતા", "કોણ અરજી કરી શકે",
        "કોણ મેળવી શકે", "કયા વિદ્યાર્થીઓ", "કોને મળે",
    ),
    "apply_procedure": (
        "કેવી રીતે", "કઈ રીતે", "પ્રક્રિયા", "અરજી કરવાની રીત", "ક્યાં", "કયા કાર્યાલય",
        "કયા વિભાગ", "જમા કરાવવી", "જમા કરવું", "સબમિટ", "મોકલવી",
    ),
}
HINDI_INTENT_TERMS = {
    "scholarship": ("छात्रवृत्ति", "छात्रवृति", "स्कॉलरशिप", "स्कालरशिप", "वजीफा"),
    "application": ("आवेदन", "अप्लाई", "आवेदन पत्र", "फॉर्म"),
    "documents": (
        "दस्तावेज़", "दस्तावेज", "कागजात", "कागज़ात", "काग़ज़ात", "कागज", "काग़ज़",
        "प्रमाणपत्र",
    ),
    "required_documents": (
        "आवश्यक", "जरूरी", "ज़रूरी", "अनिवार्य", "चाहिए", "कौन कौन से", "कौन से",
        "कौनसे", "कौन सा", "कौनसा", "किस किस", "कौन कौन से जमा", "मांगे गए",
        "जमा करने होंगे", "जमा करना होगा",
    ),
    "deadline": (
        "अंतिम तिथि", "आखिरी तिथि", "अंतिम तारीख", "आखिरी तारीख", "समय सीमा",
        "समयसीमा", "आवेदन की आखिरी तारीख", "आवेदन की अंतिम तिथि", "कब तक आवेदन",
        "आवेदन कब बंद", "अंतिम दिन",
    ),
    "when": ("कब", "कब तक", "किस तारीख", "किस दिन"),
    "amount": ("राशि", "रकम", "धनराशि", "कितने पैसे", "रुपये", "रुपए", "रुपया", "amount"),
    "amount_topic": (
        "सहायता", "आर्थिक सहायता", "सब्सिडी", "स्टाइपेंड", "वजीफा", "छात्रवृत्ति", "स्कॉलरशिप",
        "stipend", "scholarship", "assistance",
    ),
    "quantity": ("कितना", "कितनी", "कितने"),
    "eligibility": (
        "पात्रता", "योग्यता", "पात्र", "योग्य", "कौन पात्र", "कौन योग्य",
        "कौन आवेदन कर सकता", "किन छात्रों", "किसे मिल सकती", "किसके लिए पात्र",
    ),
    "apply_procedure": (
        "कैसे", "किस तरह", "प्रक्रिया", "आवेदन करने की विधि", "कहाँ", "कहां",
        "किस कार्यालय", "जमा करना", "जमा करें", "भेजें", "आवेदन कहाँ", "आवेदन कहां",
    ),
}
CANONICAL_QUESTIONS_BY_INTENT = {
    "documents": "What documents are required for the scholarship?",
    "deadline": "What is the application deadline?",
    "amount": "What is the scholarship amount?",
    "eligibility": "What are the eligibility criteria?",
    "how_to_apply": "How do I apply for the scholarship?",
}
ENGLISH_DEADLINE_CUES = (
    "application deadline",
    "deadline for application",
    "deadline for applications",
    "deadline to apply",
    "deadline for applying",
    "application closing date",
    "applications close",
    "application closes",
    "last date",
    "last day",
    "final date",
    "final day",
    "closing date",
)
ENGLISH_ELIGIBILITY_CUES = (
    "eligibility",
    "eligible",
    "qualify",
    "qualified",
    "qualification",
    "qualifications",
    "who can apply",
    "who may apply",
)
ENGLISH_AMOUNT_QUANTITY_CUES = ("amount", "how much", "money")
ENGLISH_AMOUNT_TOPIC_CUES = (
    "scholarship", "award", "financial assistance", "assistance", "stipend", "grant", "money",
)
ENGLISH_DOCUMENT_TERMS = (
    "document", "documents", "paper", "papers", "paperwork", "certificate", "proof",
)
ENGLISH_DOCUMENT_REQUIREMENT_CUES = (
    "required", "require", "need", "needed", "must", "submit", "provide", "bring", "include",
)
ENGLISH_DOCUMENT_REQUEST_CUES = ("what", "which")
ENGLISH_DOCUMENT_CONTEXT_TERMS = ("scholarship", "application", "applications")
ENGLISH_DOCUMENT_ACTION_CUES = ("submit", "provide", "bring", "include", "send")
PERSONAL_ELIGIBILITY_CUES = (
    "am i eligible",
    "am i qualified",
    "do i qualify",
    "would i qualify",
    "do i meet",
)
ENGLISH_APPLICATION_CONTEXT_TERMS = ("apply", "application", "applications")
ENGLISH_SUBMISSION_TERMS = ("submit", "submission")
LOCALIZED_WORD_SUFFIXES = (
    "ઓના", "ઓમાં", "ઓથી", "ો", "ી", "ા", "ુ", "ે", "ૈ", "ૌ", "ઓ", "ની", "નો",
    "નું", "ના", "ને", "માં", "થી",
    "ों", "ें", "े", "ो", "ी", "ि", "ा", "याँ", "यां",
)
AMOUNT_VALUE_PATTERN = re.compile(
    r"(?:₹\s*\d[\d,]*|\b(?:INR|Rs\.?|rupees?)\s*\d[\d,]*|"
    r"\b(?:amount|stipend|award|grant|worth)\b[^.\n]{0,60}\b\d[\d,]*)",
    re.IGNORECASE,
)
AMOUNT_CONTEXT_PATTERN = re.compile(
    r"\b(?:scholarship|award|amount|stipend|grant|subsidy|financial\s+assistance|assistance)\b",
    re.IGNORECASE,
)
AMOUNT_SPECIFIC_CONTEXT_PATTERN = re.compile(
    r"\b(?:scholarship\s+amount|award(?:\s+amount)?|stipend|grant|subsidy|"
    r"financial\s+assistance|assistance)\b",
    re.IGNORECASE,
)
AMOUNT_FEE_CONTEXT_PATTERN = re.compile(r"\b(?:application\s+)?fees?\b", re.IGNORECASE)
SUBSIDY_ON_LOAN_LIMIT_PATTERN = re.compile(
    r"\bsubsid(?:y|ies)\b[^.\n]{0,140}?\b(?:education(?:al)?\s+)?loans?\b"
    r"[^.\n]{0,100}?\bup\s+to\s+(?P<currency>₹|INR|Rs\.?)\s*"
    r"(?P<value>\d[\d,]*)(?P<unit>\s*(?:lakhs?|crores?)?)",
    re.IGNORECASE,
)
SUBSIDY_LOAN_LIMIT_ANSWER_TEMPLATES = {
    "hi": "नोटिस में सब्सिडी की निश्चित राशि नहीं बताई गई है; इसमें कहा गया है कि {amount} तक के शिक्षा ऋण पर सब्सिडी उपलब्ध है।",
    "gu": "નોટિસમાં સબસિડીની ચોક્કસ રકમ જણાવેલી નથી; તેમાં લખ્યું છે કે {amount} સુધીની શિક્ષણ લોન પર સબસિડી ઉપલબ્ધ છે.",
}
DEADLINE_MONTH_NAMES = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?"
)
DEADLINE_DATE_PATTERN = re.compile(
    rf"\b(?:\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{DEADLINE_MONTH_NAMES})(?:,?\s+\d{{4}})?|"
    rf"(?:{DEADLINE_MONTH_NAMES})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?|"
    r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
    re.IGNORECASE,
)
DEADLINE_CONTEXT_PATTERN = re.compile(
    r"\b(?:application deadline|deadline|due date|last date|last day|final date|"
    r"closing date|applications? close|applications? end|apply by|submit by)\b",
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
    tokens = {
        token
        for token in _unicode_word_tokens(text)
        if token not in STOP_WORDS
    }
    return _expand_retrieval_aliases(tokens)


def _unicode_word_tokens(text: str) -> list[str]:
    """Split on separators while keeping combining marks with Indic letters."""
    normalized = unicodedata.normalize("NFC", text).casefold()
    tokens: list[str] = []
    current: list[str] = []

    def finish_token() -> None:
        if current:
            tokens.append("".join(current))
            current.clear()

    for character in normalized:
        category = unicodedata.category(character)
        if category[0] in {"L", "N"} or (category[0] == "M" and current):
            current.append(character)
        else:
            finish_token()
    finish_token()
    return tokens


def _expand_retrieval_aliases(tokens: set[str]) -> set[str]:
    """Expand matched words with a small, local multilingual retrieval glossary."""
    expanded = set(tokens)
    changed = True
    while changed:
        previous_size = len(expanded)
        for aliases in RETRIEVAL_TERM_ALIAS_GROUPS:
            if expanded.intersection(aliases):
                expanded.update(aliases)
        changed = len(expanded) != previous_size
    return expanded


def _normalize_intent_text(value: str) -> str:
    """Normalize Unicode spelling and punctuation for phrase matching only."""
    normalized = unicodedata.normalize("NFC", value).casefold()
    characters = [
        " " if unicodedata.category(character).startswith(("P", "Z")) else character
        for character in normalized
    ]
    return " ".join("".join(characters).split())


def _contains_any_intent_term(text: str, terms: tuple[str, ...]) -> bool:
    words = text.split()

    def word_forms(word: str) -> set[str]:
        forms = {word}
        frontier = [word]
        while frontier:
            current = frontier.pop()
            for suffix in LOCALIZED_WORD_SUFFIXES:
                if current.endswith(suffix) and len(current) > len(suffix) + 1:
                    stem = current[:-len(suffix)]
                    if stem not in forms:
                        forms.add(stem)
                        frontier.append(stem)
        return forms

    word_variants = [word_forms(word) for word in words]
    for term in terms:
        term_words = _normalize_intent_text(term).split()
        for start in range(len(words) - len(term_words) + 1):
            if all(
                term_word in word_variants[start + offset]
                for offset, term_word in enumerate(term_words)
            ):
                return True
    return False


def _map_localized_question(
    question: str,
    script_pattern: re.Pattern[str],
    terms: dict[str, tuple[str, ...]],
) -> tuple[str, str | None]:
    if not script_pattern.search(question):
        return question, None

    text = _normalize_intent_text(question)
    has_scholarship = _contains_any_intent_term(text, terms["scholarship"])
    has_application = _contains_any_intent_term(text, terms["application"])
    has_documents = _contains_any_intent_term(text, terms["documents"])
    has_required_cue = _contains_any_intent_term(text, terms["required_documents"])
    has_amount_topic = has_scholarship or _contains_any_intent_term(text, terms["amount_topic"])

    # Check the more specific document intent first, since list questions may
    # also contain terms used by application or amount questions.
    if has_documents and has_required_cue:
        intent = "documents"
    elif _contains_any_intent_term(text, terms["deadline"]) or (
        _contains_any_intent_term(text, terms["when"])
        and has_application
    ):
        intent = "deadline"
    elif _contains_any_intent_term(text, terms["amount"]) or (
        has_amount_topic and _contains_any_intent_term(text, terms["quantity"])
    ):
        intent = "amount"
    elif _contains_any_intent_term(text, terms["eligibility"]):
        intent = "eligibility"
    elif has_application and _contains_any_intent_term(text, terms["apply_procedure"]):
        intent = "how_to_apply"
    else:
        return question, None

    return CANONICAL_QUESTIONS_BY_INTENT[intent], intent


def map_gujarati_question(question: str) -> tuple[str, str | None]:
    """Map common Gujarati scholarship intents to English retrieval queries."""
    return _map_localized_question(question, GUJARATI_SCRIPT_PATTERN, GUJARATI_INTENT_TERMS)


def map_hindi_question(question: str) -> tuple[str, str | None]:
    """Map common Hindi scholarship intents to English retrieval queries."""
    return _map_localized_question(question, HINDI_SCRIPT_PATTERN, HINDI_INTENT_TERMS)


def map_english_question(question: str) -> tuple[str, str | None]:
    """Normalize common English scholarship questions for retrieval."""
    text = _normalize_intent_text(question)
    if _contains_any_intent_term(text, ENGLISH_ELIGIBILITY_CUES):
        return CANONICAL_QUESTIONS_BY_INTENT["eligibility"], "eligibility"

    has_application_context = _contains_any_intent_term(
        text,
        ENGLISH_APPLICATION_CONTEXT_TERMS,
    )
    has_deadline_context = _contains_any_intent_term(text, ENGLISH_DEADLINE_CUES) or (
        _contains_any_intent_term(text, ("deadline", "deadlines"))
        and has_application_context
    )
    has_submission_timing = (
        _contains_any_intent_term(text, ("when",))
        and _contains_any_intent_term(text, ENGLISH_SUBMISSION_TERMS)
        and has_application_context
    )
    if has_submission_timing or (has_deadline_context and has_application_context):
        return CANONICAL_QUESTIONS_BY_INTENT["deadline"], "deadline"

    has_document_term = _contains_any_intent_term(text, ENGLISH_DOCUMENT_TERMS)
    has_document_requirement_cue = _contains_any_intent_term(
        text,
        ENGLISH_DOCUMENT_REQUIREMENT_CUES,
    )
    asks_what_to_submit_or_provide = (
        _contains_any_intent_term(text, ENGLISH_DOCUMENT_REQUEST_CUES)
        and _contains_any_intent_term(text, ENGLISH_DOCUMENT_ACTION_CUES)
        and _contains_any_intent_term(text, ENGLISH_DOCUMENT_CONTEXT_TERMS)
    )
    if (has_document_term and has_document_requirement_cue) or asks_what_to_submit_or_provide:
        return CANONICAL_QUESTIONS_BY_INTENT["documents"], "documents"

    has_amount_quantity = _contains_any_intent_term(text, ENGLISH_AMOUNT_QUANTITY_CUES)
    has_amount_topic = _contains_any_intent_term(text, ENGLISH_AMOUNT_TOPIC_CUES)
    if has_amount_quantity and has_amount_topic:
        return CANONICAL_QUESTIONS_BY_INTENT["amount"], "amount"

    return question, None


def _asks_personal_eligibility_question(question: str) -> bool:
    """Detect a question asking Saar to decide whether the user personally qualifies."""
    return _contains_any_intent_term(
        _normalize_intent_text(question),
        PERSONAL_ELIGIBILITY_CUES,
    )


def map_question_for_retrieval(question: str) -> tuple[str, str | None]:
    """Normalize supported Hindi and Gujarati intents for English document retrieval."""
    for mapper in (map_gujarati_question, map_hindi_question):
        mapped_question, intent = mapper(question)
        if intent is not None:
            return mapped_question, intent
    return map_english_question(question)


def _asks_deadline_and_eligibility(question: str) -> bool:
    """Detect the supported compound question so the answer path can clarify."""
    text = _normalize_intent_text(question)
    language_terms = (HINDI_INTENT_TERMS, GUJARATI_INTENT_TERMS)
    deadline_terms = (
        "deadline", "application deadline", "last date", "due date", "closing date",
        *(term for terms in language_terms for term in terms["deadline"]),
    )
    when_terms = (
        "when", *(term for terms in language_terms for term in terms["when"]),
    )
    application_terms = (
        "application", "apply", "submit",
        *(term for terms in language_terms for term in terms["application"]),
    )
    eligibility_terms = (
        *ENGLISH_ELIGIBILITY_CUES,
        "criteria",
        *(term for terms in language_terms for term in terms["eligibility"]),
    )

    has_deadline = _contains_any_intent_term(text, deadline_terms) or (
        _contains_any_intent_term(text, when_terms)
        and _contains_any_intent_term(text, application_terms)
    )
    return has_deadline and _contains_any_intent_term(text, eligibility_terms)


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
    normalized = unicodedata.normalize("NFC", question).casefold()
    for token in _unicode_word_tokens(normalized):
        if token in STOP_WORDS:
            continue
        family = _expand_retrieval_aliases(set(TERM_FAMILIES.get(token, {token})))
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
    normalized = unicodedata.normalize("NFC", value).casefold()
    name = " ".join(_unicode_word_tokens(normalized))
    return SECTION_HEADING_ALIASES.get(name, name)


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


def _natural_deadline_answer(body: str, response_language: str) -> str | None:
    """Format a deadline from its section while retaining its adjacent caveat."""
    date_match = DEADLINE_DATE_PATTERN.search(body)
    if not date_match:
        return None

    templates = DEADLINE_ANSWER_TEMPLATES[response_language]
    answer = templates["answer"].format(date=date_match.group(0))
    following_text = body[date_match.end():]
    caveat_match = re.match(r"\s*(\([^()\r\n]{1,120}\))", following_text)
    if not caveat_match:
        return answer

    caveat = caveat_match.group(1)
    if "fictional" in caveat.casefold():
        return answer + templates["fictional_caveat"]
    return answer + templates["other_caveat"].format(caveat=caveat)


def _natural_amount_answer(body: str) -> str | None:
    """Return a natural amount statement while preserving its source wording."""
    normalized = re.sub(r"\s+", " ", body).strip()
    normalized = LIST_MARKER_PATTERN.sub("", normalized, count=1)
    if not AMOUNT_VALUE_PATTERN.search(normalized):
        return None

    sentences = [
        sentence.strip()
        for sentence in SENTENCE_SPLIT_PATTERN.split(normalized)
        if sentence.strip()
    ]
    if not sentences:
        return None

    amount_fragment = re.match(
        r"^(?:₹|INR\b|Rs\.?\b|rupees?\b|\d|up\s+to\s+(?:₹|INR\b|Rs\.?\b|\d))",
        sentences[0],
        flags=re.IGNORECASE,
    )
    if not amount_fragment:
        return normalized

    first_sentence = f"The scholarship amount is listed as {sentences[0].rstrip(' .')}."
    remaining = [
        f'The notice also states: “{sentence}”'
        for sentence in sentences[1:]
    ]
    return " ".join([first_sentence, *remaining])


def _localized_subsidy_loan_limit_answer(body: str, response_language: str) -> str | None:
    """Explain a loan coverage ceiling without presenting it as the subsidy amount."""
    template = SUBSIDY_LOAN_LIMIT_ANSWER_TEMPLATES.get(response_language)
    match = SUBSIDY_ON_LOAN_LIMIT_PATTERN.search(body)
    if not template or not match:
        return None

    currency = match.group("currency")
    value = match.group("value")
    source_unit = match.group("unit").strip().casefold()
    localized_currency = "રૂ." if response_language == "gu" and currency != "₹" else "₹"
    localized_unit = ""
    if source_unit.startswith("lakh"):
        localized_unit = "લાખ" if response_language == "gu" else "लाख"
    elif source_unit.startswith("crore"):
        localized_unit = "કરોડ" if response_language == "gu" else "करोड़"

    amount = f"{localized_currency} {value}"
    if localized_unit:
        amount = f"{amount} {localized_unit}"
    return template.format(amount=amount)


def _eligibility_criteria_sentences(body: str) -> list[str]:
    """Keep concrete eligibility conditions and discard link/footer noise."""
    selected: list[str] = []
    seen: set[str] = set()
    for sentence in SENTENCE_SPLIT_PATTERN.split(body):
        source_text = sentence.strip()
        text = LIST_MARKER_PATTERN.sub("", source_text, count=1).strip()
        if not text or ELIGIBILITY_BOILERPLATE_PATTERN.search(text):
            continue
        tokens = _tokens(text)
        if not tokens.intersection(ELIGIBILITY_CRITERIA_TERMS):
            continue
        key = unicodedata.normalize("NFC", text).casefold()
        if key not in seen:
            selected.append(source_text)
            seen.add(key)
    return selected


def _eligibility_section_response(
    passages: list[_RankedPassage],
    response_language: str,
) -> tuple[str, list[SourcePassage]] | None:
    """Return concise, source-worded criteria with filtered page-level evidence."""
    selected_sections: dict[tuple[int, str], list[str]] = {}
    seen_sentences: set[str] = set()

    for passage in sorted(passages, key=lambda item: item.order):
        if _heading_name(passage.heading) not in SECTION_NAMES_BY_INTENT["eligibility"]:
            continue
        criteria = _eligibility_criteria_sentences(passage.body)
        if not criteria:
            continue

        key = (passage.source.page_number, passage.heading)
        section_sentences = selected_sections.setdefault(key, [])
        for sentence in criteria:
            sentence_key = unicodedata.normalize("NFC", sentence).casefold()
            if sentence_key in seen_sentences:
                continue
            section_sentences.append(sentence)
            seen_sentences.add(sentence_key)

    if not selected_sections:
        return None

    sources = [
        SourcePassage(
            page_number=page_number,
            text=f"{heading}\n" + "\n".join(sentences),
        )
        for (page_number, heading), sentences in selected_sections.items()
    ]
    bullets = [
        f"• {LIST_MARKER_PATTERN.sub('', sentence, count=1).strip()}"
        for source in sources
        for sentence in source.text.splitlines()[1:]
    ]
    answer = f"{ELIGIBILITY_ANSWER_INTRODUCTIONS[response_language]}\n" + "\n".join(bullets)
    return answer, sources


def _deadline_evidence_text(body: str, heading: str) -> str | None:
    """Keep only the sentence containing the date and its deadline context."""
    has_deadline_heading = _heading_name(heading) in SECTION_NAMES_BY_INTENT["deadline"]
    for sentence in SENTENCE_SPLIT_PATTERN.split(body):
        text = sentence.strip()
        if not DEADLINE_DATE_PATTERN.search(text):
            continue
        if has_deadline_heading or DEADLINE_CONTEXT_PATTERN.search(text):
            return text
    return None


def _mapped_section_response(
    intent: str | None,
    passages: list[_RankedPassage],
    response_language: str,
) -> tuple[str, SourcePassage] | None:
    """Build a response from the section selected by a mapped question."""
    section_names = SECTION_NAMES_BY_INTENT.get(intent or "")
    if not section_names and intent != "deadline":
        return None

    matching = [
        passage
        for passage in passages
        if passage.body.strip()
        and (
            _heading_name(passage.heading) in (section_names or set())
            or (intent == "amount" and _passage_has_explicit_amount(passage))
            or (
                intent == "deadline"
                and DEADLINE_DATE_PATTERN.search(passage.body)
                and DEADLINE_CONTEXT_PATTERN.search(passage.body)
            )
        )
    ]
    if not matching:
        return None

    selected = matching[0]
    same_section_candidates = (
        passages
        if intent == "amount"
        and _heading_name(selected.heading) in SECTION_NAMES_BY_INTENT["amount"]
        else matching
    )
    same_section = sorted(
        (
            passage for passage in same_section_candidates
            if passage.source.page_number == selected.source.page_number
            and passage.heading == selected.heading
            and passage.body.strip()
        ),
        key=lambda passage: passage.order,
    )
    body = "\n".join(passage.body.strip() for passage in same_section).strip()
    if intent == "deadline":
        body = _deadline_evidence_text(body, selected.heading)
        if body is None:
            return None
        answer = _natural_deadline_answer(body, response_language)
        if answer is None:
            return None
    elif intent == "amount":
        if _heading_name(selected.heading) not in SECTION_NAMES_BY_INTENT["amount"]:
            body = _amount_evidence_text(body)
            if body is None:
                return None
        if response_language == "en":
            answer = _natural_amount_answer(body)
            if answer is None:
                return None
        else:
            answer = _localized_subsidy_loan_limit_answer(body, response_language)
            if answer is None:
                answer = f'{EXTRACTIVE_INTRODUCTIONS[response_language]} “{body}”'
    else:
        answer = f'{EXTRACTIVE_INTRODUCTIONS[response_language]} “{body}”'
    source = SourcePassage(
        page_number=selected.source.page_number,
        text=f"{selected.heading}\n{body}" if selected.heading else body,
    )
    return answer, source


def _passage_has_explicit_amount(passage: _RankedPassage) -> bool:
    has_amount_heading = (
        _heading_name(passage.heading) in SECTION_NAMES_BY_INTENT["amount"]
    )
    for sentence in SENTENCE_SPLIT_PATTERN.split(passage.body):
        if not AMOUNT_VALUE_PATTERN.search(sentence):
            continue
        fee_only_context = (
            AMOUNT_FEE_CONTEXT_PATTERN.search(sentence)
            and not AMOUNT_SPECIFIC_CONTEXT_PATTERN.search(sentence)
        )
        if fee_only_context:
            continue
        if has_amount_heading or AMOUNT_CONTEXT_PATTERN.search(sentence):
            return True
    return False


def _amount_evidence_text(body: str) -> str | None:
    """Isolate the sentence that explicitly ties a value to an award or assistance."""
    normalized = re.sub(r"\s+", " ", body).strip()
    sentences = [
        sentence.strip()
        for sentence in SENTENCE_SPLIT_PATTERN.split(normalized)
        if sentence.strip()
    ]
    for index, sentence in enumerate(sentences):
        if not AMOUNT_VALUE_PATTERN.search(sentence):
            continue
        if not AMOUNT_CONTEXT_PATTERN.search(sentence):
            continue
        if (
            AMOUNT_FEE_CONTEXT_PATTERN.search(sentence)
            and not AMOUNT_SPECIFIC_CONTEXT_PATTERN.search(sentence)
        ):
            continue
        selected = [sentence]
        if index + 1 < len(sentences) and re.search(
            r"\b(?:subject\s+to|provided\s+that|only\s+if|unless|eligible|"
            r"eligibility|maintain|review|renewal|may\s+be\s+adjusted)\b",
            sentences[index + 1],
            flags=re.IGNORECASE,
        ):
            selected.append(sentences[index + 1])
        return " ".join(selected)
    return None


def _has_explicit_amount(passages: list[_RankedPassage]) -> bool:
    return any(_passage_has_explicit_amount(passage) for passage in passages)


def _has_explicit_deadline_evidence(passages: list[_RankedPassage]) -> bool:
    """Require both deadline context and a date value before answering a date query."""
    return any(
        DEADLINE_DATE_PATTERN.search(passage.body)
        and (
            _heading_name(passage.heading) in SECTION_NAMES_BY_INTENT["deadline"]
            or DEADLINE_CONTEXT_PATTERN.search(passage.body)
        )
        for passage in passages
    )


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
            message=(
                "No readable text was provided for this notice. If it is a scan, "
                "make sure local OCR completed successfully before asking a question."
            ),
        )

    if _asks_deadline_and_eligibility(request.question):
        return DocumentAnswerResponse(
            answer=MULTIPLE_INTENTS_CLARIFICATION[request.response_language],
            sources=[],
            status="not_found",
            mode="extractive",
        )

    retrieval_question, mapped_intent = map_question_for_retrieval(request.question)
    ranked_passages = _rank_relevant_passages(retrieval_question, pages)
    if not ranked_passages:
        return _not_found_response(response_language=request.response_language)
    if mapped_intent == "deadline" and not _has_explicit_deadline_evidence(ranked_passages):
        return _not_found_response(response_language=request.response_language)
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

    if mapped_intent == "eligibility":
        eligibility_answer = _eligibility_section_response(
            ranked_passages,
            request.response_language,
        )
        if not eligibility_answer:
            return _not_found_response(response_language=request.response_language)
        answer, sources = eligibility_answer
        return DocumentAnswerResponse(
            answer=answer,
            sources=sources,
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

    # Evon is intentionally not called for document QA until its generated
    # answers can be validated against source evidence. Deterministic answers
    # below preserve citations and cannot add unsupported document facts.
    if mapped_intent == "amount" and not mapped_section_answer:
        return _not_found_response(response_language=request.response_language)
    if mapped_section_answer:
        answer, source = mapped_section_answer
        return DocumentAnswerResponse(
            answer=answer,
            sources=[source],
            status="answered",
            mode="extractive",
        )

    evidence = _extractive_support(retrieval_question, ranked_passages)
    if not evidence:
        return _not_found_response(response_language=request.response_language)

    first_passage = evidence[0].text
    extractive_intro = EXTRACTIVE_INTRODUCTIONS[request.response_language]
    return DocumentAnswerResponse(
        answer=f'{extractive_intro} “{first_passage}”',
        sources=evidence,
        status="answered",
        mode="extractive",
    )
