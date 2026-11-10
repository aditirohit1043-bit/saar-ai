# SAAR — Made a Little Clearer.

**A multilingual document assistant for understanding official notices.**

### The problem

Official notices about scholarships, education loans, government schemes, and student benefits can be difficult to understand. Important eligibility criteria, financial limits, and application details are often buried in lengthy documents. Language barriers make accessing this information even harder.

### Our solution

Saar helps people understand official notices through a conversational interface. Users upload a PDF and ask questions in a language they feel comfortable with.

Saar extracts document content and retrieves relevant information, making lengthy notices easier to navigate.

### Key features

- **PDF processing:** Upload official notices and extract text page by page.
- **Local OCR:** Recognize text in scanned PDF pages using Tesseract.
- **Multilingual interaction:** English, Hindi, and Gujarati question-and-answer workflows.
- **Grounded Q&A:** Retrieve answers from the uploaded document.
- **Source references:** Display supporting passages and page numbers.
- **Responsible answers:** Abstain when the document does not provide sufficient evidence.
- **Privacy-conscious processing:** Documents are processed for the current session and are not permanently stored by the application workflow.

### Technology

React, Vite, JavaScript, Python, FastAPI, PyMuPDF, Tesseract OCR, and deterministic document retrieval.

### What makes Saar different?

Saar distinguishes between what a notice actually states and what it does not.

For example, when a notice mentions education loans up to ₹10 lakh, Saar explains that this is the loan limit rather than claiming that ₹10 lakh is the actual subsidy payout.

### Vision

Make official information more accessible, understandable, and approachable—one notice at a time.

**Saar — Made for the moments when paperwork feels like too much.**