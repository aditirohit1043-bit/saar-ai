import { useEffect, useRef, useState } from "react";

function BrandMark() {
  return (
    <svg
      aria-hidden="true"
      className="brand-mark"
      viewBox="0 0 44 44"
      fill="none"
    >
      <rect width="44" height="44" rx="16" fill="currentColor" />
      <path
        d="M13 27.5c5.5-.2 10.7-4.2 13.8-11.2 2.6 7 .7 14.2-5.3 16.1-3.5 1.1-6.9-.8-8.5-4.9Z"
        fill="#E6C0B8"
      />
      <path d="M14 31c3-4.1 6.6-7.2 11.3-9.4" stroke="#FFFDF8" strokeWidth="1.8" strokeLinecap="round" />
      <circle cx="29.6" cy="13.8" r="2.1" fill="#E6C0B8" />
    </svg>
  );
}

function DocumentIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 48 48" fill="none">
      <path d="M14 6.75h13.1L36 15.7v22.55A3.75 3.75 0 0 1 32.25 42h-18.5A3.75 3.75 0 0 1 10 38.25v-27.5A4 4 0 0 1 14 6.75Z" fill="#FFFDF8" stroke="currentColor" strokeWidth="1.8" />
      <path d="M27 7v8.5h8.5M17 24h12M17 29.5h12M17 35h8" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M27 7v6.5a2 2 0 0 0 2 2h6.5" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
    </svg>
  );
}

function SparkleIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 32 32" fill="none">
      <path d="M16 3.5c1.5 6.7 4.1 9.3 10.8 10.8C20.1 15.8 17.5 18.4 16 25.1 14.5 18.4 11.9 15.8 5.2 14.3 11.9 12.8 14.5 10.2 16 3.5Z" fill="currentColor" />
      <path d="M26 22c.5 2.1 1.3 2.9 3.5 3.4-2.2.5-3 1.3-3.5 3.5-.5-2.2-1.3-3-3.5-3.5 2.2-.5 3-1.3 3.5-3.4Z" fill="currentColor" />
    </svg>
  );
}

function App() {
  const [language, setLanguage] = useState("en");
  const [apiStatus, setApiStatus] = useState("checking");
  const [uploadedDocument, setUploadedDocument] = useState(null);
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [isAsking, setIsAsking] = useState(false);
  const fileInputRef = useRef(null);
  const conversationEndRef = useRef(null);

  useEffect(() => {
    let isCurrent = true;

    fetch("/api/health")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Health check failed");
        }
        return response.json();
      })
      .then((data) => {
        if (isCurrent) {
          setApiStatus(data.status === "ok" ? "connected" : "offline");
        }
      })
      .catch(() => {
        if (isCurrent) {
          setApiStatus("offline");
        }
      });

    return () => {
      isCurrent = false;
    };
  }, []);

  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({ behavior: "auto", block: "end" });
  }, [messages]);

  const statusText = {
    checking: "Checking connection",
    connected: "Saar is ready",
    offline: "API is offline",
  }[apiStatus];

  const canAskAboutDocument = Boolean(uploadedDocument?.extracted_text?.trim()) && !isUploading && !isAsking;

  async function handleDocumentSelected(event) {
    const selectedFile = event.currentTarget.files?.[0];
    event.currentTarget.value = "";

    if (!selectedFile) {
      return;
    }

    setUploadedDocument(null);
    setIsPreviewOpen(false);
    setUploadError("");
    setMessages([]);
    setQuestion("");
    setIsUploading(true);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("/api/documents/upload", {
        method: "POST",
        body: formData,
      });
      const result = await response.json().catch(() => null);

      if (!response.ok) {
        const detail = result?.detail;
        throw new Error(
          typeof detail === "string"
            ? detail
            : "We couldn't read that PDF. Please choose another file and try again.",
        );
      }

      setUploadedDocument(result);
    } catch (error) {
      if (error instanceof TypeError) {
        setUploadError("Saar couldn't reach the document service. Please check the connection and try again.");
      } else {
        setUploadError(
          error instanceof Error && error.message
            ? error.message
            : "We couldn't read that PDF. Please try again.",
        );
      }
    } finally {
      setIsUploading(false);
    }
  }

  async function handleQuestionSubmit(event) {
    event.preventDefault();
    const questionText = question.trim();

    if (!questionText || !uploadedDocument?.pages?.length || isUploading || isAsking) {
      return;
    }

    const messageId = Date.now();
    const answerId = `answer-${messageId}`;
    setQuestion("");
    setIsAsking(true);
    setMessages((currentMessages) => [
      ...currentMessages,
      { id: `question-${messageId}`, role: "user", question: questionText },
      { id: answerId, role: "assistant", pending: true },
    ]);

    try {
      const response = await fetch("/api/documents/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: questionText,
          pages: uploadedDocument.pages,
          response_language: language,
        }),
      });
      const result = await response.json().catch(() => null);

      if (!response.ok) {
        const detail = result?.detail;
        throw new Error(
          typeof detail === "string"
            ? detail
            : "Saar couldn't answer that just now. Please try again.",
        );
      }

      if (typeof result?.answer !== "string") {
        throw new Error("Saar couldn't read the answer. Please try again.");
      }

      setMessages((currentMessages) =>
        currentMessages.map((message) =>
          message.id === answerId
            ? {
                ...message,
                pending: false,
                answer: result.answer,
                sources: Array.isArray(result.sources) ? result.sources : [],
                mode: result.mode,
                notice: result.message,
              }
            : message,
        ),
      );
    } catch (error) {
      const friendlyError = error instanceof TypeError
        ? "Saar couldn't reach the question service. Please check the connection and try again."
        : error instanceof Error && error.message
          ? error.message
          : "Saar couldn't answer that just now. Please try again.";

      setMessages((currentMessages) =>
        currentMessages.map((message) =>
          message.id === answerId
            ? { ...message, pending: false, answer: friendlyError, error: true }
            : message,
        ),
      );
    } finally {
      setIsAsking(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#home" aria-label="Saar home">
          <BrandMark />
          <span className="brand-word">saar<span className="brand-period">.</span></span>
        </a>

        <div className={`connection-status connection-${apiStatus}`} role="status" aria-live="polite">
          <span className="status-dot" aria-hidden="true" />
          <span>{statusText}</span>
        </div>
      </header>

      <main id="home" className="page-content">
        <section className="intro" aria-labelledby="page-title">
          <div className="eyebrow"><span /> A calmer way to understand notices</div>
          <h1 id="page-title">Made a little clearer.</h1>
          <p>Bring a notice. We'll make sense of it together, one step at a time.</p>
        </section>

        <div className="workspace-grid">
          <aside className="side-panel" aria-label="Document and language">
            <section className="panel-card document-card" aria-labelledby="document-title">
              <div className="section-heading">
                <span className="step-number">01</span>
                <div>
                  <h2 id="document-title">Your document</h2>
                  <p>Start with an official notice</p>
                </div>
              </div>

              <div className="upload-placeholder" aria-busy={isUploading}>
                <input
                  ref={fileInputRef}
                  className="file-input"
                  type="file"
                  accept=".pdf,application/pdf"
                  aria-label="Choose a PDF notice"
                  onChange={handleDocumentSelected}
                />

                {isUploading ? (
                  <div className="upload-loading" role="status" aria-live="polite">
                    <span className="loading-mark" aria-hidden="true" />
                    <span className="upload-title">Reading your notice…</span>
                    <span className="upload-copy">Saar is looking for text in your PDF.</span>
                  </div>
                ) : (
                  <button
                    className="upload-action"
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    aria-describedby="upload-guidance"
                  >
                    <span className="document-illustration"><DocumentIcon /></span>
                    <span className="upload-title">{uploadedDocument ? "Add another notice" : "Bring your notice here"}</span>
                    <span className="upload-copy">
                      Choose an official notice in PDF format to see its text here.
                    </span>
                    <span className="upload-button-label">{uploadedDocument ? "Choose another PDF" : "Choose a PDF"}</span>
                  </button>
                )}

                <span className="upload-format" id="upload-guidance">PDF files, up to 10 MB</span>
              </div>

              {uploadError && <p className="upload-error" role="alert">{uploadError}</p>}

              {uploadedDocument && (
                <div className="upload-result" id="document-result">
                  <div className="uploaded-summary" role="status" aria-live="polite">
                    <span className="uploaded-file-icon" aria-hidden="true"><DocumentIcon /></span>
                    <div className="uploaded-file-details">
                      <span className="uploaded-label">YOUR NOTICE</span>
                      <p className="uploaded-filename" title={uploadedDocument.filename}>
                        {uploadedDocument.filename}
                      </p>
                      <span className="uploaded-pages">
                        {uploadedDocument.page_count} {uploadedDocument.page_count === 1 ? "page" : "pages"}
                      </span>
                    </div>
                  </div>

                  {uploadedDocument.message && (
                    <p className="extraction-message" role="status">{uploadedDocument.message}</p>
                  )}

                  {uploadedDocument.extracted_text && (
                    <div className="preview-disclosure">
                      <button
                        className="preview-toggle"
                        type="button"
                        aria-expanded={isPreviewOpen}
                        aria-controls="preview-content"
                        onClick={() => setIsPreviewOpen((isOpen) => !isOpen)}
                      >
                        <span>{isPreviewOpen ? "Hide extracted text" : "View extracted text"}</span>
                        <svg
                          className={`preview-toggle-chevron${isPreviewOpen ? " is-open" : ""}`}
                          aria-hidden="true"
                          viewBox="0 0 20 20"
                          fill="none"
                        >
                          <path d="m5 7.5 5 5 5-5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                      </button>

                      <section
                        className="document-preview"
                        id="preview-content"
                        aria-labelledby="preview-title"
                        hidden={!isPreviewOpen}
                      >
                        <div className="preview-heading">
                          <h3 id="preview-title">Extracted text</h3>
                          <span>Scroll to read</span>
                        </div>
                        <pre
                          className="preview-text"
                          tabIndex="0"
                          aria-label="Scrollable extracted PDF text"
                        >
                          {uploadedDocument.extracted_text}
                        </pre>
                      </section>
                    </div>
                  )}
                </div>
              )}

              <div className="next-step-note">
                <span className="note-sparkle"><SparkleIcon /></span>
                <p>Your PDF is processed for this preview and is not permanently stored.</p>
              </div>
            </section>

            <section className="panel-card language-card" aria-labelledby="language-title">
              <div className="section-heading compact-heading">
                <span className="step-number">02</span>
                <div>
                  <h2 id="language-title">Your language</h2>
                  <p>Choose what feels comfortable</p>
                </div>
              </div>
              <label className="select-label" htmlFor="language-select">Conversation language</label>
              <div className="select-wrap">
                <select
                  id="language-select"
                  value={language}
                  onChange={(event) => setLanguage(event.target.value)}
                >
                  <option value="gu">ગુજરાતી · Gujarati</option>
                  <option value="hi">हिन्दी · Hindi</option>
                  <option value="en">English</option>
                </select>
                <span className="select-chevron" aria-hidden="true">⌄</span>
              </div>
            </section>
          </aside>

          <section className="chat-card" aria-labelledby="chat-title">
            <div className="chat-header">
              <div>
                <div className="eyebrow chat-eyebrow">YOUR SAAR SPACE</div>
                <h2 id="chat-title">A conversation that starts with you.</h2>
              </div>
              <div className="chat-avatar" aria-hidden="true"><SparkleIcon /></div>
            </div>

            <div
              className="chat-body"
              role="log"
              aria-label="Conversation"
              aria-live="polite"
              aria-relevant="additions text"
            >
              <div className="day-divider"><span>YOUR FIRST VISIT</span></div>
              <div className="message-row">
                <div className="message-avatar" aria-hidden="true"><BrandMark /></div>
                <div className="message-content">
                  <span className="message-name">Saar</span>
                  <div className="assistant-message">
                    <p>Hi there. Official notices can feel like a lot. I’m here to help make the details easier to follow.</p>
                    <p>When you’re ready, add a notice and we can take it from there.</p>
                  </div>
                  <span className="message-time">A friendly place to begin</span>
                </div>
              </div>

              {messages.map((message) => (
                message.role === "user" ? (
                  <div className="message-row message-user" key={message.id}>
                    <div className="message-content">
                      <span className="message-name">You</span>
                      <div className="user-message"><p>{message.question}</p></div>
                    </div>
                  </div>
                ) : (
                  <div className="message-row message-assistant" key={message.id}>
                    <div className="message-avatar" aria-hidden="true"><BrandMark /></div>
                    <div className="message-content">
                      <span className="message-name">Saar</span>
                      {message.pending ? (
                        <div className="assistant-message answer-pending" role="status" aria-live="polite">
                          <p>Saar is looking through the relevant parts of your notice…</p>
                        </div>
                      ) : (
                        <div
                          className={`assistant-message${message.error ? " answer-error" : " answer-response"}`}
                          role={message.error ? "alert" : undefined}
                        >
                          <p>{message.answer}</p>
                        </div>
                      )}

                      {message.notice && !message.pending && (
                        <p className="answer-note">{message.notice}</p>
                      )}

                      {message.sources?.length > 0 && !message.pending && (
                        <div className="answer-sources" aria-label="Supporting passages from the notice">
                          <span className="source-heading">From your notice</span>
                          {message.sources.map((source, index) => (
                            <article className="source-passage" key={`${message.id}-source-${index}`}>
                              <span className="source-page">Page {source.page_number}</span>
                              <p>{source.text}</p>
                            </article>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )
              ))}

              {messages.length === 0 && (
                <div className="conversation-hint">
                  <span className="hint-line" />
                  <p>Your questions and answers will appear here.</p>
                  <span className="hint-line" />
                </div>
              )}
              <div className="conversation-end" ref={conversationEndRef} />
            </div>

            <div className="composer-wrap">
              <form className="composer" onSubmit={handleQuestionSubmit} aria-label="Ask about your notice">
                <input
                  className="composer-input"
                  type="text"
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder={
                    !uploadedDocument?.extracted_text
                      ? "Upload a readable PDF to begin"
                      : isAsking
                        ? "Saar is finding the right passage…"
                        : "Ask a question about your notice…"
                  }
                  aria-label="Your question about the notice"
                  disabled={!canAskAboutDocument}
                />
                <button
                  className="send-button"
                  type="submit"
                  aria-label="Send question"
                  disabled={!canAskAboutDocument || !question.trim()}
                >
                  <svg aria-hidden="true" viewBox="0 0 24 24" fill="none">
                    <path d="M5 12h14M13 6l6 6-6 6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </button>
              </form>
              <p className="composer-caption">
                {uploadedDocument?.extracted_text
                  ? "Answers are grounded in the text of your notice."
                  : "Upload a PDF with selectable text to ask Saar a question."}
              </p>
            </div>
          </section>
        </div>

        <footer className="page-footer">
          <span>Made for the moments when paperwork feels like too much.</span>
          <span className="footer-languages">ગુજરાતી <i /> हिन्दी <i /> English</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
