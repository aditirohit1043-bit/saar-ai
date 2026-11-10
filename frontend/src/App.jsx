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

function MicrophoneIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="none">
      <rect x="9" y="3" width="6" height="12" rx="3" stroke="currentColor" strokeWidth="1.7" />
      <path d="M5.5 11.5a6.5 6.5 0 0 0 13 0M12 18v3m-3.5 0h7" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="none">
      <rect x="6" y="6" width="12" height="12" rx="3" fill="currentColor" />
    </svg>
  );
}

function SpeakerIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="none">
      <path d="M4 10v4h4l5 4V6l-5 4H4Z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
      <path d="M16 9a4 4 0 0 1 0 6m2.5-8.5a7.5 7.5 0 0 1 0 11" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

const MAX_RECORDING_SECONDS = 25;
const RECORDING_MIME_TYPES = [
  "audio/ogg;codecs=opus",
  "audio/mp4;codecs=mp4a.40.2",
  "audio/webm;codecs=opus",
  "audio/webm",
];

function supportedRecordingType() {
  if (typeof MediaRecorder === "undefined" || typeof MediaRecorder.isTypeSupported !== "function") {
    return "";
  }

  return RECORDING_MIME_TYPES.find((mimeType) => MediaRecorder.isTypeSupported(mimeType)) || "";
}

function formatRecordingTime(seconds) {
  const minutes = Math.floor(seconds / 60).toString().padStart(2, "0");
  const remainingSeconds = (seconds % 60).toString().padStart(2, "0");
  return `${minutes}:${remainingSeconds}`;
}

function encodeMonoWav(audioBuffer) {
  const targetSampleRate = 16_000;
  const frameCount = Math.ceil(audioBuffer.duration * targetSampleRate);
  const dataLength = frameCount * 2;
  const wav = new ArrayBuffer(44 + dataLength);
  const view = new DataView(wav);
  const writeLabel = (offset, label) => {
    for (let index = 0; index < label.length; index += 1) {
      view.setUint8(offset + index, label.charCodeAt(index));
    }
  };

  writeLabel(0, "RIFF");
  view.setUint32(4, 36 + dataLength, true);
  writeLabel(8, "WAVE");
  writeLabel(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, targetSampleRate, true);
  view.setUint32(28, targetSampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeLabel(36, "data");
  view.setUint32(40, dataLength, true);

  const channels = Array.from({ length: audioBuffer.numberOfChannels }, (_, index) => audioBuffer.getChannelData(index));
  const sampleRatio = audioBuffer.sampleRate / targetSampleRate;

  for (let frame = 0; frame < frameCount; frame += 1) {
    const sourcePosition = frame * sampleRatio;
    const lowerIndex = Math.min(Math.floor(sourcePosition), audioBuffer.length - 1);
    const upperIndex = Math.min(lowerIndex + 1, audioBuffer.length - 1);
    const fraction = sourcePosition - lowerIndex;
    const mixedSample = channels.reduce((sum, channel) => {
      const interpolated = channel[lowerIndex] + (channel[upperIndex] - channel[lowerIndex]) * fraction;
      return sum + interpolated;
    }, 0) / channels.length;
    const safeSample = Math.max(-1, Math.min(1, mixedSample));
    view.setInt16(44 + frame * 2, safeSample < 0 ? safeSample * 0x8000 : safeSample * 0x7fff, true);
  }

  return new Blob([wav], { type: "audio/wav" });
}

async function convertWebmToWav(recording) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    throw new Error("This browser can't convert its recording format to a supported audio file.");
  }

  const audioContext = new AudioContextClass();
  try {
    const decoded = await audioContext.decodeAudioData(await recording.arrayBuffer());
    return encodeMonoWav(decoded);
  } finally {
    await audioContext.close();
  }
}

async function prepareAudioUpload(recording) {
  const mimeType = (recording.type || "").split(";")[0].toLowerCase();
  const extensions = {
    "audio/aac": "aac",
    "audio/flac": "flac",
    "audio/mp4": "m4a",
    "audio/mpeg": "mp3",
    "audio/ogg": "ogg",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
  };

  if (extensions[mimeType]) {
    return { blob: recording, filename: `saar-question.${extensions[mimeType]}` };
  }

  if (mimeType.includes("webm")) {
    return { blob: await convertWebmToWav(recording), filename: "saar-question.wav" };
  }

  throw new Error("This browser produced an audio format Saar can't send yet. Try another browser or type your question.");
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
  const [voiceAvailability, setVoiceAvailability] = useState({ supported: null, reason: "" });
  const [voiceStatus, setVoiceStatus] = useState("idle");
  const [voiceSeconds, setVoiceSeconds] = useState(0);
  const [voiceError, setVoiceError] = useState("");
  const [transcriptReady, setTranscriptReady] = useState(false);
  const [speechPlayback, setSpeechPlayback] = useState({
    messageId: null,
    requestId: 0,
    status: "idle",
    url: "",
    error: "",
  });
  const fileInputRef = useRef(null);
  const conversationEndRef = useRef(null);
  const questionInputRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const recordingChunksRef = useRef([]);
  const recordingStartedAtRef = useRef(0);
  const recordingIntervalRef = useRef(null);
  const recordingTimeoutRef = useRef(null);
  const recordingErrorRef = useRef(false);
  const speechAudioRef = useRef(null);
  const speechAudioUrlRef = useRef("");
  const speechRequestIdRef = useRef(0);

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

  useEffect(() => {
    if (!window.isSecureContext) {
      setVoiceAvailability({ supported: false, reason: "Voice input needs a secure connection, such as HTTPS or localhost." });
    } else if (!navigator.mediaDevices?.getUserMedia) {
      setVoiceAvailability({ supported: false, reason: "Microphone access isn't available in this browser." });
    } else if (typeof window.MediaRecorder !== "function") {
      setVoiceAvailability({ supported: false, reason: "Audio recording isn't supported in this browser." });
    } else {
      setVoiceAvailability({ supported: true, reason: "" });
    }

    return () => {
      window.clearInterval(recordingIntervalRef.current);
      window.clearTimeout(recordingTimeoutRef.current);
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.ondataavailable = null;
        mediaRecorderRef.current.onstop = null;
        mediaRecorderRef.current.onerror = null;
        mediaRecorderRef.current.stop();
      }
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
      speechAudioRef.current?.pause();
      if (speechAudioUrlRef.current) {
        URL.revokeObjectURL(speechAudioUrlRef.current);
      }
    };
  }, []);

  const statusText = {
    checking: "Checking connection",
    connected: "Saar is ready",
    offline: "API is offline",
  }[apiStatus];

  const hasReadableDocument = Boolean(uploadedDocument?.extracted_text?.trim());
  const canAskAboutDocument = hasReadableDocument && !isUploading && !isAsking && voiceStatus === "idle";
  const canStartRecording = hasReadableDocument && !isUploading && !isAsking && voiceStatus === "idle" && voiceAvailability.supported === true;

  function clearRecordingTimers() {
    window.clearInterval(recordingIntervalRef.current);
    window.clearTimeout(recordingTimeoutRef.current);
    recordingIntervalRef.current = null;
    recordingTimeoutRef.current = null;
  }

  function releaseMicrophone() {
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    mediaStreamRef.current = null;
  }

  function stopVoiceRecording() {
    clearRecordingTimers();
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
    } else {
      releaseMicrophone();
      setVoiceStatus("idle");
    }
  }

  async function transcribeRecording(recording) {
    try {
      const preparedAudio = await prepareAudioUpload(recording);
      const formData = new FormData();
      formData.append("audio_file", preparedAudio.blob, preparedAudio.filename);

      const response = await fetch("/api/speech/transcribe", {
        method: "POST",
        body: formData,
      });
      const result = await response.json().catch(() => null);

      if (!response.ok) {
        if (response.status === 503) {
          throw new Error("Voice transcription isn't available right now. You can type your question instead.");
        }
        if (response.status === 504) {
          throw new Error("That took too long to transcribe. Try a shorter recording or type your question.");
        }
        throw new Error(
          typeof result?.detail === "string"
            ? result.detail
            : "Saar couldn't transcribe that recording. Please try again or type your question.",
        );
      }

      const transcript = typeof result?.transcript === "string" ? result.transcript.trim() : "";
      if (!transcript || result?.status === "no_speech") {
        throw new Error("I couldn't hear clear speech in that recording. Try again or type your question.");
      }

      setQuestion(transcript);
      setTranscriptReady(true);
      setVoiceError("");
      window.requestAnimationFrame(() => questionInputRef.current?.focus());
    } catch (error) {
      const friendlyError = error instanceof TypeError
        ? "Saar couldn't reach voice transcription. Please check your connection or type your question."
        : error instanceof Error && error.message
          ? error.message
          : "Saar couldn't transcribe that recording. Please try again or type your question.";
      setVoiceError(friendlyError);
      setTranscriptReady(false);
    } finally {
      setVoiceStatus("idle");
      setVoiceSeconds(0);
    }
  }

  async function startVoiceRecording() {
    if (!canStartRecording) {
      return;
    }

    setVoiceError("");
    setTranscriptReady(false);
    setVoiceStatus("requesting");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      let recorder;
      const mimeType = supportedRecordingType();
      try {
        recorder = mimeType
          ? new MediaRecorder(stream, { mimeType, audioBitsPerSecond: 64_000 })
          : new MediaRecorder(stream);
      } catch {
        recorder = new MediaRecorder(stream);
      }

      mediaRecorderRef.current = recorder;
      recordingChunksRef.current = [];
      recordingErrorRef.current = false;
      recorder.ondataavailable = (event) => {
        if (event.data?.size) {
          recordingChunksRef.current.push(event.data);
        }
      };
      recorder.onerror = () => {
        recordingErrorRef.current = true;
        setVoiceError("The recording stopped unexpectedly. Please try again or type your question.");
        if (recorder.state !== "inactive") {
          recorder.stop();
        }
      };
      recorder.onstop = () => {
        clearRecordingTimers();
        mediaRecorderRef.current = null;
        releaseMicrophone();

        if (recordingErrorRef.current) {
          recordingErrorRef.current = false;
          setVoiceStatus("idle");
          return;
        }

        const chunks = recordingChunksRef.current;
        recordingChunksRef.current = [];
        const recordedType = recorder.mimeType || chunks[0]?.type || "";
        const recording = new Blob(chunks, { type: recordedType });
        if (!recording.size) {
          setVoiceStatus("idle");
          setVoiceError("No audio was captured. Please try recording again.");
          return;
        }

        setVoiceStatus("transcribing");
        void transcribeRecording(recording);
      };

      recorder.start();
      recordingStartedAtRef.current = Date.now();
      setVoiceSeconds(0);
      setVoiceStatus("recording");
      recordingIntervalRef.current = window.setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartedAtRef.current) / 1000);
        setVoiceSeconds(Math.min(elapsed, MAX_RECORDING_SECONDS));
      }, 500);
      recordingTimeoutRef.current = window.setTimeout(stopVoiceRecording, MAX_RECORDING_SECONDS * 1000);
    } catch (error) {
      releaseMicrophone();
      mediaRecorderRef.current = null;
      setVoiceStatus("idle");
      const errorName = error instanceof DOMException ? error.name : "";
      if (errorName === "NotAllowedError" || errorName === "SecurityError") {
        setVoiceError("Microphone access was blocked. Allow microphone permission in your browser, or type your question.");
      } else if (errorName === "NotFoundError") {
        setVoiceError("No microphone was found. Connect one or type your question instead.");
      } else {
        setVoiceError("Saar couldn't start the microphone. Please try again or type your question.");
      }
    }
  }

  function handleVoiceButton() {
    if (voiceStatus === "recording") {
      stopVoiceRecording();
    } else {
      void startVoiceRecording();
    }
  }

  function clearSpeechAudio() {
    speechRequestIdRef.current += 1;
    if (speechAudioRef.current) {
      speechAudioRef.current.pause();
      speechAudioRef.current.currentTime = 0;
      speechAudioRef.current = null;
    }
    if (speechAudioUrlRef.current) {
      URL.revokeObjectURL(speechAudioUrlRef.current);
      speechAudioUrlRef.current = "";
    }
    setSpeechPlayback({
      messageId: null,
      requestId: speechRequestIdRef.current,
      status: "idle",
      url: "",
      error: "",
    });
  }

  async function handleReadAloud(message) {
    if (speechPlayback.status === "loading") {
      return;
    }

    if (speechPlayback.messageId === message.id && speechPlayback.status === "playing") {
      speechAudioRef.current?.pause();
      return;
    }

    if (speechPlayback.messageId === message.id && speechPlayback.status === "ready" && speechAudioRef.current) {
      if (speechAudioRef.current.ended) {
        speechAudioRef.current.currentTime = 0;
      }
      try {
        await speechAudioRef.current.play();
      } catch {
        setSpeechPlayback((current) => ({
          ...current,
          status: "error",
          error: "Audio didn't start. Use the audio controls to try again.",
        }));
      }
      return;
    }

    if (speechAudioRef.current) {
      speechAudioRef.current.pause();
      speechAudioRef.current.currentTime = 0;
      speechAudioRef.current = null;
    }
    if (speechAudioUrlRef.current) {
      URL.revokeObjectURL(speechAudioUrlRef.current);
      speechAudioUrlRef.current = "";
    }

    const requestId = speechRequestIdRef.current + 1;
    speechRequestIdRef.current = requestId;
    setSpeechPlayback({ messageId: message.id, requestId, status: "loading", url: "", error: "" });

    try {
      const response = await fetch("/api/speech/synthesize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: message.answer, response_language: language }),
      });

      if (!response.ok) {
        const result = await response.json().catch(() => null);
        if (response.status === 503) {
          throw new Error("Read-aloud isn't available right now. You can still read this answer.");
        }
        if (response.status === 504) {
          throw new Error("Speech generation took too long. Please try again.");
        }
        throw new Error(
          typeof result?.detail === "string"
            ? result.detail
            : "Saar couldn't prepare audio for this answer. Please try again.",
        );
      }

      const audioBlob = await response.blob();
      if (!audioBlob.size || !audioBlob.type.toLowerCase().startsWith("audio/")) {
        throw new Error("Saar received an audio file it couldn't play. Please try again.");
      }
      if (requestId !== speechRequestIdRef.current) {
        return;
      }

      const audioUrl = URL.createObjectURL(audioBlob);
      speechAudioUrlRef.current = audioUrl;
      setSpeechPlayback({ messageId: message.id, requestId, status: "ready", url: audioUrl, error: "" });
    } catch (error) {
      if (requestId !== speechRequestIdRef.current) {
        return;
      }
      const friendlyError = error instanceof TypeError
        ? "Saar couldn't reach read-aloud. Please check your connection and try again."
        : error instanceof Error && error.message
          ? error.message
          : "Saar couldn't prepare audio for this answer. Please try again.";
      setSpeechPlayback({ messageId: message.id, requestId, status: "error", url: "", error: friendlyError });
    }
  }

  async function handleDocumentSelected(event) {
    const selectedFile = event.currentTarget.files?.[0];
    event.currentTarget.value = "";

    if (!selectedFile || voiceStatus !== "idle") {
      return;
    }

    clearSpeechAudio();
    setUploadedDocument(null);
    setIsPreviewOpen(false);
    setUploadError("");
    setMessages([]);
    setQuestion("");
    setVoiceError("");
    setTranscriptReady(false);
    setIsUploading(true);

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("response_language", language);

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

    if (!questionText || !uploadedDocument?.pages?.length || isUploading || isAsking || voiceStatus !== "idle") {
      return;
    }

    const messageId = Date.now();
    const answerId = `answer-${messageId}`;
    setQuestion("");
    setTranscriptReady(false);
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
                    disabled={voiceStatus !== "idle"}
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

                      {!message.pending && !message.error && (
                        <div className="answer-audio-actions">
                          <button
                            className="read-aloud-button"
                            type="button"
                            onClick={() => handleReadAloud(message)}
                            disabled={speechPlayback.status === "loading"}
                            aria-label={
                              speechPlayback.messageId === message.id && speechPlayback.status === "playing"
                                ? "Stop reading Saar's answer"
                                : speechPlayback.messageId === message.id && speechPlayback.status === "loading"
                                  ? "Preparing audio for Saar's answer"
                                  : speechPlayback.messageId === message.id && speechPlayback.status === "ready"
                                    ? "Play Saar's answer"
                                    : "Read Saar's answer aloud"
                            }
                            aria-pressed={speechPlayback.messageId === message.id && speechPlayback.status === "playing"}
                          >
                            <SpeakerIcon />
                            <span>
                              {speechPlayback.messageId === message.id && speechPlayback.status === "loading"
                                ? "Preparing audio…"
                                : speechPlayback.messageId === message.id && speechPlayback.status === "playing"
                                  ? "Stop reading"
                                  : speechPlayback.messageId === message.id && speechPlayback.status === "ready"
                                    ? "Play answer"
                                    : "Read aloud"}
                            </span>
                          </button>

                          {speechPlayback.messageId === message.id && speechPlayback.url && (
                            <audio
                              ref={speechAudioRef}
                              className="answer-audio-player"
                              src={speechPlayback.url}
                              controls
                              autoPlay
                              preload="none"
                              aria-label="Saar answer audio controls"
                              onPlay={() => setSpeechPlayback((current) => (
                                current.requestId === speechPlayback.requestId
                                  ? { ...current, status: "playing" }
                                  : current
                              ))}
                              onPause={() => setSpeechPlayback((current) => (
                                current.requestId === speechPlayback.requestId && current.status === "playing"
                                  ? { ...current, status: "ready" }
                                  : current
                              ))}
                              onEnded={() => setSpeechPlayback((current) => (
                                current.requestId === speechPlayback.requestId
                                  ? { ...current, status: "ready" }
                                  : current
                              ))}
                              onError={() => setSpeechPlayback((current) => (
                                current.requestId === speechPlayback.requestId
                                  ? { ...current, status: "error", error: "This browser couldn't play the generated audio." }
                                  : current
                              ))}
                            />
                          )}

                          {speechPlayback.messageId === message.id && speechPlayback.error && (
                            <p className="voice-error answer-audio-error" role="alert">{speechPlayback.error}</p>
                          )}
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
                  ref={questionInputRef}
                  className="composer-input"
                  type="text"
                  value={question}
                  onChange={(event) => {
                    setQuestion(event.target.value);
                    setTranscriptReady(false);
                  }}
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
                  className={`voice-button${voiceStatus === "recording" ? " is-recording" : ""}`}
                  type="button"
                  onClick={handleVoiceButton}
                  disabled={
                    voiceStatus === "requesting" ||
                    voiceStatus === "transcribing" ||
                    (voiceStatus !== "recording" && !canStartRecording)
                  }
                  aria-label={voiceStatus === "recording" ? "Stop recording question" : "Record a voice question"}
                  aria-pressed={voiceStatus === "recording"}
                  aria-describedby="voice-guidance"
                  title={voiceStatus === "recording" ? "Stop recording" : "Record a voice question"}
                >
                  {voiceStatus === "transcribing" ? (
                    <span className="voice-spinner" aria-hidden="true" />
                  ) : voiceStatus === "recording" ? (
                    <StopIcon />
                  ) : (
                    <MicrophoneIcon />
                  )}
                </button>
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
              <span className="sr-only" id="voice-guidance">
                Record up to {MAX_RECORDING_SECONDS} seconds. Saar will transcribe it into this editable question field for you to review before sending.
              </span>
              <span className="sr-only" role="status" aria-live="polite">
                {voiceStatus === "requesting"
                  ? "Waiting for microphone permission."
                  : voiceStatus === "recording"
                    ? "Recording started. Activate stop recording when you are finished."
                    : ""}
              </span>
              <div className="voice-feedback" aria-live="off">
                {voiceStatus === "requesting" && <p className="voice-status">Waiting for microphone permission…</p>}
                {voiceStatus === "recording" && (
                  <p className="voice-status recording-status">
                    <span className="recording-dot" aria-hidden="true" />
                    Recording {formatRecordingTime(voiceSeconds)} / {formatRecordingTime(MAX_RECORDING_SECONDS)}
                    <span className="recording-hint">Select stop when you're done speaking.</span>
                  </p>
                )}
                {voiceStatus === "transcribing" && (
                  <p className="voice-status" role="status" aria-live="polite">
                    <span className="voice-spinner" aria-hidden="true" /> Saar is transcribing your question…
                  </p>
                )}
                {transcriptReady && voiceStatus === "idle" && (
                  <p className="voice-success" role="status" aria-live="polite">
                    Transcript ready. Review or edit it above, then send when you're ready.
                  </p>
                )}
                {voiceError && <p className="voice-error" role="alert">{voiceError}</p>}
                {voiceAvailability.supported === false && hasReadableDocument && (
                  <p className="voice-unavailable" role="status">{voiceAvailability.reason} You can type your question instead.</p>
                )}
              </div>
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
