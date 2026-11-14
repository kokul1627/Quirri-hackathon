import { useEffect, useRef, useState } from 'react';
import type { Dispatch, FormEvent, SetStateAction } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';

export type ConversationTurn = { who: 'you' | 'kit'; text: string; data?: unknown; intent?: string; source?: string; citations?: Array<{ source: string; page?: number }>; confirmation_required?: boolean };
export type AssistantStage = 'idle' | 'listening' | 'transcribing' | 'thinking' | 'calling_kit_api' | 'retrieving_documents' | 'completed' | 'error';

type Props = {
  role?: 'student' | 'teacher';
  turns: ConversationTurn[];
  setTurns: Dispatch<SetStateAction<ConversationTurn[]>>;
  conversationId?: string;
  setConversationId: Dispatch<SetStateAction<string | undefined>>;
};

const chatApi = axios.create({ baseURL: `${import.meta.env.VITE_API_BASE || ''}/api`, withCredentials: true });
const STAGE_LABEL: Record<AssistantStage, string> = {
  idle: 'Ready', listening: 'Listening', transcribing: 'Transcribing with local Whisper',
  thinking: 'Thinking', calling_kit_api: 'Calling KIT API', retrieving_documents: 'Retrieving documents',
  completed: 'Completed', error: 'Error',
};

function stageForQuestion(text: string): AssistantStage {
  if (/document|lesson|rulebook|handbook|brochure|notes/i.test(text)) return 'retrieving_documents';
  if (/attendance|timetable|subject|academic|holiday|exam|leave|on.?duty|request|placement|class|course|college|fee|contact/i.test(text)) return 'calling_kit_api';
  return 'thinking';
}

function formatDuration(seconds: number): string {
  return `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${(seconds % 60).toString().padStart(2, '0')}`;
}

function recordingType(): string | undefined {
  if (typeof MediaRecorder.isTypeSupported !== 'function') return undefined;
  return ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/mp4']
    .find((type) => MediaRecorder.isTypeSupported(type));
}

function extensionFor(mime: string): string {
  if (mime.includes('ogg')) return 'ogg';
  if (mime.includes('mp4')) return 'mp4';
  return 'webm';
}

export default function KITVoiceAssistant({ role, turns, setTurns, conversationId, setConversationId }: Props) {
  const [stage, setStage] = useState<AssistantStage>('idle');
  const [duration, setDuration] = useState(0);
  const [text, setText] = useState('');
  const [transcriptDraft, setTranscriptDraft] = useState('');
  const [autoSend, setAutoSend] = useState(true);
  const [error, setError] = useState('');
  const recorder = useRef<MediaRecorder | null>(null);
  const stream = useRef<MediaStream | null>(null);
  const chunks = useRef<Blob[]>([]);
  const timer = useRef<number | undefined>();
  const startedAt = useRef(0);
  const autoSendRef = useRef(autoSend);
  const busy = ['transcribing', 'thinking', 'calling_kit_api', 'retrieving_documents'].includes(stage);
  const isRecording = stage === 'listening';

  useEffect(() => { autoSendRef.current = autoSend; }, [autoSend]);
  useEffect(() => () => {
    if (timer.current !== undefined) window.clearInterval(timer.current);
    recorder.current?.stop();
    stream.current?.getTracks().forEach((track) => track.stop());
  }, []);

  async function sendToAssistant(rawText: string, isTranscript = false) {
    const message = rawText.trim();
    if (!message || busy) return;
    setError('');
    setTranscriptDraft('');
    setTurns((current) => [...current, { who: 'you', text: message }]);
    setStage('thinking');
    try {
      await new Promise<void>((resolve) => window.setTimeout(resolve, 120));
      setStage(stageForQuestion(message));
      const { data } = await chatApi.post('/chat', { message, conversation_id: conversationId });
      setConversationId(data.conversation_id);
      setTurns((current) => [...current, { who: 'kit', text: data.answer, data: data.tool_result, intent: data.intent, source: data.source, citations: data.citations, confirmation_required: data.confirmation_required }]);
      setStage('completed');
    } catch (requestError: unknown) {
      const response = axios.isAxiosError(requestError) ? requestError.response?.data?.detail : undefined;
      if (isTranscript) setTranscriptDraft(message);
      setError(typeof response === 'string' ? response : 'Could not reach KIT Assistant. Check the backend and try again.');
      setStage('error');
    }
  }

  async function transcribe(blob: Blob, mimeType: string) {
    if (!blob.size) {
      setError('No audio was recorded. Try again or use text input.');
      setStage('error');
      return;
    }
    setStage('transcribing');
    setError('');
    const form = new FormData();
    form.append('file', blob, `recording.${extensionFor(mimeType)}`);
    try {
      const { data } = await axios.post(`${import.meta.env.VITE_API_BASE || ''}/speech/transcribe`, form, { withCredentials: true });
      const transcript = String(data.text || '').trim();
      if (!transcript) {
        setError('Whisper returned an empty transcript. Try again or type your question.');
        setStage('error');
        return;
      }
      setTranscriptDraft(transcript);
      if (autoSendRef.current) await sendToAssistant(transcript, true);
      else setStage('completed');
    } catch (requestError: unknown) {
      const response = axios.isAxiosError(requestError) ? requestError.response?.data?.detail : undefined;
      setError(typeof response === 'string' ? response : 'Transcription failed. You can type your question instead.');
      setStage('error');
    }
  }

  async function startRecording() {
    setError('');
    setTranscriptDraft('');
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setError('This browser does not support microphone recording. Use text input instead.');
      setStage('error');
      return;
    }
    try {
      const media = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.current = media;
      const mimeType = recordingType();
      const nextRecorder = new MediaRecorder(media, mimeType ? { mimeType } : undefined);
      recorder.current = nextRecorder;
      chunks.current = [];
      nextRecorder.ondataavailable = (event) => { if (event.data.size) chunks.current.push(event.data); };
      nextRecorder.onerror = () => {
        media.getTracks().forEach((track) => track.stop());
        setError('Recording failed. Check microphone access or use text input.');
        setStage('error');
      };
      nextRecorder.onstop = () => {
        media.getTracks().forEach((track) => track.stop());
        stream.current = null;
        if (timer.current !== undefined) window.clearInterval(timer.current);
        timer.current = undefined;
        const type = nextRecorder.mimeType || mimeType || 'audio/webm';
        void transcribe(new Blob(chunks.current, { type }), type);
      };
      setDuration(0);
      startedAt.current = Date.now();
      timer.current = window.setInterval(() => setDuration(Math.floor((Date.now() - startedAt.current) / 1000)), 250);
      nextRecorder.start(250);
      setStage('listening');
    } catch (permissionError: unknown) {
      const name = permissionError instanceof DOMException ? permissionError.name : '';
      setError(name === 'NotAllowedError' || name === 'SecurityError'
        ? 'Microphone permission was denied. Allow microphone access or use text input.'
        : 'Could not start the microphone. Check the device or use text input.');
      setStage('error');
    }
  }

  function stopRecording() {
    if (recorder.current?.state === 'recording') {
      recorder.current.stop();
      setStage('transcribing');
    }
  }

  function submitText(event: FormEvent) {
    event.preventDefault();
    const value = text;
    setText('');
    void sendToAssistant(value);
  }

  function clearConversation() {
    setTurns([]);
    setConversationId(undefined);
    setTranscriptDraft('');
    setText('');
    setError('');
    setStage('idle');
  }

  const suggestions = role === 'teacher' ? ['Show my classes', 'Show pending requests']
    : role === 'student' ? ['How is my attendance?', 'Show my timetable']
      : ['What courses are available?', 'Tell me about KIT'];

  return <div className="chat">
    <div className="chat-top">
      <div><b>KIT Assistant</b><small>{STAGE_LABEL[stage]}{isRecording ? ` · ${formatDuration(duration)}` : ''}</small></div>
      <div className="chat-tools"><span className={`assistant-state ${stage}`}><i />{isRecording ? `REC ${formatDuration(duration)}` : STAGE_LABEL[stage].toUpperCase()}</span><button type="button" className="clear-chat" onClick={clearConversation} disabled={busy || isRecording}>Clear conversation</button></div>
    </div>
    <div className="messages" aria-live="polite">
      {!turns.length && <div className="welcome"><div className="spark">✦</div><h3>What can I help you with?</h3><p>Ask by voice or text. Sign in for private student or teacher services.</p><div className="suggestions">{suggestions.map((item) => <button key={item} type="button" onClick={() => void sendToAssistant(item)} disabled={busy}>{item}<span>↗</span></button>)}</div></div>}
      {turns.map((turn, index) => <div className={`turn ${turn.who}`} key={`${index}-${turn.who}`}><span className="turn-label">{turn.who === 'you' ? 'YOU' : 'KIT'}</span><div className="bubble">{turn.who === 'kit' ? <div className="assistant-markdown"><ReactMarkdown>{turn.text}</ReactMarkdown></div> : turn.text}{turn.who === 'kit' && (turn.intent || turn.source) && <small className="turn-meta">{turn.intent?.replace(/_/g, ' ')}{turn.source ? ` · ${turn.source.replace(/_/g, ' ')}` : ''}</small>}{turn.citations?.length ? <div className="citations">Sources: {turn.citations.map((citation, citationIndex) => <span key={`${citation.source}-${citation.page}-${citationIndex}`}>{citation.source}{citation.page ? `, p. ${citation.page}` : ''}{citationIndex < turn.citations!.length - 1 ? ' · ' : ''}</span>)}</div> : null}{turn.data != null && <details className="tool-data"><summary>Retrieved details</summary><pre>{JSON.stringify(turn.data, null, 2)}</pre></details>}{turn.confirmation_required && <div className="confirmation-actions"><button type="button" onClick={() => void sendToAssistant('CONFIRM')} disabled={busy}>Confirm action</button><button type="button" onClick={() => void sendToAssistant('CANCEL')} disabled={busy}>Cancel</button></div>}</div></div>)}
      {busy && <div className="thinking">{STAGE_LABEL[stage]}<span>•••</span></div>}
    </div>
      {transcriptDraft && <div className="transcript-panel"><label htmlFor="transcript-edit">Recognized transcript</label><textarea id="transcript-edit" value={transcriptDraft} onChange={(event) => setTranscriptDraft(event.target.value)} disabled={busy} /><button type="button" className="send-transcript" onClick={() => void sendToAssistant(transcriptDraft, true)} disabled={busy || !transcriptDraft.trim()}>{autoSend ? 'Edit and resend transcript' : 'Send transcript to KIT'}</button></div>}
    {error && <div className="error" role="alert">{error}</div>}
    <div className="auto-send"><label><input type="checkbox" checked={autoSend} onChange={(event) => setAutoSend(event.target.checked)} disabled={busy || isRecording} /> Automatically send transcript to KIT</label></div>
    <form className="composer" onSubmit={submitText}>
      {isRecording ? <button type="button" className="mic active" onClick={stopRecording} aria-label="Stop recording">■</button> : <button type="button" className="mic" onClick={() => void startRecording()} disabled={busy} aria-label="Start recording">🎙</button>}
      <input value={text} onChange={(event) => setText(event.target.value)} placeholder="Ask KIT Assistant anything…" aria-label="Message" disabled={busy || isRecording} />
      <button className="send" type="submit" disabled={busy || isRecording || !text.trim()}>Send <span>↑</span></button>
    </form>
    <div className="chat-foot">{isRecording ? `Recording ${formatDuration(duration)} · click stop when done` : 'Local Whisper transcribes your recording. Text input remains available.'}</div>
  </div>;
}
