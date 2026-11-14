# KIT Voice Assistant

Standalone KIT Hackathon application. This project is separate from the TNPSC application.

KIT Voice Assistant combines a React/TypeScript frontend with a FastAPI backend. The backend connects to KIT APIs, locally transcribes microphone recordings with Whisper, retrieves college PDF passages from ChromaDB, and uses Gemini to write grounded text answers when configured.

## Architecture

```text
React UI
  ├─ Typed question ───────────────────────────────┐
  └─ Microphone → MediaRecorder → /speech/transcribe│
                         ↓ local Whisper transcript│
                                                    ↓
                                      FastAPI supervisor
                                       ├─ KIT API routes
                                       ├─ ChromaDB PDF retrieval
                                       └─ Gemini grounded answer
                                                    ↓
                                     Response + citations + data
```

### Frontend

- React, TypeScript, and Vite app.
- `KITVoiceAssistant.tsx` records audio using browser `MediaRecorder`; it does not use browser speech recognition or browser-native transcription.
- Sends audio as `multipart/form-data` to FastAPI `POST /speech/transcribe`.
- Shows transcript, progress, answer, errors, citations, and action confirmation controls. User can edit transcript or ask by text.
- Frontend calls `/api` and `/speech` on backend. Set `VITE_API_BASE` to backend origin for separate frontend/backend hosts; leave empty when served behind same-origin routing.

### FastAPI supervisor

`backend/services/supervisor.py` classifies request intent before selecting a fixed backend route or document search. `backend/app.py` enforces authentication, role checks, and write confirmations, then returns response metadata for frontend display.

- Policy and general college-document questions → local ChromaDB retrieval.
- Personal or current student/teacher data → authenticated KIT API routes.
- Admission fee questions can combine course API data with brochure passages.
- Unknown questions receive a clarification prompt instead of an arbitrary API call.
- Gemini only summarizes the API data and document passages supplied by the backend. It does not choose API routes or establish identity/role.
- Write actions require authorization and explicit confirmation. OD/leave requests collect missing details, create a draft, then require confirmation before submission.

### Speech and documents

- `backend/services/speech.py` loads one local CPU Whisper model per FastAPI process. `POST /speech/transcribe` accepts uploaded audio; `GET /speech/health` reports readiness.
- FFmpeg must be installed and available on `PATH` for Whisper audio decoding.
- `backend/services/knowledge_base.py` extracts PDFs with PyMuPDF, creates overlapping page-aware chunks, and stores vectors with source filename/page metadata in persistent ChromaDB collection `kit_college_documents`.
- By default, PDF input folder is `../college docs`; set `KIT_PDF_DIR` to another folder if needed. Index PDFs with the script below. ChromaDB creates its local embedding model on first indexing.

## Requirements

- Python 3.9+ (use CPU-compatible PyTorch and torchaudio builds for your Python version/platform).
- Node.js and npm.
- FFmpeg installed and on `PATH` for Whisper.
- KIT participant key for live KIT API operations.
- Gemini API key for generated answers. Without it, API-backed responses and document retrieval remain available, but natural-language generation may use backend fallback text.
- College PDF files for document indexing.

## Local setup (Windows PowerShell)

From project folder:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
Copy-Item .env.example .env
npm install
```

Edit `.env` and set required backend secrets and local paths. Never place KIT or Gemini secrets in `VITE_*` variables or frontend source. `.env` is ignored by Git.

Index local college PDFs:

```powershell
python scripts\index_college_pdfs.py
```

Run backend and frontend in separate terminals:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

```powershell
npm run dev
```

Open the Vite URL shown in terminal (normally `http://localhost:5173`). Backend API docs: `http://127.0.0.1:8000/docs`.

First startup may download Whisper checkpoint; first PDF indexing may download a local embedding model. Both need network access and disk space.

## Configuration

Copy `.env.example` to `.env`; set values for your environment. Important variables:

| Variable | Purpose |
| --- | --- |
| `KIT_BASE_URL` | KIT API base URL |
| `KIT_PARTICIPANT_KEY` | Server-side participant credential for KIT API |
| `GEMINI_API_KEY` | Server-side Gemini credential |
| `GEMINI_MODEL` | Gemini model name supported by your key/account |
| `DATABASE_URL` | Conversation and pending-action database; local default uses SQLite |
| `KIT_PDF_DIR` | Folder containing college PDFs |
| `CHROMA_DB_DIR` | Persistent local vector database directory |
| `FRONTEND_ORIGIN` | Exact frontend origin allowed by backend CORS |
| `SESSION_COOKIE_SECURE` | Set `true` when deployed behind HTTPS |
| `SPEECH_PROVIDER` | Keep `local_whisper` |
| `WHISPER_MODEL` | Local Whisper checkpoint, e.g. `tiny` or `base` |
| `WHISPER_DEVICE` | Current service requires `cpu` |
| `WHISPER_LANGUAGE` | Language hint or `auto` |

For Vite deployment, `VITE_API_BASE` is a frontend build-time setting containing only public backend URL (no secrets).

## API routes

Common app routes:

- `GET /api/health` — backend health.
- `POST /api/chat` — typed chat and supervisor routing.
- `POST /api/voice` — audio upload, transcription, and assistant workflow.
- `GET /api/history` and `GET /api/history/{conversation_id}` — authenticated chat history.
- `POST /speech/transcribe` and `GET /speech/health` — local Whisper service.
- `/api/auth/*`, `/api/student/*`, `/api/teacher/*`, `/api/public/*` — backend-mediated KIT operations.

All KIT requests and keys stay on backend. Session uses an HttpOnly cookie; authenticated routes check backend session and role. Mutations require confirmation before submission.

## Tests and build

```powershell
python -m unittest discover -s tests -v
python -m py_compile backend\app.py backend\services\supervisor.py
npm run build
```

Supervisor tests cover policy vs. personal-data routing, authenticated routes, role denial, confirmation for writes, collecting missing OD details, hybrid retrieval, Gemini failure handling, and unknown-question clarification. Tests mock KIT/Gemini requests; live integration still requires valid service credentials and network access.

## Deployment

Recommended topology: deploy React static frontend to Vercel; run FastAPI on a persistent Python host that supports CPU Whisper and durable storage. Set `VITE_API_BASE` to public HTTPS API origin, backend `FRONTEND_ORIGIN` to Vercel origin, and `SESSION_COOKIE_SECURE=true`. Backend currently sets session cookie `SameSite=Lax`; if frontend and API are cross-site, update cookie configuration to `SameSite=None; Secure`, enable credentialed CORS for exact frontend origin, and verify browser cookie behavior.

Vercel can host FastAPI functions, but this project’s local Whisper model, PyTorch dependencies, persistent ChromaDB, and SQLite writes need deployment changes. Serverless filesystem is temporary; migrate app database and vector storage to managed persistent services and verify function package, memory, and duration limits before running Whisper there. Do not deploy local `.env`, `.venv`, `node_modules`, `chroma_db`, or SQLite database as application source.

## Known boundaries

- PDF extraction uses selectable text; scanned image-only PDFs need OCR before useful retrieval.
- Whisper accuracy depends on microphone quality, recording language, and selected checkpoint.
- Local ChromaDB and SQLite are appropriate for development, not multi-instance production without persistent/shared storage.
- Some advanced KIT operations may require UI forms or permissions beyond chat. Backend authorization remains authoritative.
