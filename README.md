# 🇰🇪 SafEye — Kenya's AI-Powered Election & Media Integrity Shield

> *Kulinda Ukweli wa Kidijitali* — Protecting Digital Truth

SafEye detects deepfakes, manipulated screenshots, WhatsApp misinformation, forged documents, and AI-generated audio — built specifically for Kenya's threat landscape, legal framework, and languages.

[![Made in Kenya](https://img.shields.io/badge/Made_in-Kenya_🇰🇪-black?labelColor=BE0027&color=006600)](https://github.com/Vicmuratha/NIRU-HACKATHON)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/React-18.3-61DAFB.svg?logo=react)](https://reactjs.org/)
[![Flask](https://img.shields.io/badge/Flask-latest-000000.svg?logo=flask)](https://flask.palletsprojects.com/)
[![Vite](https://img.shields.io/badge/Vite-7.3-646CFF.svg?logo=vite)](https://vitejs.dev/)
[![License](https://img.shields.io/badge/License-ISC-green.svg)](LICENSE)

## The Problem

Kenya faces a growing AI manipulation crisis with **zero dedicated defence infrastructure**:

| Threat | Reality | Impact |
|---|---|---|
| **Election deepfakes** | Manipulated "leaked audio" and doctored images of politicians circulate before every election since 2013. By 2027, AI-generated video will be indistinguishable from real footage. | 2007/08 PEV: 1,500+ killed, 600,000+ displaced — incitement spread via SMS and radio |
| **Fake news screenshots** | Edited Citizen TV / NTV / Nation breaking news screenshots are the #1 misinformation format on Kenyan WhatsApp. | Millions exposed to fabricated "breaking news" daily |
| **WhatsApp forwards** | 67% of Kenyans receive news primarily through WhatsApp, where false health scares, political fabrications, and scam messages spread unchecked. | Reuters Institute 2024 |
| **Document forgery** | Counterfeit KRA PINs, HELB clearance letters, and edited M-Pesa screenshots used for fraud daily. | DCI Cybercrime Unit reports |

**No existing tool addresses Kenya's specific threat landscape.** Western deepfake detectors don't understand Kenyan news outlets, Swahili misinformation patterns, or our regulatory framework.

## What SafEye Does

SafEye is a **Kenya-built, Kenya-focused** AI platform with five detection modes:

| Mode | How It Works | Kenya Use Case |
|---|---|---|
| 📸 **Deepfake Detector** | EfficientNet-B4 + ELA + metadata analysis | Detect manipulated images of politicians and fake campaign material |
| 🎙️ **Audio Analyser** | MFCC analysis + spectral pattern detection | Flag manipulated "leaked audio" recordings — the primary threat is splicing, not AI voice cloning (yet) |
| 📝 **Fake News Classifier** | RoBERTa NLP + clickbait detection | Identify AI-generated and fabricated articles |
| 💬 **WhatsApp Forward Checker** | Pattern matching + Swahili hoax detection | Analyse forwarded messages for Safaricom hoaxes, political misinformation, and health scares |
| 📄 **Document & Screenshot Verifier** | OCR + ELA + format validation | Detect forged KRA PINs, HELB letters, fake M-Pesa confirmations, and edited news screenshots |

All results include:
- **Kenya legal context** (Computer Misuse & Cybercrimes Act 2018, NCIC Act, Elections Act)
- **Direct reporting links** to NCIC, DCI Cybercrime Unit, Communications Authority, IEBC
- **Bilingual warnings** in English and Swahili
- **Election context analysis** with incitement detection

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Mobile Access](#mobile-access)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Docker & Deployment](#docker--deployment)
- [Testing](#testing)
- [Demo Script](#demo-script)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Features

| Capability | Details |
|---|---|
| **Image Detection** | EfficientNet-B4 fine-tuned checkpoint (`.pth`) **or** HuggingFace `dima806/deepfake_vs_real_image_detection`, combined with Error Level Analysis, EXIF metadata checks, face texture analysis (via DeepFace), and spectral noise analysis. Weighted scoring: AI 60 %, metadata 15 %, ELA 15 %, face 7 %, noise 3 %. |
| **Audio Detection** | MFCC variance and silence-ratio heuristics via librosa. Honest about Kenya's context: the primary audio threat is **splicing** (editing real recordings), not AI voice cloning. Swahili AI audio is still detectable by ear as of 2026. |
| **Text Detection** | HuggingFace `hamzab/roberta-fake-news-classification` pipeline with clickbait keyword boosting. Local model files used when present. |
| **WhatsApp Forward Checker** | Pattern-matching engine detecting misinformation indicators in English and Swahili: Safaricom hoaxes, political fabrications, health scares, fake job listings, and M-Pesa fee rumours. Cross-referenced with AI text classification. |
| **Document & Screenshot Verifier** | OCR-powered detection of forged KRA PINs, HELB letters, fake M-Pesa confirmations, and manipulated Citizen TV / NTV / Nation breaking news screenshots. Combines OCR text validation with ELA and AI deepfake scoring. |
| **Election Shield** | Political context engine detecting mentions of politicians, ethnic incitement keywords, and media outlet impersonation. Provides bilingual (EN/SW) warnings, applicable Kenyan law references, and direct NCIC/DCI reporting links. |
| **React UI** | Single-page app built with React 18, Vite 7, Tailwind CSS 3, Framer Motion, and Lucide icons. Five analysis tabs with drag-and-drop upload, animated risk scores, and Kenya-specific threat warnings. |
| **OAuth Authentication** | Unified Flask server with Google and GitHub OAuth via Authlib, SQLite user store, session management, user profiles, scan history, and login/signup HTML templates. |
| **User Profiles** | Full profile management — edit name, bio, phone, location, organization, profile picture upload. Per-user scan statistics and history. |
| **Scan History** | Every analysis is saved to SQLite and linked to the authenticated user. Filterable by type (image, audio, text, forward, document). Deletable. |
| **Kenya Legal Framework** | All detection results reference applicable Kenyan law (CMCA 2018, NCIC Act, Elections Act, Penal Code). Direct reporting links to DCI Cybercrime, NCIC, Communications Authority, and IEBC. |
| **Docker** | CPU-optimised Dockerfile with pre-installed `torch` (CPU wheel) and `tensorflow-cpu`, Gunicorn, health check, and 600-second model-load timeout. |

---

## Architecture

```
┌─────────────────────┐        /api proxy         ┌──────────────────────────┐
│   React + Vite UI   │ ──────────────────────────▶│  Unified Backend (Flask) │
│   localhost:3000     │                            │  localhost:7860           │
│   (or tunnel URL)    │                            │                          │
└─────────────────────┘                            │  Auth (Google/GitHub)    │
                                                   │  /api/analyze/*          │
┌─────────────────────┐                            │  /api/profile            │
│   📱 Phone Access    │  ─── localtunnel ────────▶│  /api/history            │
│   https://xxx.loca.lt│                            │  /api/users              │
└─────────────────────┘                            └──────────┬───────────────┘
                                                              │
                                                     ┌────────▼────────┐
                                                     │  models/         │
                                                     │  ~1.1 GB total   │
                                                     │  SQLite (users.db)│
                                                     └─────────────────┘
```

**Unified backend** — `app.py` runs everything on port 7860: authentication (local + OAuth), detection APIs, user profiles, scan history, and admin features. The Vite dev server proxies `/api`, `/auth`, `/login`, `/logout`, and `/uploads` requests to port 7860.

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.9+ | 3.10+ recommended |
| Node.js | 18+ | |
| npm | 8+ | |
| ffmpeg | — | Required by librosa for audio processing |
| Git | — | |

**Install ffmpeg:**

```bash
# Ubuntu / Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

---

## Project Structure

```
NIRU-HACKATHON/
├── app.py                     # Unified backend (Auth + Detection + Profiles + History)
├── gunicorn.conf.py           # Production Gunicorn configuration
├── docker-compose.yml         # Docker Compose for development & production
├── Dockerfile                 # Multi-stage production Docker build
├── start-tunnel.sh            # Public tunnel script (phone access)
├── backend/
│   ├── __init__.py            # Package init
│   ├── config.py              # Centralised environment-aware configuration
│   ├── middleware.py           # Security headers, rate limiting, request logging
│   ├── errors.py              # Structured error handling & custom exceptions
│   ├── logging_config.py      # JSON/text structured logging
│   ├── app.py                 # Legacy detection API (port 7860)
│   ├── election_shield.py     # 🇰🇪 Election context analysis & incitement detection
│   ├── whatsapp_checker.py    # 🇰🇪 WhatsApp forward misinformation detector
│   ├── kenya_documents.py     # 🇰🇪 Kenyan document forgery detector (KRA, HELB, M-Pesa)
│   ├── fake_screenshot.py     # 🇰🇪 Fake news screenshot detector (Citizen TV, NTV, Nation)
│   ├── audio_context.py       # 🇰🇪 Honest Kenyan audio threat context
│   ├── data/                  # Backend data files
│   └── uploads/               # Backend uploads directory
├── models/
│   ├── download_models.py     # Azure Blob / URL model downloader script
│   ├── audio_model/           # WavLM config + weights (~361 MB)
│   ├── image_model/           # EfficientNet-B4 checkpoint (~47 MB)
│   ├── bestdeepfake/          # Alternate .pth checkpoint
│   └── text_model/            # RoBERTa fake-news weights + tokenizer (~704 MB)
├── src/
│   ├── App.tsx                # Main React component (hero, analysis panel, profile, history)
│   ├── main.tsx               # Vite entry point (with ErrorBoundary)
│   ├── styles.css             # Tailwind + custom animations
│   ├── components/
│   │   └── ErrorBoundary.tsx  # React error boundary with recovery UI
│   ├── lib/
│   │   └── api.ts             # Typed API client with timeout & error handling
│   └── types/
│       └── index.ts           # Shared TypeScript interfaces
├── static/css/                # CSS for auth templates
├── templates/                 # login.html, signup.html (Jinja2)
├── tests/
│   ├── conftest.py            # Shared fixtures & test configuration
│   ├── test_api.py            # Integration tests for Flask API routes
│   ├── test_image.py          # Unit tests for image detector
│   ├── test_audio.py          # Unit tests for audio detector
│   └── test_text.py           # Unit tests for text detector
├── data/
│   └── detection_log.json     # JSONL detection history
├── deploy_package/            # Standalone deployment build (app.py, Dockerfile, requirements.txt)
├── uploads/                   # Temporary file uploads (gitignored)
├── .env.example               # Environment variable template
├── .dockerignore              # Docker build exclusions
├── CONTRIBUTING.md            # Contribution guidelines
├── pytest.ini                 # Pytest configuration
├── requirements.txt           # Python dependencies (pinned ranges)
├── package.json               # Node.js dependencies
├── vite.config.ts             # Vite config (host: 0.0.0.0, proxy /api → :7860)
├── tailwind.config.js
├── postcss.config.js
├── tsconfig.json
└── index.html                 # Vite HTML entry (SEO + noscript)
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Vicmuratha/NIRU-HACKATHON.git
cd NIRU-HACKATHON
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
npm install
```

### 3. Download AI models (~1.1 GB)

```bash
python models/download_models.py
```

> Models are **not** included in the repo. The script downloads from Azure Blob Storage (if configured) or prints placeholder instructions. See [Environment Variables](#environment-variables) for Azure setup.

### 4. Start the unified backend

```bash
python app.py
```

Runs on **http://localhost:7860**. Handles authentication, detection APIs, user profiles, scan history, and admin features. Models are downloaded on startup if not already present.

### 5. Start the frontend

```bash
npm run dev
```

Runs on **http://localhost:3000**. API and auth calls are proxied to `localhost:7860`.

### 6. (Optional) Access from your phone — any network

```bash
# Set the tunnel URL as the frontend URL
export FRONTEND_URL=https://safeye-$(whoami).loca.lt

# Restart the backend with the tunnel URL
python app.py &

# Start the tunnel
./start-tunnel.sh
```

Open the printed URL on your phone. See [Mobile Access](#mobile-access) for full details.

---

## Mobile Access

SafEye can be accessed from your phone on **any network** using a free tunnel.

### Same Wi-Fi (no tunnel needed)

If your phone and computer are on the same Wi-Fi, just open `http://<your-ip>:3000` on your phone. Your IP is printed when the backend starts (e.g. `192.168.0.100`).

On Fedora, open the firewall first:

```bash
sudo firewall-cmd --add-port=3000/tcp --add-port=7860/tcp --permanent
sudo firewall-cmd --reload
```

### Any Network (tunnel)

Uses [localtunnel](https://theboroer.github.io/localtunnel-www/) — free, no signup required.

**Terminal 1** — Backend:
```bash
export FRONTEND_URL=https://safeye-$(whoami).loca.lt
python app.py
```

**Terminal 2** — Frontend:
```bash
npm run dev
```

**Terminal 3** — Tunnel:
```bash
./start-tunnel.sh
```

On your phone, open the printed URL (e.g. `https://safeye-blackhole.loca.lt`).

> **Note:** On first visit, localtunnel shows a splash page — click **"Click to Continue"**.

### Google OAuth on phone

For Google sign-in to work via the tunnel, add the tunnel URL to your [Google Cloud Console](https://console.cloud.google.com/) credentials:

1. **Authorized JavaScript origins:** `https://safeye-blackhole.loca.lt`
2. **Authorized redirect URIs:** `https://safeye-blackhole.loca.lt/auth/google/callback`

---

## Environment Variables

Create a `.env` file in the project root:

```bash
# ── Unified Backend (app.py) ──
FLASK_SECRET_KEY=your-secret-key
FRONTEND_URL=http://localhost:3000       # or your tunnel URL (e.g. https://safeye-user.loca.lt)
EXTRA_CORS_ORIGINS=                      # comma-separated extra allowed origins

GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
GITHUB_CLIENT_ID=...
GITHUB_CLIENT_SECRET=...

# ── Detection API ──
JWT_SECRET_KEY=your-jwt-secret

# ── Azure Blob model download ──
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
AZURE_STORAGE_CONTAINER=ai-models

AZURE_AUDIO_BLOBS=audio_model/config.json,audio_model/model.safetensors,audio_model/preprocessor_config.json
AZURE_TEXT_BLOBS=text_model/config.json,text_model/model.safetensors,text_model/tokenizer_config.json,text_model/added_tokens.json,text_model/special_tokens_map.json,text_model/spm.model
AZURE_IMAGE_BLOBS=image_model/config.json,image_model/model.safetensors,image_model/preprocessor_config.json
AZURE_IMAGE_BLOB=image_model.zip          # optional archive

# ── Public URL alternative (no connection string needed) ──
AZURE_BLOB_BASE_URL=https://<account>.blob.core.windows.net
AZURE_SAS_TOKEN=sv=...                     # optional

# ── Auto-download on API startup (for Azure App Service) ──
DOWNLOAD_MODELS_ON_STARTUP=true
IMAGE_MODEL_DIR=/code/models/image_model
TEXT_MODEL_DIR=/code/models/text_model
```

---

## API Reference

### Detection API (`backend/app.py` — port 7860)

#### Health Check

```
GET /api/health
```

```json
{ "status": "healthy", "models_loaded": true }
```

#### Image Analysis

```
POST /api/analyze/image
Content-Type: multipart/form-data
Body: file (image — jpg, png, webp)
```

```json
{
  "risk_score": 23.5,
  "verdict": "AUTHENTIC",
  "confidence": 0.92,
  "findings": [
    "🤖 AI Model: Authentic (24.5% confidence)",
    "✓ Camera: Samsung SM-G991B"
  ],
  "kenya_warnings": [],
  "details": { "ai_confidence": 24.5 }
}
```

Verdicts: `AUTHENTIC` (< 40), `REVIEW_REQUIRED` (40–65), `LIKELY_DEEPFAKE` (> 65).

#### Audio Analysis

```
POST /api/analyze/audio
Content-Type: multipart/form-data
Body: file (audio — wav, mp3, ogg, flac)
```

```json
{
  "risk_score": 75,
  "is_authentic": false,
  "confidence": 0.88,
  "findings": ["⚠️ Robotic voice texture", "⚠️ Abnormal breathing pauses"],
  "kenya_warnings": [
    { "type": "AUDIO_MANIPULATION", "severity": "HIGH", "warning": "This audio shows signs of manipulation...", "action": "Report to DCI Cybercrime" }
  ],
  "kenya_audio_context": {
    "detection_focus": "AUDIO_MANIPULATION",
    "primary_threat": { "type": "AUDIO_SPLICING", "description": "..." }
  },
  "detection_note": "In the Kenyan context, edited real audio is a far more common threat than AI-generated audio (as of 2026)."
}
```

#### Text Analysis

```
POST /api/analyze/text
Content-Type: application/json
Body: { "text": "content to analyze" }
```

```json
{
  "risk_score": 82,
  "is_authentic": false,
  "confidence": 0.91,
  "findings": ["AI Result: FAKE"]
}
```

#### WhatsApp Forward Analysis 🇰🇪

```
POST /api/analyze/forward
Content-Type: application/json
Body: { "text": "forwarded message text" }
```

```json
{
  "risk_score": 78.5,
  "verdict": "LIKELY_MISINFORMATION",
  "confidence": 0.82,
  "findings": [
    "Forward pattern score: 60%",
    "AI text analysis: 90%",
    "Hoax templates matched: 1",
    "🔍 Matched hoax: Safaricom Hoax"
  ],
  "kenya_warnings": [
    { "type": "MISINFORMATION", "severity": "HIGH", "warning": "...", "action": "Verify with PesaCheck.org" }
  ],
  "forward_analysis": {
    "hoax_matches": [{ "category": "Safaricom Hoax", "debunk": "..." }],
    "swahili_clickbait": ["usisadiki"],
    "fact_check_resources": [
      { "name": "PesaCheck", "url": "https://pesacheck.org/" }
    ]
  }
}
```

#### Document / Screenshot Analysis 🇰🇪

```
POST /api/analyze/document
Content-Type: multipart/form-data
Body: file (image of document or news screenshot)
```

```json
{
  "risk_score": 72.3,
  "verdict": "LIKELY_FORGED",
  "confidence": 0.85,
  "findings": ["📄 Document type: KRA PIN Certificate"],
  "kenya_warnings": [
    { "type": "DOCUMENT_FORGERY", "severity": "CRITICAL", "warning": "...", "action": "Verify at: https://itax.kra.go.ke" }
  ],
  "document_analysis": {
    "is_document": true,
    "document_type": "kra_pin",
    "verdict": "LIKELY_FORGED",
    "kenya_context": { "verify_at": "https://itax.kra.go.ke/KRA-Portal/", "report_to": "DCI: reportcrime@dci.go.ke" }
  },
  "screenshot_analysis": {
    "is_news_screenshot": false
  }
}
```

#### JWT Login

```
POST /api/login
Content-Type: application/json
Body: { "username": "admin", "password": "password" }
```

Returns `{ "access_token": "..." }`.

### Auth Server (`app.py` — port 7860)

All auth and profile routes are served from the **unified backend** on port 7860.

| Route | Method | Description |
|---|---|---|
| `/login` | GET, POST | Login page / form handler |
| `/signup` | GET, POST | Signup page / form handler |
| `/auth/google` | GET | Initiate Google OAuth |
| `/auth/google/callback` | GET | Google OAuth callback |
| `/auth/github` | GET | Initiate GitHub OAuth |
| `/auth/github/callback` | GET | GitHub OAuth callback |
| `/logout` | GET | Clear session |
| `/api/me` | GET | Return current user JSON |
| `/api/profile` | GET, PUT | Get or update user profile |
| `/api/profile/password` | PUT | Change password |
| `/api/profile/picture` | POST | Upload profile picture |
| `/api/history` | GET | Get user's scan history (filterable by type) |
| `/api/history/:id` | DELETE | Delete a history entry |
| `/api/users` | GET | List all registered users |

---

## Docker & Deployment

### Quick Start with Docker Compose

```bash
# Production (backend only)
cp .env.example .env    # Edit .env with your secrets
docker compose up -d

# Development (backend + frontend)
docker compose --profile dev up -d
```

### Production Dockerfile

Multi-stage build with non-root user, tini init, and Gunicorn:

```bash
docker build -t safeye .
docker run -p 7860:7860 --env-file .env safeye
```

Key details:
- **Python 3.11 slim** multi-stage build (small final image)
- CPU-only PyTorch pre-installed (avoids 5 GB+ GPU wheels)
- Non-root `safeye` user for security
- `tini` init system for proper signal handling
- Gunicorn with production config (`gunicorn.conf.py`)
- Health check on `/api/health`
- Tesseract OCR installed for document analysis

### Production Gunicorn

```bash
# Run directly with production config
gunicorn -c gunicorn.conf.py app:app
```

Features: worker recycling, request timeouts, preloaded app, structured logging.

### Environment-based Configuration

| Variable | Default | Description |
|---|---|---|
| `FLASK_ENV` | `production` | `development`, `production`, or `testing` |
| `FLASK_SECRET_KEY` | auto-generated | **Set in production!** Random 64-char string |
| `JWT_SECRET_KEY` | auto-generated | **Set in production!** Random 64-char string |
| `PORT` | `7860` | Server port |
| `WORKERS` | `2` | Gunicorn worker count |
| `RATELIMIT_ENABLED` | `true` | Enable API rate limiting |
| `RATELIMIT_ANALYSIS` | `30/minute` | Rate limit for analysis endpoints |
| `LOG_FORMAT` | `json` (prod) | `json` for production, `text` for development |

---

## Testing

```bash
# Run all tests with pytest
pytest tests/ -v

# Run integration tests only
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=backend --cov=app -v

# Run specific test class
pytest tests/test_api.py::TestHealthEndpoint -v
```

Test files:
- `tests/conftest.py` — Shared fixtures (Flask client, test files, detectors)
- `tests/test_api.py` — Integration tests (API routes, validation, security headers)
- `tests/test_image.py` — Image detector unit tests
- `tests/test_audio.py` — Audio detector unit tests
- `tests/test_text.py` — Text detector unit tests

---

## Demo Script

| Step | Duration | Action |
|---|---|---|
| **The Problem** | 30 sec | "In 2007, 1,500 Kenyans died in PEV. Incitement spread via SMS. In 2027, the weapon is AI." |
| Doctored news screenshot | 2 min | Upload a manipulated Citizen TV screenshot → detect forgery + outlet identification |
| Deepfake image | 1.5 min | Upload a deepfake → high risk score + election context warnings |
| WhatsApp forward | 1.5 min | Paste a real Safaricom hoax forward → matched hoax template + Swahili clickbait detection |
| Manipulated audio | 1 min | Upload edited audio → honest context about splicing vs AI generation |
| Document forgery | 1 min | Upload a forged KRA PIN → OCR + ELA detection + verification links |
| **Why Kenya** | 30 sec | "No existing tool is built for Kenyan news outlets, Swahili misinformation, or our legal framework." |
| **Business model** | 30 sec | B2G (NCIC/IEBC), B2B (newsrooms), B2C (free WhatsApp bot) |
| Q&A | 3 min | |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| **OAuth callback to localhost on phone** | Set `FRONTEND_URL` to your tunnel URL and update Google Cloud Console redirect URIs |
| **Models not downloading** | Ensure Azure env vars are set, or update URLs in `models/download_models.py`. Try `pip install transformers --no-cache-dir`. |
| **ffmpeg not found** | `sudo apt-get install ffmpeg` (Ubuntu) or `brew install ffmpeg` (macOS). |
| **CUDA errors on CPU machine** | `export CUDA_VISIBLE_DEVICES=""` before running. |
| **Port 7860 already in use** | Kill the existing process or change the port in `backend/app.py`. |
| **Large model files missing after clone** | Run `python models/download_models.py` — models (~1.1 GB) are not committed to git. |
| **Import errors in tests** | Ensure all Python deps are installed: `pip install -r requirements.txt`. |
| **`sqlite3.OperationalError: no such column`** | The DB migration runs on startup. Delete `users.db` and restart, or run `python app.py` which auto-migrates. |
| **Phone can't connect on local network** | Open firewall: `sudo firewall-cmd --add-port=3000/tcp --add-port=7860/tcp --permanent && sudo firewall-cmd --reload` |
| **Phone on different network** | Use the tunnel: `./start-tunnel.sh` — see [Mobile Access](#mobile-access) |
| **CORS errors from tunnel URL** | Set `EXTRA_CORS_ORIGINS=https://your-tunnel.loca.lt` or set `FRONTEND_URL` to the tunnel URL |

---

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md) for the full sprint plan. Key upcoming milestones:

- **Kenya-specific model training** — Fine-tune on Kenyan politician faces, Kenyan news screenshots, and Swahili text corpora
- **Swahili text detection** — Multilingual DistilBERT for Swahili/Sheng fake news classification
- **PesaCheck API integration** — Cross-reference detected claims with PesaCheck's verified fact-checks
- **WhatsApp Bot** — Free verification bot for citizens (send media/text → get analysis)
- **NCIC/IEBC dashboard** — Institutional tool for election monitoring at scale
- **Video support** — Frame extraction and temporal analysis for political deepfake videos
- **Persistent database** — Migrate from in-memory / flat-file storage to PostgreSQL
- **wav2vec2 audio inference** — Replace MFCC heuristics with actual AI model detection
- **CI/CD pipeline** — Automated testing and deployment

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

1. Fork the repository and create a branch from `main`
2. Make your changes and add tests where applicable
3. Run the test suite: `pytest tests/ -v`
4. Update documentation to reflect any changes
5. Submit a pull request with a clear description

---

## License

ISC

---

## Acknowledgements

- **Team NIRU** — Built at the NIRU AI Hackathon 2026
- [Hugging Face](https://huggingface.co/) — Pre-trained AI models
- [PyTorch](https://pytorch.org/) & [EfficientNet](https://arxiv.org/abs/1905.11946) — Image detection backbone
- [DeepFace](https://github.com/serengil/deepface) — Face extraction and texture analysis
- [React](https://react.dev/), [Vite](https://vitejs.dev/), [Tailwind CSS](https://tailwindcss.com/) — Frontend stack
- [Flask](https://flask.palletsprojects.com/) & [Authlib](https://authlib.org/) — Backend and OAuth
