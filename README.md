# SafEye — AI-Powered Deepfake Detection Platform

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/React-18.3-61DAFB.svg?logo=react)](https://reactjs.org/)
[![Flask](https://img.shields.io/badge/Flask-latest-000000.svg?logo=flask)](https://flask.palletsprojects.com/)
[![Vite](https://img.shields.io/badge/Vite-7.3-646CFF.svg?logo=vite)](https://vitejs.dev/)
[![License](https://img.shields.io/badge/License-ISC-green.svg)](LICENSE)

SafEye is a multi-modal AI platform that detects deepfakes and misinformation across **images**, **audio**, and **text**. It pairs a React + Vite frontend with two Flask backends — a detection API and an OAuth authentication server — and ships with pre-trained models, a Docker deployment package, and Azure Blob integration for model hosting.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
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
| **Audio Detection** | MFCC variance and silence-ratio heuristics via librosa (wav2vec2 fine-tuning planned — see Roadmap). |
| **Text Detection** | HuggingFace `hamzab/roberta-fake-news-classification` pipeline with clickbait keyword boosting. Local model files used when present. |
| **React UI** | Single-page app built with React 18, Vite 7, Tailwind CSS 3, Framer Motion, and Lucide icons. Drag-and-drop file upload with animated risk scores and Kenya-specific threat warnings. |
| **OAuth Authentication** | Standalone Flask server with Google and GitHub OAuth (via Authlib), SQLite user store, session management, and login/signup HTML templates. |
| **JWT Auth (Detection API)** | `flask-jwt-extended` token-based auth on the detection backend for programmatic access. |
| **Azure Blob Model Download** | Chunked streaming download from Azure Blob Storage with progress bar, SAS token support, and auto-download on startup for Azure App Service. |
| **Docker** | CPU-optimised Dockerfile with pre-installed `torch` (CPU wheel) and `tensorflow-cpu`, Gunicorn, health check, and 600 s model-load timeout. |
| **Kenya-Focused Warnings** | Election manipulation alerts for high-risk face deepfakes, M-Pesa fraud warnings for voice cloning detections. |

---

## Architecture

```
┌─────────────────────┐        /api proxy         ┌──────────────────────────┐
│   React + Vite UI   │ ──────────────────────────▶│  Detection API (Flask)   │
│   localhost:3000     │                            │  localhost:7860           │
└─────────────────────┘                            │                          │
                                                   │  /api/analyze/image      │
┌─────────────────────┐                            │  /api/analyze/audio      │
│  Auth Server (Flask) │                            │  /api/analyze/text       │
│  localhost:5000      │                            │  /api/health             │
│                      │                            └──────────┬───────────────┘
│  /login  /signup     │                                       │
│  /auth/google        │                              ┌────────▼────────┐
│  /auth/github        │                              │  models/         │
│  /api/me             │                              │  ~1.1 GB total   │
└─────────────────────┘                              └─────────────────┘
```

The Vite dev server proxies all `/api` requests to **port 7860** (the detection API) as configured in `vite.config.ts`.

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
├── app.py                     # OAuth authentication server (Flask, SQLite, Authlib)
├── backend/
│   ├── app.py                 # Main detection API (Flask, port 7860)
│   └── simple_app.py          # Lightweight detection API (uses logic.py)
├── logic.py                   # Shared HuggingFace pipeline wrappers (image/audio/text)
├── models/
│   ├── download_models.py     # Azure Blob / URL model downloader script
│   ├── audio_model/           # Wav2Vec2 config + weights (~361 MB)
│   ├── image_model/           # EfficientNet-B4 checkpoint (~47 MB)
│   ├── bestdeepfake/          # Alternate .pth checkpoint
│   └── text_model/            # RoBERTa fake-news weights + tokenizer (~704 MB)
├── src/
│   ├── App.tsx                # Main React component (hero, analysis panel)
│   ├── main.tsx               # Vite entry point
│   └── styles.css             # Tailwind + custom animations
├── static/css/                # CSS for auth templates
├── templates/                 # login.html, signup.html (Jinja2)
├── tests/
│   ├── test_image.py
│   ├── test_audio.py
│   └── test_text.py
├── data/
│   └── detection_log.json     # JSONL detection history
├── uploads/                   # Temporary file uploads (gitignored)
├── deploy_package/            # Production Docker build (Gunicorn on port 8000)
│   ├── Dockerfile
│   ├── app.py
│   └── requirements.txt
├── docs/
│   ├── README.md
│   └── ROADMAP.md
├── Dockerfile                 # Dev/CI Docker build (Gunicorn on port 8000)
├── requirements.txt           # Python dependencies
├── package.json               # Node.js dependencies
├── vite.config.ts             # Vite config (proxy /api → :7860)
├── tailwind.config.js
├── postcss.config.js
├── tsconfig.json
└── index.html                 # Vite HTML entry
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

### 4. Start the detection API

```bash
python backend/app.py
```

Runs on **http://localhost:7860**. On first request the AI models are lazy-loaded into memory.

### 5. (Optional) Start the auth server

```bash
python app.py
```

Runs on **http://localhost:5000**. Requires Google/GitHub OAuth keys in `.env` for social login.

### 6. Start the frontend

```bash
npm run dev
```

Runs on **http://localhost:3000**. All `/api` calls are proxied to `localhost:7860`.

---

## Environment Variables

Create a `.env` file in the project root:

```bash
# ── Auth server ──
FLASK_SECRET_KEY=your-secret-key
FRONTEND_URL=http://localhost:3000

GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
GITHUB_CLIENT_ID=...
GITHUB_CLIENT_SECRET=...

# ── Detection API ──
JWT_SECRET_KEY=your-jwt-secret
FLASK_SECRET_KEY=super_secret_key

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
    { "type": "MPESA_FRAUD", "severity": "HIGH", "warning": "Voice cloning risk", "action": "Do not authorize transactions via voice" }
  ]
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

#### JWT Login

```
POST /api/login
Content-Type: application/json
Body: { "username": "admin", "password": "password" }
```

Returns `{ "access_token": "..." }`.

### Auth Server (`app.py` — port 5000)

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

---

## Docker & Deployment

### Root Dockerfile

Builds a CPU-optimised container running `backend.simple_app:app` via Gunicorn on port 8000:

```bash
docker build -t safeye .
docker run -p 8000:8000 safeye
```

Key details:
- Python 3.9 slim base with ffmpeg, libsndfile
- CPU-only `torch` and `tensorflow-cpu` pre-installed (avoids 5 GB+ GPU wheels)
- 600 s Gunicorn timeout for model loading
- 50 MB max upload size

### deploy_package/

A self-contained deployment variant with its own `Dockerfile`, `app.py`, and `requirements.txt` for Azure App Service or similar PaaS. Runs on port 8000 with Gunicorn.

Set `DOWNLOAD_MODELS_ON_STARTUP=true` to auto-download models when the container starts.

---

## Testing

```bash
# Run all tests
python -m unittest discover tests/

# Run a specific test
python -m unittest tests/test_image.py

# Verbose output
python -m unittest -v tests/
```

Test files: `tests/test_image.py`, `tests/test_audio.py`, `tests/test_text.py`.

---

## Demo Script

| Step | Duration | Action |
|---|---|---|
| Dashboard walkthrough | 2 min | Show hero section, navigation, analysis tabs |
| Authentic image | 1 min | Upload real photo → low risk score |
| Manipulated image | 2 min | Upload deepfake → high risk score + detailed findings |
| Suspicious audio | 2 min | Upload cloned voice → Kenya fraud warning |
| Misinformation text | 1 min | Paste fake article → AI detection result |
| Analytics & impact | 1 min | Show detection statistics |
| Architecture deep-dive | 2 min | Explain model pipeline and weighted scoring |
| Q&A | 3 min | |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| **Models not downloading** | Ensure Azure env vars are set, or update URLs in `models/download_models.py`. Try `pip install transformers --no-cache-dir`. |
| **ffmpeg not found** | `sudo apt-get install ffmpeg` (Ubuntu) or `brew install ffmpeg` (macOS). |
| **CUDA errors on CPU machine** | `export CUDA_VISIBLE_DEVICES=""` before running. |
| **Port 7860 already in use** | Kill the existing process or change the port in `backend/app.py`. |
| **Large model files missing after clone** | Run `python models/download_models.py` — models (~1.1 GB) are not committed to git. |
| **Import errors in tests** | Ensure all Python deps are installed: `pip install -r requirements.txt`. |

---

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md) for the full sprint plan. Key upcoming milestones:

- **Fine-tuned models** — Replace generic weights with domain-specific training on FaceForensics++, ASVspoof, and Kenya news corpora
- **wav2vec2 audio inference** — Replace MFCC heuristics with actual AI model detection
- **Persistent database** — Migrate from in-memory / flat-file storage to SQLite/PostgreSQL
- **Video support** — Frame extraction and temporal analysis
- **Analysis history UI** — View past scan results in the React frontend
- **Swahili text detection** — Multilingual DistilBERT for Swahili news
- **CI/CD pipeline** — Automated testing and deployment
- **Rate limiting & file validation** — Abuse prevention and security hardening

---

## Contributing

1. Fork the repository and create a branch from `main`
2. Make your changes and add tests where applicable
3. Run the test suite: `python -m unittest discover tests/`
4. Update documentation to reflect any changes
5. Submit a pull request with a clear description

---

## License

ISC — see [LICENSE](LICENSE).

---

## Acknowledgments

- **Team NIRU** — Hackathon project
- [Hugging Face](https://huggingface.co/) — Pre-trained AI models
- [PyTorch](https://pytorch.org/) & [EfficientNet](https://arxiv.org/abs/1905.11946) — Image detection backbone
- [DeepFace](https://github.com/serengil/deepface) — Face extraction and texture analysis
- [React](https://react.dev/), [Vite](https://vitejs.dev/), [Tailwind CSS](https://tailwindcss.com/) — Frontend stack
- [Flask](https://flask.palletsprojects.com/) & [Authlib](https://authlib.org/) — Backend and OAuth
