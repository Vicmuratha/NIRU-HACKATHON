# SafEye — Technical Roadmap

**Period**: February 9 – February 28, 2026 (20 days)  
**Team**: NIRU  
**Repo**: github.com/Vicmuratha/NIRU-HACKATHON

---

## Current State Assessment

### What's Done

| Area | Status | Notes |
|------|--------|-------|
| React + Tailwind SPA | ✅ | Hero, Analyze panel (image/audio/text), animations |
| Flask auth server (app.py) | ✅ | Login/signup POST routes, Google & GitHub OAuth, session mgmt |
| Login & signup page UX | ✅ | Client validation, password strength meter, toggles, loading states |
| Backend detection API (backend/app.py) | ✅ | Image, audio, text endpoints on port 7860 |
| Image detector | ✅ | EfficientNet-B4 .pth + ELA + metadata + face texture + noise |
| Audio detector | ✅ | MFCC variance + silence ratio heuristics (no AI model inference) |
| Text detector | ✅ | HuggingFace roberta-fake-news pipeline + clickbait heuristics |
| Azure Blob model download | ✅ | Chunked streaming, progress bar, .env config |
| Dockerfile + deploy_package | ✅ | CPU-only torch, gunicorn, health check |
| Vite build pipeline | ✅ | Builds clean, 0 TS errors |
| Detection logging | ✅ | JSONL to data/detection_log.json |

### Gaps Identified

| # | Gap | Severity | Notes |
|---|-----|----------|-------|
| G1 | **Models not fine-tuned** — generic EfficientNet-B4, heuristic audio, generic roberta text | **Critical** | Core deliverable |
| G2 | **No persistent database** — in-memory `users_db = {}`, detection logs in flat JSON | High | Lost on restart |
| G3 | **Tests broken** — import non-existent `detectors.*` modules | High | No CI possible |
| G4 | **No rate limiting / file validation** — any file type accepted, no abuse prevention | High | Security risk |
| G5 | **Duplicate code paths** — `logic.py` + `simple_app.py` + `backend/app.py` all do detection | Medium | Maintenance burden |
| G6 | **No video support** | Medium | Feature gap |
| G7 | **No analysis history in UI** — scans are fire-and-forget | Medium | UX gap |
| G8 | **No CI/CD pipeline** | Medium | Dev velocity |
| G9 | **Missing text model blobs in Azure** — `model.safetensors` + `spm.model` | Medium | Fresh deploy breaks |
| G10 | **Auth not wired to React** — Flask and Vite on separate ports | Medium | Split UX |
| G11 | **No Swahili / Kenya-specific text model** | Low | Hackathon context |
| G12 | **Confidence scoring bug** — values > 1.0 in detection_log | Low | Incorrect output |

---

## Sprint 1 — AI Model Fine-Tuning & Data Pipeline

**Feb 9 – Feb 14 (6 days)**

> Primary goal: move all three detectors from generic pretrained weights to fine-tuned, validated models.

### Milestone 1.1 — Collect & Prepare Training Data (Feb 9–10)

| Task | Details |
|------|---------|
| Curate image dataset | FaceForensics++ (c23), DFDC subset, 140k Real-vs-Fake Faces. Target ≥ 10 K real + 10 K fake. Split 80/10/10. |
| Curate audio dataset | ASVspoof 2019/2021 LA partition + In-The-Wild voice-clone samples. Target ≥ 5 K real + 5 K fake clips. |
| Curate text dataset | LIAR, FakeNewsNet, Kenya news corpus (Nation, Standard Digital, fact-check.ke). Target ≥ 8 K real + 8 K fake articles. |
| Build preprocessing scripts | `scripts/prepare_image_data.py` — resize 380×380, augment (flip, colour jitter, JPEG artefacts). `scripts/prepare_audio_data.py` — resample 16 kHz mono, 4-sec chunks. `scripts/prepare_text_data.py` — clean, truncate 512 tokens. |
| Upload datasets to Azure | Push to `safeye` storage `datasets` container for reproducibility. |

**Deliverable**: `data/splits/{train,val,test}/{image,audio,text}/` ready on disk.

---

### Milestone 1.2 — Fine-Tune Image Model (Feb 10–12)

| Task | Details |
|------|---------|
| Fine-tune EfficientNet-B4 | Unfreeze last 3 blocks + classifier head. LR 1e-4, cosine annealing, 15 epochs. Script: `scripts/train_image.py`. |
| Evaluate on test split | Target: **>92 % accuracy, >0.95 AUC** on FF++ c23. Log confusion matrix. |
| Export & upload | Best checkpoint → `models/bestdeepfake/best_deepfake_detector.pth` → Azure `ai-models/`. |
| Quantize for CPU | `torch.quantization.quantize_dynamic` on Linear layers → ~12 MB, 2× faster inference. |

**Deliverable**: Fine-tuned image model with measured accuracy in `docs/benchmarks.md`.

---

### Milestone 1.3 — Fine-Tune Audio Model (Feb 12–13)

| Task | Details |
|------|---------|
| Replace heuristic audio detector | Fine-tune `facebook/wav2vec2-base` (already in Azure) on ASVspoof binary classification. |
| Create `scripts/train_audio.py` | Classification head on wav2vec2 pooled output. LR 3e-5, 10 epochs. |
| Evaluate | Target: **>90 % accuracy, >0.93 AUC** on ASVspoof eval set. |
| Update `backend/app.py` `UltraAudioDetector` | Replace MFCC heurists with proper model inference using fine-tuned wav2vec2. |
| Upload | Update `audio_model/` blobs in Azure. |

**Deliverable**: Audio detector using actual AI model inference, not heuristics.

---

### Milestone 1.4 — Fine-Tune Text Model (Feb 13–14)

| Task | Details |
|------|---------|
| Fine-tune roberta | Add Kenya news samples. Fine-tune `hamzab/roberta-fake-news-classification` for 5 more epochs on combined dataset. |
| Add Swahili support | Secondary lightweight classifier (multilingual DistilBERT) on Swahili news for basic coverage. |
| Evaluate | Target: **>88 % accuracy** on combined English + Swahili test set. |
| Upload missing blobs | Push `model.safetensors` (704 MB) and `spm.model` (2.4 MB) to Azure `ai-models/text_model/`. |
| Fix confidence scoring | Clamp all risk scores to [0, 1] range — fix `confidence > 1.0` bug in analysis pipeline. |

**Deliverable**: Text model handles English + basic Swahili. All model files in Azure.

---

## Sprint 2 — System Hardening & Missing Features

**Feb 15 – Feb 21 (7 days)**

### Milestone 2.1 — Database & Auth Integration (Feb 15–16)

| Task | Details |
|------|---------|
| Add SQLite (dev) / PostgreSQL (prod) via SQLAlchemy | `User` model (id, name, email, password_hash, created_at). `ScanLog` model (id, user_id, media_type, risk_score, verdict, timestamp). |
| Replace in-memory `users_db` | Both `app.py` and `backend/app.py` use DB queries. `werkzeug.security` for password hashing. |
| Persist detection logs to DB | Replace flat JSON writes with `ScanLog` inserts. Keep JSON export endpoint for backwards compat. |
| Unify auth flow | Proxy `/login`, `/signup` through Vite dev server OR serve React `dist/` from Flask in production (single origin cookies). |

**Deliverable**: Users and scan logs persist across restarts.

---

### Milestone 2.2 — Consolidate Code & Fix Tests (Feb 17–18)

| Task | Details |
|------|---------|
| Remove duplicates | Delete `logic.py` and `backend/simple_app.py`. Single detection source = `backend/app.py`. Update Dockerfile CMD. |
| Rewrite tests | Fix `tests/test_image.py` etc. to import `UltraImageDetector`, `UltraAudioDetector`, `UltraTextDetector` from `backend.app`. Add Flask test-client integration tests. |
| Pytest + coverage | Replace unittest. Add `pytest.ini`, `conftest.py`. Target: **>80 % coverage** on detection logic. |
| GitHub Actions CI | `.github/workflows/ci.yml` — ruff lint, pyright type-check, pytest, `npx vite build`. |

**Deliverable**: `pytest` passes. CI green on push.

---

### Milestone 2.3 — Security Hardening (Feb 18–19)

| Task | Details |
|------|---------|
| Rate limiting (`flask-limiter`) | 20 req/min per IP on `/api/analyze/*`, 5 req/min on `/login` POST. |
| File validation | Whitelist MIME types (jpeg, png, wav, mp3…). Reject others with 415. Max 10 MB image, 25 MB audio. |
| CSRF protection | `flask-wtf` CSRF tokens on auth forms. |
| Sanitize uploads | Strip EXIF GPS, auto-delete uploads after analysis. |

**Deliverable**: Rate-limited, MIME-validated, CSRF-protected API.

---

### Milestone 2.4 — Video Support (Feb 19–20)

| Task | Details |
|------|---------|
| Keyframe extraction | Use `ffmpeg` to extract 5 evenly-spaced keyframes from uploaded video (max 30 s, 25 MB). |
| Per-frame analysis | Run `UltraImageDetector.analyze_image()` on each frame. Overall risk = max across frames. |
| Endpoint `/api/analyze/video` | Accept `.mp4`, `.avi`, `.mov`. Return per-frame breakdown + overall verdict. |
| React "Video" tab | Add to AnalysisPanel alongside Image/Audio/Text. Show timeline of frame results. |

**Deliverable**: Video upload → keyframe extraction → per-frame deepfake analysis.

---

### Milestone 2.5 — Analysis History Dashboard (Feb 20–21)

| Task | Details |
|------|---------|
| `/api/history` endpoint | Paginated scan history for current user from DB. |
| History view in React | New tab: past scans list (date, type badge, risk score, verdict). Expandable findings. |
| Scan statistics | Total scans, threats detected, per-type breakdown. Display in user dashboard area. |

**Deliverable**: Users can review past scans and overall stats.

---

## Sprint 3 — Polish, Deploy & Demo

**Feb 22 – Feb 28 (7 days)**

### Milestone 3.1 — Production Deployment (Feb 22–23)

| Task | Details |
|------|---------|
| Unified Dockerfile | Single container: build React (`npx vite build`) → serve `dist/` from Flask static. One port. |
| Deploy to Azure App Service | Push Docker image to ACR. `az webapp create`. Set env vars for Azure Blob, DB. |
| Custom domain + HTTPS | Configure managed SSL certificate. |
| Validate cold start | Confirm model auto-download works in container. Add `/api/health` readiness probe. |

**Deliverable**: Live at production URL with HTTPS.

---

### Milestone 3.2 — API Documentation (Feb 23–24)

| Task | Details |
|------|---------|
| Swagger/OpenAPI via Flask-RESTX or Flasgger | Interactive docs at `/api/docs`. |
| Document all endpoints | Health, image, audio, text, video, history, auth. Request/response examples. |
| Postman collection | Export for easy testing by judges. |

**Deliverable**: Interactive API docs at `/api/docs`.

---

### Milestone 3.3 — Monitoring & Observability (Feb 24–25)

| Task | Details |
|------|---------|
| Structured logging (`structlog`) | JSON-formatted logs. Every scan logs duration, result, user_id. |
| Azure Application Insights | Track request latency, error rates, model load times. |
| Enhanced health endpoint | `/api/health` returns model statuses, uptime, scan count, avg response time. |

**Deliverable**: Observable production system.

---

### Milestone 3.4 — Final Model Validation & Benchmarks (Feb 25–26)

| Task | Details |
|------|---------|
| Full benchmark suite | All models vs held-out test sets. Precision, recall, F1, AUC per modality. |
| Real-world cross-validation | Recent Kenya news images, political audio, social media posts. |
| Document results | `docs/benchmarks.md` — tables, confusion matrices, sample predictions. |
| Threshold tuning | Adjust verdict thresholds (AUTHENTIC < 40, REVIEW 40–65, DEEPFAKE > 65) from benchmark data. |

**Deliverable**: Validated accuracy numbers with real test data.

---

### Milestone 3.5 — Demo Preparation & Final Documentation (Feb 26–28)

| Task | Details |
|------|---------|
| Demo script | 10-min walkthrough: signup → upload real image (low risk) → deepfake (high risk) → audio → text → video → history → API docs. |
| Test assets | Curate 2 real + 2 fake samples per modality with reliable, reproducible results. |
| Demo video | Screen capture of full flow as backup. |
| Update README | Final setup instructions, architecture diagram, feature list, benchmark results. |
| Release tag | `git tag v1.0.0` and push. |

**Deliverable**: Demo-ready system with complete documentation.

---

## Timeline Summary

```
Feb 2026

Week 1 (Feb 9–14)           Week 2 (Feb 15–21)            Week 3 (Feb 22–28)
══════════════════════       ══════════════════════════     ══════════════════════════
 M1.1 Data Collection         M2.1 Database & Auth           M3.1 Production Deploy
 M1.2 Image Fine-Tune         M2.2 Code Cleanup & CI         M3.2 API Documentation
 M1.3 Audio Fine-Tune         M2.3 Security Hardening        M3.3 Monitoring
 M1.4 Text Fine-Tune          M2.4 Video Support             M3.4 Final Benchmarks
                               M2.5 History Dashboard         M3.5 Demo Prep
```

---

## Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Fine-tuning doesn't hit accuracy targets | High | Medium | Fall back to larger pretrained models (ViT-Large, Wav2Vec2-Large). Use ensemble voting. |
| Azure cold start too slow (models ~1.1 GB) | High | High | Pre-warm with scheduled pings. Use quantized models. Consider min replicas = 1. |
| Training data licensing issues | Medium | Low | Use only open datasets (FF++ CC BY-NC, ASVspoof open, LIAR public). |
| Swahili text data scarcity | Medium | High | Start English-only. Add Swahili as best-effort with multilingual DistilBERT zero-shot. |
| Team bandwidth (20 days, many milestones) | High | Medium | Prioritise Sprint 1 (models) above all. Sprint 2/3 items can be cut. |

---

## Priority Stack (if time runs short)

1. **Must Have**: Fine-tuned image model + audio model with real inference + working demo
2. **Should Have**: Database, video support, history dashboard, CI
3. **Nice to Have**: Swahili support, Swagger docs, monitoring, rate limiting

---

*Last updated: February 8, 2026*