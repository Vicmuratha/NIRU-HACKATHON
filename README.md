# SafEye - AI-Powered Deepfake Detection Platform

> STILL IN DEVELOPMENT

SafEye is a multi-modal deepfake and misinformation detection platform for images, audio, and text. It includes a React frontend, a detection API, and an OAuth-based demo auth server.

## ✅ What’s in this repo

- **Frontend**: React + Vite UI in [src](src)
- **Detection API**: Flask app with image, audio, and text analyzers in [backend/app.py](backend/app.py)
- **Auth demo**: OAuth login server in [app.py](app.py)
- **Models**: Local model assets under [models](models)

## 🚀 Quick start (development)

### 1) Install dependencies

```bash
pip install -r requirements.txt
npm install
```

### 2) Download model assets

```bash
python models/download_models.py
```

If your model URLs are not set, update them inside [models/download_models.py](models/download_models.py).

### 3) Run the detection API

```bash
python backend/app.py
```

This starts the detection API on http://localhost:7860.

### 4) Run the auth demo server (optional)

```bash
python app.py
```

This starts the auth server on http://localhost:5000.

### 5) Run the frontend

```bash
npm run dev
```

The frontend runs on http://localhost:3000 and proxies `/api` to http://localhost:5000 as configured in [vite.config.ts](vite.config.ts). If you want the frontend to call the detection API on port 7860, update the proxy target in [vite.config.ts](vite.config.ts) or run the detection API on port 5000.

## 🔌 API endpoints (detection service)

From [backend/app.py](backend/app.py):

- `GET /api/health`
- `POST /api/analyze/image` (multipart file upload)
- `POST /api/analyze/audio` (multipart file upload)
- `POST /api/analyze/text` (JSON body with `text`)
- `GET /api/analytics`
- `GET /api/test-model`

## 🔐 Auth demo routes (optional server)

From [app.py](app.py):

- `GET /login`, `GET /signup`
- `GET /auth/google`, `GET /auth/github`
- `GET /logout`
- `GET /api/me`

## ⚙️ Configuration

Create a `.env` file in the repo root and set any of the following as needed:

```
FLASK_SECRET_KEY=your-secret-key
FRONTEND_URL=http://localhost:3000
JWT_SECRET_KEY=your-jwt-secret

GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret

GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret
```

## 🧪 Tests

Test files live in [tests](tests). You can run them with your preferred Python test runner.

## 🐳 Docker

The root [Dockerfile](Dockerfile) runs `app:app` via Gunicorn (auth demo server). If you want a container for the detection API, adjust the command to point at `backend.app:app` or run [backend/app.py](backend/app.py) directly.

## 📦 Dependencies

- Python dependencies are listed in [requirements.txt](requirements.txt)
- Frontend dependencies are listed in [package.json](package.json)

## 📁 Project layout

```
NIRU-HACKATHON/
├── app.py                 # OAuth demo server
├── backend/               # Detection API
├── models/                # Model assets + downloader
├── src/                   # React UI
├── static/                # Static assets (CSS)
├── templates/             # Auth pages
├── tests/                 # Unit tests
├── uploads/               # Temporary uploads
└── data/                  # Detection logs
```
    redis_client.setex(
        f"analysis:{file_hash}",
        3600,  # 1 hour TTL
        json.dumps(result)
    )
```

### Async Processing with Celery

```python
from celery import Celery

celery = Celery('safeye', broker='redis://localhost:6379/0')

@celery.task
def analyze_image_async(filepath):
    return image_detector.analyze_image(filepath)
```

## 🛡️ Security Best Practices

```python
# Add rate limiting
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

# Add file validation
def validate_file(file):
    # Check file size
    if file.content_length > app.config['MAX_CONTENT_LENGTH']:
        raise ValueError("File too large")

    # Check file type
    allowed_types = {'image/jpeg', 'image/png', 'audio/mpeg', 'audio/wav'}
    if file.content_type not in allowed_types:
        raise ValueError("Invalid file type")

    return True

# Add CORS security
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})
```

## 📚 API Documentation

### Endpoints

#### Health Check

- **GET** `/api/health`
- **Response**: `{"status": "healthy"}`

#### Image Analysis

- **POST** `/api/analyze/image`
- **Content-Type**: `multipart/form-data`
- **Body**: `file` (image file)
- **Response**:

```json
{
  "risk_score": 23.5,
  "is_authentic": true,
  "confidence": 0.92,
  "findings": ["Natural compression patterns detected"],
  "details": {
    "ela_score": 8.2,
    "face_verification": "PASSED",
    "metadata_integrity": "INTACT"
  }
}
```

#### Audio Analysis

- **POST** `/api/analyze/audio`
- **Content-Type**: `multipart/form-data`
- **Body**: `file` (audio file)
- **Response**:

```json
{
  "risk_score": 67.8,
  "is_authentic": false,
  "confidence": 0.88,
  "findings": ["AI voice generation artifacts detected"],
  "details": {
    "spoofing_score": 78.9,
    "spectral_analysis": "SYNTHETIC",
    "pitch_consistency": "ABNORMAL"
  }
}
```

#### Text Analysis

- **POST** `/api/analyze/text`
- **Content-Type**: `application/json`
- **Body**: `{"text": "content to analyze"}`
- **Response**:

```json
{
  "risk_score": 45.2,
  "is_authentic": true,
  "confidence": 0.85,
  "findings": ["Factual claims structure detected"],
  "details": {
    "claim_verification": "VERIFIED",
    "bias_score": 15.2,
    "credibility": "HIGH"
  }
}
```

## 🧪 Running Tests

```bash
# Run all tests
python -m unittest discover tests/

# Run specific test file
python -m unittest tests/test_image.py

# Run with verbose output
python -m unittest -v tests/
```

## 🎯 Demo Script

- Show dashboard (2 min)
- Upload authentic image → low risk score (1 min)
- Upload manipulated image → high risk score + analysis (2 min)
- Analyze suspicious audio clip (2 min)
- Check misinformation text (1 min)
- Show statistics and impact metrics (1 min)
- Explain technical architecture (2 min)
- Q&A (3 min)

## 📞 Support & Troubleshooting

### Common Issues

#### Issue: Models not downloading

**Solution**: Install with specific versions

```bash
pip install transformers==4.35.0 --no-cache-dir
```

#### Issue: ffmpeg not found

**Ubuntu**:

```bash
sudo apt-get install ffmpeg
```

**Mac**:

```bash
brew install ffmpeg
```

#### Issue: CUDA errors

**Solution**: Use CPU

```bash
export CUDA_VISIBLE_DEVICES=""
```
