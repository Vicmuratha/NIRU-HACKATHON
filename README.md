# SafEye - AI-Powered Deepfake Detection Platform

[![Development Status](https://img.shields.io/badge/Status-In%20Development-yellow)](https://github.com/Vicmuratha/NIRU-HACKATHON)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/React-18.3+-61DAFB.svg?logo=react)](https://reactjs.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000.svg?logo=flask)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-ISC-green.svg)](LICENSE)

> **⚠️ STILL IN DEVELOPMENT** - Some features may be incomplete or experimental

SafEye is a comprehensive multi-modal AI-powered platform designed to detect deepfakes and misinformation in images, audio, and text. Built for real-time analysis with high accuracy, it provides a modern React frontend, a robust Flask-based detection API, and an OAuth-based authentication server.

## 📋 Table of Contents

- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start-development)
- [Project Structure](#-whats-in-this-repo)
- [API Endpoints](#-api-endpoints-detection-service)
- [Configuration](#️-configuration)
- [Testing](#-tests)
- [Docker](#-docker)
- [API Documentation](#-api-documentation)
- [Troubleshooting](#-support--troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Features

- **Multi-Modal Detection**: Analyze images, audio, and text content for deepfakes and manipulation
- **Real-Time Analysis**: Get instant results with detailed confidence scores and risk assessments
- **Modern React UI**: Intuitive drag-and-drop interface built with React, Vite, and Tailwind CSS
- **RESTful API**: Easy integration with Flask-based backend API
- **OAuth Authentication**: Secure login with Google and GitHub integration
- **Advanced AI Models**: State-of-the-art deep learning models for accurate detection
- **Comprehensive Analytics**: Track detection statistics and analysis history

## 📦 Prerequisites

Before you begin, ensure you have the following installed:

- **Python**: 3.10 or higher
- **Node.js**: 18 or higher
- **npm**: 8 or higher
- **ffmpeg**: Required for audio processing
  - Ubuntu/Debian: `sudo apt-get install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org)
- **Git**: For cloning the repository

## 📁 What's in this repo

```
NIRU-HACKATHON/
├── src/                   # React + Vite frontend UI
├── backend/               # Flask detection API
│   └── app.py            # Image, audio, and text analyzers
├── app.py                 # OAuth authentication server
├── models/                # AI model assets and downloader
├── static/                # Static assets (CSS, images)
├── templates/             # Authentication page templates
├── tests/                 # Unit tests
├── uploads/               # Temporary file uploads
└── docs/                  # Additional documentation
```

**Key Components:**
- **Frontend**: Modern React + Vite UI with Tailwind CSS in [src](src)
- **Detection API**: Flask app with specialized analyzers in [backend/app.py](backend/app.py)
- **Auth Server**: OAuth 2.0 login server in [app.py](app.py)
- **AI Models**: Pre-trained models for deepfake detection under [models](models) (~1.1GB total)


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
For Azure Blob Storage, set these environment variables before running the script:

```
AZURE_STORAGE_CONNECTION_STRING=...
AZURE_STORAGE_CONTAINER=your-container
AZURE_AUDIO_BLOBS=audio_model/config.json,audio_model/model.safetensors,audio_model/preprocessor_config.json
AZURE_TEXT_BLOBS=text_model/config.json,text_model/model.safetensors,text_model/tokenizer_config.json,text_model/added_tokens.json,text_model/special_tokens_map.json,text_model/spm.model
AZURE_IMAGE_BLOBS=image_model/config.json,image_model/model.safetensors,image_model/preprocessor_config.json
# Optional: single blob (e.g., a zip/tar.gz archive)
AZURE_IMAGE_BLOB=image_model.zip
```

If you prefer public Blob URLs, set:

```
AZURE_BLOB_BASE_URL=https://<account>.blob.core.windows.net
AZURE_SAS_TOKEN=sv=...  # optional
```

To download models automatically when the API starts (recommended for Azure App Service), set:

```
DOWNLOAD_MODELS_ON_STARTUP=true
IMAGE_MODEL_DIR=/code/models/image_model
TEXT_MODEL_DIR=/code/models/text_model
```

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

### Need Help?

- Check the [documentation](docs/) for more detailed guides
- Review the [roadmap](docs/ROADMAP.md) for planned features
- Open an [issue](https://github.com/Vicmuratha/NIRU-HACKATHON/issues) for bug reports or feature requests

## 🤝 Contributing

We welcome contributions to SafEye! Here's how you can help:

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** and add tests if applicable
3. **Ensure the test suite passes**: `python -m unittest discover tests/`
4. **Update documentation** to reflect any changes
5. **Submit a pull request** with a clear description of your changes

### Development Guidelines

- Follow existing code style and conventions
- Write clear commit messages
- Add tests for new features
- Update the README if you change functionality

## 📄 License

This project is licensed under the ISC License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for hackathons and educational purposes
- Powered by state-of-the-art AI models from Hugging Face
- Frontend built with React, Vite, and Tailwind CSS
- Backend powered by Flask and PyTorch

---

**Note**: SafEye is still in development. Some features may be incomplete or experimental. Use responsibly and always verify critical detections manually.
