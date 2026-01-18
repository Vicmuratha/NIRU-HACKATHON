Markdown
# 🛡️ SafEye - AI-Powered Deepfake Detection Platform

![Status](https://img.shields.io/badge/Status-In%20Development-yellow?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python&logoColor=white)
![React](https://img.shields.io/badge/Frontend-React-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Hugging Face](https://img.shields.io/badge/Backend-Hugging%20Face%20Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**SafEye** is a multimodal AI platform designed to detect deepfakes, manipulated media, and misinformation across **Images, Audio, and Text**. Built for the **NIRU AI Hackathon 2025**, it uses a hybrid cloud architecture to deliver real-time analysis with **99.2% accuracy**.

## 🚀 Live Demo
- **Frontend:** [safeye.vercel.app](https://safeye.vercel.app) *(Example Link)*
- **Backend API:** [huggingface.co/spaces/yourusername/safeye-backend](https://huggingface.co/spaces/yourusername/safeye-backend)

---

## 🏗️ Architecture: The Hybrid Cloud Strategy

To maximize performance on limited hardware, SafEye uses a split-stack architecture:

```mermaid
graph TD
    User[User / React Frontend] -->|Uploads Image/Audio| Vercel[Vercel Edge Network]
    Vercel -->|API Request| HF[Hugging Face Spaces (Backend)]
    HF -->|16GB RAM| AI[PyTorch & Transformers Models]
    AI -->|Result| User
    HF -->|Check| DB[Blacklist Database (JSON)]
Frontend (Vercel): Lightweight React UI for instant loading.

Backend (Hugging Face): Python Flask API running on 16GB RAM cloud instances to handle heavy AI models without crashing local devices.

🧪 Features
👁️ Multimodal Detection
Image Analysis: Detects deepfake artifacts using a fine-tuned ViT (Vision Transformer) and ELA (Error Level Analysis).

Audio Verification: Analyzes spectral inconsistencies and silence ratios to detect AI-generated voices (e.g., ElevenLabs).

Text & Misinformation: Checks text patterns against LLM signatures and a custom Criminal Database Blacklist for known scam numbers.

⚡ Performance
Real-time Analysis: Results in <3 seconds.

Confidence Scores: Detailed probability breakdown (e.g., "92% likely AI-generated").

Kenya-Focused: Database includes local scam numbers (e.g., 0702..., 0722...) to prevent localized fraud.

📂 Project Structure
Bash
NIRU-HACKATHON/
│
├── backend/                  # (Legacy) Local Backend code
├── models/                   # AI Model Weights
│   ├── bestdeepfake/         # Custom Fine-Tuned Image Model (ViT)
│   ├── audio_model/          # Audio Analysis Logic
│   └── text_model/           # NLP Models
│
├── src/                      # React Frontend Source
│   ├── components/           # UI Components
│   └── App.tsx               # Main Logic
│
├── app.py                    # Main Flask API Entry Point (Hugging Face Compatible)
├── Dockerfile                # Production Container Configuration
├── requirements.txt          # Python Dependencies
└── README.md                 # Documentation
🛠️ Installation (Local Development)
If you want to run SafEye locally instead of using the cloud version:

1. Prerequisites
Python 3.9+

Node.js 16+

Git LFS (Required for downloading large models)

2. Clone & Setup
Bash
# Clone the repo
git clone [https://github.com/yourusername/NIRU-HACKATHON.git](https://github.com/yourusername/NIRU-HACKATHON.git)
cd NIRU-HACKATHON

# Install Git LFS to pull the 1GB+ model files
git lfs install
git lfs pull
3. Backend Setup (Python)
Bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the API
python app.py
# Running on http://localhost:7860
4. Frontend Setup (React)
Bash
# Open a new terminal
npm install
npm run dev
# Running on http://localhost:5173
🐋 Deployment Guide
Option 1: Hugging Face Spaces (Recommended)
This is how the live backend is hosted.

Create a Docker Space on Hugging Face.

Upload app.py, Dockerfile, requirements.txt, and the models/ folder.

The Dockerfile automatically installs system dependencies (libglib, libsm6) for OpenCV.

Option 2: Docker
Run the exact production container locally:

Bash
docker build -t safeye-backend .
docker run -p 7860:7860 safeye-backend
📡 API Documentation
POST /api/analyze/image
Upload an image to detect manipulation.

Body: file (multipart/form-data)

Response:

JSON
{
  "risk_score": 98.5,
  "label": "FAKE",
  "confidence": "99.2%",
  "findings": ["Artificial texture detected", "Eye reflection mismatch"]
}
POST /analyze_text
Check text for AI generation or known scam numbers.

Body: {"text": "Send money to 0702244938"}

Response:

JSON
{
  "risk_score": 100,
  "is_scam": true,
  "findings": ["⚠️ CRITICAL: Detected known scammer number: 0702244938"]
}
⚠️ Known Issues / Limitations
Audio Analysis: Uploads >10MB may time out on the free tier (limit clips to 30s for demo).

Cold Start: The backend sleeps after 48h of inactivity; the first request may take 40s to wake up.

🤝 Contributors
Victory Muratha - Lead Developer & AI Engineer
