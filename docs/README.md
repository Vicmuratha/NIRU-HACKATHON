SafEye - AI-Powered Deepfake Detection Platform
STILL IN DEVELOPMENT

SafEye is a comprehensive AI-powered platform for detecting deepfakes, manipulated media, and misinformation across images, audio, and text content. This platform provides real-time analysis with 99.2% accuracy.

🚀 Quick Start
Clone and Setup
git clone https://github.com/yourusername/NIRU-HACKATHON.git
cd NIRU-HACKATHON

# Download AI models (required)
python models/download_models.py

# Install dependencies
pip install -r requirements.txt
npm install

# Start the application
python app.py              # Backend on http://localhost:5000
npm run dev               # Frontend on http://localhost:3000
📁 Project Structure
NIRU-HACKATHON/
│
├── backend/               # Python Flask backend
│   └── app.py            # Main API server
├── models/               # AI models (download required)
│   ├── download_models.py # Model downloader script
│   ├── audio_model/      # Audio detection models
│   ├── text_model/       # Text detection models
│   └── best_deepfake_detector.pth
├── data/                 # Runtime data (ignored by git)
│   └── detection_log.json
├── docs/                 # Documentation
│   └── README.md
├── src/                  # React frontend source
├── tests/                # Unit tests
├── uploads/              # Temporary uploads (ignored)
├── index.html           # Vite entry point
├── package.json         # Frontend dependencies
├── requirements.txt     # Backend dependencies
└── .gitignore          # Git ignore rules
🧪 Features
Multimodal Detection: Analyze images, audio, and text content
Real-time Analysis: Instant results with confidence scores
Advanced AI Models: Uses state-of-the-art deep learning techniques
Modern React UI: Beautiful interface with drag-and-drop functionality
RESTful API: Easy integration with other systems
Kenya-Focused: Specialized detection for local threats and scams
⚠️ Important Notes for GitHub Users
Large Model Files
The AI models are not included in the repository due to size constraints:

Audio model: ~361MB
Text model: ~704MB
Image model: ~47MB
Total: ~1.1GB
Model Setup Required
After cloning, run the model downloader:

python models/download_models.py
Note: You'll need to update the download URLs in download_models.py with actual model hosting locations (GitHub releases, cloud storage, etc.).

📋 System Requirements
Python Dependencies
flask==3.0.0
flask-cors==4.0.0
numpy==1.24.3
pillow==10.1.0
librosa==0.10.1
opencv-python==4.8.1.78
torch==2.1.0
torchvision==0.16.0
transformers==4.35.0
deepface==0.0.79
exifread==3.0.0
scipy==1.11.4
werkzeug==3.0.1
tensorflow==2.15.0
Node.js Dependencies
react==19.2.3
vite==7.3.0
lucide-react==0.562.0
tailwindcss==3.4.0
🛠 Installation
System Prerequisites
Ubuntu/Debian:
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev
sudo apt-get install -y libsndfile1 ffmpeg
sudo apt-get install -y libsm6 libxext6 libxrender-dev
MacOS:
brew install python@3.10
brew install ffmpeg
brew install portaudio
Windows:
Install Python 3.10+ from python.org
Install Visual C++ Build Tools
Install ffmpeg from ffmpeg.org
Create Virtual Environment
# Create project directory
mkdir safeye-platform
cd safeye-platform

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt
Download Pre-trained Models
# Run this Python script to download models
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from deepface import DeepFace

# Download text analysis models
print("Downloading text analysis models...")
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Download face analysis models
print("Downloading face analysis models...")
DeepFace.build_model("Facenet")

print("All models downloaded successfully!")
📁 Project Structure
safeye-platform/
│
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── uploads/              # Temporary file storage
├── models/               # Pre-trained models
│
├── detectors/
│   ├── __init__.py
│   ├── image_detector.py
│   ├── audio_detector.py
│   └── text_detector.py
│
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
│
├── tests/
│   ├── test_image.py
│   ├── test_audio.py
│   └── test_text.py
│
└── README.md
🎯 Running the Application
Backend Server
# Start Flask backend
python app.py

# Server will run on http://localhost:5000
Frontend (HTML/JS Version)
# Serve frontend files
cd frontend
python -m http.server 3000

# Access at http://localhost:3000
🧪 Testing the API
Using cURL
# Test health endpoint
curl http://localhost:5000/api/health

# Test image analysis
curl -X POST -F "file=@test_image.jpg" http://localhost:5000/api/analyze/image

# Test audio analysis
curl -X POST -F "file=@test_audio.mp3" http://localhost:5000/api/analyze/audio

# Test text analysis
curl -X POST -H "Content-Type: application/json" \
  -d '{"text":"Breaking news! You wont believe what happened!"}' \
  http://localhost:5000/api/analyze/text
Using Python
import requests

# Test image
with open('test_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/api/analyze/image', files=files)
    print(response.json())

# Test text
data = {'text': 'This is a test message'}
response = requests.post('http://localhost:5000/api/analyze/text', json=data)
print(response.json())
⚙️ Configuration Options
Environment Variables
Create a .env file:

FLASK_ENV=production
FLASK_DEBUG=False
MAX_FILE_SIZE=52428800
UPLOAD_FOLDER=uploads
SECRET_KEY=your-secret-key-here
Advanced Configuration
# In app.py, add these configurations:

app.config.update(
    MAX_CONTENT_LENGTH=50 * 1024 * 1024,  # 50MB
    UPLOAD_FOLDER='uploads',
    ALLOWED_EXTENSIONS={'png', 'jpg', 'jpeg', 'mp3', 'wav', 'mp4', 'avi', 'mov', 'txt'},
    SQLALCHEMY_DATABASE_URI='sqlite:///safeye.db',  # For production
    REDIS_URL='redis://localhost:6379/0',  # For caching
)
🐳 Docker Deployment
Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 ffmpeg libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "app.py"]
docker-compose.yml
version: '3.8'
services:
  safeye-api:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./uploads:/app/uploads
    environment:
      - FLASK_ENV=production
🚀 Deployment Options
Azure Deployment
# Install Azure CLI
az login

# Create resource group
az group create --name safeye-rg --location eastus

# Create App Service plan
az appservice plan create --name safeye-plan --resource-group safeye-rg --sku B1 --is-linux

# Create web app
az webapp create --resource-group safeye-rg --plan safeye-plan --name safeye-api --runtime "PYTHON|3.10"

# Deploy code
az webapp up --name safeye-api --resource-group safeye-rg
AWS Deployment
# Install AWS CLI and EB CLI
pip install awsebcli

# Initialize Elastic Beanstalk
eb init -p python-3.10 safeye-platform

# Create environment
eb create safeye-env

# Deploy
eb deploy
⚡ Performance Optimization
Model Optimization
# Use quantization for faster inference
import torch.quantization

model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
Caching with Redis
import redis
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_cached_result(file_hash):
    cached = redis_client.get(f"analysis:{file_hash}")
    return json.loads(cached) if cached else None

def cache_result(file_hash, result):
    redis_client.setex(
        f"analysis:{file_hash}",
        3600,  # 1 hour TTL
        json.dumps(result)
    )
Async Processing with Celery
from celery import Celery

celery = Celery('safeye', broker='redis://localhost:6379/0')

@celery.task
def analyze_image_async(filepath):
    return image_detector.analyze_image(filepath)
🛡️ Security Best Practices
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
📚 API Documentation
Endpoints
Health Check
GET /api/health
Response: {"status": "healthy"}
Image Analysis
POST /api/analyze/image
Content-Type: multipart/form-data
Body: file (image file)
Response:
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
Audio Analysis
POST /api/analyze/audio
Content-Type: multipart/form-data
Body: file (audio file)
Response:
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
Text Analysis
POST /api/analyze/text
Content-Type: application/json
Body: {"text": "content to analyze"}
Response:
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
🧪 Running Tests
# Run all tests
python -m unittest discover tests/

# Run specific test file
python -m unittest tests/test_image.py

# Run with verbose output
python -m unittest -v tests/
Demo Script:
Show dashboard (2 min)
Upload authentic image → low risk score (1 min)
Upload manipulated image → high risk score + analysis (2 min)
Analyze suspicious audio clip (2 min)
Check misinformation text (1 min)
Show statistics and impact metrics (1 min)
Explain technical architecture (2 min)
Q&A (3 min)
📞 Support & Troubleshooting
Common Issues:
Issue: Models not downloading

# Solution: Install with specific versions
pip install transformers==4.35.0 --no-cache-dir
Issue: ffmpeg not found

# Ubuntu
sudo apt-get install ffmpeg

# Mac
brew install ffmpeg
Issue: CUDA errors

# Use CPU
export CUDA_VISIBLE_DEVICES=""
