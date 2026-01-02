# SafEye Backend - Main Application (app.py)
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    import torch
    from deepface import DeepFace
    import exifread
    from scipy import signal
    HEAVY_DEPS_AVAILABLE = True
except ImportError:
    HEAVY_DEPS_AVAILABLE = False

# Import detectors
from detectors.image_detector import ImageDeepfakeDetector
from detectors.audio_detector import AudioDeepfakeDetector
from detectors.text_detector import TextMisinformationDetector

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp3', 'wav', 'mp4', 'avi', 'mov', 'txt'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize detectors
image_detector = ImageDeepfakeDetector()
audio_detector = AudioDeepfakeDetector()
text_detector = TextMisinformationDetector()

# ============== API ENDPOINTS ==============

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'SafEye API is running'})

@app.route('/api/analyze/image', methods=['POST'])
def analyze_image():
    """Analyze image for deepfakes and manipulation"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Analyze
        results = image_detector.analyze_image(filepath)

        # Clean up
        os.remove(filepath)

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/audio', methods=['POST'])
def analyze_audio():
    """Analyze audio for deepfakes and spoofing"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Analyze
        results = audio_detector.analyze_audio(filepath)

        # Clean up
        os.remove(filepath)

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/text', methods=['POST'])
def analyze_text():
    """Analyze text for misinformation"""
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Analyze
        results = text_detector.analyze_text(text)

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get platform statistics"""
    # In production, pull from database
    stats = {
        'total_analyzed': 12847,
        'authentic': 9234,
        'manipulated': 3613,
        'accuracy_rate': 99.2
    }
    return jsonify(stats)

if __name__ == '__main__':
    print("=" * 60)
    print("SafEye - AI-Powered Deepfake Detection Platform")
    print("=" * 60)
    print("Starting server...")
    app.run(debug=True, host='localhost', port=5000)
