# SafEye Backend - High-confidence AI-assisted detection
# Competition-grade deepfake detection with AI models + advanced heuristics

import os
import warnings
import threading
import uuid
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np

# --- ADDED: Load environment variables from .env file ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, assuming env vars set manually

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variables for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from werkzeug.utils import secure_filename
from PIL import Image
import exifread

warnings.filterwarnings('ignore')

# --- FIX: Correctly get the project root directory (Current Folder) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)
CORS(app)
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'safeye-hackathon-secret-2026')
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'super_secret_key')
jwt = JWTManager(app)

# Define folders relative to the current directory
# Note: Use 'models' directory at project root to match download_models.py behavior
PROJECT_ROOT = os.path.dirname(BASE_DIR)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
DATA_FOLDER = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models') # Project root models dir

# Create them if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- FIX: Updated Model Downloader Integration ---
def download_models_on_startup():
    flag = os.getenv("DOWNLOAD_MODELS_ON_STARTUP", "").strip().lower()
    if flag not in {"1", "true", "yes"}:
        return

    try:
        import sys
        # download_models.py lives in the project-root 'models/' folder
        models_script_dir = os.path.join(os.path.dirname(BASE_DIR), 'models')
        if models_script_dir not in sys.path:
            sys.path.insert(0, models_script_dir)
        import download_models as model_downloader
        logger.info("📦 Downloading models on startup...")
        model_downloader.main()
        logger.info("✅ Model download completed")
    except ImportError as ie:
        logger.error(f"❌ Could not import 'download_models.py': {ie}")
    except Exception as e:
        logger.warning(f"⚠️ Model download skipped/failed: {e}")

download_models_on_startup()

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# ============== AUTHENTICATION ==============
@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    if data.get('username') == 'admin' and data.get('password') == 'password':
        return jsonify(access_token=create_access_token(identity='admin')), 200
    return jsonify({'error': 'Invalid credentials'}), 401

# ============== WEB AUTH PAGES (HACKATHON DEMO) ==============
users_db: Dict[str, Dict[str, str]] = {}

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email in users_db and users_db[email]['password'] == password:
            return f"Welcome back, {users_db[email]['name']}! (Redirecting to Dashboard...)"
        else:
            flash("Invalid email or password!")
            return redirect(url_for('login_page'))
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup_page():
    if request.method == 'POST':
        name = request.form['username']
        email = request.form['email']
        password = request.form['password']
        if password != request.form['confirm_password']:
            flash("Passwords do not match!")
            return redirect(url_for('signup_page'))
        if email in users_db:
            flash("Email already exists!")
            return redirect(url_for('signup_page'))
        users_db[email] = {'name': name, 'password': password}
        flash("Account created! Please log in.")
        return redirect(url_for('login_page'))
    return render_template('signup.html')

# ============== ULTRA-ACCURATE IMAGE DETECTOR ==============
class UltraImageDetector:
    def __init__(self):
        self.ai_model = None
        self.ai_processor = None
        self.lock = threading.Lock()
        logger.info("🔧 Ultra-Accurate Image Detector initialized")

    def load_ai_model(self):
        if self.ai_model is None:
            with self.lock:
                if self.ai_model is None:
                    import torch
                    from torchvision import transforms

                    local_model_dir = os.path.join(MODELS_DIR, "image_model")
                    pth_path = os.path.join(local_model_dir, "best_deepfake_detector.pth")
                    hf_config_path = os.path.join(local_model_dir, "config.json")

                    # ── Option 1: Load custom EfficientNet-B4 .pth checkpoint ──
                    if os.path.exists(pth_path) and os.path.getsize(pth_path) > 1000:
                        try:
                            from torchvision.models import efficientnet_b4
                            logger.info("📥 Loading custom EfficientNet-B4 .pth model from LOCAL")
                            model = efficientnet_b4(weights=None)
                            # Replace classifier head for 2-class (real/fake)
                            in_features = model.classifier[1].in_features
                            model.classifier[1] = torch.nn.Linear(in_features, 2)

                            ckpt = torch.load(pth_path, map_location='cpu', weights_only=False)
                            state_dict = ckpt.get('model_state_dict', ckpt)
                            model.load_state_dict(state_dict, strict=False)
                            model.eval()

                            self.ai_model = model
                            self.ai_processor = transforms.Compose([
                                transforms.Resize((380, 380)),
                                transforms.CenterCrop(380),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                            ])
                            self._model_type = 'pth'
                            logger.info("✅ Custom EfficientNet-B4 model loaded successfully")
                            return
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to load .pth model: {e}")

                    # ── Option 2: HuggingFace format (config.json + model.safetensors) ──
                    try:
                        from transformers import AutoModelForImageClassification, AutoImageProcessor
                        use_local = os.path.exists(hf_config_path)
                        logger.info(f"📥 Loading deepfake detection AI model from: {'LOCAL HF' if use_local else 'HUGGING FACE HUB'}")
                        model_source = local_model_dir if use_local else "dima806/deepfake_vs_real_image_detection"
                        self.ai_model = AutoModelForImageClassification.from_pretrained(model_source)
                        self.ai_processor = AutoImageProcessor.from_pretrained(model_source)
                        self.ai_model.eval()
                        self._model_type = 'hf'
                        logger.info("✅ HuggingFace AI model loaded successfully")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not load AI model: {e}")
                        self.ai_model = "unavailable"

    def ai_deepfake_check(self, image_path):
        self.load_ai_model()
        if self.ai_model == "unavailable":
            return {'available': False, 'fake_confidence': 0}
        try:
            import torch
            image = Image.open(image_path).convert('RGB')

            if getattr(self, '_model_type', 'hf') == 'pth':
                # Custom EfficientNet-B4
                input_tensor = self.ai_processor(image).unsqueeze(0)
                with torch.no_grad():
                    outputs = self.ai_model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=-1)
                fake_confidence = float(probs[0][1].item())
                real_confidence = float(probs[0][0].item())
            else:
                # HuggingFace model
                if max(image.size) > 512:
                    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
                inputs = self.ai_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.ai_model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                fake_confidence = float(probs[0][1].item())
                real_confidence = float(probs[0][0].item())

            return {'available': True, 'fake_confidence': fake_confidence, 'real_confidence': real_confidence}
        except Exception as e:
            logger.error(f"❌ AI detection error: {e}")
            return {'available': False, 'fake_confidence': 0}

    def error_level_analysis(self, image_path):
        try:
            original = Image.open(image_path).convert('RGB')
            temp_filename = f"temp_ela_{uuid.uuid4().hex}.jpg"
            temp_path = os.path.join(os.path.dirname(image_path), temp_filename)
            try:
                original.save(temp_path, 'JPEG', quality=90)
                compressed = Image.open(temp_path)
                original_arr = np.array(original).astype(np.float32)
                compressed_arr = np.array(compressed).astype(np.float32)
                if original_arr.shape != compressed_arr.shape:
                    compressed = compressed.resize(original.size, Image.Resampling.LANCZOS)
                    compressed_arr = np.array(compressed).astype(np.float32)
                diff = np.abs(original_arr - compressed_arr)
                ela_score = float(np.mean(diff))
                if ela_score < 2.5: risk = 95; assessment = 'EXTREMELY_CLEAN'
                elif ela_score < 5.0: risk = 80; assessment = 'VERY_CLEAN'
                elif ela_score < 8.0: risk = 55; assessment = 'CLEAN'
                elif ela_score < 15.0: risk = 30; assessment = 'MODERATE'
                else: risk = 12; assessment = 'HEAVY_COMPRESSION'
                risk = min(risk, 60)
                return {'ela_score': ela_score, 'assessment': assessment, 'risk': risk}
            finally:
                if os.path.exists(temp_path): os.remove(temp_path)
        except Exception as e:
            return {'ela_score': 15.0, 'assessment': 'UNKNOWN', 'risk': 35}

    def analyze_metadata(self, image_path):
        try:
            with open(image_path, 'rb') as f: tags = exifread.process_file(f, details=False)
            make = str(tags.get('Image Make', tags.get('EXIF Make', ''))).strip()
            model = str(tags.get('Image Model', tags.get('EXIF Model', ''))).strip()
            software = str(tags.get('Image Software', '')).lower().strip()
            trusted_brands = ['samsung', 'apple', 'iphone', 'google', 'pixel', 'huawei', 'tecno', 'infinix', 'oppo', 'xiaomi', 'vivo', 'canon', 'nikon', 'sony']
            is_trusted_camera = any(b in make.lower() or b in model.lower() for b in trusted_brands)
            was_edited = any(sw in software for sw in ['photoshop', 'gimp', 'paint.net', 'lightroom', 'affinity'])
            
            if is_trusted_camera: risk = 12 if not was_edited else 25
            elif make or model: risk = 35
            else: risk = 25
            return {'has_metadata': bool(make or model), 'is_trusted_camera': is_trusted_camera, 'camera_info': f"{make} {model}".strip() or 'None', 'was_edited': was_edited, 'editing_software': software if was_edited else None, 'metadata_count': len(tags), 'risk': risk}
        except:
            return {'has_metadata': False, 'is_trusted_camera': False, 'camera_info': 'None', 'was_edited': False, 'metadata_count': 0, 'risk': 52}

    def analyze_face_texture(self, image_path, sharpness):
        try:
            from deepface import DeepFace
            faces = DeepFace.extract_faces(image_path, enforce_detection=False)
            if not faces or len(faces) == 0: return {'faces_detected': 0, 'risk': 0, 'assessment': 'NO_FACE'}
            
            max_risk = 0
            best_assessment = 'NORMAL'
            for face_data in faces:
                face_img = face_data['face']
                if isinstance(face_img, np.ndarray):
                    if face_img.max() <= 1.0: face_img = (face_img * 255).astype(np.uint8)
                    face_std = float(np.std(face_img))
                    if sharpness > 100:
                        if face_std < 14: risk = 75; assessment = 'SYNTHETIC'
                        elif face_std < 28: risk = 68; assessment = 'SUSPICIOUSLY_SMOOTH'
                        elif face_std < 40: risk = 28; assessment = 'SMOOTH'
                        else: risk = 8; assessment = 'NATURAL'
                    else:
                        if face_std < 10: risk = 75; assessment = 'TOO_SMOOTH'
                        elif face_std < 22: risk = 35; assessment = 'SMOOTH'
                        else: risk = 12; assessment = 'NORMAL'
                    risk = min(risk, 75)
                    if risk > max_risk: max_risk = risk; best_assessment = assessment
            return {'faces_detected': len(faces), 'risk': max_risk, 'assessment': best_assessment}
        except:
            return {'faces_detected': 0, 'risk': 0, 'assessment': 'ERROR'}

    def get_sharpness(self, image_path):
        try:
            import cv2
            img = cv2.imread(image_path)
            if img is None: return 50.0
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except: return 50.0

    def noise_analysis(self, image_path):
        try:
            from scipy import fft
            img = Image.open(image_path).convert('L')
            img.thumbnail((256, 256), Image.Resampling.LANCZOS)
            img_array = np.array(img).astype(np.float32)
            noise_score = float(np.mean(np.abs(fft.fftshift(fft.fft2(img_array)))[img_array.shape[0]//3:, img_array.shape[1]//3:]))
            if noise_score < 12: return {'noise_score': noise_score, 'risk': 25, 'assessment': 'VERY_LOW'}
            elif noise_score < 20: return {'noise_score': noise_score, 'risk': 10, 'assessment': 'LOW'}
            return {'noise_score': noise_score, 'risk': 0, 'assessment': 'NORMAL'}
        except: return {'noise_score': 0, 'risk': 0, 'assessment': 'UNKNOWN'}

    def analyze_image(self, image_path):
        sharpness = self.get_sharpness(image_path)
        ai_result = self.ai_deepfake_check(image_path)
        ela_result = self.error_level_analysis(image_path)
        meta_result = self.analyze_metadata(image_path)
        face_result = self.analyze_face_texture(image_path, sharpness)
        noise_result = self.noise_analysis(image_path)

        total_risk = 0
        confidence_sum = 0
        findings = []

        # AI Model (60%)
        if ai_result['available']:
            total_risk += ai_result['fake_confidence'] * 100 * 0.60
            confidence_sum += 0.60
            findings.append(f"🤖 AI Model: {'DEEPFAKE' if ai_result['fake_confidence'] > 0.5 else 'Authentic'} ({ai_result['fake_confidence']:.1%} confidence)")

        # Metadata (15%)
        total_risk += meta_result['risk'] * 0.15; confidence_sum += 0.15
        findings.append(f"✓ Camera: {meta_result['camera_info']}" if meta_result['is_trusted_camera'] else "⚠️ No trusted camera metadata")

        # ELA (15%)
        total_risk += ela_result['risk'] * 0.15; confidence_sum += 0.15
        
        # Face Texture (7%)
        if face_result['faces_detected'] > 0:
            total_risk += face_result['risk'] * 0.07; confidence_sum += 0.07

        # Noise (3%)
        total_risk += noise_result['risk'] * 0.03; confidence_sum += 0.03

        final_risk = min(max(total_risk, 0), 100)
        verdict = "LIKELY_DEEPFAKE" if final_risk > 65 else "AUTHENTIC" if final_risk < 40 else "REVIEW_REQUIRED"
        
        kenya_warnings = []
        if final_risk > 70 and face_result['faces_detected'] > 0:
            kenya_warnings.append({'type': 'ELECTION_MANIPULATION', 'severity': 'CRITICAL', 'warning': 'Political deepfake risk', 'action': 'Verify source'})

        return {'risk_score': round(final_risk, 1), 'verdict': verdict, 'confidence': round(max(0.6, confidence_sum), 2), 'findings': findings, 'kenya_warnings': kenya_warnings, 'details': {'ai_confidence': round(ai_result['fake_confidence']*100, 1) if ai_result['available'] else 0}}

# ============== ULTRA-ACCURATE AUDIO DETECTOR ==============
class UltraAudioDetector:
    def __init__(self):
        self.sample_rate = 16000
        logger.info("🔧 Ultra-Accurate Audio Detector initialized")

    def analyze_audio(self, audio_path):
        import librosa
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            if len(y) < 2048: y = np.pad(y, (0, 2048 - len(y)))
            
            # Simple Features
            mfcc_var = float(np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)))
            rms = librosa.feature.rms(y=y)[0]
            silence_ratio = float(np.sum(rms < (np.mean(rms) * 0.12)) / len(rms))
            
            risk = 0
            findings = []
            
            if mfcc_var < 150: risk += 75; findings.append("⚠️ Robotic voice texture")
            elif mfcc_var < 400: risk += 40; findings.append("⚠️ Smooth voice texture")
            else: findings.append("✓ Natural voice variation")
            
            if not (0.02 < silence_ratio < 0.15): risk += 30; findings.append("⚠️ Abnormal breathing pauses")
            
            risk = min(risk, 98)
            kenya_warnings = []
            if risk > 65:
                kenya_warnings.append({'type': 'MPESA_FRAUD', 'severity': 'HIGH', 'warning': 'Voice cloning risk', 'action': 'Do not authorize transactions via voice'})

            return {'risk_score': risk, 'is_authentic': risk < 50, 'confidence': 0.88, 'findings': findings, 'kenya_warnings': kenya_warnings}
        except Exception as e:
            return {'risk_score': 0, 'is_authentic': True, 'error': str(e)}

# ============== ULTRA-ACCURATE TEXT DETECTOR ==============
class UltraTextDetector:
    def __init__(self):
        self.pipeline = None
        self.lock = threading.Lock()
        logger.info("🔧 Ultra-Accurate Text Detector initialized")

    def analyze_text(self, text):
        with self.lock:
            if self.pipeline is None:
                from transformers import pipeline
                # --- FIX: Check the 'models' subdirectory ---
                local_model_dir = os.path.join(MODELS_DIR, "text_model")
                use_local = os.path.exists(os.path.join(local_model_dir, "config.json"))
                model_source = local_model_dir if use_local else "hamzab/roberta-fake-news-classification"
                self.pipeline = pipeline("text-classification", model=model_source, tokenizer=model_source)

        ai_result = self.pipeline(text[:512])[0]
        is_fake = ai_result['label'] in ['FAKE', 'LABEL_0', '0']
        confidence = ai_result['score']
        risk = int(confidence * 100) if is_fake else int((1 - confidence) * 100)
        
        # Simple heuristics
        txt_lower = text.lower()
        clickbait_count = sum(1 for kw in ['exposed', 'shocking', 'secret'] if kw in txt_lower)
        if clickbait_count > 0: risk = min(risk + 20, 96)
        
        return {'risk_score': risk, 'is_authentic': risk < 50, 'confidence': confidence, 'findings': [f"AI Result: {ai_result['label']}"]}

# ============== INITIALIZATION ==============
image_detector = UltraImageDetector()
audio_detector = UltraAudioDetector()
text_detector = UltraTextDetector()

#API ENDPOINTS 
@app.route('/api/analyze/image', methods=['POST'])
def analyze_image():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}_{secure_filename(file.filename)}")
    file.save(filepath)
    try:
        result = image_detector.analyze_image(filepath)
        return jsonify(result)
    finally:
        if os.path.exists(filepath): os.remove(filepath)

@app.route('/api/analyze/audio', methods=['POST'])
def analyze_audio():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}_{secure_filename(file.filename)}")
    file.save(filepath)
    try:
        result = audio_detector.analyze_audio(filepath)
        return jsonify(result)
    finally:
        if os.path.exists(filepath): os.remove(filepath)

@app.route('/api/analyze/text', methods=['POST'])
def analyze_text():
    data = request.get_json()
    result = text_detector.analyze_text(data.get('text', ''))
    return jsonify(result)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'models_loaded': image_detector.ai_model is not None})

if __name__ == '__main__':
    print("🚀 Starting SafEye Server...")
    # HOST must be 0.0.0.0 for external access (Ngrok/Azure)
    app.run(host='0.0.0.0', port=7860)