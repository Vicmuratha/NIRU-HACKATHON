# SafEye Backend - High-confidence AI-assisted detection
# Competition-grade deepfake detection with AI models + advanced heuristics

import os
import warnings
import threading
import uuid
import json
import sqlite3
import logging
from datetime import datetime, timezone
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

from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session, g
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from werkzeug.utils import secure_filename
from PIL import Image
import exifread

# Kenya-specific modules
import sys
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.election_shield import analyze_election_context
from backend.whatsapp_checker import analyze_forward
from backend.kenya_documents import analyze_kenya_document, detect_document_type
from backend.audio_context import get_audio_kenya_context
from backend.fake_screenshot import detect_news_screenshot, run_ela as screenshot_ela
from backend.middleware import init_security, rate_limit, validate_file_upload, validate_json_body

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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
app.config['RATELIMIT_ENABLED'] = True
app.config['RATELIMIT_DEFAULT'] = '60/minute'
app.config['SECURITY_HEADERS'] = {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'X-XSS-Protection': '1; mode=block',
    'Referrer-Policy': 'strict-origin-when-cross-origin',
}

# ── Attach security middleware (headers, rate limiting, request logging) ──
init_security(app)

# ══════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════

DB_PATH = os.path.join(BASE_DIR, 'safeye.db')


def get_db():
    """Get a database connection for the current request."""
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()


def init_db():
    """Create all tables if they don't exist."""
    db = sqlite3.connect(DB_PATH)
    db.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP DEFAULT NULL
        )
    ''')
    db.execute('''
        CREATE TABLE IF NOT EXISTS detection_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            detection_type TEXT NOT NULL,
            filename TEXT,
            risk_score REAL,
            verdict TEXT,
            confidence REAL,
            findings TEXT,
            kenya_warnings TEXT,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    db.execute('''
        CREATE INDEX IF NOT EXISTS idx_history_user_id
        ON detection_history(user_id)
    ''')
    db.execute('''
        CREATE INDEX IF NOT EXISTS idx_history_created_at
        ON detection_history(created_at)
    ''')
    db.commit()
    db.close()
    logger.info(f"Database initialised at {DB_PATH}")


init_db()


def save_detection(detection_type, filename, result, user_id=None):
    """Persist a detection result to the database."""
    try:
        db = get_db()
        db.execute(
            '''INSERT INTO detection_history
               (user_id, detection_type, filename, risk_score, verdict, confidence, findings, kenya_warnings, details)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                user_id,
                detection_type,
                filename,
                result.get('risk_score'),
                result.get('verdict'),
                result.get('confidence'),
                json.dumps(result.get('findings', [])),
                json.dumps(result.get('kenya_warnings', [])),
                json.dumps(result.get('details', {})),
            )
        )
        db.commit()
    except Exception as e:
        logger.error(f"Failed to save detection: {e}")


# ============== AUTHENTICATION ==============
@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    if data.get('username') == 'admin' and data.get('password') == 'password':
        return jsonify(access_token=create_access_token(identity='admin')), 200
    return jsonify({'error': 'Invalid credentials'}), 401

# ============== WEB AUTH PAGES (PERSISTENT - SQLite) ==============

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        if user and user['password'] == password:
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            db.execute('UPDATE users SET last_login = ? WHERE id = ?',
                       (datetime.now(timezone.utc).isoformat(), user['id']))
            db.commit()
            return f"Welcome back, {user['name']}! (Redirecting to Dashboard...)"
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
        db = get_db()
        existing = db.execute('SELECT id FROM users WHERE email = ?', (email,)).fetchone()
        if existing:
            flash("Email already exists!")
            return redirect(url_for('signup_page'))
        db.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
                   (name, email, password))
        db.commit()
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

        final_risk = min(max(total_risk, 0.0), 100.0)
        confidence_sum = min(max(confidence_sum, 0.0), 1.0)
        verdict = "LIKELY_DEEPFAKE" if final_risk > 65 else "AUTHENTIC" if final_risk < 40 else "REVIEW_REQUIRED"
        
        kenya_warnings = []
        if final_risk > 70 and face_result['faces_detected'] > 0:
            kenya_warnings.append({'type': 'ELECTION_MANIPULATION', 'severity': 'CRITICAL', 'warning': 'This image may be a deepfake of a public figure. Manipulated images of politicians circulate on WhatsApp ahead of elections. Verify with official sources before sharing.', 'action': 'Report to NCIC: complaints@cohesion.or.ke | DCI: reportcrime@dci.go.ke'})
        if final_risk > 50:
            kenya_warnings.append({'type': 'MEDIA_MANIPULATION', 'severity': 'HIGH', 'warning': 'This image shows signs of manipulation. In Kenya, doctored news screenshots and fake campaign posters are common misinformation vectors.', 'action': 'Verify at the original news outlet website. Check PesaCheck.org for fact-checks.'})

        return {'risk_score': round(final_risk, 1), 'verdict': verdict, 'confidence': round(min(max(0.6, confidence_sum), 1.0), 2), 'findings': findings, 'kenya_warnings': kenya_warnings, 'details': {'ai_confidence': round(min(ai_result['fake_confidence'] * 100, 100.0), 1) if ai_result['available'] else 0, 'ela_score': round(ela_result.get('ela_score', 0), 1)}}

# ============== ULTRA-ACCURATE AUDIO DETECTOR ==============
class UltraAudioDetector:
    def __init__(self):
        self.sample_rate = 16000
        logger.info("🔧 Ultra-Accurate Audio Detector initialized")

    def analyze_audio(self, audio_path):
        import librosa
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
        except Exception as e:
            logger.error(f"Failed to load audio file: {e}")
            return {
                'risk_score': 0,
                'is_authentic': True,
                'confidence': 0.0,
                'findings': ['❌ Could not read audio file — it may be corrupted or in an unsupported format.'],
                'kenya_warnings': [],
                'error': f'Audio load failed: {str(e)}'
            }

        # Reject extremely short audio (< 0.5 seconds)
        duration = len(y) / sr
        if duration < 0.5:
            return {
                'risk_score': 0,
                'is_authentic': True,
                'confidence': 0.0,
                'findings': [f'❌ Audio too short ({duration:.1f}s) — need at least 0.5 seconds for analysis.'],
                'kenya_warnings': [],
                'error': 'Audio duration below minimum threshold (0.5s)'
            }

        try:
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
            
            risk = min(max(risk, 0), 98)
            confidence = round(min(max(0.5, 1.0 - (abs(mfcc_var - 500) / 1000)), 1.0), 2)
            kenya_warnings = []
            
            # Kenya-specific audio context (honest about the real threat)
            language = 'english'  # default; Swahili AI audio is still poor quality
            kenya_audio_ctx = get_audio_kenya_context(language, risk)
            
            if risk > 65:
                kenya_warnings.append({
                    'type': 'AUDIO_MANIPULATION',
                    'severity': 'HIGH',
                    'warning': 'This audio shows signs of manipulation. In Kenya, "leaked audio" of politicians is a common tactic — verify with official channels before sharing.',
                    'action': 'Report to DCI Cybercrime: reportcrime@dci.go.ke | Verify with at least 2 news outlets'
                })
            elif risk > 40:
                kenya_warnings.append({
                    'type': 'AUDIO_SUSPICIOUS',
                    'severity': 'MEDIUM',
                    'warning': 'This audio has some manipulation indicators. The most common audio manipulation in Kenya is splicing real recordings, not AI generation.',
                    'action': 'Verify the full context of this recording before sharing'
                })

            return {
                'risk_score': min(max(risk, 0), 100),
                'is_authentic': risk < 50,
                'confidence': confidence,
                'findings': findings,
                'kenya_warnings': kenya_warnings,
                'kenya_audio_context': kenya_audio_ctx,
                'detection_note': kenya_audio_ctx.get('detection_note', '')
            }
        except Exception as e:
            logger.error(f"Audio analysis error: {e}")
            return {
                'risk_score': 0,
                'is_authentic': True,
                'confidence': 0.0,
                'findings': [f'❌ Analysis failed: {str(e)}'],
                'kenya_warnings': [],
                'error': str(e)
            }

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
        
        # Clamp all output values to valid ranges (fixes G12)
        risk = min(max(risk, 0), 100)
        confidence = min(max(confidence, 0.0), 1.0)
        
        return {'risk_score': risk, 'is_authentic': risk < 50, 'confidence': round(confidence, 2), 'findings': [f"AI Result: {ai_result['label']}"]}

# ============== INITIALIZATION ==============
image_detector = UltraImageDetector()
audio_detector = UltraAudioDetector()
text_detector = UltraTextDetector()

# ============== ALLOWED FILE TYPES ==============
ALLOWED_IMAGE_TYPES = {'png', 'jpg', 'jpeg', 'webp'}
ALLOWED_AUDIO_TYPES = {'wav', 'mp3', 'ogg', 'flac'}

# ============== API ENDPOINTS ==============
@app.route('/api/analyze/image', methods=['POST'])
@rate_limit('30/minute')
def analyze_image():
    file, err = validate_file_upload('file', allowed_extensions=ALLOWED_IMAGE_TYPES)
    if err:
        return err
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}_{secure_filename(file.filename)}")
    file.save(filepath)
    try:
        result = image_detector.analyze_image(filepath)
        save_detection('image', file.filename, result, user_id=session.get('user_id'))
        return jsonify(result)
    finally:
        if os.path.exists(filepath): os.remove(filepath)

@app.route('/api/analyze/audio', methods=['POST'])
@rate_limit('30/minute')
def analyze_audio():
    file, err = validate_file_upload('file', allowed_extensions=ALLOWED_AUDIO_TYPES)
    if err:
        return err
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}_{secure_filename(file.filename)}")
    file.save(filepath)
    try:
        result = audio_detector.analyze_audio(filepath)
        save_detection('audio', file.filename, result, user_id=session.get('user_id'))
        return jsonify(result)
    finally:
        if os.path.exists(filepath): os.remove(filepath)

@app.route('/api/analyze/text', methods=['POST'])
@rate_limit('30/minute')
def analyze_text():
    data, err = validate_json_body('text')
    if err:
        return err
    result = text_detector.analyze_text(data['text'])
    save_detection('text', None, result, user_id=session.get('user_id'))
    return jsonify(result)

# ============== WHATSAPP FORWARD CHECKER ==============
@app.route('/api/analyze/forward', methods=['POST'])
def analyze_whatsapp_forward():
    """Analyse a WhatsApp forward for misinformation patterns."""
    data = request.get_json()
    text = (data or {}).get('text', '')

    if not text or len(text.strip()) < 10:
        return jsonify({'error': 'Text too short to analyse (min 10 characters)'}), 400

    # Run forward pattern analysis
    forward_result = analyze_forward(text)

    # Also run AI text detection
    ai_result = text_detector.analyze_text(text)

    # Combine scores: forward patterns (40%) + AI detection (60%)
    combined_score = min(100, (
        forward_result['forward_risk_score'] * 0.4
        + ai_result.get('risk_score', 0) * 0.6
    ))

    # Build Kenya warnings
    kenya_warnings = []
    if combined_score > 65:
        kenya_warnings.append({
            'type': 'MISINFORMATION',
            'severity': 'HIGH',
            'warning': 'This message has strong misinformation indicators. 67% of Kenyans get news via WhatsApp — do not forward unverified content.',
            'action': 'Verify with PesaCheck.org or Africa Check before sharing'
        })
    if forward_result['hoax_matches']:
        for hoax in forward_result['hoax_matches']:
            kenya_warnings.append({
                'type': hoax['category'].upper().replace(' ', '_'),
                'severity': 'HIGH',
                'warning': hoax['debunk'],
                'action': 'Do not forward this message'
            })

    # Election context
    election_ctx = analyze_election_context(text, 'text', combined_score)
    if election_ctx.get('election_relevant'):
        for w in election_ctx.get('warnings', []):
            kenya_warnings.append({
                'type': 'ELECTION_MISINFORMATION',
                'severity': election_ctx['risk_level'],
                'warning': w.get('en', ''),
                'action': '; '.join(election_ctx.get('recommendations', [])[:2])
            })

    return jsonify({
        'risk_score': round(combined_score, 1),
        'verdict': (
            'LIKELY_MISINFORMATION' if combined_score > 65
            else 'SUSPICIOUS' if combined_score > 40
            else 'APPEARS_GENUINE'
        ),
        'confidence': round(max(ai_result.get('confidence', 0.5), 0.6), 2),
        'findings': [
            f'Forward pattern score: {forward_result["forward_risk_score"]}%',
            f'AI text analysis: {ai_result.get("risk_score", 0)}%',
            f'Hoax templates matched: {len(forward_result["hoax_matches"])}',
            f'Swahili clickbait keywords: {len(forward_result.get("swahili_clickbait", []))}',
        ] + [f'🔍 Matched hoax: {h["category"]}' for h in forward_result['hoax_matches']],
        'kenya_warnings': kenya_warnings,
        'forward_analysis': forward_result,
        'is_authentic': combined_score < 40,
        'details': {
            'forward_score': forward_result['forward_risk_score'],
            'ai_score': ai_result.get('risk_score', 0),
            'hoax_count': len(forward_result['hoax_matches']),
        }
    })

# ============== DOCUMENT FORGERY CHECKER ==============
@app.route('/api/analyze/document', methods=['POST'])
def analyze_document():
    """Analyse an uploaded image for Kenyan document forgery."""
    if 'file' not in request.files:
        return jsonify({'error': 'Image file required'}), 400

    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}_{secure_filename(file.filename)}")
    file.save(filepath)

    try:
        # Run standard image analysis
        image_result = image_detector.analyze_image(filepath)

        # OCR the image for document text
        ocr_text = ''
        try:
            import pytesseract
            ocr_text = pytesseract.image_to_string(Image.open(filepath))
        except ImportError:
            logger.warning('pytesseract not installed — document OCR unavailable')
        except Exception as e:
            logger.warning(f'OCR failed: {e}')

        # Run Kenya document analysis
        ela_score = image_result.get('details', {}).get('ela_score', 0)
        doc_result = analyze_kenya_document(
            text=ocr_text,
            image_risk_score=image_result.get('risk_score', 0),
            ela_score=ela_score,
        )

        # Also check for news screenshot
        screenshot_result = detect_news_screenshot(
            ocr_text=ocr_text,
            ela_score=ela_score,
            ai_deepfake_score=image_result.get('risk_score', 0),
        )

        # Build combined result
        kenya_warnings = image_result.get('kenya_warnings', [])

        if doc_result.get('is_document') and doc_result.get('verdict') != 'APPEARS_GENUINE':
            kenya_warnings.append({
                'type': 'DOCUMENT_FORGERY',
                'severity': 'CRITICAL' if doc_result.get('verdict') == 'LIKELY_FORGED' else 'HIGH',
                'warning': f"{doc_result['document_name']}: {doc_result.get('kenya_context', {}).get('message_en', 'Possible forgery detected')}",
                'action': f"Verify at: {doc_result.get('kenya_context', {}).get('verify_at', 'N/A')} | {doc_result.get('kenya_context', {}).get('report_to', '')}"
            })

        if screenshot_result.get('is_news_screenshot') and screenshot_result.get('verdict') != 'APPEARS_GENUINE':
            outlet = screenshot_result.get('detected_outlet', {})
            kenya_warnings.append({
                'type': 'FAKE_NEWS_SCREENSHOT',
                'severity': 'CRITICAL' if screenshot_result['verdict'] == 'LIKELY_MANIPULATED' else 'HIGH',
                'warning': f"This appears to be a manipulated {outlet.get('name', 'news')} screenshot. Edited news screenshots are the #1 misinformation format on Kenyan WhatsApp.",
                'action': screenshot_result.get('action', {}).get('en', f"Verify at {outlet.get('verify_url', '')}")
            })

        # Use the higher risk score
        risk_score = image_result.get('risk_score', 0)
        if doc_result.get('is_document'):
            risk_score = max(risk_score, doc_result.get('risk_score', 0))

        return jsonify({
            'risk_score': round(risk_score, 1),
            'verdict': doc_result.get('verdict', image_result.get('verdict', 'REVIEW_REQUIRED')),
            'confidence': image_result.get('confidence', 0.6),
            'findings': image_result.get('findings', []) + [
                f"📄 Document type: {doc_result.get('document_name', 'Unknown')}" if doc_result.get('is_document') else '📄 No recognised Kenyan document detected',
                f"🗞️ News outlet: {screenshot_result.get('detected_outlet', {}).get('name', 'None')}" if screenshot_result.get('is_news_screenshot') else '',
            ],
            'kenya_warnings': kenya_warnings,
            'document_analysis': doc_result,
            'screenshot_analysis': screenshot_result,
            'ocr_text_preview': ocr_text[:300] if ocr_text else None,
            'is_authentic': risk_score < 40,
            'details': image_result.get('details', {}),
        })
    except Exception as e:
        logger.error(f'Document analysis error: {e}')
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

# ============== DETECTION HISTORY API ==============
@app.route('/api/history', methods=['GET'])
def detection_history():
    """Return recent detection history. Optionally filter by user session."""
    limit = min(int(request.args.get('limit', 50)), 200)
    user_id = session.get('user_id')

    db = get_db()
    if user_id:
        rows = db.execute(
            'SELECT * FROM detection_history WHERE user_id = ? ORDER BY created_at DESC LIMIT ?',
            (user_id, limit)
        ).fetchall()
    else:
        rows = db.execute(
            'SELECT * FROM detection_history ORDER BY created_at DESC LIMIT ?',
            (limit,)
        ).fetchall()

    history = []
    for row in rows:
        history.append({
            'id': row['id'],
            'detection_type': row['detection_type'],
            'filename': row['filename'],
            'risk_score': row['risk_score'],
            'verdict': row['verdict'],
            'confidence': row['confidence'],
            'findings': json.loads(row['findings']) if row['findings'] else [],
            'kenya_warnings': json.loads(row['kenya_warnings']) if row['kenya_warnings'] else [],
            'details': json.loads(row['details']) if row['details'] else {},
            'created_at': row['created_at'],
        })

    return jsonify({'history': history, 'count': len(history)})


@app.route('/api/history/stats', methods=['GET'])
def detection_stats():
    """Return aggregate detection statistics."""
    db = get_db()
    total = db.execute('SELECT COUNT(*) as cnt FROM detection_history').fetchone()['cnt']
    by_type = db.execute(
        'SELECT detection_type, COUNT(*) as cnt, AVG(risk_score) as avg_risk '
        'FROM detection_history GROUP BY detection_type'
    ).fetchall()

    return jsonify({
        'total_detections': total,
        'by_type': [
            {
                'type': row['detection_type'],
                'count': row['cnt'],
                'avg_risk_score': round(row['avg_risk'], 1) if row['avg_risk'] else 0,
            }
            for row in by_type
        ],
    })


# ============== API v1 VERSIONED ROUTES ==============
# Mirror analysis endpoints under /api/v1/ for forward compatibility.
# Legacy /api/analyze/* routes remain for backward compatibility.

@app.route('/api/v1/analyze/image', methods=['POST'])
@rate_limit('30/minute')
def analyze_image_v1():
    return analyze_image()


@app.route('/api/v1/analyze/audio', methods=['POST'])
@rate_limit('30/minute')
def analyze_audio_v1():
    return analyze_audio()


@app.route('/api/v1/analyze/text', methods=['POST'])
@rate_limit('30/minute')
def analyze_text_v1():
    return analyze_text()


@app.route('/api/v1/health', methods=['GET'])
def health_v1():
    return health()


@app.route('/api/v1/history', methods=['GET'])
def history_v1():
    return detection_history()


@app.route('/api/v1/history/stats', methods=['GET'])
def stats_v1():
    return detection_stats()


@app.route('/api/v1/docs', methods=['GET'])
def api_docs():
    """Return OpenAPI-style documentation for the SafEye API."""
    return jsonify({
        'openapi': '3.0.0',
        'info': {
            'title': 'SafEye API',
            'version': '1.0.0',
            'description': 'AI-powered deepfake, audio manipulation, and misinformation detection API tailored for the Kenyan information ecosystem.',
        },
        'basePath': '/api/v1',
        'endpoints': {
            'POST /api/v1/analyze/image': {
                'description': 'Analyse an image for deepfake indicators',
                'content_type': 'multipart/form-data',
                'parameters': {'file': 'Image file (png, jpg, jpeg, webp)'},
                'response': {'risk_score': 'float 0-100', 'verdict': 'AUTHENTIC | REVIEW_REQUIRED | LIKELY_DEEPFAKE', 'confidence': 'float 0-1', 'findings': 'list[str]', 'kenya_warnings': 'list[object]'},
            },
            'POST /api/v1/analyze/audio': {
                'description': 'Analyse an audio file for manipulation indicators',
                'content_type': 'multipart/form-data',
                'parameters': {'file': 'Audio file (wav, mp3, ogg, flac)'},
                'response': {'risk_score': 'float 0-100', 'is_authentic': 'bool', 'confidence': 'float 0-1', 'findings': 'list[str]', 'kenya_warnings': 'list[object]'},
            },
            'POST /api/v1/analyze/text': {
                'description': 'Analyse text for AI-generated or fake news indicators',
                'content_type': 'application/json',
                'parameters': {'text': 'string (required)'},
                'response': {'risk_score': 'int 0-100', 'is_authentic': 'bool', 'confidence': 'float 0-1', 'findings': 'list[str]'},
            },
            'POST /api/v1/analyze/forward': {
                'description': 'Analyse a WhatsApp forward for misinformation patterns',
                'content_type': 'application/json',
                'parameters': {'text': 'string (min 10 chars)'},
                'response': {'risk_score': 'float 0-100', 'verdict': 'APPEARS_GENUINE | SUSPICIOUS | LIKELY_MISINFORMATION', 'forward_analysis': 'object'},
            },
            'POST /api/v1/analyze/document': {
                'description': 'Analyse an image for Kenyan document forgery',
                'content_type': 'multipart/form-data',
                'parameters': {'file': 'Image file'},
                'response': {'risk_score': 'float 0-100', 'verdict': 'string', 'document_analysis': 'object', 'screenshot_analysis': 'object'},
            },
            'GET /api/v1/health': {
                'description': 'Health check — returns server status and loaded modules',
            },
            'GET /api/v1/history': {
                'description': 'Retrieve detection history (optionally filtered by authenticated user)',
                'parameters': {'limit': 'int (default 50, max 200)'},
            },
            'GET /api/v1/history/stats': {
                'description': 'Aggregate detection statistics by type',
            },
        },
    })


# ============== HEALTH CHECK ==============
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': image_detector.ai_model is not None,
        'platform': 'SafEye Kenya',
        'version': '2.0.0',
        'kenya_modules': {
            'election_shield': True,
            'whatsapp_checker': True,
            'document_verifier': True,
            'news_screenshot_detector': True,
            'audio_context': True,
        }
    })

if __name__ == '__main__':
    print("🚀 Starting SafEye Server...")
    # HOST must be 0.0.0.0 for external access (Ngrok/Azure)
    app.run(host='0.0.0.0', port=7860)