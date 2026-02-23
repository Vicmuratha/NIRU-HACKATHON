"""
SafEye — Unified Backend  v3.1
Production-grade Flask application:
  • Authentication (local + OAuth)
  • AI-powered deepfake / misinformation detection
  • User profiles & detection history
  • Kenya election-integrity modules
"""

import os
import sqlite3
import uuid
import json
import warnings
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from functools import wraps

import numpy as np

# ── Load .env early ──
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Suppress TF noise before any TF import ──
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# ── Structured logging (must come before other app imports) ──
from backend.logging_config import setup_logging
from backend.config import get_config

_cfg = get_config()
setup_logging(level=_cfg.LOG_LEVEL, fmt=_cfg.LOG_FORMAT)

import logging
logger = logging.getLogger(__name__)

from flask import (
    Flask, render_template, url_for, redirect, session,
    jsonify, request, flash, g, send_from_directory
)
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from werkzeug.utils import secure_filename
from PIL import Image

try:
    import exifread
except ImportError:
    exifread = None

try:
    from authlib.integrations.flask_client import OAuth
    HAS_OAUTH = True
except ImportError:
    HAS_OAUTH = False

# ── Security & error handling ──
from backend.middleware import init_security, rate_limit, validate_file_upload, validate_json_body
from backend.errors import init_error_handlers, SafEyeError, AuthenticationError, ValidationError, AnalysisError

# ─── Kenya-specific modules ───
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.election_shield import analyze_election_context
from backend.whatsapp_checker import analyze_forward
from backend.kenya_documents import analyze_kenya_document, detect_document_type
from backend.audio_context import get_audio_kenya_context
from backend.fake_screenshot import detect_news_screenshot, run_ela as screenshot_ela

# ══════════════════════════════════════════════════════════════
#  FLASK APP FACTORY
# ══════════════════════════════════════════════════════════════

config = get_config()
FRONTEND_URL = config.FRONTEND_URL

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)

# ── Apply config object ──
app.config.from_object(config)
app.secret_key = config.SECRET_KEY

# ── Production security middleware & error handlers ──
init_security(app)
init_error_handlers(app)

# ── CORS ──
_cors_origins = [
    'http://localhost:3000',
    'http://localhost:7860',
    'http://127.0.0.1:3000',
    'http://127.0.0.1:7860',
]
if FRONTEND_URL not in _cors_origins:
    _cors_origins.append(FRONTEND_URL)
_extra_origins = os.getenv('EXTRA_CORS_ORIGINS', '')
if _extra_origins:
    _cors_origins.extend([o.strip() for o in _extra_origins.split(',') if o.strip()])

CORS(app, supports_credentials=True, origins=_cors_origins)
jwt = JWTManager(app)

# Allow OAuth over HTTP only in development
if app.debug:
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# ─── OAuth (Google + GitHub) ───
if HAS_OAUTH:
    oauth = OAuth(app)
    google = oauth.register(
        name='google',
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid email profile'}
    )
    github = oauth.register(
        name='github',
        client_id=os.getenv("GITHUB_CLIENT_ID"),
        client_secret=os.getenv("GITHUB_CLIENT_SECRET"),
        access_token_url='https://github.com/login/oauth/access_token',
        authorize_url='https://github.com/login/oauth/authorize',
        client_kwargs={'scope': 'user:email'},
    )

# ─── File Upload Config ───
PROJECT_ROOT = BASE_DIR
UPLOAD_FOLDER = config.UPLOAD_FOLDER
DATA_FOLDER = os.path.join(BASE_DIR, 'data')
MODELS_DIR = config.MODELS_DIR

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

# ══════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════

DB_PATH = config.DATABASE_PATH if config.DATABASE_PATH != ':memory:' else os.path.join(BASE_DIR, 'users.db')


def get_db():
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
            bio TEXT DEFAULT '',
            phone TEXT DEFAULT '',
            location TEXT DEFAULT '',
            organization TEXT DEFAULT '',
            role TEXT DEFAULT 'user',
            profile_picture TEXT DEFAULT '',
            auth_provider TEXT DEFAULT 'local',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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

    # ─── Migrate existing users table if needed ───
    cursor = db.execute("PRAGMA table_info(users)")
    columns = [row[1] for row in cursor.fetchall()]
    new_columns = {
        'bio': "TEXT DEFAULT ''",
        'phone': "TEXT DEFAULT ''",
        'location': "TEXT DEFAULT ''",
        'organization': "TEXT DEFAULT ''",
        'role': "TEXT DEFAULT 'user'",
        'profile_picture': "TEXT DEFAULT ''",
        'auth_provider': "TEXT DEFAULT 'local'",
        'updated_at': "TIMESTAMP DEFAULT NULL",
        'last_login': "TIMESTAMP DEFAULT NULL",
    }
    for col_name, col_def in new_columns.items():
        if col_name not in columns:
            try:
                db.execute(f'ALTER TABLE users ADD COLUMN {col_name} {col_def}')
            except Exception:
                pass

    # Backfill updated_at for existing rows
    db.execute('UPDATE users SET updated_at = created_at WHERE updated_at IS NULL')

    db.commit()
    db.close()


init_db()


from werkzeug.security import generate_password_hash, check_password_hash

def hash_password(password):
    return generate_password_hash(password)


def get_current_user_id():
    """Get user ID from session."""
    user = session.get('user')
    if not user:
        return None
    email = user.get('email')
    if not email:
        return None
    db = get_db()
    row = db.execute('SELECT id FROM users WHERE email = ?', (email,)).fetchone()
    return row['id'] if row else None


def save_detection_history(user_id, detection_type, filename, result):
    """Save a detection result to history."""
    if not user_id:
        return
    try:
        db = get_db()
        db.execute('''
            INSERT INTO detection_history
            (user_id, detection_type, filename, risk_score, verdict, confidence, findings, kenya_warnings, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            detection_type,
            filename or 'Unknown',
            result.get('risk_score', 0),
            result.get('verdict', 'UNKNOWN'),
            result.get('confidence', 0),
            json.dumps(result.get('findings', [])),
            json.dumps(result.get('kenya_warnings', [])),
            json.dumps(result.get('details', {})),
        ))
        db.commit()
    except Exception as e:
        logger.error(f"Failed to save detection history: {e}")


# ══════════════════════════════════════════════════════════════
#  MODEL DOWNLOAD (STARTUP)
# ══════════════════════════════════════════════════════════════

def download_models_on_startup():
    flag = os.getenv("DOWNLOAD_MODELS_ON_STARTUP", "").strip().lower()
    if flag not in {"1", "true", "yes"}:
        return
    try:
        models_script_dir = os.path.join(BASE_DIR, 'models')
        if models_script_dir not in sys.path:
            sys.path.insert(0, models_script_dir)
        import download_models as model_downloader
        logger.info("Downloading models on startup...")
        model_downloader.main()
        logger.info("Model download completed")
    except ImportError as ie:
        logger.error(f"Could not import download_models.py: {ie}")
    except Exception as e:
        logger.warning(f"Model download skipped/failed: {e}")


download_models_on_startup()

# ══════════════════════════════════════════════════════════════
#  DETECTORS (from backend/app.py)
# ══════════════════════════════════════════════════════════════

class UltraImageDetector:
    def __init__(self):
        self.ai_model = None
        self.ai_processor = None
        self.lock = threading.Lock()
        self._model_type = 'hf'
        logger.info("Ultra-Accurate Image Detector initialized")

    def load_ai_model(self):
        if self.ai_model is None:
            with self.lock:
                if self.ai_model is None:
                    import torch
                    from torchvision import transforms

                    local_model_dir = os.path.join(MODELS_DIR, "image_model")
                    pth_path = os.path.join(local_model_dir, "best_deepfake_detector.pth")
                    hf_config_path = os.path.join(local_model_dir, "config.json")

                    if os.path.exists(pth_path) and os.path.getsize(pth_path) > 1000:
                        try:
                            from torchvision.models import efficientnet_b4
                            logger.info("Loading custom EfficientNet-B4 .pth model")
                            model = efficientnet_b4(weights=None)
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
                            logger.info("Custom EfficientNet-B4 model loaded")
                            return
                        except Exception as e:
                            logger.warning(f"Failed to load .pth model: {e}")

                    try:
                        from transformers import AutoModelForImageClassification, AutoImageProcessor
                        use_local = os.path.exists(hf_config_path)
                        model_source = local_model_dir if use_local else "dima806/deepfake_vs_real_image_detection"
                        logger.info(f"Loading image model from: {'LOCAL' if use_local else 'HuggingFace'}")
                        self.ai_model = AutoModelForImageClassification.from_pretrained(model_source)
                        self.ai_processor = AutoImageProcessor.from_pretrained(model_source)
                        self.ai_model.eval()
                        self._model_type = 'hf'
                        logger.info("HuggingFace image model loaded")
                    except Exception as e:
                        logger.warning(f"Could not load AI model: {e}")
                        self.ai_model = "unavailable"

    def ai_deepfake_check(self, image_path):
        self.load_ai_model()
        if self.ai_model == "unavailable":
            return {'available': False, 'fake_confidence': 0}
        try:
            import torch
            image = Image.open(image_path).convert('RGB')
            if self._model_type == 'pth':
                input_tensor = self.ai_processor(image).unsqueeze(0)
                with torch.no_grad():
                    outputs = self.ai_model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=-1)
                fake_confidence = float(probs[0][1].item())
                real_confidence = float(probs[0][0].item())
            else:
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
            logger.error(f"AI detection error: {e}")
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
                if ela_score < 2.5:
                    risk = 95; assessment = 'EXTREMELY_CLEAN'
                elif ela_score < 5.0:
                    risk = 80; assessment = 'VERY_CLEAN'
                elif ela_score < 8.0:
                    risk = 55; assessment = 'CLEAN'
                elif ela_score < 15.0:
                    risk = 30; assessment = 'MODERATE'
                else:
                    risk = 12; assessment = 'HEAVY_COMPRESSION'
                risk = min(risk, 60)
                return {'ela_score': ela_score, 'assessment': assessment, 'risk': risk}
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception:
            return {'ela_score': 15.0, 'assessment': 'UNKNOWN', 'risk': 35}

    def analyze_metadata(self, image_path):
        if not exifread:
            return {'has_metadata': False, 'is_trusted_camera': False, 'camera_info': 'None',
                    'was_edited': False, 'metadata_count': 0, 'risk': 52}
        try:
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)
            make = str(tags.get('Image Make', tags.get('EXIF Make', ''))).strip()
            model = str(tags.get('Image Model', tags.get('EXIF Model', ''))).strip()
            software = str(tags.get('Image Software', '')).lower().strip()
            trusted_brands = ['samsung', 'apple', 'iphone', 'google', 'pixel', 'huawei',
                              'tecno', 'infinix', 'oppo', 'xiaomi', 'vivo', 'canon', 'nikon', 'sony']
            is_trusted_camera = any(b in make.lower() or b in model.lower() for b in trusted_brands)
            was_edited = any(sw in software for sw in ['photoshop', 'gimp', 'paint.net', 'lightroom', 'affinity'])
            if is_trusted_camera:
                risk = 12 if not was_edited else 25
            elif make or model:
                risk = 35
            else:
                risk = 25
            return {
                'has_metadata': bool(make or model), 'is_trusted_camera': is_trusted_camera,
                'camera_info': f"{make} {model}".strip() or 'None', 'was_edited': was_edited,
                'editing_software': software if was_edited else None,
                'metadata_count': len(tags), 'risk': risk
            }
        except Exception:
            return {'has_metadata': False, 'is_trusted_camera': False, 'camera_info': 'None',
                    'was_edited': False, 'metadata_count': 0, 'risk': 52}

    def analyze_face_texture(self, image_path, sharpness):
        try:
            from deepface import DeepFace
            faces = DeepFace.extract_faces(image_path, enforce_detection=False)
            if not faces:
                return {'faces_detected': 0, 'risk': 0, 'assessment': 'NO_FACE'}
            max_risk = 0
            best_assessment = 'NORMAL'
            for face_data in faces:
                face_img = face_data['face']
                if isinstance(face_img, np.ndarray):
                    if face_img.max() <= 1.0:
                        face_img = (face_img * 255).astype(np.uint8)
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
                    if risk > max_risk:
                        max_risk = risk; best_assessment = assessment
            return {'faces_detected': len(faces), 'risk': max_risk, 'assessment': best_assessment}
        except Exception:
            return {'faces_detected': 0, 'risk': 0, 'assessment': 'ERROR'}

    def get_sharpness(self, image_path):
        try:
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                return 50.0
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:
            return 50.0

    def noise_analysis(self, image_path):
        try:
            from scipy import fft
            img = Image.open(image_path).convert('L')
            img.thumbnail((256, 256), Image.Resampling.LANCZOS)
            img_array = np.array(img).astype(np.float32)
            noise_score = float(np.mean(np.abs(
                fft.fftshift(fft.fft2(img_array))
            )[img_array.shape[0] // 3:, img_array.shape[1] // 3:]))
            if noise_score < 12:
                return {'noise_score': noise_score, 'risk': 25, 'assessment': 'VERY_LOW'}
            elif noise_score < 20:
                return {'noise_score': noise_score, 'risk': 10, 'assessment': 'LOW'}
            return {'noise_score': noise_score, 'risk': 0, 'assessment': 'NORMAL'}
        except Exception:
            return {'noise_score': 0, 'risk': 0, 'assessment': 'UNKNOWN'}

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

        if ai_result['available']:
            total_risk += ai_result['fake_confidence'] * 100 * 0.60
            confidence_sum += 0.60
            findings.append(
                f"AI Model: {'DEEPFAKE' if ai_result['fake_confidence'] > 0.5 else 'Authentic'} "
                f"({ai_result['fake_confidence']:.1%} confidence)"
            )

        total_risk += meta_result['risk'] * 0.15
        confidence_sum += 0.15
        findings.append(
            f"Camera: {meta_result['camera_info']}" if meta_result['is_trusted_camera']
            else "No trusted camera metadata"
        )

        total_risk += ela_result['risk'] * 0.15
        confidence_sum += 0.15

        if face_result['faces_detected'] > 0:
            total_risk += face_result['risk'] * 0.07
            confidence_sum += 0.07

        total_risk += noise_result['risk'] * 0.03
        confidence_sum += 0.03

        final_risk = min(max(total_risk, 0), 100)
        verdict = "LIKELY_DEEPFAKE" if final_risk > 65 else "AUTHENTIC" if final_risk < 40 else "REVIEW_REQUIRED"

        kenya_warnings = []
        if final_risk > 70 and face_result['faces_detected'] > 0:
            kenya_warnings.append({
                'type': 'ELECTION_MANIPULATION', 'severity': 'CRITICAL',
                'warning': 'This image may be a deepfake of a public figure. Verify with official sources before sharing.',
                'action': 'Report to NCIC: complaints@cohesion.or.ke | DCI: reportcrime@dci.go.ke'
            })
        if final_risk > 50:
            kenya_warnings.append({
                'type': 'MEDIA_MANIPULATION', 'severity': 'HIGH',
                'warning': 'This image shows signs of manipulation. Doctored news screenshots and fake campaign posters are common misinformation vectors.',
                'action': 'Verify at the original news outlet website. Check PesaCheck.org for fact-checks.'
            })

        return {
            'risk_score': round(final_risk, 1),
            'verdict': verdict,
            'confidence': round(max(0.6, confidence_sum), 2),
            'findings': findings,
            'kenya_warnings': kenya_warnings,
            'details': {
                'ai_confidence': round(ai_result['fake_confidence'] * 100, 1) if ai_result['available'] else 0,
                'ela_score': round(ela_result.get('ela_score', 0), 1)
            }
        }


class UltraAudioDetector:
    def __init__(self):
        self.sample_rate = 16000
        self.model = None
        self.feature_extractor = None
        self.lock = threading.Lock()
        logger.info("Ultra-Accurate Audio Detector initialized")

    def _load_model(self):
        """Lazy-load the WavLM deepfake detection model on first use."""
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
        model_dir = os.path.join(MODELS_DIR, "audio_model")
        if os.path.exists(os.path.join(model_dir, "config.json")):
            logger.info("Loading WavLM audio deepfake model from local files...")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)
            self.model = AutoModelForAudioClassification.from_pretrained(model_dir)
        else:
            logger.info("Loading WavLM audio deepfake model from HuggingFace...")
            model_name = "Hemg/wavlm-base-deepfake-detection"
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            self.model = AutoModelForAudioClassification.from_pretrained(model_name)
        self.model.eval()
        logger.info("WavLM audio model loaded successfully")

    def analyze_audio(self, audio_path):
        import librosa
        import torch
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            if len(y) < 2048:
                y = np.pad(y, (0, 2048 - len(y)))

            # --- AI Model Inference (primary signal — 70% weight) ---
            ai_risk = 50.0
            ai_confidence = 0.5
            ai_available = False
            try:
                with self.lock:
                    self._load_model()
                # Process audio through WavLM feature extractor
                inputs = self.feature_extractor(
                    y, sampling_rate=self.sample_rate, return_tensors="pt", padding=True
                )
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                probs = torch.nn.functional.softmax(logits, dim=-1)[0]
                # Model outputs: index 0 = real/bonafide, index 1 = fake/spoof
                num_labels = probs.shape[0]
                if num_labels >= 2:
                    fake_prob = float(probs[1])
                    real_prob = float(probs[0])
                else:
                    fake_prob = float(probs[0])
                    real_prob = 1.0 - fake_prob
                ai_risk = fake_prob * 100.0
                ai_confidence = max(fake_prob, real_prob)
                ai_available = True
                logger.info(f"Audio AI model: fake_prob={fake_prob:.3f}, real_prob={real_prob:.3f}")
            except Exception as e:
                logger.warning(f"Audio AI model inference failed, falling back to heuristics: {e}")

            # --- Heuristic signals (secondary — 30% weight) ---
            mfcc_var = float(np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)))
            rms = librosa.feature.rms(y=y)[0]
            silence_ratio = float(np.sum(rms < (np.mean(rms) * 0.12)) / len(rms))

            heuristic_risk = 0
            findings = []

            if mfcc_var < 150:
                heuristic_risk += 75; findings.append("Robotic voice texture detected (MFCC analysis)")
            elif mfcc_var < 400:
                heuristic_risk += 40; findings.append("Smooth voice texture (MFCC analysis)")
            else:
                findings.append("Natural voice variation detected")

            if not (0.02 < silence_ratio < 0.15):
                heuristic_risk += 30; findings.append("Abnormal breathing/pause pattern")

            heuristic_risk = min(heuristic_risk, 100)

            # --- Combine signals ---
            if ai_available:
                final_risk = (ai_risk * 0.70) + (heuristic_risk * 0.30)
                findings.insert(0, f"WavLM neural model confidence: {ai_confidence:.1%}")
            else:
                final_risk = heuristic_risk
                findings.insert(0, "Heuristic analysis only (model unavailable)")

            final_risk = min(round(final_risk, 1), 98)
            confidence = round(ai_confidence if ai_available else max(0.55, min(0.75, heuristic_risk / 100)), 2)

            verdict = 'LIKELY_DEEPFAKE' if final_risk > 65 else 'AUTHENTIC' if final_risk < 40 else 'REVIEW_REQUIRED'
            kenya_warnings = []

            language = 'english'
            kenya_audio_ctx = get_audio_kenya_context(language, final_risk)

            if final_risk > 65:
                kenya_warnings.append({
                    'type': 'AUDIO_MANIPULATION', 'severity': 'HIGH',
                    'warning': 'This audio shows signs of manipulation. "Leaked audio" of politicians is a common tactic — verify with official channels.',
                    'action': 'Report to DCI Cybercrime: reportcrime@dci.go.ke | Verify with at least 2 news outlets'
                })
            elif final_risk > 40:
                kenya_warnings.append({
                    'type': 'AUDIO_SUSPICIOUS', 'severity': 'MEDIUM',
                    'warning': 'This audio has some manipulation indicators.',
                    'action': 'Verify the full context of this recording before sharing'
                })

            return {
                'risk_score': final_risk,
                'is_authentic': final_risk < 50,
                'verdict': verdict,
                'confidence': confidence,
                'findings': findings,
                'kenya_warnings': kenya_warnings,
                'kenya_audio_context': kenya_audio_ctx,
                'detection_note': kenya_audio_ctx.get('detection_note', ''),
                'details': {
                    'ai_model_used': ai_available,
                    'ai_fake_probability': round(ai_risk, 1) if ai_available else None,
                    'mfcc_variance': round(mfcc_var, 1),
                    'silence_ratio': round(silence_ratio, 3)
                }
            }
        except Exception as e:
            return {'risk_score': 0, 'is_authentic': True, 'verdict': 'ERROR', 'error': str(e),
                    'confidence': 0, 'findings': [], 'kenya_warnings': [], 'details': {}}


class UltraTextDetector:
    def __init__(self):
        self.pipeline = None
        self.lock = threading.Lock()
        logger.info("Ultra-Accurate Text Detector initialized")

    def analyze_text(self, text):
        with self.lock:
            if self.pipeline is None:
                from transformers import pipeline
                local_model_dir = os.path.join(MODELS_DIR, "text_model")
                use_local = os.path.exists(os.path.join(local_model_dir, "config.json"))
                model_source = local_model_dir if use_local else "hamzab/roberta-fake-news-classification"
                self.pipeline = pipeline("text-classification", model=model_source, tokenizer=model_source)

        ai_result = self.pipeline(text[:512])[0]
        is_fake = ai_result['label'] in ['FAKE', 'LABEL_0', '0']
        confidence = ai_result['score']
        risk = int(confidence * 100) if is_fake else int((1 - confidence) * 100)

        txt_lower = text.lower()
        clickbait_count = sum(1 for kw in ['exposed', 'shocking', 'secret'] if kw in txt_lower)
        if clickbait_count > 0:
            risk = min(risk + 20, 96)

        return {
            'risk_score': risk,
            'is_authentic': risk < 50,
            'verdict': 'LIKELY_DEEPFAKE' if risk > 65 else 'AUTHENTIC' if risk < 40 else 'REVIEW_REQUIRED',
            'confidence': confidence,
            'findings': [f"AI Result: {ai_result['label']}"],
            'kenya_warnings': [],
            'details': {'ai_label': ai_result['label'], 'ai_score': round(confidence, 3)}
        }


# Initialize detectors
image_detector = UltraImageDetector()
audio_detector = UltraAudioDetector()
text_detector = UltraTextDetector()

# ══════════════════════════════════════════════════════════════
#  AUTH ROUTES (WEB — Templates)
# ══════════════════════════════════════════════════════════════

@app.route('/')
def home():
    user = session.get('user')
    if user:
        return redirect(FRONTEND_URL)
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        if not email or not password:
            flash('Please fill in all fields', 'error')
            return redirect(url_for('login'))

        db = get_db()
        user = db.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        if not user or not check_password_hash(user['password'], password):
            flash('Invalid email or password', 'error')
            return redirect(url_for('login'))

        # Update last_login
        db.execute('UPDATE users SET last_login = ? WHERE id = ?', (datetime.now(timezone.utc).isoformat(), user['id']))
        db.commit()

        session['user'] = {
            'name': user['name'],
            'email': email,
            'picture': user['profile_picture'] or None
        }
        return redirect(url_for('home'))

    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not username or not email or not password or not confirm_password:
            flash('Please fill in all fields', 'error')
            return redirect(url_for('signup'))

        if len(username) < 2:
            flash('Name must be at least 2 characters', 'error')
            return redirect(url_for('signup'))

        if '@' not in email or '.' not in email:
            flash('Please enter a valid email address', 'error')
            return redirect(url_for('signup'))

        if len(password) < 8:
            flash('Password must be at least 8 characters', 'error')
            return redirect(url_for('signup'))

        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('signup'))

        db = get_db()
        existing = db.execute('SELECT id FROM users WHERE email = ?', (email,)).fetchone()
        if existing:
            flash('An account with this email already exists', 'error')
            return redirect(url_for('signup'))

        db.execute(
            'INSERT INTO users (name, email, password, auth_provider, created_at) VALUES (?, ?, ?, ?, ?)',
            (username, email, hash_password(password), 'local', datetime.now(timezone.utc).isoformat())
        )
        db.commit()

        flash('Account created successfully! Please sign in.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


# ─── OAuth Routes ───
if HAS_OAUTH:
    @app.route('/auth/google')
    def login_google():
        redirect_uri = FRONTEND_URL + '/auth/google/callback'
        return google.authorize_redirect(redirect_uri)

    @app.route('/auth/google/callback')
    def google_callback():
        token = google.authorize_access_token()
        user_info = token['userinfo']

        db = get_db()
        existing = db.execute('SELECT * FROM users WHERE email = ?', (user_info['email'],)).fetchone()
        if not existing:
            db.execute(
                'INSERT INTO users (name, email, password, profile_picture, auth_provider, created_at) VALUES (?, ?, ?, ?, ?, ?)',
                (user_info['name'], user_info['email'], '', user_info.get('picture', ''), 'google', datetime.now(timezone.utc).isoformat())
            )
        else:
            db.execute(
                'UPDATE users SET last_login = ?, profile_picture = COALESCE(NULLIF(?, ""), profile_picture) WHERE email = ?',
                (datetime.now(timezone.utc).isoformat(), user_info.get('picture', ''), user_info['email'])
            )
        db.commit()

        session['user'] = {
            'name': user_info['name'],
            'email': user_info['email'],
            'picture': user_info.get('picture')
        }
        return redirect(FRONTEND_URL)

    @app.route('/auth/github')
    def login_github():
        redirect_uri = FRONTEND_URL + '/auth/github/callback'
        return github.authorize_redirect(redirect_uri)

    @app.route('/auth/github/callback')
    def github_callback():
        token = github.authorize_access_token()
        resp = github.get('user')
        user_info = resp.json()

        name = user_info.get('name') or user_info.get('login')
        email = user_info.get('email') or f"{user_info.get('login')}@github.local"
        picture = user_info.get('avatar_url', '')

        db = get_db()
        existing = db.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        if not existing:
            db.execute(
                'INSERT INTO users (name, email, password, profile_picture, auth_provider, created_at) VALUES (?, ?, ?, ?, ?, ?)',
                (name, email, '', picture, 'github', datetime.now(timezone.utc).isoformat())
            )
        else:
            db.execute(
                'UPDATE users SET last_login = ?, profile_picture = COALESCE(NULLIF(?, ""), profile_picture) WHERE email = ?',
                (datetime.now(timezone.utc).isoformat(), picture, email)
            )
        db.commit()

        session['user'] = {
            'name': name,
            'email': email,
            'picture': picture
        }
        return redirect(FRONTEND_URL)


# ══════════════════════════════════════════════════════════════
#  API — AUTH
# ══════════════════════════════════════════════════════════════

@app.route('/api/me')
def get_current_user():
    user = session.get('user')
    if not user:
        return jsonify({'user': None}), 200
    return jsonify({'user': user}), 200


@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    email = (data or {}).get('email', '').strip().lower()
    password = (data or {}).get('password', '')

    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400

    db = get_db()
    user = db.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    if not user or not check_password_hash(user['password'], password):
        return jsonify({'error': 'Invalid credentials'}), 401

    db.execute('UPDATE users SET last_login = ? WHERE id = ?', (datetime.now(timezone.utc).isoformat(), user['id']))
    db.commit()

    session['user'] = {
        'name': user['name'],
        'email': user['email'],
        'picture': user['profile_picture'] or None
    }

    token = create_access_token(identity=user['email'])
    return jsonify({
        'access_token': token,
        'user': {
            'name': user['name'],
            'email': user['email'],
            'picture': user['profile_picture'] or None
        }
    }), 200


# ══════════════════════════════════════════════════════════════
#  API — USER PROFILE
# ══════════════════════════════════════════════════════════════

@app.route('/api/profile', methods=['GET'])
def get_profile():
    """Get the current user's full profile."""
    user = session.get('user')
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401

    db = get_db()
    row = db.execute(
        'SELECT id, name, email, bio, phone, location, organization, role, '
        'profile_picture, auth_provider, created_at, updated_at, last_login '
        'FROM users WHERE email = ?',
        (user['email'],)
    ).fetchone()

    if not row:
        return jsonify({'error': 'User not found'}), 404

    # Get detection stats
    user_id = row['id']
    stats = db.execute('''
        SELECT
            COUNT(*) as total_scans,
            COALESCE(SUM(CASE WHEN verdict = 'LIKELY_DEEPFAKE' THEN 1 ELSE 0 END), 0) as threats_detected,
            COALESCE(SUM(CASE WHEN verdict = 'AUTHENTIC' THEN 1 ELSE 0 END), 0) as authentic_count,
            COALESCE(SUM(CASE WHEN detection_type = 'image' THEN 1 ELSE 0 END), 0) as image_scans,
            COALESCE(SUM(CASE WHEN detection_type = 'audio' THEN 1 ELSE 0 END), 0) as audio_scans,
            COALESCE(SUM(CASE WHEN detection_type = 'text' THEN 1 ELSE 0 END), 0) as text_scans,
            COALESCE(SUM(CASE WHEN detection_type = 'forward' THEN 1 ELSE 0 END), 0) as forward_scans,
            COALESCE(SUM(CASE WHEN detection_type = 'document' THEN 1 ELSE 0 END), 0) as document_scans,
            COALESCE(AVG(risk_score), 0) as avg_risk_score
        FROM detection_history WHERE user_id = ?
    ''', (user_id,)).fetchone()

    return jsonify({
        'profile': {
            'id': row['id'],
            'name': row['name'],
            'email': row['email'],
            'bio': row['bio'] or '',
            'phone': row['phone'] or '',
            'location': row['location'] or '',
            'organization': row['organization'] or '',
            'role': row['role'] or 'user',
            'profile_picture': row['profile_picture'] or '',
            'auth_provider': row['auth_provider'] or 'local',
            'created_at': row['created_at'],
            'updated_at': row['updated_at'],
            'last_login': row['last_login'],
        },
        'stats': {
            'total_scans': stats['total_scans'],
            'threats_detected': stats['threats_detected'],
            'authentic_count': stats['authentic_count'],
            'image_scans': stats['image_scans'],
            'audio_scans': stats['audio_scans'],
            'text_scans': stats['text_scans'],
            'forward_scans': stats['forward_scans'],
            'document_scans': stats['document_scans'],
            'avg_risk_score': round(stats['avg_risk_score'], 1),
        }
    }), 200


@app.route('/api/profile', methods=['PUT'])
def update_profile():
    """Update the current user's profile."""
    user = session.get('user')
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    db = get_db()
    row = db.execute('SELECT id FROM users WHERE email = ?', (user['email'],)).fetchone()
    if not row:
        return jsonify({'error': 'User not found'}), 404

    # Only allow updating certain fields
    allowed_fields = ['name', 'bio', 'phone', 'location', 'organization']
    updates = []
    values = []

    for field in allowed_fields:
        if field in data:
            updates.append(f'{field} = ?')
            values.append(data[field])

    if not updates:
        return jsonify({'error': 'No valid fields to update'}), 400

    updates.append('updated_at = ?')
    values.append(datetime.now(timezone.utc).isoformat())
    values.append(row['id'])

    db.execute(
        f'UPDATE users SET {", ".join(updates)} WHERE id = ?',
        values
    )
    db.commit()

    # Update session if name changed
    if 'name' in data:
        session['user']['name'] = data['name']

    return jsonify({'message': 'Profile updated successfully'}), 200


@app.route('/api/profile/password', methods=['PUT'])
def change_password():
    """Change the current user's password."""
    user = session.get('user')
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401

    data = request.get_json()
    current_password = (data or {}).get('current_password', '')
    new_password = (data or {}).get('new_password', '')

    if not current_password or not new_password:
        return jsonify({'error': 'Current and new password required'}), 400

    if len(new_password) < 8:
        return jsonify({'error': 'New password must be at least 8 characters'}), 400

    db = get_db()
    row = db.execute('SELECT id, password, auth_provider FROM users WHERE email = ?', (user['email'],)).fetchone()

    if not row:
        return jsonify({'error': 'User not found'}), 404

    if row['auth_provider'] != 'local':
        return jsonify({'error': 'Cannot change password for OAuth accounts'}), 400

    if not check_password_hash(row['password'], current_password):
        return jsonify({'error': 'Current password is incorrect'}), 401

    db.execute(
        'UPDATE users SET password = ?, updated_at = ? WHERE id = ?',
        (hash_password(new_password), datetime.now(timezone.utc).isoformat(), row['id'])
    )
    db.commit()

    return jsonify({'message': 'Password changed successfully'}), 200


@app.route('/api/profile/picture', methods=['POST'])
def upload_profile_picture():
    """Upload a profile picture."""
    user = session.get('user')
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    ext = os.path.splitext(secure_filename(file.filename))[1].lower()
    if ext not in ['.png', '.jpg', '.jpeg', '.webp']:
        return jsonify({'error': 'Invalid image format'}), 400

    # Save the file
    profile_pics_dir = os.path.join(UPLOAD_FOLDER, 'profile_pictures')
    os.makedirs(profile_pics_dir, exist_ok=True)

    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(profile_pics_dir, filename)
    file.save(filepath)

    # Update DB
    picture_url = f'/uploads/profile_pictures/{filename}'
    db = get_db()
    db.execute(
        'UPDATE users SET profile_picture = ?, updated_at = ? WHERE email = ?',
        (picture_url, datetime.now(timezone.utc).isoformat(), user['email'])
    )
    db.commit()

    # Update session
    session['user']['picture'] = picture_url

    return jsonify({'message': 'Profile picture updated', 'picture_url': picture_url}), 200


# ══════════════════════════════════════════════════════════════
#  API — DETECTION HISTORY
# ══════════════════════════════════════════════════════════════

@app.route('/api/history', methods=['GET'])
def get_detection_history():
    """Get detection history for the current user."""
    user = session.get('user')
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401

    user_id = get_current_user_id()
    if not user_id:
        return jsonify({'error': 'User not found'}), 404

    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    detection_type = request.args.get('type', None)

    per_page = min(per_page, 100)
    offset = (page - 1) * per_page

    db = get_db()

    query = 'SELECT * FROM detection_history WHERE user_id = ?'
    count_query = 'SELECT COUNT(*) as total FROM detection_history WHERE user_id = ?'
    params = [user_id]
    count_params = [user_id]

    if detection_type:
        query += ' AND detection_type = ?'
        count_query += ' AND detection_type = ?'
        params.append(detection_type)
        count_params.append(detection_type)

    query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
    params.extend([per_page, offset])

    rows = db.execute(query, params).fetchall()
    total = db.execute(count_query, count_params).fetchone()['total']

    history = []
    for row in rows:
        history.append({
            'id': row['id'],
            'detection_type': row['detection_type'],
            'filename': row['filename'],
            'risk_score': row['risk_score'],
            'verdict': row['verdict'],
            'confidence': row['confidence'],
            'findings': json.loads(row['findings'] or '[]'),
            'kenya_warnings': json.loads(row['kenya_warnings'] or '[]'),
            'details': json.loads(row['details'] or '{}'),
            'created_at': row['created_at'],
        })

    return jsonify({
        'history': history,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': total,
            'pages': (total + per_page - 1) // per_page
        }
    }), 200


@app.route('/api/history/<int:history_id>', methods=['DELETE'])
def delete_history_item(history_id):
    """Delete a detection history item."""
    user = session.get('user')
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401

    user_id = get_current_user_id()
    db = get_db()
    db.execute('DELETE FROM detection_history WHERE id = ? AND user_id = ?', (history_id, user_id))
    db.commit()

    return jsonify({'message': 'History item deleted'}), 200


# ══════════════════════════════════════════════════════════════
#  API — ALL USERS (for profile page user listing)
# ══════════════════════════════════════════════════════════════

@app.route('/api/users', methods=['GET'])
def get_all_users():
    """Get all signed-up users (for profile listing)."""
    user = session.get('user')
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401

    db = get_db()
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    per_page = min(per_page, 200)
    offset = (page - 1) * per_page

    total = db.execute('SELECT COUNT(*) as total FROM users').fetchone()['total']
    rows = db.execute(
        'SELECT id, name, email, bio, location, organization, role, profile_picture, '
        'auth_provider, created_at, last_login '
        'FROM users ORDER BY created_at DESC LIMIT ? OFFSET ?',
        (per_page, offset)
    ).fetchall()

    users = []
    for row in rows:
        scan_count = db.execute(
            'SELECT COUNT(*) as cnt FROM detection_history WHERE user_id = ?', (row['id'],)
        ).fetchone()['cnt']

        users.append({
            'id': row['id'],
            'name': row['name'],
            'email': row['email'],
            'bio': row['bio'] or '',
            'location': row['location'] or '',
            'organization': row['organization'] or '',
            'role': row['role'] or 'user',
            'profile_picture': row['profile_picture'] or '',
            'auth_provider': row['auth_provider'] or 'local',
            'created_at': row['created_at'],
            'last_login': row['last_login'],
            'total_scans': scan_count,
        })

    return jsonify({
        'users': users,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': total,
            'pages': (total + per_page - 1) // per_page
        }
    }), 200


# ══════════════════════════════════════════════════════════════
#  API — DETECTION ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.route('/api/analyze/image', methods=['POST'])
@rate_limit(config.RATELIMIT_ANALYSIS)
def analyze_image():
    file, err = validate_file_upload('file', config.ALLOWED_IMAGE_EXTENSIONS)
    if err:
        return err

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}_{secure_filename(file.filename)}")
    file.save(filepath)
    try:
        result = image_detector.analyze_image(filepath)
        user_id = get_current_user_id()
        save_detection_history(user_id, 'image', file.filename, result)
        return jsonify(result)
    except Exception as e:
        logger.error("Image analysis error: %s", e, exc_info=True)
        return jsonify({'error': 'Image analysis failed. Please try again.'}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/api/analyze/audio', methods=['POST'])
@rate_limit(config.RATELIMIT_ANALYSIS)
def analyze_audio():
    file, err = validate_file_upload('file', config.ALLOWED_AUDIO_EXTENSIONS)
    if err:
        return err

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}_{secure_filename(file.filename)}")
    file.save(filepath)
    try:
        result = audio_detector.analyze_audio(filepath)
        user_id = get_current_user_id()
        save_detection_history(user_id, 'audio', file.filename, result)
        return jsonify(result)
    except Exception as e:
        logger.error("Audio analysis error: %s", e, exc_info=True)
        return jsonify({'error': 'Audio analysis failed. Please try again.'}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/api/analyze/text', methods=['POST'])
@rate_limit(config.RATELIMIT_ANALYSIS)
def analyze_text():
    data, err = validate_json_body('text')
    if err:
        return err
    text = data['text']
    if len(text.strip()) < 5:
        return jsonify({'error': 'Text too short (minimum 5 characters)'}), 400

    try:
        result = text_detector.analyze_text(text)
        user_id = get_current_user_id()
        save_detection_history(user_id, 'text', f'text_{len(text)}_chars', result)
        return jsonify(result)
    except Exception as e:
        logger.error("Text analysis error: %s", e, exc_info=True)
        return jsonify({'error': 'Text analysis failed. Please try again.'}), 500


@app.route('/api/analyze/forward', methods=['POST'])
@rate_limit(config.RATELIMIT_ANALYSIS)
def analyze_whatsapp_forward():
    """Analyse a WhatsApp forward for misinformation patterns."""
    data, err = validate_json_body('text')
    if err:
        return err
    text = data['text']

    if len(text.strip()) < 10:
        return jsonify({'error': 'Text too short to analyse (minimum 10 characters)'}), 400

    forward_result = analyze_forward(text)
    ai_result = text_detector.analyze_text(text)

    combined_score = min(100, (
        forward_result['forward_risk_score'] * 0.4
        + ai_result.get('risk_score', 0) * 0.6
    ))

    kenya_warnings = []
    if combined_score > 65:
        kenya_warnings.append({
            'type': 'MISINFORMATION', 'severity': 'HIGH',
            'warning': '67% of Kenyans get news via WhatsApp — do not forward unverified content.',
            'action': 'Verify with PesaCheck.org or Africa Check before sharing'
        })
    if forward_result.get('hoax_matches'):
        for hoax in forward_result['hoax_matches']:
            kenya_warnings.append({
                'type': hoax['category'].upper().replace(' ', '_'), 'severity': 'HIGH',
                'warning': hoax['debunk'], 'action': 'Do not forward this message'
            })

    election_ctx = analyze_election_context(text, 'text', combined_score)
    if election_ctx.get('election_relevant'):
        for w in election_ctx.get('warnings', []):
            kenya_warnings.append({
                'type': 'ELECTION_MISINFORMATION', 'severity': election_ctx['risk_level'],
                'warning': w.get('en', ''),
                'action': '; '.join(election_ctx.get('recommendations', [])[:2])
            })

    result = {
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
            f'Hoax templates matched: {len(forward_result.get("hoax_matches", []))}',
        ],
        'kenya_warnings': kenya_warnings,
        'forward_analysis': forward_result,
        'is_authentic': combined_score < 40,
        'details': {
            'forward_score': forward_result['forward_risk_score'],
            'ai_score': ai_result.get('risk_score', 0),
            'hoax_count': len(forward_result.get('hoax_matches', [])),
        }
    }

    user_id = get_current_user_id()
    save_detection_history(user_id, 'forward', f'forward_{len(text)}_chars', result)
    return jsonify(result)


@app.route('/api/analyze/document', methods=['POST'])
@rate_limit(config.RATELIMIT_ANALYSIS)
def analyze_document():
    """Analyse an uploaded image for Kenyan document forgery."""
    file, err = validate_file_upload('file', config.ALLOWED_DOC_EXTENSIONS)
    if err:
        return err

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}_{secure_filename(file.filename)}")
    file.save(filepath)

    try:
        image_result = image_detector.analyze_image(filepath)
        ocr_text = ''
        try:
            import pytesseract
            ocr_text = pytesseract.image_to_string(Image.open(filepath))
        except ImportError:
            logger.warning('pytesseract not installed — document OCR unavailable')
        except Exception as e:
            logger.warning(f'OCR failed: {e}')

        ela_score = image_result.get('details', {}).get('ela_score', 0)
        doc_result = analyze_kenya_document(
            text=ocr_text,
            image_risk_score=image_result.get('risk_score', 0),
            ela_score=ela_score,
        )

        screenshot_result = detect_news_screenshot(
            ocr_text=ocr_text,
            ela_score=ela_score,
            ai_deepfake_score=image_result.get('risk_score', 0),
        )

        kenya_warnings = image_result.get('kenya_warnings', [])
        if doc_result.get('is_document') and doc_result.get('verdict') != 'APPEARS_GENUINE':
            kenya_warnings.append({
                'type': 'DOCUMENT_FORGERY',
                'severity': 'CRITICAL' if doc_result.get('verdict') == 'LIKELY_FORGED' else 'HIGH',
                'warning': f"{doc_result['document_name']}: {doc_result.get('kenya_context', {}).get('message_en', 'Possible forgery detected')}",
                'action': f"Verify at: {doc_result.get('kenya_context', {}).get('verify_at', 'N/A')}"
            })
        if screenshot_result.get('is_news_screenshot') and screenshot_result.get('verdict') != 'APPEARS_GENUINE':
            outlet = screenshot_result.get('detected_outlet', {})
            kenya_warnings.append({
                'type': 'FAKE_NEWS_SCREENSHOT',
                'severity': 'CRITICAL' if screenshot_result['verdict'] == 'LIKELY_MANIPULATED' else 'HIGH',
                'warning': f"Manipulated {outlet.get('name', 'news')} screenshot detected.",
                'action': screenshot_result.get('action', {}).get('en', f"Verify at {outlet.get('verify_url', '')}")
            })

        risk_score = image_result.get('risk_score', 0)
        if doc_result.get('is_document'):
            risk_score = max(risk_score, doc_result.get('risk_score', 0))

        result = {
            'risk_score': round(risk_score, 1),
            'verdict': doc_result.get('verdict', image_result.get('verdict', 'REVIEW_REQUIRED')),
            'confidence': image_result.get('confidence', 0.6),
            'findings': image_result.get('findings', []) + [
                f"Document type: {doc_result.get('document_name', 'Unknown')}" if doc_result.get('is_document') else 'No recognised Kenyan document detected',
            ],
            'kenya_warnings': kenya_warnings,
            'document_analysis': doc_result,
            'screenshot_analysis': screenshot_result,
            'is_authentic': risk_score < 40,
            'details': image_result.get('details', {}),
        }

        user_id = get_current_user_id()
        save_detection_history(user_id, 'document', file.filename, result)
        return jsonify(result)
    except Exception as e:
        logger.error(f'Document analysis error: {e}')
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


# ══════════════════════════════════════════════════════════════
#  STATIC FILE SERVING
# ══════════════════════════════════════════════════════════════

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# ══════════════════════════════════════════════════════════════
#  HEALTH CHECK
# ══════════════════════════════════════════════════════════════

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'models': {
            'image_loaded': image_detector.ai_model is not None and image_detector.ai_model != 'unavailable',
            'audio_loaded': audio_detector.model is not None,
            'text_loaded': text_detector.pipeline is not None,
        },
        'platform': 'SafEye Kenya',
        'version': config.APP_VERSION,
        'environment': os.getenv('FLASK_ENV', 'production'),
        'modules': {
            'auth': True,
            'profile': True,
            'detection_history': True,
            'election_shield': True,
            'whatsapp_checker': True,
            'document_verifier': True,
            'news_screenshot_detector': True,
            'audio_context': True,
        }
    })


# ══════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    port = int(os.getenv('PORT', 7860))
    host = os.getenv('HOST', '0.0.0.0')
    is_debug = os.getenv('FLASK_ENV', 'production').lower() == 'development'

    logger.info("SafEye v%s starting on %s:%d (debug=%s)", config.APP_VERSION, host, port, is_debug)
    print("=" * 55)
    print(f"  SafEye — Unified Backend v{config.APP_VERSION}")
    print("  Auth + Detection API + Profiles + History")
    print(f"  Environment: {os.getenv('FLASK_ENV', 'production')}")
    print(f"  Running on http://{host}:{port}")
    print("=" * 55)
    app.run(host=host, port=port, debug=is_debug)
