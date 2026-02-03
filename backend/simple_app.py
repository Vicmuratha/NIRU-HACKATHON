import os
import sys
import uuid
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from logic import analyze_media, error_result
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

@app.route("/")
def home():
    return jsonify({
        "system": "SafEye - AI Detection API",
        "version": "2.1 (Auto-Download Models)",
        "endpoints": {
            "/api/analyze/image": "POST - Image detection",
            "/api/analyze/audio": "POST - Audio detection",
            "/api/analyze/text": "POST - Text detection (file or JSON)"
        }
    })

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })

@app.route("/api/analyze/image", methods=["POST"])
def analyze_image():
    filepath = None
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        ext = os.path.splitext(secure_filename(file.filename))[1].lower()
        if ext not in [".png", ".jpg", ".jpeg", ".webp"]:
            return jsonify({"error": "Invalid image format"}), 400

        unique_filename = f"{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
        file.save(filepath)

        result = analyze_media(file.filename, filepath)
        return jsonify(result)
    except Exception as e:
        return jsonify(error_result(str(e))), 500
    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass

@app.route("/api/analyze/audio", methods=["POST"])
def analyze_audio():
    filepath = None
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        ext = os.path.splitext(secure_filename(file.filename))[1].lower()
        if ext not in [".mp3", ".wav", ".ogg", ".flac"]:
            return jsonify({"error": "Invalid audio format"}), 400

        unique_filename = f"{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
        file.save(filepath)

        result = analyze_media(file.filename, filepath)
        return jsonify(result)
    except Exception as e:
        return jsonify(error_result(str(e))), 500
    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass

@app.route("/api/analyze/text", methods=["POST"])
def analyze_text():
    filepath = None
    try:
        if "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "No file selected"}), 400

            ext = os.path.splitext(secure_filename(file.filename))[1].lower()
            if ext != ".txt":
                return jsonify({"error": "Invalid text format"}), 400

            unique_filename = f"{uuid.uuid4().hex}{ext}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
            file.save(filepath)
            result = analyze_media(file.filename, filepath)
            return jsonify(result)

        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        unique_filename = f"{uuid.uuid4().hex}.txt"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)

        result = analyze_media("input.txt", filepath)
        return jsonify(result)
    except Exception as e:
        return jsonify(error_result(str(e))), 500
    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
