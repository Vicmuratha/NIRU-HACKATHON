# SafEye Backend - Main Application Entry Point
# This file imports the Flask app from backend/app.py
# Run this file to start the server: python app.py
# OR run directly: python backend/app.py

import sys
import os
import importlib.util

# Get the path to backend/app.py
backend_app_path = os.path.join(os.path.dirname(__file__), 'backend', 'app.py')

# Load the module from backend/app.py
spec = importlib.util.spec_from_file_location("backend_app", backend_app_path)
backend_app = importlib.util.module_from_spec(spec)
sys.modules["backend_app"] = backend_app
spec.loader.exec_module(backend_app)

# Get the Flask app instance
app = backend_app.app

if __name__ == '__main__':
    print("=" * 60)
    print("SafEye - AI-Powered Deepfake Detection Platform")
    print("=" * 60)
    print("Starting server...")
    print("Note: Using backend/app.py as the main application")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
