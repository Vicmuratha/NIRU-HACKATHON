import os
import hashlib
import sqlite3
from flask import Flask, render_template, url_for, redirect, session, jsonify, request, flash, g
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv

# 1. Load keys from .env file
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_URL = os.getenv('FRONTEND_URL', 'http://localhost:3000')
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or os.getenv("SECRET_KEY", "super_secret_hackathon_key")

# Allow OAuth over HTTP for local testing (REMOVE THIS IN PRODUCTION)
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# 2. Configure OAuth
oauth = OAuth(app)

# --- GOOGLE CONFIGURATION ---
google = oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

# --- GITHUB CONFIGURATION ---
# (If you haven't added GitHub keys to .env yet, this will just stay inactive)
github = oauth.register(
    name='github',
    client_id=os.getenv("GITHUB_CLIENT_ID"),
    client_secret=os.getenv("GITHUB_CLIENT_SECRET"),
    access_token_url='https://github.com/login/oauth/access_token',
    authorize_url='https://github.com/login/oauth/authorize',
    client_kwargs={'scope': 'user:email'},
)

# ─── SQLite Database ───
DB_PATH = os.path.join(BASE_DIR, 'users.db')

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
    db = sqlite3.connect(DB_PATH)
    db.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    db.commit()
    db.close()

init_db()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

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
        if not user or user['password'] != hash_password(password):
            flash('Invalid email or password', 'error')
            return redirect(url_for('login'))

        session['user'] = {
            'name': user['name'],
            'email': email,
            'picture': None
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

        # Validation
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

        # Store user
        db.execute(
            'INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
            (username, email, hash_password(password))
        )
        db.commit()

        flash('Account created successfully! Please sign in.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/api/me')
def get_current_user():
    user = session.get('user')
    if not user:
        return jsonify({'user': None}), 200
    return jsonify({'user': user}), 200

# --- GOOGLE ROUTES ---
@app.route('/auth/google')
def login_google():
    # Redirect to Google's Login Page
    redirect_uri = url_for('google_callback', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/auth/google/callback')
def google_callback():
    token = google.authorize_access_token()
    user_info = token['userinfo']
    
    # Save user to session
    session['user'] = {
        'name': user_info['name'],
        'email': user_info['email'],
        'picture': user_info.get('picture')
    }
    return redirect(url_for('home'))

# --- GITHUB ROUTES ---
@app.route('/auth/github')
def login_github():
    redirect_uri = url_for('github_callback', _external=True)
    return github.authorize_redirect(redirect_uri)

@app.route('/auth/github/callback')
def github_callback():
    token = github.authorize_access_token()
    resp = github.get('user')
    user_info = resp.json()
    
    session['user'] = {
        'name': user_info.get('name') or user_info.get('login'),
        'email': user_info.get('email'),
        'picture': user_info.get('avatar_url')
    }
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
