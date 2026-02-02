import os
from flask import Flask, render_template, url_for, redirect, session
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

# Dummy Database
users_db = {}

@app.route('/')
def home():
    user = session.get('user')
    if user:
        return redirect(FRONTEND_URL)
    return redirect(url_for('login'))

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

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
