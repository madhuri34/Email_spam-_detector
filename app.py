import os
import re
import joblib
import numpy as np
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import random
from datetime import datetime
from collections import deque

# NLTK imports for text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Text Preprocessing Setup ---
try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
except Exception as e:
    print(f"Error downloading NLTK stopwords: {e}")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK tokenizer...")
    nltk.download('punkt')
except Exception as e:
    print(f"Error downloading NLTK tokenizer: {e}")

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_strong_random_secret_key'

# --- Flask-Login Setup ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# In-memory storage for users and predictions
users = {
    "testuser": {"password": generate_password_hash("password123"), "id": "1", "username": "testuser"}
}
user_id_counter = 1
# Use a deque for efficient adding and removing of recent predictions
recent_predictions = deque(maxlen=10) # Stores the last 10 predictions

class User(UserMixin):
    def __init__(self, user_id, username):
        self.id = user_id
        self.username = username

    def get_id(self):
        return str(self.id)

    def get_username(self):
        return self.username

@login_manager.user_loader
def load_user(user_id):
    for user_data in users.values():
        if user_data["id"] == user_id:
            return User(user_data["id"], user_data["username"])
    return None

# --- Load Model and Vectorizer ---
MODEL_DIR = 'model'
CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'spam_classifier.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')

model, vectorizer = None, None
try:
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    if os.path.exists(CLASSIFIER_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(CLASSIFIER_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("Model and vectorizer loaded successfully.")
    else:
        print("Warning: Model or vectorizer files not found.")
except Exception as e:
    print(f"Error loading model files: {e}")

# --- Text Preprocessing Function ---
def preprocess_text(text):
    if not text: return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

# --- Flask Routes ---
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username, password = request.form.get('username'), request.form.get('password')
        user_data = users.get(username)
        if user_data and check_password_hash(user_data['password'], password):
            user = User(user_data['id'], user_data['username'])
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    global user_id_counter
    if request.method == 'POST':
        username, password = request.form.get('username'), request.form.get('password')
        if not username or not password:
            flash('Please fill out all fields.', 'danger')
            return render_template('register.html')
        if username in users:
            flash('Username already taken.', 'danger')
            return render_template('register.html')
        user_id_counter += 1
        users[username] = {"password": generate_password_hash(password), "id": str(user_id_counter), "username": username}
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=current_user.get_username())

@app.route('/dashboard_data')
@login_required
def dashboard_data():
    total_emails = random.randint(1000, 5000)
    spam_count = random.randint(int(total_emails * 0.1), int(total_emails * 0.4))
    ham_count = total_emails - spam_count

    daily_data = []
    for i in range(7):
        daily_total = random.randint(100, 300)
        daily_spam = random.randint(10, 80)
        if daily_spam > daily_total: daily_spam = random.randint(0, daily_total)
        daily_ham = daily_total - daily_spam
        daily_data.append({"day": f"Day {i+1}", "total": daily_total, "spam": daily_spam, "ham": daily_ham})

    return jsonify({
        'total_emails': total_emails,
        'spam_count': spam_count,
        'ham_count': ham_count,
        'daily_stats': daily_data,
        'recent_predictions': list(recent_predictions) # Return a list of the recent predictions
    })

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'GET':
        return render_template('predict.html')

    if not hasattr(model, 'predict_proba') or not model or not vectorizer:
        return jsonify({'error': 'Model is not loaded.'}), 500

    try:
        message = request.json.get('message', '') if request.is_json else request.form.get('message', '')
        if not message:
            return jsonify({'error': 'No message provided.'}), 400

        processed_message = preprocess_text(message)
        vectorized_message = vectorizer.transform([processed_message])
        probabilities = model.predict_proba(vectorized_message)[0]
        prediction_index = np.argmax(probabilities)
        confidence = probabilities[prediction_index]

        result = "Spam" if prediction_index == 1 else "Not Spam (Ham)"
        confidence_percent = f"{confidence * 100:.1f}%"

        # Store the prediction
        prediction_record = {
            "message": message,
            "prediction": result,
            "confidence": confidence_percent,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        recent_predictions.appendleft(prediction_record) # Add to the left of the deque

        return jsonify({'prediction': result, 'confidence': confidence_percent, 'message': message})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
Footer
©
