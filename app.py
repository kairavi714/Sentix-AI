import os
import joblib
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask import jsonify

app = Flask(__name__)
app.secret_key = 'sentix-dev-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sentixai.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load the local Machine Learning model & TF-IDF vectorizer
print("Loading local sentiment model into memory...")
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')
print("Model ready!")

# --- Database Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    auto_save_history = db.Column(db.Boolean, default=True)
    strict_categorization = db.Column(db.Boolean, default=False)
    logs = db.relationship('SentimentLog', backref='user', lazy=True)

class SentimentLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    input_text = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(50), nullable=False)
    empathetic_response = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

with app.app_context():
    db.create_all()

# --- Local ML Inference Function ---
def analyze_text_local(text):
    # Transform text to TF-IDF feature matrix
    features = vectorizer.transform([text])
    
    # Calculate probability distribution across classes
    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)  # e.g., ['Negative', 'Neutral', 'Positive']
    
    prob_dict = {cls: int(round(prob * 100)) for cls, prob in zip(classes, probabilities)}
    
    pos = prob_dict.get('Positive', 0)
    neg = prob_dict.get('Negative', 0)
    neu = prob_dict.get('Neutral', 0)
    
    # Ensure total adds up cleanly to 100%
    total = pos + neg + neu
    if total != 100 and total > 0:
        neu += (100 - total)

    # Determine dominant sentiment
    primary = max(prob_dict, key=prob_dict.get)

    # Contextual insight mapping
    if primary == "Positive":
        response = f"The model detected high positive indicators ({pos}% confidence), reflecting an uplifting and favorable sentiment."
    elif primary == "Negative":
        response = f"The model detected strong negative signals ({neg}% confidence), pointing toward frustration or distress."
    else:
        response = f"The model determined this statement to be predominantly neutral ({neu}% confidence), indicating an objective or balanced tone."

    return {
        "primary": primary,
        "pos": max(0, pos),
        "neg": max(0, neg),
        "neu": max(0, neu),
        "response": response
    }

# --- Routes ---
@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if not user:
            user = User(username=username, password=password)
            db.session.add(user)
            db.session.commit()
            
        if user.password == password:
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid credentials")
            
    return render_template('login.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    result = None
    if request.method == 'POST':
        user_text = request.form.get('user_text')
        if user_text and user_text.strip():
            data = analyze_text_local(user_text)

            # 1. Fetch the current user to read their settings
            user = User.query.get(session['user_id'])
            
            # 2. Strict Categorization Math
            max_prob = max(data['pos'], data['neg'], data['neu'])
            if user.strict_categorization and max_prob < 40:
                data['primary'] = "Neutral"
                data['response'] = "The model detected conflicting emotions. Due to strict categorization, this is marked Neutral."

            # 3. Auto Save History Toggle
            if user.auto_save_history:
                log = SentimentLog(
                    input_text=user_text,
                    sentiment=data['primary'],
                    empathetic_response=data['response'],
                    user_id=session['user_id']
                )
                db.session.add(log)
                db.session.commit()
            
            result = {
                'input': user_text,
                'sentiment': data['primary'],
                'pos': data['pos'],
                'neg': data['neg'],
                'neu': data['neu'],
                'response': data['response']
            }
            
    return render_template('dashboard.html', result=result)

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    logs = SentimentLog.query.filter_by(user_id=session['user_id']).all()
    return render_template('history.html', logs=logs)

@app.route('/settings')
def settings():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('settings.html')

@app.route('/update_setting', methods=['POST'])
def update_setting():
    if 'user_id' not in session:
        return jsonify({"status": "error"}), 401
    
    user = User.query.get(session['user_id'])
    data = request.get_json()
    
    if data['setting'] == 'auto_save':
        user.auto_save_history = data['value']
    elif data['setting'] == 'strict_mode':
        user.strict_categorization = data['value']
        
    db.session.commit()
    return jsonify({"status": "success"})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)