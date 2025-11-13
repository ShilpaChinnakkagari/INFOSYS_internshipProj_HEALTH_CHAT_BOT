from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, date, timedelta
import json
import os
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from transformers import MarianMTModel, MarianTokenizer
import re
from functools import wraps
import csv

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'dev-secret-key-2023'
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL') or 'sqlite:///health_chatbot.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Load translation model
try:
    model_name = "Helsinki-NLP/opus-mt-en-hi"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    translation_model = MarianMTModel.from_pretrained(model_name)
    print("Translation model loaded successfully!")
except Exception as e:
    print(f"Error loading translation model: {e}")
    translation_model = None
    tokenizer = None

def translate_to_hindi(text):
    """Translate English text to Hindi using MarianMT"""
    if translation_model is None or tokenizer is None:
        return text
    
    try:
        # Split text into sentences for better translation
        sentences = re.split(r'[.!?]+', text)
        translated_sentences = []
        
        for sentence in sentences:
            if sentence.strip():
                # Tokenize and translate
                inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
                translated = translation_model.generate(**inputs)
                translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
                translated_sentences.append(translated_text)
        
        return ' '.join(translated_sentences)
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def is_hindi_text(text):
    """Check if text contains Hindi characters"""
    hindi_chars = set('अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहळक्षज्ञ')
    return any(char in hindi_chars for char in text)

def add_disclaimer(response, is_hindi=False):
    """Add appropriate disclaimer to the response"""
    disclaimer_en = "\n\n---\n**Disclaimer:** This health advice is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition."
    
    disclaimer_hi = "\n\n---\n**अस्वीकरण:** यह स्वास्थ्य सलाह केवल सूचनात्मक उद्देश्यों के लिए है और यह पेशेवर चिकित्सकीय सलाह, निदान, या उपचार का विकल्प नहीं है। किसी भी चिकित्सकीय स्थिति के संबंध में आपके कोई भी प्रश्न हो तो हमेशा अपने चिकित्सक या अन्य योग्य स्वास्थ्य सेवा प्रदाता की सलाह लें।"
    
    return response + (disclaimer_hi if is_hindi else disclaimer_en)

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    location = db.Column(db.String(200), nullable=False)
    language = db.Column(db.String(10), default='en')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)

    # Relationships
    health_scores = db.relationship('HealthScore', backref='user', lazy=True)
    chat_history = db.relationship('ChatHistory', backref='user', lazy=True)
    feedback = db.relationship('ChatFeedback', backref='user', lazy=True)
    emergencies = db.relationship('EmergencyLog', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class HealthScore(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    score = db.Column(db.Integer, nullable=False)
    date = db.Column(db.Date, default=date.today)
    notes = db.Column(db.Text)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship for feedback
    feedback = db.relationship('ChatFeedback', backref='chat', lazy=True, uselist=False)

class HealthTip(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(100), nullable=False)
    symptoms = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'))

class EmergencyLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    location = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(50), default='triggered')

class ChatFeedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    chat_id = db.Column(db.Integer, db.ForeignKey('chat_history.id'), nullable=False)
    feedback = db.Column(db.String(10))  # 'thumbs_up', 'thumbs_down'
    reason = db.Column(db.Text)  # Optional reason for thumbs down
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class SystemAnalytics(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    metric_name = db.Column(db.String(100), nullable=False)
    metric_value = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Admin required decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        if not current_user.is_admin:
            flash('Access denied. Admin privileges required.', 'danger')
            return redirect(url_for('user_dashboard'))
        return f(*args, **kwargs)
    return decorated_function

def init_db():
    with app.app_context():
        try:
            # Drop all tables and recreate them
            db.drop_all()
            db.create_all()
            print("Database tables created successfully!")
            
            # Create admin user
            if not User.query.filter_by(email='admin@healthbot.com').first():
                admin = User(
                    email='admin@healthbot.com',
                    name='Admin',
                    age=30,
                    location='HQ',
                    is_admin=True
                )
                admin.set_password('admin123')
                db.session.add(admin)
                db.session.commit()
                print("Admin user created!")

            # Create sample health tips
            if not HealthTip.query.first():
                sample_tips = [
                    HealthTip(
                        title='Migraine Relief',
                        content='For migraine relief:\n• Rest in a quiet, dark room\n• Apply cold compresses to your head\n• Stay hydrated\n• Avoid bright lights and loud sounds\n• Consider over-the-counter pain relievers\n• Practice relaxation techniques',
                        category='head_pain',
                        symptoms='migraine,headache,head pain,सिरदर्द,माइग्रेन',
                        created_by=1
                    ),
                    HealthTip(
                        title='Fever Management',
                        content='For fever management:\n• Rest and drink plenty of fluids\n• Take acetaminophen or ibuprofen as directed\n• Use a cool compress on your forehead\n• Monitor your temperature regularly\n• Seek medical help if fever is above 103°F or lasts more than 3 days',
                        category='fever',
                        symptoms='fever,temperature,hot,बुखार,तापमान',
                        created_by=1
                    ),
                    HealthTip(
                        title='Cold and Flu Care',
                        content='For cold and flu symptoms:\n• Get plenty of rest\n• Drink warm fluids like tea or soup\n• Use a humidifier\n• Gargle with salt water for sore throat\n• Take over-the-counter cold medications\n• Wash hands frequently to prevent spread',
                        category='cold',
                        symptoms='cold,flu,cough,sneezing,जुकाम,खांसी,फ्लू',
                        created_by=1
                    ),
                    HealthTip(
                        title='Stomach Pain Relief',
                        content='For stomach pain:\n• Rest and avoid solid foods\n• Drink clear fluids\n• Apply heat to abdomen\n• Avoid spicy or fatty foods\n• Consider antacids if needed\n• See doctor if pain is severe',
                        category='stomach',
                        symptoms='stomach pain,abdominal pain,पेट दर्द,उदर पीड़ा',
                        created_by=1
                    ),
                    HealthTip(
                        title='Cough Relief',
                        content='For cough relief:\n• Drink warm liquids like honey tea\n• Use a humidifier\n• Try cough drops or lozenges\n• Avoid irritants like smoke\n• Get plenty of rest\n• See doctor if cough persists more than a week',
                        category='respiratory',
                        symptoms='cough,coughing,खांसी,कफ',
                        created_by=1
                    ),
                    HealthTip(
                        title='Sore Throat Care',
                        content='For sore throat:\n• Gargle with warm salt water\n• Drink warm liquids\n• Use throat lozenges\n• Avoid smoking and alcohol\n• Rest your voice\n• Use a humidifier',
                        category='throat',
                        symptoms='sore throat,throat pain,गला खराब,गले में दर्द',
                        created_by=1
                    ),
                    HealthTip(
                        title='Body Aches Relief',
                        content='For body aches:\n• Rest and relax\n• Take warm baths\n• Use heating pads\n• Gentle stretching\n• Over-the-counter pain relievers\n• Stay hydrated',
                        category='pain',
                        symptoms='body ache,muscle pain,शरीर में दर्द,मांसपेशियों में दर्द',
                        created_by=1
                    ),
                    HealthTip(
                        title='Vomiting Relief',
                        content='For vomiting:\n• Rest and avoid solid foods\n• Sip clear fluids slowly\n• Try ginger tea or crackers\n• Avoid strong smells\n• Use BRAT diet (Bananas, Rice, Applesauce, Toast)\n• See doctor if vomiting persists more than 24 hours',
                        category='digestive',
                        symptoms='vomiting,nausea,उल्टी,मतली',
                        created_by=1
                    )
                ]
                for tip in sample_tips:
                    db.session.add(tip)
                db.session.commit()
                print("Sample health tips created!")
                
        except Exception as e:
            print(f"Error initializing database: {e}")
            db.session.rollback()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        if current_user.is_admin:
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('user_dashboard'))
        
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            if user.is_admin:
                return redirect(next_page or url_for('admin_dashboard'))
            return redirect(next_page or url_for('user_dashboard'))
        flash('Invalid email or password', 'danger')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('user_dashboard'))
        
    if request.method == 'POST':
        try:
            existing_user = User.query.filter_by(email=request.form.get('email')).first()
            if existing_user:
                flash('Email already exists', 'danger')
                return render_template('signup.html')
                
            user = User(
                email=request.form.get('email'),
                name=request.form.get('name'),
                age=int(request.form.get('age')),
                location=request.form.get('location'),
                language=request.form.get('language', 'en')
            )
            user.set_password(request.form.get('password'))
            db.session.add(user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('Error creating account. Please try again.', 'danger')
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# User Routes
@app.route('/user/dashboard')
@login_required
def user_dashboard():
    if current_user.is_admin:
        return redirect(url_for('admin_dashboard'))
    
    try:
        health_scores = HealthScore.query.filter_by(user_id=current_user.id).order_by(HealthScore.date.desc()).limit(10).all()
        chat_history = ChatHistory.query.filter_by(user_id=current_user.id).order_by(ChatHistory.timestamp.desc()).limit(50).all()
        
        chart_url = generate_health_chart(current_user.id)
        
        return render_template('user_dashboard.html', 
                             health_scores=health_scores,
                             chat_history=chat_history,
                             chart_url=chart_url)
    except Exception as e:
        print(f"User dashboard error: {e}")
        flash('Error loading dashboard', 'danger')
        return render_template('user_dashboard.html', 
                             health_scores=[],
                             chat_history=[],
                             chart_url=None)

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    try:
        message = request.json.get('message')
        if not message:
            return jsonify({'response': 'Please enter a message.'})
            
        response = generate_chat_response(message, current_user)
        
        chat_entry = ChatHistory(
            user_id=current_user.id,
            message=message,
            response=response
        )
        db.session.add(chat_entry)
        db.session.commit()
        
        return jsonify({
            'response': response,
            'chat_id': chat_entry.id  # Return chat_id for feedback
        })
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({'response': 'Sorry, I encountered an error. Please try again.', 'chat_id': None})

@app.route('/chat/feedback', methods=['POST'])
@login_required
def chat_feedback():
    try:
        data = request.get_json()
        feedback = ChatFeedback(
            user_id=current_user.id,
            chat_id=data['chat_id'],
            feedback=data['feedback'],
            reason=data.get('reason', '')  # Optional reason field
        )
        db.session.add(feedback)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        print(f"Feedback error: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/chat/delete/<int:chat_id>', methods=['DELETE'])
@login_required
def delete_chat(chat_id):
    try:
        chat = ChatHistory.query.filter_by(id=chat_id, user_id=current_user.id).first()
        if chat:
            # Delete associated feedback first
            ChatFeedback.query.filter_by(chat_id=chat_id).delete()
            db.session.delete(chat)
            db.session.commit()
            return jsonify({'success': True})
        return jsonify({'success': False, 'message': 'Chat not found'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/chat/clear_all', methods=['DELETE'])
@login_required
def clear_all_chats():
    try:
        # Delete all chats and their feedback for the current user
        chat_ids = [chat.id for chat in ChatHistory.query.filter_by(user_id=current_user.id).all()]
        ChatFeedback.query.filter(ChatFeedback.chat_id.in_(chat_ids)).delete()
        ChatHistory.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/emergency', methods=['POST'])
@login_required
def emergency():
    try:
        emergency_log = EmergencyLog(
            user_id=current_user.id,
            location=current_user.location
        )
        db.session.add(emergency_log)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Emergency services have been notified! Help is on the way.',
            'hospital': {
                'name': 'City General Hospital',
                'address': '123 Medical Center Dr',
                'phone': '+1-555-EMERGENCY',
                'distance': '2.3 miles'
            }
        })
    except Exception as e:
        print(f"Emergency error: {e}")
        return jsonify({
            'success': False,
            'message': 'Emergency service temporarily unavailable.'
        })

@app.route('/health/score', methods=['POST'])
@login_required
def update_health_score():
    try:
        score = request.json.get('score')
        notes = request.json.get('notes', '')
        
        if not score:
            return jsonify({'success': False, 'message': 'Score is required'})
            
        health_score = HealthScore(
            user_id=current_user.id,
            score=int(score),
            notes=notes,
            date=date.today()
        )
        db.session.add(health_score)
        db.session.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"Health score error: {e}")
        return jsonify({'success': False, 'message': 'Error updating health score'})

@app.route('/health/score/<int:score_id>', methods=['DELETE'])
@login_required
def delete_health_score(score_id):
    try:
        score = HealthScore.query.filter_by(id=score_id, user_id=current_user.id).first()
        if score:
            db.session.delete(score)
            db.session.commit()
            return jsonify({'success': True})
        return jsonify({'success': False, 'message': 'Score not found'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/profile/update', methods=['POST'])
@login_required
def update_profile():
    try:
        data = request.get_json()
        current_user.name = data.get('name', current_user.name)
        current_user.age = data.get('age', current_user.age)
        current_user.location = data.get('location', current_user.location)
        current_user.language = data.get('language', current_user.language)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)})

# Admin Routes
@app.route('/admin/dashboard')
@login_required
@admin_required
def admin_dashboard():
    try:
        total_users = User.query.filter_by(is_admin=False).count()
        total_chats = ChatHistory.query.count()
        total_health_scores = HealthScore.query.count()
        emergency_count = EmergencyLog.query.count()
        recent_emergencies = EmergencyLog.query.order_by(EmergencyLog.timestamp.desc()).limit(5).all()
        
        user_growth_chart = generate_user_growth_chart()
        health_score_chart = generate_health_score_chart()
        
        return render_template('admin_dashboard.html', 
                             total_users=total_users,
                             total_chats=total_chats,
                             total_health_scores=total_health_scores,
                             emergency_count=emergency_count,
                             recent_emergencies=recent_emergencies,
                             user_growth_chart=user_growth_chart,
                             health_score_chart=health_score_chart)
    except Exception as e:
        print(f"Admin dashboard error: {e}")
        flash('Error loading admin dashboard', 'danger')
        return render_template('admin_dashboard.html',
                             total_users=0,
                             total_chats=0,
                             total_health_scores=0,
                             emergency_count=0,
                             recent_emergencies=[],
                             user_growth_chart=None,
                             health_score_chart=None)

@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    users = User.query.filter_by(is_admin=False).all()
    users_data = []
    
    for user in users:
        # Get user statistics
        chat_count = ChatHistory.query.filter_by(user_id=user.id).count()
        emergency_count = EmergencyLog.query.filter_by(user_id=user.id).count()
        latest_score = HealthScore.query.filter_by(user_id=user.id).order_by(HealthScore.date.desc()).first()
        
        # Get feedback counts
        thumbs_up = ChatFeedback.query.filter_by(user_id=user.id, feedback='thumbs_up').count()
        thumbs_down = ChatFeedback.query.filter_by(user_id=user.id, feedback='thumbs_down').count()
        
        users_data.append({
            'user': user,
            'chat_count': chat_count,
            'emergency_count': emergency_count,
            'latest_score': latest_score.score if latest_score else 'N/A',
            'thumbs_up': thumbs_up,
            'thumbs_down': thumbs_down
        })
    
    return render_template('admin_users.html', users_data=users_data)

@app.route('/admin/user/<int:user_id>')
@login_required
@admin_required
def admin_user_detail(user_id):
    user = User.query.get_or_404(user_id)
    
    # Get user activity data
    chats = ChatHistory.query.filter_by(user_id=user_id).order_by(ChatHistory.timestamp.desc()).limit(10).all()
    health_scores = HealthScore.query.filter_by(user_id=user_id).order_by(HealthScore.date.desc()).limit(10).all()
    emergencies = EmergencyLog.query.filter_by(user_id=user_id).order_by(EmergencyLog.timestamp.desc()).limit(5).all()
    feedbacks = ChatFeedback.query.filter_by(user_id=user_id).order_by(ChatFeedback.created_at.desc()).limit(10).all()
    
    # Calculate statistics
    total_chats = ChatHistory.query.filter_by(user_id=user_id).count()
    total_feedback = ChatFeedback.query.filter_by(user_id=user_id).count()
    positive_feedback = ChatFeedback.query.filter_by(user_id=user_id, feedback='thumbs_up').count()
    negative_feedback = ChatFeedback.query.filter_by(user_id=user_id, feedback='thumbs_down').count()
    
    # Generate health chart
    chart_url = generate_health_chart(user_id)
    
    return render_template('admin_user_detail.html', 
                         user=user,
                         chats=chats,
                         health_scores=health_scores,
                         emergencies=emergencies,
                         feedbacks=feedbacks,
                         total_chats=total_chats,
                         total_feedback=total_feedback,
                         positive_feedback=positive_feedback,
                         negative_feedback=negative_feedback,
                         chart_url=chart_url)

@app.route('/admin/delete_user/<int:user_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_user(user_id):
    try:
        user = User.query.get(user_id)
        if user and not user.is_admin:
            # Delete related records
            HealthScore.query.filter_by(user_id=user_id).delete()
            ChatHistory.query.filter_by(user_id=user_id).delete()
            EmergencyLog.query.filter_by(user_id=user_id).delete()
            ChatFeedback.query.filter_by(user_id=user_id).delete()
            
            db.session.delete(user)
            db.session.commit()
            return jsonify({'success': True, 'message': 'User deleted successfully'})
        return jsonify({'success': False, 'message': 'User not found or cannot delete admin'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/admin/export_users')
@login_required
@admin_required
def export_users():
    users = User.query.filter_by(is_admin=False).all()
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['ID', 'Name', 'Email', 'Age', 'Location', 'Language', 'Registration Date', 
                    'Total Chats', 'Health Score', 'Emergencies', 'Thumbs Up', 'Thumbs Down', 'Last Active'])
    
    for user in users:
        chat_count = ChatHistory.query.filter_by(user_id=user.id).count()
        latest_score = HealthScore.query.filter_by(user_id=user.id).order_by(HealthScore.date.desc()).first()
        emergency_count = EmergencyLog.query.filter_by(user_id=user.id).count()
        thumbs_up = ChatFeedback.query.filter_by(user_id=user.id, feedback='thumbs_up').count()
        thumbs_down = ChatFeedback.query.filter_by(user_id=user.id, feedback='thumbs_down').count()
        last_chat = ChatHistory.query.filter_by(user_id=user.id).order_by(ChatHistory.timestamp.desc()).first()
        
        writer.writerow([
            user.id,
            user.name,
            user.email,
            user.age,
            user.location,
            user.language,
            user.created_at.strftime('%Y-%m-%d'),
            chat_count,
            latest_score.score if latest_score else 'N/A',
            emergency_count,
            thumbs_up,
            thumbs_down,
            last_chat.timestamp.strftime('%Y-%m-%d %H:%M') if last_chat else 'Never'
        ])
    
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'users_export_{datetime.now().strftime("%Y%m%d")}.csv'
    )

@app.route('/admin/health-tips')
@login_required
@admin_required
def admin_health_tips():
    tips = HealthTip.query.all()
    return render_template('admin_health_tips.html', tips=tips)

@app.route('/admin/add_health_tip', methods=['POST'])
@login_required
@admin_required
def add_health_tip():
    try:
        tip = HealthTip(
            title=request.json.get('title'),
            content=request.json.get('content'),
            category=request.json.get('category'),
            symptoms=request.json.get('symptoms'),
            created_by=current_user.id
        )
        db.session.add(tip)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/admin/update_health_tip/<int:tip_id>', methods=['PUT'])
@login_required
@admin_required
def update_health_tip(tip_id):
    try:
        tip = HealthTip.query.get(tip_id)
        if tip:
            tip.title = request.json.get('title', tip.title)
            tip.content = request.json.get('content', tip.content)
            tip.category = request.json.get('category', tip.category)
            tip.symptoms = request.json.get('symptoms', tip.symptoms)
            db.session.commit()
            return jsonify({'success': True})
        return jsonify({'success': False, 'message': 'Tip not found'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/admin/delete_health_tip/<int:tip_id>', methods=['DELETE'])
@login_required
@admin_required
def delete_health_tip(tip_id):
    try:
        tip = HealthTip.query.get(tip_id)
        if tip:
            db.session.delete(tip)
            db.session.commit()
            return jsonify({'success': True})
        return jsonify({'success': False, 'message': 'Tip not found'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/admin/analytics')
@login_required
@admin_required
def admin_analytics():
    total_users = User.query.filter_by(is_admin=False).count()
    total_chats = ChatHistory.query.count()
    total_health_scores = HealthScore.query.count()
    emergency_count = EmergencyLog.query.count()
    
    user_growth_chart = generate_user_growth_chart()
    health_dist_chart = generate_health_score_distribution_chart()
    
    return render_template('admin_analytics.html',
                         total_users=total_users,
                         total_chats=total_chats,
                         total_health_scores=total_health_scores,
                         emergency_count=emergency_count,
                         user_growth_chart=user_growth_chart,
                         health_dist_chart=health_dist_chart)

@app.route('/admin/analytics/data')
@login_required
@admin_required
def analytics_data():
    # Calculate real-time stats
    total_users = User.query.filter_by(is_admin=False).count()
    total_queries = ChatHistory.query.count()
    queries_today = ChatHistory.query.filter(
        ChatHistory.timestamp >= datetime.utcnow().date()
    ).count()
    
    thumbs_up = ChatFeedback.query.filter_by(feedback='thumbs_up').count()
    thumbs_down = ChatFeedback.query.filter_by(feedback='thumbs_down').count()
    total_feedback = thumbs_up + thumbs_down
    
    positive_rate = round((thumbs_up / total_feedback * 100), 2) if total_feedback > 0 else 0
    feedback_rate = round((total_feedback / total_queries * 100), 2) if total_queries > 0 else 0
    
    # Calculate active users today
    today = datetime.utcnow().date()
    active_users = db.session.query(ChatHistory.user_id).filter(
        ChatHistory.timestamp >= today
    ).distinct().count()
    
    # Calculate average health score
    avg_health_score = db.session.query(db.func.avg(HealthScore.score)).scalar()
    avg_health_score = round(avg_health_score, 1) if avg_health_score else 0
    
    # Generate real chart data
    chart_data = generate_real_chart_data()
    
    return jsonify({
        'stats': {
            'total_users': total_users,
            'total_queries': total_queries,
            'queries_today': queries_today,
            'thumbs_up': thumbs_up,
            'thumbs_down': thumbs_down,
            'positive_rate': positive_rate,
            'feedback_rate': feedback_rate,
            'active_users': active_users,
            'avg_health_score': avg_health_score,
            'emergency_count': EmergencyLog.query.count()
        },
        'charts': chart_data
    })

@app.route('/admin/analytics/feedback_reasons')
@login_required
@admin_required
def feedback_reasons():
    """Get common feedback reasons for analysis"""
    negative_feedbacks = ChatFeedback.query.filter_by(feedback='thumbs_down').filter(ChatFeedback.reason != '').all()
    
    reasons = {}
    for feedback in negative_feedbacks:
        reason = feedback.reason.lower().strip()
        if reason:
            # Group similar reasons
            if 'unclear' in reason or 'confusing' in reason or 'not clear' in reason:
                key = 'Unclear/Confusing Advice'
            elif 'address' in reason or 'relevant' in reason or 'symptoms' in reason:
                key = 'Not Relevant to Symptoms'
            elif 'detailed' in reason or 'more info' in reason or 'specific' in reason:
                key = 'Needs More Detailed Info'
            elif 'professional' in reason or 'doctor' in reason or 'medical' in reason:
                key = 'Needs Professional Advice'
            elif 'understand' in reason or 'language' in reason or 'hindi' in reason or 'english' in reason:
                key = 'Language/Understanding Issues'
            else:
                key = 'Other Reasons'
            
            reasons[key] = reasons.get(key, 0) + 1
    
    return jsonify({
        'reasons': reasons
    })

@app.route('/admin/analytics/feedback_insights')
@login_required
@admin_required
def feedback_insights():
    """Get detailed feedback insights"""
    # Calculate feedback rates
    total_chats = ChatHistory.query.count()
    total_feedback = ChatFeedback.query.count()
    thumbs_up = ChatFeedback.query.filter_by(feedback='thumbs_up').count()
    thumbs_down = ChatFeedback.query.filter_by(feedback='thumbs_down').count()
    
    # Feedback over time (last 7 days)
    feedback_trend = []
    for i in range(6, -1, -1):
        date = datetime.utcnow().date() - timedelta(days=i)
        day_up = ChatFeedback.query.filter(
            ChatFeedback.feedback == 'thumbs_up',
            db.func.date(ChatFeedback.created_at) == date
        ).count()
        day_down = ChatFeedback.query.filter(
            ChatFeedback.feedback == 'thumbs_down',
            db.func.date(ChatFeedback.created_at) == date
        ).count()
        
        feedback_trend.append({
            'date': date.strftime('%m/%d'),
            'thumbs_up': day_up,
            'thumbs_down': day_down
        })
    
    # Most helpful tips (based on thumbs up)
    helpful_tips = db.session.query(
        ChatHistory.response,
        db.func.count(ChatFeedback.id).label('thumbs_up_count')
    ).join(ChatFeedback, ChatHistory.id == ChatFeedback.chat_id)\
     .filter(ChatFeedback.feedback == 'thumbs_up')\
     .group_by(ChatHistory.response)\
     .order_by(db.desc('thumbs_up_count'))\
     .limit(5)\
     .all()
    
    # Least helpful tips (based on thumbs down)
    unhelpful_tips = db.session.query(
        ChatHistory.response,
        db.func.count(ChatFeedback.id).label('thumbs_down_count')
    ).join(ChatFeedback, ChatHistory.id == ChatFeedback.chat_id)\
     .filter(ChatFeedback.feedback == 'thumbs_down')\
     .group_by(ChatHistory.response)\
     .order_by(db.desc('thumbs_down_count'))\
     .limit(5)\
     .all()
    
    return jsonify({
        'feedback_stats': {
            'total_feedback': total_feedback,
            'thumbs_up': thumbs_up,
            'thumbs_down': thumbs_down,
            'feedback_rate': round((total_feedback / total_chats * 100), 2) if total_chats > 0 else 0,
            'satisfaction_rate': round((thumbs_up / total_feedback * 100), 2) if total_feedback > 0 else 0
        },
        'feedback_trend': feedback_trend,
        'helpful_tips': [
            {'response': tip[0][:100] + '...' if len(tip[0]) > 100 else tip[0], 'count': tip[1]}
            for tip in helpful_tips
        ],
        'unhelpful_tips': [
            {'response': tip[0][:100] + '...' if len(tip[0]) > 100 else tip[0], 'count': tip[1]}
            for tip in unhelpful_tips
        ]
    })

@app.route('/admin/analytics/recent_feedback')
@login_required
@admin_required
def recent_feedback():
    """Get recent feedback activity"""
    recent_feedbacks = db.session.query(
        ChatFeedback,
        ChatHistory.response,
        User.name
    ).join(ChatHistory, ChatFeedback.chat_id == ChatHistory.id)\
     .join(User, ChatFeedback.user_id == User.id)\
     .order_by(ChatFeedback.created_at.desc())\
     .limit(20)\
     .all()
    
    feedback_data = []
    for feedback, response, user_name in recent_feedbacks:
        feedback_data.append({
            'user_id': feedback.user_id,
            'user_name': user_name,
            'feedback': feedback.feedback,
            'reason': feedback.reason,
            'response_preview': response[:100] + '...' if len(response) > 100 else response,
            'timestamp': feedback.created_at.isoformat()
        })
    
    return jsonify({
        'recent_feedback': feedback_data
    })

@app.route('/admin/generate_report/<report_type>')
@login_required
@admin_required
def generate_report(report_type):
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        
        title = Paragraph(f"Health Wellness Chatbot - {report_type.title()} Report", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 0.25*inch))
        
        if report_type == 'users':
            users = User.query.filter_by(is_admin=False).all()
            data = [['ID', 'Name', 'Email', 'Age', 'Location', 'Joined', 'Health Score']]
            for user in users:
                latest_score = HealthScore.query.filter_by(user_id=user.id).order_by(HealthScore.date.desc()).first()
                score = latest_score.score if latest_score else 'N/A'
                data.append([
                    str(user.id), user.name, user.email, str(user.age),
                    user.location, user.created_at.strftime('%Y-%m-%d'), str(score)
                ])
            
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 7)
            ]))
            elements.append(table)
            
        elif report_type == 'emergencies':
            emergencies = EmergencyLog.query.order_by(EmergencyLog.timestamp.desc()).all()
            data = [['User ID', 'Location', 'Timestamp', 'Status']]
            for emergency in emergencies:
                data.append([
                    str(emergency.user_id),
                    emergency.location,
                    emergency.timestamp.strftime('%Y-%m-%d %H:%M'),
                    emergency.status
                ])
            
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(table)
        
        doc.build(elements)
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f'{report_type}_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
            mimetype='application/pdf'
        )
        
    except Exception as e:
        flash(f'Error generating report: {str(e)}', 'danger')
        return redirect(url_for('admin_analytics'))

@app.route('/admin/settings')
@login_required
@admin_required
def admin_settings():
    return render_template('admin_settings.html')

# Helper Functions
def generate_chat_response(message, user):
    message_lower = message.lower()
    
    # Check if message is in Hindi
    message_is_hindi = is_hindi_text(message)
    
    # Check health tips from database for ALL matching symptoms
    tips = HealthTip.query.all()
    matching_tips = []
    
    for tip in tips:
        if tip.symptoms:
            symptoms = [s.strip().lower() for s in tip.symptoms.split(',')]
            for symptom in symptoms:
                if symptom and symptom in message_lower:
                    matching_tips.append(tip)
                    break  # Avoid adding same tip multiple times
    
    # Define pre-translated Hindi responses to avoid translation issues
    hindi_responses = {
        'multiple_symptoms': "आपके लक्षणों के आधार पर, यहाँ व्यापक सिफारिशें हैं:\n\n",
        'general_advice': "💡 **एक से अधिक लक्षणों के लिए सामान्य सलाह:**\n• अपने शरीर को ठीक होने में मदद के लिए भरपूर आराम करें\n• पानी और गर्म तरल पदार्थ पीकर हाइड्रेटेड रहें\n• अपने लक्षणों पर नज़र रखें और किसी भी बदलाव को नोट करें\n• जब तक आप बेहतर महसूस न करें तब तक strenuous गतिविधियों से बचें\n• पौष्टिक, आसानी से पचने वाले खाद्य पदार्थ खाएं\n\n",
        'medical_attention': "🚨 **चिकित्सकीय सहायता कब लें:**\n• लक्षण 3-4 दिनों के बाद बिगड़ते हैं या सुधारते नहीं हैं\n• तेज बुखार (101°F/38.3°C से ऊपर) होता है\n• सांस लेने में कठिनाई या गंभीर दर्द होता है\n• आप भ्रम या चक्कर महसूस करते हैं\n",
        'greeting': f"नमस्ते {user.name}! मैं आपका स्वास्थ्य सहायक हूं। अंग्रेजी या हिंदी में अपने लक्षण बताएं, और मैं मददगार सलाह प्रदान करूंगा।",
        'no_symptoms': "मैं समझता हूं कि आप ठीक महसूस नहीं कर रहे हैं। क्या आप अपने लक्षणों के बारे में अधिक विस्तार से बता सकते हैं? उदाहरण के लिए, आप 'सिरदर्द और बुखार' या 'खांसी और गले में खराश' कह सकते हैं।"
    }
    
    # If multiple tips found, combine them into comprehensive advice
    if len(matching_tips) > 1:
        # Get the main symptoms mentioned
        detected_symptoms = []
        for tip in matching_tips:
            if tip.title:
                detected_symptoms.append(tip.title.replace(' Relief', '').replace(' Care', '').replace(' Management', ''))
        
        symptom_list = ", ".join(detected_symptoms[:-1]) + " और " + detected_symptoms[-1] if len(detected_symptoms) > 1 else detected_symptoms[0]
        
        if message_is_hindi:
            # Build Hindi response without translation
            combined_response = f"आपके {symptom_list} के लक्षणों के आधार पर, यहाँ व्यापक सिफारिशें हैं:\n\n"
            
            for i, tip in enumerate(matching_tips, 1):
                combined_response += f"📍 {tip.title}:\n"
                # Use pre-translated content or translate if needed
                if 'headache' in tip.symptoms or 'migraine' in tip.symptoms:
                    combined_response += "• शांत, अंधेरे कमरे में आराम करें\n• अपने सिर पर ठंडा कंप्रेस लगाएं\n• हाइड्रेटेड रहें\n• तेज रोशनी और तेज आवाज से बचें\n• दर्द निवारक दवाओं पर विचार करें\n• विश्राम तकनीकों का अभ्यास करें\n\n"
                elif 'fever' in tip.symptoms:
                    combined_response += "• आराम करें और भरपूर तरल पदार्थ पिएं\n• निर्देशानुसार एसिटामिनोफेन या आइबुप्रोफेन लें\n• अपने माथे पर ठंडा कंप्रेस लगाएं\n• अपने तापमान की नियमित रूप से निगरानी करें\n• यदि बुखार 103°F से ऊपर है या 3 दिन से अधिक रहता है तो चिकित्सकीय सहायता लें\n\n"
                elif 'cold' in tip.symptoms:
                    combined_response += "• भरपूर आराम करें\n• चाय या सूप जैसे गर्म तरल पदार्थ पिएं\n• ह्यूमिडिफायर का उपयोग करें\n• गले में खराश के लिए नमक के पानी से गरारे करें\n• ओवर-द-काउंटर कोल्ड की दवाएं लें\n• फैलाव को रोकने के लिए बार-बार हाथ धोएं\n\n"
                else:
                    # Safe translation with error handling
                    try:
                        translated_content = translate_to_hindi(tip.content)
                        combined_response += translated_content + "\n\n"
                    except:
                        combined_response += tip.content + "\n\n"
            
            combined_response += hindi_responses['general_advice']
            combined_response += hindi_responses['medical_attention']
            combined_response = add_disclaimer(combined_response, True)
            return combined_response
        else:
            # English response
            combined_response = f"Based on your symptoms of {symptom_list}, here are comprehensive recommendations:\n\n"
            
            for i, tip in enumerate(matching_tips, 1):
                combined_response += f"📍 {tip.title}:\n{tip.content}\n\n"
            
            combined_response += "💡 **General Advice for Multiple Symptoms:**\n"
            combined_response += "• Get plenty of rest to help your body recover\n"
            combined_response += "• Stay well-hydrated with water and warm fluids\n"
            combined_response += "• Monitor your symptoms and note any changes\n"
            combined_response += "• Avoid strenuous activities until you feel better\n"
            combined_response += "• Eat nutritious, easy-to-digest foods\n\n"
            
            combined_response += "🚨 **When to Seek Medical Attention:**\n"
            combined_response += "• Symptoms worsen or don't improve after 3-4 days\n"
            combined_response += "• High fever (above 101°F/38.3°C) develops\n"
            combined_response += "• Difficulty breathing or severe pain occurs\n"
            combined_response += "• You experience confusion or dizziness\n"
            
            combined_response = add_disclaimer(combined_response, False)
            return combined_response
    
    # If only one tip found, return it normally
    elif len(matching_tips) == 1:
        tip = matching_tips[0]
        
        if message_is_hindi:
            # Use pre-translated Hindi content for common symptoms
            if 'headache' in tip.symptoms or 'migraine' in tip.symptoms:
                response_content = "माइग्रेन/सिरदर्द से राहत के लिए:\n• शांत, अंधेरे कमरे में आराम करें\n• अपने सिर पर ठंडा कंप्रेस लगाएं\n• हाइड्रेटेड रहें\n• तेज रोशनी और तेज आवाज से बचें\n• दर्द निवारक दवाओं पर विचार करें\n• विश्राम तकनीकों का अभ्यास करें"
            elif 'fever' in tip.symptoms:
                response_content = "बुखार प्रबंधन के लिए:\n• आराम करें और भरपूर तरल पदार्थ पिएं\n• निर्देशानुसार एसिटामिनोफेन या आइबुप्रोफेन लें\n• अपने माथे पर ठंडा कंप्रेस लगाएं\n• अपने तापमान की नियमित रूप से निगरानी करें\n• यदि बुखार 103°F से ऊपर है या 3 दिन से अधिक रहता है तो चिकित्सकीय सहायता लें"
            elif 'cold' in tip.symptoms:
                response_content = "जुकाम और फ्लू के लक्षणों के लिए:\n• भरपूर आराम करें\n• चाय या सूप जैसे गर्म तरल पदार्थ पिएं\n• ह्यूमिडिफायर का उपयोग करें\n• गले में खराश के लिए नमक के पानी से गरारे करें\n• ओवर-द-काउंटर कोल्ड की दवाएं लें\n• फैलाव को रोकने के लिए बार-बार हाथ धोएं"
            elif 'vomiting' in tip.symptoms:
                response_content = "उल्टी से राहत के लिए:\n• आराम करें और ठोस खाद्य पदार्थों से बचें\n• धीरे-धीरे स्पष्ट तरल पदार्थ पिएं\n• अदरक की चाय या क्रैकर्स आज़माएं\n• तेज़ गंध से बचें\n• BRAT आहार का उपयोग करें (केले, चावल, सेब की चटनी, टोस्ट)\n• यदि उल्टी 24 घंटे से अधिक समय तक बनी रहती है तो डॉक्टर को दिखाएं"
            else:
                # Safe translation
                try:
                    response_content = translate_to_hindi(tip.content)
                except:
                    response_content = tip.content
        else:
            response_content = tip.content
        
        response_content = add_disclaimer(response_content, message_is_hindi)
        return response_content
    
    # Default responses for specific symptoms (pre-translated)
    health_advice = {
        'fever': "बुखार के लिए:\n• आराम करें और भरपूर तरल पदार्थ पिएं\n• निर्देशानुसार एसिटामिनोफेन या आइबुप्रोफेन लें\n• अपने माथे पर ठंडा कंप्रेस लगाएं\n• अपने तापमान की नियमित रूप से निगरानी करें\n• यदि बुखार 103°F से ऊपर है या 3 दिन से अधिक रहता है तो चिकित्सकीय सहायता लें",
        'headache': "सिरदर्द के लिए:\n• अंधेरे कमरे में आराम करें\n• हाइड्रेटेड रहें\n• तेज रोशनी जैसे ट्रिगर्स से बचें\n• दर्द निवारक दवा पर विचार करें\n• ठंडा कंप्रेस लगाएं",
        'migraine': "माइग्रेन के लिए:\n• शांत अंधेरे कमरे में आराम करें\n• ठंडे कंप्रेस लगाएं\n• हाइड्रेटेड रहें\n• तेज रोशनी और तेज आवाज से बचें\n• दवा पर विचार करें",
        'cold': "जुकाम के लिए:\n• भरपूर आराम करें\n• गर्म तरल पदार्थ पिएं\n• ह्यूमिडिफायर का उपयोग करें\n• ओवर-द-काउंटर दवाएं लें\n• हाथों को बार-बार धोएं",
        'cough': "खांसी के लिए:\n• शहद की चाय जैसे गर्म तरल पदार्थ पिएं\n• ह्यूमिडिफायर का उपयोग करें\n• कफ ड्रॉप्स आज़माएं\n• धुएं जैसे उत्तेजक पदार्थों से बचें\n• भरपूर आराम करें",
        'stomach': "पेट दर्द के लिए:\n• आराम करें और ठोस खाद्य पदार्थों से बचें\n• स्पष्ट तरल पदार्थ पिएं\n• पेट पर गर्मी लगाएं\n• मसालेदार या वसायुक्त खाद्य पदार्थों से बचें\n• यदि गंभीर है तो डॉक्टर को दिखाएं",
        'body ache': "शरीर में दर्द के लिए:\n• आराम करें और आराम करें\n• गर्म स्नान करें\n• हीटिंग पैड का उपयोग करें\n• हल्का स्ट्रेचिंग करें\n• यदि आवश्यक हो तो दर्द निवारक दवाएं लें",
        'sore throat': "गले में खराश के लिए:\n• गर्म नमक के पानी से गरारे करें\n• गर्म तरल पदार्थ पिएं\n• गले की लोज़ेंजेस का उपयोग करें\n• धूम्रपान और शराब से बचें\n• अपनी आवाज़ को आराम दें",
        'vomiting': "उल्टी के लिए:\n• आराम करें और ठोस खाद्य पदार्थों से बचें\n• धीरे-धीरे स्पष्ट तरल पदार्थ पिएं\n• अदरक की चाय या क्रैकर्स आज़माएं\n• तेज़ गंध से बचें\n• BRAT आहार का उपयोग करें\n• यदि उल्टी 24 घंटे से अधिक समय तक बनी रहती है तो डॉक्टर को दिखाएं",
        'बुखार': "बुखार के लिए:\n• आराम करें और भरपूर तरल पदार्थ पिएं\n• निर्देशानुसार एसिटामिनोफेन या आइबुप्रोफेन लें\n• अपने माथे पर ठंडा कंप्रेस लगाएं\n• अपने तापमान की नियमित रूप से निगरानी करें\n• यदि बुखार 103°F से ऊपर है या 3 दिन से अधिक रहता है तो चिकित्सकीय सहायता लें",
        'सिरदर्द': "सिरदर्द के लिए:\n• अंधेरे कमरे में आराम करें\n• हाइड्रेटेड रहें\n• तेज रोशनी जैसे ट्रिगर्स से बचें\n• दर्द निवारक दवा पर विचार करें\n• ठंडा कंप्रेस लगाएं",
        'खांसी': "खांसी के लिए:\n• शहद की चाय जैसे गर्म तरल पदार्थ पिएं\n• ह्यूमिडिफायर का उपयोग करें\n• कफ ड्रॉप्स आज़माएं\n• धुएं जैसे उत्तेजक पदार्थों से बचें\n• भरपूर आराम करें",
        'जुकाम': "जुकाम के लिए:\n• भरपूर आराम करें\n• गर्म तरल पदार्थ पिएं\n• ह्यूमिडिफायर का उपयोग करें\n• ओवर-द-काउंटर दवाएं लें\n• हाथों को बार-बार धोएं",
        'पेट दर्द': "पेट दर्द के लिए:\n• आराम करें और ठोस खाद्य पदार्थों से बचें\n• स्पष्ट तरल पदार्थ पिएं\n• पेट पर गर्मी लगाएं\n• मसालेदार या वसायुक्त खाद्य पदार्थों से बचें\n• यदि गंभीर है तो डॉक्टर को दिखाएं",
        'उल्टी': "उल्टी के लिए:\n• आराम करें और ठोस खाद्य पदार्थों से बचें\n• धीरे-धीरे स्पष्ट तरल पदार्थ पिएं\n• अदरक की चाय या क्रैकर्स आज़माएं\n• तेज़ गंध से बचें\n• BRAT आहार का उपयोग करें\n• यदि उल्टी 24 घंटे से अधिक समय तक बनी रहती है तो डॉक्टर को दिखाएं",
    }
    
    # Check for symptoms in the default advice
    for symptom, advice in health_advice.items():
        if symptom in message_lower:
            final_advice = add_disclaimer(advice, message_is_hindi)
            return final_advice
    
    # Greetings and other responses
    if any(word in message_lower for word in ['hello', 'hi', 'hey', 'नमस्ते', 'हैलो']):
        if message_is_hindi:
            greeting = hindi_responses['greeting']
        else:
            greeting = f"Hello {user.name}! I'm your health assistant. Describe your symptoms in English or Hindi, and I'll provide helpful advice."
        
        greeting = add_disclaimer(greeting, message_is_hindi)
        return greeting
    else:
        if message_is_hindi:
            response = hindi_responses['no_symptoms']
        else:
            response = "I understand you're not feeling well. Could you describe your symptoms in more detail? For example, you can say 'headache and fever' or 'cough with sore throat'."
        
        response = add_disclaimer(response, message_is_hindi)
        return response

def generate_health_chart(user_id):
    try:
        scores = HealthScore.query.filter_by(user_id=user_id).order_by(HealthScore.date).limit(10).all()
        if not scores:
            return None
            
        dates = [score.date.strftime('%m/%d') for score in scores]
        values = [score.score for score in scores]
        
        plt.figure(figsize=(8, 4))
        plt.plot(dates, values, marker='o', linewidth=2, markersize=6, color='#007bff')
        plt.title('Health Score Progress', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Health Score')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.ylim(0, 100)
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        print(f"Chart error: {e}")
        return None

def generate_user_growth_chart():
    try:
        # This is sample data - you can replace with actual database queries
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        users = [10, 25, 45, 70, 100, User.query.filter_by(is_admin=False).count()]
        
        plt.figure(figsize=(6, 4))
        plt.plot(months, users, marker='o', linewidth=2, color='green')
        plt.title('User Growth', fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('Users')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        print(f"User growth chart error: {e}")
        return None

def generate_health_score_chart():
    try:
        # Sample data - replace with actual database queries
        labels = ['Excellent (80-100)', 'Good (60-79)', 'Poor (0-59)']
        sizes = [45, 35, 20]
        colors = ['#28a745', '#ffc107', '#dc3545']
        
        plt.figure(figsize=(6, 4))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Health Score Distribution', fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        print(f"Health score chart error: {e}")
        return None

def generate_health_score_distribution_chart():
    try:
        excellent = HealthScore.query.filter(HealthScore.score >= 80).count()
        good = HealthScore.query.filter(HealthScore.score >= 60, HealthScore.score < 80).count()
        poor = HealthScore.query.filter(HealthScore.score < 60).count()
        
        labels = ['Excellent (80-100)', 'Good (60-79)', 'Poor (0-59)']
        sizes = [excellent, good, poor]
        colors = ['#28a745', '#ffc107', '#dc3545']
        
        plt.figure(figsize=(6, 4))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Health Score Distribution', fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        print(f"Health distribution chart error: {e}")
        return None

def generate_real_chart_data():
    """Generate real chart data from database"""
    # Last 7 days data
    dates = []
    daily_queries = []
    daily_feedback = []
    
    for i in range(6, -1, -1):
        date = datetime.utcnow().date() - timedelta(days=i)
        dates.append(date.strftime('%m/%d'))
        
        # Count queries for this day
        day_queries = ChatHistory.query.filter(
            db.func.date(ChatHistory.timestamp) == date
        ).count()
        daily_queries.append(day_queries)
        
        # Count feedback for this day
        day_feedback = ChatFeedback.query.filter(
            db.func.date(ChatFeedback.created_at) == date
        ).count()
        daily_feedback.append(day_feedback)
    
    # Feedback distribution
    thumbs_up = ChatFeedback.query.filter_by(feedback='thumbs_up').count()
    thumbs_down = ChatFeedback.query.filter_by(feedback='thumbs_down').count()
    no_feedback = ChatHistory.query.count() - (thumbs_up + thumbs_down)
    
    # Health score distribution
    excellent = HealthScore.query.filter(HealthScore.score >= 80).count()
    good = HealthScore.query.filter(HealthScore.score >= 60, HealthScore.score < 80).count()
    poor = HealthScore.query.filter(HealthScore.score < 60).count()
    
    return {
        'feedback_thumbs_up': thumbs_up,
        'feedback_thumbs_down': thumbs_down,
        'feedback_none': no_feedback,
        'trend_dates': dates,
        'daily_queries': daily_queries,
        'daily_feedback': daily_feedback,
        'health_excellent': excellent,
        'health_good': good,
        'health_poor': poor
    }

if __name__ == '__main__':
    with app.app_context():
        init_db()
    app.run(debug=True, host='127.0.0.1', port=5000)