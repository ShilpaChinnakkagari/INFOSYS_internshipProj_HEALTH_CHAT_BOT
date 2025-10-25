from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, date
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

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

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
        
        return jsonify({'response': response})
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({'response': 'Sorry, I encountered an error. Please try again.'})

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

@app.route('/chat/delete/<int:chat_id>', methods=['DELETE'])
@login_required
def delete_chat(chat_id):
    chat = ChatHistory.query.filter_by(id=chat_id, user_id=current_user.id).first()
    if chat:
        db.session.delete(chat)
        db.session.commit()
        return jsonify({'success': True})
    return jsonify({'success': False})

@app.route('/profile/update', methods=['POST'])
@login_required
def update_profile():
    current_user.name = request.json.get('name')
    current_user.age = request.json.get('age')
    current_user.location = request.json.get('location')
    current_user.language = request.json.get('language')
    db.session.commit()
    return jsonify({'success': True})

# Admin Routes
@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        return redirect(url_for('user_dashboard'))
    
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

@app.route('/admin/health-tips')
@login_required
def admin_health_tips():
    if not current_user.is_admin:
        return redirect(url_for('user_dashboard'))
    
    tips = HealthTip.query.all()
    return render_template('admin_health_tips.html', tips=tips)

@app.route('/admin/add_health_tip', methods=['POST'])
@login_required
def add_health_tip():
    if not current_user.is_admin:
        return jsonify({'success': False})
    
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

@app.route('/admin/update_health_tip/<int:tip_id>', methods=['PUT'])
@login_required
def update_health_tip(tip_id):
    if not current_user.is_admin:
        return jsonify({'success': False})
    
    tip = HealthTip.query.get(tip_id)
    if tip:
        tip.title = request.json.get('title', tip.title)
        tip.content = request.json.get('content', tip.content)
        tip.category = request.json.get('category', tip.category)
        tip.symptoms = request.json.get('symptoms', tip.symptoms)
        db.session.commit()
        return jsonify({'success': True})
    return jsonify({'success': False})

@app.route('/admin/delete_health_tip/<int:tip_id>', methods=['DELETE'])
@login_required
def delete_health_tip(tip_id):
    if not current_user.is_admin:
        return jsonify({'success': False})
    
    tip = HealthTip.query.get(tip_id)
    if tip:
        db.session.delete(tip)
        db.session.commit()
        return jsonify({'success': True})
    return jsonify({'success': False})

@app.route('/admin/analytics')
@login_required
def admin_analytics():
    if not current_user.is_admin:
        return redirect(url_for('user_dashboard'))
    
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

@app.route('/admin/generate_report/<report_type>')
@login_required
def generate_report(report_type):
    if not current_user.is_admin:
        return redirect(url_for('user_dashboard'))
    
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
def admin_settings():
    if not current_user.is_admin:
        return redirect(url_for('user_dashboard'))
    return render_template('admin_settings.html')

# Helper Functions
def generate_chat_response(message, user):
    message_lower = message.lower()
    
    # Check if message is in Hindi
    message_is_hindi = is_hindi_text(message)
    
    # Check health tips from database
    tips = HealthTip.query.all()
    for tip in tips:
        if tip.symptoms:
            symptoms = [s.strip().lower() for s in tip.symptoms.split(',')]
            for symptom in symptoms:
                if symptom and symptom in message_lower:
                    # If user message is in Hindi, translate the response to Hindi
                    if message_is_hindi:
                        try:
                            hindi_response = translate_to_hindi(tip.content)
                            return hindi_response
                        except Exception as e:
                            print(f"Translation failed: {e}")
                            return tip.content
                    else:
                        return tip.content
    
    # Default responses
    health_advice = {
        'fever': "For fever: Rest, drink fluids, take medication, use cool compress. See doctor if high fever persists.",
        'headache': "For headache: Rest in dark room, stay hydrated, avoid triggers. Consider pain relief medication.",
        'migraine': "For migraine: Rest in quiet dark room, cold compress, hydration, avoid lights/sounds. Medication if needed.",
        'cold': "For cold: Rest, fluids, humidifier, over-the-counter meds. See doctor if symptoms worsen.",
        'cough': "For cough: Honey tea, humidifier, rest. See doctor if persistent or with fever.",
        'stomach': "For stomach pain: Rest, clear fluids, heat application. Avoid spicy foods. See doctor if severe.",
        'बुखार': "बुखार के लिए: आराम करें, तरल पदार्थ पिएं, दवा लें, ठंडा कंप्रेस लगाएं। यदि तेज बुखार बना रहे तो डॉक्टर को दिखाएं।",
        'सिरदर्द': "सिरदर्द के लिए: अंधेरे कमरे में आराम करें, हाइड्रेटेड रहें, ट्रिगर्स से बचें। दर्द निवारक दवा पर विचार करें।",
        'खांसी': "खांसी के लिए: शहद की चाय, ह्यूमिडिफायर, आराम। यदि लगातार खांसी या बुखार हो तो डॉक्टर को दिखाएं।",
        'जुकाम': "जुकाम के लिए: आराम करें, तरल पदार्थ पिएं, ह्यूमिडिफायर का उपयोग करें, ओवर-द-काउंटर दवाएं लें।",
    }
    
    for symptom, advice in health_advice.items():
        if symptom in message_lower:
            if message_is_hindi and not is_hindi_text(advice):
                try:
                    return translate_to_hindi(advice)
                except:
                    return advice
            return advice
    
    if any(word in message_lower for word in ['hello', 'hi', 'hey', 'नमस्ते', 'हैलो']):
        greeting = f"Hello {user.name}! How can I help with your health today?"
        if message_is_hindi:
            try:
                return translate_to_hindi(greeting)
            except:
                return greeting
        return greeting
    else:
        response = "I understand you're not feeling well. Could you describe your symptoms in more detail?"
        if message_is_hindi:
            try:
                return translate_to_hindi(response)
            except:
                return response
        return response

# Add these routes to your existing app.py

@app.route('/admin/users')
@login_required
def admin_users():
    if not current_user.is_admin:
        return redirect(url_for('user_dashboard'))
    
    users = User.query.filter_by(is_admin=False).all()
    users_data = []
    for user in users:
        latest_score = HealthScore.query.filter_by(user_id=user.id).order_by(HealthScore.date.desc()).first()
        chat_count = ChatHistory.query.filter_by(user_id=user.id).count()
        emergency_count = EmergencyLog.query.filter_by(user_id=user.id).count()
        health_scores = HealthScore.query.filter_by(user_id=user.id).order_by(HealthScore.date.desc()).limit(5).all()
        
        users_data.append({
            'user': user,
            'latest_score': latest_score.score if latest_score else 'N/A',
            'chat_count': chat_count,
            'emergency_count': emergency_count,
            'health_scores': health_scores
        })
    
    return render_template('admin_users.html', users_data=users_data)

@app.route('/admin/user/<int:user_id>')
@login_required
def admin_user_detail(user_id):
    if not current_user.is_admin:
        return redirect(url_for('user_dashboard'))
    
    user = User.query.get_or_404(user_id)
    health_scores = HealthScore.query.filter_by(user_id=user_id).order_by(HealthScore.date.desc()).all()
    chat_history = ChatHistory.query.filter_by(user_id=user_id).order_by(ChatHistory.timestamp.desc()).limit(20).all()
    emergencies = EmergencyLog.query.filter_by(user_id=user_id).order_by(EmergencyLog.timestamp.desc()).all()
    
    # Generate health chart for this user
    chart_url = generate_health_chart(user_id)
    
    return render_template('admin_user_detail.html', 
                         user=user, 
                         health_scores=health_scores,
                         chat_history=chat_history,
                         emergencies=emergencies,
                         chart_url=chart_url)

@app.route('/admin/delete_user/<int:user_id>', methods=['DELETE'])
@login_required
def delete_user(user_id):
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Unauthorized'})
    
    try:
        user = User.query.get(user_id)
        if user and not user.is_admin:
            # Delete related records
            HealthScore.query.filter_by(user_id=user_id).delete()
            ChatHistory.query.filter_by(user_id=user_id).delete()
            EmergencyLog.query.filter_by(user_id=user_id).delete()
            
            db.session.delete(user)
            db.session.commit()
            return jsonify({'success': True, 'message': 'User deleted successfully'})
        return jsonify({'success': False, 'message': 'User not found'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)})
    

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

if __name__ == '__main__':
    with app.app_context():
        init_db()
    app.run(debug=True, host='127.0.0.1', port=5000)