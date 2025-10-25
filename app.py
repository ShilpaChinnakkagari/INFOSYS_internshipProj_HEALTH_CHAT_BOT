from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, date
import json
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import io

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'dev-secret-key-2023'
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL') or 'sqlite:///health_chatbot.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

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

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def init_db():
    with app.app_context():
        db.create_all()
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

        # Create sample health tips
        if not HealthTip.query.first():
            sample_tips = [
                HealthTip(
                    title='Migraine Relief',
                    content='For migraine relief:\n• Rest in a quiet, dark room\n• Apply cold compresses to your head\n• Stay hydrated\n• Avoid bright lights and loud sounds\n• Consider over-the-counter pain relievers\n• Practice relaxation techniques',
                    category='head_pain',
                    symptoms='migraine,headache,head pain',
                    created_by=1
                ),
                HealthTip(
                    title='Fever Management',
                    content='For fever management:\n• Rest and drink plenty of fluids\n• Take acetaminophen or ibuprofen as directed\n• Use a cool compress on your forehead\n• Monitor your temperature regularly\n• Seek medical help if fever is above 103°F or lasts more than 3 days',
                    category='fever',
                    symptoms='fever,temperature,hot',
                    created_by=1
                ),
                HealthTip(
                    title='Cold and Flu Care',
                    content='For cold and flu symptoms:\n• Get plenty of rest\n• Drink warm fluids like tea or soup\n• Use a humidifier\n• Gargle with salt water for sore throat\n• Take over-the-counter cold medications\n• Wash hands frequently to prevent spread',
                    category='cold',
                    symptoms='cold,flu,cough,sneezing',
                    created_by=1
                )
            ]
            for tip in sample_tips:
                db.session.add(tip)
            db.session.commit()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user)
            if user.is_admin:
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('user_dashboard'))
        flash('Invalid credentials', 'danger')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
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
        flash('Registration successful!', 'success')
        return redirect(url_for('login'))
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
    
    health_scores = HealthScore.query.filter_by(user_id=current_user.id).order_by(HealthScore.date.desc()).limit(10).all()
    chat_history = ChatHistory.query.filter_by(user_id=current_user.id).order_by(ChatHistory.timestamp.desc()).limit(50).all()
    
    return render_template('user_dashboard.html', 
                         health_scores=health_scores,
                         chat_history=chat_history)

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    message = request.json.get('message')
    response = generate_chat_response(message, current_user)
    
    chat_entry = ChatHistory(
        user_id=current_user.id,
        message=message,
        response=response
    )
    db.session.add(chat_entry)
    db.session.commit()
    
    return jsonify({'response': response})

@app.route('/health/score', methods=['POST'])
@login_required
def update_health_score():
    score = request.json.get('score')
    notes = request.json.get('notes', '')
    
    health_score = HealthScore(
        user_id=current_user.id,
        score=int(score),
        notes=notes,
        date=date.today()
    )
    db.session.add(health_score)
    db.session.commit()
    
    return jsonify({'success': True})

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
    total_users = User.query.filter_by(is_admin=False).count()
    total_chats = ChatHistory.query.count()
    total_health_scores = HealthScore.query.count()
    return render_template('admin_dashboard.html', 
                         total_users=total_users,
                         total_chats=total_chats,
                         total_health_scores=total_health_scores)

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

@app.route('/admin/analytics')
@login_required
def admin_analytics():
    if not current_user.is_admin:
        return redirect(url_for('user_dashboard'))
    
    total_users = User.query.filter_by(is_admin=False).count()
    total_chats = ChatHistory.query.count()
    total_health_scores = HealthScore.query.count()
    recent_users = User.query.filter_by(is_admin=False).order_by(User.created_at.desc()).limit(5).all()
    recent_chats = ChatHistory.query.order_by(ChatHistory.timestamp.desc()).limit(10).all()
    
    return render_template('admin_analytics.html',
                         total_users=total_users,
                         total_chats=total_chats,
                         total_health_scores=total_health_scores,
                         recent_users=recent_users,
                         recent_chats=recent_chats)

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
        elements.append(Paragraph(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
        elements.append(Spacer(1, 0.25*inch))
        
        if report_type == 'users':
            users = User.query.filter_by(is_admin=False).all()
            data = [['ID', 'Name', 'Email', 'Age', 'Location', 'Joined']]
            for user in users:
                data.append([
                    str(user.id),
                    user.name,
                    user.email,
                    str(user.age),
                    user.location,
                    user.created_at.strftime('%Y-%m-%d')
                ])
            
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(table)
            
        elif report_type == 'chats':
            chats = ChatHistory.query.order_by(ChatHistory.timestamp.desc()).limit(50).all()
            data = [['User ID', 'Message', 'Response', 'Timestamp']]
            for chat in chats:
                data.append([
                    str(chat.user_id),
                    chat.message[:50] + '...' if len(chat.message) > 50 else chat.message,
                    chat.response[:50] + '...' if len(chat.response) > 50 else chat.response,
                    chat.timestamp.strftime('%Y-%m-%d %H:%M')
                ])
            
            table = Table(data, colWidths=[1*inch, 2*inch, 2*inch, 1.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8)
            ]))
            elements.append(table)
            
        elif report_type == 'health':
            health_scores = HealthScore.query.order_by(HealthScore.date.desc()).limit(50).all()
            data = [['User ID', 'Score', 'Date', 'Notes']]
            for score in health_scores:
                data.append([
                    str(score.user_id),
                    str(score.score),
                    score.date.strftime('%Y-%m-%d'),
                    score.notes[:30] + '...' if score.notes and len(score.notes) > 30 else (score.notes or 'N/A')
                ])
            
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
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
    
    # Check health tips from database
    tips = HealthTip.query.all()
    for tip in tips:
        if tip.symptoms:
            symptoms = [s.strip().lower() for s in tip.symptoms.split(',')]
            for symptom in symptoms:
                if symptom and symptom in message_lower:
                    return tip.content
    
    # Default responses
    health_advice = {
        'fever': "For fever: Rest, drink fluids, take medication, use cool compress. See doctor if high fever persists.",
        'headache': "For headache: Rest in dark room, stay hydrated, avoid triggers. Consider pain relief medication.",
        'migraine': "For migraine: Rest in quiet dark room, cold compress, hydration, avoid lights/sounds. Medication if needed.",
        'cold': "For cold: Rest, fluids, humidifier, over-the-counter meds. See doctor if symptoms worsen.",
        'cough': "For cough: Honey tea, humidifier, rest. See doctor if persistent or with fever.",
    }
    
    for symptom, advice in health_advice.items():
        if symptom in message_lower:
            return advice
    
    if any(word in message_lower for word in ['hello', 'hi', 'hey']):
        return f"Hello {user.name}! How can I help with your health today?"
    else:
        return "I understand you're not feeling well. Could you describe your symptoms in more detail?"

if __name__ == '__main__':
    init_db()
    app.run(debug=True)