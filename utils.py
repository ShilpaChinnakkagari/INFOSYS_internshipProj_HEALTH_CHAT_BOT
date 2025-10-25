import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from datetime import datetime, timedelta
import requests
from flask import current_app

def generate_health_graph(scores_data):
    """Generate health score progression graph"""
    try:
        plt.figure(figsize=(10, 6))
        dates = scores_data['dates']
        scores = scores_data['scores']
        
        plt.plot(dates, scores, marker='o', linewidth=2, markersize=8)
        plt.title('Health Score Progression', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Health Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        
        # Convert to base64 for HTML embedding
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        print(f"Error generating graph: {e}")
        return None

def calculate_health_score(user_data, recent_symptoms=None):
    """Calculate health score based on various factors"""
    base_score = 80  # Starting score
    
    # Age factor
    age = user_data.get('age', 30)
    if age < 30:
        base_score += 5
    elif age > 60:
        base_score -= 5
    
    # Recent activity factor (simplified)
    if recent_symptoms:
        base_score -= len(recent_symptoms) * 2
    
    # Ensure score stays within reasonable bounds
    return max(0, min(100, base_score))

def send_emergency_alert(user_info, location):
    """Send emergency alert to hospital services"""
    # This is a placeholder for actual emergency service integration
    # In production, integrate with services like Twilio for calls/SMS
    
    alert_data = {
        'user_name': user_info['name'],
        'user_age': user_info['age'],
        'location': location,
        'timestamp': datetime.utcnow().isoformat(),
        'emergency_type': 'medical'
    }
    
    # Log the emergency (in real implementation, send to emergency services)
    print(f"EMERGENCY ALERT: {alert_data}")
    
    return True

def get_health_tips(category='general', language='en'):
    """Get health tips based on category and language"""
    # This could be expanded to query the database
    tips = {
        'general': {
            'en': [
                "Stay hydrated by drinking at least 8 glasses of water daily",
                "Aim for 7-9 hours of quality sleep each night",
                "Include fruits and vegetables in every meal",
                "Take short breaks from sitting every 30 minutes"
            ],
            'es': [
                "Manténgase hidratado bebiendo al menos 8 vasos de agua al día",
                "Procure dormir 7-9 horas de calidad cada noche",
                "Incluya frutas y verduras en cada comida",
                "Tome descansos cortos de estar sentado cada 30 minutos"
            ]
        },
        'exercise': {
            'en': [
                "Start with 30 minutes of moderate exercise daily",
                "Include both cardio and strength training",
                "Stretch before and after workouts",
                "Find activities you enjoy to stay motivated"
            ]
        }
    }
    
    return tips.get(category, tips['general']).get(language, tips['general']['en'])

def validate_health_data(age, weight, height):
    """Validate basic health data"""
    errors = []
    
    if not (0 < age < 150):
        errors.append("Please enter a valid age")
    
    if weight and not (0 < weight < 500):  # kg
        errors.append("Please enter a valid weight")
    
    if height and not (0 < height < 300):  # cm
        errors.append("Please enter a valid height")
    
    return errors