# app.py - Main Flask Application
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
from model_train import EpidemicPredictor
from utils import DataProcessor, RiskAssessment

app = Flask(__name__)
CORS(app)

# Initialize components
data_processor = DataProcessor()
risk_assessor = RiskAssessment()
predictor = EpidemicPredictor()

# Load trained model
model_loaded = predictor.load_model()
if not model_loaded:
    print("⚠️ No trained model found. Training new model...")
    X, y, _ = predictor.load_and_preprocess_data()
    predictor.train_model(X, y)
    predictor.save_model()
    print("✅ New model trained and ready")
else:
    print("✅ Trained model loaded successfully")

# Load dataset
try:
    df = pd.read_csv('epidemic_dataset_extended.csv')
    print(f"✅ Dataset loaded: {len(df)} records")
except FileNotFoundError:
    print("⚠️ Dataset not found. Creating sample data...")
    df = data_processor.create_sample_dataset()
    df.to_csv('epidemic_dataset_extended.csv', index=False)

@app.route('/')
def index():
    """Serve the frontend"""
    return render_template ('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_epidemic():
    """Main prediction endpoint"""
    try:
        data = request.json

        required_fields = ['population', 'current_cases', 'transmission_rate', 'vaccination_rate', 'mobility_index', 'region']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        processed_data = data_processor.validate_input_data(data)
        features = predictor.preprocess_input(processed_data)
        prediction = predictor.predict(features)

        risk_level = risk_assessor.calculate_risk_level([prediction], processed_data['population'])
        doubling_time = risk_assessor.calculate_doubling_time(processed_data['transmission_rate'], processed_data.get('recovery_rate', 0.07))

        response = {
            'prediction': [int(prediction)],
            'risk_level': risk_level,
            'doubling_time': doubling_time,
            'peak_cases': int(prediction),
            'total_predicted_cases': int(prediction),
            'confidence': round(predictor.best_score, 3),
            'timestamp': datetime.now().isoformat()
        }

        log_prediction(data, response)
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/historical-data')
def get_historical_data():
    """Get historical epidemic data for analysis"""
    try:
        region = request.args.get('region', 'global')
        days = int(request.args.get('days', 30))

        filtered_df = df[df['region'] == region].tail(days) if region != 'global' else df.tail(days)

        historical_data = {
            'dates': filtered_df['date'].tolist(),
            'cases': filtered_df['cases'].tolist(),
            'recovered': filtered_df['recovered'].tolist(),
            'deaths': filtered_df['deaths'].tolist(),
            'transmission_rate': filtered_df['transmission_rate'].tolist()
        }

        return jsonify(historical_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/real-time-metrics')
def get_real_time_metrics():
    """Get current real-time metrics"""
    try:
        region = request.args.get('region', 'global')
        latest_data = df[df['region'] == region].iloc[-1] if region != 'global' else df.iloc[-1]

        metrics = {
            'total_cases': int(latest_data['cases']),
            'active_cases': int(latest_data['cases'] - latest_data['recovered'] - latest_data['deaths']),
            'recovery_rate': round((latest_data['recovered'] / latest_data['cases']) * 100, 1),
            'mortality_rate': round((latest_data['deaths'] / latest_data['cases']) * 100, 2),
            'growth_rate': calculate_growth_rate(region),
            'last_updated': datetime.now().isoformat()
        }

        return jsonify(metrics)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/regions')
def get_regions():
    """Get available regions for prediction"""
    try:
        regions = df['region'].unique().tolist()
        return jsonify({'regions': regions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-performance')
def get_model_performance():
    """Get model performance metrics"""
    try:
        performance = {
            'accuracy': round(predictor.best_score, 3),
            'loss': round(1 - predictor.best_score, 3),
            'training_samples': len(df),
            'last_trained': datetime.now().isoformat(),
            'prediction_window': 7,
            'model_version': '1.0.0'
        }
        return jsonify(performance)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/update-data', methods=['POST'])
def update_data():
    """Update dataset with new data"""
    try:
        new_data = request.json
        processed_data = data_processor.validate_new_data(new_data)

        global df
        df = pd.concat([df, pd.DataFrame([processed_data])], ignore_index=True)
        df.to_csv('epidemic_dataset_extended.csv', index=False)

        return jsonify({'status': 'success', 'message': 'Data updated successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def calculate_growth_rate(region):
    """Calculate current growth rate"""
    region_data = df[df['region'] == region].tail(7) if region != 'global' else df.tail(7)
    if len(region_data) < 2:
        return 0
    current_cases = region_data.iloc[-1]['cases']
    prev_cases = region_data.iloc[-2]['cases']
    return round(((current_cases - prev_cases) / prev_cases) * 100, 2)

def log_prediction(input_data, prediction_result):
    """Log predictions for monitoring and analysis"""
    os.makedirs('logs', exist_ok=True)
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'input': input_data,
        'prediction': prediction_result,
        'model_version': '1.0.0'
    }
    with open('logs/predictions.json', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
