# model_train.py - Advanced AI Model Training
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

class EpidemicPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.best_model = None
        self.best_score = 0
        
    def load_and_preprocess_data(self, filepath='epidemic_dataset_better.csv'):
        """Load and preprocess the epidemic dataset"""
        try:
            df = pd.read_csv(filepath)
            print(f"✅ Loaded {len(df)} records from dataset")
        except FileNotFoundError:
            print("⚠️  Dataset not found. Creating sample data...")
            df = self.create_sample_dataset()
            df.to_csv(filepath, index=False)
        
        # Feature engineering
        df = self.engineer_features(df)
        
        # Prepare features and target
        feature_columns = ['population', 'current_cases', 'transmission_rate', 'vaccination_rate', 
                          'mobility_index', 'temperature', 'humidity', 'population_density',
                          'healthcare_capacity', 'region_encoded', 'season_encoded']
        
        self.feature_columns = feature_columns
        
        # Handle missing values
        df[feature_columns] = df[feature_columns].fillna(df[feature_columns].mean())
        
        X = df[feature_columns]
        y = df['future_cases']  # Predict cases 7 days ahead
        
        return X, y, df
    
    def engineer_features(self, df):
        """Create additional features for better prediction"""
        # Encode categorical variables
        le_region = LabelEncoder()
        le_season = LabelEncoder()
        
        df['region_encoded'] = le_region.fit_transform(df['region'])
        df['season_encoded'] = le_season.fit_transform(df['season'])
        
        # Store encoders
        self.encoders['region'] = le_region
        self.encoders['season'] = le_season
        
        # Calculate derived features
        df['cases_per_capita'] = df['current_cases'] / df['population']
        df['effective_reproduction_rate'] = df['transmission_rate'] * (1 - df['vaccination_rate']/100)
        df['healthcare_strain'] = df['current_cases'] / df['healthcare_capacity']
        
        # Add time-based features
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        return df
    
    def train_model(self, X, y):
        """Train multiple models and select the best one"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['main'] = scaler
        
        # Define models
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'neural_network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }
        
        # Train and evaluate models
        for name, model in models.items():
            print(f"\n🔄 Training {name}...")
            
            if name == 'neural_network':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Evaluate model
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"   MSE: {mse:.4f}")
            print(f"   R²: {r2:.4f}")
            print(f"   MAE: {mae:.4f}")
            
            # Cross-validation
            if name == 'neural_network':
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            print(f"   CV R² Score: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
            
            # Store model
            self.models[name] = model
            
            # Track best model
            if r2 > self.best_score:
                self.best_score = r2
                self.best_model = model
                print(f"   ✅ New best model: {name}")
        
        return self.best_model
    
    def predict(self, input_data):
        """Make prediction using the best model"""
        if self.best_model is None:
            raise ValueError("No trained model available")
        
        # Preprocess input
        if isinstance(input_data, dict):
            input_data = self.preprocess_input(input_data)
        
        # Scale if neural network
        if isinstance(self.best_model, MLPRegressor):
            input_data = self.scalers['main'].transform([input_data])
            prediction = self.best_model.predict(input_data)[0]
        else:
            prediction = self.best_model.predict([input_data])[0]
        
        return max(0, prediction)  # Ensure non-negative prediction
    
    def preprocess_input(self, data):
        """Preprocess input data for prediction"""
        # Create feature vector
        features = []
        
        # Basic features
        features.extend([
            data['population'],
            data['current_cases'],
            data['transmission_rate'],
            data['vaccination_rate'],
            data['mobility_index']
        ])
        
        # Default values for missing features
        features.extend([
            data.get('temperature', 20),  # Default temperature
            data.get('humidity', 50),     # Default humidity
            data.get('population_density', data['population'] / 1000),  # Estimated density
            data.get('healthcare_capacity', data['population'] * 0.003),  # 3 beds per 1000
        ])
        
        # Encode categorical variables
        region_encoded = 0
        if data['region'] in self.encoders['region'].classes_:
            region_encoded = self.encoders['region'].transform([data['region']])[0]
        
        season_encoded = 0  # Default to winter
        if 'season' in data and data['season'] in self.encoders['season'].classes_:
            season_encoded = self.encoders['season'].transform([data['season']])[0]
        
        features.extend([region_encoded, season_encoded])
        
        return features
    
    def create_sample_dataset(self):
        """Create a sample dataset for training"""
        np.random.seed(42)
        n_samples = 5000
        
        regions = ['north-america', 'europe', 'asia', 'africa', 'south-america', 'oceania']
        seasons = ['spring', 'summer', 'autumn', 'winter']
        
        data = {
            'date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
            'region': np.random.choice(regions, n_samples),
            'season': np.random.choice(seasons, n_samples),
            'population': np.random.randint(100000, 50000000, n_samples),
            'current_cases': np.random.randint(10, 100000, n_samples),
            'transmission_rate': np.random.uniform(0.5, 4.0, n_samples),
            'vaccination_rate': np.random.uniform(0, 95, n_samples),
            'mobility_index': np.random.randint(1, 11, n_samples),
            'temperature': np.random.uniform(-10, 40, n_samples),
            'humidity': np.random.uniform(20, 90, n_samples),
            'population_density': np.random.uniform(10, 10000, n_samples),
            'healthcare_capacity': np.random.randint(100, 50000, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Generate synthetic future cases based on epidemiological model
        df['future_cases'] = self.generate_synthetic_targets(df)
        
        return df
    
    def generate_synthetic_targets(self, df):
        """Generate synthetic target values using epidemiological principles"""
        targets = []
        
        for _, row in df.iterrows():
            # Basic SIR model simulation
            beta = row['transmission_rate'] * (row['mobility_index'] / 10) * (1 - row['vaccination_rate'] / 100)
            gamma = 0.07  # Recovery rate
            
            # Environmental factors
            temp_factor = 1.0 - (abs(row['temperature'] - 20) / 100)  # Optimal at 20°C
            humidity_factor = 1.0 - (abs(row['humidity'] - 50) / 200)  # Optimal at 50%
            
            # Healthcare capacity effect
            capacity_factor = max(0.5, 1.0 - (row['current_cases'] / row['healthcare_capacity']))
            
            # Calculate expected cases in 7 days
            r_effective = beta * temp_factor * humidity_factor * capacity_factor
            future_cases = row['current_cases'] * (1 + r_effective - gamma) ** 7
            
            # Add some noise
            noise = np.random.normal(0, future_cases * 0.1)
            targets.append(max(0, future_cases + noise))
        
        return targets
    
    def save_model(self, filepath='models/epidemic_model.pkl'):
        """Save the trained model"""
        model_data = {
            'best_model': self.best_model,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_columns': self.feature_columns,
            'best_score': self.best_score
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath='models/epidemic_model.pkl'):
        """Load a trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.best_model = model_data['best_model']
            self.scalers = model_data['scalers']
            self.encoders = model_data['encoders']
            self.feature_columns = model_data['feature_columns']
            self.best_score = model_data['best_score']
            
            print(f"✅ Model loaded from {filepath}")
            return True
        except FileNotFoundError:
            print(f"⚠️  Model file not found: {filepath}")
            return False

def main():
    """Main training script"""
    print("🚀 Starting Epidemic Prediction Model Training...")
    
    # Initialize predictor
    predictor = EpidemicPredictor()
    
    # Load and preprocess data
    X, y, df = predictor.load_and_preprocess_data()
    
    # Train model
    best_model = predictor.train_model(X, y)
    
    # Save model
    predictor.save_model()
    
    print(f"\n✅ Training completed! Best model R² score: {predictor.best_score:.4f}")
    print(f"📊 Model type: {type(best_model).__name__}")
    
    return predictor

if __name__ == "__main__":
    main()