# utils.py - Utility functions for epidemic prediction
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data preprocessing and validation"""
    
    def __init__(self):
        self.required_fields = ['population', 'current_cases', 'transmission_rate', 
                               'vaccination_rate', 'mobility_index', 'region']
        self.optional_fields = ['temperature', 'humidity', 'population_density', 
                               'healthcare_capacity', 'season']
    
    def validate_input_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean input data"""
        validated_data = {}
        
        # Check required fields
        for field in self.required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
            validated_data[field] = data[field]
        
        # Validate numeric fields
        numeric_fields = ['population', 'current_cases', 'transmission_rate', 
                         'vaccination_rate', 'mobility_index']
        
        for field in numeric_fields:
            if field in validated_data:
                try:
                    validated_data[field] = float(validated_data[field])
                except ValueError:
                    raise ValueError(f"Invalid numeric value for {field}")
        
        # Validate ranges
        if validated_data['population'] <= 0:
            raise ValueError("Population must be positive")
        
        if validated_data['current_cases'] < 0:
            raise ValueError("Current cases cannot be negative")
        
        if validated_data['transmission_rate'] < 0:
            raise ValueError("Transmission rate cannot be negative")
        
        if not 0 <= validated_data['vaccination_rate'] <= 100:
            raise ValueError("Vaccination rate must be between 0 and 100")
        
        if not 1 <= validated_data['mobility_index'] <= 10:
            raise ValueError("Mobility index must be between 1 and 10")
        
        # Add optional fields with defaults
        validated_data.update({
            'temperature': data.get('temperature', 20.0),
            'humidity': data.get('humidity', 50.0),
            'population_density': data.get('population_density', 
                                         validated_data['population'] / 1000),
            'healthcare_capacity': data.get('healthcare_capacity', 
                                          validated_data['population'] * 0.003),
            'season': data.get('season', self._get_current_season())
        })
        
        return validated_data
    
    def preprocess_prediction_data(self, data: Dict[str, Any]) -> np.ndarray:
        """Preprocess data for model prediction"""
        validated_data = self.validate_input_data(data)
        
        # Convert to feature array (order matters for model)
        features = np.array([
            validated_data['population'],
            validated_data['current_cases'],
            validated_data['transmission_rate'],
            validated_data['vaccination_rate'],
            validated_data['mobility_index'],
            validated_data['temperature'],
            validated_data['humidity'],
            validated_data['population_density'],
            validated_data['healthcare_capacity'],
            self._encode_region(validated_data['region']),
            self._encode_season(validated_data['season'])
        ])
        
        return features.reshape(1, -1)
    
    def validate_new_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate new data for dataset updates"""
        validated_data = self.validate_input_data(data)
        
        # Add timestamp
        validated_data['timestamp'] = datetime.now().isoformat()
        validated_data['date'] = datetime.now().strftime('%Y-%m-%d')
        
        return validated_data
    
    def create_sample_dataset(self, n_samples: int = 1000) -> pd.DataFrame:
        """Create sample dataset for testing"""
        np.random.seed(42)
        
        regions = ['north-america', 'europe', 'asia', 'africa', 'south-america', 'oceania']
        seasons = ['spring', 'summer', 'autumn', 'winter']
        
        data = {
            'date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
            'region': np.random.choice(regions, n_samples),
            'season': np.random.choice(seasons, n_samples),
            'population': np.random.randint(100000, 50000000, n_samples),
            'current_cases': np.random.randint(10, 100000, n_samples),
            'cases': np.random.randint(10, 100000, n_samples),
            'recovered': np.random.randint(5, 80000, n_samples),
            'deaths': np.random.randint(0, 5000, n_samples),
            'transmission_rate': np.random.uniform(0.5, 4.0, n_samples),
            'vaccination_rate': np.random.uniform(0, 95, n_samples),
            'mobility_index': np.random.randint(1, 11, n_samples),
            'temperature': np.random.uniform(-10, 40, n_samples),
            'humidity': np.random.uniform(20, 90, n_samples),
            'population_density': np.random.uniform(10, 10000, n_samples),
            'healthcare_capacity': np.random.randint(100, 50000, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Generate future cases based on epidemiological model
        df['future_cases'] = self._generate_future_cases(df)
        
        logger.info(f"Created sample dataset with {n_samples} records")
        return df
    
    def _generate_future_cases(self, df: pd.DataFrame) -> List[float]:
        """Generate synthetic future cases"""
        future_cases = []
        
        for _, row in df.iterrows():
            # SIR model parameters
            beta = row['transmission_rate'] * (row['mobility_index'] / 10) * (1 - row['vaccination_rate'] / 100)
            gamma = 0.07  # Recovery rate
            
            # Environmental factors
            temp_factor = 1.0 - (abs(row['temperature'] - 20) / 100)
            humidity_factor = 1.0 - (abs(row['humidity'] - 50) / 200)
            
            # Healthcare capacity effect
            capacity_factor = max(0.5, 1.0 - (row['current_cases'] / row['healthcare_capacity']))
            
            # Calculate projected cases
            r_effective = beta * temp_factor * humidity_factor * capacity_factor
            projected = row['current_cases'] * (1 + r_effective - gamma) ** 7
            
            # Add noise
            noise = np.random.normal(0, projected * 0.1)
            future_cases.append(max(0, projected + noise))
        
        return future_cases
    
    def _encode_region(self, region: str) -> int:
        """Encode region to numeric value"""
        region_mapping = {
            'north-america': 0, 'europe': 1, 'asia': 2, 
            'africa': 3, 'south-america': 4, 'oceania': 5, 'global': 6
        }
        return region_mapping.get(region, 0)
    
    def _encode_season(self, season: str) -> int:
        """Encode season to numeric value"""
        season_mapping = {'spring': 0, 'summer': 1, 'autumn': 2, 'winter': 3}
        return season_mapping.get(season, 0)
    
    def _get_current_season(self) -> str:
        """Get current season based on date"""
        month = datetime.now().month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'

class RiskAssessment:
    """Handles risk assessment and calculations"""
    
    def __init__(self):
        self.risk_thresholds = {
            'low': 0.01,     # 1% of population
            'medium': 0.05,   # 5% of population
            'high': 0.10      # 10% of population
        }
    
    def calculate_risk_level(self, prediction: np.ndarray, population: int) -> str:
        """Calculate risk level based on prediction"""
        if isinstance(prediction, (list, np.ndarray)):
            peak_cases = np.max(prediction)
        else:
            peak_cases = prediction
        
        infection_rate = peak_cases / population
        
        if infection_rate <= self.risk_thresholds['low']:
            return 'low'
        elif infection_rate <= self.risk_thresholds['medium']:
            return 'medium'
        else:
            return 'high'
    
    def calculate_doubling_time(self, transmission_rate: float, recovery_rate: float = 0.07) -> float:
        """Calculate doubling time for epidemic"""
        growth_rate = transmission_rate - recovery_rate
        if growth_rate <= 0:
            return float('inf')
        return np.log(2) / growth_rate
    
    def calculate_herd_immunity_threshold(self, transmission_rate: float) -> float:
        """Calculate herd immunity threshold"""
        return 1 - (1 / transmission_rate)
    
    def assess_healthcare_capacity(self, predicted_cases: np.ndarray, 
                                 healthcare_capacity: int) -> Dict[str, Any]:
        """Assess healthcare system capacity"""
        peak_cases = np.max(predicted_cases)
        capacity_utilization = (peak_cases * 0.1) / healthcare_capacity  # Assume 10% need hospitalization
        
        return {
            'peak_cases': int(peak_cases),
            'capacity_utilization': min(capacity_utilization, 2.0),  # Cap at 200%
            'capacity_exceeded': capacity_utilization > 1.0,
            'additional_capacity_needed': max(0, int(peak_cases * 0.1 - healthcare_capacity))
        }

class EpidemicMetrics:
    """Calculate various epidemic metrics"""
    
    @staticmethod
    def calculate_reproduction_number(new_cases: List[float], 
                                    generation_time: float = 7.0) -> float:
        """Calculate effective reproduction number (Rt)"""
        if len(new_cases) < 2:
            return 0.0
        
        recent_cases = sum(new_cases[-int(generation_time):])
        previous_cases = sum(new_cases[-int(2*generation_time):-int(generation_time)])
        
        if previous_cases == 0:
            return 0.0
        
        return recent_cases / previous_cases
    
    @staticmethod
    def calculate_attack_rate(total_cases: int, population: int) -> float:
        """Calculate attack rate (proportion of population infected)"""
        return (total_cases / population) * 100
    
    @staticmethod
    def calculate_case_fatality_rate(deaths: int, cases: int) -> float:
        """Calculate case fatality rate"""
        if cases == 0:
            return 0.0
        return (deaths / cases) * 100
    
    @staticmethod
    def calculate_growth_rate(cases: List[float], days: int = 7) -> float:
        """Calculate growth rate over specified days"""
        if len(cases) < days + 1:
            return 0.0
        
        current = cases[-1]
        previous = cases[-(days + 1)]
        
        if previous == 0:
            return 0.0
        
        return ((current / previous) ** (1/days) - 1) * 100

class DataValidator:
    """Validate and clean data inputs"""
    
    @staticmethod
    def validate_positive_number(value: Any, field_name: str) -> float:
        """Validate that a value is a positive number"""
        try:
            num_value = float(value)
            if num_value < 0:
                raise ValueError(f"{field_name} must be non-negative")
            return num_value
        except (ValueError, TypeError):
            raise ValueError(f"{field_name} must be a valid number")
    
    @staticmethod
    def validate_percentage(value: Any, field_name: str) -> float:
        """Validate that a value is a valid percentage (0-100)"""
        try:
            num_value = float(value)
            if not 0 <= num_value <= 100:
                raise ValueError(f"{field_name} must be between 0 and 100")
            return num_value
        except (ValueError, TypeError):
            raise ValueError(f"{field_name} must be a valid percentage")
    
    @staticmethod
    def validate_region(region: str) -> str:
        """Validate region input"""
        valid_regions = ['north-america', 'europe', 'asia', 'africa', 
                        'south-america', 'oceania', 'global']
        if region.lower() not in valid_regions:
            raise ValueError(f"Invalid region. Must be one of: {', '.join(valid_regions)}")
        return region.lower()

class ConfigManager:
    """Manage application configuration"""
    
    def __init__(self, config_file: str = 'config.json'):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'model': {
                'prediction_window': 7,
                'retrain_interval': 24,  # hours
                'min_accuracy': 0.8
            },
            'api': {
                'rate_limit': 100,  # requests per hour
                'cache_ttl': 300,   # seconds
                'timeout': 30       # seconds
            },
            'data': {
                'max_records': 100000,
                'backup_interval': 24,  # hours
                'data_retention': 365   # days
            },
            'alerts': {
                'high_risk_threshold': 0.1,
                'capacity_threshold': 0.9,
                'growth_rate_threshold': 0.2
            }
        }
    
    def save_config(self):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, default)
            if value is None:
                return default
        return value

class AlertSystem:
    """Handle epidemic alerts and notifications"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.alert_history = []
    
    def check_alerts(self, prediction_result: Dict[str, Any], 
                    input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for alert conditions"""
        alerts = []
        
        # High risk alert
        if prediction_result['risk_level'] == 'high':
            alerts.append({
                'type': 'high_risk',
                'severity': 'critical',
                'message': f"High epidemic risk detected for {input_data['region']}",
                'timestamp': datetime.now().isoformat()
            })
        
        # Healthcare capacity alert
        peak_cases = prediction_result['peak_cases']
        healthcare_capacity = input_data.get('healthcare_capacity', 
                                           input_data['population'] * 0.003)
        
        if peak_cases * 0.1 > healthcare_capacity * 0.9:  # 90% capacity threshold
            alerts.append({
                'type': 'capacity_warning',
                'severity': 'warning',
                'message': f"Healthcare capacity may be exceeded in {input_data['region']}",
                'timestamp': datetime.now().isoformat()
            })
        
        # Rapid growth alert
        if prediction_result.get('doubling_time', float('inf')) < 3:
            alerts.append({
                'type': 'rapid_growth',
                'severity': 'warning',
                'message': f"Rapid case growth detected in {input_data['region']}",
                'timestamp': datetime.now().isoformat()
            })
        
        # Store alert history
        self.alert_history.extend(alerts)
        
        return alerts
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [alert for alert in self.alert_history 
                if datetime.fromisoformat(alert['timestamp']) > cutoff_time]

class CacheManager:
    """Simple in-memory cache for API responses"""
    
    def __init__(self, ttl: int = 300):
        self.cache = {}
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set cached value"""
        self.cache[key] = (value, datetime.now())
    
    def clear(self):
        """Clear all cached values"""
        self.cache.clear()
    
    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)

# Utility functions
def create_cache_key(data: Dict[str, Any]) -> str:
    """Create cache key from input data"""
    key_parts = [
        str(data.get('region', 'global')),
        str(data.get('population', 0)),
        str(data.get('current_cases', 0)),
        str(data.get('transmission_rate', 0)),
        str(data.get('vaccination_rate', 0)),
        str(data.get('mobility_index', 0))
    ]
    return '_'.join(key_parts)

def format_number(number: float) -> str:
    """Format number with appropriate units"""
    if number >= 1_000_000:
        return f"{number/1_000_000:.1f}M"
    elif number >= 1_000:
        return f"{number/1_000:.1f}K"
    else:
        return str(int(number))

def calculate_confidence_interval(prediction: float, 
                                confidence_level: float = 0.95) -> tuple:
    """Calculate confidence interval for prediction"""
    # Simplified confidence interval calculation
    margin = prediction * 0.1 * (confidence_level / 0.95)
    return (max(0, prediction - margin), prediction + margin)

def export_data_to_csv(data: List[Dict[str, Any]], 
                      filename: str = 'epidemic_data.csv'):
    """Export data to CSV file"""
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    logger.info(f"Data exported to {filename}")

def import_data_from_csv(filename: str) -> List[Dict[str, Any]]:
    """Import data from CSV file"""
    try:
        df = pd.read_csv(filename)
        return df.to_dict('records')
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return []

# Initialize global instances
config_manager = ConfigManager()
cache_manager = CacheManager(ttl=config_manager.get('api.cache_ttl', 300))
alert_system = AlertSystem(config_manager)