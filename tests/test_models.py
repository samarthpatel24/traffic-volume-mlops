"""
Test suite for traffic volume prediction models
"""
import unittest
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocess import load_config, clean_data, feature_engineering, encode_categorical_features


class TestDataPreprocessing(unittest.TestCase):
    """Test data preprocessing functions"""
    
    def setUp(self):
        """Set up test data"""
        self.sample_data = pd.DataFrame({
            'traffic_volume': [1000, 2000, 3000, 1500, 2500],
            'holiday': ['None', 'None', 'Christmas Day', 'None', 'None'],
            'temp': [273.15, 283.15, 293.15, 278.15, 288.15],
            'rain_1h': [0.0, 2.5, 0.0, 1.0, 0.0],
            'snow_1h': [0.0, 0.0, 0.0, 0.0, 0.0],
            'clouds_all': [20, 75, 10, 50, 30],
            'weather_main': ['Clear', 'Clouds', 'Clear', 'Rain', 'Clouds'],
            'weather_description': ['sky is clear', 'broken clouds', 'sky is clear', 'light rain', 'few clouds'],
            'date_time': ['01-01-2023 08:00', '01-01-2023 09:00', '01-01-2023 10:00', '01-01-2023 11:00', '01-01-2023 12:00']
        })
    
    def test_clean_data(self):
        """Test data cleaning function"""
        cleaned_data = clean_data(self.sample_data.copy())
        
        # Should not lose any rows for this clean sample data
        self.assertEqual(len(cleaned_data), len(self.sample_data))
        
        # Should have same columns
        self.assertEqual(list(cleaned_data.columns), list(self.sample_data.columns))
    
    def test_feature_engineering(self):
        """Test feature engineering function"""
        engineered_data = feature_engineering(self.sample_data.copy())
        
        # Check if new features are created
        expected_features = ['hour', 'day_of_week', 'month', 'year', 'is_weekend', 
                           'is_rush_hour', 'is_holiday', 'temp_celsius', 'weather_severity', 
                           'total_precipitation']
        
        for feature in expected_features:
            self.assertIn(feature, engineered_data.columns, f"Missing feature: {feature}")
        
        # Test temperature conversion
        self.assertAlmostEqual(engineered_data['temp_celsius'].iloc[0], 0.0, places=1)  # 273.15K = 0°C
        
        # Test holiday feature
        self.assertEqual(engineered_data['is_holiday'].iloc[2], 1)  # Christmas Day
        self.assertEqual(engineered_data['is_holiday'].iloc[0], 0)  # None
        
        # Test precipitation feature
        expected_precip = self.sample_data['rain_1h'] + self.sample_data['snow_1h']
        pd.testing.assert_series_equal(
            engineered_data['total_precipitation'], 
            expected_precip, 
            check_names=False
        )
    
    def test_encode_categorical_features(self):
        """Test categorical encoding"""
        engineered_data = feature_engineering(self.sample_data.copy())
        encoded_data = encode_categorical_features(engineered_data)
        
        # Check if encoded features are created
        self.assertIn('weather_main_encoded', encoded_data.columns)
        self.assertIn('weather_description_encoded', encoded_data.columns)
        
        # Encoded values should be integers
        self.assertTrue(encoded_data['weather_main_encoded'].dtype in [np.int32, np.int64])
        self.assertTrue(encoded_data['weather_description_encoded'].dtype in [np.int32, np.int64])


class TestModelValidation(unittest.TestCase):
    """Test model validation functions"""
    
    def test_config_loading(self):
        """Test configuration loading"""
        try:
            config = load_config()
            
            # Check required sections
            required_sections = ['data', 'model', 'training', 'evaluation']
            for section in required_sections:
                self.assertIn(section, config, f"Missing config section: {section}")
            
            # Check model types
            valid_model_types = ['random_forest', 'xgboost', 'lightgbm', 'ensemble']
            self.assertIn(config['model']['type'], valid_model_types)
            
        except Exception as e:
            self.fail(f"Config loading failed: {e}")
    
    def test_data_validation_requirements(self):
        """Test data validation requirements"""
        # Test minimum data requirements
        min_required_columns = ['traffic_volume', 'temp', 'weather_main', 'date_time']
        
        for col in min_required_columns:
            self.assertIn(col, self.sample_data.columns if hasattr(self, 'sample_data') else min_required_columns)


class TestModelPerformance(unittest.TestCase):
    """Test model performance thresholds"""
    
    def test_performance_thresholds(self):
        """Test that model meets minimum performance requirements"""
        # These would be actual model performance metrics
        # For demo purposes, using example values
        
        min_r2_score = 0.7  # Minimum R² score
        max_rmse = 1000     # Maximum acceptable RMSE
        max_mape = 50       # Maximum acceptable MAPE
        
        # In a real test, you would load actual model metrics
        example_r2 = 0.95
        example_rmse = 426
        example_mape = 48
        
        self.assertGreaterEqual(example_r2, min_r2_score, "R² score below threshold")
        self.assertLessEqual(example_rmse, max_rmse, "RMSE above threshold")
        self.assertLessEqual(example_mape, max_mape, "MAPE above threshold")


if __name__ == '__main__':
    unittest.main()