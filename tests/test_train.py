import pytest
import json
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import load_config, train_model

class TestTraining:
    
    def test_config_file_loading(self):
        """Test that configuration file loads successfully"""
        config_path = 'config/config.json'
        
        # Test that file exists
        assert os.path.exists(config_path), "Config file does not exist"
        
        # Test that file loads successfully
        config = load_config(config_path)
        assert isinstance(config, dict), "Config should be a dictionary"
    
    def test_required_hyperparameters_exist(self):
        """Test that all required hyperparameters exist in config"""
        config = load_config('config/config.json')
        
        # Check required parameters exist
        required_params = ['C', 'solver', 'max_iter']
        for param in required_params:
            assert param in config, f"Required parameter '{param}' missing from config"
    
    def test_hyperparameter_types(self):
        """Test that hyperparameters have correct data types"""
        config = load_config('config/config.json')
        
        # Check data types
        assert isinstance(config['C'], (int, float)), "C should be numeric"
        assert isinstance(config['solver'], str), "solver should be string"
        assert isinstance(config['max_iter'], int), "max_iter should be integer"
    
    def test_model_creation(self):
        """Test that training function returns LogisticRegression object"""
        # Load test data
        digits = load_digits()
        X, y = digits.data, digits.target
        
        # Load config
        config = load_config('config/config.json')
        
        # Train model
        model = train_model(X, y, config)
        
        # Test model type
        assert isinstance(model, LogisticRegression), "Should return LogisticRegression object"
        
        # Test that model is fitted
        assert hasattr(model, 'coef_'), "Model should be fitted (have coef_ attribute)"
        assert hasattr(model, 'classes_'), "Model should be fitted (have classes_ attribute)"
    
    def test_model_accuracy_threshold(self):
        """Test that model achieves minimum accuracy threshold"""
        # Load data
        digits = load_digits()
        X, y = digits.data, digits.target
        
        # Load config and train model
        config = load_config('config/config.json')
        model = train_model(X, y, config)
        
        # Calculate accuracy
        accuracy = model.score(X, y)
        
        # Test accuracy threshold (should be reasonable for digits dataset)
        assert accuracy > 0.8, f"Model accuracy {accuracy:.4f} is below threshold of 0.8"
        print(f"Model achieved accuracy: {accuracy:.4f}")
