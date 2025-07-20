import json
import pickle
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def train_model(X, y, config):
    """Train logistic regression model with given config"""
    model = LogisticRegression(
        C=config['C'],
        solver=config['solver'],
        max_iter=config['max_iter'],
        random_state=config.get('random_state', 42)
    )
    model.fit(X, y)
    return model

def main():
    # Load dataset
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Load configuration
    config = load_config('config/config.json')
    
    # Train model
    model = train_model(X, y, config)
    
    # Calculate accuracy
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    print(f"Training completed!")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Loss: {1 - accuracy:.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'model_train.pkl')
    print("Model saved as model_train.pkl")

if __name__ == "__main__":
    main()
