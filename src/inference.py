import joblib
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def load_model(model_path):
    """Load trained model from file"""
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Model file {model_path} not found!")
        return None

def main():
    # Load the same dataset used for training
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Load trained model
    model = load_model('model_train.pkl')
    
    if model is None:
        return
    
    # Make predictions
    predictions = model.predict(X)
    
    # Calculate metrics
    accuracy = accuracy_score(y, predictions)
    
    print("="*50)
    print("INFERENCE RESULTS")
    print("="*50)
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Total predictions: {len(predictions)}")
    
    # Show classification report
    print("\nClassification Report:")
    print(classification_report(y, predictions))
    
    # Show some example predictions
    print("\nSample Predictions:")
    for i in range(10):
        print(f"Actual: {y[i]}, Predicted: {predictions[i]}")

if __name__ == "__main__":
    main()
