"""
Example demonstrating the usage of the Bias Detection module.

This example creates a synthetic dataset with potential bias and analyzes it
using the BiasDetector class.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.bias_detection.bias_detector import BiasDetector

def create_biased_dataset(n_samples: int = 1000, bias_factor: float = 0.7) -> tuple:
    """
    Create a synthetic dataset with intentional bias for demonstration.
    
    Args:
        n_samples: Number of samples to generate
        bias_factor: Strength of bias (0 to 1)
        
    Returns:
        Tuple of (features, labels, protected_attributes)
    """
    # Generate protected attributes
    gender = np.random.choice(['male', 'female'], size=n_samples)
    age = np.random.randint(18, 70, size=n_samples)
    
    # Generate features with bias
    income = np.random.normal(50000, 20000, n_samples)
    # Introduce bias: higher income for one gender
    income[gender == 'male'] *= 1.2
    
    education_years = np.random.normal(16, 3, n_samples)
    # Introduce bias: more education years for one gender
    education_years[gender == 'male'] *= 1.1
    
    # Create biased labels
    probability = 0.5 + bias_factor * (
        (gender == 'male').astype(float) * 0.3 +
        (age > 40).astype(float) * 0.2
    )
    labels = np.random.binomial(1, probability)
    
    # Create feature matrix
    features = pd.DataFrame({
        'income': income,
        'education_years': education_years,
    })
    
    # Create protected attributes DataFrame
    protected_attributes = pd.DataFrame({
        'gender': gender,
        'age': age
    })
    
    return features, labels, protected_attributes

def main():
    # Create synthetic dataset
    print("Creating synthetic dataset with bias...")
    features, labels, protected_attributes = create_biased_dataset()
    
    # Split the data
    X_train, X_test, y_train, y_test, protected_train, protected_test = train_test_split(
        features, labels, protected_attributes, test_size=0.2, random_state=42
    )
    
    # Train a simple model
    print("\nTraining a RandomForest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Initialize bias detector
    print("\nAnalyzing bias in model predictions...")
    bias_detector = BiasDetector(sensitive_features=['gender', 'age'])
    
    # Analyze bias
    bias_metrics = bias_detector.analyze_bias(
        y_true=y_test,
        y_pred=y_pred,
        protected_attributes=protected_test
    )
    
    # Print results
    print("\nBias Analysis Results:")
    print(f"Demographic Parity: {bias_metrics.demographic_parity:.3f}")
    print(f"Equal Opportunity: {bias_metrics.equal_opportunity:.3f}")
    print(f"Disparate Impact: {bias_metrics.disparate_impact:.3f}")
    
    print("\nGroup-specific metrics:")
    for feature, metrics in bias_metrics.group_fairness.items():
        print(f"\n{feature.capitalize()} groups:")
        for group, group_metrics in metrics.items():
            print(f"  {group}:")
            for metric_name, value in group_metrics.items():
                print(f"    {metric_name}: {value:.3f}")

if __name__ == "__main__":
    main() 