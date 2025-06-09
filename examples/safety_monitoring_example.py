"""
Example demonstrating the usage of the Safety Monitoring module.

This example shows how to set up safety constraints and monitor model predictions
to ensure they satisfy safety requirements.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.safety_monitoring.safety_monitor import SafetyMonitor, SafetyViolationError

def create_synthetic_data(n_samples: int = 1000) -> tuple:
    """
    Create synthetic data for a medical treatment dosage prediction model.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (features, target)
    """
    np.random.seed(42)
    
    # Patient features
    age = np.random.normal(50, 15, n_samples)
    weight = np.random.normal(70, 15, n_samples)
    severity = np.random.uniform(0, 1, n_samples)
    
    # Calculate safe dosage based on features
    safe_dosage = (
        0.5 * weight / 70 +  # Base dosage adjusted for weight
        0.2 * severity +     # Increase for severity
        0.1 * (age / 50)    # Small adjustment for age
    )
    
    # Create feature matrix
    X = pd.DataFrame({
        'age': age,
        'weight': weight,
        'severity': severity
    })
    
    return X, safe_dosage

def define_safety_constraints():
    """Define safety constraints for the medical dosage model."""
    
    def max_dosage_constraint(data, predictions):
        """Ensure dosage doesn't exceed maximum safe limit."""
        max_safe_dosage = 2.0
        return np.all(predictions <= max_safe_dosage)
    
    def weight_adjusted_constraint(data, predictions):
        """Ensure dosage is appropriate for patient weight."""
        max_dosage_per_kg = 0.05
        weight_adjusted_limit = data['weight'] * max_dosage_per_kg
        return np.all(predictions <= weight_adjusted_limit)
    
    def age_safety_constraint(data, predictions):
        """Additional safety checks for elderly patients."""
        elderly_mask = data['age'] > 70
        if not np.any(elderly_mask):
            return True
        return np.all(predictions[elderly_mask] <= 1.0)
    
    constraints = {
        'max_dosage': max_dosage_constraint,
        'weight_adjusted': weight_adjusted_constraint,
        'elderly_safety': age_safety_constraint
    }
    
    thresholds = {
        'max_dosage': 2.0,
        'weight_adjusted': 0.05,
        'elderly_safety': 1.0
    }
    
    return constraints, thresholds

def custom_alert_handler(alert_info):
    """Custom handler for safety alerts."""
    print(f"\nSAFETY ALERT at {alert_info['timestamp']}:")
    print(f"Constraint violated: {alert_info['constraint']}")
    print(f"Details: {alert_info['details']}")

def main():
    # Create synthetic dataset
    print("Creating synthetic medical dosage dataset...")
    X, y = create_synthetic_data()
    
    # Train a simple model
    print("\nTraining dosage prediction model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Initialize safety monitor
    print("\nInitializing safety monitoring system...")
    constraints, thresholds = define_safety_constraints()
    safety_monitor = SafetyMonitor(
        constraints=constraints,
        thresholds=thresholds,
        alert_callback=custom_alert_handler
    )
    
    # Make predictions
    print("\nMaking predictions and checking safety constraints...")
    predictions = model.predict(X)
    
    # Check safety constraints
    try:
        metrics = safety_monitor.check_constraints(
            X,
            predictions,
            raise_on_violation=True
        )
        print("\nAll safety constraints satisfied!")
    except SafetyViolationError as e:
        print(f"\nSafety violation detected: {str(e)}")
    
    # Enforce safety constraints
    print("\nEnforcing safety constraints on predictions...")
    safe_predictions = safety_monitor.enforce_constraints(
        X,
        predictions,
        fallback_strategy='conservative'
    )
    
    # Compare original and safe predictions
    print("\nPrediction Statistics:")
    print(f"Original - Mean: {predictions.mean():.3f}, Max: {predictions.max():.3f}")
    print(f"Safe     - Mean: {safe_predictions.mean():.3f}, Max: {safe_predictions.max():.3f}")
    
    # Add a new constraint
    print("\nAdding new safety constraint...")
    def severity_constraint(data, predictions):
        """Ensure higher dosage for severe cases."""
        severe_mask = data['severity'] > 0.8
        if not np.any(severe_mask):
            return True
        return np.all(predictions[severe_mask] >= 0.5)
    
    safety_monitor.add_constraint(
        name='severity_requirement',
        constraint_fn=severity_constraint,
        threshold=0.5
    )
    
    # Generate safety report
    print("\nFinal Safety Report:")
    print(safety_monitor.get_safety_report())

if __name__ == "__main__":
    main() 