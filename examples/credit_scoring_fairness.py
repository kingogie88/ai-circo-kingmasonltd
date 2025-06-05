"""
Example demonstrating fair credit scoring using the responsible AI implementation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.bias_detection.bias_detector import BiasDetector
from src.fairness_metrics.fairness_calculator import FairnessCalculator
from src.explainability.shap_explainer import ShapExplainer
from src.privacy_protection.differential_privacy import DifferentialPrivacy
from src.safety_monitoring.model_monitoring import ModelMonitor

def load_sample_data():
    """Load and preprocess sample credit scoring data."""
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    age = np.random.normal(40, 15, n_samples)
    income = np.random.lognormal(10, 1, n_samples)
    employment_years = np.random.poisson(5, n_samples)
    debt_ratio = np.random.beta(2, 5, n_samples)
    
    # Generate sensitive attributes
    gender = np.random.binomial(1, 0.5, n_samples)  # 0: female, 1: male
    race = np.random.choice(['A', 'B', 'C'], n_samples, p=[0.5, 0.3, 0.2])
    
    # Generate target (credit approval) with some bias
    base_prob = 0.7 * (income / income.max()) + 0.3 * (employment_years / employment_years.max())
    gender_bias = 0.1 * gender  # Intentional bias for demonstration
    race_bias = np.where(race == 'A', 0.1, np.where(race == 'B', 0, -0.1))
    
    approval_prob = base_prob + gender_bias + race_bias
    approval = (approval_prob > np.random.random(n_samples)).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'age': age,
        'income': income,
        'employment_years': employment_years,
        'debt_ratio': debt_ratio,
        'gender': gender,
        'race': race,
        'approved': approval
    })
    
    return data

def main():
    """Main function demonstrating responsible AI implementation."""
    print("Loading and preparing data...")
    data = load_sample_data()
    
    # Split features and target
    X = data.drop(['approved'], axis=1)
    y = data['approved']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Initialize responsible AI components
    bias_detector = BiasDetector(sensitive_features=['gender', 'race'])
    fairness_calc = FairnessCalculator(sensitive_features=['gender', 'race'])
    
    # Train model
    print("\nTraining model...")
    model.fit(X_train, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Bias detection
    print("\nPerforming bias detection...")
    bias_metrics = bias_detector.evaluate_bias(
        data=X_test,
        predictions=y_pred,
        target_col='approved'
    )
    print(bias_detector.get_bias_report())
    
    # Fairness assessment
    print("\nAssessing fairness metrics...")
    fairness_metrics = fairness_calc.evaluate_fairness(
        data=X_test,
        predictions=y_pred,
        target_col='approved'
    )
    print(fairness_calc.get_fairness_report())
    
    # Model explainability
    print("\nGenerating model explanations...")
    explainer = ShapExplainer(model, model_type="tree")
    shap_values = explainer.explain_dataset(X_test)
    feature_importance = explainer.get_feature_importance()
    print("\nFeature Importance:")
    for feature, importance in feature_importance.items():
        print(f"- {feature}: {importance:.4f}")
    
    # Privacy protection
    print("\nApplying differential privacy...")
    dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
    X_private = dp.fit_transform(X_test)
    print(dp.get_privacy_report())
    
    # Safety monitoring
    print("\nSetting up safety monitoring...")
    monitor = ModelMonitor(
        model=model,
        feature_names=X.columns.tolist(),
        safety_constraints={
            'age': (18, 100),
            'income': (0, 1e7),
            'debt_ratio': (0, 1)
        }
    )
    
    # Check safety constraints
    is_safe, violations = monitor.check_safety_constraints(X_test)
    if not is_safe:
        print("\nSafety violations detected:")
        for violation in violations:
            print(f"- {violation}")
    
    # Monitor performance
    performance_score = monitor.monitor_performance(X_test, y_test)
    print(f"\nModel performance score: {performance_score:.4f}")
    
    # Check prediction safety
    pred_safe, warnings = monitor.check_prediction_safety(y_pred_proba)
    if not pred_safe:
        print("\nPrediction safety warnings:")
        for warning in warnings:
            print(f"- {warning}")
    
    # Get improvement suggestions
    print("\nSuggested improvements:")
    for suggestion in fairness_calc.suggest_improvements():
        print(f"- {suggestion}")

if __name__ == "__main__":
    main() 