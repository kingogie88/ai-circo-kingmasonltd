"""
Example demonstrating the usage of the Bias Detection module.

This example shows how to use the BiasDetector class to analyze and detect
potential biases in a machine learning model for loan approval predictions.
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

def create_synthetic_loan_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create synthetic loan application dataset with potential biases.
    
    This creates a dataset that intentionally contains some biases
    to demonstrate the bias detection capabilities.
    """
    np.random.seed(42)
    
    # Create base features
    data = pd.DataFrame({
        'age': np.random.normal(40, 10, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'credit_score': np.random.normal(700, 50, n_samples),
        'years_employed': np.random.normal(10, 5, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'race': np.random.choice(['White', 'Black', 'Asian', 'Hispanic'], n_samples),
        'education': np.random.choice(
            ['High School', 'Bachelor', 'Master', 'PhD'],
            n_samples
        )
    })
    
    # Introduce some biased relationships
    data.loc[data['gender'] == 'F', 'income'] *= 0.8  # Gender pay gap
    data.loc[data['race'] == 'Black', 'credit_score'] *= 0.9  # Historical bias
    
    # Create target variable with bias
    target = (
        0.3 * data['age'] +
        0.4 * data['income'] / 10000 +
        0.2 * data['credit_score'] / 100 +
        0.1 * data['years_employed']
    )
    
    # Add more bias in target
    target[data['gender'] == 'F'] *= 0.9
    target[data['race'] == 'Black'] *= 0.9
    
    data['loan_approved'] = (target > target.mean()).astype(int)
    
    return data

def main():
    print("Bias Detection Example")
    print("=" * 50)
    
    # Create dataset
    print("\nCreating synthetic loan application dataset...")
    data = create_synthetic_loan_data()
    
    # Split features and target
    X = data.drop('loan_approved', axis=1)
    y = data['loan_approved']
    
    # Train a simple model
    print("\nTraining a RandomForest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(
        X.select_dtypes(include=[np.number]),  # Only numeric features
        y
    )
    
    # Generate predictions
    data['loan_approved_pred'] = model.predict(
        X.select_dtypes(include=[np.number])
    )
    
    # Initialize bias detector
    print("\nInitializing BiasDetector...")
    detector = BiasDetector(
        sensitive_features=['gender', 'race', 'education'],
        target_column='loan_approved',
        prediction_column='loan_approved_pred'
    )
    
    # Generate comprehensive bias report
    print("\nGenerating bias report...")
    report = detector.generate_bias_report(data)
    
    # Print results
    print("\nBias Detection Results:")
    print("-" * 30)
    
    print("\n1. Overall Bias Metrics:")
    metrics = report['bias_metrics']
    print(f"- Demographic Parity Ratio: {metrics.demographic_parity_ratio:.3f}")
    print(f"- Equal Opportunity Ratio: {metrics.equal_opportunity_ratio:.3f}")
    print(f"- Disparate Impact Ratio: {metrics.disparate_impact_ratio:.3f}")
    print(f"- Statistical Parity Difference: {metrics.statistical_parity_difference:.3f}")
    
    print("\n2. Group-specific Metrics:")
    for feature, group_metrics in metrics.group_fairness_metrics.items():
        print(f"\n{feature} groups:")
        for group, values in group_metrics.items():
            print(f"  {group}:")
            for metric, value in values.items():
                print(f"    - {metric}: {value:.3f}")
    
    print("\n3. Intersectional Bias Metrics:")
    for metric, value in metrics.intersectional_metrics.items():
        print(f"- {metric}: {value:.3f}")
    
    print("\n4. Data Representation Analysis:")
    for feature, rep_metrics in report['data_representation'].items():
        print(f"\n{feature}:")
        print(f"- Entropy: {rep_metrics['entropy']:.3f}")
        print(f"- Max Representation Ratio: {rep_metrics['max_representation_ratio']:.3f}")
        print("- Distribution:")
        for group, prop in rep_metrics['distribution'].items():
            print(f"  {group}: {prop:.3f}")
    
    print("\n5. Recommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"{i}. {rec}")

if __name__ == "__main__":
    main() 