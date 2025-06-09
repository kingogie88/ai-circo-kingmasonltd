"""
Example demonstrating the usage of the Fairness Metrics Calculator module.

This example creates a synthetic dataset with potential fairness issues and analyzes it
using the FairnessCalculator class.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.fairness_metrics.fairness_calculator import FairnessCalculator

def create_biased_dataset(n_samples: int = 1000, bias_strength: float = 0.3) -> tuple:
    """
    Create a synthetic dataset with intentional fairness issues for demonstration.
    
    Args:
        n_samples: Number of samples to generate
        bias_strength: Strength of the bias (0 to 1)
        
    Returns:
        Tuple of (features, labels, protected_attributes)
    """
    # Generate protected attributes
    race = np.random.choice(['group_A', 'group_B', 'group_C'], size=n_samples)
    age = np.random.randint(18, 70, size=n_samples)
    
    # Generate features with bias
    income = np.random.normal(50000, 20000, n_samples)
    # Introduce bias: higher income for certain groups
    income[race == 'group_A'] *= (1 + bias_strength)
    income[age > 50] *= (1 + bias_strength/2)
    
    credit_score = np.random.normal(700, 100, n_samples)
    # Introduce bias: better credit scores for certain groups
    credit_score[race == 'group_A'] += 50 * bias_strength
    credit_score[age > 50] += 25 * bias_strength
    
    # Create biased labels (loan approval)
    base_probability = 0.5 + bias_strength * (
        (race == 'group_A').astype(float) * 0.3 +
        (age > 50).astype(float) * 0.2
    )
    labels = np.random.binomial(1, base_probability)
    
    # Create feature matrix
    features = pd.DataFrame({
        'income': income,
        'credit_score': credit_score,
    })
    
    # Create protected attributes DataFrame
    protected_attributes = pd.DataFrame({
        'race': race,
        'age': age
    })
    
    return features, labels, protected_attributes

def main():
    # Create synthetic dataset
    print("Creating synthetic dataset with fairness issues...")
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
    
    # Initialize fairness calculator
    print("\nAnalyzing fairness metrics...")
    fairness_calc = FairnessCalculator(protected_attributes=['race', 'age'])
    
    # Calculate fairness metrics
    fairness_metrics = fairness_calc.evaluate_fairness(
        y_true=y_test,
        y_pred=y_pred,
        protected_features=protected_test
    )
    
    # Print results
    print("\nFairness Analysis Results:")
    print(f"Demographic Parity Ratio: {fairness_metrics.demographic_parity_ratio:.3f}")
    print(f"Equal Opportunity Ratio: {fairness_metrics.equal_opportunity_ratio:.3f}")
    print(f"Equalized Odds Ratio: {fairness_metrics.equalized_odds_ratio:.3f}")
    
    print("\nGroup-specific metrics:")
    for attribute, metrics in fairness_metrics.group_fairness_metrics.items():
        print(f"\n{attribute.capitalize()} groups:")
        for group, group_metrics in metrics.items():
            print(f"\n  {group}:")
            for metric_name, value in group_metrics.items():
                print(f"    {metric_name}: {value:.3f}")

if __name__ == "__main__":
    main() 