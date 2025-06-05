"""
Example demonstrating the usage of the Fairness Metrics module.

This example shows how to use the FairnessCalculator class to analyze and measure
fairness metrics in a machine learning model for credit scoring predictions.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.fairness_metrics.fairness_calculator import FairnessCalculator

def create_synthetic_credit_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create synthetic credit scoring dataset with potential fairness issues.
    
    This creates a dataset that intentionally contains some fairness concerns
    to demonstrate the fairness metrics capabilities.
    """
    np.random.seed(42)
    
    # Create base features
    data = pd.DataFrame({
        'age': np.random.normal(40, 10, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'credit_score': np.random.normal(700, 50, n_samples),
        'debt_to_income': np.random.normal(0.3, 0.1, n_samples),
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
        0.1 * (1 - data['debt_to_income'])
    )
    
    # Add more bias in target
    target[data['gender'] == 'F'] *= 0.9
    target[data['race'] == 'Black'] *= 0.9
    
    data['credit_approved'] = (target > target.mean()).astype(int)
    
    return data

def plot_fairness_metrics(metrics: dict):
    """Create visualizations for fairness metrics."""
    # Plot overall metrics
    plt.figure(figsize=(10, 6))
    overall_metrics = metrics['overall_metrics']
    plt.bar(overall_metrics.keys(), overall_metrics.values())
    plt.title('Overall Fairness Metrics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('overall_fairness_metrics.png')
    plt.close()
    
    # Plot group metrics
    for protected_attr, group_metrics in metrics['group_metrics'].items():
        plt.figure(figsize=(12, 6))
        
        # Prepare data for plotting
        groups = list(group_metrics.keys())
        metrics_names = list(group_metrics[groups[0]].keys())
        
        x = np.arange(len(groups))
        width = 0.15
        multiplier = 0
        
        # Plot each metric for each group
        for metric in metrics_names:
            offset = width * multiplier
            values = [group_metrics[group][metric] for group in groups]
            plt.bar(x + offset, values, width, label=metric)
            multiplier += 1
        
        plt.xlabel('Groups')
        plt.title(f'Group Metrics for {protected_attr}')
        plt.xticks(x + width * (len(metrics_names) - 1) / 2, groups)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'group_metrics_{protected_attr}.png')
        plt.close()

def main():
    print("Fairness Metrics Example")
    print("=" * 50)
    
    # Create dataset
    print("\nCreating synthetic credit scoring dataset...")
    data = create_synthetic_credit_data()
    
    # Split features and target
    X = data.drop('credit_approved', axis=1)
    y = data['credit_approved']
    
    # Train a simple model
    print("\nTraining a RandomForest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(
        X.select_dtypes(include=[np.number]),  # Only numeric features
        y
    )
    
    # Generate predictions
    data['credit_approved_pred'] = model.predict(
        X.select_dtypes(include=[np.number])
    )
    
    # Initialize fairness calculator
    print("\nInitializing FairnessCalculator...")
    calculator = FairnessCalculator(
        protected_attributes=['gender', 'race', 'education'],
        target_column='credit_approved',
        prediction_column='credit_approved_pred'
    )
    
    # Generate comprehensive fairness report
    print("\nGenerating fairness report...")
    report = calculator.generate_fairness_report(data)
    
    # Create visualizations
    print("\nCreating fairness visualizations...")
    plot_fairness_metrics(report)
    
    # Print results
    print("\nFairness Analysis Results:")
    print("-" * 30)
    
    print("\n1. Overall Fairness Metrics:")
    for metric, value in report['overall_metrics'].items():
        print(f"- {metric}: {value:.3f}")
    
    print("\n2. Group-specific Metrics:")
    for attr, group_metrics in report['group_metrics'].items():
        print(f"\n{attr} groups:")
        for group, metrics in group_metrics.items():
            print(f"  {group}:")
            for metric, value in metrics.items():
                print(f"    - {metric}: {value:.3f}")
    
    print("\n3. Individual Fairness Metrics:")
    for metric, value in report['individual_metrics'].items():
        print(f"- {metric}: {value:.3f}")
    
    print("\n4. Recommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"{i}. {rec}")
    
    print("\nVisualization files have been saved:")
    print("- overall_fairness_metrics.png")
    print("- group_metrics_gender.png")
    print("- group_metrics_race.png")
    print("- group_metrics_education.png")

if __name__ == "__main__":
    main() 