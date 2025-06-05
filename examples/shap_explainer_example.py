"""
Example demonstrating the usage of the SHAP-based Explainability module.

This example shows how to use the ShapExplainer class to analyze and explain
predictions of a machine learning model for customer churn prediction.
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

from src.explainability.shap_explainer import ShapExplainer

def create_synthetic_churn_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create synthetic customer churn dataset.
    
    This creates a realistic dataset for demonstrating model explainability
    in a customer churn prediction context.
    """
    np.random.seed(42)
    
    # Create base features
    data = pd.DataFrame({
        'tenure_months': np.random.normal(30, 15, n_samples),
        'monthly_charges': np.random.normal(70, 20, n_samples),
        'total_charges': np.random.normal(2000, 800, n_samples),
        'age': np.random.normal(45, 15, n_samples),
        'num_services': np.random.randint(1, 6, n_samples),
        'support_calls': np.random.poisson(2, n_samples),
        'payment_delay': np.random.exponential(1, n_samples),
        'usage_decline': np.random.normal(0, 1, n_samples),
        'contract_type': np.random.choice(
            ['Monthly', 'One Year', 'Two Year'],
            n_samples,
            p=[0.5, 0.3, 0.2]
        ),
        'internet_service': np.random.choice(
            ['DSL', 'Fiber Optic', 'No'],
            n_samples,
            p=[0.4, 0.5, 0.1]
        )
    })
    
    # Create relationships
    data['total_charges'] = (
        data['monthly_charges'] * data['tenure_months'] +
        np.random.normal(0, 100, n_samples)
    )
    
    # Add some realistic patterns
    data.loc[data['contract_type'] == 'Monthly', 'payment_delay'] *= 1.5
    data.loc[data['internet_service'] == 'Fiber Optic', 'monthly_charges'] *= 1.3
    
    # Create target variable (churn)
    churn_prob = (
        0.3 * (data['payment_delay'] > 2).astype(int) +
        0.2 * (data['support_calls'] > 3).astype(int) +
        0.2 * (data['usage_decline'] > 0.5).astype(int) +
        0.2 * (data['contract_type'] == 'Monthly').astype(int) +
        0.1 * (data['tenure_months'] < 12).astype(int)
    )
    
    data['churn'] = (churn_prob > 0.5).astype(int)
    
    return data

def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for modeling."""
    # Create copy to avoid modifying original data
    df = data.copy()
    
    # Encode categorical variables
    df = pd.get_dummies(
        df,
        columns=['contract_type', 'internet_service'],
        prefix=['contract', 'internet']
    )
    
    return df

def main():
    print("SHAP Explainability Example")
    print("=" * 50)
    
    # Create dataset
    print("\nCreating synthetic customer churn dataset...")
    data = create_synthetic_churn_data()
    
    # Prepare features
    print("\nPreparing features...")
    features = prepare_features(data.drop('churn', axis=1))
    target = data['churn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42
    )
    
    # Train model
    print("\nTraining a RandomForest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Initialize explainer
    print("\nInitializing ShapExplainer...")
    explainer = ShapExplainer(
        model=model,
        feature_names=features.columns.tolist(),
        output_path="churn_explanations"
    )
    
    # Fit explainer
    print("\nFitting explainer...")
    explainer.fit(X_train, model_type="tree")
    
    # Generate explanations
    print("\nGenerating SHAP explanations...")
    explanation = explainer.explain_predictions(X_test, num_samples=100)
    
    # Create visualizations
    print("\nCreating explanation visualizations...")
    explainer.plot_explanations(explanation, X_test)
    
    # Generate report
    print("\nGenerating explanation report...")
    report = explainer.generate_explanation_report(explanation, X_test)
    
    # Print results
    print("\nModel Explanation Results:")
    print("-" * 30)
    
    print("\n1. Global Feature Importance:")
    importance_df = pd.DataFrame(report['global_importance']['top_features'])
    print(importance_df)
    
    print("\n2. Feature Interactions:")
    if report['feature_interactions'] is not None:
        interactions_df = pd.DataFrame(report['feature_interactions'])
        print("\nTop feature interactions:")
        print(interactions_df.head())
    else:
        print("No significant feature interactions detected.")
    
    print("\n3. Sample Explanations:")
    for i, explanation in report['sample_explanations'].items():
        print(f"\nSample {i + 1}:")
        for feature, impact in sorted(
            explanation.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]:
            print(f"  {feature}: {impact:.3f}")
    
    print("\n4. Summary Statistics:")
    print("Expected value:", report['summary_statistics']['expected_value'])
    print("\nMean feature impacts:")
    for feature, impact in zip(
        features.columns,
        report['summary_statistics']['mean_impact']
    ):
        print(f"  {feature}: {impact:.3f}")
    
    print("\n5. Recommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"{i}. {rec}")
    
    print("\nVisualization files have been saved in the 'churn_explanations' directory:")
    print("- shap_summary.png")
    print("- feature_importance.png")
    print("- feature_interactions.png")
    print("- force_plot_[0-4].png")
    print("- decision_plot.png")
    
    # Save explainer for later use
    print("\nSaving explainer...")
    explainer.save("churn_explainer.joblib")

if __name__ == "__main__":
    main() 