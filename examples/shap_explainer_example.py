"""
Example demonstrating the usage of the SHAP-based Model Explainability module.

This example trains a simple model on the diabetes dataset and explains its predictions
using SHAP values.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_diabetes
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.explainability.shap_explainer import ShapExplainer

def prepare_diabetes_dataset():
    """
    Prepare the diabetes dataset for demonstration.
    
    Returns:
        Tuple of (features, target, feature_names)
    """
    # Load diabetes dataset
    diabetes = load_diabetes()
    X = diabetes.data
    y = (diabetes.target > diabetes.target.mean()).astype(int)  # Convert to binary classification
    feature_names = diabetes.feature_names
    
    return X, y, feature_names

def main():
    # Prepare dataset
    print("Loading and preparing the diabetes dataset...")
    X, y, feature_names = prepare_diabetes_dataset()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train a random forest classifier
    print("\nTraining a RandomForest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Initialize SHAP explainer
    print("\nInitializing SHAP explainer...")
    explainer = ShapExplainer(model, feature_names)
    
    # Explain a single instance
    print("\nExplaining a single prediction...")
    instance_to_explain = X_test[0]
    explanation = explainer.explain_instance(
        instance_to_explain,
        background_data=X_train
    )
    
    print("\nSingle Instance Explanation:")
    print(explanation.explanation_text)
    
    print("\nFeature Importance:")
    for feature, importance in sorted(
        explanation.feature_importance.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    ):
        print(f"{feature}: {importance:.4f}")
    
    # Explain the entire test dataset
    print("\nGenerating global explanations for the test dataset...")
    global_explanation = explainer.explain_dataset(
        X_test,
        background_data=X_train,
        max_display=10
    )
    
    print("\nGlobal Feature Importance:")
    for feature, importance in global_explanation['global_importance'].items():
        print(f"{feature}: {importance:.4f}")
    
    print("\nBase value for predictions:", global_explanation['base_value'])

if __name__ == "__main__":
    main() 