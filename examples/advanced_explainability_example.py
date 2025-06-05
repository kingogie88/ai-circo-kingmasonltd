"""
Example demonstrating the usage of advanced explainability tools (LIT and PAIR).

This example shows how to use both LIT for text model interpretability and
PAIR for interactive visualizations.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.explainability.lit_explainer import LitExplainer
from src.explainability.pair_visualizer import PairVisualizer, VisualizationConfig

def create_text_classification_data():
    """Create synthetic text classification dataset."""
    texts = [
        "This product is amazing!",
        "I'm very disappointed with the service",
        "The quality is excellent",
        "Would not recommend this",
        "Great customer support"
    ]
    labels = [1, 0, 1, 0, 1]  # 1 for positive, 0 for negative
    return texts, labels

def create_tabular_data():
    """Create synthetic tabular dataset."""
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'age': np.random.normal(40, 10, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'credit_score': np.random.normal(700, 50, n_samples),
        'years_employed': np.random.normal(10, 5, n_samples)
    })
    
    # Create target variable based on features
    target = (
        0.3 * data['age'] +
        0.4 * data['income'] / 10000 +
        0.2 * data['credit_score'] / 100 +
        0.1 * data['years_employed']
    )
    data['loan_approved'] = (target > target.mean()).astype(int)
    
    return data

def main():
    # Text Classification Example with LIT
    print("Text Classification Example with LIT")
    print("-" * 50)
    
    # Load text classification model
    model_name = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create LIT explainer
    lit_explainer = LitExplainer(model, model_name)
    
    # Generate explanation for a sample text
    text = "This product exceeded all my expectations!"
    explanation = lit_explainer.explain_text(text)
    print(f"LIT Explanation for: '{text}'")
    print(explanation.interpretation)
    print()
    
    # Tabular Data Example with PAIR
    print("Tabular Data Example with PAIR")
    print("-" * 50)
    
    # Create and prepare tabular data
    data = create_tabular_data()
    X = data.drop('loan_approved', axis=1)
    y = data['loan_approved']
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Create PAIR visualizer
    visualizer = PairVisualizer()
    
    # Generate feature importance plot
    importance_fig = visualizer.feature_importance_plot(
        feature_names=X.columns.tolist(),
        importance_scores=model.feature_importances_
    )
    importance_fig.write_html("feature_importance.html")
    print("Feature importance plot saved as 'feature_importance.html'")
    
    # Generate feature correlation plot
    correlation_fig = visualizer.feature_correlation_plot(X)
    correlation_fig.write_html("feature_correlations.html")
    print("Feature correlation plot saved as 'feature_correlations.html'")
    
    # Generate prediction distribution plot
    predictions = model.predict_proba(X)[:, 1]
    dist_fig = visualizer.prediction_distribution_plot(
        predictions=predictions,
        actual_values=y
    )
    dist_fig.write_html("prediction_distribution.html")
    print("Prediction distribution plot saved as 'prediction_distribution.html'")
    
    # Generate dimensionality reduction plot
    dim_fig = visualizer.dimensionality_reduction_plot(
        data=X,
        labels=y,
        method='tsne',
        title="t-SNE Visualization of Loan Approval Data"
    )
    dim_fig.write_html("dimensionality_reduction.html")
    print("Dimensionality reduction plot saved as 'dimensionality_reduction.html'")

if __name__ == "__main__":
    main() 