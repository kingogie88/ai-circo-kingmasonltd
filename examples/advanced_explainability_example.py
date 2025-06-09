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
        "This product is amazing and works perfectly",
        "Terrible experience, would not recommend",
        "Average product, nothing special",
        "Best purchase I've ever made",
        "Complete waste of money"
    ]
    labels = [1, 0, 0.5, 1, 0]  # Sentiment scores
    
    return texts, labels

def create_tabular_data(n_samples: int = 1000):
    """Create synthetic tabular dataset."""
    np.random.seed(42)
    
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.uniform(0, 10, n_samples),
        'feature4': np.random.choice(['A', 'B', 'C'], n_samples)
    })
    
    # Create target based on features
    target = (
        0.3 * data['feature1'] +
        0.5 * data['feature2'] +
        0.2 * data['feature3']
    )
    target = (target > target.mean()).astype(int)
    
    return data, target

def demonstrate_lit():
    """Demonstrate LIT explainer usage."""
    print("\nDemonstrating LIT Explainer...")
    
    # Load text classification model
    model_name = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create LIT explainer
    explainer = LitExplainer(
        model=model,
        model_name=model_name,
        vocab=tokenizer.get_vocab()
    )
    
    # Get example text and explanation
    text = "This is a great example of natural language processing"
    explanation = explainer.explain_text(text)
    
    print("\nLIT Explanation:")
    print(explanation.interpretation)
    
    # Start LIT UI server (commented out for example)
    # explainer.serve_ui()

def demonstrate_pair():
    """Demonstrate PAIR visualizer usage."""
    print("\nDemonstrating PAIR Visualizer...")
    
    # Create synthetic data
    data, target = create_tabular_data()
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(data.drop('feature4', axis=1), target)
    
    # Create PAIR visualizer
    visualizer = PairVisualizer(
        model=model,
        feature_names=['feature1', 'feature2', 'feature3'],
        config=VisualizationConfig(
            plot_width=1000,
            plot_height=600
        )
    )
    
    # Create feature importance plot
    importance_scores = dict(zip(
        ['feature1', 'feature2', 'feature3'],
        model.feature_importances_
    ))
    fig_importance = visualizer.create_feature_importance_plot(importance_scores)
    
    # Create prediction scatter plot
    fig_scatter = visualizer.create_prediction_scatter(
        data=data,
        predictions=model.predict_proba(data.drop('feature4', axis=1))[:, 1],
        x_feature='feature1',
        y_feature='feature2',
        color_by='feature3'
    )
    
    # Create embedding plot
    fig_embedding = visualizer.create_embedding_plot(
        data=data.drop('feature4', axis=1),
        method='tsne',
        color_by='feature3'
    )
    
    # Create what-if plot
    fig_whatif = visualizer.create_what_if_plot(
        feature='feature1',
        range_min=data['feature1'].min(),
        range_max=data['feature1'].max(),
        reference_point=data.iloc[0].drop('feature4').values
    )
    
    # Save plots (in practice, these would be displayed in a notebook or web app)
    fig_importance.write_html("feature_importance.html")
    fig_scatter.write_html("prediction_scatter.html")
    fig_embedding.write_html("embedding_plot.html")
    fig_whatif.write_html("whatif_plot.html")
    
    print("\nVisualization files created:")
    print("- feature_importance.html")
    print("- prediction_scatter.html")
    print("- embedding_plot.html")
    print("- whatif_plot.html")

def main():
    print("Advanced Explainability Example")
    print("==============================")
    
    # Demonstrate LIT
    demonstrate_lit()
    
    # Demonstrate PAIR
    demonstrate_pair()

if __name__ == "__main__":
    main() 