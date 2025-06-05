"""
Tests for the explainability components (LIT and PAIR).
"""

import unittest
import numpy as np
import pandas as pd
from src.explainability.lit_explainer import LitExplainer, LitExplanation
from src.explainability.pair_visualizer import PairVisualizer, VisualizationConfig

class MockModel:
    """Mock model for testing."""
    def predict(self, texts):
        return [0.8 for _ in texts]

class TestLitExplainer(unittest.TestCase):
    """Test cases for LIT explainer."""
    
    def setUp(self):
        self.model = MockModel()
        self.explainer = LitExplainer(self.model, "mock-model")
    
    def test_explain_text(self):
        """Test text explanation generation."""
        text = "This is a test text"
        explanation = self.explainer.explain_text(text)
        
        self.assertIsInstance(explanation, LitExplanation)
        self.assertEqual(len(explanation.tokens), len(text.split()))
        self.assertIsInstance(explanation.interpretation, str)

class TestPairVisualizer(unittest.TestCase):
    """Test cases for PAIR visualizer."""
    
    def setUp(self):
        self.visualizer = PairVisualizer()
        
        # Create sample data
        np.random.seed(42)
        self.feature_names = ['feature1', 'feature2']
        self.importance_scores = [0.7, 0.3]
        self.data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100)
        })
    
    def test_feature_importance_plot(self):
        """Test feature importance plot generation."""
        fig = self.visualizer.feature_importance_plot(
            self.feature_names,
            self.importance_scores
        )
        self.assertIsNotNone(fig)
    
    def test_feature_correlation_plot(self):
        """Test feature correlation plot generation."""
        fig = self.visualizer.feature_correlation_plot(self.data)
        self.assertIsNotNone(fig)
    
    def test_prediction_distribution_plot(self):
        """Test prediction distribution plot generation."""
        predictions = np.random.normal(0, 1, 100)
        actuals = np.random.normal(0, 1, 100)
        
        fig = self.visualizer.prediction_distribution_plot(
            predictions,
            actuals
        )
        self.assertIsNotNone(fig)
    
    def test_dimensionality_reduction_plot(self):
        """Test dimensionality reduction plot generation."""
        data = np.random.normal(0, 1, (100, 5))
        labels = np.random.choice([0, 1], 100)
        
        fig = self.visualizer.dimensionality_reduction_plot(
            data,
            labels
        )
        self.assertIsNotNone(fig)

if __name__ == '__main__':
    unittest.main() 