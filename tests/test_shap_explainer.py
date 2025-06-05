"""
Tests for the SHAP-based Explainability module.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
import shutil
import tempfile
from src.explainability.shap_explainer import ShapExplainer, ShapExplanation

class TestShapExplainer(unittest.TestCase):
    """Test cases for ShapExplainer class."""
    
    def setUp(self):
        """Set up test data and model."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 5
        
        # Create synthetic dataset
        self.X = pd.DataFrame(
            np.random.normal(0, 1, (n_samples, n_features)),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Create target with known relationships
        self.y = (
            0.3 * self.X['feature_0'] +
            0.5 * self.X['feature_1'] +
            0.2 * self.X['feature_2'] +
            np.random.normal(0, 0.1, n_samples)
        )
        self.y = (self.y > self.y.mean()).astype(int)
        
        # Train a simple model
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X, self.y)
        
        # Create temporary directory for outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Initialize explainer
        self.explainer = ShapExplainer(
            model=self.model,
            feature_names=self.X.columns.tolist(),
            output_path=self.test_dir
        )
        
        # Fit explainer
        self.explainer.fit(self.X, model_type="tree")
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test explainer initialization."""
        self.assertIsNotNone(self.explainer)
        self.assertEqual(self.explainer.feature_names, self.X.columns.tolist())
        self.assertTrue(os.path.exists(self.test_dir))
    
    def test_fit(self):
        """Test explainer fitting."""
        # Test with different model types
        for model_type in ["tree", "kernel"]:
            explainer = ShapExplainer(
                model=self.model,
                feature_names=self.X.columns.tolist(),
                output_path=self.test_dir
            )
            explainer.fit(self.X, model_type=model_type)
            self.assertIsNotNone(explainer.explainer)
        
        # Test with invalid model type
        with self.assertRaises(ValueError):
            self.explainer.fit(self.X, model_type="invalid")
    
    def test_explain_predictions(self):
        """Test prediction explanation generation."""
        explanation = self.explainer.explain_predictions(self.X)
        
        self.assertIsInstance(explanation, ShapExplanation)
        self.assertIsInstance(explanation.shap_values, np.ndarray)
        self.assertEqual(explanation.shap_values.shape, self.X.shape)
        self.assertIsInstance(explanation.feature_importance, pd.DataFrame)
        self.assertEqual(
            len(explanation.feature_importance),
            len(self.X.columns)
        )
        
        # Test with sample limit
        num_samples = 10
        explanation = self.explainer.explain_predictions(
            self.X,
            num_samples=num_samples
        )
        self.assertEqual(explanation.shap_values.shape[0], num_samples)
    
    def test_plot_explanations(self):
        """Test explanation visualization generation."""
        explanation = self.explainer.explain_predictions(self.X)
        self.explainer.plot_explanations(explanation, self.X)
        
        # Check if visualization files were created
        expected_files = [
            "shap_summary.png",
            "feature_importance.png",
            "decision_plot.png"
        ]
        for i in range(min(5, len(self.X))):
            expected_files.append(f"force_plot_{i}.png")
        
        for file in expected_files:
            self.assertTrue(
                os.path.exists(os.path.join(self.test_dir, file))
            )
    
    def test_explanation_report(self):
        """Test explanation report generation."""
        explanation = self.explainer.explain_predictions(self.X)
        report = self.explainer.generate_explanation_report(explanation, self.X)
        
        # Check report structure
        self.assertIn('global_importance', report)
        self.assertIn('feature_interactions', report)
        self.assertIn('sample_explanations', report)
        self.assertIn('summary_statistics', report)
        self.assertIn('recommendations', report)
        
        # Check global importance
        self.assertIn('feature_importance', report['global_importance'])
        self.assertIn('top_features', report['global_importance'])
        
        # Check sample explanations
        self.assertTrue(len(report['sample_explanations']) > 0)
        for sample in report['sample_explanations'].values():
            self.assertEqual(len(sample), len(self.X.columns))
        
        # Check recommendations
        self.assertTrue(len(report['recommendations']) > 0)
        self.assertTrue(
            all(isinstance(rec, str) for rec in report['recommendations'])
        )
    
    def test_feature_importance(self):
        """Test feature importance calculation."""
        explanation = self.explainer.explain_predictions(self.X)
        importance = explanation.feature_importance
        
        self.assertEqual(len(importance), len(self.X.columns))
        self.assertTrue(all(importance['importance'] >= 0))
        self.assertTrue(
            importance['importance'].iloc[0] >= importance['importance'].iloc[-1]
        )
    
    def test_feature_interactions(self):
        """Test feature interaction calculation."""
        explanation = self.explainer.explain_predictions(self.X)
        interactions = explanation.feature_interactions
        
        if interactions is not None:
            self.assertEqual(interactions.shape, (len(self.X.columns),) * 2)
            self.assertTrue(np.allclose(interactions, interactions.T))
    
    def test_save_load(self):
        """Test explainer saving and loading."""
        save_path = os.path.join(self.test_dir, "explainer.joblib")
        
        # Save explainer
        self.explainer.save(save_path)
        self.assertTrue(os.path.exists(save_path))
        
        # Load explainer
        loaded_explainer = ShapExplainer.load(save_path)
        self.assertEqual(
            loaded_explainer.feature_names,
            self.explainer.feature_names
        )
        
        # Test loaded explainer
        explanation = loaded_explainer.explain_predictions(self.X)
        self.assertIsInstance(explanation, ShapExplanation)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with unfitted explainer
        explainer = ShapExplainer(
            model=self.model,
            feature_names=self.X.columns.tolist(),
            output_path=self.test_dir
        )
        with self.assertRaises(ValueError):
            explainer.explain_predictions(self.X)
        
        # Test with empty data
        empty_data = pd.DataFrame(columns=self.X.columns)
        with self.assertRaises(Exception):
            self.explainer.explain_predictions(empty_data)
        
        # Test with missing features
        invalid_data = self.X.drop(columns=['feature_0'])
        with self.assertRaises(Exception):
            self.explainer.explain_predictions(invalid_data)
    
    def test_background_samples(self):
        """Test background sample handling."""
        # Test with different numbers of background samples
        for n_samples in [10, 50, 200]:
            explainer = ShapExplainer(
                model=self.model,
                feature_names=self.X.columns.tolist(),
                output_path=self.test_dir,
                background_samples=n_samples
            )
            explainer.fit(self.X, model_type="tree")
            explanation = explainer.explain_predictions(self.X)
            self.assertIsInstance(explanation, ShapExplanation)

if __name__ == '__main__':
    unittest.main() 