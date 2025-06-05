"""
Tests for the Bias Detection module.
"""

import unittest
import numpy as np
import pandas as pd
from src.bias_detection.bias_detector import BiasDetector, BiasMetrics

class TestBiasDetector(unittest.TestCase):
    """Test cases for BiasDetector class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create synthetic dataset with known biases
        self.data = pd.DataFrame({
            'gender': np.random.choice(['M', 'F'], n_samples),
            'race': np.random.choice(['White', 'Black'], n_samples),
            'age': np.random.normal(40, 10, n_samples),
            'income': np.random.normal(50000, 20000, n_samples)
        })
        
        # Add target with intentional bias
        target = (
            0.3 * (self.data['age'] - self.data['age'].mean()) +
            0.7 * (self.data['income'] - self.data['income'].mean())
        )
        
        # Introduce bias
        target[self.data['gender'] == 'F'] *= 0.7
        target[self.data['race'] == 'Black'] *= 0.8
        
        self.data['target'] = (target > target.mean()).astype(int)
        self.data['prediction'] = self.data['target'].copy()
        
        # Initialize detector
        self.detector = BiasDetector(
            sensitive_features=['gender', 'race'],
            target_column='target',
            prediction_column='prediction'
        )
    
    def test_demographic_parity(self):
        """Test demographic parity calculation."""
        dp_ratio = self.detector._calculate_demographic_parity(self.data)
        self.assertIsInstance(dp_ratio, float)
        self.assertTrue(0 <= dp_ratio <= 1)
    
    def test_equal_opportunity(self):
        """Test equal opportunity calculation."""
        eo_ratio = self.detector._calculate_equal_opportunity(self.data)
        self.assertIsInstance(eo_ratio, float)
        self.assertTrue(0 <= eo_ratio <= 1)
    
    def test_disparate_impact(self):
        """Test disparate impact calculation."""
        di_ratio = self.detector._calculate_disparate_impact(self.data)
        self.assertIsInstance(di_ratio, float)
        self.assertTrue(0 <= di_ratio <= 1)
    
    def test_statistical_parity_difference(self):
        """Test statistical parity difference calculation."""
        sp_diff = self.detector._calculate_statistical_parity_difference(self.data)
        self.assertIsInstance(sp_diff, float)
        self.assertTrue(0 <= sp_diff <= 1)
    
    def test_group_fairness(self):
        """Test group fairness metrics calculation."""
        metrics = self.detector._calculate_group_fairness(self.data)
        
        self.assertIsInstance(metrics, dict)
        self.assertEqual(set(metrics.keys()), {'gender', 'race'})
        
        for feature_metrics in metrics.values():
            for group_metrics in feature_metrics.values():
                self.assertIn('precision', group_metrics)
                self.assertIn('recall', group_metrics)
                self.assertIn('false_positive_rate', group_metrics)
                self.assertIn('false_negative_rate', group_metrics)
    
    def test_intersectional_bias(self):
        """Test intersectional bias calculation."""
        metrics = self.detector._calculate_intersectional_bias(self.data)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('max_disparity', metrics)
        self.assertIn('ratio_disparity', metrics)
        self.assertIn('std_deviation', metrics)
    
    def test_data_representation(self):
        """Test data representation analysis."""
        metrics = self.detector.analyze_data_representation(self.data)
        
        self.assertIsInstance(metrics, dict)
        self.assertEqual(set(metrics.keys()), {'gender', 'race'})
        
        for feature_metrics in metrics.values():
            self.assertIn('entropy', feature_metrics)
            self.assertIn('max_representation_ratio', feature_metrics)
            self.assertIn('distribution', feature_metrics)
    
    def test_bias_report(self):
        """Test comprehensive bias report generation."""
        report = self.detector.generate_bias_report(self.data)
        
        self.assertIsInstance(report, dict)
        self.assertIn('bias_metrics', report)
        self.assertIn('data_representation', report)
        self.assertIn('recommendations', report)
        
        self.assertIsInstance(report['bias_metrics'], BiasMetrics)
        self.assertIsInstance(report['recommendations'], list)
    
    def test_recommendations(self):
        """Test bias mitigation recommendations generation."""
        recommendations = self.detector._generate_recommendations(self.data)
        
        self.assertIsInstance(recommendations, list)
        self.assertTrue(len(recommendations) > 0)
        self.assertTrue(all(isinstance(rec, str) for rec in recommendations))
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with single sensitive feature
        detector = BiasDetector(
            sensitive_features=['gender'],
            target_column='target'
        )
        metrics = detector.calculate_bias_metrics(self.data)
        self.assertIsInstance(metrics, BiasMetrics)
        
        # Test with empty data
        empty_data = pd.DataFrame(columns=self.data.columns)
        with self.assertRaises(Exception):
            self.detector.calculate_bias_metrics(empty_data)
        
        # Test with missing columns
        invalid_data = self.data.drop('target', axis=1)
        with self.assertRaises(Exception):
            self.detector.calculate_bias_metrics(invalid_data)

if __name__ == '__main__':
    unittest.main() 