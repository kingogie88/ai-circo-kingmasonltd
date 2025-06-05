"""
Tests for the Fairness Metrics module.
"""

import unittest
import numpy as np
import pandas as pd
from src.fairness_metrics.fairness_calculator import FairnessCalculator, FairnessMetrics

class TestFairnessCalculator(unittest.TestCase):
    """Test cases for FairnessCalculator class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create synthetic dataset with known fairness issues
        self.data = pd.DataFrame({
            'gender': np.random.choice(['M', 'F'], n_samples),
            'race': np.random.choice(['White', 'Black'], n_samples),
            'age': np.random.normal(40, 10, n_samples),
            'income': np.random.normal(50000, 20000, n_samples),
            'credit_score': np.random.normal(700, 50, n_samples)
        })
        
        # Add target with intentional bias
        target = (
            0.3 * (self.data['age'] - self.data['age'].mean()) / self.data['age'].std() +
            0.4 * (self.data['income'] - self.data['income'].mean()) / self.data['income'].std() +
            0.3 * (self.data['credit_score'] - self.data['credit_score'].mean()) / self.data['credit_score'].std()
        )
        
        # Introduce bias
        target[self.data['gender'] == 'F'] *= 0.7
        target[self.data['race'] == 'Black'] *= 0.8
        
        self.data['target'] = (target > target.mean()).astype(int)
        self.data['prediction'] = self.data['target'].copy()
        
        # Initialize calculator
        self.calculator = FairnessCalculator(
            protected_attributes=['gender', 'race'],
            target_column='target',
            prediction_column='prediction'
        )
    
    def test_demographic_parity(self):
        """Test demographic parity calculation."""
        parity = self.calculator.calculate_demographic_parity(self.data)
        self.assertIsInstance(parity, float)
        self.assertTrue(0 <= parity <= 1)
        
        # Test with perfect parity
        perfect_data = self.data.copy()
        perfect_data['prediction'] = 1
        perfect_parity = self.calculator.calculate_demographic_parity(perfect_data)
        self.assertAlmostEqual(perfect_parity, 1.0)
    
    def test_equal_opportunity(self):
        """Test equal opportunity calculation."""
        opportunity = self.calculator.calculate_equal_opportunity(self.data)
        self.assertIsInstance(opportunity, float)
        self.assertTrue(0 <= opportunity <= 1)
        
        # Test with perfect equality
        perfect_data = self.data.copy()
        perfect_data['prediction'] = perfect_data['target']
        perfect_opportunity = self.calculator.calculate_equal_opportunity(perfect_data)
        self.assertAlmostEqual(perfect_opportunity, 1.0)
    
    def test_predictive_parity(self):
        """Test predictive parity calculation."""
        parity = self.calculator.calculate_predictive_parity(self.data)
        self.assertIsInstance(parity, float)
        self.assertTrue(0 <= parity <= 1)
        
        # Test with perfect predictions
        perfect_data = self.data.copy()
        perfect_data['prediction'] = perfect_data['target']
        perfect_parity = self.calculator.calculate_predictive_parity(perfect_data)
        self.assertAlmostEqual(perfect_parity, 1.0)
    
    def test_treatment_equality(self):
        """Test treatment equality calculation."""
        equality = self.calculator.calculate_treatment_equality(self.data)
        self.assertIsInstance(equality, float)
        self.assertTrue(0 <= equality <= 1)
        
        # Test with perfect equality
        perfect_data = self.data.copy()
        perfect_data['prediction'] = perfect_data['target']
        perfect_equality = self.calculator.calculate_treatment_equality(perfect_data)
        self.assertAlmostEqual(perfect_equality, 1.0)
    
    def test_disparate_impact(self):
        """Test disparate impact calculation."""
        impact = self.calculator.calculate_disparate_impact(self.data)
        self.assertIsInstance(impact, float)
        self.assertTrue(0 <= impact <= 1)
        
        # Test with no impact
        no_impact_data = self.data.copy()
        no_impact_data['prediction'] = 1
        no_impact = self.calculator.calculate_disparate_impact(no_impact_data)
        self.assertAlmostEqual(no_impact, 1.0)
    
    def test_group_metrics(self):
        """Test group fairness metrics calculation."""
        metrics = self.calculator.calculate_group_metrics(self.data)
        
        self.assertIsInstance(metrics, dict)
        self.assertEqual(set(metrics.keys()), {'gender', 'race'})
        
        for feature_metrics in metrics.values():
            for group_metrics in feature_metrics.values():
                self.assertIn('precision', group_metrics)
                self.assertIn('recall', group_metrics)
                self.assertIn('false_positive_rate', group_metrics)
                self.assertIn('false_negative_rate', group_metrics)
                self.assertIn('positive_predictive_value', group_metrics)
                self.assertIn('negative_predictive_value', group_metrics)
    
    def test_individual_metrics(self):
        """Test individual fairness metrics calculation."""
        metrics = self.calculator.calculate_individual_metrics(self.data)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('consistency', metrics)
        self.assertIn('individual_fairness', metrics)
        
        self.assertTrue(0 <= metrics['consistency'] <= 1)
        self.assertTrue(0 <= metrics['individual_fairness'] <= 1)
    
    def test_fairness_report(self):
        """Test comprehensive fairness report generation."""
        report = self.calculator.generate_fairness_report(self.data)
        
        self.assertIsInstance(report, dict)
        self.assertIn('overall_metrics', report)
        self.assertIn('group_metrics', report)
        self.assertIn('individual_metrics', report)
        self.assertIn('recommendations', report)
        
        # Check overall metrics
        overall_metrics = report['overall_metrics']
        self.assertIn('demographic_parity', overall_metrics)
        self.assertIn('equal_opportunity', overall_metrics)
        self.assertIn('predictive_parity', overall_metrics)
        self.assertIn('treatment_equality', overall_metrics)
        self.assertIn('disparate_impact', overall_metrics)
    
    def test_recommendations(self):
        """Test fairness improvement recommendations generation."""
        report = self.calculator.generate_fairness_report(self.data)
        recommendations = report['recommendations']
        
        self.assertIsInstance(recommendations, list)
        self.assertTrue(len(recommendations) > 0)
        self.assertTrue(all(isinstance(rec, str) for rec in recommendations))
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with single protected attribute
        calculator = FairnessCalculator(
            protected_attributes=['gender'],
            target_column='target'
        )
        report = calculator.generate_fairness_report(self.data)
        self.assertIsInstance(report, dict)
        
        # Test with empty data
        empty_data = pd.DataFrame(columns=self.data.columns)
        with self.assertRaises(Exception):
            self.calculator.generate_fairness_report(empty_data)
        
        # Test with missing columns
        invalid_data = self.data.drop('target', axis=1)
        with self.assertRaises(Exception):
            self.calculator.generate_fairness_report(invalid_data)
        
        # Test with all same predictions
        same_pred_data = self.data.copy()
        same_pred_data['prediction'] = 1
        report = self.calculator.generate_fairness_report(same_pred_data)
        self.assertEqual(report['overall_metrics']['demographic_parity'], 1.0)
    
    def test_threshold_sensitivity(self):
        """Test sensitivity to fairness threshold."""
        # Initialize calculator with different threshold
        strict_calculator = FairnessCalculator(
            protected_attributes=['gender', 'race'],
            target_column='target',
            prediction_column='prediction',
            threshold=0.95
        )
        
        # Generate reports with different thresholds
        normal_report = self.calculator.generate_fairness_report(self.data)
        strict_report = strict_calculator.generate_fairness_report(self.data)
        
        # Strict threshold should generate more recommendations
        self.assertTrue(
            len(strict_report['recommendations']) >= 
            len(normal_report['recommendations'])
        )

if __name__ == '__main__':
    unittest.main() 