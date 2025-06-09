"""
Tests for the Bias Detection module.
"""

import numpy as np
import pandas as pd
import pytest
from src.bias_detection.bias_detector import BiasDetector, BiasMetrics

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    # Create biased dataset for testing
    data = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(0, 1, n_samples),
        'sensitive_attr': np.random.binomial(1, 0.7, n_samples)
    })
    
    # Create biased predictions
    base_prob = 0.7 * (data['feature_1'] > 0) + 0.3 * (data['feature_2'] > 0)
    bias = 0.2 * data['sensitive_attr']  # Intentional bias
    predictions = ((base_prob + bias) > 0.5).astype(int)
    
    return data, predictions

def test_bias_detector_initialization():
    """Test BiasDetector initialization."""
    sensitive_features = ['gender', 'age']
    detector = BiasDetector(sensitive_features)
    assert detector.sensitive_features == sensitive_features

def test_demographic_parity():
    """Test demographic parity calculation."""
    detector = BiasDetector(['group'])
    
    # Create test data with clear bias
    y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    protected_attributes = pd.DataFrame({
        'group': ['A'] * 5 + ['B'] * 5
    })
    
    parity = detector.compute_demographic_parity(y_pred, protected_attributes)
    assert parity == 1.0  # Perfect disparity (group A: 100% positive, group B: 0% positive)

def test_equal_opportunity():
    """Test equal opportunity calculation."""
    detector = BiasDetector(['group'])
    
    # Create test data
    y_true = np.array([1, 1, 1, 0, 0, 1, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 0])
    protected_attributes = pd.DataFrame({
        'group': ['A'] * 5 + ['B'] * 5
    })
    
    opportunity = detector.compute_equal_opportunity(y_true, y_pred, protected_attributes)
    assert 0 <= opportunity <= 1

def test_disparate_impact():
    """Test disparate impact calculation."""
    detector = BiasDetector(['group'])
    
    # Create test data with clear disparate impact
    y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    protected_attributes = pd.DataFrame({
        'group': ['A'] * 5 + ['B'] * 5
    })
    
    impact = detector.compute_disparate_impact(y_pred, protected_attributes)
    assert impact == 0.0  # Complete disparate impact

def test_analyze_bias():
    """Test comprehensive bias analysis."""
    detector = BiasDetector(['group'])
    
    # Create test data
    y_true = np.array([1, 1, 1, 0, 0, 1, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 0])
    protected_attributes = pd.DataFrame({
        'group': ['A'] * 5 + ['B'] * 5
    })
    
    metrics = detector.analyze_bias(y_true, y_pred, protected_attributes)
    
    assert isinstance(metrics, BiasMetrics)
    assert hasattr(metrics, 'demographic_parity')
    assert hasattr(metrics, 'equal_opportunity')
    assert hasattr(metrics, 'disparate_impact')
    assert hasattr(metrics, 'group_fairness')
    
    # Check that metrics are within valid ranges
    assert 0 <= metrics.demographic_parity <= 1
    assert 0 <= metrics.equal_opportunity <= 1
    assert 0 <= metrics.disparate_impact <= 1

def test_edge_cases():
    """Test edge cases and potential error conditions."""
    detector = BiasDetector(['group'])
    
    # Test empty data
    with pytest.raises(Exception):
        detector.compute_demographic_parity(
            np.array([]),
            pd.DataFrame({'group': []})
        )
    
    # Test single group
    y_pred = np.array([1, 0, 1])
    protected_attributes = pd.DataFrame({
        'group': ['A'] * 3
    })
    
    parity = detector.compute_demographic_parity(y_pred, protected_attributes)
    assert parity == 0  # No disparity with single group
    
    # Test all zeros
    y_pred = np.array([0, 0, 0, 0])
    protected_attributes = pd.DataFrame({
        'group': ['A', 'A', 'B', 'B']
    })
    
    impact = detector.compute_disparate_impact(y_pred, protected_attributes)
    assert impact == 1  # No impact when all predictions are the same

def test_demographic_disparity(sample_data):
    """Test demographic disparity calculation."""
    data, predictions = sample_data
    detector = BiasDetector(sensitive_features=['sensitive_attr'])
    
    disparities = detector.calculate_demographic_disparity(
        data=data,
        predictions=predictions,
        target_col='target'
    )
    
    assert isinstance(disparities, dict)
    assert 'sensitive_attr' in disparities
    assert 0 <= disparities['sensitive_attr'] <= 1

def test_evaluate_bias(sample_data):
    """Test comprehensive bias evaluation."""
    data, predictions = sample_data
    detector = BiasDetector(sensitive_features=['sensitive_attr'])
    
    metrics = detector.evaluate_bias(
        data=data,
        predictions=predictions,
        target_col='target'
    )
    
    assert isinstance(metrics, dict)
    assert 'demographic_disparity' in metrics
    assert 'equal_opportunity' in metrics
    assert 'disparate_impact' in metrics

def test_bias_report(sample_data):
    """Test bias report generation."""
    data, predictions = sample_data
    detector = BiasDetector(sensitive_features=['sensitive_attr'])
    
    # First test without evaluation
    report = detector.get_bias_report()
    assert isinstance(report, str)
    assert "No bias metrics have been calculated yet" in report
    
    # Test after evaluation
    detector.evaluate_bias(
        data=data,
        predictions=predictions,
        target_col='target'
    )
    report = detector.get_bias_report()
    assert isinstance(report, str)
    assert "Bias Detection Report" in report
    assert "demographic_disparity" in report.lower()
    assert "equal_opportunity" in report.lower()
    assert "disparate_impact" in report.lower()

def test_mitigation_suggestions(sample_data):
    """Test mitigation suggestions generation."""
    data, predictions = sample_data
    detector = BiasDetector(sensitive_features=['sensitive_attr'])
    
    # First test without evaluation
    suggestions = detector.suggest_mitigations()
    assert isinstance(suggestions, list)
    assert len(suggestions) == 1
    assert "No bias metrics available" in suggestions[0]
    
    # Test after evaluation
    detector.evaluate_bias(
        data=data,
        predictions=predictions,
        target_col='target'
    )
    suggestions = detector.suggest_mitigations()
    assert isinstance(suggestions, list)
    assert len(suggestions) > 0
    assert all(isinstance(s, str) for s in suggestions) 