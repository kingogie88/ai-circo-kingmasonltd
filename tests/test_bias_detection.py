"""
Tests for the bias detection module.
"""

import numpy as np
import pandas as pd
import pytest
from src.bias_detection.bias_detector import BiasDetector

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
    detector = BiasDetector(sensitive_features=['sensitive_attr'])
    assert detector.sensitive_features == ['sensitive_attr']
    assert isinstance(detector.metrics, dict)
    assert len(detector.metrics) == 0

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

def test_equal_opportunity(sample_data):
    """Test equal opportunity calculation."""
    data, predictions = sample_data
    detector = BiasDetector(sensitive_features=['sensitive_attr'])
    
    metrics = detector.calculate_equal_opportunity(
        data=data,
        predictions=predictions,
        target_col='target'
    )
    
    assert isinstance(metrics, dict)
    assert 'sensitive_attr' in metrics
    assert 0 <= metrics['sensitive_attr'] <= 1

def test_disparate_impact(sample_data):
    """Test disparate impact calculation."""
    data, predictions = sample_data
    detector = BiasDetector(sensitive_features=['sensitive_attr'])
    
    impact = detector.calculate_disparate_impact(
        data=data,
        predictions=predictions
    )
    
    assert isinstance(impact, dict)
    assert 'sensitive_attr' in impact
    assert 0 <= impact['sensitive_attr'] <= 1

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

def test_edge_cases():
    """Test edge cases and error handling."""
    detector = BiasDetector(sensitive_features=['sensitive_attr'])
    
    # Test empty data
    empty_data = pd.DataFrame(columns=['sensitive_attr', 'feature'])
    empty_predictions = np.array([])
    
    with pytest.raises(ValueError):
        detector.evaluate_bias(
            data=empty_data,
            predictions=empty_predictions,
            target_col='target'
        )
    
    # Test mismatched data and predictions
    data = pd.DataFrame({'sensitive_attr': [0, 1], 'feature': [1, 2]})
    predictions = np.array([0])
    
    with pytest.raises(ValueError):
        detector.evaluate_bias(
            data=data,
            predictions=predictions,
            target_col='target'
        )
    
    # Test invalid sensitive feature
    detector = BiasDetector(sensitive_features=['invalid_feature'])
    data = pd.DataFrame({'sensitive_attr': [0, 1], 'feature': [1, 2]})
    predictions = np.array([0, 1])
    
    with pytest.raises(KeyError):
        detector.evaluate_bias(
            data=data,
            predictions=predictions,
            target_col='target'
        ) 