"""
Tests for the Fairness Metrics Calculator module.
"""

import numpy as np
import pandas as pd
import pytest
from src.fairness_metrics.fairness_calculator import FairnessCalculator, FairnessMetrics

def test_fairness_calculator_initialization():
    """Test FairnessCalculator initialization."""
    protected_attributes = ['race', 'age']
    calculator = FairnessCalculator(protected_attributes)
    assert calculator.protected_attributes == protected_attributes

def test_demographic_parity_ratio():
    """Test demographic parity ratio calculation."""
    calculator = FairnessCalculator(['group'])
    
    # Create test data with clear disparity
    y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    protected_features = pd.DataFrame({
        'group': ['A'] * 5 + ['B'] * 5
    })
    
    ratio = calculator.calculate_demographic_parity_ratio(y_pred, protected_features, 'group')
    assert ratio == 0.0  # Complete disparity (group A: 100% positive, group B: 0% positive)

def test_equal_opportunity_ratio():
    """Test equal opportunity ratio calculation."""
    calculator = FairnessCalculator(['group'])
    
    # Create test data
    y_true = np.array([1, 1, 1, 0, 0, 1, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    protected_features = pd.DataFrame({
        'group': ['A'] * 5 + ['B'] * 5
    })
    
    ratio = calculator.calculate_equal_opportunity_ratio(
        y_true, y_pred, protected_features, 'group'
    )
    assert 0 <= ratio <= 1

def test_equalized_odds_ratio():
    """Test equalized odds ratio calculation."""
    calculator = FairnessCalculator(['group'])
    
    # Create test data
    y_true = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 0, 0, 1, 1, 1])
    protected_features = pd.DataFrame({
        'group': ['A'] * 5 + ['B'] * 5
    })
    
    ratio = calculator.calculate_equalized_odds_ratio(
        y_true, y_pred, protected_features, 'group'
    )
    assert 0 <= ratio <= 1

def test_group_metrics():
    """Test group-specific metrics calculation."""
    calculator = FairnessCalculator(['group'])
    
    # Create test data
    y_true = np.array([1, 1, 1, 0, 0, 1, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 0])
    protected_features = pd.DataFrame({
        'group': ['A'] * 5 + ['B'] * 5
    })
    
    metrics = calculator.calculate_group_metrics(
        y_true, y_pred, protected_features, 'group'
    )
    
    for group_metrics in metrics.values():
        assert 'accuracy' in group_metrics
        assert 'precision' in group_metrics
        assert 'recall' in group_metrics
        assert 'false_positive_rate' in group_metrics
        assert 'selection_rate' in group_metrics
        
        for value in group_metrics.values():
            assert 0 <= value <= 1

def test_evaluate_fairness():
    """Test comprehensive fairness evaluation."""
    calculator = FairnessCalculator(['group'])
    
    # Create test data
    y_true = np.array([1, 1, 1, 0, 0, 1, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 0])
    protected_features = pd.DataFrame({
        'group': ['A'] * 5 + ['B'] * 5
    })
    
    metrics = calculator.evaluate_fairness(y_true, y_pred, protected_features)
    
    assert isinstance(metrics, FairnessMetrics)
    assert hasattr(metrics, 'demographic_parity_ratio')
    assert hasattr(metrics, 'equal_opportunity_ratio')
    assert hasattr(metrics, 'equalized_odds_ratio')
    assert hasattr(metrics, 'group_fairness_metrics')
    
    # Check that metrics are within valid ranges
    assert 0 <= metrics.demographic_parity_ratio <= 1
    assert 0 <= metrics.equal_opportunity_ratio <= 1
    assert 0 <= metrics.equalized_odds_ratio <= 1

def test_edge_cases():
    """Test edge cases and potential error conditions."""
    calculator = FairnessCalculator(['group'])
    
    # Test empty data
    with pytest.raises(Exception):
        calculator.calculate_demographic_parity_ratio(
            np.array([]),
            pd.DataFrame({'group': []})
        )
    
    # Test single group
    y_pred = np.array([1, 0, 1])
    protected_features = pd.DataFrame({
        'group': ['A'] * 3
    })
    
    ratio = calculator.calculate_demographic_parity_ratio(
        y_pred, protected_features, 'group'
    )
    assert ratio == 1.0  # Perfect ratio with single group
    
    # Test all zeros
    y_pred = np.array([0, 0, 0, 0])
    protected_features = pd.DataFrame({
        'group': ['A', 'A', 'B', 'B']
    })
    
    ratio = calculator.calculate_demographic_parity_ratio(
        y_pred, protected_features, 'group'
    )
    assert ratio == 1.0  # Perfect ratio when all predictions are the same 