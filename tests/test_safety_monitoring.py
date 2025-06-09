"""Tests for the safety monitoring system."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from src.safety_monitoring.safety_monitor import SafetyMonitor, SafetyMetrics

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = pd.DataFrame({
        'feature1': np.random.randn(5),
        'feature2': np.random.randn(5) + 3
    })
    predictions = np.random.rand(5)
    return data, predictions

@pytest.fixture
def sample_constraints():
    """Create sample constraints for testing."""
    def feature_based_constraint(data, predictions):
        return data['feature2'].mean() < 3.5
        
    def max_value_constraint(data, predictions):
        return all(p <= 0.8 for p in predictions)
        
    constraints = {
        'feature_based': feature_based_constraint,
        'max_value': max_value_constraint
    }
    thresholds = {
        'feature_based': 0.1,
        'max_value': 0.8
    }
    return constraints, thresholds

def test_safety_monitor_initialization(sample_constraints):
    """Test safety monitor initialization."""
    constraints, thresholds = sample_constraints
    monitor = SafetyMonitor(constraints, thresholds)
    assert isinstance(monitor, SafetyMonitor)
    assert monitor.constraints == constraints
    assert monitor.thresholds == thresholds

def test_check_constraints(sample_data, sample_constraints):
    """Test constraint checking functionality."""
    data, predictions = sample_data
    constraints, thresholds = sample_constraints
    monitor = SafetyMonitor(constraints, thresholds)
    
    metrics = monitor.check_constraints(data, predictions)
    assert isinstance(metrics, SafetyMetrics)
    assert metrics.safety_score >= 0
    assert metrics.safety_score <= 1
    assert isinstance(metrics.constraint_violations, dict)

def test_enforce_constraints(sample_data, sample_constraints):
    """Test constraint enforcement."""
    data, predictions = sample_data
    constraints, thresholds = sample_constraints
    monitor = SafetyMonitor(constraints, thresholds)
    
    safe_predictions = monitor.enforce_constraints(data, predictions)
    assert isinstance(safe_predictions, np.ndarray)
    assert len(safe_predictions) == len(predictions)
    assert all(p <= thresholds['max_value'] for p in safe_predictions)

def test_constraint_management(sample_constraints):
    """Test adding and removing constraints."""
    constraints, thresholds = sample_constraints
    monitor = SafetyMonitor(constraints, thresholds)
    
    def new_constraint(data, predictions):
        return True
        
    monitor.add_constraint('new_constraint', new_constraint, 0.5)
    assert 'new_constraint' in monitor.constraints
    assert 'new_constraint' in monitor.thresholds
    
    monitor.remove_constraint('new_constraint')
    assert 'new_constraint' not in monitor.constraints
    assert 'new_constraint' not in monitor.thresholds

def test_safety_report(sample_data, sample_constraints):
    """Test safety report generation."""
    data, predictions = sample_data
    constraints, thresholds = sample_constraints
    monitor = SafetyMonitor(constraints, thresholds)
    
    metrics = monitor.check_constraints(data, predictions)
    report = monitor.get_safety_report()
    assert isinstance(report, str)
    assert "Safety Monitoring Report" in report
    assert "Constraint Violations:" in report

def test_alert_handling(sample_data, sample_constraints):
    """Test alert handling system."""
    data, predictions = sample_data
    constraints, thresholds = sample_constraints
    monitor = SafetyMonitor(constraints, thresholds)
    
    metrics = monitor.check_constraints(data, predictions)
    assert hasattr(metrics, 'alert_history')
    assert isinstance(metrics.alert_history, list)

def test_edge_cases(sample_constraints):
    """Test edge cases and error handling."""
    constraints, thresholds = sample_constraints
    monitor = SafetyMonitor(constraints, thresholds)
    
    # Test with empty data
    empty_data = pd.DataFrame()
    empty_predictions = np.array([])
    metrics = monitor.check_constraints(empty_data, empty_predictions)
    assert metrics.safety_score >= 0.0  # Empty data should not cause errors
    assert metrics.safety_score <= 1.0  # Safety score should be normalized

def test_safety_score_calculation(sample_data, sample_constraints):
    """Test safety score calculation."""
    data, predictions = sample_data
    constraints, thresholds = sample_constraints
    monitor = SafetyMonitor(constraints, thresholds)
    
    # Test with safe predictions
    safe_predictions = np.ones(len(predictions)) * 0.5
    metrics = monitor.check_constraints(data, safe_predictions.copy())
    assert metrics.safety_score >= 0.0
    assert metrics.safety_score <= 1.0 