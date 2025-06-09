"""
Tests for the Safety Monitoring module.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from src.safety_monitoring.safety_monitor import (
    SafetyMonitor,
    SafetyMetrics,
    SafetyViolationError
)

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.uniform(0, 10, n_samples)
    })
    
    predictions = np.random.uniform(0, 1, n_samples)
    
    return data, predictions

@pytest.fixture
def sample_constraints():
    """Create sample safety constraints for testing."""
    def max_value_constraint(data, predictions):
        return np.all(predictions <= 0.8)
    
    def feature_based_constraint(data, predictions):
        return np.all(predictions <= data['feature2'] * 0.1)
    
    constraints = {
        'max_value': max_value_constraint,
        'feature_based': feature_based_constraint
    }
    
    thresholds = {
        'max_value': 0.8,
        'feature_based': 0.1
    }
    
    return constraints, thresholds

def test_safety_monitor_initialization(sample_constraints):
    """Test SafetyMonitor initialization."""
    constraints, thresholds = sample_constraints
    monitor = SafetyMonitor(constraints, thresholds)
    
    assert monitor.constraints == constraints
    assert monitor.thresholds == thresholds
    assert monitor.total_checks == 0
    assert len(monitor.violation_counts) == len(constraints)
    assert all(count == 0 for count in monitor.violation_counts.values())

def test_check_constraints(sample_data, sample_constraints):
    """Test constraint checking functionality."""
    data, predictions = sample_data
    constraints, thresholds = sample_constraints
    monitor = SafetyMonitor(constraints, thresholds)
    
    # Test with safe predictions
    safe_predictions = np.ones_like(predictions) * 0.5
    metrics = monitor.check_constraints(data, safe_predictions)
    
    assert isinstance(metrics, SafetyMetrics)
    assert metrics.safety_score > 0
    assert all(count == 0 for count in metrics.constraint_violations.values())
    
    # Test with unsafe predictions
    unsafe_predictions = np.ones_like(predictions) * 0.9
    with pytest.raises(SafetyViolationError):
        monitor.check_constraints(data, unsafe_predictions, raise_on_violation=True)

def test_enforce_constraints(sample_data, sample_constraints):
    """Test constraint enforcement."""
    data, predictions = sample_data
    constraints, thresholds = sample_constraints
    monitor = SafetyMonitor(constraints, thresholds)
    
    # Create unsafe predictions
    unsafe_predictions = np.ones_like(predictions) * 0.9
    
    # Test conservative fallback
    safe_predictions = monitor.enforce_constraints(
        data,
        unsafe_predictions,
        fallback_strategy='conservative'
    )
    assert np.all(safe_predictions <= 0.8)
    
    # Test previous fallback
    safe_predictions = monitor.enforce_constraints(
        data,
        unsafe_predictions,
        fallback_strategy='previous'
    )
    assert safe_predictions.shape == predictions.shape
    
    # Test invalid fallback strategy
    with pytest.raises(ValueError):
        monitor.enforce_constraints(data, predictions, fallback_strategy='invalid')

def test_constraint_management(sample_constraints):
    """Test adding and removing constraints."""
    constraints, thresholds = sample_constraints
    monitor = SafetyMonitor(constraints, thresholds)
    
    # Add new constraint
    def new_constraint(data, predictions):
        return np.all(predictions >= 0)
    
    monitor.add_constraint('non_negative', new_constraint, 0.0)
    assert 'non_negative' in monitor.constraints
    assert 'non_negative' in monitor.thresholds
    assert 'non_negative' in monitor.violation_counts
    
    # Try to add duplicate constraint
    with pytest.raises(ValueError):
        monitor.add_constraint('non_negative', new_constraint, 0.0)
    
    # Remove constraint
    monitor.remove_constraint('non_negative')
    assert 'non_negative' not in monitor.constraints
    assert 'non_negative' not in monitor.thresholds
    assert 'non_negative' not in monitor.violation_counts
    
    # Try to remove non-existent constraint
    with pytest.raises(ValueError):
        monitor.remove_constraint('non_existent')

def test_safety_report(sample_data, sample_constraints):
    """Test safety report generation."""
    data, predictions = sample_data
    constraints, thresholds = sample_constraints
    monitor = SafetyMonitor(constraints, thresholds)
    
    # Generate some violations
    unsafe_predictions = np.ones_like(predictions) * 0.9
    try:
        monitor.check_constraints(data, unsafe_predictions)
    except SafetyViolationError:
        pass
    
    report = monitor.get_safety_report()
    assert isinstance(report, str)
    assert "Safety Monitoring Report" in report
    assert "Total Checks:" in report
    assert "Constraint Violations:" in report
    
    for constraint in constraints:
        assert constraint in report

def test_alert_handling(sample_data, sample_constraints):
    """Test alert handling functionality."""
    data, predictions = sample_data
    constraints, thresholds = sample_constraints
    
    alerts = []
    def test_alert_handler(alert_info):
        alerts.append(alert_info)
    
    monitor = SafetyMonitor(
        constraints,
        thresholds,
        alert_callback=test_alert_handler
    )
    
    # Generate violation
    unsafe_predictions = np.ones_like(predictions) * 0.9
    try:
        monitor.check_constraints(data, unsafe_predictions)
    except SafetyViolationError:
        pass
    
    assert len(alerts) > 0
    assert all(isinstance(alert['timestamp'], datetime) for alert in alerts)
    assert all('constraint' in alert for alert in alerts)
    assert all('details' in alert for alert in alerts)

def test_edge_cases(sample_constraints):
    """Test edge cases and error handling."""
    constraints, thresholds = sample_constraints
    monitor = SafetyMonitor(constraints, thresholds)
    
    # Test empty data
    empty_data = pd.DataFrame()
    empty_predictions = np.array([])
    
    with pytest.raises(Exception):
        monitor.check_constraints(empty_data, empty_predictions)
    
    # Test mismatched data and predictions
    mismatched_data = pd.DataFrame({'feature1': [1, 2, 3]})
    mismatched_predictions = np.array([1, 2])
    
    with pytest.raises(Exception):
        monitor.check_constraints(mismatched_data, mismatched_predictions)

def test_safety_score_calculation(sample_data, sample_constraints):
    """Test safety score calculation."""
    data, predictions = sample_data
    constraints, thresholds = sample_constraints
    monitor = SafetyMonitor(constraints, thresholds)
    
    # Test perfect score
    safe_predictions = np.ones_like(predictions) * 0.5
    metrics = monitor.check_constraints(data, safe_predictions)
    assert metrics.safety_score == 1.0
    
    # Test imperfect score
    unsafe_predictions = np.ones_like(predictions) * 0.9
    try:
        metrics = monitor.check_constraints(data, unsafe_predictions)
    except SafetyViolationError:
        pass
    
    assert monitor._calculate_safety_score() < 1.0 