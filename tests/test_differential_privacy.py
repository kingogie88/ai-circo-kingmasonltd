"""
Tests for the Differential Privacy module.
"""

import numpy as np
import pandas as pd
import pytest
from src.privacy_protection.differential_privacy import (
    DifferentialPrivacy,
    PrivacyParams,
    PrivacyMetrics
)

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = pd.DataFrame({
        'sensitive_col1': np.random.normal(0, 1, n_samples),
        'sensitive_col2': np.random.normal(10, 2, n_samples),
        'non_sensitive': np.random.randint(0, 100, n_samples)
    })
    
    return data

def test_initialization():
    """Test DifferentialPrivacy initialization."""
    dp = DifferentialPrivacy(epsilon=0.1, delta=1e-6, sensitivity=2.0)
    
    assert isinstance(dp.params, PrivacyParams)
    assert dp.params.epsilon == 0.1
    assert dp.params.delta == 1e-6
    assert dp.params.sensitivity == 2.0
    assert dp.epsilon_used == 0.0

def test_add_gaussian_noise():
    """Test Gaussian noise addition."""
    dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    noisy_data, metrics = dp.add_gaussian_noise(data)
    
    assert isinstance(noisy_data, np.ndarray)
    assert isinstance(metrics, PrivacyMetrics)
    assert noisy_data.shape == data.shape
    assert metrics.epsilon_used > 0
    assert metrics.noise_scale > 0
    assert "(1.0, 1e-05)-differential privacy" in metrics.privacy_guarantee

def test_privatize_dataset(sample_data):
    """Test dataset privatization."""
    dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
    sensitive_columns = ['sensitive_col1', 'sensitive_col2']
    
    private_data, metrics = dp.privatize_dataset(
        sample_data,
        sensitive_columns=sensitive_columns
    )
    
    assert isinstance(private_data, pd.DataFrame)
    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == set(sensitive_columns)
    assert all(isinstance(m, PrivacyMetrics) for m in metrics.values())
    
    # Check that non-sensitive column remains unchanged
    pd.testing.assert_series_equal(
        sample_data['non_sensitive'],
        private_data['non_sensitive']
    )
    
    # Check that sensitive columns are modified
    for col in sensitive_columns:
        assert not np.array_equal(
            sample_data[col].values,
            private_data[col].values
        )

def test_privatize_aggregation():
    """Test private aggregation operations."""
    dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Test mean
    private_mean, mean_metrics = dp.privatize_aggregation(data, operation='mean')
    assert isinstance(private_mean, float)
    assert isinstance(mean_metrics, PrivacyMetrics)
    
    # Test sum
    private_sum, sum_metrics = dp.privatize_aggregation(data, operation='sum')
    assert isinstance(private_sum, float)
    assert isinstance(sum_metrics, PrivacyMetrics)
    
    # Test invalid operation
    with pytest.raises(ValueError):
        dp.privatize_aggregation(data, operation='invalid')

def test_privacy_budget_tracking():
    """Test privacy budget tracking."""
    dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Perform multiple operations
    dp.add_gaussian_noise(data, epsilon=0.3)
    dp.privatize_aggregation(data, epsilon=0.4)
    
    assert dp.epsilon_used == 0.7
    assert dp.params.epsilon - dp.epsilon_used == 0.3

def test_edge_cases(sample_data):
    """Test edge cases and error handling."""
    dp = DifferentialPrivacy()
    
    # Test empty data
    empty_data = np.array([])
    with pytest.raises(Exception):
        dp.add_gaussian_noise(empty_data)
    
    # Test non-numeric column
    sample_data['non_numeric'] = ['a', 'b', 'c'] * (len(sample_data) // 3 + 1)
    with pytest.raises(ValueError):
        dp.privatize_dataset(sample_data, sensitive_columns=['non_numeric'])
    
    # Test epsilon = 0
    with pytest.raises(Exception):
        DifferentialPrivacy(epsilon=0.0)

def test_privacy_report():
    """Test privacy report generation."""
    dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
    report = dp.get_privacy_report()
    
    assert isinstance(report, str)
    assert "Differential Privacy Report" in report
    assert "Total Privacy Budget" in report
    assert "Privacy Budget Used" in report
    assert "Remaining Budget" in report
    assert "Privacy Guarantee" in report

def test_noise_scale_properties():
    """Test properties of the added noise."""
    dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
    data = np.random.normal(0, 1, 1000)
    
    noisy_data, metrics = dp.add_gaussian_noise(data)
    noise = noisy_data - data
    
    # Check that noise is roughly zero-centered
    assert abs(np.mean(noise)) < 0.1
    
    # Check that noise scale matches the reported scale
    assert abs(np.std(noise) - metrics.noise_scale) < 0.1 