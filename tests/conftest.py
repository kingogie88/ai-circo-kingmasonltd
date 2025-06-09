"""Test configuration for the plastic recycling system."""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Set up test environment."""
    # Create models directory if it doesn't exist
    models_dir = project_root / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # Create empty model file if it doesn't exist
    model_path = models_dir / 'yolov8_plastic.pt'
    if not model_path.exists():
        model_path.touch()
    
    # Set environment variables
    os.environ['TESTING'] = 'true'
    os.environ['MODEL_PATH'] = str(model_path)
    
    yield
    
    # Cleanup is handled by pytest

@pytest.fixture
def sample_image():
    """Create a sample test image."""
    return np.zeros((100, 100, 3), dtype=np.uint8)

@pytest.fixture
def mock_gpio(monkeypatch):
    """Mock GPIO for testing without hardware."""
    class MockGPIO:
        BCM = 'BCM'
        OUT = 'OUT'
        IN = 'IN'
        HIGH = 1
        LOW = 0
        PUD_UP = 'PUD_UP'
        
        def __init__(self):
            self.mode = None
            self.pins = {}
            self.cleanup_called = False
            
        def setmode(self, mode):
            self.mode = mode
            
        def setup(self, pin, direction, pull_up_down=None):
            self.pins[pin] = {'direction': direction, 'value': self.HIGH}
            
        def input(self, pin):
            return self.pins.get(pin, {}).get('value', self.HIGH)
            
        def output(self, pin, value):
            if pin in self.pins:
                self.pins[pin]['value'] = value
                
        def cleanup(self, pins=None):
            self.cleanup_called = True
    
    monkeypatch.setattr('RPi.GPIO', MockGPIO)
    return MockGPIO()

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

@pytest.fixture
def mock_logger(monkeypatch):
    """Mock logger for testing."""
    mock = MagicMock()
    monkeypatch.setattr("loguru.logger", mock)
    return mock 