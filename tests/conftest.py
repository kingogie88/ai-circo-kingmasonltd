"""Test configuration for the plastic recycling system."""

import os
import sys
import pytest
import numpy as np
from pathlib import Path

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
        
        @staticmethod
        def setmode(mode):
            pass
            
        @staticmethod
        def setup(pin, mode, pull_up_down=None):
            pass
            
        @staticmethod
        def output(pin, value):
            pass
            
        @staticmethod
        def cleanup():
            pass
    
    monkeypatch.setattr('RPi.GPIO', MockGPIO)
    return MockGPIO() 