"""Test configuration for the plastic recycling system."""

import pytest
import os
import sys

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment."""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Create empty model file if it doesn't exist
    model_path = 'models/yolov8_plastic.pt'
    if not os.path.exists(model_path):
        with open(model_path, 'w') as f:
            f.write('')
    
    yield
    
    # Cleanup is handled by pytest 