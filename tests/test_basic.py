"""Basic tests for the plastic recycling system."""

import os
import sys
import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

def test_imports():
    """Test that core modules can be imported."""
    try:
        import numpy as np
        import cv2
        from PIL import Image
        assert np is not None, "NumPy import failed"
        assert cv2 is not None, "OpenCV import failed"
        assert Image is not None, "PIL import failed"
    except ImportError as e:
        pytest.fail(f"Import error: {str(e)}")

def test_basic_numpy():
    """Test basic NumPy functionality."""
    try:
        import numpy as np
        arr = np.array([1, 2, 3])
        assert len(arr) == 3, "Array length mismatch"
        assert arr.sum() == 6, "Array sum incorrect"
    except Exception as e:
        pytest.fail(f"NumPy test failed: {str(e)}")

def test_basic_opencv():
    """Test basic OpenCV functionality."""
    try:
        import cv2
        import numpy as np
        # Create a simple test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        assert img.shape == (100, 100, 3), "Image shape incorrect"
    except Exception as e:
        pytest.fail(f"OpenCV test failed: {str(e)}")

@pytest.mark.skip(reason="Requires model file")
def test_plastic_detector_init():
    """Test plastic detector initialization."""
    try:
        from src.vision.plastic_detector import PlasticDetector
        detector = PlasticDetector(model_path="models/yolov8_plastic.pt")
        assert detector is not None, "Detector initialization failed"
        assert detector.model_path == "models/yolov8_plastic.pt", "Model path mismatch"
        assert detector.confidence_threshold == 0.85, "Confidence threshold mismatch"
    except Exception as e:
        pytest.fail(f"Plastic detector test failed: {str(e)}")

@pytest.mark.skip(reason="Requires GPIO")
def test_robot_controller_init():
    """Test robot controller initialization."""
    try:
        from src.robotics.controller import RobotController
        arm_config = {"base": 0, "elbow": 0, "wrist": 0}
        conveyor_config = {"speed": 0.5}
        controller = RobotController(arm_config, conveyor_config)
        assert controller is not None, "Controller initialization failed"
        assert controller.arm_config == arm_config, "Arm config mismatch"
        assert controller.conveyor_config == conveyor_config, "Conveyor config mismatch"
    except Exception as e:
        pytest.fail(f"Robot controller test failed: {str(e)}")

@pytest.mark.skip(reason="Requires GPIO")
def test_safety_system_init():
    """Test safety system initialization."""
    try:
        from src.safety_monitoring.safety_system import SafetySystem
        safety = SafetySystem()
        assert safety is not None, "Safety system initialization failed"
        assert safety.timeout == 0.1, "Timeout value mismatch"
        assert safety.check_interval == 0.5, "Check interval mismatch"
        assert not safety.monitoring, "Monitoring should be False initially"
    except Exception as e:
        pytest.fail(f"Safety system test failed: {str(e)}")
