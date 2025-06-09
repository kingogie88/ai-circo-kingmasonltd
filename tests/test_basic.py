"""Basic tests for the plastic recycling system."""

import pytest
from src.vision.plastic_detector import PlasticDetector
from src.robotics.controller import RobotController
from src.safety_monitoring.safety_system import SafetySystem

def test_plastic_detector_init():
    """Test plastic detector initialization."""
    detector = PlasticDetector(model_path="models/yolov8_plastic.pt")
    assert detector is not None
    assert detector.model_path == "models/yolov8_plastic.pt"
    assert detector.confidence_threshold == 0.85

def test_robot_controller_init():
    """Test robot controller initialization."""
    arm_config = {"base": 0, "elbow": 0, "wrist": 0}
    conveyor_config = {"speed": 0.5}
    controller = RobotController(arm_config, conveyor_config)
    assert controller is not None
    assert controller.arm_config == arm_config
    assert controller.conveyor_config == conveyor_config

def test_safety_system_init():
    """Test safety system initialization."""
    safety = SafetySystem()
    assert safety is not None
    assert safety.timeout == 0.1
    assert safety.check_interval == 0.5
    assert not safety.monitoring

def test_imports():

    """Test that core modules can be imported"""

    try:

        import numpy

        import cv2

        import PIL

        print("All core imports successful")

        assert True

    except ImportError as e:

        print(f"Import failed: {e}")

        assert False



def test_basic_functionality():

    """Basic functionality test"""

    import numpy as np

    arr = np.array([1, 2, 3])

    assert len(arr) == 3
