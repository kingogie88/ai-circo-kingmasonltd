"""Tests for vision functionality."""

import pytest
import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from src.vision import ImageProcessor

@pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not installed")
def test_opencv_basic():
    """Test OpenCV basic functionality."""
    # Create a simple test image
    img = ImageProcessor.create_test_image(100, 100)
    assert img.shape == (100, 100, 3)
    assert img.dtype == np.uint8
    
    # Test BGR conversion
    bgr_img = ImageProcessor.convert_to_cv2(img)
    assert bgr_img.shape == (100, 100, 3)
    assert bgr_img.dtype == np.uint8

def test_pillow_basic():
    """Test PIL/Pillow basic functionality."""
    # Create a test image
    img_array = ImageProcessor.create_test_image(100, 100)
    img = ImageProcessor.convert_to_pil(img_array)
    assert img.size == (100, 100)
    assert img.mode == 'RGB'
    
    # Test size calculation
    width, height = ImageProcessor.get_image_size(img_array)
    assert width == 100
    assert height == 100