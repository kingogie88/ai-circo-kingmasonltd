"""Tests for vision functionality."""

import cv2
import numpy as np
from PIL import Image
import pytest
from src.vision.image_processor import ImageProcessor

def test_opencv_basic():
    """Test OpenCV basic functionality."""
    # Create a simple test image
    img = ImageProcessor.create_test_image(100, 100)
    assert img.shape == (100, 100, 3)
    assert img.dtype == np.uint8

def test_pillow_basic():
    """Test PIL/Pillow basic functionality."""
    # Create a test image
    img_array = ImageProcessor.create_test_image(100, 100)
    img = ImageProcessor.convert_to_pil(img_array)
    assert img.size == (100, 100)
    assert img.mode == 'RGB'