"""Basic image processing functionality."""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple

class ImageProcessor:
    """Basic image processing class."""
    
    @staticmethod
    def create_test_image(width: int = 100, height: int = 100) -> np.ndarray:
        """Create a test image.
        
        Args:
            width: Image width
            height: Image height
            
        Returns:
            Numpy array containing the test image
        """
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    @staticmethod
    def convert_to_pil(image: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image.
        
        Args:
            image: Input numpy array
            
        Returns:
            PIL Image object
        """
        return Image.fromarray(image)
    
    @staticmethod
    def convert_to_cv2(image: np.ndarray) -> np.ndarray:
        """Convert RGB image to BGR for OpenCV.
        
        Args:
            image: Input numpy array in RGB format
            
        Returns:
            Numpy array in BGR format
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    @staticmethod
    def get_image_size(image: np.ndarray) -> Tuple[int, int]:
        """Get image dimensions.
        
        Args:
            image: Input numpy array
            
        Returns:
            Tuple of (width, height)
        """
        height, width = image.shape[:2]
        return width, height 