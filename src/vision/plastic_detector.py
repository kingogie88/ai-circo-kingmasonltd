"""
Computer vision module for plastic type detection.
"""

import cv2
import numpy as np
import torch
from loguru import logger
from ultralytics import YOLO

class PlasticDetector:
    """Class for detecting and classifying plastic types."""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.85):
        """Initialize the plastic detector.
        
        Args:
            model_path: Path to the YOLOv8 model file
            confidence_threshold: Minimum confidence score for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.camera = None
        self.plastic_types = [
            'PET', 'HDPE', 'PVC', 'LDPE', 'PP', 'PS', 'OTHER'
        ]

    async def initialize(self):
        """Initialize the vision system."""
        try:
            # Load YOLO model
            self.model = YOLO(self.model_path)
            logger.info("Loaded plastic detection model")
            
            # Initialize camera
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                raise RuntimeError("Failed to open camera")
            logger.info("Initialized camera")
            
        except Exception as e:
            logger.error(f"Failed to initialize vision system: {e}")
            raise

    async def detect_plastic(self, frame: np.ndarray) -> dict:
        """Detect plastic types in a frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            Dictionary containing detection results
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
            
        try:
            # Run inference
            results = self.model(frame)[0]
            
            detections = []
            for r in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                if score > self.confidence_threshold:
                    detections.append({
                        'type': self.plastic_types[int(class_id)],
                        'confidence': float(score),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    })
            
            return {
                'detections': detections,
                'frame_width': frame.shape[1],
                'frame_height': frame.shape[0]
            }
            
        except Exception as e:
            logger.error(f"Error during plastic detection: {e}")
            raise

    async def get_frame(self) -> np.ndarray:
        """Get a frame from the camera.
        
        Returns:
            Camera frame as numpy array
        """
        if self.camera is None:
            raise RuntimeError("Camera not initialized")
            
        ret, frame = self.camera.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
            
        return frame

    async def shutdown(self):
        """Shutdown the vision system."""
        try:
            if self.camera is not None:
                self.camera.release()
            logger.info("Vision system shut down")
        except Exception as e:
            logger.error(f"Error during vision system shutdown: {e}")
            raise 