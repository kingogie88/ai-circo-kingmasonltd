"""
AI Vision Module for Plastic Classification using YOLOv11
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class PlasticClassifier:
    PLASTIC_CLASSES = {
        0: "PET",
        1: "HDPE",
        2: "PVC",
        3: "LDPE",
        4: "PP",
        5: "PS",
        6: "OTHER"
    }

    def __init__(self, model_path: str = "models/plastic_classifier.pt"):
        """Initialize the plastic classifier with YOLOv11."""
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        logger.info(f"Initialized PlasticClassifier on device: {self.device}")

    def _load_model(self) -> YOLO:
        """Load the YOLOv11 model."""
        try:
            model = YOLO(self.model_path)
            model.to(self.device)
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the image for model inference."""
        # Resize image to model input size
        image = cv2.resize(image, (640, 640))
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        return image

    def predict(self, image: np.ndarray) -> List[Dict]:
        """
        Perform plastic classification on an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of dictionaries containing detection results
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Perform inference
            results = self.model(processed_image)
            
            # Process results
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    detection = {
                        "class": self.PLASTIC_CLASSES[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                    }
                    detections.append(detection)
            
            return detections
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return []

    def analyze_quality(self, image: np.ndarray, detection: Dict) -> Dict:
        """
        Analyze the quality of detected plastic item.
        
        Args:
            image: Original image
            detection: Detection result from predict()
            
        Returns:
            Dictionary containing quality metrics
        """
        try:
            x1, y1, x2, y2 = map(int, detection["bbox"])
            roi = image[y1:y2, x1:x2]
            
            # Calculate quality metrics
            quality_metrics = {
                "contamination_level": self._assess_contamination(roi),
                "color": self._analyze_color(roi),
                "size": self._calculate_size(roi),
                "degradation": self._assess_degradation(roi)
            }
            
            return quality_metrics
        
        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            return {}

    def _assess_contamination(self, roi: np.ndarray) -> float:
        """Assess contamination level in the region of interest."""
        # Implement contamination detection logic
        # This is a placeholder implementation
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contamination = 1.0 - (np.count_nonzero(thresh) / thresh.size)
        return float(contamination)

    def _analyze_color(self, roi: np.ndarray) -> Dict:
        """Analyze the color distribution in the region of interest."""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculate color histograms
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        
        # Normalize histograms
        h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX)
        s_hist = cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX)
        v_hist = cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX)
        
        return {
            "hue_distribution": h_hist.flatten().tolist(),
            "saturation_distribution": s_hist.flatten().tolist(),
            "value_distribution": v_hist.flatten().tolist()
        }

    def _calculate_size(self, roi: np.ndarray) -> Dict:
        """Calculate size metrics of the detected plastic item."""
        height, width = roi.shape[:2]
        area = height * width
        aspect_ratio = width / height if height > 0 else 0
        
        return {
            "width_pixels": width,
            "height_pixels": height,
            "area_pixels": area,
            "aspect_ratio": aspect_ratio
        }

    def _assess_degradation(self, roi: np.ndarray) -> float:
        """Assess the degradation level of the plastic item."""
        # Implement degradation assessment logic
        # This is a placeholder implementation using edge detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.count_nonzero(edges) / edges.size
        
        # Normalize to 0-1 range where 1 indicates high degradation
        degradation_score = min(edge_density * 5, 1.0)
        return float(degradation_score) 