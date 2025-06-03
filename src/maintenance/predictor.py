"""
Predictive Maintenance Engine for monitoring system health and predicting failures
"""

import logging
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import mlflow

logger = logging.getLogger(__name__)

class MaintenancePredictor:
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the maintenance predictor."""
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.lstm_model = self._build_lstm_model()
        self.model_path = model_path
        
        if model_path:
            self.load_models(model_path)
            
        # Initialize MLflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("maintenance_prediction")
        
        logger.info("Initialized MaintenancePredictor")

    def _build_lstm_model(self) -> Sequential:
        """Build LSTM model for time series prediction."""
        model = Sequential([
            LSTM(64, input_shape=(24, 8), return_sequences=True),
            LSTM(32),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, 
              sensor_data: pd.DataFrame,
              failure_events: pd.DataFrame,
              epochs: int = 100) -> Dict:
        """
        Train the predictive models using historical sensor data.
        
        Args:
            sensor_data: DataFrame with sensor readings
            failure_events: DataFrame with historical failure events
            epochs: Number of training epochs
            
        Returns:
            Dict with training metrics
        """
        try:
            with mlflow.start_run():
                # Log parameters
                mlflow.log_param("epochs", epochs)
                mlflow.log_param("lstm_layers", 2)
                
                # Prepare data for anomaly detection
                X = self.scaler.fit_transform(sensor_data)
                
                # Train anomaly detector
                self.anomaly_detector.fit(X)
                
                # Prepare data for LSTM
                X_lstm, y_lstm = self._prepare_sequences(
                    sensor_data,
                    failure_events
                )
                
                # Train LSTM model
                history = self.lstm_model.fit(
                    X_lstm, y_lstm,
                    epochs=epochs,
                    validation_split=0.2,
                    verbose=1
                )
                
                # Log metrics
                mlflow.log_metrics({
                    "final_loss": history.history['loss'][-1],
                    "final_accuracy": history.history['accuracy'][-1]
                })
                
                # Save models
                if self.model_path:
                    self.save_models(self.model_path)
                
                return {
                    "loss": history.history['loss'][-1],
                    "accuracy": history.history['accuracy'][-1]
                }
                
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {}

    def predict_failures(self, 
                        current_data: pd.DataFrame,
                        prediction_horizon: int = 7) -> Dict:
        """
        Predict potential failures within the specified time horizon.
        
        Args:
            current_data: Current sensor readings
            prediction_horizon: Number of days to look ahead
            
        Returns:
            Dict with failure predictions and confidence scores
        """
        try:
            # Scale data
            X = self.scaler.transform(current_data)
            
            # Detect current anomalies
            anomaly_scores = self.anomaly_detector.score_samples(X)
            
            # Prepare sequences for LSTM
            X_seq = self._prepare_prediction_sequence(current_data)
            
            # Get failure probabilities
            failure_probs = self.lstm_model.predict(X_seq)
            
            # Calculate remaining useful life
            rul = self._estimate_rul(failure_probs, anomaly_scores)
            
            return {
                "failure_probability": float(failure_probs.mean()),
                "anomaly_score": float(anomaly_scores.mean()),
                "estimated_rul": int(rul),
                "prediction_horizon": prediction_horizon,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {}

    def _prepare_sequences(self, 
                         sensor_data: pd.DataFrame,
                         failure_events: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training."""
        sequence_length = 24  # 24 hours of data
        X, y = [], []
        
        # Create sequences
        for i in range(len(sensor_data) - sequence_length):
            sequence = sensor_data.iloc[i:i + sequence_length].values
            # Check if failure occurred within next 7 days
            end_time = sensor_data.index[i + sequence_length]
            failure = failure_events[
                (failure_events.index > end_time) & 
                (failure_events.index <= end_time + timedelta(days=7))
            ].any()
            
            X.append(sequence)
            y.append(1 if failure else 0)
            
        return np.array(X), np.array(y)

    def _prepare_prediction_sequence(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare sequence for prediction."""
        sequence = data.iloc[-24:].values
        return np.array([sequence])

    def _estimate_rul(self, 
                     failure_probs: np.ndarray,
                     anomaly_scores: np.ndarray) -> int:
        """Estimate Remaining Useful Life in days."""
        # Combine probabilities and anomaly scores
        failure_indicator = (failure_probs.mean() + 
                           (1 - anomaly_scores.mean())) / 2
        
        # Convert to days (simple linear mapping)
        max_rul = 30  # Maximum RUL prediction
        rul = int((1 - failure_indicator) * max_rul)
        
        return max(0, min(rul, max_rul))

    def save_models(self, path: str):
        """Save trained models."""
        try:
            # Save LSTM model
            self.lstm_model.save(f"{path}/lstm_model")
            
            # Save scaler and anomaly detector
            np.save(f"{path}/scaler.npy", self.scaler.get_params())
            np.save(f"{path}/isolation_forest.npy", 
                   self.anomaly_detector.get_params())
            
            logger.info(f"Models saved to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")

    def load_models(self, path: str):
        """Load trained models."""
        try:
            # Load LSTM model
            self.lstm_model = tf.keras.models.load_model(f"{path}/lstm_model")
            
            # Load scaler and anomaly detector
            scaler_params = np.load(f"{path}/scaler.npy", allow_pickle=True)
            self.scaler.set_params(**scaler_params.item())
            
            forest_params = np.load(f"{path}/isolation_forest.npy", 
                                  allow_pickle=True)
            self.anomaly_detector.set_params(**forest_params.item())
            
            logger.info(f"Models loaded from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")

    def analyze_component_health(self, 
                               component_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyze health of individual components.
        
        Args:
            component_data: Dict with component names and their sensor data
            
        Returns:
            Dict with health scores for each component
        """
        health_scores = {}
        
        for component, data in component_data.items():
            try:
                # Scale data
                X = self.scaler.transform(data)
                
                # Get anomaly scores
                anomaly_scores = self.anomaly_detector.score_samples(X)
                
                # Calculate health score (0-100)
                health_score = 100 * (1 + anomaly_scores.mean())
                health_scores[component] = min(100, max(0, health_score))
                
            except Exception as e:
                logger.error(f"Health analysis failed for {component}: {e}")
                health_scores[component] = None
        
        return health_scores 