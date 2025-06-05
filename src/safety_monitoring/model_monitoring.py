"""
Model Monitoring module for safety checks and performance monitoring.
"""

from typing import Dict, List, Optional, Union, Tuple, Callable
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.base import BaseEstimator
import logging
from datetime import datetime

class ModelMonitor:
    """A class for monitoring model safety and performance."""
    
    def __init__(
        self,
        model: BaseEstimator,
        feature_names: List[str],
        safety_constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        performance_threshold: float = 0.7,
        drift_threshold: float = 0.1
    ):
        """
        Initialize the ModelMonitor.

        Args:
            model: The model to monitor
            feature_names: List of feature names
            safety_constraints: Dictionary mapping features to (min, max) constraints
            performance_threshold: Minimum acceptable performance score
            drift_threshold: Maximum acceptable drift score
        """
        self.model = model
        self.feature_names = feature_names
        self.safety_constraints = safety_constraints or {}
        self.performance_threshold = performance_threshold
        self.drift_threshold = drift_threshold
        
        # Initialize monitoring metrics
        self.metrics_history: Dict[str, List[float]] = {
            'performance_score': [],
            'drift_score': [],
            'safety_violations': [],
            'timestamp': []
        }
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def check_safety_constraints(
        self,
        data: Union[np.ndarray, pd.DataFrame]
    ) -> Tuple[bool, List[str]]:
        """
        Check if data meets safety constraints.

        Args:
            data: Input data to check

        Returns:
            Tuple of (is_safe, violation_messages)
        """
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=self.feature_names)
        
        violations = []
        is_safe = True
        
        for feature, (min_val, max_val) in self.safety_constraints.items():
            if feature not in data.columns:
                continue
            
            # Check minimum constraint
            min_violations = data[data[feature] < min_val]
            if len(min_violations) > 0:
                is_safe = False
                violations.append(
                    f"Feature '{feature}' has {len(min_violations)} values below "
                    f"minimum threshold {min_val}"
                )
            
            # Check maximum constraint
            max_violations = data[data[feature] > max_val]
            if len(max_violations) > 0:
                is_safe = False
                violations.append(
                    f"Feature '{feature}' has {len(max_violations)} values above "
                    f"maximum threshold {max_val}"
                )
        
        return is_safe, violations

    def detect_drift(
        self,
        reference_data: Union[np.ndarray, pd.DataFrame],
        current_data: Union[np.ndarray, pd.DataFrame],
        method: str = "ks_test"
    ) -> Tuple[float, Dict[str, float]]:
        """
        Detect data drift between reference and current data.

        Args:
            reference_data: Reference data distribution
            current_data: Current data distribution
            method: Drift detection method ('ks_test' or 'wasserstein')

        Returns:
            Tuple of (drift_score, feature_drift_scores)
        """
        from scipy import stats
        from scipy.stats import wasserstein_distance
        
        # Convert to DataFrames if necessary
        if isinstance(reference_data, np.ndarray):
            reference_data = pd.DataFrame(reference_data, columns=self.feature_names)
        if isinstance(current_data, np.ndarray):
            current_data = pd.DataFrame(current_data, columns=self.feature_names)
        
        feature_drift_scores = {}
        
        for feature in self.feature_names:
            if method == "ks_test":
                # Kolmogorov-Smirnov test
                statistic, _ = stats.ks_2samp(
                    reference_data[feature],
                    current_data[feature]
                )
                feature_drift_scores[feature] = statistic
            elif method == "wasserstein":
                # Wasserstein distance
                distance = wasserstein_distance(
                    reference_data[feature],
                    current_data[feature]
                )
                feature_drift_scores[feature] = distance
            else:
                raise ValueError(f"Unsupported drift detection method: {method}")
        
        # Overall drift score is the maximum feature drift
        drift_score = max(feature_drift_scores.values())
        
        return drift_score, feature_drift_scores

    def monitor_performance(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        metric: str = "roc_auc"
    ) -> float:
        """
        Monitor model performance using specified metric.

        Args:
            X: Input features
            y: True labels
            metric: Performance metric to use

        Returns:
            Performance score
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if metric == "roc_auc":
            y_pred = self.model.predict_proba(X)[:, 1]
            return roc_auc_score(y, y_pred)
        else:
            raise ValueError(f"Unsupported performance metric: {metric}")

    def check_prediction_safety(
        self,
        predictions: np.ndarray,
        confidence_threshold: float = 0.9
    ) -> Tuple[bool, List[str]]:
        """
        Check if model predictions are safe based on confidence.

        Args:
            predictions: Model predictions (probabilities)
            confidence_threshold: Minimum confidence threshold

        Returns:
            Tuple of (is_safe, warning_messages)
        """
        warnings = []
        is_safe = True
        
        # Check prediction confidence
        if predictions.ndim == 2:  # For probability predictions
            confidence = np.max(predictions, axis=1)
            low_confidence = confidence < confidence_threshold
            
            if np.any(low_confidence):
                is_safe = False
                warnings.append(
                    f"Found {np.sum(low_confidence)} predictions with confidence "
                    f"below threshold {confidence_threshold}"
                )
        
        return is_safe, warnings

    def update_metrics(
        self,
        performance_score: float,
        drift_score: float,
        num_violations: int
    ) -> None:
        """
        Update monitoring metrics history.

        Args:
            performance_score: Current performance score
            drift_score: Current drift score
            num_violations: Number of safety violations
        """
        self.metrics_history['performance_score'].append(performance_score)
        self.metrics_history['drift_score'].append(drift_score)
        self.metrics_history['safety_violations'].append(num_violations)
        self.metrics_history['timestamp'].append(datetime.now())

    def get_monitoring_report(self) -> str:
        """
        Generate a comprehensive monitoring report.

        Returns:
            String containing the monitoring report
        """
        if not self.metrics_history['timestamp']:
            return "No monitoring data available yet."
        
        report = ["Model Safety Monitoring Report"]
        report.append("============================")
        
        # Latest metrics
        report.append("\nLatest Metrics:")
        report.append(f"- Performance Score: {self.metrics_history['performance_score'][-1]:.4f}")
        report.append(f"- Drift Score: {self.metrics_history['drift_score'][-1]:.4f}")
        report.append(f"- Safety Violations: {self.metrics_history['safety_violations'][-1]}")
        report.append(f"- Timestamp: {self.metrics_history['timestamp'][-1]}")
        
        # Performance trend
        perf_trend = np.mean(np.diff(self.metrics_history['performance_score']))
        trend_direction = "improving" if perf_trend > 0 else "degrading" if perf_trend < 0 else "stable"
        report.append(f"\nPerformance Trend: {trend_direction} ({perf_trend:.4f} per update)")
        
        # Alerts
        report.append("\nAlerts:")
        if self.metrics_history['performance_score'][-1] < self.performance_threshold:
            report.append("⚠️ Performance below threshold!")
        if self.metrics_history['drift_score'][-1] > self.drift_threshold:
            report.append("⚠️ Significant data drift detected!")
        if self.metrics_history['safety_violations'][-1] > 0:
            report.append("⚠️ Safety violations present!")
        
        if len(report) == 4:  # No alerts
            report.append("✅ No alerts - model operating within safe parameters")
        
        return "\n".join(report)

    def suggest_actions(self) -> List[str]:
        """
        Suggest actions based on monitoring results.

        Returns:
            List of suggested actions
        """
        if not self.metrics_history['timestamp']:
            return ["No monitoring data available for suggestions."]
        
        suggestions = []
        
        # Performance-based suggestions
        latest_performance = self.metrics_history['performance_score'][-1]
        if latest_performance < self.performance_threshold:
            suggestions.extend([
                "Model performance below threshold:",
                "- Consider retraining the model",
                "- Review recent data quality",
                "- Analyze feature importance changes"
            ])
        
        # Drift-based suggestions
        latest_drift = self.metrics_history['drift_score'][-1]
        if latest_drift > self.drift_threshold:
            suggestions.extend([
                "Significant data drift detected:",
                "- Investigate feature distribution changes",
                "- Update reference data if changes are expected",
                "- Consider adaptive retraining"
            ])
        
        # Safety violation suggestions
        latest_violations = self.metrics_history['safety_violations'][-1]
        if latest_violations > 0:
            suggestions.extend([
                "Safety violations present:",
                "- Review and adjust safety constraints",
                "- Implement additional validation checks",
                "- Consider feature preprocessing improvements"
            ])
        
        # Trend-based suggestions
        if len(self.metrics_history['performance_score']) > 1:
            perf_trend = np.mean(np.diff(self.metrics_history['performance_score']))
            if perf_trend < 0:
                suggestions.extend([
                    "Performance degradation trend detected:",
                    "- Analyze root causes of degradation",
                    "- Review feature engineering pipeline",
                    "- Consider model architecture updates"
                ])
        
        if not suggestions:
            suggestions.append(
                "Model operating within normal parameters. Continue regular monitoring."
            )
        
        return suggestions 