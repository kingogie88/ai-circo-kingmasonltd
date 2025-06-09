"""
Metrics Tracking Module for Responsible AI Framework.

This module provides comprehensive metrics tracking capabilities including:
- Real-time metrics monitoring
- Historical metrics storage
- Alert generation
- Metric visualization
- Performance tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

@dataclass
class MetricAlert:
    """Container for metric alerts."""
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime
    severity: str
    message: str

class MetricsTracker:
    """Main class for tracking and monitoring RAI metrics."""
    
    def __init__(
        self,
        storage_path: str = "metrics",
        alert_thresholds: Optional[Dict[str, float]] = None,
        history_size: int = 1000
    ):
        """
        Initialize MetricsTracker.
        
        Args:
            storage_path: Path to store metrics data
            alert_thresholds: Dictionary of metric thresholds
            history_size: Number of historical records to maintain
        """
        self.storage_path = Path(storage_path)
        self.alert_thresholds = alert_thresholds or {
            'bias': 0.8,
            'fairness': 0.8,
            'privacy': 0.7,
            'performance': 0.9
        }
        self.history_size = history_size
        self.logger = logging.getLogger('RAI.MetricsTracker')
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics_history = pd.DataFrame()
        self._load_metrics_history()
    
    def track_metrics(
        self,
        metrics: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> List[MetricAlert]:
        """
        Track new metrics and generate alerts if needed.
        
        Args:
            metrics: Dictionary of metrics to track
            timestamp: Optional timestamp for metrics
            
        Returns:
            List of generated alerts
        """
        try:
            # Add timestamp
            current_time = timestamp or datetime.now()
            metrics['timestamp'] = current_time
            
            # Add to history
            self.metrics_history = pd.concat([
                self.metrics_history,
                pd.DataFrame([metrics])
            ]).tail(self.history_size)
            
            # Save updated history
            self._save_metrics_history()
            
            # Generate alerts
            alerts = self._check_alerts(metrics)
            
            # Log tracking
            self.logger.info(
                f"Tracked metrics at {current_time}. "
                f"Generated {len(alerts)} alerts."
            )
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error tracking metrics: {str(e)}")
            raise
    
    def get_metric_history(
        self,
        metric_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get historical metrics data.
        
        Args:
            metric_name: Optional specific metric to retrieve
            start_time: Optional start time for filtering
            end_time: Optional end time for filtering
            
        Returns:
            DataFrame containing metric history
        """
        history = self.metrics_history.copy()
        
        # Apply filters
        if metric_name:
            if metric_name not in history.columns:
                raise ValueError(f"Metric {metric_name} not found in history")
            history = history[['timestamp', metric_name]]
        
        if start_time:
            history = history[history['timestamp'] >= start_time]
        
        if end_time:
            history = history[history['timestamp'] <= end_time]
        
        return history
    
    def plot_metric_trends(
        self,
        metric_names: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Generate visualization of metric trends.
        
        Args:
            metric_names: List of metrics to plot
            start_time: Optional start time for filtering
            end_time: Optional end time for filtering
            save_path: Optional path to save visualization
        """
        try:
            # Get metric history
            history = self.get_metric_history(
                start_time=start_time,
                end_time=end_time
            )
            
            if metric_names is None:
                metric_names = [
                    col for col in history.columns
                    if col != 'timestamp' and np.issubdtype(
                        history[col].dtype,
                        np.number
                    )
                ]
            
            # Create visualization
            plt.figure(figsize=(12, 6))
            for metric in metric_names:
                if metric in history.columns:
                    plt.plot(
                        history['timestamp'],
                        history[metric],
                        label=metric
                    )
            
            plt.title('Metric Trends Over Time')
            plt.xlabel('Timestamp')
            plt.ylabel('Metric Value')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting metric trends: {str(e)}")
            raise
    
    def generate_metrics_report(
        self,
        include_alerts: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive metrics report.
        
        Args:
            include_alerts: Whether to include recent alerts
            
        Returns:
            Dictionary containing metrics report
        """
        try:
            # Get latest metrics
            latest_metrics = self.metrics_history.iloc[-1].to_dict()
            
            # Calculate statistics
            stats = {}
            for metric in self.alert_thresholds.keys():
                if metric in self.metrics_history.columns:
                    metric_data = self.metrics_history[metric].dropna()
                    stats[metric] = {
                        'mean': metric_data.mean(),
                        'std': metric_data.std(),
                        'min': metric_data.min(),
                        'max': metric_data.max(),
                        'trend': self._calculate_trend(metric_data)
                    }
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'current_metrics': latest_metrics,
                'statistics': stats,
                'thresholds': self.alert_thresholds
            }
            
            if include_alerts:
                report['recent_alerts'] = self._get_recent_alerts()
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating metrics report: {str(e)}")
            raise
    
    def _check_alerts(
        self,
        metrics: Dict[str, Any]
    ) -> List[MetricAlert]:
        """Generate alerts for metrics that exceed thresholds."""
        alerts = []
        
        for metric_name, threshold in self.alert_thresholds.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                if isinstance(value, (int, float)):
                    if value < threshold:
                        severity = 'high' if value < threshold * 0.8 else 'medium'
                        alerts.append(
                            MetricAlert(
                                metric_name=metric_name,
                                current_value=value,
                                threshold=threshold,
                                timestamp=metrics['timestamp'],
                                severity=severity,
                                message=(
                                    f"{metric_name} value {value:.3f} is below "
                                    f"threshold {threshold:.3f}"
                                )
                            )
                        )
        
        return alerts
    
    def _calculate_trend(self, data: pd.Series) -> str:
        """Calculate trend direction for a metric."""
        if len(data) < 2:
            return 'stable'
        
        # Calculate percentage change
        change = (data.iloc[-1] - data.iloc[-2]) / data.iloc[-2] * 100
        
        if abs(change) < 1:
            return 'stable'
        elif change > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def _get_recent_alerts(self) -> List[Dict[str, Any]]:
        """Get recent alerts from storage."""
        alerts_file = self.storage_path / 'recent_alerts.json'
        
        if alerts_file.exists():
            with open(alerts_file) as f:
                return json.load(f)
        return []
    
    def _save_metrics_history(self) -> None:
        """Save metrics history to storage."""
        history_file = self.storage_path / 'metrics_history.csv'
        self.metrics_history.to_csv(history_file, index=False)
    
    def _load_metrics_history(self) -> None:
        """Load metrics history from storage."""
        history_file = self.storage_path / 'metrics_history.csv'
        
        if history_file.exists():
            self.metrics_history = pd.read_csv(
                history_file,
                parse_dates=['timestamp']
            )
        else:
            self.metrics_history = pd.DataFrame(columns=['timestamp'])
    
    def set_alert_threshold(
        self,
        metric_name: str,
        threshold: float
    ) -> None:
        """
        Set alert threshold for a metric.
        
        Args:
            metric_name: Name of the metric
            threshold: New threshold value
        """
        self.alert_thresholds[metric_name] = threshold
        self.logger.info(
            f"Updated alert threshold for {metric_name} to {threshold}"
        )
    
    def add_metric_annotation(
        self,
        timestamp: datetime,
        message: str,
        metric_name: Optional[str] = None
    ) -> None:
        """
        Add annotation to metric history.
        
        Args:
            timestamp: Timestamp for annotation
            message: Annotation message
            metric_name: Optional specific metric to annotate
        """
        annotations_file = self.storage_path / 'annotations.json'
        
        try:
            if annotations_file.exists():
                with open(annotations_file) as f:
                    annotations = json.load(f)
            else:
                annotations = []
            
            annotations.append({
                'timestamp': timestamp.isoformat(),
                'message': message,
                'metric_name': metric_name
            })
            
            with open(annotations_file, 'w') as f:
                json.dump(annotations, f, indent=2)
            
            self.logger.info(f"Added annotation: {message}")
            
        except Exception as e:
            self.logger.error(f"Error adding annotation: {str(e)}")
            raise 