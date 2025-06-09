"""
Safety Monitoring Module for Responsible AI Implementation.

This module provides tools for monitoring and enforcing safety constraints
in AI systems during both training and inference.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import logging

@dataclass
class SafetyMetrics:
    """Container for safety monitoring metrics."""
    constraint_violations: Dict[str, int]
    violation_rates: Dict[str, float]
    alert_history: List[Dict[str, Any]]
    safety_score: float

class SafetyMonitor:
    """Main class for monitoring and enforcing AI system safety."""
    
    def __init__(
        self,
        constraints: Dict[str, Callable],
        thresholds: Dict[str, float],
        alert_callback: Optional[Callable] = None
    ):
        """
        Initialize SafetyMonitor.
        
        Args:
            constraints: Dictionary mapping constraint names to constraint functions
            thresholds: Dictionary mapping constraint names to violation thresholds
            alert_callback: Optional callback function for handling alerts
        """
        self.constraints = constraints
        self.thresholds = thresholds
        self.alert_callback = alert_callback or self._default_alert_handler
        
        self.violation_counts = {name: 0 for name in constraints}
        self.total_checks = 0
        self.alert_history = []
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def check_constraints(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        predictions: np.ndarray,
        raise_on_violation: bool = False
    ) -> SafetyMetrics:
        """
        Check if the predictions satisfy safety constraints.
        
        Args:
            data: Input data
            predictions: Model predictions
            raise_on_violation: Whether to raise an exception on constraint violation
            
        Returns:
            SafetyMetrics object containing monitoring results
        """
        self.total_checks += 1
        violations = {}
        
        for name, constraint_fn in self.constraints.items():
            try:
                constraint_satisfied = constraint_fn(data, predictions)
                if not constraint_satisfied:
                    self.violation_counts[name] += 1
                    violations[name] = True
                    
                    alert_info = {
                        'timestamp': datetime.now(),
                        'constraint': name,
                        'details': f"Constraint violation detected: {name}"
                    }
                    self.alert_history.append(alert_info)
                    self.alert_callback(alert_info)
                    
                    if raise_on_violation:
                        raise SafetyViolationError(
                            f"Safety constraint violation: {name}"
                        )
            except Exception as e:
                self.logger.error(f"Error checking constraint {name}: {str(e)}")
                violations[name] = True
        
        # Calculate violation rates and safety score
        violation_rates = {
            name: count / self.total_checks
            for name, count in self.violation_counts.items()
        }
        
        safety_score = 1.0 - sum(violation_rates.values()) / len(violation_rates)
        
        return SafetyMetrics(
            constraint_violations=self.violation_counts.copy(),
            violation_rates=violation_rates,
            alert_history=self.alert_history.copy(),
            safety_score=safety_score
        )
    
    def enforce_constraints(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        predictions: np.ndarray,
        fallback_strategy: str = 'conservative'
    ) -> np.ndarray:
        """
        Enforce safety constraints by modifying predictions if necessary.
        
        Args:
            data: Input data
            predictions: Model predictions
            fallback_strategy: Strategy for handling violations ('conservative' or 'previous')
            
        Returns:
            Modified predictions that satisfy safety constraints
        """
        safe_predictions = predictions.copy()
        
        for name, constraint_fn in self.constraints.items():
            if not constraint_fn(data, safe_predictions):
                self.logger.warning(f"Enforcing constraint: {name}")
                
                if fallback_strategy == 'conservative':
                    # Apply conservative predictions
                    safe_predictions = self._apply_conservative_fallback(
                        safe_predictions,
                        self.thresholds[name]
                    )
                elif fallback_strategy == 'previous':
                    # Use last known safe prediction
                    safe_predictions = self._apply_previous_fallback(
                        safe_predictions
                    )
                else:
                    raise ValueError(
                        "fallback_strategy must be 'conservative' or 'previous'"
                    )
        
        return safe_predictions
    
    def add_constraint(
        self,
        name: str,
        constraint_fn: Callable,
        threshold: float
    ):
        """
        Add a new safety constraint.
        
        Args:
            name: Name of the constraint
            constraint_fn: Function implementing the constraint check
            threshold: Violation threshold for the constraint
        """
        if name in self.constraints:
            raise ValueError(f"Constraint {name} already exists")
        
        self.constraints[name] = constraint_fn
        self.thresholds[name] = threshold
        self.violation_counts[name] = 0
        
        self.logger.info(f"Added new safety constraint: {name}")
    
    def remove_constraint(self, name: str):
        """
        Remove a safety constraint.
        
        Args:
            name: Name of the constraint to remove
        """
        if name not in self.constraints:
            raise ValueError(f"Constraint {name} does not exist")
        
        del self.constraints[name]
        del self.thresholds[name]
        del self.violation_counts[name]
        
        self.logger.info(f"Removed safety constraint: {name}")
    
    def get_safety_report(self) -> str:
        """
        Generate a safety monitoring report.
        
        Returns:
            String containing the safety report
        """
        report = [
            "Safety Monitoring Report",
            "======================",
            f"Total Checks: {self.total_checks}",
            f"Overall Safety Score: {self._calculate_safety_score():.2%}\n",
            "Constraint Violations:",
        ]
        
        for name in self.constraints:
            violations = self.violation_counts[name]
            rate = violations / self.total_checks if self.total_checks > 0 else 0
            report.append(
                f"  {name}:"
                f"\n    Violations: {violations}"
                f"\n    Rate: {rate:.2%}"
                f"\n    Threshold: {self.thresholds[name]}"
            )
        
        if self.alert_history:
            report.extend([
                "\nRecent Alerts:",
                *[f"  {alert['timestamp']}: {alert['details']}"
                  for alert in self.alert_history[-5:]]
            ])
        
        return "\n".join(report)
    
    def _calculate_safety_score(self) -> float:
        """Calculate overall safety score."""
        if self.total_checks == 0:
            return 1.0
            
        violation_rates = [
            count / self.total_checks
            for count in self.violation_counts.values()
        ]
        return 1.0 - sum(violation_rates) / len(violation_rates)
    
    def _default_alert_handler(self, alert_info: Dict[str, Any]):
        """Default handler for safety alerts."""
        self.logger.warning(
            f"Safety Alert: {alert_info['constraint']} - {alert_info['details']}"
        )
    
    def _apply_conservative_fallback(
        self,
        predictions: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """Apply conservative predictions as fallback."""
        return np.clip(predictions, 0, threshold)
    
    def _apply_previous_fallback(
        self,
        predictions: np.ndarray
    ) -> np.ndarray:
        """Use previous safe predictions as fallback."""
        if hasattr(self, 'last_safe_predictions'):
            return self.last_safe_predictions
        return predictions

class SafetyViolationError(Exception):
    """Exception raised when a safety constraint is violated."""
    pass 