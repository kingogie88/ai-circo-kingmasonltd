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

class SafetyMetrics:
    """Class to hold safety monitoring metrics."""
    
    def __init__(self):
        """Initialize safety metrics."""
        self.safety_score: float = 1.0
        self.constraint_violations: Dict[str, bool] = {}
        self.alert_history: List[Dict] = []
        self.timestamp = datetime.now()

    def add_violation(self, constraint_name: str, violated: bool):
        """Add a constraint violation."""
        self.constraint_violations[constraint_name] = violated
        if violated:
            self.alert_history.append({
                'constraint': constraint_name,
                'timestamp': datetime.now()
            })

    def calculate_safety_score(self):
        """Calculate overall safety score."""
        if not self.constraint_violations:
            return 1.0
        violations = sum(1 for v in self.constraint_violations.values() if v)
        total = len(self.constraint_violations)
        self.safety_score = 1.0 - (violations / total)
        return self.safety_score

class SafetyMonitor:
    """Class for monitoring system safety."""
    
    def __init__(self, constraints: Dict[str, Callable], thresholds: Dict[str, float]):
        """Initialize safety monitor.
        
        Args:
            constraints: Dictionary of constraint functions
            thresholds: Dictionary of threshold values for constraints
        """
        self.constraints = constraints
        self.thresholds = thresholds
        self.metrics = SafetyMetrics()
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def check_constraints(self, data: pd.DataFrame, predictions: np.ndarray) -> SafetyMetrics:
        """Check all safety constraints.
        
        Args:
            data: Input data
            predictions: Model predictions
            
        Returns:
            SafetyMetrics object with results
        """
        metrics = SafetyMetrics()
        
        for name, constraint in self.constraints.items():
            try:
                result = constraint(data, predictions)
                metrics.add_violation(name, not result)
            except Exception as e:
                metrics.add_violation(name, True)
                
        metrics.calculate_safety_score()
        return metrics

    def enforce_constraints(self, data: pd.DataFrame, predictions: np.ndarray) -> np.ndarray:
        """Enforce safety constraints on predictions.
        
        Args:
            data: Input data
            predictions: Model predictions
            
        Returns:
            Modified predictions that satisfy constraints
        """
        safe_predictions = predictions.copy()
        
        # Apply max value constraint if it exists
        if 'max_value' in self.thresholds:
            max_allowed = self.thresholds['max_value']
            safe_predictions = np.minimum(safe_predictions, max_allowed)
            
        return safe_predictions

    def add_constraint(self, name: str, constraint: Callable, threshold: float):
        """Add a new safety constraint.
        
        Args:
            name: Constraint name
            constraint: Constraint function
            threshold: Threshold value
        """
        self.constraints[name] = constraint
        self.thresholds[name] = threshold
        self.logger.info(f"Added new safety constraint: {name}")

    def remove_constraint(self, name: str):
        """Remove a safety constraint.
        
        Args:
            name: Name of constraint to remove
        """
        self.constraints.pop(name, None)
        self.thresholds.pop(name, None)
        self.logger.info(f"Removed safety constraint: {name}")

    def get_safety_report(self) -> str:
        """Generate a safety monitoring report.
        
        Returns:
            Formatted safety report string
        """
        metrics = self.metrics
        report = ["Safety Monitoring Report"]
        report.append("-" * 30)
        report.append(f"Generated at: {metrics.timestamp}")
        report.append(f"Safety Score: {metrics.safety_score:.2f}")
        report.append("\nConstraint Violations:")
        
        for name, violated in metrics.constraint_violations.items():
            status = "VIOLATED" if violated else "OK"
            report.append(f"- {name}: {status}")
            
        if metrics.alert_history:
            report.append("\nRecent Alerts:")
            for alert in metrics.alert_history[-5:]:
                report.append(f"- {alert['constraint']} at {alert['timestamp']}")
                
        return "\n".join(report)

class SafetyViolationError(Exception):
    """Exception raised when a safety constraint is violated."""
    pass 