"""
Fairness Metrics Calculator Module for Responsible AI Implementation.

This module provides tools for calculating various fairness metrics to assess
and quantify algorithmic fairness across different demographic groups.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from sklearn.metrics import confusion_matrix

@dataclass
class FairnessMetrics:
    """Container for various fairness metrics."""
    demographic_parity_ratio: float
    equal_opportunity_ratio: float
    equalized_odds_ratio: float
    group_fairness_metrics: Dict[str, Dict[str, float]]

class FairnessCalculator:
    """Main class for calculating various fairness metrics."""
    
    def __init__(self, protected_attributes: List[str]):
        """
        Initialize FairnessCalculator.
        
        Args:
            protected_attributes: List of column names representing protected attributes
        """
        self.protected_attributes = protected_attributes
    
    def calculate_demographic_parity_ratio(
        self,
        y_pred: np.ndarray,
        protected_features: pd.DataFrame,
        attribute: str
    ) -> float:
        """
        Calculate demographic parity ratio between groups.
        
        Args:
            y_pred: Model predictions
            protected_features: DataFrame containing protected attribute values
            attribute: The protected attribute to analyze
            
        Returns:
            Demographic parity ratio (1.0 = perfect parity)
        """
        groups = protected_features[attribute].unique()
        acceptance_rates = []
        
        for group in groups:
            mask = protected_features[attribute] == group
            acceptance_rate = np.mean(y_pred[mask])
            acceptance_rates.append(acceptance_rate)
            
        return min(acceptance_rates) / max(acceptance_rates) if max(acceptance_rates) > 0 else 1.0
    
    def calculate_equal_opportunity_ratio(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_features: pd.DataFrame,
        attribute: str
    ) -> float:
        """
        Calculate equal opportunity ratio between groups.
        
        Args:
            y_true: Ground truth labels
            y_pred: Model predictions
            protected_features: DataFrame containing protected attribute values
            attribute: The protected attribute to analyze
            
        Returns:
            Equal opportunity ratio (1.0 = perfect equality)
        """
        groups = protected_features[attribute].unique()
        true_positive_rates = []
        
        for group in groups:
            mask = protected_features[attribute] == group
            tn, fp, fn, tp = confusion_matrix(y_true[mask], y_pred[mask]).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            true_positive_rates.append(tpr)
            
        return min(true_positive_rates) / max(true_positive_rates) if max(true_positive_rates) > 0 else 1.0
    
    def calculate_equalized_odds_ratio(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_features: pd.DataFrame,
        attribute: str
    ) -> float:
        """
        Calculate equalized odds ratio between groups.
        
        Args:
            y_true: Ground truth labels
            y_pred: Model predictions
            protected_features: DataFrame containing protected attribute values
            attribute: The protected attribute to analyze
            
        Returns:
            Equalized odds ratio (1.0 = perfect equality)
        """
        groups = protected_features[attribute].unique()
        odds_differences = []
        
        for group in groups:
            mask = protected_features[attribute] == group
            tn, fp, fn, tp = confusion_matrix(y_true[mask], y_pred[mask]).ravel()
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            odds_differences.append((tpr, fpr))
        
        # Calculate the maximum difference in both TPR and FPR
        tpr_ratio = min(x[0] for x in odds_differences) / max(x[0] for x in odds_differences) if max(x[0] for x in odds_differences) > 0 else 1.0
        fpr_ratio = min(x[1] for x in odds_differences) / max(x[1] for x in odds_differences) if max(x[1] for x in odds_differences) > 0 else 1.0
        
        return min(tpr_ratio, fpr_ratio)
    
    def calculate_group_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_features: pd.DataFrame,
        attribute: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate detailed metrics for each group.
        
        Args:
            y_true: Ground truth labels
            y_pred: Model predictions
            protected_features: DataFrame containing protected attribute values
            attribute: The protected attribute to analyze
            
        Returns:
            Dictionary containing metrics for each group
        """
        groups = protected_features[attribute].unique()
        group_metrics = {}
        
        for group in groups:
            mask = protected_features[attribute] == group
            tn, fp, fn, tp = confusion_matrix(y_true[mask], y_pred[mask]).ravel()
            
            metrics = {
                'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'selection_rate': np.mean(y_pred[mask])
            }
            group_metrics[str(group)] = metrics
            
        return group_metrics
    
    def evaluate_fairness(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_features: pd.DataFrame
    ) -> FairnessMetrics:
        """
        Perform comprehensive fairness evaluation.
        
        Args:
            y_true: Ground truth labels
            y_pred: Model predictions
            protected_features: DataFrame containing protected attribute values
            
        Returns:
            FairnessMetrics object containing various fairness measurements
        """
        attribute = self.protected_attributes[0]  # Using first protected attribute for primary metrics
        
        demographic_parity = self.calculate_demographic_parity_ratio(
            y_pred, protected_features, attribute
        )
        equal_opportunity = self.calculate_equal_opportunity_ratio(
            y_true, y_pred, protected_features, attribute
        )
        equalized_odds = self.calculate_equalized_odds_ratio(
            y_true, y_pred, protected_features, attribute
        )
        
        # Calculate group-specific metrics for all protected attributes
        group_metrics = {}
        for attr in self.protected_attributes:
            group_metrics[attr] = self.calculate_group_metrics(
                y_true, y_pred, protected_features, attr
            )
        
        return FairnessMetrics(
            demographic_parity_ratio=demographic_parity,
            equal_opportunity_ratio=equal_opportunity,
            equalized_odds_ratio=equalized_odds,
            group_fairness_metrics=group_metrics
        ) 