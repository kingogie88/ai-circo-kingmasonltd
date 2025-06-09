"""
Bias Detection Module for Responsible AI Implementation.

This module provides tools for detecting and measuring various types of bias in ML models
and datasets.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

@dataclass
class BiasMetrics:
    """Container for various bias metrics."""
    demographic_parity: float
    equal_opportunity: float
    disparate_impact: float
    group_fairness: Dict[str, float]

class BiasDetector:
    """Main class for detecting and measuring bias in ML models and datasets."""
    
    def __init__(self, sensitive_features: List[str]):
        """
        Initialize BiasDetector.
        
        Args:
            sensitive_features: List of column names representing protected attributes
        """
        self.sensitive_features = sensitive_features
        
    def compute_demographic_parity(
        self,
        y_pred: np.ndarray,
        protected_attributes: pd.DataFrame
    ) -> float:
        """
        Compute demographic parity - difference in prediction rates across groups.
        
        Args:
            y_pred: Model predictions
            protected_attributes: DataFrame containing protected attribute values
            
        Returns:
            Demographic parity score (0 = perfect parity)
        """
        groups = protected_attributes[self.sensitive_features[0]].unique()
        pred_rates = []
        
        for group in groups:
            mask = protected_attributes[self.sensitive_features[0]] == group
            pred_rate = np.mean(y_pred[mask])
            pred_rates.append(pred_rate)
            
        return np.max(pred_rates) - np.min(pred_rates)
    
    def compute_equal_opportunity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attributes: pd.DataFrame
    ) -> float:
        """
        Compute equal opportunity - difference in true positive rates across groups.
        
        Args:
            y_true: Ground truth labels
            y_pred: Model predictions
            protected_attributes: DataFrame containing protected attribute values
            
        Returns:
            Equal opportunity difference (0 = perfect equality)
        """
        groups = protected_attributes[self.sensitive_features[0]].unique()
        tpr_rates = []
        
        for group in groups:
            mask = protected_attributes[self.sensitive_features[0]] == group
            tn, fp, fn, tp = confusion_matrix(y_true[mask], y_pred[mask]).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tpr_rates.append(tpr)
            
        return np.max(tpr_rates) - np.min(tpr_rates)
    
    def compute_disparate_impact(
        self,
        y_pred: np.ndarray,
        protected_attributes: pd.DataFrame
    ) -> float:
        """
        Compute disparate impact - ratio of prediction rates between groups.
        
        Args:
            y_pred: Model predictions
            protected_attributes: DataFrame containing protected attribute values
            
        Returns:
            Disparate impact ratio (1 = no impact)
        """
        groups = protected_attributes[self.sensitive_features[0]].unique()
        pred_rates = []
        
        for group in groups:
            mask = protected_attributes[self.sensitive_features[0]] == group
            pred_rate = np.mean(y_pred[mask])
            pred_rates.append(pred_rate)
            
        return min(pred_rates) / max(pred_rates) if max(pred_rates) > 0 else 1
    
    def analyze_bias(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attributes: pd.DataFrame
    ) -> BiasMetrics:
        """
        Perform comprehensive bias analysis.
        
        Args:
            y_true: Ground truth labels
            y_pred: Model predictions
            protected_attributes: DataFrame containing protected attribute values
            
        Returns:
            BiasMetrics object containing various bias measurements
        """
        demographic_parity = self.compute_demographic_parity(y_pred, protected_attributes)
        equal_opportunity = self.compute_equal_opportunity(y_true, y_pred, protected_attributes)
        disparate_impact = self.compute_disparate_impact(y_pred, protected_attributes)
        
        # Compute group-specific metrics
        group_fairness = {}
        for feature in self.sensitive_features:
            groups = protected_attributes[feature].unique()
            group_metrics = {}
            for group in groups:
                mask = protected_attributes[feature] == group
                group_metrics[str(group)] = {
                    'prediction_rate': np.mean(y_pred[mask]),
                    'accuracy': np.mean(y_pred[mask] == y_true[mask])
                }
            group_fairness[feature] = group_metrics
            
        return BiasMetrics(
            demographic_parity=demographic_parity,
            equal_opportunity=equal_opportunity,
            disparate_impact=disparate_impact,
            group_fairness=group_fairness
        )

    def calculate_demographic_disparity(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        target_col: str
    ) -> Dict[str, float]:
        """
        Calculate demographic disparity across different groups.

        Args:
            data: DataFrame containing the features
            predictions: Model predictions
            target_col: Name of the target column

        Returns:
            Dictionary containing disparity metrics for each sensitive feature
        """
        disparities = {}
        
        for feature in self.sensitive_features:
            groups = data[feature].unique()
            group_metrics = {}
            
            for group in groups:
                mask = data[feature] == group
                group_pred_rate = predictions[mask].mean()
                group_metrics[str(group)] = group_pred_rate
            
            # Calculate max disparity between groups
            max_disparity = max(group_metrics.values()) - min(group_metrics.values())
            disparities[feature] = max_disparity
            
        return disparities

    def calculate_equal_opportunity(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        target_col: str
    ) -> Dict[str, float]:
        """
        Calculate equal opportunity difference across groups.

        Args:
            data: DataFrame containing the features
            predictions: Model predictions
            target_col: Name of the target column

        Returns:
            Dictionary containing equal opportunity metrics for each sensitive feature
        """
        equal_opp_metrics = {}
        
        for feature in self.sensitive_features:
            groups = data[feature].unique()
            group_metrics = {}
            
            for group in groups:
                mask = (data[feature] == group) & (data[target_col] == 1)
                true_positive_rate = (predictions[mask] == 1).mean()
                group_metrics[str(group)] = true_positive_rate
            
            # Calculate max difference in true positive rates
            max_diff = max(group_metrics.values()) - min(group_metrics.values())
            equal_opp_metrics[feature] = max_diff
            
        return equal_opp_metrics

    def calculate_disparate_impact(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        reference_group: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate disparate impact ratio for each sensitive feature.

        Args:
            data: DataFrame containing the features
            predictions: Model predictions
            reference_group: Optional reference group for comparison

        Returns:
            Dictionary containing disparate impact metrics for each sensitive feature
        """
        impact_metrics = {}
        
        for feature in self.sensitive_features:
            groups = data[feature].unique()
            group_metrics = {}
            
            for group in groups:
                mask = data[feature] == group
                positive_rate = predictions[mask].mean()
                group_metrics[str(group)] = positive_rate
            
            if reference_group is None:
                reference_rate = max(group_metrics.values())
            else:
                reference_rate = group_metrics[reference_group]
            
            # Calculate disparate impact ratios
            impact_ratios = {
                group: rate / reference_rate 
                for group, rate in group_metrics.items()
            }
            
            # Store minimum ratio (worst case)
            impact_metrics[feature] = min(impact_ratios.values())
            
        return impact_metrics

    def evaluate_bias(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        target_col: str,
        reference_group: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Comprehensive bias evaluation combining multiple metrics.

        Args:
            data: DataFrame containing the features
            predictions: Model predictions
            target_col: Name of the target column
            reference_group: Optional reference group for comparison

        Returns:
            Dictionary containing all bias metrics
        """
        metrics = {
            "demographic_disparity": self.calculate_demographic_disparity(
                data, predictions, target_col
            ),
            "equal_opportunity": self.calculate_equal_opportunity(
                data, predictions, target_col
            ),
            "disparate_impact": self.calculate_disparate_impact(
                data, predictions, reference_group
            )
        }
        
        return metrics

    def get_bias_report(self) -> str:
        """
        Generate a human-readable report of bias metrics.

        Returns:
            String containing formatted bias report
        """
        if not self.metrics:
            return "No bias metrics have been calculated yet. Run evaluate_bias() first."
        
        report = []
        report.append("Bias Detection Report")
        report.append("===================")
        
        for metric_name, metric_values in self.metrics.items():
            report.append(f"\n{metric_name.replace('_', ' ').title()}:")
            for feature, value in metric_values.items():
                report.append(f"  {feature}: {value:.4f}")
        
        return "\n".join(report)

    def suggest_mitigations(self) -> List[str]:
        """
        Suggest potential mitigation strategies based on detected bias.

        Returns:
            List of suggested mitigation strategies
        """
        if not self.metrics:
            return ["No bias metrics available. Run evaluate_bias() first."]
        
        suggestions = []
        
        # Check demographic disparity
        for feature, value in self.metrics["demographic_disparity"].items():
            if value > 0.1:  # Threshold for concerning disparity
                suggestions.append(
                    f"High demographic disparity detected in {feature}. "
                    "Consider: \n"
                    "- Reweighting training samples\n"
                    "- Applying preprocessing techniques like resampling\n"
                    "- Using adversarial debiasing during training"
                )
        
        # Check equal opportunity
        for feature, value in self.metrics["equal_opportunity"].items():
            if value > 0.1:  # Threshold for concerning difference
                suggestions.append(
                    f"Equal opportunity violation detected in {feature}. "
                    "Consider: \n"
                    "- Post-processing techniques to equalize opportunity\n"
                    "- Adjusting classification thresholds per group\n"
                    "- Collecting additional training data for underrepresented groups"
                )
        
        # Check disparate impact
        for feature, value in self.metrics["disparate_impact"].items():
            if value < 0.8:  # Standard 80% rule
                suggestions.append(
                    f"Disparate impact detected in {feature}. "
                    "Consider: \n"
                    "- Applying preprocessing techniques for fair representation\n"
                    "- Using constrained optimization during training\n"
                    "- Implementing post-processing corrections"
                )
        
        if not suggestions:
            suggestions.append(
                "No significant bias detected based on current thresholds. "
                "Continue monitoring and consider stricter thresholds if needed."
            )
        
        return suggestions 