"""
Fairness Calculator module for computing comprehensive fairness metrics.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

class FairnessCalculator:
    """A class for calculating various fairness metrics for ML models."""
    
    def __init__(self, sensitive_features: List[str]):
        """
        Initialize the FairnessCalculator.

        Args:
            sensitive_features: List of column names representing protected attributes
        """
        self.sensitive_features = sensitive_features
        self.metrics: Dict[str, Dict[str, float]] = {}

    def calculate_group_metrics(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        target_col: str
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Calculate basic metrics for each group.

        Args:
            data: DataFrame containing the features
            predictions: Model predictions
            target_col: Name of the target column

        Returns:
            Dictionary containing metrics for each group
        """
        group_metrics = {}
        
        for feature in self.sensitive_features:
            feature_metrics = {}
            groups = data[feature].unique()
            
            for group in groups:
                mask = data[feature] == group
                group_preds = predictions[mask]
                group_true = data.loc[mask, target_col].values
                
                tn, fp, fn, tp = confusion_matrix(
                    group_true, group_preds, labels=[0, 1]
                ).ravel()
                
                # Calculate basic rates
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                
                feature_metrics[str(group)] = {
                    "true_positive_rate": tpr,
                    "true_negative_rate": tnr,
                    "false_positive_rate": fpr,
                    "false_negative_rate": fnr,
                    "positive_predictive_value": ppv
                }
            
            group_metrics[feature] = feature_metrics
        
        return group_metrics

    def calculate_demographic_parity(
        self,
        predictions: np.ndarray,
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate demographic parity difference for each sensitive feature.

        Args:
            predictions: Model predictions
            data: DataFrame containing the features

        Returns:
            Dictionary containing demographic parity metrics
        """
        parity_metrics = {}
        
        for feature in self.sensitive_features:
            groups = data[feature].unique()
            group_rates = {}
            
            for group in groups:
                mask = data[feature] == group
                positive_rate = predictions[mask].mean()
                group_rates[str(group)] = positive_rate
            
            # Calculate max difference in positive rates
            max_diff = max(group_rates.values()) - min(group_rates.values())
            parity_metrics[feature] = max_diff
        
        return parity_metrics

    def calculate_equalized_odds(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        target_col: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate equalized odds metrics.

        Args:
            data: DataFrame containing the features
            predictions: Model predictions
            target_col: Name of the target column

        Returns:
            Dictionary containing equalized odds metrics
        """
        odds_metrics = {}
        
        group_metrics = self.calculate_group_metrics(data, predictions, target_col)
        
        for feature, metrics in group_metrics.items():
            tpr_values = [m["true_positive_rate"] for m in metrics.values()]
            fpr_values = [m["false_positive_rate"] for m in metrics.values()]
            
            odds_metrics[feature] = {
                "tpr_difference": max(tpr_values) - min(tpr_values),
                "fpr_difference": max(fpr_values) - min(fpr_values)
            }
        
        return odds_metrics

    def calculate_predictive_parity(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        target_col: str
    ) -> Dict[str, float]:
        """
        Calculate predictive parity metrics.

        Args:
            data: DataFrame containing the features
            predictions: Model predictions
            target_col: Name of the target column

        Returns:
            Dictionary containing predictive parity metrics
        """
        parity_metrics = {}
        
        group_metrics = self.calculate_group_metrics(data, predictions, target_col)
        
        for feature, metrics in group_metrics.items():
            ppv_values = [m["positive_predictive_value"] for m in metrics.values()]
            max_diff = max(ppv_values) - min(ppv_values)
            parity_metrics[feature] = max_diff
        
        return parity_metrics

    def calculate_individual_fairness(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        distance_threshold: float = 0.1
    ) -> float:
        """
        Calculate individual fairness metric based on similar individuals.

        Args:
            data: DataFrame containing the features
            predictions: Model predictions
            distance_threshold: Threshold for considering individuals similar

        Returns:
            Individual fairness score
        """
        from sklearn.metrics.pairwise import euclidean_distances
        
        # Normalize numerical features
        numerical_features = data.select_dtypes(include=[np.number]).columns
        normalized_data = (data[numerical_features] - data[numerical_features].mean()) / \
                         data[numerical_features].std()
        
        # Calculate pairwise distances
        distances = euclidean_distances(normalized_data)
        
        # Find similar individuals
        similar_pairs = np.where(distances < distance_threshold)
        
        if len(similar_pairs[0]) == 0:
            return 1.0  # Perfect fairness if no similar pairs found
        
        # Calculate prediction differences for similar individuals
        prediction_differences = np.abs(
            predictions[similar_pairs[0]] - predictions[similar_pairs[1]]
        )
        
        # Return average consistency of predictions
        return 1 - prediction_differences.mean()

    def evaluate_fairness(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        target_col: str
    ) -> Dict[str, Union[Dict[str, float], float]]:
        """
        Comprehensive fairness evaluation combining multiple metrics.

        Args:
            data: DataFrame containing the features
            predictions: Model predictions
            target_col: Name of the target column

        Returns:
            Dictionary containing all fairness metrics
        """
        metrics = {
            "demographic_parity": self.calculate_demographic_parity(
                predictions, data
            ),
            "equalized_odds": self.calculate_equalized_odds(
                data, predictions, target_col
            ),
            "predictive_parity": self.calculate_predictive_parity(
                data, predictions, target_col
            ),
            "individual_fairness": self.calculate_individual_fairness(
                data, predictions
            )
        }
        
        self.metrics = metrics
        return metrics

    def get_fairness_report(self) -> str:
        """
        Generate a human-readable report of fairness metrics.

        Returns:
            String containing formatted fairness report
        """
        if not self.metrics:
            return "No fairness metrics have been calculated yet. Run evaluate_fairness() first."
        
        report = []
        report.append("Fairness Evaluation Report")
        report.append("========================")
        
        for metric_name, metric_values in self.metrics.items():
            report.append(f"\n{metric_name.replace('_', ' ').title()}:")
            
            if isinstance(metric_values, dict):
                if isinstance(next(iter(metric_values.values())), dict):
                    # Handle nested metrics (e.g., equalized odds)
                    for feature, values in metric_values.items():
                        report.append(f"  {feature}:")
                        for submetric, value in values.items():
                            report.append(f"    {submetric}: {value:.4f}")
                else:
                    # Handle single-level metrics
                    for feature, value in metric_values.items():
                        report.append(f"  {feature}: {value:.4f}")
            else:
                # Handle scalar metrics (e.g., individual fairness)
                report.append(f"  Score: {metric_values:.4f}")
        
        return "\n".join(report)

    def suggest_improvements(self) -> List[str]:
        """
        Suggest potential improvements based on fairness metrics.

        Returns:
            List of suggested improvements
        """
        if not self.metrics:
            return ["No fairness metrics available. Run evaluate_fairness() first."]
        
        suggestions = []
        
        # Check demographic parity
        for feature, value in self.metrics["demographic_parity"].items():
            if value > 0.1:
                suggestions.append(
                    f"High demographic disparity in {feature}. Consider:\n"
                    "- Applying preprocessing techniques\n"
                    "- Using fairness constraints during training\n"
                    "- Implementing post-processing corrections"
                )
        
        # Check equalized odds
        for feature, values in self.metrics["equalized_odds"].items():
            if values["tpr_difference"] > 0.1 or values["fpr_difference"] > 0.1:
                suggestions.append(
                    f"Equalized odds violations in {feature}. Consider:\n"
                    "- Adjusting classification thresholds\n"
                    "- Using adversarial debiasing\n"
                    "- Implementing equalized odds post-processing"
                )
        
        # Check predictive parity
        for feature, value in self.metrics["predictive_parity"].items():
            if value > 0.1:
                suggestions.append(
                    f"Predictive parity issues in {feature}. Consider:\n"
                    "- Balancing training data\n"
                    "- Using fairness-aware learning algorithms\n"
                    "- Implementing calibrated predictions"
                )
        
        # Check individual fairness
        if self.metrics["individual_fairness"] < 0.9:
            suggestions.append(
                "Low individual fairness score. Consider:\n"
                "- Reviewing feature engineering process\n"
                "- Implementing similarity-based regularization\n"
                "- Using individual fairness constraints"
            )
        
        if not suggestions:
            suggestions.append(
                "All fairness metrics are within acceptable ranges. "
                "Continue monitoring and consider stricter thresholds if needed."
            )
        
        return suggestions 