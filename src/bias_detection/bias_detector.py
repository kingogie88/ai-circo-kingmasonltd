"""
Bias Detection Module for Responsible AI Implementation.

This module provides comprehensive bias detection capabilities including:
- Statistical bias metrics
- Group fairness analysis
- Intersectional bias detection
- Data representation analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from sklearn.metrics import confusion_matrix
import scipy.stats as stats

@dataclass
class BiasMetrics:
    """Container for bias detection metrics."""
    demographic_parity_ratio: float
    equal_opportunity_ratio: float
    disparate_impact_ratio: float
    statistical_parity_difference: float
    group_fairness_metrics: Dict[str, Dict[str, float]]
    intersectional_metrics: Dict[str, float]

class BiasDetector:
    """Main class for detecting and analyzing bias in ML models."""
    
    def __init__(
        self,
        sensitive_features: List[str],
        target_column: str,
        prediction_column: Optional[str] = None,
        threshold: float = 0.8
    ):
        """
        Initialize BiasDetector.
        
        Args:
            sensitive_features: List of column names containing protected attributes
            target_column: Name of the target variable column
            prediction_column: Name of the model predictions column
            threshold: Threshold for determining bias significance (default: 0.8)
        """
        self.sensitive_features = sensitive_features
        self.target_column = target_column
        self.prediction_column = prediction_column or f"{target_column}_pred"
        self.threshold = threshold
        
    def calculate_bias_metrics(
        self,
        data: pd.DataFrame,
        group_by: Optional[str] = None
    ) -> BiasMetrics:
        """
        Calculate comprehensive bias metrics.
        
        Args:
            data: DataFrame containing features, targets, and predictions
            group_by: Optional column name to group results by
            
        Returns:
            BiasMetrics object containing all calculated metrics
        """
        metrics = {}
        
        # Calculate demographic parity
        dp_ratio = self._calculate_demographic_parity(data)
        
        # Calculate equal opportunity
        eo_ratio = self._calculate_equal_opportunity(data)
        
        # Calculate disparate impact
        di_ratio = self._calculate_disparate_impact(data)
        
        # Calculate statistical parity difference
        sp_diff = self._calculate_statistical_parity_difference(data)
        
        # Calculate group fairness metrics
        group_metrics = self._calculate_group_fairness(data)
        
        # Calculate intersectional metrics
        intersectional = self._calculate_intersectional_bias(data)
        
        return BiasMetrics(
            demographic_parity_ratio=dp_ratio,
            equal_opportunity_ratio=eo_ratio,
            disparate_impact_ratio=di_ratio,
            statistical_parity_difference=sp_diff,
            group_fairness_metrics=group_metrics,
            intersectional_metrics=intersectional
        )
    
    def analyze_data_representation(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze representation in training data.
        
        Args:
            data: DataFrame containing features and sensitive attributes
            
        Returns:
            Dictionary containing representation metrics for each sensitive feature
        """
        representation_metrics = {}
        
        for feature in self.sensitive_features:
            # Calculate distribution metrics
            value_counts = data[feature].value_counts(normalize=True)
            
            # Calculate entropy (measure of distribution uniformity)
            entropy = stats.entropy(value_counts)
            
            # Calculate representation ratios
            max_ratio = value_counts.max() / value_counts.min()
            
            representation_metrics[feature] = {
                'entropy': entropy,
                'max_representation_ratio': max_ratio,
                'distribution': value_counts.to_dict()
            }
        
        return representation_metrics
    
    def generate_bias_report(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate comprehensive bias analysis report.
        
        Args:
            data: DataFrame containing all necessary columns
            
        Returns:
            Dictionary containing detailed bias analysis results
        """
        report = {
            'bias_metrics': self.calculate_bias_metrics(data),
            'data_representation': self.analyze_data_representation(data),
            'recommendations': self._generate_recommendations(data)
        }
        
        return report
    
    def _calculate_demographic_parity(self, data: pd.DataFrame) -> float:
        """Calculate demographic parity ratio."""
        predictions = data[self.prediction_column]
        ratios = []
        
        for feature in self.sensitive_features:
            groups = data[feature].unique()
            group_ratios = []
            
            for group in groups:
                group_pred = predictions[data[feature] == group].mean()
                group_ratios.append(group_pred)
            
            ratios.append(min(group_ratios) / max(group_ratios))
        
        return np.mean(ratios)
    
    def _calculate_equal_opportunity(self, data: pd.DataFrame) -> float:
        """Calculate equal opportunity ratio."""
        positive_label = data[self.target_column] == 1
        ratios = []
        
        for feature in self.sensitive_features:
            groups = data[feature].unique()
            group_ratios = []
            
            for group in groups:
                mask = (data[feature] == group) & positive_label
                true_positive_rate = (
                    data.loc[mask, self.prediction_column] == 1
                ).mean()
                group_ratios.append(true_positive_rate)
            
            ratios.append(min(group_ratios) / max(group_ratios))
        
        return np.mean(ratios)
    
    def _calculate_disparate_impact(self, data: pd.DataFrame) -> float:
        """Calculate disparate impact ratio."""
        predictions = data[self.prediction_column]
        ratios = []
        
        for feature in self.sensitive_features:
            groups = data[feature].unique()
            group_ratios = []
            
            for group in groups:
                group_pred = predictions[data[feature] == group].mean()
                group_ratios.append(group_pred)
            
            ratios.append(min(group_ratios) / max(group_ratios))
        
        return np.mean(ratios)
    
    def _calculate_statistical_parity_difference(
        self,
        data: pd.DataFrame
    ) -> float:
        """Calculate statistical parity difference."""
        predictions = data[self.prediction_column]
        differences = []
        
        for feature in self.sensitive_features:
            groups = data[feature].unique()
            group_preds = []
            
            for group in groups:
                group_pred = predictions[data[feature] == group].mean()
                group_preds.append(group_pred)
            
            differences.append(max(group_preds) - min(group_preds))
        
        return np.mean(differences)
    
    def _calculate_group_fairness(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Calculate group-specific fairness metrics."""
        metrics = {}
        
        for feature in self.sensitive_features:
            groups = data[feature].unique()
            group_metrics = {}
            
            for group in groups:
                mask = data[feature] == group
                group_data = data[mask]
                
                # Calculate confusion matrix
                cm = confusion_matrix(
                    group_data[self.target_column],
                    group_data[self.prediction_column]
                )
                
                # Calculate metrics
                tn, fp, fn, tp = cm.ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                group_metrics[str(group)] = {
                    'precision': precision,
                    'recall': recall,
                    'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                    'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
                }
            
            metrics[feature] = group_metrics
        
        return metrics
    
    def _calculate_intersectional_bias(
        self,
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate intersectional bias metrics."""
        if len(self.sensitive_features) < 2:
            return {}
        
        # Create intersectional groups
        data['intersectional_group'] = data[self.sensitive_features].apply(
            lambda x: '_'.join(x.astype(str)),
            axis=1
        )
        
        # Calculate prediction rates for each intersectional group
        group_rates = data.groupby('intersectional_group')[
            self.prediction_column
        ].mean()
        
        # Calculate metrics
        metrics = {
            'max_disparity': group_rates.max() - group_rates.min(),
            'ratio_disparity': group_rates.min() / group_rates.max(),
            'std_deviation': group_rates.std()
        }
        
        return metrics
    
    def _generate_recommendations(
        self,
        data: pd.DataFrame
    ) -> List[str]:
        """Generate bias mitigation recommendations."""
        recommendations = []
        metrics = self.calculate_bias_metrics(data)
        
        # Check demographic parity
        if metrics.demographic_parity_ratio < self.threshold:
            recommendations.append(
                "Consider applying pre-processing techniques to balance "
                "prediction rates across demographic groups."
            )
        
        # Check equal opportunity
        if metrics.equal_opportunity_ratio < self.threshold:
            recommendations.append(
                "Model shows unequal true positive rates across groups. "
                "Consider applying post-processing calibration."
            )
        
        # Check disparate impact
        if metrics.disparate_impact_ratio < self.threshold:
            recommendations.append(
                "Significant disparate impact detected. Review feature "
                "selection and consider fairness constraints during training."
            )
        
        # Check data representation
        rep_metrics = self.analyze_data_representation(data)
        for feature, metrics in rep_metrics.items():
            if metrics['max_representation_ratio'] > 3:
                recommendations.append(
                    f"Significant imbalance detected in {feature}. "
                    "Consider data augmentation or resampling techniques."
                )
        
        return recommendations 