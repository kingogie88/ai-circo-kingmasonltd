"""
Fairness Metrics Calculator Module for Responsible AI Implementation.

This module provides comprehensive fairness metrics calculation including:
- Group fairness metrics
- Individual fairness metrics
- Equality of opportunity
- Predictive parity
- Treatment equality
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from sklearn.metrics import confusion_matrix, roc_auc_score
import scipy.stats as stats

@dataclass
class FairnessMetrics:
    """Container for fairness metrics results."""
    demographic_parity: float
    equal_opportunity: float
    predictive_parity: float
    treatment_equality: float
    disparate_impact: float
    group_metrics: Dict[str, Dict[str, float]]
    individual_metrics: Dict[str, float]

class FairnessCalculator:
    """Main class for calculating fairness metrics."""
    
    def __init__(
        self,
        protected_attributes: List[str],
        target_column: str,
        prediction_column: Optional[str] = None,
        threshold: float = 0.8
    ):
        """
        Initialize FairnessCalculator.
        
        Args:
            protected_attributes: List of protected attribute column names
            target_column: Name of the target variable column
            prediction_column: Name of the model predictions column
            threshold: Threshold for fairness metrics (default: 0.8)
        """
        self.protected_attributes = protected_attributes
        self.target_column = target_column
        self.prediction_column = prediction_column or f"{target_column}_pred"
        self.threshold = threshold
    
    def calculate_all_metrics(
        self,
        data: pd.DataFrame
    ) -> FairnessMetrics:
        """
        Calculate comprehensive fairness metrics.
        
        Args:
            data: DataFrame containing all necessary columns
            
        Returns:
            FairnessMetrics object containing all calculated metrics
        """
        return FairnessMetrics(
            demographic_parity=self.calculate_demographic_parity(data),
            equal_opportunity=self.calculate_equal_opportunity(data),
            predictive_parity=self.calculate_predictive_parity(data),
            treatment_equality=self.calculate_treatment_equality(data),
            disparate_impact=self.calculate_disparate_impact(data),
            group_metrics=self.calculate_group_metrics(data),
            individual_metrics=self.calculate_individual_metrics(data)
        )
    
    def calculate_demographic_parity(
        self,
        data: pd.DataFrame
    ) -> float:
        """
        Calculate demographic parity (statistical parity).
        
        Measures whether predictions are independent of protected attributes.
        """
        predictions = data[self.prediction_column]
        parities = []
        
        for attr in self.protected_attributes:
            groups = data[attr].unique()
            group_rates = []
            
            for group in groups:
                group_pred = predictions[data[attr] == group].mean()
                group_rates.append(group_pred)
            
            # Calculate min/max ratio
            parity = min(group_rates) / max(group_rates)
            parities.append(parity)
        
        return np.mean(parities)
    
    def calculate_equal_opportunity(
        self,
        data: pd.DataFrame
    ) -> float:
        """
        Calculate equal opportunity.
        
        Measures whether true positive rates are equal across protected groups.
        """
        opportunities = []
        
        for attr in self.protected_attributes:
            groups = data[attr].unique()
            group_tprs = []
            
            for group in groups:
                mask = (data[attr] == group) & (data[self.target_column] == 1)
                tpr = (data.loc[mask, self.prediction_column] == 1).mean()
                group_tprs.append(tpr)
            
            # Calculate min/max ratio
            opportunity = min(group_tprs) / max(group_tprs)
            opportunities.append(opportunity)
        
        return np.mean(opportunities)
    
    def calculate_predictive_parity(
        self,
        data: pd.DataFrame
    ) -> float:
        """
        Calculate predictive parity.
        
        Measures whether precision is equal across protected groups.
        """
        parities = []
        
        for attr in self.protected_attributes:
            groups = data[attr].unique()
            group_precisions = []
            
            for group in groups:
                mask = data[attr] == group
                group_data = data[mask]
                
                # Calculate precision
                true_pos = ((group_data[self.prediction_column] == 1) & 
                          (group_data[self.target_column] == 1)).sum()
                pred_pos = (group_data[self.prediction_column] == 1).sum()
                
                precision = true_pos / pred_pos if pred_pos > 0 else 0
                group_precisions.append(precision)
            
            # Calculate min/max ratio
            parity = min(group_precisions) / max(group_precisions)
            parities.append(parity)
        
        return np.mean(parities)
    
    def calculate_treatment_equality(
        self,
        data: pd.DataFrame
    ) -> float:
        """
        Calculate treatment equality.
        
        Measures whether false positive and false negative ratios are equal
        across protected groups.
        """
        equalities = []
        
        for attr in self.protected_attributes:
            groups = data[attr].unique()
            group_ratios = []
            
            for group in groups:
                mask = data[attr] == group
                group_data = data[mask]
                
                # Calculate confusion matrix
                cm = confusion_matrix(
                    group_data[self.target_column],
                    group_data[self.prediction_column]
                )
                
                # Calculate FP/FN ratio
                tn, fp, fn, tp = cm.ravel()
                ratio = fp / fn if fn > 0 else np.inf
                group_ratios.append(ratio)
            
            # Calculate ratio of ratios
            valid_ratios = [r for r in group_ratios if r != np.inf]
            if valid_ratios:
                equality = min(valid_ratios) / max(valid_ratios)
                equalities.append(equality)
        
        return np.mean(equalities) if equalities else 0
    
    def calculate_disparate_impact(
        self,
        data: pd.DataFrame
    ) -> float:
        """
        Calculate disparate impact.
        
        Measures the ratio of positive prediction rates between groups.
        """
        impacts = []
        
        for attr in self.protected_attributes:
            groups = data[attr].unique()
            group_rates = []
            
            for group in groups:
                mask = data[attr] == group
                rate = (data.loc[mask, self.prediction_column] == 1).mean()
                group_rates.append(rate)
            
            # Calculate min/max ratio
            impact = min(group_rates) / max(group_rates)
            impacts.append(impact)
        
        return np.mean(impacts)
    
    def calculate_group_metrics(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate group-specific fairness metrics.
        
        Returns detailed metrics for each protected group.
        """
        metrics = {}
        
        for attr in self.protected_attributes:
            groups = data[attr].unique()
            group_metrics = {}
            
            for group in groups:
                mask = data[attr] == group
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
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                
                group_metrics[str(group)] = {
                    'precision': precision,
                    'recall': recall,
                    'false_positive_rate': fpr,
                    'false_negative_rate': fnr,
                    'positive_predictive_value': precision,
                    'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0
                }
            
            metrics[attr] = group_metrics
        
        return metrics
    
    def calculate_individual_metrics(
        self,
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate individual fairness metrics.
        
        Measures consistency and individual fairness properties.
        """
        metrics = {}
        
        # Calculate consistency score
        metrics['consistency'] = self._calculate_consistency(data)
        
        # Calculate individual fairness score
        metrics['individual_fairness'] = self._calculate_individual_fairness(data)
        
        return metrics
    
    def _calculate_consistency(
        self,
        data: pd.DataFrame,
        k: int = 5
    ) -> float:
        """Calculate consistency score using k-nearest neighbors."""
        from sklearn.neighbors import NearestNeighbors
        
        # Select numeric features only
        numeric_data = data.select_dtypes(include=[np.number])
        numeric_data = numeric_data.drop(
            [self.target_column, self.prediction_column],
            axis=1,
            errors='ignore'
        )
        
        if numeric_data.empty:
            return 0.0
        
        # Normalize data
        normalized_data = (numeric_data - numeric_data.mean()) / numeric_data.std()
        
        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(normalized_data)
        distances, indices = nbrs.kneighbors(normalized_data)
        
        # Calculate consistency
        predictions = data[self.prediction_column].values
        consistency_scores = []
        
        for i in range(len(data)):
            neighbors = indices[i][1:]  # Exclude self
            neighbor_preds = predictions[neighbors]
            consistency = np.mean(neighbor_preds == predictions[i])
            consistency_scores.append(consistency)
        
        return np.mean(consistency_scores)
    
    def _calculate_individual_fairness(
        self,
        data: pd.DataFrame
    ) -> float:
        """
        Calculate individual fairness score.
        
        Measures whether similar individuals receive similar predictions.
        """
        from sklearn.metrics.pairwise import euclidean_distances
        
        # Select numeric features only
        numeric_data = data.select_dtypes(include=[np.number])
        numeric_data = numeric_data.drop(
            [self.target_column, self.prediction_column],
            axis=1,
            errors='ignore'
        )
        
        if numeric_data.empty:
            return 0.0
        
        # Normalize data
        normalized_data = (numeric_data - numeric_data.mean()) / numeric_data.std()
        
        # Calculate pairwise distances
        distances = euclidean_distances(normalized_data)
        predictions = data[self.prediction_column].values
        
        # Calculate prediction differences
        pred_diff = np.abs(
            predictions.reshape(-1, 1) - predictions.reshape(1, -1)
        )
        
        # Calculate correlation between distances and prediction differences
        mask = np.triu(np.ones_like(distances), k=1).astype(bool)
        correlation = stats.spearmanr(
            distances[mask],
            pred_diff[mask]
        ).correlation
        
        # Convert to fairness score (higher correlation is better)
        return max(0, correlation)
    
    def generate_fairness_report(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate comprehensive fairness analysis report.
        
        Args:
            data: DataFrame containing all necessary columns
            
        Returns:
            Dictionary containing detailed fairness analysis results
        """
        metrics = self.calculate_all_metrics(data)
        
        report = {
            'overall_metrics': {
                'demographic_parity': metrics.demographic_parity,
                'equal_opportunity': metrics.equal_opportunity,
                'predictive_parity': metrics.predictive_parity,
                'treatment_equality': metrics.treatment_equality,
                'disparate_impact': metrics.disparate_impact
            },
            'group_metrics': metrics.group_metrics,
            'individual_metrics': metrics.individual_metrics,
            'recommendations': self._generate_recommendations(metrics)
        }
        
        return report
    
    def _generate_recommendations(
        self,
        metrics: FairnessMetrics
    ) -> List[str]:
        """Generate fairness improvement recommendations."""
        recommendations = []
        
        # Check demographic parity
        if metrics.demographic_parity < self.threshold:
            recommendations.append(
                "Consider applying pre-processing techniques to balance "
                "prediction rates across demographic groups."
            )
        
        # Check equal opportunity
        if metrics.equal_opportunity < self.threshold:
            recommendations.append(
                "Model shows unequal true positive rates across groups. "
                "Consider post-processing calibration techniques."
            )
        
        # Check predictive parity
        if metrics.predictive_parity < self.threshold:
            recommendations.append(
                "Precision varies significantly across groups. "
                "Consider adjusting classification thresholds."
            )
        
        # Check treatment equality
        if metrics.treatment_equality < self.threshold:
            recommendations.append(
                "Error rates are imbalanced across groups. "
                "Consider error rate balancing techniques."
            )
        
        # Check disparate impact
        if metrics.disparate_impact < self.threshold:
            recommendations.append(
                "Significant disparate impact detected. "
                "Consider applying fairness constraints during training."
            )
        
        return recommendations 