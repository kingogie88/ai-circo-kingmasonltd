"""
Privacy Protection Module for Responsible AI Implementation.

This module provides comprehensive privacy protection capabilities including:
- Differential privacy
- Data anonymization
- Privacy-preserving data transformations
- Privacy risk assessment
- Privacy budget management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import hashlib
from sklearn.preprocessing import StandardScaler
import warnings
from scipy import stats

@dataclass
class PrivacyMetrics:
    """Container for privacy metrics results."""
    k_anonymity: int
    l_diversity: float
    t_closeness: float
    privacy_risk_score: float
    attribute_disclosure_risk: Dict[str, float]
    identification_risk: Dict[str, float]

class PrivacyProtector:
    """Main class for privacy protection and risk assessment."""
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        sensitive_attributes: Optional[List[str]] = None,
        quasi_identifiers: Optional[List[str]] = None,
        random_state: int = 42
    ):
        """
        Initialize PrivacyProtector.
        
        Args:
            epsilon: Privacy budget for differential privacy
            delta: Delta parameter for differential privacy
            sensitive_attributes: List of sensitive attribute names
            quasi_identifiers: List of quasi-identifier attribute names
            random_state: Random seed for reproducibility
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitive_attributes = sensitive_attributes or []
        self.quasi_identifiers = quasi_identifiers or []
        self.random_state = random_state
        np.random.seed(random_state)
    
    def anonymize_data(
        self,
        data: pd.DataFrame,
        method: str = "differential_privacy"
    ) -> pd.DataFrame:
        """
        Apply privacy-preserving transformations to data.
        
        Args:
            data: DataFrame to anonymize
            method: Anonymization method ('differential_privacy', 'k_anonymity',
                   or 'generalization')
            
        Returns:
            Anonymized DataFrame
        """
        if method == "differential_privacy":
            return self._apply_differential_privacy(data)
        elif method == "k_anonymity":
            return self._apply_k_anonymity(data)
        elif method == "generalization":
            return self._apply_generalization(data)
        else:
            raise ValueError(
                f"Unsupported anonymization method: {method}. "
                "Use 'differential_privacy', 'k_anonymity', or 'generalization'."
            )
    
    def assess_privacy_risks(
        self,
        data: pd.DataFrame
    ) -> PrivacyMetrics:
        """
        Assess privacy risks in the data.
        
        Args:
            data: DataFrame to assess
            
        Returns:
            PrivacyMetrics object containing risk assessment results
        """
        # Calculate k-anonymity
        k_anonymity = self._calculate_k_anonymity(data)
        
        # Calculate l-diversity
        l_diversity = self._calculate_l_diversity(data)
        
        # Calculate t-closeness
        t_closeness = self._calculate_t_closeness(data)
        
        # Calculate attribute disclosure risks
        attr_risks = self._calculate_attribute_disclosure_risks(data)
        
        # Calculate identification risks
        id_risks = self._calculate_identification_risks(data)
        
        # Calculate overall privacy risk score
        risk_score = self._calculate_privacy_risk_score(
            k_anonymity,
            l_diversity,
            t_closeness,
            attr_risks,
            id_risks
        )
        
        return PrivacyMetrics(
            k_anonymity=k_anonymity,
            l_diversity=l_diversity,
            t_closeness=t_closeness,
            privacy_risk_score=risk_score,
            attribute_disclosure_risk=attr_risks,
            identification_risk=id_risks
        )
    
    def generate_privacy_report(
        self,
        data: pd.DataFrame,
        metrics: PrivacyMetrics
    ) -> Dict[str, Any]:
        """
        Generate comprehensive privacy assessment report.
        
        Args:
            data: Original DataFrame
            metrics: PrivacyMetrics object
            
        Returns:
            Dictionary containing detailed privacy assessment results
        """
        report = {
            'privacy_metrics': {
                'k_anonymity': metrics.k_anonymity,
                'l_diversity': metrics.l_diversity,
                't_closeness': metrics.t_closeness,
                'privacy_risk_score': metrics.privacy_risk_score
            },
            'attribute_risks': metrics.attribute_disclosure_risk,
            'identification_risks': metrics.identification_risk,
            'recommendations': self._generate_recommendations(metrics),
            'privacy_budget': {
                'epsilon': self.epsilon,
                'delta': self.delta,
                'remaining_budget': self._calculate_remaining_budget()
            }
        }
        
        return report
    
    def _apply_differential_privacy(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply differential privacy using Laplace mechanism."""
        anonymized_data = data.copy()
        
        # Apply differential privacy to numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            sensitivity = (data[col].max() - data[col].min()) / len(data)
            noise_scale = sensitivity / self.epsilon
            noise = np.random.laplace(0, noise_scale, len(data))
            anonymized_data[col] = data[col] + noise
        
        # Apply differential privacy to categorical columns
        categorical_cols = data.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            # Use exponential mechanism for categorical data
            unique_values = data[col].unique()
            probabilities = np.exp(
                self.epsilon * np.random.random(len(unique_values)) / 2
            )
            probabilities /= probabilities.sum()
            
            anonymized_data[col] = data[col].map(
                dict(zip(unique_values, np.random.permutation(unique_values)))
            )
        
        return anonymized_data
    
    def _apply_k_anonymity(
        self,
        data: pd.DataFrame,
        k: int = 5
    ) -> pd.DataFrame:
        """Apply k-anonymity through generalization and suppression."""
        anonymized_data = data.copy()
        
        if not self.quasi_identifiers:
            warnings.warn(
                "No quasi-identifiers specified. K-anonymity may not be effective."
            )
            return anonymized_data
        
        # Group by quasi-identifiers
        groups = data.groupby(self.quasi_identifiers).size()
        small_groups = groups[groups < k].index
        
        # Suppress small groups
        for group in small_groups:
            mask = True
            for attr, value in zip(self.quasi_identifiers, group):
                mask = mask & (anonymized_data[attr] == value)
            
            # Generalize or suppress values
            for attr in self.quasi_identifiers:
                anonymized_data.loc[mask, attr] = '*'
        
        return anonymized_data
    
    def _apply_generalization(
        self,
        data: pd.DataFrame,
        num_bins: int = 5
    ) -> pd.DataFrame:
        """Apply generalization to quasi-identifiers."""
        anonymized_data = data.copy()
        
        for col in self.quasi_identifiers:
            if data[col].dtype in [np.float64, np.int64]:
                # Bin numeric data
                anonymized_data[col] = pd.qcut(
                    data[col],
                    q=num_bins,
                    labels=False,
                    duplicates='drop'
                )
            else:
                # Hash categorical data
                anonymized_data[col] = data[col].apply(
                    lambda x: hashlib.sha256(
                        str(x).encode()
                    ).hexdigest()[:8]
                )
        
        return anonymized_data
    
    def _calculate_k_anonymity(self, data: pd.DataFrame) -> int:
        """Calculate k-anonymity level."""
        if not self.quasi_identifiers:
            return len(data)
        
        # Count occurrences of each combination of quasi-identifiers
        groups = data.groupby(self.quasi_identifiers).size()
        return int(groups.min()) if not groups.empty else 1
    
    def _calculate_l_diversity(self, data: pd.DataFrame) -> float:
        """Calculate l-diversity."""
        if not self.sensitive_attributes or not self.quasi_identifiers:
            return float('inf')
        
        diversities = []
        
        # Calculate diversity for each sensitive attribute
        for sensitive_attr in self.sensitive_attributes:
            groups = data.groupby(self.quasi_identifiers)[sensitive_attr]
            group_diversities = groups.nunique()
            diversities.append(float(group_diversities.min()))
        
        return min(diversities) if diversities else float('inf')
    
    def _calculate_t_closeness(self, data: pd.DataFrame) -> float:
        """Calculate t-closeness."""
        if not self.sensitive_attributes or not self.quasi_identifiers:
            return 0.0
        
        closeness_values = []
        
        for sensitive_attr in self.sensitive_attributes:
            if data[sensitive_attr].dtype in [np.float64, np.int64]:
                # For numeric attributes, use EMD
                overall_dist = data[sensitive_attr].values
                
                groups = data.groupby(self.quasi_identifiers)[sensitive_attr]
                for _, group_values in groups:
                    if len(group_values) > 0:
                        emd = stats.wasserstein_distance(
                            group_values,
                            overall_dist
                        )
                        closeness_values.append(emd)
            else:
                # For categorical attributes, use value distribution difference
                overall_dist = data[sensitive_attr].value_counts(normalize=True)
                
                groups = data.groupby(self.quasi_identifiers)[sensitive_attr]
                for _, group_values in groups:
                    if len(group_values) > 0:
                        group_dist = group_values.value_counts(normalize=True)
                        diff = abs(overall_dist - group_dist).sum() / 2
                        closeness_values.append(diff)
        
        return max(closeness_values) if closeness_values else 0.0
    
    def _calculate_attribute_disclosure_risks(
        self,
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate attribute disclosure risks."""
        risks = {}
        
        for attr in self.sensitive_attributes:
            # Calculate uniqueness of values
            value_counts = data[attr].value_counts(normalize=True)
            entropy = stats.entropy(value_counts)
            max_entropy = np.log(len(value_counts))
            
            # Normalize risk score
            risk = 1 - (entropy / max_entropy if max_entropy > 0 else 0)
            risks[attr] = risk
        
        return risks
    
    def _calculate_identification_risks(
        self,
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate identification risks for quasi-identifiers."""
        risks = {}
        
        for attr in self.quasi_identifiers:
            # Calculate uniqueness and frequency of values
            value_counts = data[attr].value_counts()
            uniqueness = (value_counts == 1).sum() / len(data)
            
            # Calculate average group size
            avg_group_size = len(data) / len(value_counts)
            
            # Combine metrics into risk score
            risk = 0.6 * uniqueness + 0.4 * (1 / avg_group_size)
            risks[attr] = risk
        
        return risks
    
    def _calculate_privacy_risk_score(
        self,
        k_anonymity: int,
        l_diversity: float,
        t_closeness: float,
        attr_risks: Dict[str, float],
        id_risks: Dict[str, float]
    ) -> float:
        """Calculate overall privacy risk score."""
        # Normalize k-anonymity (higher is better)
        k_score = 1 - (1 / k_anonymity if k_anonymity > 0 else 1)
        
        # Normalize l-diversity (higher is better)
        l_score = 1 - (1 / l_diversity if l_diversity > 0 else 1)
        
        # T-closeness is already normalized (lower is better)
        t_score = 1 - t_closeness
        
        # Average attribute and identification risks
        avg_attr_risk = np.mean(list(attr_risks.values()))
        avg_id_risk = np.mean(list(id_risks.values()))
        
        # Combine scores with weights
        weights = {
            'k_anonymity': 0.3,
            'l_diversity': 0.2,
            't_closeness': 0.2,
            'attr_risk': 0.15,
            'id_risk': 0.15
        }
        
        risk_score = (
            weights['k_anonymity'] * k_score +
            weights['l_diversity'] * l_score +
            weights['t_closeness'] * t_score +
            weights['attr_risk'] * (1 - avg_attr_risk) +
            weights['id_risk'] * (1 - avg_id_risk)
        )
        
        return risk_score
    
    def _calculate_remaining_budget(self) -> float:
        """Calculate remaining privacy budget."""
        # Simple implementation - could be extended for more sophisticated tracking
        return self.epsilon
    
    def _generate_recommendations(
        self,
        metrics: PrivacyMetrics
    ) -> List[str]:
        """Generate privacy improvement recommendations."""
        recommendations = []
        
        # Check k-anonymity
        if metrics.k_anonymity < 5:
            recommendations.append(
                "K-anonymity is low. Consider increasing generalization "
                "or suppression of quasi-identifiers."
            )
        
        # Check l-diversity
        if metrics.l_diversity < 3:
            recommendations.append(
                "L-diversity is low. Consider additional protections "
                "for sensitive attributes."
            )
        
        # Check t-closeness
        if metrics.t_closeness > 0.5:
            recommendations.append(
                "T-closeness is high. Consider additional anonymization "
                "of sensitive attributes."
            )
        
        # Check attribute disclosure risks
        high_risk_attrs = [
            attr for attr, risk in metrics.attribute_disclosure_risk.items()
            if risk > 0.7
        ]
        if high_risk_attrs:
            recommendations.append(
                f"High disclosure risk for attributes: {', '.join(high_risk_attrs)}. "
                "Consider additional privacy protections."
            )
        
        # Check identification risks
        high_risk_ids = [
            attr for attr, risk in metrics.identification_risk.items()
            if risk > 0.7
        ]
        if high_risk_ids:
            recommendations.append(
                f"High identification risk for quasi-identifiers: "
                f"{', '.join(high_risk_ids)}. Consider additional generalization."
            )
        
        # Check overall privacy risk
        if metrics.privacy_risk_score < 0.6:
            recommendations.append(
                "Overall privacy risk is high. Consider applying stronger "
                "privacy-preserving transformations."
            )
        
        return recommendations 