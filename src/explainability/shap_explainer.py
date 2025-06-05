"""
SHAP-based Model Explainability Module for Responsible AI Implementation.

This module provides comprehensive model explainability capabilities using SHAP
(SHapley Additive exPlanations) including:
- Global feature importance
- Local prediction explanations
- Feature interaction analysis
- Decision plots
- Force plots
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import shap
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.base import BaseEstimator
import joblib
import os

@dataclass
class ShapExplanation:
    """Container for SHAP explanation results."""
    shap_values: np.ndarray
    expected_value: Union[float, List[float]]
    feature_importance: pd.DataFrame
    feature_interactions: Optional[pd.DataFrame]
    sample_explanations: Dict[int, Dict[str, float]]

class ShapExplainer:
    """Main class for SHAP-based model explanations."""
    
    def __init__(
        self,
        model: BaseEstimator,
        feature_names: List[str],
        output_path: str = "explanations",
        background_samples: int = 100
    ):
        """
        Initialize ShapExplainer.
        
        Args:
            model: Trained model to explain
            feature_names: List of feature names
            output_path: Path to save visualizations
            background_samples: Number of background samples for SHAP
        """
        self.model = model
        self.feature_names = feature_names
        self.output_path = output_path
        self.background_samples = background_samples
        self.explainer = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
    
    def fit(
        self,
        data: pd.DataFrame,
        model_type: str = "tree"
    ) -> None:
        """
        Fit the SHAP explainer.
        
        Args:
            data: Training data for background distribution
            model_type: Type of model ('tree', 'linear', or 'kernel')
        """
        # Sample background data
        if len(data) > self.background_samples:
            background_data = data.sample(
                n=self.background_samples,
                random_state=42
            )
        else:
            background_data = data
        
        # Initialize appropriate SHAP explainer
        if model_type == "tree":
            self.explainer = shap.TreeExplainer(
                self.model,
                data=background_data,
                feature_names=self.feature_names
            )
        elif model_type == "linear":
            self.explainer = shap.LinearExplainer(
                self.model,
                background_data,
                feature_names=self.feature_names
            )
        elif model_type == "kernel":
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba
                if hasattr(self.model, 'predict_proba')
                else self.model.predict,
                background_data,
                feature_names=self.feature_names
            )
        else:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                "Use 'tree', 'linear', or 'kernel'."
            )
    
    def explain_predictions(
        self,
        data: pd.DataFrame,
        num_samples: Optional[int] = None
    ) -> ShapExplanation:
        """
        Generate SHAP explanations for predictions.
        
        Args:
            data: Data to explain predictions for
            num_samples: Number of samples to explain (None for all)
            
        Returns:
            ShapExplanation object containing all explanation components
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        # Sample data if requested
        if num_samples is not None and num_samples < len(data):
            data = data.sample(n=num_samples, random_state=42)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(data)
        expected_value = self.explainer.expected_value
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
            if len(shap_values.shape) == 3:  # Multi-class
                # Use mean absolute SHAP values across classes
                shap_values = np.abs(shap_values).mean(axis=0)
        
        # Calculate feature importance
        importance = self._calculate_feature_importance(shap_values)
        
        # Calculate feature interactions if possible
        interactions = self._calculate_feature_interactions(data)
        
        # Generate sample explanations
        sample_explanations = self._generate_sample_explanations(
            data,
            shap_values
        )
        
        return ShapExplanation(
            shap_values=shap_values,
            expected_value=expected_value,
            feature_importance=importance,
            feature_interactions=interactions,
            sample_explanations=sample_explanations
        )
    
    def plot_explanations(
        self,
        explanation: ShapExplanation,
        data: pd.DataFrame
    ) -> None:
        """
        Generate and save explanation visualizations.
        
        Args:
            explanation: ShapExplanation object
            data: Original data used for explanations
        """
        # 1. Summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            explanation.shap_values,
            data,
            feature_names=self.feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, "shap_summary.png"))
        plt.close()
        
        # 2. Bar plot of feature importance
        plt.figure(figsize=(10, 6))
        explanation.feature_importance.plot(
            kind='bar',
            title='Feature Importance (mean |SHAP value|)'
        )
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, "feature_importance.png"))
        plt.close()
        
        # 3. Feature interactions heatmap
        if explanation.feature_interactions is not None:
            plt.figure(figsize=(12, 8))
            shap.plots.heatmap(
                explanation.feature_interactions,
                show=False
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_path, "feature_interactions.png")
            )
            plt.close()
        
        # 4. Individual prediction explanations
        for i in range(min(5, len(data))):  # Plot first 5 samples
            plt.figure(figsize=(12, 4))
            shap.force_plot(
                explanation.expected_value,
                explanation.shap_values[i],
                data.iloc[i],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_path, f"force_plot_{i}.png")
            )
            plt.close()
        
        # 5. Decision plot
        plt.figure(figsize=(10, 12))
        shap.decision_plot(
            explanation.expected_value,
            explanation.shap_values[:10],  # First 10 samples
            data.iloc[:10],
            feature_names=self.feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, "decision_plot.png"))
        plt.close()
    
    def generate_explanation_report(
        self,
        explanation: ShapExplanation,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation report.
        
        Args:
            explanation: ShapExplanation object
            data: Original data used for explanations
            
        Returns:
            Dictionary containing detailed explanation results
        """
        report = {
            'global_importance': {
                'feature_importance': explanation.feature_importance.to_dict(),
                'top_features': explanation.feature_importance.head(5).to_dict()
            },
            'feature_interactions': (
                explanation.feature_interactions.to_dict()
                if explanation.feature_interactions is not None
                else None
            ),
            'sample_explanations': explanation.sample_explanations,
            'summary_statistics': {
                'mean_impact': np.abs(explanation.shap_values).mean(axis=0),
                'max_impact': np.abs(explanation.shap_values).max(axis=0),
                'expected_value': explanation.expected_value
            },
            'recommendations': self._generate_recommendations(explanation)
        }
        
        return report
    
    def _calculate_feature_importance(
        self,
        shap_values: np.ndarray
    ) -> pd.DataFrame:
        """Calculate global feature importance from SHAP values."""
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(shap_values).mean(axis=0)
        })
        return importance.sort_values('importance', ascending=False)
    
    def _calculate_feature_interactions(
        self,
        data: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """Calculate feature interaction strengths if supported."""
        try:
            # Only calculate for tree models (other models too computationally expensive)
            if isinstance(self.explainer, shap.TreeExplainer):
                interactions = self.explainer.shap_interaction_values(data)
                
                # Calculate interaction strengths
                interaction_matrix = np.zeros(
                    (len(self.feature_names), len(self.feature_names))
                )
                
                for i in range(len(self.feature_names)):
                    for j in range(len(self.feature_names)):
                        interaction_matrix[i, j] = np.abs(
                            interactions[:, i, j]
                        ).mean()
                
                return pd.DataFrame(
                    interaction_matrix,
                    index=self.feature_names,
                    columns=self.feature_names
                )
            return None
        except:
            return None
    
    def _generate_sample_explanations(
        self,
        data: pd.DataFrame,
        shap_values: np.ndarray
    ) -> Dict[int, Dict[str, float]]:
        """Generate detailed explanations for sample predictions."""
        explanations = {}
        
        for i in range(min(5, len(data))):  # Explain first 5 samples
            sample_explanation = {}
            for j, feature in enumerate(self.feature_names):
                sample_explanation[feature] = float(shap_values[i, j])
            explanations[i] = sample_explanation
        
        return explanations
    
    def _generate_recommendations(
        self,
        explanation: ShapExplanation
    ) -> List[str]:
        """Generate model improvement recommendations based on explanations."""
        recommendations = []
        
        # Analyze feature importance
        top_features = explanation.feature_importance.head(3)['feature'].tolist()
        recommendations.append(
            f"Top 3 most important features: {', '.join(top_features)}. "
            "Consider focusing on these for model improvements."
        )
        
        # Check for potential redundant features
        low_importance = explanation.feature_importance.tail(
            len(self.feature_names) // 4
        )['feature'].tolist()
        if low_importance:
            recommendations.append(
                f"Consider removing or combining low-importance features: "
                f"{', '.join(low_importance)}"
            )
        
        # Analyze feature interactions
        if explanation.feature_interactions is not None:
            strong_interactions = []
            for i in range(len(self.feature_names)):
                for j in range(i + 1, len(self.feature_names)):
                    if explanation.feature_interactions.iloc[i, j] > 0.1:
                        strong_interactions.append(
                            (self.feature_names[i], self.feature_names[j])
                        )
            
            if strong_interactions:
                recommendations.append(
                    "Strong feature interactions detected between: " +
                    ", ".join([f"{a}-{b}" for a, b in strong_interactions])
                )
        
        return recommendations
    
    def save(self, path: str) -> None:
        """Save the explainer to disk."""
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: str) -> 'ShapExplainer':
        """Load explainer from disk."""
        return joblib.load(path) 