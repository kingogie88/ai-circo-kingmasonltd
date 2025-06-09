"""
SHAP-based Model Explainability Module for Responsible AI Implementation.

This module provides tools for explaining model predictions using SHAP (SHapley Additive exPlanations).
"""

import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

@dataclass
class ShapExplanation:
    """Container for SHAP explanation results."""
    feature_importance: Dict[str, float]
    shap_values: np.ndarray
    base_value: float
    explanation_text: str

class ShapExplainer:
    """Main class for explaining model predictions using SHAP."""
    
    def __init__(self, model: Any, feature_names: List[str]):
        """
        Initialize ShapExplainer.
        
        Args:
            model: The trained model to explain
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        
    def _initialize_explainer(self, background_data: np.ndarray):
        """
        Initialize the SHAP explainer based on model type.
        
        Args:
            background_data: Representative dataset for SHAP explainer
        """
        try:
            # Try Tree explainer first (for tree-based models)
            self.explainer = shap.TreeExplainer(self.model, background_data)
        except:
            try:
                # Try Kernel explainer as fallback
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba if hasattr(self.model, 'predict_proba')
                    else self.model.predict,
                    background_data
                )
            except Exception as e:
                raise ValueError(f"Could not initialize SHAP explainer: {str(e)}")
    
    def explain_instance(
        self,
        instance: np.ndarray,
        background_data: np.ndarray
    ) -> ShapExplanation:
        """
        Generate SHAP explanation for a single instance.
        
        Args:
            instance: The instance to explain
            background_data: Representative dataset for SHAP explainer
            
        Returns:
            ShapExplanation object containing the explanation
        """
        if self.explainer is None:
            self._initialize_explainer(background_data)
        
        # Reshape instance if needed
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(instance)
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # For multi-class, take positive class
            shap_values = shap_values[1]
        
        # Get base value
        base_value = (
            self.explainer.expected_value[1]
            if isinstance(self.explainer.expected_value, (list, np.ndarray))
            else self.explainer.expected_value
        )
        
        # Calculate feature importance
        feature_importance = dict(zip(
            self.feature_names,
            np.abs(shap_values[0])
        ))
        
        # Generate explanation text
        explanation_text = self._generate_explanation_text(
            feature_importance,
            shap_values[0],
            base_value
        )
        
        return ShapExplanation(
            feature_importance=feature_importance,
            shap_values=shap_values,
            base_value=base_value,
            explanation_text=explanation_text
        )
    
    def explain_dataset(
        self,
        data: np.ndarray,
        background_data: np.ndarray,
        max_display: int = 10
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for a dataset.
        
        Args:
            data: The dataset to explain
            background_data: Representative dataset for SHAP explainer
            max_display: Maximum number of features to include in summary
            
        Returns:
            Dictionary containing global explanations
        """
        if self.explainer is None:
            self._initialize_explainer(background_data)
        
        # Calculate SHAP values for all instances
        shap_values = self.explainer.shap_values(data)
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Calculate global feature importance
        global_importance = np.abs(shap_values).mean(axis=0)
        feature_importance = dict(zip(
            self.feature_names,
            global_importance
        ))
        
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Generate summary
        summary = {
            'global_importance': dict(sorted_features[:max_display]),
            'shap_values': shap_values,
            'base_value': (
                self.explainer.expected_value[1]
                if isinstance(self.explainer.expected_value, (list, np.ndarray))
                else self.explainer.expected_value
            )
        }
        
        return summary
    
    def _generate_explanation_text(
        self,
        feature_importance: Dict[str, float],
        shap_values: np.ndarray,
        base_value: float
    ) -> str:
        """
        Generate human-readable explanation text.
        
        Args:
            feature_importance: Dictionary of feature importance values
            shap_values: SHAP values for the instance
            base_value: Base value for the model
            
        Returns:
            Human-readable explanation text
        """
        # Sort features by absolute importance
        sorted_features = sorted(
            zip(self.feature_names, shap_values),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Generate explanation
        explanation = [
            "Model Prediction Explanation:",
            f"Base prediction value: {base_value:.3f}\n",
            "Top contributing features:"
        ]
        
        for feature, value in sorted_features[:5]:  # Show top 5 features
            impact = "increased" if value > 0 else "decreased"
            explanation.append(
                f"- {feature} {impact} the prediction by {abs(value):.3f}"
            )
        
        return "\n".join(explanation)

    def plot_feature_importance(
        self,
        max_display: int = 20,
        plot_type: str = "bar"
    ) -> None:
        """
        Plot feature importance based on SHAP values.

        Args:
            max_display: Maximum number of features to display
            plot_type: Type of plot ('bar' or 'beeswarm')
        """
        if self.shap_values is None:
            raise ValueError("No SHAP values available. Run explain_dataset() first.")
        
        plt.figure(figsize=(10, 6))
        
        if plot_type == "bar":
            shap.summary_plot(
                self.shap_values,
                features=self.background_data if isinstance(self.background_data, pd.DataFrame) else None,
                feature_names=self.feature_names,
                plot_type="bar",
                max_display=max_display,
                show=False
            )
        elif plot_type == "beeswarm":
            shap.summary_plot(
                self.shap_values,
                features=self.background_data if isinstance(self.background_data, pd.DataFrame) else None,
                feature_names=self.feature_names,
                max_display=max_display,
                show=False
            )
        else:
            raise ValueError("plot_type must be either 'bar' or 'beeswarm'")
        
        plt.tight_layout()
        plt.show()

    def plot_feature_dependence(
        self,
        feature_idx: Union[int, str],
        interaction_idx: Optional[Union[int, str]] = None
    ) -> None:
        """
        Plot feature dependence and interaction plots.

        Args:
            feature_idx: Index or name of the feature to plot
            interaction_idx: Index or name of the interaction feature
        """
        if self.shap_values is None:
            raise ValueError("No SHAP values available. Run explain_dataset() first.")
        
        # Convert feature names to indices if necessary
        if isinstance(feature_idx, str):
            feature_idx = self.feature_names.index(feature_idx)
        if isinstance(interaction_idx, str):
            interaction_idx = self.feature_names.index(interaction_idx)
        
        plt.figure(figsize=(10, 6))
        
        if interaction_idx is not None:
            shap.dependence_plot(
                feature_idx,
                self.shap_values,
                self.background_data if isinstance(self.background_data, pd.DataFrame) else None,
                interaction_index=interaction_idx,
                feature_names=self.feature_names,
                show=False
            )
        else:
            shap.dependence_plot(
                feature_idx,
                self.shap_values,
                self.background_data if isinstance(self.background_data, pd.DataFrame) else None,
                feature_names=self.feature_names,
                show=False
            )
        
        plt.tight_layout()
        plt.show()

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get global feature importance based on mean absolute SHAP values.

        Returns:
            Dictionary mapping features to their importance scores
        """
        if self.shap_values is None:
            raise ValueError("No SHAP values available. Run explain_dataset() first.")
        
        # Calculate mean absolute SHAP values for each feature
        importance_values = np.abs(self.shap_values).mean(axis=0)
        
        # Create importance dictionary
        importance_dict = dict(zip(self.feature_names, importance_values))
        
        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    def generate_explanation_report(
        self,
        instance: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        top_k: int = 5
    ) -> str:
        """
        Generate a human-readable explanation report for a single instance.

        Args:
            instance: The instance to explain
            feature_names: List of feature names
            top_k: Number of top features to include in the report

        Returns:
            String containing the explanation report
        """
        explanation = self.explain_instance(instance, self.background_data)
        
        # Sort features by absolute SHAP value
        sorted_features = sorted(
            explanation.feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Generate report
        report = ["Model Prediction Explanation"]
        report.append("=========================")
        report.append(f"\nTop {top_k} Most Influential Features:")
        
        for feature, shap_value in sorted_features[:top_k]:
            impact = "increased" if shap_value > 0 else "decreased"
            report.append(
                f"- {feature}: {impact} the prediction by {abs(shap_value):.4f} units"
            )
        
        # Add summary
        total_impact = sum(abs(v) for v in explanation.feature_importance.values())
        report.append(f"\nTotal Feature Impact: {total_impact:.4f}")
        
        return "\n".join(report) 