"""
SHAP (SHapley Additive exPlanations) explainer module for model interpretability.
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator

class ShapExplainer:
    """A class for explaining model predictions using SHAP values."""
    
    def __init__(
        self,
        model: BaseEstimator,
        background_data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        model_type: str = "tree"
    ):
        """
        Initialize the SHAP explainer.

        Args:
            model: The trained model to explain
            background_data: Background data for SHAP explainer initialization
            model_type: Type of model ('tree', 'linear', 'kernel', or 'deep')
        """
        self.model = model
        self.background_data = background_data
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        
        self._initialize_explainer()

    def _initialize_explainer(self):
        """Initialize the appropriate SHAP explainer based on model type."""
        if self.model_type == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == "linear":
            self.explainer = shap.LinearExplainer(self.model, self.background_data)
        elif self.model_type == "kernel":
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba 
                if hasattr(self.model, "predict_proba") 
                else self.model.predict,
                self.background_data
            )
        elif self.model_type == "deep":
            self.explainer = shap.DeepExplainer(self.model, self.background_data)
        else:
            raise ValueError(
                f"Unsupported model type: {self.model_type}. "
                "Supported types are: 'tree', 'linear', 'kernel', 'deep'"
            )

    def explain_instance(
        self,
        instance: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Generate SHAP explanations for a single instance.

        Args:
            instance: The instance to explain
            feature_names: List of feature names

        Returns:
            Dictionary mapping features to their SHAP values
        """
        # Ensure instance is in the correct format
        if isinstance(instance, pd.DataFrame):
            self.feature_names = feature_names or instance.columns.tolist()
            instance = instance.values
        else:
            self.feature_names = feature_names or [f"feature_{i}" for i in range(instance.shape[-1])]
        
        # Reshape if needed
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(instance)
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        # Create explanation dictionary
        explanation = dict(zip(self.feature_names, shap_values[0]))
        return explanation

    def explain_dataset(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Generate SHAP explanations for a dataset.

        Args:
            data: The dataset to explain
            feature_names: List of feature names

        Returns:
            Array of SHAP values for the dataset
        """
        # Handle DataFrame input
        if isinstance(data, pd.DataFrame):
            self.feature_names = feature_names or data.columns.tolist()
            data = data.values
        else:
            self.feature_names = feature_names or [f"feature_{i}" for i in range(data.shape[1])]
        
        # Calculate SHAP values
        self.shap_values = self.explainer.shap_values(data)
        
        # Handle different SHAP value formats
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1] if len(self.shap_values) > 1 else self.shap_values[0]
        
        return self.shap_values

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
        explanation = self.explain_instance(instance, feature_names)
        
        # Sort features by absolute SHAP value
        sorted_features = sorted(
            explanation.items(),
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
        total_impact = sum(abs(v) for v in explanation.values())
        report.append(f"\nTotal Feature Impact: {total_impact:.4f}")
        
        return "\n".join(report) 