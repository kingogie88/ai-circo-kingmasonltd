"""
PAIR (People + AI Research) Visualization Module for Responsible AI Implementation.

This module provides interactive visualization tools based on Google's PAIR initiative
for better model understanding and what-if analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    plot_width: int = 800
    plot_height: int = 600
    color_scale: str = 'viridis'
    point_size: int = 5
    opacity: float = 0.7

class PairVisualizer:
    """Main class for creating interactive model visualizations."""
    
    def __init__(
        self,
        config: Optional[VisualizationConfig] = None
    ):
        """Initialize PairVisualizer."""
        self.config = config or VisualizationConfig()
    
    def feature_importance_plot(
        self,
        feature_names: List[str],
        importance_scores: List[float],
        title: str = "Feature Importance"
    ) -> go.Figure:
        """Create feature importance visualization."""
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        })
        df = df.sort_values('Importance', ascending=True)
        
        fig = go.Figure(go.Bar(
            x=df['Importance'],
            y=df['Feature'],
            orientation='h'
        ))
        
        fig.update_layout(
            title=title,
            width=self.config.plot_width,
            height=self.config.plot_height
        )
        
        return fig
    
    def prediction_distribution_plot(
        self,
        predictions: List[float],
        actual_values: Optional[List[float]] = None,
        title: str = "Prediction Distribution"
    ) -> go.Figure:
        """Create prediction distribution visualization."""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=predictions,
            name="Predictions",
            opacity=self.config.opacity
        ))
        
        if actual_values is not None:
            fig.add_trace(go.Histogram(
                x=actual_values,
                name="Actual Values",
                opacity=self.config.opacity
            ))
        
        fig.update_layout(
            title=title,
            width=self.config.plot_width,
            height=self.config.plot_height,
            barmode='overlay'
        )
        
        return fig
    
    def feature_correlation_plot(
        self,
        data: pd.DataFrame,
        title: str = "Feature Correlations"
    ) -> go.Figure:
        """Create feature correlation heatmap."""
        corr_matrix = data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale=self.config.color_scale
        ))
        
        fig.update_layout(
            title=title,
            width=self.config.plot_width,
            height=self.config.plot_height
        )
        
        return fig
    
    def dimensionality_reduction_plot(
        self,
        data: np.ndarray,
        labels: Optional[List[Any]] = None,
        method: str = 'tsne',
        title: Optional[str] = None
    ) -> go.Figure:
        """Create dimensionality reduction visualization."""
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:
            reducer = PCA(n_components=2)
        
        reduced_data = reducer.fit_transform(data)
        
        if title is None:
            title = f"{method.upper()} Visualization"
        
        if labels is not None:
            fig = px.scatter(
                x=reduced_data[:, 0],
                y=reduced_data[:, 1],
                color=labels,
                title=title
            )
        else:
            fig = px.scatter(
                x=reduced_data[:, 0],
                y=reduced_data[:, 1],
                title=title
            )
        
        fig.update_layout(
            width=self.config.plot_width,
            height=self.config.plot_height
        )
        
        return fig
    
    def partial_dependence_plot(
        self,
        feature_values: List[float],
        predictions: List[float],
        feature_name: str,
        title: Optional[str] = None
    ) -> go.Figure:
        """Create partial dependence plot."""
        if title is None:
            title = f"Partial Dependence Plot for {feature_name}"
        
        fig = go.Figure(go.Scatter(
            x=feature_values,
            y=predictions,
            mode='lines+markers',
            name=feature_name
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=feature_name,
            yaxis_title="Predicted Value",
            width=self.config.plot_width,
            height=self.config.plot_height
        )
        
        return fig 