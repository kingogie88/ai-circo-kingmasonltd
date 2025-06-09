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
        model: Any,
        feature_names: List[str],
        config: Optional[VisualizationConfig] = None
    ):
        """
        Initialize PairVisualizer.
        
        Args:
            model: The trained model to visualize
            feature_names: List of feature names
            config: Optional visualization configuration
        """
        self.model = model
        self.feature_names = feature_names
        self.config = config or VisualizationConfig()
    
    def create_feature_importance_plot(
        self,
        importance_scores: Dict[str, float]
    ) -> go.Figure:
        """
        Create interactive feature importance visualization.
        
        Args:
            importance_scores: Dictionary mapping features to importance scores
            
        Returns:
            Plotly figure object
        """
        # Sort features by importance
        sorted_features = sorted(
            importance_scores.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        features, scores = zip(*sorted_features)
        
        # Create bar plot
        fig = go.Figure(data=[
            go.Bar(
                x=features,
                y=scores,
                marker_color=scores,
                marker_colorscale=self.config.color_scale
            )
        ])
        
        # Update layout
        fig.update_layout(
            title="Feature Importance Visualization",
            xaxis_title="Features",
            yaxis_title="Importance Score",
            width=self.config.plot_width,
            height=self.config.plot_height
        )
        
        return fig
    
    def create_prediction_scatter(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        x_feature: str,
        y_feature: str,
        color_by: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive scatter plot of predictions.
        
        Args:
            data: Input data
            predictions: Model predictions
            x_feature: Feature for x-axis
            y_feature: Feature for y-axis
            color_by: Optional feature to color points by
            
        Returns:
            Plotly figure object
        """
        # Create scatter plot
        fig = px.scatter(
            data,
            x=x_feature,
            y=y_feature,
            color=color_by if color_by else None,
            color_continuous_scale=self.config.color_scale,
            size_max=self.config.point_size,
            opacity=self.config.opacity
        )
        
        # Add predictions as marker size
        fig.update_traces(
            marker=dict(size=predictions * self.config.point_size)
        )
        
        # Update layout
        fig.update_layout(
            title=f"Prediction Visualization: {x_feature} vs {y_feature}",
            width=self.config.plot_width,
            height=self.config.plot_height
        )
        
        return fig
    
    def create_embedding_plot(
        self,
        data: pd.DataFrame,
        method: str = 'tsne',
        perplexity: int = 30,
        color_by: Optional[str] = None
    ) -> go.Figure:
        """
        Create dimensionality reduction plot.
        
        Args:
            data: Input data
            method: 'tsne' or 'pca'
            perplexity: Perplexity for t-SNE
            color_by: Optional feature to color points by
            
        Returns:
            Plotly figure object
        """
        # Perform dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=42
            )
        else:  # PCA
            reducer = PCA(n_components=2)
        
        # Fit and transform
        embedding = reducer.fit_transform(data)
        
        # Create plot data
        plot_data = pd.DataFrame({
            'x': embedding[:, 0],
            'y': embedding[:, 1],
            'color': data[color_by] if color_by else None
        })
        
        # Create scatter plot
        fig = px.scatter(
            plot_data,
            x='x',
            y='y',
            color='color' if color_by else None,
            color_continuous_scale=self.config.color_scale,
            opacity=self.config.opacity
        )
        
        # Update layout
        fig.update_layout(
            title=f"{method.upper()} Embedding Visualization",
            xaxis_title=f"{method.upper()} 1",
            yaxis_title=f"{method.upper()} 2",
            width=self.config.plot_width,
            height=self.config.plot_height
        )
        
        return fig
    
    def create_what_if_plot(
        self,
        feature: str,
        range_min: float,
        range_max: float,
        n_points: int = 100,
        reference_point: Optional[np.ndarray] = None
    ) -> go.Figure:
        """
        Create what-if analysis plot for a feature.
        
        Args:
            feature: Feature to analyze
            range_min: Minimum value for feature range
            range_max: Maximum value for feature range
            n_points: Number of points to evaluate
            reference_point: Optional reference point
            
        Returns:
            Plotly figure object
        """
        # Generate feature values
        feature_values = np.linspace(range_min, range_max, n_points)
        
        # Create prediction data
        if reference_point is not None:
            predictions = []
            for value in feature_values:
                point = reference_point.copy()
                point[self.feature_names.index(feature)] = value
                predictions.append(self.model.predict([point])[0])
        else:
            # Simple single feature prediction
            predictions = self.model.predict(
                feature_values.reshape(-1, 1)
            )
        
        # Create line plot
        fig = go.Figure(data=[
            go.Scatter(
                x=feature_values,
                y=predictions,
                mode='lines',
                name='Model Predictions'
            )
        ])
        
        # Add reference point if provided
        if reference_point is not None:
            fig.add_trace(
                go.Scatter(
                    x=[reference_point[self.feature_names.index(feature)]],
                    y=[self.model.predict([reference_point])[0]],
                    mode='markers',
                    name='Reference Point',
                    marker=dict(size=10, color='red')
                )
            )
        
        # Update layout
        fig.update_layout(
            title=f"What-If Analysis: Impact of {feature}",
            xaxis_title=feature,
            yaxis_title="Prediction",
            width=self.config.plot_width,
            height=self.config.plot_height
        )
        
        return fig 