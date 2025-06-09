"""
Responsible AI Manager - Main orchestration layer for the RAI framework.

This module provides a unified interface to all RAI components including:
- Bias detection and mitigation
- Fairness metrics and monitoring
- Privacy protection
- Model explainability
- Governance and compliance
"""

import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import pandas as pd
from datetime import datetime

from ..bias_detection.bias_detector import BiasDetector
from ..fairness_metrics.fairness_calculator import FairnessCalculator
from ..privacy_protection.privacy_protector import PrivacyProtector
from ..explainability.shap_explainer import ShapExplainer

@dataclass
class RAIConfig:
    """Configuration for Responsible AI components."""
    sensitive_attributes: List[str]
    target_column: str
    prediction_column: Optional[str] = None
    quasi_identifiers: Optional[List[str]] = None
    privacy_epsilon: float = 1.0
    privacy_delta: float = 1e-5
    fairness_threshold: float = 0.8
    background_samples: int = 100
    output_path: str = "rai_outputs"

class ResponsibleAIManager:
    """Main class for orchestrating Responsible AI components."""
    
    def __init__(
        self,
        config: RAIConfig,
        model: Any = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize ResponsibleAIManager.
        
        Args:
            config: RAIConfig object containing configuration
            model: Trained model to analyze (optional)
            logger: Logger instance (optional)
        """
        self.config = config
        self.model = model
        self.logger = logger or self._setup_logger()
        
        # Initialize components
        self.bias_detector = BiasDetector(
            sensitive_features=config.sensitive_attributes,
            target_column=config.target_column,
            prediction_column=config.prediction_column
        )
        
        self.fairness_calculator = FairnessCalculator(
            protected_attributes=config.sensitive_attributes,
            target_column=config.target_column,
            prediction_column=config.prediction_column,
            threshold=config.fairness_threshold
        )
        
        self.privacy_protector = PrivacyProtector(
            epsilon=config.privacy_epsilon,
            delta=config.privacy_delta,
            sensitive_attributes=config.sensitive_attributes,
            quasi_identifiers=config.quasi_identifiers
        )
        
        if model is not None:
            self.explainer = ShapExplainer(
                model=model,
                feature_names=config.sensitive_attributes + 
                            (config.quasi_identifiers or []),
                output_path=config.output_path,
                background_samples=config.background_samples
            )
        else:
            self.explainer = None
        
        self.logger.info("ResponsibleAIManager initialized successfully")
    
    def analyze_data(
        self,
        data: pd.DataFrame,
        generate_report: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive RAI analysis on data.
        
        Args:
            data: DataFrame to analyze
            generate_report: Whether to generate a comprehensive report
            
        Returns:
            Dictionary containing analysis results
        """
        self.logger.info("Starting comprehensive RAI analysis")
        results = {}
        
        try:
            # Bias detection
            self.logger.info("Running bias detection")
            bias_report = self.bias_detector.generate_bias_report(data)
            results['bias_analysis'] = bias_report
            
            # Fairness metrics
            self.logger.info("Calculating fairness metrics")
            fairness_report = self.fairness_calculator.generate_fairness_report(data)
            results['fairness_analysis'] = fairness_report
            
            # Privacy assessment
            self.logger.info("Assessing privacy risks")
            privacy_metrics = self.privacy_protector.assess_privacy_risks(data)
            privacy_report = self.privacy_protector.generate_privacy_report(
                data,
                privacy_metrics
            )
            results['privacy_analysis'] = privacy_report
            
            # Model explainability
            if self.explainer is not None and self.model is not None:
                self.logger.info("Generating model explanations")
                self.explainer.fit(data)
                explanation = self.explainer.explain_predictions(data)
                explanation_report = self.explainer.generate_explanation_report(
                    explanation,
                    data
                )
                results['explainability_analysis'] = explanation_report
            
            if generate_report:
                report = self._generate_comprehensive_report(results)
                results['comprehensive_report'] = report
            
            self.logger.info("RAI analysis completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during RAI analysis: {str(e)}")
            raise
        
        return results
    
    def mitigate_issues(
        self,
        data: pd.DataFrame,
        mitigation_config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Apply automated mitigation strategies.
        
        Args:
            data: DataFrame to apply mitigations to
            mitigation_config: Configuration for mitigation strategies
            
        Returns:
            DataFrame with mitigations applied
        """
        self.logger.info("Starting automated mitigation")
        mitigated_data = data.copy()
        
        try:
            # Apply privacy protections if configured
            if mitigation_config.get('apply_privacy', False):
                self.logger.info("Applying privacy protections")
                mitigated_data = self.privacy_protector.anonymize_data(
                    mitigated_data,
                    method=mitigation_config.get('privacy_method', 'differential_privacy')
                )
            
            # Log mitigation results
            self.logger.info("Mitigation completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during mitigation: {str(e)}")
            raise
        
        return mitigated_data
    
    def monitor_metrics(
        self,
        data: pd.DataFrame,
        threshold_config: Dict[str, float]
    ) -> Dict[str, bool]:
        """
        Monitor RAI metrics against thresholds.
        
        Args:
            data: DataFrame to monitor
            threshold_config: Configuration of metric thresholds
            
        Returns:
            Dictionary of threshold violations
        """
        self.logger.info("Starting metrics monitoring")
        violations = {}
        
        try:
            # Check bias metrics
            bias_report = self.bias_detector.generate_bias_report(data)
            bias_threshold = threshold_config.get('bias_threshold', 0.8)
            violations['bias'] = any(
                metric < bias_threshold
                for metric in bias_report['bias_metrics'].__dict__.values()
                if isinstance(metric, (int, float))
            )
            
            # Check fairness metrics
            fairness_report = self.fairness_calculator.generate_fairness_report(data)
            fairness_threshold = threshold_config.get('fairness_threshold', 0.8)
            violations['fairness'] = any(
                metric < fairness_threshold
                for metric in fairness_report['overall_metrics'].values()
            )
            
            # Check privacy metrics
            privacy_metrics = self.privacy_protector.assess_privacy_risks(data)
            privacy_threshold = threshold_config.get('privacy_threshold', 0.7)
            violations['privacy'] = (
                privacy_metrics.privacy_risk_score > privacy_threshold
            )
            
            self.logger.info("Metrics monitoring completed")
            
        except Exception as e:
            self.logger.error(f"Error during metrics monitoring: {str(e)}")
            raise
        
        return violations
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('ResponsibleAI')
        logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('responsible_ai.log')
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        
        return logger
    
    def _generate_comprehensive_report(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive RAI report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'bias_detected': any(
                    metric < self.config.fairness_threshold
                    for metric in results['bias_analysis']['bias_metrics'].__dict__.values()
                    if isinstance(metric, (int, float))
                ),
                'fairness_issues': any(
                    metric < self.config.fairness_threshold
                    for metric in results['fairness_analysis']['overall_metrics'].values()
                ),
                'privacy_risk_level': results['privacy_analysis']['privacy_metrics']['privacy_risk_score'],
                'model_explained': 'explainability_analysis' in results
            },
            'detailed_results': results,
            'recommendations': self._compile_recommendations(results)
        }
        
        return report
    
    def _compile_recommendations(
        self,
        results: Dict[str, Any]
    ) -> List[str]:
        """Compile recommendations from all analyses."""
        recommendations = []
        
        # Add bias recommendations
        if 'bias_analysis' in results:
            recommendations.extend(results['bias_analysis'].get('recommendations', []))
        
        # Add fairness recommendations
        if 'fairness_analysis' in results:
            recommendations.extend(results['fairness_analysis'].get('recommendations', []))
        
        # Add privacy recommendations
        if 'privacy_analysis' in results:
            recommendations.extend(results['privacy_analysis'].get('recommendations', []))
        
        # Add explainability recommendations
        if 'explainability_analysis' in results:
            recommendations.extend(
                results['explainability_analysis'].get('recommendations', [])
            )
        
        return recommendations 