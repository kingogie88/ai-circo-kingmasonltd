"""
Configuration Management Module for Responsible AI Framework.

This module provides configuration management capabilities including:
- Loading configurations from files
- Validating configurations
- Managing different configuration profiles
- Environment-specific settings
"""

import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from dataclasses import asdict
from .responsible_ai_manager import RAIConfig

class ConfigurationError(Exception):
    """Custom exception for configuration errors."""
    pass

class ConfigManager:
    """Configuration management for the RAI framework."""
    
    def __init__(
        self,
        config_dir: str = "config",
        environment: str = "development"
    ):
        """
        Initialize ConfigManager.
        
        Args:
            config_dir: Directory containing configuration files
            environment: Current environment (development, staging, production)
        """
        self.config_dir = Path(config_dir)
        self.environment = environment
        self.logger = logging.getLogger('RAI.ConfigManager')
        
        # Create config directory if it doesn't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configuration templates
        self.default_configs = {
            'development': {
                'logging': {
                    'level': 'DEBUG',
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    'file': 'rai_dev.log'
                },
                'monitoring': {
                    'enabled': True,
                    'interval': 60,
                    'alert_threshold': 0.7
                },
                'privacy': {
                    'epsilon': 1.0,
                    'delta': 1e-5,
                    'method': 'differential_privacy'
                },
                'fairness': {
                    'threshold': 0.8,
                    'metrics': ['demographic_parity', 'equal_opportunity']
                },
                'explainability': {
                    'background_samples': 100,
                    'output_format': ['html', 'json']
                }
            },
            'production': {
                'logging': {
                    'level': 'INFO',
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    'file': 'rai_prod.log'
                },
                'monitoring': {
                    'enabled': True,
                    'interval': 300,
                    'alert_threshold': 0.8
                },
                'privacy': {
                    'epsilon': 0.1,
                    'delta': 1e-6,
                    'method': 'differential_privacy'
                },
                'fairness': {
                    'threshold': 0.9,
                    'metrics': ['demographic_parity', 'equal_opportunity']
                },
                'explainability': {
                    'background_samples': 1000,
                    'output_format': ['json']
                }
            }
        }
    
    def load_config(
        self,
        config_name: str,
        environment: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_name: Name of the configuration to load
            environment: Optional environment override
            
        Returns:
            Dictionary containing configuration
        """
        env = environment or self.environment
        config_file = self.config_dir / f"{config_name}_{env}.yaml"
        
        try:
            if config_file.exists():
                with open(config_file) as f:
                    config = yaml.safe_load(f)
            else:
                # Use default config if file doesn't exist
                config = self.default_configs.get(env, {})
                self.save_config(config_name, config, env)
            
            # Validate configuration
            self._validate_config(config)
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise ConfigurationError(f"Failed to load configuration: {str(e)}")
    
    def save_config(
        self,
        config_name: str,
        config: Dict[str, Any],
        environment: Optional[str] = None
    ) -> None:
        """
        Save configuration to file.
        
        Args:
            config_name: Name of the configuration
            config: Configuration dictionary to save
            environment: Optional environment override
        """
        env = environment or self.environment
        config_file = self.config_dir / f"{config_name}_{env}.yaml"
        
        try:
            # Validate configuration before saving
            self._validate_config(config)
            
            # Save configuration
            with open(config_file, 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=False)
            
            self.logger.info(f"Configuration saved to {config_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
            raise ConfigurationError(f"Failed to save configuration: {str(e)}")
    
    def create_rai_config(
        self,
        config: Dict[str, Any]
    ) -> RAIConfig:
        """
        Create RAIConfig object from configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            RAIConfig object
        """
        try:
            required_fields = [
                'sensitive_attributes',
                'target_column'
            ]
            
            # Check required fields
            for field in required_fields:
                if field not in config:
                    raise ConfigurationError(f"Missing required field: {field}")
            
            # Create RAIConfig object
            rai_config = RAIConfig(
                sensitive_attributes=config['sensitive_attributes'],
                target_column=config['target_column'],
                prediction_column=config.get('prediction_column'),
                quasi_identifiers=config.get('quasi_identifiers'),
                privacy_epsilon=config.get('privacy', {}).get('epsilon', 1.0),
                privacy_delta=config.get('privacy', {}).get('delta', 1e-5),
                fairness_threshold=config.get('fairness', {}).get('threshold', 0.8),
                background_samples=config.get('explainability', {}).get('background_samples', 100),
                output_path=config.get('output_path', 'rai_outputs')
            )
            
            return rai_config
            
        except Exception as e:
            self.logger.error(f"Error creating RAIConfig: {str(e)}")
            raise ConfigurationError(f"Failed to create RAIConfig: {str(e)}")
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        required_sections = ['logging', 'monitoring', 'privacy', 'fairness']
        
        # Check required sections
        for section in required_sections:
            if section not in config:
                raise ConfigurationError(f"Missing required section: {section}")
        
        # Validate logging configuration
        if 'logging' in config:
            if 'level' not in config['logging']:
                raise ConfigurationError("Missing logging level")
            if config['logging']['level'] not in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
                raise ConfigurationError("Invalid logging level")
        
        # Validate monitoring configuration
        if 'monitoring' in config:
            if 'enabled' not in config['monitoring']:
                raise ConfigurationError("Missing monitoring enabled flag")
            if not isinstance(config['monitoring']['interval'], (int, float)):
                raise ConfigurationError("Invalid monitoring interval")
        
        # Validate privacy configuration
        if 'privacy' in config:
            if 'epsilon' not in config['privacy']:
                raise ConfigurationError("Missing privacy epsilon")
            if not isinstance(config['privacy']['epsilon'], (int, float)):
                raise ConfigurationError("Invalid privacy epsilon")
        
        # Validate fairness configuration
        if 'fairness' in config:
            if 'threshold' not in config['fairness']:
                raise ConfigurationError("Missing fairness threshold")
            if not isinstance(config['fairness']['threshold'], (int, float)):
                raise ConfigurationError("Invalid fairness threshold")
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get default configuration for current environment."""
        return self.default_configs.get(self.environment, {})
    
    def merge_configs(
        self,
        base_config: Dict[str, Any],
        override_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge two configurations, with override taking precedence.
        
        Args:
            base_config: Base configuration
            override_config: Configuration to override with
            
        Returns:
            Merged configuration dictionary
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if (
                key in merged and
                isinstance(merged[key], dict) and
                isinstance(value, dict)
            ):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged 