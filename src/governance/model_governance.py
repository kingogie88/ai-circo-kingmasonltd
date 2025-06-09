"""
Model Governance Module for Responsible AI Framework.

This module provides comprehensive model governance capabilities including:
- Model versioning and lifecycle management
- Compliance tracking
- Audit trails
- Approval workflows
- Policy enforcement
"""

import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import json
from pathlib import Path
import hashlib
from dataclasses import dataclass
import yaml
import joblib

@dataclass
class ModelMetadata:
    """Container for model metadata."""
    model_id: str
    version: str
    name: str
    description: str
    created_at: datetime
    created_by: str
    framework: str
    type: str
    input_features: List[str]
    target_features: List[str]
    performance_metrics: Dict[str, float]
    fairness_metrics: Dict[str, float]
    privacy_metrics: Dict[str, float]
    approval_status: str
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None

@dataclass
class AuditLog:
    """Container for audit log entries."""
    timestamp: datetime
    action: str
    user: str
    model_id: str
    version: str
    details: Dict[str, Any]

class ModelGovernance:
    """Main class for model governance and compliance."""
    
    def __init__(
        self,
        storage_path: str = "governance",
        config_path: Optional[str] = None
    ):
        """
        Initialize ModelGovernance.
        
        Args:
            storage_path: Path to store governance data
            config_path: Path to governance configuration file
        """
        self.storage_path = Path(storage_path)
        self.logger = logging.getLogger('RAI.ModelGovernance')
        
        # Create storage directories
        self.models_path = self.storage_path / 'models'
        self.audit_path = self.storage_path / 'audit'
        self.policy_path = self.storage_path / 'policies'
        
        for path in [self.models_path, self.audit_path, self.policy_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        self.logger.info("ModelGovernance initialized successfully")
    
    def register_model(
        self,
        model: Any,
        metadata: ModelMetadata,
        save_model: bool = True
    ) -> str:
        """
        Register a new model version.
        
        Args:
            model: Model object to register
            metadata: ModelMetadata object
            save_model: Whether to save model artifact
            
        Returns:
            Model ID
        """
        try:
            # Generate model ID if not provided
            if not metadata.model_id:
                metadata.model_id = self._generate_model_id(metadata)
            
            # Validate metadata
            self._validate_metadata(metadata)
            
            # Save model if requested
            if save_model:
                model_path = self.models_path / f"{metadata.model_id}_{metadata.version}"
                joblib.dump(model, model_path)
            
            # Save metadata
            self._save_metadata(metadata)
            
            # Log registration
            self._add_audit_log(
                AuditLog(
                    timestamp=datetime.now(),
                    action="register_model",
                    user=metadata.created_by,
                    model_id=metadata.model_id,
                    version=metadata.version,
                    details={
                        "name": metadata.name,
                        "framework": metadata.framework,
                        "type": metadata.type
                    }
                )
            )
            
            self.logger.info(
                f"Registered model {metadata.name} "
                f"version {metadata.version}"
            )
            
            return metadata.model_id
            
        except Exception as e:
            self.logger.error(f"Error registering model: {str(e)}")
            raise
    
    def approve_model(
        self,
        model_id: str,
        version: str,
        approver: str,
        approval_notes: Optional[str] = None
    ) -> None:
        """
        Approve a model version for deployment.
        
        Args:
            model_id: Model ID
            version: Model version
            approver: Name of approver
            approval_notes: Optional approval notes
        """
        try:
            # Load metadata
            metadata = self._load_metadata(model_id, version)
            
            # Check if model can be approved
            if metadata.approval_status == 'approved':
                raise ValueError("Model is already approved")
            
            # Update approval status
            metadata.approval_status = 'approved'
            metadata.approved_by = approver
            metadata.approved_at = datetime.now()
            
            # Save updated metadata
            self._save_metadata(metadata)
            
            # Log approval
            self._add_audit_log(
                AuditLog(
                    timestamp=datetime.now(),
                    action="approve_model",
                    user=approver,
                    model_id=model_id,
                    version=version,
                    details={"notes": approval_notes}
                )
            )
            
            self.logger.info(
                f"Approved model {model_id} version {version}"
            )
            
        except Exception as e:
            self.logger.error(f"Error approving model: {str(e)}")
            raise
    
    def get_model_history(
        self,
        model_id: str
    ) -> List[ModelMetadata]:
        """
        Get version history for a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            List of ModelMetadata objects
        """
        try:
            metadata_files = list(self.models_path.glob(f"{model_id}_*.yaml"))
            history = []
            
            for file in metadata_files:
                with open(file) as f:
                    metadata_dict = yaml.safe_load(f)
                    metadata = ModelMetadata(**metadata_dict)
                    history.append(metadata)
            
            return sorted(history, key=lambda x: x.version)
            
        except Exception as e:
            self.logger.error(f"Error getting model history: {str(e)}")
            raise
    
    def get_audit_trail(
        self,
        model_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[AuditLog]:
        """
        Get audit trail for models.
        
        Args:
            model_id: Optional model ID to filter by
            start_time: Optional start time for filtering
            end_time: Optional end time for filtering
            
        Returns:
            List of AuditLog objects
        """
        try:
            audit_files = list(self.audit_path.glob("*.json"))
            audit_logs = []
            
            for file in audit_files:
                with open(file) as f:
                    logs = json.load(f)
                    for log in logs:
                        log['timestamp'] = datetime.fromisoformat(log['timestamp'])
                        audit_log = AuditLog(**log)
                        
                        # Apply filters
                        if model_id and audit_log.model_id != model_id:
                            continue
                        if start_time and audit_log.timestamp < start_time:
                            continue
                        if end_time and audit_log.timestamp > end_time:
                            continue
                        
                        audit_logs.append(audit_log)
            
            return sorted(audit_logs, key=lambda x: x.timestamp)
            
        except Exception as e:
            self.logger.error(f"Error getting audit trail: {str(e)}")
            raise
    
    def check_compliance(
        self,
        metadata: ModelMetadata
    ) -> Dict[str, bool]:
        """
        Check model compliance against policies.
        
        Args:
            metadata: ModelMetadata object
            
        Returns:
            Dictionary of compliance check results
        """
        try:
            compliance_results = {}
            
            # Load policies
            policies = self._load_policies()
            
            # Check performance requirements
            if 'performance' in policies:
                performance_compliant = all(
                    metadata.performance_metrics.get(metric, 0) >= threshold
                    for metric, threshold in policies['performance'].items()
                )
                compliance_results['performance'] = performance_compliant
            
            # Check fairness requirements
            if 'fairness' in policies:
                fairness_compliant = all(
                    metadata.fairness_metrics.get(metric, 0) >= threshold
                    for metric, threshold in policies['fairness'].items()
                )
                compliance_results['fairness'] = fairness_compliant
            
            # Check privacy requirements
            if 'privacy' in policies:
                privacy_compliant = all(
                    metadata.privacy_metrics.get(metric, 0) >= threshold
                    for metric, threshold in policies['privacy'].items()
                )
                compliance_results['privacy'] = privacy_compliant
            
            return compliance_results
            
        except Exception as e:
            self.logger.error(f"Error checking compliance: {str(e)}")
            raise
    
    def _generate_model_id(self, metadata: ModelMetadata) -> str:
        """Generate unique model ID."""
        base = f"{metadata.name}_{metadata.framework}_{metadata.type}"
        return hashlib.md5(base.encode()).hexdigest()[:8]
    
    def _validate_metadata(self, metadata: ModelMetadata) -> None:
        """Validate model metadata."""
        required_fields = [
            'name',
            'version',
            'framework',
            'type',
            'input_features',
            'target_features'
        ]
        
        for field in required_fields:
            if not getattr(metadata, field):
                raise ValueError(f"Missing required metadata field: {field}")
    
    def _save_metadata(self, metadata: ModelMetadata) -> None:
        """Save model metadata to file."""
        metadata_file = (
            self.models_path /
            f"{metadata.model_id}_{metadata.version}.yaml"
        )
        
        with open(metadata_file, 'w') as f:
            yaml.safe_dump(metadata.__dict__, f)
    
    def _load_metadata(
        self,
        model_id: str,
        version: str
    ) -> ModelMetadata:
        """Load model metadata from file."""
        metadata_file = self.models_path / f"{model_id}_{version}.yaml"
        
        if not metadata_file.exists():
            raise ValueError(f"Model {model_id} version {version} not found")
        
        with open(metadata_file) as f:
            metadata_dict = yaml.safe_load(f)
            return ModelMetadata(**metadata_dict)
    
    def _add_audit_log(self, log: AuditLog) -> None:
        """Add entry to audit log."""
        log_file = self.audit_path / f"{log.timestamp.date()}.json"
        
        # Load existing logs
        if log_file.exists():
            with open(log_file) as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Add new log
        log_dict = {
            'timestamp': log.timestamp.isoformat(),
            'action': log.action,
            'user': log.user,
            'model_id': log.model_id,
            'version': log.version,
            'details': log.details
        }
        logs.append(log_dict)
        
        # Save updated logs
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def _load_config(
        self,
        config_path: Optional[str]
    ) -> Dict[str, Any]:
        """Load governance configuration."""
        if config_path:
            with open(config_path) as f:
                return yaml.safe_load(f)
        
        return {
            'approval_required': True,
            'version_format': 'semver',
            'audit_retention_days': 365
        }
    
    def _load_policies(self) -> Dict[str, Any]:
        """Load compliance policies."""
        policy_file = self.policy_path / 'policies.yaml'
        
        if policy_file.exists():
            with open(policy_file) as f:
                return yaml.safe_load(f)
        
        return {
            'performance': {
                'accuracy': 0.8,
                'f1_score': 0.7
            },
            'fairness': {
                'demographic_parity': 0.8,
                'equal_opportunity': 0.8
            },
            'privacy': {
                'epsilon': 1.0
            }
        } 