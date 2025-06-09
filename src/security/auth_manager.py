"""
Authentication and Authorization Module for Responsible AI Framework.

This module provides comprehensive security capabilities including:
- User authentication
- Role-based access control
- Permission management
- Security audit logging
- Token management
"""

import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import json
from pathlib import Path
import hashlib
import jwt
from dataclasses import dataclass
import yaml
import secrets
import bcrypt

@dataclass
class User:
    """Container for user information."""
    username: str
    email: str
    role: str
    permissions: List[str]
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True

@dataclass
class Role:
    """Container for role information."""
    name: str
    description: str
    permissions: List[str]
    created_at: datetime

class AuthManager:
    """Main class for authentication and authorization management."""
    
    def __init__(
        self,
        storage_path: str = "security",
        config_path: Optional[str] = None,
        token_expiry: int = 24  # hours
    ):
        """
        Initialize AuthManager.
        
        Args:
            storage_path: Path to store security data
            config_path: Path to security configuration file
            token_expiry: Token expiry time in hours
        """
        self.storage_path = Path(storage_path)
        self.logger = logging.getLogger('RAI.AuthManager')
        self.token_expiry = token_expiry
        
        # Create storage directories
        self.users_path = self.storage_path / 'users'
        self.roles_path = self.storage_path / 'roles'
        self.audit_path = self.storage_path / 'audit'
        
        for path in [self.users_path, self.roles_path, self.audit_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize secret key
        self.secret_key = self._load_or_generate_secret()
        
        self.logger.info("AuthManager initialized successfully")
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: str
    ) -> User:
        """
        Create a new user.
        
        Args:
            username: Username
            email: Email address
            password: Password
            role: Role name
            
        Returns:
            Created User object
        """
        try:
            # Check if user exists
            if self._user_exists(username):
                raise ValueError(f"User {username} already exists")
            
            # Validate role
            if not self._role_exists(role):
                raise ValueError(f"Role {role} does not exist")
            
            # Get role permissions
            role_data = self._load_role(role)
            
            # Create user object
            user = User(
                username=username,
                email=email,
                role=role,
                permissions=role_data.permissions,
                created_at=datetime.now()
            )
            
            # Hash password
            password_hash = bcrypt.hashpw(
                password.encode(),
                bcrypt.gensalt()
            )
            
            # Save user data
            self._save_user(user, password_hash)
            
            # Log creation
            self._add_security_log(
                "create_user",
                username,
                details={"role": role}
            )
            
            return user
            
        except Exception as e:
            self.logger.error(f"Error creating user: {str(e)}")
            raise
    
    def authenticate(
        self,
        username: str,
        password: str
    ) -> Optional[str]:
        """
        Authenticate user and return token.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            JWT token if authentication successful, None otherwise
        """
        try:
            # Load user data
            user_file = self.users_path / f"{username}.yaml"
            if not user_file.exists():
                return None
            
            with open(user_file) as f:
                user_data = yaml.safe_load(f)
            
            # Verify password
            if not bcrypt.checkpw(
                password.encode(),
                user_data['password_hash'].encode()
            ):
                return None
            
            # Create user object
            user = User(**{
                k: v for k, v in user_data.items()
                if k != 'password_hash'
            })
            
            # Update last login
            user.last_login = datetime.now()
            self._save_user(user, user_data['password_hash'])
            
            # Generate token
            token = self._generate_token(user)
            
            # Log authentication
            self._add_security_log(
                "authenticate",
                username,
                details={"success": True}
            )
            
            return token
            
        except Exception as e:
            self.logger.error(f"Error during authentication: {str(e)}")
            return None
    
    def verify_token(self, token: str) -> Optional[User]:
        """
        Verify JWT token and return user.
        
        Args:
            token: JWT token
            
        Returns:
            User object if token valid, None otherwise
        """
        try:
            # Decode token
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=['HS256']
            )
            
            # Load user
            username = payload['sub']
            user_file = self.users_path / f"{username}.yaml"
            
            if not user_file.exists():
                return None
            
            with open(user_file) as f:
                user_data = yaml.safe_load(f)
            
            return User(**{
                k: v for k, v in user_data.items()
                if k != 'password_hash'
            })
            
        except jwt.InvalidTokenError:
            return None
        except Exception as e:
            self.logger.error(f"Error verifying token: {str(e)}")
            return None
    
    def check_permission(
        self,
        user: User,
        permission: str
    ) -> bool:
        """
        Check if user has permission.
        
        Args:
            user: User object
            permission: Permission to check
            
        Returns:
            True if user has permission, False otherwise
        """
        return permission in user.permissions
    
    def create_role(
        self,
        name: str,
        description: str,
        permissions: List[str]
    ) -> Role:
        """
        Create a new role.
        
        Args:
            name: Role name
            description: Role description
            permissions: List of permissions
            
        Returns:
            Created Role object
        """
        try:
            # Check if role exists
            if self._role_exists(name):
                raise ValueError(f"Role {name} already exists")
            
            # Create role object
            role = Role(
                name=name,
                description=description,
                permissions=permissions,
                created_at=datetime.now()
            )
            
            # Save role
            self._save_role(role)
            
            # Log creation
            self._add_security_log(
                "create_role",
                name,
                details={"permissions": permissions}
            )
            
            return role
            
        except Exception as e:
            self.logger.error(f"Error creating role: {str(e)}")
            raise
    
    def get_user_roles(self) -> Dict[str, Role]:
        """
        Get all roles.
        
        Returns:
            Dictionary of role name to Role object
        """
        try:
            roles = {}
            role_files = self.roles_path.glob("*.yaml")
            
            for file in role_files:
                with open(file) as f:
                    role_data = yaml.safe_load(f)
                    roles[role_data['name']] = Role(**role_data)
            
            return roles
            
        except Exception as e:
            self.logger.error(f"Error getting roles: {str(e)}")
            raise
    
    def _generate_token(self, user: User) -> str:
        """Generate JWT token for user."""
        payload = {
            'sub': user.username,
            'role': user.role,
            'permissions': user.permissions,
            'exp': datetime.utcnow() + timedelta(hours=self.token_expiry)
        }
        
        return jwt.encode(
            payload,
            self.secret_key,
            algorithm='HS256'
        )
    
    def _save_user(self, user: User, password_hash: bytes) -> None:
        """Save user data to file."""
        user_file = self.users_path / f"{user.username}.yaml"
        
        user_data = user.__dict__
        user_data['password_hash'] = password_hash.decode()
        
        with open(user_file, 'w') as f:
            yaml.safe_dump(user_data, f)
    
    def _save_role(self, role: Role) -> None:
        """Save role data to file."""
        role_file = self.roles_path / f"{role.name}.yaml"
        
        with open(role_file, 'w') as f:
            yaml.safe_dump(role.__dict__, f)
    
    def _load_role(self, name: str) -> Role:
        """Load role data from file."""
        role_file = self.roles_path / f"{name}.yaml"
        
        if not role_file.exists():
            raise ValueError(f"Role {name} not found")
        
        with open(role_file) as f:
            role_data = yaml.safe_load(f)
            return Role(**role_data)
    
    def _user_exists(self, username: str) -> bool:
        """Check if user exists."""
        return (self.users_path / f"{username}.yaml").exists()
    
    def _role_exists(self, name: str) -> bool:
        """Check if role exists."""
        return (self.roles_path / f"{name}.yaml").exists()
    
    def _load_or_generate_secret(self) -> str:
        """Load or generate secret key."""
        secret_file = self.storage_path / 'secret.key'
        
        if secret_file.exists():
            with open(secret_file) as f:
                return f.read().strip()
        
        # Generate new secret
        secret = secrets.token_hex(32)
        
        with open(secret_file, 'w') as f:
            f.write(secret)
        
        return secret
    
    def _load_config(
        self,
        config_path: Optional[str]
    ) -> Dict[str, Any]:
        """Load security configuration."""
        if config_path:
            with open(config_path) as f:
                return yaml.safe_load(f)
        
        return {
            'password_min_length': 8,
            'password_require_special': True,
            'password_require_numbers': True,
            'max_login_attempts': 3,
            'lockout_duration': 30  # minutes
        }
    
    def _add_security_log(
        self,
        action: str,
        user: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add security audit log entry."""
        log_file = self.audit_path / f"{datetime.now().date()}.json"
        
        # Load existing logs
        if log_file.exists():
            with open(log_file) as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Add new log
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'user': user,
            'details': details or {}
        }
        logs.append(log_entry)
        
        # Save updated logs
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2) 