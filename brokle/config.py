"""
Configuration management for Brokle SDK.
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv


def validate_environment_name(env: str) -> None:
    """
    Validate environment name according to rules.

    Rules:
    - Must be lowercase
    - Maximum 40 characters
    - Cannot start with "brokle" prefix
    - Cannot be empty

    Args:
        env: Environment name to validate

    Raises:
        ValueError: If environment name is invalid
    """
    if not env:
        raise ValueError("Environment name cannot be empty")

    if len(env) > 40:
        raise ValueError(f"Environment name too long: {len(env)} characters (max 40)")

    if env != env.lower():
        raise ValueError("Environment name must be lowercase")

    if env.startswith("brokle"):
        raise ValueError("Environment name cannot start with 'brokle' prefix")


def sanitize_environment_name(env: str) -> str:
    """
    Sanitize environment name to follow rules.

    Args:
        env: Environment name to sanitize

    Returns:
        Sanitized environment name
    """
    if not env:
        return "default"

    # Convert to lowercase
    env = env.lower()

    # Truncate if too long
    if len(env) > 40:
        env = env[:40]

    # Validate that environment doesn't start with brokle prefix
    if env.startswith("brokle"):
        raise ValueError("Environment name cannot start with 'brokle' prefix")

    return env or "default"


class Config(BaseModel):
    """Configuration for Brokle SDK."""
    
    # Core configuration
    api_key: Optional[str] = Field(default=None, description="Brokle API key")
    host: str = Field(default="http://localhost:8000", description="Brokle host URL")
    project_id: Optional[str] = Field(default=None, description="Project ID")
    environment: str = Field(default="default", description="Environment name")
    
    # OpenTelemetry configuration
    otel_enabled: bool = Field(default=True, description="Enable OpenTelemetry integration")
    otel_endpoint: Optional[str] = Field(default=None, description="OpenTelemetry endpoint")
    otel_service_name: str = Field(default="brokle-sdk", description="OpenTelemetry service name")
    otel_headers: Optional[Dict[str, str]] = Field(default=None, description="OpenTelemetry headers")
    
    # Telemetry settings
    telemetry_enabled: bool = Field(default=True, description="Enable telemetry collection")
    telemetry_batch_size: int = Field(default=100, description="Telemetry batch size")
    telemetry_flush_interval: int = Field(default=10000, description="Telemetry flush interval (ms)")
    
    # HTTP settings
    timeout: int = Field(default=30, description="HTTP timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    
    # Feature flags
    cache_enabled: bool = Field(default=True, description="Enable caching")
    routing_enabled: bool = Field(default=True, description="Enable intelligent routing")
    evaluation_enabled: bool = Field(default=True, description="Enable evaluation")
    
    # Debug settings
    debug: bool = Field(default=False, description="Enable debug mode")
    
    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v: Optional[str]) -> Optional[str]:
        """Validate API key format."""
        if v and not v.startswith('ak_'):
            raise ValueError('API key must start with "ak_"')
        return v
    
    @field_validator('host')
    @classmethod
    def validate_host(cls, v: str) -> str:
        """Validate host URL format."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Host must start with http:// or https://')
        return v.rstrip('/')

    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment name according to rules."""
        validate_environment_name(v)
        return v
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        load_dotenv()
        
        return cls(
            api_key=os.getenv('BROKLE_API_KEY'),
            host=os.getenv('BROKLE_HOST', 'http://localhost:8000'),
            project_id=os.getenv('BROKLE_PROJECT_ID'),
            environment=os.getenv('BROKLE_ENVIRONMENT', 'default'),
            
            # OpenTelemetry
            otel_enabled=os.getenv('BROKLE_OTEL_ENABLED', 'true').lower() == 'true',
            otel_endpoint=os.getenv('BROKLE_OTEL_ENDPOINT'),
            otel_service_name=os.getenv('BROKLE_OTEL_SERVICE_NAME', 'brokle-sdk'),
            
            # Telemetry
            telemetry_enabled=os.getenv('BROKLE_TELEMETRY_ENABLED', 'true').lower() == 'true',
            telemetry_batch_size=int(os.getenv('BROKLE_TELEMETRY_BATCH_SIZE', '100')),
            telemetry_flush_interval=int(os.getenv('BROKLE_TELEMETRY_FLUSH_INTERVAL', '10000')),
            
            # HTTP
            timeout=int(os.getenv('BROKLE_TIMEOUT', '30')),
            max_retries=int(os.getenv('BROKLE_MAX_RETRIES', '3')),
            
            # Features
            cache_enabled=os.getenv('BROKLE_CACHE_ENABLED', 'true').lower() == 'true',
            routing_enabled=os.getenv('BROKLE_ROUTING_ENABLED', 'true').lower() == 'true',
            evaluation_enabled=os.getenv('BROKLE_EVALUATION_ENABLED', 'true').lower() == 'true',
            
            # Debug
            debug=os.getenv('BROKLE_DEBUG', 'false').lower() == 'true',
        )
    
    def validate(self) -> None:
        """Validate configuration."""
        if not self.api_key:
            raise ValueError("API key is required")
        if not self.project_id:
            raise ValueError("Project ID is required")
    
    def get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': f'brokle-python/0.1.0',
        }
        
        if self.api_key:
            headers['X-API-Key'] = self.api_key
        
        if self.project_id:
            headers['X-Project-ID'] = self.project_id
        
        headers['X-Environment'] = self.environment
        
        return headers


# Global configuration instance
_global_config: Optional[Config] = None


def configure(
    api_key: Optional[str] = None,
    host: Optional[str] = None,
    project_id: Optional[str] = None,
    environment: Optional[str] = None,
    **kwargs: Any
) -> Config:
    """Configure Brokle SDK globally."""
    global _global_config
    
    if _global_config is None:
        _global_config = Config.from_env()
    
    # Update with provided values
    if api_key is not None:
        _global_config.api_key = api_key
    if host is not None:
        _global_config.host = host
    if project_id is not None:
        _global_config.project_id = project_id
    if environment is not None:
        _global_config.environment = environment
    
    # Update other kwargs
    for key, value in kwargs.items():
        if hasattr(_global_config, key):
            setattr(_global_config, key, value)
    
    return _global_config


def get_config() -> Config:
    """Get current configuration."""
    global _global_config
    
    if _global_config is None:
        _global_config = Config.from_env()
    
    return _global_config


def reset_config() -> None:
    """Reset configuration to defaults."""
    global _global_config
    _global_config = None


def is_configured() -> bool:
    """Check if SDK is configured."""
    config = get_config()
    try:
        config.validate()
        return True
    except ValueError:
        return False