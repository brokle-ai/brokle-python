"""
Configuration management for Brokle SDK.
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv


class Config(BaseModel):
    """Configuration for Brokle SDK."""
    
    # Core configuration
    public_key: Optional[str] = Field(default=None, description="Brokle public key")
    host: str = Field(default="http://localhost:8000", description="Brokle host URL")
    secret_key: Optional[str] = Field(default=None, description="Brokle secret key")
    environment: str = Field(default="production", description="Environment name")
    
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
    
    @field_validator('public_key')
    @classmethod
    def validate_public_key(cls, v: Optional[str]) -> Optional[str]:
        """Validate public key format."""
        if v and not v.startswith('pk_'):
            raise ValueError('Public key must start with "pk_"')
        return v
    
    @field_validator('host')
    @classmethod
    def validate_host(cls, v: str) -> str:
        """Validate host URL format."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Host must start with http:// or https://')
        return v.rstrip('/')
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        load_dotenv()
        
        return cls(
            public_key=os.getenv('BROKLE_PUBLIC_KEY'),
            host=os.getenv('BROKLE_HOST', 'http://localhost:8000'),
            secret_key=os.getenv('BROKLE_SECRET_KEY'),
            environment=os.getenv('BROKLE_ENVIRONMENT', 'production'),
            
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
        if not self.public_key:
            raise ValueError("Public key is required")
        if not self.secret_key:
            raise ValueError("Secret key is required")
    
    def get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': f'brokle-python/0.1.0',
        }
        
        if self.public_key:
            headers['X-Public-Key'] = self.public_key

        if self.secret_key:
            headers['X-Secret-Key'] = self.secret_key
        
        headers['X-Environment'] = self.environment
        
        return headers


# Global configuration instance
_global_config: Optional[Config] = None


def configure(
    public_key: Optional[str] = None,
    host: Optional[str] = None,
    secret_key: Optional[str] = None,
    environment: Optional[str] = None,
    **kwargs: Any
) -> Config:
    """Configure Brokle SDK globally."""
    global _global_config
    
    if _global_config is None:
        _global_config = Config.from_env()
    
    # Update with provided values
    if public_key is not None:
        _global_config.public_key = public_key
    if host is not None:
        _global_config.host = host
    if secret_key is not None:
        _global_config.secret_key = secret_key
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