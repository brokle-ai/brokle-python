"""
Configuration management for Brokle SDK.
"""

import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


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
    host: str = Field(default="http://localhost:8080", description="Brokle host URL")
    environment: str = Field(default="default", description="Environment name")

    # OpenTelemetry configuration
    otel_enabled: bool = Field(
        default=True, description="Enable OpenTelemetry integration"
    )
    otel_endpoint: Optional[str] = Field(
        default=None, description="OpenTelemetry endpoint"
    )
    otel_service_name: str = Field(
        default="brokle-sdk", description="OpenTelemetry service name"
    )
    otel_headers: Optional[Dict[str, str]] = Field(
        default=None, description="OpenTelemetry headers"
    )

    # Telemetry settings
    telemetry_enabled: bool = Field(
        default=True, description="Enable telemetry collection"
    )

    # Batch telemetry settings (unified /v1/telemetry/batch API)
    batch_max_size: int = Field(
        default=100, description="Maximum events per batch", ge=1, le=1000
    )
    batch_flush_interval: float = Field(
        default=5.0, description="Batch flush interval in seconds", ge=0.1, le=60.0
    )
    batch_enable_deduplication: bool = Field(
        default=True, description="Enable ULID-based event deduplication"
    )
    batch_deduplication_ttl: int = Field(
        default=3600, description="Deduplication cache TTL in seconds", ge=60, le=86400
    )
    batch_use_redis_cache: bool = Field(
        default=True, description="Use Redis for distributed deduplication"
    )
    batch_fail_on_duplicate: bool = Field(
        default=False, description="Fail entire batch on duplicate events"
    )

    # Debug settings
    debug: bool = Field(default=False, description="Enable debug logging")

    # HTTP settings
    timeout: int = Field(default=30, description="HTTP timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")

    # Feature flags
    cache_enabled: bool = Field(default=True, description="Enable caching")
    routing_enabled: bool = Field(
        default=True, description="Enable intelligent routing"
    )
    evaluation_enabled: bool = Field(default=True, description="Enable evaluation")

    # Debug settings
    debug: bool = Field(default=False, description="Enable debug mode")

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: Optional[str]) -> Optional[str]:
        """Validate API key format."""
        if v and not v.startswith("bk_"):
            raise ValueError('API key must start with "bk_"')
        return v

    @field_validator("host")
    @classmethod
    def validate_host(cls, v: str) -> str:
        """Validate host URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("Host must start with http:// or https://")
        return v.rstrip("/")

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment name according to rules."""
        validate_environment_name(v)
        return v

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        load_dotenv()

        return cls(
            api_key=os.getenv("BROKLE_API_KEY"),
            host=os.getenv("BROKLE_HOST", "http://localhost:8080"),
            environment=os.getenv("BROKLE_ENVIRONMENT", "default"),
            # OpenTelemetry
            otel_enabled=os.getenv("BROKLE_OTEL_ENABLED", "true").lower() == "true",
            otel_endpoint=os.getenv("BROKLE_OTEL_ENDPOINT"),
            otel_service_name=os.getenv("BROKLE_OTEL_SERVICE_NAME", "brokle-sdk"),
            # Telemetry
            telemetry_enabled=os.getenv("BROKLE_TELEMETRY_ENABLED", "true").lower()
            == "true",
            # Batch telemetry
            batch_max_size=int(os.getenv("BROKLE_BATCH_MAX_SIZE", "100")),
            batch_flush_interval=float(os.getenv("BROKLE_BATCH_FLUSH_INTERVAL", "5.0")),
            batch_enable_deduplication=os.getenv("BROKLE_BATCH_ENABLE_DEDUPLICATION", "true").lower() == "true",
            batch_deduplication_ttl=int(os.getenv("BROKLE_BATCH_DEDUPLICATION_TTL", "3600")),
            batch_use_redis_cache=os.getenv("BROKLE_BATCH_USE_REDIS_CACHE", "true").lower() == "true",
            batch_fail_on_duplicate=os.getenv("BROKLE_BATCH_FAIL_ON_DUPLICATE", "false").lower() == "true",
            # HTTP
            timeout=int(os.getenv("BROKLE_TIMEOUT", "30")),
            max_retries=int(os.getenv("BROKLE_MAX_RETRIES", "3")),
            # Features
            cache_enabled=os.getenv("BROKLE_CACHE_ENABLED", "true").lower() == "true",
            routing_enabled=os.getenv("BROKLE_ROUTING_ENABLED", "true").lower()
            == "true",
            evaluation_enabled=os.getenv("BROKLE_EVALUATION_ENABLED", "true").lower()
            == "true",
            # Debug
            debug=os.getenv("BROKLE_DEBUG", "false").lower() == "true",
        )

    def validate(self) -> None:
        """Validate configuration."""
        if not self.api_key:
            raise ValueError("API key is required")

    def get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"brokle-python/0.1.0",
        }

        if self.api_key:
            headers["X-API-Key"] = self.api_key

        headers["X-Environment"] = self.environment

        return headers


# Note: Global configuration functions (configure, get_config, reset_config) have been removed
# in favor of direct instantiation and environment variable fallback.
# Use Brokle(api_key=...) or get_client() instead.
