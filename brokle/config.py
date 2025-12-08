"""
Configuration management for Brokle OpenTelemetry SDK.

Supports both programmatic configuration and environment variable-based configuration
following the 12-factor app pattern.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class BrokleConfig:
    """
    Configuration for Brokle OpenTelemetry SDK.

    All parameters can be set programmatically or via environment variables.
    Environment variables take precedence over default values but not over
    explicit programmatic configuration.
    """

    # ========== Required Configuration ==========
    api_key: str
    """Brokle API key (required, must start with 'bk_')"""

    # ========== Connection Configuration ==========
    base_url: str = "http://localhost:8080"
    """Brokle API base URL"""

    timeout: int = 30
    """HTTP request timeout in seconds"""

    # ========== Project Configuration ==========
    environment: str = "default"
    """Environment tag (e.g., 'production', 'staging', 'development')"""

    release: Optional[str] = None
    """Release identifier for deployment tracking (e.g., 'v2.1.24', git commit hash)"""

    version: Optional[str] = None
    """Trace-level version for A/B testing experiments (e.g., 'experiment-A', 'control')"""

    # ========== Tracing Control ==========
    tracing_enabled: bool = True
    """Enable/disable tracing (if False, all calls become no-ops)"""

    # ========== Metrics Configuration ==========
    metrics_enabled: bool = True
    """Enable/disable metrics collection (if False, no metrics are recorded)"""

    metrics_export_interval: float = 60.0
    """Interval in seconds between metric exports (default: 60s)"""

    # ========== Logs Configuration ==========
    logs_enabled: bool = False
    """Enable/disable OTLP log export (opt-in, default: False). Set to True to enable."""

    sample_rate: float = 1.0
    """
    Sampling rate for traces (0.0 to 1.0, default: 1.0 = 100%).

    Uses OpenTelemetry's TraceIdRatioBased sampler to ensure entire traces
    are sampled together (not individual spans). The decision is deterministic
    based on trace_id hash.

    Example: sample_rate=0.1 means ~10% of traces will be sampled (all spans
    in those traces). The same trace_id always produces the same decision.
    """

    debug: bool = False
    """Enable debug logging"""

    # ========== Privacy & Masking ==========
    mask: Optional[Callable[[Any], Any]] = None
    """Optional function to mask sensitive data before sending to backend"""

    # ========== Batch Configuration ==========
    flush_at: int = 100
    """Maximum batch size before flush (1-1000, default: 100)"""

    flush_interval: float = 5.0
    """Maximum delay in seconds before flush (0.1-60.0, default: 5.0)"""

    max_queue_size: int = 2048
    """Maximum queue size for pending spans"""

    export_timeout: int = 30000
    """Export timeout in milliseconds (default: 30000 = 30s)"""

    # ========== OTLP Export Configuration ==========
    use_protobuf: bool = True
    """Use Protobuf format for OTLP export (True) or JSON (False)"""

    compression: Optional[str] = "gzip"
    """Compression algorithm: 'gzip', 'deflate', or None"""

    # ========== Transport Configuration ==========
    transport: str = "http"
    """Transport protocol: 'http' (default) or 'grpc' (requires grpc extras)"""

    grpc_endpoint: Optional[str] = None
    """Explicit gRPC endpoint (default: derived from base_url with port 4317)"""

    # ========== Internal ==========
    _validated: bool = field(default=False, init=False, repr=False)
    """Internal flag to track validation status"""

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
        self._validated = True

    def validate(self):
        """
        Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate API key
        if not self.api_key:
            raise ValueError("api_key is required")
        if not self.api_key.startswith("bk_"):
            raise ValueError("api_key must start with 'bk_'")
        if len(self.api_key) < 10:
            raise ValueError("api_key is too short (minimum 10 characters)")

        # Validate base_url
        if not self.base_url:
            raise ValueError("base_url is required")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")

        # Validate environment
        if not self.environment:
            raise ValueError("environment cannot be empty")
        if len(self.environment) > 40:
            raise ValueError("environment must be 40 characters or less")
        if not self.environment.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "environment must contain only alphanumeric characters, hyphens, and underscores"
            )

        # Validate sample_rate
        if not 0.0 <= self.sample_rate <= 1.0:
            raise ValueError("sample_rate must be between 0.0 and 1.0")

        # Validate metrics_export_interval
        if not 1.0 <= self.metrics_export_interval <= 300.0:
            raise ValueError(
                "metrics_export_interval must be between 1.0 and 300.0 seconds"
            )

        # Validate flush_at
        if not 1 <= self.flush_at <= 1000:
            raise ValueError("flush_at must be between 1 and 1000")

        # Validate flush_interval
        if not 0.1 <= self.flush_interval <= 60.0:
            raise ValueError("flush_interval must be between 0.1 and 60.0 seconds")

        # Validate timeout
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

        # Validate compression
        if self.compression not in (None, "gzip", "deflate"):
            raise ValueError("compression must be 'gzip', 'deflate', or None")

        # Validate transport
        if self.transport not in ("http", "grpc"):
            raise ValueError("transport must be 'http' or 'grpc'")

        # Validate max_queue_size
        if self.max_queue_size < 1:
            raise ValueError("max_queue_size must be at least 1")

        # Validate export_timeout
        if self.export_timeout < 1000:
            raise ValueError("export_timeout must be at least 1000 milliseconds")

    @classmethod
    def from_env(cls, **overrides) -> "BrokleConfig":
        """
        Create configuration from environment variables.

        Environment variables:
            BROKLE_API_KEY - API key (required)
            BROKLE_BASE_URL - Base URL (default: http://localhost:8080)
            BROKLE_ENVIRONMENT - Environment tag (default: "default")
            BROKLE_RELEASE - Release identifier for deployment tracking
            BROKLE_VERSION - Trace-level version for A/B testing experiments
            BROKLE_TRACING_ENABLED - Enable tracing (default: true)
            BROKLE_METRICS_ENABLED - Enable metrics collection (default: true)
            BROKLE_METRICS_EXPORT_INTERVAL - Metrics export interval in seconds (default: 60.0)
            BROKLE_LOGS_ENABLED - Enable OTLP log export (opt-in, default: false)
            BROKLE_SAMPLE_RATE - Sampling rate (default: 1.0)
            BROKLE_DEBUG - Enable debug logging (default: false)
            BROKLE_FLUSH_AT - Batch size (default: 100)
            BROKLE_FLUSH_INTERVAL - Flush interval in seconds (default: 5.0)
            BROKLE_TIMEOUT - HTTP timeout in seconds (default: 30)
            BROKLE_USE_PROTOBUF - Use Protobuf format (default: true)
            BROKLE_COMPRESSION - Compression algorithm (default: "gzip")
            BROKLE_TRANSPORT - Transport protocol: "http" or "grpc" (default: "http")
            BROKLE_GRPC_ENDPOINT - Explicit gRPC endpoint (default: derived from base_url)

        Args:
            **overrides: Override specific configuration values

        Returns:
            BrokleConfig instance

        Raises:
            ValueError: If required environment variables are missing or invalid
        """
        # Required
        api_key = overrides.get("api_key") or os.getenv("BROKLE_API_KEY")
        if not api_key:
            raise ValueError(
                "BROKLE_API_KEY environment variable is required. "
                "Get your API key from https://app.brokle.ai/settings/api-keys"
            )

        # Connection
        base_url = overrides.get("base_url") or os.getenv(
            "BROKLE_BASE_URL", "http://localhost:8080"
        )
        timeout = int(overrides.get("timeout") or os.getenv("BROKLE_TIMEOUT", "30"))

        # Project
        environment = overrides.get("environment") or os.getenv(
            "BROKLE_ENVIRONMENT", "default"
        )
        release = overrides.get("release") or os.getenv("BROKLE_RELEASE")
        version = overrides.get("version") or os.getenv("BROKLE_VERSION")

        # Tracing control
        tracing_enabled = cls._parse_bool(
            overrides.get("tracing_enabled"),
            os.getenv("BROKLE_TRACING_ENABLED", "true"),
        )
        sample_rate = float(
            overrides.get("sample_rate") or os.getenv("BROKLE_SAMPLE_RATE", "1.0")
        )
        debug = cls._parse_bool(
            overrides.get("debug"), os.getenv("BROKLE_DEBUG", "false")
        )

        # Metrics configuration
        metrics_enabled = cls._parse_bool(
            overrides.get("metrics_enabled"),
            os.getenv("BROKLE_METRICS_ENABLED", "true"),
        )
        metrics_export_interval = float(
            overrides.get("metrics_export_interval")
            or os.getenv("BROKLE_METRICS_EXPORT_INTERVAL", "60.0")
        )

        # Logs configuration (opt-in, default: false)
        logs_enabled = cls._parse_bool(
            overrides.get("logs_enabled"), os.getenv("BROKLE_LOGS_ENABLED", "false")
        )

        # Batch configuration
        flush_at = int(overrides.get("flush_at") or os.getenv("BROKLE_FLUSH_AT", "100"))
        flush_interval = float(
            overrides.get("flush_interval") or os.getenv("BROKLE_FLUSH_INTERVAL", "5.0")
        )
        max_queue_size = int(
            overrides.get("max_queue_size")
            or os.getenv("BROKLE_MAX_QUEUE_SIZE", "2048")
        )
        export_timeout = int(
            overrides.get("export_timeout")
            or os.getenv("BROKLE_EXPORT_TIMEOUT", "30000")
        )

        # OTLP configuration
        use_protobuf = cls._parse_bool(
            overrides.get("use_protobuf"), os.getenv("BROKLE_USE_PROTOBUF", "true")
        )
        compression = overrides.get("compression") or os.getenv(
            "BROKLE_COMPRESSION", "gzip"
        )
        if compression == "none":
            compression = None

        # Transport configuration
        transport = overrides.get("transport") or os.getenv("BROKLE_TRANSPORT", "http")
        grpc_endpoint = overrides.get("grpc_endpoint") or os.getenv(
            "BROKLE_GRPC_ENDPOINT"
        )

        # Privacy (only from overrides, not environment)
        mask = overrides.get("mask")

        return cls(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            environment=environment,
            release=release,
            version=version,
            tracing_enabled=tracing_enabled,
            metrics_enabled=metrics_enabled,
            metrics_export_interval=metrics_export_interval,
            logs_enabled=logs_enabled,
            sample_rate=sample_rate,
            debug=debug,
            mask=mask,
            flush_at=flush_at,
            flush_interval=flush_interval,
            max_queue_size=max_queue_size,
            export_timeout=export_timeout,
            use_protobuf=use_protobuf,
            compression=compression,
            transport=transport,
            grpc_endpoint=grpc_endpoint,
        )

    @staticmethod
    def _parse_bool(override_value: Optional[bool], env_value: str) -> bool:
        """
        Parse boolean value from override or environment variable.

        Args:
            override_value: Explicit override value (takes precedence)
            env_value: Environment variable string value

        Returns:
            Boolean value
        """
        if override_value is not None:
            return bool(override_value)

        # Parse environment variable
        env_lower = env_value.lower().strip()
        return env_lower in ("true", "1", "yes", "on", "enabled")

    def get_otlp_endpoint(self) -> str:
        """Get the OTLP traces endpoint URL (OpenTelemetry standard)."""
        base = self.base_url.rstrip("/")
        return f"{base}/v1/traces"

    def get_headers(self) -> dict:
        """Get HTTP headers for OTLP export."""
        headers = {
            "X-API-Key": self.api_key,
        }

        # Add environment header if not default
        if self.environment != "default":
            headers["X-Brokle-Environment"] = self.environment

        return headers

    def __repr__(self) -> str:
        """Safe string representation (masks API key)."""
        masked_key = (
            f"{self.api_key[:7]}...{self.api_key[-4:]}"
            if len(self.api_key) > 11
            else "***"
        )
        return (
            f"BrokleConfig("
            f"api_key='{masked_key}', "
            f"base_url='{self.base_url}', "
            f"environment='{self.environment}', "
            f"tracing_enabled={self.tracing_enabled}, "
            f"metrics_enabled={self.metrics_enabled}, "
            f"logs_enabled={self.logs_enabled}, "
            f"sample_rate={self.sample_rate})"
        )
