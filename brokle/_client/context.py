"""
Context-Aware Client Management - Brokle SDK

This module provides LangFuse-style context-aware client management using ContextVar
for thread-safe, multi-project client isolation. Prevents data leakage between
different projects and environments in multi-tenant applications.

Key Features:
- Thread-safe client isolation using ContextVar
- Multi-project safety prevents data cross-contamination
- Automatic project validation and switching
- Context inheritance for nested operations
- Production-safe with comprehensive validation

Based on LangFuse's get_client() pattern but enhanced for Brokle's multi-environment needs.
"""

import logging
import warnings
from contextvars import ContextVar
from typing import Optional, Dict, Any, Set
import threading
from dataclasses import dataclass, field
import time

from .client import Brokle
from ..config import Config
from ..exceptions import ConfigurationError, AuthenticationError

logger = logging.getLogger(__name__)

# Context variable for thread-safe client storage
_brokle_client_context: ContextVar[Optional['Brokle']] = ContextVar(
    'brokle_client_context',
    default=None
)

# Global client registry for tracking and validation
_client_registry: Dict[str, 'Brokle'] = {}
_registry_lock = threading.Lock()


@dataclass
class ClientContext:
    """Tracks client context for safety and debugging"""
    client: 'Brokle'
    project_id: str
    environment: str
    api_key_hash: str
    created_at: float = field(default_factory=time.time)
    usage_count: int = 0

    def __post_init__(self):
        self.api_key_hash = self._hash_api_key(self.client.config.api_key)

    def _hash_api_key(self, api_key: str) -> str:
        """Create safe hash of API key for tracking"""
        if not api_key:
            return "none"
        return f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "short_key"


class ContextAwareClientManager:
    """
    Manages Brokle clients with context awareness and multi-project safety.

    Implements LangFuse's pattern of context-aware client management while
    adding Brokle-specific features like environment tags and project isolation.
    """

    def __init__(self):
        self._active_contexts: Dict[str, ClientContext] = {}
        self._validation_warnings: Set[str] = set()

    def get_client(
        self,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        project_id: Optional[str] = None,
        environment: Optional[str] = None,
        config: Optional[Config] = None,
        **kwargs
    ) -> 'Brokle':
        """
        Get or create a context-aware Brokle client.

        This is the main entry point for client access, providing:
        - Thread-safe client isolation
        - Multi-project data safety
        - Automatic context validation
        - Configuration inheritance

        Args:
            api_key: Brokle API key (overrides environment/context)
            host: Brokle host URL (overrides environment/context)
            project_id: Project ID (overrides environment/context)
            environment: Environment tag (overrides environment/context)
            config: Complete configuration object
            **kwargs: Additional client configuration

        Returns:
            Brokle client instance, context-aware and thread-safe

        Raises:
            ConfigurationError: If configuration is invalid or incomplete
            AuthenticationError: If API key validation fails
        """

        # Check for existing context client first
        current_client = _brokle_client_context.get()

        # Determine if we can reuse the current client
        if current_client and self._can_reuse_client(
            current_client, api_key, host, project_id, environment
        ):
            self._track_client_usage(current_client)
            return current_client

        # Create new client with proper configuration
        new_client = self._create_context_client(
            api_key=api_key,
            host=host,
            project_id=project_id,
            environment=environment,
            config=config,
            **kwargs
        )

        # Set in context for thread safety
        _brokle_client_context.set(new_client)

        # Track in registry for validation
        self._register_client_context(new_client)

        return new_client

    def _can_reuse_client(
        self,
        client: 'Brokle',
        api_key: Optional[str],
        host: Optional[str],
        project_id: Optional[str],
        environment: Optional[str]
    ) -> bool:
        """
        Determine if existing client can be reused safely.

        Critical for multi-project safety - prevents data leakage between
        different projects, environments, or authentication contexts.
        """
        if not client or not client.config:
            return False

        config = client.config

        # Check API key compatibility
        if api_key and api_key != config.api_key:
            self._warn_context_mismatch("API key mismatch detected")
            return False

        # Check host compatibility
        if host and host != config.host:
            self._warn_context_mismatch("Host mismatch detected")
            return False

        # Check project ID compatibility
        if project_id and project_id != config.project_id:
            self._warn_context_mismatch("Project ID mismatch detected")
            return False

        # Check environment compatibility
        if environment and environment != config.environment:
            self._warn_context_mismatch("Environment mismatch detected")
            return False

        return True

    def _create_context_client(
        self,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        project_id: Optional[str] = None,
        environment: Optional[str] = None,
        config: Optional[Config] = None,
        **kwargs
    ) -> 'Brokle':
        """Create new client with proper context configuration"""

        try:
            # Use provided config or create from parameters with environment fallback
            if config:
                client_config = config
            else:
                # Start with environment variables as base, then overlay provided parameters
                from ..config import Config
                client_config = Config.from_env()

                # Override with any explicitly provided parameters
                if api_key is not None:
                    client_config.api_key = api_key
                if host is not None:
                    client_config.host = host
                if project_id is not None:
                    client_config.project_id = project_id
                if environment is not None:
                    client_config.environment = environment

                # Apply any additional kwargs
                for key, value in kwargs.items():
                    if hasattr(client_config, key) and value is not None:
                        setattr(client_config, key, value)

            # Validate configuration
            self._validate_client_config(client_config)

            # Create client
            client = Brokle(config=client_config)

            logger.debug(
                f"Created new context client for project {client_config.project_id}, "
                f"environment {client_config.environment}"
            )

            return client

        except Exception as e:
            logger.error(f"Failed to create context client: {e}")
            raise ConfigurationError(f"Client creation failed: {e}")

    def _validate_client_config(self, config: Config):
        """Validate and apply safety fallbacks for client configuration"""

        # Apply same safety logic as Brokle.__init__ for missing credentials
        if not config.api_key:
            logger.warning(
                "Authentication error: Brokle client initialized without api_key. "
                "Provide an api_key parameter or set BROKLE_API_KEY environment variable."
            )
            # Continue with disabled tracing similar to original logic
            config.api_key = "ak_fake"  # Must start with "ak_" to pass validation
            config.otel_enabled = False

        if not config.project_id:
            logger.warning(
                "Configuration error: Brokle client initialized without project_id. "
                "Provide a project_id parameter or set BROKLE_PROJECT_ID environment variable."
            )
            config.project_id = "fake"
            config.otel_enabled = False

        # Validate environment tag format
        if config.environment:
            from .._utils.validation import validate_environment
            try:
                validate_environment(config.environment)
            except Exception as e:
                raise ConfigurationError(f"Invalid environment tag: {e}")

    def _register_client_context(self, client: 'Brokle'):
        """Register client in global registry for tracking"""

        with _registry_lock:
            context_key = self._get_context_key(client)

            context = ClientContext(
                client=client,
                project_id=client.config.project_id,
                environment=client.config.environment,
                api_key_hash=""  # Will be set in __post_init__
            )

            self._active_contexts[context_key] = context
            _client_registry[context_key] = client

            logger.debug(f"Registered client context: {context_key}")

    def _track_client_usage(self, client: 'Brokle'):
        """Track client usage for monitoring and debugging"""

        context_key = self._get_context_key(client)

        if context_key in self._active_contexts:
            self._active_contexts[context_key].usage_count += 1

    def _get_context_key(self, client: 'Brokle') -> str:
        """Generate unique context key for client tracking"""
        config = client.config
        return f"{config.project_id}:{config.environment}:{hash(config.api_key)}"

    def _warn_context_mismatch(self, message: str):
        """Issue context mismatch warning (once per type)"""

        if message not in self._validation_warnings:
            warnings.warn(
                f"Context mismatch: {message}. Creating new client for safety.",
                UserWarning,
                stacklevel=3
            )
            self._validation_warnings.add(message)
            logger.warning(f"Context safety: {message}")

    def clear_context(self):
        """Clear current context client (useful for testing)"""
        _brokle_client_context.set(None)
        logger.debug("Cleared client context")

    def get_context_info(self) -> Dict[str, Any]:
        """Get information about current context for debugging"""

        current_client = _brokle_client_context.get()

        if not current_client:
            return {"context": "no_client", "clients_active": len(self._active_contexts)}

        context_key = self._get_context_key(current_client)
        context = self._active_contexts.get(context_key)

        return {
            "context": "active_client",
            "project_id": current_client.config.project_id,
            "environment": current_client.config.environment,
            "host": current_client.config.host,
            "usage_count": context.usage_count if context else 0,
            "created_at": context.created_at if context else 0,
            "clients_active": len(self._active_contexts)
        }

    def cleanup_inactive_contexts(self, max_age_seconds: int = 3600):
        """Clean up old inactive contexts (useful for long-running applications)"""

        current_time = time.time()
        to_remove = []

        with _registry_lock:
            for key, context in self._active_contexts.items():
                if current_time - context.created_at > max_age_seconds:
                    to_remove.append(key)

            for key in to_remove:
                del self._active_contexts[key]
                if key in _client_registry:
                    del _client_registry[key]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} inactive client contexts")


# Global context manager instance
_context_manager = ContextAwareClientManager()


def get_client(
    api_key: Optional[str] = None,
    host: Optional[str] = None,
    project_id: Optional[str] = None,
    environment: Optional[str] = None,
    config: Optional[Config] = None,
    **kwargs
) -> 'Brokle':
    """
    Get context-aware Brokle client with multi-project safety.

    This is the main entry point for accessing Brokle clients. It provides:
    - Thread-safe client isolation using ContextVar
    - Multi-project data safety
    - Automatic configuration from environment variables
    - Context validation and warning system

    Args:
        api_key: Brokle API key (overrides BROKLE_API_KEY)
        host: Brokle host URL (overrides BROKLE_HOST)
        project_id: Project ID (overrides BROKLE_PROJECT_ID)
        environment: Environment tag (overrides BROKLE_ENVIRONMENT)
        config: Complete configuration object
        **kwargs: Additional client configuration

    Returns:
        Brokle client instance

    Example:
        # Use environment variables
        client = get_client()

        # Override specific settings
        client = get_client(
            project_id="proj_custom",
            environment="staging"
        )

        # Use in async context (inherits context)
        async def my_function():
            client = get_client()  # Same client as parent context
            return await client.chat.create(...)
    """
    return _context_manager.get_client(
        api_key=api_key,
        host=host,
        project_id=project_id,
        environment=environment,
        config=config,
        **kwargs
    )


def clear_context():
    """Clear current client context (useful for testing and cleanup)"""
    _context_manager.clear_context()


def get_context_info() -> Dict[str, Any]:
    """Get information about current client context for debugging"""
    return _context_manager.get_context_info()


def cleanup_contexts(max_age_seconds: int = 3600):
    """Clean up old inactive contexts"""
    _context_manager.cleanup_inactive_contexts(max_age_seconds)


# Export public API
__all__ = [
    'get_client',
    'clear_context',
    'get_context_info',
    'cleanup_contexts',
    'ContextAwareClientManager',
    'ClientContext',
]