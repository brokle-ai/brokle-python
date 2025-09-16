"""
Advanced automatic instrumentation system for popular AI/ML libraries.

Inspired by Optik's comprehensive instrumentation approach with
smart discovery, dynamic patching, and performance monitoring.
"""

import importlib
import logging
import sys
import threading
import traceback
from types import ModuleType
from typing import Dict, List, Optional, Set, Callable, Any, Type, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .registry import InstrumentationRegistry
from .error_handlers import InstrumentationError, LibraryNotAvailableError

logger = logging.getLogger(__name__)


class LibraryStatus(Enum):
    """Status of library instrumentation."""
    NOT_FOUND = "not_found"
    FOUND = "found"
    INSTRUMENTED = "instrumented"
    FAILED = "failed"
    DISABLED = "disabled"


@dataclass
class LibraryInfo:
    """Information about a discovered library."""
    name: str
    module_name: str
    version: Optional[str] = None
    status: LibraryStatus = LibraryStatus.NOT_FOUND
    instrumentation_class: Optional[Type] = None
    error: Optional[str] = None
    instrumented_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_available(self) -> bool:
        """Check if library is available for instrumentation."""
        return self.status in [LibraryStatus.FOUND, LibraryStatus.INSTRUMENTED]


class AutoInstrumentationEngine:
    """
    Advanced auto-instrumentation engine for AI/ML libraries.

    Provides smart discovery, dynamic instrumentation, and comprehensive
    monitoring of popular libraries in the Python ecosystem.
    """

    def __init__(self):
        self.registry = InstrumentationRegistry()
        self._discovered_libraries: Dict[str, LibraryInfo] = {}
        self._instrumentation_classes: Dict[str, Type] = {}
        self._lock = threading.Lock()
        self._auto_instrument_enabled = True
        self._discovery_cache: Dict[str, bool] = {}

        # Register built-in instrumentations
        self._register_builtin_instrumentations()

    def _register_builtin_instrumentations(self) -> None:
        """Register built-in instrumentation classes."""
        try:
            from .openai_instrumentation import OpenAIInstrumentation
            self.register_instrumentation("openai", OpenAIInstrumentation)
        except ImportError:
            pass

        try:
            from .anthropic_instrumentation import AnthropicInstrumentation
            self.register_instrumentation("anthropic", AnthropicInstrumentation)
        except ImportError:
            pass

        try:
            from .langchain_instrumentation import LangChainInstrumentation
            self.register_instrumentation("langchain", LangChainInstrumentation)
        except ImportError:
            pass

        # Register additional instrumentations
        additional_instrumentations = {
            "requests": "RequestsInstrumentation",
            "httpx": "HttpxInstrumentation",
            "urllib3": "Urllib3Instrumentation",
            "psycopg2": "PostgreSQLInstrumentation",
            "redis": "RedisInstrumentation",
            "transformers": "TransformersInstrumentation",
            "torch": "PyTorchInstrumentation",
            "tensorflow": "TensorFlowInstrumentation",
            "sklearn": "ScikitLearnInstrumentation",
            "numpy": "NumpyInstrumentation",
            "pandas": "PandasInstrumentation",
            "fastapi": "FastAPIInstrumentation",
            "flask": "FlaskInstrumentation",
            "django": "DjangoInstrumentation",
            "celery": "CeleryInstrumentation",
            "asyncio": "AsyncioInstrumentation",
        }

        for lib_name, class_name in additional_instrumentations.items():
            try:
                # Try to import from our instrumentations module
                mod = importlib.import_module(f".{lib_name}_instrumentation", package=__package__)
                instrumentation_class = getattr(mod, class_name)
                self.register_instrumentation(lib_name, instrumentation_class)
            except (ImportError, AttributeError):
                # Create placeholder for future implementation
                logger.debug(f"Instrumentation for {lib_name} not yet implemented")

    def register_instrumentation(self, library_name: str, instrumentation_class: Type) -> None:
        """Register an instrumentation class for a library."""
        with self._lock:
            self._instrumentation_classes[library_name] = instrumentation_class
            logger.debug(f"Registered instrumentation for {library_name}")

    def discover_libraries(self, libraries: Optional[List[str]] = None) -> Dict[str, LibraryInfo]:
        """
        Discover available libraries in the environment.

        Args:
            libraries: Specific libraries to discover, or None for all known libraries

        Returns:
            Dictionary of library name to LibraryInfo
        """
        if libraries is None:
            libraries = list(self._instrumentation_classes.keys())

        discovered = {}

        for lib_name in libraries:
            lib_info = self._discover_library(lib_name)
            discovered[lib_name] = lib_info
            self._discovered_libraries[lib_name] = lib_info

        logger.info(f"Discovered {len(discovered)} libraries: {list(discovered.keys())}")
        return discovered

    def _discover_library(self, library_name: str) -> LibraryInfo:
        """Discover a specific library."""
        # Check cache first
        if library_name in self._discovery_cache:
            if not self._discovery_cache[library_name]:
                return LibraryInfo(
                    name=library_name,
                    module_name=library_name,
                    status=LibraryStatus.NOT_FOUND
                )

        # Try to import the library
        try:
            module = importlib.import_module(library_name)
            version = self._get_library_version(module, library_name)

            lib_info = LibraryInfo(
                name=library_name,
                module_name=library_name,
                version=version,
                status=LibraryStatus.FOUND,
                instrumentation_class=self._instrumentation_classes.get(library_name),
                metadata={
                    "module_path": getattr(module, "__file__", None),
                    "package_path": getattr(module, "__path__", None)
                }
            )

            self._discovery_cache[library_name] = True
            logger.debug(f"Discovered {library_name} v{version}")
            return lib_info

        except ImportError as e:
            self._discovery_cache[library_name] = False
            return LibraryInfo(
                name=library_name,
                module_name=library_name,
                status=LibraryStatus.NOT_FOUND,
                error=str(e)
            )

    def _get_library_version(self, module: ModuleType, library_name: str) -> Optional[str]:
        """Extract version from imported module."""
        # Common version attribute names
        version_attrs = ["__version__", "VERSION", "version"]

        for attr in version_attrs:
            if hasattr(module, attr):
                version = getattr(module, attr)
                if isinstance(version, str):
                    return version
                elif hasattr(version, "__str__"):
                    return str(version)

        # Try package metadata
        try:
            import importlib.metadata
            return importlib.metadata.version(library_name)
        except (ImportError, Exception):
            pass

        return None

    def auto_instrument_all(
        self,
        libraries: Optional[List[str]] = None,
        force_discover: bool = False
    ) -> Dict[str, LibraryInfo]:
        """
        Automatically instrument all available libraries.

        Args:
            libraries: Specific libraries to instrument, or None for all
            force_discover: Force rediscovery of libraries

        Returns:
            Dictionary of instrumentation results
        """
        if not self._auto_instrument_enabled:
            logger.info("Auto-instrumentation is disabled")
            return {}

        # Discover libraries if needed
        if force_discover or not self._discovered_libraries:
            self.discover_libraries(libraries)

        instrumentation_results = {}

        # Instrument each discovered library
        for lib_name, lib_info in self._discovered_libraries.items():
            if libraries and lib_name not in libraries:
                continue

            if lib_info.status == LibraryStatus.FOUND and lib_info.instrumentation_class:
                try:
                    result = self.instrument_library(lib_name)
                    instrumentation_results[lib_name] = result
                except Exception as e:
                    logger.error(f"Failed to instrument {lib_name}: {e}")
                    lib_info.status = LibraryStatus.FAILED
                    lib_info.error = str(e)
                    instrumentation_results[lib_name] = lib_info

        return instrumentation_results

    def instrument_library(self, library_name: str) -> LibraryInfo:
        """
        Instrument a specific library.

        Args:
            library_name: Name of library to instrument

        Returns:
            Updated LibraryInfo with instrumentation status
        """
        with self._lock:
            # Get library info
            lib_info = self._discovered_libraries.get(library_name)
            if not lib_info:
                lib_info = self._discover_library(library_name)

            if lib_info.status != LibraryStatus.FOUND:
                raise LibraryNotAvailableError(f"Library {library_name} not available for instrumentation")

            if not lib_info.instrumentation_class:
                raise InstrumentationError(f"No instrumentation class registered for {library_name}")

            try:
                # Create and register instrumentation
                instrumentation = lib_info.instrumentation_class()
                self.registry.register(library_name, instrumentation)

                # Update status
                lib_info.status = LibraryStatus.INSTRUMENTED
                lib_info.instrumented_at = datetime.utcnow()
                lib_info.error = None

                logger.info(f"Successfully instrumented {library_name}")
                return lib_info

            except Exception as e:
                # Update status with error
                lib_info.status = LibraryStatus.FAILED
                lib_info.error = str(e)

                logger.error(f"Failed to instrument {library_name}: {e}")
                raise InstrumentationError(f"Instrumentation failed for {library_name}: {e}")

    def uninstrument_library(self, library_name: str) -> bool:
        """
        Remove instrumentation from a library.

        Args:
            library_name: Name of library to uninstrument

        Returns:
            True if successfully uninstrumented
        """
        with self._lock:
            try:
                # Unregister from registry
                self.registry.unregister(library_name)

                # Update library info
                if library_name in self._discovered_libraries:
                    lib_info = self._discovered_libraries[library_name]
                    if lib_info.status == LibraryStatus.INSTRUMENTED:
                        lib_info.status = LibraryStatus.FOUND
                        lib_info.instrumented_at = None
                        lib_info.error = None

                logger.info(f"Successfully uninstrumented {library_name}")
                return True

            except Exception as e:
                logger.error(f"Failed to uninstrument {library_name}: {e}")
                return False

    def get_instrumentation_status(self) -> Dict[str, Any]:
        """Get comprehensive instrumentation status."""
        with self._lock:
            status_counts = {}
            for lib_info in self._discovered_libraries.values():
                status = lib_info.status.value
                status_counts[status] = status_counts.get(status, 0) + 1

            instrumented_libraries = [
                {
                    "name": lib_info.name,
                    "version": lib_info.version,
                    "instrumented_at": lib_info.instrumented_at.isoformat() if lib_info.instrumented_at else None
                }
                for lib_info in self._discovered_libraries.values()
                if lib_info.status == LibraryStatus.INSTRUMENTED
            ]

            failed_libraries = [
                {
                    "name": lib_info.name,
                    "error": lib_info.error
                }
                for lib_info in self._discovered_libraries.values()
                if lib_info.status == LibraryStatus.FAILED
            ]

            return {
                "auto_instrument_enabled": self._auto_instrument_enabled,
                "total_libraries_discovered": len(self._discovered_libraries),
                "status_counts": status_counts,
                "instrumented_libraries": instrumented_libraries,
                "failed_libraries": failed_libraries,
                "registry_status": self.registry.get_status()
            }

    def enable_auto_instrumentation(self) -> None:
        """Enable automatic instrumentation."""
        self._auto_instrument_enabled = True
        logger.info("Auto-instrumentation enabled")

    def disable_auto_instrumentation(self) -> None:
        """Disable automatic instrumentation."""
        self._auto_instrument_enabled = False
        logger.info("Auto-instrumentation disabled")

    def clear_discovery_cache(self) -> None:
        """Clear library discovery cache."""
        with self._lock:
            self._discovery_cache.clear()
            logger.info("Discovery cache cleared")

    def get_library_info(self, library_name: str) -> Optional[LibraryInfo]:
        """Get information about a specific library."""
        return self._discovered_libraries.get(library_name)

    def get_all_libraries(self) -> Dict[str, LibraryInfo]:
        """Get information about all discovered libraries."""
        return self._discovered_libraries.copy()

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all instrumented libraries."""
        health_status = {}

        for lib_name, lib_info in self._discovered_libraries.items():
            if lib_info.status == LibraryStatus.INSTRUMENTED:
                try:
                    # Get instrumentation from registry
                    instrumentation = self.registry.get_instrumentation(lib_name)
                    if instrumentation and hasattr(instrumentation, "health_check"):
                        health_result = instrumentation.health_check()
                        health_status[lib_name] = {
                            "status": "healthy",
                            "details": health_result
                        }
                    else:
                        health_status[lib_name] = {
                            "status": "unknown",
                            "details": "Health check not implemented"
                        }
                except Exception as e:
                    health_status[lib_name] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }

        return {
            "overall_status": "healthy" if all(
                h.get("status") == "healthy" for h in health_status.values()
            ) else "degraded",
            "libraries": health_status
        }


# Global auto-instrumentation engine
_auto_engine: Optional[AutoInstrumentationEngine] = None
_engine_lock = threading.Lock()


def get_auto_engine() -> AutoInstrumentationEngine:
    """Get global auto-instrumentation engine."""
    global _auto_engine

    with _engine_lock:
        if _auto_engine is None:
            _auto_engine = AutoInstrumentationEngine()

        return _auto_engine


def auto_instrument(
    libraries: Optional[List[str]] = None,
    force: bool = False
) -> Dict[str, LibraryInfo]:
    """
    Automatically instrument available libraries.

    Args:
        libraries: Specific libraries to instrument, or None for all
        force: Force rediscovery and instrumentation

    Returns:
        Dictionary of instrumentation results
    """
    engine = get_auto_engine()
    return engine.auto_instrument_all(libraries=libraries, force_discover=force)


def discover_libraries(libraries: Optional[List[str]] = None) -> Dict[str, LibraryInfo]:
    """
    Discover available libraries.

    Args:
        libraries: Specific libraries to discover, or None for all

    Returns:
        Dictionary of discovered libraries
    """
    engine = get_auto_engine()
    return engine.discover_libraries(libraries)


def get_instrumentation_status() -> Dict[str, Any]:
    """Get comprehensive instrumentation status."""
    engine = get_auto_engine()
    return engine.get_instrumentation_status()


def instrument_library(library_name: str) -> LibraryInfo:
    """Instrument a specific library."""
    engine = get_auto_engine()
    return engine.instrument_library(library_name)


def uninstrument_library(library_name: str) -> bool:
    """Uninstrument a specific library."""
    engine = get_auto_engine()
    return engine.uninstrument_library(library_name)


def enable_auto_instrumentation() -> None:
    """Enable automatic instrumentation."""
    engine = get_auto_engine()
    engine.enable_auto_instrumentation()


def disable_auto_instrumentation() -> None:
    """Disable automatic instrumentation."""
    engine = get_auto_engine()
    engine.disable_auto_instrumentation()


def health_check() -> Dict[str, Any]:
    """Perform health check on instrumented libraries."""
    engine = get_auto_engine()
    return engine.health_check()