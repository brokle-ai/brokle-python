"""
Auto-instrumentation registry and activation system for Brokle SDK.

This module provides a centralized registry for managing auto-instrumentation
of popular LLM libraries with configurable activation.
"""

import logging
from typing import Dict, List, Optional, Set, Callable
from enum import Enum

from .error_handlers import (
    InstrumentationError,
    LibraryNotAvailableError,
    ErrorSeverity,
    get_error_handler,
    safe_operation,
    instrumentation_context
)

from .openai_instrumentation import (
    OpenAIInstrumentation,
    instrument_openai,
    uninstrument_openai,
    is_openai_instrumented
)
from .anthropic_instrumentation import (
    AnthropicInstrumentation,
    instrument_anthropic,
    uninstrument_anthropic,
    is_anthropic_instrumented
)
from .langchain_instrumentation import (
    LangChainInstrumentation,
    instrument_langchain,
    uninstrument_langchain,
    is_langchain_instrumented
)

logger = logging.getLogger(__name__)


class InstrumentationStatus(Enum):
    """Status of instrumentation for a library."""
    NOT_AVAILABLE = "not_available"  # Library not installed
    AVAILABLE = "available"          # Library available but not instrumented
    INSTRUMENTED = "instrumented"    # Library instrumented successfully
    FAILED = "failed"               # Instrumentation failed


class LibraryInstrumentation:
    """Configuration for a library's instrumentation."""

    def __init__(
        self,
        name: str,
        instrument_func: Callable[[], bool],
        uninstrument_func: Callable[[], bool],
        is_instrumented_func: Callable[[], bool],
        is_available_func: Callable[[], bool],
        description: str = "",
        auto_instrument: bool = True
    ):
        self.name = name
        self.instrument_func = instrument_func
        self.uninstrument_func = uninstrument_func
        self.is_instrumented_func = is_instrumented_func
        self.is_available_func = is_available_func
        self.description = description
        self.auto_instrument = auto_instrument

    def get_status(self) -> InstrumentationStatus:
        """Get current instrumentation status."""
        if not self.is_available_func():
            return InstrumentationStatus.NOT_AVAILABLE

        if self.is_instrumented_func():
            return InstrumentationStatus.INSTRUMENTED

        return InstrumentationStatus.AVAILABLE

    @safe_operation("registry", "instrument_library", ErrorSeverity.MEDIUM)
    def instrument(self) -> bool:
        """Instrument this library."""
        error_handler = get_error_handler()

        with instrumentation_context("registry", f"instrument_{self.name}", ErrorSeverity.MEDIUM):
            if not self.is_available_func():
                raise LibraryNotAvailableError(
                    f"{self.name} library not available for instrumentation",
                    severity=ErrorSeverity.MEDIUM,
                    library=self.name,
                    operation="instrument"
                )

            if self.is_instrumented_func():
                logger.info(f"{self.name} already instrumented")
                return True

            # Check if this library has been failing recently
            if not error_handler.is_operation_healthy("registry", f"instrument_{self.name}"):
                logger.warning(f"Skipping {self.name} instrumentation - circuit breaker open")
                return False

            result = self.instrument_func()
            if result:
                logger.info(f"Successfully instrumented {self.name}")
                # Reset error tracking on success
                error_handler.reset_errors("registry", f"instrument_{self.name}")
            else:
                raise InstrumentationError(
                    f"Instrumentation function returned False for {self.name}",
                    severity=ErrorSeverity.MEDIUM,
                    library=self.name,
                    operation="instrument"
                )

            return result

    @safe_operation("registry", "uninstrument_library", ErrorSeverity.LOW)
    def uninstrument(self) -> bool:
        """Remove instrumentation for this library."""
        with instrumentation_context("registry", f"uninstrument_{self.name}", ErrorSeverity.LOW):
            if not self.is_instrumented_func():
                logger.debug(f"{self.name} not currently instrumented")
                return True

            result = self.uninstrument_func()
            if result:
                logger.info(f"Successfully uninstrumented {self.name}")
                # Reset error tracking on successful uninstrumentation
                error_handler = get_error_handler()
                error_handler.reset_errors("registry", f"uninstrument_{self.name}")
            else:
                raise InstrumentationError(
                    f"Uninstrumentation function returned False for {self.name}",
                    severity=ErrorSeverity.LOW,
                    library=self.name,
                    operation="uninstrument"
                )

            return result


class InstrumentationRegistry:
    """Registry for managing auto-instrumentation of LLM libraries."""

    def __init__(self):
        self._libraries: Dict[str, LibraryInstrumentation] = {}
        self._setup_default_libraries()

    def _setup_default_libraries(self):
        """Setup default library instrumentations."""
        # OpenAI instrumentation
        openai_instrumentation = OpenAIInstrumentation()
        self._libraries["openai"] = LibraryInstrumentation(
            name="openai",
            instrument_func=instrument_openai,
            uninstrument_func=uninstrument_openai,
            is_instrumented_func=is_openai_instrumented,
            is_available_func=openai_instrumentation.is_available,
            description="OpenAI Python library for GPT models",
            auto_instrument=True
        )

        # Anthropic instrumentation
        anthropic_instrumentation = AnthropicInstrumentation()
        self._libraries["anthropic"] = LibraryInstrumentation(
            name="anthropic",
            instrument_func=instrument_anthropic,
            uninstrument_func=uninstrument_anthropic,
            is_instrumented_func=is_anthropic_instrumented,
            is_available_func=anthropic_instrumentation.is_available,
            description="Anthropic Python library for Claude models",
            auto_instrument=True
        )

        # LangChain instrumentation
        langchain_instrumentation = LangChainInstrumentation()
        self._libraries["langchain"] = LibraryInstrumentation(
            name="langchain",
            instrument_func=instrument_langchain,
            uninstrument_func=uninstrument_langchain,
            is_instrumented_func=is_langchain_instrumented,
            is_available_func=langchain_instrumentation.is_available,
            description="LangChain framework for LLM applications",
            auto_instrument=True
        )

    def register_library(self, library: LibraryInstrumentation) -> None:
        """Register a new library for instrumentation."""
        self._libraries[library.name] = library
        logger.info(f"Registered instrumentation for {library.name}")

    def unregister_library(self, name: str) -> bool:
        """Unregister a library from instrumentation."""
        if name in self._libraries:
            # Uninstrument first if needed
            library = self._libraries[name]
            library.uninstrument()

            del self._libraries[name]
            logger.info(f"Unregistered instrumentation for {name}")
            return True

        return False

    def get_library(self, name: str) -> Optional[LibraryInstrumentation]:
        """Get library instrumentation by name."""
        return self._libraries.get(name)

    def list_libraries(self) -> List[str]:
        """List all registered library names."""
        return list(self._libraries.keys())

    def get_status(self, name: Optional[str] = None) -> Dict[str, InstrumentationStatus]:
        """Get instrumentation status for libraries."""
        if name:
            library = self._libraries.get(name)
            if library:
                return {name: library.get_status()}
            return {}

        return {
            name: library.get_status()
            for name, library in self._libraries.items()
        }

    def get_available_libraries(self) -> List[str]:
        """Get list of available (installed) libraries."""
        return [
            name for name, library in self._libraries.items()
            if library.is_available_func()
        ]

    def get_instrumented_libraries(self) -> List[str]:
        """Get list of currently instrumented libraries."""
        return [
            name for name, library in self._libraries.items()
            if library.is_instrumented_func()
        ]

    def instrument_library(self, name: str) -> bool:
        """Instrument a specific library."""
        library = self._libraries.get(name)
        if not library:
            logger.error(f"Library {name} not registered")
            return False

        return library.instrument()

    def uninstrument_library(self, name: str) -> bool:
        """Remove instrumentation for a specific library."""
        library = self._libraries.get(name)
        if not library:
            logger.error(f"Library {name} not registered")
            return False

        return library.uninstrument()

    def instrument_all(self, only_available: bool = True, only_auto: bool = True) -> Dict[str, bool]:
        """Instrument all registered libraries."""
        results = {}

        for name, library in self._libraries.items():
            # Skip if not available and only_available is True
            if only_available and not library.is_available_func():
                logger.debug(f"Skipping {name} - not available")
                results[name] = False
                continue

            # Skip if auto_instrument is False and only_auto is True
            if only_auto and not library.auto_instrument:
                logger.debug(f"Skipping {name} - auto_instrument disabled")
                results[name] = False
                continue

            results[name] = library.instrument()

        return results

    def uninstrument_all(self) -> Dict[str, bool]:
        """Remove instrumentation for all libraries."""
        results = {}

        for name, library in self._libraries.items():
            results[name] = library.uninstrument()

        return results

    def auto_instrument(
        self,
        libraries: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """Automatically instrument available libraries."""
        if libraries is None:
            # Instrument all available libraries with auto_instrument=True
            target_libraries = [
                name for name, lib in self._libraries.items()
                if lib.auto_instrument and lib.is_available_func()
            ]
        else:
            # Instrument only specified libraries
            target_libraries = [
                name for name in libraries
                if name in self._libraries and self._libraries[name].is_available_func()
            ]

        if exclude:
            target_libraries = [name for name in target_libraries if name not in exclude]

        results = {}
        for name in target_libraries:
            results[name] = self.instrument_library(name)

        # Log summary
        successful = [name for name, success in results.items() if success]
        failed = [name for name, success in results.items() if not success]

        if successful:
            logger.info(f"Successfully instrumented: {', '.join(successful)}")

        if failed:
            logger.warning(f"Failed to instrument: {', '.join(failed)}")

        return results

    def get_instrumentation_summary(self) -> Dict[str, Dict[str, any]]:
        """Get comprehensive instrumentation summary."""
        summary = {}
        error_handler = get_error_handler()

        for name, library in self._libraries.items():
            try:
                status = library.get_status()
                is_healthy = error_handler.is_operation_healthy("registry", f"instrument_{name}")

                summary[name] = {
                    "status": status.value,
                    "description": library.description,
                    "auto_instrument": library.auto_instrument,
                    "available": library.is_available_func(),
                    "instrumented": library.is_instrumented_func(),
                    "healthy": is_healthy,
                    "error_summary": error_handler.get_error_summary(name) if not is_healthy else {}
                }
            except Exception as e:
                logger.warning(f"Failed to get status for {name}: {e}")
                summary[name] = {
                    "status": "error",
                    "description": library.description,
                    "auto_instrument": library.auto_instrument,
                    "available": False,
                    "instrumented": False,
                    "healthy": False,
                    "error_summary": {"status_error": str(e)}
                }

        return summary

    def get_health_report(self) -> Dict[str, any]:
        """Get detailed health report for all instrumentation operations."""
        error_handler = get_error_handler()
        summary = self.get_instrumentation_summary()

        # Calculate overall health metrics
        total_libraries = len(self._libraries)
        healthy_libraries = sum(1 for lib in summary.values() if lib["healthy"])
        instrumented_libraries = sum(1 for lib in summary.values() if lib["instrumented"])
        available_libraries = sum(1 for lib in summary.values() if lib["available"])

        health_score = (healthy_libraries / total_libraries) * 100 if total_libraries > 0 else 0

        return {
            "overall_health": {
                "score": round(health_score, 2),
                "total_libraries": total_libraries,
                "healthy_libraries": healthy_libraries,
                "instrumented_libraries": instrumented_libraries,
                "available_libraries": available_libraries
            },
            "library_details": summary,
            "error_summary": error_handler.get_error_summary(),
            "circuit_breaker_states": {
                name: error_handler.get_circuit_breaker(f"registry.instrument_{name}").state
                for name in self._libraries.keys()
            }
        }

    def reset_all_errors(self) -> None:
        """Reset all error tracking and circuit breakers."""
        error_handler = get_error_handler()

        for name in self._libraries.keys():
            error_handler.reset_errors("registry", f"instrument_{name}")
            error_handler.reset_errors("registry", f"uninstrument_{name}")
            error_handler.reset_errors(name)  # Reset library-specific errors too

        logger.info("Reset all instrumentation error tracking")

    def print_status(self) -> None:
        """Print instrumentation status in a readable format."""
        print("\n=== Brokle Auto-Instrumentation Status ===")

        summary = self.get_instrumentation_summary()
        health_report = self.get_health_report()

        for name, info in summary.items():
            status_symbol = {
                "instrumented": "✅",
                "available": "⚪",
                "not_available": "❌",
                "failed": "🔴",
                "error": "💥"
            }.get(info["status"], "❓")

            health_symbol = "🔴" if not info["healthy"] else ""
            auto_marker = " (auto)" if info["auto_instrument"] else ""

            print(f"{status_symbol}{health_symbol} {name}{auto_marker}: {info['status']}")
            if info["description"]:
                print(f"    📝 {info['description']}")

            # Show error information if unhealthy
            if not info["healthy"] and info["error_summary"]:
                print(f"    ⚠️  Issues detected - check logs for details")

        # Overall health summary
        overall = health_report["overall_health"]
        print(f"\n📊 Health Summary:")
        print(f"   Overall Health: {overall['score']}%")
        print(f"   Libraries: {overall['instrumented_libraries']}/{overall['available_libraries']} instrumented, {overall['healthy_libraries']}/{overall['total_libraries']} healthy")

        # Circuit breaker status
        open_breakers = [name for name, state in health_report["circuit_breaker_states"].items() if state == "open"]
        if open_breakers:
            print(f"   🚨 Circuit Breakers Open: {', '.join(open_breakers)}")

        print("=" * 50)

    def print_health_report(self) -> None:
        """Print detailed health report."""
        print("\n=== Brokle Auto-Instrumentation Health Report ===")

        health_report = self.get_health_report()

        # Overall health
        overall = health_report["overall_health"]
        print(f"\n🏥 Overall Health Score: {overall['score']}%")
        print(f"   📊 Libraries: {overall['total_libraries']} total, {overall['available_libraries']} available")
        print(f"   ✅ Status: {overall['instrumented_libraries']} instrumented, {overall['healthy_libraries']} healthy")

        # Circuit breaker details
        print(f"\n⚡ Circuit Breaker Status:")
        for name, state in health_report["circuit_breaker_states"].items():
            state_symbol = {"closed": "🟢", "open": "🔴", "half-open": "🟡"}.get(state, "❓")
            print(f"   {state_symbol} {name}: {state}")

        # Error summary
        error_summary = health_report["error_summary"]
        if error_summary.get("error_counts"):
            print(f"\n❗ Recent Errors:")
            for operation, count in error_summary["error_counts"].items():
                print(f"   {operation}: {count} errors")

        print("=" * 60)


# Global registry instance
_registry = InstrumentationRegistry()


def get_registry() -> InstrumentationRegistry:
    """Get the global instrumentation registry."""
    return _registry


def auto_instrument(
    libraries: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None
) -> Dict[str, bool]:
    """
    Automatically instrument available LLM libraries.

    Args:
        libraries: Specific libraries to instrument (default: all available)
        exclude: Libraries to exclude from instrumentation

    Returns:
        Dict mapping library names to instrumentation success status

    Examples:
        # Instrument all available libraries
        auto_instrument()

        # Instrument only OpenAI and Anthropic
        auto_instrument(libraries=["openai", "anthropic"])

        # Instrument all except LangChain
        auto_instrument(exclude=["langchain"])
    """
    return _registry.auto_instrument(libraries=libraries, exclude=exclude)


def uninstrument_all() -> Dict[str, bool]:
    """Remove instrumentation from all libraries."""
    return _registry.uninstrument_all()


def get_status() -> Dict[str, InstrumentationStatus]:
    """Get instrumentation status for all libraries."""
    return _registry.get_status()


def print_status() -> None:
    """Print instrumentation status in a readable format."""
    _registry.print_status()


def print_health_report() -> None:
    """Print detailed health report for instrumentation."""
    _registry.print_health_report()


def get_health_report() -> Dict[str, any]:
    """Get detailed health report for all instrumentation operations."""
    return _registry.get_health_report()


def reset_all_errors() -> None:
    """Reset all error tracking and circuit breakers."""
    _registry.reset_all_errors()


def instrument(library: str) -> bool:
    """Instrument a specific library."""
    return _registry.instrument_library(library)


def uninstrument(library: str) -> bool:
    """Remove instrumentation from a specific library."""
    return _registry.uninstrument_library(library)