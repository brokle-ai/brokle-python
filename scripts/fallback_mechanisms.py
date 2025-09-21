#!/usr/bin/env python3
"""
Graceful Fallback Mechanisms for Brokle SDK Architecture Migration

This module provides comprehensive fallback strategies to ensure zero-downtime
migration from the old integration framework to the new 3-pattern architecture.

Key Features:
1. Import fallback detection and graceful degradation
2. Version compatibility validation with warnings
3. Feature availability detection with alternatives
4. Runtime safety with telemetry and logging
5. Production-ready error handling and recovery
"""

import logging
import sys
import traceback
import functools
import importlib
from typing import Any, Dict, List, Optional, Callable, Union, Type
from dataclasses import dataclass, field
import warnings
from contextlib import contextmanager
import time


# Configure fallback logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior"""
    enable_warnings: bool = True
    enable_telemetry: bool = True
    strict_mode: bool = False  # Fail fast vs graceful degradation
    timeout_seconds: float = 30.0
    max_retries: int = 3
    fallback_cache_ttl: int = 300  # 5 minutes


@dataclass
class FallbackReport:
    """Report of fallback actions taken"""
    timestamp: float = field(default_factory=time.time)
    fallbacks_triggered: List[str] = field(default_factory=list)
    warnings_issued: List[str] = field(default_factory=list)
    errors_handled: List[str] = field(default_factory=list)
    performance_impact: Dict[str, float] = field(default_factory=dict)


class GracefulFallbackManager:
    """
    Comprehensive fallback management for SDK migration.

    Handles import failures, version incompatibilities, and feature degradation
    with production-safe recovery mechanisms.
    """

    def __init__(self, config: Optional[FallbackConfig] = None):
        self.config = config or FallbackConfig()
        self.report = FallbackReport()
        self._import_cache: Dict[str, Any] = {}
        self._fallback_registry: Dict[str, Callable] = {}

    def safe_import(self,
                   module_name: str,
                   fallback: Optional[Callable] = None,
                   required: bool = False) -> Any:
        """
        Safely import a module with fallback options.

        Args:
            module_name: Module to import (e.g., 'brokle.integrations.openai')
            fallback: Fallback function to call if import fails
            required: Whether to raise exception if both import and fallback fail

        Returns:
            Imported module or fallback result
        """
        # Check cache first
        cache_key = f"import:{module_name}"
        if cache_key in self._import_cache:
            return self._import_cache[cache_key]

        try:
            module = importlib.import_module(module_name)
            self._import_cache[cache_key] = module
            return module

        except ImportError as e:
            self._handle_import_failure(module_name, e, fallback, required)

            if fallback:
                try:
                    result = fallback()
                    self._import_cache[cache_key] = result
                    return result
                except Exception as fallback_error:
                    self._handle_fallback_failure(module_name, fallback_error, required)

            if required:
                raise ImportError(f"Required module {module_name} not available and no fallback provided")

            return None

    def _handle_import_failure(self, module_name: str, error: ImportError,
                             fallback: Optional[Callable], required: bool):
        """Handle import failures with appropriate logging and telemetry"""
        message = f"Failed to import {module_name}: {error}"
        self.report.errors_handled.append(message)

        if self.config.enable_warnings:
            if fallback:
                warnings.warn(f"Using fallback for {module_name}: {error}", UserWarning)
            else:
                warnings.warn(f"Module {module_name} not available: {error}", UserWarning)

        if self.config.enable_telemetry:
            logger.warning(message, extra={
                'module_name': module_name,
                'error_type': type(error).__name__,
                'has_fallback': fallback is not None,
                'required': required
            })

    def _handle_fallback_failure(self, module_name: str, error: Exception, required: bool):
        """Handle fallback failures"""
        message = f"Fallback for {module_name} also failed: {error}"
        self.report.errors_handled.append(message)

        if self.config.enable_telemetry:
            logger.error(message, extra={
                'module_name': module_name,
                'fallback_error': str(error),
                'required': required
            })

    def with_fallback(self,
                     primary_func: Callable,
                     fallback_func: Optional[Callable] = None,
                     error_types: tuple = (Exception,),
                     max_retries: Optional[int] = None) -> Callable:
        """
        Decorator to add fallback behavior to any function.

        Args:
            primary_func: Primary function to attempt
            fallback_func: Fallback function if primary fails
            error_types: Tuple of exception types to catch
            max_retries: Maximum retry attempts

        Returns:
            Decorated function with fallback behavior
        """
        retries = max_retries or self.config.max_retries

        @functools.wraps(primary_func)
        def wrapper(*args, **kwargs):
            last_error = None

            for attempt in range(retries + 1):
                try:
                    start_time = time.time()
                    result = primary_func(*args, **kwargs)

                    # Track performance
                    duration = time.time() - start_time
                    self.report.performance_impact[f"{primary_func.__name__}_success"] = duration

                    return result

                except error_types as e:
                    last_error = e

                    if attempt < retries:
                        self._log_retry(primary_func.__name__, attempt + 1, e)
                        time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                        continue

                    # All retries exhausted, try fallback
                    if fallback_func:
                        return self._execute_fallback(
                            primary_func.__name__, fallback_func,
                            args, kwargs, e
                        )

                    # No fallback available
                    self._handle_final_failure(primary_func.__name__, e)

                    if self.config.strict_mode:
                        raise

                    return None

            return None

        return wrapper

    def _log_retry(self, func_name: str, attempt: int, error: Exception):
        """Log retry attempts"""
        if self.config.enable_telemetry:
            logger.info(f"Retrying {func_name} (attempt {attempt}): {error}")

    def _execute_fallback(self, func_name: str, fallback_func: Callable,
                         args: tuple, kwargs: dict, primary_error: Exception) -> Any:
        """Execute fallback function and track metrics"""
        try:
            start_time = time.time()
            result = fallback_func(*args, **kwargs)

            # Track fallback usage
            duration = time.time() - start_time
            self.report.performance_impact[f"{func_name}_fallback"] = duration
            self.report.fallbacks_triggered.append(f"{func_name} -> {fallback_func.__name__}")

            if self.config.enable_warnings:
                warnings.warn(f"Using fallback for {func_name}: {primary_error}", UserWarning)

            return result

        except Exception as fallback_error:
            self._handle_fallback_failure(func_name, fallback_error, required=True)
            if self.config.strict_mode:
                raise
            return None

    def _handle_final_failure(self, func_name: str, error: Exception):
        """Handle final failure when all options exhausted"""
        message = f"All attempts failed for {func_name}: {error}"
        self.report.errors_handled.append(message)

        if self.config.enable_telemetry:
            logger.error(message, extra={
                'function_name': func_name,
                'error_type': type(error).__name__,
                'traceback': traceback.format_exc()
            })

    @contextmanager
    def compatibility_mode(self, feature_name: str):
        """
        Context manager for compatibility mode operations.

        Provides enhanced error handling and telemetry for migration scenarios.
        """
        start_time = time.time()

        try:
            if self.config.enable_telemetry:
                logger.info(f"Entering compatibility mode for {feature_name}")

            yield self

        except Exception as e:
            self._handle_compatibility_error(feature_name, e)
            if self.config.strict_mode:
                raise

        finally:
            duration = time.time() - start_time
            self.report.performance_impact[f"compatibility_{feature_name}"] = duration

            if self.config.enable_telemetry:
                logger.info(f"Exiting compatibility mode for {feature_name} (duration: {duration:.3f}s)")

    def _handle_compatibility_error(self, feature_name: str, error: Exception):
        """Handle errors in compatibility mode"""
        message = f"Compatibility mode error for {feature_name}: {error}"
        self.report.errors_handled.append(message)

        if self.config.enable_telemetry:
            logger.error(message, extra={
                'feature_name': feature_name,
                'error_type': type(error).__name__
            })

    def register_fallback(self, primary_name: str, fallback_func: Callable):
        """Register a fallback function for a specific primary function"""
        self._fallback_registry[primary_name] = fallback_func

    def get_fallback_report(self) -> FallbackReport:
        """Get comprehensive report of all fallback actions"""
        return self.report

    def print_fallback_summary(self):
        """Print human-readable summary of fallback usage"""
        print("\n🛡️  GRACEFUL FALLBACK SUMMARY")
        print("=" * 50)

        print(f"\n📊 Fallbacks Triggered: {len(self.report.fallbacks_triggered)}")
        for fallback in self.report.fallbacks_triggered:
            print(f"  ↳ {fallback}")

        print(f"\n⚠️  Warnings Issued: {len(self.report.warnings_issued)}")
        for warning in self.report.warnings_issued:
            print(f"  ⚠️  {warning}")

        print(f"\n❌ Errors Handled: {len(self.report.errors_handled)}")
        for error in self.report.errors_handled:
            print(f"  ❌ {error}")

        print(f"\n⏱️  Performance Impact:")
        for operation, duration in self.report.performance_impact.items():
            print(f"  {operation:<30}: {duration:.3f}s")


# Global fallback manager instance
_global_fallback_manager = GracefulFallbackManager()


def safe_import(module_name: str, fallback: Optional[Callable] = None, required: bool = False) -> Any:
    """Global function for safe imports"""
    return _global_fallback_manager.safe_import(module_name, fallback, required)


def with_fallback(fallback_func: Optional[Callable] = None,
                 error_types: tuple = (Exception,)) -> Callable:
    """Global decorator for fallback behavior"""
    def decorator(func):
        return _global_fallback_manager.with_fallback(func, fallback_func, error_types)
    return decorator


@contextmanager
def compatibility_mode(feature_name: str):
    """Global context manager for compatibility mode"""
    with _global_fallback_manager.compatibility_mode(feature_name):
        yield


def get_fallback_report() -> FallbackReport:
    """Get global fallback report"""
    return _global_fallback_manager.get_fallback_report()


def print_fallback_summary():
    """Print global fallback summary"""
    _global_fallback_manager.print_fallback_summary()


# Example usage and testing functions
def demo_fallback_mechanisms():
    """Demonstrate fallback mechanisms"""
    print("🚀 Demonstrating graceful fallback mechanisms...")

    # Test 1: Safe import with fallback
    def create_stub_openai():
        """Stub OpenAI client for fallback"""
        class StubOpenAI:
            def __init__(self, *args, **kwargs):
                print("⚠️  Using stub OpenAI client (original not available)")

            def chat_completions_create(self, *args, **kwargs):
                return {"choices": [{"message": {"content": "Fallback response"}}]}

        return StubOpenAI

    # Try to import OpenAI with fallback
    openai_module = safe_import('openai', fallback=create_stub_openai)
    print(f"OpenAI import result: {type(openai_module)}")

    # Test 2: Function with fallback
    @with_fallback(fallback_func=lambda x: f"Fallback result for {x}")
    def risky_function(value):
        if value == "fail":
            raise ValueError("Intentional failure")
        return f"Success: {value}"

    print(f"Risky function (success): {risky_function('test')}")
    print(f"Risky function (fallback): {risky_function('fail')}")

    # Test 3: Compatibility mode
    with compatibility_mode("migration_test"):
        print("Operating in compatibility mode...")
        # Simulate some migration logic here
        pass

    # Print summary
    print_fallback_summary()


if __name__ == "__main__":
    demo_fallback_mechanisms()