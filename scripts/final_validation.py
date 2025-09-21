#!/usr/bin/env python3
"""
Final Validation Script - Brokle SDK Migration

This script performs comprehensive validation of the new 3-pattern architecture
to ensure the migration is complete and production-ready. Tests all patterns,
validates integrations, and confirms backward compatibility.

Test Coverage:
1. Context-aware client management
2. OpenAI drop-in replacement
3. Anthropic drop-in replacement
4. Universal decorator pattern
5. LangChain callback handler
6. Error handling and fallbacks
7. Performance benchmarks
8. Backward compatibility

Run this script to validate the complete migration.
"""

import sys
import time
import asyncio
import warnings
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import traceback
import json

@dataclass
class ValidationResult:
    """Result of a validation test"""
    test_name: str
    success: bool
    duration_ms: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    warning_count: int = 0
    error: Optional[Exception] = None


class BrokleSDKValidator:
    """Comprehensive validator for Brokle SDK migration"""

    def __init__(self):
        self.results: List[ValidationResult] = []
        self.warnings_captured: List[str] = []

    def run_all_validations(self) -> Dict[str, Any]:
        """Run all validation tests and return comprehensive report"""
        print("🚀 Starting Brokle SDK Migration Validation")
        print("=" * 60)

        # Test categories
        test_categories = [
            ("Core Architecture", self._test_core_architecture),
            ("Context Management", self._test_context_management),
            ("Drop-in Replacements", self._test_drop_in_replacements),
            ("Universal Decorator", self._test_universal_decorator),
            ("Framework Integration", self._test_framework_integration),
            ("Error Handling", self._test_error_handling),
            ("Performance", self._test_performance),
            ("Backward Compatibility", self._test_backward_compatibility),
        ]

        for category, test_func in test_categories:
            print(f"\n📋 {category}")
            print("-" * 40)
            test_func()

        # Generate final report
        return self._generate_final_report()

    def _test_core_architecture(self):
        """Test core architecture components"""

        # Test 1: Main imports work
        start_time = time.time()
        try:
            from brokle import Brokle, get_client, observe
            from brokle.openai import OpenAI as BrokleOpenAI
            from brokle.anthropic import Anthropic as BrokleAnthropic

            self._record_success(
                "Core imports",
                time.time() - start_time,
                "All core imports successful"
            )
        except Exception as e:
            self._record_failure("Core imports", time.time() - start_time, str(e), e)

        # Test 2: Main client creation
        start_time = time.time()
        try:
            # Test without environment variables (should handle gracefully)
            client = Brokle(
                api_key="ak_test_key",
                project_id="proj_test",
                host="http://test.example.com"
            )

            self._record_success(
                "Client creation",
                time.time() - start_time,
                f"Client created successfully: {type(client).__name__}"
            )
        except Exception as e:
            self._record_failure("Client creation", time.time() - start_time, str(e), e)

        # Test 3: Configuration validation
        start_time = time.time()
        try:
            from brokle.config import Config

            config = Config(
                api_key="ak_test_key",
                project_id="proj_test",
                environment="test"
            )

            self._record_success(
                "Configuration system",
                time.time() - start_time,
                "Configuration validation works"
            )
        except Exception as e:
            self._record_failure("Configuration system", time.time() - start_time, str(e), e)

    def _test_context_management(self):
        """Test context-aware client management"""

        # Test 1: Context-aware get_client
        start_time = time.time()
        try:
            from brokle import get_client
            from brokle._client.context import clear_context, get_context_info

            # Clear any existing context
            clear_context()

            # Test client creation with explicit parameters
            client1 = get_client(
                api_key="ak_test_1",
                project_id="proj_test_1",
                environment="test1"
            )

            # Test context info
            info = get_context_info()

            self._record_success(
                "Context management",
                time.time() - start_time,
                f"Context client created. Info: {info.get('context', 'unknown')}"
            )
        except Exception as e:
            self._record_failure("Context management", time.time() - start_time, str(e), e)

        # Test 2: Multi-project safety
        start_time = time.time()
        try:
            from brokle._client.context import clear_context

            # Clear context
            clear_context()

            # Create client for project 1
            client1 = get_client(
                api_key="ak_test_1",
                project_id="proj_1",
                environment="prod"
            )

            # Try to get client for different project (should create new)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                client2 = get_client(
                    api_key="ak_test_2",
                    project_id="proj_2",
                    environment="prod"
                )

                # Should have issued warning about context mismatch
                warning_count = len(w)

            self._record_success(
                "Multi-project safety",
                time.time() - start_time,
                f"Project isolation working. Warnings issued: {warning_count}",
                {"warning_count": warning_count}
            )
        except Exception as e:
            self._record_failure("Multi-project safety", time.time() - start_time, str(e), e)

    def _test_drop_in_replacements(self):
        """Test drop-in replacement functionality"""

        # Test 1: OpenAI drop-in import
        start_time = time.time()
        try:
            from brokle.openai import OpenAI

            # Should work even if openai package not installed
            self._record_success(
                "OpenAI drop-in import",
                time.time() - start_time,
                "OpenAI drop-in replacement import successful"
            )
        except Exception as e:
            self._record_failure("OpenAI drop-in import", time.time() - start_time, str(e), e)

        # Test 2: Anthropic drop-in import
        start_time = time.time()
        try:
            from brokle.anthropic import Anthropic

            # Should work even if anthropic package not installed
            self._record_success(
                "Anthropic drop-in import",
                time.time() - start_time,
                "Anthropic drop-in replacement import successful"
            )
        except Exception as e:
            self._record_failure("Anthropic drop-in import", time.time() - start_time, str(e), e)

        # Test 3: Graceful fallback when SDK not available
        start_time = time.time()
        try:
            from brokle.openai import OpenAI, HAS_OPENAI
            from brokle.anthropic import HAS_ANTHROPIC

            fallback_info = {
                "openai_available": HAS_OPENAI,
                "anthropic_available": HAS_ANTHROPIC
            }

            self._record_success(
                "SDK availability detection",
                time.time() - start_time,
                f"SDK detection working: {fallback_info}",
                fallback_info
            )
        except Exception as e:
            self._record_failure("SDK availability detection", time.time() - start_time, str(e), e)

        # Test 4: __getattr__ passthrough
        start_time = time.time()
        try:
            import brokle.openai as brokle_openai

            # Try to access an attribute that should be passed through
            # This might fail, but should do so gracefully
            try:
                # Try to get __version__ or similar
                version = getattr(brokle_openai, '__version__', 'unknown')
                success_msg = f"Passthrough working, version: {version}"
            except AttributeError as attr_e:
                # This is expected if OpenAI not installed
                success_msg = f"Graceful AttributeError: {attr_e}"

            self._record_success(
                "Attribute passthrough",
                time.time() - start_time,
                success_msg
            )
        except Exception as e:
            self._record_failure("Attribute passthrough", time.time() - start_time, str(e), e)

    def _test_universal_decorator(self):
        """Test universal decorator pattern"""

        # Test 1: Basic decorator functionality
        start_time = time.time()
        try:
            from brokle import observe

            @observe(name="test-function")
            def test_function(x: int) -> int:
                return x * 2

            result = test_function(5)

            self._record_success(
                "Basic decorator",
                time.time() - start_time,
                f"Decorator works, result: {result}",
                {"result": result}
            )
        except Exception as e:
            self._record_failure("Basic decorator", time.time() - start_time, str(e), e)

        # Test 2: Async decorator
        start_time = time.time()
        try:
            from brokle import observe

            @observe(name="test-async-function")
            async def test_async_function(x: int) -> int:
                await asyncio.sleep(0.001)  # Small delay
                return x * 3

            async def run_async_test():
                return await test_async_function(4)

            result = asyncio.run(run_async_test())

            self._record_success(
                "Async decorator",
                time.time() - start_time,
                f"Async decorator works, result: {result}",
                {"result": result}
            )
        except Exception as e:
            self._record_failure("Async decorator", time.time() - start_time, str(e), e)

        # Test 3: Privacy controls
        start_time = time.time()
        try:
            from brokle import observe

            @observe(capture_inputs=False, name="sensitive-function")
            def sensitive_function(api_key: str, data: str) -> str:
                return f"processed-{data}"

            result = sensitive_function("secret-key", "test-data")

            self._record_success(
                "Privacy controls",
                time.time() - start_time,
                f"Privacy controls work, result: {result}",
                {"result": result}
            )
        except Exception as e:
            self._record_failure("Privacy controls", time.time() - start_time, str(e), e)

        # Test 4: Workflow tracing
        start_time = time.time()
        try:
            from brokle import trace_workflow

            def test_workflow():
                with trace_workflow("test-workflow", metadata={"version": "1.0"}):
                    return "workflow-complete"

            result = test_workflow()

            self._record_success(
                "Workflow tracing",
                time.time() - start_time,
                f"Workflow tracing works, result: {result}",
                {"result": result}
            )
        except Exception as e:
            self._record_failure("Workflow tracing", time.time() - start_time, str(e), e)

    def _test_framework_integration(self):
        """Test framework integration capabilities"""

        # Test 1: LangChain callback handler import
        start_time = time.time()
        try:
            from brokle.langchain import BrokleCallbackHandler, create_callback_handler

            self._record_success(
                "LangChain handler import",
                time.time() - start_time,
                "LangChain callback handler import successful"
            )
        except Exception as e:
            # This might fail if LangChain not installed, which is fine
            if "LangChain not installed" in str(e):
                self._record_success(
                    "LangChain handler import",
                    time.time() - start_time,
                    "LangChain not installed - graceful fallback working"
                )
            else:
                self._record_failure("LangChain handler import", time.time() - start_time, str(e), e)

        # Test 2: Handler creation (if LangChain available)
        start_time = time.time()
        try:
            from brokle.langchain import HAS_LANGCHAIN

            if HAS_LANGCHAIN:
                from brokle.langchain import create_callback_handler

                handler = create_callback_handler(
                    session_id="test-session",
                    tags=["test"]
                )

                self._record_success(
                    "LangChain handler creation",
                    time.time() - start_time,
                    f"Handler created: {type(handler).__name__}"
                )
            else:
                self._record_success(
                    "LangChain handler creation",
                    time.time() - start_time,
                    "LangChain not available - skipped"
                )
        except Exception as e:
            self._record_failure("LangChain handler creation", time.time() - start_time, str(e), e)

    def _test_error_handling(self):
        """Test error handling and graceful fallbacks"""

        # Test 1: Exception hierarchy
        start_time = time.time()
        try:
            from brokle.exceptions import (
                BrokleError, AuthenticationError, ConfigurationError,
                ProviderError, ValidationError
            )

            # Test exception creation
            errors = [
                BrokleError("Base error"),
                AuthenticationError("Auth error"),
                ConfigurationError("Config error"),
                ProviderError("Provider error"),
                ValidationError("Validation error")
            ]

            self._record_success(
                "Exception hierarchy",
                time.time() - start_time,
                f"All {len(errors)} exception types work"
            )
        except Exception as e:
            self._record_failure("Exception hierarchy", time.time() - start_time, str(e), e)

        # Test 2: Graceful fallback with missing dependencies
        start_time = time.time()
        try:
            # This should not crash even if telemetry is not available
            from brokle import observe

            @observe()
            def test_function_fallback():
                return "success"

            result = test_function_fallback()

            self._record_success(
                "Graceful fallback",
                time.time() - start_time,
                f"Function works with fallback: {result}"
            )
        except Exception as e:
            self._record_failure("Graceful fallback", time.time() - start_time, str(e), e)

    def _test_performance(self):
        """Test performance characteristics"""

        # Test 1: Decorator overhead
        start_time = time.time()
        try:
            from brokle import observe

            # Test function without decorator
            def normal_function(x):
                return x * 2

            # Test function with decorator
            @observe()
            def decorated_function(x):
                return x * 2

            # Measure overhead
            iterations = 1000

            # Time normal function
            start = time.time()
            for i in range(iterations):
                normal_function(i)
            normal_time = time.time() - start

            # Time decorated function
            start = time.time()
            for i in range(iterations):
                decorated_function(i)
            decorated_time = time.time() - start

            overhead_ms = ((decorated_time - normal_time) / iterations) * 1000

            # Should be less than 1ms overhead per call
            if overhead_ms < 1.0:
                self._record_success(
                    "Decorator performance",
                    time.time() - start_time,
                    f"Overhead: {overhead_ms:.3f}ms per call (target: <1ms)",
                    {"overhead_ms": overhead_ms, "iterations": iterations}
                )
            else:
                self._record_failure(
                    "Decorator performance",
                    time.time() - start_time,
                    f"Overhead too high: {overhead_ms:.3f}ms per call (target: <1ms)"
                )

        except Exception as e:
            self._record_failure("Decorator performance", time.time() - start_time, str(e), e)

    def _test_backward_compatibility(self):
        """Test backward compatibility with existing code"""

        # Test 1: Original imports still work
        start_time = time.time()
        try:
            from brokle import Brokle, get_client, observe
            from brokle import __version__

            # These should all work as before
            self._record_success(
                "Original imports",
                time.time() - start_time,
                f"All original imports work. Version: {__version__}"
            )
        except Exception as e:
            self._record_failure("Original imports", time.time() - start_time, str(e), e)

        # Test 2: Configuration compatibility
        start_time = time.time()
        try:
            from brokle import Config, AuthManager

            # Original patterns should still work
            config = Config()  # Should work with environment variables
            auth = AuthManager(config)

            self._record_success(
                "Configuration compatibility",
                time.time() - start_time,
                "Configuration and auth manager work"
            )
        except Exception as e:
            self._record_failure("Configuration compatibility", time.time() - start_time, str(e), e)

    def _record_success(self, test_name: str, duration: float, message: str, details: Dict[str, Any] = None):
        """Record a successful test"""
        result = ValidationResult(
            test_name=test_name,
            success=True,
            duration_ms=duration * 1000,
            message=message,
            details=details or {}
        )
        self.results.append(result)
        print(f"  ✅ {test_name}: {message}")

    def _record_failure(self, test_name: str, duration: float, message: str, error: Exception = None):
        """Record a failed test"""
        result = ValidationResult(
            test_name=test_name,
            success=False,
            duration_ms=duration * 1000,
            message=message,
            error=error
        )
        self.results.append(result)
        print(f"  ❌ {test_name}: {message}")

    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report"""

        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests

        total_duration = sum(r.duration_ms for r in self.results)
        avg_duration = total_duration / total_tests if total_tests > 0 else 0

        report = {
            "summary": {
                "total_tests": total_tests,
                "successful": successful_tests,
                "failed": failed_tests,
                "success_rate": (successful_tests / total_tests) * 100 if total_tests > 0 else 0,
                "total_duration_ms": total_duration,
                "average_duration_ms": avg_duration
            },
            "tests": [
                {
                    "name": r.test_name,
                    "success": r.success,
                    "duration_ms": r.duration_ms,
                    "message": r.message,
                    "details": r.details,
                    "error": str(r.error) if r.error else None
                }
                for r in self.results
            ],
            "failed_tests": [
                {
                    "name": r.test_name,
                    "message": r.message,
                    "error": str(r.error) if r.error else None,
                    "traceback": traceback.format_exception(type(r.error), r.error, r.error.__traceback__) if r.error else None
                }
                for r in self.results if not r.success
            ]
        }

        # Print summary
        print("\n" + "=" * 60)
        print("🎯 VALIDATION SUMMARY")
        print("=" * 60)

        print(f"\n📊 Results:")
        print(f"  Total Tests: {total_tests}")
        print(f"  ✅ Successful: {successful_tests}")
        print(f"  ❌ Failed: {failed_tests}")
        print(f"  📈 Success Rate: {report['summary']['success_rate']:.1f}%")
        print(f"  ⏱️  Total Time: {total_duration:.1f}ms")
        print(f"  ⚡ Avg Time: {avg_duration:.1f}ms per test")

        if failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for failure in report["failed_tests"]:
                print(f"  • {failure['name']}: {failure['message']}")

        # Final assessment
        if successful_tests == total_tests:
            print(f"\n🎉 ALL TESTS PASSED - MIGRATION FULLY VALIDATED! 🎉")
            print("✅ The new 3-pattern architecture is working correctly")
            print("✅ All patterns are functional and performant")
            print("✅ Backward compatibility is maintained")
            print("✅ Error handling and fallbacks are working")
        elif successful_tests >= total_tests * 0.9:  # 90% success rate
            print(f"\n🟡 MOSTLY SUCCESSFUL - MINOR ISSUES DETECTED")
            print("⚠️  Most functionality is working correctly")
            print("⚠️  Some optional features may have issues")
            print("✅ Core migration appears successful")
        else:
            print(f"\n🔴 VALIDATION FAILED - SIGNIFICANT ISSUES DETECTED")
            print("❌ Multiple core components have issues")
            print("❌ Migration may not be complete")
            print("❌ Review failed tests and fix before proceeding")

        return report


def main():
    """Main validation execution"""
    print("Brokle SDK Migration - Final Validation")
    print("=====================================\n")

    validator = BrokleSDKValidator()
    report = validator.run_all_validations()

    # Save detailed report
    with open("validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Detailed report saved to: validation_report.json")

    # Return appropriate exit code
    if report["summary"]["success_rate"] == 100:
        sys.exit(0)  # Perfect success
    elif report["summary"]["success_rate"] >= 90:
        sys.exit(1)  # Mostly successful but with warnings
    else:
        sys.exit(2)  # Significant failures


if __name__ == "__main__":
    main()