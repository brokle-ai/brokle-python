#!/usr/bin/env python3
"""
Provider Compatibility Matrix for Brokle SDK Architecture Migration

This script validates compatibility between:
1. Provider SDK versions (OpenAI, Anthropic, etc.)
2. Framework versions (LangChain, LlamaIndex, etc.)
3. Brokle instrumentation patterns
4. Python runtime versions

Ensures graceful degradation and version compatibility.
"""

import sys
import importlib
import pkg_resources
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from packaging import version
import subprocess
import json


@dataclass
class VersionInfo:
    """Version information for a package"""
    name: str
    installed: Optional[str] = None
    required_min: Optional[str] = None
    required_max: Optional[str] = None
    compatible: bool = False
    features_supported: List[str] = field(default_factory=list)


@dataclass
class CompatibilityReport:
    """Complete compatibility analysis"""
    providers: Dict[str, VersionInfo] = field(default_factory=dict)
    frameworks: Dict[str, VersionInfo] = field(default_factory=dict)
    runtime_compatible: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class CompatibilityMatrix:
    """Comprehensive compatibility validation for Brokle SDK"""

    # Provider compatibility requirements
    PROVIDER_REQUIREMENTS = {
        'openai': {
            'min_version': '1.0.0',
            'max_version': '2.0.0',
            'drop_in_features': ['chat.completions', 'completions', 'embeddings'],
            'breaking_changes': {
                '1.0.0': 'Async client API changes',
                '1.2.0': 'Response format updates'
            }
        },
        'anthropic': {
            'min_version': '0.5.0',
            'max_version': '1.0.0',
            'drop_in_features': ['messages.create', 'completions.create'],
            'breaking_changes': {
                '0.7.0': 'Message format changes'
            }
        },
        'google-generativeai': {
            'min_version': '0.3.0',
            'max_version': '1.0.0',
            'drop_in_features': ['chat', 'generate_text'],
            'breaking_changes': {}
        },
        'cohere': {
            'min_version': '4.0.0',
            'max_version': '5.0.0',
            'drop_in_features': ['chat', 'generate'],
            'breaking_changes': {}
        }
    }

    # Framework compatibility requirements
    FRAMEWORK_REQUIREMENTS = {
        'langchain': {
            'min_version': '0.1.0',
            'max_version': '1.0.0',
            'integration_method': 'callback_handler',
            'supported_chains': ['llm_chain', 'conversation_chain', 'sequential_chain']
        },
        'langchain-community': {
            'min_version': '0.0.10',
            'max_version': '1.0.0',
            'integration_method': 'callback_handler',
            'supported_chains': ['retrieval_qa', 'stuff_documents_chain']
        },
        'llama-index': {
            'min_version': '0.9.0',
            'max_version': '1.0.0',
            'integration_method': 'callback_manager',
            'supported_features': ['query_engine', 'chat_engine']
        }
    }

    def __init__(self):
        self.report = CompatibilityReport()

    def validate_complete_compatibility(self) -> CompatibilityReport:
        """Run comprehensive compatibility validation"""
        print("🔍 Starting provider compatibility validation...")

        # Check Python runtime
        self._validate_python_runtime()

        # Validate all providers
        for provider_name, requirements in self.PROVIDER_REQUIREMENTS.items():
            self._validate_provider_compatibility(provider_name, requirements)

        # Validate all frameworks
        for framework_name, requirements in self.FRAMEWORK_REQUIREMENTS.items():
            self._validate_framework_compatibility(framework_name, requirements)

        # Generate recommendations
        self._generate_recommendations()

        # Print summary
        self._print_compatibility_summary()

        return self.report

    def _validate_python_runtime(self):
        """Validate Python runtime compatibility"""
        python_version = sys.version_info
        min_required = (3, 8)
        max_supported = (3, 12)

        if python_version < min_required:
            self.report.runtime_compatible = False
            self.report.errors.append(
                f"Python {python_version.major}.{python_version.minor} not supported. "
                f"Minimum required: {min_required[0]}.{min_required[1]}"
            )
        elif python_version > max_supported:
            self.report.warnings.append(
                f"Python {python_version.major}.{python_version.minor} not tested. "
                f"Latest tested: {max_supported[0]}.{max_supported[1]}"
            )

    def _validate_provider_compatibility(self, provider_name: str, requirements: Dict):
        """Validate specific provider compatibility"""
        version_info = VersionInfo(
            name=provider_name,
            required_min=requirements['min_version'],
            required_max=requirements['max_version'],
            features_supported=requirements['drop_in_features']
        )

        try:
            # Check if package is installed
            installed_version = self._get_installed_version(provider_name)
            version_info.installed = installed_version

            if installed_version:
                # Check version compatibility
                min_ver = version.parse(requirements['min_version'])
                max_ver = version.parse(requirements['max_version'])
                installed_ver = version.parse(installed_version)

                version_info.compatible = min_ver <= installed_ver < max_ver

                if not version_info.compatible:
                    if installed_ver < min_ver:
                        self.report.errors.append(
                            f"{provider_name} {installed_version} too old. "
                            f"Minimum required: {requirements['min_version']}"
                        )
                    else:
                        self.report.warnings.append(
                            f"{provider_name} {installed_version} may be incompatible. "
                            f"Latest tested: {requirements['max_version']}"
                        )

                # Check for breaking changes
                for breaking_version, description in requirements.get('breaking_changes', {}).items():
                    if installed_ver >= version.parse(breaking_version):
                        self.report.warnings.append(
                            f"{provider_name}: {description} (since v{breaking_version})"
                        )

        except Exception as e:
            self.report.warnings.append(f"Could not validate {provider_name}: {str(e)}")

        self.report.providers[provider_name] = version_info

    def _validate_framework_compatibility(self, framework_name: str, requirements: Dict):
        """Validate framework compatibility"""
        version_info = VersionInfo(
            name=framework_name,
            required_min=requirements['min_version'],
            required_max=requirements['max_version']
        )

        try:
            installed_version = self._get_installed_version(framework_name)
            version_info.installed = installed_version

            if installed_version:
                min_ver = version.parse(requirements['min_version'])
                max_ver = version.parse(requirements['max_version'])
                installed_ver = version.parse(installed_version)

                version_info.compatible = min_ver <= installed_ver < max_ver

                if not version_info.compatible:
                    self.report.warnings.append(
                        f"{framework_name} {installed_version} may need custom integration"
                    )

        except Exception as e:
            # Framework not installed is often fine
            self.report.warnings.append(f"Framework {framework_name} not installed")

        self.report.frameworks[framework_name] = version_info

    def _get_installed_version(self, package_name: str) -> Optional[str]:
        """Get installed version of a package"""
        try:
            return pkg_resources.get_distribution(package_name).version
        except pkg_resources.DistributionNotFound:
            return None
        except Exception:
            return None

    def _generate_recommendations(self):
        """Generate actionable recommendations"""
        self.report.recommendations.append("🎯 MIGRATION RECOMMENDATIONS:")

        # Provider recommendations
        for provider_name, info in self.report.providers.items():
            if info.installed and not info.compatible:
                if info.installed and info.required_min:
                    installed_ver = version.parse(info.installed)
                    required_ver = version.parse(info.required_min)
                    if installed_ver < required_ver:
                        self.report.recommendations.append(
                            f"  📦 Upgrade {provider_name}: pip install '{provider_name}>={info.required_min}'"
                        )

        # Drop-in replacement safety
        safe_providers = [name for name, info in self.report.providers.items()
                         if info.installed and info.compatible]

        if safe_providers:
            self.report.recommendations.append(
                f"  ✅ Safe for drop-in replacement: {', '.join(safe_providers)}"
            )

        # Framework integration
        installed_frameworks = [name for name, info in self.report.frameworks.items()
                              if info.installed]

        if installed_frameworks:
            self.report.recommendations.append(
                f"  🔗 Frameworks requiring callback integration: {', '.join(installed_frameworks)}"
            )

    def _print_compatibility_summary(self):
        """Print comprehensive compatibility summary"""
        print("\n📊 PROVIDER COMPATIBILITY MATRIX")
        print("=" * 60)

        # Provider status
        print("\n🏢 AI Providers:")
        for name, info in self.report.providers.items():
            status = "✅" if info.compatible else "⚠️" if info.installed else "➖"
            installed = info.installed or "Not installed"
            required = f"{info.required_min} - {info.required_max}"
            print(f"  {status} {name:<20} : {installed:<15} (req: {required})")

        # Framework status
        print("\n🧩 Frameworks:")
        for name, info in self.report.frameworks.items():
            status = "✅" if info.compatible else "⚠️" if info.installed else "➖"
            installed = info.installed or "Not installed"
            print(f"  {status} {name:<20} : {installed}")

        # Summary counts
        compatible_providers = sum(1 for info in self.report.providers.values() if info.compatible)
        total_providers = len(self.report.providers)

        print(f"\n📈 COMPATIBILITY SUMMARY:")
        print(f"  🎯 Compatible providers: {compatible_providers}/{total_providers}")
        print(f"  ⚠️  Warnings: {len(self.report.warnings)}")
        print(f"  ❌ Errors: {len(self.report.errors)}")

        # Print warnings and errors
        if self.report.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.report.warnings:
                print(f"    {warning}")

        if self.report.errors:
            print("\n❌ ERRORS:")
            for error in self.report.errors:
                print(f"    {error}")

        # Print recommendations
        if self.report.recommendations:
            print("\n" + "\n".join(self.report.recommendations))

    def generate_installation_matrix(self) -> Dict[str, str]:
        """Generate pip install commands for compatible versions"""
        install_commands = {}

        for provider_name, info in self.report.providers.items():
            if info.required_min and info.required_max:
                install_commands[provider_name] = (
                    f"pip install '{provider_name}>={info.required_min},<{info.required_max}'"
                )

        return install_commands

    def validate_drop_in_safety(self, provider_name: str) -> Tuple[bool, List[str]]:
        """Validate if drop-in replacement is safe for specific provider"""
        safety_checks = []
        is_safe = True

        provider_info = self.report.providers.get(provider_name)
        if not provider_info:
            return False, ["Provider not found in compatibility matrix"]

        if not provider_info.installed:
            return False, ["Provider not installed"]

        if not provider_info.compatible:
            is_safe = False
            safety_checks.append("Version compatibility issues detected")

        # Check for known breaking changes
        requirements = self.PROVIDER_REQUIREMENTS.get(provider_name, {})
        breaking_changes = requirements.get('breaking_changes', {})

        if provider_info.installed:
            installed_ver = version.parse(provider_info.installed)
            for breaking_version, description in breaking_changes.items():
                if installed_ver >= version.parse(breaking_version):
                    safety_checks.append(f"Breaking change: {description}")

        return is_safe, safety_checks


def main():
    """Main execution function"""
    print("🚀 Starting comprehensive compatibility validation...")

    matrix = CompatibilityMatrix()
    report = matrix.validate_complete_compatibility()

    # Generate installation matrix
    install_commands = matrix.generate_installation_matrix()

    print("\n📋 INSTALLATION COMMANDS:")
    for provider, command in install_commands.items():
        print(f"  {command}")

    # Validate drop-in safety for key providers
    print("\n🛡️  DROP-IN REPLACEMENT SAFETY:")
    for provider in ['openai', 'anthropic']:
        is_safe, checks = matrix.validate_drop_in_safety(provider)
        status = "✅ SAFE" if is_safe else "⚠️  CAUTION"
        print(f"  {status}: {provider}")
        for check in checks:
            print(f"    - {check}")

    print("\n✅ Compatibility validation completed!")
    return report


if __name__ == "__main__":
    main()