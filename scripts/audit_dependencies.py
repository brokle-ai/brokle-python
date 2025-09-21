#!/usr/bin/env python3
"""
Dependency Audit Script for Brokle SDK Architecture Migration

This script audits the current codebase to understand:
1. External dependencies and their usage patterns
2. Internal module interdependencies
3. Integration framework complexity
4. Provider-specific code locations

Phase 0 validation before architectural changes.
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import importlib.util


@dataclass
class DependencyReference:
    """Tracks where and how a dependency is used"""
    file_path: str
    line_number: int
    import_type: str  # 'import', 'from_import', 'dynamic'
    usage_context: str  # 'class_method', 'function', 'module_level'
    specific_usage: str  # actual code snippet


@dataclass
class ModuleAnalysis:
    """Complete analysis of a Python module"""
    file_path: str
    imports: List[str] = field(default_factory=list)
    external_deps: List[str] = field(default_factory=list)
    internal_deps: List[str] = field(default_factory=list)
    class_definitions: List[str] = field(default_factory=list)
    function_definitions: List[str] = field(default_factory=list)
    complexity_score: int = 0
    lines_of_code: int = 0


class DependencyAuditor:
    """Comprehensive dependency auditor for Brokle SDK"""

    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.brokle_path = self.root_path / "brokle"
        self.dependencies: Dict[str, List[DependencyReference]] = defaultdict(list)
        self.modules: Dict[str, ModuleAnalysis] = {}
        self.integration_complexity = {}

    def audit_with_explicit_tracking(self) -> Dict[str, List[DependencyReference]]:
        """Main audit function with comprehensive tracking"""
        print("🔍 Starting comprehensive dependency audit...")

        # Audit all Python files in brokle package
        for py_file in self.brokle_path.rglob("*.py"):
            if py_file.name == "__pycache__":
                continue

            try:
                self._analyze_file(py_file)
            except Exception as e:
                print(f"⚠️  Error analyzing {py_file}: {e}")

        # Generate reports
        self._generate_dependency_report()
        self._analyze_integration_complexity()
        self._identify_removal_candidates()

        return dict(self.dependencies)

    def _analyze_file(self, file_path: Path):
        """Analyze a single Python file for dependencies"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)
            relative_path = str(file_path.relative_to(self.root_path))

            analysis = ModuleAnalysis(
                file_path=relative_path,
                lines_of_code=len(content.splitlines())
            )

            # Walk AST to find all imports and usage
            for node in ast.walk(tree):
                self._process_ast_node(node, analysis, content.splitlines())

            self.modules[relative_path] = analysis

        except Exception as e:
            print(f"⚠️  Failed to parse {file_path}: {e}")

    def _process_ast_node(self, node: ast.AST, analysis: ModuleAnalysis, lines: List[str]):
        """Process individual AST nodes for dependency tracking"""

        if isinstance(node, ast.Import):
            for alias in node.names:
                self._track_import(alias.name, 'import', node, analysis, lines)

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                full_name = f"{module}.{alias.name}" if module else alias.name
                self._track_import(full_name, 'from_import', node, analysis, lines)

        elif isinstance(node, ast.ClassDef):
            analysis.class_definitions.append(node.name)
            analysis.complexity_score += 2

        elif isinstance(node, ast.FunctionDef):
            analysis.function_definitions.append(node.name)
            analysis.complexity_score += 1

        elif isinstance(node, ast.Call):
            # Track dynamic imports and complex usage
            if isinstance(node.func, ast.Name) and node.func.id == 'importlib':
                analysis.complexity_score += 3

    def _track_import(self, import_name: str, import_type: str, node: ast.AST,
                     analysis: ModuleAnalysis, lines: List[str]):
        """Track specific import usage"""

        # Categorize as external or internal
        if self._is_external_dependency(import_name):
            analysis.external_deps.append(import_name)
            dep_ref = DependencyReference(
                file_path=analysis.file_path,
                line_number=getattr(node, 'lineno', 0),
                import_type=import_type,
                usage_context='module_level',
                specific_usage=lines[getattr(node, 'lineno', 1) - 1] if lines else ""
            )
            self.dependencies[import_name].append(dep_ref)
        else:
            analysis.internal_deps.append(import_name)

        analysis.imports.append(import_name)

    def _is_external_dependency(self, import_name: str) -> bool:
        """Determine if import is external dependency"""
        external_patterns = [
            'openai', 'anthropic', 'google', 'cohere', 'langchain', 'llama_index',
            'opentelemetry', 'pydantic', 'httpx', 'requests', 'wrapt', 'asyncio',
            'typing', 'dataclasses', 'abc', 'contextlib', 'functools', 'inspect',
            'threading', 'logging', 'json', 'time', 'uuid', 'os', 'sys', 'pathlib'
        ]

        return any(import_name.startswith(pattern) for pattern in external_patterns)

    def _generate_dependency_report(self):
        """Generate comprehensive dependency usage report"""
        print("\n📊 DEPENDENCY USAGE REPORT")
        print("=" * 50)

        # Most used external dependencies
        sorted_deps = sorted(self.dependencies.items(),
                           key=lambda x: len(x[1]), reverse=True)

        print("\n🔗 Top External Dependencies:")
        for dep, refs in sorted_deps[:10]:
            print(f"  {dep:<25} : {len(refs):>3} usages")

        # Integration framework usage
        integration_files = [f for f in self.modules.keys()
                           if 'integrations' in f]

        print(f"\n🧩 Integration Framework Files: {len(integration_files)}")
        total_integration_loc = sum(self.modules[f].lines_of_code
                                  for f in integration_files)
        print(f"   Total Lines of Code: {total_integration_loc}")

        # Complex modules (high dependency count)
        complex_modules = sorted(self.modules.values(),
                               key=lambda x: x.complexity_score, reverse=True)

        print("\n⚡ Most Complex Modules:")
        for module in complex_modules[:5]:
            print(f"  {module.file_path:<40} : {module.complexity_score:>3} complexity")

    def _analyze_integration_complexity(self):
        """Analyze complexity of current integration framework"""
        print("\n🔧 INTEGRATION FRAMEWORK ANALYSIS")
        print("=" * 50)

        framework_patterns = {
            'BaseInstrumentation': 0,
            'InstrumentationEngine': 0,
            'auto_instrument': 0,
            'registry': 0,
            'wrapt': 0,
            'monkey_patch': 0
        }

        for module in self.modules.values():
            content_path = self.root_path / module.file_path
            if content_path.exists():
                try:
                    with open(content_path, 'r') as f:
                        content = f.read()
                        for pattern in framework_patterns:
                            framework_patterns[pattern] += content.count(pattern)
                except:
                    continue

        print("\n🏗️  Framework Pattern Usage:")
        for pattern, count in framework_patterns.items():
            if count > 0:
                print(f"  {pattern:<25} : {count:>3} occurrences")

        self.integration_complexity = framework_patterns

    def _identify_removal_candidates(self):
        """Identify files/modules that can be safely removed"""
        print("\n🗑️  REMOVAL CANDIDATES")
        print("=" * 50)

        # Files in integrations directory
        integration_files = [f for f in self.modules.keys()
                           if 'integrations' in f and 'brokle/integrations' in f]

        print(f"\n📁 Integration Framework Files ({len(integration_files)}):")
        total_loc = 0
        for file_path in integration_files:
            module = self.modules[file_path]
            total_loc += module.lines_of_code
            print(f"  {file_path:<50} : {module.lines_of_code:>4} LOC")

        print(f"\n💾 Total removable LOC: {total_loc}")
        print(f"🎯 Complexity reduction: {sum(self.modules[f].complexity_score for f in integration_files)}")

        # Suggest migration strategy
        print("\n📋 MIGRATION STRATEGY:")
        print("  1. Archive integrations/ directory")
        print("  2. Extract reusable utilities to _utils/")
        print("  3. Implement new 3-pattern architecture")
        print("  4. Add comprehensive validation and testing")


def main():
    """Main execution function"""
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
    else:
        # Default to script's parent directory
        root_path = Path(__file__).parent.parent

    print(f"🚀 Auditing Brokle SDK at: {root_path}")

    auditor = DependencyAuditor(str(root_path))
    dependencies = auditor.audit_with_explicit_tracking()

    print(f"\n✅ Audit completed. Found {len(dependencies)} external dependencies.")
    print("📄 See detailed analysis above for migration guidance.")

    return dependencies


if __name__ == "__main__":
    main()