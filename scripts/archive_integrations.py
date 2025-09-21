#!/usr/bin/env python3
"""
Archive Integrations Directory - Brokle SDK Migration

This script safely archives the existing integrations framework while
extracting reusable utilities for the new 3-pattern architecture.

Steps:
1. Create timestamped backup of integrations/
2. Analyze code for reusable utilities
3. Extract useful patterns and utilities to _utils/
4. Generate migration report
5. Safely remove old framework

Ensures zero data loss and provides clear migration path.
"""

import os
import shutil
import ast
import time
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
import json


@dataclass
class UtilityFunction:
    """Represents a reusable utility function"""
    name: str
    file_path: str
    line_start: int
    line_end: int
    source_code: str
    dependencies: List[str] = field(default_factory=list)
    category: str = ""
    reusability_score: int = 0


@dataclass
class ArchiveReport:
    """Report of archival process"""
    timestamp: str
    files_archived: List[str] = field(default_factory=list)
    utilities_extracted: List[UtilityFunction] = field(default_factory=list)
    patterns_identified: Dict[str, List[str]] = field(default_factory=dict)
    total_loc_archived: int = 0
    reusable_loc: int = 0


class IntegrationsArchiver:
    """Comprehensive archiver for integrations framework"""

    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.integrations_path = self.root_path / "brokle" / "integrations"
        self.utils_path = self.root_path / "brokle" / "_utils"
        self.archive_path = self.root_path / "archived_integrations"
        self.report = ArchiveReport(timestamp=time.strftime("%Y%m%d_%H%M%S"))

    def archive_with_utility_extraction(self) -> ArchiveReport:
        """Main archival process with utility extraction"""
        print("🗂️  Starting integrations archival process...")

        # Step 1: Analyze for reusable utilities
        self._analyze_reusable_utilities()

        # Step 2: Create timestamped backup
        self._create_timestamped_backup()

        # Step 3: Extract utilities to _utils/
        self._extract_utilities()

        # Step 4: Generate comprehensive report
        self._generate_archive_report()

        # Step 5: Prepare for safe removal (but don't remove yet)
        self._prepare_removal_plan()

        print(f"✅ Archival completed. Report saved to archive_report.json")
        return self.report

    def _analyze_reusable_utilities(self):
        """Analyze integrations code for reusable utilities"""
        print("🔍 Analyzing code for reusable utilities...")

        utility_categories = {
            'decorators': ['decorator', 'wrapper', 'trace'],
            'http_utils': ['request', 'response', 'http', 'client'],
            'validation': ['validate', 'check', 'verify'],
            'telemetry': ['span', 'metric', 'log', 'otel'],
            'error_handling': ['error', 'exception', 'handle'],
            'provider_utils': ['provider', 'api', 'endpoint'],
            'caching': ['cache', 'redis', 'memory'],
            'async_utils': ['async', 'await', 'asyncio']
        }

        for py_file in self.integrations_path.rglob("*.py"):
            if py_file.name.startswith("__"):
                continue

            try:
                utilities = self._extract_utilities_from_file(py_file, utility_categories)
                self.report.utilities_extracted.extend(utilities)
            except Exception as e:
                print(f"⚠️  Error analyzing {py_file}: {e}")

        # Sort utilities by reusability score
        self.report.utilities_extracted.sort(key=lambda u: u.reusability_score, reverse=True)

        print(f"📊 Found {len(self.report.utilities_extracted)} potentially reusable utilities")

    def _extract_utilities_from_file(self, file_path: Path, categories: Dict[str, List[str]]) -> List[UtilityFunction]:
        """Extract utilities from a single file"""
        utilities = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    utility = self._analyze_function_node(node, file_path, lines, categories)
                    if utility:
                        utilities.append(utility)

        except Exception as e:
            print(f"⚠️  Error parsing {file_path}: {e}")

        return utilities

    def _analyze_function_node(self, node: ast.FunctionDef, file_path: Path,
                             lines: List[str], categories: Dict[str, List[str]]) -> Optional[UtilityFunction]:
        """Analyze a function node for reusability"""

        # Skip private functions (unless they're clearly utilities)
        if node.name.startswith('_') and not any(keyword in node.name.lower()
                                               for keywords in categories.values()
                                               for keyword in keywords):
            return None

        # Extract function source
        line_start = node.lineno - 1
        line_end = node.end_lineno if hasattr(node, 'end_lineno') else line_start + 10

        source_lines = lines[line_start:line_end]
        source_code = '\n'.join(source_lines)

        # Categorize function
        category = self._categorize_function(node.name, source_code, categories)

        # Calculate reusability score
        reusability_score = self._calculate_reusability_score(node, source_code)

        # Skip low-scoring utilities
        if reusability_score < 3:
            return None

        # Extract dependencies
        dependencies = self._extract_dependencies(node)

        return UtilityFunction(
            name=node.name,
            file_path=str(file_path.relative_to(self.root_path)),
            line_start=line_start + 1,
            line_end=line_end,
            source_code=source_code,
            dependencies=dependencies,
            category=category,
            reusability_score=reusability_score
        )

    def _categorize_function(self, func_name: str, source_code: str,
                           categories: Dict[str, List[str]]) -> str:
        """Categorize function based on name and content"""
        func_name_lower = func_name.lower()
        source_lower = source_code.lower()

        for category, keywords in categories.items():
            if any(keyword in func_name_lower for keyword in keywords):
                return category
            if any(keyword in source_lower for keyword in keywords):
                return category

        return "misc"

    def _calculate_reusability_score(self, node: ast.FunctionDef, source_code: str) -> int:
        """Calculate reusability score for a function"""
        score = 0

        # Base score for non-private functions
        if not node.name.startswith('_'):
            score += 2

        # Points for having docstring
        if ast.get_docstring(node):
            score += 2

        # Points for having type hints
        if node.returns or any(arg.annotation for arg in node.args.args):
            score += 1

        # Points for utility keywords
        utility_keywords = ['util', 'helper', 'decorator', 'wrapper', 'validate', 'format']
        if any(keyword in node.name.lower() for keyword in utility_keywords):
            score += 2

        # Points for being self-contained (few dependencies)
        external_deps = sum(1 for line in source_code.split('\n')
                          if line.strip().startswith('from ') and 'brokle' not in line)
        if external_deps <= 2:
            score += 1

        # Points for reasonable length (not too short, not too long)
        line_count = len(source_code.split('\n'))
        if 5 <= line_count <= 50:
            score += 1

        return score

    def _extract_dependencies(self, node: ast.FunctionDef) -> List[str]:
        """Extract dependencies from function node"""
        dependencies = []

        for child in ast.walk(node):
            if isinstance(child, ast.Import):
                for alias in child.names:
                    dependencies.append(alias.name)
            elif isinstance(child, ast.ImportFrom) and child.module:
                dependencies.append(child.module)

        return list(set(dependencies))

    def _create_timestamped_backup(self):
        """Create timestamped backup of integrations directory"""
        print("💾 Creating timestamped backup...")

        backup_name = f"integrations_backup_{self.report.timestamp}"
        backup_path = self.archive_path / backup_name

        # Create archive directory
        self.archive_path.mkdir(exist_ok=True)

        # Copy entire integrations directory
        shutil.copytree(self.integrations_path, backup_path)

        # Count files and lines
        for py_file in backup_path.rglob("*.py"):
            self.report.files_archived.append(str(py_file.relative_to(backup_path)))
            try:
                with open(py_file, 'r') as f:
                    self.report.total_loc_archived += len(f.readlines())
            except:
                pass

        print(f"📁 Backup created at: {backup_path}")
        print(f"📊 Archived {len(self.report.files_archived)} files, {self.report.total_loc_archived} LOC")

    def _extract_utilities(self):
        """Extract high-value utilities to _utils/ directory"""
        print("🔧 Extracting reusable utilities...")

        # Create _utils directory structure
        self.utils_path.mkdir(exist_ok=True)
        (self.utils_path / "__init__.py").touch()

        # Group utilities by category
        by_category = {}
        for utility in self.report.utilities_extracted:
            if utility.reusability_score >= 5:  # Only extract high-value utilities
                category = utility.category
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(utility)

        # Create category files
        for category, utilities in by_category.items():
            self._create_utility_file(category, utilities)

        print(f"🎯 Extracted {sum(len(utils) for utils in by_category.values())} high-value utilities")

    def _create_utility_file(self, category: str, utilities: List[UtilityFunction]):
        """Create a utility file for a specific category"""
        file_path = self.utils_path / f"{category}.py"

        header = f'''"""
{category.replace('_', ' ').title()} utilities extracted from integrations framework.

This file contains reusable utilities that were identified during the
migration from the old integrations framework to the new 3-pattern architecture.

Auto-generated on {self.report.timestamp}
"""

'''

        imports = set()
        functions = []

        for utility in utilities:
            # Add utility source with metadata
            func_source = f'''
# Extracted from: {utility.file_path}:{utility.line_start}
# Reusability score: {utility.reusability_score}
# Dependencies: {', '.join(utility.dependencies)}
{utility.source_code}

'''
            functions.append(func_source)

            # Collect imports
            imports.update(utility.dependencies)

        # Write file
        with open(file_path, 'w') as f:
            f.write(header)

            # Write imports
            if imports:
                f.write("# Dependencies\n")
                for imp in sorted(imports):
                    if not imp.startswith('brokle'):
                        f.write(f"import {imp}\n")
                f.write("\n")

            # Write functions
            f.write("".join(functions))

        print(f"📄 Created {file_path} with {len(utilities)} utilities")

    def _generate_archive_report(self):
        """Generate comprehensive archive report"""
        print("📄 Generating archive report...")

        # Calculate reusable LOC
        self.report.reusable_loc = sum(
            len(u.source_code.split('\n'))
            for u in self.report.utilities_extracted
            if u.reusability_score >= 5
        )

        # Identify patterns
        self.report.patterns_identified = {
            "high_value_utilities": [u.name for u in self.report.utilities_extracted if u.reusability_score >= 7],
            "decorator_patterns": [u.name for u in self.report.utilities_extracted if u.category == "decorators"],
            "telemetry_utils": [u.name for u in self.report.utilities_extracted if u.category == "telemetry"],
            "provider_patterns": [u.name for u in self.report.utilities_extracted if u.category == "provider_utils"]
        }

        # Save report as JSON
        report_file = self.root_path / "archive_report.json"
        with open(report_file, 'w') as f:
            json.dump({
                "timestamp": self.report.timestamp,
                "files_archived": self.report.files_archived,
                "total_loc_archived": self.report.total_loc_archived,
                "reusable_loc": self.report.reusable_loc,
                "utilities_count": len(self.report.utilities_extracted),
                "high_value_utilities": len([u for u in self.report.utilities_extracted if u.reusability_score >= 5]),
                "patterns_identified": self.report.patterns_identified,
                "utilities_by_category": {
                    cat: len([u for u in self.report.utilities_extracted if u.category == cat])
                    for cat in set(u.category for u in self.report.utilities_extracted)
                }
            }, f, indent=2)

    def _prepare_removal_plan(self):
        """Prepare plan for safe removal (but don't execute)"""
        print("📋 Preparing removal plan...")

        removal_plan = {
            "files_to_remove": [str(p.relative_to(self.root_path)) for p in self.integrations_path.rglob("*.py")],
            "backup_location": str(self.archive_path),
            "utilities_extracted": len(self.report.utilities_extracted),
            "safe_to_remove": True,
            "removal_command": f"rm -rf {self.integrations_path}",
            "recovery_command": f"cp -r {self.archive_path}/integrations_backup_{self.report.timestamp}/* {self.integrations_path}/"
        }

        with open(self.root_path / "removal_plan.json", 'w') as f:
            json.dump(removal_plan, f, indent=2)

        print("⚠️  Removal plan created but NOT executed")
        print("🔄 To proceed with removal, run the command from removal_plan.json")

    def print_archive_summary(self):
        """Print human-readable archive summary"""
        print("\n📊 INTEGRATION ARCHIVAL SUMMARY")
        print("=" * 50)

        print(f"\n💾 Archive Details:")
        print(f"  📁 Files archived: {len(self.report.files_archived)}")
        print(f"  📄 Total LOC archived: {self.report.total_loc_archived}")
        print(f"  🎯 Reusable LOC extracted: {self.report.reusable_loc}")

        print(f"\n🔧 Utilities Extracted:")
        print(f"  📦 Total utilities found: {len(self.report.utilities_extracted)}")
        print(f"  ⭐ High-value utilities: {len([u for u in self.report.utilities_extracted if u.reusability_score >= 5])}")

        print(f"\n📂 Utilities by Category:")
        by_category = {}
        for u in self.report.utilities_extracted:
            by_category[u.category] = by_category.get(u.category, 0) + 1

        for category, count in sorted(by_category.items()):
            print(f"  {category:<15}: {count:>3} utilities")

        print(f"\n🔝 Top Utilities (by reusability score):")
        top_utilities = sorted(self.report.utilities_extracted,
                             key=lambda u: u.reusability_score, reverse=True)[:5]
        for utility in top_utilities:
            print(f"  {utility.name:<25} : {utility.reusability_score:>2} points ({utility.category})")


def main():
    """Main execution function"""
    root_path = Path(__file__).parent.parent
    print(f"🚀 Starting integrations archival at: {root_path}")

    archiver = IntegrationsArchiver(str(root_path))
    report = archiver.archive_with_utility_extraction()

    archiver.print_archive_summary()

    print(f"\n✅ Archival process completed successfully!")
    print(f"📄 Detailed report saved to: archive_report.json")
    print(f"💾 Backup location: archived_integrations/")
    print(f"🔧 Extracted utilities: brokle/_utils/")

    return report


if __name__ == "__main__":
    main()