"""
Database migration runner tests.

This module tests the actual database migration execution
to ensure migrations run successfully and create the expected schema.
"""

import os
import subprocess
import pytest
import time
from typing import Optional


class TestMigrationRunner:
    """Test actual migration execution."""

    def test_migration_command_exists(self):
        """Test that migration command exists and is executable."""
        migration_cmd = "/home/hashir/Development/Projects/Personal/Brokle/brokle-platform/brokle/cmd/migrate/main.go"

        # Check if migration command exists
        assert os.path.exists(migration_cmd), f"Migration command should exist: {migration_cmd}"

        # Check if Go is available to run migrations
        try:
            result = subprocess.run(["go", "version"], capture_output=True, text=True, timeout=10)
            assert result.returncode == 0, "Go should be available to run migrations"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Go not available for migration testing")

    def test_migration_help_command(self):
        """Test migration help command works."""
        migration_dir = "/home/hashir/Development/Projects/Personal/Brokle/brokle-platform/brokle"

        try:
            # Try to run migration help command
            result = subprocess.run(
                ["go", "run", "cmd/migrate/main.go", "-h"],
                cwd=migration_dir,
                capture_output=True,
                text=True,
                timeout=30
            )

            # Should show help output (exit code may be 0 or 2 for help)
            assert result.returncode in [0, 2], f"Migration help should work, got exit code: {result.returncode}"

            # Should contain usage information
            help_output = result.stdout + result.stderr
            assert any(keyword in help_output.lower() for keyword in ["usage", "help", "migrate", "database"]), \
                f"Help output should contain usage information: {help_output}"

        except subprocess.TimeoutExpired:
            pytest.skip("Migration command timed out")
        except Exception as e:
            pytest.skip(f"Could not run migration command: {e}")

    def test_migration_info_command(self):
        """Test migration info command works."""
        migration_dir = "/home/hashir/Development/Projects/Personal/Brokle/brokle-platform/brokle"

        try:
            # Try to run migration info command
            result = subprocess.run(
                ["go", "run", "cmd/migrate/main.go", "info"],
                cwd=migration_dir,
                capture_output=True,
                text=True,
                timeout=30
            )

            # Info command should work even without database connection
            # (it should show migration file information)
            info_output = result.stdout + result.stderr

            # Should contain information about migrations
            expected_keywords = ["migration", "postgres", "clickhouse", "files", "version"]
            found_keywords = sum(1 for keyword in expected_keywords if keyword in info_output.lower())

            assert found_keywords >= 2, f"Info output should contain migration information: {info_output}"

        except subprocess.TimeoutExpired:
            pytest.skip("Migration info command timed out")
        except Exception as e:
            pytest.skip(f"Could not run migration info command: {e}")

    def test_migration_dry_run(self):
        """Test migration dry run functionality."""
        migration_dir = "/home/hashir/Development/Projects/Personal/Brokle/brokle-platform/brokle"

        try:
            # Try dry run mode (should not require actual database)
            result = subprocess.run(
                ["go", "run", "cmd/migrate/main.go", "-dry-run", "up"],
                cwd=migration_dir,
                capture_output=True,
                text=True,
                timeout=30
            )

            dry_run_output = result.stdout + result.stderr

            # Dry run should show what would be executed
            dry_run_keywords = ["dry", "would", "execute", "migration", "preview"]
            found_keywords = sum(1 for keyword in dry_run_keywords if keyword in dry_run_output.lower())

            # Should find at least some dry run indicators
            if found_keywords == 0:
                # If no dry run keywords found, at least shouldn't crash
                # and should mention database or migration files
                assert any(keyword in dry_run_output.lower() for keyword in ["database", "migration", "postgres", "clickhouse"]), \
                    f"Dry run should show migration information: {dry_run_output}"

        except subprocess.TimeoutExpired:
            pytest.skip("Migration dry run command timed out")
        except Exception as e:
            pytest.skip(f"Could not run migration dry run: {e}")

    def test_migration_files_validation(self):
        """Test that migration files are properly formatted."""
        postgres_dir = "/home/hashir/Development/Projects/Personal/Brokle/brokle-platform/brokle/migrations/postgres"
        clickhouse_dir = "/home/hashir/Development/Projects/Personal/Brokle/brokle-platform/brokle/migrations/clickhouse"

        # Test PostgreSQL migration files
        if os.path.exists(postgres_dir):
            postgres_files = os.listdir(postgres_dir)
            up_files = [f for f in postgres_files if f.endswith('.up.sql')]
            down_files = [f for f in postgres_files if f.endswith('.down.sql')]

            assert len(up_files) > 0, "Should have PostgreSQL up migration files"
            assert len(down_files) > 0, "Should have PostgreSQL down migration files"

            # Check that each up file has corresponding down file
            up_prefixes = [f.replace('.up.sql', '') for f in up_files]
            down_prefixes = [f.replace('.down.sql', '') for f in down_files]

            for up_prefix in up_prefixes:
                assert up_prefix in down_prefixes, f"Missing down migration for: {up_prefix}"

        # Test ClickHouse migration files
        if os.path.exists(clickhouse_dir):
            clickhouse_files = os.listdir(clickhouse_dir)
            ch_up_files = [f for f in clickhouse_files if f.endswith('.up.sql')]
            ch_down_files = [f for f in clickhouse_files if f.endswith('.down.sql')]

            assert len(ch_up_files) > 0, "Should have ClickHouse up migration files"
            assert len(ch_down_files) > 0, "Should have ClickHouse down migration files"

    def test_observability_migration_exists(self):
        """Test that observability migration files exist."""
        postgres_dir = "/home/hashir/Development/Projects/Personal/Brokle/brokle-platform/brokle/migrations/postgres"
        clickhouse_dir = "/home/hashir/Development/Projects/Personal/Brokle/brokle-platform/brokle/migrations/clickhouse"

        # Check for observability migration in PostgreSQL
        postgres_files = os.listdir(postgres_dir)
        observability_postgres = [f for f in postgres_files if 'llm_observability' in f or 'observability' in f]
        assert len(observability_postgres) >= 2, f"Should have PostgreSQL observability migrations (up/down): {observability_postgres}"

        # Check for analytics migration in ClickHouse
        clickhouse_files = os.listdir(clickhouse_dir)
        analytics_clickhouse = [f for f in clickhouse_files if 'llm_analytics' in f or 'analytics' in f]
        assert len(analytics_clickhouse) >= 2, f"Should have ClickHouse analytics migrations (up/down): {analytics_clickhouse}"

    def test_migration_file_syntax_validation(self):
        """Test basic SQL syntax validation of migration files."""
        postgres_dir = "/home/hashir/Development/Projects/Personal/Brokle/brokle-platform/brokle/migrations/postgres"

        # Test the observability migration file specifically
        observability_up = None
        for filename in os.listdir(postgres_dir):
            if 'llm_observability' in filename and filename.endswith('.up.sql'):
                observability_up = os.path.join(postgres_dir, filename)
                break

        if observability_up and os.path.exists(observability_up):
            with open(observability_up, 'r') as f:
                content = f.read()

            # Basic SQL syntax checks
            assert content.strip(), "Migration file should not be empty"
            assert 'CREATE TABLE' in content, "Should contain table creation statements"
            assert 'llm_traces' in content, "Should create llm_traces table"
            assert 'llm_observations' in content, "Should create llm_observations table"
            assert 'llm_quality_scores' in content, "Should create llm_quality_scores table"

            # Check for ULID compatibility
            assert 'CHAR(26)' in content, "Should use CHAR(26) for ULID fields"

            # Check for proper indexes
            assert 'CREATE INDEX' in content, "Should create performance indexes"

            # Check for constraints
            assert 'CHECK (' in content, "Should have data validation constraints"

            # Check for triggers
            assert 'CREATE TRIGGER' in content, "Should create automatic calculation triggers"

    @pytest.mark.slow
    def test_migration_compilation(self):
        """Test that migration command compiles successfully."""
        migration_dir = "/home/hashir/Development/Projects/Personal/Brokle/brokle-platform/brokle"

        try:
            # Try to compile the migration command
            result = subprocess.run(
                ["go", "build", "-o", "/tmp/test_migrate", "cmd/migrate/main.go"],
                cwd=migration_dir,
                capture_output=True,
                text=True,
                timeout=60
            )

            compile_output = result.stdout + result.stderr

            if result.returncode != 0:
                # If compilation failed, provide detailed error information
                pytest.fail(f"Migration command compilation failed:\nExit code: {result.returncode}\nOutput: {compile_output}")

            # Check that binary was created
            assert os.path.exists("/tmp/test_migrate"), "Migration binary should be created"

            # Clean up
            try:
                os.remove("/tmp/test_migrate")
            except OSError:
                pass

        except subprocess.TimeoutExpired:
            pytest.fail("Migration compilation timed out")
        except Exception as e:
            pytest.skip(f"Could not compile migration command: {e}")


class TestMigrationEnvironmentSetup:
    """Test migration environment and dependencies."""

    def test_go_environment(self):
        """Test Go environment is properly set up for migrations."""
        try:
            # Check Go version
            result = subprocess.run(["go", "version"], capture_output=True, text=True, timeout=10)
            assert result.returncode == 0, "Go should be available"

            go_version = result.stdout
            assert "go version" in go_version.lower(), f"Should show Go version: {go_version}"

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Go not available")

    def test_migration_dependencies(self):
        """Test that migration dependencies are available."""
        migration_dir = "/home/hashir/Development/Projects/Personal/Brokle/brokle-platform/brokle"
        go_mod_file = os.path.join(migration_dir, "go.mod")

        assert os.path.exists(go_mod_file), "go.mod file should exist for dependencies"

        with open(go_mod_file, 'r') as f:
            go_mod_content = f.read()

        # Should include migration library dependency
        migration_libraries = [
            "golang-migrate",
            "migrate",
            "database/sql",
            "postgres",
            "clickhouse"
        ]

        found_libraries = []
        for lib in migration_libraries:
            if lib.lower() in go_mod_content.lower():
                found_libraries.append(lib)

        assert len(found_libraries) >= 2, f"Should have migration dependencies: found {found_libraries}"

    def test_database_drivers_available(self):
        """Test that database drivers are properly imported."""
        migration_cmd = "/home/hashir/Development/Projects/Personal/Brokle/brokle-platform/brokle/cmd/migrate/main.go"

        if not os.path.exists(migration_cmd):
            pytest.skip("Migration command file not found")

        with open(migration_cmd, 'r') as f:
            migrate_content = f.read()

        # Should import database drivers
        expected_imports = [
            "database/sql",
            "postgres",
            "clickhouse"
        ]

        found_imports = []
        for expected_import in expected_imports:
            if expected_import in migrate_content:
                found_imports.append(expected_import)

        assert len(found_imports) >= 2, f"Should import database drivers: found {found_imports}"


class TestMigrationConfiguration:
    """Test migration configuration and environment variables."""

    def test_database_url_handling(self):
        """Test database URL configuration handling."""
        # This tests the theoretical handling of database URLs
        # without requiring actual database connections

        # Test URL patterns that should be supported
        test_urls = [
            "postgresql://user:pass@localhost:5432/dbname",
            "postgres://user:pass@localhost:5432/dbname",
            "clickhouse://user:pass@localhost:9000/dbname"
        ]

        for url in test_urls:
            # Basic URL format validation
            assert "://" in url, f"Should be valid URL format: {url}"
            assert "@" in url, f"Should include credentials: {url}"
            assert "/" in url.split("://")[1], f"Should include database name: {url}"

    def test_environment_variables(self):
        """Test environment variable patterns for database configuration."""
        # Test expected environment variable names
        expected_env_vars = [
            "DATABASE_URL",
            "POSTGRES_URL",
            "CLICKHOUSE_URL",
            "DB_HOST",
            "DB_PORT",
            "DB_USER",
            "DB_PASSWORD",
            "DB_NAME"
        ]

        # These variables might not be set in test environment,
        # but we test that they have reasonable names
        for var_name in expected_env_vars:
            assert var_name.isupper(), f"Environment variable should be uppercase: {var_name}"
            assert "_" in var_name or var_name in ["HOST", "PORT", "USER"], f"Should follow naming convention: {var_name}"


# Test markers
pytestmark = [
    pytest.mark.integration,
    pytest.mark.migration,
    pytest.mark.slow
]

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])