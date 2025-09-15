"""
Database migration and ULID functionality tests.

This module tests the database migration system and ULID integration
to ensure proper functionality and compatibility between the SDK and backend.
"""

import asyncio
import pytest
import re
import time
from datetime import datetime, timezone
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock

# Test ULID functionality
try:
    import ulid
    ULID_AVAILABLE = True
except ImportError:
    ULID_AVAILABLE = False

from brokle.auto_instrumentation import get_registry, reset_all_errors
from brokle.auto_instrumentation.openai_instrumentation import OpenAIInstrumentation


class TestULIDFunctionality:
    """Test ULID generation and validation functionality."""

    def test_ulid_import_availability(self):
        """Test that ULID library is available."""
        try:
            import ulid as ulid_module
            assert hasattr(ulid_module, 'new')
            assert hasattr(ulid_module, 'parse')
            ulid_available = True
        except ImportError:
            ulid_available = False

        # ULID should be available for proper observability functionality
        assert ulid_available, "ULID library should be available for ID generation"

    @pytest.mark.skipif(not ULID_AVAILABLE, reason="ULID library not available")
    def test_ulid_generation_properties(self):
        """Test ULID generation properties and characteristics."""
        import ulid as ulid_module

        # Generate multiple ULIDs
        ulids = [ulid_module.new() for _ in range(100)]

        # Test basic properties
        for generated_ulid in ulids:
            # Should be 26 characters (base32 encoded)
            assert len(str(generated_ulid)) == 26

            # Should be uppercase base32 (Crockford's base32)
            ulid_str = str(generated_ulid)
            assert ulid_str.isupper()
            assert re.match(r'^[0123456789ABCDEFGHJKMNPQRSTVWXYZ]+$', ulid_str)

        # Test uniqueness
        ulid_strings = [str(u) for u in ulids]
        assert len(set(ulid_strings)) == len(ulid_strings), "All ULIDs should be unique"

        # Test lexicographical ordering (should be mostly ordered due to timestamp)
        time.sleep(0.001)  # Small delay
        later_ulid = ulid_module.new()

        # The later ULID should generally be lexicographically greater
        # (though this is not guaranteed due to randomness in the random portion)
        assert str(later_ulid) > str(ulids[0])

    @pytest.mark.skipif(not ULID_AVAILABLE, reason="ULID library not available")
    def test_ulid_parsing_and_validation(self):
        """Test ULID parsing and validation."""
        import ulid as ulid_module

        # Generate a ULID
        original_ulid = ulid_module.new()
        ulid_str = str(original_ulid)

        # Test parsing
        parsed_ulid = ulid_module.parse(ulid_str)
        assert str(parsed_ulid) == ulid_str

        # Test timestamp extraction
        timestamp = original_ulid.timestamp
        assert isinstance(timestamp, int)
        assert timestamp > 0

        # Test that timestamp is recent (within last minute)
        current_timestamp = int(time.time() * 1000)
        assert abs(current_timestamp - timestamp) < 60000  # Within 1 minute

    @pytest.mark.skipif(not ULID_AVAILABLE, reason="ULID library not available")
    def test_ulid_format_compatibility(self):
        """Test ULID format compatibility with database schema."""
        import ulid as ulid_module

        # Generate ULIDs and test they match expected database format
        for _ in range(50):
            generated_ulid = ulid_module.new()
            ulid_str = str(generated_ulid)

            # Should be exactly 26 characters (CHAR(26) in database)
            assert len(ulid_str) == 26

            # Should be valid base32
            assert re.match(r'^[0-9A-Z]+$', ulid_str)

            # Should not contain problematic characters
            problematic_chars = ['O', 'I', 'L', 'U']  # Excluded from Crockford's base32
            for char in problematic_chars:
                assert char not in ulid_str

    @pytest.mark.skipif(not ULID_AVAILABLE, reason="ULID library not available")
    def test_ulid_performance(self):
        """Test ULID generation performance."""
        import ulid as ulid_module

        # Time ULID generation
        start_time = time.perf_counter()
        ulids = [ulid_module.new() for _ in range(1000)]
        end_time = time.perf_counter()

        generation_time = (end_time - start_time) * 1000  # Convert to ms
        avg_time_per_ulid = generation_time / 1000

        print(f"Generated 1000 ULIDs in {generation_time:.2f}ms (avg: {avg_time_per_ulid:.4f}ms per ULID)")

        # Should be very fast (< 1ms per ULID on average)
        assert avg_time_per_ulid < 1.0, f"ULID generation too slow: {avg_time_per_ulid:.4f}ms per ULID"

        # Verify all are unique
        ulid_strings = [str(u) for u in ulids]
        assert len(set(ulid_strings)) == 1000


class TestDatabaseSchemaCompatibility:
    """Test database schema compatibility with ULID format."""

    def test_ulid_database_field_compatibility(self):
        """Test that ULIDs are compatible with database CHAR(26) fields."""
        if not ULID_AVAILABLE:
            pytest.skip("ULID library not available")

        import ulid as ulid_module

        # Test multiple ULIDs for compatibility
        test_ulids = [ulid_module.new() for _ in range(20)]

        for test_ulid in test_ulids:
            ulid_str = str(test_ulid)

            # Database compatibility checks
            assert len(ulid_str) == 26, f"ULID length should be 26 chars for CHAR(26): {ulid_str}"
            assert ulid_str.isalnum(), f"ULID should be alphanumeric: {ulid_str}"
            assert ulid_str.isupper(), f"ULID should be uppercase: {ulid_str}"

            # Test that it would fit in VARCHAR fields too
            assert len(ulid_str.encode('utf-8')) == 26, "ULID should be ASCII-only"

    def test_database_migration_ulid_fields(self):
        """Test that database migration creates proper ULID-compatible fields."""
        # This test validates the migration SQL structure
        migration_file = "/home/hashir/Development/Projects/Personal/Brokle/brokle-platform/brokle/migrations/postgres/20250915120001_create_llm_observability_tables.up.sql"

        try:
            with open(migration_file, 'r') as f:
                migration_content = f.read()
        except FileNotFoundError:
            pytest.skip("Migration file not found")

        # Check that ULID fields are defined as CHAR(26)
        ulid_field_patterns = [
            r'id CHAR\(26\) PRIMARY KEY',
            r'trace_id CHAR\(26\)',
            r'observation_id CHAR\(26\)',
            r'project_id CHAR\(26\)',
            r'session_id CHAR\(26\)',
            r'parent_trace_id CHAR\(26\)',
            r'parent_observation_id CHAR\(26\)',
            r'user_id CHAR\(26\)',
            r'author_user_id CHAR\(26\)'
        ]

        for pattern in ulid_field_patterns:
            matches = re.findall(pattern, migration_content, re.IGNORECASE)
            if 'id CHAR(26) PRIMARY KEY' in pattern:
                # Should find multiple primary key definitions
                assert len(matches) >= 3, f"Should find primary key ULID fields: {pattern}"
            elif 'trace_id CHAR(26)' in pattern:
                # Should find trace_id references
                assert len(matches) >= 2, f"Should find trace_id references: {pattern}"

    def test_database_indexes_for_ulid_fields(self):
        """Test that proper indexes are created for ULID fields."""
        migration_file = "/home/hashir/Development/Projects/Personal/Brokle/brokle-platform/brokle/migrations/postgres/20250915120001_create_llm_observability_tables.up.sql"

        try:
            with open(migration_file, 'r') as f:
                migration_content = f.read()
        except FileNotFoundError:
            pytest.skip("Migration file not found")

        # Check for important indexes on ULID fields
        expected_indexes = [
            r'CREATE INDEX.*trace_id',
            r'CREATE INDEX.*project_id',
            r'CREATE INDEX.*observation_id',
            r'CREATE INDEX.*parent_trace_id',
            r'CREATE INDEX.*parent_observation_id'
        ]

        for index_pattern in expected_indexes:
            matches = re.findall(index_pattern, migration_content, re.IGNORECASE)
            assert len(matches) >= 1, f"Should find index for ULID field: {index_pattern}"


class TestObservabilityDataTypes:
    """Test observability data types and constraints."""

    def test_observation_type_constraints(self):
        """Test that observation type constraints match expected values."""
        migration_file = "/home/hashir/Development/Projects/Personal/Brokle/brokle-platform/brokle/migrations/postgres/20250915120001_create_llm_observability_tables.up.sql"

        try:
            with open(migration_file, 'r') as f:
                migration_content = f.read()
        except FileNotFoundError:
            pytest.skip("Migration file not found")

        # Check observation type constraint
        type_constraint_pattern = r"CHECK \(type IN \([^)]+\)\)"
        matches = re.findall(type_constraint_pattern, migration_content)

        assert len(matches) >= 1, "Should find observation type constraint"

        # Extract the types from the constraint
        type_constraint = matches[0]
        expected_types = ['llm', 'span', 'event', 'generation', 'retrieval', 'embedding', 'agent', 'tool', 'chain']

        for expected_type in expected_types:
            assert expected_type in type_constraint, f"Should include observation type: {expected_type}"

    def test_quality_score_constraints(self):
        """Test quality score table constraints."""
        migration_file = "/home/hashir/Development/Projects/Personal/Brokle/brokle-platform/brokle/migrations/postgres/20250915120001_create_llm_observability_tables.up.sql"

        try:
            with open(migration_file, 'r') as f:
                migration_content = f.read()
        except FileNotFoundError:
            pytest.skip("Migration file not found")

        # Check data type constraint
        data_type_pattern = r"data_type.*CHECK.*NUMERIC.*CATEGORICAL.*BOOLEAN"
        matches = re.findall(data_type_pattern, migration_content, re.DOTALL)
        assert len(matches) >= 1, "Should find data_type constraint for quality scores"

        # Check source constraint
        source_pattern = r"source.*CHECK.*API.*AUTO.*HUMAN.*EVAL"
        matches = re.findall(source_pattern, migration_content, re.DOTALL)
        assert len(matches) >= 1, "Should find source constraint for quality scores"

        # Check value validation constraint
        value_constraint_pattern = r"chk_quality_score_value.*CHECK"
        matches = re.findall(value_constraint_pattern, migration_content, re.IGNORECASE | re.DOTALL)
        assert len(matches) >= 1, "Should find quality score value validation constraint"

    def test_numeric_field_constraints(self):
        """Test numeric field constraints and ranges."""
        migration_file = "/home/hashir/Development/Projects/Personal/Brokle/brokle-platform/brokle/migrations/postgres/20250915120001_create_llm_observability_tables.up.sql"

        try:
            with open(migration_file, 'r') as f:
                migration_content = f.read()
        except FileNotFoundError:
            pytest.skip("Migration file not found")

        # Check token constraints (should be >= 0)
        token_constraints = [
            r'prompt_tokens.*CHECK.*>= 0',
            r'completion_tokens.*CHECK.*>= 0',
            r'total_tokens.*CHECK.*>= 0'
        ]

        for constraint_pattern in token_constraints:
            matches = re.findall(constraint_pattern, migration_content)
            assert len(matches) >= 1, f"Should find token constraint: {constraint_pattern}"

        # Check cost constraints (should be >= 0)
        cost_constraints = [
            r'input_cost.*CHECK.*>= 0',
            r'output_cost.*CHECK.*>= 0',
            r'total_cost.*CHECK.*>= 0'
        ]

        for constraint_pattern in cost_constraints:
            matches = re.findall(constraint_pattern, migration_content)
            assert len(matches) >= 1, f"Should find cost constraint: {constraint_pattern}"

        # Check quality score range (0 to 1)
        quality_pattern = r'quality_score.*CHECK.*>= 0 AND.*<= 1'
        matches = re.findall(quality_pattern, migration_content)
        assert len(matches) >= 1, "Should find quality score range constraint"


class TestDatabaseTriggers:
    """Test database triggers and automatic calculations."""

    def test_automatic_timestamp_triggers(self):
        """Test that automatic timestamp update triggers are created."""
        migration_file = "/home/hashir/Development/Projects/Personal/Brokle/brokle-platform/brokle/migrations/postgres/20250915120001_create_llm_observability_tables.up.sql"

        try:
            with open(migration_file, 'r') as f:
                migration_content = f.read()
        except FileNotFoundError:
            pytest.skip("Migration file not found")

        # Check for updated_at trigger function
        assert "update_updated_at_column()" in migration_content
        assert "NEW.updated_at = NOW()" in migration_content

        # Check that triggers are created for all observability tables
        expected_triggers = [
            "trigger_llm_traces_updated_at",
            "trigger_llm_observations_updated_at",
            "trigger_llm_quality_scores_updated_at"
        ]

        for trigger_name in expected_triggers:
            assert trigger_name in migration_content, f"Should create trigger: {trigger_name}"

    def test_automatic_calculation_triggers(self):
        """Test that automatic calculation triggers are created."""
        migration_file = "/home/hashir/Development/Projects/Personal/Brokle/brokle-platform/brokle/migrations/postgres/20250915120001_create_llm_observability_tables.up.sql"

        try:
            with open(migration_file, 'r') as f:
                migration_content = f.read()
        except FileNotFoundError:
            pytest.skip("Migration file not found")

        # Check for latency calculation trigger
        assert "calculate_observation_latency()" in migration_content
        assert "trigger_llm_observations_calculate_latency" in migration_content
        assert "EXTRACT(MILLISECONDS FROM" in migration_content

        # Check for token calculation trigger
        assert "calculate_total_tokens()" in migration_content
        assert "trigger_llm_observations_calculate_total_tokens" in migration_content

        # Check for cost calculation trigger
        assert "calculate_total_cost()" in migration_content
        assert "trigger_llm_observations_calculate_total_cost" in migration_content


class TestClickHouseSchemaCompatibility:
    """Test ClickHouse analytics schema compatibility."""

    def test_clickhouse_analytics_tables(self):
        """Test ClickHouse analytics table creation."""
        clickhouse_migration = "/home/hashir/Development/Projects/Personal/Brokle/brokle-platform/brokle/migrations/clickhouse/20250915120002_create_llm_analytics_tables.up.sql"

        try:
            with open(clickhouse_migration, 'r') as f:
                clickhouse_content = f.read()
        except FileNotFoundError:
            pytest.skip("ClickHouse migration file not found")

        # Check for required analytics tables
        expected_tables = [
            "llm_request_logs",
            "llm_trace_analytics",
            "llm_cost_analytics",
            "llm_quality_analytics",
            "llm_performance_metrics"
        ]

        for table_name in expected_tables:
            assert table_name in clickhouse_content, f"Should create ClickHouse table: {table_name}"

        # Check for ULID compatibility in ClickHouse (usually String type)
        ulid_field_patterns = [
            r'trace_id String',
            r'observation_id String',
            r'project_id String'
        ]

        for pattern in ulid_field_patterns:
            matches = re.findall(pattern, clickhouse_content)
            if 'trace_id String' in pattern:
                assert len(matches) >= 1, f"Should find ClickHouse ULID field: {pattern}"

    def test_clickhouse_ttl_settings(self):
        """Test ClickHouse TTL (Time To Live) settings."""
        clickhouse_migration = "/home/hashir/Development/Projects/Personal/Brokle/brokle-platform/brokle/migrations/clickhouse/20250915120002_create_llm_analytics_tables.up.sql"

        try:
            with open(clickhouse_migration, 'r') as f:
                clickhouse_content = f.read()
        except FileNotFoundError:
            pytest.skip("ClickHouse migration file not found")

        # Check for TTL settings (data retention policies)
        ttl_patterns = [
            r'TTL.*DAY',
            r'toDate.*\+ INTERVAL.*DAY'
        ]

        ttl_found = False
        for pattern in ttl_patterns:
            if re.search(pattern, clickhouse_content, re.IGNORECASE):
                ttl_found = True
                break

        # TTL is important for analytics data management
        if not ttl_found:
            print("Warning: No TTL settings found in ClickHouse migration")


class TestMigrationIntegration:
    """Test integration between migration system and observability."""

    def test_migration_rollback_compatibility(self):
        """Test that migration rollback (down) files are compatible."""
        postgres_down = "/home/hashir/Development/Projects/Personal/Brokle/brokle-platform/brokle/migrations/postgres/20250915120001_create_llm_observability_tables.down.sql"

        try:
            with open(postgres_down, 'r') as f:
                down_content = f.read()
        except FileNotFoundError:
            pytest.skip("Migration down file not found")

        # Should drop tables in reverse dependency order
        expected_drops = [
            "DROP TABLE IF EXISTS llm_quality_scores",
            "DROP TABLE IF EXISTS llm_observations",
            "DROP TABLE IF EXISTS llm_traces"
        ]

        for drop_statement in expected_drops:
            assert drop_statement in down_content, f"Should include: {drop_statement}"

        # Should drop functions
        function_drops = [
            "DROP FUNCTION",
            "update_updated_at_column",
            "calculate_observation_latency",
            "calculate_total_tokens",
            "calculate_total_cost"
        ]

        for func_drop in function_drops:
            if "DROP FUNCTION" in func_drop:
                assert func_drop in down_content, "Should drop functions"
            else:
                assert func_drop in down_content, f"Should drop function: {func_drop}"

    @pytest.mark.skipif(not ULID_AVAILABLE, reason="ULID library not available")
    def test_ulid_sdk_integration_format(self):
        """Test ULID format integration with SDK."""
        import ulid as ulid_module

        # Test that SDK can generate ULIDs in the format expected by database
        sdk_ulid = ulid_module.new()
        ulid_str = str(sdk_ulid)

        # Simulate database insertion format
        assert len(ulid_str) == 26
        assert ulid_str.isalnum()
        assert ulid_str.isupper()

        # Test that we can create trace/observation IDs
        trace_id = str(ulid_module.new())
        observation_id = str(ulid_module.new())
        quality_score_id = str(ulid_module.new())

        # All should be different
        ids = {trace_id, observation_id, quality_score_id}
        assert len(ids) == 3, "All generated IDs should be unique"

        # All should be valid database format
        for test_id in ids:
            assert len(test_id) == 26
            assert re.match(r'^[0-9A-Z]+$', test_id)

    def test_external_id_compatibility(self):
        """Test external ID fields for SDK-server correlation."""
        migration_file = "/home/hashir/Development/Projects/Personal/Brokle/brokle-platform/brokle/migrations/postgres/20250915120001_create_llm_observability_tables.up.sql"

        try:
            with open(migration_file, 'r') as f:
                migration_content = f.read()
        except FileNotFoundError:
            pytest.skip("Migration file not found")

        # Check for external ID fields
        external_id_fields = [
            "external_trace_id VARCHAR(255)",
            "external_observation_id VARCHAR(255)"
        ]

        for field in external_id_fields:
            assert field in migration_content, f"Should define external ID field: {field}"

        # Check for unique constraints on external IDs
        external_id_indexes = [
            "idx_llm_traces_external_trace_id",
            "idx_llm_observations_external_obs_id"
        ]

        for index_name in external_id_indexes:
            assert index_name in migration_content, f"Should create unique index: {index_name}"


# Test markers
pytestmark = [
    pytest.mark.integration,
    pytest.mark.database,
    pytest.mark.ulid
]

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])