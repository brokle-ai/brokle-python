"""
Test hierarchical tracing capabilities of the @observe() decorator.

This test suite verifies that Brokle's @observe() decorator provides
automatic parent-child span relationships without requiring trace_workflow.
"""

import pytest

from brokle import observe
from brokle.observability.spans import get_current_span


class TestHierarchicalTracing:
    """Test automatic hierarchical tracing functionality."""

    def setup_method(self):
        """Setup for each test method."""
        self.span_info = []

    def collect_span_info(self, role: str):
        """Helper to collect span information."""
        current = get_current_span()
        self.span_info.append(
            (
                role,
                current.span_id if current else None,
                current.parent_span_id if current else None,
            )
        )

    def test_parent_child_relationship(self):
        """Test basic parent-child span relationship."""

        @observe(name="parent-span")
        def parent_function():
            """Parent function."""
            self.collect_span_info("parent")
            result = child_function("test-data")
            return f"parent({result})"

        @observe(name="child-span")
        def child_function(data):
            """Child function."""
            self.collect_span_info("child")
            return f"processed({data})"

        # Execute the workflow
        result = parent_function()

        # Verify result
        assert result == "parent(processed(test-data))"
        assert len(self.span_info) == 2

        # Extract spans by role
        spans_by_role = {
            role: (span_id, parent_id) for role, span_id, parent_id in self.span_info
        }

        # Verify parent span has no parent
        parent_span_id, parent_parent_id = spans_by_role["parent"]
        assert parent_span_id is not None
        assert parent_parent_id is None

        # Verify child span has parent as its parent
        child_span_id, child_parent_id = spans_by_role["child"]
        assert child_span_id is not None
        assert child_parent_id == parent_span_id

    def test_grandparent_parent_child_hierarchy(self):
        """Test three-level hierarchy: grandparent -> parent -> child."""

        @observe(name="grandparent-span")
        def grandparent_workflow():
            """Grandparent function."""
            self.collect_span_info("grandparent")
            result = parent_workflow()
            return f"grandparent({result})"

        @observe(name="parent-span")
        def parent_workflow():
            """Parent function."""
            self.collect_span_info("parent")
            result = child_function("data")
            return f"parent({result})"

        @observe(name="child-span")
        def child_function(data):
            """Child function."""
            self.collect_span_info("child")
            return f"child({data})"

        # Execute the workflow
        result = grandparent_workflow()

        # Verify result
        assert result == "grandparent(parent(child(data)))"
        assert len(self.span_info) == 3

        # Extract spans by role
        spans_by_role = {
            role: (span_id, parent_id) for role, span_id, parent_id in self.span_info
        }

        grandparent_span_id, grandparent_parent_id = spans_by_role["grandparent"]
        parent_span_id, parent_parent_id = spans_by_role["parent"]
        child_span_id, child_parent_id = spans_by_role["child"]

        # Verify hierarchy
        assert grandparent_span_id is not None
        assert grandparent_parent_id is None  # Root span

        assert parent_span_id is not None
        assert parent_parent_id == grandparent_span_id  # Parent's parent is grandparent

        assert child_span_id is not None
        assert child_parent_id == parent_span_id  # Child's parent is parent

    def test_multiple_children_same_parent(self):
        """Test multiple children having the same parent span."""

        @observe(name="parent-workflow")
        def parent_workflow():
            """Parent function that calls multiple children."""
            self.collect_span_info("parent")

            result1 = child_function_1("data1")
            result2 = child_function_2("data2")

            return f"parent({result1}, {result2})"

        @observe(name="child1-span")
        def child_function_1(data):
            """First child function."""
            self.collect_span_info("child1")
            return f"child1({data})"

        @observe(name="child2-span")
        def child_function_2(data):
            """Second child function."""
            self.collect_span_info("child2")
            return f"child2({data})"

        # Execute the workflow
        result = parent_workflow()

        # Verify result
        assert result == "parent(child1(data1), child2(data2))"
        assert len(self.span_info) == 3

        # Extract spans by role
        spans_by_role = {
            role: (span_id, parent_id) for role, span_id, parent_id in self.span_info
        }

        parent_span_id, parent_parent_id = spans_by_role["parent"]
        child1_span_id, child1_parent_id = spans_by_role["child1"]
        child2_span_id, child2_parent_id = spans_by_role["child2"]

        # Verify parent is root
        assert parent_span_id is not None
        assert parent_parent_id is None

        # Verify both children have same parent
        assert child1_span_id is not None
        assert child1_parent_id == parent_span_id

        assert child2_span_id is not None
        assert child2_parent_id == parent_span_id

    def test_complex_workflow(self):
        """Test complex workflow with hierarchical tracing."""

        @observe(name="ai-analysis-workflow")
        def ai_analysis_workflow(prompt: str):
            """Main AI analysis workflow."""
            self.collect_span_info("workflow")

            # Step 1: Preprocess
            processed = preprocess_data(prompt)

            # Step 2: Generate
            generated = generate_response(processed)

            # Step 3: Post-process
            final = postprocess_result(generated)

            return final

        @observe(name="preprocess")
        def preprocess_data(data: str):
            """Preprocessing step."""
            self.collect_span_info("preprocess")
            return data.strip().lower()

        @observe(name="generate")
        def generate_response(data: str):
            """Generation step."""
            self.collect_span_info("generate")
            # Simulate nested generation
            enhanced = enhance_generation(data)
            return f"generated({enhanced})"

        @observe(name="enhance")
        def enhance_generation(data: str):
            """Enhancement sub-step."""
            self.collect_span_info("enhance")
            return f"enhanced({data})"

        @observe(name="postprocess")
        def postprocess_result(result: str):
            """Post-processing step."""
            self.collect_span_info("postprocess")
            return f"final({result})"

        # Execute the workflow
        result = ai_analysis_workflow("  Test Prompt  ")

        # Verify result structure
        assert "final(generated(enhanced(test prompt)))" in result
        assert len(self.span_info) == 5

        # Extract spans by role
        spans_by_role = {
            role: (span_id, parent_id) for role, span_id, parent_id in self.span_info
        }

        # Verify hierarchy
        workflow_span_id = spans_by_role["workflow"][0]
        preprocess_parent = spans_by_role["preprocess"][1]
        generate_parent = spans_by_role["generate"][1]
        enhance_parent = spans_by_role["enhance"][1]
        postprocess_parent = spans_by_role["postprocess"][1]

        # Main workflow is root
        assert spans_by_role["workflow"][1] is None

        # Direct children of workflow
        assert preprocess_parent == workflow_span_id
        assert generate_parent == workflow_span_id
        assert postprocess_parent == workflow_span_id

        # enhance is child of generate
        generate_span_id = spans_by_role["generate"][0]
        assert enhance_parent == generate_span_id

    def test_no_parent_context_creates_root_span(self):
        """Test that calling @observe without parent context creates root span."""

        @observe(name="root-function")
        def root_function():
            """Function called without parent context."""
            self.collect_span_info("root")
            return "root-result"

        # Execute without any parent context
        result = root_function()

        assert result == "root-result"
        assert len(self.span_info) == 1

        # Should be root span (no parent)
        span_id, parent_id = self.span_info[0][1:3]
        assert span_id is not None
        assert parent_id is None

    def test_span_restoration_after_function_exit(self):
        """Test that previous span is restored after function exits."""

        @observe(name="outer-function")
        def outer_function():
            """Outer function."""
            self.collect_span_info("outer")

            # Call inner function
            inner_result = inner_function()

            # Check that we're back to outer span after inner completes
            self.collect_span_info("outer-after")

            return f"outer({inner_result})"

        @observe(name="inner-function")
        def inner_function():
            """Inner function."""
            self.collect_span_info("inner")
            return "inner-result"

        # Execute
        result = outer_function()

        assert result == "outer(inner-result)"
        assert len(self.span_info) == 3

        # Both "outer" and "outer-after" should have the same span context
        outer_span_info = [
            info for info in self.span_info if info[0].startswith("outer")
        ]
        assert len(outer_span_info) == 2
        assert outer_span_info[0][1] == outer_span_info[1][1]  # Same span ID
        assert outer_span_info[0][2] == outer_span_info[1][2]  # Same parent ID
