"""
Tests for OpenTelemetry Resource creation with release/version attributes.

Verifies that release and version attributes are correctly set on the Resource
for both tracing-only, metrics-only, and combined deployments.

Regression test for: Metrics resource drops release/version when tracing is off
"""

from unittest.mock import MagicMock, patch

import pytest


class TestOtelResourceAttributes:
    """Test OpenTelemetry Resource attribute handling in Brokle client."""

    @patch("brokle.client.BrokleMeterProvider")
    @patch("brokle.client.TracerProvider")
    @patch("brokle.client.BrokleSpanProcessor")
    @patch("brokle.client.create_exporter_for_config")
    def test_metrics_only_deployment_has_release_version(
        self,
        mock_create_exporter,
        mock_processor,
        mock_tracer_provider,
        mock_meter_provider,
    ):
        """
        Ensure release/version in resource when tracing disabled but metrics enabled.

        This is the key regression test: previously, when tracing_enabled=False,
        the resource was created without release/version attributes.
        """
        from brokle.client import Brokle
        from brokle.types import Attrs

        # Mock meter provider to capture the resource
        mock_meter_instance = MagicMock()
        mock_meter_instance.get_meter.return_value = MagicMock()
        mock_meter_provider.return_value = mock_meter_instance

        # Create client with tracing disabled but metrics enabled
        client = Brokle(
            api_key="bk_test_key_1234567890",
            tracing_enabled=False,
            metrics_enabled=True,
            release="v1.2.3",
            version="experiment-A",
        )

        # Tracer provider should NOT be created (tracing disabled)
        mock_tracer_provider.assert_not_called()

        # Meter provider SHOULD be created
        mock_meter_provider.assert_called_once()

        # Get the resource passed to MeterProvider
        call_kwargs = mock_meter_provider.call_args.kwargs
        resource = call_kwargs["resource"]

        # CRITICAL: Verify release and version are in the resource
        assert Attrs.BROKLE_RELEASE in resource.attributes
        assert Attrs.BROKLE_VERSION in resource.attributes
        assert resource.attributes[Attrs.BROKLE_RELEASE] == "v1.2.3"
        assert resource.attributes[Attrs.BROKLE_VERSION] == "experiment-A"

    @patch("brokle.client.BrokleMeterProvider")
    @patch("brokle.client.TracerProvider")
    @patch("brokle.client.BrokleSpanProcessor")
    @patch("brokle.client.create_exporter_for_config")
    def test_tracing_only_deployment_has_release_version(
        self,
        mock_create_exporter,
        mock_processor,
        mock_tracer_provider,
        mock_meter_provider,
    ):
        """
        Ensure release/version in resource when metrics disabled but tracing enabled.
        """
        from brokle.client import Brokle
        from brokle.types import Attrs

        # Mock tracer provider to capture the resource
        mock_tracer_instance = MagicMock()
        mock_tracer_instance.get_tracer.return_value = MagicMock()
        mock_tracer_provider.return_value = mock_tracer_instance

        # Create client with tracing enabled but metrics disabled
        client = Brokle(
            api_key="bk_test_key_1234567890",
            tracing_enabled=True,
            metrics_enabled=False,
            release="v2.0.0",
            version="control",
        )

        # Tracer provider SHOULD be created
        mock_tracer_provider.assert_called_once()

        # Meter provider should NOT be created (metrics disabled)
        mock_meter_provider.assert_not_called()

        # Get the resource passed to TracerProvider
        call_kwargs = mock_tracer_provider.call_args.kwargs
        resource = call_kwargs["resource"]

        # Verify release and version are in the resource
        assert Attrs.BROKLE_RELEASE in resource.attributes
        assert Attrs.BROKLE_VERSION in resource.attributes
        assert resource.attributes[Attrs.BROKLE_RELEASE] == "v2.0.0"
        assert resource.attributes[Attrs.BROKLE_VERSION] == "control"

    @patch("brokle.client.BrokleMeterProvider")
    @patch("brokle.client.TracerProvider")
    @patch("brokle.client.BrokleSpanProcessor")
    @patch("brokle.client.create_exporter_for_config")
    def test_both_providers_share_same_resource(
        self,
        mock_create_exporter,
        mock_processor,
        mock_tracer_provider,
        mock_meter_provider,
    ):
        """
        Verify tracer and meter providers use identical resource with release/version.
        """
        from brokle.client import Brokle
        from brokle.types import Attrs

        # Mock both providers
        mock_tracer_instance = MagicMock()
        mock_tracer_instance.get_tracer.return_value = MagicMock()
        mock_tracer_provider.return_value = mock_tracer_instance

        mock_meter_instance = MagicMock()
        mock_meter_instance.get_meter.return_value = MagicMock()
        mock_meter_provider.return_value = mock_meter_instance

        # Create client with both tracing and metrics enabled
        client = Brokle(
            api_key="bk_test_key_1234567890",
            tracing_enabled=True,
            metrics_enabled=True,
            release="v3.0.0",
            version="both-enabled",
        )

        # Both providers should be created
        mock_tracer_provider.assert_called_once()
        mock_meter_provider.assert_called_once()

        # Get resources from both providers
        tracer_resource = mock_tracer_provider.call_args.kwargs["resource"]
        meter_resource = mock_meter_provider.call_args.kwargs["resource"]

        # Resources should be the same instance (shared resource)
        assert tracer_resource is meter_resource

        # Both should have release and version
        assert tracer_resource.attributes[Attrs.BROKLE_RELEASE] == "v3.0.0"
        assert tracer_resource.attributes[Attrs.BROKLE_VERSION] == "both-enabled"

    @patch("brokle.client.BrokleMeterProvider")
    @patch("brokle.client.TracerProvider")
    @patch("brokle.client.BrokleSpanProcessor")
    @patch("brokle.client.create_exporter_for_config")
    def test_resource_without_release_version(
        self,
        mock_create_exporter,
        mock_processor,
        mock_tracer_provider,
        mock_meter_provider,
    ):
        """
        Verify resource is created correctly when release/version are not provided.
        """
        from brokle.client import Brokle
        from brokle.types import Attrs

        # Mock meter provider
        mock_meter_instance = MagicMock()
        mock_meter_instance.get_meter.return_value = MagicMock()
        mock_meter_provider.return_value = mock_meter_instance

        # Create client without release/version
        client = Brokle(
            api_key="bk_test_key_1234567890",
            tracing_enabled=False,
            metrics_enabled=True,
            # No release or version
        )

        # Meter provider should be created
        mock_meter_provider.assert_called_once()

        # Get the resource
        resource = mock_meter_provider.call_args.kwargs["resource"]

        # Resource should NOT have release or version attributes
        assert Attrs.BROKLE_RELEASE not in resource.attributes
        assert Attrs.BROKLE_VERSION not in resource.attributes

    @patch("brokle.client.BrokleMeterProvider")
    @patch("brokle.client.TracerProvider")
    @patch("brokle.client.BrokleSpanProcessor")
    @patch("brokle.client.create_exporter_for_config")
    def test_resource_with_only_release(
        self,
        mock_create_exporter,
        mock_processor,
        mock_tracer_provider,
        mock_meter_provider,
    ):
        """
        Verify resource works correctly when only release is provided.
        """
        from brokle.client import Brokle
        from brokle.types import Attrs

        # Mock meter provider
        mock_meter_instance = MagicMock()
        mock_meter_instance.get_meter.return_value = MagicMock()
        mock_meter_provider.return_value = mock_meter_instance

        # Create client with only release (no version)
        client = Brokle(
            api_key="bk_test_key_1234567890",
            tracing_enabled=False,
            metrics_enabled=True,
            release="v1.0.0",
            # No version
        )

        # Get the resource
        resource = mock_meter_provider.call_args.kwargs["resource"]

        # Resource should have release but NOT version
        assert Attrs.BROKLE_RELEASE in resource.attributes
        assert resource.attributes[Attrs.BROKLE_RELEASE] == "v1.0.0"
        assert Attrs.BROKLE_VERSION not in resource.attributes

    @patch("brokle.client.BrokleMeterProvider")
    @patch("brokle.client.TracerProvider")
    @patch("brokle.client.BrokleSpanProcessor")
    @patch("brokle.client.create_exporter_for_config")
    def test_resource_with_only_version(
        self,
        mock_create_exporter,
        mock_processor,
        mock_tracer_provider,
        mock_meter_provider,
    ):
        """
        Verify resource works correctly when only version is provided.
        """
        from brokle.client import Brokle
        from brokle.types import Attrs

        # Mock meter provider
        mock_meter_instance = MagicMock()
        mock_meter_instance.get_meter.return_value = MagicMock()
        mock_meter_provider.return_value = mock_meter_instance

        # Create client with only version (no release)
        client = Brokle(
            api_key="bk_test_key_1234567890",
            tracing_enabled=False,
            metrics_enabled=True,
            version="experiment-B",
            # No release
        )

        # Get the resource
        resource = mock_meter_provider.call_args.kwargs["resource"]

        # Resource should have version but NOT release
        assert Attrs.BROKLE_VERSION in resource.attributes
        assert resource.attributes[Attrs.BROKLE_VERSION] == "experiment-B"
        assert Attrs.BROKLE_RELEASE not in resource.attributes


class TestGetClientConfigForwarding:
    """Test that get_client() properly forwards all BrokleConfig fields."""

    def test_get_client_forwards_transport_setting(self):
        """
        Verify get_client() forwards transport setting to Brokle client.

        Regression test for: get_client drops new transport and metrics settings
        """
        from unittest.mock import patch

        import brokle.client

        # Reset singleton
        brokle.client._global_client = None

        with patch.dict(
            "os.environ",
            {"BROKLE_API_KEY": "bk_test_key_1234567890"},
            clear=False,
        ):
            with patch("brokle.client.Brokle") as mock_brokle:
                mock_brokle.return_value = MagicMock()

                # Call get_client with transport override
                from brokle.client import get_client

                client = get_client(transport="grpc")

                # Verify Brokle was called with config object
                mock_brokle.assert_called_once()
                call_kwargs = mock_brokle.call_args.kwargs

                # Should have config parameter, not individual fields
                assert "config" in call_kwargs
                config = call_kwargs["config"]
                assert config.transport == "grpc"

        # Cleanup
        brokle.client._global_client = None

    def test_get_client_forwards_metrics_export_interval(self):
        """
        Verify get_client() forwards metrics_export_interval setting.
        """
        from unittest.mock import patch

        import brokle.client

        # Reset singleton
        brokle.client._global_client = None

        with patch.dict(
            "os.environ",
            {"BROKLE_API_KEY": "bk_test_key_1234567890"},
            clear=False,
        ):
            with patch("brokle.client.Brokle") as mock_brokle:
                mock_brokle.return_value = MagicMock()

                from brokle.client import get_client

                client = get_client(metrics_export_interval=30.0)

                mock_brokle.assert_called_once()
                config = mock_brokle.call_args.kwargs["config"]
                assert config.metrics_export_interval == 30.0

        # Cleanup
        brokle.client._global_client = None

    def test_get_client_forwards_version(self):
        """
        Verify get_client() forwards version setting for A/B testing.
        """
        from unittest.mock import patch

        import brokle.client

        # Reset singleton
        brokle.client._global_client = None

        with patch.dict(
            "os.environ",
            {"BROKLE_API_KEY": "bk_test_key_1234567890"},
            clear=False,
        ):
            with patch("brokle.client.Brokle") as mock_brokle:
                mock_brokle.return_value = MagicMock()

                from brokle.client import get_client

                client = get_client(version="experiment-A")

                mock_brokle.assert_called_once()
                config = mock_brokle.call_args.kwargs["config"]
                assert config.version == "experiment-A"

        # Cleanup
        brokle.client._global_client = None

    def test_get_client_forwards_grpc_endpoint(self):
        """
        Verify get_client() forwards grpc_endpoint setting.
        """
        from unittest.mock import patch

        import brokle.client

        # Reset singleton
        brokle.client._global_client = None

        with patch.dict(
            "os.environ",
            {"BROKLE_API_KEY": "bk_test_key_1234567890"},
            clear=False,
        ):
            with patch("brokle.client.Brokle") as mock_brokle:
                mock_brokle.return_value = MagicMock()

                from brokle.client import get_client

                client = get_client(transport="grpc", grpc_endpoint="localhost:4317")

                mock_brokle.assert_called_once()
                config = mock_brokle.call_args.kwargs["config"]
                assert config.transport == "grpc"
                assert config.grpc_endpoint == "localhost:4317"

        # Cleanup
        brokle.client._global_client = None

    def test_get_client_forwards_max_queue_size(self):
        """
        Verify get_client() forwards max_queue_size setting.
        """
        from unittest.mock import patch

        import brokle.client

        # Reset singleton
        brokle.client._global_client = None

        with patch.dict(
            "os.environ",
            {"BROKLE_API_KEY": "bk_test_key_1234567890"},
            clear=False,
        ):
            with patch("brokle.client.Brokle") as mock_brokle:
                mock_brokle.return_value = MagicMock()

                from brokle.client import get_client

                client = get_client(max_queue_size=4096)

                mock_brokle.assert_called_once()
                config = mock_brokle.call_args.kwargs["config"]
                assert config.max_queue_size == 4096

        # Cleanup
        brokle.client._global_client = None

    def test_get_client_forwards_export_timeout(self):
        """
        Verify get_client() forwards export_timeout setting.
        """
        from unittest.mock import patch

        import brokle.client

        # Reset singleton
        brokle.client._global_client = None

        with patch.dict(
            "os.environ",
            {"BROKLE_API_KEY": "bk_test_key_1234567890"},
            clear=False,
        ):
            with patch("brokle.client.Brokle") as mock_brokle:
                mock_brokle.return_value = MagicMock()

                from brokle.client import get_client

                client = get_client(export_timeout=60000)

                mock_brokle.assert_called_once()
                config = mock_brokle.call_args.kwargs["config"]
                assert config.export_timeout == 60000

        # Cleanup
        brokle.client._global_client = None

    def test_get_client_reads_transport_from_env(self):
        """
        Verify get_client() reads BROKLE_TRANSPORT from environment.
        """
        from unittest.mock import patch

        import brokle.client

        # Reset singleton
        brokle.client._global_client = None

        with patch.dict(
            "os.environ",
            {
                "BROKLE_API_KEY": "bk_test_key_1234567890",
                "BROKLE_TRANSPORT": "grpc",
            },
            clear=False,
        ):
            with patch("brokle.client.Brokle") as mock_brokle:
                mock_brokle.return_value = MagicMock()

                from brokle.client import get_client

                client = get_client()

                mock_brokle.assert_called_once()
                config = mock_brokle.call_args.kwargs["config"]
                assert config.transport == "grpc"

        # Cleanup
        brokle.client._global_client = None

    def test_get_client_reads_metrics_export_interval_from_env(self):
        """
        Verify get_client() reads BROKLE_METRICS_EXPORT_INTERVAL from environment.
        """
        from unittest.mock import patch

        import brokle.client

        # Reset singleton
        brokle.client._global_client = None

        with patch.dict(
            "os.environ",
            {
                "BROKLE_API_KEY": "bk_test_key_1234567890",
                "BROKLE_METRICS_EXPORT_INTERVAL": "15.0",
            },
            clear=False,
        ):
            with patch("brokle.client.Brokle") as mock_brokle:
                mock_brokle.return_value = MagicMock()

                from brokle.client import get_client

                client = get_client()

                mock_brokle.assert_called_once()
                config = mock_brokle.call_args.kwargs["config"]
                assert config.metrics_export_interval == 15.0

        # Cleanup
        brokle.client._global_client = None

    @patch("brokle.client.BrokleMeterProvider")
    @patch("brokle.client.TracerProvider")
    @patch("brokle.client.BrokleSpanProcessor")
    @patch("brokle.client.create_exporter_for_config")
    def test_get_client_includes_release_version_in_resource(
        self,
        mock_create_exporter,
        mock_processor,
        mock_tracer_provider,
        mock_meter_provider,
    ):
        """
        Verify release/version from config appear in OTEL resource when using get_client().

        Regression test for: Release/version dropped when Brokle is built from config
        When using config= parameter path (used by get_client), the resource attributes
        must be sourced from self.config, not local parameters.
        """
        import brokle.client
        from brokle.client import get_client
        from brokle.types import Attrs

        # Reset singleton
        brokle.client._global_client = None

        # Mock tracer provider to capture the resource
        mock_tracer_instance = MagicMock()
        mock_tracer_instance.get_tracer.return_value = MagicMock()
        mock_tracer_provider.return_value = mock_tracer_instance

        with patch.dict(
            "os.environ",
            {"BROKLE_API_KEY": "bk_test_key_for_resource_test"},
            clear=False,
        ):
            # Call get_client with release and version
            client = get_client(
                release="v2.0.0",
                version="experiment-B",
                tracing_enabled=True,
                metrics_enabled=False,
            )

            # Verify TracerProvider was called
            mock_tracer_provider.assert_called_once()

            # Get the resource passed to TracerProvider
            call_kwargs = mock_tracer_provider.call_args.kwargs
            resource = call_kwargs["resource"]

            # CRITICAL: Verify release and version from config are in the resource
            assert Attrs.BROKLE_RELEASE in resource.attributes
            assert Attrs.BROKLE_VERSION in resource.attributes
            assert resource.attributes[Attrs.BROKLE_RELEASE] == "v2.0.0"
            assert resource.attributes[Attrs.BROKLE_VERSION] == "experiment-B"

        # Cleanup
        brokle.client._global_client = None

    @patch("brokle.client.BrokleMeterProvider")
    @patch("brokle.client.TracerProvider")
    @patch("brokle.client.BrokleSpanProcessor")
    @patch("brokle.client.create_exporter_for_config")
    def test_get_client_reads_release_version_from_env(
        self,
        mock_create_exporter,
        mock_processor,
        mock_tracer_provider,
        mock_meter_provider,
    ):
        """
        Verify BROKLE_RELEASE and BROKLE_VERSION env vars appear in resource via get_client().
        """
        import brokle.client
        from brokle.client import get_client
        from brokle.types import Attrs

        # Reset singleton
        brokle.client._global_client = None

        # Mock tracer provider to capture the resource
        mock_tracer_instance = MagicMock()
        mock_tracer_instance.get_tracer.return_value = MagicMock()
        mock_tracer_provider.return_value = mock_tracer_instance

        with patch.dict(
            "os.environ",
            {
                "BROKLE_API_KEY": "bk_test_key_for_env_resource",
                "BROKLE_RELEASE": "v3.1.4",
                "BROKLE_VERSION": "canary",
            },
            clear=False,
        ):
            # Call get_client - should read release/version from env
            client = get_client(
                tracing_enabled=True,
                metrics_enabled=False,
            )

            # Verify TracerProvider was called
            mock_tracer_provider.assert_called_once()

            # Get the resource passed to TracerProvider
            call_kwargs = mock_tracer_provider.call_args.kwargs
            resource = call_kwargs["resource"]

            # Verify release and version from env are in the resource
            assert resource.attributes[Attrs.BROKLE_RELEASE] == "v3.1.4"
            assert resource.attributes[Attrs.BROKLE_VERSION] == "canary"

        # Cleanup
        brokle.client._global_client = None
