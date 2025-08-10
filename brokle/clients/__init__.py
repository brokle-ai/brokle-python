"""
Client modules for Brokle Platform.

This module contains AI-focused clients for developers and internal backend integration clients.
"""

# AI-focused clients (public API)
from .ai_analytics import AIAnalyticsClient

# Backend integration clients (internal use)
from .telemetry import TelemetryClient
from .cache import CacheClient
from .cost import CostClient
from .ml import MLClient

__all__ = [
    # Public AI-focused clients
    "AIAnalyticsClient",
    
    # Internal backend clients (used automatically by platform)
    "TelemetryClient",
    "CacheClient", 
    "CostClient",
    "MLClient",
]