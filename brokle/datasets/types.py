"""
Type definitions for the datasets module.

This module provides type definitions for dataset operations.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DatasetData:
    """
    Dataset data from API response.

    Attributes:
        id: Unique identifier for the dataset
        name: Dataset name
        description: Dataset description
        metadata: Additional metadata
        created_at: ISO timestamp when created
        updated_at: ISO timestamp when last updated
    """

    id: str
    name: str
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetData":
        """Create DatasetData from API response dict."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description"),
            metadata=data.get("metadata"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )
