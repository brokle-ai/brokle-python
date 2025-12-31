"""
Datasets Module

Provides dataset management for Brokle evaluations.

Usage:
    >>> from brokle import Brokle
    >>>
    >>> client = Brokle(api_key="bk_...")
    >>>
    >>> # Create dataset
    >>> dataset = client.datasets.create(
    ...     name="qa-pairs",
    ...     description="Question-answer test cases"
    ... )
    >>>
    >>> # Insert items
    >>> dataset.insert([
    ...     {"input": {"q": "2+2?"}, "expected": {"a": "4"}},
    ... ])
    >>>
    >>> # Iterate with auto-pagination
    >>> for item in dataset:
    ...     print(item.input, item.expected)
"""

from ._managers import AsyncDatasetsManager, DatasetsManager
from .dataset import AsyncDataset, Dataset, DatasetItem, DatasetItemInput
from .exceptions import DatasetError
from .types import DatasetData

__all__ = [
    # Managers
    "DatasetsManager",
    "AsyncDatasetsManager",
    # Dataset classes
    "Dataset",
    "AsyncDataset",
    "DatasetItem",
    "DatasetItemInput",
    # Types
    "DatasetData",
    # Exceptions
    "DatasetError",
]
