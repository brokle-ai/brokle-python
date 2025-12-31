"""
Dataset Management

Provides Dataset and AsyncDataset classes for managing evaluation datasets.
Datasets are collections of input/expected pairs used for systematic evaluation.

Sync Usage:
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
    ...     {"input": {"question": "What is 2+2?"}, "expected": {"answer": "4"}},
    ... ])
    >>>
    >>> # Iterate with auto-pagination
    >>> for item in dataset:
    ...     print(item.input, item.expected)

Async Usage:
    >>> async with AsyncBrokle(api_key="bk_...") as client:
    ...     dataset = await client.datasets.create(name="test")
    ...     await dataset.insert([{"input": {"q": "test"}}])
    ...     async for item in dataset:
    ...         print(item.input)
"""

from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from .._http import AsyncHTTPClient, SyncHTTPClient, unwrap_response
from .exceptions import DatasetError


@dataclass
class DatasetItem:
    """
    A single item in a dataset.

    Attributes:
        id: Unique identifier for the item
        dataset_id: ID of the parent dataset
        input: Input data for evaluation (arbitrary dict)
        expected: Expected output for comparison (optional)
        metadata: Additional metadata (optional)
        created_at: ISO timestamp when created
    """

    id: str
    dataset_id: str
    input: Dict[str, Any]
    expected: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetItem":
        """Create DatasetItem from API response dict."""
        return cls(
            id=data["id"],
            dataset_id=data["dataset_id"],
            input=data.get("input", {}),
            expected=data.get("expected"),
            metadata=data.get("metadata"),
            created_at=data.get("created_at"),
        )


DatasetItemInput = Union[Dict[str, Any], DatasetItem]


class Dataset:
    """
    A dataset for evaluation (sync).

    Supports batch insert and auto-pagination for iteration.
    Uses SyncHTTPClient internally - no event loop involvement.

    Example:
        >>> dataset = client.datasets.create(name="my-dataset")
        >>> dataset.insert([
        ...     {"input": {"text": "hello"}, "expected": {"label": "greeting"}},
        ... ])
        >>> for item in dataset:
        ...     print(item.input, item.expected)
    """

    def __init__(
        self,
        id: str,
        name: str,
        description: Optional[str],
        metadata: Optional[Dict[str, Any]],
        created_at: str,
        updated_at: str,
        _http_client: SyncHTTPClient,
        _debug: bool = False,
    ):
        """
        Initialize Dataset.

        Args:
            id: Dataset ID
            name: Dataset name
            description: Dataset description
            metadata: Additional metadata
            created_at: Creation timestamp
            updated_at: Last update timestamp
            _http_client: Internal HTTP client (injected by manager)
            _debug: Enable debug logging
        """
        self._id = id
        self._name = name
        self._description = description
        self._metadata = metadata
        self._created_at = created_at
        self._updated_at = updated_at
        self._http = _http_client
        self._debug = _debug

    @property
    def id(self) -> str:
        """Dataset ID."""
        return self._id

    @property
    def name(self) -> str:
        """Dataset name."""
        return self._name

    @property
    def description(self) -> Optional[str]:
        """Dataset description."""
        return self._description

    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        """Dataset metadata."""
        return self._metadata

    @property
    def created_at(self) -> str:
        """Creation timestamp."""
        return self._created_at

    @property
    def updated_at(self) -> str:
        """Last update timestamp."""
        return self._updated_at

    def _log(self, message: str, *args: Any) -> None:
        """Log debug messages."""
        if self._debug:
            print(f"[Brokle Dataset] {message}", *args)

    def _normalize_item(self, item: DatasetItemInput) -> Dict[str, Any]:
        """Normalize item input to API format."""
        if isinstance(item, DatasetItem):
            result: Dict[str, Any] = {"input": item.input}
            if item.expected is not None:
                result["expected"] = item.expected
            if item.metadata is not None:
                result["metadata"] = item.metadata
            return result
        elif isinstance(item, dict):
            if "input" not in item:
                raise ValueError("Item dict must have 'input' key")
            return item
        else:
            raise TypeError(f"Item must be dict or DatasetItem, got {type(item)}")

    def insert(self, items: List[DatasetItemInput]) -> int:
        """
        Insert items into the dataset.

        Args:
            items: List of items to insert. Each item can be:
                - A dict with 'input' (required), 'expected' (optional), 'metadata' (optional)
                - A DatasetItem instance

        Returns:
            Number of items created

        Raises:
            DatasetError: If the API request fails
            ValueError: If item format is invalid

        Example:
            >>> dataset.insert([
            ...     {"input": {"q": "2+2?"}, "expected": {"a": "4"}},
            ... ])
            1
        """
        if not items:
            return 0

        normalized = [self._normalize_item(item) for item in items]
        self._log(f"Inserting {len(normalized)} items into dataset {self._id}")

        try:
            raw_response = self._http.post(
                f"/v1/datasets/{self._id}/items",
                json={"items": normalized},
            )
            data = unwrap_response(raw_response, resource_type="DatasetItems")
            return int(data.get("created", len(normalized)))
        except ValueError as e:
            raise DatasetError(f"Failed to insert items: {e}")
        except Exception as e:
            raise DatasetError(f"Failed to insert items: {e}")

    def get_items(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> List[DatasetItem]:
        """
        Fetch items with pagination.

        Args:
            limit: Maximum number of items to return (default: 50)
            offset: Number of items to skip (default: 0)

        Returns:
            List of DatasetItem objects

        Raises:
            DatasetError: If the API request fails

        Example:
            >>> items = dataset.get_items(limit=10, offset=0)
            >>> for item in items:
            ...     print(item.input)
        """
        self._log(f"Fetching items from dataset {self._id}: limit={limit}, offset={offset}")

        try:
            raw_response = self._http.get(
                f"/v1/datasets/{self._id}/items",
                params={"limit": limit, "offset": offset},
            )
            data = unwrap_response(raw_response, resource_type="DatasetItems")
            items_data = data.get("items", [])
            return [DatasetItem.from_dict(item) for item in items_data]
        except ValueError as e:
            raise DatasetError(f"Failed to fetch items: {e}")
        except Exception as e:
            raise DatasetError(f"Failed to fetch items: {e}")

    def __iter__(self) -> Iterator[DatasetItem]:
        """
        Auto-paginating iterator over all items.

        Transparently fetches pages as needed.

        Example:
            >>> for item in dataset:
            ...     print(item.input, item.expected)
        """
        offset = 0
        limit = 50
        while True:
            items = self.get_items(limit=limit, offset=offset)
            if not items:
                break
            yield from items
            if len(items) < limit:
                break
            offset += limit

    def __len__(self) -> int:
        """
        Return total item count.

        Note: This requires an API call to fetch the count.

        Example:
            >>> len(dataset)
            42
        """
        try:
            raw_response = self._http.get(
                f"/v1/datasets/{self._id}/items",
                params={"limit": 1, "offset": 0},
            )
            data = unwrap_response(raw_response, resource_type="DatasetItems")
            return int(data.get("total", 0))
        except Exception:
            return 0

    def __repr__(self) -> str:
        """String representation."""
        return f"Dataset(id='{self._id}', name='{self._name}')"


class AsyncDataset:
    """
    A dataset for evaluation (async).

    Supports batch insert and auto-pagination for async iteration.
    Uses AsyncHTTPClient internally.

    Example:
        >>> dataset = await client.datasets.create(name="my-dataset")
        >>> await dataset.insert([
        ...     {"input": {"text": "hello"}, "expected": {"label": "greeting"}},
        ... ])
        >>> async for item in dataset:
        ...     print(item.input, item.expected)
    """

    def __init__(
        self,
        id: str,
        name: str,
        description: Optional[str],
        metadata: Optional[Dict[str, Any]],
        created_at: str,
        updated_at: str,
        _http_client: AsyncHTTPClient,
        _debug: bool = False,
    ):
        """
        Initialize AsyncDataset.

        Args:
            id: Dataset ID
            name: Dataset name
            description: Dataset description
            metadata: Additional metadata
            created_at: Creation timestamp
            updated_at: Last update timestamp
            _http_client: Internal async HTTP client (injected by manager)
            _debug: Enable debug logging
        """
        self._id = id
        self._name = name
        self._description = description
        self._metadata = metadata
        self._created_at = created_at
        self._updated_at = updated_at
        self._http = _http_client
        self._debug = _debug

    @property
    def id(self) -> str:
        """Dataset ID."""
        return self._id

    @property
    def name(self) -> str:
        """Dataset name."""
        return self._name

    @property
    def description(self) -> Optional[str]:
        """Dataset description."""
        return self._description

    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        """Dataset metadata."""
        return self._metadata

    @property
    def created_at(self) -> str:
        """Creation timestamp."""
        return self._created_at

    @property
    def updated_at(self) -> str:
        """Last update timestamp."""
        return self._updated_at

    def _log(self, message: str, *args: Any) -> None:
        """Log debug messages."""
        if self._debug:
            print(f"[Brokle AsyncDataset] {message}", *args)

    def _normalize_item(self, item: DatasetItemInput) -> Dict[str, Any]:
        """Normalize item input to API format."""
        if isinstance(item, DatasetItem):
            result: Dict[str, Any] = {"input": item.input}
            if item.expected is not None:
                result["expected"] = item.expected
            if item.metadata is not None:
                result["metadata"] = item.metadata
            return result
        elif isinstance(item, dict):
            if "input" not in item:
                raise ValueError("Item dict must have 'input' key")
            return item
        else:
            raise TypeError(f"Item must be dict or DatasetItem, got {type(item)}")

    async def insert(self, items: List[DatasetItemInput]) -> int:
        """
        Insert items into the dataset (async).

        Args:
            items: List of items to insert. Each item can be:
                - A dict with 'input' (required), 'expected' (optional), 'metadata' (optional)
                - A DatasetItem instance

        Returns:
            Number of items created

        Raises:
            DatasetError: If the API request fails
            ValueError: If item format is invalid

        Example:
            >>> await dataset.insert([
            ...     {"input": {"q": "2+2?"}, "expected": {"a": "4"}},
            ... ])
            1
        """
        if not items:
            return 0

        normalized = [self._normalize_item(item) for item in items]
        self._log(f"Inserting {len(normalized)} items into dataset {self._id}")

        try:
            raw_response = await self._http.post(
                f"/v1/datasets/{self._id}/items",
                json={"items": normalized},
            )
            data = unwrap_response(raw_response, resource_type="DatasetItems")
            return int(data.get("created", len(normalized)))
        except ValueError as e:
            raise DatasetError(f"Failed to insert items: {e}")
        except Exception as e:
            raise DatasetError(f"Failed to insert items: {e}")

    async def get_items(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> List[DatasetItem]:
        """
        Fetch items with pagination (async).

        Args:
            limit: Maximum number of items to return (default: 50)
            offset: Number of items to skip (default: 0)

        Returns:
            List of DatasetItem objects

        Raises:
            DatasetError: If the API request fails

        Example:
            >>> items = await dataset.get_items(limit=10, offset=0)
            >>> for item in items:
            ...     print(item.input)
        """
        self._log(f"Fetching items from dataset {self._id}: limit={limit}, offset={offset}")

        try:
            raw_response = await self._http.get(
                f"/v1/datasets/{self._id}/items",
                params={"limit": limit, "offset": offset},
            )
            data = unwrap_response(raw_response, resource_type="DatasetItems")
            items_data = data.get("items", [])
            return [DatasetItem.from_dict(item) for item in items_data]
        except ValueError as e:
            raise DatasetError(f"Failed to fetch items: {e}")
        except Exception as e:
            raise DatasetError(f"Failed to fetch items: {e}")

    async def __aiter__(self) -> AsyncIterator[DatasetItem]:
        """
        Auto-paginating async iterator over all items.

        Transparently fetches pages as needed.

        Example:
            >>> async for item in dataset:
            ...     print(item.input, item.expected)
        """
        offset = 0
        limit = 50
        while True:
            items = await self.get_items(limit=limit, offset=offset)
            if not items:
                break
            for item in items:
                yield item
            if len(items) < limit:
                break
            offset += limit

    async def count(self) -> int:
        """
        Return total item count (async).

        Example:
            >>> total = await dataset.count()
            >>> print(f"Dataset has {total} items")
        """
        try:
            raw_response = await self._http.get(
                f"/v1/datasets/{self._id}/items",
                params={"limit": 1, "offset": 0},
            )
            data = unwrap_response(raw_response, resource_type="DatasetItems")
            return int(data.get("total", 0))
        except Exception:
            return 0

    def __repr__(self) -> str:
        """String representation."""
        return f"AsyncDataset(id='{self._id}', name='{self._name}')"
