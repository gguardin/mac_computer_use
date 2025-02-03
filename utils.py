"""Utility functions for the application."""

import json
from typing import Any, Union


def truncate_for_logging(data: Any, max_length: int = 50) -> Any:
    """Truncate data for logging purposes.

    Args:
        data: The data to truncate. Can be any JSON-serializable type.
        max_length: Maximum length for string values before truncation.

    Returns:
        Truncated version of the data, safe for logging.
    """
    if isinstance(data, dict):
        return {k: truncate_for_logging(v, max_length) for k, v in data.items()}
    elif isinstance(data, list):
        return [truncate_for_logging(item, max_length) for item in data]
    elif isinstance(data, str) and len(data) > max_length:
        # Check if it looks like base64 (long string without spaces)
        if len(data) > 100 and " " not in data:
            return data[:max_length] + "..."
        # For other strings, try to break at word boundaries
        return data[:max_length].rsplit(" ", 1)[0] + "..."
    return data


def safe_dumps(obj: Any, **kwargs) -> str:
    """Safely dump an object to JSON string with truncation for logging.

    Args:
        obj: The object to serialize
        **kwargs: Additional arguments passed to json.dumps

    Returns:
        JSON string with truncated content
    """
    if "indent" not in kwargs:
        kwargs["indent"] = 2
    return json.dumps(truncate_for_logging(obj), **kwargs)
