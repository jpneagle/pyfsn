"""Error handling utilities for pyfsn.

Provides exception classes and error handling helpers for
robust file system navigation.
"""

from pathlib import Path
from typing import Callable


class PyfsnError(Exception):
    """Base exception for pyfsn errors."""

    pass


class ScanError(PyfsnError):
    """Exception raised when scanning fails."""

    def __init__(self, path: Path, reason: str) -> None:
        """Initialize scan error.

        Args:
            path: Path that failed to scan
            reason: Reason for failure
        """
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to scan {path}: {reason}")


class LayoutError(PyfsnError):
    """Exception raised when layout calculation fails."""

    def __init__(self, reason: str) -> None:
        """Initialize layout error.

        Args:
            reason: Reason for failure
        """
        self.reason = reason
        super().__init__(f"Layout calculation failed: {reason}")


class RenderError(PyfsnError):
    """Exception raised when rendering fails."""

    def __init__(self, reason: str) -> None:
        """Initialize render error.

        Args:
            reason: Reason for failure
        """
        self.reason = reason
        super().__init__(f"Rendering failed: {reason}")


class ValidationError(PyfsnError):
    """Exception raised when input validation fails."""

    def __init__(self, field: str, value: object, expected: str) -> None:
        """Initialize validation error.

        Args:
            field: Field name that failed validation
            value: Invalid value
            expected: Expected type/description
        """
        self.field = field
        self.value = value
        self.expected = expected
        super().__init__(f"Validation failed for '{field}': expected {expected}, got {value!r}")


class FileOpenError(PyfsnError):
    """ファイルオープンエラー

    OSのデフォルトアプリケーションでファイルを開く際に発生するエラー。
    """

    def __init__(self, path: Path, reason: str) -> None:
        """Initialize FileOpenError.

        Args:
            path: 開けなかったファイルのパス
            reason: エラーの理由
        """
        self.path = path
        self.reason = reason
        super().__init__(f"Cannot open file {path}: {reason}")


def safe_path(path: Path | str) -> Path:
    """Safely convert to Path object, handling errors.

    Args:
        path: Path to convert

    Returns:
        Path object

    Raises:
        ValidationError: If path is invalid
    """
    try:
        result = Path(path).resolve()
        # Check if path exists (optional, depending on use case)
        return result
    except Exception as e:
        raise ValidationError("path", path, f"valid path: {e}")


def validate_directory(path: Path) -> None:
    """Validate that a path is a directory.

    Args:
        path: Path to validate

    Raises:
        ValidationError: If path is not a directory
    """
    if not path.is_dir():
        raise ValidationError("path", path, "directory")


def validate_range(value: int | float, min_val: int | float, max_val: int | float, name: str = "value") -> None:
    """Validate that a value is within range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of the value for error messages

    Raises:
        ValidationError: If value is out of range
    """
    if not (min_val <= value <= max_val):
        raise ValidationError(name, value, f"value between {min_val} and {max_val}")


def handle_errors(
    error_types: tuple[type[Exception], ...] = (Exception,),
    default_return: object = None,
    on_error: Callable[[Exception], None] | None = None,
) -> Callable:
    """Decorator for handling errors in functions.

    Args:
        error_types: Tuple of exception types to catch
        default_return: Value to return on error
        on_error: Optional callback to call with the exception

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_types as e:
                if on_error:
                    on_error(e)
                if default_return is not None:
                    return default_return
                raise
        return wrapper
    return decorator
