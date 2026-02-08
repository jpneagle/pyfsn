"""Node class representing a file system entry."""

import os
import stat
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Self


class NodeType(Enum):
    """Type of file system node."""

    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"


@dataclass
class Node:
    """Represents a file or directory in the file system.

    Attributes:
        path: Absolute path to the file/directory
        name: Base name of the file/directory
        size: Size in bytes (0 for directories)
        type: NodeType indicating what kind of node this is
        permissions: Unix permission mode (e.g., 0o755)
        mtime: Modification time as Unix timestamp
        children: List of child nodes (only for directories)
        parent: Parent node (None for root)
        is_loaded: Whether children have been loaded (for lazy loading)
        is_expanded: Whether this directory is expanded in the view
    """

    path: Path
    name: str
    size: int
    type: NodeType
    permissions: int
    mtime: float
    children: list[Self] = field(default_factory=list)
    parent: Self | None = None
    is_loaded: bool = False
    is_expanded: bool = False

    def __hash__(self) -> int:
        """Hash based on path."""
        return hash(self.path)

    def __eq__(self, other: object) -> bool:
        """Equality based on path."""
        if not isinstance(other, Node):
            return NotImplemented
        return self.path == other.path

    @classmethod
    def from_path(cls, path: Path, parent: Self | None = None) -> Self:
        """Create a Node from a filesystem path.

        Args:
            path: Path to the file/directory
            parent: Parent node (if any)

        Returns:
            A new Node instance populated with metadata from the filesystem
        """
        try:
            stat_info = path.lstat()
            size = stat_info.st_size
            permissions = stat.S_IMODE(stat_info.st_mode)
            mtime = stat_info.st_mtime
        except OSError:
            # Handle permission errors or broken symlinks
            size = 0
            permissions = 0
            mtime = 0.0

        # Determine node type
        if path.is_symlink():
            node_type = NodeType.SYMLINK
        elif path.is_dir():
            node_type = NodeType.DIRECTORY
        else:
            node_type = NodeType.FILE

        return cls(
            path=path.resolve(),
            name=path.name,
            size=size,
            type=node_type,
            permissions=permissions,
            mtime=mtime,
            parent=parent,
        )

    @property
    def is_directory(self) -> bool:
        """Check if this node is a directory."""
        return self.type == NodeType.DIRECTORY

    @property
    def is_image_file(self) -> bool:
        """Check if this node is an image file.

        Returns True for common image formats (PNG, JPG, JPEG, GIF, BMP, WebP, etc.).
        """
        if self.type != NodeType.FILE:
            return False

        # Get file extension
        suffix = self.path.suffix.lower()

        # Common image file extensions
        image_extensions = {
            '.png', '.jpg', '.jpeg', '.jpe', '.jfif',
            '.gif', '.bmp', '.webp', '.svg',
            '.ico', '.tiff', '.tif', '.psd',
            '.raw', '.cr2', '.nef', '.arw',
            '.heic', '.heif', '.avif',
        }

        return suffix in image_extensions

    @property
    def is_video_file(self) -> bool:
        """Check if this node is a video file.

        Returns True for common video formats (MP4, AVI, MKV, MOV, WEBM, etc.).
        """
        if self.type != NodeType.FILE:
            return False

        # Get file extension
        suffix = self.path.suffix.lower()

        # Common video file extensions
        video_extensions = {
            '.mp4', '.avi', '.mkv', '.mov', '.wmv',
            '.flv', '.webm', '.m4v', '.mpg', '.mpeg',
            '.3gp', '.ogv', '.ts', '.m2ts', '.mts',
            '.vob', '.rm', '.rmvb', '.asf', '.divx',
            '.xvid', '.f4v', '.mxf', '.qt',
        }

        return suffix in video_extensions

    @property
    def is_file(self) -> bool:
        """Check if this node is a file."""
        return self.type == NodeType.FILE

    @property
    def is_symlink(self) -> bool:
        """Check if this node is a symlink."""
        return self.type == NodeType.SYMLINK

    @property
    def depth(self) -> int:
        """Get the depth of this node in the tree (root = 0)."""
        depth = 0
        current = self.parent
        while current is not None:
            depth += 1
            current = current.parent
        return depth

    @property
    def total_size(self) -> int:
        """Get total size including all descendants (for directories)."""
        if self.is_file:
            return self.size
        return sum(child.total_size for child in self.children)

    @property
    def file_count(self) -> int:
        """Get total number of files in this subtree."""
        if self.is_file:
            return 1
        return sum(child.file_count for child in self.children)

    @property
    def directory_count(self) -> int:
        """Get total number of directories in this subtree."""
        if not self.is_directory:
            return 0
        return 1 + sum(child.directory_count for child in self.children)

    def add_child(self, child: Self) -> None:
        """Add a child node to this directory.

        Args:
            child: Child node to add
        """
        if not self.is_directory:
            raise ValueError("Cannot add children to non-directory nodes")
        child.parent = self
        self.children.append(child)

    def remove_child(self, child: Self) -> bool:
        """Remove a child node from this directory.

        Args:
            child: Child node to remove

        Returns:
            True if child was removed, False if not found
        """
        try:
            self.children.remove(child)
            child.parent = None
            return True
        except ValueError:
            return False

    def find_child_by_name(self, name: str) -> Self | None:
        """Find a direct child by name.

        Args:
            name: Name of the child to find

        Returns:
            The child node if found, None otherwise
        """
        for child in self.children:
            if child.name == name:
                return child
        return None

    def get_all_descendants(self) -> list[Self]:
        """Get all descendant nodes recursively.

        Returns:
            List of all descendant nodes (not including self)
        """
        descendants: list[Node] = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_all_descendants())
        return descendants

    def invalidate_children(self) -> None:
        """Mark children as unloaded for lazy reloading."""
        self.children.clear()
        self.is_loaded = False

    def __repr__(self) -> str:
        """String representation of the node."""
        return f"Node({self.type.value}, {self.name}, depth={self.depth})"
