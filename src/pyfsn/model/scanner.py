"""Async filesystem scanner using QThread."""

import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QThread, pyqtSignal

from pyfsn.model.node import Node, NodeType


class ScanProgress:
    """Progress information for a scan operation."""

    def __init__(
        self,
        current_path: Path,
        nodes_found: int,
        is_complete: bool = False,
        error: str | None = None,
    ) -> None:
        """Initialize scan progress.

        Args:
            current_path: Path currently being scanned
            nodes_found: Total number of nodes found so far
            is_complete: Whether the scan is complete
            error: Error message if an error occurred
        """
        self.current_path = current_path
        self.nodes_found = nodes_found
        self.is_complete = is_complete
        self.error = error


class ScannerWorker(QThread):
    """Worker thread for scanning directories asynchronously."""

    # Signals
    progress = pyqtSignal(object)  # Emits ScanProgress
    node_found = pyqtSignal(Node)  # Emits newly discovered nodes
    finished = pyqtSignal(Node)  # Emits root node when complete
    error = pyqtSignal(str)  # Emits error messages

    def __init__(
        self,
        root_path: Path,
        lazy_load: bool = True,
        lazy_depth: int = 2,
        max_workers: int = 4,
    ) -> None:
        """Initialize the scanner worker.

        Args:
            root_path: Root path to scan
            lazy_load: Whether to use lazy loading for deep hierarchies
            lazy_depth: Depth at which to start lazy loading
            max_workers: Maximum number of threads for parallel scanning
        """
        super().__init__()
        self.root_path = root_path
        self.lazy_load = lazy_load
        self.lazy_depth = lazy_depth
        self.max_workers = max_workers
        self._is_running = True

    def stop(self) -> None:
        """Stop the scanning process."""
        self._is_running = False

    def run(self) -> None:
        """Run the scan operation."""
        try:
            root_node = self._scan_node(self.root_path, depth=0)
            self.finished.emit(root_node)
        except Exception as e:
            self.error.emit(f"Scan failed: {e}")
        finally:
            self.progress.emit(
                ScanProgress(
                    current_path=self.root_path,
                    nodes_found=0,
                    is_complete=True,
                )
            )

    def _scan_node(self, path: Path, depth: int, parent: Node | None = None) -> Node:
        """Scan a single node and optionally its children.

        Args:
            path: Path to scan
            depth: Current depth in the tree
            parent: Parent node

        Returns:
            The scanned node
        """
        if not self._is_running:
            raise InterruptedError("Scan interrupted")

        # Create the node
        node = Node.from_path(path, parent=parent)
        self.node_found.emit(node)

        # For directories, scan children
        if node.is_directory:
            if self.lazy_load and depth >= self.lazy_depth:
                # Mark as not loaded for lazy loading
                node.is_loaded = False
            else:
                self._scan_children(node, depth)
                node.is_loaded = True

        return node

    def _scan_children(self, node: Node, depth: int) -> None:
        """Scan children of a directory node.

        Args:
            node: Directory node to scan children for
            depth: Current depth in the tree
        """
        try:
            entries = self._get_sorted_entries(node.path)
            self.progress.emit(
                ScanProgress(
                    current_path=node.path,
                    nodes_found=len(node.children),
                )
            )

            for entry in entries:
                if not self._is_running:
                    break

                child_path = node.path / entry
                child_node = self._scan_node(child_path, depth + 1, parent=node)
                node.add_child(child_node)

        except PermissionError:
            # Skip directories we can't read
            pass
        except OSError as e:
            self.error.emit(f"Error reading {node.path}: {e}")

    def _get_sorted_entries(self, path: Path) -> list[str]:
        """Get directory entries sorted by type and name.

        Directories come first, then files, both sorted alphabetically.

        Args:
            path: Directory path to read

        Returns:
            Sorted list of entry names
        """
        try:
            entries = list(path.iterdir())
        except PermissionError:
            return []

        # Separate directories and files
        dirs: list[str] = []
        files: list[str] = []

        for entry in entries:
            name = entry.name
            # Skip hidden files/directories
            if name.startswith("."):
                continue

            if entry.is_dir():
                dirs.append(name)
            else:
                files.append(name)

        # Sort and combine
        return sorted(dirs) + sorted(files)


class Scanner:
    """High-level scanner interface."""

    def __init__(self, lazy_load: bool = True, lazy_depth: int = 2) -> None:
        """Initialize the scanner.

        Args:
            lazy_load: Whether to use lazy loading for deep hierarchies
            lazy_depth: Depth at which to start lazy loading
        """
        self.lazy_load = lazy_load
        self.lazy_depth = lazy_depth
        self._worker: ScannerWorker | None = None
        self._executor = ThreadPoolExecutor(max_workers=4)

    def scan(self, path: Path) -> Node:
        """Synchronously scan a path.

        Args:
            path: Path to scan

        Returns:
            Root node of the scanned tree
        """
        self._worker = None
        root = Node.from_path(path)

        if root.is_directory:
            self._load_children_sync(root, 0)

        return root

    def scan_async(
        self,
        path: Path,
        on_progress: Callable[[ScanProgress], None] | None = None,
        on_node_found: Callable[[Node], None] | None = None,
        on_finished: Callable[[Node], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> ScannerWorker:
        """Asynchronously scan a path.

        Args:
            path: Path to scan
            on_progress: Callback for progress updates
            on_node_found: Callback when a new node is found
            on_finished: Callback when scan is complete
            on_error: Callback for errors

        Returns:
            The scanner worker thread
        """
        self._worker = ScannerWorker(
            path, lazy_load=self.lazy_load, lazy_depth=self.lazy_depth
        )

        if on_progress is not None:
            self._worker.progress.connect(on_progress)
        if on_node_found is not None:
            self._worker.node_found.connect(on_node_found)
        if on_finished is not None:
            self._worker.finished.connect(on_finished)
        if on_error is not None:
            self._worker.error.connect(on_error)

        self._worker.start()
        return self._worker

    def stop(self) -> None:
        """Stop any ongoing scan."""
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait()

    def load_children(self, node: Node) -> None:
        """Load children for a lazy-loaded node.

        Args:
            node: Node to load children for
        """
        if not node.is_directory or node.is_loaded:
            return

        self._load_children_sync(node, node.depth)
        node.is_loaded = True

    def _load_children_sync(self, node: Node, depth: int) -> None:
        """Synchronously load children for a node.

        Args:
            node: Node to load children for
            depth: Current depth
        """
        try:
            entries = self._get_sorted_entries(node.path)

            for entry in entries:
                child_path = node.path / entry
                child_node = Node.from_path(child_path, parent=node)
                node.add_child(child_node)

                # Recursively load directory children if within lazy_depth
                if child_node.is_directory:
                    if self.lazy_load and depth + 1 >= self.lazy_depth:
                        # Mark as not loaded for lazy loading
                        child_node.is_loaded = False
                    else:
                        # Recursively load children
                        self._load_children_sync(child_node, depth + 1)
                        child_node.is_loaded = True

        except PermissionError:
            # Skip directories we can't read
            pass

    def _get_sorted_entries(self, path: Path) -> list[str]:
        """Get directory entries sorted by type and name."""
        try:
            entries = list(path.iterdir())
        except PermissionError:
            return []

        dirs: list[str] = []
        files: list[str] = []

        for entry in entries:
            name = entry.name
            if name.startswith("."):
                continue

            if entry.is_dir():
                dirs.append(name)
            else:
                files.append(name)

        return sorted(dirs) + sorted(files)

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self._executor.shutdown(wait=False)
