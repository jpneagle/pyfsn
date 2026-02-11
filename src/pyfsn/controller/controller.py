"""Main controller for the application.

Coordinates between the Model, View, and Layout layers, handling
user input and managing application state.
"""

from pathlib import Path
from typing import Callable
import subprocess
import sys

from PyQt6.QtCore import QObject, pyqtSignal, QThread, QTimer
from PyQt6.QtWidgets import QMessageBox

from pyfsn.model.node import Node, NodeType
from pyfsn.model.scanner import Scanner, ScannerWorker, ScanProgress
from pyfsn.errors import FileOpenError
from pyfsn.layout.engine import LayoutEngine, LayoutConfig, LayoutResult
from pyfsn.view.renderer import Renderer
from pyfsn.view.camera import CameraMode
from pyfsn.view.main_window import MainWindow
from pyfsn.controller.input_handler import InputHandler


class Controller(QObject):
    """Main application controller.

    Manages the application state and coordinates between layers.
    """

    # Signals for UI updates
    scene_loaded = pyqtSignal(int)  # Emits node count
    node_selected = pyqtSignal(object)  # Emits selected node
    node_focused = pyqtSignal(object)  # Emits focused node
    scan_progress = pyqtSignal(str)  # Emits status message
    scan_complete = pyqtSignal()  # Emits when scan is done
    navigation_state_changed = pyqtSignal(bool, bool)  # Emits (can_go_back, can_go_forward)

    def __init__(self, root_path: Path) -> None:
        """Initialize controller.

        Args:
            root_path: Root directory path to visualize
        """
        super().__init__()

        self._root_path = root_path
        self._original_root_path = root_path
        self._root_node: Node | None = None

        # Create main window
        self._window = MainWindow(root_path)

        # Get renderer from window
        self._renderer = self._window.renderer
        self._camera = self._renderer.camera

        # Set tooltip reference on renderer for hover functionality
        self._renderer.set_tooltip(self._window.file_tooltip)

        # Create input handler
        self._input_handler = InputHandler(self._camera, self._renderer)
        self._renderer.set_input_handler(self._input_handler)

        # Create layout engine
        self._layout_config = LayoutConfig(
            node_size=1.0,
            dir_size=2.0,
            spacing=0.5,
            max_depth=5,
        )
        self._layout_engine = LayoutEngine(self._layout_config)

        # Create scanner
        self._scanner = Scanner()
        self._scan_worker: ScannerWorker | None = None

        # Scene data
        self._nodes: dict[str, Node] = {}
        self._positions: dict[str, object] = {}
        self._layout_result: LayoutResult | None = None

        # Selection state
        self._selected_nodes: set[Node] = set()
        self._focused_node: Node | None = None

        # Search state
        self._search_results: list[Node] = []
        self._current_search_index = 0

        # Navigation history state
        self._back_stack: list[Path] = []
        self._forward_stack: list[Path] = []

        # Filter state (Workstream F - Advanced filtering)
        self._active_filters: dict = {}
        self._filtered_nodes: dict[str, Node] = {}

        # Text overlay update timer
        self._overlay_timer = QTimer()
        self._overlay_timer.timeout.connect(self._update_text_overlay)
        self._overlay_timer.start(100)  # Update labels 10 times per second

        # Connect signals
        self._connect_signals()
        self._connect_input_handler()

        # Install event filter on renderer
        self._renderer.installEventFilter(self)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self.scene_loaded.connect(self._window.update_stats)
        self.scan_progress.connect(self._window.set_status_message)

        # Connect window signals
        self._window.directory_changed.connect(self._change_directory)
        self._window.search_requested.connect(self._perform_search)
        self._window.file_tree.node_selected.connect(self._on_tree_node_selected)
        self._window.tree_node_double_clicked.connect(self._on_tree_node_double_clicked)

        # Connect navigation history signals
        self._window.go_back_requested.connect(self.go_back)
        self._window.go_forward_requested.connect(self.go_forward)

        # Connect navigation state signal
        self.navigation_state_changed.connect(self._window.update_navigation_state)

        # Connect window signals
        self._window.refresh_requested.connect(self.refresh)

        # Connect filter panel signal (Workstream F - Advanced filtering)
        self._window.filter_changed.connect(self._apply_filters)

    def _connect_input_handler(self) -> None:
        """Connect input handler callbacks."""
        self._input_handler.set_node_clicked_callback(self._on_node_clicked)
        self._input_handler.set_node_focused_callback(self._on_node_focused)
        self._input_handler.set_selection_changed_callback(self._on_selection_changed)
        self._input_handler.set_navigate_next_callback(self._select_next_node)
        self._input_handler.set_navigate_previous_callback(self._select_previous_node)
        self._input_handler.set_camera_mode_changed_callback(self._on_camera_mode_changed)

    def start(self) -> None:
        """Start the application - begin scanning."""
        self._start_scan()

    def show(self) -> None:
        """Show the main window."""
        self._window.show()

    # Event filtering

    def eventFilter(self, obj, event) -> bool:
        """Filter events from the renderer.

        Args:
            obj: Object sending the event
            event: Event object

        Returns:
            True if event was handled
        """
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QMouseEvent, QWheelEvent, QKeyEvent

        if obj == self._renderer:
            event_type = event.type()

            if event_type == QEvent.Type.MouseButtonPress:
                return self._input_handler.mouse_press_event(event)
            elif event_type == QEvent.Type.MouseButtonRelease:
                return self._input_handler.mouse_release_event(event)
            elif event_type == QEvent.Type.MouseMove:
                return self._input_handler.mouse_move_event(event)
            elif event_type == QEvent.Type.Wheel:
                return self._input_handler.wheel_event(event)
            elif event_type == QEvent.Type.MouseButtonDblClick:
                return self._input_handler.mouse_double_click_event(event)
            elif event_type == QEvent.Type.KeyPress:
                return self._input_handler.key_press_event(event)
            elif event_type == QEvent.Type.KeyRelease:
                return self._input_handler.key_release_event(event)

        return super().eventFilter(obj, event)

    # Directory management

    def _change_directory(self, path: Path) -> None:
        """Change the root directory.

        Args:
            path: New root directory path
        """
        # Push current path to back stack before changing
        if self._root_path != path:
            self._back_stack.append(self._root_path)
            # Clear forward stack when navigating to new path
            self._forward_stack.clear()

        self._root_path = path
        self._window.set_root_path(path)

        # Clear current scene
        self._nodes.clear()
        self._positions.clear()
        self._selected_nodes.clear()
        self._focused_node = None
        
        # Clear renderer selection (to remove spotlights)
        if self._renderer:
            self._renderer.clear_selection()

        # Start new scan
        self._start_scan()

        # Update navigation state
        self._emit_navigation_state()

    # Scanning

    def _start_scan(self) -> None:
        """Start scanning the root directory."""
        self.scan_progress.emit(f"Scanning {self._root_path}...")

        # Create root node
        self._root_node = Node.from_path(self._root_path)
        self._nodes[str(self._root_node.path)] = self._root_node

        # Start async scan - ScannerWorker is already a QThread
        self._scan_worker = ScannerWorker(self._root_path)

        # Connect worker signals
        self._scan_worker.progress.connect(self._on_scan_progress)
        self._scan_worker.finished.connect(self._on_scan_complete)
        self._scan_worker.error.connect(self._on_scan_error)

        # Start the worker thread
        self._scan_worker.start()

    def _on_scan_progress(self, progress: ScanProgress) -> None:
        """Handle scan progress update.

        Args:
            progress: ScanProgress object with current status
        """
        # Create message from progress info
        message = f"Scanning: {progress.current_path} ({progress.nodes_found} nodes)"
        self.scan_progress.emit(message)

    def _on_scan_complete(self, root: Node) -> None:
        """Handle scan completion.

        Args:
            root: Root node with all children loaded
        """
        # Cleanup worker thread
        if self._scan_worker:
            self._scan_worker.quit()
            self._scan_worker.wait()
            self._scan_worker = None

        # Update root node with scanned tree (has children populated)
        self._root_node = root

        # Build node dictionary
        self._nodes.clear()
        self._nodes[str(root.path)] = root
        for descendant in root.get_all_descendants():
            self._nodes[str(descendant.path)] = descendant

        # Calculate layout
        self._calculate_layout()

        # Load into renderer
        self._load_scene()

        # Position camera to view the scene
        self._reset_camera_to_scene()

        # Load into renderer (this also updates camera in renderer)
        self._load_scene()

        # Update input handler with scene data
        self._input_handler.set_scene_data(self._nodes, self._positions)

        # Load file tree
        self._window.file_tree.load_tree(root)

        # Emit completion signal
        self.scan_complete.emit()
        self.scan_progress.emit(f"Loaded {len(self._nodes)} items from {self._root_path}")

    def _on_scan_error(self, error_message: str) -> None:
        """Handle scan error.

        Args:
            error_message: Error message
        """
        self.scan_progress.emit(f"Error: {error_message}")

    # Layout

    def _calculate_layout(self) -> None:
        """Calculate 3D layout for the scanned nodes."""
        if self._root_node is None:
            return

        self.scan_progress.emit("Calculating layout...")
        self._layout_result = self._layout_engine.calculate_layout(self._root_node)
        self._positions = self._layout_result.positions

    # Rendering

    def _load_scene(self) -> None:
        """Load the scene into the renderer."""
        if self._layout_result is None:
            return

        # Get selected paths
        selected_paths = {str(n.path) for n in self._selected_nodes}

        # Load into renderer
        self._renderer.load_layout(self._layout_result, self._nodes, selected_paths)

        # Update mini map with scene data
        self._update_mini_map()

        # Emit scene loaded signal
        self.scene_loaded.emit(len(self._nodes))

    def _reset_camera_to_scene(self) -> None:
        """Reset camera to view the entire scene."""
        import numpy as np

        if self._layout_result is None or self._layout_result.bounds is None:
            return

        bounds = self._layout_result.bounds

        # Use root directory position as target (not scene bounds center)
        # This ensures the clicked directory is centered in view
        root_path = str(self._root_node.path)
        if root_path in self._positions:
            root_pos = self._positions[root_path]
            target_center = root_pos.center
        else:
            # Fallback to scene bounds center if root position not found
            target_center = bounds.position.center

        # Calculate appropriate distance based on scene size
        size = max(
            bounds.position.width,
            bounds.position.height,
            bounds.position.depth,
        )
        distance = size * 2.0 + 10.0  # Ensure we're far enough to see everything

        # Position camera to look at root directory center from a diagonal angle
        camera_pos = np.array([
            target_center[0] + distance * 0.5,
            target_center[1] + distance * 0.5,
            target_center[2] - distance,  # Z negative to look at positive Z objects
        ], dtype=np.float32)

        target_pos = np.array(target_center, dtype=np.float32)

        # Update camera state using public API
        self._camera.set_position_target(camera_pos, target_pos)

    # Text overlay (labels)

    def _update_text_overlay(self) -> None:
        """Update the text overlay with node labels."""
        if not self._window.show_labels or self._layout_result is None:
            return

        overlay = self._window.text_overlay
        if overlay is None:
            return

        # Collect labels for visible nodes
        labels = []
        focused_label = None
        focused_pos = None

        for path_str, position in self._positions.items():
            if path_str not in self._nodes:
                continue

            node = self._nodes[path_str]

            # Get screen position
            center = position.center
            import numpy as np
            world_pos = np.array([center[0], center[1], center[2]], dtype=np.float32)
            screen_pos = self._renderer.get_screen_position(world_pos)

            if screen_pos is None:
                continue  # Behind camera or off-screen

            x, y = screen_pos

            # Check if this is the focused node
            if self._focused_node and node == self._focused_node:
                focused_label = node.name
                focused_pos = screen_pos
            elif node.is_file:
                # Show only file names (directories shown on ground as handwritten style)
                labels.append((node.name, x, y))

        # Update overlay
        if focused_label and focused_pos:
            overlay.set_focused_label(focused_label, focused_pos[0], focused_pos[1])
        else:
            overlay.set_focused_label(None)

        overlay.set_labels(labels)

    # Search

    def _perform_search(self, query: str) -> None:
        """Perform a search for nodes by name with spotlight visualization.

        Phase 3.2: Spotlight Search Visualization

        Args:
            query: Search query string
        """
        if not query or not self._nodes:
            self._search_results.clear()
            # Clear spotlight visualization
            if hasattr(self._renderer, 'clear_spotlight_search'):
                self._renderer.clear_spotlight_search()
            return

        query_lower = query.lower()
        self._search_results = [
            node for node in self._nodes.values()
            if query_lower in node.name.lower()
        ]

        if self._search_results:
            self._current_search_index = 0

            # Phase 3.2: Start spotlight search visualization
            matching_nodes = set(self._search_results)
            if hasattr(self._renderer, 'start_spotlight_search'):
                self._renderer.start_spotlight_search(query, matching_nodes)

            self._show_search_result()
            self.scan_progress.emit(f"Found {len(self._search_results)} results for '{query}'")
        else:
            self.scan_progress.emit(f"No results found for '{query}'")
            # Clear spotlight when no results
            if hasattr(self._renderer, 'clear_spotlight_search'):
                self._renderer.clear_spotlight_search()

    def _show_search_result(self) -> None:
        """Show the current search result."""
        if not self._search_results:
            return

        node = self._search_results[self._current_search_index]

        # Focus on the node
        if hasattr(self._renderer, 'snap_camera_to_node'):
            self._renderer.snap_camera_to_node(id(node))

        # Select the node
        self._selected_nodes.clear()
        self._selected_nodes.add(node)
        self._focused_node = node

        # Update visual selection
        self._update_selection_visual()

        # Update file tree
        self._window.file_tree.select_node(node)

        self.scan_progress.emit(f"Result {self._current_search_index + 1}/{len(self._search_results)}: {node.name}")

    def next_search_result(self) -> None:
        """Navigate to the next search result."""
        if not self._search_results:
            return

        self._current_search_index = (self._current_search_index + 1) % len(self._search_results)
        self._show_search_result()

    def previous_search_result(self) -> None:
        """Navigate to the previous search result."""
        if not self._search_results:
            return

        self._current_search_index = (self._current_search_index - 1) % len(self._search_results)
        self._show_search_result()

    # Filter methods (Workstream F - Advanced filtering)

    def _apply_filters(self, filters: dict) -> None:
        """Apply advanced filters to the node set.

        Args:
            filters: Dictionary with filter criteria including:
                - name_contains: str
                - min_size: int | None
                - max_size: int | None
                - min_mtime: float | None
                - show_files: bool
                - show_dirs: bool
                - show_symlinks: bool
                - include_ancestors: bool
        """
        self._active_filters = filters

        if not filters:
            # No filters, show all nodes
            self._filtered_nodes.clear()
            self._load_scene()
            self.scan_progress.emit(f"Showing all {len(self._nodes)} items")
            return

        # Filter nodes based on criteria
        self._filtered_nodes.clear()
        matched_nodes: dict[str, Node] = {}
        filtered_count = 0

        for path_str, node in self._nodes.items():
            # Apply type filters
            if not filters.get('show_files', True) and node.is_file:
                continue
            if not filters.get('show_dirs', True) and node.is_directory:
                continue
            if not filters.get('show_symlinks', True) and node.is_symlink:
                continue

            # Apply name filter
            name_contains = filters.get('name_contains', '')
            if name_contains and name_contains.lower() not in node.name.lower():
                continue

            # Apply size filters
            min_size = filters.get('min_size')
            max_size = filters.get('max_size')
            if min_size is not None and node.size < min_size:
                continue
            if max_size is not None and node.size > max_size:
                continue

            # Apply age filter
            min_mtime = filters.get('min_mtime')
            if min_mtime is not None and node.mtime < min_mtime:
                continue

            # Node passed all filters
            matched_nodes[path_str] = node
            filtered_count += 1

        # Include ancestors if enabled
        if filters.get('include_ancestors', True):
            for path_str, node in matched_nodes.items():
                # Add the matched node
                self._filtered_nodes[path_str] = node

                # Add all ancestors
                current = node.parent
                while current is not None:
                    ancestor_path = str(current.path)
                    if ancestor_path not in self._filtered_nodes:
                        self._filtered_nodes[ancestor_path] = current
                    current = current.parent
        else:
            # No ancestor inclusion, just use matched nodes
            self._filtered_nodes = matched_nodes.copy()

        # Update the scene with filtered nodes
        self._load_filtered_scene()

        # Update status message
        filter_desc = self._get_filter_description(filters)
        ancestor_note = " + ancestors" if filters.get('include_ancestors', True) and len(self._filtered_nodes) > filtered_count else ""
        self.scan_progress.emit(f"Showing {filtered_count}/{len(self._nodes)} items{filter_desc}{ancestor_note}")

    def _get_filter_description(self, filters: dict) -> str:
        """Get a human-readable description of active filters.

        Args:
            filters: Filter criteria dictionary

        Returns:
            Description string
        """
        parts = []

        if filters.get('name_contains'):
            parts.append(f"name:'{filters['name_contains']}'")

        if filters.get('min_size') or filters.get('max_size'):
            min_size = filters.get('min_size', 0)
            max_size = filters.get('max_size', '∞')
            parts.append(f"size:{min_size}-{max_size}")

        if filters.get('min_mtime'):
            import time
            days = int((time.time() - filters['min_mtime']) / (24 * 60 * 60))
            parts.append(f"age:<{days}d")

        type_parts = []
        if filters.get('show_files', True):
            type_parts.append("files")
        if filters.get('show_dirs', True):
            type_parts.append("dirs")
        if filters.get('show_symlinks', True):
            type_parts.append("symlinks")

        if len(type_parts) < 3:
            parts.append(f"type:{','.join(type_parts)}")

        return f" ({', '.join(parts)})" if parts else ""

    def _load_filtered_scene(self) -> None:
        """Load the filtered scene into the renderer.

        Only shows nodes that match the current filters.
        Maintains wire connections between filtered nodes.
        """
        if not self._filtered_nodes:
            # No nodes match filter, show empty scene
            if hasattr(self._renderer, 'load_layout'):
                empty_result = LayoutResult(positions={}, connections=[], bounds=None)
                self._renderer.load_layout(empty_result, {}, set())
            return

        # Create filtered positions dict
        filtered_positions = {
            path_str: self._positions[path_str]
            for path_str in self._filtered_nodes.keys()
            if path_str in self._positions
        }

        # Filter connections: only include connections where both endpoints are in the filtered set
        filtered_connections = []
        if self._layout_result and self._layout_result.connections:
            filtered_paths = set(self._filtered_nodes.keys())
            for parent_path, child_path in self._layout_result.connections:
                if parent_path in filtered_paths and child_path in filtered_paths:
                    filtered_connections.append((parent_path, child_path))

        # Get selected paths from filtered nodes
        selected_paths = {
            str(n.path) for n in self._selected_nodes
            if str(n.path) in self._filtered_nodes
        }

        # Load into renderer
        if hasattr(self._renderer, 'load_layout'):
            # Create a filtered layout result with connections
            filtered_result = LayoutResult(
                positions=filtered_positions,
                connections=filtered_connections,
                bounds=None
            )
            self._renderer.load_layout(filtered_result, self._filtered_nodes, selected_paths)

        # Update input handler with filtered scene data
        self._input_handler.set_scene_data(self._filtered_nodes, filtered_positions)

    # Input callbacks

    def _on_node_clicked(self, node: Node, is_double_click: bool) -> None:
        """Handle node click from input handler.

        Args:
            node: Clicked node
            is_double_click: Whether this was a double-click
        """
        self.node_selected.emit(node)

        # Snap camera to selected node (single click)
        if hasattr(self._renderer, 'snap_camera_to_node'):
            self._renderer.snap_camera_to_node(id(node))

        if is_double_click:
            if node.is_directory:
                # Navigate to directory (change current directory)
                self._change_directory(node.path)

            elif node.is_file:
                # Open file with default application
                try:
                    self._open_file(node)
                    self.scan_progress.emit(f"Opened: {node.name}")
                except FileOpenError as e:
                    self.scan_progress.emit(f"Error: {e.reason}")
                    # Show error dialog
                    QMessageBox.warning(
                        self._window,
                        "Cannot Open File",
                        f"Failed to open {node.name}:\n{e.reason}"
                    )

    def _open_file(self, node: Node) -> None:
        """Open a file with the default application.

        Args:
            node: File node to open

        Raises:
            FileOpenError: If file cannot be opened
        """
        file_path = node.path

        # Validation
        if not file_path.exists():
            raise FileOpenError(file_path, "File does not exist")

        if not file_path.is_file():
            raise FileOpenError(file_path, "Path is not a file")

        # Open file with platform-specific command
        try:
            if sys.platform == "darwin":  # macOS
                subprocess.run(["open", str(file_path)], check=True)
            elif sys.platform == "win32":  # Windows
                subprocess.run(["start", "", str(file_path)], shell=True, check=True)
            else:  # Linux and other Unix-like systems
                subprocess.run(["xdg-open", str(file_path)], check=True)
        except subprocess.CalledProcessError as e:
            raise FileOpenError(file_path, f"Failed to open file: {e}")
        except FileNotFoundError:
            raise FileOpenError(file_path, "Default application launcher not found on this system")
        except Exception as e:
            raise FileOpenError(file_path, f"Unexpected error: {e}")

    def _on_node_focused(self, node: Node) -> None:
        """Handle node focus from input handler.

        Args:
            node: Focused node
        """
        self._focused_node = node

    def _on_camera_mode_changed(self, mode) -> None:
        """Handle camera mode change from input handler.

        Args:
            mode: New camera mode
        """
        self._window._control_panel.set_camera_mode_display(mode)

    def _on_tree_node_selected(self, node: Node) -> None:
        """Handle node selection from file tree.

        Args:
            node: Selected node
        """
        self._focused_node = node
        self._selected_nodes.clear()
        self._selected_nodes.add(node)

        # Update visual selection
        self._update_selection_visual()

        # Navigate to node in 3D view
        if hasattr(self._renderer, 'snap_camera_to_node'):
            self._renderer.snap_camera_to_node(id(node))

    def _on_tree_node_double_clicked(self, node: Node) -> None:
        """Handle node double-click from file tree.

        Args:
            node: Double-clicked node
        """
        if node.is_file:
            # Open file with default application
            try:
                self._open_file(node)
                self.scan_progress.emit(f"Opened: {node.name}")
            except FileOpenError as e:
                self.scan_progress.emit(f"Error: {e.reason}")
                # Show error dialog
                QMessageBox.warning(
                    self._window,
                    "Cannot Open File",
                    f"Failed to open {node.name}:\n{e.reason}"
                )

    def _on_selection_changed(self, selected_nodes: set[Node]) -> None:
        """Handle selection change from input handler.

        Args:
            selected_nodes: Set of selected nodes
        """
        self._selected_nodes = selected_nodes
        self._window.update_stats(len(self._nodes), len(selected_nodes))

    def _update_selection_visual(self) -> None:
        """Update visual selection state."""
        selected_paths = {str(n.path) for n in self._selected_nodes}

        if hasattr(self._renderer, 'set_selection'):
            from pyfsn.model.node import NodeType
            self._renderer.set_selection(selected_paths, self._nodes)

        # Update mini map selection
        self._update_mini_map_selection()

    def _update_mini_map(self) -> None:
        """Update the mini map with current scene data."""
        mini_map = self._window.mini_map
        if mini_map is None:
            return

        # Get selected paths
        selected_paths = {str(n.path) for n in self._selected_nodes}

        # Update mini map
        mini_map.set_scene_data(
            positions=self._positions,
            nodes=self._nodes,
            camera=self._camera,
            root_path=str(self._root_node.path) if self._root_node else None,
            current_path=str(self._root_path),
            selected_paths=selected_paths,
        )

    def _update_mini_map_selection(self) -> None:
        """Update only the selection state on the mini map."""
        mini_map = self._window.mini_map
        if mini_map is None:
            return

        selected_paths = {str(n.path) for n in self._selected_nodes}
        mini_map.set_selection(selected_paths)

    def _get_ordered_nodes(self) -> list[Node]:
        """Get list of nodes in depth-first order (visual tree order).
        
        Returns:
            List of nodes
        """
        if not self._root_node:
            return []
            
        nodes = [self._root_node]
        nodes.extend(self._root_node.get_all_descendants())
        return nodes

    def _select_next_node(self) -> None:
        """Select the next node in the tree."""
        if not self._root_node:
            return

        all_nodes = self._get_ordered_nodes()
        if not all_nodes:
            return

        # Find current selection index
        current_index = -1
        if self._focused_node:
            try:
                current_index = all_nodes.index(self._focused_node)
            except ValueError:
                pass
        
        # Select next
        next_index = (current_index + 1) % len(all_nodes)
        next_node = all_nodes[next_index]
        self._select_node(next_node)

    def _select_previous_node(self) -> None:
        """Select the previous node in the tree."""
        if not self._root_node:
            return

        all_nodes = self._get_ordered_nodes()
        if not all_nodes:
            return

        # Find current selection index
        current_index = -1
        if self._focused_node:
            try:
                current_index = all_nodes.index(self._focused_node)
            except ValueError:
                pass
        
        # Select previous
        prev_index = (current_index - 1) % len(all_nodes)
        prev_node = all_nodes[prev_index]
        self._select_node(prev_node)

    def _select_node(self, node: Node) -> None:
        """Select and focus a specific node.
        
        Args:
            node: Node to select
        """
        self._focused_node = node
        self._selected_nodes.clear()
        self._selected_nodes.add(node)
        
        self.node_selected.emit(node)
        self.node_focused.emit(node)
        
        self._update_selection_visual()
        
        if hasattr(self._renderer, 'snap_camera_to_node'):
            self._renderer.snap_camera_to_node(id(node))
            
        self._window.file_tree.select_node(node)

    # Public API

    @property
    def window(self) -> MainWindow:
        """Get the main window."""
        return self._window

    @property
    def root_node(self) -> Node | None:
        """Get the root node."""
        return self._root_node

    def refresh(self) -> None:
        """Refresh the current view."""
        if self._root_node:
            self._root_node.invalidate_children()
            self._start_scan()

    def navigate_to(self, path: Path) -> None:
        """Navigate to a specific path.

        Args:
            path: Path to navigate to
        """
        path_str = str(path.resolve())
        if path_str in self._nodes:
            node = self._nodes[path_str]
            if hasattr(self._renderer, 'snap_camera_to_node'):
                self._renderer.snap_camera_to_node(id(node))

    def set_camera_mode(self, mode: CameraMode) -> None:
        """Set camera navigation mode.

        Args:
            mode: Camera mode to set
        """
        self._camera.set_mode(mode)

    # Navigation history

    def can_go_back(self) -> bool:
        """Check if we can go back in history.

        Returns:
            True if there are items in the back stack
        """
        return len(self._back_stack) > 0

    def can_go_forward(self) -> bool:
        """Check if we can go forward in history.

        Returns:
            True if there are items in the forward stack
        """
        return len(self._forward_stack) > 0

    def go_back(self) -> None:
        """Navigate to the previous directory in history."""
        if not self.can_go_back():
            return

        # Pop from back stack
        previous_path = self._back_stack.pop()

        # Push current path to forward stack
        self._forward_stack.append(self._root_path)

        # Navigate to previous path
        self._navigate_to_path(previous_path)

        # Update navigation state
        self._emit_navigation_state()

    def go_forward(self) -> None:
        """Navigate to the next directory in history."""
        if not self.can_go_forward():
            return

        # Pop from forward stack
        next_path = self._forward_stack.pop()

        # Push current path to back stack
        self._back_stack.append(self._root_path)

        # Navigate to next path
        self._navigate_to_path(next_path)

        # Update navigation state
        self._emit_navigation_state()

    def _navigate_to_path(self, path: Path) -> None:
        """Navigate to a path without modifying history stacks.

        Args:
            path: Path to navigate to
        """
        self._root_path = path
        self._window.set_root_path(path)

        # Clear current scene
        self._nodes.clear()
        self._positions.clear()
        self._selected_nodes.clear()
        self._focused_node = None

        # Start new scan
        self._start_scan()

    def _emit_navigation_state(self) -> None:
        """Emit navigation state change signal."""
        self.navigation_state_changed.emit(self.can_go_back(), self.can_go_forward())

