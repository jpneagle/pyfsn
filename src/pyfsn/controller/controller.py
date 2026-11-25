"""Main controller for the application.

Coordinates between the Model, View, and Layout layers, handling
user input and managing application state.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QObject, pyqtSignal, QThread, QTimer, QPoint
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QMessageBox, QMenu, QApplication

from pyfsn.model.node import Node, NodeType
from pyfsn.model.scanner import Scanner, ScannerWorker, ScanProgress
from pyfsn.errors import FileOpenError
from pyfsn.layout.engine import (
    DirHeightMode,
    HeightMode,
    LayoutConfig,
    LayoutEngine,
    LayoutResult,
)
from pyfsn.view.renderer import Renderer, ColorMode
from pyfsn.view.camera import CameraMode
from pyfsn.view.main_window import MainWindow
from pyfsn.view.sound import SoundManager
from pyfsn.view.theme import DEFAULT_THEME
from pyfsn.view.theme_manager import get_theme_manager
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

    def __init__(
        self,
        root_path: Path,
        show_hidden: bool = False,
        lazy_depth: int = 2,
        layout_config: LayoutConfig | None = None,
        height_mode: HeightMode | None = None,
        dir_height_mode: DirHeightMode | None = None,
        landscape: str | None = None,
    ) -> None:
        """Initialize controller.

        Args:
            root_path: Root directory path to visualize
            show_hidden: Whether to include hidden files and directories
            lazy_depth: Depth at which the scanner starts lazy loading
            layout_config: Layout configuration (defaults if None)
            height_mode: Height mode override (persisted choice if None)
            dir_height_mode: Directory height mode override (persisted choice if None)
            landscape: Landscape name override (persisted choice if None)
        """
        super().__init__()

        self._root_path = root_path
        self._original_root_path = root_path
        self._root_node: Node | None = None
        self._show_hidden_cli = show_hidden
        self._lazy_depth = lazy_depth

        # Restore view preferences before building the window/scanner so that
        # CLI flags can override persisted settings when explicitly provided.
        from PyQt6.QtCore import QSettings
        self._settings = QSettings("pyfsn", "pyfsn")
        self._show_hidden = self._settings.value("view/show_hidden", False, type=bool)
        if show_hidden:
            # CLI explicitly requested hidden files
            self._show_hidden = True
        self._color_mode = self._settings.value("view/color_mode", "age", type=str)

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

        # Create layout engine. Height modes come from the command line when
        # given, otherwise from the persisted View menu choice.
        self._layout_config = layout_config or LayoutConfig()
        if height_mode is None:
            height_mode = self._enum_setting(
                "view/height_mode", HeightMode, self._layout_config.height_mode
            )
        if dir_height_mode is None:
            dir_height_mode = self._enum_setting(
                "view/dir_height_mode", DirHeightMode, self._layout_config.dir_height_mode
            )
        self._layout_config.height_mode = height_mode
        self._layout_config.dir_height_mode = dir_height_mode
        self._layout_engine = LayoutEngine(self._layout_config)
        self._window.set_height_mode_checked(height_mode.value)
        self._window.set_dir_height_mode_checked(dir_height_mode.value)

        # Sky/ground colours
        if landscape is None:
            landscape = self._settings.value("view/landscape", "grass", type=str)
        if not self._renderer.set_landscape(landscape):
            landscape = "grass"
        self._window.set_landscape_checked(landscape)

        # Directories the user expanded by visiting them; replayed after a
        # rescan so the landscape does not collapse on refresh
        self._expanded_paths: set[str] = set()
        # Keep the camera where it is when the next scan completes (refresh)
        self._keep_camera_on_scan = False

        # Create scanner
        self._scanner = Scanner(lazy_depth=lazy_depth, show_hidden=self._show_hidden)
        self._scan_worker: ScannerWorker | None = None

        # Scene data
        self._nodes: dict[str, Node] = {}
        self._positions: dict[str, object] = {}
        self._layout_result: LayoutResult | None = None

        # (positions dict, names, anchor points) of the file labels; rebuilt
        # whenever the renderer is given a new positions dict
        self._file_label_cache = None

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

        # Selection sync guard to prevent 3D <-> tree feedback loops
        self._updating_selection = False

        # Sound effects (disabled by default)
        self._sound = SoundManager()

        # Set theme on renderer and listen for theme changes
        self._theme_manager = get_theme_manager()
        self._theme_manager.load_preferences()
        self._renderer.set_theme(self._theme_manager.current_theme)
        self._theme_manager.theme_changed.connect(self._renderer.set_theme)
        # Sync the theme menu to the loaded theme
        self._window.set_active_theme(self._current_theme_key())

        # Restore persisted view preferences (colorblind palette, sound)
        if self._settings.value("view/colorblind", False, type=bool):
            self._renderer.set_colorblind_mode(True)
            self._window.set_colorblind_checked(True)
        if self._settings.value("view/sound", False, type=bool):
            self._sound.enabled = True

        # Sync color mode and hidden-files UI state
        self._renderer.set_color_mode(ColorMode(self._color_mode))
        self._window.set_color_mode_checked(self._color_mode)
        self._window.set_color_mode_legend(
            self._color_mode,
            self._renderer.type_color_map if self._color_mode == "type" else None,
        )
        self._window.set_show_hidden_checked(self._show_hidden)

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
        self.node_selected.connect(self._on_node_selected_status)

        # Connect window signals
        self._window.directory_changed.connect(self._change_directory)
        self._window.search_requested.connect(self._perform_search)
        self._window.next_search_result_requested.connect(self.next_search_result)
        self._window.previous_search_result_requested.connect(self.previous_search_result)
        self._window.file_tree.node_selected.connect(self._on_tree_node_selected)
        self._window.tree_node_double_clicked.connect(self._on_tree_node_double_clicked)
        self._window.tree_selection_changed.connect(self._on_tree_selection_changed)

        # Connect navigation history signals
        self._window.go_back_requested.connect(self.go_back)
        self._window.go_forward_requested.connect(self.go_forward)

        # Connect navigation state signal
        self.navigation_state_changed.connect(self._window.update_navigation_state)

        # Connect window signals
        self._window.refresh_requested.connect(self.refresh)

        # Connect filter panel signal (Workstream F - Advanced filtering)
        self._window.filter_changed.connect(self._apply_filters)

        # Sound toggle
        self._window.sound_toggled.connect(self._on_sound_toggled)

        # Theme selection
        self._window.theme_selected.connect(self._on_theme_selected)

        # Colorblind palette toggle
        self._window.colorblind_toggled.connect(self._on_colorblind_toggled)

        # Show hidden files toggle
        self._window.show_hidden_toggled.connect(self._on_show_hidden_toggled)

        # Color mode selection
        self._window.color_mode_selected.connect(self._on_color_mode_selected)

        # Height modes and view presets (fsn Display menu / control panel)
        self._window.height_mode_selected.connect(self._on_height_mode_selected)
        self._window.dir_height_mode_selected.connect(self._on_dir_height_mode_selected)
        self._window.view_preset_requested.connect(self._on_view_preset_requested)
        self._window.landscape_selected.connect(self._on_landscape_selected)

        # Bookmarks
        self._window.bookmark_add_requested.connect(
            lambda: self._window.add_bookmark(self._root_path)
        )
        self._window.bookmark_selected.connect(self._on_bookmark_selected)

        # Recent directories
        self._window.recent_dir_selected.connect(self._on_recent_dir_selected)

        # File tree context menu
        self._window.tree_context_menu_requested.connect(self._show_context_menu)

        # Mini map click-to-navigate
        if self._window.mini_map is not None:
            self._window.mini_map.map_clicked.connect(self._on_mini_map_clicked)

    def _connect_input_handler(self) -> None:
        """Connect input handler callbacks."""
        self._input_handler.set_node_clicked_callback(self._on_node_clicked)
        self._input_handler.set_node_focused_callback(self._on_node_focused)
        self._input_handler.set_selection_changed_callback(self._on_selection_changed)
        self._input_handler.set_navigate_next_callback(self._select_next_node)
        self._input_handler.set_navigate_previous_callback(self._select_previous_node)
        self._input_handler.set_camera_mode_changed_callback(self._on_camera_mode_changed)
        self._input_handler.set_context_menu_callback(self._show_context_menu)

    def start(self) -> None:
        """Start the application - begin scanning."""
        self._window.add_recent_directory(self._root_path)
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

        self._navigate_to_path(path)
        self._window.add_recent_directory(path)

        # Update navigation state
        self._emit_navigation_state()

    # Scanning

    def _start_scan(self) -> None:
        """Start scanning the root directory."""
        self._window.set_loading(True)
        self.scan_progress.emit(f"Scanning {self._root_path}...")

        # Create root node
        self._root_node = Node.from_path(self._root_path)
        self._nodes[str(self._root_node.path)] = self._root_node

        # Start async scan through the scanner so its options
        # (lazy_depth, show_hidden) are applied
        self._scan_worker = self._scanner.scan_async(
            self._root_path,
            on_progress=self._on_scan_progress,
            on_finished=self._on_scan_complete,
            on_error=self._on_scan_error,
        )

    def _on_scan_progress(self, progress: ScanProgress) -> None:
        """Handle scan progress update.

        Args:
            progress: ScanProgress object with current status
        """
        # The worker's final "complete" tick carries no information and would
        # overwrite the "Loaded N items" message
        if progress.is_complete:
            return

        # Create concise message from progress info
        current_name = progress.current_path.name or str(progress.current_path)
        message = f"Scanning {current_name}... ({progress.nodes_found} nodes)"
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

        # Re-open the directories the user had expanded before this scan
        self._replay_expanded_paths()

        # Calculate layout
        self._calculate_layout()

        # Position camera to view the scene before loading it, so the
        # renderer picks up the final camera state
        if self._keep_camera_on_scan:
            self._keep_camera_on_scan = False
        else:
            self._reset_camera_to_scene()

        # Load into renderer
        self._load_scene()

        # Update input handler with scene data
        self._input_handler.set_scene_data(self._nodes, self._positions)

        # Load file tree
        self._window.file_tree.load_tree(root)

        # Re-apply active filters to the freshly scanned nodes so the
        # filter panel state stays in sync after navigation/refresh
        if self._active_filters:
            self._apply_filters(self._active_filters)

        # Emit completion signal
        self._window.set_loading(False)
        self.scan_complete.emit()
        self.scan_progress.emit(f"Loaded {len(self._nodes)} items from {self._root_path}")

    def _on_scan_error(self, error_message: str) -> None:
        """Handle scan error.

        Args:
            error_message: Error message
        """
        self._window.set_loading(False)
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
        """Move the camera to the front view: the root and its first row.

        Like fsn's "reset"/"front view": a low viewpoint in front of the root
        looking into the landscape, rather than a fit of the whole tree
        (which would make everything tiny on a wide file system).
        """
        import math
        import numpy as np

        if self._layout_result is None or self._root_node is None:
            return

        root_pos = self._positions.get(str(self._root_node.path))
        if root_pos is None:
            return

        # Width worth framing: the root and its row of children, capped so a
        # directory with hundreds of children does not push the camera away
        slot = self._layout_result.slot_widths.get(str(self._root_node.path), root_pos.width)
        frame_width = max(root_pos.width + 6.0, min(slot, 70.0))

        root_cx, _, root_cz = root_pos.center
        target = np.array(
            [root_cx, 0.0, root_cz - self._layout_config.level_gap],
            dtype=np.float32,
        )

        fov_rad = math.radians(self._camera.state.fov)
        distance = max((frame_width / 2.0) / math.tan(fov_rad / 2.0) * 0.8, 14.0)

        # Low enough above the ground that the horizon and sky are in view
        elevation = math.radians(17.0)
        camera_pos = target + np.array(
            [0.0, distance * math.sin(elevation), distance * math.cos(elevation)],
            dtype=np.float32,
        )
        self._camera.set_position_target(camera_pos, target)

    def _birds_eye_view(self) -> None:
        """Look at the whole landscape from high above (fsn "birds eye")."""
        import math
        import numpy as np

        if self._layout_result is None or self._layout_result.bounds is None:
            return

        bounds = self._layout_result.bounds.position
        center = np.array(bounds.center, dtype=np.float32)
        center[1] = 0.0
        half_extent = max(bounds.width, bounds.depth) / 2.0
        fov_rad = math.radians(self._camera.state.fov)
        distance = max(half_extent / math.tan(fov_rad / 2.0) * 1.15, 20.0)

        # Nearly straight down, keeping "up the screen" = deeper into the tree
        elevation = math.radians(80.0)
        camera_pos = center + np.array(
            [0.0, distance * math.sin(elevation), distance * math.cos(elevation)],
            dtype=np.float32,
        )
        self._camera.max_orbit_distance = max(self._camera.max_orbit_distance, distance * 1.2)
        self._camera.set_position_target(camera_pos, center)

    def _on_view_preset_requested(self, preset: str) -> None:
        """Handle View > Reset View / Bird's Eye View."""
        if hasattr(self._renderer, "_is_animating"):
            self._renderer._is_animating = False
        if self._camera.mode != CameraMode.ORBIT:
            self._camera.set_mode(CameraMode.ORBIT)
            self._on_camera_mode_changed(CameraMode.ORBIT)
        if preset == "birds_eye":
            self._birds_eye_view()
        else:
            self._reset_camera_to_scene()

    # Landscape expansion (single persistent landscape, like fsn)

    def _enum_setting(self, key: str, enum_cls, default):
        """Read an enum-valued preference, falling back on unknown values."""
        raw = self._settings.value(key, default.value, type=str)
        try:
            return enum_cls(raw)
        except ValueError:
            return default

    def _load_directory_children(self, node: Node) -> bool:
        """Scan a not-yet-loaded directory and register its children.

        Returns:
            True if the directory was loaded by this call
        """
        if not node.is_directory or node.is_loaded:
            return False

        try:
            self._scanner.load_children(node)
        except OSError as e:
            # Leave it marked as loaded-but-empty so we do not retry on every click
            node.is_loaded = True
            self.scan_progress.emit(f"Cannot read {node.name}: {e}")

        for descendant in node.get_all_descendants():
            self._nodes[str(descendant.path)] = descendant

        # The layout only draws max_depth levels below the root; make sure an
        # expansion the user asked for is never silently cut off
        needed_depth = node.depth + 1
        if needed_depth > self._layout_config.max_depth:
            self._layout_config.max_depth = needed_depth
        return True

    def _replay_expanded_paths(self) -> None:
        """Re-load directories that were expanded before a rescan."""
        if not self._expanded_paths:
            return
        # Parents first: a child can only be found once its parent is loaded
        for path_str in sorted(self._expanded_paths, key=len):
            node = self._nodes.get(path_str)
            if node is not None:
                self._load_directory_children(node)

    def _expand_directory(self, node: Node) -> bool:
        """Load a directory on demand and grow the landscape around it.

        The camera is carried along with the directory, so the pedestal the
        user clicked stays where it was on screen while its neighbours move
        aside to make room.

        Returns:
            True if the landscape changed
        """
        if not self._load_directory_children(node):
            return False

        self._expanded_paths.add(str(node.path))
        self._rebuild_scene(anchor=node)
        self._window.file_tree.add_children(node)
        self.scan_progress.emit(
            f"Loaded {node.name}: {len(node.children)} items ({len(self._nodes)} total)"
        )
        return True

    def _rebuild_scene(self, anchor: Node | None = None) -> None:
        """Recompute the layout and reload the scene without rescanning.

        Args:
            anchor: Node the camera should stay attached to while everything
                else is rearranged (None keeps the camera where it is)
        """
        if self._root_node is None:
            return

        import numpy as np

        anchor_path = str(anchor.path) if anchor is not None else None
        old_pos = self._positions.get(anchor_path) if anchor_path else None
        old_center = np.array(old_pos.center, dtype=np.float32) if old_pos else None

        self._calculate_layout()

        if old_center is not None:
            new_pos = self._positions.get(anchor_path)
            if new_pos is not None:
                delta = np.array(new_pos.center, dtype=np.float32) - old_center
                # Heights can change with the scene-wide scale; follow on the ground only
                delta[1] = 0.0
                self._camera.translate(delta)

        if self._active_filters:
            # Re-filters the node set and loads the filtered scene
            self._apply_filters(self._active_filters)
            self._update_mini_map()
        else:
            self._load_scene()
            self._input_handler.set_scene_data(self._nodes, self._positions)

    def _zoom_to_directory(self, node: Node, close_up: bool = False) -> None:
        """Fly the camera to a directory."""
        if not hasattr(self._renderer, 'snap_camera_to_node'):
            return
        if close_up:
            position = self._positions.get(str(node.path))
            size = max(position.width, position.depth) if position else 2.0
            self._renderer.snap_camera_to_node(id(node), distance=size * 1.3 + 4.0)
        else:
            self._renderer.snap_camera_to_node(id(node))

    def _prune_directory(self, node: Node) -> None:
        """Hide the contents of a directory (fsn "Prune")."""
        self._layout_engine.pruned_paths.add(str(node.path))
        self._rebuild_scene(anchor=node)
        self.scan_progress.emit(f"Pruned {node.name}")

    def _unprune_directory(self, node: Node) -> None:
        """Show the contents of a pruned directory again."""
        self._layout_engine.pruned_paths.discard(str(node.path))
        self._rebuild_scene(anchor=node)
        self.scan_progress.emit(f"Restored {node.name}")

    def _unprune_all(self) -> None:
        """Show the contents of all pruned directories again."""
        self._layout_engine.pruned_paths.clear()
        self._rebuild_scene()
        self.scan_progress.emit("Restored all pruned directories")

    def _on_landscape_selected(self, key: str) -> None:
        """Switch the sky/ground colours and persist the choice."""
        if not self._renderer.set_landscape(key):
            return
        self._settings.setValue("view/landscape", key)
        self._window.set_landscape_checked(key)
        self.scan_progress.emit(f"Landscape: {key}")

    def _on_height_mode_selected(self, mode: str) -> None:
        """Switch how sizes map to heights and persist the choice."""
        try:
            self._layout_config.height_mode = HeightMode(mode)
        except ValueError:
            return
        self._settings.setValue("view/height_mode", mode)
        self._window.set_height_mode_checked(mode)
        self._rebuild_scene()
        self.scan_progress.emit(f"Height: {mode}")

    def _on_dir_height_mode_selected(self, mode: str) -> None:
        """Switch what pedestal height stands for and persist the choice."""
        try:
            self._layout_config.dir_height_mode = DirHeightMode(mode)
        except ValueError:
            return
        self._settings.setValue("view/dir_height_mode", mode)
        self._window.set_dir_height_mode_checked(mode)
        self._rebuild_scene()
        self.scan_progress.emit(f"Directory height: {mode}")

    # Text overlay (labels)

    # File names are only drawn near the camera: far away they are unreadable
    # and pile up on top of each other
    FILE_LABEL_DISTANCE = 45.0
    FILE_LABEL_LIMIT = 150

    def _file_label_arrays(self):
        """Names and anchor points (top centre) of all file boxes in the scene."""
        import numpy as np

        positions = self._renderer._positions
        if self._file_label_cache is not None and self._file_label_cache[0] is positions:
            return self._file_label_cache[1], self._file_label_cache[2]

        nodes = self._renderer._nodes
        names: list[str] = []
        points: list[tuple[float, float, float]] = []
        for path_str, position in positions.items():
            node = nodes.get(path_str)
            if node is None or node.is_directory:
                continue
            names.append(node.name)
            points.append((
                position.x + position.width / 2,
                position.y + position.height,
                position.z + position.depth / 2,
            ))
        array = np.array(points, dtype=np.float32).reshape(-1, 3)
        self._file_label_cache = (positions, names, array)
        return names, array

    def _update_text_overlay(self) -> None:
        """Update the text overlay with the names of the files near the camera."""
        import math
        import numpy as np

        if not self._window.show_labels or self._layout_result is None:
            return

        overlay = self._window.text_overlay
        if overlay is None:
            return

        names, points = self._file_label_arrays()
        labels: list[tuple[str, int, int]] = []

        if len(names):
            state = self._camera.state
            forward = state.target - state.position
            norm = float(np.linalg.norm(forward))
            width, height = self._renderer.width(), self._renderer.height()
            if norm > 1e-6 and width > 0 and height > 0:
                forward = forward / norm
                right = np.cross(forward, state.up)
                right_norm = float(np.linalg.norm(right))
                if right_norm > 1e-6:
                    right = right / right_norm
                    up = np.cross(right, forward)

                    # Project every anchor at once (same maths as
                    # Renderer.get_screen_position)
                    to_points = points - state.position
                    view_z = to_points @ forward
                    distance = np.linalg.norm(to_points, axis=1)
                    candidates = np.nonzero(
                        (view_z > 0.1) & (distance < self.FILE_LABEL_DISTANCE)
                    )[0]
                    if candidates.size:
                        scale = 1.0 / math.tan(math.radians(state.fov) / 2.0)
                        aspect = width / height
                        z = view_z[candidates]
                        ndc_x = (to_points[candidates] @ right) / z * (scale / aspect)
                        ndc_y = (to_points[candidates] @ up) / z * scale
                        on_screen = (np.abs(ndc_x) <= 1.0) & (np.abs(ndc_y) <= 1.0)
                        candidates = candidates[on_screen]
                        ndc_x, ndc_y = ndc_x[on_screen], ndc_y[on_screen]
                        # Nearest first, so the cap drops the farthest names
                        order = np.argsort(distance[candidates])[: self.FILE_LABEL_LIMIT]
                        for i in order:
                            labels.append((
                                names[int(candidates[i])],
                                int((ndc_x[i] + 1.0) * 0.5 * width),
                                int((1.0 - ndc_y[i]) * 0.5 * height),
                            ))

        # The focused node keeps its highlighted name at any distance
        focused_pos = None
        if self._focused_node is not None:
            position = self._renderer._positions.get(str(self._focused_node.path))
            if position is not None:
                focused_pos = self._renderer.get_screen_position(
                    np.array(position.center, dtype=np.float32)
                )
        if focused_pos is not None:
            name = self._focused_node.name
            labels = [label for label in labels if label[0] != name]
            overlay.set_focused_label(name, focused_pos[0], focused_pos[1])
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
            self._window.set_search_result_count(0, 0)
            # Clear spotlight visualization
            if hasattr(self._renderer, 'clear_spotlight_search'):
                self._renderer.clear_spotlight_search()
            return

        query_lower = query.lower()
        matches = [
            node for node in self._nodes.values()
            if query_lower in node.name.lower()
        ]

        def _match_rank(node: Node) -> int:
            name = node.name.lower()
            if name == query_lower:
                return 0  # 完全一致
            if name.startswith(query_lower):
                return 1  # 前方一致
            return 2  # 部分一致

        self._search_results = sorted(matches, key=_match_rank)

        if self._search_results:
            self._current_search_index = 0

            # Phase 3.2: Start spotlight search visualization
            matching_nodes = set(self._search_results)
            if hasattr(self._renderer, 'start_spotlight_search'):
                self._renderer.start_spotlight_search(query, matching_nodes)

            self._show_search_result()
            self.scan_progress.emit(f"Found {len(self._search_results)} results for '{query}'")
        else:
            self._window.set_search_result_count(0, 0)
            self.scan_progress.emit(f"No results found for '{query}'")
            # Clear spotlight when no results
            if hasattr(self._renderer, 'clear_spotlight_search'):
                self._renderer.clear_spotlight_search()

    def _show_search_result(self) -> None:
        """Show the current search result."""
        if not self._search_results:
            self._window.set_search_result_count(0, 0)
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

        self._window.set_search_result_count(
            self._current_search_index + 1, len(self._search_results)
        )
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
            self._window.set_filter_status(None)
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

        # Update status message and persistent filter indicator
        filter_desc = self._get_filter_description(filters)
        ancestor_note = " + ancestors" if filters.get('include_ancestors', True) and len(self._filtered_nodes) > filtered_count else ""
        status_text = f"Showing {filtered_count}/{len(self._nodes)} items{filter_desc}{ancestor_note}"
        self._window.set_filter_status(status_text)
        self.scan_progress.emit(status_text)

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
        self._sound.play_click()

        if node.is_directory:
            # fsn: selecting a directory zooms to it. A directory that has not
            # been scanned yet is loaded first and the landscape grows in
            # place; the scene reload must come before the camera animation,
            # because loading a layout cancels a running animation.
            self._expand_directory(node)
            if is_double_click:
                # Close-up of the directory (fsn "warps" here; pyfsn stays in
                # the landscape). Re-rooting is available from the context menu.
                self._sound.play_navigate()
            self._zoom_to_directory(node, close_up=is_double_click)
        elif not is_double_click and hasattr(self._renderer, 'snap_camera_to_node'):
            self._renderer.snap_camera_to_node(id(node))

        if is_double_click:
            if node.is_file:
                # Open file with default application
                try:
                    self._open_file(node)
                    self._window.show_toast(f"Opened: {node.name}")
                    self.scan_progress.emit(f"Opened: {node.name}")
                except FileOpenError as e:
                    self._show_file_open_error(node, e)

    def _show_file_open_error(self, node: Node, e: FileOpenError) -> None:
        """Display file open error in status bar and dialog."""
        self.scan_progress.emit(f"Error: {e.reason}")
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
                # os.startfile avoids the shell, so special characters in
                # file names are never interpreted as commands
                os.startfile(str(file_path))
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

    def _on_node_selected_status(self, node: Node) -> None:
        """Update status bar with selected node path."""
        self._window.set_selected_node_path(str(node.path))

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

        # Load the directory on demand, like a click in the 3D view
        if node.is_directory:
            self._expand_directory(node)

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
                self._show_file_open_error(node, e)

    def _on_tree_selection_changed(self, selected_nodes: set[Node]) -> None:
        """Handle selection change from file tree.

        Args:
            selected_nodes: Set of selected nodes
        """
        if self._updating_selection:
            return

        self._updating_selection = True
        self._selected_nodes = selected_nodes.copy()
        self._window.update_stats(len(self._nodes), len(selected_nodes))

        if selected_nodes:
            node = next(iter(selected_nodes))
            self._focused_node = node
            self._window.set_selected_node_path(str(node.path))
            # Update renderer selection (sync 3D view)
            selected_paths = {str(n.path) for n in selected_nodes}
            if hasattr(self._renderer, 'set_selection'):
                self._renderer.set_selection(selected_paths, self._nodes)
        else:
            self._focused_node = None
            self._window.set_selected_node_path(None)
            if hasattr(self._renderer, 'clear_selection'):
                self._renderer.clear_selection()

        self._updating_selection = False

    def _on_selection_changed(self, selected_nodes: set[Node]) -> None:
        """Handle selection change from input handler.

        Args:
            selected_nodes: Set of selected nodes
        """
        self._selected_nodes = selected_nodes
        self._window.update_stats(len(self._nodes), len(selected_nodes))
        if selected_nodes:
            # Display path of the most recently focused/selected node if available
            node = self._focused_node or next(iter(selected_nodes))
            self._window.set_selected_node_path(str(node.path))
        else:
            self._window.set_selected_node_path(None)

        # Sync tree selection without re-entering this handler
        if not self._updating_selection:
            self._updating_selection = True
            self._window.set_tree_selection(selected_nodes)
            self._updating_selection = False

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

    # Context menu

    def _show_context_menu(self, node, screen_pos) -> None:
        """Show right-click context menu for a node or background.

        Args:
            node: Node at click position, or None for background
            screen_pos: Screen position for menu
        """
        menu = QMenu(self._window)

        if node is None:
            # Background context menu
            if self.can_go_back():
                menu.addAction("Go Up", self.go_back)
            menu.addAction("Refresh", self.refresh)
            mode_menu = menu.addMenu("Camera Mode")
            mode_menu.addAction("Orbit", lambda: self.set_camera_mode(CameraMode.ORBIT))
            mode_menu.addAction("Fly", lambda: self.set_camera_mode(CameraMode.FLY))
            if self._layout_engine.pruned_paths:
                menu.addAction("Restore All Pruned", self._unprune_all)
        elif node.is_directory:
            menu.addAction("Set as Root", lambda: self._change_directory(node.path))
            if str(node.path) in self._layout_engine.pruned_paths:
                menu.addAction("Restore (Unprune)", lambda: self._unprune_directory(node))
            elif node is not self._root_node:
                menu.addAction("Prune", lambda: self._prune_directory(node))
            menu.addSeparator()
            menu.addAction("Copy Path", lambda: self._copy_path(node))
            menu.addAction("Reveal in Finder", lambda: self._reveal_in_finder(node))
            menu.addAction("Open Terminal Here", lambda: self._open_in_terminal(node))
            menu.addAction("Bookmark", lambda: self._window.add_bookmark(node.path))
            menu.addSeparator()
            menu.addAction("Rename...", lambda: self._rename_node(node))
            menu.addAction("Move to Trash", lambda: self._delete_node(node))
        else:
            menu.addAction("Open", lambda: self._open_file_safe(node))
            menu.addSeparator()
            menu.addAction("Copy Path", lambda: self._copy_path(node))
            menu.addAction("Reveal in Finder", lambda: self._reveal_in_finder(node))
            menu.addSeparator()
            menu.addAction("Rename...", lambda: self._rename_node(node))
            menu.addAction("Move to Trash", lambda: self._delete_node(node))

        menu.exec(screen_pos)

    def _copy_path(self, node: Node) -> None:
        """Copy node path to clipboard."""
        QApplication.clipboard().setText(str(node.path))
        self.scan_progress.emit(f"Copied: {node.path}")

    def _reveal_in_finder(self, node: Node) -> None:
        """Reveal node in system file manager."""
        path = str(node.path)
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", "-R", path], check=True)
            elif sys.platform == "win32":
                subprocess.run(["explorer", "/select,", path], check=True)
            else:
                # Linux: open parent directory
                parent = str(node.path.parent)
                subprocess.run(["xdg-open", parent], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.scan_progress.emit(f"Could not reveal: {path}")

    def _open_in_terminal(self, node: Node) -> None:
        """Open a terminal at the given directory."""
        path = str(node.path) if node.is_directory else str(node.path.parent)
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", "-a", "Terminal", path], check=True)
            elif sys.platform == "win32":
                # Set the working directory via cwd instead of a "cd" command
                # string, which breaks on spaces and shell metacharacters
                subprocess.Popen(
                    ["cmd"],
                    cwd=path,
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                )
            else:
                subprocess.Popen(["x-terminal-emulator", "--working-directory", path])
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.scan_progress.emit(f"Could not open terminal at: {path}")

    def _open_file_safe(self, node: Node) -> None:
        """Open file with error handling for context menu."""
        try:
            self._open_file(node)
            self.scan_progress.emit(f"Opened: {node.name}")
        except FileOpenError as e:
            self._show_file_open_error(node, e)

    # File operations (rename / delete)

    def _rename_node(self, node: Node) -> None:
        """Rename a file or directory after prompting for a new name."""
        from PyQt6.QtWidgets import QInputDialog

        old_path = Path(node.path)
        new_name, ok = QInputDialog.getText(
            self._window, "Rename", "New name:", text=old_path.name
        )
        if not ok or not new_name or new_name == old_path.name:
            return

        new_path = old_path.with_name(new_name)
        if new_path.exists():
            QMessageBox.warning(
                self._window, "Rename Failed", f"'{new_name}' already exists."
            )
            return
        try:
            old_path.rename(new_path)
        except OSError as e:
            QMessageBox.critical(self._window, "Rename Failed", str(e))
            return

        self._window.show_toast(f"Renamed to: {new_name}")
        self.scan_progress.emit(f"Renamed to: {new_name}")
        self.refresh()

    def _delete_node(self, node: Node) -> None:
        """Move a file or directory to the trash after confirmation."""
        path = Path(node.path)
        reply = QMessageBox.question(
            self._window,
            "Move to Trash",
            f"Move '{path.name}' to the trash?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        if not self._move_to_trash(path):
            return

        self._window.show_toast(f"Moved to trash: {path.name}")
        self.scan_progress.emit(f"Moved to trash: {path.name}")
        self.refresh()

    def _move_to_trash(self, path: Path) -> bool:
        """Move a path to the OS trash. Returns True on success."""
        # Prefer send2trash if available (cross-platform, recoverable)
        try:
            from send2trash import send2trash

            send2trash(str(path))
            return True
        except ImportError:
            pass
        except OSError as e:
            QMessageBox.critical(self._window, "Delete Failed", str(e))
            return False

        # macOS native fallback via Finder
        try:
            if sys.platform == "darwin":
                # Escape backslashes and quotes so the path cannot break
                # out of the AppleScript string literal
                escaped = str(path).replace("\\", "\\\\").replace('"', '\\"')
                script = (
                    'tell application "Finder" to move POSIX file '
                    f'"{escaped}" to trash'
                )
                subprocess.run(["osascript", "-e", script], check=True)
                return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        QMessageBox.warning(
            self._window,
            "Trash Unavailable",
            "Could not move to trash. Install 'send2trash' "
            "(pip install Send2Trash) to enable safe deletion.",
        )
        return False

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
            # Same landscape, same viewpoint: only the contents are re-read
            self._keep_camera_on_scan = True
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

        # Clear current scene state. The filtered node set and search
        # results reference nodes from the old scene, so they must be
        # dropped as well (active filter criteria are kept and re-applied
        # once the new scan completes).
        self._nodes.clear()
        self._positions.clear()
        self._selected_nodes.clear()
        self._focused_node = None
        self._filtered_nodes.clear()
        self._search_results.clear()
        self._current_search_index = 0

        # Clear renderer selection and search spotlights
        if self._renderer:
            self._renderer.clear_selection()
            if hasattr(self._renderer, 'clear_spotlight_search'):
                self._renderer.clear_spotlight_search()

        # Start new scan
        self._start_scan()

    def _emit_navigation_state(self) -> None:
        """Emit navigation state change signal."""
        self.navigation_state_changed.emit(self.can_go_back(), self.can_go_forward())

    # Theme / colorblind / bookmarks / mini-map handlers

    def _current_theme_key(self) -> str:
        """Return the registry key for the currently active theme."""
        from pyfsn.view.theme import BUILTIN_THEMES
        current = self._theme_manager.current_theme
        for key, theme in BUILTIN_THEMES.items():
            if theme is current:
                return key
        return current.name.lower().replace(" ", "_")

    def _on_sound_toggled(self, enabled: bool) -> None:
        """Toggle UI sound effects and persist preference."""
        self._sound.enabled = enabled
        self._settings.setValue("view/sound", enabled)

    def _on_theme_selected(self, key: str) -> None:
        """Apply a theme chosen from the View > Theme menu and persist it."""
        try:
            self._theme_manager.set_theme(key)
        except KeyError:
            return
        self._theme_manager.save_preferences()
        self._window.set_active_theme(key)
        self.scan_progress.emit(f"Theme: {self._theme_manager.theme_name}")

    def _on_colorblind_toggled(self, enabled: bool) -> None:
        """Toggle the colorblind-friendly age palette."""
        self._renderer.set_colorblind_mode(enabled)
        self._window.set_colorblind_checked(enabled)
        self._settings.setValue("view/colorblind", enabled)

    def _on_show_hidden_toggled(self, enabled: bool) -> None:
        """Toggle hidden file visibility and refresh the view."""
        self._show_hidden = enabled
        self._scanner.show_hidden = enabled
        self._settings.setValue("view/show_hidden", enabled)
        self.scan_progress.emit("Hidden files: " + ("shown" if enabled else "hidden"))
        self.refresh()

    def _on_color_mode_selected(self, mode: str) -> None:
        """Switch file cube color mode and persist preference."""
        try:
            new_mode = ColorMode(mode.lower())
        except ValueError:
            return
        self._color_mode = mode
        self._renderer.set_color_mode(new_mode)
        self._window.set_color_mode_checked(mode)
        self._window.set_color_mode_legend(
            mode,
            self._renderer.type_color_map if mode == "type" else None,
        )
        self._settings.setValue("view/color_mode", mode)
        self.scan_progress.emit(f"Color mode: {mode.capitalize()}")

    def _on_bookmark_selected(self, path: Path) -> None:
        """Navigate to a bookmarked directory."""
        if path.exists():
            self._change_directory(path)
        else:
            self.scan_progress.emit(f"Bookmark not found: {path}")

    def _on_recent_dir_selected(self, path: Path) -> None:
        """Navigate to a recently used directory."""
        if path.exists():
            self._change_directory(path)
        else:
            self.scan_progress.emit(f"Recent directory not found: {path}")

    def _on_mini_map_clicked(self, world_x: float, world_z: float) -> None:
        """Move the camera to look at the clicked mini-map location."""
        import numpy as np

        # Find the nearest node to the clicked world position (XZ plane)
        nearest_node = None
        nearest_dist = float("inf")
        for path, position in self._positions.items():
            cx, _cy, cz = position.center
            dist = (cx - world_x) ** 2 + (cz - world_z) ** 2
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_node = self._nodes.get(path)

        if nearest_node is not None and hasattr(self._renderer, "snap_camera_to_node"):
            self._renderer.snap_camera_to_node(id(nearest_node))
        else:
            # Fall back to panning the orbit target to the clicked point
            target = np.array([world_x, 0.0, world_z], dtype=np.float32)
            self._camera._state.target = target
            self._camera._update_orbit_from_position()

