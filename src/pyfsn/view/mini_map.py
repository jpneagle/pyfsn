"""2D Mini Map (Radar View) for file system visualization.

Provides a top-down view of the 3D scene showing node positions
and camera frustum.
"""

import math
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPolygonF
from PyQt6.QtWidgets import QWidget

if TYPE_CHECKING:
    from pyfsn.view.camera import Camera


class MiniMap(QWidget):
    """2D radar-style mini map showing top-down view of the file system.

    Displays:
    - Node positions as dots (top-down XZ plane view)
    - Current directory highlighted
    - Camera view frustum outline
    - Zoom level indicator
    """

    # Visual style constants
    DEFAULT_SIZE = 200  # Width and height in pixels
    MARGIN = 10  # Internal margin for drawing

    # Colors (cyberpunk/SGI fsn inspired)
    BG_COLOR = QColor(10, 15, 25, 220)  # Dark semi-transparent background
    BORDER_COLOR = QColor(60, 100, 120)  # Cyan-ish border
    GRID_COLOR = QColor(40, 60, 70, 100)  # Faint grid lines

    # Node colors
    ROOT_COLOR = QColor(255, 200, 50)  # Yellow for root
    DIR_COLOR = QColor(100, 180, 255)  # Light blue for directories
    FILE_COLOR = QColor(150, 150, 150)  # Gray for files
    SELECTED_COLOR = QColor(255, 255, 100)  # Bright yellow for selection
    CURRENT_DIR_COLOR = QColor(50, 255, 100)  # Green for current directory

    # Camera frustum color
    FRUSTUM_COLOR = QColor(255, 100, 50, 180)  # Orange semi-transparent

    def __init__(self, parent=None) -> None:
        """Initialize the mini map.

        Args:
            parent: Parent widget (typically the Renderer)
        """
        super().__init__(parent)

        # Make widget transparent for mouse events and background
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Scene data (will be updated by controller/renderer)
        self._positions: dict[str, object] = {}  # path -> Position object
        self._nodes: dict[str, object] = {}  # path -> Node object
        self._camera: object | None = None  # Camera reference
        self._root_path: str | None = None
        self._current_path: str | None = None  # Current directory path
        self._selected_paths: set[str] = set()

        # View bounds (auto-calculated from scene)
        self._scene_bounds = {
            'min_x': -50.0, 'max_x': 50.0,
            'min_z': -100.0, 'max_z': 50.0
        }

        # Fixed size for the mini map
        self.setFixedSize(self.DEFAULT_SIZE, self.DEFAULT_SIZE)

        # Update timer (throttle updates for performance)
        self._update_pending = False
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._do_update)
        self._update_timer.setSingleShot(True)

    def set_scene_data(
        self,
        positions: dict[str, object],
        nodes: dict[str, object],
        camera: object,
        root_path: str | None = None,
        current_path: str | None = None,
        selected_paths: set[str] | None = None,
    ) -> None:
        """Update the scene data for the mini map.

        Args:
            positions: Dictionary mapping paths to Position objects
            nodes: Dictionary mapping paths to Node objects
            camera: Camera object
            root_path: Path string of the root directory
            current_path: Path string of current directory
            selected_paths: Set of selected path strings
        """
        self._positions = positions
        self._nodes = nodes
        self._camera = camera
        self._root_path = root_path
        self._current_path = current_path
        self._selected_paths = selected_paths if selected_paths else set()

        # Recalculate scene bounds
        self._calculate_scene_bounds()

        # Schedule update (throttled)
        self._schedule_update()

    def set_selection(self, selected_paths: set[str]) -> None:
        """Update selected nodes.

        Args:
            selected_paths: Set of selected path strings
        """
        self._selected_paths = selected_paths
        self._schedule_update()

    def set_current_directory(self, path: str | None) -> None:
        """Update the current directory.

        Args:
            path: Current directory path string
        """
        self._current_path = path
        self._schedule_update()

    def _calculate_scene_bounds(self) -> None:
        """Calculate scene bounds from current positions."""
        if not self._positions:
            return

        # Initialize with first position
        first_pos = next(iter(self._positions.values()))
        min_x = max_x = first_pos.center[0]
        min_z = max_z = first_pos.center[2]

        # Find bounds
        for pos in self._positions.values():
            center = pos.center
            x, z = center[0], center[2]
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_z = min(min_z, z)
            max_z = max(max_z, z)

        # Add padding (20% on each side)
        padding_x = (max_x - min_x) * 0.2 if max_x != min_x else 10.0
        padding_z = (max_z - min_z) * 0.2 if max_z != min_z else 10.0

        self._scene_bounds = {
            'min_x': min_x - padding_x,
            'max_x': max_x + padding_x,
            'min_z': min_z - padding_z,
            'max_z': max_z + padding_z,
        }

    def _schedule_update(self) -> None:
        """Schedule a delayed update (throttled)."""
        if not self._update_pending:
            self._update_pending = True
            self._update_timer.start(50)  # 50ms throttle = ~20 FPS max

    def _do_update(self) -> None:
        """Perform the actual update."""
        self._update_pending = False
        self.update()

    def paintEvent(self, event) -> None:
        """Paint the mini map.

        Args:
            event: Paint event
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw background
        self._draw_background(painter)

        # Draw grid
        self._draw_grid(painter)

        # Draw nodes
        self._draw_nodes(painter)

        # Draw camera frustum
        self._draw_camera_frustum(painter)

        # Draw zoom indicator
        self._draw_zoom_indicator(painter)

    def _draw_background(self, painter: QPainter) -> None:
        """Draw the mini map background.

        Args:
            painter: QPainter instance
        """
        # Rounded rectangle background
        painter.setBrush(self.BG_COLOR)
        painter.setPen(QPen(self.BORDER_COLOR, 2))
        painter.drawRoundedRect(
            self.MARGIN, self.MARGIN,
            self.width() - 2 * self.MARGIN,
            self.height() - 2 * self.MARGIN,
            8, 8
        )

    def _draw_grid(self, painter: QPainter) -> None:
        """Draw a subtle grid on the mini map.

        Args:
            painter: QPainter instance
        """
        bounds = self._scene_bounds
        draw_rect = self._get_draw_rect()

        # Calculate grid spacing (adaptive based on scene size)
        scene_width = bounds['max_x'] - bounds['min_x']
        grid_cells = 5  # Target number of grid cells
        grid_spacing_world = scene_width / grid_cells
        # Round to nice number
        grid_spacing_world = max(1.0, 10 ** math.floor(math.log10(grid_spacing_world)))

        # Draw grid lines
        painter.setPen(QPen(self.GRID_COLOR, 1))

        # Vertical lines (X in world space)
        x_start = math.floor(bounds['min_x'] / grid_spacing_world) * grid_spacing_world
        x = x_start
        while x <= bounds['max_x']:
            map_x = self._world_to_map_x(x)
            painter.drawLine(int(map_x), int(draw_rect.top()), int(map_x), int(draw_rect.bottom()))
            x += grid_spacing_world

        # Horizontal lines (Z in world space)
        z_start = math.floor(bounds['min_z'] / grid_spacing_world) * grid_spacing_world
        z = z_start
        while z <= bounds['max_z']:
            map_z = self._world_to_map_z(z)
            painter.drawLine(int(draw_rect.left()), int(map_z), int(draw_rect.right()), int(map_z))
            z += grid_spacing_world

    def _draw_nodes(self, painter: QPainter) -> None:
        """Draw nodes on the mini map.

        Args:
            painter: QPainter instance
        """
        if not self._positions:
            return

        # Draw all nodes as dots
        for path, pos in self._positions.items():
            if path not in self._nodes:
                continue

            node = self._nodes[path]
            center = pos.center
            map_pos = self._world_to_map(center[0], center[2])

            # Determine color
            if path == self._current_path:
                color = self.CURRENT_DIR_COLOR
                radius = 4
            elif path == self._root_path:
                color = self.ROOT_COLOR
                radius = 4
            elif path in self._selected_paths:
                color = self.SELECTED_COLOR
                radius = 3
            elif hasattr(node, 'is_directory') and node.is_directory:
                color = self.DIR_COLOR
                radius = 2
            else:
                color = self.FILE_COLOR
                radius = 1.5

            # Draw node
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(map_pos[0], map_pos[1]), radius, radius)

    def _draw_camera_frustum(self, painter: QPainter) -> None:
        """Draw the camera view frustum on the mini map.

        Args:
            painter: QPainter instance
        """
        if self._camera is None:
            return

        camera_state = self._camera.state
        pos = camera_state.position
        target = camera_state.target

        # Project camera position to map (top-down, so we use X and Z)
        cam_map_pos = self._world_to_map(pos[0], pos[2])
        target_map_pos = self._world_to_map(target[0], target[2])

        # Calculate frustum outline
        # Get camera direction
        direction = target - pos
        direction = direction / np.linalg.norm(direction)

        # Calculate frustum width at a distance
        fov_rad = math.radians(camera_state.fov)
        aspect = 1.0  # Assume square for mini map
        frustum_height = 2 * math.tan(fov_rad / 2)

        # Sample frustum at a few distances
        frustum_distances = [10.0, 50.0, 100.0]
        polygon_points = [QPointF(cam_map_pos[0], cam_map_pos[1])]

        for distance in frustum_distances:
            # Calculate frustum width at this distance
            half_width = distance * frustum_height * aspect / 2

            # Forward vector (in XZ plane)
            forward_xz = np.array([direction[0], direction[2]])
            forward_xz = forward_xz / (np.linalg.norm(forward_xz) + 1e-6)

            # Perpendicular vector (for frustum width)
            perp_xz = np.array([-forward_xz[1], forward_xz[0]])

            # Calculate frustum edge points
            for sign in [-1, 1]:
                edge_point_3d = pos[:2] + forward_xz * distance + perp_xz * (sign * half_width)
                # Project to map coordinates
                map_x, map_z = self._world_to_map(edge_point_3d[0], edge_point_3d[1])
                polygon_points.append(QPointF(map_x, map_z))

        # Draw frustum polygon
        painter.setBrush(QBrush(self.FRUSTUM_COLOR))
        painter.setPen(QPen(self.FRUSTUM_COLOR, 1))
        painter.drawPolygon(QPolygonF(polygon_points))

        # Draw camera position indicator
        painter.setBrush(self.SELECTED_COLOR)
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawEllipse(QPointF(cam_map_pos[0], cam_map_pos[1]), 3, 3)

    def _draw_zoom_indicator(self, painter: QPainter) -> None:
        """Draw a zoom level indicator.

        Args:
            painter: QPainter instance
        """
        if self._camera is None:
            return

        # Calculate zoom level (based on orbit distance)
        camera_state = self._camera.state
        pos = camera_state.position
        target = camera_state.target
        distance = np.linalg.norm(pos - target)

        # Map distance to zoom level (1-10)
        zoom_level = max(1, min(10, int(100 / distance)))

        # Draw in corner
        font = QFont("Arial", 9)
        painter.setFont(font)
        painter.setPen(QColor(200, 200, 200))

        text = f"Zoom: {zoom_level}x"
        painter.drawText(
            self.width() - self.MARGIN - 60,
            self.height() - self.MARGIN - 5,
            text
        )

    def _get_draw_rect(self) -> 'QtCore.QRectF':
        """Get the drawing rectangle (background area).

        Returns:
            QRectF of the drawable area
        """
        from PyQt6.QtCore import QRectF
        return QRectF(
            self.MARGIN + 2,
            self.MARGIN + 2,
            self.width() - 2 * self.MARGIN - 4,
            self.height() - 2 * self.MARGIN - 4
        )

    def _world_to_map(self, x: float, z: float) -> tuple[float, float]:
        """Convert world coordinates to mini map coordinates.

        Args:
            x: World X coordinate
            z: World Z coordinate

        Returns:
            Tuple of (map_x, map_y) coordinates
        """
        bounds = self._scene_bounds
        draw_rect = self._get_draw_rect()

        # Normalize to [0, 1]
        norm_x = (x - bounds['min_x']) / (bounds['max_x'] - bounds['min_x'] + 1e-6)
        norm_z = (z - bounds['min_z']) / (bounds['max_z'] - bounds['min_z'] + 1e-6)

        # Map to draw rectangle
        # Note: Z in world maps to Y in map (top-down view)
        map_x = draw_rect.left() + norm_x * draw_rect.width()
        map_y = draw_rect.bottom() - norm_z * draw_rect.height()  # Invert Z for Y

        return (map_x, map_y)

    def _world_to_map_x(self, x: float) -> float:
        """Convert world X to map X coordinate.

        Args:
            x: World X coordinate

        Returns:
            Map X coordinate
        """
        bounds = self._scene_bounds
        draw_rect = self._get_draw_rect()

        norm_x = (x - bounds['min_x']) / (bounds['max_x'] - bounds['min_x'] + 1e-6)
        return draw_rect.left() + norm_x * draw_rect.width()

    def _world_to_map_z(self, z: float) -> float:
        """Convert world Z to map Y coordinate.

        Args:
            z: World Z coordinate

        Returns:
            Map Y coordinate
        """
        bounds = self._scene_bounds
        draw_rect = self._get_draw_rect()

        norm_z = (z - bounds['min_z']) / (bounds['max_z'] - bounds['min_z'] + 1e-6)
        return draw_rect.bottom() - norm_z * draw_rect.height()
