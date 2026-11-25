"""OpenGL widget for rendering 3D file system visualization.

Uses PyOpenGL with Legacy OpenGL 2.1 for Mac compatibility.
"""

import math
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np
from PyQt6.QtCore import QTimer, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPainter, QFont, QColor, QFontMetrics
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

from OpenGL.GL import *
from OpenGL.GLU import gluPerspective, gluLookAt

from pyfsn.model.node import Node, NodeType
from pyfsn.view.camera import Camera
from pyfsn.view.performance import PerformanceMonitor, FrustumCuller, LevelOfDetail
from pyfsn.view.spotlight import SpotlightSearch
from pyfsn.view.bloom import SimpleBloom


class ColorMode(Enum):
    """Color mode for file visualization."""
    TYPE = "type"  # Color by file type
    AGE = "age"    # Color by file age (SGI fsn style)


@dataclass(frozen=True)
class Landscape:
    """Colours of the sky and the ground (fsn "-landscape" option)."""
    name: str
    sky_bottom: tuple[float, float, float]   # At the horizon
    sky_top: tuple[float, float, float]      # High above
    ground_near: tuple[float, float, float]
    ground_far: tuple[float, float, float]
    grid: tuple[float, float, float, float]


# The landscape names are fsn's; the colours are pyfsn's own interpretation
# of the man page descriptions (a meadow by day, an ocean at sunset, a desert
# at sunrise, black space, and the muted "indigo" variant of grass).
LANDSCAPES: dict[str, Landscape] = {
    "grass": Landscape("Grass", (0.6, 0.8, 1.0), (0.15, 0.25, 0.65),
                       (0.2, 0.45, 0.2), (0.15, 0.38, 0.15), (0.28, 0.48, 0.28, 0.25)),
    "ocean": Landscape("Ocean", (1.0, 0.55, 0.3), (0.25, 0.1, 0.4),
                       (0.05, 0.25, 0.45), (0.1, 0.35, 0.5), (0.2, 0.45, 0.6, 0.25)),
    "desert": Landscape("Desert", (1.0, 0.8, 0.6), (0.3, 0.5, 0.85),
                        (0.76, 0.6, 0.38), (0.68, 0.5, 0.3), (0.82, 0.68, 0.46, 0.25)),
    "space": Landscape("Space", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                       (0.03, 0.03, 0.06), (0.03, 0.03, 0.06), (0.2, 0.3, 0.6, 0.35)),
    "indigo": Landscape("Indigo", (0.45, 0.6, 0.8), (0.2, 0.25, 0.5),
                        (0.2, 0.36, 0.25), (0.16, 0.3, 0.22), (0.28, 0.42, 0.32, 0.25)),
}


def _unit_cube_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit cube as quad vertices, per-vertex normals and edge line vertices."""
    faces = [
        ((0, 0, 1), [(-.5, -.5, .5), (.5, -.5, .5), (.5, .5, .5), (-.5, .5, .5)]),
        ((0, 0, -1), [(.5, -.5, -.5), (-.5, -.5, -.5), (-.5, .5, -.5), (.5, .5, -.5)]),
        ((0, 1, 0), [(-.5, .5, .5), (.5, .5, .5), (.5, .5, -.5), (-.5, .5, -.5)]),
        ((0, -1, 0), [(-.5, -.5, -.5), (.5, -.5, -.5), (.5, -.5, .5), (-.5, -.5, .5)]),
        ((1, 0, 0), [(.5, -.5, .5), (.5, -.5, -.5), (.5, .5, -.5), (.5, .5, .5)]),
        ((-1, 0, 0), [(-.5, -.5, -.5), (-.5, -.5, .5), (-.5, .5, .5), (-.5, .5, -.5)]),
    ]
    quads = np.array([v for _, verts in faces for v in verts], dtype=np.float32)
    normals = np.array([n for n, verts in faces for _ in verts], dtype=np.float32)
    corners = [(-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)]  # (x, z) around the box
    edges = []
    for i in range(4):
        (x0, z0), (x1, z1) = corners[i], corners[(i + 1) % 4]
        edges += [(x0, -.5, z0), (x1, -.5, z1)]  # bottom ring
        edges += [(x0, .5, z0), (x1, .5, z1)]    # top ring
        edges += [(x0, -.5, z0), (x0, .5, z0)]   # vertical
    return quads, normals, np.array(edges, dtype=np.float32)


_CUBE_QUADS, _CUBE_NORMALS, _CUBE_EDGES = _unit_cube_arrays()


@dataclass
class _CubeBatch:
    """All cubes of one kind as flat vertex arrays (drawn in a few GL calls)."""
    count: int
    centers: np.ndarray       # (N, 3) for per-frame distance LOD
    max_scales: np.ndarray    # (N,)
    vertices: np.ndarray      # (N*24, 3)
    normals: np.ndarray       # (N*24, 3)
    colors: np.ndarray        # (N*24, 4)
    edge_vertices: np.ndarray  # (N*24, 3)
    edge_colors: np.ndarray    # (N*24, 4)


@dataclass
class CubeInstance:
    """Single cube instance data."""
    position: np.ndarray  # [x, y, z]
    scale: np.ndarray     # [w, h, d]
    color: np.ndarray     # [r, g, b, a]
    shininess: float = 30.0  # Specular shininess (0-128), default for file cubes
    emission: float = 0.0    # Emission intensity for glow effect (0.0 - 1.0+)
    path: str = ""           # Path of the node this instance stands for


class Renderer(QOpenGLWidget):
    """OpenGL widget for rendering 3D file system visualization."""

    # Signals
    node_clicked = pyqtSignal(Node, bool)  # Node, is_double_click
    node_focused = pyqtSignal(Node)
    selection_changed = pyqtSignal(set)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        # Camera
        self.camera = Camera()

        # Performance monitoring
        self._performance_monitor = PerformanceMonitor()

        # Optimization: Frustum culling and LOD (PR-5)
        self._frustum_culler = FrustumCuller()
        self._lod = LevelOfDetail(distances=[15.0, 40.0, 80.0, 150.0])
        self._lod_edge_threshold = 50.0  # Distance threshold for skipping edge rendering
        self._lod_small_cube_threshold = 100.0  # Distance threshold for skipping small cubes

        # Node data
        self._nodes: dict[str, Node] = {}
        self._positions: dict[str, object] = {}
        self._cubes: list[CubeInstance] = []
        self._platforms: list[CubeInstance] = []  # Directories are platforms
        self._connections: list[list[np.ndarray]] = []  # Polyline segments (L-shaped routing)
        self._connection_metadata: list[tuple[str, str]] = []  # (parent_path, child_path) for each connection (PR-4)
        self._slot_widths: dict[str, float] = {}  # Ground strip width per directory (for labels)
        self._pruned_paths: set[str] = set()
        # Directory label quads, rebuilt lazily after each load_layout
        self._label_cache: list[tuple[int, float, float, float, float]] | None = None
        # Sky and ground colours
        self._landscape: Landscape = LANDSCAPES["grass"]
        self._fog_dirty = False
        # Batched vertex arrays for platforms and file cubes; None = rebuild
        self._batches: tuple[_CubeBatch, _CubeBatch] | None = None
        # Half diagonal of the laid-out scene (for clip planes and fog)
        self._scene_radius: float = 50.0
        # Ground plane extent (centre x, centre z, half size), follows the layout bounds
        self._ground_extent: tuple[float, float, float] = (0.0, 0.0, 500.0)
        self._node_to_cube: dict[str, int] = {}
        self._node_to_platform: dict[str, int] = {}

        # Selection state
        self._selected_paths: set[str] = set()
        self._focused_path: str | None = None

        # Wire/connection highlighting (PR-4: Track highlighted connections)
        self._selected_connections: set[int] = set()  # Indices of highlighted connections

        # Text texture cache for directory labels on platforms
        self._text_textures: dict[str, int] = {}  # text -> texture_id

        # Color mode (SGI fsn style: age-based or type-based)
        self._color_mode: ColorMode = ColorMode.AGE

        # Colorblind-friendly age palette toggle
        self._colorblind: bool = False

        # Colors for different node types
        self._colors = {
            "directory": np.array([0.2, 0.6, 1.0, 1.0], dtype=np.float32),  # Bright blue (platform)
            "file": np.array([0.8, 0.8, 0.8, 1.0], dtype=np.float32),
            "symlink": np.array([0.3, 1.0, 0.3, 1.0], dtype=np.float32),
            "selected": np.array([1.0, 1.0, 0.3, 1.0], dtype=np.float32),
            "focused": np.array([1.0, 0.8, 0.0, 1.0], dtype=np.float32),
            # Directories whose contents have not been scanned yet / are pruned
            "unloaded": np.array([0.32, 0.40, 0.55, 1.0], dtype=np.float32),
            "pruned": np.array([0.45, 0.35, 0.50, 1.0], dtype=np.float32),
        }

        # Type-mode category colors (RGBA, alpha will be set to 0.8 for file blocks)
        self._type_colors = {
            "image": np.array([1.0, 0.4, 0.7, 1.0], dtype=np.float32),      # Pink
            "video": np.array([1.0, 0.2, 0.2, 1.0], dtype=np.float32),      # Red
            "audio": np.array([1.0, 0.6, 0.2, 1.0], dtype=np.float32),      # Orange
            "document": np.array([1.0, 0.85, 0.2, 1.0], dtype=np.float32),  # Yellow
            "code": np.array([0.2, 0.9, 0.6, 1.0], dtype=np.float32),       # Green/Teal
            "archive": np.array([0.7, 0.5, 0.3, 1.0], dtype=np.float32),    # Brown
            "data": np.array([0.4, 0.7, 1.0, 1.0], dtype=np.float32),       # Light blue
            "other": np.array([0.6, 0.6, 0.7, 1.0], dtype=np.float32),      # Gray
        }

        # Animation timer
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._on_timer)
        self._update_timer.start(16)  # ~60 FPS

        # Mouse tracking
        self.setMouseTracking(True)

        # Tooltip overlay reference (set by controller)
        self._tooltip = None

        # Spotlight search visualization
        self._spotlight = SpotlightSearch()
        self._path_to_cube_index: dict[str, int] = {}  # Maps path to cube index for spotlight
        self._path_to_platform_index: dict[str, int] = {}  # Maps path to platform index for spotlight

        # Bloom effect for cyberpunk visual style
        self._bloom = SimpleBloom(intensity=0.3)

        # Animation time for shader effects
        self._animation_time = 0.0

        # Current theme (for theme-specific rendering)
        self._theme = None

        # Input handler
        self._input_handler = None

        # OpenGL state
        self._initialized = False

    def _calculate_type_color(self, node: Node) -> np.ndarray:
        """Calculate color based on file type category.

        Args:
            node: Node to calculate color for.

        Returns:
            RGBA color array.
        """
        category = node.file_category
        return self._type_colors.get(category, self._type_colors["other"]).copy()

    @property
    def type_color_map(self) -> dict[str, np.ndarray]:
        """Return the mapping of file category names to colors."""
        return self._type_colors.copy()

    def _calculate_age_color(self, mtime: float) -> np.ndarray:
        """Calculate color based on file age (SGI fsn style).

        Args:
            mtime: File modification time (Unix timestamp)

        Returns:
            RGBA color array
        """
        now = time.time()
        age_days = (now - mtime) / 86400.0  # Seconds to days

        if self._colorblind:
            # Colorblind-friendly palette (blue->yellow luminance ramp, deuteranopia-safe)
            if age_days < 1:
                return np.array([0.27, 0.47, 0.71, 1.0], dtype=np.float32)  # Blue
            elif age_days < 7:
                return np.array([0.40, 0.76, 0.65, 1.0], dtype=np.float32)  # Teal
            elif age_days < 30:
                return np.array([0.93, 0.93, 0.42, 1.0], dtype=np.float32)  # Pale yellow
            elif age_days < 365:
                return np.array([0.96, 0.62, 0.30, 1.0], dtype=np.float32)  # Sand/orange
            else:
                return np.array([0.55, 0.34, 0.29, 1.0], dtype=np.float32)  # Muted brown

        if age_days < 1:           # Less than 24 hours
            return np.array([0.2, 1.0, 0.2, 1.0], dtype=np.float32)  # Bright green
        elif age_days < 7:         # Less than 1 week
            return np.array([0.3, 0.9, 1.0, 1.0], dtype=np.float32)  # Cyan/Sky blue
        elif age_days < 30:        # Less than 1 month
            return np.array([0.9, 0.9, 0.1, 1.0], dtype=np.float32)  # Bright Yellow
        elif age_days < 365:       # Less than 1 year
            return np.array([1.0, 0.5, 0.1, 1.0], dtype=np.float32)  # Orange
        else:                      # More than 1 year
            return np.array([0.7, 0.2, 0.1, 1.0], dtype=np.float32)  # Reddish Brown

    def set_colorblind_mode(self, enabled: bool) -> None:
        """Toggle the colorblind-friendly age palette and recolor cubes.

        Args:
            enabled: Whether to use the colorblind-friendly palette
        """
        if self._colorblind == enabled:
            return
        self._colorblind = enabled

        # Recolor all non-selected file cubes that use age coloring
        for path, idx in self._node_to_cube.items():
            if path in self._selected_paths:
                continue
            node = self._nodes.get(path)
            if node and node.type != NodeType.SYMLINK and self._color_mode == ColorMode.AGE:
                self._cubes[idx].color = self._file_base_color(node)
        self._batches = None
        self.update()

    @property
    def colorblind_mode(self) -> bool:
        """Get whether the colorblind-friendly palette is active."""
        return self._colorblind

    def set_theme(self, theme) -> None:
        """Set the current theme for theme-specific rendering."""
        from pyfsn.view.theme import Theme
        self._theme = theme
        self._batches = None

    def _calculate_emission(self, node: Node) -> float:
        """Calculate emission intensity based on file type and git status.

        Args:
            node: Node to calculate emission for

        Returns:
            Emission intensity (0.0 - 1.0)
        """
        # SGI Classic theme: no emission/glow effects
        if self._theme and hasattr(self._theme, 'static_wires') and self._theme.static_wires:
            return 0.0

        emission = 0.0

        # File type-based emission (cyberpunk style)
        if node.is_directory:
            emission = 0.0  # Directories don't glow by default
        else:
            # Get file extension from node's path
            ext = node.path.suffix.lower() if node.path.suffix else ""

            # High emission for code files (cyberpunk theme)
            if ext in {'.py', '.js', '.ts', '.rs', '.go', '.java', '.cpp', '.h', '.c'}:
                emission = 0.6
            elif ext in {'.md', '.txt', '.rst', '.adoc'}:
                emission = 0.2
            elif ext in {'.json', '.yaml', '.yml', '.toml', '.xml'}:
                emission = 0.3
            elif ext in {'.sh', '.bash', '.zsh', '.fish'}:
                emission = 0.5
            else:
                emission = 0.1

        # Git status-based emission boost
        if hasattr(node, 'git_status') and node.git_status:
            # Modified files glow more
            if node.git_status in {'M', 'MM', 'AM', 'TM'}:
                emission += 0.4
            # Added files have a special glow
            elif node.git_status in {'A', '??'}:
                emission += 0.3
            # Deleted files have a dim glow
            elif node.git_status == 'D':
                emission = max(emission * 0.3, 0.1)

        # Selected items get extra emission
        if str(node.path) in self._selected_paths:
            emission += 0.3

        return min(emission, 1.0)

    def _platform_base_color(self, path: str, node: Node | None) -> np.ndarray:
        """Unselected colour of a directory pedestal."""
        if path in self._pruned_paths:
            return self._colors["pruned"].copy()
        if node is not None and node.is_directory and not node.is_loaded:
            return self._colors["unloaded"].copy()
        return self._colors["directory"].copy()

    def _file_base_color(self, node: Node) -> np.ndarray:
        """Unselected colour of a file box (age or type based)."""
        if node.type == NodeType.SYMLINK:
            color = self._colors["symlink"].copy()
        elif self._color_mode == ColorMode.AGE:
            color = self._calculate_age_color(node.mtime)
        else:
            color = self._calculate_type_color(node)
        # Semi-transparent file blocks (SGI fsn style translucency)
        color[3] = 0.8
        return color

    def initializeGL(self) -> None:
        """Initialize OpenGL resources."""
        glClearColor(0.1, 0.1, 0.15, 1.0)
        glEnable(GL_DEPTH_TEST)

        # Enable OpenGL lighting (SGI fsn style specular highlights)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_NORMALIZE)

        # Directional light from upper-right-front
        glLightfv(GL_LIGHT0, GL_POSITION, [0.5, 1.0, 0.8, 0.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.35, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.75, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 0.9, 1.0])

        # Set up depth fog (SGI fsn style)
        self._setup_fog()

        self._initialized = True

    def _setup_fog(self) -> None:
        """Set up depth fog for distance fade effect."""
        glEnable(GL_FOG)
        glFogi(GL_FOG_MODE, GL_LINEAR)
        # Fade into the colour of the sky at the horizon, so the far edge of
        # the ground melts into it instead of ending in a dark band
        glFogfv(GL_FOG_COLOR, [*self._sky_colors[0], 1.0])
        glFogf(GL_FOG_START, 80.0)
        glFogf(GL_FOG_END, 600.0)
        glHint(GL_FOG_HINT, GL_NICEST)

    def _update_view_range(self) -> None:
        """Fit the clip planes and the fog to the camera's distance from the scene."""
        state = self.camera.state
        center_x, center_z, _half = self._ground_extent
        scene_radius = self._scene_radius
        to_center = math.hypot(
            float(state.position[0]) - center_x, float(state.position[2]) - center_z
        )
        altitude = max(float(state.position[1]), 0.0)
        reach = math.hypot(to_center + scene_radius, altitude)

        far = max(1000.0, reach * 1.3)
        # Keep the depth buffer usable: push the near plane out when high up
        near = min(max(0.1, altitude * 0.02), 5.0)
        state.far = far
        state.near = near

        aspect = self.width() / max(1, self.height())
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(state.fov, aspect, near, far)
        glMatrixMode(GL_MODELVIEW)

        # Fog starts beyond the nearby pedestals and never hides the scene
        # the camera was pulled back to look at
        glFogf(GL_FOG_START, max(80.0, reach * 0.6))
        glFogf(GL_FOG_END, max(600.0, reach * 2.0))

    def resizeGL(self, w: int, h: int) -> None:
        """Handle viewport resize."""
        glViewport(0, 0, int(w * self.devicePixelRatio()), int(h * self.devicePixelRatio()))
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = w / max(1, h)
        gluPerspective(self.camera.state.fov, aspect, self.camera.state.near, self.camera.state.far)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self) -> None:
        """Render the scene."""
        self._performance_monitor.start_frame()

        if not self._initialized:
            self._performance_monitor.end_frame()
            return

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if self._fog_dirty:
            self._fog_dirty = False
            self._setup_fog()

        # The landscape can be far wider than a fixed far plane / fog range:
        # fit both to how far the camera is from the far side of the scene
        self._update_view_range()

        # Update frustum culling (PR-5: Optimization)
        aspect = self.width() / max(1, self.height())
        view_matrix = self.camera.view_matrix
        proj_matrix = self.camera.projection_matrix(aspect)
        self._frustum_culler.update_from_camera(view_matrix, proj_matrix, aspect)

        # Draw sky gradient first (fullscreen background, no lighting)
        glDisable(GL_LIGHTING)
        self._draw_sky_gradient()

        glLoadIdentity()

        # Apply camera view
        pos = self.camera.state.position
        target = self.camera.state.target
        up = self.camera.state.up
        gluLookAt(pos[0], pos[1], pos[2],
                  target[0], target[1], target[2],
                  up[0], up[1], up[2])

        # Draw ground and connections without lighting
        self._draw_ground_grid()
        self._draw_connections()

        # Enable lighting for 3D objects
        glEnable(GL_LIGHTING)

        if self._spotlight.is_active:
            # Search spotlight recolours every node each frame: draw one by one
            # === Opaque pass: platforms (directories) ===
            for platform in self._platforms:
                self._draw_cube_with_spotlight(platform)
            glDisable(GL_LIGHTING)
            for platform in self._platforms:
                self._draw_cube_edges(platform)
            glEnable(GL_LIGHTING)

            # Draw directory labels on platforms (SGI fsn style)
            glDisable(GL_LIGHTING)
            self._draw_directory_labels()
            glEnable(GL_LIGHTING)

            # === Translucent pass: file blocks (alpha=0.8) ===
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDepthMask(GL_FALSE)

            for cube in self._cubes:
                self._draw_cube_with_spotlight(cube)

            glDepthMask(GL_TRUE)
            glDisable(GL_BLEND)

            # Draw file edges on top (opaque, crisp lines over translucent faces)
            glDisable(GL_LIGHTING)
            for cube in self._cubes:
                self._draw_cube_edges(cube)
        else:
            self._draw_scene_batched()
            glDisable(GL_LIGHTING)

        # Draw spotlights for selection (transparent)
        self._draw_selection_spotlights()

        # Draw search spotlights (Phase 3.2)
        self._draw_search_spotlights()

        self._performance_monitor.end_frame()
        self._performance_monitor.set_instance_count(len(self._cubes) + len(self._platforms))
        self._performance_monitor.set_draw_calls(len(self._cubes) + len(self._platforms))

    def _draw_ground_grid(self) -> None:
        """Draw the ground plane with grid lines (FSN classic style)."""
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        center_x, center_z, ground_size = self._ground_extent
        ground_y = -0.01  # Just below y=0 labels
        x0, x1 = center_x - ground_size, center_x + ground_size
        z0, z1 = center_z - ground_size, center_z + ground_size

        # Ground plane (Rich green gradient, FSN grass-like)
        glBegin(GL_QUADS)
        glColor4f(*self._landscape.ground_near, 1.0)
        glVertex3f(x0, ground_y, z0)
        glVertex3f(x1, ground_y, z0)
        glColor4f(*self._landscape.ground_far, 1.0)
        glVertex3f(x1, ground_y, z1)
        glVertex3f(x0, ground_y, z1)
        glEnd()

        # Grid lines - subtle, on a fixed world lattice so they do not swim
        glLineWidth(1.0)
        glColor4f(*self._landscape.grid)  # Very subtle grid
        grid_spacing = 20.0
        # Keep the line count bounded on very wide landscapes
        while ground_size / grid_spacing > 60:
            grid_spacing *= 2.0

        glBegin(GL_LINES)
        first_x = math.ceil(x0 / grid_spacing)
        last_x = math.floor(x1 / grid_spacing)
        for i in range(first_x, last_x + 1):
            offset = i * grid_spacing
            glVertex3f(offset, ground_y + 0.01, z0)
            glVertex3f(offset, ground_y + 0.01, z1)
        first_z = math.ceil(z0 / grid_spacing)
        last_z = math.floor(z1 / grid_spacing)
        for i in range(first_z, last_z + 1):
            offset = i * grid_spacing
            glVertex3f(x0, ground_y + 0.01, offset)
            glVertex3f(x1, ground_y + 0.01, offset)
        glEnd()

        glDisable(GL_BLEND)

    @property
    def _sky_colors(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Sky colours: (at the horizon, high above)."""
        return self._landscape.sky_bottom, self._landscape.sky_top

    def set_landscape(self, key: str) -> bool:
        """Switch the sky/ground colours.

        Returns:
            False if the landscape name is unknown
        """
        landscape = LANDSCAPES.get(key)
        if landscape is None:
            return False
        self._landscape = landscape
        self._fog_dirty = True  # Fog colour follows the horizon; needs the GL context
        self.update()
        return True

    def _horizon_ndc_y(self) -> float:
        """Screen height of the horizon in normalized device coordinates."""
        state = self.camera.state
        forward = state.target - state.position
        norm = float(np.linalg.norm(forward))
        if norm < 1e-6:
            return 0.0
        # Positive pitch = looking down = horizon above the screen centre
        pitch = -math.asin(max(-1.0, min(1.0, float(forward[1]) / norm)))
        if abs(pitch) > math.radians(89.0):
            return 4.0 if pitch > 0 else -4.0
        y = math.tan(pitch) / math.tan(math.radians(state.fov) / 2.0)
        return max(-4.0, min(4.0, y))

    def _draw_sky_gradient(self) -> None:
        """Draw sky gradient as a fullscreen quad (FSN classic style)."""
        glDisable(GL_DEPTH_TEST)
        
        # Save current matrices
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # The gradient hangs on the real horizon, so the sky stays put when
        # the camera tilts (bottom colour at the horizon, top colour above it)
        horizon = self._horizon_ndc_y()
        bottom_color, top_color = self._sky_colors
        top = max(horizon + 1.2, 1.0)
        glBegin(GL_QUADS)
        # Below the horizon: plain horizon colour (normally hidden by the ground)
        glColor4f(*bottom_color, 1.0)
        glVertex3f(-1.0, -1.0, -0.999)
        glVertex3f(1.0, -1.0, -0.999)
        glVertex3f(1.0, horizon, -0.999)
        glVertex3f(-1.0, horizon, -0.999)
        # Horizon up to the zenith colour
        glVertex3f(-1.0, horizon, -0.999)
        glVertex3f(1.0, horizon, -0.999)
        glColor4f(*top_color, 1.0)
        glVertex3f(1.0, top, -0.999)
        glVertex3f(-1.0, top, -0.999)
        glEnd()

        # Restore matrices
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
        glEnable(GL_DEPTH_TEST)

    def _draw_connections(self) -> None:
        """Draw wire connections between platforms with highlighting and pulse effect (PR-4)."""
        # Check if theme uses static wires (SGI Classic style)
        use_static = self._theme and hasattr(self._theme, 'static_wires') and self._theme.static_wires

        if use_static:
            self._draw_static_connections()
            return

        # Wire pulse configuration
        pulse_speed = 2.0
        pulse_intensity = 0.3

        # Calculate pulse factor (0.0 to 1.0)
        pulse_phase = (self._animation_time * pulse_speed) % 3.0
        pulse = math.sin(pulse_phase) * 0.5 + 0.5  # Normalize to 0-1

        # Draw non-selected connections first (normal style with pulse)
        glLineWidth(2.0)

        # Interpolate color for pulse effect
        # Base: white, Pulse: cyan-magenta
        base_r, base_g, base_b = 1.0, 1.0, 1.0
        pulse_r = 0.0 * pulse + 1.0 * (1 - pulse)
        pulse_g = 1.0 * pulse + 0.0 * (1 - pulse)
        pulse_b = 1.0 * pulse + 1.0 * (1 - pulse)

        r = base_r + (pulse_r - base_r) * pulse_intensity * pulse
        g = base_g + (pulse_g - base_g) * pulse_intensity * pulse
        b = base_b + (pulse_b - base_b) * pulse_intensity * pulse

        glColor4f(r, g, b, 0.4)

        for i, polyline in enumerate(self._connections):
            if i not in self._selected_connections:
                glBegin(GL_LINE_STRIP)
                for pt in polyline:
                    glVertex3fv(pt)
                glEnd()

        # Draw selected connections with highlight style and stronger pulse (PR-4)
        if self._selected_connections:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)

            # Main wire with enhanced pulse
            glLineWidth(4.0)

            # Enhanced pulse for selected wires
            selected_pulse = (pulse * 0.5 + 0.5)  # 0.25 to 0.75 range
            sel_r = 1.0
            sel_g = 1.0 - selected_pulse * 0.2  # Slight yellow variation
            sel_b = 0.6 + selected_pulse * 0.3  # Cyan shift

            glColor4f(sel_r, sel_g, sel_b, 0.9)

            for i in self._selected_connections:
                if i < len(self._connections):
                    polyline = self._connections[i]
                    glBegin(GL_LINE_STRIP)
                    for pt in polyline:
                        glVertex3fv(pt)
                    glEnd()

            # Add animated glow effect for selected wires
            glLineWidth(8.0)

            # Cyberpunk glow color animation
            glow_pulse = (math.sin(self._animation_time * 3.0) * 0.5 + 0.5)
            glow_r = 0.0 + glow_pulse * 0.3
            glow_g = 1.0 - glow_pulse * 0.2
            glow_b = 1.0

            glColor4f(glow_r, glow_g, glow_b, 0.2 + glow_pulse * 0.2)

            for i in self._selected_connections:
                if i < len(self._connections):
                    polyline = self._connections[i]
                    glBegin(GL_LINE_STRIP)
                    for pt in polyline:
                        glVertex3fv(pt)
                    glEnd()

            # Add traveling pulse effect along the wire
            glLineWidth(2.0)
            travel_speed = 3.0
            travel_phase = (self._animation_time * travel_speed) % 1.0

            glColor4f(0.0, 1.0, 1.0, 0.6)  # Bright cyan

            # Draw small segments for traveling pulse
            for i in self._selected_connections:
                if i < len(self._connections):
                    polyline = self._connections[i]
                    length = self._polyline_length(polyline)
                    if length > 0:
                        pulse_pos, direction = self._interpolate_polyline(polyline, travel_phase)

                        segment_length = min(length * 0.1, 2.0)
                        seg_start = pulse_pos - direction * segment_length * 0.5
                        seg_end = pulse_pos + direction * segment_length * 0.5

                        glBegin(GL_LINES)
                        glVertex3fv(seg_start)
                        glVertex3fv(seg_end)
                        glEnd()

            glDisable(GL_BLEND)

    def _draw_static_connections(self) -> None:
        """Draw simple static wire connections (SGI Classic style - no pulse/animation)."""
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Draw non-selected connections as simple white lines
        glLineWidth(1.5)
        glColor4f(1.0, 1.0, 1.0, 0.6)

        for i, polyline in enumerate(self._connections):
            if i not in self._selected_connections:
                glBegin(GL_LINE_STRIP)
                for pt in polyline:
                    glVertex3fv(pt)
                glEnd()

        # Draw selected connections with highlight (but no animation)
        if self._selected_connections:
            glLineWidth(3.0)
            glColor4f(1.0, 1.0, 0.6, 0.9)

            for i in self._selected_connections:
                if i < len(self._connections):
                    polyline = self._connections[i]
                    glBegin(GL_LINE_STRIP)
                    for pt in polyline:
                        glVertex3fv(pt)
                    glEnd()

        glDisable(GL_BLEND)

    def _draw_selection_spotlights(self) -> None:
        """Draw spotlights over selected items."""
        if not self._selected_paths:
            return

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        for path in self._selected_paths:
            # Check if it's a file cube
            if path in self._node_to_cube:
                idx = self._node_to_cube[path]
                cube = self._cubes[idx]
                self._draw_spotlight(cube.position, height=8.0, radius=2.0)
            
            # Directories get no cone: like in fsn, a selected directory and
            # the line leading to it brighten instead
        
        glDisable(GL_BLEND)

    def _draw_spotlight(self, target_pos: np.ndarray, height: float, radius: float) -> None:
        """Draw a conical spotlight."""
        x, y, z = target_pos
        
        # Cone tip (light source)
        tip_y = y + height
        
        # Draw translucent light cone
        glColor4f(1.0, 1.0, 0.8, 0.2)  # Pale yellow, transparent
        
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(x, tip_y, z)  # Tip
        
        segments = 16
        for i in range(segments + 1):
            angle = 2.0 * math.pi * i / segments
            dx = radius * math.cos(angle)
            dz = radius * math.sin(angle)
            glVertex3f(x + dx, y, z + dz)
            
        glEnd()
        
        # Draw base circle highlight (brighter)
        glLineWidth(2.0)
        glColor4f(1.0, 1.0, 0.5, 0.6)
        
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            angle = 2.0 * math.pi * i / segments
            dx = radius * math.cos(angle)
            dz = radius * math.sin(angle)
            glVertex3f(x + dx, y + 0.05, z + dz)
        glEnd()

    # ------------------------------------------------------------------
    # Batched drawing: all cubes of the scene in a handful of GL calls
    # ------------------------------------------------------------------

    def _build_batch(self, instances: list[CubeInstance], bright_edges: bool) -> _CubeBatch:
        """Flatten cube instances into vertex arrays."""
        count = len(instances)
        if count == 0:
            empty3 = np.zeros((0, 3), dtype=np.float32)
            empty4 = np.zeros((0, 4), dtype=np.float32)
            return _CubeBatch(0, empty3, np.zeros(0, dtype=np.float32), empty3, empty3,
                              empty4, empty3, empty4)

        centers = np.array([c.position for c in instances], dtype=np.float32)
        scales = np.array([c.scale for c in instances], dtype=np.float32)
        colors = np.array([c.color for c in instances], dtype=np.float32)
        emission = np.array([c.emission for c in instances], dtype=np.float32)

        # Bake the glow that _draw_cube applies per cube
        lit = colors.copy()
        glowing = emission > 0.01
        if self._bloom.enabled and glowing.any():
            glow = (emission * self._bloom.intensity * 2.0)[:, None]
            lit[:, :3] = np.where(glowing[:, None], np.minimum(1.0, lit[:, :3] + glow), lit[:, :3])
        lit[:, :3] = np.minimum(1.0, lit[:, :3] * (1.0 + emission[:, None] * 0.3))

        vertices = (centers[:, None, :] + _CUBE_QUADS[None, :, :] * scales[:, None, :]).reshape(-1, 3)
        normals = np.tile(_CUBE_NORMALS, (count, 1))
        vertex_colors = np.repeat(lit, 24, axis=0)

        edge_vertices = (centers[:, None, :] + _CUBE_EDGES[None, :, :] * scales[:, None, :]).reshape(-1, 3)
        edge = colors.copy()
        if bright_edges:
            edge[:, :3] = np.minimum(1.0, edge[:, :3] * 0.5 + 0.5)
        else:
            edge[:, :3] *= 0.3
        edge_colors = np.repeat(edge, 24, axis=0)

        return _CubeBatch(
            count=count,
            centers=centers,
            max_scales=scales.max(axis=1),
            vertices=np.ascontiguousarray(vertices, dtype=np.float32),
            normals=np.ascontiguousarray(normals, dtype=np.float32),
            colors=np.ascontiguousarray(vertex_colors, dtype=np.float32),
            edge_vertices=np.ascontiguousarray(edge_vertices, dtype=np.float32),
            edge_colors=np.ascontiguousarray(edge_colors, dtype=np.float32),
        )

    def _draw_batch_faces(self, batch: _CubeBatch, specular: float, shininess: float) -> None:
        """Draw the lit faces of a batch."""
        if batch.count == 0:
            return
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [specular, specular, specular, 1.0])
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess)
        glVertexPointer(3, GL_FLOAT, 0, batch.vertices)
        glNormalPointer(GL_FLOAT, 0, batch.normals)
        glColorPointer(4, GL_FLOAT, 0, batch.colors)
        glDrawArrays(GL_QUADS, 0, batch.count * 24)

    def _draw_batch_edges(self, batch: _CubeBatch, line_width: float) -> None:
        """Draw the outlines of the cubes that are close enough to matter."""
        if batch.count == 0:
            return
        camera_pos = self.camera.state.position
        distances = np.linalg.norm(batch.centers - camera_pos, axis=1)
        visible = distances <= self._lod_edge_threshold
        visible &= ~((batch.max_scales < 0.5) & (distances > self._lod_small_cube_threshold * 0.3))
        selected = np.nonzero(visible)[0]
        if selected.size == 0:
            return
        indices = (selected[:, None] * 24 + np.arange(24)[None, :]).astype(np.uint32).ravel()

        glLineWidth(line_width)
        glVertexPointer(3, GL_FLOAT, 0, batch.edge_vertices)
        glColorPointer(4, GL_FLOAT, 0, batch.edge_colors)
        glDrawElements(GL_LINES, indices.size, GL_UNSIGNED_INT, indices)

    def _draw_scene_batched(self) -> None:
        """Draw platforms, labels and file cubes from the batched arrays.

        Same passes and look as the per-cube path, but the cost no longer
        grows with one Python call per cube.
        """
        if self._batches is None:
            self._batches = (
                self._build_batch(self._platforms, bright_edges=True),
                self._build_batch(self._cubes, bright_edges=False),
            )
        platforms, files = self._batches

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        def lit_pass(batch: _CubeBatch, specular: float, shininess: float) -> None:
            # Vertex colours drive ambient+diffuse. _draw_cube uses 40% of the
            # colour as ambient, so scale the ambient light instead.
            glEnable(GL_LIGHTING)
            glEnable(GL_COLOR_MATERIAL)
            glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
            glLightfv(GL_LIGHT0, GL_AMBIENT, [0.12, 0.12, 0.14, 1.0])
            glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.08, 0.08, 0.08, 1.0])
            glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, [0.0, 0.0, 0.0, 1.0])
            glEnableClientState(GL_NORMAL_ARRAY)
            self._draw_batch_faces(batch, specular, shininess)
            glDisableClientState(GL_NORMAL_ARRAY)
            glDisable(GL_COLOR_MATERIAL)
            glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.35, 1.0])
            glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
            glDisable(GL_LIGHTING)

        def edge_pass(batch: _CubeBatch, line_width: float) -> None:
            glEnable(GL_POLYGON_OFFSET_LINE)
            glPolygonOffset(-1.0, -1.0)
            self._draw_batch_edges(batch, line_width)
            glDisable(GL_POLYGON_OFFSET_LINE)

        # Opaque pass: platforms (matte)
        lit_pass(platforms, 0.2, 5.0)
        edge_pass(platforms, 2.0)

        # Directory names on the ground (uses its own immediate-mode quads)
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        self._draw_directory_labels()
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        # Translucent pass: file blocks (glossy)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)
        lit_pass(files, 0.8, 50.0)
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)
        edge_pass(files, 1.5)

        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

    def _draw_cube(self, cube: CubeInstance) -> None:
        """Draw a single cube with OpenGL material-based lighting."""
        x, y, z = cube.position
        w, h, d = cube.scale
        r, g, b, a = cube.color

        # Apply emission-based glow effect
        r, g, b, a = self._bloom.apply_glow(r, g, b, a, cube.emission)

        # Set material properties (OpenGL lighting handles shading per-face via normals)
        diffuse = [r, g, b, a]
        ambient = [r * 0.4, g * 0.4, b * 0.4, a]
        shininess = cube.shininess

        # Files get glossy specular (plastic look), platforms get matte
        if shininess > 10:
            specular = [0.8, 0.8, 0.8, 1.0]
        else:
            specular = [0.2, 0.2, 0.2, 1.0]

        # Emission for glow effects
        emission_val = cube.emission * 0.3
        emission_color = [r * emission_val, g * emission_val, b * emission_val, 1.0]

        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse)
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular)
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, min(shininess, 128.0))
        glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, emission_color)

        glPushMatrix()
        glTranslatef(x, y, z)
        glScalef(w, h, d)

        # Draw cube faces with normals (lighting calculates shading)
        glBegin(GL_QUADS)
        # Front face
        glNormal3f(0, 0, 1)
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        # Back face
        glNormal3f(0, 0, -1)
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(-0.5, 0.5, -0.5)
        glVertex3f(0.5, 0.5, -0.5)
        # Top face
        glNormal3f(0, 1, 0)
        glVertex3f(-0.5, 0.5, 0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(0.5, 0.5, -0.5)
        glVertex3f(-0.5, 0.5, -0.5)
        # Bottom face
        glNormal3f(0, -1, 0)
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(-0.5, -0.5, 0.5)
        # Right face
        glNormal3f(1, 0, 0)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(0.5, 0.5, -0.5)
        glVertex3f(0.5, 0.5, 0.5)
        # Left face
        glNormal3f(-1, 0, 0)
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        glVertex3f(-0.5, 0.5, -0.5)
        glEnd()

        # Reset emission to zero
        glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, [0.0, 0.0, 0.0, 1.0])

        glPopMatrix()

    def _draw_cube_with_spotlight(self, cube: CubeInstance) -> None:
        """Draw a single cube with spotlight search effects applied.

        Args:
            cube: CubeInstance to draw
        """
        # Apply spotlight effects if active
        if self._spotlight.is_active:
            # Find the path for this cube
            path = cube.path
            if path:
                # Get spotlight-modified color
                original_color = cube.color.copy()
                modified_color = self._spotlight.get_node_color(path, original_color)

                # Create a temporary cube with modified color
                spotlight_cube = CubeInstance(
                    position=cube.position,
                    scale=cube.scale,
                    color=modified_color,
                    shininess=cube.shininess
                )
                self._draw_cube(spotlight_cube)
                return

        # No spotlight active, draw normally
        self._draw_cube(cube)

    def _draw_search_spotlights(self) -> None:
        """Draw cone spotlights over matching search results.

        Phase 3.2: Spotlight Search Visualization
        """
        if not self._spotlight.is_active:
            return

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Draw spotlights for matching cubes
        for path, idx in self._path_to_cube_index.items():
            if self._spotlight.should_draw_spotlight(path):
                params = self._spotlight.get_spotlight_params(path)
                if params and idx < len(self._cubes):
                    height, radius, alpha = params
                    cube = self._cubes[idx]
                    self._draw_search_spotlight(cube.position, height, radius, alpha)

        # Draw spotlights for matching platforms
        for path, idx in self._path_to_platform_index.items():
            if self._spotlight.should_draw_spotlight(path):
                params = self._spotlight.get_spotlight_params(path)
                if params and idx < len(self._platforms):
                    height, radius, alpha = params
                    platform = self._platforms[idx]
                    self._draw_search_spotlight(platform.position, height * 1.5, radius * 2, alpha)

        glDisable(GL_BLEND)

    def _draw_search_spotlight(self, target_pos: np.ndarray, height: float,
                               radius: float, alpha: float) -> None:
        """Draw a conical search spotlight.

        Args:
            target_pos: Target position (center base)
            height: Height of the cone
            radius: Radius of the cone base
            alpha: Alpha value (transparency)
        """
        x, y, z = target_pos

        # Cone tip (light source)
        tip_y = y + height

        # Draw translucent light cone with cyan tint (search theme)
        glColor4f(0.5, 0.8, 1.0, alpha * 0.3)  # Cyan, transparent

        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(x, tip_y, z)  # Tip

        segments = 16
        for i in range(segments + 1):
            angle = 2.0 * math.pi * i / segments
            dx = radius * math.cos(angle)
            dz = radius * math.sin(angle)
            glVertex3f(x + dx, y, z + dz)

        glEnd()

        # Draw base circle highlight (brighter cyan ring)
        glLineWidth(2.0)
        glColor4f(0.3, 1.0, 1.0, alpha * 0.8)  # Bright cyan

        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            angle = 2.0 * math.pi * i / segments
            dx = radius * math.cos(angle)
            dz = radius * math.sin(angle)
            glVertex3f(x + dx, y + 0.05, z + dz)
        glEnd()

    def _draw_cube_edges(self, cube: CubeInstance) -> None:
        """Draw cube edges for better definition.

        PR-5: Uses LOD to skip edge rendering for distant objects.
        """
        # Calculate distance to camera for LOD (PR-5)
        camera_pos = self.camera.state.position
        distance = np.linalg.norm(cube.position - camera_pos)

        # Skip edge rendering for distant objects (PR-5 optimization)
        if distance > self._lod_edge_threshold:
            return

        # Also skip edges for very small cubes at medium distance
        max_scale = max(cube.scale)
        if max_scale < 0.5 and distance > self._lod_small_cube_threshold * 0.3:
            return

        x, y, z = cube.position
        w, h, d = cube.scale

        # Use polygon offset to prevent Z-fighting
        glEnable(GL_POLYGON_OFFSET_LINE)
        glPolygonOffset(-1.0, -1.0)  # Pull lines toward camera

        # Dark edge color - computed from cube color
        r, g, b, a = cube.color

        # Check if this is a platform (pedestal)
        # Platforms have distinct edge behavior in FSN (brighter/outlined)
        is_platform = cube.scale[1] < 1.0 and cube.scale[0] > 2.0

        if is_platform:
            # Highlight edges for pedestals (white-ish bright edges, SGI fsn style)
            edge_color = [min(1.0, r * 0.5 + 0.5), min(1.0, g * 0.5 + 0.5), min(1.0, b * 0.5 + 0.5), a]
            glLineWidth(2.0)  # Thicker edges for platforms
        else:
            # Standard dark edges for file cubes
            edge_color = [r * 0.3, g * 0.3, b * 0.3, a]
            glLineWidth(1.5)
            
        glColor4fv(edge_color)

        glPushMatrix()
        glTranslatef(x, y, z)
        glScalef(w, h, d)

        # Draw cube edges
        glBegin(GL_LINES)
        # Bottom face edges
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(-0.5, -0.5, -0.5)

        # Top face edges
        glVertex3f(-0.5, 0.5, -0.5)
        glVertex3f(0.5, 0.5, -0.5)
        glVertex3f(0.5, 0.5, -0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        glVertex3f(-0.5, 0.5, -0.5)

        # Vertical edges
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(-0.5, 0.5, -0.5)
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(0.5, 0.5, -0.5)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        glEnd()

        glPopMatrix()

        glDisable(GL_POLYGON_OFFSET_LINE)

    def load_layout(self, layout_result, nodes: dict[str, Node], selection: set[str] | None = None) -> None:
        """Load nodes from layout result for rendering.

        Args:
            layout_result: LayoutResult from LayoutEngine with positions
            nodes: Dictionary mapping path strings to Node objects
            selection: Set of selected node paths
        """
        self._cubes.clear()
        self._platforms.clear()
        self._connections.clear()
        self._connection_metadata.clear()  # PR-4: Clear connection metadata
        self._node_to_cube.clear()
        self._node_to_platform.clear()
        self._path_to_cube_index.clear()
        self._path_to_platform_index.clear()
        self._nodes = nodes
        self._positions = layout_result.positions
        self._slot_widths = getattr(layout_result, "slot_widths", {}) or {}
        self._pruned_paths = set(getattr(layout_result, "pruned", ()) or ())
        self._label_cache = None
        self._batches = None
        self._update_ground_extent(layout_result.positions)

        # Cancel any in-progress camera animation so it doesn't override
        # the camera position set after scene load
        self._is_animating = False

        selection = selection or set()

        for path_str, position in layout_result.positions.items():
            if path_str not in nodes:
                continue
            node = nodes[path_str]
            if node.type == NodeType.DIRECTORY:
                self._load_directory_instance(path_str, node, position, selection)
            else:
                self._load_file_instance(path_str, node, position, selection)

        self._build_connections(layout_result, nodes)
        self.update()

    def _load_directory_instance(self, path_str: str, node: Node, position, selection: set[str]) -> None:
        """Create and register a platform (directory) render instance."""
        platform_w = position.width
        platform_d = position.depth

        # FSN style: pedestal height comes from the layout (size of the
        # files in the directory, see LayoutConfig.dir_height_mode)
        height = position.height

        # "Grow Up" model: Base at y=0, extends upwards
        pos_array = np.array([
            position.x + position.width / 2,
            height / 2,  # Center is at radius (height/2) above ground (0)
            position.z + position.depth / 2,
        ], dtype=np.float32)

        scale_array = np.array([
            platform_w,
            height,
            platform_d,
        ], dtype=np.float32)

        if path_str in selection:
            color = self._colors["selected"].copy()
        else:
            color = self._platform_base_color(path_str, node)

        emission = self._calculate_emission(node)

        platform = CubeInstance(position=pos_array, scale=scale_array, color=color, shininess=5.0,
                                emission=emission, path=path_str)
        self._node_to_platform[path_str] = len(self._platforms)
        self._path_to_platform_index[path_str] = len(self._platforms)
        self._platforms.append(platform)

    def _load_file_instance(self, path_str: str, node: Node, position, selection: set[str]) -> None:
        """Create and register a cube (file) render instance."""
        height = position.height

        # The layout already places files on top of their pedestal
        pos_array = np.array([
            position.x + position.width / 2,
            position.y + height / 2,
            position.z + position.depth / 2,
        ], dtype=np.float32)

        scale_array = np.array([
            position.width * 0.5,
            height,
            position.depth * 0.5,
        ], dtype=np.float32)

        if path_str in selection:
            color = self._colors["selected"].copy()
            color[3] = 0.8
        else:
            color = self._file_base_color(node)

        emission = self._calculate_emission(node)

        cube = CubeInstance(position=pos_array, scale=scale_array, color=color, shininess=50.0,
                            emission=emission, path=path_str)
        self._node_to_cube[path_str] = len(self._cubes)
        self._path_to_cube_index[path_str] = len(self._cubes)
        self._cubes.append(cube)

    def _build_connections(self, layout_result, nodes: dict[str, Node]) -> None:
        """Build wire connections between directory platforms with L-shaped routing."""
        for parent_path, child_path in layout_result.connections:
            if parent_path not in nodes or child_path not in nodes:
                continue

            parent_node = nodes[parent_path]
            child_node = nodes[child_path]

            if parent_node.type == NodeType.DIRECTORY and child_node.type == NodeType.DIRECTORY:
                parent_pos = layout_result.positions[parent_path]
                child_pos = layout_result.positions[child_path]

                parent_cx = parent_pos.x + parent_pos.width / 2
                child_cx = child_pos.x + child_pos.width / 2
                parent_min_z = parent_pos.z  # Back face
                child_max_z = child_pos.z + child_pos.depth  # Front face

                start = np.array([parent_cx, 0.05, parent_min_z], dtype=np.float32)
                end = np.array([child_cx, 0.05, child_max_z], dtype=np.float32)

                if abs(parent_cx - child_cx) < 0.01:
                    polyline = [start, end]
                else:
                    mid_z = (parent_min_z + child_max_z) / 2.0
                    elbow1 = np.array([parent_cx, 0.05, mid_z], dtype=np.float32)
                    elbow2 = np.array([child_cx, 0.05, mid_z], dtype=np.float32)
                    polyline = [start, elbow1, elbow2, end]

                self._connections.append(polyline)
                self._connection_metadata.append((parent_path, child_path))

    def _interpolate_polyline(self, polyline: list[np.ndarray], t: float) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate position and direction along a polyline at parameter t in [0,1]."""
        if len(polyline) < 2:
            return polyline[0], np.array([0, 0, 1], dtype=np.float32)

        # Calculate cumulative segment lengths
        segments = []
        total_length = 0.0
        for j in range(len(polyline) - 1):
            seg_len = float(np.linalg.norm(polyline[j + 1] - polyline[j]))
            segments.append(seg_len)
            total_length += seg_len

        if total_length < 1e-6:
            return polyline[0], np.array([0, 0, 1], dtype=np.float32)

        target_dist = t * total_length
        accumulated = 0.0
        for j, seg_len in enumerate(segments):
            if accumulated + seg_len >= target_dist or j == len(segments) - 1:
                local_t = (target_dist - accumulated) / seg_len if seg_len > 1e-6 else 0.0
                local_t = max(0.0, min(1.0, local_t))
                pos = polyline[j] + (polyline[j + 1] - polyline[j]) * local_t
                direction = polyline[j + 1] - polyline[j]
                norm = float(np.linalg.norm(direction))
                if norm > 1e-6:
                    direction = direction / norm
                return pos, direction
            accumulated += seg_len

        return polyline[-1], np.array([0, 0, 1], dtype=np.float32)

    def _polyline_length(self, polyline: list[np.ndarray]) -> float:
        """Calculate total length of a polyline."""
        total = 0.0
        for j in range(len(polyline) - 1):
            total += float(np.linalg.norm(polyline[j + 1] - polyline[j]))
        return total

    # Properties for compatibility with existing code

    @property
    def cube_geometry(self):
        """Compatibility property - returns self for instance_count access."""
        return self

    @property
    def instance_count(self) -> int:
        """Get number of cube instances."""
        return len(self._cubes)

    @property
    def ctx(self):
        """Compatibility property - returns True when initialized."""
        return self._initialized

    @property
    def instanced_vao(self):
        """Compatibility property."""
        return self._initialized

    @property
    def instanced_program(self):
        """Compatibility property."""
        return self._initialized

    # Tooltip setter

    def set_input_handler(self, handler) -> None:
        """Set the input handler for continuous updates.

        Args:
            handler: InputHandler instance
        """
        self._input_handler = handler

    def set_tooltip(self, tooltip_widget) -> None:
        """Set the tooltip widget reference for hover functionality.
        
        Args:
            tooltip_widget: The tooltip widget to display hover information
        """
        self._tooltip = tooltip_widget

    # Selection methods

    def get_node_at_position(self, x: int, y: int) -> Node | None:
        """Get the node at the given screen position using ray casting.

        Args:
            x: Screen X coordinate
            y: Screen Y coordinate

        Returns:
            Node at position or None
        """
        return self.raycast_find_node(x, y, self._nodes, self._positions)

    def raycast_find_node(self, screen_x: int, screen_y: int,
                         nodes: dict[str, Node],
                         positions: dict[str, object]) -> Node | None:
        """Find node at screen position using ray-AABB intersection.

        Args:
            screen_x: Screen X coordinate
            screen_y: Screen Y coordinate
            nodes: Dictionary mapping path strings to Node objects
            positions: Dictionary mapping path strings to Position objects

        Returns:
            Node at position or None
        """
        if not positions:
            return None

        # Get ray from camera
        width, height = self.width(), self.height()
        ray_dir = self.camera.get_ray_direction(screen_x, screen_y, width, height)
        ray_origin = self.camera.state.position

        # Find closest intersection
        closest_node = None
        closest_t = float('inf')

        for path_str, position in positions.items():
            if path_str not in nodes:
                continue

            node = nodes[path_str]

            # Get AABB bounds - match actual rendered positions (Grow Up model)
            if node.is_directory:
                # Platform: rendered from y=0 upwards
                min_y = 0.0
                max_y = position.height
                
                # Platform width/depth from layout position
                min_x = position.x
                max_x = position.x + position.width
                min_z = position.z
                max_z = position.z + position.depth
                
                min_bounds = np.array([min_x, min_y, min_z], dtype=np.float32)
                max_bounds = np.array([max_x, max_y, max_z], dtype=np.float32)

            else:
                # File: rendered on top of parent pedestal
                height = position.height
                
                # The layout position already sits on the parent pedestal
                min_y = position.y
                max_y = position.y + height
                
                # File width/depth (scaled by 0.8 as in load_layout)
                width_scaled = position.width * 0.8
                depth_scaled = position.depth * 0.8

                min_x = position.x + (position.width - width_scaled) / 2
                max_x = min_x + width_scaled
                min_z = position.z + (position.depth - depth_scaled) / 2
                max_z = min_z + depth_scaled

                min_bounds = np.array([min_x, min_y, min_z], dtype=np.float32)
                max_bounds = np.array([max_x, max_y, max_z], dtype=np.float32)

            # Ray-AABB intersection
            t = self._ray_aabb_intersect(ray_origin, ray_dir, min_bounds, max_bounds)
            if t is not None and t < closest_t:
                closest_t = t
                closest_node = node

        return closest_node

    def pick_connection(self, screen_x: int, screen_y: int) -> str | None:
        """Find the wire under a screen position.

        Wires lie on the ground, so the pick ray is intersected with the
        ground plane and compared against every wire segment.

        Returns:
            Path of the directory the wire leads to, or None
        """
        if not self._connections:
            return None

        ray_dir = self.camera.get_ray_direction(screen_x, screen_y, self.width(), self.height())
        origin = self.camera.state.position
        if ray_dir[1] >= -1e-6:
            return None  # Looking at the sky
        t = (0.05 - float(origin[1])) / float(ray_dir[1])
        if t <= 0:
            return None
        hit_x = float(origin[0]) + float(ray_dir[0]) * t
        hit_z = float(origin[2]) + float(ray_dir[2]) * t

        # Thin lines are hard to hit from far away: widen the tolerance with distance
        tolerance = max(0.25, t * 0.012)

        best_path = None
        best_dist = tolerance
        for polyline, (_parent, child_path) in zip(self._connections, self._connection_metadata):
            for a, b in zip(polyline, polyline[1:]):
                ax, az = float(a[0]), float(a[2])
                dx, dz = float(b[0]) - ax, float(b[2]) - az
                length_sq = dx * dx + dz * dz
                if length_sq < 1e-9:
                    continue
                u = ((hit_x - ax) * dx + (hit_z - az) * dz) / length_sq
                u = max(0.0, min(1.0, u))
                dist = math.hypot(hit_x - (ax + dx * u), hit_z - (az + dz * u))
                # Siblings share the bus segment; on a tie prefer the last
                # segment, which belongs to one directory only
                if dist < best_dist:
                    best_dist = dist
                    best_path = child_path
        return best_path

    def _ray_aabb_intersect(self, ray_origin: np.ndarray, ray_dir: np.ndarray,
                           min_bounds: np.ndarray, max_bounds: np.ndarray) -> float | None:
        """Compute ray-AABB intersection using slab method.

        Args:
            ray_origin: Ray origin point
            ray_dir: Normalized ray direction
            min_bounds: AABB minimum bounds
            max_bounds: AABB maximum bounds

        Returns:
            Distance to intersection or None if no intersection
        """
        # Avoid division by zero
        epsilon = 1e-6

        t_min = 0.0  # Near plane
        t_max = float('inf')

        for i in range(3):
            if abs(ray_dir[i]) < epsilon:
                # Ray is parallel to this slab
                if ray_origin[i] < min_bounds[i] or ray_origin[i] > max_bounds[i]:
                    return None
            else:
                inv_d = 1.0 / ray_dir[i]
                t1 = (min_bounds[i] - ray_origin[i]) * inv_d
                t2 = (max_bounds[i] - ray_origin[i]) * inv_d

                if t1 > t2:
                    t1, t2 = t2, t1

                t_min = max(t_min, t1)
                t_max = min(t_max, t2)

                if t_min > t_max:
                    return None

        return t_min if t_min > 0 else None

    def snap_camera_to_node(self, node_id: int, distance: float | None = None) -> None:
        """Snap camera to focus on a node.

        Args:
            node_id: ID of the node to focus on
            distance: Optional camera distance (auto-calculated if None)
        """
        # Find node by ID
        target_node = None
        target_path = None

        for path_str, node in self._nodes.items():
            if id(node) == node_id:
                target_node = node
                target_path = path_str
                break

        if target_node is None or target_path not in self._positions:
            return

        position = self._positions[target_path]

        # Calculate bbox center, accounting for actual rendered Y for files
        block_top = None
        center = np.array(position.center, dtype=np.float32)
        if not target_node.is_directory:
            block_top = position.y + position.height

        # Determine distance based on node type
        if distance is None:
            if target_node.is_directory:
                # Distance for directories (platforms) - larger view
                max_dim = max(position.width, position.depth)
                distance = max_dim * 2.5 + 10.0
            else:
                # Distance for files - closer view
                max_dim = max(position.width, position.height, position.depth)
                distance = max_dim * 3.0 + 5.0

        # For files, calculate camera end position with proper elevation above block top
        end_pos = None
        if block_top is not None:
            current_pos = self.camera.state.position.copy()

            # Direction from file center to current camera (keeps camera on same side)
            horiz = current_pos - center
            horiz[1] = 0.0
            horiz_norm = np.linalg.norm(horiz)
            if horiz_norm > 0.001:
                horiz = horiz / horiz_norm
            else:
                horiz = np.array([0.0, 0.0, 1.0], dtype=np.float32)

            # Place camera at ~30-degree elevation (sin30=0.5, cos30=0.866)
            end_pos = center + horiz * (distance * 0.866)
            end_pos[1] = center[1] + distance * 0.5

            # Ensure camera stays above block top with clearance
            min_cam_y = block_top + 2.0
            if end_pos[1] < min_cam_y:
                end_pos[1] = min_cam_y

        # Directory navigation: cinematic flythrough (800ms)
        # File selection: quick snap (300ms)
        is_directory = target_node.is_directory
        duration = 800 if is_directory else 300
        self._animate_camera_to(center, distance, end_pos=end_pos, duration_ms=duration,
                                cinematic=is_directory)

    def _animate_camera_to(self, target: np.ndarray, distance: float,
                          end_pos: np.ndarray | None = None,
                          duration_ms: int = 300,
                          cinematic: bool = False) -> None:
        """Animate camera to look at a target position.

        Args:
            target: Target position to look at
            distance: Distance from target
            end_pos: Optional explicit camera end position (auto-calculated if None)
            duration_ms: Animation duration in milliseconds
            cinematic: Use cinematic flythrough with arc trajectory
        """
        # Store animation state
        self._animation_start_time = time.time()
        self._animation_duration = duration_ms / 1000.0
        self._animation_start_pos = self.camera.state.position.copy()
        self._animation_end_target = target

        # In Fly mode, state.target is just eye+front (distance=1), which makes
        # target interpolation erratic.  Synthesise a start_target at the same
        # distance as end_target so the "look-at" sweeps smoothly.
        from pyfsn.view.camera import CameraMode
        if self.camera.mode == CameraMode.FLY:
            end_dist = float(np.linalg.norm(target - self._animation_start_pos))
            if end_dist < 0.01:
                end_dist = 1.0
            front = self.camera.state.target - self._animation_start_pos
            front_norm = float(np.linalg.norm(front))
            if front_norm > 1e-6:
                front = front / front_norm
            else:
                front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            self._animation_start_target = self._animation_start_pos + front * end_dist
        else:
            self._animation_start_target = self.camera.state.target.copy()

        if end_pos is not None:
            self._animation_end_pos = end_pos
        else:
            # Calculate end position based on current direction
            direction = self._animation_start_pos - self._animation_start_target
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            else:
                direction = np.array([1.0, 1.0, 1.0], dtype=np.float32)
                direction = direction / np.linalg.norm(direction)

            self._animation_end_pos = target + direction * distance

        # Build waypoints for cinematic flythrough
        self._animation_cinematic = cinematic
        if cinematic:
            self._build_flythrough_waypoints()

        # Enable animation flag
        self._is_animating = True

        # Ensure timer is running
        if not self._update_timer.isActive():
            self._update_timer.start(16)

    def _build_flythrough_waypoints(self) -> None:
        """Build control point for cinematic camera flythrough arc.

        Uses a quadratic Bezier curve with a single elevated control point
        at the midpoint for a smooth arc trajectory without segment boundaries.
        """
        start = self._animation_start_pos.copy()
        end = self._animation_end_pos.copy()
        travel_dist = np.linalg.norm(end - start)

        # Control point: midpoint elevated above the higher endpoint
        control = (start + end) * 0.5
        arc_height = min(travel_dist * 0.15, 8.0)
        control[1] = max(start[1], end[1]) + arc_height

        self._animation_control_point = control

    def _on_timer(self) -> None:
        """Timer callback for animation."""
        # Process input updates (Fly Mode movement)
        if self._input_handler:
            self._input_handler.update()

        # Update animation time for shader effects
        self._animation_time += 0.016  # ~60 FPS

        # Handle camera animation
        if hasattr(self, '_is_animating') and self._is_animating:
            elapsed = time.time() - self._animation_start_time
            progress = min(1.0, elapsed / self._animation_duration)

            # Global ease-in-out
            t = progress
            ease = t * t * (3.0 - 2.0 * t)

            # Calculate camera position
            if hasattr(self, '_animation_cinematic') and self._animation_cinematic and hasattr(self, '_animation_control_point'):
                # Quadratic Bezier: P(t) = (1-t)²·start + 2(1-t)t·control + t²·end
                s = self._animation_start_pos
                c = self._animation_control_point
                e = self._animation_end_pos
                inv = 1.0 - ease
                new_pos = inv * inv * s + 2.0 * inv * ease * c + ease * ease * e
            else:
                new_pos = (1 - ease) * self._animation_start_pos + ease * self._animation_end_pos

            # Target always interpolates linearly
            new_target = (1 - ease) * self._animation_start_target + ease * self._animation_end_target

            self.camera.set_position_target(new_pos, new_target)

            # Check if animation complete
            if progress >= 1.0:
                self._is_animating = False
                self.node_focused.emit(self._get_node_at_target(self._animation_end_target))

        self.update()

    def _get_node_at_target(self, target: np.ndarray) -> Node | None:
        """Get node closest to target position.

        Args:
            target: Target position

        Returns:
            Closest node or None
        """
        closest_node = None
        closest_dist = float('inf')

        for path_str, position in self._positions.items():
            if path_str not in self._nodes:
                continue

            center = np.array([
                position.x + position.width / 2,
                position.y + position.height / 2,
                position.z + position.depth / 2,
            ], dtype=np.float32)

            dist = np.linalg.norm(center - target)
            if dist < closest_dist:
                closest_dist = dist
                closest_node = self._nodes[path_str]

        return closest_node

    def set_selection(self, selected_paths: set[str], nodes: dict[str, Node]) -> None:
        """Set the current selection.

        Args:
            selected_paths: Set of selected node paths
            nodes: Dictionary mapping path strings to Node objects
        """
        self._selected_paths = selected_paths.copy()

        # Update colors for all cubes
        for path, idx in self._node_to_cube.items():
            if path in self._selected_paths:
                color = self._colors["selected"].copy()
                color[3] = 0.8
                self._cubes[idx].color = color
            else:
                # Reset to default color based on color mode
                node = nodes.get(path)
                if node:
                    self._cubes[idx].color = self._file_base_color(node)

        # Update colors for all platforms
        for path, idx in self._node_to_platform.items():
            if path in self._selected_paths:
                self._platforms[idx].color = self._colors["selected"].copy()
            else:
                self._platforms[idx].color = self._platform_base_color(path, nodes.get(path))

        # PR-4: Calculate which connections should be highlighted
        # Like in fsn, the line *leading to* a selected directory brightens
        self._selected_connections.clear()
        for i, (_parent_path, child_path) in enumerate(self._connection_metadata):
            if child_path in self._selected_paths:
                self._selected_connections.add(i)

        self._batches = None
        self.selection_changed.emit(self._selected_paths)
        self.update()

    def set_color_mode(self, mode: ColorMode) -> None:
        """Set the color mode for file visualization.

        Args:
            mode: ColorMode.AGE for age-based colors, ColorMode.TYPE for type-based
        """
        if self._color_mode == mode:
            return

        self._color_mode = mode

        # Update all file cube colors
        for path, idx in self._node_to_cube.items():
            if path in self._selected_paths:
                continue  # Keep selection color
            
            node = self._nodes.get(path)
            if node:
                self._cubes[idx].color = self._file_base_color(node)

        self._batches = None
        self.update()

    @property
    def color_mode(self) -> ColorMode:
        """Get current color mode."""
        return self._color_mode

    def select_node(self, path: str) -> None:
        """Select a node by path.

        Args:
            path: Path string of node to select
        """
        self.set_selection({path}, self._nodes)

    def clear_selection(self) -> None:
        """Clear all selections."""
        for path in self._selected_paths:
            # Handle file cubes
            if path in self._node_to_cube:
                idx = self._node_to_cube[path]
                node = self._nodes.get(path)
                if node:
                    self._cubes[idx].color = self._file_base_color(node)
            # Handle directory platforms
            elif path in self._node_to_platform:
                idx = self._node_to_platform[path]
                self._platforms[idx].color = self._platform_base_color(path, self._nodes.get(path))
        self._batches = None
        self._selected_paths.clear()
        self._selected_connections.clear()  # PR-4: Clear highlighted connections
        self.selection_changed.emit(self._selected_paths)
        self.update()

    def focus_node(self, path: str) -> None:
        """Focus camera on a node.

        Args:
            path: Path string of node to focus
        """
        if path in self._positions:
            pos = self._positions[path]
            center = np.array([
                pos.x + pos.width / 2,
                pos.y + pos.height / 2,
                pos.z + pos.depth / 2,
            ], dtype=np.float32)
            self.camera._state.target = center
            self._focused_path = path
            self.update()

    @property
    def performance_stats(self) -> dict:
        """Get performance statistics.

        Returns:
            Dictionary with FPS, frame time, etc.
        """
        return {
            "fps": self._performance_monitor.metrics.fps,
            "frame_time_ms": self._performance_monitor.metrics.frame_time_ms,
            "instance_count": len(self._cubes),
            "draw_calls": len(self._cubes),
        }

    def set_root_path(self, path) -> None:
        """Set the root path for visualization.

        Args:
            path: Root path (Path object)
        """
        # Store for reference - actual loading done via load_layout
        self._root_path = path

    def set_camera_mode(self, mode) -> None:
        """Set camera navigation mode.

        Args:
            mode: CameraMode enum value
        """
        self.camera.set_mode(mode)

    # Spotlight Search API (Phase 3.2)

    def start_spotlight_search(self, query: str, matching_nodes: set[Node]) -> None:
        """Start a spotlight search with visualization.

        Args:
            query: Search query string
            matching_nodes: Set of nodes that match the search
        """
        self._spotlight.start_search(query, matching_nodes)
        self.update()

    def update_spotlight_results(self, matching_nodes: set[Node]) -> None:
        """Update spotlight search results.

        Args:
            matching_nodes: Set of nodes that match the search
        """
        self._spotlight.update_results(matching_nodes)
        self.update()

    def clear_spotlight_search(self) -> None:
        """Clear the spotlight search visualization."""
        self._spotlight.clear_search()
        self.update()

    @property
    def spotlight(self) -> SpotlightSearch:
        """Get the spotlight search manager."""
        return self._spotlight

    @property
    def is_spotlight_active(self) -> bool:
        """Check if spotlight search is currently active."""
        return self._spotlight.is_active

    def get_screen_position(self, world_pos: np.ndarray) -> tuple[int, int] | None:
        """Convert world position to screen coordinates.

        Args:
            world_pos: World position as [x, y, z] array

        Returns:
            Screen (x, y) tuple or None if behind camera
        """
        # Get camera matrices
        pos = self.camera.state.position
        target = self.camera.state.target
        up = self.camera.state.up

        # View direction
        forward = target - pos
        forward = forward / np.linalg.norm(forward)

        # Check if point is in front of camera
        to_point = world_pos - pos
        if np.dot(forward, to_point) < 0:
            return None

        # Simple projection (Legacy OpenGL compatible)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up_corrected = np.cross(right, forward)

        # Project to view space
        view_x = np.dot(to_point, right)
        view_y = np.dot(to_point, up_corrected)
        view_z = np.dot(to_point, forward)

        if view_z <= 0:
            return None

        # Apply perspective
        # gluPerspective uses vertical FOV
        aspect = self.width() / max(1, self.height())
        fov_rad = math.radians(self.camera.state.fov)
        scale = 1.0 / math.tan(fov_rad / 2.0)

        # NDC calculation
        # X: needs to be scaled by (scale / aspect)
        # Y: needs to be scaled by (scale)
        ndc_x = (view_x / view_z) * (scale / aspect)
        ndc_y = (view_y / view_z) * scale

        # Convert to screen coords
        screen_x = int((ndc_x + 1.0) * 0.5 * self.width())
        screen_y = int((1.0 - ndc_y) * 0.5 * self.height())

        return (screen_x, screen_y)

    def is_node_visible(self, path: str) -> bool:
        """Check if a node is visible within the camera frustum (PR-5).

        This is used to optimize label/tooltip rendering by only
        updating text overlays for visible nodes.

        Args:
            path: Path string of the node to check

        Returns:
            True if the node is visible within the frustum
        """
        if path not in self._positions:
            return False

        position = self._positions[path]

        # Get bounding box from Position object
        min_bounds = np.array([position.min_x, position.min_y, position.min_z], dtype=np.float32)
        max_bounds = np.array([position.max_x, position.max_y, position.max_z], dtype=np.float32)

        # Use frustum culler to check visibility
        return self._frustum_culler.is_box_visible(min_bounds, max_bounds)

    # Directory label rendering (SGI fsn style: text on platforms)

    def _create_text_texture(self, text: str, font_size: int = 32) -> tuple[int, int, int]:
        """Create a texture from text using QPainter.

        Args:
            text: Text to render
            font_size: Font size in pixels

        Returns:
            Tuple of (texture_id, width, height)
        """
        # Check cache first
        cache_key = f"{text}:{font_size}"
        if cache_key in self._text_textures:
            # Return cached texture with stored dimensions
            return self._text_textures[cache_key]

        # Use handwriting-style font (SGI fsn style)
        # Try different fonts based on platform
        font_families = ["Comic Sans MS", "Bradley Hand", "Papyrus", "Brush Script MT", "Arial"]
        font = QFont(font_families[0], font_size)
        font.setBold(True)

        # Create image for text
        metrics = QFontMetrics(font)
        text_width = metrics.horizontalAdvance(text)
        text_height = metrics.height()

        # Create image with padding
        padding = 4
        img_width = text_width + padding * 2
        img_height = text_height + padding * 2

        # Ensure minimum size
        img_width = max(img_width, 32)
        img_height = max(img_height, 16)

        image = QImage(img_width, img_height, QImage.Format.Format_RGBA8888)
        image.fill(QColor(0, 0, 0, 0))  # Transparent background

        # Draw text
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(font)

        # Handwritten style: white/tan color with slight shadow
        painter.setPen(QColor(255, 248, 220))  # Warm white/cream (SGI fsn style)
        painter.drawText(padding, padding + metrics.ascent(), text)

        painter.end()

        # Convert to bytes
        ptr = image.bits()
        if ptr is not None:
            # Calculate size in bytes (PyQt6 compatible)
            byte_count = img_width * img_height * 4  # RGBA = 4 bytes per pixel
            data = ptr.asstring(byte_count)
        else:
            # Fallback: create empty data
            data = bytes(byte_count)

        # Create OpenGL texture
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img_width, img_height,
                     0, GL_BGRA, GL_UNSIGNED_BYTE, data)

        # Cache the texture with dimensions
        self._text_textures[cache_key] = (texture_id, img_width, img_height)
        return texture_id, img_width, img_height

    def _draw_directory_labels(self) -> None:
        """Draw directory names on the ground next to platforms (SGI fsn style)."""
        if not self._platforms or not self._nodes:
            return

        # Label quads only change when the layout changes
        if self._label_cache is None:
            self._label_cache = self._build_label_cache()

        # Enable blending for text
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Enable texturing
        glEnable(GL_TEXTURE_2D)

        # Disable depth write for text (so it doesn't interfere)
        glDepthMask(GL_FALSE)

        # Save current matrices
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()

        # Draw each label at its calculated position
        for texture_id, label_x, label_z, quad_width, quad_height in self._label_cache:
            # Position text on the ground, just above the grid lines
            text_y = 0.02

            # Save current matrix
            glPushMatrix()

            # Translate to ground position
            glTranslatef(label_x, text_y, label_z)

            # Scale to appropriate size
            glScalef(quad_width, 1.0, quad_height)

            # Bind texture and draw quad
            glBindTexture(GL_TEXTURE_2D, texture_id)
            glColor4f(1.0, 1.0, 1.0, 1.0)  # White with full alpha

            # Draw quad centered at origin (after scale)
            glBegin(GL_QUADS)
            # Texture coords: U and V both corrected for +Z camera direction
            glTexCoord2f(0, 0)
            glVertex3f(-0.5, 0, -0.5)
            glTexCoord2f(1, 0)
            glVertex3f(0.5, 0, -0.5)
            glTexCoord2f(1, 1)
            glVertex3f(0.5, 0, 0.5)
            glTexCoord2f(0, 1)
            glVertex3f(-0.5, 0, 0.5)
            glEnd()

            glPopMatrix()

        # Restore depth write
        glDepthMask(GL_TRUE)

        # Restore matrices
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

        # Disable texturing and blending
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)

    def check_collision(self, position: np.ndarray, radius: float = 0.5) -> bool:
        """Check if a position collides with any geometry.
        
        Args:
            position: Position to check [x, y, z]
            radius: Radius of the object (camera)
            
        Returns:
            True if collision detected
        """
        # 1. Ground collision
        # Ground plane is at y=-0.5; keep camera above ground surface + radius
        if position[1] < -0.5 + radius:
            return True
            
        # 2. Platform collision (AABB)
        # Only check if we are low enough to hit platforms
        # Platforms are usually thin, maybe y=0 to y=1?
        # Let's check platform scale from typical layout
        
        # Helper for AABB overlap
        def check_aabb(pos, box_pos, box_scale, margin):
            # Box bounds
            box_min = box_pos - box_scale * 0.5 - margin
            box_max = box_pos + box_scale * 0.5 + margin
            
            # Point vs Box
            return (pos[0] >= box_min[0] and pos[0] <= box_max[0] and
                    pos[1] >= box_min[1] and pos[1] <= box_max[1] and
                    pos[2] >= box_min[2] and pos[2] <= box_max[2])

        # Check platforms
        margin = radius
        for platform in self._platforms:
            if check_aabb(position, platform.position, platform.scale, margin):
                return True
                
        # Check files (only if close to ground/platforms)
        # Optimize: only check if y is within typical file height range
        if position[1] < 20.0:  # Optimization
            for cube in self._cubes:
                if check_aabb(position, cube.position, cube.scale, margin):
                    return True
                    
        return False

    # Geometry of the directory name written on the ground in front of a pedestal
    LABEL_HEIGHT = 0.9   # Preferred text height (world units)
    LABEL_FRONT_GAP = 0.25  # Distance from the pedestal's front edge
    LABEL_WIRE_GAP = 0.4    # Distance from the wire that enters the pedestal's centre
    LABEL_REACH = 1.8       # How far past its own strip a label may extend (see below)

    def _build_label_cache(self) -> list[tuple[int, float, float, float, float]]:
        """Compute the ground quad for every directory name.

        The name is written in front of the pedestal, to the right of the wire
        that leads to it.  It is shrunk to fit the ground strip owned by the
        directory's sub-tree, so labels never overlap each other or a pedestal
        and no collision search is needed.

        Returns:
            List of (texture_id, center_x, center_z, width, height)
        """
        labels: list[tuple[int, float, float, float, float]] = []
        for path_str in self._node_to_platform:
            node = self._nodes.get(path_str)
            position = self._positions.get(path_str)
            if node is None or position is None or not node.name:
                continue

            texture_id, tex_width, tex_height = self._create_text_texture(node.name, font_size=28)
            aspect = tex_width / max(tex_height, 1)

            # Free ground to the right of the wire: the rest of this strip plus
            # most of the gap to the next strip. Staying out of the next strip
            # matters: a cousin there can sit at a different depth.
            slot = self._slot_widths.get(path_str, position.width)
            max_width = max(0.6, slot / 2 + self.LABEL_REACH - self.LABEL_WIRE_GAP)
            # Bigger pedestals get bigger names, but stay clear of the wire bus
            quad_height = min(max(self.LABEL_HEIGHT, position.depth * 0.3), 1.8)
            quad_width = quad_height * aspect
            if quad_width > max_width:
                quad_width = max_width
                quad_height = quad_width / aspect

            center_x = position.x + position.width / 2 + self.LABEL_WIRE_GAP + quad_width / 2
            center_z = position.z + position.depth + self.LABEL_FRONT_GAP + quad_height / 2
            labels.append((texture_id, center_x, center_z, quad_width, quad_height))

        return labels

    def _update_ground_extent(self, positions: dict) -> None:
        """Make the ground plane cover the whole layout (plus a wide margin)."""
        if not positions:
            self._ground_extent = (0.0, 0.0, 500.0)
            self._scene_radius = 50.0
            return
        min_x = min(p.x for p in positions.values())
        max_x = max(p.x + p.width for p in positions.values())
        min_z = min(p.z for p in positions.values())
        max_z = max(p.z + p.depth for p in positions.values())
        self._scene_radius = math.hypot(max_x - min_x, max_z - min_z) / 2
        half = max(max_x - min_x, max_z - min_z) * 1.5 + 400.0
        self._ground_extent = ((min_x + max_x) / 2, (min_z + max_z) / 2, max(500.0, half))
        # Let the orbit camera pull back far enough to see all of it
        self.camera.max_orbit_distance = max(500.0, half * 1.5)

    # Bloom control methods

    def set_bloom_enabled(self, enabled: bool) -> None:
        """Enable or disable bloom effect.

        Args:
            enabled: True to enable bloom, False to disable
        """
        self._bloom.set_enabled(enabled)
        self.update()

    def is_bloom_enabled(self) -> bool:
        """Check if bloom effect is enabled.

        Returns:
            True if bloom is enabled
        """
        return self._bloom.enabled

    def set_bloom_intensity(self, intensity: float) -> None:
        """Set bloom intensity.

        Args:
            intensity: Bloom intensity (0.0 - 1.0)
        """
        self._bloom.intensity = max(0.0, min(1.0, intensity))
        self.update()

    def get_bloom_intensity(self) -> float:
        """Get current bloom intensity.

        Returns:
            Current bloom intensity
        """
        return self._bloom.intensity

    @property
    def animation_time(self) -> float:
        """Get current animation time for shader effects."""
        return self._animation_time
