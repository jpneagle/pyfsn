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


@dataclass
class CubeInstance:
    """Single cube instance data."""
    position: np.ndarray  # [x, y, z]
    scale: np.ndarray     # [w, h, d]
    color: np.ndarray     # [r, g, b, a]
    shininess: float = 30.0  # Specular shininess (0-128), default for file cubes
    emission: float = 0.0    # Emission intensity for glow effect (0.0 - 1.0+)


class Renderer(QOpenGLWidget):
    """OpenGL widget for rendering 3D file system visualization."""

    # Signals
    node_clicked = pyqtSignal(Node, bool)  # Node, is_double_click
    node_focused = pyqtSignal(Node)
    selection_changed = pyqtSignal(set)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)

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
        self._connections: list[tuple[np.ndarray, np.ndarray]] = []  # Line segments
        self._connection_metadata: list[tuple[str, str]] = []  # (parent_path, child_path) for each connection (PR-4)
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

        # Colors for different node types
        self._colors = {
            "directory": np.array([0.2, 0.6, 1.0, 1.0], dtype=np.float32),  # Bright blue (platform)
            "file": np.array([0.8, 0.8, 0.8, 1.0], dtype=np.float32),
            "symlink": np.array([0.3, 1.0, 0.3, 1.0], dtype=np.float32),
            "selected": np.array([1.0, 1.0, 0.3, 1.0], dtype=np.float32),
            "focused": np.array([1.0, 0.8, 0.0, 1.0], dtype=np.float32),
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

        # Input handler
        self._input_handler = None

        # OpenGL state
        self._initialized = False

    def _calculate_age_color(self, mtime: float) -> np.ndarray:
        """Calculate color based on file age (SGI fsn style).

        Args:
            mtime: File modification time (Unix timestamp)

        Returns:
            RGBA color array
        """
        now = time.time()
        age_days = (now - mtime) / 86400.0  # Seconds to days

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

    def _calculate_emission(self, node: Node) -> float:
        """Calculate emission intensity based on file type and git status.

        Args:
            node: Node to calculate emission for

        Returns:
            Emission intensity (0.0 - 1.0)
        """
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

    def _calculate_pedestal_height(self, node: Node) -> float:
        """Calculate pedestal height based on directory content size (FSN style).

        In original FSN, pedestal height = total file sizes in directory.
        Uses logarithmic scale for visual clarity.
        """
        total_size = node.total_size
        if total_size <= 0:
            return 0.3  # Minimum height for empty directories

        # Logarithmic scale: 1KB ≈ 3, 1MB ≈ 6, 1GB ≈ 9
        log_size = math.log10(max(total_size, 1))
        # Scale: 1KB -> ~0.9, 1MB -> ~1.5, 1GB -> ~2.1
        height = 0.3 + log_size * 0.2
        return min(max(height, 0.3), 3.0)

    def initializeGL(self) -> None:
        """Initialize OpenGL resources."""
        glClearColor(0.1, 0.1, 0.15, 1.0)
        glEnable(GL_DEPTH_TEST)
        
        # Disable lighting completely to preserve colors
        glDisable(GL_LIGHTING)

        # Set up depth fog (SGI fsn style)
        self._setup_fog()

        self._initialized = True

    def _setup_fog(self) -> None:
        """Set up depth fog for distance fade effect."""
        glEnable(GL_FOG)
        glFogi(GL_FOG_MODE, GL_LINEAR)
        glFogfv(GL_FOG_COLOR, [0.1, 0.1, 0.15, 1.0])  # Match background color
        glFogf(GL_FOG_START, 100.0)
        glFogf(GL_FOG_END, 500.0)
        glHint(GL_FOG_HINT, GL_NICEST)

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

        # Update frustum culling (PR-5: Optimization)
        aspect = self.width() / max(1, self.height())
        view_matrix = self.camera.view_matrix
        proj_matrix = self.camera.projection_matrix(aspect)
        self._frustum_culler.update_from_camera(view_matrix, proj_matrix, aspect)

        # Draw sky gradient first (fullscreen background)
        self._draw_sky_gradient()

        glLoadIdentity()

        # Apply camera view
        pos = self.camera.state.position
        target = self.camera.state.target
        up = self.camera.state.up
        gluLookAt(pos[0], pos[1], pos[2],
                  target[0], target[1], target[2],
                  up[0], up[1], up[2])

        # Draw ground in 3D space
        self._draw_ground_grid()

        # Draw connections
        self._draw_connections()

        # Draw platforms first (opaque) - with spotlight effects
        for platform in self._platforms:
            self._draw_cube_with_spotlight(platform)
            self._draw_cube_edges(platform)

        # Draw directory labels on platforms (SGI fsn style)
        self._draw_directory_labels()

        # Render all cubes (files) - with spotlight effects
        for cube in self._cubes:
            self._draw_cube_with_spotlight(cube)
            self._draw_cube_edges(cube)

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

        ground_size = 500.0
        ground_y = -0.01  # Just below y=0 labels

        # Ground plane (Rich green gradient, FSN grass-like)
        glBegin(GL_QUADS)
        glColor4f(0.25, 0.5, 0.25, 1.0)  # Near green
        glVertex3f(-ground_size, ground_y, -ground_size)
        glVertex3f(ground_size, ground_y, -ground_size)
        glColor4f(0.18, 0.4, 0.18, 1.0)  # Far green (gradient)
        glVertex3f(ground_size, ground_y, ground_size)
        glVertex3f(-ground_size, ground_y, ground_size)
        glEnd()

        # Grid lines - more subtle and sparser
        glLineWidth(1.0)
        glColor4f(0.28, 0.48, 0.28, 0.25)  # Very subtle grid
        grid_spacing = 40.0
        grid_range = 10

        glBegin(GL_LINES)
        for i in range(-grid_range, grid_range + 1):
            offset = i * grid_spacing
            # Lines parallel to X-axis
            glVertex3f(-ground_size, ground_y + 0.01, offset)
            glVertex3f(ground_size, ground_y + 0.01, offset)
            # Lines parallel to Z-axis
            glVertex3f(offset, ground_y + 0.01, -ground_size)
            glVertex3f(offset, ground_y + 0.01, ground_size)
        glEnd()

        glDisable(GL_BLEND)

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
        
        # Draw fullscreen quad with gradient (FSN original style - bright sky)
        glBegin(GL_QUADS)
        # Bottom (horizon) - bright cyan/blue
        glColor4f(0.6, 0.8, 1.0, 1.0)
        glVertex3f(-1.0, -1.0, -0.999)
        glVertex3f(1.0, -1.0, -0.999)
        # Top - medium blue (not too dark)
        glColor4f(0.3, 0.5, 0.9, 1.0)
        glVertex3f(1.0, 1.0, -0.999)
        glVertex3f(-1.0, 1.0, -0.999)
        glEnd()
        
        # Restore matrices
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
        glEnable(GL_DEPTH_TEST)

    def _draw_connections(self) -> None:
        """Draw wire connections between platforms with highlighting and pulse effect (PR-4)."""
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

        glBegin(GL_LINES)
        for i, (start, end) in enumerate(self._connections):
            if i not in self._selected_connections:
                glVertex3fv(start)
                glVertex3fv(end)
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

            glBegin(GL_LINES)
            for i in self._selected_connections:
                if i < len(self._connections):
                    start, end = self._connections[i]
                    glVertex3fv(start)
                    glVertex3fv(end)
            glEnd()

            # Add animated glow effect for selected wires
            glLineWidth(8.0)

            # Cyberpunk glow color animation
            glow_pulse = (math.sin(self._animation_time * 3.0) * 0.5 + 0.5)
            glow_r = 0.0 + glow_pulse * 0.3
            glow_g = 1.0 - glow_pulse * 0.2
            glow_b = 1.0

            glColor4f(glow_r, glow_g, glow_b, 0.2 + glow_pulse * 0.2)

            glBegin(GL_LINES)
            for i in self._selected_connections:
                if i < len(self._connections):
                    start, end = self._connections[i]
                    glVertex3fv(start)
                    glVertex3fv(end)
            glEnd()

            # Add traveling pulse effect along the wire
            glLineWidth(2.0)
            travel_speed = 3.0
            travel_phase = (self._animation_time * travel_speed) % 1.0

            glColor4f(0.0, 1.0, 1.0, 0.6)  # Bright cyan

            # Draw small segments for traveling pulse
            for i in self._selected_connections:
                if i < len(self._connections):
                    start, end = self._connections[i]
                    direction = end - start
                    length = np.linalg.norm(direction)
                    if length > 0:
                        direction = direction / length
                        # Calculate pulse position along the wire
                        pulse_pos = start + direction * (length * travel_phase)

                        # Draw a small glowing segment at the pulse position
                        segment_length = min(length * 0.1, 2.0)
                        seg_start = pulse_pos - direction * segment_length * 0.5
                        seg_end = pulse_pos + direction * segment_length * 0.5

                        glBegin(GL_LINES)
                        glVertex3fv(seg_start)
                        glVertex3fv(seg_end)
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
            
            # Check if it's a platform
            elif path in self._node_to_platform:
                idx = self._node_to_platform[path]
                platform = self._platforms[idx]
                self._draw_spotlight(platform.position, height=12.0, radius=4.0)
        
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

    def _draw_cube(self, cube: CubeInstance) -> None:
        """Draw a single cube with emission-based glow and 3D face shading."""
        x, y, z = cube.position
        w, h, d = cube.scale
        r, g, b, a = cube.color

        # Apply emission-based glow effect
        r, g, b, a = self._bloom.apply_glow(r, g, b, a, cube.emission)

        # Per-face shading factors (simulate simple directional lighting)
        top_f = 1.0     # Top face: brightest
        front_f = 0.85  # Front face: slightly dimmer
        side_f = 0.7    # Side faces: medium
        back_f = 0.55   # Back face: darker
        bottom_f = 0.4  # Bottom face: darkest

        glPushMatrix()
        glTranslatef(x, y, z)
        glScalef(w, h, d)

        # Draw cube faces
        glBegin(GL_QUADS)
        # Front face
        glColor4f(r * front_f, g * front_f, b * front_f, a)
        glNormal3f(0, 0, 1)
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        # Back face
        glColor4f(r * back_f, g * back_f, b * back_f, a)
        glNormal3f(0, 0, -1)
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(-0.5, 0.5, -0.5)
        glVertex3f(0.5, 0.5, -0.5)
        # Top face
        glColor4f(r * top_f, g * top_f, b * top_f, a)
        glNormal3f(0, 1, 0)
        glVertex3f(-0.5, 0.5, 0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(0.5, 0.5, -0.5)
        glVertex3f(-0.5, 0.5, -0.5)
        # Bottom face
        glColor4f(r * bottom_f, g * bottom_f, b * bottom_f, a)
        glNormal3f(0, -1, 0)
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(-0.5, -0.5, 0.5)
        # Right face
        glColor4f(r * side_f, g * side_f, b * side_f, a)
        glNormal3f(1, 0, 0)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(0.5, 0.5, -0.5)
        glVertex3f(0.5, 0.5, 0.5)
        # Left face
        glColor4f(r * side_f, g * side_f, b * side_f, a)
        glNormal3f(-1, 0, 0)
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        glVertex3f(-0.5, 0.5, -0.5)
        glEnd()

        glPopMatrix()

    def _draw_cube_with_spotlight(self, cube: CubeInstance) -> None:
        """Draw a single cube with spotlight search effects applied.

        Args:
            cube: CubeInstance to draw
        """
        # Apply spotlight effects if active
        if self._spotlight.is_active:
            # Find the path for this cube
            path = self._find_path_for_cube(cube)
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

    def _find_path_for_cube(self, cube: CubeInstance) -> str | None:
        """Find the path string for a given cube instance.

        Args:
            cube: CubeInstance to find path for

        Returns:
            Path string or None
        """
        # Search in cubes
        for path, idx in self._path_to_cube_index.items():
            if idx < len(self._cubes) and self._cubes[idx] is cube:
                return path

        # Search in platforms
        for path, idx in self._path_to_platform_index.items():
            if idx < len(self._platforms) and self._platforms[idx] is cube:
                return path

        return None

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

        glLineWidth(1.5)

        # Dark edge color - computed from cube color
        r, g, b, a = cube.color
        
        # Check if this is a platform (pedestal)
        # Platforms have distinct edge behavior in FSN (brighter/outlined)
        is_platform = cube.scale[1] < 1.0 and cube.scale[0] > 2.0
        
        if is_platform:
            # Highlight edges for pedestals (bright blue/white tint)
            edge_color = [min(1.0, r * 1.3), min(1.0, g * 1.3), min(1.0, b * 1.3), a]
        else:
            # Standard dark edges for file cubes
            edge_color = [r * 0.3, g * 0.3, b * 0.3, a]
            
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

        selection = selection or set()

        for path_str, position in layout_result.positions.items():
            if path_str not in nodes:
                continue

            node = nodes[path_str]

            if node.type == NodeType.DIRECTORY:
                # Platforms (Directories) - Size calculated by LayoutEngine
                platform_w = position.width
                platform_d = position.depth
                
                # FSN style: pedestal height proportional to content size (Phase 2)
                height = self._calculate_pedestal_height(node)
                
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

                # Color: Blue for platform (directory color)
                if path_str in selection:
                    color = self._colors["selected"].copy()
                else:
                    color = self._colors["directory"].copy()

                # Calculate emission for platform
                emission = self._calculate_emission(node)

                platform = CubeInstance(position=pos_array, scale=scale_array, color=color, shininess=5.0, emission=emission)
                self._node_to_platform[path_str] = len(self._platforms)
                self._path_to_platform_index[path_str] = len(self._platforms)
                self._platforms.append(platform)
                
            else:
                # File blocks - height from layout engine (SGI fsn style)
                height = position.height

                # Calculate parent pedestal height to place file on top
                parent_height = 0.0
                if node.parent and self._calculate_pedestal_height:
                     parent_height = self._calculate_pedestal_height(node.parent)

                # Position files on top of parent pedestal
                pos_array = np.array([
                    position.x + position.width / 2,
                    parent_height + height / 2,
                    position.z + position.depth / 2,
                ], dtype=np.float32)

                scale_array = np.array([
                    position.width * 0.8,
                    height,
                    position.depth * 0.8,
                ], dtype=np.float32)

                if path_str in selection:
                    color = self._colors["selected"].copy()
                elif node.type == NodeType.SYMLINK:
                    color = self._colors["symlink"].copy()
                elif self._color_mode == ColorMode.AGE:
                    # SGI fsn style: color by file age
                    color = self._calculate_age_color(node.mtime)
                else:
                    # Type-based color (default gray for files)
                    color = np.array([0.6, 0.6, 0.7, 1.0], dtype=np.float32)

                # Calculate emission for file cube
                emission = self._calculate_emission(node)

                # File cubes have higher shininess (more glossy surface)
                cube = CubeInstance(position=pos_array, scale=scale_array, color=color, shininess=50.0, emission=emission)
                self._node_to_cube[path_str] = len(self._cubes)
                self._path_to_cube_index[path_str] = len(self._cubes)
                self._cubes.append(cube)

        # Build connections
        for parent_path, child_path in layout_result.connections:
            if parent_path not in nodes or child_path not in nodes:
                continue
            
            # Only connect directory to directory (Platform to Platform)
            # Feature 2 requirement: "Connect parent directory platform to child directory platform"
            parent_node = nodes[parent_path]
            child_node = nodes[child_path]
            
            if parent_node.type == NodeType.DIRECTORY and child_node.type == NodeType.DIRECTORY:
                parent_pos = layout_result.positions[parent_path]
                child_pos = layout_result.positions[child_path]
                
                # Connect Back Edge of Parent to Front Edge of Child
                # Parent is "in front" (larger Z), Child is "behind" (smaller/more negative Z)
                # We connect Parent Min Z to Child Max Z
                # Height: Connect along the ground (y = 0.05) to be visible above ground plane
                
                start_pos = np.array([
                    parent_pos.x + parent_pos.width / 2,
                    0.05,
                    parent_pos.z  # Back face (Min Z)
                ], dtype=np.float32)
                
                end_pos = np.array([
                    child_pos.x + child_pos.width / 2,
                    0.05,
                    child_pos.z + child_pos.depth  # Front face (Max Z)
                ], dtype=np.float32)
                
                self._connections.append((start_pos, end_pos))
                self._connection_metadata.append((parent_path, child_path))  # PR-4: Store metadata

        self.update()

    def _on_timer(self) -> None:
        """Timer callback for animation."""
        self.update()

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
                # Height is calculated by _calculate_pedestal_height
                height = self._calculate_pedestal_height(node)
                
                min_y = 0.0
                max_y = height
                
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
                
                # Calculate parent pedestal height to determine base Y
                parent_height = 0.0
                if node.parent and self._calculate_pedestal_height:
                     parent_height = self._calculate_pedestal_height(node.parent)
                
                min_y = parent_height
                max_y = parent_height + height
                
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

        # Calculate bbox center
        center = np.array([
            position.x + position.width / 2,
            position.y + position.height / 2,
            position.z + position.depth / 2,
        ], dtype=np.float32)

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

        # Animate camera to new position over ~300ms
        self._animate_camera_to(center, distance, duration_ms=300)

    def _animate_camera_to(self, target: np.ndarray, distance: float,
                          duration_ms: int = 300) -> None:
        """Animate camera to look at a target position.

        Args:
            target: Target position to look at
            distance: Distance from target
            duration_ms: Animation duration in milliseconds
        """
        # Store animation state
        self._animation_start_time = time.time()
        self._animation_duration = duration_ms / 1000.0
        self._animation_start_pos = self.camera.state.position.copy()
        self._animation_start_target = self.camera.state.target.copy()
        self._animation_end_target = target

        # Calculate end position based on current direction
        direction = self._animation_start_pos - self._animation_start_target
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        else:
            direction = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            direction = direction / np.linalg.norm(direction)

        self._animation_end_pos = target + direction * distance

        # Enable animation flag
        self._is_animating = True

        # Ensure timer is running
        if not self._update_timer.isActive():
            self._update_timer.start(16)

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

            # Ease-in-out function
            t = progress
            ease = t * t * (3.0 - 2.0 * t)

            # Interpolate position and target
            new_pos = (1 - ease) * self._animation_start_pos + ease * self._animation_end_pos
            new_target = (1 - ease) * self._animation_start_target + ease * self._animation_end_target

            self.camera._state.position = new_pos
            self.camera._state.target = new_target
            self.camera._update_orbit_from_position()

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
                self._cubes[idx].color = self._colors["selected"].copy()
            else:
                # Reset to default color based on color mode
                node = nodes.get(path)
                if node:
                    if node.type == NodeType.SYMLINK:
                        self._cubes[idx].color = self._colors["symlink"].copy()
                    elif self._color_mode == ColorMode.AGE:
                        self._cubes[idx].color = self._calculate_age_color(node.mtime)
                    else:
                        self._cubes[idx].color = self._colors["file"].copy()

        # Update colors for all platforms
        for path, idx in self._node_to_platform.items():
            if path in self._selected_paths:
                self._platforms[idx].color = self._colors["selected"].copy()
            else:
                self._platforms[idx].color = self._colors["directory"].copy()

        # PR-4: Calculate which connections should be highlighted
        # A connection is highlighted if either endpoint (parent or child directory) is selected
        self._selected_connections.clear()
        for i, (parent_path, child_path) in enumerate(self._connection_metadata):
            if parent_path in self._selected_paths or child_path in self._selected_paths:
                self._selected_connections.add(i)

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
                if node.type == NodeType.SYMLINK:
                    self._cubes[idx].color = self._colors["symlink"].copy()
                elif self._color_mode == ColorMode.AGE:
                    self._cubes[idx].color = self._calculate_age_color(node.mtime)
                else:
                    self._cubes[idx].color = self._colors["file"].copy()

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
                    if node.type == NodeType.SYMLINK:
                        self._cubes[idx].color = self._colors["symlink"].copy()
                    elif self._color_mode == ColorMode.AGE:
                        self._cubes[idx].color = self._calculate_age_color(node.mtime)
                    else:
                        self._cubes[idx].color = self._colors["file"].copy()
            # Handle directory platforms
            elif path in self._node_to_platform:
                idx = self._node_to_platform[path]
                self._platforms[idx].color = self._colors["directory"].copy()
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

        # First, calculate label positions with collision avoidance
        label_positions = self._calculate_label_positions()

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
        for path_str, (label_x, label_z, quad_width, quad_height) in label_positions.items():
            if path_str not in self._nodes:
                continue

            node = self._nodes[path_str]
            if not node.is_directory:
                continue

            # Create text texture and get dimensions
            texture_id, _, _ = self._create_text_texture(node.name, font_size=28)

            # Position text on the ground (y=0)
            text_y = 0.01  # Just above ground

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
            # Texture coords: flip U to fix mirror effect
            glTexCoord2f(1, 1)
            glVertex3f(-0.5, 0, -0.5)  # Bottom-left
            glTexCoord2f(0, 1)
            glVertex3f(0.5, 0, -0.5)  # Bottom-right
            glTexCoord2f(0, 0)
            glVertex3f(0.5, 0, 0.5)  # Top-right
            glTexCoord2f(1, 0)
            glVertex3f(-0.5, 0, 0.5)  # Top-left
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
        if position[1] < radius:  # Assuming ground is at y=0 (or -0.5)
            # Actually ground is at -0.5, but we want to stay above y=0.5
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

    def _calculate_label_positions(self) -> dict[str, tuple[float, float, float, float]]:
        """Calculate label positions with collision avoidance.

        Returns:
            Dictionary mapping path_str to (x, z, width, height)
        """
        from dataclasses import dataclass

        @dataclass
        class LabelRect:
            x: float  # Center X
            z: float  # Center Z
            width: float  # Total width
            height: float  # Total height

            @property
            def min_x(self) -> float:
                return self.x - self.width / 2

            @property
            def max_x(self) -> float:
                return self.x + self.width / 2

            @property
            def min_z(self) -> float:
                return self.z - self.height / 2

            @property
            def max_z(self) -> float:
                return self.z + self.height / 2

            def collides_with(self, other: 'LabelRect') -> bool:
                """Check if this label collides with another."""
                return not (
                    self.max_x < other.min_x or self.min_x > other.max_x or
                    self.max_z < other.min_z or self.min_z > other.max_z
                )

        # Collect all directory platforms
        directories = []
        for path_str, platform_idx in self._node_to_platform.items():
            if path_str not in self._nodes:
                continue
            node = self._nodes[path_str]
            if not node.is_directory:
                continue
            position = self._positions.get(path_str)
            if not position:
                continue
            directories.append((path_str, node, position))

        # Sort by size (larger first) to reduce fragmentation
        directories.sort(key=lambda x: x[2].width * x[2].depth, reverse=True)

        # Calculate label positions
        placed_labels: dict[str, LabelRect] = {}
        label_positions: dict[str, tuple[float, float, float, float]] = {}

        for path_str, node, position in directories:
            # Get texture dimensions
            _, tex_width, tex_height = self._create_text_texture(node.name, font_size=28)

            # Calculate scale
            platform_size = min(position.width, position.depth)
            aspect_ratio = tex_width / tex_height
            quad_width = platform_size * 0.8
            quad_height = quad_width / aspect_ratio

            # Platform bounds
            plat_min_x = position.x
            plat_max_x = position.x + position.width
            plat_min_z = position.z
            plat_max_z = position.z + position.depth
            plat_center_x = (plat_min_x + plat_max_x) / 2
            plat_center_z = (plat_min_z + plat_max_z) / 2

            # Generate candidate positions around the platform
            candidates = []
            gap = 0.5

            # Right side
            candidates.append((plat_max_x + gap + quad_width / 2, plat_center_z, 0))
            # Left side
            candidates.append((plat_min_x - gap - quad_width / 2, plat_center_z, 1))
            # Bottom side
            candidates.append((plat_center_x, plat_max_z + gap + quad_height / 2, 2))
            # Top side
            candidates.append((plat_center_x, plat_min_z - gap - quad_height / 2, 3))

            # Add further candidates (spiral outward)
            for ring in range(1, 5):
                offset = ring * 2.0
                candidates.append((plat_max_x + offset + quad_width / 2, plat_center_z, 10 + ring))
                candidates.append((plat_min_x - offset - quad_width / 2, plat_center_z, 10 + ring + 5))
                candidates.append((plat_center_x, plat_max_z + offset + quad_height / 2, 10 + ring + 10))
                candidates.append((plat_center_x, plat_min_z - offset - quad_height / 2, 10 + ring + 15))
                # Diagonal positions
                candidates.append((plat_max_x + offset + quad_width / 2, plat_max_z + offset + quad_height / 2, 20 + ring))
                candidates.append((plat_min_x - offset - quad_width / 2, plat_max_z + offset + quad_height / 2, 20 + ring + 5))
                candidates.append((plat_max_x + offset + quad_width / 2, plat_min_z - offset - quad_height / 2, 20 + ring + 10))
                candidates.append((plat_min_x - offset - quad_width / 2, plat_min_z - offset - quad_height / 2, 20 + ring + 15))

            # Sort candidates by priority
            candidates.sort(key=lambda x: x[2])

            # Find best candidate position
            best_pos = None
            for cand_x, cand_z, _ in candidates:
                label_rect = LabelRect(cand_x, cand_z, quad_width, quad_height)

                # Check collision with platforms
                collides_with_platform = False
                for _, _, plat_pos in directories:
                    plat_rect = LabelRect(
                        (plat_pos.x + plat_pos.width) / 2,
                        (plat_pos.z + plat_pos.depth) / 2,
                        plat_pos.width + 0.3,
                        plat_pos.depth + 0.3
                    )
                    if label_rect.collides_with(plat_rect):
                        collides_with_platform = True
                        break

                if collides_with_platform:
                    continue

                # Check collision with other placed labels
                collides_with_label = False
                for placed_rect in placed_labels.values():
                    if label_rect.collides_with(placed_rect):
                        collides_with_label = True
                        break

                if not collides_with_label:
                    best_pos = (cand_x, cand_z)
                    break

            # If no valid position found, use the right side with large offset
            if best_pos is None:
                best_pos = (plat_max_x + quad_width + 2.0, plat_center_z)

            # Store the label position
            placed_labels[path_str] = LabelRect(best_pos[0], best_pos[1], quad_width, quad_height)
            label_positions[path_str] = (best_pos[0], best_pos[1], quad_width, quad_height)

        return label_positions

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
