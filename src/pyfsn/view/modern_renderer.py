"""ModernGL-based OpenGL renderer for 3D file system visualization.

This is a modern replacement for the legacy PyOpenGL renderer. It uses:
- ModernGL for Core Profile OpenGL 3.3+ context management
- Programmable pipeline with custom shaders
- Efficient buffer management for instanced rendering
- Modern OpenGL features (VAOs, VBOs, UBOs)

DESIGN DECISIONS:
1. Parallel Implementation: This renderer is built alongside the existing
   renderer.py (legacy PyOpenGL) to avoid breaking existing functionality.
   The two renderers can be swapped via configuration once complete.

2. Shader-based Rendering: Unlike the fixed-function pipeline in the legacy
   renderer, this uses custom GLSL shaders for maximum flexibility and
   modern GPU features.

3. Instanced Rendering: File cubes will be rendered using instanced drawing
   (glDrawElementsInstanced) for significantly better performance with
   thousands of files.

4. Resource Management: A ResourceManager class will handle shader programs,
   VAOs, VBOs, and texture lifecycles to prevent memory leaks.

5. Picking System: GPU-based color picking will replace the raycasting
   approach for more accurate and performant node selection.

6. Camera System: Will reuse the existing Camera class from view.camera
   to maintain compatibility with the existing controller code.

MIGRATION PATH:
Phase 1: Basic scaffold + resource management
Phase 2: Core rendering (cubes, platforms, connections)
Phase 3: Picking and interaction
Phase 4: Advanced features (fog, themes, post-processing)
"""

import logging
import math
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import QTimer, pyqtSignal, Qt
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

from pyfsn.model.node import Node, NodeType
from pyfsn.view.camera import Camera
from pyfsn.view.shader_loader import ShaderLoader
from pyfsn.view.performance import PerformanceMonitor, FrustumCuller, LevelOfDetail

logger = logging.getLogger(__name__)

# ModernGL imports - will be properly initialized after OpenGL context creation
try:
    import moderngl
    MODERNGL_AVAILABLE = True
except ImportError:
    MODERNGL_AVAILABLE = False


class ColorMode(Enum):
    """Color mode for file visualization.

    Note: This mirrors the ColorMode from renderer.py for compatibility.
    """
    TYPE = "type"  # Color by file type
    AGE = "age"    # Color by file age (SGI fsn style)


@dataclass
class CubeInstance:
    """Single cube instance data for GPU rendering.

    DESIGN DECISION: This structure is designed to be uploaded directly
    to a GPU buffer for instanced rendering. The layout matches the
    shader's instance attribute layout.
    """
    position: np.ndarray  # [x, y, z] - center position
    scale: np.ndarray     # [w, h, d] - dimensions
    color: np.ndarray     # [r, g, b, a] - RGBA color


class GeometryManager:
    """Manages geometry VBOs and VAOs for rendering.

    DESIGN DECISION: Separated from ResourceManager to focus specifically
    on geometry data (vertices, indices, normals) separate from instance data.
    """

    def __init__(self, ctx: 'moderngl.Context') -> None:
        """Initialize the geometry manager.

        Args:
            ctx: ModernGL context for creating GPU resources
        """
        self._ctx = ctx
        self._vertex_vbos: dict[str, Any] = {}
        self._index_buffers: dict[str, Any] = {}
        self._vaos: dict[str, Any] = {}

    def create_cube_geometry(self) -> tuple:
        """Create cube geometry VBOs and index buffer.

        Returns:
            Tuple of (vertex_vbo, normal_vbo, index_buffer)
        """
        # Cube vertices (8 corners)
        vertices = np.array([
            # Front face
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
            # Back face
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
        ], dtype=np.float32)

        # Cube indices (triangles, 12 faces * 3 vertices = 36 indices)
        indices = np.array([
            # Front face
            0, 1, 2, 2, 3, 0,
            # Right face
            1, 5, 6, 6, 2, 1,
            # Back face
            5, 4, 7, 7, 6, 5,
            # Left face
            4, 0, 3, 3, 7, 4,
            # Bottom face
            4, 5, 1, 1, 0, 4,
            # Top face
            3, 2, 6, 6, 7, 3,
        ], dtype=np.uint32)

        # Normals for each face (for flat shading)
        normals = np.array([
            # Front (4 vertices)
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            # Back (4 vertices)
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
        ], dtype=np.float32)

        vertex_vbo = self._ctx.buffer(vertices.tobytes())
        normal_vbo = self._ctx.buffer(normals.tobytes())
        index_buffer = self._ctx.buffer(indices.tobytes())

        return vertex_vbo, normal_vbo, index_buffer

    def create_edge_geometry(self) -> Any:
        """Create edge line geometry for cube outlines.

        Returns:
            VBO with edge vertices
        """
        edge_vertices = np.array([
            # Bottom face edges
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5], [0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5],
            [-0.5, -0.5, 0.5], [-0.5, -0.5, -0.5],
            # Top face edges
            [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5], [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5],
            # Vertical edges
            [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5], [0.5, 0.5, -0.5],
            [0.5, -0.5, 0.5], [0.5, 0.5, 0.5],
            [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5],
        ], dtype=np.float32)

        return self._ctx.buffer(edge_vertices.tobytes())

    def create_ground_geometry(self) -> Any:
        """Create ground plane geometry.

        Returns:
            VBO with ground vertices including texture coordinates
        """
        ground_size = 500.0
        vertices = np.array([
            [-ground_size, -0.5, -ground_size, 0.0, 0.0],
            [ground_size, -0.5, -ground_size, ground_size, 0.0],
            [ground_size, -0.5, ground_size, ground_size, ground_size],
            [-ground_size, -0.5, ground_size, 0.0, ground_size],
        ], dtype=np.float32)

        return self._ctx.buffer(vertices.tobytes())

    def cleanup(self) -> None:
        """Release all geometry resources."""
        for vbo in self._vertex_vbos.values():
            vbo.release()
        for ibo in self._index_buffers.values():
            ibo.release()
        for vao in self._vaos.values():
            vao.release()

        self._vertex_vbos.clear()
        self._index_buffers.clear()
        self._vaos.clear()


class ModernGLRenderer(QOpenGLWidget):
    """ModernGL-based 3D file system visualizer.

    This renderer provides a modern OpenGL 3.3+ implementation using
    programmable shaders and efficient instanced rendering.

    DESIGN DECISION: Extends QOpenGLWidget for Qt integration, similar to
    the legacy Renderer. This allows seamless swapping in the controller.

    SIGNALS:
        node_clicked: Emitted when a node is clicked (Node, is_double_click)
        node_focused: Emitted when camera focuses on a node
        selection_changed: Emitted when selection changes (set of paths)
    """

    # Signals - matching legacy renderer for compatibility
    node_clicked = pyqtSignal(Node, bool)  # Node, is_double_click
    node_focused = pyqtSignal(Node)
    selection_changed = pyqtSignal(set)

    def __init__(self, parent=None) -> None:
        """Initialize the ModernGL renderer.

        DESIGN DECISION: Most initialization happens in initializeGL()
        after the OpenGL context is created. ModernGL requires a valid
        context to create GPU resources.
        """
        if not MODERNGL_AVAILABLE:
            raise ImportError(
                "ModernGL is not installed. Install with: pip install moderngl moderngl-window scipy"
            )

        super().__init__(parent)

        # Camera - reuse existing Camera class for compatibility
        self.camera = Camera()

        # Performance monitoring
        self._performance_monitor = PerformanceMonitor()

        # Optimization: Frustum culling and LOD (PR-5)
        self._frustum_culler = FrustumCuller()
        self._lod = LevelOfDetail(distances=[15.0, 40.0, 80.0, 150.0])
        self._lod_edge_threshold = 50.0  # Distance threshold for skipping edge rendering
        self._lod_small_cube_threshold = 100.0  # Distance threshold for skipping small cubes

        # ModernGL context (created in initializeGL)
        self._ctx: 'moderngl.Context | None' = None
        self._shader_loader: 'ShaderLoader | None' = None
        self._geometry_manager: 'GeometryManager | None' = None

        # Shader programs
        self._cube_program = None
        self._wire_program = None
        self._ground_program = None
        self._sky_program = None

        # Fog settings (SGI fsn style)
        self._fog_start = 100.0
        self._fog_end = 500.0
        self._fog_color = np.array([0.1, 0.1, 0.15, 1.0], dtype=np.float32)
        self._identity_matrix = np.identity(4, dtype=np.float32)

        # Node data - mirrors legacy renderer structure
        self._nodes: dict[str, Node] = {}
        self._positions: dict[str, object] = {}
        self._cubes: list[CubeInstance] = []
        self._platforms: list[CubeInstance] = []
        self._connections: list[tuple[np.ndarray, np.ndarray]] = []
        self._connection_metadata: list[tuple[str, str]] = []

        # Mapping for node lookup
        self._node_to_cube: dict[str, int] = {}
        self._node_to_platform: dict[str, int] = {}

        # Selection state
        self._selected_paths: set[str] = set()
        self._selected_connections: set[int] = set()
        self._focused_path: str | None = None

        # Color mode (SGI fsn style)
        self._color_mode: ColorMode = ColorMode.AGE

        # Colors - matching legacy renderer
        self._colors = {
            "directory": np.array([0.2, 0.6, 1.0, 1.0], dtype=np.float32),
            "file": np.array([0.8, 0.8, 0.8, 1.0], dtype=np.float32),
            "symlink": np.array([0.3, 1.0, 0.3, 1.0], dtype=np.float32),
            "selected": np.array([1.0, 1.0, 0.3, 1.0], dtype=np.float32),
            "focused": np.array([1.0, 0.8, 0.0, 1.0], dtype=np.float32),
        }

        # Instance data buffers (CPU side for updates)
        max_instances = 100000
        self._cube_positions = np.zeros((max_instances, 3), dtype=np.float32)
        self._cube_scales = np.zeros((max_instances, 3), dtype=np.float32)
        self._cube_colors = np.zeros((max_instances, 4), dtype=np.float32)

        # Current instance counts
        self._cube_instance_count = 0
        self._platform_instance_count = 0

        # Geometry VBOs (created in initializeGL)
        self._cube_vertex_vbo = None
        self._cube_normal_vbo = None
        self._cube_index_buffer = None
        self._edge_vertex_vbo = None
        self._ground_vbo = None

        # Instance data buffers (GPU side)
        self._cube_position_buffer = None
        self._cube_scale_buffer = None
        self._cube_color_buffer = None

        # Animation timer for 60 FPS updates
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._on_timer)
        self._update_timer.start(16)

        # Mouse tracking for hover effects
        self.setMouseTracking(True)

        # Tooltip widget reference (set by controller)
        self._tooltip = None

        # Initialization state
        self._initialized = False

    def initializeGL(self) -> None:
        """Initialize ModernGL context and GPU resources.

        DESIGN DECISION: This is called by Qt when the OpenGL context
        is first created. We create the ModernGL context here and
        initialize all GPU resources.

        The ModernGL context is created from the existing Qt context,
        allowing seamless integration with QOpenGLWidget.
        """
        # Create ModernGL context from Qt's OpenGL context
        self._ctx = moderngl.create_context()

        # Set OpenGL defaults
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.clear_color(*self._fog_color.tolist())

        # Create shader loader
        self._shader_loader = ShaderLoader(self._ctx)

        # Create geometry manager
        self._geometry_manager = GeometryManager(self._ctx)

        # Get shader directory
        shader_dir = Path(__file__).parent / "shaders"

        # Load shader programs
        try:
            # Cube instancing shader
            self._cube_program = self._shader_loader.load_program(
                vertex_shader=str(shader_dir / "cube.vert"),
                fragment_shader=str(shader_dir / "cube.frag"),
                cache_key="cube"
            )

            # Wire/line shader (with pulse effects)
            self._wire_program = self._shader_loader.load_program(
                vertex_shader=str(shader_dir / "wire.vert"),
                fragment_shader=str(shader_dir / "wire.frag"),
                cache_key="wire"
            )

            # Ground shader
            self._ground_program = self._shader_loader.load_program(
                vertex_shader=str(shader_dir / "ground.vert"),
                fragment_shader=str(shader_dir / "ground.frag"),
                cache_key="ground"
            )

            # Sky shader
            self._sky_program = self._shader_loader.load_program(
                vertex_shader=str(shader_dir / "sky.vert"),
                fragment_shader=str(shader_dir / "sky.frag"),
                cache_key="sky"
            )

            logger.info("All shader programs loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load shaders: {e}")
            raise

        # Create geometry buffers
        self._create_geometry()

        # Create instance data buffers
        self._create_instance_buffers()

        self._initialized = True
        logger.info("ModernGL renderer initialized")

    def _create_geometry(self) -> None:
        """Create geometry VBOs for rendering."""
        # Cube geometry
        self._cube_vertex_vbo, self._cube_normal_vbo, self._cube_index_buffer = \
            self._geometry_manager.create_cube_geometry()

        # Edge geometry
        self._edge_vertex_vbo = self._geometry_manager.create_edge_geometry()

        # Ground geometry
        self._ground_vbo = self._geometry_manager.create_ground_geometry()

    def _create_instance_buffers(self) -> None:
        """Create instance data buffers for instanced rendering."""
        max_instances = 100000
        self._cube_position_buffer = self._ctx.buffer(
            np.zeros((max_instances, 3), dtype=np.float32).tobytes()
        )
        self._cube_scale_buffer = self._ctx.buffer(
            np.zeros((max_instances, 3), dtype=np.float32).tobytes()
        )
        self._cube_color_buffer = self._ctx.buffer(
            np.zeros((max_instances, 4), dtype=np.float32).tobytes()
        )

    def resizeGL(self, w: int, h: int) -> None:
        """Handle viewport resize.

        Args:
            w: New viewport width
            h: New viewport height

        DESIGN DECISION: ModernGL requires explicit viewport updates
        when the window is resized.
        """
        if self._ctx is None:
            return

        # Update viewport with device pixel ratio for high-DPI displays
        dpr = self.devicePixelRatio()
        self._ctx.viewport = (0, 0, int(w * dpr), int(h * dpr))

        # TODO: Update projection matrix in Phase 2
        # self._update_projection_matrix(w, h)

    def paintGL(self) -> None:
        """Render the scene.

        DESIGN DECISION: This is called by Qt on each update. The render
        sequence follows the legacy renderer for visual consistency:
        1. Sky gradient background
        2. Ground grid
        3. Connections (parent-child wires)
        4. Platforms (directories)
        5. Cubes (files)
        6. Selection effects (spotlights, highlights)
        """
        self._performance_monitor.start_frame()

        if not self._initialized or self._ctx is None:
            self._performance_monitor.end_frame()
            return

        # Clear buffers
        self._ctx.clear(*self._fog_color.tolist())

        # Update frustum culling (PR-5: Optimization)
        aspect = self.width() / max(1, self.height())
        view_matrix = self.camera.view_matrix
        proj_matrix = self.camera.projection_matrix(aspect)
        self._frustum_culler.update_from_camera(view_matrix, proj_matrix, aspect)

        # Upload instance data to GPU
        self._upload_cube_instances()

        # Draw sky gradient first (fullscreen background)
        self._draw_sky_gradient()

        # Draw ground in 3D space
        self._draw_ground_grid(view_matrix, proj_matrix)

        # Draw connections
        self._draw_connections(view_matrix, proj_matrix)

        # Draw platforms first (opaque)
        if self._platform_instance_count > 0:
            self._draw_cubes(
                0, self._platform_instance_count,
                view_matrix, proj_matrix
            )

        # Draw all cubes (files)
        if self._cube_instance_count > 0:
            self._draw_cubes(
                self._platform_instance_count,
                self._cube_instance_count,
                view_matrix, proj_matrix
            )

        # Draw cube edges
        self._draw_cube_edges(view_matrix, proj_matrix)

        # Draw spotlights for selection (transparent)
        self._draw_selection_spotlights(view_matrix, proj_matrix)

        self._performance_monitor.end_frame()
        self._performance_monitor.set_instance_count(
            self._cube_instance_count + self._platform_instance_count
        )
        self._performance_monitor.set_draw_calls(
            3  # sky, ground, cubes (instanced)
        )

    def _draw_sky_gradient(self) -> None:
        """Draw sky gradient as a fullscreen quad (FSN classic style)."""
        self._ctx.disable(moderngl.DEPTH_TEST)

        self._sky_program.use()
        # No uniforms needed for sky

        # Draw fullscreen quad (4 vertices)
        vao = self._ctx.vertex_array(self._sky_program, [])
        vao.render(vertices=4)

        self._ctx.enable(moderngl.DEPTH_TEST)

    def _draw_ground_grid(self, view_matrix: np.ndarray, proj_matrix: np.ndarray) -> None:
        """Draw the ground plane with grid lines (FSN classic style)."""
        self._ground_program.use()

        # Set uniforms
        self._ground_program["u_view"].write(view_matrix.tobytes())
        self._ground_program["u_projection"].write(proj_matrix.tobytes())
        self._ground_program["u_model"].write(self._identity_matrix.tobytes())
        self._ground_program["u_camera_position"].write(self.camera.state.position.tobytes())
        self._ground_program["u_fog_start"] = self._fog_start
        self._ground_program["u_fog_end"] = self._fog_end
        self._ground_program["u_fog_color"].write(self._fog_color.tobytes())

        # Create VAO and render
        vao = self._ctx.vertex_array(
            self._ground_program,
            [(self._ground_vbo, "3f 2f", "in_position", "in_texcoord")]
        )
        vao.render(mode=4)  # Triangle fan (mode=4 in ModernGL)

    def _draw_cubes(
        self,
        offset: int,
        count: int,
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray
    ) -> None:
        """Draw instanced cubes.

        Args:
            offset: Starting instance index
            count: Number of instances to draw
            view_matrix: View matrix
            proj_matrix: Projection matrix
        """
        self._cube_program.use()

        # Set uniforms
        self._cube_program["u_view"].write(view_matrix.tobytes())
        self._cube_program["u_projection"].write(proj_matrix.tobytes())
        self._cube_program["u_model"].write(self._identity_matrix.tobytes())
        self._cube_program["u_camera_position"].write(self.camera.state.position.tobytes())
        self._cube_program["u_fog_start"] = self._fog_start
        self._cube_program["u_fog_end"] = self._fog_end
        self._cube_program["u_fog_color"].write(self._fog_color.tobytes())

        # Create VAO with instancing
        position_buffer = self._ctx.buffer(
            self._cube_positions[offset:offset + count].tobytes()
        )
        scale_buffer = self._ctx.buffer(
            self._cube_scales[offset:offset + count].tobytes()
        )
        color_buffer = self._ctx.buffer(
            self._cube_colors[offset:offset + count].tobytes()
        )

        vao = self._ctx.vertex_array(
            self._cube_program,
            [
                (self._cube_vertex_vbo, "3f", "in_position"),
                (self._cube_normal_vbo, "3f", "in_normal"),
                (position_buffer, "3f/i", "in_instance_position"),
                (scale_buffer, "3f/i", "in_instance_scale"),
                (color_buffer, "4f/i", "in_instance_color"),
            ],
            self._cube_index_buffer,
        )

        vao.render(instances=count)

    def _draw_cube_edges(self, view_matrix: np.ndarray, proj_matrix: np.ndarray) -> None:
        """Draw cube edges for better definition.

        PR-5: Uses LOD to skip edge rendering for distant objects.
        """
        # Calculate distance to camera for LOD (PR-5)
        camera_pos = self.camera.state.position

        # Filter instances that need edge rendering
        edges_to_draw = []
        edge_colors = []

        for i in range(self._platform_instance_count):
            pos = self._cube_positions[i]
            scale = self._cube_scales[i]
            distance = np.linalg.norm(pos - camera_pos)

            # Skip edge rendering for distant objects (PR-5 optimization)
            if distance > self._lod_edge_threshold:
                continue

            # Also skip edges for very small cubes at medium distance
            max_scale = max(scale)
            if max_scale < 0.5 and distance > self._lod_small_cube_threshold * 0.3:
                continue

            # Dark edge color - computed from cube color but darker
            color = self._cube_colors[i]
            edge_color = np.array([
                color[0] * 0.3,
                color[1] * 0.3,
                color[2] * 0.3,
                color[3]
            ], dtype=np.float32)

            edges_to_draw.append((pos, scale))
            edge_colors.append(edge_color)

        for i in range(self._platform_instance_count, self._platform_instance_count + self._cube_instance_count):
            pos = self._cube_positions[i]
            scale = self._cube_scales[i]
            distance = np.linalg.norm(pos - camera_pos)

            # Skip edge rendering for distant objects
            if distance > self._lod_edge_threshold:
                continue

            # Also skip edges for very small cubes at medium distance
            max_scale = max(scale)
            if max_scale < 0.5 and distance > self._lod_small_cube_threshold * 0.3:
                continue

            # Dark edge color
            color = self._cube_colors[i]
            edge_color = np.array([
                color[0] * 0.3,
                color[1] * 0.3,
                color[2] * 0.3,
                color[3]
            ], dtype=np.float32)

            edges_to_draw.append((pos, scale))
            edge_colors.append(edge_color)

        if not edges_to_draw:
            return

        # Draw each edge (not instanced, since we have varying colors)
        # Note: For better performance, this could be batched
        for (pos, scale), color in zip(edges_to_draw, edge_colors):
            # Create transform matrix
            transform = np.identity(4, dtype=np.float32)
            transform[0, 0] = scale[0]
            transform[1, 1] = scale[1]
            transform[2, 2] = scale[2]
            transform[0, 3] = pos[0]
            transform[1, 3] = pos[1]
            transform[2, 3] = pos[2]

            # Simple vertex/fragment shader for edges
            # For now, we'll use the wire program without instancing
            # Create a simple color buffer
            color_buffer = self._ctx.buffer(np.tile(color, 24).tobytes())  # 24 vertices

            vao = self._ctx.vertex_array(
                self._wire_program,
                [
                    (self._edge_vertex_vbo, "3f", "in_position"),
                    (color_buffer, "4f/i", "in_color"),
                ]
            )

            # Set uniforms for wire program
            self._wire_program["u_view"].write(view_matrix.tobytes())
            self._wire_program["u_projection"].write(proj_matrix.tobytes())
            self._wire_program["u_model"].write(transform.tobytes())
            self._wire_program["u_camera_position"].write(camera_pos.tobytes())
            self._wire_program["u_fog_start"] = self._fog_start
            self._wire_program["u_fog_end"] = self._fog_end
            self._wire_program["u_fog_color"].write(self._fog_color.tobytes())

            vao.render(mode=1)  # Lines

    def _draw_connections(self, view_matrix: np.ndarray, proj_matrix: np.ndarray) -> None:
        """Draw wire connections between platforms with highlighting for selected nodes (PR-4)."""
        if not self._connections:
            return

        self._wire_program.use()

        # Set uniforms
        self._wire_program["u_view"].write(view_matrix.tobytes())
        self._wire_program["u_projection"].write(proj_matrix.tobytes())
        self._wire_program["u_model"].write(self._identity_matrix.tobytes())
        self._wire_program["u_camera_position"].write(self.camera.state.position.tobytes())
        self._wire_program["u_fog_start"] = self._fog_start
        self._wire_program["u_fog_end"] = self._fog_end
        self._wire_program["u_fog_color"].write(self._fog_color.tobytes())

        # Draw non-selected connections first (normal style)
        unselected_vertices = []
        unselected_colors = []
        for i, (start, end) in enumerate(self._connections):
            if i not in self._selected_connections:
                unselected_vertices.extend([start, end])
                unselected_colors.extend([
                    [1.0, 1.0, 1.0, 0.4],  # Dim white
                    [1.0, 1.0, 1.0, 0.4],
                ])

        if unselected_vertices:
            vertices = np.array(unselected_vertices, dtype=np.float32)
            colors = np.array(unselected_colors, dtype=np.float32)

            vbo = self._ctx.buffer(vertices.tobytes())
            color_buffer = self._ctx.buffer(colors.tobytes())

            vao = self._ctx.vertex_array(
                self._wire_program,
                [
                    (vbo, "3f", "in_position"),
                    (color_buffer, "4f/i", "in_color"),
                ]
            )
            vao.render(mode=1)  # Lines

        # Draw selected connections with highlight style (PR-4)
        if self._selected_connections:
            self._ctx.enable(moderngl.BLEND)
            self._ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

            # Thicker lines (selected)
            selected_vertices = []
            selected_colors = []
            for i in self._selected_connections:
                if i < len(self._connections):
                    start, end = self._connections[i]
                    selected_vertices.extend([start, end])
                    selected_colors.extend([
                        [1.0, 1.0, 0.6, 0.9],  # Bright yellow-white
                        [1.0, 1.0, 0.6, 0.9],
                    ])

            if selected_vertices:
                vertices = np.array(selected_vertices, dtype=np.float32)
                colors = np.array(selected_colors, dtype=np.float32)

                vbo = self._ctx.buffer(vertices.tobytes())
                color_buffer = self._ctx.buffer(colors.tobytes())

                vao = self._ctx.vertex_array(
                    self._wire_program,
                    [
                        (vbo, "3f", "in_position"),
                        (color_buffer, "4f/i", "in_color"),
                    ]
                )
                vao.render(mode=1)  # Lines

            self._ctx.disable(moderngl.BLEND)

    def _draw_selection_spotlights(self, view_matrix: np.ndarray, proj_matrix: np.ndarray) -> None:
        """Draw spotlights over selected items.

        TODO: Implement spotlight geometry and rendering
        """
        pass

    def _upload_cube_instances(self) -> None:
        """Upload instance data to GPU."""
        total_count = self._platform_instance_count + self._cube_instance_count
        if total_count > 0:
            self._cube_position_buffer.write(
                self._cube_positions[:total_count].tobytes()
            )
            self._cube_scale_buffer.write(
                self._cube_scales[:total_count].tobytes()
            )
            self._cube_color_buffer.write(
                self._cube_colors[:total_count].tobytes()
            )

    def _on_timer(self) -> None:
        """Timer callback for 60 FPS updates."""
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

    def load_layout(self, layout_result, nodes: dict[str, Node],
                   selection: set[str] | None = None) -> None:
        """Load nodes from layout result for rendering.

        DESIGN DECISION: This mirrors the legacy renderer's API for
        compatibility with the existing controller.

        The layout data is converted to GPU-ready buffers for efficient
        instanced rendering.

        Args:
            layout_result: LayoutResult from LayoutEngine with positions
            nodes: Dictionary mapping path strings to Node objects
            selection: Set of selected node paths
        """
        # Clear existing data
        self._cubes.clear()
        self._platforms.clear()
        self._connections.clear()
        self._connection_metadata.clear()
        self._node_to_cube.clear()
        self._node_to_platform.clear()

        self._nodes = nodes
        self._positions = layout_result.positions

        # Reset instance counts
        self._platform_instance_count = 0
        self._cube_instance_count = 0

        selection = selection or set()

        for path_str, position in layout_result.positions.items():
            if path_str not in nodes:
                continue

            node = nodes[path_str]

            if node.type == NodeType.DIRECTORY:
                # Platforms (Directories) - Size calculated by LayoutEngine (Feature 4)
                platform_w = position.width
                platform_d = position.depth

                height = 0.2  # Thin platform

                pos_array = np.array([
                    position.x + position.width / 2,
                    -height / 2,  # Top of platform is at y=0
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

                platform = CubeInstance(position=pos_array, scale=scale_array, color=color)
                self._node_to_platform[path_str] = len(self._platforms)
                self._platforms.append(platform)

                # Add to instance arrays
                idx = self._platform_instance_count
                self._cube_positions[idx] = pos_array
                self._cube_scales[idx] = scale_array
                self._cube_colors[idx] = color
                self._platform_instance_count += 1

            else:
                # File blocks - height from layout engine (SGI fsn style)
                height = position.height

                # Position files on top of y=0 (where platform top is)
                pos_array = np.array([
                    position.x + position.width / 2,
                    height / 2,
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

                cube = CubeInstance(position=pos_array, scale=scale_array, color=color)
                self._node_to_cube[path_str] = len(self._cubes)
                self._cubes.append(cube)

                # Add to instance arrays (after platforms)
                idx = self._platform_instance_count + self._cube_instance_count
                self._cube_positions[idx] = pos_array
                self._cube_scales[idx] = scale_array
                self._cube_colors[idx] = color
                self._cube_instance_count += 1

        # Build connections
        for parent_path, child_path in layout_result.connections:
            if parent_path not in nodes or child_path not in nodes:
                continue

            # Only connect directory to directory (Platform to Platform)
            parent_node = nodes[parent_path]
            child_node = nodes[child_path]

            if parent_node.type == NodeType.DIRECTORY and child_node.type == NodeType.DIRECTORY:
                parent_pos = layout_result.positions[parent_path]
                child_pos = layout_result.positions[child_path]

                # Connect Back Edge of Parent to Front Edge of Child
                start_pos = np.array([
                    parent_pos.x + parent_pos.width / 2,
                    -0.1,
                    parent_pos.z  # Back face (Min Z)
                ], dtype=np.float32)

                end_pos = np.array([
                    child_pos.x + child_pos.width / 2,
                    -0.1,
                    child_pos.z + child_pos.depth  # Front face (Max Z)
                ], dtype=np.float32)

                self._connections.append((start_pos, end_pos))
                self._connection_metadata.append((parent_path, child_path))

        # Update selected connections based on selection
        self._selected_connections.clear()
        for i, (parent_path, child_path) in enumerate(self._connection_metadata):
            if parent_path in selection or child_path in selection:
                self._selected_connections.add(i)

        self.update()

    def set_selection(self, selected_paths: set[str], nodes: dict[str, Node]) -> None:
        """Set the current selection.

        DESIGN DECISION: In the ModernGL renderer, selection updates
        will trigger GPU buffer updates rather than iterating through
        all instances on CPU.

        Args:
            selected_paths: Set of selected node paths
            nodes: Dictionary mapping path strings to Node objects
        """
        self._selected_paths = selected_paths.copy()
        self._nodes = nodes

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

        # Update GPU instance data
        self._update_instance_colors()

        # PR-4: Calculate which connections should be highlighted
        self._selected_connections.clear()
        for i, (parent_path, child_path) in enumerate(self._connection_metadata):
            if parent_path in self._selected_paths or child_path in self._selected_paths:
                self._selected_connections.add(i)

        self.selection_changed.emit(self._selected_paths)
        self.update()

    def _update_instance_colors(self) -> None:
        """Update instance color arrays from current cube/platform data."""
        for i, platform in enumerate(self._platforms):
            self._cube_colors[i] = platform.color

        for i, cube in enumerate(self._cubes):
            idx = self._platform_instance_count + i
            self._cube_colors[idx] = cube.color

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
        self._selected_connections.clear()

        # Update GPU instance data
        self._update_instance_colors()

        self.selection_changed.emit(self._selected_paths)
        self.update()

    def select_node(self, path: str) -> None:
        """Select a node by path.

        Args:
            path: Path string of node to select
        """
        self.set_selection({path}, self._nodes)

    def set_color_mode(self, mode: ColorMode) -> None:
        """Set the color mode for file visualization.

        Args:
            mode: ColorMode.AGE or ColorMode.TYPE
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

        # Update GPU instance data
        self._update_instance_colors()
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

    # Compatibility properties for existing controller code

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
        """Return ModernGL context (or None if not initialized)."""
        return self._ctx

    @property
    def instanced_vao(self):
        """Compatibility property - returns True when initialized."""
        return self._initialized

    @property
    def instanced_program(self):
        """Compatibility property - returns True when initialized."""
        return self._initialized

    # Tooltip and interaction methods

    def set_tooltip(self, tooltip_widget) -> None:
        """Set the tooltip widget reference for hover functionality.

        Args:
            tooltip_widget: The tooltip widget to display hover information
        """
        self._tooltip = tooltip_widget

    def get_node_at_position(self, x: int, y: int) -> Node | None:
        """Get the node at the given screen position.

        DESIGN DECISION: Uses CPU raycasting for now. Phase 3 will implement
        GPU-based color picking for more accurate and performant selection.

        Args:
            x: Screen X coordinate
            y: Screen Y coordinate

        Returns:
            Node at position or None
        """
        return self.raycast_find_node(x, y, self._nodes, self._positions)

    def set_root_path(self, path) -> None:
        """Set the root path for visualization.

        Args:
            path: Root path (Path object)
        """
        self._root_path = path

    def set_camera_mode(self, mode) -> None:
        """Set camera navigation mode.

        Args:
            mode: CameraMode enum value
        """
        self.camera.set_mode(mode)

    @property
    def color_mode(self) -> ColorMode:
        """Get current color mode."""
        return self._color_mode

    @property
    def performance_stats(self) -> dict:
        """Get performance statistics.

        Returns:
            Dictionary with FPS, frame time, instance count, etc.
        """
        return {
            "fps": self._performance_monitor.metrics.fps,
            "frame_time_ms": self._performance_monitor.metrics.frame_time_ms,
            "instance_count": self._cube_instance_count + self._platform_instance_count,
            "draw_calls": 3,  # sky, ground, cubes (instanced)
        }

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

        # Simple projection
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
        aspect = self.width() / max(1, self.height())
        fov_rad = math.radians(self.camera.state.fov)
        scale = 1.0 / math.tan(fov_rad / 2.0)

        # NDC calculation
        ndc_x = (view_x / view_z) * (scale / aspect)
        ndc_y = (view_y / view_z) * scale

        # Convert to screen coords
        screen_x = int((ndc_x + 1.0) * 0.5 * self.width())
        screen_y = int((1.0 - ndc_y) * 0.5 * self.height())

        return (screen_x, screen_y)

    def is_node_visible(self, path: str) -> bool:
        """Check if a node is visible within the camera frustum.

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

    def raycast_find_node(self, screen_x: int, screen_y: int,
                         nodes: dict[str, Node],
                         positions: dict[str, object]) -> Node | None:
        """Find node at screen position using ray casting.

        DESIGN DECISION: This will be replaced with GPU-based picking
        in Phase 3. The CPU raycasting from the legacy renderer is
        kept here as a fallback.

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

            # Get AABB bounds - match actual rendered positions
            if node.is_directory:
                # Platform: rendered at y = -height/2 (top at y=0)
                platform_h = 0.2
                min_y = -platform_h / 2
                max_y = platform_h / 2
                min_bounds = np.array([position.min_x, min_y, position.min_z], dtype=np.float32)
                max_bounds = np.array([position.max_x, max_y, position.max_z], dtype=np.float32)
            else:
                # File: rendered at y = height/2 (centered on platform top)
                height = position.height
                width_scaled = position.width * 0.8
                depth_scaled = position.depth * 0.8

                min_x = position.x + (position.width - width_scaled) / 2
                max_x = min_x + width_scaled
                min_z = position.z + (position.depth - depth_scaled) / 2
                max_z = min_z + depth_scaled
                min_y = 0.0  # File cube base at y=0
                max_y = height  # File cube top at y=height

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

    def _calculate_age_color(self, mtime: float) -> np.ndarray:
        """Calculate color based on file age (SGI fsn style).

        Args:
            mtime: File modification time (Unix timestamp)

        Returns:
            RGBA color array
        """
        now = time.time()
        age_days = (now - mtime) / 86400.0

        if age_days < 1:
            return np.array([0.2, 1.0, 0.2, 1.0], dtype=np.float32)
        elif age_days < 7:
            return np.array([0.2, 0.8, 1.0, 1.0], dtype=np.float32)
        elif age_days < 30:
            return np.array([0.8, 0.8, 0.2, 1.0], dtype=np.float32)
        elif age_days < 365:
            return np.array([1.0, 0.6, 0.2, 1.0], dtype=np.float32)
        else:
            return np.array([0.6, 0.3, 0.2, 1.0], dtype=np.float32)

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
