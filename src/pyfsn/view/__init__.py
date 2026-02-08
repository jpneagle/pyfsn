"""View layer for pyfsn 3D file system visualization.

This module provides the rendering and visualization components:

- Renderer: QOpenGLWidget subclass with Legacy OpenGL 2.1 (PyOpenGL)
- Camera: 3D camera with Orbit navigation mode
- CubeGeometry: GPU instanced cube mesh data (ModernGL asset, currently unused)
- MainWindow: Main application window with menu bar and controls
- TextOverlay: 2D text overlay for node labels
- ControlPanel: Side panel with camera controls
- SearchBar: Search input widget
- FileTreeWidget: Hierarchical file tree view
- Theme: Theme dataclass for visual styling
- ThemeManager: Manager for theme switching and persistence

ModernGL Resource Management (Phase 1.2):
- ShaderLoader: Load, compile, and cache shader programs
- BufferManager: Manage VBO, VAO, EBO GPU buffers
- TextureManager: Manage 2D textures with caching

Picking System (Phase 1.4):
- PickingSystem: CPU-side ray-AABB intersection for object selection
- Ray: 3D ray representation for raycasting
- AABB: Axis-aligned bounding box for intersection tests
"""

from pyfsn.view.renderer import Renderer
from pyfsn.view.modern_renderer import ModernGLRenderer
from pyfsn.view.camera import Camera, CameraMode, CameraState
from pyfsn.view.cube_geometry import CubeGeometry, CubeInstance, get_default_colors
from pyfsn.view.main_window import (
    MainWindow,
    TextOverlay,
    ControlPanel,
    SearchBar,
    FileTreeWidget,
)
from pyfsn.view.theme import (
    Theme,
    SGI_CLASSIC,
    DARK_MODE,
    CYBERPUNK,
    SOLARIZED,
    FOREST,
    OCEAN,
    DEFAULT_THEME,
    BUILTIN_THEMES,
    get_theme,
    list_themes,
    register_theme,
    blend_colors,
    adjust_brightness,
    hex_to_rgba,
)
from pyfsn.view.theme_manager import ThemeManager, get_theme_manager

# ModernGL resource management (Phase 1.2)
from pyfsn.view.shader_loader import (
    ShaderLoader,
    ShaderCompilationError,
    ProgramLinkError,
)
from pyfsn.view.buffer_manager import (
    BufferManager,
    BufferInfo,
    VertexArrayInfo,
)
from pyfsn.view.texture_manager import (
    TextureManager,
    TextureLoadError,
    TextureInfo,
)

# Picking system (Phase 1.4)
from pyfsn.view.picking import (
    Ray,
    AABB,
    ray_aabb_intersect,
    screen_to_ray,
    PickingSystem,
    raycast_find_node_with_camera,
)

# Spotlight search (Phase 3.2)
from pyfsn.view.spotlight import SpotlightSearch, SpotlightAnimation

__all__ = [
    "Renderer",
    "ModernGLRenderer",
    "Camera",
    "CameraMode",
    "CameraState",
    "CubeGeometry",
    "CubeInstance",
    "get_default_colors",
    "MainWindow",
    "TextOverlay",
    "ControlPanel",
    "SearchBar",
    "FileTreeWidget",
    # Theme exports
    "Theme",
    "SGI_CLASSIC",
    "DARK_MODE",
    "CYBERPUNK",
    "SOLARIZED",
    "FOREST",
    "OCEAN",
    "DEFAULT_THEME",
    "BUILTIN_THEMES",
    "get_theme",
    "list_themes",
    "register_theme",
    "blend_colors",
    "adjust_brightness",
    "hex_to_rgba",
    "ThemeManager",
    "get_theme_manager",
    # ModernGL resource management exports
    "ShaderLoader",
    "ShaderCompilationError",
    "ProgramLinkError",
    "BufferManager",
    "BufferInfo",
    "VertexArrayInfo",
    "TextureManager",
    "TextureLoadError",
    "TextureInfo",
    # Picking exports
    "Ray",
    "AABB",
    "ray_aabb_intersect",
    "screen_to_ray",
    "PickingSystem",
    "raycast_find_node_with_camera",
    # Spotlight search exports (Phase 3.2)
    "SpotlightSearch",
    "SpotlightAnimation",
]
