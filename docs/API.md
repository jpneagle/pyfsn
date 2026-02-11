# pyfsn API Documentation

## Overview

pyfsn (Python File System Navigator) is a 3D interactive file system visualization tool built with Python, PyQt6, and **PyOpenGL (Legacy OpenGL 2.1)**.

注:
- `pyfsn.view.cube_geometry` / `pyfsn.view.shaders` は ModernGL 向けの資産です（`ModernGLRenderer` で使用）。Legacy `Renderer` では未使用です。
- テーマシステム、ブルームエフェクト、ミニマップ、スポットライト検索等の追加モジュールについては本ドキュメント末尾を参照してください。

## Architecture

The application follows a Model-View-Controller (MVC) pattern:

### Model Layer (`pyfsn.model`)

#### `Node`
Represents a file or directory in the file system.

```python
from pyfsn.model import Node, NodeType

node = Node.from_path(Path("/some/path"))
print(node.type)  # NodeType.FILE, NodeType.DIRECTORY, or NodeType.SYMLINK
print(node.is_directory)  # True/False
print(node.children)  # List of child nodes
```

**Key Properties:**
- `path: Path` - Absolute path
- `name: str` - Base name
- `size: int` - Size in bytes
- `type: NodeType` - File type enum
- `permissions: int` - Unix permissions
- `mtime: float` - Modification time
- `children: list[Node]` - Child nodes
- `parent: Node | None` - Parent node
- `is_image_file: bool` - True if the node is an image file (PNG, JPG, GIF, etc.)
- `is_video_file: bool` - True if the node is a video file (MP4, AVI, MKV, etc.)

**Key Methods:**
- `Node.from_path(path: Path) -> Node` - Create node from filesystem
- `add_child(child: Node) -> None` - Add child to directory
- `get_all_descendants() -> list[Node]` - Get all descendants recursively

#### `Scanner`
Async filesystem scanner using QThread.

```python
from pyfsn.model import Scanner

scanner = Scanner()
worker = scanner.scan_async(
    Path("/some/path"),
    on_progress=lambda p: print(f"Scanning: {p.current_path}"),
    on_finished=lambda root: print(f"Found {len(root.children)} items"),
    on_error=lambda err: print(f"Error: {err}")
)
```

### View Layer (`pyfsn.view`)

#### `Renderer`
QOpenGLWidget subclass with **PyOpenGL (Legacy OpenGL 2.1)** rendering.

```python
from pyfsn.view import Renderer

renderer = Renderer()
renderer.load_layout(layout_result, nodes, selection)
renderer.set_camera_mode(CameraMode.ORBIT)
```

**Key Methods:**
- `load_layout(layout_result, nodes, selection)` - Load scene for rendering
- `set_camera_mode(mode)` - Set camera navigation mode
- `snap_camera_to_node(node_id, distance)` - Focus camera on node
- `raycast_find_node(x, y, nodes, positions)` - Find node at screen position
- `get_screen_position(world_position)` - Project world to screen coords

**Optimization Features:**
- **FrustumCuller**: 部分接続済み（`paintGL` で更新、`is_node_visible` で使用）
- **LevelOfDetail**: 部分接続済み（エッジ描画の距離スキップ、小キューブスキップ）
- **Wire Highlighting**: Selected connections are highlighted
- **Bloom / Emissive**: `SimpleBloom` による発光効果（接続済み）
- **Collision Detection**: Fly mode の AABB 衝突検出（`check_collision`）

#### `Camera`
3D camera with Orbit and Fly modes.

```python
from pyfsn.view import Camera, CameraMode

camera = Camera()

# Orbit mode (default)
camera.set_mode(CameraMode.ORBIT)
camera.orbit_rotate(dx, dy)
camera.orbit_zoom(delta)
camera.orbit_pan(dx, dy, width, height)

# Fly mode (FPS-style)
camera.set_mode(CameraMode.FLY)
camera.fly_move(dx=0, dy=0, dz=1.0)  # Move forward
camera.fly_look(dx, dy)  # Look around
move_vec = camera.get_fly_move_vector(dx, dy, dz, speed_multiplier=2.0)
```

**Modes:**
- `CameraMode.ORBIT` - Rotate around focal point
- `CameraMode.FLY` - FPS-style movement with WASD + mouse look

#### `MainWindow`
Main application window with controls and menu bar.

```python
from pyfsn.view import MainWindow
from pathlib import Path

window = MainWindow(Path("/some/path"))
window.show()
```

**Signals:**
- `directory_changed(Path)` - Emitted when user selects new directory
- `search_requested(str)` - Emitted when user searches

**Properties:**
- `renderer` - Access to the Renderer widget
- `text_overlay` - Access to text label overlay
- `file_tree` - Access to file tree widget
- `file_tooltip` - Access to ImagePreviewTooltip for media previews

#### `ImagePreviewTooltip`
Enhanced tooltip with image and video preview support.

```python
from pyfsn.view.main_window import ImagePreviewTooltip

tooltip = ImagePreviewTooltip(parent_widget)
tooltip.show_for_node(node, x, y)
```

**Features:**
- **Image Preview**: Shows scaled preview (max 320×240) for image files
- **Video Thumbnails**: Extracts and displays frame at 25% of video duration
- **Play Icon Overlay**: Visual indicator for video files
- **Caching**: Pre-scales and caches media for performance
- **Graceful Degradation**: Works without OpenCV (shows message instead of thumbnail)

**Supported Image Formats:**
PNG, JPG, JPEG, GIF, BMP, WebP, SVG, ICO, TIFF, PSD, RAW, HEIC, AVIF

**Supported Video Formats:**
MP4, AVI, MKV, MOV, WMV, FLV, WebM, M4V, MPG, MPEG, 3GP, OGV, TS, M2TS

**Methods:**
- `show_for_node(node, x, y)` - Show tooltip at position for node
- `hide()` - Hide the tooltip
- `clear_cache()` - Clear cached media data

**Dependencies:**
- Images: Qt built-in (QPixmap, QImage)
- Videos: opencv-python (optional)

### Controller Layer (`pyfsn.controller`)

#### `Controller`
Main application coordinator.

```python
from pyfsn.controller import Controller
from pathlib import Path

controller = Controller(Path("/some/path"))
controller.start()
controller.show()
```

**Key Methods:**
- `start()` - Begin scanning and loading
- `refresh()` - Refresh current view
- `navigate_to(path)` - Navigate to specific path
- `set_camera_mode(mode)` - Change camera mode

**Signals:**
- `scene_loaded(int)` - Emitted with node count when scene loads
- `node_selected(Node)` - Emitted when node is selected
- `scan_progress(str)` - Emitted with status messages

### Layout Layer (`pyfsn.layout`)

#### `LayoutEngine`
Calculates 3D positions for file system nodes.

```python
from pyfsn.layout import LayoutEngine, LayoutConfig

config = LayoutConfig(
    node_size=1.0,
    dir_size=2.0,
    spacing=0.5,
    max_depth=5
)
engine = LayoutEngine(config)
result = engine.calculate_layout(root_node)

# Access positions
position = result.positions[str(node.path)]
```

**Configuration:**
- `node_size: float` - Base size for file nodes
- `dir_size: float` - Base size for directory nodes
- `spacing: float` - Spacing between nodes
- `max_depth: int` - Maximum depth to visualize
- `placement_strategy` - Strategy for placing subdirectories

## Performance Optimization

### Utilities (`pyfsn.view.performance`)

`pyfsn.view.performance` に以下のユーティリティが存在します:

- `FrustumCuller`: 視錐台カリング（部分接続済み: `paintGL` 内で `update_from_camera` を呼び、`is_node_visible` で使用）
- `LevelOfDetail`: 距離ベース LOD（部分接続済み: エッジ描画の距離スキップとして使用）
- `ProgressiveLoader`: ノードをバッチで段階ロードするユーティリティ（未接続）

#### ProgressiveLoader example

```python
from pyfsn.view.performance import ProgressiveLoader

loader = ProgressiveLoader(batch_size=1000)
loader.batch_loaded.connect(lambda loaded, total: print(f"Loaded {loaded}/{total}"))
loader.loading_complete.connect(lambda: print("Loading complete"))
loader.start_loading(nodes_list)
```

### Performance Monitoring

Monitor rendering performance:

```python
from pyfsn.view.performance import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.metrics_updated.connect(lambda metrics: print(f"FPS: {metrics.fps}"))
```

## Controls Reference

### Mouse Controls (Orbit Mode)
- **Left-drag**: Rotate camera
- **Right-drag**: Pan camera
- **Shift+Left-drag**: Pan camera (macOS trackpad alternative)
- **Middle-drag**: Pan camera
- **Scroll**: Zoom in/out
- **Click**: Select node (Ctrl=toggle, Shift=add)
- **Double-click directory**: Focus on directory
- **Double-click file**: Open file with default application

### Mouse Controls (Fly Mode)
- **Left-drag**: Look around (rotate view)
- **Right-drag**: Look around (rotate view)
- **Click**: Select node
- **Double-click**: Navigate/Open

### Keyboard Controls (Fly Mode)
- **W/S**: Forward / Backward
- **A/D**: Strafe Left / Right
- **Q/E**: Down / Up
- **Shift** (hold): Sprint (2x speed)
- Flyモードはコントロールパネルのボタンで切り替え。衝突検出あり。

### Menu Shortcuts
- **Ctrl+O**: Open directory
- **Ctrl+T**: Toggle file tree
- **Ctrl+L**: Toggle labels
- **Ctrl+F**: Toggle filter panel
- **F5**: Refresh

## Extending pyfsn

### Custom Layout Strategies

```python
from pyfsn.layout import PlacementStrategy

class CustomStrategy:
    def calculate_positions(nodes, parent_pos):
        # Your custom layout logic
        return positions

# Configure via LayoutConfig(placement_strategy=...) and/or extend LayoutEngine.
```

### Custom Node Colors

```python
from pyfsn.view import get_default_colors

colors = get_default_colors()
colors["my_type"] = np.array([1.0, 0.5, 0.0, 1.0])
```

### Custom Shaders

`pyfsn/view/shaders.py` および `pyfsn/view/shaders/` ディレクトリには ModernGL 向けの GLSL シェーダーが含まれています。
`ModernGLRenderer` で使用されます。Legacy `Renderer`（PyOpenGL / Immediate Mode）では `SimpleBloom` による簡易エミッシブ効果が代わりに使用されています。

詳細は [ADVANCED_EFFECTS.md](ADVANCED_EFFECTS.md) を参照してください。

## Error Handling

```python
from pyfsn.errors import (
    ScanError,
    LayoutError,
    RenderError,
    ValidationError,
    handle_errors,
)

@handle_errors(error_types=(ScanError,), default_return=None)
def safe_scan(path):
    return scanner.scan(path)
```

## Additional Modules

### Theme System (`pyfsn.view.theme`, `pyfsn.view.theme_manager`)

テーマの定義と管理。

```python
from pyfsn.view.theme import Theme
from pyfsn.view.theme_manager import ThemeManager

manager = ThemeManager()
manager.set_theme("cyberpunk")  # SGI_CLASSIC, DARK_MODE, CYBERPUNK, SOLARIZED, FOREST, OCEAN
```

**Built-in Themes:**
- `SGI_CLASSIC` - 既存のSGI fsnスタイル配色
- `DARK_MODE` - 暗色背景
- `CYBERPUNK` - ネオンピンク/シアン
- `SOLARIZED` / `FOREST` / `OCEAN` - その他のプリセット

**Signals:**
- `theme_changed(Theme)` - テーマ変更時
- `bloom_changed(bool)` - ブルーム有効/無効時
- `grid_changed(bool)` - グリッド表示変更時

### Bloom & Emissive Effects (`pyfsn.view.bloom`)

```python
from pyfsn.view.bloom import SimpleBloom

bloom = SimpleBloom(intensity=0.3)
bloom.set_enabled(True)
r, g, b, a = bloom.apply_glow(r, g, b, a, emission)
```

詳細は [ADVANCED_EFFECTS.md](ADVANCED_EFFECTS.md) を参照。

### Spotlight Search (`pyfsn.view.spotlight`)

```python
from pyfsn.view.spotlight import SpotlightSearch

spotlight = SpotlightSearch()
spotlight.start_search("query")
# Matching nodes: full opacity, non-matching: dimmed to 30%
```

### Mini Map (`pyfsn.view.mini_map`)

```python
from pyfsn.view.mini_map import MiniMap

mini_map = MiniMap(parent_widget)
# 2D radar-style overview with camera frustum visualization
```

### ModernGL Renderer (`pyfsn.view.modern_renderer`)

Legacy `Renderer` の代替として、ModernGL ベースのインスタンシング描画を提供。

```python
from pyfsn.view.modern_renderer import ModernGLRenderer

renderer = ModernGLRenderer()
# Shader-based pipeline with instanced rendering
```

注: `ModernGLRenderer` を使用するには `pip install -e ".[modern]"` が必要です。

### Picking System (`pyfsn.view.picking`)

```python
from pyfsn.view.picking import PickingSystem, Ray, AABB, ray_aabb_intersect

picking = PickingSystem()
picking.set_scene_data(nodes, positions)
node = picking.pick(screen_x, screen_y, view_matrix, proj_matrix, width, height)
```

### Resource Managers

- **ShaderLoader** (`pyfsn.view.shader_loader`): シェーダーのコンパイル・キャッシュ管理
- **BufferManager** (`pyfsn.view.buffer_manager`): VBO/VAO/EBO の管理
- **TextureManager** (`pyfsn.view.texture_manager`): テクスチャのロード・管理

## Thread Safety

- Scanning operations run in a separate QThread
- UI updates must happen on the main thread
- Use Qt signals to communicate between threads
