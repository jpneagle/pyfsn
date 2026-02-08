# pyfsn - Python File System Navigator 技術仕様書

## 1. プロジェクト概要

### 1.1 目的

SGI IRIXのファイルシステム可視化ツール fsn をPythonで忠実に再現する。ディレクトリ構造を3D空間で可視化し、直感的なナビゲーションを提供する。

### 1.2 主な機能

| 機能 | 説明 |
|------|------|
| **3D可視化** | ディレクトリごとの浮遊プラットフォーム、ワイヤー接続、ファイルキューブ |
| **GPUレンダリング** | PyOpenGL (Legacy OpenGL 2.1 / Compatibility) による描画（固定機能＋即時モード） |
| **カメラモード** | Orbit（回転）- マウスドラッグで操作 |
| **ナビゲーション履歴** | Back/Forward（メニュー・プログラム内部で管理） |
| **リアルタイム検索** | ファイル名のインクリメンタルサーチ |
| **高度なフィルタリング** | サイズ・年齢・種別による絞り込み（FilterPanel） |
| **インタラクション** | スポットライト強調表示、ラベル表示、ファイル情報ツールチップ、ファイル年齢凡例 |
| **メディアプレビュー** | 画像・動画ファイルのサムネイルプレビュー（ホバー時表示） |
| **ファイルツリー** | 右ドックの階層ツリー（QTreeWidget）で選択・移動を補助 |

### 1.3 パフォーマンス目標

| 指標 | 目標値 |
|------|--------|
| フレームレート | 30 FPS以上（〜10,000ノード目安、環境依存） |
| 起動時間 | 2秒以内（ルート表示まで） |
| 対応ファイル数 | 数万ノード程度（現状: 即時モードのため環境依存、将来最適化余地あり） |

---

## 2. 技術スタック

### 2.1 採用技術

| コンポーネント | 技術 | 理由 |
|---------------|------|------|
| **言語** | Python 3.10+ | 型ヒント、Pattern Matching等の活用 |
| **GUIフレームワーク** | PyQt6 | ネイティブウィンドウ、メニュー、ダイアログ |
| **3Dレンダリング** | PyOpenGL 3.1+ | macOS等での互換性重視、Legacy OpenGL 2.1採用 |
| **将来/実験** | ModernGL 5.9+ | `CubeGeometry`/`shaders.py` は存在するが、現状の `Renderer` には未接続 |
| **数値計算** | numpy | 行列演算、座標計算の高速化 |
| **ファイル操作** | pathlib, os | 標準ライブラリ |

### 2.2 依存関係

```
pyfsn/
├── PyQt6>=6.6.0      # GUIフレームワーク
├── PyOpenGL>=3.1.0   # OpenGLバインディング（Legacy OpenGL 2.1サポート）
└── numpy>=1.24.0     # 数値計算
```

#### オプション依存関係

```
pyfsn/[video]  # 動画プレビュー機能
├── opencv-python>=4.8.0  # 動画サムネイル抽出

pyfsn/[legacy]  # GPUアクセラレーション
├── PyOpenGL-accelerate>=3.1.0

pyfsn/[modern]  # ModernGLレンダラー（実験中）
├── ModernGL>=5.9.0
├── moderngl-window>=2.4.0
└── scipy>=1.11.0
```

注:
- 現状の `src/pyfsn/view/renderer.py` は **PyOpenGL (Legacy OpenGL 2.1)** を使用。
- macOS環境との互換性を重視し、固定機能パイプライン + 即時モード描画を採用。
- `cube_geometry.py` / `shaders.py` は ModernGL 向けアセット（将来の最適化用）として保持されているが、現在のレンダラーには未接続。
- **メディアプレビュー**: 画像プレビューはQt組み込み機能（追加依存なし）、動画サムネイルはOpenCV（オプション）を使用。

---

## 3. システムアーキテクチャ

### 3.1 MVCパターン

```
┌─────────────────────────────────────────────────────────────┐
│                     Controller Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │   Controller │◄─┤InputHandler  │  │     Error        │ │
│  └───────┬──────┘  └──────────────┘  │    Handling      │ │
│          │                               └──────────────────┘ │
├──────────┼───────────────────────────────────────────────────┤
│          │                                               │
│  ┌───────▼──────┐                              ┌──────────▼───────┐
│  │ Model Layer  │                              │  View Layer      │
│  │ ┌──────────┐ │                              │ ┌──────────────┐ │
│  │ │   Node   │ │                              │ │   Renderer   │ │
│  │ ├──────────┤ │                              │ ├──────────────┤ │
│  │ │ Scanner  │ │                              │ │    Camera    │ │
│  │ └──────────┘ │                              │ ├──────────────┤ │
│  └──────────────┘                              │ │ CubeGeometry │ │
│                                                │ ├──────────────┤ │
│  ┌──────────────┐                              │ │ MainWindow   │ │
│  │Layout Engine │◄─────────────────────────────│ └──────────────┘ │
│  │ ┌──────────┐ │                              │ ┌──────────────┐ │
│  │ │ Position │ │                              │ │Performance   │ │
│  │ ├──────────┤ │                              │ │Monitoring    │ │
│  │ │Box/Bound │ │                              │ └──────────────┘ │
│  │ └──────────┘ │                              │                  │
│  └──────────────┘                              └──────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 ディレクトリ構造

```
pyfsn/
├── pyproject.toml           # プロジェクト設定
├── README.md                # ユーザーマニュアル
├── src/pyfsn/
│   ├── __init__.py
│   ├── __main__.py          # エントリーポイント
│   ├── run.py               # CLIエントリーポイント（pyfsn コマンド）
│   ├── errors.py            # 例外クラス定義
│   ├── model/               # データ層
│   │   ├── __init__.py
│   │   ├── node.py          # Nodeクラス
│   │   └── scanner.py       # 非同期スキャナー
│   ├── layout/              # レイアウト層
│   │   ├── __init__.py
│   │   ├── position.py      # 3D座標クラス
│   │   ├── box.py           # バウンディングボックス
│   │   └── engine.py        # レイアウト計算エンジン
│   ├── view/                # 描画層
│   │   ├── __init__.py
│   │   ├── renderer.py      # Legacy OpenGLレンダラー
│   │   ├── modern_renderer.py # ModernGLレンダラー（代替バックエンド）
│   │   ├── camera.py        # カメラシステム
│   │   ├── cube_geometry.py # ModernGL用インスタンシング
│   │   ├── shaders.py       # ModernGL用GLSLシェーダー定義
│   │   ├── shader_loader.py # シェーダーコンパイル・キャッシュ
│   │   ├── shaders/         # GLSLシェーダーファイル
│   │   │   ├── cube.vert/frag      # キューブシェーダー
│   │   │   ├── emissive.vert/frag  # エミッシブマテリアルシェーダー
│   │   │   ├── ground.vert/frag    # 地面シェーダー
│   │   │   ├── sky.vert/frag       # スカイグラデーションシェーダー
│   │   │   └── wire.vert/frag      # ワイヤー接続シェーダー
│   │   ├── bloom.py         # ブルーム＆エミッシブエフェクト
│   │   ├── spotlight.py     # スポットライト検索視覚化
│   │   ├── filter_panel.py  # 高度なフィルタリングパネル
│   │   ├── mini_map.py      # 2Dミニマップ（レーダービュー）
│   │   ├── theme.py         # テーマ定義（SGI Classic, Dark Mode, Cyberpunk等）
│   │   ├── theme_manager.py # テーマ管理＆永続化
│   │   ├── buffer_manager.py # VBO/VAO/EBO管理
│   │   ├── texture_manager.py # テクスチャ管理
│   │   ├── picking.py       # Ray-AABBピッキングシステム
│   │   ├── performance.py   # パフォーマンス監視
│   │   ├── effects_demo.py  # エフェクトデモアプリケーション
│   │   └── main_window.py   # メインウィンドウ
│   └── controller/          # 制御層
│       ├── __init__.py
│       ├── input_handler.py # 入力処理
│       └── controller.py    # メインコントローラー
├── tests/                   # テストスイート
│   ├── test_spotlight.py    # スポットライト機能の単体テスト
│   └── test_spotlight_demo.py  # スポットライト機能の対話型デモ
└── docs/
    ├── API.md               # APIドキュメント
    ├── SPEC.md              # 本技術仕様書
    ├── MEDIA_PREVIEW.md     # メディアプレビュー機能ドキュメント
    ├── ADVANCED_EFFECTS.md  # 高度なシェーダー/エフェクトドキュメント
    ├── EFFECTS_IMPLEMENTATION_SUMMARY.md # エフェクト実装サマリー
    └── screenshot.png       # アプリケーションスクリーンショット
```

---

## 4. 詳細機能仕様

### 4.1 可視化ルール (Visual Metaphor)

#### 4.1.1 視覚表現

| 要素 | 表現 | 意味 |
|------|------|------|
| **ディレクトリ** | 浮遊プラットフォーム | ファイル群を載せる台座、サブディレクトリの親 |
| **ファイル** | 直方体（キューブ） | プラットフォーム上に配置されるファイル |
| **高さ** | ファイルはサイズ対数スケール、プラットフォームは薄い板 | `LayoutEngine` が \(min=0.2, max=5.0\) の範囲で高さを算出。プラットフォーム厚みは `Renderer` 側で 0.2 固定 |
| **色** | 状態/種別/年齢に依存 | 既定は「更新時刻（年齢）に基づく色」＋選択/フォーカス色 |
| **階層接続** | ワイヤー（白線） | 親プラットフォームと子プラットフォームを接続 |
| **強調表示** | コーンスポットライト | 選択されたアイテム上空からの照明効果 |
| **メディアプレビュー** | ツールチップ内サムネイル | 画像・動画ファイルにホバー時にプレビュー表示 |

#### 4.1.2 カラースキーム

| タイプ | 色 (RGBA) |
|--------|-----------|
| ディレクトリ（プラットフォーム） | (0.2, 0.6, 1.0, 1.0) - 明るい青 |
| ファイル（既定） | 年齢ベース（例: 24h未満=緑, 1週未満=シアン, 1ヶ月未満=黄, 1年未満=橙, それ以上=茶） |
| シンボリックリンク | (0.3, 1.0, 0.3, 1.0) - 緑 |
| 選択 | (1.0, 1.0, 0.3, 1.0) - 黄 |
| フォーカス | (1.0, 0.8, 0.0, 1.0) - 金 |
| ワイヤー | (1.0, 1.0, 1.0, 0.8) - 白（やや透過） |
| スポットライト（コーン） | (1.0, 1.0, 0.8, 0.2) - 淡黄（透過） |

### 4.2 レイアウトアルゴリズム

#### 4.2.1 FSNスタイル 後退レイアウト (Receding Layout)

サブディレクトリは**負のZ方向（画面奥）**に配置され、階層が深くなるほど遠ざかるFSN特有のビジュアルを再現。

```
カメラ (Z+)
    ↓
[Root Platform]  ← Z = 0付近
      │ (Wire)
      ▼
   [Child]       ← Z = -10付近
      │
      ▼
  [Grandchild]   ← Z = -20付近
```

#### 4.2.2 動的プラットフォームサイジング

プラットフォームサイズはファイル数から自動計算:

```
Grid_Cols = Ceil(Sqrt(ファイル数))
Cell_Size = File_Width + Spacing
Platform_Width = (Grid_Cols * Cell_Size) + (Padding * 2)
```

#### 4.2.3 ワイヤー接続（エッジ接続）

親プラットフォームの**背面エッジ**（Min Z）から子プラットフォームの**前面エッジ**（Max Z）にワイヤーを接続し、プラットフォームを貫通しない。

#### 4.2.4 2Dデンドログラム (Overview)

現状のUIは **2Dデンドログラム表示は未実装**。代わりに `MainWindow` の **ファイルツリードック（QTreeWidget）** をナビゲーション補助として利用する。

### 4.3 インタラクション

#### 4.3.1 ナビゲーション

| 入力 | アクション |
|------|-----------|
| 左ドラッグ | カメラ回転（Orbitモード） |
| 右ドラッグ | パン |
| Shift+左ドラッグ | パン（macOSトラックパッド代替） |
| 中ドラッグ | パン |
| ホイール | ズーム |
| 左クリック | 3Dビューでノード選択（Ray-AABBピッキング） |
| ダブルクリック | ディレクトリへフォーカス＋選択 |
| ファイルをダブルクリック | ファイルをOSデフォルトアプリで開く |
| **ファイルツリーでダブルクリック** | **ファイルを開く/ディレクトリを移動** |

補足:
- `Renderer` は Ray-AABB intersection（Slab method）による正確な3Dピッキングを実装済み。
- `snap_camera_to_node()` はイージングアニメーション（300ms）付きで実装済み。
- ノードの選択/移動は **3Dビュー** と **ファイルツリードック** の両方から可能。
- **右ドラッグパン**: 実装済み。2ボタンマウスやトラックパッドユーザー向け。
- **Shift+左ドラッグパン**: macOSトラックパッドユーザー向け代替操作として実装済み。

#### 4.3.2 キーボード操作

現在、キーボードショートカットは未実装です。すべての操作はマウスとメニューバーから行います。

| ショートカット | 機能 |
|------|------|
| Ctrl+O | ディレクトリを開く |
| F5 | 現在のディレクトリを再スキャン |
| Ctrl+T | ファイルツリーパネルの表示切り替え |
| Ctrl+F | フィルタパネルの表示切り替え |
| Ctrl+L | ノードラベルの表示切り替え |
| Ctrl+Q | アプリケーション終了 |

注: 3Dビュー内でのキーボードによるカメラ操作（F, R, Esc等）は現在未実装です。

---

## 5. 各層の詳細仕様

### 5.1 Model Layer

#### 5.1.1 Nodeクラス

```python
@dataclass
class NodeType(Enum):
    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"

@dataclass
class Node:
    """ファイルシステムノードを表す"""

    path: Path                    # ファイルパス
    name: str                     # ファイル名
    size: int                     # バイト単位
    type: NodeType                # ノードタイプ
    permissions: int              # ファイルパーミッション
    mtime: float                  # 更新時刻（Unixタイムスタンプ）
    children: list[Node]          # 子ノード（ディレクトリのみ）
    parent: Node | None = None    # 親ノード
    is_loaded: bool = False       # 子ノードがロード済みか
    is_expanded: bool = False     # UIで展開されているか

    @classmethod
    def from_path(cls, path: Path, parent: Self | None = None) -> Self: ...

    @property
    def is_directory(self) -> bool: ...

    @property
    def is_file(self) -> bool: ...

    @property
    def is_symlink(self) -> bool: ...

    @property
    def is_image_file(self) -> bool:
        """画像ファイルかどうかを判定（PNG, JPG, GIF, WebP, SVG, HEIC, AVIF等）"""
        ...

    @property
    def is_video_file(self) -> bool:
        """動画ファイルかどうかを判定（MP4, AVI, MKV, MOV, WebM等）"""
        ...

    @property
    def depth(self) -> int: ...

    @property
    def total_size(self) -> int: ...

    @property
    def file_count(self) -> int: ...

    @property
    def directory_count(self) -> int: ...

    def add_child(self, child: Node) -> None: ...

    def remove_child(self, child: Node) -> bool: ...

    def find_child_by_name(self, name: str) -> Node | None: ...

    def get_all_descendants(self) -> list[Node]: ...

    def invalidate_children(self) -> None: ...
```

#### 5.1.2 Scannerクラス

```python
class ScanProgress:
    """スキャン進捗情報"""

    current_path: Path            # 現在スキャン中のパス
    nodes_found: int              # 発見されたノード数
    is_complete: bool = False     # スキャン完了フラグ
    error: str | None = None      # エラーメッセージ

class ScannerWorker(QThread):
    """非同期ファイルシステムスキャナー（ワーカースレッド）"""

    progress = pyqtSignal(object)  # ScanProgress
    node_found = pyqtSignal(Node)  # 新規ノード発見
    finished = pyqtSignal(Node)    # ルートノード
    error = pyqtSignal(str)        # エラーメッセージ

    def __init__(
        self,
        root_path: Path,
        lazy_load: bool = True,
        lazy_depth: int = 2,
        max_workers: int = 4,
    ): ...

    def run(self) -> None: ...
    def stop(self) -> None: ...

class Scanner:
    """ハイレベルスキャナーインターフェース"""

    def __init__(self, lazy_load: bool = True, lazy_depth: int = 2): ...

    def scan(self, path: Path) -> Node:
        """同期スキャン（再帰ロード対応）"""
        ...

    def scan_async(
        self,
        path: Path,
        on_progress: Callable[[ScanProgress], None] | None = None,
        on_node_found: Callable[[Node], None] | None = None,
        on_finished: Callable[[Node], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> ScannerWorker:
        """非同期スキャン（再帰ロード対応）
        
        Returns:
            ScannerWorker: ワーカースレッド
        """
        ...

    def stop(self) -> None: ...

    def load_children(self, node: Node) -> None:
        """遅延ロード：ノードの子を読み込み（再帰ロード対応）"""
        ...
```

### 5.2 Layout Engine

#### 5.2.1 Positionクラス

```python
@dataclass
class Position:
    """3D空間での位置と寸法"""

    x: float      # X座標
    y: float      # Y座標（高さ）
    z: float      # Z座標
    width: float  # 幅
    height: float # 高さ
    depth: float  # 奥行き

    @property
    def center(self) -> tuple[float, float, float]: ...

    def contains(self, point: tuple[float, float, float]) -> bool: ...
```

#### 5.2.2 LayoutEngineクラス

```python
class PlacementStrategy(Enum):
    """サブディレクトリ配置戦略"""
    GRID = "grid"            # グリッド配置
    RADIAL = "radial"        # 放射配置
    SPIRAL = "spiral"        # スパイラル配置
    HEURISTIC = "heuristic"  # ヒューリスティック配置

@dataclass
class LayoutConfig:
    """レイアウトエンジン設定"""
    node_size: float = 1.0                         # ファイルノードの基本サイズ
    dir_size: float = 2.0                          # ディレクトリノードの基本サイズ
    spacing: float = 0.5                           # ノード間のスペーシング
    padding: float = 0.2                           # パディング
    max_depth: int = 5                             # 最大探索深度
    placement_strategy: PlacementStrategy = PlacementStrategy.GRID
    grid_size: float = 3.0                         # グリッドサイズ
    connection_width: float = 0.1                  # 接続線の幅

@dataclass
class LayoutResult:
    """レイアウト計算結果"""
    positions: dict[str, Position]       # パス → 位置のマッピング
    connections: list[tuple[str, str]]   # 接続（親子パスのペア）
    bounds: BoundingBox | None = None    # 全体のバウンディングボックス

class LayoutEngine:
    """3Dレイアウト計算エンジン"""

    def __init__(self, config: LayoutConfig | None = None): ...

    def calculate_layout(self, root: Node) -> LayoutResult:
        """ノードツリーからレイアウトを計算"""
        ...

    def _create_root_position(self, node: Node) -> Position: ...

    def _position_children(
        self, parent: Node, parent_pos: Position, result: LayoutResult
    ) -> None: ...

    def _position_directories(
        self, directories: list[Node], parent_pos: Position, result: LayoutResult
    ) -> None: ...

    def _position_files(
        self, files: list[Node], parent_pos: Position, result: LayoutResult
    ) -> None: ...

    def _calculate_grid_positions(
        self, nodes: list[Node], parent_pos: Position, on_platform: bool = False
    ) -> list[Position]: ...

    def _calculate_radial_positions(
        self, nodes: list[Node], parent_pos: Position
    ) -> list[Position]: ...

    def _calculate_spiral_positions(
        self, nodes: list[Node], parent_pos: Position
    ) -> list[Position]: ...

    def _calculate_heuristic_positions(
        self, nodes: list[Node], parent_pos: Position
    ) -> list[Position]: ...

    def _resolve_collision(self, pos: Position) -> Position: ...
```

### 5.3 View Layer

#### 5.3.1 Cameraクラス

```python
class CameraMode(Enum):
    ORBIT = "orbit"   # 注視点周りの回転

@dataclass
class CameraState:
    position: np.ndarray    # [x, y, z] 例: [0, 10, 20] (FSN: 高角度)
    target: np.ndarray      # 注視点 [x, y, z] 例: [0, 0, -50] (遠方を見る)
    up: np.ndarray          # 上方向ベクトル
    fov: float = 45.0       # 視野角
    near: float = 0.1       # 近クリップ面
    far: float = 1000.0     # 遠クリップ面

class Camera:
    """3Dカメラシステム（Orbitモードのみ）"""

    def __init__(self) -> None: ...

    @property
    def view_matrix(self) -> np.ndarray: ...

    @property
    def projection_matrix(self, aspect_ratio: float) -> np.ndarray: ...

    def set_mode(self, mode: CameraMode) -> None: ...

    def orbit_rotate(self, dx: int, dy: int) -> None: ...

    def orbit_zoom(self, delta: float) -> None: ...

    def orbit_pan(self, dx: int, dy: int, view_width: int, view_height: int) -> None: ...

    def set_position_target(self, position: np.ndarray, target: np.ndarray) -> None: ...

    def get_ray_direction(
        self, screen_x: float, screen_y: float, width: int, height: int
    ) -> np.ndarray: ...
```

注:
- 現在はOrbitモードのみをサポート。Fly/Snapモードは削除済み。
- ビュープリセット（Bird's Eye、Front view）は未実装。

#### 5.3.2 CubeGeometryクラス（ModernGL用アセット）

```python
@dataclass
class CubeInstance:
    position: np.ndarray  # [x, y, z]
    scale: np.ndarray     # [width, height, depth]
    color: np.ndarray     # [r, g, b, a]

class CubeGeometry:
    """GPUインスタンシングによるキューブ描画（ModernGL用）"""

    VERTICES: np.ndarray   # キューブの頂点（8個）
    INDICES: np.ndarray    # インデックス（36個、12三角形）
    NORMALS: np.ndarray    # 法線ベクトル

    def __init__(self, ctx) -> None: ...

    def add_instance(self, instance: CubeInstance) -> int: ...

    def update_instance(self, index: int, instance: CubeInstance) -> None: ...

    def upload_instances(self) -> None: ...

    def create_vertex_array(self, program) -> VertexArray: ...

    def render(self, vao: VertexArray) -> None: ...
```

注: 現在のRenderer（Legacy OpenGL 2.1）では未使用。将来の高速化パス実装時に利用予定。

#### 5.3.3 Rendererクラス

```python
@dataclass
class CubeInstance:
    """キューブインスタンスデータ"""
    position: np.ndarray  # [x, y, z]
    scale: np.ndarray     # [width, height, depth]
    color: np.ndarray     # [r, g, b, a]
    shininess: float = 30.0  # スペキュラー光沢度（0-128）、ファイルキューブのデフォルト

class Renderer(QOpenGLWidget):
    """OpenGLレンダリングウィジェット"""

    # シグナル
    node_clicked = pyqtSignal(Node, bool)   # (ノード, ダブルクリックか)
    node_focused = pyqtSignal(Node)         # フォーカスされたノード
    selection_changed = pyqtSignal(set)     # 選択パスのセット

    def __init__(self, parent: QWidget = None) -> None: ...

    def initializeGL(self) -> None: ...

    def paintGL(self) -> None: ...

    def resizeGL(self, w: int, h: int) -> None: ...

    def load_layout(
        self,
        layout_result: LayoutResult,
        nodes: dict[str, Node],
        selection: set[str] | None = None,
    ) -> None: ...

    def get_node_at_position(self, x: int, y: int) -> Node | None:
        """レイキャストで指定座標のノードを取得（実装済み）"""
        ...

    def raycast_find_node(self, x: int, y: int, nodes: dict[str, Node], positions: dict[str, Position]) -> Node | None:
        """Ray-AABB intersection による3Dピッキング（実装済み: Slab method使用）"""
        ...

    def snap_camera_to_node(self, node_id: int, distance: float | None = None) -> None:
        """カメラをノードにスナップ（実装済み: 300msイージングアニメーション付き）"""
        ...

    def set_selection(self, selected_paths: set[str], nodes: dict[str, Node]) -> None: ...

    def select_node(self, path: str) -> None: ...

    def clear_selection(self) -> None: ...

    def focus_node(self, path: str) -> None: ...

    def set_camera_mode(self, mode: CameraMode) -> None: ...

    def get_screen_position(self, world_pos: np.ndarray) -> tuple[int, int] | None:
        """3D位置をスクリーン座標に変換（ラベル表示用）"""
        ...

    @property
    def performance_stats(self) -> dict:
        """パフォーマンス統計を取得"""
        ...
```

**実装済み描画機能（Sprint 1）:**
- **デプスフォグ**: `glFog(GL_FOG_MODE, GL_LINEAR)` による距離フェード
- **地面グリッド**: `_draw_ground_grid()` による等間隔グリッドライン
- **スカイグラデーション**: `_draw_sky_gradient()` によるフルスクリーングラデーション
- **キューブエッジ**: `_draw_cube_edges()` によるエッジハイライト（`glPolygonOffset` でZ-fighting回避）

注: ライティングは `glDisable(GL_LIGHTING)` により無効化されています（v1.6.0で無効化）。色は直接指定（`glColor4f`）で設定されます。エミッシブ効果は `SimpleBloom.apply_glow()` により色を加算ブースト方式で実現しています。

### 5.4 Controller Layer

#### 5.4.1 InputHandlerクラス

```python
class MouseButton:
    """マウスボタン識別子"""
    LEFT = Qt.MouseButton.LeftButton
    MIDDLE = Qt.MouseButton.MiddleButton
    RIGHT = Qt.MouseButton.RightButton

class KeyModifier:
    """キーボード修飾キー識別子"""
    SHIFT = Qt.KeyboardModifier.ShiftModifier
    CONTROL = Qt.KeyboardModifier.ControlModifier
    ALT = Qt.KeyboardModifier.AltModifier
    META = Qt.KeyboardModifier.MetaModifier

@dataclass
class InputState:
    """現在の入力状態"""
    mouse_position: QPoint = QPoint(0, 0)
    mouse_pressed: set[MouseButton] = None
    modifiers: set[KeyModifier] = None
    last_mouse_position: QPoint = None
    mouse_drag_start: QPoint = None
    is_dragging: bool = False
    drag_threshold: int = 5

    def update_modifiers(self, modifiers: Qt.KeyboardModifiers) -> None: ...

    @property
    def shift_pressed(self) -> bool: ...

    @property
    def control_pressed(self) -> bool: ...

@dataclass
class Action:
    """入力アクション定義"""
    name: str
    mouse_button: MouseButton | None = None
    key: int | None = None
    modifiers: set[KeyModifier] | None = None
    on_press: Callable[[], None] | None = None
    on_release: Callable[[], None] | None = None
    on_drag: Callable[[int, int], None] | None = None
    on_click: Callable[[int, int], None] | None = None
    on_double_click: Callable[[int, int], None] | None = None

class InputHandler:
    """マウス/キーボード入力処理"""

    def __init__(self, camera: Camera, renderer: object) -> None: ...

    def set_scene_data(self, nodes: dict[str, Node], positions: dict[str, object]) -> None: ...

    def set_node_clicked_callback(self, callback: Callable[[Node, bool], None]) -> None: ...

    def set_node_focused_callback(self, callback: Callable[[Node], None]) -> None: ...

    def set_selection_changed_callback(self, callback: Callable[[set[Node]], None]) -> None: ...

    def mouse_press_event(self, event: QMouseEvent) -> bool: ...

    def mouse_release_event(self, event: QMouseEvent) -> bool: ...

    def mouse_move_event(self, event: QMouseEvent) -> bool: ...

    def wheel_event(self, event: QWheelEvent) -> bool: ...

    def mouse_double_click_event(self, event: QMouseEvent) -> bool: ...

    def key_press_event(self, event: QKeyEvent) -> bool: ...

    def key_release_event(self, event: QKeyEvent) -> bool: ...
```

#### 5.4.2 Controllerクラス

```python
class Controller(QObject):
    """メインアプリケーションコントローラー"""

    # シグナル
    scene_loaded = pyqtSignal(int)           # ノード数
    node_selected = pyqtSignal(object)       # 選択されたノード
    node_focused = pyqtSignal(object)        # フォーカスされたノード
    scan_progress = pyqtSignal(str)          # スキャン進捗メッセージ
    scan_complete = pyqtSignal()             # スキャン完了
    navigation_state_changed = pyqtSignal(bool, bool)  # (can_go_back, can_go_forward)

    def __init__(self, root_path: Path) -> None: ...

    def start(self) -> None:
        """アプリケーション開始（スキャン開始）"""
        ...

    def show(self) -> None:
        """メインウィンドウ表示"""
        ...

    def refresh(self) -> None: ...

    def navigate_to(self, path: Path) -> None: ...

    def go_back(self) -> None: ...

    def go_forward(self) -> None: ...

    def can_go_back(self) -> bool: ...

    def can_go_forward(self) -> bool: ...

    def next_search_result(self) -> None: ...

    def previous_search_result(self) -> None: ...
    
    def _apply_filters(self, filters: dict) -> None:
        """フィルタを適用してノードセットを絞り込む"""
        ...
    
    def _load_filtered_scene(self) -> None:
        """フィルタされたノードのみを描画する

        文脈保持機能:
        - connections維持: フィルタ後のノード間の接続を保持
        - 祖先ノード包含: マッチしたノードの親ディレクトリを自動的に表示
        """
        ...

    @property
    def window(self) -> MainWindow: ...

    @property
    def root_node(self) -> Node | None: ...

    def _on_scan_complete(self, root_node: Node) -> None: ...

    def _on_scan_progress(self, progress: ScanProgress) -> None: ...

    def _on_node_clicked(self, node: Node, is_double_click: bool) -> None: ...

    def _on_tree_node_selected(self, node: Node) -> None: ...
```

### 5.5 MainWindow Layer

#### 5.5.1 MainWindowクラス

```python
class TextOverlay(QWidget):
    """3Dビューの上にテキストラベルを描画するウィジェット"""

    def set_labels(self, labels: list[tuple[str, int, int]]) -> None: ...

    def set_focused_label(self, text: str | None, x: int | None, y: int | None) -> None: ...

    def clear(self) -> None: ...

class FileTooltipOverlay(QWidget):
    """ホバー位置にファイル情報を表示するツールチップ（SGI fsn風）"""

    def show_for_node(self, node: Node, x: int, y: int) -> None: ...
    def hide_tooltip(self) -> None: ...

class ImagePreviewTooltip(FileTooltipOverlay):
    """画像・動画プレビュー機能付きツールチップ

    画像ファイル（PNG, JPG等）にホバー時にサムネイルを表示。
    動画ファイル（MP4, AVI等）にホバー時にサムネイルを表示（OpenCV使用時）。

    機能:
    - 最大320×240ピクセルのプレビュー
    - アスペクト比保持
    - キャッシュ機能（パフォーマンス向上）
    - 動画には再生アイコンオーバーレイ
    - OpenCVなしでの動作可能（動画はメッセージ表示）
    """

    def show_for_node(self, node: Node, x: int, y: int) -> None: ...
    def hide(self) -> None: ...
    def clear_cache(self) -> None: ...

class FileAgeLegend(QWidget):
    """ファイル年齢の凡例を左下に表示するオーバーレイ"""

    def paintEvent(self, event) -> None: ...

class FilterPanel(QWidget):
    """高度なフィルタリングパネル

    サイズ範囲、年齢範囲、ファイル種別による絞り込み機能を提供。

    文脈保持機能:
    - include_ancestors: マッチしたノードの親ディレクトリを自動的に含める
    """

    filter_changed = pyqtSignal(dict)  # フィルタ条件の変更を通知

    def __init__(self, parent=None) -> None: ...
    def get_filters(self) -> dict: ...
    def clear_filters(self) -> None: ...

class ControlPanel(QWidget):
    """カメラ・ビューコントロールパネル"""

    # シグナル
    refresh_requested = pyqtSignal()
    navigate_up = pyqtSignal()
    navigate_home = pyqtSignal()
    show_tree_toggled = pyqtSignal(bool)
    filter_panel_toggled = pyqtSignal(bool)
    labels_toggled = pyqtSignal(bool)

    def update_stats(self, node_count: int, selected_count: int = 0) -> None: ...

    def set_camera_mode_display(self, mode: CameraMode) -> None:
        """カメラモード表示を更新する"""
        ...

class SearchBar(QLineEdit):
    """ファイル/フォルダ検索バー"""

    search_requested = pyqtSignal(str)

class FileTreeWidget(QTreeWidget):
    """ファイル階層ツリー表示ウィジェット"""

    node_selected = pyqtSignal(object)  # Node

    def load_tree(self, root_node: Node) -> None: ...

    def select_node(self, node: Node) -> None: ...

class MainWindow(QMainWindow):
    """メインアプリケーションウィンドウ"""

    # シグナル
    directory_changed = pyqtSignal(Path)
    search_requested = pyqtSignal(str)
    refresh_requested = pyqtSignal()
    go_back_requested = pyqtSignal()
    go_forward_requested = pyqtSignal()
    filter_changed = pyqtSignal(dict)

    def __init__(self, root_path: Path) -> None: ...

    @property
    def renderer(self) -> Renderer: ...

    @property
    def text_overlay(self) -> TextOverlay: ...

    @property
    def file_tooltip(self) -> FileTooltipOverlay | None: ...

    @property
    def file_tree(self) -> FileTreeWidget: ...

    def update_stats(self, node_count: int, selected_count: int = 0) -> None: ...
    def set_status_message(self, message: str) -> None: ...
    def set_root_path(self, path: Path) -> None: ...
```

---

## 6. パフォーマンス最適化

### 6.1 現状の描画方式（PyOpenGL Legacy OpenGL 2.1）

```
Legacy OpenGL (Compatibility Profile):
- `Renderer` は `glBegin(GL_QUADS)` によるキューブ描画と、`glBegin(GL_LINES)` によるワイヤー描画を行う。
- 即時モード（Immediate mode）中心の実装で、描画負荷はノード数に比例して増加する。
- macOS/古いGPU環境との互換性を最優先した設計。
- ライティングは固定機能パイプライン（GL_LIGHTING、GL_LIGHT0、GL_LIGHT1）を使用。
```

実装済み機能:
- Ray-AABB ピッキング（Slab method）
- カメラスナップアニメーション（イージング付き）
- スポットライト強調表示（選択ノード）
- 年齢ベースの色分け（AGEモード）
- **デプスフォグ**（Sprint 1）
- **地面グリッド**（Sprint 1）
- **スカイグラデーション**（Sprint 1）
- **キューブエッジハイライト**（Sprint 1）
- **2ライトライティング**（Sprint 1で実装、v1.6.0で無効化）
- **スペキュラーハイライト**（Sprint 1で実装、v1.6.0で無効化）
- **エミッシブエフェクト**（SimpleBloomによる発光効果）
- **ワイヤーパルスアニメーション**（接続線のサイバーパンク風アニメーション）

将来の最適化候補:
- `view/cube_geometry.py` と `view/shaders.py` は ModernGL のインスタンシング経路向けに実装済み。
- ModernGL バックエンドを追加することで、10万ノード規模でも高FPSを実現可能（現状は未接続）。

### 6.2 パフォーマンス監視

```python
@dataclass
class PerformanceMetrics:
    """レンダリングパフォーマンス指標"""
    fps: float = 0.0
    frame_time_ms: float = 0.0
    draw_calls: int = 0
    instance_count: int = 0
    visible_instances: int = 0
    triangle_count: int = 0

class PerformanceMonitor(QObject):
    """パフォーマンス監視"""

    metrics_updated = pyqtSignal(object)  # PerformanceMetrics

    def __init__(self, window_size: int = 60) -> None: ...

    def start_frame(self) -> None: ...

    def end_frame(self) -> float: ...

    def set_draw_calls(self, count: int) -> None: ...

    def set_instance_count(self, count: int) -> None: ...

    def set_visible_instances(self, count: int) -> None: ...

    @property
    def metrics(self) -> PerformanceMetrics: ...

    @property
    def current_fps(self) -> float: ...
```

### 6.3 視錐台カリング

```python
class FrustumCuller:
    """視錐台カリングで画面外のノードをスキップ"""

    def __init__(self) -> None: ...

    def update_from_camera(
        self,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
        aspect_ratio: float,
    ) -> None:
        """カメラ行列から視錐台平面を更新"""
        ...

    def is_box_visible(
        self,
        min_bounds: np.ndarray,
        max_bounds: np.ndarray,
    ) -> bool:
        """バウンディングボックスが視界内にあるか判定"""
        ...

    def is_sphere_visible(
        self,
        center: np.ndarray,
        radius: float,
    ) -> bool:
        """球体が視界内にあるか判定"""
        ...
```

### 6.4 レベルオブディテール (LOD)

```python
class LevelOfDetail:
    """距離に応じた描画品質の調整"""

    def __init__(self, distances: list[float] | None = None) -> None: ...

    def get_lod_level(self, distance: float) -> int:
        """距離からLODレベルを取得（0=最高品質）"""
        ...

    def should_render(self, distance: float, max_distance: float = 200.0) -> bool:
        """描画すべきか判定"""
        ...

    def get_lod_scale(self, lod_level: int) -> float:
        """LODレベルのスケール係数を取得"""
        ...
```

### 6.5 プログレッシブローディング

```python
class ProgressiveLoader(QObject):
    """大規模ディレクトリの段階的ロード"""

    batch_loaded = pyqtSignal(int, int)  # (loaded_count, total_count)
    loading_complete = pyqtSignal()

    def __init__(self, batch_size: int = 1000, batch_delay_ms: int = 16) -> None: ...

    def start_loading(self, nodes: list[tuple[str, object]]) -> None: ...

    def cancel(self) -> None: ...

    @property
    def progress(self) -> tuple[int, int]: ...

    @property
    def is_loading(self) -> bool: ...
```

---

## 7. テスト

### 7.1 テストスイート構成

現在のテストディレクトリ構成:

```
tests/
├── test_spotlight.py       # スポットライト検索の単体テスト
└── test_spotlight_demo.py  # スポットライト機能の対話型デモ
```

### 7.2 単体テスト

**test_spotlight.py** - スポットライト検索機能の単体テスト

```python
# テスト実行
python tests/test_spotlight.py

# 期待される出力
✓ Spotlight initialization test passed
✓ Spotlight start search test passed
✓ Spotlight opacity test passed
✓ Spotlight desaturation test passed
✓ Spotlight clear test passed
✓ Spotlight animation test passed
✓ Spotlight parameters test passed
✓ Spotlight update results test passed
```

テスト項目:
- 初期化テスト
- 検索開始テスト
- 不透明度計算テスト
- 彩度低下テスト
- クリアテスト
- アニメーションテスト
- パラメータテスト
- 結果更新テスト

### 7.3 対話型デモ

**test_spotlight_demo.py** - スポットライト機能の対話型デモ

```python
# デモ実行
python tests/test_spotlight_demo.py
```

デモ内容:
1. pyfsnソースディレクトリをロード
2. 'view' ファイルの検索（ビュー関連ファイルをハイライト）
3. '.py' ファイルの検索（Pythonファイル）
4. 'controller' ディレクトリの検索
5. 検索結果間のナビゲーション
6. 検索のクリア

### 7.4 今後のテスト拡張

| モジュール | テスト予定 | 優先度 |
|-----------|-----------|--------|
| Node | ファイル種別検出、プロパティ | 中 |
| LayoutEngine | レイアウト計算、配置戦略 | 中 |
| Renderer | レンダリング、ピッキング | 低 |
| Camera | カメラ操作、ビュー行列 | 低 |
| Scanner | 非同期スキャン、エラーハンドリング | 中 |
| Controller | ナビゲーション、検索統合 | 低 |

---

## 8. エラーハンドリング

### 7.1 例外クラス

```python
class PyfsnError(Exception):
    """ベース例外クラス"""

class ScanError(PyfsnError):
    """スキャンエラー"""
    path: Path
    reason: str

    def __init__(self, path: Path, reason: str) -> None: ...

class LayoutError(PyfsnError):
    """レイアウトエラー"""
    reason: str

    def __init__(self, reason: str) -> None: ...

class RenderError(PyfsnError):
    """レンダリングエラー"""
    reason: str

    def __init__(self, reason: str) -> None: ...

class ValidationError(PyfsnError):
    """入力検証エラー"""
    field: str
    value: object
    expected: str

    def __init__(self, field: str, value: object, expected: str) -> None: ...
```

### 7.2 ヘルパー関数

```python
def safe_path(path: Path | str) -> Path:
    """安全にPathオブジェクトに変換"""
    ...

def validate_directory(path: Path) -> None:
    """パスがディレクトリであることを検証"""
    ...

def validate_range(
    value: int | float,
    min_val: int | float,
    max_val: int | float,
    name: str = "value"
) -> None:
    """値が範囲内であることを検証"""
    ...

def handle_errors(
    error_types: tuple[type[Exception], ...] = (Exception,),
    default_return: object = None,
    on_error: Callable[[Exception], None] | None = None,
) -> Callable:
    """エラーハンドリングデコレーター"""
    ...
```

### 7.3 エラー処理戦略

| エラー種別 | 対処 |
|-----------|------|
| パーミッション拒否 | 該当ディレクトリをスキップ |
| 壊れたシンボリックリンク | 警告表示してスキップ |
| メモリ不足 | ロード深度を制限 |
| OpenGL初期化失敗 | エラーダイアログを表示して終了 |
| 入力検証エラー | ValidationError例外を送出 |

---

## 9. シェーダー仕様（ModernGL向けアセット）

この章のGLSLは `view/shaders.py` に存在する **ModernGL向け資産（現状: 未接続）** を記述する。現状の `Renderer` は PyOpenGL 固定機能パイプライン（OpenGL 2.1）を使用しているため、ここで示すシェーダーは実行経路としては使用されない。

将来的にModernGLバックエンドを追加する際は、これらのシェーダーを利用してGPUインスタンシングによる高速描画が可能。

### 8.1 頂点シェーダー

```glsl
#version 330

// 頂点属性
in vec3 in_position;      // キューブ頂点
in vec3 in_normal;        // 法線

// インスタンス属性
in vec3 in_instance_position;  // インスタンス位置
in vec3 in_instance_scale;     // インスタンスサイズ
in vec4 in_instance_color;     // インスタンス色

// Uniform
uniform mat4 u_view;
uniform mat4 u_projection;
uniform mat4 u_model;

// フラグメントシェーダーへ
out vec3 v_normal;
out vec3 v_position;
out vec4 v_color;

void main() {
    // スケール適用
    vec3 scaled = in_position * in_instance_scale;
    // 移動適用
    vec3 world_pos = scaled + in_instance_position;

    vec4 view_pos = u_view * u_model * vec4(world_pos, 1.0);
    gl_Position = u_projection * view_pos;

    v_normal = mat3(u_model) * in_normal;
    v_position = world_pos;
    v_color = in_instance_color;
}
```

### 8.2 フラグメントシェーダー

```glsl
#version 330

in vec3 v_normal;
in vec3 v_position;
in vec4 v_color;

out vec4 frag_color;

void main() {
    vec3 normal = normalize(v_normal);

    // 方向光（右上前方から）
    vec3 light_dir = normalize(vec3(1.0, 1.0, 1.0));

    // 環境光 + 拡散光
    float ambient = 0.4;
    float diffuse = max(dot(normal, light_dir), 0.0) * 0.6;
    float lighting = ambient + diffuse;

    vec3 lit_color = v_color.rgb * lighting;

    // エッジの暗め（奥行き感）
    float edge = 1.0 - abs(dot(normal, vec3(0.0, 0.0, 1.0)));
    lit_color *= (1.0 - edge * 0.1);

    frag_color = vec4(lit_color, v_color.a);

    if (frag_color.a < 0.01) discard;
}
```

---

## 10. 拡張ポイント

### 9.1 将来の拡張

| 機能 | 優先度 | 説明 |
|------|--------|------|
| ファイル操作 | 中 | 3D空間からファイル削除/移動 |
| プラグインシステム | 低 | カスタムビジュアライゼーション |
| VR対応 | 低 | Oculus/Viveサポート |
| ネットワークドライブ | 中 | SMB/NFSマウント |
| ファイルプレビュー | 中 | 3D空間での画像プレビュー |

### 9.2 設定拡張

```python
# 現状: 設定は主にコード内定数と `LayoutConfig` に集約されている。
# 以下は「実装に存在する設定項目」の抜粋。

@dataclass
class LayoutConfig:
    node_size: float = 1.0
    dir_size: float = 2.0
    spacing: float = 0.5
    padding: float = 0.2
    max_depth: int = 5
    placement_strategy: PlacementStrategy = PlacementStrategy.GRID
    grid_size: float = 3.0
    connection_width: float = 0.1
    use_size_height: bool = True
    min_height: float = 0.2
    max_height: float = 5.0
    height_scale: float = 0.3

# カメラ/描画の細かな調整値は `Camera` / `Renderer` の内部定数として保持されている
# 例: _movement_speed, _rotation_speed, _zoom_speed, _damping, 色定義, スポットライト高さ/半径 など。
```

---

## 11. 付録

### 10.1 用語集

| 用語 | 説明 |
|------|------|
| **GPUインスタンシング** | 同一メッシュを異なる位置/サイズ/色で一度のドローコールで描画する技術（ModernGL資産として準備済み） |
| **視錐台カリング** | 視野外のオブジェクトを描画から除外する最適化（FrustumCullerクラス実装済み） |
| **LOD (Level of Detail)** | 距離に応じて描画品質を変える技術（LevelOfDetailクラス実装済み） |
| **レイキャスティング** | 2Dスクリーン座標から3D空間への ray を飛ばし、交差オブジェクトを検出（Ray-AABB Slab method実装済み） |
| **Slab method** | Ray-AABB交差判定の業界標準アルゴリズム。3軸のスラブ（平行平面対）との交差を計算 |
| **イージング** | アニメーションの加減速を滑らかにする手法（カメラスナップで使用） |

### 10.2 参考文献

- SGI IRIX fsn ドキュメント
- ModernGL Documentation（任意/実験）: https://moderngl.readthedocs.io/
- PyQt6 Documentation: https://www.riverbankcomputing.com/static/Docs/PyQt6/
- PyOpenGL Documentation: http://pyopengl.sourceforge.net/documentation/
- OpenGL Instancing Tutorial

---

## 12. 実装済み主要機能一覧

### ✅ 完全動作
- **3D可視化**: FSN風の浮遊プラットフォーム＋キューブ＋ワイヤー
- **後退レイアウト**: 子ディレクトリは負のZ方向に配置
- **対数スケール高さ**: ファイルサイズに応じた高さ（log10スケール）
- **年齢ベース色分け**: mtimeに基づく5段階の色（24h/1w/1m/1y/それ以上）
- **3Dピッキング**: ファイルのAABBをレンダリング位置に合わせて正確にピッキング
- **ファイル名常時表示**: すべてのファイルとディレクトリ名を表示
- **ディレクトリ名表示**: SGI fsnスタイルの手書き風テキストを地面に配置（衝突回避付き）
- **ファイルツリーファイルオープン**: ダブルクリックでファイルを開く
- **カメラスナップアクション**: ダブルクリックでイージングアニメーション（300ms）付きフォーカス
- **Orbitカメラモード**: 注視点周りの回転（マウスドラッグ操作）
- **ナビゲーション履歴**: Back/Forward（内部で管理、UIから操作可能）
- **スポットライト**: 選択ノード上に半透明コーン表示
- **ファイルツリー**: QTreeWidget による階層表示＋同期選択
- **リアルタイム検索**: インクリメンタルサーチ
- **高度なフィルタリング**: サイズ・年齢・種別による絞り込み（FilterPanel）
- **ファイル年齢凡例**: 左下に表示される色分け凡例（FileAgeLegend）
- **非同期スキャン**: QThreadによるバックグラウンド読み込み
- **遅延ロード**: 深い階層は必要時に展開（再帰ロード対応）
- **ファイルオープン**: ダブルクリックでファイルをOSデフォルトアプリで開く
- **エラーハンドリング**: ファイルオープン失敗時のエラーダイアログ表示
- **デプスフォグ**: 距離に応じたフォグ効果（リニアフォグ）
- **地面グリッド**: FSNスタイルのグリッドライン
- **スカイグラデーション**: 上部が明るく下方がやや暗いグラデーション背景
- **キューブエッジハイライト**: エッジの強調表示（Z-fighting回避付き）
- **オーバーレイ自動リサイズ**: ウィンドウサイズ変更に追従
- **ホバーツールチップ**: ファイル情報をホバー時に表示
- **カプセル化改善**: パブリックAPIの整理
- **メディアプレビュー**: 画像・動画ファイルのホバープレビュー
- **右ドラッグパン**: 右マウスボタンドラッグによるカメラパン
- **Shift+左ドラッグパン**: macOSトラックパッド向け代替操作
- **テーマシステム**: SGI Classic, Dark Mode, Cyberpunk等の切り替え（ThemeManager）
- **ブルーム/エミッシブ**: ファイル種別に基づく発光効果（SimpleBloom）
- **ワイヤーパルス**: 接続線のアニメーション効果
- **2Dミニマップ**: レーダースタイルの俯瞰ビュー（MiniMap）
- **ModernGLレンダラー**: 代替描画バックエンド（ModernGLRenderer）
- **ピッキングシステム**: Ray-AABB交差判定の独立モジュール（PickingSystem）

### 🎮 操作系
- **マウス**: 左ドラッグ回転、右ドラッグパン、中ドラッグパン、ホイールズーム
- **Shift+左ドラッグ**: macOSトラックパッド向けパン代替操作
- **3Dクリック**: 左クリック選択、ダブルクリックフォーカス/ナビゲート
- **メニューショートカット**: Ctrl+O（開く）、F5（更新）、Ctrl+T（ツリー）、Ctrl+F（フィルタ）、Ctrl+L（ラベル）
- **修飾キー**: Ctrl+クリック（トグル）、Shift+クリック（追加選択）

### ❌ 未実装機能
- **キーボードによる3Dビュー操作**: F（スナップ）、R（リセット）、Esc（選択解除）等
- **ビュープリセット**: Bird's Eye、Front view

### 🔧 最適化機能（実装済み、一部未接続）
- **PerformanceMonitor**: FPS/フレームタイム計測
- **FrustumCuller**: 視錐台カリング（未接続）
- **LevelOfDetail**: 距離ベースLOD（未接続）
- **ProgressiveLoader**: プログレッシブロード（未接続）

### 📦 将来の拡張資産（実装済み、未使用）
- **ModernGL資産**: `cube_geometry.py`、`shaders.py`（GPUインスタンシング用）
- **カスタムシェーダー**: GLSL 330 vertex/fragment shaders

---

## 13. 変更履歴

### v1.9.0 (2026-02-09) - メディアプレビュー機能

**新機能**
- 画像ファイルにホバー時にサムネイルを表示する機能を実装
  - 対応形式: PNG, JPG, GIF, BMP, WebP, SVG, ICO, TIFF, PSD, RAW, HEIC, AVIF 等（15+形式）
  - Qt組み込み機能を使用（追加依存なし）
  - 最大320×240ピクセルでプレビュー
  - アスペクト比を保持して自動スケーリング
  - 画像寸法情報を表示

- 動画ファイルにホバー時にサムネイルを表示する機能を実装
  - 対応形式: MP4, AVI, MKV, MOV, WMV, FLV, WebM, M4V, MPG, MPEG, 3GP, OGV, TS 等（15+形式）
  - OpenCV（オプション）を使用して動画の25%位置からフレーム抽出
  - 再生アイコンオーバーレイで動画を識別
  - 紫色背景で画像と区別
  - 動画解像度情報を表示
  - OpenCV未インストール時は「Video (OpenCV not available)」と表示

**追加コンポーネント**
- `Node.is_image_file`: 画像ファイル検出プロパティ
- `Node.is_video_file`: 動画ファイル検出プロパティ
- `ImagePreviewTooltip`: 画像・動画プレビューツールチップクラス
  - `_load_image_if_needed()`: 画像のロードと事前スケーリング
  - `_load_video_thumbnail_if_needed()`: 動画サムネイルの抽出
  - `clear_cache()`: キャッシュクリアメソッド

**依存関係の追加**
- `pyproject.toml` に `[video]` オプション依存関係を追加
  - `opencv-python>=4.8.0`

**ドキュメント更新**
- `README.md`: メディアプレビュー機能の説明とインストール手順を追加
- `docs/API.md`: `is_image_file`, `is_video_file`, `ImagePreviewTooltip` のAPIドキュメントを追加
- `docs/MEDIA_PREVIEW.md` (新規): メディアプレビュー機能の詳細ドキュメント

### v1.8.1 (2026-02-09) - カメラセンタリング修正

**バグ修正**
- ディレクトリをダブルクリックして移動した際、カメラがディレクトリの中心に来ない問題を修正
- シーン境界全体の中心ではなく、ルートディレクトリの位置をカメラターゲットにするように変更

**変更**
- `Controller._reset_camera_to_scene()`: ルートディレクトリの中心をターゲットに使用

### v1.8.0 (2026-02-09) - ビジュアル機能強化

**改善**
- 3Dビューでファイルが選択できるようになった（ピッキングAABB修正）
- ファイル名を常に表示するように変更
- ディレクトリ名をプラットフォーム横の地面に手書き風フォントで表示
- 衝突判定でラベル同士が重ならないように動的配置

**追加コンポーネント**
- `Renderer._create_text_texture()`: テキストテクスチャ生成
- `Renderer._draw_directory_labels()`: ディレクトリ名描画
- `Renderer._calculate_label_positions()`: 衝突回避付きラベル配置

**変更**
- `Renderer.raycast_find_node()`: ファイルAABB計算を修正
- `Controller._update_text_overlay()`: すべてのノード名を表示
- `FileTreeWidget`: ダブルクリックシグナルを追加

### v1.7.0 (2026-02-09) - ファイルオープン機能の実装

**新機能**
- ファイルをダブルクリックでOSデフォルトアプリケーションで開く機能を実装
- クロスプラットフォーム対応（macOS、Windows、Linux）
- ファイルオープン失敗時のエラーダイアログ表示

**追加コンポーネント**
- `FileOpenError`: ファイルオープン専用の例外クラス
- `Controller._open_file()`: ファイルオープン処理メソッド

**変更**
- `Controller._on_node_clicked()`: ファイルダブルクリック時の処理を追加

### v1.6.0 (2026-02-08) - 仕様整理とクリーンアップ
**不要機能の削除**
- Fly/Snapカメラモードの削除（Orbitモードのみに統一）
- キーボードショートカット（F, R, Esc等）の削除（マウス操作に集約）
- ライティングの無効化（`glDisable(GL_LIGHTING)`）

**実装済み機能のSPEC反映**
- FilterPanel（高度なフィルタリング）の仕様追加
- FileAgeLegend（ファイル年齢凡例）の仕様追加
- ディレクトリ色を明るい青 `(0.2, 0.6, 1.0)` に設定
- LayoutConfigデフォルト値をSPEC準拠に修正（`spacing=0.5`, `padding=0.2`, `grid_size=3.0`）

**バグ修正**
- `Controller._on_camera_mode_changed()` で未定義変数 `node` を参照していたバグを修正

### v1.5.0 (2026-02-08) - Sprint 1 完了
**バグ修正・コード品質向上（Phase 1）**
- `_on_timer` 二重定義の修正（アニメーション処理を統合）
- `ControlPanel.set_camera_mode()` メソッドの追加
- Scanner 再帰ロードの修正（同期版でも再帰するように）
- オーバーレイのリサイズ追従の改善
- カプセル化の改善（パブリックAPIの追加）

**SGI fsn 忠実度向上（Phase 2/3）**
- デプスフォグの実装（`glFog` によるリニアフォグ）
- 地面グリッドラインの追加（`GL_LINES` によるグリッド描画）
- スカイグラデーションの実装（フルスクリーンクワッドによるグラデーション）
- キューブエッジハイライトの追加（`GL_LINES` + `glPolygonOffset` によるZ-fighting回避）
- ライティング実装（`GL_LIGHT1` 追加、スペキュラーハイライト有効化）- v1.6.0で無効化

### v1.4.0 (2026-02-08)
- 現在のソースコードに合わせて仕様を更新
- Flyモードを削除（Orbitモードのみ）
- ビュープリセット（Bird's Eye、Front view）を追加
- ナビゲーション履歴（Back/Forward）を追加
- 右ドラッグパンを追加
- キーボード操作を更新（1/2キー削除、Tab/Alt+B/Alt+F追加）
- カメラモードUIボタンを削除

### v1.3.0 (2026-02-08)
- 実装状況を正確に反映（3Dピッキング、スナップ、Flyモードの記述を修正）
- Snapモードを削除（snap_to()はアクションとして機能）
- Flyモードのvelocity更新を修正（camera.update()呼び出しを追加）
- 依存関係を整理（PyOpenGL必須、ModernGLは将来用）

### v1.2.1 (2026-02-08)
- 実装実態に合わせてUI/描画/最適化章を整理

### v1.0.0 (2026-02-07)
- 初版作成

---

*仕様書バージョン: 1.9.1*
*作成日: 2026-02-07*
*最終更新: 2026-02-11 (ドキュメントとソースの不一致修正 - ディレクトリ構造更新、未実装機能リスト修正、BackgroundMode削除、ライティング状態整理)*
