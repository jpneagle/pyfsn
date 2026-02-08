"""Main application window for pyfsn.

Provides the top-level window containing the renderer and
UI controls for the file system navigator.
"""

from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QStatusBar,
    QMenuBar,
    QMenu,
    QFileDialog,
    QLineEdit,
    QDockWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QSplitter,
    QTextEdit,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QAction, QKeyEvent, QPainter, QColor, QFont, QPixmap, QImage
import sys

from pyfsn.view.renderer import Renderer
from pyfsn.view.camera import CameraMode
from pyfsn.view.filter_panel import FilterPanel
from pyfsn.view.mini_map import MiniMap


class TextOverlay(QWidget):
    """Text overlay widget for rendering labels on top of the 3D view.

    Provides 2D text rendering for node names and information.
    """

    def __init__(self, parent=None) -> None:
        """Initialize text overlay.

        Args:
            parent: Parent widget (should be the Renderer)
        """
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._labels: list[tuple[str, int, int]] = []  # (text, x, y)
        self._focused_text: str | None = None
        self._focused_pos: tuple[int, int] | None = None

    def set_labels(self, labels: list[tuple[str, int, int]]) -> None:
        """Set text labels to render.

        Args:
            labels: List of (text, x, y) tuples
        """
        self._labels = labels
        self.update()

    def set_focused_label(self, text: str | None, x: int | None = None, y: int | None = None) -> None:
        """Set the focused/hovered label.

        Args:
            text: Text to display
            x: X position (or None)
            y: Y position (or None)
        """
        self._focused_text = text
        self._focused_pos = (x, y) if x is not None and y is not None else None
        self.update()

    def clear(self) -> None:
        """Clear all labels."""
        self._labels.clear()
        self._focused_text = None
        self._focused_pos = None
        self.update()

    def paintEvent(self, event) -> None:
        """Paint the text overlay."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw regular labels
        font = QFont("Arial", 9)
        painter.setFont(font)
        painter.setPen(QColor(200, 200, 200))

        for text, x, y in self._labels:
            painter.drawText(x, y, text)

        # Draw focused label (larger, brighter)
        if self._focused_text and self._focused_pos:
            x, y = self._focused_pos
            font_bold = QFont("Arial", 11)
            font_bold.setBold(True)
            painter.setFont(font_bold)
            painter.setPen(QColor(255, 255, 200))

            # Draw background for better readability
            metrics = painter.fontMetrics()
            text_width = metrics.horizontalAdvance(self._focused_text)
            text_height = metrics.height()

            painter.setBrush(QColor(0, 0, 0, 180))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(x - 4, y - text_height + 4, text_width + 8, text_height + 4, 4, 4)

            painter.setPen(QColor(255, 255, 200))
            painter.drawText(x, y, self._focused_text)


class FileAgeLegend(QWidget):
    """Legend overlay showing file age color coding (bottom-left corner)."""

    def __init__(self, parent=None) -> None:
        """Initialize file age legend.

        Args:
            parent: Parent widget (should be the Renderer)
        """
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def paintEvent(self, event) -> None:
        """Paint the legend."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Position at bottom-left with margin
        margin = 10
        x = margin
        y = self.height() - 130 - margin

        # Dark semi-transparent background with rounded corners (like original)
        painter.setBrush(QColor(30, 40, 30, 200))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(x, y, 120, 130, 8, 8)

        # Title
        font = QFont("Arial", 11, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(x + 12, y + 24, "File Age")

        # Legend entries
        font = QFont("Arial", 10)
        painter.setFont(font)

        entries = [
            (QColor(51, 255, 51), "< 24h"),
            (QColor(51, 204, 255), "< 7d"),
            (QColor(204, 204, 51), "< 30d"),
            (QColor(255, 153, 51), "< 365d"),
            (QColor(153, 76, 51), ">= 365d"),
        ]

        for i, (color, label) in enumerate(entries):
            y_pos = y + 40 + i * 17
            # Color swatch (larger and clearer)
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(x + 12, y_pos, 20, 14, 2, 2)
            # Label
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(x + 38, y_pos + 11, label)


class FileTooltipOverlay(QWidget):
    """Tooltip overlay for displaying file information on hover (SGI fsn style).

    Shows file name, size, permissions, and modification time.
    """

    def __init__(self, parent=None) -> None:
        """Initialize file tooltip overlay.

        Args:
            parent: Parent widget (should be the Renderer)
        """
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setVisible(False)

        from pyfsn.model.node import Node
        self._node: Node | None = None
        self._position: tuple[int, int] = (0, 0)

    def show_for_node(self, node, x: int, y: int) -> None:
        """Show tooltip for a node at the given position.

        Args:
            node: Node to show information for
            x: Screen X position
            y: Screen Y position
        """
        self._node = node
        self._position = (x + 15, y + 15)  # Offset from cursor
        self.setVisible(True)
        self.update()

    def hide_tooltip(self) -> None:
        """Hide the tooltip."""
        self._node = None
        self.setVisible(False)

    def _format_size(self, size: int) -> str:
        """Format file size for display."""
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        elif size < 1024 * 1024 * 1024:
            return f"{size / (1024 * 1024):.1f} MB"
        else:
            return f"{size / (1024 * 1024 * 1024):.1f} GB"

    def _format_permissions(self, permissions: int) -> str:
        """Format permissions as rwxrwxrwx string."""
        mode_chars = ""
        for i in range(8, -1, -1):
            bit = (permissions >> i) & 1
            if i % 3 == 2:
                mode_chars += "r" if bit else "-"
            elif i % 3 == 1:
                mode_chars += "w" if bit else "-"
            else:
                mode_chars += "x" if bit else "-"
        return mode_chars

    def _format_time(self, mtime: float) -> str:
        """Format modification time."""
        from datetime import datetime
        dt = datetime.fromtimestamp(mtime)
        return dt.strftime("%Y-%m-%d %H:%M")

    def paintEvent(self, event) -> None:
        """Paint the tooltip with improved positioning (PR-4)."""
        if not self._node:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Build tooltip text lines
        lines = [
            self._node.name,
            f"Size: {self._format_size(self._node.size)}",
            f"Permissions: {self._format_permissions(self._node.permissions)}",
            f"Modified: {self._format_time(self._node.mtime)}",
        ]

        # Calculate background size
        font = QFont("Monospace", 10)
        painter.setFont(font)
        metrics = painter.fontMetrics()

        max_width = max(metrics.horizontalAdvance(line) for line in lines)
        line_height = metrics.height()
        padding = 8
        margin = 10  # Extra margin from screen edges
        total_height = line_height * len(lines) + padding * 2
        total_width = max_width + padding * 2

        x, y = self._position

        # PR-4: Improved tooltip positioning - handle all screen edges
        # Ensure tooltip doesn't overflow right edge
        if x + total_width > self.width() - margin:
            x = self.width() - total_width - margin

        # Ensure tooltip doesn't overflow left edge
        if x < margin:
            x = margin

        # Ensure tooltip doesn't overflow bottom edge
        if y + total_height > self.height() - margin:
            y = self.height() - total_height - margin

        # Ensure tooltip doesn't overflow top edge
        if y < margin:
            y = margin

        # Draw background with subtle shadow effect (PR-4: improved visual)
        painter.setBrush(QColor(20, 20, 30, 235))
        painter.setPen(QColor(100, 100, 120))
        painter.drawRoundedRect(x, y, total_width, total_height, 6, 6)

        # Draw text
        painter.setPen(QColor(220, 220, 230))
        text_y = y + padding + line_height - metrics.descent()

        # First line (name) is bold
        font_bold = QFont("Monospace", 10)
        font_bold.setBold(True)
        painter.setFont(font_bold)
        painter.setPen(QColor(255, 255, 200))
        painter.drawText(x + padding, text_y, lines[0])

        # Rest of lines
        painter.setFont(font)
        painter.setPen(QColor(180, 180, 190))
        for i, line in enumerate(lines[1:], 1):
            text_y = y + padding + line_height * (i + 1) - metrics.descent()
            painter.drawText(x + padding, text_y, line)


class ImagePreviewTooltip(FileTooltipOverlay):
    """Enhanced tooltip with image preview and video thumbnails.

    When hovering over an image file, shows a preview of the image.
    When hovering over a video file, shows a thumbnail from the video with a play icon overlay.
    Also displays standard file information.

    Video thumbnails are generated using OpenCV (optional dependency).
    """

    # Maximum size for image preview
    MAX_IMAGE_WIDTH = 320
    MAX_IMAGE_HEIGHT = 240

    def __init__(self, parent=None) -> None:
        """Initialize image preview tooltip.

        Args:
            parent: Parent widget (should be the Renderer)
        """
        super().__init__(parent)

        # Image/Video preview data
        self._cached_pixmap: QPixmap | None = None
        self._cached_path: str | None = None
        self._media_info: str | None = None  # Image/Video dimensions info
        self._scaled_pixmap: QPixmap | None = None  # Cached scaled preview
        self._is_video: bool = False  # Whether current preview is from video

        # Try to import OpenCV for video thumbnails (optional dependency)
        self._cv2 = None
        try:
            import cv2
            self._cv2 = cv2
        except ImportError:
            pass

    def show_for_node(self, node, x: int, y: int) -> None:
        """Show tooltip for a node at the given position.

        Args:
            node: Node to show information for
            x: Screen X position
            y: Screen Y position
        """
        self._node = node
        self._position = (x + 15, y + 15)  # Offset from cursor

        # Check file type and load appropriate preview
        if hasattr(node, 'is_image_file') and node.is_image_file:
            self._load_image_if_needed(node)
        elif hasattr(node, 'is_video_file') and node.is_video_file:
            self._load_video_thumbnail_if_needed(node)
        else:
            # Clear cached preview for other file types
            self._cached_pixmap = None
            self._scaled_pixmap = None
            self._media_info = None
            self._is_video = False

        self.setVisible(True)
        self.update()

    def _load_image_if_needed(self, node) -> None:
        """Load and cache the image if not already cached.

        Args:
            node: Node to load image for
        """
        path_str = str(node.path)

        # Check if we need to reload
        if self._cached_path == path_str and self._cached_pixmap is not None and not self._is_video:
            return

        try:
            # Load image using QPixmap
            self._cached_pixmap = QPixmap(str(node.path))

            if self._cached_pixmap.isNull():
                # Failed to load image
                self._cached_pixmap = None
                self._scaled_pixmap = None
                self._media_info = None
            else:
                # Get image info
                self._media_info = f"{self._cached_pixmap.width()}×{self._cached_pixmap.height()} px"
                self._cached_path = path_str
                self._is_video = False
                # Pre-scale the image for performance
                self._scaled_pixmap = self._scale_preview(self._cached_pixmap)

        except Exception:
            # Failed to load image
            self._cached_pixmap = None
            self._scaled_pixmap = None
            self._media_info = None
            self._cached_path = None
            self._is_video = False

    def _scale_preview(self, pixmap: QPixmap) -> QPixmap:
        """Scale image or video thumbnail to fit within maximum dimensions while maintaining aspect ratio.

        Args:
            pixmap: Original pixmap

        Returns:
            Scaled pixmap
        """
        return pixmap.scaled(
            self.MAX_IMAGE_WIDTH,
            self.MAX_IMAGE_HEIGHT,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

    def _load_video_thumbnail_if_needed(self, node) -> None:
        """Load and cache a video thumbnail if not already cached.

        Extracts a frame from the video at 25% position to get a representative thumbnail.

        Args:
            node: Video Node to generate thumbnail for
        """
        path_str = str(node.path)

        # Check if we need to reload
        if self._cached_path == path_str and self._cached_pixmap is not None and self._is_video:
            return

        self._is_video = False
        self._cached_pixmap = None
        self._scaled_pixmap = None
        self._media_info = None

        # Check if OpenCV is available
        if self._cv2 is None:
            self._media_info = "Video (OpenCV not available)"
            self._cached_path = path_str
            return

        try:
            # Open video file
            cap = self._cv2.VideoCapture(str(node.path))

            if not cap.isOpened():
                self._media_info = "Video (unreadable)"
                self._cached_path = path_str
                return

            # Get video properties
            width = int(cap.get(self._cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(self._cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(self._cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(self._cv2.CAP_PROP_FPS)

            # Seek to 25% of video to get a representative frame
            target_frame = int(frame_count * 0.25) if frame_count > 0 else 0
            cap.set(self._cv2.CAP_PROP_POS_FRAMES, target_frame)

            # Read frame
            ret, frame = cap.read()
            cap.release()

            if ret and frame is not None:
                # Convert BGR to RGB
                frame_rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

                # Create QPixmap from numpy array
                h, w, ch = frame_rgb.shape
                bytes_per_line = 3 * w
                q_img = QImage(
                    frame_rgb.data,
                    w,
                    h,
                    bytes_per_line,
                    QImage.Format.Format_RGB888
                )
                self._cached_pixmap = QPixmap.fromImage(q_img)

                if not self._cached_pixmap.isNull():
                    duration_str = ""
                    if frame_count > 0 and fps > 0:
                        duration = frame_count / fps
                        minutes = int(duration // 60)
                        secs = int(duration % 60)
                        duration_str = f" ({minutes}:{secs:02d})"

                    self._media_info = f"{width}×{height} px @ {fps:.1f}fps{duration_str}"
                    self._scaled_pixmap = self._scale_preview(self._cached_pixmap)
                    self._is_video = True
                    self._cached_path = path_str
                else:
                    self._media_info = f"{width}×{height} px @ {fps:.1f}fps"
            else:
                self._media_info = f"{width}×{height} px (no thumbnail)"

            self._cached_path = path_str

        except Exception:
            # Failed to load video
            self._media_info = "Video (thumbnail error)"
            self._cached_path = path_str

    def paintEvent(self, event) -> None:
        """Paint the tooltip with optional image/video preview."""
        if not self._node:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Check if we have a preview (image or video thumbnail)
        has_preview = self._scaled_pixmap is not None

        # Build tooltip text lines
        lines = [
            self._node.name,
            f"Size: {self._format_size(self._node.size)}",
        ]

        # Add media info if available
        if self._media_info:
            label = "Video:" if self._is_video else "Dimensions:"
            lines.append(f"{label} {self._media_info}")

        lines.extend([
            f"Permissions: {self._format_permissions(self._node.permissions)}",
            f"Modified: {self._format_time(self._node.mtime)}",
        ])

        # Calculate text dimensions
        font = QFont("Monospace", 10)
        painter.setFont(font)
        metrics = painter.fontMetrics()

        max_text_width = max(metrics.horizontalAdvance(line) for line in lines)
        line_height = metrics.height()
        text_padding = 8
        margin = 10  # Extra margin from screen edges

        # Calculate preview area dimensions
        preview_height = 0
        preview_width = 0
        preview_padding = 0

        if has_preview:
            preview_width = self._scaled_pixmap.width()
            preview_height = self._scaled_pixmap.height()
            preview_padding = 10  # Padding around preview

        # Total dimensions - use max of text width and preview width
        content_width = max(max_text_width, preview_width)
        total_width = content_width + text_padding * 2
        total_height = preview_height + preview_padding + (line_height * len(lines)) + text_padding * 2

        # Position tooltip
        x, y = self._position

        # Get available space (parent widget dimensions)
        parent_width = self.width()
        parent_height = self.height()

        # Handle edge clamping - ensure tooltip fits within parent bounds
        if x + total_width > parent_width - margin:
            x = parent_width - total_width - margin
        if x < margin:
            x = margin
        if y + total_height > parent_height - margin:
            # Try to position above cursor instead of below
            y = y - total_height - 30  # Move above cursor with extra offset
            if y < margin:
                y = margin  # Still doesn't fit, clamp to top
        if y < margin:
            y = margin

        # Draw background with purple tint for videos
        if self._is_video:
            painter.setBrush(QColor(30, 20, 40, 245))  # Purple tint for video
        else:
            painter.setBrush(QColor(20, 20, 30, 245))  # Standard dark background
        painter.setPen(QColor(100, 100, 120))
        painter.drawRoundedRect(x, y, total_width, total_height, 8, 8)

        # Current Y position for drawing
        current_y = y

        # Draw preview if available
        if has_preview:
            # Center preview horizontally
            preview_x = x + (total_width - preview_width) // 2

            # Draw preview with border for videos
            if self._is_video:
                # Add a subtle border for videos
                painter.setBrush(QColor(0, 0, 0, 100))
                painter.setPen(QColor(200, 150, 255, 180))
                painter.drawRect(preview_x - 1, current_y + text_padding - 1,
                              preview_width + 2, preview_height + 2)

            painter.drawPixmap(preview_x, current_y + text_padding, self._scaled_pixmap)

            # Draw play icon overlay for videos
            if self._is_video:
                cx = preview_x + preview_width // 2
                cy = current_y + text_padding + preview_height // 2
                play_size = 24

                # Draw play triangle
                painter.setBrush(QColor(255, 255, 255, 220))
                painter.setPen(Qt.PenStyle.NoPen)
                from PyQt6.QtCore import QPointF
                from PyQt6.QtGui import QPolygonF
                triangle = QPolygonF([
                    QPointF(cx - play_size // 3, cy - play_size // 2),
                    QPointF(cx - play_size // 3, cy + play_size // 2),
                    QPointF(cx + play_size // 2, cy),
                ])
                painter.drawPolygon(triangle)

            current_y += preview_height + preview_padding

        # Draw text information
        text_y = current_y + text_padding + line_height - metrics.descent()

        # First line (name) is bold
        font_bold = QFont("Monospace", 10)
        font_bold.setBold(True)
        painter.setFont(font_bold)
        painter.setPen(QColor(255, 255, 200))
        painter.drawText(x + text_padding, text_y, lines[0])

        # Rest of lines
        painter.setFont(font)
        painter.setPen(QColor(180, 180, 190))
        for i, line in enumerate(lines[1:], 1):
            text_y = current_y + text_padding + line_height * (i + 1) - metrics.descent()
            painter.drawText(x + text_padding, text_y, line)

    def clear_cache(self) -> None:
        """Clear cached preview data."""
        self._cached_pixmap = None
        self._scaled_pixmap = None
        self._cached_path = None
        self._media_info = None
        self._is_video = False


class ControlPanel(QWidget):
    """Control panel widget with camera and view controls."""

    # Signals
    refresh_requested = pyqtSignal()
    navigate_up = pyqtSignal()
    navigate_home = pyqtSignal()
    show_tree_toggled = pyqtSignal(bool)
    filter_panel_toggled = pyqtSignal(bool)
    labels_toggled = pyqtSignal(bool)

    def __init__(self, parent=None) -> None:
        """Initialize control panel."""
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the control panel UI."""
        self.setFixedWidth(220)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Title
        title = QLabel("<b>pyfsn</b>")
        title.setStyleSheet("font-size: 14px;")
        layout.addWidget(title)

        layout.addSpacing(8)

        # Camera mode display (hidden, for internal use only)
        self._camera_mode_label = QLabel("Orbit")
        self._camera_mode_label.setVisible(False)
        
        # Navigation buttons
        layout.addWidget(QLabel("<b>Navigation:</b>"))

        nav_layout = QHBoxLayout()
        self._up_btn = QPushButton("↑")
        self._up_btn.setFixedWidth(40)
        self._up_btn.setToolTip("Navigate to parent directory")
        nav_layout.addWidget(self._up_btn)

        self._home_btn = QPushButton("⌂")
        self._home_btn.setFixedWidth(40)
        self._home_btn.setToolTip("Navigate to root directory")
        nav_layout.addWidget(self._home_btn)

        nav_layout.addStretch()
        layout.addLayout(nav_layout)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setToolTip("Rescan current directory (F5)")
        layout.addWidget(self._refresh_btn)

        # Connect navigation signals
        self._up_btn.clicked.connect(self.navigate_up.emit)
        self._home_btn.clicked.connect(self.navigate_home.emit)
        self._refresh_btn.clicked.connect(self.refresh_requested.emit)

        layout.addSpacing(12)

        # View options
        layout.addWidget(QLabel("<b>View Options:</b>"))

        self._show_tree_btn = QPushButton("Show Tree")
        self._show_tree_btn.setCheckable(True)
        self._show_tree_btn.setChecked(False)  # Default: OFF
        self._show_tree_btn.setToolTip("Toggle file tree panel")
        layout.addWidget(self._show_tree_btn)

        self._filter_panel_btn = QPushButton("Filter Panel")
        self._filter_panel_btn.setCheckable(True)
        self._filter_panel_btn.setChecked(False)  # Default: OFF
        self._filter_panel_btn.setToolTip("Toggle filter panel")
        layout.addWidget(self._filter_panel_btn)

        self._show_labels_btn = QPushButton("Show Labels")
        self._show_labels_btn.setCheckable(True)
        self._show_labels_btn.setChecked(True)
        self._show_labels_btn.setToolTip("Toggle node name labels")
        layout.addWidget(self._show_labels_btn)

        # Connect view option signals
        self._show_tree_btn.clicked.connect(self.show_tree_toggled.emit)
        self._filter_panel_btn.clicked.connect(self.filter_panel_toggled.emit)
        self._show_labels_btn.clicked.connect(self.labels_toggled.emit)

        layout.addSpacing(12)
        
        # Info section
        layout.addWidget(QLabel("<b>Controls:</b>"))
        info = QLabel(
            "<b>Mouse:</b><br>"
            "Left-drag: Rotate<br>"
            "Right-drag: Pan<br>"
            "Shift+Left-drag: Pan<br>"
            "Middle-drag: Pan<br>"
            "Scroll: Zoom<br>"
            "Click: Select<br>"
            "Dbl-click dir: Navigate<br>"
            "Dbl-click file: Open"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(info)

        layout.addStretch()

        # Stats section
        self._stats_label = QLabel("<b>Stats:</b><br>Nodes: 0")
        self._stats_label.setStyleSheet("font-size: 10px;")
        layout.addWidget(self._stats_label)

    def set_camera_mode_display(self, mode: CameraMode) -> None:
        """Update camera mode display.
        
        Args:
            mode: Current camera mode
        """
        mode_names = {
            CameraMode.ORBIT: "Orbit",
        }
        self._camera_mode_label.setText(mode_names.get(mode, "Unknown"))
    
    def update_stats(self, node_count: int, selected_count: int = 0) -> None:
        """Update statistics display.

        Args:
            node_count: Total number of nodes
            selected_count: Number of selected nodes
        """
        self._stats_label.setText(
            f"<b>Stats:</b><br>Nodes: {node_count}<br>Selected: {selected_count}"
        )


class SearchBar(QLineEdit):
    """Search bar for finding nodes by name."""

    search_requested = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        """Initialize search bar."""
        super().__init__(parent)
        self.setPlaceholderText("Search files and folders...")
        self.setClearButtonEnabled(True)
        self.textChanged.connect(self._on_text_changed)

    def _on_text_changed(self, text: str) -> None:
        """Handle text change - emit search signal with debounce."""
        # Using a simple debounce timer
        if hasattr(self, '_debounce_timer'):
            self._debounce_timer.stop()

        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(lambda: self.search_requested.emit(text))
        self._debounce_timer.start(300)  # 300ms debounce


class FileTreeWidget(QTreeWidget):
    """Tree widget showing the file system hierarchy."""

    node_selected = pyqtSignal(object)
    node_double_clicked = pyqtSignal(object)

    # Role for storing node reference in item data
    NODE_ROLE = Qt.ItemDataRole.UserRole

    def __init__(self, parent=None) -> None:
        """Initialize file tree widget."""
        super().__init__(parent)
        self.setHeaderLabels(["Name", "Size", "Type"])
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        self.setColumnWidth(0, 200)
        self.setColumnWidth(1, 80)
        self.setColumnWidth(2, 80)

        # Connect selection signal
        self.itemClicked.connect(self._on_item_clicked)
        # Connect double-click signal
        self.itemDoubleClicked.connect(self._on_item_double_clicked)

        # Map path strings to tree items for quick lookup
        self._path_to_item: dict[str, QTreeWidgetItem] = {}

    def load_tree(self, root_node) -> None:
        """Load tree from root node.

        Args:
            root_node: Root Node object
        """
        self.clear()
        self._path_to_item.clear()

        root_item = self._create_item(root_node)
        self.addTopLevelItem(root_item)
        self._load_children(root_node, root_item)

        self.expandAll()

    def _load_children(self, node, parent_item: QTreeWidgetItem) -> None:
        """Recursively load children.

        Args:
            node: Parent node
            parent_item: Parent tree item
        """
        for child in node.children:
            child_item = self._create_item(child)
            parent_item.addChild(child_item)

            if child.is_directory and child.is_loaded:
                self._load_children(child, child_item)

    def _create_item(self, node) -> QTreeWidgetItem:
        """Create a tree item from a node.

        Args:
            node: Node object

        Returns:
            QTreeWidgetItem
        """
        from pyfsn.model.node import NodeType

        # Format size
        if node.is_directory:
            size_str = ""
        else:
            size = node.size
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size / (1024 * 1024):.1f} MB"

        # Determine type string
        if node.type == NodeType.DIRECTORY:
            type_str = "Directory"
        elif node.type == NodeType.SYMLINK:
            type_str = "Symlink"
        else:
            # Get extension
            if "." in node.name:
                ext = node.name.rsplit(".", 1)[-1].upper()
                type_str = f"{ext} File"
            else:
                type_str = "File"

        item = QTreeWidgetItem([node.name, size_str, type_str])
        # Store node reference in item data
        item.setData(0, self.NODE_ROLE, node)

        # Map path to item for quick lookup
        self._path_to_item[str(node.path)] = item

        return item

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """Handle item click.

        Args:
            item: Clicked tree item
            column: Column index
        """
        node = item.data(0, self.NODE_ROLE)
        if node:
            self.node_selected.emit(node)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """Handle item double-click.

        Args:
            item: Double-clicked tree item
            column: Column index
        """
        node = item.data(0, self.NODE_ROLE)
        if node:
            self.node_double_clicked.emit(node)

    def select_node(self, node) -> None:
        """Select and show a node in the tree.

        Args:
            node: Node to select
        """
        # Look up item by path
        path_str = str(node.path)
        item = self._path_to_item.get(path_str)

        if item:
            self.setCurrentItem(item)
            self.scrollToItem(item)
            # Expand parent to ensure item is visible
            parent = item.parent()
            if parent:
                parent.setExpanded(True)


class MainWindow(QMainWindow):
    """Main window for the pyfsn application."""

    # Signals
    directory_changed = pyqtSignal(Path)
    search_requested = pyqtSignal(str)
    refresh_requested = pyqtSignal()
    go_back_requested = pyqtSignal()
    go_forward_requested = pyqtSignal()
    filter_changed = pyqtSignal(dict)
    tree_node_double_clicked = pyqtSignal(object)

    def __init__(self, root_path: Path) -> None:
        """Initialize main window.

        Args:
            root_path: Root directory path to visualize
        """
        super().__init__()

        self._root_path = root_path
        self._original_root_path = root_path  # Store for home navigation
        self._renderer = None
        self._text_overlay = None
        self._file_tooltip = None  # SGI fsn style tooltip
        self._mini_map = None  # 2D radar mini map
        self._control_panel = None
        self._file_tree = None
        self._search_bar = None
        self._show_labels = True

        self._setup_ui()
        self._setup_menu_bar()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        self.setWindowTitle(f"pyfsn - {self._root_path}")
        self.resize(1400, 900)

        # Create central widget with splitter
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create search bar at top
        self._search_bar = SearchBar()
        self._search_bar.setStyleSheet("padding: 8px; background: #2a2a2a; color: white;")
        main_layout.addWidget(self._search_bar)

        # Create main splitter for resizable panels
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter, stretch=1)

        # Create container for renderer
        view_container = QWidget()
        view_layout = QVBoxLayout(view_container)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(0)

        # Renderer container (for overlays)
        renderer_container = QWidget()
        renderer_layout = QVBoxLayout(renderer_container)
        renderer_layout.setContentsMargins(0, 0, 0, 0)
        
        self._renderer = Renderer(self)
        self._renderer.set_root_path(self._root_path)
        
        # Text overlay
        self._text_overlay = TextOverlay(self._renderer)
        self._text_overlay.setGeometry(0, 0, self._renderer.width(), self._renderer.height())
        self._text_overlay.raise_()
        
        # File tooltip overlay with image preview support
        self._file_tooltip = ImagePreviewTooltip(self._renderer)
        self._file_tooltip.setGeometry(0, 0, self._renderer.width(), self._renderer.height())
        self._file_tooltip.raise_()
        
        # File age legend overlay (bottom-left)
        self._file_age_legend = FileAgeLegend(self._renderer)
        self._file_age_legend.setGeometry(0, 0, self._renderer.width(), self._renderer.height())
        self._file_age_legend.raise_()

        # Mini map overlay (bottom-right)
        self._mini_map = MiniMap(self._renderer)
        self._mini_map.move(self._renderer.width() - self._mini_map.width() - 10,
                           self._renderer.height() - self._mini_map.height() - 10)
        self._mini_map.raise_()

        renderer_layout.addWidget(self._renderer)
        view_layout.addWidget(renderer_container)

        main_splitter.addWidget(view_container)

        # Create control panel
        self._control_panel = ControlPanel()
        main_splitter.addWidget(self._control_panel)

        # Create file tree dock
        self._file_tree_dock = QDockWidget("File Tree", self)
        self._file_tree_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self._file_tree = FileTreeWidget()
        self._file_tree_dock.setWidget(self._file_tree)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._file_tree_dock)
        self._file_tree_dock.hide()  # Hidden by default

        # Create filter panel dock (Workstream F - Advanced filtering)
        self._filter_panel_dock = QDockWidget("Filter Panel", self)
        self._filter_panel_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self._filter_panel = FilterPanel()
        self._filter_panel_dock.setWidget(self._filter_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._filter_panel_dock)
        self._filter_panel_dock.hide()  # Hidden by default

        # Set initial splitter sizes
        main_splitter.setSizes([1000, 200])

        # Create status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage(f"Root: {self._root_path}")

    @property
    def file_tooltip(self) -> ImagePreviewTooltip | None:
        """Get the file tooltip overlay widget with image preview support."""
        return self._file_tooltip

    def _setup_menu_bar(self) -> None:
        """Set up the menu bar."""
        menubar = self.menuBar()
        menubar.setStyleSheet("QMenuBar { background: #2a2a2a; color: white; }"

                              "QMenuBar::item { background: transparent; }"

                              "QMenuBar::item:selected { background: #3a3a3a; }"

                              "QMenu { background: #2a2a2a; color: white; }"

                              "QMenu::item:selected { background: #3a3a3a; }")

        # File menu
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open Directory...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_directory)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        refresh_action = QAction("&Refresh", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self._refresh_requested)
        file_menu.addAction(refresh_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        self._toggle_tree_action = QAction("&File Tree", self)
        self._toggle_tree_action.setCheckable(True)
        self._toggle_tree_action.setChecked(False)  # Default: OFF
        self._toggle_tree_action.setShortcut("Ctrl+T")
        self._toggle_tree_action.triggered.connect(self._toggle_file_tree)
        view_menu.addAction(self._toggle_tree_action)

        self._toggle_filters_action = QAction("&Filter Panel", self)
        self._toggle_filters_action.setCheckable(True)
        self._toggle_filters_action.setChecked(False)  # Default: OFF
        self._toggle_filters_action.setShortcut("Ctrl+F")
        self._toggle_filters_action.triggered.connect(self._toggle_filter_panel)
        view_menu.addAction(self._toggle_filters_action)

        self._toggle_labels_action = QAction("&Node Labels", self)
        self._toggle_labels_action.setCheckable(True)
        self._toggle_labels_action.setChecked(True)
        self._toggle_labels_action.setShortcut("Ctrl+L")
        self._toggle_labels_action.triggered.connect(self._toggle_labels)
        view_menu.addAction(self._toggle_labels_action)

        view_menu.addSeparator()

        reset_view_action = QAction("&Reset View", self)
        reset_view_action.triggered.connect(self._reset_view)
        view_menu.addAction(reset_view_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        # Control panel signals
        self._control_panel.refresh_requested.connect(self._refresh_requested)
        self._control_panel.navigate_up.connect(self._navigate_up)
        self._control_panel.navigate_home.connect(self._navigate_home)
        self._control_panel.show_tree_toggled.connect(self._toggle_file_tree)
        self._control_panel.filter_panel_toggled.connect(self._toggle_filter_panel)
        self._control_panel.labels_toggled.connect(self._toggle_labels)

        # Search bar signal
        self._search_bar.search_requested.connect(self.search_requested.emit)

        # File tree signals
        self._file_tree.node_selected.connect(self._on_tree_node_selected)
        self._file_tree.node_double_clicked.connect(self._on_tree_node_double_clicked)

        # Filter panel signal
        self._filter_panel.filter_changed.connect(self.filter_changed.emit)

    # Menu actions

    def _open_directory(self) -> None:
        """Open a directory via file dialog."""
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Directory",
            str(self._root_path),
        )
        if path:
            self.directory_changed.emit(Path(path))

    def _refresh_requested(self) -> None:
        """Handle refresh request."""
        # Emit refresh_requested signal (connected to controller.refresh())
        self.refresh_requested.emit()

    def _navigate_up(self) -> None:
        """Navigate to parent directory."""
        parent = self._root_path.parent
        if parent != self._root_path and parent.exists():
            self.directory_changed.emit(parent)

    def _navigate_home(self) -> None:
        """Navigate to original root."""
        if self._original_root_path != self._root_path:
            self.directory_changed.emit(self._original_root_path)

    def _toggle_file_tree(self) -> None:
        """Toggle file tree visibility."""
        visible = not self._file_tree_dock.isVisible()
        self._file_tree_dock.setVisible(visible)
        self._toggle_tree_action.setChecked(visible)
        self._control_panel._show_tree_btn.setChecked(visible)

    def _toggle_filter_panel(self) -> None:
        """Toggle filter panel visibility."""
        visible = not self._filter_panel_dock.isVisible()
        self._filter_panel_dock.setVisible(visible)
        self._toggle_filters_action.setChecked(visible)
        self._control_panel._filter_panel_btn.setChecked(visible)

    def _toggle_labels(self) -> None:
        """Toggle node labels visibility."""
        self._show_labels = not self._show_labels
        if not self._show_labels:
            self._text_overlay.clear()
        self._toggle_labels_action.setChecked(self._show_labels)
        self._control_panel._show_labels_btn.setChecked(self._show_labels)

    def _reset_view(self) -> None:
        """Reset camera to default view."""
        if self._renderer:
            import numpy as np
            self._renderer.camera._state.position = np.array([10.0, 10.0, 10.0], dtype=np.float32)
            self._renderer.camera._state.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self._renderer.camera._update_orbit_from_position()
            self._renderer.camera.set_mode(CameraMode.ORBIT)

    def _show_about(self) -> None:
        """Show about dialog."""
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.about(
            self,
            "About pyfsn",
            "<h3>pyfsn - Python File System Navigator</h3>"
            "<p>A 3D interactive file system visualization tool inspired by SGI IRIX fsn.</p>"
            "<p>Version 0.1.0</p>"
            "<p>Camera Controls:</p>"
            "<ul>"
            "<li>Left-drag: Rotate camera (Orbit mode)</li>"
            "<li>Right-drag: Pan camera</li>"
            "<li>Shift+Left-drag: Pan camera (macOS trackpad)</li>"
            "<li>Middle-drag: Pan camera</li>"
            "<li>Scroll: Zoom in/out</li>"
            "<li>Click: Select node</li>"
            "<li>Double-click directory: Navigate to directory</li>"
            "<li>Double-click file: Open with default application</li>"
            "</ul>"
            "<p>Menu Shortcuts:</p>"
            "<ul>"
            "<li>Ctrl+O: Open directory</li>"
            "<li>F5: Refresh</li>"
            "<li>Ctrl+T: Toggle file tree</li>"
            "<li>Ctrl+F: Toggle filter panel</li>"
            "<li>Ctrl+L: Toggle node labels</li>"
            "<li>Ctrl+Q: Exit</li>"
            "</ul>"
        )

    def _on_tree_node_selected(self, node) -> None:
        """Handle node selection from file tree.

        Args:
            node: Selected node
        """
        # Navigate to node in 3D view
        if self._renderer and hasattr(self._renderer, 'snap_camera_to_node'):
            self._renderer.snap_camera_to_node(id(node))

    def _on_tree_node_double_clicked(self, node) -> None:
        """Handle node double-click from file tree.

        Args:
            node: Double-clicked node
        """
        # Emit signal for controller to handle
        self.tree_node_double_clicked.emit(node)

    def _set_camera_mode(self, mode: CameraMode) -> None:
        """Set camera mode.

        Args:
            mode: Camera mode to set
        """
        if self._renderer:
            self._renderer.set_camera_mode(mode)
            self._control_panel.set_camera_mode_display(mode)

    # Public API

    @property
    def renderer(self) -> Renderer:
        """Get the renderer widget."""
        return self._renderer

    @property
    def text_overlay(self) -> TextOverlay:
        """Get the text overlay widget."""
        return self._text_overlay

    @property
    def file_tree(self) -> FileTreeWidget:
        """Get the file tree widget."""
        return self._file_tree

    @property
    def show_labels(self) -> bool:
        """Get whether labels are shown."""
        return self._show_labels

    def update_stats(self, node_count: int, selected_count: int = 0) -> None:
        """Update statistics display.

        Args:
            node_count: Total number of nodes
            selected_count: Number of selected nodes
        """
        self._control_panel.update_stats(node_count, selected_count)

    @property
    def mini_map(self) -> MiniMap | None:
        """Get the mini map widget."""
        return self._mini_map

    def set_status_message(self, message: str) -> None:
        """Set status bar message.

        Args:
            message: Message to display
        """
        self._status_bar.showMessage(message)

    def set_root_path(self, path: Path) -> None:
        """Update the root path.

        Args:
            path: New root path
        """
        self._root_path = path
        self.setWindowTitle(f"pyfsn - {self._root_path}")

    def update_navigation_state(self, can_go_back: bool, can_go_forward: bool) -> None:
        """Update navigation button states.
        
        Args:
            can_go_back: Whether back navigation is available
            can_go_forward: Whether forward navigation is available
        """
        # TODO: Enable/disable back/forward buttons when they are added to UI
        pass

    def resizeEvent(self, event) -> None:
        """Handle window resize - update overlay geometry."""
        super().resizeEvent(event)
        if self._text_overlay and self._renderer:
            self._text_overlay.setGeometry(0, 0, self._renderer.width(), self._renderer.height())
        if hasattr(self, '_file_tooltip') and self._file_tooltip and self._renderer:
            self._file_tooltip.setGeometry(0, 0, self._renderer.width(), self._renderer.height())
        if hasattr(self, '_file_age_legend') and self._file_age_legend and self._renderer:
            self._file_age_legend.setGeometry(0, 0, self._renderer.width(), self._renderer.height())
        if hasattr(self, '_mini_map') and self._mini_map and self._renderer:
            # Update mini map geometry
            self._mini_map.setGeometry(0, 0, self._renderer.width(), self._renderer.height())
            # Reposition mini map to bottom-right corner
            self._mini_map.move(self._renderer.width() - self._mini_map.width() - 10,
                               self._renderer.height() - self._mini_map.height() - 10)
