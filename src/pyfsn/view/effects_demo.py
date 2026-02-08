"""Demo script for cyberpunk visual effects.

This script demonstrates the advanced shaders and post-processing effects
implemented for the pyfsn project, including:

1. Emissive materials - file type-based glow
2. Wire pulse effect - animated connections
3. Bloom effect - post-processing glow

Usage:
    python -m pyfsn.view.effects_demo
"""

import sys
import time
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QSlider
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from pyfsn.model.node import Node, NodeType
from pyfsn.layout.engine import LayoutEngine
from pyfsn.controller.controller import Controller
from pyfsn.view.renderer import Renderer, ColorMode


class EffectsDemoWindow(QMainWindow):
    """Demo window showcasing cyberpunk visual effects."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("pyfsn - Cyberpunk Effects Demo")
        self.resize(1200, 800)

        # Create central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Create info label
        self.info_label = QLabel(
            "<b>Cyberpunk Effects Demo</b><br>"
            "Emissive materials | Wire pulse | Bloom effect<br>"
            "<br>"
            "Controls:<br>"
            "- Bloom: Adjust intensity slider<br>"
            "- Navigate: Left-click drag to rotate, Right-click drag to pan<br>"
            "- Scroll: Zoom in/out<br>"
            "- Double-click: Navigate to directory"
        )
        self.info_label.setStyleSheet("padding: 10px; background: #1a1a2e; color: #00ffcc;")
        layout.addWidget(self.info_label)

        # Create renderer
        self.renderer = Renderer()
        layout.addWidget(self.renderer, stretch=1)

        # Create bloom control slider
        bloom_layout = QVBoxLayout()
        bloom_label = QLabel("Bloom Intensity:")
        bloom_label.setStyleSheet("color: #00ffcc;")
        bloom_layout.addWidget(bloom_label)

        self.bloom_slider = QSlider(Qt.Orientation.Horizontal)
        self.bloom_slider.setRange(0, 100)
        self.bloom_slider.setValue(30)
        self.bloom_slider.valueChanged.connect(self._on_bloom_changed)
        bloom_layout.addWidget(self.bloom_slider)

        layout.addLayout(bloom_layout)

        # Create test file system
        self._create_test_filesystem()

        # Timer for status updates
        self._start_time = time.time()
        self.renderer._update_timer.timeout.connect(self._update_status)

    def _create_test_filesystem(self):
        """Create a test file system for demonstration."""
        # Create mock nodes with various file types
        nodes = {
            "/root": self._create_node("/root", NodeType.DIRECTORY),
            "/root/src": self._create_node("/root/src", NodeType.DIRECTORY),
            "/root/src/main.py": self._create_node("/root/src/main.py", NodeType.FILE, ".py"),
            "/root/src/utils.py": self._create_node("/root/src/utils.py", NodeType.FILE, ".py"),
            "/root/src/config.json": self._create_node("/root/src/config.json", NodeType.FILE, ".json"),
            "/root/docs": self._create_node("/root/docs", NodeType.DIRECTORY),
            "/root/docs/README.md": self._create_node("/root/docs/README.md", NodeType.FILE, ".md"),
            "/root/tests": self._create_node("/root/tests", NodeType.DIRECTORY),
            "/root/tests/test_main.py": self._create_node("/root/tests/test_main.py", NodeType.FILE, ".py"),
        }

        # Create layout
        layout_engine = LayoutEngine()
        layout_result = layout_engine.layout(nodes)

        # Load into renderer
        self.renderer.load_layout(layout_result, nodes)

        # Set color mode to age for variety
        self.renderer.set_color_mode(ColorMode.AGE)

    def _create_node(self, path: str, node_type: NodeType, suffix: str = "") -> Node:
        """Create a mock node for testing.

        Args:
            path: Node path
            node_type: Type of node
            suffix: File suffix (for files)

        Returns:
            Node object
        """
        from pyfsn.model.node import Node

        # Create node with mock data
        node = Node(
            path=Path(path),
            name=Path(path).name,
            type=node_type,
            size=1000 if node_type == NodeType.FILE else 0,
            mtime=time.time() - np.random.randint(0, 30 * 86400),  # Random age
            suffix=suffix,
        )
        return node

    def _on_bloom_changed(self, value: int):
        """Handle bloom intensity slider change.

        Args:
            value: Slider value (0-100)
        """
        intensity = value / 100.0
        self.renderer.set_bloom_intensity(intensity)
        self.info_label.setText(
            f"<b>Cyberpunk Effects Demo</b><br>"
            f"Emissive materials | Wire pulse | Bloom effect<br>"
            f"<br>"
            f"Controls:<br>"
            f"- Bloom: {intensity:.2f}<br>"
            f"- Navigate: Left-click drag to rotate, Right-click drag to pan<br>"
            f"- Scroll: Zoom in/out<br>"
            f"- Double-click: Navigate to directory"
        )

    def _update_status(self):
        """Update status information."""
        elapsed = time.time() - self._start_time
        stats = self.renderer.performance_stats

        # Update info with performance stats
        current_text = self.info_label.text()
        if "Bloom:" in current_text:
            # Preserve bloom info
            lines = current_text.split("<br>")
            base_text = "<br>".join(lines[:4])
        else:
            base_text = (
                "<b>Cyberpunk Effects Demo</b><br>"
                "Emissive materials | Wire pulse | Bloom effect<br>"
                "<br>"
            )

        self.info_label.setText(
            f"{base_text}"
            f"Controls:<br>"
            f"- Bloom: {self.renderer.get_bloom_intensity():.2f}<br>"
            f"- Navigate: Left-click drag to rotate, Right-click drag to pan<br>"
            f"- Scroll: Zoom in/out<br>"
            f"- Double-click: Navigate to directory<br>"
            f"<br>"
            f"Performance: {stats['fps']:.1f} FPS | "
            f"{stats['instance_count']} instances | "
            f"{stats['draw_calls']} draw calls"
        )


def main():
    """Run the effects demo."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark theme for cyberpunk feel
    app.setPalette(QApplication.style().standardPalette())

    window = EffectsDemoWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
