"""Main entry point for pyfsn."""

import argparse
import sys
from pathlib import Path

# Fix for Python 3.14+ recursion limit issue with standard library imports
# See: https://github.com/python/cpython/issues/...
sys.setrecursionlimit(3000)

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

from pyfsn.controller.controller import Controller
from pyfsn.layout.engine import LayoutConfig, LayoutEngine, PlacementStrategy
from pyfsn.view.renderer import ColorMode


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        prog="pyfsn",
        description="Python File System Navigator - 3D interactive file visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path.cwd(),
        help="Root directory path to visualize (default: current directory)",
    )
    parser.add_argument(
        "--renderer",
        choices=["auto", "modern", "legacy"],
        default="auto",
        help="Renderer to use: auto (detect best), modern (ModernGL), legacy (PyOpenGL) (default: auto)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        metavar="N",
        help="Maximum depth to visualize (default: 5)",
    )
    parser.add_argument(
        "--lazy-depth",
        type=int,
        default=2,
        metavar="N",
        help="Depth at which to start lazy loading (default: 2)",
    )
    parser.add_argument(
        "--show-hidden",
        action="store_true",
        help="Show hidden files and directories (starting with '.')",
    )
    parser.add_argument(
        "--color-mode",
        choices=["type", "age"],
        default="age",
        help="Color mode for file visualization (default: age)",
    )
    parser.add_argument(
        "--msaa",
        type=int,
        choices=[0, 2, 4, 8],
        default=4,
        metavar="N",
        help="Multisample anti-aliasing samples (default: 4)",
    )
    parser.add_argument(
        "--no-tooltips",
        action="store_true",
        help="Disable tooltips on hover",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Disable node name labels",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point.

    Returns:
        Exit code
    """
    args = parse_args()

    # Resolve the root path
    root_path = args.path.resolve()

    # Validate the path
    if not root_path.exists():
        print(f"Error: Path '{root_path}' does not exist", file=sys.stderr)
        return 1

    if not root_path.is_dir():
        print(f"Error: Path '{root_path}' is not a directory", file=sys.stderr)
        return 1

    # Create Qt application
    app = QApplication(sys.argv)

    # Configure OpenGL surface format based on renderer choice
    from PyQt6.QtGui import QSurfaceFormat
    fmt = QSurfaceFormat()

    renderer = args.renderer
    if renderer == "legacy":
        # Legacy OpenGL 2.1 for PyOpenGL - never attempts ModernGL
        fmt.setVersion(2, 1)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
    elif renderer == "modern":
        # Modern OpenGL 3.2+ core for ModernGL
        fmt.setVersion(3, 2)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    else:  # auto
        # Auto-detect: use legacy OpenGL 2.1 for maximum compatibility
        # Can be upgraded to ModernGL detection in the future
        fmt.setVersion(2, 1)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)

    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSamples(args.msaa)
    QSurfaceFormat.setDefaultFormat(fmt)

    # Create layout config with CLI options
    layout_config = LayoutConfig(
        node_size=1.0,
        dir_size=2.0,
        spacing=0.5,
        padding=0.2,
        max_depth=args.max_depth,
        placement_strategy=PlacementStrategy.GRID,
        grid_size=3.0,
        connection_width=0.1,
        use_size_height=True,
        min_height=0.2,
        max_height=5.0,
        height_scale=0.3,
    )

    # Create and start controller
    controller = Controller(root_path)

    # Apply CLI configuration
    controller._layout_config = layout_config
    controller._layout_engine = LayoutEngine(layout_config)

    # Set scanner options
    controller._scanner._lazy_depth = args.lazy_depth

    # Set color mode
    color_mode = ColorMode.TYPE if args.color_mode == "type" else ColorMode.AGE
    controller._renderer.set_color_mode(color_mode)

    # Set label visibility
    if args.no_labels:
        controller._window._show_labels = False
        controller._window._control_panel._show_labels_btn.setChecked(False)

    # Set tooltip visibility (if supported)
    if args.no_tooltips:
        # Disable tooltips on the renderer
        controller._renderer.setToolTip("")

    controller.start()
    controller.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
