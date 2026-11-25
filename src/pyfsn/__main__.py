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
from pyfsn.layout.engine import DirHeightMode, HeightMode, LayoutConfig
from pyfsn.view.renderer import LANDSCAPES, ColorMode


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
        default=None,
        help="Root directory path to visualize (default: last used or current directory)",
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
        "--height",
        choices=[m.value for m in HeightMode],
        default=None,
        help="How size maps to height: none, linear, exaggerated (sqrt, fsn style), "
        "log (default: last used or exaggerated)",
    )
    parser.add_argument(
        "--dir-height",
        choices=[m.value for m in DirHeightMode],
        default=None,
        help="What pedestal height stands for: flat, files (files directly inside), "
        "children (including sub-directories) (default: last used or files)",
    )
    parser.add_argument(
        "--landscape",
        choices=list(LANDSCAPES),
        default=None,
        help="Sky and ground colours (default: last used or grass)",
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
    parser.add_argument(
        "--no-sound",
        action="store_true",
        help="Disable UI sound effects",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point.

    Returns:
        Exit code
    """
    args = parse_args()

    # Resolve the root path. With no explicit path, prefer the last used
    # directory (persisted across sessions), falling back to the cwd.
    if args.path is not None:
        root_path = args.path.resolve()
    else:
        from PyQt6.QtCore import QSettings

        last = QSettings("pyfsn", "pyfsn").value("last_path", "", type=str)
        candidate = Path(last) if last else Path.cwd()
        root_path = candidate.resolve() if candidate.exists() else Path.cwd().resolve()

    # Validate the path
    if not root_path.exists():
        print(f"Error: Path '{root_path}' does not exist", file=sys.stderr)
        return 1

    if not root_path.is_dir():
        print(f"Error: Path '{root_path}' is not a directory", file=sys.stderr)
        return 1

    # Create Qt application
    app = QApplication(sys.argv)

    # Set application icon
    from PyQt6.QtGui import QIcon
    icon_path = Path(__file__).parent / "icon.png"
    
    # Handle PyInstaller bundled resources
    if hasattr(sys, '_MEIPASS'):
        # Check for bundled icon in the extracted resource directory
        bundled_icon = Path(sys._MEIPASS) / "pyfsn" / "icon.png"
        if bundled_icon.exists():
            icon_path = bundled_icon
        else:
            bundled_icon_ico = Path(sys._MEIPASS) / "pyfsn" / "icon.ico"
            if bundled_icon_ico.exists():
                icon_path = bundled_icon_ico

    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

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

    # Create layout config with CLI options. Height modes left unset on the
    # command line fall back to the persisted View menu choice.
    layout_config = LayoutConfig(max_depth=args.max_depth)

    # Create and start controller
    controller = Controller(
        root_path,
        show_hidden=args.show_hidden,
        lazy_depth=args.lazy_depth,
        layout_config=layout_config,
        height_mode=HeightMode(args.height) if args.height else None,
        dir_height_mode=DirHeightMode(args.dir_height) if args.dir_height else None,
        landscape=args.landscape,
    )

    # Set color mode
    color_mode = ColorMode.TYPE if args.color_mode == "type" else ColorMode.AGE
    controller._renderer.set_color_mode(color_mode)

    # Set label visibility
    if args.no_labels:
        controller._window._show_labels = False
        controller._window._control_panel._show_labels_btn.setChecked(False)

    # Disable sound effects if requested
    if args.no_sound:
        controller._sound.enabled = False

    # Set tooltip visibility (if supported)
    if args.no_tooltips:
        # Disable tooltips on the renderer
        controller._renderer.setToolTip("")

    controller.start()
    controller.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
