"""Theme system for pyfsn 3D file system visualization.

Provides theme presets and theme data structures for controlling
the visual appearance of the 3D visualization.

Themes are compatible with both legacy (PyOpenGL) and modern (ModernGL) renderers.
All colors are stored as normalized RGBA tuples (0.0-1.0).
"""

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Theme:
    """Visual theme configuration for the 3D file system visualization.

    All color values are normalized RGBA tuples in the range [0.0, 1.0].

    Attributes:
        name: Human-readable theme name
        background_color: Sky/viewport background color (RGBA)
        platform_color: Directory platform color (RGBA)
        wire_color: Connection wire color (RGBA)
        selection_color: Selected node highlight color (RGBA)
        spotlight_color: Selection spotlight color (RGBA)
        enable_bloom: Enable bloom/glow post-processing effect
        enable_grid: Enable ground grid rendering
        fog_color: Depth fog color (RGBA), defaults to background_color
        fog_start: Distance at which fog begins
        fog_end: Distance at which fog reaches full density
    """

    name: str
    background_color: tuple[float, float, float, float]
    platform_color: tuple[float, float, float, float]
    wire_color: tuple[float, float, float, float]
    selection_color: tuple[float, float, float, float]
    spotlight_color: tuple[float, float, float, float]
    enable_bloom: bool = False
    enable_grid: bool = True
    fog_color: tuple[float, float, float, float] | None = None
    fog_start: float = 100.0
    fog_end: float = 500.0

    def __post_init__(self):
        """Set default fog color to match background if not specified."""
        if self.fog_color is None:
            self.fog_color = self.background_color


# Theme Presets

# SGI Classic theme - matches the original fsn aesthetic
SGI_CLASSIC = Theme(
    name="SGI Classic",
    background_color=(0.3, 0.5, 0.9, 1.0),  # Medium blue sky
    platform_color=(0.2, 0.6, 1.0, 1.0),   # Bright blue platforms
    wire_color=(1.0, 1.0, 1.0, 0.4),       # Semi-transparent white wires
    selection_color=(1.0, 1.0, 0.3, 1.0),  # Yellow selection
    spotlight_color=(1.0, 1.0, 0.8, 0.2),  # Pale yellow spotlight
    enable_bloom=False,
    enable_grid=True,
    fog_color=(0.1, 0.1, 0.15, 1.0),
)

# Dark Mode theme - reduced eye strain for dark environments
DARK_MODE = Theme(
    name="Dark Mode",
    background_color=(0.08, 0.08, 0.12, 1.0),  # Very dark blue-gray
    platform_color=(0.25, 0.35, 0.45, 1.0),    # Muted blue-gray
    wire_color=(0.6, 0.7, 0.8, 0.3),           # Dim bluish-white
    selection_color=(1.0, 0.8, 0.4, 1.0),      # Warm amber
    spotlight_color=(1.0, 0.9, 0.6, 0.15),     # Soft amber spotlight
    enable_bloom=False,
    enable_grid=True,
    fog_color=(0.05, 0.05, 0.08, 1.0),
)

# Cyberpunk theme - high contrast neon aesthetics
CYBERPUNK = Theme(
    name="Cyberpunk",
    background_color=(0.02, 0.0, 0.05, 1.0),   # Nearly black
    platform_color=(0.15, 0.1, 0.2, 1.0),     # Dark purple-gray
    wire_color=(0.0, 1.0, 1.0, 0.6),          # Cyan neon
    selection_color=(1.0, 0.0, 0.8, 1.0),     # Magenta selection
    spotlight_color=(1.0, 0.0, 0.8, 0.25),    # Magenta spotlight
    enable_bloom=True,                         # Strong bloom effect
    enable_grid=True,
    fog_color=(0.01, 0.0, 0.03, 1.0),
    fog_start=50.0,
    fog_end=300.0,
)

# Solarized theme - based on the Solarized color palette
SOLARIZED = Theme(
    name="Solarized",
    background_color=(0.0, 0.17, 0.21, 1.0),  # Base03
    platform_color=(0.27, 0.43, 0.45, 1.0),   # Base1
    wire_color=(0.35, 0.42, 0.43, 0.4),       # Base00
    selection_color=(0.86, 0.2, 0.18, 1.0),   # Red (selection)
    spotlight_color=(0.94, 0.27, 0.2, 0.2),   # Orange spotlight
    enable_bloom=False,
    enable_grid=True,
    fog_color=(0.0, 0.12, 0.15, 1.0),
)

# Forest theme - nature-inspired green tones
FOREST = Theme(
    name="Forest",
    background_color=(0.05, 0.15, 0.1, 1.0),  # Dark green
    platform_color=(0.2, 0.5, 0.3, 1.0),      # Medium green
    wire_color=(0.6, 0.8, 0.6, 0.35),         # Light green
    selection_color=(1.0, 0.9, 0.3, 1.0),     # Yellow selection
    spotlight_color=(1.0, 0.95, 0.5, 0.18),   # Warm yellow spotlight
    enable_bloom=False,
    enable_grid=True,
    fog_color=(0.03, 0.1, 0.07, 1.0),
)

# Ocean theme - deep sea blue tones
OCEAN = Theme(
    name="Ocean",
    background_color=(0.0, 0.05, 0.15, 1.0),  # Deep blue
    platform_color=(0.0, 0.3, 0.5, 1.0),      # Sea blue
    wire_color=(0.4, 0.7, 0.9, 0.35),         # Foam white
    selection_color=(0.0, 0.9, 1.0, 1.0),     # Cyan selection
    spotlight_color=(0.0, 0.85, 0.95, 0.2),   # Cyan spotlight
    enable_bloom=False,
    enable_grid=True,
    fog_color=(0.0, 0.03, 0.1, 1.0),
)

# Default theme registry - available themes in the application
BUILTIN_THEMES: dict[str, Theme] = {
    "sgi_classic": SGI_CLASSIC,
    "dark_mode": DARK_MODE,
    "cyberpunk": CYBERPUNK,
    "solarized": SOLARIZED,
    "forest": FOREST,
    "ocean": OCEAN,
}

# Default theme used when no preference is set
DEFAULT_THEME = SGI_CLASSIC


def get_theme(name: str) -> Theme:
    """Get a theme by name.

    Args:
        name: Theme name (e.g., "sgi_classic", "cyberpunk")

    Returns:
        Theme object

    Raises:
        KeyError: If theme name is not found
    """
    return BUILTIN_THEMES[name]


def list_themes() -> list[str]:
    """List all available theme names.

    Returns:
        List of theme identifier strings
    """
    return list(BUILTIN_THEMES.keys())


def register_theme(theme: Theme) -> None:
    """Register a custom theme.

    Args:
        theme: Theme object to register

    Note:
        Custom themes can be registered at runtime and will be
        available alongside built-in themes.
    """
    BUILTIN_THEMES[theme.name.lower().replace(" ", "_")] = theme


# Color helper functions

def blend_colors(
    color1: tuple[float, float, float, float],
    color2: tuple[float, float, float, float],
    factor: float
) -> tuple[float, float, float, float]:
    """Blend two colors by a factor.

    Args:
        color1: First color (RGBA)
        color2: Second color (RGBA)
        factor: Blend factor (0.0 = color1, 1.0 = color2)

    Returns:
        Blended color (RGBA)
    """
    factor = max(0.0, min(1.0, factor))
    return tuple(
        c1 + (c2 - c1) * factor
        for c1, c2 in zip(color1, color2)
    )


def adjust_brightness(
    color: tuple[float, float, float, float],
    factor: float
) -> tuple[float, float, float, float]:
    """Adjust color brightness.

    Args:
        color: Input color (RGBA)
        factor: Brightness factor (1.0 = no change, >1.0 = brighter, <1.0 = darker)

    Returns:
        Adjusted color (RGBA)
    """
    r, g, b, a = color
    return (
        min(1.0, r * factor),
        min(1.0, g * factor),
        min(1.0, b * factor),
        a,
    )


def hex_to_rgba(hex_color: str) -> tuple[float, float, float, float]:
    """Convert hex color string to normalized RGBA tuple.

    Args:
        hex_color: Hex color string (e.g., "#FF5500" or "FF5500")

    Returns:
        RGBA tuple with values in [0.0, 1.0]
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return (r, g, b, 1.0)
    elif len(hex_color) == 8:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        a = int(hex_color[6:8], 16) / 255.0
        return (r, g, b, a)
    else:
        raise ValueError(f"Invalid hex color: {hex_color}")
