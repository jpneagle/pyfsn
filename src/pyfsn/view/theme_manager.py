"""Theme manager for pyfsn 3D file system visualization.

Manages theme switching, persistence, and change notifications.
"""

import json
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

from pyfsn.view.theme import (
    Theme,
    DEFAULT_THEME,
    BUILTIN_THEMES,
    list_themes,
    get_theme,
    register_theme,
)


class ThemeManager(QObject):
    """Manager for visual themes in the 3D file system visualization.

    Handles theme selection, switching, persistence, and change notifications.
    Emits signals when the theme changes so UI components can update.

    Usage:
        manager = ThemeManager()
        manager.load_preferences()  # Load saved theme
        manager.set_theme("cyberpunk")  # Switch theme
    """

    # Signal emitted when theme changes
    theme_changed = pyqtSignal(Theme)

    # Signal emitted when bloom setting changes
    bloom_changed = pyqtSignal(bool)

    # Signal emitted when grid visibility changes
    grid_changed = pyqtSignal(bool)

    def __init__(self, parent: QObject | None = None) -> None:
        """Initialize the theme manager.

        Args:
            parent: Parent QObject (optional)
        """
        super().__init__(parent)

        self._current_theme: Theme = DEFAULT_THEME
        self._preferences_path: Path = Path.home() / ".config" / "pyfsn" / "theme_preferences.json"

    @property
    def current_theme(self) -> Theme:
        """Get the currently active theme."""
        return self._current_theme

    @property
    def theme_name(self) -> str:
        """Get the name of the currently active theme."""
        return self._current_theme.name

    def set_theme(self, name: str) -> Theme:
        """Set the current theme by name.

        Args:
            name: Theme name (e.g., "sgi_classic", "cyberpunk")

        Returns:
            The newly set Theme object

        Raises:
            KeyError: If theme name is not found

        Emits:
            theme_changed: With the new theme
            bloom_changed: With the bloom enablement state
            grid_changed: With the grid visibility state
        """
        theme = get_theme(name)
        self._current_theme = theme

        # Emit signals for UI updates
        self.theme_changed.emit(theme)
        self.bloom_changed.emit(theme.enable_bloom)
        self.grid_changed.emit(theme.enable_grid)

        return theme

    def set_theme_object(self, theme: Theme) -> Theme:
        """Set the current theme from a Theme object.

        Args:
            theme: Theme object to set as current

        Returns:
            The same Theme object

        Emits:
            theme_changed: With the new theme
            bloom_changed: With the bloom enablement state
            grid_changed: With the grid visibility state

        Note:
            This method is useful for custom themes that may not be
            registered in the built-in theme registry.
        """
        self._current_theme = theme

        # Emit signals for UI updates
        self.theme_changed.emit(theme)
        self.bloom_changed.emit(theme.enable_bloom)
        self.grid_changed.emit(theme.enable_grid)

        return theme

    def list_available_themes(self) -> list[str]:
        """Get list of all available theme names.

        Returns:
            List of theme identifier strings
        """
        return list_themes()

    def get_theme_info(self, name: str) -> dict[str, Any]:
        """Get information about a theme.

        Args:
            name: Theme name

        Returns:
            Dictionary with theme information:
                - name: Theme display name
                - enable_bloom: Bloom enabled
                - enable_grid: Grid enabled
                - colors: Dict of color names to hex values
        """
        theme = get_theme(name)

        def rgba_to_hex(color: tuple[float, float, float, float]) -> str:
            r, g, b, a = color
            return f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}{int(a * 255):02X}"

        return {
            "name": theme.name,
            "enable_bloom": theme.enable_bloom,
            "enable_grid": theme.enable_grid,
            "colors": {
                "background": rgba_to_hex(theme.background_color),
                "platform": rgba_to_hex(theme.platform_color),
                "wire": rgba_to_hex(theme.wire_color),
                "selection": rgba_to_hex(theme.selection_color),
                "spotlight": rgba_to_hex(theme.spotlight_color),
                "fog": rgba_to_hex(theme.fog_color),
            },
        }

    def save_preferences(self, path: Path | None = None) -> None:
        """Save current theme preferences to disk.

        Args:
            path: Optional custom path for preferences file.
                  Defaults to ~/.config/pyfsn/theme_preferences.json
        """
        save_path = path or self._preferences_path

        # Create parent directories if needed
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Find theme key for current theme
        theme_key = None
        for key, theme in BUILTIN_THEMES.items():
            if theme is self._current_theme:
                theme_key = key
                break

        # If it's a custom theme, save by name
        if theme_key is None:
            theme_key = self._current_theme.name.lower().replace(" ", "_")

        preferences = {
            "theme": theme_key,
            "version": "1.0",
        }

        with open(save_path, "w") as f:
            json.dump(preferences, f, indent=2)

    def load_preferences(self, path: Path | None = None) -> Theme:
        """Load theme preferences from disk.

        Args:
            path: Optional custom path for preferences file.
                  Defaults to ~/.config/pyfsn/theme_preferences.json

        Returns:
            The loaded theme, or DEFAULT_THEME if loading fails

        Note:
            Emits theme_changed signal if a theme is successfully loaded.
        """
        load_path = path or self._preferences_path

        if not load_path.exists():
            # No preferences saved, use default
            self._current_theme = DEFAULT_THEME
            return DEFAULT_THEME

        try:
            with open(load_path, "r") as f:
                preferences = json.load(f)

            theme_name = preferences.get("theme")
            if theme_name and theme_name in BUILTIN_THEMES:
                return self.set_theme(theme_name)
            else:
                # Unknown theme, use default
                self._current_theme = DEFAULT_THEME
                return DEFAULT_THEME

        except (json.JSONDecodeError, OSError, KeyError):
            # Error loading preferences, use default
            self._current_theme = DEFAULT_THEME
            return DEFAULT_THEME

    def reset_to_default(self) -> Theme:
        """Reset to the default theme.

        Returns:
            The default theme

        Emits:
            theme_changed: With the default theme
        """
        return self.set_theme_object(DEFAULT_THEME)

    def create_custom_theme(
        self,
        name: str,
        base_theme: str | None = None,
        **kwargs
    ) -> Theme:
        """Create and register a custom theme.

        Args:
            name: Human-readable theme name
            base_theme: Optional theme to use as base (copies values)
            **kwargs: Theme attributes to override:
                - background_color: RGBA tuple
                - platform_color: RGBA tuple
                - wire_color: RGBA tuple
                - selection_color: RGBA tuple
                - spotlight_color: RGBA tuple
                - enable_bloom: bool
                - enable_grid: bool
                - fog_color: RGBA tuple (optional)
                - fog_start: float
                - fog_end: float

        Returns:
            The newly created Theme object

        Example:
            manager.create_custom_theme(
                "My Theme",
                base_theme="dark_mode",
                platform_color=(1.0, 0.5, 0.0, 1.0),
                enable_bloom=True
            )
        """
        # Start with base theme or defaults
        if base_theme:
            base = get_theme(base_theme)
            theme_dict = {
                "name": name,
                "background_color": base.background_color,
                "platform_color": base.platform_color,
                "wire_color": base.wire_color,
                "selection_color": base.selection_color,
                "spotlight_color": base.spotlight_color,
                "enable_bloom": base.enable_bloom,
                "enable_grid": base.enable_grid,
                "fog_color": base.fog_color,
                "fog_start": base.fog_start,
                "fog_end": base.fog_end,
            }
        else:
            theme_dict = {
                "name": name,
                "background_color": (0.1, 0.1, 0.15, 1.0),
                "platform_color": (0.5, 0.5, 0.5, 1.0),
                "wire_color": (1.0, 1.0, 1.0, 0.5),
                "selection_color": (1.0, 1.0, 0.0, 1.0),
                "spotlight_color": (1.0, 1.0, 0.8, 0.2),
                "enable_bloom": False,
                "enable_grid": True,
                "fog_color": None,
                "fog_start": 100.0,
                "fog_end": 500.0,
            }

        # Apply overrides
        theme_dict.update(kwargs)

        # Create and register theme
        theme = Theme(**theme_dict)
        theme_key = name.lower().replace(" ", "_")
        register_theme(theme)

        return theme


# Global singleton instance
_global_manager: ThemeManager | None = None


def get_theme_manager() -> ThemeManager:
    """Get the global theme manager singleton.

    Returns:
        The global ThemeManager instance

    Note:
        The singleton is created on first call.
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = ThemeManager()
    return _global_manager
