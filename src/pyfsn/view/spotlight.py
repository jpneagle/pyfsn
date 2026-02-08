"""Spotlight search visualization for highlighting matching nodes.

Provides visual search functionality that dims non-matching nodes and
highlights matching ones with a "spotlight" effect.
"""

import time
import numpy as np
from dataclasses import dataclass

from pyfsn.model.node import Node


@dataclass
class SpotlightAnimation:
    """Animation state for spotlight effects."""
    start_time: float
    duration: float
    from_opacity: float
    to_opacity: float


class SpotlightSearch:
    """Manages search visualization with spotlight effects.

    When a search is active:
    - Matching nodes are highlighted with full opacity and enhanced emission
    - Non-matching nodes are dimmed to 30% opacity and desaturated
    - Optional cone spotlight models appear above matching nodes
    """

    def __init__(self):
        """Initialize the spotlight search manager."""
        # Search state
        self._is_active: bool = False
        self._matching_paths: set[str] = set()
        self._search_query: str = ""

        # Animation state
        self._animation_start_time: float | None = None
        self._animation_duration: float = 0.3  # 300ms fade in/out
        self._current_opacity: float = 1.0  # Global opacity multiplier

        # Spotlight visual settings
        self._dim_opacity = 0.3  # Opacity for non-matching nodes
        self._match_opacity = 1.0  # Opacity for matching nodes
        self._desaturate_non_matches = True  # Grayscale for non-matches

        # Cone spotlight settings
        self._show_cone_spotlights = True
        self._cone_height = 6.0
        self._cone_radius = 1.5

    def start_search(self, query: str, matching_nodes: set[Node]) -> None:
        """Start a new search with spotlight visualization.

        Args:
            query: Search query string
            matching_nodes: Set of nodes that match the search
        """
        self._search_query = query
        self._matching_paths = {str(node.path) for node in matching_nodes}
        self._is_active = True

        # Start animation from current opacity
        self._animation_start_time = time.time()

    def update_results(self, matching_nodes: set[Node]) -> None:
        """Update search results without resetting animation.

        Args:
            matching_nodes: Set of nodes that match the search
        """
        self._matching_paths = {str(node.path) for node in matching_nodes}

    def clear_search(self) -> None:
        """Clear the search and reset visualization."""
        if self._is_active:
            # Set inactive flag to trigger fade-out animation
            self._is_active = False
            # Start fade-out animation
            self._animation_start_time = time.time()
        else:
            # Already inactive, just clear
            self._matching_paths.clear()
            self._search_query = ""

    def is_match(self, path: str) -> bool:
        """Check if a node path matches the current search.

        Args:
            path: Node path string

        Returns:
            True if the path matches the search
        """
        return path in self._matching_paths

    def get_node_opacity(self, path: str) -> float:
        """Get the opacity multiplier for a node based on match status.

        Args:
            path: Node path string

        Returns:
            Opacity multiplier (0.0-1.0)
        """
        # Use the property which accounts for animation state
        if not self.is_active:
            return 1.0

        # Get animation progress
        animation_factor = self._get_animation_factor()

        if self.is_match(path):
            # Matching node: full opacity
            return self._match_opacity * animation_factor
        else:
            # Non-matching node: dimmed
            base_opacity = self._dim_opacity
            # Ensure minimum visibility for structure
            return max(0.1, base_opacity * animation_factor)

    def get_node_color(self, path: str, original_color: np.ndarray) -> np.ndarray:
        """Get the modified color for a node based on match status.

        Args:
            path: Node path string
            original_color: Original RGBA color array

        Returns:
            Modified RGBA color array
        """
        # Use the property which accounts for animation state
        if not self.is_active:
            return original_color

        color = original_color.copy()

        if not self.is_match(path) and self._desaturate_non_matches:
            # Desaturate non-matching nodes
            # Convert to grayscale while preserving alpha
            luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            color[0] = luminance
            color[1] = luminance
            color[2] = luminance

        # Apply opacity
        opacity = self.get_node_opacity(path)
        color[3] = original_color[3] * opacity

        return color

    def should_draw_spotlight(self, path: str) -> bool:
        """Check if a cone spotlight should be drawn for a node.

        Args:
            path: Node path string

        Returns:
            True if a spotlight should be drawn
        """
        return (
            self._is_active and
            self._show_cone_spotlights and
            self.is_match(path) and
            self._get_animation_factor() > 0.5
        )

    def get_spotlight_params(self, path: str) -> tuple[float, float, float] | None:
        """Get spotlight parameters for a matching node.

        Args:
            path: Node path string

        Returns:
            Tuple of (height, radius, alpha) or None
        """
        if not self.should_draw_spotlight(path):
            return None

        # Fade in spotlight based on animation
        animation_factor = self._get_animation_factor()
        alpha = 0.3 * animation_factor  # Max 0.3 alpha

        return (self._cone_height, self._cone_radius, alpha)

    def _get_animation_factor(self) -> float:
        """Get current animation progress factor (0.0-1.0).

        Returns:
            Animation factor
        """
        if self._animation_start_time is None:
            return 1.0 if self._is_active else 0.0

        elapsed = time.time() - self._animation_start_time
        progress = min(elapsed / self._animation_duration, 1.0)

        # Ease-in-out function
        t = progress
        ease = t * t * (3.0 - 2.0 * t)

        if self._is_active:
            # Fade in
            if progress >= 1.0:
                self._animation_start_time = None
                return 1.0
            return ease
        else:
            # Fade out
            if progress >= 1.0:
                self._matching_paths.clear()
                self._search_query = ""
                self._animation_start_time = None  # Clear animation time
                return 0.0
            return 1.0 - ease

    @property
    def is_active(self) -> bool:
        """Check if spotlight search is currently active."""
        return self._is_active or (self._animation_start_time is not None)

    @property
    def match_count(self) -> int:
        """Get the number of matching nodes."""
        return len(self._matching_paths)

    @property
    def search_query(self) -> str:
        """Get the current search query."""
        return self._search_query
