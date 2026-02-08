"""Performance monitoring and optimization utilities.

Provides tools for monitoring FPS, optimizing rendering, and
implementing progressive loading for large file systems.
"""

import time
from dataclasses import dataclass, field
from collections import deque
from typing import Callable

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, QTimer


@dataclass
class PerformanceMetrics:
    """Performance metrics for rendering."""

    fps: float = 0.0
    frame_time_ms: float = 0.0
    draw_calls: int = 0
    instance_count: int = 0
    visible_instances: int = 0
    triangle_count: int = 0


class PerformanceMonitor(QObject):
    """Monitor and track rendering performance."""

    metrics_updated = pyqtSignal(object)  # Emits PerformanceMetrics

    def __init__(self, window_size: int = 60) -> None:
        """Initialize performance monitor.

        Args:
            window_size: Number of frames to average for FPS calculation
        """
        super().__init__()

        self._window_size = window_size
        self._frame_times: deque[float] = deque(maxlen=window_size)
        self._last_frame_time = time.perf_counter()

        self._metrics = PerformanceMetrics()
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_metrics)
        self._update_timer.start(250)  # Update 4 times per second

    def start_frame(self) -> None:
        """Mark the start of a frame."""
        self._last_frame_time = time.perf_counter()

    def end_frame(self) -> float:
        """Mark the end of a frame and return frame time.

        Returns:
            Frame time in milliseconds
        """
        current_time = time.perf_counter()
        frame_time = current_time - self._last_frame_time
        frame_time_ms = frame_time * 1000.0

        self._frame_times.append(frame_time)
        return frame_time_ms

    def _update_metrics(self) -> None:
        """Update and emit performance metrics."""
        if not self._frame_times:
            return

        # Calculate average FPS
        avg_frame_time = sum(self._frame_times) / len(self._frame_times)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

        self._metrics.fps = fps
        self._metrics.frame_time_ms = avg_frame_time * 1000.0

        self.metrics_updated.emit(self._metrics)

    def set_draw_calls(self, count: int) -> None:
        """Set the number of draw calls.

        Args:
            count: Number of draw calls
        """
        self._metrics.draw_calls = count

    def set_instance_count(self, count: int) -> None:
        """Set the total instance count.

        Args:
            count: Total number of instances
        """
        self._metrics.instance_count = count

    def set_visible_instances(self, count: int) -> None:
        """Set the number of visible instances.

        Args:
            count: Number of visible instances
        """
        self._metrics.visible_instances = count

    def set_triangle_count(self, count: int) -> None:
        """Set the triangle count.

        Args:
            count: Number of triangles rendered
        """
        self._metrics.triangle_count = count

    @property
    def metrics(self) -> PerformanceMetrics:
        """Get current metrics."""
        return self._metrics

    @property
    def current_fps(self) -> float:
        """Get current FPS."""
        return self._metrics.fps


class FrustumCuller:
    """Frustum culling for visibility determination.

    Determines which nodes are visible within the camera's view frustum.
    """

    def __init__(self) -> None:
        """Initialize frustum culler."""
        self._planes: list[tuple[np.ndarray, float]] = []  # (normal, distance)

    def update_from_camera(self, view_matrix: np.ndarray, projection_matrix: np.ndarray, aspect_ratio: float) -> None:
        """Update frustum planes from camera matrices.

        Args:
            view_matrix: View matrix
            projection_matrix: Projection matrix
            aspect_ratio: Aspect ratio
        """


        # Combined view-projection matrix
        vp = projection_matrix @ view_matrix

        # Extract frustum planes from VP matrix
        # Each plane is (normal, distance) where plane is defined as dot(normal, point) + distance = 0
        self._planes = []

        # Left plane (column 0 + column 3)
        normal = np.array([vp[0, 3] + vp[0, 0], vp[1, 3] + vp[1, 0], vp[2, 3] + vp[2, 0]], dtype=np.float32)
        length = np.linalg.norm(normal)
        if length > 0:
            self._planes.append((normal / length, (vp[3, 3] + vp[3, 0]) / length))

        # Right plane (column 3 - column 0)
        normal = np.array([vp[0, 3] - vp[0, 0], vp[1, 3] - vp[1, 0], vp[2, 3] - vp[2, 0]], dtype=np.float32)
        length = np.linalg.norm(normal)
        if length > 0:
            self._planes.append((normal / length, (vp[3, 3] - vp[3, 0]) / length))

        # Bottom plane (column 1 + column 3)
        normal = np.array([vp[0, 3] + vp[0, 1], vp[1, 3] + vp[1, 1], vp[2, 3] + vp[2, 1]], dtype=np.float32)
        length = np.linalg.norm(normal)
        if length > 0:
            self._planes.append((normal / length, (vp[3, 3] + vp[3, 1]) / length))

        # Top plane (column 3 - column 1)
        normal = np.array([vp[0, 3] - vp[0, 1], vp[1, 3] - vp[1, 1], vp[2, 3] - vp[2, 1]], dtype=np.float32)
        length = np.linalg.norm(normal)
        if length > 0:
            self._planes.append((normal / length, (vp[3, 3] - vp[3, 1]) / length))

        # Near plane (column 2 + column 3)
        normal = np.array([vp[0, 3] + vp[0, 2], vp[1, 3] + vp[1, 2], vp[2, 3] + vp[2, 2]], dtype=np.float32)
        length = np.linalg.norm(normal)
        if length > 0:
            self._planes.append((normal / length, (vp[3, 3] + vp[3, 2]) / length))

        # Far plane (column 3 - column 2)
        normal = np.array([vp[0, 3] - vp[0, 2], vp[1, 3] - vp[1, 2], vp[2, 3] - vp[2, 2]], dtype=np.float32)
        length = np.linalg.norm(normal)
        if length > 0:
            self._planes.append((normal / length, (vp[3, 3] - vp[3, 2]) / length))

    def is_box_visible(self, min_bounds: np.ndarray, max_bounds: np.ndarray) -> bool:
        """Check if a bounding box is visible.

        Args:
            min_bounds: Minimum bounds [x, y, z]
            max_bounds: Maximum bounds [x, y, z]

        Returns:
            True if the box is visible
        """


        # Get the 8 corners of the box
        corners = np.array([
            [min_bounds[0], min_bounds[1], min_bounds[2]],
            [max_bounds[0], min_bounds[1], min_bounds[2]],
            [min_bounds[0], max_bounds[1], min_bounds[2]],
            [max_bounds[0], max_bounds[1], min_bounds[2]],
            [min_bounds[0], min_bounds[1], max_bounds[2]],
            [max_bounds[0], min_bounds[1], max_bounds[2]],
            [min_bounds[0], max_bounds[1], max_bounds[2]],
            [max_bounds[0], max_bounds[1], max_bounds[2]],
        ], dtype=np.float32)

        # Check if all corners are outside any plane
        for normal, distance in self._planes:
            all_outside = True
            for corner in corners:
                if np.dot(normal, corner) + distance >= 0:
                    all_outside = False
                    break
            if all_outside:
                return False  # Box is completely outside this plane

        return True  # Box is potentially visible

    def is_sphere_visible(self, center: np.ndarray, radius: float) -> bool:
        """Check if a sphere is visible.

        Args:
            center: Sphere center [x, y, z]
            radius: Sphere radius

        Returns:
            True if the sphere is visible
        """


        for normal, distance in self._planes:
            # Distance from sphere center to plane
            dist = np.dot(normal, center) + distance
            if dist < -radius:
                return False  # Sphere is completely outside this plane

        return True  # Sphere is potentially visible


class LevelOfDetail:
    """Level of Detail (LOD) system for optimizing distant rendering."""

    def __init__(self, distances: list[float] | None = None) -> None:
        """Initialize LOD system.

        Args:
            distances: Distance thresholds for each LOD level
        """
        self._distances = distances or [10.0, 30.0, 60.0, 100.0]

    def get_lod_level(self, distance: float) -> int:
        """Get LOD level for a given distance.

        Args:
            distance: Distance from camera

        Returns:
            LOD level (0 = highest quality, higher = lower quality)
        """
        for i, threshold in enumerate(self._distances):
            if distance < threshold:
                return i
        return len(self._distances)

    def should_render(self, distance: float, max_distance: float = 200.0) -> bool:
        """Check if an object should be rendered at given distance.

        Args:
            distance: Distance from camera
            max_distance: Maximum render distance

        Returns:
            True if object should be rendered
        """
        return distance < max_distance

    def get_lod_scale(self, lod_level: int) -> float:
        """Get scale factor for a LOD level.

        Args:
            lod_level: LOD level

        Returns:
            Scale multiplier
        """
        scales = [1.0, 0.8, 0.6, 0.4, 0.2]
        return scales[lod_level] if lod_level < len(scales) else 0.1


class ProgressiveLoader(QObject):
    """Progressive loader for large file systems.

    Loads and renders nodes in batches to maintain responsiveness.
    """

    batch_loaded = pyqtSignal(int, int)  # Emits (loaded_count, total_count)
    loading_complete = pyqtSignal()

    def __init__(self, batch_size: int = 1000, batch_delay_ms: int = 16) -> None:
        """Initialize progressive loader.

        Args:
            batch_size: Number of nodes to load per batch
            batch_delay_ms: Delay between batches in milliseconds
        """
        super().__init__()

        self._batch_size = batch_size
        self._batch_delay_ms = batch_delay_ms

        self._pending_nodes: list[tuple[str, object]] = []
        self._loaded_count = 0
        self._is_loading = False

        self._timer = QTimer()
        self._timer.timeout.connect(self._load_next_batch)

    def start_loading(self, nodes: list[tuple[str, object]]) -> None:
        """Start progressive loading.

        Args:
            nodes: List of (path_str, node) tuples to load
        """
        self._pending_nodes = list(nodes)
        self._loaded_count = 0
        self._is_loading = True

        if self._pending_nodes:
            self._load_next_batch()

    def _load_next_batch(self) -> None:
        """Load the next batch of nodes."""
        if not self._pending_nodes:
            self._is_loading = False
            self.loading_complete.emit()
            self._timer.stop()
            return

        # Process batch
        batch = self._pending_nodes[:self._batch_size]
        self._pending_nodes = self._pending_nodes[self._batch_size:]

        # This would trigger loading into the renderer
        # For now, just emit progress
        self._loaded_count += len(batch)

        # Continue loading if more pending
        if self._pending_nodes:
            self.batch_loaded.emit(self._loaded_count, self._loaded_count + len(self._pending_nodes))
            if not self._timer.isActive():
                self._timer.start(self._batch_delay_ms)
        else:
            self.loading_complete.emit()
            self._timer.stop()

    def cancel(self) -> None:
        """Cancel loading."""
        self._timer.stop()
        self._pending_nodes.clear()
        self._is_loading = False

    @property
    def is_loading(self) -> bool:
        """Check if currently loading."""
        return self._is_loading

    @property
    def progress(self) -> tuple[int, int]:
        """Get loading progress.

        Returns:
            Tuple of (loaded_count, total_count)
        """
        total = self._loaded_count + len(self._pending_nodes)
        return (self._loaded_count, total)

