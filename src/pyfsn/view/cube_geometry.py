"""Cube geometry and instancing setup for GPU rendering.

Provides cube mesh data and manages instanced rendering attributes
(position, scale, color) for efficient visualization of many nodes.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CubeInstance:
    """Single cube instance attributes."""

    position: np.ndarray  # [x, y, z]
    scale: np.ndarray  # [width, height, depth]
    color: np.ndarray  # [r, g, b, a]


class CubeGeometry:
    """Cube mesh geometry with instancing support."""

    # Cube vertices (8 corners)
    VERTICES = np.array([
        # Front face
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
        # Back face
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
    ], dtype=np.float32)

    # Cube indices (triangles, 12 faces * 3 vertices = 36 indices)
    INDICES = np.array([
        # Front face
        0, 1, 2, 2, 3, 0,
        # Right face
        1, 5, 6, 6, 2, 1,
        # Back face
        5, 4, 7, 7, 6, 5,
        # Left face
        4, 0, 3, 3, 7, 4,
        # Bottom face
        4, 5, 1, 1, 0, 4,
        # Top face
        3, 2, 6, 6, 7, 3,
    ], dtype=np.uint32)

    # Normals for each face (for flat shading)
    NORMALS = np.array([
        # Front (4 vertices)
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        # Back (4 vertices)
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float32)

    def __init__(self, ctx) -> None:
        """Initialize cube geometry with ModernGL context.

        Args:
            ctx: ModernGL context
        """
        import moderngl

        self.ctx = ctx

        # Create base cube VBO
        self.vertex_vbo = ctx.buffer(self.VERTICES.tobytes())
        self.normal_vbo = ctx.buffer(self.NORMALS.tobytes())
        self.index_buffer = ctx.buffer(self.INDICES.tobytes())

        # Instance data buffers (dynamic)
        self.max_instances = 100000
        self._position_buffer = ctx.buffer(
            np.zeros((self.max_instances, 3), dtype=np.float32).tobytes()
        )
        self._scale_buffer = ctx.buffer(
            np.zeros((self.max_instances, 3), dtype=np.float32).tobytes()
        )
        self._color_buffer = ctx.buffer(
            np.zeros((self.max_instances, 4), dtype=np.float32).tobytes()
        )

        # Current instance count
        self._instance_count = 0

        # Instance data arrays (CPU side for updates)
        self._positions = np.zeros((self.max_instances, 3), dtype=np.float32)
        self._scales = np.zeros((self.max_instances, 3), dtype=np.float32)
        self._colors = np.zeros((self.max_instances, 4), dtype=np.float32)

    @property
    def instance_count(self) -> int:
        """Get current number of instances."""
        return self._instance_count

    def clear_instances(self) -> None:
        """Clear all instances."""
        self._instance_count = 0

    def add_instance(self, instance: CubeInstance) -> int:
        """Add a single cube instance.

        Args:
            instance: CubeInstance with position, scale, color

        Returns:
            Instance index
        """
        if self._instance_count >= self.max_instances:
            raise RuntimeError(f"Maximum instances ({self.max_instances}) reached")

        idx = self._instance_count
        self._positions[idx] = instance.position
        self._scales[idx] = instance.scale
        self._colors[idx] = instance.color
        self._instance_count += 1
        return idx

    def add_instances(self, instances: list[CubeInstance]) -> None:
        """Add multiple cube instances.

        Args:
            instances: List of CubeInstance objects
        """
        count = len(instances)
        if self._instance_count + count > self.max_instances:
            raise RuntimeError(f"Would exceed maximum instances ({self.max_instances})")

        for instance in instances:
            self.add_instance(instance)

    def update_instance(self, index: int, instance: CubeInstance) -> None:
        """Update an existing instance.

        Args:
            index: Instance index
            instance: New CubeInstance data
        """
        if 0 <= index < self._instance_count:
            self._positions[index] = instance.position
            self._scales[index] = instance.scale
            self._colors[index] = instance.color

    def remove_instance(self, index: int) -> None:
        """Remove an instance by swapping with last.

        Args:
            index: Instance index to remove
        """
        if 0 <= index < self._instance_count:
            # Swap with last
            last_idx = self._instance_count - 1
            if index != last_idx:
                self._positions[index] = self._positions[last_idx]
                self._scales[index] = self._scales[last_idx]
                self._colors[index] = self._colors[last_idx]
            self._instance_count -= 1

    def upload_instances(self) -> None:
        """Upload instance data to GPU.

        Call this after modifying instances to sync with GPU.
        """
        if self._instance_count > 0:
            self._position_buffer.write(
                self._positions[:self._instance_count].tobytes()
            )
            self._scale_buffer.write(
                self._scales[:self._instance_count].tobytes()
            )
            self._color_buffer.write(
                self._colors[:self._instance_count].tobytes()
            )

    def create_vertex_array(self, program) -> object:
        """Create vertex array object for rendering.

        Args:
            program: ModernGL program with vertex shader

        Returns:
            ModernGL vertex array object
        """
        return self.ctx.vertex_array(
            program,
            [
                (self.vertex_vbo, "3f", "in_position"),
                (self.normal_vbo, "3f", "in_normal"),
                (self._position_buffer, "3f/i", "in_instance_position"),
                (self._scale_buffer, "3f/i", "in_instance_scale"),
                (self._color_buffer, "4f/i", "in_instance_color"),
            ],
            self.index_buffer,
        )

    def render(self, vao: object) -> None:
        """Render all instances.

        Args:
            vao: Vertex array object from create_vertex_array
        """
        if self._instance_count > 0:
            vao.render(instances=self._instance_count)


def get_default_colors() -> dict[str, np.ndarray]:
    """Get default colors for different file types.

    Returns:
        Dictionary mapping file type to RGBA color
    """
    return {
        "directory": np.array([0.2, 0.6, 1.0, 1.0], dtype=np.float32),  # Blue
        "file": np.array([0.8, 0.8, 0.8, 1.0], dtype=np.float32),  # Gray
        "symlink": np.array([0.3, 1.0, 0.3, 1.0], dtype=np.float32),  # Green
        "executable": np.array([1.0, 0.5, 0.2, 1.0], dtype=np.float32),  # Orange
        "hidden": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),  # Transparent gray
        "selected": np.array([1.0, 1.0, 0.3, 1.0], dtype=np.float32),  # Yellow
        "focused": np.array([1.0, 0.8, 0.0, 1.0], dtype=np.float32),  # Gold
        "connection": np.array([0.3, 0.3, 0.3, 0.5], dtype=np.float32),  # Dark gray
    }
