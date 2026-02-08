"""Picking and raycasting system for 3D object selection.

Provides CPU-side ray-AABB intersection tests and screen-to-world
coordinate conversion for object picking in the 3D visualization.
Works with ModernGL by using matrix transformations instead of
gluUnProject.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


@dataclass
class Ray:
    """3D ray for raycasting operations.

    Attributes:
        origin: Ray origin point [x, y, z]
        direction: Normalized direction vector [dx, dy, dz]
    """

    origin: np.ndarray  # [x, y, z]
    direction: np.ndarray  # [dx, dy, dz] - normalized

    def __post_init__(self) -> None:
        """Ensure direction is normalized."""
        norm = np.linalg.norm(self.direction)
        if norm > 1e-6:
            self.direction = self.direction / norm

    def at(self, t: float) -> np.ndarray:
        """Get point along ray at distance t.

        Args:
            t: Distance along ray

        Returns:
            Point at distance t from origin
        """
        return self.origin + self.direction * t


class AABB(NamedTuple):
    """Axis-Aligned Bounding Box.

    Attributes:
        min_bounds: Minimum corner [x, y, z]
        max_bounds: Maximum corner [x, y, z]
    """

    min_bounds: np.ndarray  # [x, y, z]
    max_bounds: np.ndarray  # [x, y, z]

    def center(self) -> np.ndarray:
        """Get center point of AABB."""
        return (self.min_bounds + self.max_bounds) / 2.0

    def size(self) -> np.ndarray:
        """Get size of AABB."""
        return self.max_bounds - self.min_bounds


def ray_aabb_intersect(ray: Ray, aabb: AABB) -> float | None:
    """Compute ray-AABB intersection using slab method.

    This is the standard algorithm for ray-box intersection.
    It finds the entry and exit points of the ray through each
    axis-aligned slab (pair of parallel planes) and computes
    the overlap.

    Args:
        ray: Ray to test
        aabb: Axis-aligned bounding box

    Returns:
        Distance to intersection point (t) or None if no intersection.
        Returns the smallest positive t (closest intersection in front
        of ray origin).
    """
    epsilon = 1e-6

    t_min = 0.0  # Near plane (don't count intersections behind origin)
    t_max = float('inf')

    for i in range(3):
        if abs(ray.direction[i]) < epsilon:
            # Ray is parallel to this slab
            # Check if ray origin is within the slab bounds
            if ray.origin[i] < aabb.min_bounds[i] or ray.origin[i] > aabb.max_bounds[i]:
                return None
        else:
            # Compute intersection distances for this slab
            inv_d = 1.0 / ray.direction[i]
            t1 = (aabb.min_bounds[i] - ray.origin[i]) * inv_d
            t2 = (aabb.max_bounds[i] - ray.origin[i]) * inv_d

            # Ensure t1 is the closer intersection
            if t1 > t2:
                t1, t2 = t2, t1

            # Update the overall intersection interval
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)

            # Check if interval is empty (no intersection)
            if t_min > t_max:
                return None

    # Return t_min if it's positive (intersection in front of ray origin)
    return t_min if t_min > 0 else None


def screen_to_ray(
    screen_x: float,
    screen_y: float,
    width: int,
    height: int,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
) -> Ray:
    """Convert screen coordinates to world-space ray.

    This replaces gluUnProject with pure matrix math. The process is:
    1. Convert screen coords to Normalized Device Coordinates (NDC)
    2. Inverse transform by projection matrix to get view-space ray
    3. Inverse transform by view matrix to get world-space ray

    Args:
        screen_x: Screen X coordinate (0 to width)
        screen_y: Screen Y coordinate (0 to height, top=0)
        width: Viewport width in pixels
        height: Viewport height in pixels
        view_matrix: 4x4 view matrix (column-major)
        proj_matrix: 4x4 projection matrix (column-major)

    Returns:
        Ray with origin at camera position and direction pointing
        into the scene at the screen coordinates
    """
    # Convert screen coordinates to NDC (-1 to 1)
    # Screen Y is inverted (0 at top, height at bottom)
    ndc_x = (2.0 * screen_x / width) - 1.0
    ndc_y = 1.0 - (2.0 * screen_y / height)

    # Create clip-space point on near plane
    # Z = -1 in NDC is the near plane in OpenGL's right-handed coordinate system
    near_ndc = np.array([ndc_x, ndc_y, -1.0, 1.0], dtype=np.float32)

    # Inverse projection to get view-space point
    inv_proj = np.linalg.inv(proj_matrix)
    view_pos = inv_proj @ near_ndc

    # Perspective divide: After inverse projection, we need to divide by w
    # to account for the perspective transformation
    if abs(view_pos[3]) > 1e-6:
        view_pos = view_pos / view_pos[3]

    # Create far point for ray direction (same NDC x,y, but z=1 for far plane)
    far_ndc = np.array([ndc_x, ndc_y, 1.0, 1.0], dtype=np.float32)
    view_far = inv_proj @ far_ndc
    if abs(view_far[3]) > 1e-6:
        view_far = view_far / view_far[3]

    # Inverse view matrix to get world-space
    inv_view = np.linalg.inv(view_matrix)

    # Camera position (world space) is the inverse view's translation component
    # Actually, camera position = - (rotation^T * translation)
    # But simpler: transform the origin (0,0,0,1) through inverse view
    origin_clip = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    origin_world = inv_view @ origin_clip
    ray_origin = origin_world[:3]

    # Transform near and far points to world space
    near_world = inv_view @ view_pos
    far_world = inv_view @ view_far

    # Ray direction is from near to far (normalized)
    ray_direction = far_world[:3] - near_world[:3]
    ray_direction = ray_direction / np.linalg.norm(ray_direction)

    return Ray(origin=ray_origin, direction=ray_direction)


def get_node_aabb(node_type: str, position: object) -> AABB:
    """Get AABB bounds for a node based on its type and layout position.

    This matches the actual rendered positions in the legacy renderer:
    - Directories are rendered as thin platforms at y=0
    - Files are rendered as cubes sitting on top of the platform

    Args:
        node_type: NodeType enum value ('directory' or 'file')
        position: Position object with x, y, z, width, height, depth

    Returns:
        AABB bounds for the node
    """
    if node_type == 'directory':
        # Platform: thin at y=0, height=0.2
        platform_h = 0.2
        min_bounds = np.array([
            position.x,
            -platform_h / 2,
            position.z,
        ], dtype=np.float32)
        max_bounds = np.array([
            position.x + position.width,
            platform_h / 2,
            position.z + position.depth,
        ], dtype=np.float32)
    else:
        # File: cube on top of platform at y=0
        # File cubes are scaled to 80% of their layout dimensions
        width_scaled = position.width * 0.8
        depth_scaled = position.depth * 0.8
        height = position.height

        # Center the scaled cube within the layout bounds
        min_bounds = np.array([
            position.x + (position.width - width_scaled) / 2,
            0.0,  # Base at y=0 (on top of platform)
            position.z + (position.depth - depth_scaled) / 2,
        ], dtype=np.float32)
        max_bounds = np.array([
            min_bounds[0] + width_scaled,
            height,  # Top at y=height
            min_bounds[2] + depth_scaled,
        ], dtype=np.float32)

    return AABB(min_bounds=min_bounds, max_bounds=max_bounds)


def find_closest_intersection(
    ray: Ray,
    nodes: dict[str, object],
    positions: dict[str, object],
) -> object | None:
    """Find the closest node intersected by a ray.

    Args:
        ray: World-space ray to test
        nodes: Dictionary mapping path strings to Node objects
        positions: Dictionary mapping path strings to Position objects

    Returns:
        The closest Node that intersects the ray, or None
    """
    closest_node = None
    closest_t = float('inf')

    for path_str, position in positions.items():
        if path_str not in nodes:
            continue

        node = nodes[path_str]

        # Get AABB for this node
        if hasattr(node, 'type'):
            # Node has type attribute
            from pyfsn.model.node import NodeType
            node_type_str = 'directory' if node.type == NodeType.DIRECTORY else 'file'
        else:
            # Fallback: check by path
            node_type_str = 'directory' if hasattr(node, 'is_directory') and node.is_directory else 'file'

        aabb = get_node_aabb(node_type_str, position)

        # Test intersection
        t = ray_aabb_intersect(ray, aabb)
        if t is not None and t < closest_t:
            closest_t = t
            closest_node = node

    return closest_node


def raycast_find_node_with_camera(
    screen_x: int,
    screen_y: int,
    nodes: dict[str, object],
    positions: dict[str, object],
    camera: object,
    width: int,
    height: int,
) -> object | None:
    """Find node at screen position using camera matrices.

    This is a convenience function that matches the legacy renderer's
    raycast_find_node signature but uses matrix-based ray generation
    instead of gluUnProject.

    Args:
        screen_x: Screen X coordinate
        screen_y: Screen Y coordinate
        nodes: Dictionary mapping path strings to Node objects
        positions: Dictionary mapping path strings to Position objects
        camera: Camera object with view_matrix and projection_matrix()
        width: Viewport width
        height: Viewport height

    Returns:
        Node at position or None
    """
    # Get camera matrices
    view_matrix = camera.view_matrix
    aspect = width / max(1, height)
    proj_matrix = camera.projection_matrix(aspect)

    # Convert screen to ray
    ray = screen_to_ray(screen_x, screen_y, width, height, view_matrix, proj_matrix)

    # Find closest intersection
    return find_closest_intersection(ray, nodes, positions)


class PickingSystem:
    """Centralized picking system for 3D object selection.

    This class coordinates the picking process, converting screen
    coordinates to world rays and finding intersected objects.
    """

    def __init__(self) -> None:
        """Initialize picking system."""
        self._nodes: dict[str, object] = {}
        self._positions: dict[str, object] = {}

    def set_scene_data(self, nodes: dict[str, object], positions: dict[str, object]) -> None:
        """Set scene data for picking.

        Args:
            nodes: Dictionary mapping path strings to Node objects
            positions: Dictionary mapping path strings to Position objects
        """
        self._nodes = nodes
        self._positions = positions

    def raycast_find_node(
        self,
        screen_x: int,
        screen_y: int,
        nodes: dict[str, object],
        positions: dict[str, object],
    ) -> object | None:
        """Find node at screen position using raycasting.

        This method matches the legacy renderer's signature for compatibility.
        Use this method when you want to pass nodes/positions directly
        instead of setting them via set_scene_data.

        Args:
            screen_x: Screen X coordinate
            screen_y: Screen Y coordinate
            nodes: Dictionary mapping path strings to Node objects
            positions: Dictionary mapping path strings to Position objects

        Returns:
            Node at position or None

        Note:
            This method is a compatibility wrapper. It requires view_matrix
            and proj_matrix to be set externally or requires the caller
            to use the pick() method instead. For modern renderer usage,
            prefer the pick() method with explicit matrix parameters.
        """
        # This method is provided for API compatibility with the legacy renderer.
        # For actual picking, you need to call pick() with matrix parameters.
        # This is a no-op here - the actual implementation in pick() requires matrices.
        raise NotImplementedError(
            "Use pick() method with view_matrix and proj_matrix parameters. "
            "This method exists only for API compatibility."
        )

    def pick(
        self,
        screen_x: float,
        screen_y: float,
        width: int,
        height: int,
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray,
    ) -> object | None:
        """Pick object at screen coordinates.

        Args:
            screen_x: Screen X coordinate
            screen_y: Screen Y coordinate
            width: Viewport width
            height: Viewport height
            view_matrix: View matrix from camera
            proj_matrix: Projection matrix from camera

        Returns:
            Picked Node object or None
        """
        # Convert screen to ray
        ray = screen_to_ray(screen_x, screen_y, width, height, view_matrix, proj_matrix)

        # Find closest intersection
        return find_closest_intersection(ray, self._nodes, self._positions)

    def ray_at_screen(
        self,
        screen_x: float,
        screen_y: float,
        width: int,
        height: int,
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray,
    ) -> Ray:
        """Get world-space ray for screen coordinates (for debugging).

        Args:
            screen_x: Screen X coordinate
            screen_y: Screen Y coordinate
            width: Viewport width
            height: Viewport height
            view_matrix: View matrix from camera
            proj_matrix: Projection matrix from camera

        Returns:
            World-space ray
        """
        return screen_to_ray(screen_x, screen_y, width, height, view_matrix, proj_matrix)
