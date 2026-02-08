"""Camera system for 3D navigation.

Supports Orbit camera mode:
- Orbit: Rotate around a focal point
"""

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class CameraMode(Enum):
    """Camera navigation modes."""

    ORBIT = "orbit"


@dataclass
class CameraState:
    """Camera position and orientation state."""

    position: np.ndarray  # [x, y, z]
    target: np.ndarray  # Focal point [x, y, z]
    up: np.ndarray  # Up vector [x, y, z]
    fov: float = 45.0  # Field of view in degrees
    near: float = 0.1
    far: float = 1000.0


class Camera:
    """3D Camera with multiple navigation modes."""

    def __init__(self) -> None:
        """Initialize camera with default state."""
        self.mode = CameraMode.ORBIT
        # FSN-style: high angle, looking down into the receding distance (-Z)
        self._state = CameraState(
            position=np.array([0.0, 10.0, 20.0], dtype=np.float32),  # High, behind root
            target=np.array([0.0, 0.0, -50.0], dtype=np.float32),    # Looking far forward/down
            up=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        self._orbit_distance = np.linalg.norm(self._state.position - self._state.target)
        self._orbit_yaw = math.atan2(
            self._state.position[0] - self._state.target[0],
            self._state.position[2] - self._state.target[2],
        )
        self._orbit_pitch = math.asin(
            (self._state.position[1] - self._state.target[1]) / self._orbit_distance
        )
        self._rotation_speed = 0.003
        self._zoom_speed = 0.1

    @property
    def state(self) -> CameraState:
        """Get current camera state."""
        return self._state

    @property
    def view_matrix(self) -> np.ndarray:
        """Get view matrix as 4x4 numpy array (column-major for ModernGL)."""
        return self._compute_view_matrix()

    def projection_matrix(self, aspect_ratio: float) -> np.ndarray:
        """Get projection matrix as 4x4 numpy array (column-major for ModernGL)."""
        return self._compute_projection_matrix(aspect_ratio)

    def _compute_view_matrix(self) -> np.ndarray:
        """Compute view matrix from camera state."""
        eye = self._state.position
        target = self._state.target
        up = self._state.up

        # Forward vector
        f = target - eye
        f = f / np.linalg.norm(f)

        # Right vector
        s = np.cross(f, up)
        s = s / np.linalg.norm(s)

        # Up vector (recalculated)
        u = np.cross(s, f)

        # View matrix - standard OpenGL layout
        # When using numpy with tobytes(), this creates the correct column-major layout
        view = np.identity(4, dtype=np.float32)
        view[0, 0] = s[0]
        view[0, 1] = s[1]
        view[0, 2] = s[2]
        view[1, 0] = u[0]
        view[1, 1] = u[1]
        view[1, 2] = u[2]
        view[2, 0] = -f[0]
        view[2, 1] = -f[1]
        view[2, 2] = -f[2]
        view[0, 3] = -np.dot(s, eye)
        view[1, 3] = -np.dot(u, eye)
        view[2, 3] = np.dot(f, eye)

        return view

    def _compute_projection_matrix(self, aspect_ratio: float) -> np.ndarray:
        """Compute perspective projection matrix."""
        fov_rad = math.radians(self._state.fov)
        f = 1.0 / math.tan(fov_rad / 2.0)

        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / aspect_ratio
        proj[1, 1] = f
        proj[2, 2] = (self._state.far + self._state.near) / (self._state.near - self._state.far)
        proj[3, 2] = -1.0
        proj[2, 3] = (2.0 * self._state.far * self._state.near) / (self._state.near - self._state.far)

        return proj

    def set_mode(self, mode: CameraMode) -> None:
        """Set camera navigation mode."""
        self.mode = mode
        if mode == CameraMode.ORBIT:
            self._update_orbit_from_position()

    def orbit_rotate(self, dx: int, dy: int) -> None:
        """Rotate camera around target in orbit mode."""
        if self.mode != CameraMode.ORBIT:
            return

        self._orbit_yaw -= dx * self._rotation_speed
        self._orbit_pitch -= dy * self._rotation_speed

        # Clamp pitch to avoid gimbal lock
        self._orbit_pitch = max(-math.pi / 2 + 0.01, min(math.pi / 2 - 0.01, self._orbit_pitch))

        self._update_position_from_orbit()

    def orbit_zoom(self, delta: float) -> None:
        """Zoom camera in orbit mode."""
        if self.mode != CameraMode.ORBIT:
            return

        self._orbit_distance *= 1.0 - delta * self._zoom_speed
        self._orbit_distance = max(1.0, min(500.0, self._orbit_distance))
        self._update_position_from_orbit()

    def orbit_pan(self, dx: int, dy: int, view_width: int, view_height: int) -> None:
        """Pan camera in orbit mode."""
        if self.mode != CameraMode.ORBIT:
            return

        # Calculate pan amount based on distance
        pan_speed = self._orbit_distance * 0.001

        # Get camera right and up vectors
        forward = self._state.target - self._state.position
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, self._state.up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        # Pan target
        pan_x = -dx * pan_speed * right
        pan_y = dy * pan_speed * up

        self._state.target += pan_x + pan_y
        self._update_position_from_orbit()

    def _update_position_from_orbit(self) -> None:
        """Update camera position from orbit parameters."""
        x = self._orbit_distance * math.cos(self._orbit_pitch) * math.sin(self._orbit_yaw)
        y = self._orbit_distance * math.sin(self._orbit_pitch)
        z = self._orbit_distance * math.cos(self._orbit_pitch) * math.cos(self._orbit_yaw)

        self._state.position = self._state.target + np.array([x, y, z], dtype=np.float32)

    def _update_orbit_from_position(self) -> None:
        """Update orbit parameters from current position."""
        diff = self._state.position - self._state.target
        self._orbit_distance = np.linalg.norm(diff)

        if self._orbit_distance > 0:
            self._orbit_yaw = math.atan2(diff[0], diff[2])
            self._orbit_pitch = math.asin(diff[1] / self._orbit_distance)

    def set_position_target(self, position: np.ndarray, target: np.ndarray) -> None:
        """Set camera position and target simultaneously.
        
        Args:
            position: New camera position
            target: New camera target
        """
        self._state.position = position.astype(np.float32)
        self._state.target = target.astype(np.float32)
        self._update_orbit_from_position()

    def get_ray_direction(self, screen_x: float, screen_y: float, width: int, height: int) -> np.ndarray:
        """Get ray direction from screen coordinates for raycasting.

        Args:
            screen_x: X coordinate in screen space (0 to width)
            screen_y: Y coordinate in screen space (0 to height)
            width: Viewport width
            height: Viewport height

        Returns:
            Normalized ray direction vector
        """
        # Normalize to NDC (-1 to 1)
        ndc_x = (2.0 * screen_x / width) - 1.0
        ndc_y = 1.0 - (2.0 * screen_y / height)

        # Inverse projection
        fov_rad = math.radians(self._state.fov)
        aspect_ratio = width / height
        tan_half_fov = math.tan(fov_rad / 2.0)

        ray_clip = np.array([
            ndc_x * aspect_ratio * tan_half_fov,
            ndc_y * tan_half_fov,
            -1.0,
            0.0,
        ], dtype=np.float32)

        # Inverse view (just need rotation part for direction)
        forward = self._state.target - self._state.position
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, self._state.up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        ray_world = (
            right * ray_clip[0] +
            up * ray_clip[1] +
            -forward * ray_clip[2]
        )

        return ray_world / np.linalg.norm(ray_world)
