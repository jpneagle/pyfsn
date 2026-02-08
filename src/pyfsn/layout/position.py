"""Position and coordinate classes for 3D layout."""

from dataclasses import dataclass


@dataclass
class Position:
    """3D position with dimensions.

    Attributes:
        x: X coordinate
        y: Y coordinate
        z: Z coordinate
        width: Width of the node (size along X axis)
        height: Height of the node (size along Y axis)
        depth: Depth of the node (size along Z axis)
    """

    x: float
    y: float
    z: float
    width: float
    height: float
    depth: float

    @property
    def center(self) -> tuple[float, float, float]:
        """Get the center point of the position."""
        return (
            self.x + self.width / 2,
            self.y + self.height / 2,
            self.z + self.depth / 2,
        )

    @property
    def min_x(self) -> float:
        """Minimum X coordinate."""
        return self.x

    @property
    def max_x(self) -> float:
        """Maximum X coordinate."""
        return self.x + self.width

    @property
    def min_y(self) -> float:
        """Minimum Y coordinate."""
        return self.y

    @property
    def max_y(self) -> float:
        """Maximum Y coordinate."""
        return self.y + self.height

    @property
    def min_z(self) -> float:
        """Minimum Z coordinate."""
        return self.z

    @property
    def max_z(self) -> float:
        """Maximum Z coordinate."""
        return self.z + self.depth

    def translate(self, dx: float, dy: float, dz: float) -> "Position":
        """Create a new position translated by the given amounts."""
        return Position(
            x=self.x + dx,
            y=self.y + dy,
            z=self.z + dz,
            width=self.width,
            height=self.height,
            depth=self.depth,
        )

    def contains_point(self, px: float, py: float, pz: float) -> bool:
        """Check if a point is inside this position."""
        return (
            self.min_x <= px <= self.max_x
            and self.min_y <= py <= self.max_y
            and self.min_z <= pz <= self.max_z
        )

    def distance_to(self, other: "Position") -> float:
        """Calculate the distance between centers of two positions."""
        c1 = self.center
        c2 = other.center
        return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2) ** 0.5

    def __repr__(self) -> str:
        """String representation."""
        return f"Position(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f}, " f"w={self.width:.2f}, h={self.height:.2f}, d={self.depth:.2f})"
