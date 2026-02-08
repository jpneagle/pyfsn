"""Bounding box for collision detection."""

from dataclasses import dataclass

from pyfsn.layout.position import Position


@dataclass
class BoundingBox:
    """Axis-aligned bounding box for collision detection.

    Attributes:
        position: The position this bounding box represents
        padding: Extra space to add around the position for collision
    """

    position: Position
    padding: float = 0.0

    @property
    def min_x(self) -> float:
        """Minimum X coordinate including padding."""
        return self.position.min_x - self.padding

    @property
    def max_x(self) -> float:
        """Maximum X coordinate including padding."""
        return self.position.max_x + self.padding

    @property
    def min_y(self) -> float:
        """Minimum Y coordinate including padding."""
        return self.position.min_y - self.padding

    @property
    def max_y(self) -> float:
        """Maximum Y coordinate including padding."""
        return self.position.max_y + self.padding

    @property
    def min_z(self) -> float:
        """Minimum Z coordinate including padding."""
        return self.position.min_z - self.padding

    @property
    def max_z(self) -> float:
        """Maximum Z coordinate including padding."""
        return self.position.max_z + self.padding

    def intersects(self, other: "BoundingBox") -> bool:
        """Check if this bounding box intersects with another.

        Args:
            other: Other bounding box to check intersection with

        Returns:
            True if the boxes intersect, False otherwise
        """
        return (
            self.min_x < other.max_x
            and self.max_x > other.min_x
            and self.min_y < other.max_y
            and self.max_y > other.min_y
            and self.min_z < other.max_z
            and self.max_z > other.min_z
        )

    def contains_point(self, x: float, y: float, z: float) -> bool:
        """Check if a point is inside this bounding box.

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate

        Returns:
            True if the point is inside, False otherwise
        """
        return (
            self.min_x <= x <= self.max_x
            and self.min_y <= y <= self.max_y
            and self.min_z <= z <= self.max_z
        )

    @classmethod
    def from_position(cls, position: Position, padding: float = 0.1) -> "BoundingBox":
        """Create a bounding box from a position.

        Args:
            position: Position to create bounding box for
            padding: Padding to add around the position

        Returns:
            A new BoundingBox instance
        """
        return cls(position=position, padding=padding)
