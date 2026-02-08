"""Layout engine for 3D file system visualization."""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from pyfsn.layout.box import BoundingBox
from pyfsn.layout.position import Position
from pyfsn.model.node import Node, NodeType


class PlacementStrategy(Enum):
    """Strategy for placing subdirectories."""

    GRID = "grid"  # Grid-based placement
    RADIAL = "radial"  # Radial/circular placement
    SPIRAL = "spiral"  # Spiral placement
    HEURISTIC = "heuristic"  # Heuristic-based placement


@dataclass
class LayoutConfig:
    """Configuration for the layout engine.

    Attributes:
        node_size: Base size for file nodes
        dir_size: Base size for directory nodes
        spacing: Spacing between nodes
        padding: Padding around nodes
        max_depth: Maximum depth to visualize
        placement_strategy: Strategy for placing subdirectories
        grid_size: Size of grid cells for grid placement
        connection_width: Width of connection wires
        use_size_height: Whether to use file size for height calculation
        min_height: Minimum height for nodes
        max_height: Maximum height for nodes
        height_scale: Scale factor for height calculation (log scale)
    """

    node_size: float = 1.0
    dir_size: float = 2.0
    spacing: float = 0.5
    padding: float = 0.2
    max_depth: int = 5
    placement_strategy: PlacementStrategy = PlacementStrategy.GRID
    grid_size: float = 3.0
    connection_width: float = 0.1
    # SGI fsn style height settings
    use_size_height: bool = True
    min_height: float = 0.2
    max_height: float = 5.0
    height_scale: float = 0.3


@dataclass
class LayoutResult:
    """Result of a layout operation.

    Attributes:
        positions: Dictionary mapping node paths to positions
        connections: List of (parent_path, child_path) tuples for connections
        bounds: Overall bounding box of the layout
    """

    positions: dict[str, Position] = field(default_factory=dict)
    connections: list[tuple[str, str]] = field(default_factory=list)
    bounds: BoundingBox | None = None


class LayoutEngine:
    """Engine for calculating 3D positions for file system nodes."""

    def __init__(self, config: LayoutConfig | None = None) -> None:
        """Initialize the layout engine.

        Args:
            config: Layout configuration (uses defaults if None)
        """
        self.config = config or LayoutConfig()
        self._occupied_boxes: list[BoundingBox] = []

    def _calculate_node_height(self, node: Node) -> float:
        """Calculate node height based on file size (SGI fsn style).

        Uses logarithmic scale for visual clarity.

        Args:
            node: Node to calculate height for

        Returns:
            Height value for the node
        """
        if not self.config.use_size_height:
            return self.config.min_height

        if node.is_directory:
            # Directory height based on total size of contents
            size = node.total_size
        else:
            size = node.size

        if size <= 0:
            return self.config.min_height

        # Logarithmic scale: log10(1 byte) = 0, log10(1KB) ≈ 3, log10(1MB) ≈ 6, log10(1GB) ≈ 9
        log_size = math.log10(max(size, 1))
        height = self.config.min_height + (log_size * self.config.height_scale)
        return min(max(height, self.config.min_height), self.config.max_height)

    def _calculate_directory_spacing(self, node: Node) -> float:
        """Calculate spacing for a directory based on its child count.

        Directories with more children need more space to prevent crowding.
        Uses sqrt scale to avoid excessive spacing for very large directories.

        Args:
            node: Directory node to calculate spacing for

        Returns:
            Spacing value for this directory's children
        """
        if not node.is_directory:
            return self.config.spacing

        # Count total descendants (not just direct children)
        child_count = len(node.children)
        
        # Base spacing plus additional based on child count
        # sqrt scale: 1 child = base, 4 children = 2x, 16 children = 4x
        # Doubled base_spacing for wider separation
        base_spacing = self.config.grid_size * 2.0
        child_factor = 1.0 + math.sqrt(max(child_count - 1, 0)) * 0.8
        
        return base_spacing * child_factor

    def calculate_layout(self, root: Node) -> LayoutResult:
        """Calculate the 3D layout for a file system tree.

        Args:
            root: Root node of the file system tree

        Returns:
            LayoutResult containing positions and connections
        """
        self._occupied_boxes.clear()
        result = LayoutResult()

        # Position the root at origin
        root_pos = self._create_root_position(root)
        result.positions[str(root.path)] = root_pos
        self._add_occupied_box(root_pos)

        # Recursively position children
        self._position_children(root, root_pos, result)

        # Calculate overall bounds
        result.bounds = self._calculate_bounds(result.positions)

        return result

    def _create_root_position(self, node: Node) -> Position:
        """Create a position for the root node.

        Args:
            node: Root node

        Returns:
            Position at origin with appropriate size
        """
        size = self.config.dir_size if node.is_directory else self.config.node_size
        height = self._calculate_node_height(node)
        return Position(
            x=-size / 2,
            y=0,  # Base at y=0
            z=-size / 2,
            width=size,
            height=height,
            depth=size,
        )

    def _position_children(
        self,
        parent: Node,
        parent_pos: Position,
        result: LayoutResult,
    ) -> None:
        """Position all children of a node.

        Args:
            parent: Parent node
            parent_pos: Position of the parent
            result: Layout result to update
        """
        if not parent.children:
            return

        # Filter children by depth
        visible_children = [
            c for c in parent.children if c.depth <= self.config.max_depth
        ]

        if not visible_children:
            return

        # Separate directories and files
        directories = [c for c in visible_children if c.is_directory]
        files = [c for c in visible_children if c.is_file]

        if files:
            count = len(files)
            # Use same grid logic as calculate_grid_positions
            grid_cols = math.ceil(math.sqrt(count))
            
            # Match cell size logic in _calculate_grid_positions for on_platform=True
            cell_size = self.config.node_size + self.config.spacing
            
            # Calculate required platform size
            # Platform_Width = (File_Width + Padding) * Grid_Rows
            # We add a little extra padding for aesthetics so files don't touch the edge exactly
            edge_padding = self.config.padding
            platform_width = (grid_cols * cell_size) + (edge_padding * 2)
            platform_depth = platform_width
            
            # Update parent position - preserve center!
            old_width = parent_pos.width
            old_depth = parent_pos.depth
            
            # Center shift (new_pos - old_pos = (old_size - new_size) / 2)
            shift_x = (old_width - platform_width) / 2
            shift_z = (old_depth - platform_depth) / 2
            
            parent_pos.x += shift_x
            parent_pos.z += shift_z
            parent_pos.width = platform_width
            parent_pos.depth = platform_depth
            
            result.positions[str(parent.path)] = parent_pos

        # 2. Position files on parent platform
        if files:
            self._position_files(files, parent_pos, result)

        # 3. Position directories outside (connected by wires)
        if directories:
            self._position_directories(directories, parent_pos, result)

    def _position_directories(
        self,
        directories: list[Node],
        parent_pos: Position,
        result: LayoutResult,
    ) -> None:
        """Position directory nodes.

        Args:
            directories: List of directory nodes
            parent_pos: Position of the parent
            result: Layout result to update
        """
        strategy = self.config.placement_strategy

        if strategy == PlacementStrategy.GRID:
            positions = self._calculate_grid_positions(directories, parent_pos, on_platform=False)
        elif strategy == PlacementStrategy.RADIAL:
            positions = self._calculate_radial_positions(directories, parent_pos)
        elif strategy == PlacementStrategy.SPIRAL:
            positions = self._calculate_spiral_positions(directories, parent_pos)
        else:  # HEURISTIC
            positions = self._calculate_heuristic_positions(directories, parent_pos)

        for i, node in enumerate(directories):
            pos = positions[i]
            result.positions[str(node.path)] = pos
            result.connections.append((str(node.parent.path), str(node.path)))
            self._add_occupied_box(pos)

            # Recursively position children
            self._position_children(node, pos, result)

    def _position_files(
        self,
        files: list[Node],
        parent_pos: Position,
        result: LayoutResult,
    ) -> None:
        """Position file nodes.

        Args:
            files: List of file nodes
            parent_pos: Position of the parent
            result: Layout result to update
        """
        positions = self._calculate_grid_positions(files, parent_pos, on_platform=True)

        for i, node in enumerate(files):
            pos = positions[i]
            result.positions[str(node.path)] = pos
            result.connections.append((str(node.parent.path), str(node.path)))
            self._add_occupied_box(pos)

    def _calculate_grid_positions(
        self,
        nodes: list[Node],
        parent_pos: Position,
        on_platform: bool = False,
    ) -> list[Position]:
        """Calculate grid-based positions for nodes (X-Z plane).

        Args:
            nodes: Nodes to position
            parent_pos: Parent position
            on_platform: Whether to place nodes ON the parent platform (centered)
                         or outside (starting from Z edge)

        Returns:
            List of positions for the nodes
        """
        positions = []
        count = len(nodes)
        if count == 0:
            return []

        # Calculate grid dimensions
        grid_cols = math.ceil(math.sqrt(count))
        
        if on_platform:
            # Files on platform use fixed spacing
            cell_size = self.config.node_size + self.config.spacing
            # Grid total size
            grid_width = grid_cols * cell_size
            # Center on parent platform
            start_x = parent_pos.x + (parent_pos.width - grid_width) / 2
            grid_depth_actual = math.ceil(count / grid_cols) * cell_size
            start_z = parent_pos.z + (parent_pos.depth - grid_depth_actual) / 2
            z_step = 1.0  # Positive growth
            
            for i, node in enumerate(nodes):
                row = i // grid_cols
                col = i % grid_cols
                size = self.config.node_size
                height = self._calculate_node_height(node)
                x = start_x + col * cell_size + (cell_size - size) / 2
                z = start_z + row * cell_size + (cell_size - size) / 2
                y = 0.0
                pos = Position(x=x, y=y, z=z, width=size, height=height, depth=size)
                positions.append(pos)
        else:
            # Directories use dynamic spacing based on their child count
            # Calculate spacing for each directory
            dir_spacings = [self._calculate_directory_spacing(n) for n in nodes]
            max_spacing = max(dir_spacings) if dir_spacings else self.config.grid_size
            
            # Use the maximum spacing plus dir_size to create uniform grid with adequate spacing
            cell_size = max_spacing + self.config.dir_size
            grid_width = grid_cols * cell_size
            
            # Center X relative to parent
            start_x = parent_pos.x + (parent_pos.width - grid_width) / 2
            
            # Start from the back edge of parent, moving further back
            # Increased gap to prevent parent-child overlap
            base_gap = self.config.spacing * 4.0  # Doubled gap between parent and children
            start_z = parent_pos.z - base_gap
            
            start_y = 0.0
            
            for i, node in enumerate(nodes):
                row = i // grid_cols
                col = i % grid_cols
                size = self.config.dir_size
                height = self._calculate_node_height(node)
                
                x = start_x + col * cell_size + (cell_size - size) / 2
                # Negative Z growth (away from parent)
                z = start_z - (row + 1) * cell_size + (cell_size - size) / 2
                y = start_y
                
                pos = Position(x=x, y=y, z=z, width=size, height=height, depth=size)
                pos = self._resolve_collision(pos)
                positions.append(pos)

        return positions

    def _calculate_radial_positions(
        self,
        nodes: list[Node],
        parent_pos: Position,
    ) -> list[Position]:
        """Calculate radial positions for nodes.

        Args:
            nodes: Nodes to position
            parent_pos: Parent position

        Returns:
            List of positions for the nodes
        """
        positions = []
        count = len(nodes)
        radius = self.config.dir_size + self.config.spacing

        for i, node in enumerate(nodes):
            angle = (2 * math.pi * i) / count

            size = self.config.dir_size if node.is_directory else self.config.node_size
            height = self._calculate_node_height(node)

            # Position in a circle around the parent
            center_x, center_y, _ = parent_pos.center
            x = center_x + radius * math.cos(angle) - size / 2
            y = center_y + radius * math.sin(angle) - size / 2
            # Recede in Z
            z = parent_pos.z - self.config.spacing - size

            pos = Position(x=x, y=y, z=z, width=size, height=height, depth=size)
            pos = self._resolve_collision(pos)
            positions.append(pos)

        return positions

    def _calculate_spiral_positions(
        self,
        nodes: list[Node],
        parent_pos: Position,
    ) -> list[Position]:
        """Calculate spiral positions for nodes.

        Args:
            nodes: Nodes to position
            parent_pos: Parent position

        Returns:
            List of positions for the nodes
        """
        positions = []
        count = len(nodes)
        spacing = self.config.spacing

        for i, node in enumerate(nodes):
            # Spiral parameters
            angle = 0.5 * i  # Angle increases with index
            radius = spacing * (1 + 0.1 * i)  # Radius increases slowly

            size = self.config.dir_size if node.is_directory else self.config.node_size
            height = self._calculate_node_height(node)

            center_x, center_y, _ = parent_pos.center
            x = center_x + radius * math.cos(angle) - size / 2
            y = center_y + radius * math.sin(angle) - size / 2
            # Recede in Z
            z = parent_pos.z - spacing - (0.1 * i) - size

            pos = Position(x=x, y=y, z=z, width=size, height=height, depth=size)
            pos = self._resolve_collision(pos)
            positions.append(pos)

        return positions

    def _calculate_heuristic_positions(
        self,
        nodes: list[Node],
        parent_pos: Position,
    ) -> list[Position]:
        """Calculate heuristic-based positions for nodes.

        Uses a smart placement algorithm that considers:
        - Node size (larger directories get more space)
        - Available space (avoids collisions)
        - Visual grouping (related items stay close)

        Args:
            nodes: Nodes to position
            parent_pos: Parent position

        Returns:
            List of positions for the nodes
        """
        positions = []

        # Sort nodes by size (largest first)
        sorted_nodes = sorted(nodes, key=lambda n: n.total_size, reverse=True)

        # Calculate available space "behind" parent (Negative Z)
        start_z = parent_pos.z - self.config.spacing
        center_x, center_y, _ = parent_pos.center

        for i, node in enumerate(sorted_nodes):
            size = self.config.dir_size if node.is_directory else self.config.node_size
            height = self._calculate_node_height(node)

            # Base position
            x = center_x - size / 2
            y = center_y - size / 2
            z = start_z - size # Start considering the node's depth

            # Add offset based on index to spread nodes
            offset_amount = self.config.dir_size + self.config.spacing

            # Arrange in layers based on depth and size
            layer = i // 4  # 4 nodes per layer
            in_layer = i % 4

            # Move further back for each layer
            z -= layer * offset_amount

            # Arrange within layer
            if in_layer == 0:
                x -= offset_amount
                y -= offset_amount
            elif in_layer == 1:
                x += offset_amount
                y -= offset_amount
            elif in_layer == 2:
                x -= offset_amount
                y += offset_amount
            else:
                x += offset_amount
                y += offset_amount

            pos = Position(x=x, y=y, z=z, width=size, height=height, depth=size)

            # Resolve collisions by finding nearest free space
            pos = self._resolve_collision(pos)
            positions.append(pos)

        # Restore original order
        original_order = {id(n): i for i, n in enumerate(nodes)}
        ordered_positions = [None] * len(nodes)
        for i, node in enumerate(sorted_nodes):
            ordered_positions[original_order[id(node)]] = positions[i]

        return [p for p in ordered_positions if p is not None]

    def _resolve_collision(self, pos: Position) -> Position:
        """Resolve collision by moving the position to free space.

        Args:
            pos: Position to check

        Returns:
            Non-colliding position
        """
        # Use larger padding for collision resolution
        collision_padding = self.config.padding * 3
        max_iterations = 20
        
        for iteration in range(max_iterations):
            box = BoundingBox.from_position(pos, padding=collision_padding)
            has_collision = False
            
            # Check for collisions with all occupied boxes
            for occupied in self._occupied_boxes:
                if box.intersects(occupied):
                    has_collision = True
                    
                    # Calculate overlap in both X and Z dimensions
                    overlap_x = min(box.max_x, occupied.max_x) - max(box.min_x, occupied.min_x)
                    overlap_z = min(box.max_z, occupied.max_z) - max(box.min_z, occupied.min_z)
                    
                    # Move in the direction with less overlap (prefer Z for FSN style)
                    if overlap_z < overlap_x:
                        # Move in Z direction (prefer -Z to maintain receding layout)
                        if pos.z > occupied.position.z:
                            # Move forward (more positive Z)
                            new_z = occupied.max_z + collision_padding
                        else:
                            # Move backward (more negative Z)
                            new_z = occupied.min_z - collision_padding - pos.depth
                        pos = Position(
                            x=pos.x,
                            y=pos.y,
                            z=new_z,
                            width=pos.width,
                            height=pos.height,
                            depth=pos.depth,
                        )
                    else:
                        # Move in X direction
                        if pos.x > occupied.position.x:
                            # Move right
                            new_x = occupied.max_x + collision_padding
                        else:
                            # Move left
                            new_x = occupied.min_x - collision_padding - pos.width
                        pos = Position(
                            x=new_x,
                            y=pos.y,
                            z=pos.z,
                            width=pos.width,
                            height=pos.height,
                            depth=pos.depth,
                        )
                    break  # Re-check from start after moving
            
            if not has_collision:
                # No collisions found, position is safe
                break
        
        # If we couldn't resolve after max iterations, push far away
        if iteration == max_iterations - 1:
            pos = Position(
                x=pos.x + (iteration * 10.0),
                y=pos.y,
                z=pos.z - (iteration * 10.0),
                width=pos.width,
                height=pos.height,
                depth=pos.depth,
            )
                
        return pos

    def _add_occupied_box(self, pos: Position) -> None:
        """Add a position to the occupied boxes list.

        Args:
            pos: Position to add
        """
        box = BoundingBox.from_position(pos, padding=self.config.padding)
        self._occupied_boxes.append(box)

    def _calculate_bounds(self, positions: dict[str, Position]) -> BoundingBox:
        """Calculate the overall bounding box of all positions.

        Args:
            positions: Dictionary of positions

        Returns:
            Bounding box containing all positions
        """
        if not positions:
            raise ValueError("No positions to calculate bounds for")

        min_x = min(p.min_x for p in positions.values())
        max_x = max(p.max_x for p in positions.values())
        min_y = min(p.min_y for p in positions.values())
        max_y = max(p.max_y for p in positions.values())
        min_z = min(p.min_z for p in positions.values())
        max_z = max(p.max_z for p in positions.values())

        # Create a position that encompasses all nodes
        bounds_pos = Position(
            x=min_x,
            y=min_y,
            z=min_z,
            width=max_x - min_x,
            height=max_y - min_y,
            depth=max_z - min_z,
        )

        return BoundingBox.from_position(bounds_pos, padding=0)
