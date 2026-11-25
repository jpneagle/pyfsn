"""Layout engine for 3D file system visualization.

The layout follows the landscape of SGI fsn: every directory is a pedestal,
its files stand on top of it as boxes, and the sub-directories of a directory
form a row behind it.  Each sub-tree owns a strip of the ground that is as
wide as everything below it, so pedestals never overlap and no collision
resolution is needed.

The layout is computed in two passes:

1. *measure* (bottom-up): pedestal footprint from the file count, then the
   width of the whole sub-tree.
2. *place* (top-down): every directory is centred on its strip and the
   sub-directories of one directory share the same front edge, right behind
   their parent.
"""

import math
from dataclasses import dataclass, field
from enum import Enum

from pyfsn.layout.box import BoundingBox
from pyfsn.layout.position import Position
from pyfsn.model.node import Node


class HeightMode(Enum):
    """How sizes are turned into heights (fsn "Display > Height" menu)."""

    NONE = "none"  # Everything at the minimum height
    LINEAR = "linear"  # Proportional to size
    EXAGGERATED = "exaggerated"  # Proportional to size ** height_power (fsn default: sqrt)
    LOG = "log"  # log10 scale (pyfsn extension, shows small files best)


class DirHeightMode(Enum):
    """What a pedestal's height stands for (fsn "Display > Directory Height")."""

    FLAT = "flat"  # All pedestals flat
    FILES_ONLY = "files"  # Size of the files directly inside the directory
    INCLUDING_CHILDREN = "children"  # Files plus all sub-directories (like du)


@dataclass
class LayoutConfig:
    """Configuration for the layout engine.

    Attributes:
        node_size: Footprint of a file cell
        dir_size: Minimum footprint of a directory pedestal
        spacing: Spacing between file cells on a pedestal
        padding: Margin between the file grid and the pedestal edge
        max_depth: Maximum depth (relative to the root) to lay out
        sibling_gap: Gap between neighbouring sub-trees
        level_gap: Gap between a row of directories and the row behind it
        height_mode: How sizes map to heights
        dir_height_mode: What a pedestal's height stands for
        height_power: Exponent used by HeightMode.EXAGGERATED (0.5 = sqrt)
        min_height: Height of an empty file (files always keep a minimum)
        max_file_height: Height of the largest file in the scene
        min_dir_height: Height of an empty (or flat) pedestal
        max_dir_height: Height of the largest pedestal in the scene
    """

    node_size: float = 1.0
    dir_size: float = 2.0
    spacing: float = 0.5
    padding: float = 0.2
    max_depth: int = 5
    sibling_gap: float = 2.0
    level_gap: float = 5.0
    height_mode: HeightMode = HeightMode.EXAGGERATED
    dir_height_mode: DirHeightMode = DirHeightMode.FILES_ONLY
    height_power: float = 0.5
    min_height: float = 0.15
    max_file_height: float = 4.0
    min_dir_height: float = 0.15
    max_dir_height: float = 2.0


@dataclass
class LayoutResult:
    """Result of a layout operation.

    Attributes:
        positions: Dictionary mapping node paths to positions.  A directory's
            position is its pedestal (y=0, height = pedestal height); a file's
            position already sits on top of its pedestal.
        connections: List of (parent_path, child_path) tuples for connections
        bounds: Overall bounding box of the layout
        slot_widths: Width of the ground strip owned by each directory's
            sub-tree (used to fit directory labels without overlap)
        pruned: Paths of directories whose contents are hidden (pruned)
    """

    positions: dict[str, Position] = field(default_factory=dict)
    connections: list[tuple[str, str]] = field(default_factory=list)
    bounds: BoundingBox | None = None
    slot_widths: dict[str, float] = field(default_factory=dict)
    pruned: set[str] = field(default_factory=set)


@dataclass
class _Measure:
    """Per-directory measurements gathered by the bottom-up pass."""

    node: Node
    level: int
    files: list[Node]
    dirs: list["_Measure"]
    cols: int
    width: float  # Pedestal width (X)
    depth: float  # Pedestal depth (Z)
    slot: float  # Width of the whole sub-tree strip
    files_size: int  # Size of the files directly inside
    tree_size: int  # files_size plus all laid-out descendants


class LayoutEngine:
    """Engine for calculating 3D positions for file system nodes."""

    def __init__(self, config: LayoutConfig | None = None) -> None:
        """Initialize the layout engine.

        Args:
            config: Layout configuration (uses defaults if None)
        """
        self.config = config or LayoutConfig()
        # Paths of pruned directories: shown as a pedestal, contents hidden.
        self.pruned_paths: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_layout(self, root: Node) -> LayoutResult:
        """Calculate the 3D layout for a file system tree.

        Args:
            root: Root node of the file system tree

        Returns:
            LayoutResult containing positions and connections
        """
        result = LayoutResult()

        if not root.is_directory:
            height = self._scaled_height(
                root.size, max(root.size, 1), self.config.min_height, self.config.max_file_height
            )
            size = self.config.node_size
            result.positions[str(root.path)] = Position(
                x=-size / 2, y=0.0, z=-size / 2, width=size, height=height, depth=size
            )
            result.bounds = self._calculate_bounds(result.positions)
            return result

        # Pass 1: measure bottom-up
        measure = self._measure(root, 0)

        # Scene-wide scale references so that heights are comparable
        all_measures = list(self._iter_measures(measure))
        max_file_size = max(
            (f.size for m in all_measures for f in m.files), default=0
        )
        if self.config.dir_height_mode == DirHeightMode.INCLUDING_CHILDREN:
            max_dir_size = max((m.tree_size for m in all_measures), default=0)
        else:
            max_dir_size = max((m.files_size for m in all_measures), default=0)

        # Pass 2: place top-down, the root centred on the origin
        self._place(
            measure, 0.0, measure.depth / 2, max_file_size, max_dir_size, result
        )

        result.pruned = {
            p for p in self.pruned_paths if p in result.positions
        }
        result.bounds = self._calculate_bounds(result.positions)
        return result

    def directory_size(self, node: Node) -> int:
        """Size a pedestal stands for under the current directory height mode."""
        if self.config.dir_height_mode == DirHeightMode.INCLUDING_CHILDREN:
            return node.total_size
        return node.files_size

    # ------------------------------------------------------------------
    # Pass 1: measure
    # ------------------------------------------------------------------

    def _measure(self, node: Node, level: int) -> _Measure:
        """Measure a directory and (recursively) its sub-tree."""
        cfg = self.config
        files: list[Node] = []
        dirs: list[_Measure] = []

        visible = (
            level < cfg.max_depth and str(node.path) not in self.pruned_paths
        )
        if visible:
            for child in node.children:
                if child.is_directory:
                    dirs.append(self._measure(child, level + 1))
                else:
                    # Regular files and symlinks both stand on the pedestal
                    files.append(child)

        count = len(files)
        cell = cfg.node_size + cfg.spacing
        if count:
            # Slightly wider than deep, like the rows of files in fsn
            cols = max(1, math.ceil(math.sqrt(count * 1.6)))
            cols = min(cols, count)
            rows = math.ceil(count / cols)
        else:
            cols, rows = 0, 0

        width = max(cfg.dir_size, cols * cell + cfg.padding * 2)
        depth = max(cfg.dir_size, rows * cell + cfg.padding * 2)

        children_width = sum(d.slot for d in dirs)
        if dirs:
            children_width += cfg.sibling_gap * (len(dirs) - 1)
        slot = max(width, children_width)

        files_size = sum(f.size for f in files)
        tree_size = files_size + sum(d.tree_size for d in dirs)

        return _Measure(
            node=node,
            level=level,
            files=files,
            dirs=dirs,
            cols=cols,
            width=width,
            depth=depth,
            slot=slot,
            files_size=files_size,
            tree_size=tree_size,
        )

    def _iter_measures(self, measure: _Measure):
        """Iterate a measure tree without recursion limits on wide trees."""
        stack = [measure]
        while stack:
            current = stack.pop()
            yield current
            stack.extend(current.dirs)

    # ------------------------------------------------------------------
    # Pass 2: place
    # ------------------------------------------------------------------

    def _place(
        self,
        measure: _Measure,
        center_x: float,
        front_z: float,
        max_file_size: int,
        max_dir_size: int,
        result: LayoutResult,
    ) -> None:
        """Place a directory, its files and its sub-directories.

        Args:
            measure: Measurements of the directory
            center_x: Centre of the ground strip owned by its sub-tree
            front_z: Z of its front edge (the edge facing the root)
        """
        cfg = self.config
        node = measure.node
        path = str(node.path)

        # Pedestal
        if cfg.dir_height_mode == DirHeightMode.FLAT:
            pedestal_height = cfg.min_dir_height
        else:
            dir_size = (
                measure.tree_size
                if cfg.dir_height_mode == DirHeightMode.INCLUDING_CHILDREN
                else measure.files_size
            )
            pedestal_height = self._scaled_height(
                dir_size, max_dir_size, cfg.min_dir_height, cfg.max_dir_height
            )

        pedestal = Position(
            x=center_x - measure.width / 2,
            y=0.0,
            z=front_z - measure.depth,
            width=measure.width,
            height=pedestal_height,
            depth=measure.depth,
        )
        result.positions[path] = pedestal
        result.slot_widths[path] = measure.slot

        # Files: a centred grid on top of the pedestal, first file front-left
        if measure.files:
            cell = cfg.node_size + cfg.spacing
            cols = measure.cols
            rows = math.ceil(len(measure.files) / cols)
            grid_w = cols * cell
            grid_d = rows * cell
            start_x = pedestal.x + (pedestal.width - grid_w) / 2
            # Front edge of the grid (max Z); rows recede towards -Z
            front = pedestal.z + pedestal.depth - (pedestal.depth - grid_d) / 2
            inset = (cell - cfg.node_size) / 2

            for i, file_node in enumerate(measure.files):
                row, col = divmod(i, cols)
                height = self._scaled_height(
                    file_node.size, max_file_size, cfg.min_height, cfg.max_file_height
                )
                result.positions[str(file_node.path)] = Position(
                    x=start_x + col * cell + inset,
                    y=pedestal_height,
                    z=front - (row + 1) * cell + inset,
                    width=cfg.node_size,
                    height=height,
                    depth=cfg.node_size,
                )
                result.connections.append((path, str(file_node.path)))

        # Sub-directories: one row right behind this pedestal, centred on the
        # strip. A deep pedestal only pushes back its own children.
        if measure.dirs:
            child_front = pedestal.z - cfg.level_gap
            children_width = sum(d.slot for d in measure.dirs) + cfg.sibling_gap * (
                len(measure.dirs) - 1
            )
            cursor = center_x - children_width / 2
            for child in measure.dirs:
                child_center = cursor + child.slot / 2
                result.connections.append((path, str(child.node.path)))
                self._place(
                    child, child_center, child_front, max_file_size, max_dir_size, result
                )
                cursor += child.slot + cfg.sibling_gap

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _scaled_height(
        self, size: int, max_size: int, min_height: float, max_height: float
    ) -> float:
        """Map a size to a height between min_height and max_height."""
        mode = self.config.height_mode
        if mode == HeightMode.NONE or size <= 0 or max_size <= 0:
            return min_height

        if mode == HeightMode.LOG:
            # log10(1 KB) ≈ 3, log10(1 GB) ≈ 9 → spread over the height range
            fraction = min(math.log10(max(size, 1)) / 10.0, 1.0)
        else:
            fraction = min(size / max_size, 1.0)
            if mode == HeightMode.EXAGGERATED:
                fraction = fraction ** self.config.height_power

        return min_height + (max_height - min_height) * fraction

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

        bounds_pos = Position(
            x=min_x,
            y=min_y,
            z=min_z,
            width=max_x - min_x,
            height=max_y - min_y,
            depth=max_z - min_z,
        )

        return BoundingBox.from_position(bounds_pos, padding=0)
