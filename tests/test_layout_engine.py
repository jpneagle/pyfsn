"""Tests for the fsn-style tree layout engine."""

from pathlib import Path

import pytest

from pyfsn.layout.engine import DirHeightMode, HeightMode, LayoutConfig, LayoutEngine
from pyfsn.model.node import Node, NodeType


def make_dir(name: str, parent: Node | None = None) -> Node:
    path = (parent.path / name) if parent else Path("/") / name
    node = Node(path=path, name=name, size=0, type=NodeType.DIRECTORY,
                permissions=0o755, mtime=0.0, is_loaded=True)
    if parent is not None:
        parent.add_child(node)
    return node


def make_file(name: str, parent: Node, size: int = 100,
              node_type: NodeType = NodeType.FILE) -> Node:
    node = Node(path=parent.path / name, name=name, size=size, type=node_type,
                permissions=0o644, mtime=0.0)
    parent.add_child(node)
    return node


def all_dirs(node: Node) -> list[Node]:
    result = [node]
    for child in node.children:
        if child.is_directory:
            result.extend(all_dirs(child))
    return result


def overlap_xz(a, b) -> bool:
    eps = 1e-6
    return (a.min_x < b.max_x - eps and b.min_x < a.max_x - eps
            and a.min_z < b.max_z - eps and b.min_z < a.max_z - eps)


def skewed_tree() -> Node:
    """One heavy branch (deep and wide) next to tiny siblings."""
    root = make_dir("root")
    for i in range(3):
        make_file(f"r{i}.txt", root, size=10 * (i + 1))
    heavy = make_dir("heavy", root)
    for i in range(500):
        make_file(f"h{i}.bin", heavy, size=i + 1)
    for i in range(12):
        sub = make_dir(f"sub{i}", heavy)
        for j in range(i):
            make_file(f"f{j}", sub)
        if i % 3 == 0:
            deep = make_dir("deep", sub)
            for j in range(40):
                make_file(f"d{j}", deep)
            make_dir("deeper", deep)
    for i in range(5):
        make_dir(f"small{i}", root)
    return root


def layout(root: Node, **config):
    config.setdefault("max_depth", 10)
    return LayoutEngine(LayoutConfig(**config)).calculate_layout(root)


def test_no_pedestals_overlap():
    root = skewed_tree()
    result = layout(root)
    dirs = [result.positions[str(d.path)] for d in all_dirs(root)]
    for i, a in enumerate(dirs):
        for b in dirs[i + 1:]:
            assert not overlap_xz(a, b), f"{a} overlaps {b}"


def test_children_are_behind_parent_and_share_a_front_edge():
    root = skewed_tree()
    result = layout(root)
    for directory in all_dirs(root):
        parent_pos = result.positions[str(directory.path)]
        fronts = set()
        for child in directory.children:
            if not child.is_directory:
                continue
            child_pos = result.positions[str(child.path)]
            # Entirely behind the parent's back edge (towards -Z)
            assert child_pos.max_z < parent_pos.min_z
            fronts.add(round(child_pos.max_z, 6))
        assert len(fronts) <= 1


def test_subtree_stays_inside_its_strip():
    root = skewed_tree()
    result = layout(root)

    def check(directory: Node) -> None:
        pos = result.positions[str(directory.path)]
        slot = result.slot_widths[str(directory.path)]
        center = pos.x + pos.width / 2
        left, right = center - slot / 2 - 1e-6, center + slot / 2 + 1e-6
        for sub in all_dirs(directory):
            sub_pos = result.positions[str(sub.path)]
            assert left <= sub_pos.min_x and sub_pos.max_x <= right
        for child in directory.children:
            if child.is_directory:
                check(child)

    check(root)


def test_files_stand_on_their_pedestal():
    root = skewed_tree()
    make_file("link", root, size=5, node_type=NodeType.SYMLINK)
    result = layout(root)
    for directory in all_dirs(root):
        pedestal = result.positions[str(directory.path)]
        assert pedestal.y == 0.0
        files = [c for c in directory.children if not c.is_directory]
        boxes = [result.positions[str(f.path)] for f in files]
        for box in boxes:
            assert pedestal.min_x <= box.min_x and box.max_x <= pedestal.max_x
            assert pedestal.min_z <= box.min_z and box.max_z <= pedestal.max_z
            assert box.y == pytest.approx(pedestal.height)
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                assert not overlap_xz(a, b)


def test_pedestal_height_counts_only_direct_files_by_default():
    root = make_dir("root")
    small = make_dir("small", root)
    make_file("a", small, size=10)
    big_child = make_dir("big_child", small)
    make_file("huge", big_child, size=1_000_000)

    files_only = layout(root)
    assert (files_only.positions[str(small.path)].height
            < files_only.positions[str(big_child.path)].height)

    including = layout(root, dir_height_mode=DirHeightMode.INCLUDING_CHILDREN)
    assert (including.positions[str(small.path)].height
            >= including.positions[str(big_child.path)].height)

    flat = layout(root, dir_height_mode=DirHeightMode.FLAT)
    assert (flat.positions[str(small.path)].height
            == flat.positions[str(big_child.path)].height)


def test_height_modes():
    root = make_dir("root")
    small = make_file("small", root, size=100)
    large = make_file("large", root, size=10_000)

    def heights(mode: HeightMode) -> tuple[float, float]:
        result = layout(root, height_mode=mode)
        return (result.positions[str(small.path)].height,
                result.positions[str(large.path)].height)

    config = LayoutConfig()
    none_small, none_large = heights(HeightMode.NONE)
    assert none_small == none_large == config.min_height

    lin_small, lin_large = heights(HeightMode.LINEAR)
    exa_small, exa_large = heights(HeightMode.EXAGGERATED)
    assert lin_large == pytest.approx(config.max_file_height)
    assert exa_large == pytest.approx(config.max_file_height)
    # Square root grows small files more than linear does
    assert lin_small < exa_small < exa_large

    log_small, log_large = heights(HeightMode.LOG)
    assert config.min_height < log_small < log_large <= config.max_file_height


def test_max_depth_and_prune_hide_contents():
    root = skewed_tree()
    heavy = root.find_child_by_name("heavy")

    shallow = layout(root, max_depth=1)
    assert str(heavy.path) in shallow.positions
    assert str(heavy.children[0].path) not in shallow.positions

    engine = LayoutEngine(LayoutConfig(max_depth=10))
    engine.pruned_paths.add(str(heavy.path))
    pruned = engine.calculate_layout(root)
    assert str(heavy.path) in pruned.positions
    assert pruned.pruned == {str(heavy.path)}
    for descendant in heavy.get_all_descendants():
        assert str(descendant.path) not in pruned.positions
    # A pruned directory shrinks back to an empty pedestal
    assert pruned.positions[str(heavy.path)].width == LayoutConfig().dir_size


def test_expanding_a_directory_keeps_the_layout_valid():
    root = make_dir("root")
    a = make_dir("a", root)
    b = make_dir("b", root)
    before = layout(root)
    width_before = before.slot_widths[str(root.path)]

    for i in range(8):
        make_dir(f"n{i}", a)
    after = layout(root)
    assert after.slot_widths[str(root.path)] > width_before
    dirs = [after.positions[str(d.path)] for d in all_dirs(root)]
    for i, p in enumerate(dirs):
        for q in dirs[i + 1:]:
            assert not overlap_xz(p, q)
    # Siblings keep their order along X
    assert after.positions[str(a.path)].x < after.positions[str(b.path)].x


def test_symlink_keeps_its_own_identity(tmp_path):
    """A symlink must not share the path (identity) of the directory it points to."""
    from pyfsn.model.scanner import Scanner

    (tmp_path / "target").mkdir()
    (tmp_path / "target" / "file.txt").write_text("x")
    (tmp_path / "link").symlink_to(tmp_path / "target", target_is_directory=True)

    root = Scanner(lazy_load=False).scan(tmp_path.resolve())
    by_name = {child.name: child for child in root.children}
    assert by_name["link"].type == NodeType.SYMLINK
    assert by_name["link"].path != by_name["target"].path

    result = layout(root)
    # The real directory keeps its pedestal; the link is a separate box on the root
    assert result.positions[str(by_name["target"].path)].y == 0.0
    assert result.positions[str(by_name["link"].path)].y > 0.0
