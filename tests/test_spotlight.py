#!/usr/bin/env python3
"""Unit tests for Spotlight Search Visualization (Phase 3.2).

Tests the spotlight search functionality including:
- Match detection
- Opacity calculations
- Color modification (desaturation)
- Animation states
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from pyfsn.view.spotlight import SpotlightSearch
from pyfsn.model.node import Node, NodeType


def test_spotlight_initialization():
    """Test spotlight initialization."""
    spotlight = SpotlightSearch()

    assert not spotlight.is_active
    assert spotlight.match_count == 0
    assert spotlight.search_query == ""

    print("✓ Spotlight initialization test passed")


def test_spotlight_start_search():
    """Test starting a spotlight search."""
    spotlight = SpotlightSearch()

    # Create mock nodes
    node1 = Node(Path("/test/test1.py"), "test1.py", 1024, NodeType.FILE, 0o644, 1234567890.0)
    node2 = Node(Path("/test/test2.py"), "test2.py", 2048, NodeType.FILE, 0o644, 1234567891.0)
    node3 = Node(Path("/test/other.txt"), "other.txt", 512, NodeType.FILE, 0o644, 1234567892.0)

    matching_nodes = {node1, node2}
    spotlight.start_search("test", matching_nodes)

    assert spotlight.is_active
    assert spotlight.match_count == 2
    assert spotlight.search_query == "test"
    assert spotlight.is_match(str(node1.path))
    assert spotlight.is_match(str(node2.path))
    assert not spotlight.is_match(str(node3.path))

    print("✓ Spotlight start search test passed")


def test_spotlight_opacity():
    """Test opacity calculation for matching/non-matching nodes."""
    spotlight = SpotlightSearch()

    # Create mock nodes
    node1 = Node(Path("/test/match.py"), "match.py", 1024, NodeType.FILE, 0o644, 1234567890.0)
    node2 = Node(Path("/test/nomatch.txt"), "nomatch.txt", 512, NodeType.FILE, 0o644, 1234567891.0)

    matching_nodes = {node1}
    spotlight.start_search("match", matching_nodes)

    # Wait for animation to complete
    time.sleep(0.4)

    # Matching node should have full opacity
    match_opacity = spotlight.get_node_opacity(str(node1.path))
    assert match_opacity >= 0.9, f"Match opacity should be ~1.0, got {match_opacity}"

    # Non-matching node should be dimmed
    nomatch_opacity = spotlight.get_node_opacity(str(node2.path))
    assert nomatch_opacity <= 0.4, f"Non-match opacity should be ~0.3, got {nomatch_opacity}"

    print("✓ Spotlight opacity test passed")


def test_spotlight_desaturation():
    """Test color desaturation for non-matching nodes."""
    spotlight = SpotlightSearch()

    # Create mock nodes
    node1 = Node(Path("/test/match.py"), "match.py", 1024, NodeType.FILE, 0o644, 1234567890.0)
    node2 = Node(Path("/test/nomatch.txt"), "nomatch.txt", 512, NodeType.FILE, 0o644, 1234567891.0)

    matching_nodes = {node1}
    spotlight.start_search("match", matching_nodes)

    # Wait for animation to complete
    time.sleep(0.4)

    # Test with a bright red color
    original_color = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)

    # Matching node should keep original color
    match_color = spotlight.get_node_color(str(node1.path), original_color)
    assert np.allclose(match_color[:3], original_color[:3], atol=0.01), \
        "Matching node should keep original color"

    # Non-matching node should be desaturated (grayscale)
    nomatch_color = spotlight.get_node_color(str(node2.path), original_color)
    assert abs(nomatch_color[0] - nomatch_color[1]) < 0.1, \
        "Non-matching node should be desaturated (grayscale)"
    assert abs(nomatch_color[1] - nomatch_color[2]) < 0.1, \
        "Non-matching node should be desaturated (grayscale)"

    print("✓ Spotlight desaturation test passed")


def test_spotlight_clear():
    """Test clearing spotlight search."""
    spotlight = SpotlightSearch()

    # Create mock nodes
    node1 = Node(Path("/test/test.py"), "test.py", 1024, NodeType.FILE, 0o644, 1234567890.0)

    matching_nodes = {node1}
    spotlight.start_search("test", matching_nodes)

    assert spotlight.is_active

    # Clear the search
    spotlight.clear_search()

    # Wait for fade-out animation to complete (0.3s + buffer)
    time.sleep(0.5)

    # Trigger animation progress check
    spotlight.get_node_opacity(str(node1.path))

    # After animation, should be fully inactive
    assert not spotlight.is_active, "Spotlight should be inactive after clear"
    assert spotlight.match_count == 0, "Match count should be 0 after clear"

    print("✓ Spotlight clear test passed")


def test_spotlight_animation():
    """Test spotlight animation timing."""
    spotlight = SpotlightSearch()

    # Create mock nodes
    node1 = Node(Path("/test/test.py"), "test.py", 1024, NodeType.FILE, 0o644, 1234567890.0)

    matching_nodes = {node1}
    spotlight.start_search("test", matching_nodes)

    # Immediately after start, animation should be in progress
    opacity = spotlight.get_node_opacity(str(node1.path))
    assert 0.0 < opacity < 1.0, "Animation should be in progress"

    # Wait for animation to complete
    time.sleep(0.4)

    # After animation, should be at full opacity
    opacity = spotlight.get_node_opacity(str(node1.path))
    assert opacity >= 0.9, "Animation should be complete"

    print("✓ Spotlight animation test passed")


def test_spotlight_params():
    """Test spotlight cone parameters."""
    spotlight = SpotlightSearch()

    # Create mock nodes
    node1 = Node(Path("/test/test.py"), "test.py", 1024, NodeType.FILE, 0o644, 1234567890.0)

    matching_nodes = {node1}
    spotlight.start_search("test", matching_nodes)

    # Wait for animation
    time.sleep(0.4)

    # Should draw spotlight for matching node
    assert spotlight.should_draw_spotlight(str(node1.path))

    # Get spotlight parameters
    params = spotlight.get_spotlight_params(str(node1.path))
    assert params is not None
    height, radius, alpha = params
    assert height > 0
    assert radius > 0
    assert 0.0 < alpha <= 1.0

    print("✓ Spotlight parameters test passed")


def test_spotlight_update_results():
    """Test updating search results without resetting animation."""
    spotlight = SpotlightSearch()

    # Create mock nodes
    node1 = Node(Path("/test/test1.py"), "test1.py", 1024, NodeType.FILE, 0o644, 1234567890.0)
    node2 = Node(Path("/test/test2.py"), "test2.py", 2048, NodeType.FILE, 0o644, 1234567891.0)
    node3 = Node(Path("/test/test3.py"), "test3.py", 3072, NodeType.FILE, 0o644, 1234567892.0)

    # Start with one match
    spotlight.start_search("test", {node1})
    assert spotlight.match_count == 1

    # Update with more matches (no animation reset)
    spotlight.update_results({node1, node2, node3})
    assert spotlight.match_count == 3
    assert spotlight.is_match(str(node1.path))
    assert spotlight.is_match(str(node2.path))
    assert spotlight.is_match(str(node3.path))

    print("✓ Spotlight update results test passed")


def run_all_tests():
    """Run all spotlight tests."""
    print("=== Running Spotlight Search Tests ===\n")

    test_spotlight_initialization()
    test_spotlight_start_search()
    test_spotlight_opacity()
    test_spotlight_desaturation()
    test_spotlight_clear()
    test_spotlight_animation()
    test_spotlight_params()
    test_spotlight_update_results()

    print("\n=== All Spotlight Tests Passed! ===")


if __name__ == "__main__":
    run_all_tests()
