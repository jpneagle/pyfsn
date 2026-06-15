#!/usr/bin/env python3
"""Unit tests for Phase 1 UX improvements.

Tests helper logic and simple widget behavior:
- Status bar path truncation
- SearchBar result count label/button state
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PyQt6.QtWidgets import QApplication

from pyfsn.view.main_window import MainWindow, SearchBar


def test_truncate_path_short_path_unchanged():
    """Short paths should not be truncated."""
    path = "/Users/test/project"
    assert MainWindow._truncate_path(path, 80) == path


def test_truncate_path_long_path():
    """Long paths should be truncated with ellipsis in the middle."""
    path = "/Users/test/very/long/path/to/some/deep/nested/file.txt"
    truncated = MainWindow._truncate_path(path, 30)
    assert len(truncated) <= 30
    assert "..." in truncated
    assert truncated.endswith("file.txt")


def test_truncate_path_very_long():
    """Extremely long paths still produce a bounded result."""
    path = "/a" * 500
    truncated = MainWindow._truncate_path(path, 80)
    assert len(truncated) <= 80
    assert "..." in truncated


def test_search_bar_result_count_updates_state():
    """SearchBar buttons and label reflect the result count."""
    app = QApplication.instance() or QApplication(sys.argv)
    bar = SearchBar()

    bar.set_result_count(0, 0)
    assert not bar._prev_btn.isEnabled()
    assert not bar._next_btn.isEnabled()
    assert bar._count_label.text() == ""

    bar.set_result_count(2, 10)
    assert bar._prev_btn.isEnabled()
    assert bar._next_btn.isEnabled()
    assert bar._count_label.text() == "2 / 10"


from pyfsn.view.filter_panel import FilterPanel


def test_parse_human_readable_sizes():
    """FilterPanel parses plain bytes and human-readable size units."""
    assert FilterPanel._parse_size("1024") == 1024
    assert FilterPanel._parse_size("1KB") == 1024
    assert FilterPanel._parse_size("1.5mb") == int(1.5 * 1024 ** 2)
    assert FilterPanel._parse_size(" 2 GB ") == 2 * 1024 ** 3
    assert FilterPanel._parse_size("bad") is None
    assert FilterPanel._parse_size("") is None
