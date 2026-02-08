#!/usr/bin/env python3
"""Demo script for Spotlight Search Visualization (Phase 3.2).

This script demonstrates the spotlight search feature that:
1. Highlights matching nodes with full opacity
2. Dims non-matching nodes to 30% opacity and desaturates them
3. Shows cone spotlights above matching nodes
4. Provides smooth fade in/out animations

Usage:
    python tests/test_spotlight_demo.py
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

from pyfsn.controller.controller import Controller
from pyfsn.model.node import Node


def demo_spotlight_search():
    """Demonstrate spotlight search visualization."""
    app = QApplication(sys.argv)

    # Use a test directory with some files
    test_dir = Path(__file__).parent.parent / "src" / "pyfsn"

    # Create controller
    controller = Controller(test_dir)

    # Show window
    controller.show()

    # Wait for initial scan to complete
    def start_demo():
        """Start the demo after initial scan."""
        print("=== Spotlight Search Demo ===")
        print("Scene loaded. Starting demo in 2 seconds...")

        # Schedule demo steps
        QTimer.singleShot(2000, lambda: demo_step_1(controller))
        QTimer.singleShot(5000, lambda: demo_step_2(controller))
        QTimer.singleShot(9000, lambda: demo_step_3(controller))
        QTimer.singleShot(13000, lambda: demo_step_4(controller))
        QTimer.singleShot(17000, lambda: demo_step_5(controller))

    # Connect to scan complete signal
    controller.scan_complete.connect(start_demo)

    # Start the controller
    controller.start()

    sys.exit(app.exec())


def demo_step_1(controller: Controller):
    """Demo Step 1: Search for 'view' files."""
    print("\n[Step 1] Searching for 'view' files...")
    print("Expected: Files/directories with 'view' in name are highlighted")
    print("         Others are dimmed and desaturated")
    controller._perform_search('view')


def demo_step_2(controller: Controller):
    """Demo Step 2: Search for 'py' files (Python files)."""
    print("\n[Step 2] Searching for '.py' files...")
    print("Expected: Python files are highlighted with cone spotlights")
    controller._perform_search('.py')


def demo_step_3(controller: Controller):
    """Demo Step 3: Search for directories."""
    print("\n[Step 3] Searching for 'controller'...")
    print("Expected: Controller directory is highlighted")
    controller._perform_search('controller')


def demo_step_4(controller: Controller):
    """Demo Step 4: Navigate between results."""
    print("\n[Step 4] Navigating between search results...")
    print("Expected: Camera moves to each matching node")

    for i in range(3):
        QTimer.singleShot(i * 1500, controller.next_search_result)


def demo_step_5(controller: Controller):
    """Demo Step 5: Clear search."""
    print("\n[Step 5] Clearing search...")
    print("Expected: All nodes return to normal visibility")
    controller._perform_search('')  # Empty query clears search

    print("\n=== Demo Complete ===")
    print("You can now test the search feature manually:")
    print("  - Type in the search bar at the top")
    print("  - Watch matching nodes highlight")
    print("  - Non-matching nodes will dim")
    print("  - Cone spotlights appear above matches")


if __name__ == "__main__":
    demo_spotlight_search()
