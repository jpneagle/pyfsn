"""Input handler for mouse and keyboard events.

Manages user interaction with the 3D file system visualization including
camera controls, object selection, and navigation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from PyQt6.QtCore import Qt, QPoint, QTimer
from PyQt6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent, QKeyEvent
import numpy as np

from pyfsn.view.camera import Camera, CameraMode
from pyfsn.model.node import Node


class MouseButton(Enum):
    """Mouse button identifiers."""

    LEFT = Qt.MouseButton.LeftButton
    MIDDLE = Qt.MouseButton.MiddleButton
    RIGHT = Qt.MouseButton.RightButton


class KeyModifier(Enum):
    """Keyboard modifier identifiers."""

    SHIFT = Qt.KeyboardModifier.ShiftModifier
    CONTROL = Qt.KeyboardModifier.ControlModifier
    ALT = Qt.KeyboardModifier.AltModifier
    META = Qt.KeyboardModifier.MetaModifier


@dataclass
class InputState:
    """Current input state."""

    mouse_position: QPoint = QPoint(0, 0)
    mouse_pressed: set[MouseButton] = None
    modifiers: set[KeyModifier] = None
    last_mouse_position: QPoint = None
    mouse_drag_start: QPoint = None
    is_dragging: bool = False
    drag_threshold: int = 5

    def __post_init__(self) -> None:
        if self.mouse_pressed is None:
            self.mouse_pressed = set()
        if self.modifiers is None:
            self.modifiers = set()
        if self.last_mouse_position is None:
            self.last_mouse_position = QPoint(0, 0)
        if self.mouse_drag_start is None:
            self.mouse_drag_start = QPoint(0, 0)

    def update_modifiers(self, modifiers: Qt.KeyboardModifier) -> None:
        """Update modifier state from Qt modifiers."""
        self.modifiers.clear()
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            self.modifiers.add(KeyModifier.SHIFT)
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            self.modifiers.add(KeyModifier.CONTROL)
        if modifiers & Qt.KeyboardModifier.AltModifier:
            self.modifiers.add(KeyModifier.ALT)
        if modifiers & Qt.KeyboardModifier.MetaModifier:
            self.modifiers.add(KeyModifier.META)

    @property
    def shift_pressed(self) -> bool:
        return KeyModifier.SHIFT in self.modifiers

    @property
    def control_pressed(self) -> bool:
        return KeyModifier.CONTROL in self.modifiers

    @property
    def alt_pressed(self) -> bool:
        return KeyModifier.ALT in self.modifiers


@dataclass
class Action:
    """Input action definition."""

    name: str
    mouse_button: MouseButton | None = None
    key: int | None = None
    modifiers: set[KeyModifier] | None = None
    on_press: Callable[[], None] | None = None
    on_release: Callable[[], None] | None = None
    on_drag: Callable[[int, int], None] | None = None
    on_click: Callable[[int, int], None] | None = None
    on_double_click: Callable[[int, int], None] | None = None


class InputHandler:
    """Handler for mouse and keyboard input events.

    Connects UI events to camera controls and object selection.
    """

    def __init__(self, camera: Camera, renderer: object) -> None:
        """Initialize input handler.

        Args:
            camera: Camera object to control
            renderer: Renderer widget for raycasting
        """
        self._camera = camera
        self._renderer = renderer
        self._state = InputState()

        # Callbacks
        self._on_node_clicked: Callable[[Node, bool], None] | None = None
        self._on_node_focused: Callable[[Node], None] | None = None
        self._on_selection_changed: Callable[[set[Node]], None] | None = None
        self._on_navigate_next: Callable[[], None] | None = None
        self._on_navigate_previous: Callable[[], None] | None = None
        self._camera_mode_changed_callback: Callable[[CameraMode], None] | None = None

        # Selection state
        self._selected_nodes: set[int] = set()
        self._focused_node: Node | None = None

        # Double click detection
        self._last_click_time = 0
        self._last_click_pos = QPoint()
        self._double_click_interval = 300  # ms

        # Reference to nodes and positions (set by controller)
        self._nodes: dict[str, Node] = {}
        self._positions: dict[str, object] = {}

        # Hover detection
        self._hovered_node: Node | None = None
        self._hover_debounce_timer = QTimer()
        self._hover_debounce_timer.setSingleShot(True)
        self._hover_debounce_timer.timeout.connect(self._update_hover)
        self._last_hover_pos: QPoint | None = None
        
        # Key tracking
        self._keys_pressed: set[int] = set()

    # Public API

    def set_scene_data(self, nodes: dict[str, Node], positions: dict[str, object]) -> None:
        """Set scene data for raycasting.

        Args:
            nodes: Dictionary mapping path strings to Node objects
            positions: Dictionary mapping path strings to Position objects
        """
        self._nodes = nodes
        self._positions = positions

    def set_node_clicked_callback(self, callback: Callable[[Node, bool], None]) -> None:
        """Set callback for node click events.

        Args:
            callback: Function receiving (node, is_double_click)
        """
        self._on_node_clicked = callback

    def set_node_focused_callback(self, callback: Callable[[Node], None]) -> None:
        """Set callback for node focus events.

        Args:
            callback: Function receiving focused node
        """
        self._on_node_focused = callback

    def set_selection_changed_callback(self, callback: Callable[[set[Node]], None]) -> None:
        """Set callback for selection changes.

        Args:
            callback: Function receiving set of selected nodes
        """
        self._on_selection_changed = callback

    def set_navigate_next_callback(self, callback: Callable[[], None]) -> None:
        """Set callback for next navigation (Tab).

        Args:
            callback: Function to call
        """
        self._on_navigate_next = callback

    def set_navigate_previous_callback(self, callback: Callable[[], None]) -> None:
        """Set callback for previous navigation (Shift+Tab).

        Args:
            callback: Function to call
        """
        self._on_navigate_previous = callback

    def set_camera_mode_changed_callback(self, callback: Callable[[CameraMode], None]) -> None:
        """Set callback for camera mode changes.

        Args:
            callback: Function receiving new camera mode
        """
        self._camera_mode_changed_callback = callback

    # Mouse event handlers

    def mouse_press_event(self, event: QMouseEvent) -> bool:
        """Handle mouse press events.

        Args:
            event: Mouse event

        Returns:
            True if event was handled
        """
        self._state.update_modifiers(event.modifiers())
        self._state.mouse_position = event.pos()

        button = MouseButton(event.button())
        self._state.mouse_pressed.add(button)
        self._state.mouse_drag_start = event.pos()

        # Check for node click
        if button == MouseButton.LEFT:
            node = self._raycast_find_node(event.pos())
            if node:
                self._handle_node_click(node, False)
                return True
            else:
                # Click on background clears selection
                self._clear_selection()
        
        # Always consume Right click to ensure we get drag events
        elif button == MouseButton.RIGHT:
            return True

        return False

    def mouse_release_event(self, event: QMouseEvent) -> bool:
        """Handle mouse release events.

        Args:
            event: Mouse event

        Returns:
            True if event was handled
        """
        self._state.update_modifiers(event.modifiers())
        self._state.mouse_position = event.pos()

        button = MouseButton(event.button())
        if button in self._state.mouse_pressed:
            self._state.mouse_pressed.remove(button)

        # Check for click (not drag)
        if button == MouseButton.LEFT:
            if not self._state.is_dragging:
                # Already handled in press event
                pass
            self._state.is_dragging = False

        # Reset drag state when any button is released
        if not any(self._state.mouse_pressed):
            self._state.is_dragging = False

        return False

    def mouse_move_event(self, event: QMouseEvent) -> bool:
        """Handle mouse move events.

        Args:
            event: Mouse event

        Returns:
            True if event was handled
        """
        self._state.update_modifiers(event.modifiers())

        dx = event.pos().x() - self._state.last_mouse_position.x()
        dy = event.pos().y() - self._state.last_mouse_position.y()

        # Check for drag threshold
        if MouseButton.LEFT in self._state.mouse_pressed:
            dist = (event.pos() - self._state.mouse_drag_start).manhattanLength()
            if dist > self._state.drag_threshold:
                self._state.is_dragging = True

        # Handle camera rotation/pan based on button
        if MouseButton.LEFT in self._state.mouse_pressed and self._state.is_dragging:
            if self._state.shift_pressed:
                # Pan camera (macOS alternative for trackpad users)
                if self._camera.mode == CameraMode.ORBIT:
                    w, h = self._renderer.width(), self._renderer.height()
                    self._camera.orbit_pan(dx, dy, w, h)
                return True
            elif not self._state.control_pressed:
                # Rotate camera (Orbit mode only)
                if self._camera.mode == CameraMode.ORBIT:
                    self._camera.orbit_rotate(dx, dy)
                elif self._camera.mode == CameraMode.FLY:
                    # Allow Left Drag to look in Fly Mode (like Orbit rotate)
                    self._camera.fly_look(dx, dy)
                return True
            else:
                # Control + Left Drag (fallback for Right Drag)
                if self._camera.mode == CameraMode.FLY:
                    self._camera.fly_look(dx, dy)
                    return True

        elif MouseButton.MIDDLE in self._state.mouse_pressed:
            # Pan camera
            if self._camera.mode == CameraMode.ORBIT:
                w, h = self._renderer.width(), self._renderer.height()
                self._camera.orbit_pan(dx, dy, w, h)
            return True

        elif MouseButton.RIGHT in self._state.mouse_pressed:
            # Pan camera (right-drag pan)
            if self._camera.mode == CameraMode.ORBIT:
                w, h = self._renderer.width(), self._renderer.height()
                self._camera.orbit_pan(dx, dy, w, h)
            elif self._camera.mode == CameraMode.FLY:
                # Look around
                self._camera.fly_look(dx, dy)
            return True

        # Handle hover detection (only when not dragging)
        if not self._state.is_dragging and not any(self._state.mouse_pressed):
            self._schedule_hover_update(event.pos())

        self._state.last_mouse_position = event.pos()
        return False

    def _schedule_hover_update(self, pos: QPoint) -> None:
        """Schedule hover update with debouncing.

        Args:
            pos: Mouse position
        """
        self._last_hover_pos = pos
        self._hover_debounce_timer.start(75)  # 75ms debounce

    def _update_hover(self) -> None:
        """Update hovered node after debounce."""
        if self._last_hover_pos is None:
            return

        node = self._raycast_find_node(self._last_hover_pos)

        # Emit signal if hover changed
        if node != self._hovered_node:
            self._hovered_node = node

            # Update tooltip via renderer's tooltip reference
            tooltip = getattr(self._renderer, '_tooltip', None)
            if tooltip:
                if node:
                    tooltip.show_for_node(
                        node,
                        self._last_hover_pos.x(),
                        self._last_hover_pos.y()
                    )
                else:
                    tooltip.hide_tooltip()

    def wheel_event(self, event: QWheelEvent) -> bool:
        """Handle mouse wheel events.

        Args:
            event: Wheel event

        Returns:
            True if event was handled
        """
        self._state.update_modifiers(event.modifiers())

        # Get scroll delta
        delta = event.angleDelta().y()
        scroll_amount = delta / 120.0  # Normalize to standard wheel click

        # Zoom (Orbit mode only)
        if self._camera.mode == CameraMode.ORBIT:
            self._camera.orbit_zoom(scroll_amount)

        return True

    def mouse_double_click_event(self, event: QMouseEvent) -> bool:
        """Handle mouse double-click events.

        Args:
            event: Mouse event

        Returns:
            True if event was handled
        """
        self._state.update_modifiers(event.modifiers())
        self._state.mouse_position = event.pos()

        if event.button() == Qt.MouseButton.LeftButton:
            node = self._raycast_find_node(event.pos())
            if node:
                # Double-click on directory or file
                self._handle_node_click(node, True)
                return True

        return False

    # Keyboard event handlers
    
    def key_press_event(self, event: QKeyEvent) -> bool:
        """Handle key press events.

        Args:
            event: Key event

        Returns:
            True if event was handled
        """
        self._state.update_modifiers(event.modifiers())

        key = event.key()
        
        
        # Track keys for Fly Mode
        if key == Qt.Key.Key_W:
            self._keys_pressed.add(Qt.Key.Key_W)
        elif key == Qt.Key.Key_S:
            self._keys_pressed.add(Qt.Key.Key_S)
        elif key == Qt.Key.Key_A:
            self._keys_pressed.add(Qt.Key.Key_A)
        elif key == Qt.Key.Key_D:
            self._keys_pressed.add(Qt.Key.Key_D)
        elif key == Qt.Key.Key_Q:
            self._keys_pressed.add(Qt.Key.Key_Q)
        elif key == Qt.Key.Key_E:
            self._keys_pressed.add(Qt.Key.Key_E)

        return False

    def key_release_event(self, event: QKeyEvent) -> bool:
        """Handle key release events.

        Args:
            event: Key event

        Returns:
            True if event was handled
        """
        key = event.key()
        
        if key in self._keys_pressed:
            self._keys_pressed.remove(key)
            
        return False
        
    def update(self) -> None:
        """Process continuous input (called every frame)."""
        if self._camera.mode != CameraMode.FLY:
            return
            
        dx = 0.0
        dy = 0.0
        dz = 0.0
        
        if Qt.Key.Key_W in self._keys_pressed:
            dz += 1.0
        if Qt.Key.Key_S in self._keys_pressed:
            dz -= 1.0
        if Qt.Key.Key_A in self._keys_pressed:
            dx -= 1.0
        if Qt.Key.Key_D in self._keys_pressed:
            dx += 1.0
        if Qt.Key.Key_Q in self._keys_pressed:
            dy -= 1.0
        if Qt.Key.Key_E in self._keys_pressed:
            dy += 1.0
            
        # Sprint with Shift
        speed_mult = 2.0 if self._state.shift_pressed else 1.0
        
        if dx != 0 or dy != 0 or dz != 0:
            # Calculate full movement vector
            move_vec = self._camera.get_fly_move_vector(dx, dy, dz, speed_multiplier=speed_mult)
            current_pos = self._camera.state.position

            # Try moving all axes; allow sliding by testing axes independently
            target_pos = current_pos + move_vec.astype(np.float32)
            if not self._renderer.check_collision(target_pos):
                self._camera.state.position = target_pos
            else:
                # Try X component
                move_x = np.array([move_vec[0], 0, 0], dtype=np.float32)
                if not self._renderer.check_collision(current_pos + move_x):
                    self._camera.state.position += move_x
                    current_pos = self._camera.state.position

                # Try Z component
                move_z = np.array([0, 0, move_vec[2]], dtype=np.float32)
                if not self._renderer.check_collision(current_pos + move_z):
                    self._camera.state.position += move_z
                    current_pos = self._camera.state.position

                # Try Y component
                move_y = np.array([0, move_vec[1], 0], dtype=np.float32)
                if not self._renderer.check_collision(current_pos + move_y):
                    self._camera.state.position += move_y

        # Clamp camera above ground regardless of input (handles initial state / gravity)
        GROUND_Y = -0.5
        CAMERA_RADIUS = 0.5
        min_y = GROUND_Y + CAMERA_RADIUS
        if self._camera.state.position[1] < min_y:
            pos = self._camera.state.position.copy()
            pos[1] = min_y
            self._camera.state.position = pos

    # Private methods

    def _raycast_find_node(self, pos: QPoint) -> Node | None:
        """Find node at screen position using raycasting.

        Supports both legacy renderer (with built-in raycast_find_node method)
        and modern renderer (using PickingSystem).

        Args:
            pos: Screen position

        Returns:
            Node at position or None
        """
        # Try legacy renderer's built-in method first
        if hasattr(self._renderer, 'raycast_find_node'):
            return self._renderer.raycast_find_node(pos.x(), pos.y(), self._nodes, self._positions)

        # Try modern renderer's picking system
        if hasattr(self._renderer, 'picking_system'):
            picking_system = self._renderer.picking_system
            picking_system.set_scene_data(self._nodes, self._positions)

            # Get matrices from camera
            view_matrix = self._camera.view_matrix
            aspect = self._renderer.width() / max(1, self._renderer.height())
            proj_matrix = self._camera.projection_matrix(aspect)

            return picking_system.pick(
                pos.x(), pos.y(),
                self._renderer.width(),
                self._renderer.height(),
                view_matrix,
                proj_matrix
            )

        return None

    def _handle_node_click(self, node: Node, is_double_click: bool) -> None:
        """Handle node click event.

        Args:
            node: Clicked node
            is_double_click: Whether this is a double-click
        """
        self._focused_node = node

        # Handle selection
        node_id = id(node)
        if self._state.control_pressed:
            # Toggle selection with Ctrl
            if node_id in self._selected_nodes:
                self._selected_nodes.remove(node_id)
            else:
                self._selected_nodes.add(node_id)
        elif self._state.shift_pressed:
            # Add to selection with Shift
            self._selected_nodes.add(node_id)
        else:
            # Replace selection
            self._selected_nodes.clear()
            self._selected_nodes.add(node_id)

        # Update visual selection
        self._update_selection_visual()

        # Handle double-click camera focus for directories
        if is_double_click and node.is_directory:
            self._focus_on_node(node)

        # Notify callbacks (controller handles both directory and file double-click)
        if self._on_node_clicked:
            self._on_node_clicked(node, is_double_click)

    def _focus_on_node(self, node: Node) -> None:
        """Focus camera on a node.

        Args:
            node: Node to focus on
        """
        if hasattr(self._renderer, 'snap_camera_to_node'):
            self._renderer.snap_camera_to_node(id(node), distance=5.0)

        if self._on_node_focused:
            self._on_node_focused(node)

    def _clear_selection(self) -> None:
        """Clear all selection."""
        self._selected_nodes.clear()
        self._focused_node = None
        self._update_selection_visual()

    def _update_selection_visual(self) -> None:
        """Update visual selection state."""
        selected_paths = {str(n.path) for n in [node for node in self._nodes.values() if id(node) in self._selected_nodes]}

        if hasattr(self._renderer, 'set_selection'):
            self._renderer.set_selection(selected_paths, self._nodes)

        if self._on_selection_changed:
            selected_nodes = [node for node in self._nodes.values() if id(node) in self._selected_nodes]
            self._on_selection_changed(set(selected_nodes))

    def _reset_view(self) -> None:
        """Reset camera to default view."""
        import numpy as np

        self._camera._state.position = np.array([10.0, 10.0, 10.0], dtype=np.float32)
        self._camera._state.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._camera._update_orbit_from_position()
        self._camera.set_mode(CameraMode.ORBIT)
