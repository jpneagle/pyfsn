"""Filter panel widget for advanced file system filtering.

Provides classic fsn-like filtering beyond name search including:
- Size range filtering
- Age range filtering
- Type filtering (files/directories/symlinks)
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QComboBox,
    QSpinBox,
    QCheckBox,
    QGroupBox,
    QScrollArea,
)
from PyQt6.QtCore import pyqtSignal


class FilterPanel(QWidget):
    """Advanced filtering panel for file system visualization.

    Provides classic fsn-like filtering beyond name search including:
    - Size range filtering
    - Age range filtering
    - Type filtering (files/directories/symlinks)
    """

    # Signals
    filter_changed = pyqtSignal(dict)  # Emits filter criteria dict
    filters_cleared = pyqtSignal()

    def __init__(self, parent=None) -> None:
        """Initialize filter panel."""
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the filter panel UI."""
        self.setFixedWidth(250)

        # Create scroll area for filter content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        # Container widget
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Title
        title = QLabel("<b>Filters</b>")
        title.setStyleSheet("font-size: 14px;")
        layout.addWidget(title)

        # Name filter section (existing search)
        name_group = QGroupBox("Name")
        name_layout = QVBoxLayout()
        self._name_filter = QLineEdit()
        self._name_filter.setPlaceholderText("Contains...")
        self._name_filter.textChanged.connect(self._on_filter_changed)
        name_layout.addWidget(self._name_filter)
        name_group.setLayout(name_layout)
        layout.addWidget(name_group)

        # Size filter section
        size_group = QGroupBox("Size (bytes)")
        size_layout = QVBoxLayout()

        # Min size
        min_size_layout = QHBoxLayout()
        min_size_layout.addWidget(QLabel("Min:"))
        self._min_size_input = QLineEdit()
        self._min_size_input.setPlaceholderText("0")
        self._min_size_input.textChanged.connect(self._on_filter_changed)
        min_size_layout.addWidget(self._min_size_input)
        size_layout.addLayout(min_size_layout)

        # Max size
        max_size_layout = QHBoxLayout()
        max_size_layout.addWidget(QLabel("Max:"))
        self._max_size_input = QLineEdit()
        self._max_size_input.setPlaceholderText("Unlimited")
        self._max_size_input.textChanged.connect(self._on_filter_changed)
        max_size_layout.addWidget(self._max_size_input)
        size_layout.addLayout(max_size_layout)

        # Human-friendly size presets
        presets_layout = QHBoxLayout()
        self._size_preset_combo = QComboBox()
        self._size_preset_combo.addItems([
            "Any size",
            "> 1 KB",
            "> 100 KB",
            "> 1 MB",
            "> 100 MB",
            "> 1 GB",
            "Custom"
        ])
        self._size_preset_combo.currentIndexChanged.connect(self._on_size_preset_changed)
        presets_layout.addWidget(self._size_preset_combo)
        size_layout.addLayout(presets_layout)

        size_group.setLayout(size_layout)
        layout.addWidget(size_group)

        # Age filter section
        age_group = QGroupBox("Age")
        age_layout = QVBoxLayout()

        # Age presets
        self._age_preset_combo = QComboBox()
        self._age_preset_combo.addItems([
            "Any time",
            "Last 24 hours",
            "Last 7 days",
            "Last 30 days",
            "Last 90 days",
            "Last year",
            "Custom"
        ])
        self._age_preset_combo.currentIndexChanged.connect(self._on_age_preset_changed)
        age_layout.addWidget(self._age_preset_combo)

        # Custom age inputs (days)
        custom_age_layout = QHBoxLayout()
        custom_age_layout.addWidget(QLabel("Days:"))
        self._min_days_input = QSpinBox()
        self._min_days_input.setRange(0, 36500)
        self._min_days_input.setValue(0)
        self._min_days_input.valueChanged.connect(self._on_filter_changed)
        custom_age_layout.addWidget(self._min_days_input)
        age_layout.addLayout(custom_age_layout)

        age_group.setLayout(age_layout)
        layout.addWidget(age_group)

        # Type filter section
        type_group = QGroupBox("Type")
        type_layout = QVBoxLayout()

        self._show_files_cb = QCheckBox("Files")
        self._show_files_cb.setChecked(True)
        self._show_files_cb.stateChanged.connect(self._on_filter_changed)
        type_layout.addWidget(self._show_files_cb)

        self._show_dirs_cb = QCheckBox("Directories")
        self._show_dirs_cb.setChecked(True)
        self._show_dirs_cb.stateChanged.connect(self._on_filter_changed)
        type_layout.addWidget(self._show_dirs_cb)

        self._show_symlinks_cb = QCheckBox("Symlinks")
        self._show_symlinks_cb.setChecked(True)
        self._show_symlinks_cb.stateChanged.connect(self._on_filter_changed)
        type_layout.addWidget(self._show_symlinks_cb)

        type_group.setLayout(type_layout)
        layout.addWidget(type_group)

        # Context preservation section
        context_group = QGroupBox("Context")
        context_layout = QVBoxLayout()

        self._include_ancestors_cb = QCheckBox("Include parent directories")
        self._include_ancestors_cb.setChecked(True)
        self._include_ancestors_cb.setToolTip(
            "When filtering, include all parent directories of matched nodes "
            "to preserve context and wire connections."
        )
        self._include_ancestors_cb.stateChanged.connect(self._on_filter_changed)
        context_layout.addWidget(self._include_ancestors_cb)

        context_group.setLayout(context_layout)
        layout.addWidget(context_group)

        # Actions
        actions_layout = QHBoxLayout()

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_filters)
        actions_layout.addWidget(clear_btn)

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._on_filter_changed)
        actions_layout.addWidget(apply_btn)

        layout.addLayout(actions_layout)

        # Add stretch to push content to top
        layout.addStretch()

        # Set scroll content
        scroll.setWidget(container)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

    def _on_size_preset_changed(self, index: int) -> None:
        """Handle size preset combo change.

        Args:
            index: Selected index
        """
        # Size presets in bytes
        size_presets = {
            0: (0, None),           # Any size
            1: (1024, None),        # > 1 KB
            2: (102400, None),      # > 100 KB
            3: (1048576, None),     # > 1 MB
            4: (104857600, None),   # > 100 MB
            5: (1073741824, None),  # > 1 GB
        }

        if index in size_presets:
            min_size, max_size = size_presets[index]
            self._min_size_input.setText(str(min_size) if min_size > 0 else "")
            self._max_size_input.setText(str(max_size) if max_size else "")

        self._on_filter_changed()

    def _on_age_preset_changed(self, index: int) -> None:
        """Handle age preset combo change.

        Args:
            index: Selected index
        """
        # Age presets in days
        age_presets = {
            0: 0,      # Any time
            1: 1,      # Last 24 hours
            2: 7,      # Last 7 days
            3: 30,     # Last 30 days
            4: 90,     # Last 90 days
            5: 365,    # Last year
        }

        if index in age_presets:
            self._min_days_input.setValue(age_presets[index])

        self._on_filter_changed()

    def _on_filter_changed(self) -> None:
        """Handle any filter change."""
        filters = self.get_filters()
        self.filter_changed.emit(filters)

    def _clear_filters(self) -> None:
        """Clear all filters."""
        self._name_filter.clear()
        self._min_size_input.clear()
        self._max_size_input.clear()
        self._size_preset_combo.setCurrentIndex(0)
        self._age_preset_combo.setCurrentIndex(0)
        self._min_days_input.setValue(0)
        self._show_files_cb.setChecked(True)
        self._show_dirs_cb.setChecked(True)
        self._show_symlinks_cb.setChecked(True)
        self._include_ancestors_cb.setChecked(True)
        self.filters_cleared.emit()

    def get_filters(self) -> dict:
        """Get current filter criteria.

        Returns:
            Dictionary with filter criteria
        """
        import time

        filters = {
            'name_contains': self._name_filter.text().strip(),
            'min_size': None,
            'max_size': None,
            'min_mtime': None,
            'show_files': self._show_files_cb.isChecked(),
            'show_dirs': self._show_dirs_cb.isChecked(),
            'show_symlinks': self._show_symlinks_cb.isChecked(),
            'include_ancestors': self._include_ancestors_cb.isChecked(),
        }

        # Parse size filters
        try:
            if self._min_size_input.text().strip():
                filters['min_size'] = int(self._min_size_input.text())
            if self._max_size_input.text().strip():
                filters['max_size'] = int(self._max_size_input.text())
        except ValueError:
            pass  # Invalid input, ignore

        # Calculate min mtime from days
        days = self._min_days_input.value()
        if days > 0:
            filters['min_mtime'] = time.time() - (days * 24 * 60 * 60)

        return filters

    def has_active_filters(self) -> bool:
        """Check if any filters are active.

        Returns:
            True if any filter is active
        """
        filters = self.get_filters()
        return (
            bool(filters['name_contains']) or
            filters['min_size'] is not None or
            filters['max_size'] is not None or
            filters['min_mtime'] is not None or
            not (filters['show_files'] and
                 filters['show_dirs'] and
                 filters['show_symlinks'])
        )
