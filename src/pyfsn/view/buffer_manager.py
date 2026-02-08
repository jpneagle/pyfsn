"""Buffer management for ModernGL.

This module provides the BufferManager class for managing VBOs, VAOs,
and other GPU buffers using ModernGL.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BufferInfo:
    """Information about a managed buffer."""
    buffer: object  # ModernGL buffer object
    size: int  # Buffer size in bytes
    usage: str  # Usage hint (static, dynamic, stream)


@dataclass
class VertexArrayInfo:
    """Information about a managed vertex array object."""
    vao: object  # ModernGL vertex array object
    program: object  # Associated shader program
    buffers: list[str]  # Names of associated buffers


class BufferManager:
    """Manage GPU buffers (VBO, VAO, EBO) for ModernGL rendering.

    This class provides methods for creating and managing vertex buffers,
    index buffers, and vertex array objects. It tracks all created buffers
    and ensures proper cleanup on destruction.

    Supports instanced rendering buffers for efficient batch rendering.

    Example:
        ```python
        import moderngl
        import numpy as np
        from pyfsn.view.buffer_manager import BufferManager

        ctx = moderngl.create_context()
        manager = BufferManager(ctx)

        # Create vertex buffer
        vertices = np.array([
            [-0.5, -0.5, 0.0],
            [ 0.5, -0.5, 0.0],
            [ 0.0,  0.5, 0.0],
        ], dtype=np.float32)
        vbo = manager.create_vertex_buffer(vertices, name="cube_vertices")

        # Create index buffer
        indices = np.array([0, 1, 2], dtype=np.uint32)
        ebo = manager.create_index_buffer(indices, name="cube_indices")

        # Create VAO
        vao = manager.create_vertex_array(program, vbo, ebo, name="cube_vao")
        ```
    """

    def __init__(self, ctx) -> None:
        """Initialize the buffer manager.

        Args:
            ctx: ModernGL context
        """
        self._ctx = ctx
        self._buffers: dict[str, BufferInfo] = {}
        self._vaos: dict[str, VertexArrayInfo] = {}

    def create_vertex_buffer(
        self,
        data: np.ndarray,
        name: Optional[str] = None,
        dynamic: bool = False
    ) -> object:
        """Create a vertex buffer object (VBO).

        Args:
            data: Vertex data as numpy array
            name: Optional name for buffer tracking
            dynamic: True if data will change frequently

        Returns:
            ModernGL buffer object

        Raises:
            ValueError: If data is not a valid numpy array
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Vertex data must be a numpy array")

        # Determine buffer usage
        usage = "dynamic" if dynamic else "static"

        # Create buffer
        buffer = self._ctx.buffer(data.tobytes())

        # Track buffer
        buffer_info = BufferInfo(
            buffer=buffer,
            size=data.nbytes,
            usage=usage
        )

        if name is not None:
            self._buffers[name] = buffer_info
            logger.debug(f"Created vertex buffer '{name}': {data.nbytes} bytes")
        else:
            logger.debug(f"Created anonymous vertex buffer: {data.nbytes} bytes")

        return buffer

    def create_index_buffer(
        self,
        data: np.ndarray,
        name: Optional[str] = None,
        dynamic: bool = False
    ) -> object:
        """Create an element buffer object (EBO/index buffer).

        Args:
            data: Index data as numpy array (typically uint32)
            name: Optional name for buffer tracking
            dynamic: True if data will change frequently

        Returns:
            ModernGL buffer object

        Raises:
            ValueError: If data is not a valid numpy array
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Index data must be a numpy array")

        # Ensure data type is compatible
        if data.dtype not in [np.uint8, np.uint16, np.uint32]:
            logger.warning(f"Index buffer data type {data.dtype} may not be supported")

        # Determine buffer usage
        usage = "dynamic" if dynamic else "static"

        # Create buffer
        buffer = self._ctx.buffer(data.tobytes())

        # Track buffer
        buffer_info = BufferInfo(
            buffer=buffer,
            size=data.nbytes,
            usage=usage
        )

        if name is not None:
            self._buffers[name] = buffer_info
            logger.debug(f"Created index buffer '{name}': {data.nbytes} bytes")
        else:
            logger.debug(f"Created anonymous index buffer: {data.nbytes} bytes")

        return buffer

    def create_instance_buffer(
        self,
        data: np.ndarray,
        name: Optional[str] = None,
        dynamic: bool = True
    ) -> object:
        """Create a buffer for instanced rendering attributes.

        This is a convenience method for creating buffers that will be
        used for per-instance data (position, scale, color, etc.) in
        instanced rendering.

        Args:
            data: Instance data as numpy array
            name: Optional name for buffer tracking
            dynamic: True if data will change frequently (default: True)

        Returns:
            ModernGL buffer object
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Instance data must be a numpy array")

        # Instance buffers are usually dynamic
        buffer = self.create_vertex_buffer(data, name=name, dynamic=dynamic)

        if name is not None:
            logger.debug(f"Created instance buffer '{name}': {len(data)} instances")

        return buffer

    def create_vertex_array(
        self,
        program: object,
        buffers: dict[str, tuple[object, str, list[str]]],
        index_buffer: Optional[object] = None,
        name: Optional[str] = None
    ) -> object:
        """Create a vertex array object (VAO) with specified attribute layout.

        Args:
            program: ModernGL shader program
            buffers: Dictionary mapping attribute names to (buffer, format, attribs) tuples
                - buffer: ModernGL buffer object
                - format: Format string (e.g., "3f", "4f", "3f 3f")
                - attribs: List of attribute names in program
            index_buffer: Optional index buffer for indexed rendering
            name: Optional name for VAO tracking

        Returns:
            ModernGL vertex array object

        Example:
            ```python
            vao = manager.create_vertex_array(
                program=program,
                buffers={
                    "in_position": (vbo, "3f", ["in_position"]),
                    "in_normal": (vbo, "3f 12x", ["in_normal"]),
                    "in_instance_position": (instance_vbo, "3f", ["in_instance_position"]),
                    "in_instance_scale": (instance_vbo, "3f 12x", ["in_instance_scale"]),
                    "in_instance_color": (instance_vbo, "4f", ["in_instance_color"]),
                },
                index_buffer=ebo,
                name="cube_vao"
            )
            ```
        """
        # Build VAO content list for ModernGL
        vao_content = []
        for attr_name, (buffer, format_str, attribs) in buffers.items():
            vao_content.append((buffer, format_str, *attribs))

        # Create VAO
        if index_buffer is not None:
            vao = self._ctx.vertex_array(program, vao_content, index_buffer)
        else:
            vao = self._ctx.vertex_array(program, vao_content)

        # Track VAO and associated buffers
        buffer_names = [name for name in self._buffers
                       if any(b == self._buffers[name].buffer for _, b, *_ in vao_content)]

        vao_info = VertexArrayInfo(
            vao=vao,
            program=program,
            buffers=buffer_names
        )

        if name is not None:
            self._vaos[name] = vao_info
            logger.debug(f"Created VAO '{name}' with {len(buffer_names)} buffers")
        else:
            logger.debug(f"Created anonymous VAO with {len(buffer_names)} buffers")

        return vao

    def create_vertex_array_simple(
        self,
        program: object,
        vertex_buffer: object,
        format_str: str,
        attributes: list[str],
        index_buffer: Optional[object] = None,
        instance_buffer: Optional[tuple[object, str, list[str]]] = None,
        name: Optional[str] = None
    ) -> object:
        """Create a VAO with simple single-buffer configuration.

        Convenience method for common case of one vertex buffer
        (and optionally one instance buffer).

        Args:
            program: ModernGL shader program
            vertex_buffer: Vertex buffer object
            format_str: Format string for vertex attributes
            attributes: List of attribute names
            index_buffer: Optional index buffer
            instance_buffer: Optional (buffer, format, attributes) for instancing
            name: Optional name for VAO tracking

        Returns:
            ModernGL vertex array object
        """
        # Build buffer dict
        buffers = {
            attributes[0]: (vertex_buffer, format_str, attributes)
        }

        # Add instance buffer if provided
        if instance_buffer is not None:
            ibuf, iformat, iattrs = instance_buffer
            buffers[iattrs[0]] = (ibuf, iformat, iattrs)

        return self.create_vertex_array(program, buffers, index_buffer, name)

    def update_buffer(
        self,
        buffer: object,
        data: np.ndarray,
        offset: int = 0
    ) -> None:
        """Update buffer data.

        Args:
            buffer: ModernGL buffer to update
            data: New data as numpy array
            offset: Byte offset for partial update

        Raises:
            ValueError: If data is not a valid numpy array
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array")

        buffer.write(data.tobytes(), offset)
        logger.debug(f"Updated buffer: {data.nbytes} bytes at offset {offset}")

    def update_buffer_named(self, name: str, data: np.ndarray, offset: int = 0) -> bool:
        """Update a named buffer.

        Args:
            name: Buffer name
            data: New data as numpy array
            offset: Byte offset for partial update

        Returns:
            True if buffer was found and updated, False otherwise
        """
        if name not in self._buffers:
            logger.warning(f"Buffer '{name}' not found for update")
            return False

        buffer_info = self._buffers[name]
        self.update_buffer(buffer_info.buffer, data, offset)
        buffer_info.size = max(buffer_info.size, offset + data.nbytes)
        return True

    def get_buffer(self, name: str) -> Optional[object]:
        """Get a named buffer.

        Args:
            name: Buffer name

        Returns:
            ModernGL buffer object or None if not found
        """
        buffer_info = self._buffers.get(name)
        return buffer_info.buffer if buffer_info else None

    def get_vao(self, name: str) -> Optional[object]:
        """Get a named VAO.

        Args:
            name: VAO name

        Returns:
            ModernGL VAO object or None if not found
        """
        vao_info = self._vaos.get(name)
        return vao_info.vao if vao_info else None

    def has_buffer(self, name: str) -> bool:
        """Check if a buffer exists.

        Args:
            name: Buffer name

        Returns:
            True if buffer exists, False otherwise
        """
        return name in self._buffers

    def has_vao(self, name: str) -> bool:
        """Check if a VAO exists.

        Args:
            name: VAO name

        Returns:
            True if VAO exists, False otherwise
        """
        return name in self._vaos

    def release_buffer(self, name: str) -> bool:
        """Release a named buffer.

        Args:
            name: Buffer name

        Returns:
            True if buffer was found and released, False otherwise
        """
        if name not in self._buffers:
            logger.warning(f"Buffer '{name}' not found for release")
            return False

        buffer_info = self._buffers.pop(name)
        buffer_info.buffer.release()
        logger.debug(f"Released buffer: {name}")
        return True

    def release_vao(self, name: str) -> bool:
        """Release a named VAO.

        Args:
            name: VAO name

        Returns:
            True if VAO was found and released, False otherwise
        """
        if name not in self._vaos:
            logger.warning(f"VAO '{name}' not found for release")
            return False

        vao_info = self._vaos.pop(name)
        vao_info.vao.release()
        logger.debug(f"Released VAO: {name}")
        return True

    def clear(self) -> None:
        """Release all managed buffers and VAOs."""
        # Release VAOs first (they reference buffers)
        for name in list(self._vaos.keys()):
            self.release_vao(name)

        # Release buffers
        for name in list(self._buffers.keys()):
            self.release_buffer(name)

        logger.debug("Released all buffers and VAOs")

    def get_buffer_info(self, name: str) -> Optional[BufferInfo]:
        """Get information about a managed buffer.

        Args:
            name: Buffer name

        Returns:
            BufferInfo or None if not found
        """
        return self._buffers.get(name)

    def get_vao_info(self, name: str) -> Optional[VertexArrayInfo]:
        """Get information about a managed VAO.

        Args:
            name: VAO name

        Returns:
            VertexArrayInfo or None if not found
        """
        return self._vaos.get(name)

    @property
    def buffer_count(self) -> int:
        """Get number of managed buffers."""
        return len(self._buffers)

    @property
    def vao_count(self) -> int:
        """Get number of managed VAOs."""
        return len(self._vaos)

    def __del__(self) -> None:
        """Cleanup when manager is destroyed."""
        try:
            self.clear()
        except Exception:
            pass  # Ignore errors during cleanup
