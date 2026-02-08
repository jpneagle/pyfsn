"""Texture management for ModernGL.

This module provides the TextureManager class for managing 2D textures,
texture caching, and texture cleanup for ModernGL rendering.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class TextureInfo:
    """Information about a managed texture."""
    texture: object  # ModernGL texture object
    width: int  # Texture width
    height: int  # Texture height
    components: int  # Number of color components (1-4)
    size: int  # Texture size in bytes


class TextureLoadError(Exception):
    """Raised when texture loading fails."""

    def __init__(self, path: str, message: str) -> None:
        """Initialize the error.

        Args:
            path: Path to texture file
            message: Error message
        """
        self.path = path
        self.message = message
        super().__init__(f"Failed to load texture '{path}': {message}")


class TextureManager:
    """Manage 2D textures for ModernGL rendering.

    This class provides methods for creating textures from files,
    numpy arrays, or raw data. It caches textures to avoid redundant
    loading and supports automatic cleanup.

    Ideal for managing label textures, icons, and other 2D assets
    in the file system visualization.

    Example:
        ```python
        import moderngl
        from pyfsn.view.texture_manager import TextureManager

        ctx = moderngl.create_context()
        manager = TextureManager(ctx)

        # Load texture from file
        texture = manager.load_texture("textures/icon.png", name="icon")

        # Create texture from numpy array
        data = np.zeros((256, 256, 4), dtype=np.uint8)
        texture = manager.create_texture(data, name="label_texture")

        # Get cached texture
        texture = manager.get_texture("icon")
        ```
    """

    def __init__(self, ctx) -> None:
        """Initialize the texture manager.

        Args:
            ctx: ModernGL context
        """
        self._ctx = ctx
        self._textures: dict[str, TextureInfo] = {}

        # Default texture parameters
        self._default_filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._default_wrap = (moderngl.CLAMP_TO_EDGE, moderngl.CLAMP_TO_EDGE)

    def create_texture(
        self,
        data: np.ndarray,
        name: Optional[str] = None,
        components: int = 4,
        filter: Optional[Tuple[int, int]] = None,
        wrap: Optional[Tuple[int, int]] = None,
        mipmap: bool = False
    ) -> object:
        """Create a 2D texture from numpy array data.

        Args:
            data: Image data as numpy array (height, width, components)
            name: Optional name for texture tracking
            components: Number of color components (1=grayscale, 2=GA, 3=RGB, 4=RGBA)
            filter: Optional (min_filter, mag_filter) tuple
            wrap: Optional (wrap_s, wrap_t) tuple
            mipmap: Whether to generate mipmaps

        Returns:
            ModernGL texture object

        Raises:
            ValueError: If data is not a valid numpy array
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Texture data must be a numpy array")

        if data.ndim != 3:
            raise ValueError(f"Texture data must be 3D array (H, W, C), got {data.ndim}D")

        height, width = data.shape[:2]

        # Convert data to uint8 if needed
        if data.dtype != np.uint8:
            data = np.clip(data * 255, 0, 255).astype(np.uint8)

        # Create texture
        texture = self._ctx.texture(
            (width, height),
            components,
            data.tobytes()
        )

        # Set texture parameters
        if filter is not None:
            texture.filter = filter
        else:
            texture.filter = self._default_filter

        if wrap is not None:
            texture.wrap = wrap
        else:
            texture.wrap = self._default_wrap

        # Generate mipmaps if requested
        if mipmap:
            texture.build_mipmaps()

        # Track texture
        texture_info = TextureInfo(
            texture=texture,
            width=width,
            height=height,
            components=components,
            size=data.nbytes
        )

        if name is not None:
            self._textures[name] = texture_info
            logger.debug(f"Created texture '{name}': {width}x{height}, {components} components")
        else:
            logger.debug(f"Created anonymous texture: {width}x{height}, {components} components")

        return texture

    def load_texture(
        self,
        path: str,
        name: Optional[str] = None,
        components: Optional[int] = None,
        filter: Optional[Tuple[int, int]] = None,
        wrap: Optional[Tuple[int, int]] = None,
        mipmap: bool = False,
        cache: bool = True
    ) -> object:
        """Load a 2D texture from file.

        Supports common image formats (PNG, JPEG, etc.) via PIL/Pillow.

        Args:
            path: Path to texture file
            name: Optional name for texture tracking (defaults to filename)
            components: Optional number of components (auto-detected if None)
            filter: Optional (min_filter, mag_filter) tuple
            wrap: Optional (wrap_s, wrap_t) tuple
            mipmap: Whether to generate mipmaps
            cache: Whether to cache the texture

        Returns:
            ModernGL texture object

        Raises:
            TextureLoadError: If loading fails
        """
        # Use filename as default name
        if name is None:
            name = Path(path).stem

        # Check cache first
        if cache and name in self._textures:
            logger.debug(f"Retrieved cached texture: {name}")
            return self._textures[name].texture

        # Load image
        try:
            image = Image.open(path)
        except IOError as e:
            raise TextureLoadError(path, str(e)) from e

        # Convert to RGB/RGBA if needed
        if components is None:
            # Auto-detect based on image mode
            mode = image.mode
            if mode == 'L':
                components = 1
            elif mode == 'LA':
                components = 2
            elif mode == 'RGB':
                components = 3
            elif mode == 'RGBA':
                components = 4
            else:
                # Convert to RGBA
                image = image.convert('RGBA')
                components = 4
        else:
            # Convert to requested format
            if components == 1:
                image = image.convert('L')
            elif components == 2:
                image = image.convert('LA')
            elif components == 3:
                image = image.convert('RGB')
            elif components == 4:
                image = image.convert('RGBA')

        # Convert to numpy array
        data = np.array(image, dtype=np.uint8)

        # Create texture
        texture = self.create_texture(
            data,
            name=name if cache else None,
            components=components,
            filter=filter,
            wrap=wrap,
            mipmap=mipmap
        )

        logger.debug(f"Loaded texture from '{path}': {image.width}x{image.height}")

        return texture

    def create_texture_from_bytes(
        self,
        data: bytes,
        size: Tuple[int, int],
        name: Optional[str] = None,
        components: int = 4,
        filter: Optional[Tuple[int, int]] = None,
        wrap: Optional[Tuple[int, int]] = None,
        mipmap: bool = False
    ) -> object:
        """Create a texture from raw bytes.

        Args:
            data: Raw image data as bytes
            size: Texture size as (width, height) tuple
            name: Optional name for texture tracking
            components: Number of color components
            filter: Optional (min_filter, mag_filter) tuple
            wrap: Optional (wrap_s, wrap_t) tuple
            mipmap: Whether to generate mipmaps

        Returns:
            ModernGL texture object
        """
        # Create texture directly from bytes
        texture = self._ctx.texture(size, components, data)

        # Set texture parameters
        if filter is not None:
            texture.filter = filter
        else:
            texture.filter = self._default_filter

        if wrap is not None:
            texture.wrap = wrap
        else:
            texture.wrap = self._default_wrap

        # Generate mipmaps if requested
        if mipmap:
            texture.build_mipmaps()

        # Track texture
        width, height = size
        texture_info = TextureInfo(
            texture=texture,
            width=width,
            height=height,
            components=components,
            size=len(data)
        )

        if name is not None:
            self._textures[name] = texture_info
            logger.debug(f"Created texture from bytes '{name}': {width}x{height}, {components} components")

        return texture

    def create_text_texture(
        self,
        text: str,
        font_size: int = 32,
        font_name: str = "Arial",
        text_color: Tuple[int, int, int, int] = (255, 255, 255, 255),
        background_color: Tuple[int, int, int, int] = (0, 0, 0, 0),
        padding: int = 4,
        name: Optional[str] = None
    ) -> object:
        """Create a texture from text using PIL.

        This is a convenience method for creating label textures
        for directory names and other text labels.

        Args:
            text: Text to render
            font_size: Font size in pixels
            font_name: Font name
            text_color: RGBA color for text
            background_color: RGBA color for background
            padding: Padding around text in pixels
            name: Optional name for texture tracking

        Returns:
            ModernGL texture object
        """
        try:
            from PIL import ImageDraw, ImageFont
        except ImportError:
            raise ImportError("Pillow is required for text texture creation")

        # Load font
        try:
            font = ImageFont.truetype(font_name, font_size)
        except IOError:
            # Fallback to default font
            font = ImageFont.load_default()

        # Calculate text size
        try:
            # Pillow 10.0.0+
            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            ascent = bbox[3] - bbox[1]
        except AttributeError:
            # Older Pillow versions
            text_width, text_height = font.getsize(text)
            ascent = text_height

        # Calculate image size with padding
        img_width = text_width + padding * 2
        img_height = text_height + padding * 2

        # Create image
        image = Image.new('RGBA', (img_width, img_height), background_color)
        draw = ImageDraw.Draw(image)

        # Draw text
        draw.text((padding, padding), text, font=font, fill=text_color)

        # Convert to numpy array and create texture
        data = np.array(image, dtype=np.uint8)

        # Use text as default name
        if name is None:
            # Create a safe name from text
            safe_text = "".join(c for c in text if c.isalnum() or c in ('-', '_')).rstrip()
            name = f"text_{safe_text}" if safe_text else f"text_{hash(text) & 0xffffffff:08x}"

        return self.create_texture(data, name=name)

    def update_texture(
        self,
        texture: object,
        data: np.ndarray,
        offset: Tuple[int, int] = (0, 0),
        size: Optional[Tuple[int, int]] = None
    ) -> None:
        """Update texture data.

        Args:
            texture: ModernGL texture to update
            data: New image data as numpy array
            offset: (x, y) offset for partial update
            size: Optional (width, height) of region to update

        Raises:
            ValueError: If data is not a valid numpy array
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Texture data must be a numpy array")

        # Convert data to uint8 if needed
        if data.dtype != np.uint8:
            data = np.clip(data * 255, 0, 255).astype(np.uint8)

        # Determine size from data if not specified
        if size is None:
            size = (data.shape[1], data.shape[0])

        # Update texture
        texture.write(data.tobytes(), offset, size)
        logger.debug(f"Updated texture at {offset}: {size[0]}x{size[1]}")

    def update_texture_named(
        self,
        name: str,
        data: np.ndarray,
        offset: Tuple[int, int] = (0, 0),
        size: Optional[Tuple[int, int]] = None
    ) -> bool:
        """Update a named texture.

        Args:
            name: Texture name
            data: New image data as numpy array
            offset: (x, y) offset for partial update
            size: Optional (width, height) of region to update

        Returns:
            True if texture was found and updated, False otherwise
        """
        if name not in self._textures:
            logger.warning(f"Texture '{name}' not found for update")
            return False

        texture_info = self._textures[name]
        self.update_texture(texture_info.texture, data, offset, size)
        return True

    def get_texture(self, name: str) -> Optional[object]:
        """Get a named texture.

        Args:
            name: Texture name

        Returns:
            ModernGL texture object or None if not found
        """
        texture_info = self._textures.get(name)
        return texture_info.texture if texture_info else None

    def has_texture(self, name: str) -> bool:
        """Check if a texture exists.

        Args:
            name: Texture name

        Returns:
            True if texture exists, False otherwise
        """
        return name in self._textures

    def release_texture(self, name: str) -> bool:
        """Release a named texture.

        Args:
            name: Texture name

        Returns:
            True if texture was found and released, False otherwise
        """
        if name not in self._textures:
            logger.warning(f"Texture '{name}' not found for release")
            return False

        texture_info = self._textures.pop(name)
        texture_info.texture.release()
        logger.debug(f"Released texture: {name}")
        return True

    def clear(self) -> None:
        """Release all managed textures."""
        for name in list(self._textures.keys()):
            self.release_texture(name)
        logger.debug("Released all textures")

    def get_texture_info(self, name: str) -> Optional[TextureInfo]:
        """Get information about a managed texture.

        Args:
            name: Texture name

        Returns:
            TextureInfo or None if not found
        """
        return self._textures.get(name)

    @property
    def texture_count(self) -> int:
        """Get number of managed textures."""
        return len(self._textures)

    @property
    def total_size(self) -> int:
        """Get total size of all managed textures in bytes."""
        return sum(info.size for info in self._textures.values())

    def set_default_filter(self, min_filter: int, mag_filter: int) -> None:
        """Set default filter for new textures.

        Args:
            min_filter: Minification filter (e.g., moderngl.LINEAR, moderngl.NEAREST)
            mag_filter: Magnification filter
        """
        self._default_filter = (min_filter, mag_filter)

    def set_default_wrap(self, wrap_s: int, wrap_t: int) -> None:
        """Set default wrap mode for new textures.

        Args:
            wrap_s: Horizontal wrap mode
            wrap_t: Vertical wrap mode
        """
        self._default_wrap = (wrap_s, wrap_t)

    def __del__(self) -> None:
        """Cleanup when manager is destroyed."""
        try:
            self.clear()
        except Exception:
            pass  # Ignore errors during cleanup


# Import moderngl for filter/wrap constants
try:
    import moderngl
except ImportError:
    # Create dummy constants if moderngl not available
    class _DummyConst:
        def __init__(self, val: int):
            self.value = val
    moderngl = type('obj', (object,), {
        'LINEAR': _DummyConst(9729),
        'NEAREST': _DummyConst(9728),
        'CLAMP_TO_EDGE': _DummyConst(33071),
        'REPEAT': _DummyConst(10497),
        'MIRRORED_REPEAT': _DummyConst(33648),
    })
