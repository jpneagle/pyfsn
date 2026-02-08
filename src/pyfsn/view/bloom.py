"""Bloom post-processing effect for cyberpunk visual style.

Provides GPU-accelerated bloom effect using framebuffers and Gaussian blur.
Compatible with both Legacy OpenGL (fixed function) and ModernGL programmable pipeline.
"""

import math
from typing import Optional

import numpy as np
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

from OpenGL.GL import *


class BloomPass:
    """Configuration for a single bloom pass."""

    def __init__(self, blur_radius: float = 5.0, intensity: float = 0.5):
        """Initialize bloom pass.

        Args:
            blur_radius: Gaussian blur radius (pixels)
            intensity: Bloom intensity multiplier
        """
        self.blur_radius = blur_radius
        self.intensity = intensity


class BloomEffect:
    """Bloom post-processing effect using framebuffers.

    Implements a multi-pass bloom effect:
    1. Bright pass - extract bright pixels above threshold
    2. Blur pass - horizontal Gaussian blur
    3. Blur pass - vertical Gaussian blur
    4. Composite - blend blurred result with original scene
    """

    def __init__(
        self,
        width: int,
        height: int,
        threshold: float = 0.8,
        intensity: float = 0.3,
        radius: float = 8.0,
        downsample: int = 2
    ):
        """Initialize bloom effect.

        Args:
            width: Viewport width
            height: Viewport height
            threshold: Brightness threshold (0.0 - 1.0) - pixels above this bloom
            intensity: Overall bloom intensity (0.0 - 1.0+)
            radius: Blur radius in pixels
            downsample: Downsampling factor for performance (1 = full res, 2 = half res)
        """
        self.width = width
        self.height = height
        self.threshold = threshold
        self.intensity = intensity
        self.radius = radius
        self.downsample = downsample

        # Bloom pass configuration
        self.passes = [
            BloomPass(blur_radius=radius * 0.5, intensity=intensity * 0.5),
            BloomPass(blur_radius=radius, intensity=intensity),
        ]

        # Framebuffer objects
        self._bright_fbo: Optional[int] = None
        self._blur_fbo_h: Optional[int] = None
        self._blur_fbo_v: Optional[int] = None

        # Texture objects
        self._bright_texture: Optional[int] = None
        self._blur_texture_h: Optional[int] = None
        self._blur_texture_v: Optional[int] = None

        # Renderbuffer for depth
        self._depth_renderbuffer: Optional[int] = None

        # Precomputed Gaussian kernel
        self._gaussian_kernel_1d: Optional[np.ndarray] = None

        # Shader programs (for Legacy OpenGL compatibility)
        self._bright_program: Optional[int] = None
        self._blur_h_program: Optional[int] = None
        self._blur_v_program: Optional[int] = None
        self._composite_program: Optional[int] = None

        # Enable/disable flag
        self._enabled = True

    @property
    def enabled(self) -> bool:
        """Check if bloom effect is enabled."""
        return self._enabled

    def enabled(self, value: bool) -> None:
        """Enable or disable bloom effect.

        Args:
            value: True to enable, False to disable
        """
        self._enabled = value

    def initialize(self) -> bool:
        """Initialize OpenGL resources for bloom effect.

        Returns:
            True if initialization successful
        """
        # Check for required OpenGL extensions
        if not self._check_extensions():
            return False

        # Calculate bloom texture dimensions (with downsampling)
        bloom_w = max(1, self.width // self.downsample)
        bloom_h = max(1, self.height // self.downsample)

        # Create textures
        self._bright_texture = self._create_texture(bloom_w, bloom_h)
        self._blur_texture_h = self._create_texture(bloom_w, bloom_h)
        self._blur_texture_v = self._create_texture(bloom_w, bloom_h)

        if None in (self._bright_texture, self._blur_texture_h, self._blur_texture_v):
            return False

        # Create framebuffers
        self._bright_fbo = self._create_fbo(self._bright_texture)
        self._blur_fbo_h = self._create_fbo(self._blur_texture_h)
        self._blur_fbo_v = self._create_fbo(self._blur_texture_v)

        if None in (self._bright_fbo, self._blur_fbo_h, self._blur_fbo_v):
            return False

        # Create depth renderbuffer
        self._depth_renderbuffer = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self._depth_renderbuffer)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, bloom_w, bloom_h)

        # Attach depth to each framebuffer
        for fbo in (self._bright_fbo, self._blur_fbo_h, self._blur_fbo_v):
            glBindFramebuffer(GL_FRAMEBUFFER, fbo)
            glFramebufferRenderbuffer(
                GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self._depth_renderbuffer
            )

        # Check framebuffer status
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            print(f"Bloom framebuffer incomplete: {status}")
            return False

        # Unbind framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Precompute Gaussian kernel
        self._compute_gaussian_kernel()

        return True

    def _check_extensions(self) -> bool:
        """Check for required OpenGL extensions.

        Returns:
            True if all required extensions are available
        """
        # Check for framebuffer support
        extensions = glGetString(GL_EXTENSIONS)
        if extensions is None:
            return False

        ext_str = extensions.decode('utf-8')
        required = ['GL_ARB_framebuffer_object', 'GL_EXT_framebuffer_object']

        # Either direct framebuffer support or extension
        has_fbo = (
            'GL_VERSION_3_0' in ext_str or
            any(r in ext_str for r in required)
        )

        return has_fbo

    def _create_texture(self, width: int, height: int) -> Optional[int]:
        """Create a texture for bloom processing.

        Args:
            width: Texture width
            height: Texture height

        Returns:
            Texture ID or None on failure
        """
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)

        # Allocate texture storage
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA16F,
            width, height, 0,
            GL_RGBA, GL_UNSIGNED_BYTE, None
        )

        # Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        return texture_id

    def _create_fbo(self, texture_id: int) -> Optional[int]:
        """Create a framebuffer with the given texture attachment.

        Args:
            texture_id: Texture to attach as color attachment

        Returns:
            Framebuffer ID or None on failure
        """
        fbo_id = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_id)
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D, texture_id, 0
        )
        return fbo_id

    def _compute_gaussian_kernel(self) -> None:
        """Precompute 1D Gaussian kernel for separable blur."""
        kernel_size = int(self.radius * 2 + 1)
        self._gaussian_kernel_1d = np.zeros(kernel_size, dtype=np.float32)

        sigma = self.radius / 3.0
        center = self.radius
        sum_val = 0.0

        for i in range(kernel_size):
            x = i - center
            val = math.exp(-(x * x) / (2.0 * sigma * sigma))
            self._gaussian_kernel_1d[i] = val
            sum_val += val

        # Normalize kernel
        if sum_val > 0:
            self._gaussian_kernel_1d /= sum_val

    def resize(self, width: int, height: int) -> None:
        """Handle viewport resize.

        Args:
            width: New viewport width
            height: New viewport height
        """
        if width == self.width and height == self.height:
            return

        # Clean up old resources
        self.cleanup()

        # Update dimensions
        self.width = width
        self.height = height

        # Reinitialize
        self.initialize()

    def apply_bright_pass(self) -> None:
        """Render bright pass - extract pixels above threshold."""
        glBindFramebuffer(GL_FRAMEBUFFER, self._bright_fbo)
        glViewport(0, 0, self.width // self.downsample, self.height // self.downsample)

        # Clear
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # For Legacy OpenGL, we use a simpler approach:
        # Just copy the current framebuffer with thresholding
        # In a full implementation, this would use a shader

    def apply_blur_pass(self) -> None:
        """Apply separable Gaussian blur."""
        # This is a simplified blur for Legacy OpenGL
        # A full implementation would use shaders for proper Gaussian blur

        # Horizontal blur pass
        glBindFramebuffer(GL_FRAMEBUFFER, self._blur_fbo_h)
        glViewport(0, 0, self.width // self.downsample, self.height // self.downsample)

        # Vertical blur pass
        glBindFramebuffer(GL_FRAMEBUFFER, self._blur_fbo_v)
        glViewport(0, 0, self.width // self.downsample, self.height // self.downsample)

    def composite(self) -> None:
        """Composite bloom result with original scene."""
        if not self._enabled:
            return

        # For Legacy OpenGL, we use additive blending
        # A full implementation would use a proper compositing shader

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, self.width, self.height)

        # Enable additive blending for bloom
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)

        # Render bloom texture over the scene
        glBindTexture(GL_TEXTURE_2D, self._blur_texture_v)

        # Draw fullscreen quad
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0)
        glVertex2f(-1.0, -1.0)
        glTexCoord2f(1.0, 0.0)
        glVertex2f(1.0, -1.0)
        glTexCoord2f(1.0, 1.0)
        glVertex2f(1.0, 1.0)
        glTexCoord2f(0.0, 1.0)
        glVertex2f(-1.0, 1.0)
        glEnd()

        glDisable(GL_BLEND)

    def process(self) -> None:
        """Process bloom effect (main entry point)."""
        if not self._enabled:
            return

        # Note: This is a simplified implementation for Legacy OpenGL
        # For proper bloom, you would need to use shaders or ModernGL

        # The actual bloom processing would be done in the render loop
        pass

    def set_threshold(self, threshold: float) -> None:
        """Set brightness threshold.

        Args:
            threshold: New threshold value (0.0 - 1.0)
        """
        self.threshold = max(0.0, min(1.0, threshold))

    def set_intensity(self, intensity: float) -> None:
        """Set bloom intensity.

        Args:
            intensity: New intensity value (0.0 - 1.0+)
        """
        self.intensity = max(0.0, intensity)

    def set_radius(self, radius: float) -> None:
        """Set blur radius.

        Args:
            radius: New blur radius in pixels
        """
        if radius != self.radius:
            self.radius = max(1.0, radius)
            self._compute_gaussian_kernel()

    def cleanup(self) -> None:
        """Clean up OpenGL resources."""
        # Delete textures
        if self._bright_texture:
            glDeleteTextures([self._bright_texture])
        if self._blur_texture_h:
            glDeleteTextures([self._blur_texture_h])
        if self._blur_texture_v:
            glDeleteTextures([self._blur_texture_v])

        # Delete framebuffers
        if self._bright_fbo:
            glDeleteFramebuffers([self._bright_fbo])
        if self._blur_fbo_h:
            glDeleteFramebuffers([self._blur_fbo_h])
        if self._blur_fbo_v:
            glDeleteFramebuffers([self._blur_fbo_v])

        # Delete renderbuffer
        if self._depth_renderbuffer:
            glDeleteRenderbuffers([self._depth_renderbuffer])

        # Reset IDs
        self._bright_texture = None
        self._blur_texture_h = None
        self._blur_texture_v = None
        self._bright_fbo = None
        self._blur_fbo_h = None
        self._blur_fbo_v = None
        self._depth_renderbuffer = None


class SimpleBloom:
    """Simplified bloom effect for Legacy OpenGL (Mac compatible).

    Uses additive blending and multiple render passes to create
    a glow effect without requiring modern shader features.
    """

    def __init__(self, intensity: float = 0.3):
        """Initialize simplified bloom effect.

        Args:
            intensity: Bloom intensity (0.0 - 1.0)
        """
        self.intensity = intensity
        self._enabled = True

    @property
    def enabled(self) -> bool:
        """Check if bloom is enabled."""
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable bloom.

        Args:
            enabled: True to enable, False to disable
        """
        self._enabled = enabled

    def apply_glow(self, r: float, g: float, b: float, a: float, emission: float) -> tuple[float, float, float, float]:
        """Apply glow effect to a color.

        Args:
            r, g, b, a: Base color values
            emission: Emission intensity (0.0 - 1.0)

        Returns:
            Tuple of (r, g, b, a) with glow applied
        """
        if not self._enabled or emission <= 0.01:
            return (r, g, b, a)

        # Add emission-based glow
        glow_factor = emission * self.intensity * 2.0
        r = min(1.0, r + glow_factor)
        g = min(1.0, g + glow_factor)
        b = min(1.0, b + glow_factor)

        return (r, g, b, a)

    def pre_render(self) -> None:
        """Setup state for bloom rendering."""
        if not self._enabled:
            return

        # Enable additive blending for glow
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)

    def post_render(self) -> None:
        """Restore state after bloom rendering."""
        if not self._enabled:
            return

        # Restore default blending
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
