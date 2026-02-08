"""Shader loading and management for ModernGL.

This module provides the ShaderLoader class for loading, compiling,
and caching shader programs using ModernGL.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ShaderCompilationError(Exception):
    """Raised when shader compilation fails."""

    def __init__(self, shader_type: str, message: str) -> None:
        """Initialize the error.

        Args:
            shader_type: Type of shader (vertex, fragment, geometry)
            message: Error message from shader compilation
        """
        self.shader_type = shader_type
        self.message = message
        super().__init__(f"Shader compilation error ({shader_type}): {message}")


class ProgramLinkError(Exception):
    """Raised when shader program linking fails."""

    def __init__(self, message: str) -> None:
        """Initialize the error.

        Args:
            message: Error message from program linking
        """
        self.message = message
        super().__init__(f"Program linking error: {message}")


class ShaderLoader:
    """Load, compile, and cache shader programs for ModernGL.

    This class provides methods for loading shaders from files or strings,
    compiling them, and linking them into complete programs. It caches
    compiled programs to avoid redundant compilation.

    Example:
        ```python
        import moderngl
        from pyfsn.view.shader_loader import ShaderLoader

        ctx = moderngl.create_context()
        loader = ShaderLoader(ctx)

        # Load shader program
        program = loader.load_program(
            vertex_shader="shaders/cube.vert.glsl",
            fragment_shader="shaders/cube.frag.glsl"
        )

        # Or load from string
        program = loader.load_program_from_string(
            vertex_source=vertex_src,
            fragment_source=fragment_src
        )
        ```
    """

    def __init__(self, ctx) -> None:
        """Initialize the shader loader.

        Args:
            ctx: ModernGL context
        """
        self._ctx = ctx
        self._program_cache: dict[str, object] = {}
        self._shader_cache: dict[tuple[str, str], object] = {}

    def load_shader(
        self,
        source: str,
        shader_type: str,
        cache_key: Optional[str] = None
    ) -> object:
        """Load and compile a single shader.

        Args:
            source: Shader source code or file path
            shader_type: Type of shader ('vertex', 'fragment', 'geometry')
            cache_key: Optional cache key to store/retrieve compiled shader

        Returns:
            Compiled ModernGL shader object

        Raises:
            ShaderCompilationError: If shader compilation fails
        """
        # Check cache first
        if cache_key is not None and (cache_key, shader_type) in self._shader_cache:
            return self._shader_cache[(cache_key, shader_type)]

        # Load source from file if it's a path
        shader_source = self._load_shader_source(source)

        # Determine ModernGL shader type
        gl_shader_type = self._get_shader_type(shader_type)

        # Compile shader
        try:
            shader = self._ctx.shader(gl_shader_type, shader_source)
            logger.debug(f"Compiled {shader_type} shader successfully")
        except Exception as e:
            raise ShaderCompilationError(shader_type, str(e)) from e

        # Cache the shader if key provided
        if cache_key is not None:
            self._shader_cache[(cache_key, shader_type)] = shader

        return shader

    def load_program(
        self,
        vertex_shader: str,
        fragment_shader: str,
        geometry_shader: Optional[str] = None,
        cache_key: Optional[str] = None,
        defines: Optional[dict[str, str]] = None
    ) -> object:
        """Load and link a shader program from files.

        Args:
            vertex_shader: Vertex shader source or file path
            fragment_shader: Fragment shader source or file path
            geometry_shader: Optional geometry shader source or file path
            cache_key: Optional cache key to store/retrieve program
            defines: Optional preprocessor definitions to inject

        Returns:
            Linked ModernGL program object

        Raises:
            ShaderCompilationError: If shader compilation fails
            ProgramLinkError: If program linking fails
        """
        # Check cache first
        if cache_key is not None and cache_key in self._program_cache:
            return self._program_cache[cache_key]

        # Load and compile each shader
        vs = self.load_shader(vertex_shader, "vertex")
        fs = self.load_shader(fragment_shader, "fragment")

        gs = None
        if geometry_shader is not None:
            gs = self.load_shader(geometry_shader, "geometry")

        # Link program
        try:
            if gs is not None:
                program = self._ctx.program(vertex_shader=vs, fragment_shader=fs, geometry_shader=gs)
            else:
                program = self._ctx.program(vertex_shader=vs, fragment_shader=fs)
            logger.debug(f"Linked program successfully")
        except Exception as e:
            raise ProgramLinkError(str(e)) from e

        # Cache the program if key provided
        if cache_key is not None:
            self._program_cache[cache_key] = program

        return program

    def load_program_from_string(
        self,
        vertex_source: str,
        fragment_source: str,
        geometry_source: Optional[str] = None,
        cache_key: Optional[str] = None,
        defines: Optional[dict[str, str]] = None
    ) -> object:
        """Load and link a shader program from source strings.

        Args:
            vertex_source: Vertex shader source code
            fragment_source: Fragment shader source code
            geometry_source: Optional geometry shader source code
            cache_key: Optional cache key to store/retrieve program
            defines: Optional preprocessor definitions to inject

        Returns:
            Linked ModernGL program object

        Raises:
            ShaderCompilationError: If shader compilation fails
            ProgramLinkError: If program linking fails
        """
        # Inject preprocessor definitions if provided
        if defines:
            vertex_source = self._inject_defines(vertex_source, defines)
            fragment_source = self._inject_defines(fragment_source, defines)
            if geometry_source:
                geometry_source = self._inject_defines(geometry_source, defines)

        # Check cache first
        if cache_key is not None and cache_key in self._program_cache:
            return self._program_cache[cache_key]

        # Compile shaders from source strings
        try:
            vs = self._ctx.shader(self._get_shader_type("vertex"), vertex_source)
            fs = self._ctx.shader(self._get_shader_type("fragment"), fragment_source)
        except Exception as e:
            raise ShaderCompilationError("vertex/fragment", str(e)) from e

        gs = None
        if geometry_source is not None:
            try:
                gs = self._ctx.shader(self._get_shader_type("geometry"), geometry_source)
            except Exception as e:
                raise ShaderCompilationError("geometry", str(e)) from e

        # Link program
        try:
            if gs is not None:
                program = self._ctx.program(vertex_shader=vs, fragment_shader=fs, geometry_shader=gs)
            else:
                program = self._ctx.program(vertex_shader=vs, fragment_shader=fs)
            logger.debug(f"Linked program from string successfully")
        except Exception as e:
            raise ProgramLinkError(str(e)) from e

        # Cache the program if key provided
        if cache_key is not None:
            self._program_cache[cache_key] = program

        return program

    def get_program(self, cache_key: str) -> Optional[object]:
        """Get a cached program by cache key.

        Args:
            cache_key: Cache key used when loading the program

        Returns:
            Cached ModernGL program object or None if not found
        """
        return self._program_cache.get(cache_key)

    def has_program(self, cache_key: str) -> bool:
        """Check if a program is cached.

        Args:
            cache_key: Cache key to check

        Returns:
            True if program is cached, False otherwise
        """
        return cache_key in self._program_cache

    def clear_cache(self) -> None:
        """Clear all cached shaders and programs.

        Note: This does not release GPU resources; the ModernGL context
        will handle cleanup when destroyed.
        """
        self._program_cache.clear()
        self._shader_cache.clear()
        logger.debug("Cleared shader and program cache")

    def release_program(self, cache_key: str) -> bool:
        """Release a specific cached program.

        Args:
            cache_key: Cache key of program to release

        Returns:
            True if program was found and released, False otherwise
        """
        if cache_key in self._program_cache:
            program = self._program_cache.pop(cache_key)
            program.release()
            logger.debug(f"Released program: {cache_key}")
            return True
        return False

    def _load_shader_source(self, source: str) -> str:
        """Load shader source from file path or return as-is.

        Args:
            source: Shader source code or file path

        Returns:
            Shader source code
        """
        # Check if it's a file path
        path = Path(source)
        if path.exists() and path.suffix in ['.vert', '.frag', '.geom', '.glsl', '.vs', '.fs']:
            try:
                return path.read_text(encoding='utf-8')
            except IOError as e:
                logger.error(f"Failed to read shader file: {source}")
                raise

        # Return as-is (already source code)
        return source

    def _get_shader_type(self, shader_type: str) -> str:
        """Map shader type string to ModernGL shader type.

        Args:
            shader_type: Shader type string

        Returns:
            ModernGL shader type constant

        Raises:
            ValueError: If shader type is unknown
        """
        type_map = {
            'vertex': 'vertex_shader',
            'fragment': 'fragment_shader',
            'geometry': 'geometry_shader',
        }

        if shader_type not in type_map:
            raise ValueError(f"Unknown shader type: {shader_type}")

        return type_map[shader_type]

    def _inject_defines(self, source: str, defines: dict[str, str]) -> str:
        """Inject preprocessor definitions into shader source.

        Args:
            source: Shader source code
            defines: Dictionary of preprocessor definitions

        Returns:
            Shader source with definitions injected
        """
        if not defines:
            return source

        # Build #define lines
        define_lines = [f"#define {key} {value}\n" for key, value in defines.items()]

        # Find version directive or insert at beginning
        lines = source.split('\n')
        insert_index = 0

        for i, line in enumerate(lines):
            if line.strip().startswith('#version'):
                insert_index = i + 1
                break

        # Insert defines
        lines.insert(insert_index, ''.join(define_lines))

        return '\n'.join(lines)

    def __del__(self) -> None:
        """Cleanup when loader is destroyed."""
        self.clear_cache()
