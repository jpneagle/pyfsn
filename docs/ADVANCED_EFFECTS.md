# Advanced Shaders & Post-Processing Effects

This document describes the cyberpunk visual effects implemented in pyfsn, including emissive materials, wire pulse animations, and bloom post-processing.

## Overview

The pyfsn project features advanced visual effects inspired by cyberpunk aesthetics:

1. **Emissive Materials** - File type-based glow effects
2. **Wire Pulse Effect** - Animated connections with traveling pulses
3. **Bloom Effect** - Post-processing glow for bright elements

## Emissive Materials

### Overview

Emissive materials cause certain files to "glow" based on their type and git status. This creates a cyberpunk visual style where code files and modified items stand out.

### Emission Levels by File Type

| File Extension | Emission Level | Visual Effect |
|----------------|----------------|---------------|
| `.py`, `.js`, `.ts`, `.rs`, `.go`, `.java`, `.cpp`, `.h`, `.c` | 0.6 (High) | Bright glow for code files |
| `.sh`, `.bash`, `.zsh`, `.fish` | 0.5 (Medium-High) | Noticeable glow for scripts |
| `.json`, `.yaml`, `.yml`, `.toml`, `.xml` | 0.3 (Medium) | Subtle glow for config files |
| `.md`, `.txt`, `.rst`, `.adoc` | 0.2 (Low) | Faint glow for documentation |
| Other files | 0.1 (Minimal) | Very subtle glow |

### Git Status Modifiers

Files under git version control receive additional emission based on their status:

| Git Status | Emission Modifier | Visual Effect |
|------------|-------------------|---------------|
| Modified (`M`, `MM`, `AM`, `TM`) | +0.4 | Strong glow for modified files |
| Added (`A`, `??`) | +0.3 | Noticeable glow for new files |
| Deleted (`D`) | * 0.3 | Dim glow for deleted files |
| Selected | +0.3 | Extra glow for selected items |

### Implementation

The emission effect is implemented through:

1. **`CubeInstance.emission`** field - stores emission intensity per instance
2. **`_calculate_emission()`** method - computes emission based on file type and git status
3. **`SimpleBloom.apply_glow()`** - applies emission-based color boost during rendering

### Usage

```python
# Enable/disable bloom (includes emission effect)
renderer.set_bloom_enabled(True)

# Adjust bloom intensity (affects emission visibility)
renderer.set_bloom_intensity(0.5)

# Emission is automatically calculated based on file type
# No manual configuration needed
```

## Wire Pulse Effect

### Overview

The wire pulse effect creates animated "data packets" traveling along connection wires between directories. Selected connections receive enhanced pulse effects with cyberpunk colors.

### Visual Characteristics

- **Base Effect**: Subtle cyan-magenta color shift on all connections
- **Selected Connections**: Thicker wires with enhanced glow
- **Traveling Pulse**: Bright cyan segments that travel along selected wires
- **Glow Effect**: Multi-layered rendering with additive blending

### Configuration

The pulse effect is configured with these parameters:

```python
# In _draw_connections()
pulse_speed = 2.0        # Speed of pulse animation
pulse_intensity = 0.3    # Strength of pulse effect
travel_speed = 3.0       # Speed of traveling pulse segment
```

### Implementation Details

1. **Time-based Animation**: Uses `_animation_time` for smooth animation
2. **Multi-pass Rendering**:
   - Base wire with pulse color shift
   - Selected wire with enhanced glow
   - Traveling pulse segment
3. **Additive Blending**: Uses `GL_ONE` blending mode for glow effect

### Performance Considerations

- Pulse calculations are done on CPU (minimal overhead)
- Only selected connections receive the full multi-pass effect
- Optimized with early-out for non-selected connections

## Bloom Effect

### Overview

The bloom effect creates a soft glow around bright elements in the scene. This is implemented using a simplified approach compatible with Legacy OpenGL (2.1) for Mac compatibility.

### Implementation

Two bloom implementations are provided:

1. **`BloomEffect`** - Full-featured bloom with framebuffers (requires OpenGL 3.0+)
2. **`SimpleBloom`** - Simplified bloom for Legacy OpenGL (default)

### SimpleBloom API

```python
# Create bloom effect
bloom = SimpleBloom(intensity=0.3)

# Enable/disable
bloom.set_enabled(True)
is_enabled = bloom.enabled

# Apply glow to a color
r, g, b, a = bloom.apply_glow(r, g, b, a, emission)
```

### Configuration

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `intensity` | 0.0 - 1.0+ | 0.3 | Overall bloom strength |
| `threshold` | 0.0 - 1.0 | 0.8 | Brightness threshold for bloom |
| `radius` | 1.0 - 20.0+ | 8.0 | Blur radius in pixels |

### Performance

The simplified bloom implementation has minimal performance impact:

- No additional render passes required
- No framebuffer operations
- Just additive blending during normal rendering

## Shader Files

The following GLSL shaders are provided for future ModernGL integration:

### Emissive Shaders

- **`src/pyfsn/view/shaders/emissive.vert`** - Vertex shader with emission support
- **`src/pyfsn/view/shaders/emissive.frag`** - Fragment shader with cyberpunk color palette

Features:
- Per-instance emission intensity
- Fresnel-like rim lighting
- Animated pulse effect
- Cyberpunk color palette (cyan, magenta, yellow, green-cyan, orange, purple)

### Wire Shaders

- **`src/pyfsn/view/shaders/wire.vert`** - Wireframe vertex shader
- **`src/pyfsn/view/shaders/wire.frag`** - Wireframe fragment shader with pulse

Features:
- Time-based pulse animation
- Directional pulse along wire
- Color variation based on position
- Enhanced effects for selected connections

## Demo

A demo application is provided to showcase all effects:

```bash
python -m pyfsn.view.effects_demo
```

The demo features:
- Interactive 3D visualization
- Bloom intensity slider
- Performance statistics
- Various file types demonstrating emission

## Performance Characteristics

### Emission Effect

- **CPU Impact**: Negligible (< 1% per frame)
- **GPU Impact**: Minimal (just color multiplication)
- **Memory Impact**: ~4 bytes per instance (float emission value)

### Wire Pulse Effect

- **CPU Impact**: ~2-3% per frame (animation calculations)
- **GPU Impact**: Low for base effect, moderate for selected connections
- **Scalability**: O(n) where n = number of connections

### Bloom Effect (Simple)

- **CPU Impact**: Negligible
- **GPU Impact**: Minimal (additive blending only)
- **Scalability**: O(1) - constant time

## Enabling/Disabling Effects

```python
# Enable/disable bloom (includes emission glow)
renderer.set_bloom_enabled(True)

# Adjust bloom intensity
renderer.set_bloom_intensity(0.5)

# Wire pulse is always active when connections are visible
# No separate control needed
```

## Future Enhancements

Potential improvements for future versions:

1. **Full Shader-Based Bloom** - Proper Gaussian blur with framebuffers
2. **Configurable Emission Colors** - User-defined color schemes
3. **Pulse Pattern Options** - Different pulse animations (sine, square, etc.)
4. **Per-Connection Emission** - Git status-based wire glow
5. **Performance Profiling** - Built-in FPS and timing diagnostics

## Troubleshooting

### Bloom Not Visible

- Ensure bloom is enabled: `renderer.set_bloom_enabled(True)`
- Check intensity: `renderer.get_bloom_intensity()` should be > 0
- Verify files have emission: Check file types match emission table

### Pulse Effect Not Smooth

- Check timer is running: `renderer._update_timer.isActive()`
- Verify `_animation_time` is incrementing
- Monitor FPS: Should be 60 FPS for smooth animation

### Performance Issues

- Reduce bloom intensity
- Reduce number of visible connections
- Check if bloom is enabled (disable if not needed)
- Monitor instance count in performance stats
