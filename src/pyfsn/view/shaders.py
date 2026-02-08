"""Shader programs for 3D rendering.

Provides vertex and fragment shaders for instanced cube rendering
with lighting and perspective projection.
"""

# Vertex shader for instanced cube rendering
VERTEX_SHADER = """
#version 330

// Per-vertex attributes
in vec3 in_position;
in vec3 in_normal;

// Per-instance attributes
in vec3 in_instance_position;
in vec3 in_instance_scale;
in vec4 in_instance_color;

// Uniforms
uniform mat4 u_view;
uniform mat4 u_projection;
uniform mat4 u_model;

// Outputs to fragment shader
out vec3 v_normal;
out vec3 v_position;
out vec4 v_color;

void main() {
    // Apply instance scale
    vec3 scaled_position = in_position * in_instance_scale;

    // Apply instance translation
    vec3 world_position = scaled_position + in_instance_position;

    // Calculate final position
    vec4 view_position = u_view * u_model * vec4(world_position, 1.0);
    gl_Position = u_projection * view_position;

    // Pass normal (apply model matrix for correct lighting)
    v_normal = mat3(u_model) * in_normal;

    // Pass world position for lighting
    v_position = world_position;

    // Pass instance color
    v_color = in_instance_color;
}
"""

# Fragment shader with simple directional lighting
FRAGMENT_SHADER = """
#version 330

// Inputs from vertex shader
in vec3 v_normal;
in vec3 v_position;
in vec4 v_color;

// Output
out vec4 frag_color;

void main() {
    // Normalize normal
    vec3 normal = normalize(v_normal);

    // Directional light (coming from top-right-front)
    vec3 light_dir = normalize(vec3(1.0, 1.0, 1.0));

    // Ambient lighting
    float ambient = 0.4;

    // Diffuse lighting
    float diffuse = max(dot(normal, light_dir), 0.0) * 0.6;

    // Combined lighting
    float lighting = ambient + diffuse;

    // Apply lighting to color
    vec3 lit_color = v_color.rgb * lighting;

    // Simple edge darkening for depth perception
    float edge_factor = 1.0 - abs(dot(normal, vec3(0.0, 0.0, 1.0)));
    lit_color *= (1.0 - edge_factor * 0.1);

    // Output final color
    frag_color = vec4(lit_color, v_color.a);

    // Handle transparency
    if (frag_color.a < 0.01) {
        discard;
    }
}
"""

# Wireframe vertex shader (for connection lines)
WIREFRAME_VERTEX_SHADER = """
#version 330

in vec3 in_position;
in vec4 in_color;

uniform mat4 u_view;
uniform mat4 u_projection;
uniform mat4 u_model;

out vec4 v_color;

void main() {
    vec4 world_position = u_model * vec4(in_position, 1.0);
    vec4 view_position = u_view * world_position;
    gl_Position = u_projection * view_position;
    v_color = in_color;
}
"""

# Wireframe fragment shader
WIREFRAME_FRAGMENT_SHADER = """
#version 330

in vec4 v_color;
out vec4 frag_color;

void main() {
    frag_color = v_color;
    if (frag_color.a < 0.01) {
        discard;
    }
}
"""

# Selection outline vertex shader
OUTLINE_VERTEX_SHADER = """
#version 330

in vec3 in_position;

uniform mat4 u_view;
uniform mat4 u_projection;
uniform mat4 u_model;
uniform vec3 u_instance_position;
uniform vec3 u_instance_scale;

void main() {
    // Slightly scaled for outline effect
    vec3 scaled_position = in_position * u_instance_scale * 1.05;
    vec3 world_position = scaled_position + u_instance_position;

    vec4 view_position = u_view * u_model * vec4(world_position, 1.0);
    gl_Position = u_projection * view_position;
}
"""

# Selection outline fragment shader
OUTLINE_FRAGMENT_SHADER = """
#version 330

out vec4 frag_color;

void main() {
    // Bright yellow outline
    frag_color = vec4(1.0, 0.9, 0.2, 1.0);
}
"""


def get_instancing_shader_source() -> tuple[str, str]:
    """Get vertex and fragment shader sources for instanced rendering.

    Returns:
        Tuple of (vertex_shader, fragment_shader)
    """
    return VERTEX_SHADER, FRAGMENT_SHADER


def get_wireframe_shader_source() -> tuple[str, str]:
    """Get vertex and fragment shader sources for wireframe rendering.

    Returns:
        Tuple of (vertex_shader, fragment_shader)
    """
    return WIREFRAME_VERTEX_SHADER, WIREFRAME_FRAGMENT_SHADER


def get_outline_shader_source() -> tuple[str, str]:
    """Get vertex and fragment shader sources for outline rendering.

    Returns:
        Tuple of (vertex_shader, fragment_shader)
    """
    return OUTLINE_VERTEX_SHADER, OUTLINE_FRAGMENT_SHADER
