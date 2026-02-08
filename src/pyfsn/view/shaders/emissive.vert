#version 330

// Per-vertex attributes
in vec3 in_position;
in vec3 in_normal;

// Per-instance attributes
in vec3 in_instance_position;
in vec3 in_instance_scale;
in vec4 in_instance_color;
in float in_instance_emission;  // Emission intensity (0.0 - 1.0+)

// Uniforms
uniform mat4 u_view;
uniform mat4 u_projection;
uniform mat4 u_model;
uniform float u_time;  // For animated effects

// Outputs to fragment shader
out vec3 v_normal;
out vec3 v_world_position;
out vec3 v_view_position;
out vec4 v_color;
out float v_emission;
out float v_time;

void main() {
    // Apply instance scale
    vec3 scaled_position = in_position * in_instance_scale;

    // Apply instance translation
    vec3 world_position = scaled_position + in_instance_position;

    // Calculate final position
    vec4 view_pos = u_view * u_model * vec4(world_position, 1.0);
    gl_Position = u_projection * view_pos;

    // Pass normal (apply model matrix for correct lighting)
    v_normal = mat3(u_model) * in_normal;

    // Pass positions
    v_world_position = world_position;
    v_view_position = view_pos.xyz;

    // Pass instance color
    v_color = in_instance_color;

    // Pass emission intensity
    v_emission = in_instance_emission;

    // Pass time for fragment shader animations
    v_time = u_time;
}
