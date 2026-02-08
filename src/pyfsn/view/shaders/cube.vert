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

    // Pass normal (apply model matrix for correct lighting if needed)
    v_normal = mat3(u_model) * in_normal;

    // Pass world position for fog
    v_position = world_position;

    // Pass instance color
    v_color = in_instance_color;
}
