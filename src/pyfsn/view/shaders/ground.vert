#version 330

// Per-vertex attributes
in vec3 in_position;
in vec2 in_texcoord;

// Uniforms
uniform mat4 u_view;
uniform mat4 u_projection;
uniform mat4 u_model;

// Outputs to fragment shader
out vec2 v_texcoord;
out vec3 v_position;

void main() {
    vec4 world_position = u_model * vec4(in_position, 1.0);
    vec4 view_position = u_view * world_position;
    gl_Position = u_projection * view_position;
    v_texcoord = in_texcoord;
    v_position = world_position.xyz;
}
