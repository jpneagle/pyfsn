#version 330

// Per-vertex attributes
in vec3 in_position;
in vec3 in_color;

// Per-instance attributes for wire segments
in vec3 in_instance_start;
in vec3 in_instance_end;
in vec4 in_instance_color;
in float in_instance_selected;  // 0.0 = normal, 1.0 = selected/highlighted

// Uniforms
uniform mat4 u_view;
uniform mat4 u_projection;
uniform mat4 u_model;
uniform float u_time;  // For pulse animation

// Outputs to fragment shader
out vec3 v_world_position;
out vec4 v_color;
out float v_selected;
out float v_time;
out vec3 v_start;
out vec3 v_end;

void main() {
    // Calculate world position from line segment
    vec3 world_position = in_instance_start + (in_instance_end - in_instance_start) * in_position.x;

    // Apply model transformation
    vec4 world_pos = u_model * vec4(world_position, 1.0);
    vec4 view_pos = u_view * world_pos;
    gl_Position = u_projection * view_pos;

    // Pass data to fragment shader
    v_world_position = world_pos.xyz;
    v_color = in_instance_color;
    v_selected = in_instance_selected;
    v_time = u_time;
    v_start = in_instance_start;
    v_end = in_instance_end;
}
