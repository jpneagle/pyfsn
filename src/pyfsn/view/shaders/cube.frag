#version 330

// Inputs from vertex shader
in vec3 v_normal;
in vec3 v_position;
in vec4 v_color;

// Uniforms
uniform vec3 u_camera_position;
uniform float u_fog_start;
uniform float u_fog_end;
uniform vec4 u_fog_color;

// Output
out vec4 frag_color;

void main() {
    // Apply color directly (no lighting like legacy renderer)
    vec3 final_color = v_color.rgb;

    // Apply depth fog (SGI fsn style)
    float distance = length(v_position - u_camera_position);
    float fog_factor = clamp((distance - u_fog_start) / (u_fog_end - u_fog_start), 0.0, 1.0);

    // Mix color with fog
    vec3 fogged_color = mix(final_color, u_fog_color.rgb, fog_factor);

    // Output final color
    frag_color = vec4(fogged_color, v_color.a);

    // Handle transparency
    if (frag_color.a < 0.01) {
        discard;
    }
}
