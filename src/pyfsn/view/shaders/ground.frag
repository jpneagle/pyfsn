#version 330

// Inputs from vertex shader
in vec2 v_texcoord;
in vec3 v_position;

// Uniforms
uniform vec3 u_camera_position;
uniform float u_fog_start;
uniform float u_fog_end;
uniform vec4 u_fog_color;

// Output
out vec4 frag_color;

void main() {
    // Dark green ground with grid pattern
    vec2 grid = abs(fract(v_texcoord * 20.0) - 0.5) / fwidth(v_texcoord * 20.0);
    float line = min(grid.x, grid.y);
    float grid_alpha = 1.0 - min(line, 1.0);

    // Base green color
    vec3 ground_color = vec3(0.2, 0.4, 0.2);
    vec3 grid_color = vec3(0.3, 0.5, 0.3);

    // Mix base and grid
    vec3 final_color = mix(ground_color, grid_color, grid_alpha * 0.5);

    // Apply depth fog
    float distance = length(v_position - u_camera_position);
    float fog_factor = clamp((distance - u_fog_start) / (u_fog_end - u_fog_start), 0.0, 1.0);

    // Mix color with fog
    vec3 fogged_color = mix(final_color, u_fog_color.rgb, fog_factor);

    frag_color = vec4(fogged_color, 0.9);
}
