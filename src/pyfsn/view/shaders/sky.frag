#version 330

// Inputs from vertex shader
in vec2 v_texcoord;

// Output
out vec4 frag_color;

void main() {
    // FSN original style - bright sky gradient
    // Bottom (horizon) - bright cyan/blue
    vec3 horizon_color = vec3(0.6, 0.8, 1.0);
    // Top - medium blue (not too dark)
    vec3 zenith_color = vec3(0.3, 0.5, 0.9);

    // Vertical gradient
    vec3 sky_color = mix(horizon_color, zenith_color, v_texcoord.y);

    frag_color = vec4(sky_color, 1.0);
}
