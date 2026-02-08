#version 330

// Inputs from vertex shader
in vec3 v_normal;
in vec3 v_world_position;
in vec3 v_view_position;
in vec4 v_color;
in float v_emission;
in float v_time;

// Output
out vec4 frag_color;

// Uniforms
uniform vec3 u_camera_position;
uniform float u_bloom_threshold;

// Emission color palette (cyberpunk style)
const vec3 EMISSION_COLORS[6] = vec3[6](
    vec3(0.0, 1.0, 1.0),    // Cyan
    vec3(1.0, 0.0, 1.0),    // Magenta
    vec3(1.0, 1.0, 0.0),    // Yellow
    vec3(0.0, 1.0, 0.5),    // Green-cyan
    vec3(1.0, 0.5, 0.0),    // Orange
    vec3(0.5, 0.0, 1.0)     // Purple
);

void main() {
    // Normalize normal
    vec3 normal = normalize(v_normal);

    // Directional light (coming from top-right-front)
    vec3 light_dir = normalize(vec3(1.0, 1.0, 1.0));

    // Ambient lighting
    float ambient = 0.3;

    // Diffuse lighting
    float diffuse = max(dot(normal, light_dir), 0.0) * 0.5;

    // Combined base lighting
    float lighting = ambient + diffuse;

    // Apply lighting to base color
    vec3 base_color = v_color.rgb * lighting;

    // Emission calculation
    vec3 emission_color = vec3(0.0);

    if (v_emission > 0.01) {
        // Select emission color based on position (creates variety)
        int color_index = int(abs(v_world_position.x + v_world_position.z) * 2.0) % 6;
        vec3 glow_color = EMISSION_COLORS[color_index];

        // Animate emission intensity slightly
        float pulse = 0.8 + 0.2 * sin(v_time * 2.0 + v_world_position.x);

        // Combine emission with color
        emission_color = glow_color * v_emission * pulse * 2.0;

        // Add rim lighting effect (Fresnel-like)
        vec3 view_dir = normalize(u_camera_position - v_world_position);
        float rim_factor = 1.0 - max(dot(normal, view_dir), 0.0);
        rim_factor = pow(rim_factor, 3.0);
        emission_color += glow_color * v_emission * rim_factor * 0.5;
    }

    // Combine base color with emission
    vec3 final_color = base_color + emission_color;

    // Calculate brightness for bloom threshold
    float brightness = max(max(final_color.r, final_color.g), final_color.b);

    // Output final color
    frag_color = vec4(final_color, v_color.a);

    // Discard fully transparent fragments
    if (frag_color.a < 0.01) {
        discard;
    }
}
