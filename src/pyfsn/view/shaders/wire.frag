#version 330

// Inputs from vertex shader
in vec3 v_world_position;
in vec4 v_color;
in float v_selected;
in float v_time;
in vec3 v_start;
in vec3 v_end;

// Output
out vec4 frag_color;

// Uniforms
uniform float u_pulse_speed;
uniform float u_pulse_intensity;

void main() {
    // Base color
    vec3 base_color = v_color.rgb;
    float alpha = v_color.a;

    // Calculate position along the wire (0.0 to 1.0)
    vec3 wire_dir = normalize(v_end - v_start);
    float wire_length = length(v_end - v_start);
    float t = dot(v_world_position - v_start, wire_dir) / wire_length;

    // Pulse effect - animated glow traveling along the wire
    float pulse_phase = mod(v_time * u_pulse_speed - t * 3.0, 3.0);
    float pulse = smoothstep(0.0, 0.3, pulse_phase) * smoothstep(0.6, 0.3, pulse_phase);

    // Cyberpunk colors for pulse
    vec3 pulse_color = mix(
        vec3(0.0, 1.0, 1.0),  // Cyan
        vec3(1.0, 0.0, 1.0),  // Magenta
        sin(v_time * 0.5)
    );

    // Apply pulse effect
    float pulse_amount = u_pulse_intensity * pulse;
    if (v_selected > 0.5) {
        // Selected wires have stronger pulse
        pulse_amount *= 2.0;
        base_color = mix(base_color, vec3(1.0, 1.0, 0.8), 0.5);  // Yellow-white tint
    }

    vec3 final_color = mix(base_color, pulse_color, pulse_amount);

    // Add glow at edges (simulated)
    float edge_glow = abs(sin(t * 20.0 + v_time * 3.0)) * 0.1;
    final_color += pulse_color * edge_glow * pulse_amount;

    // Increase brightness for selected connections
    if (v_selected > 0.5) {
        final_color += vec3(0.2, 0.2, 0.1);
        alpha = min(alpha * 1.5, 1.0);
    }

    frag_color = vec4(final_color, alpha);

    if (frag_color.a < 0.01) {
        discard;
    }
}
