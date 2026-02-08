#version 330

// Fullscreen quad (no attributes needed)
out vec2 v_texcoord;

void main() {
    // Generate fullscreen quad vertices
    vec2 position;
    int vertex_id = gl_VertexID;

    // Triangle strip: 4 vertices
    if (vertex_id == 0) {
        position = vec2(-1.0, -1.0);
        v_texcoord = vec2(0.0, 0.0);
    } else if (vertex_id == 1) {
        position = vec2(1.0, -1.0);
        v_texcoord = vec2(1.0, 0.0);
    } else if (vertex_id == 2) {
        position = vec2(-1.0, 1.0);
        v_texcoord = vec2(0.0, 1.0);
    } else {
        position = vec2(1.0, 1.0);
        v_texcoord = vec2(1.0, 1.0);
    }

    gl_Position = vec4(position, 0.999, 1.0);  // Near far clip
}
