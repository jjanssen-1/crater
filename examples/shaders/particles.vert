#version 450

layout(binding = 0) readonly buffer Particles {
    vec4 positions[];
};

void main() {
    gl_Position = vec4(positions[gl_VertexIndex].xy, 0.0, 1.0);
    gl_PointSize = 2.0;
}
