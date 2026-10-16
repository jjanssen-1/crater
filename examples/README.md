# crater examples

## triangle (header-only, no window system)

The minimal demonstration of the declarative style.

```bash
crater generate examples/triangle.yaml -o examples/generated/triangle.hpp
g++ -std=c++17 -I examples/generated -I <vulkan-headers>/include \
    -fsyntax-only examples/main.cpp
```

## glfw_triangle (full Vulkan triangle with GLFW)

A complete, runnable triangle: instance with GLFW's extension list,
runtime queue family selection (graphics required, compute preferred,
`samplerAnisotropy` required), device creation, surface capability
selection (format/present mode preference lists), render pass, graphics
pipeline, swapchain and sync objects — all declared in
`glfw_triangle.yaml`. The imperative parts (render loop, command buffer
recording, per-swapchain-image views/framebuffers) stay explicit in
`glfw_triangle.cpp`.

### Prerequisites

- Python ≥ 3.10 with `pip install .` (this repo)
- Vulkan SDK (provides `vk.xml`, `vulkan.h`, `glslc`, the loader)
- GLFW 3.3+ (development package)
- C++17 compiler

### Build & run

```bash
# 1. generate the C++ header from the YAML spec
crater generate examples/glfw_triangle.yaml -o examples/generated/glfw_triangle.hpp

# 2. compile the shaders
glslc examples/shaders/triangle.vert -o examples/generated/triangle.vert.spv
glslc examples/shaders/triangle.frag -o examples/generated/triangle.frag.spv

# 3. build (adjust include paths if your SDK is not in the default search path)
g++ -std=c++17 examples/glfw_triangle.cpp -o triangle \
    -I examples/generated -I $VULKAN_SDK/include \
    -lglfw -lvulkan

# 4. run from the repository root (shader paths are relative)
./triangle
```

CMake equivalent (recommended for real projects):

```cmake
cmake_minimum_required(VERSION 3.16)
project(triangle CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(glfw3 REQUIRED)
find_package(Vulkan REQUIRED)

add_executable(triangle examples/glfw_triangle.cpp)
target_include_directories(triangle PRIVATE examples/generated)
target_link_libraries(triangle PRIVATE glfw Vulkan::Vulkan)
```

### What is declarative, what is not

Everything in `glfw_triangle.yaml` is created and destroyed by the
generated classes, in dependency order, with capability values selected
at runtime per the declared rules. Three things remain explicit in the
C++:

1. **Physical device choice** — the first device is picked; preferring a
   discrete GPU or asking the user is application policy.
2. **Swapchain image views / framebuffers** — their count depends on
   `vkGetSwapchainImagesKHR`, which is runtime-dependent; expressing
   "one per image" declaratively is planned.
3. **Command buffers and the render loop** — recording is imperative by
   nature and a later phase for crater.

### Verified in CI-less development

The generated header and `glfw_triangle.cpp` were compile-checked with
`g++ -fsyntax-only` against the official Vulkan headers and GLFW's
public headers (no GPU available in the sandbox). Linking and rendering
need a real machine — please report any issues you hit at runtime.

## quad (feature-rich example)

An animated textured quad exercising the broader feature set beyond the
triangle:

- multi-selection queue families (graphics + transfer from ONE query)
- pNext feature queries (core + Vulkan 1.1/1.2 features chained)
- descriptors: pool + set layout + set, updated with UBO + sampler
- uniform buffer with bound memory, updated per frame (rotation)
- vertex + index buffers uploaded through a staging buffer
- texture: image + view + sampler, checkerboard uploaded via barriers
- graphics AND compute pipelines in one class
- cross-class references (pipeline layout consumes the descriptor set
  layout from another class)
- namespace + per-frame call functions

```bash
crater generate examples/quad.yaml -o examples/generated/quad.hpp
glslc examples/shaders/quad.vert -o examples/generated/quad.vert.spv
glslc examples/shaders/quad.frag -o examples/generated/quad.frag.spv
glslc examples/shaders/quad.comp -o examples/generated/quad.comp.spv
g++ -std=c++17 examples/quad.cpp -o quad \
    -I examples -I $VULKAN_SDK/include -lglfw -lvulkan
./quad
```

The imperative parts remain: memory binding (runtime memory type
selection), uploads (one-time copy commands), descriptor writes,
recording, and the render loop - everything declarable is in the yaml.

## particles (VMA + compute example)

A GPU particle simulation exercising the machinery the quad didn't:

- **VMA**: the allocator as an ordinary object; buffers with persistent
  host mappings (initial positions and per-frame uniforms are written
  through `VmaAllocationInfo::pMappedData` - no vkMapMemory anywhere)
- **a compute pipeline that is actually dispatched** each frame,
  integrating 8192 particle positions in a storage buffer before the
  graphics pass reads them (chained semaphores, separate command
  buffers per pass)
- timestamp queries, pNext feature chains on the device, a surface
  format preference with a real fallback candidate
- point-list rendering with the storage buffer as a vertex buffer

```bash
crater generate examples/particles.yaml -o examples/generated/particles.hpp
glslc examples/shaders/particles.comp -o examples/generated/particles.comp.spv
glslc examples/shaders/particles.vert -o examples/generated/particles.vert.spv
glslc examples/shaders/particles.frag -o examples/generated/particles.frag.spv
g++ -std=c++17 examples/particles.cpp -o particles \
    -I examples -I $VULKAN_SDK/include -I <vma-headers> -lglfw -lvulkan
./particles
