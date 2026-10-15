// Example usage of the generated triangle.hpp.
//
// Build:
//   crater generate examples/triangle.yaml -o examples/generated/triangle.hpp
//   g++ -std=c++17 -I examples/generated -I <vulkan-headers>/include examples/main.cpp

#include <cstdint>
#include <vector>

#include "generated/triangle.hpp"

int main() {
    // In a real application these come from a window system binding and a
    // file loader; they are exactly the values crater asks the caller for.
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    std::vector<uint32_t> vertexShader = {0x07230203, 0x00010000, 0x0008000b};
    std::vector<uint32_t> fragmentShader = {0x07230203, 0x00010000, 0x0008000b};

    VulkanInstance instance;

    // Selects a queue family (graphics required, compute preferred) and
    // queries the supported features.
    DeviceCapabilities deviceCapabilities(physicalDevice);

    // Device creation consumes the selected queue family index and the
    // queried features.
    VulkanDevice device(deviceCapabilities, physicalDevice);

    // Queries surface formats/present modes/capabilities at runtime and
    // selects values according to the preference lists in triangle.yaml.
    SurfaceCapabilities capabilities(physicalDevice, surface);

    // Everything else is declared in triangle.yaml: shader modules,
    // pipeline layout, render pass, graphics pipeline, swapchain and
    // synchronization objects, wired to the selected capability values.
    TrianglePipeline pipeline(
        capabilities,
        device,
        vertexShader.size() * sizeof(uint32_t), vertexShader.data(),
        fragmentShader.size() * sizeof(uint32_t), fragmentShader.data());

    // All Vulkan objects declared in triangle.yaml are alive here and are
    // destroyed automatically in reverse creation order. The capability
    // selections are available as capabilities.surfaceFormat,
    // capabilities.presentMode and capabilities.surfaceCapabilities.

    return 0;
}
