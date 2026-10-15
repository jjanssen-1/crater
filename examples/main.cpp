// Example usage of the generated triangle.hpp.
//
// Build:
//   crater generate examples/triangle.yaml -o examples/generated/triangle.hpp
//   g++ -std=c++17 -I examples/generated -I <vulkan-headers>/include examples/main.cpp

#include <cstdint>
#include <vector>

#include "triangle.hpp"

int main() {
    // In a real application these come from a window system binding and a
    // file loader; they are exactly the values crater asks the caller for.
    VkExtent2D extent{800, 600};
    VkFormat format = VK_FORMAT_B8G8R8A8_SRGB;

    std::vector<uint32_t> vertexShader = {0x07230203, 0x00010000, 0x0008000b};
    std::vector<uint32_t> fragmentShader = {0x07230203, 0x00010000, 0x0008000b};

    // Physical device selection is application logic; here we simply
    // pick one (enumeration is not object creation, so it stays explicit).
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;

    VulkanInstance instance;
    VulkanDevice device(physicalDevice);
    TrianglePipeline pipeline(
        device,
        vertexShader.size() * sizeof(uint32_t), vertexShader.data(),
        fragmentShader.size() * sizeof(uint32_t), fragmentShader.data(),
        nullptr,
        extent,
        format);

    // All Vulkan objects declared in triangle.yaml are alive here:
    //   instance.instance, device.device, pipeline.vertexModule,
    //   pipeline.fragmentModule, pipeline.layout, pipeline.renderPass,
    //   pipeline.pipeline, pipeline.swapchain, pipeline.imageAvailable,
    //   pipeline.renderFinished, pipeline.inFlight
    // and are destroyed automatically in reverse creation order.

    return 0;
}
