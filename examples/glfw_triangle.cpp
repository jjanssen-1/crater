// Full Vulkan triangle with GLFW, using crater-generated classes.
//
// Build (see examples/README.md for details):
//   crater generate examples/glfw_triangle.yaml -o examples/generated/glfw_triangle.hpp
//   glslc examples/shaders/triangle.vert -o examples/generated/triangle.vert.spv
//   glslc examples/shaders/triangle.frag -o examples/generated/triangle.frag.spv
//   g++ -std=c++17 examples/glfw_triangle.cpp -o triangle \
//       -I examples/generated -I <vulkan-headers>/include \
//       -lglfw -lvulkan

#define GLFW_INCLUDE_VULKAN
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "glfw_triangle.hpp"

static std::vector<char> readFile(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error(std::string("cannot open ") + path);
    size_t size = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(size);
    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(size));
    return buffer;
}

static VkPhysicalDevice pickPhysicalDevice(VkInstance instance) {
    // Keep it simple: the first device. A real application would prefer a
    // discrete GPU or ask the user; that choice stays explicit with crater.
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count == 0) throw std::runtime_error("no Vulkan device found");
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());
    return devices[0];
}

// ---------------------------------------------------------------------------
// Swapchain image views and framebuffers.
//
// crater does not own these yet: they depend on the number of swapchain
// images, which is only known at runtime (vkGetSwapchainImagesKHR).
// Expressing "one framebuffer per swapchain image" declaratively is a
// planned capability; for now this part stays explicit, like command
// buffer recording.
// ---------------------------------------------------------------------------
struct FrameResources {
    std::vector<VkImage> images;
    std::vector<VkImageView> views;
    std::vector<VkFramebuffer> framebuffers;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

    void create(VkDevice device, const TrianglePipeline& pipeline, VkFormat pipelineFormat, VkExtent2D extent, uint32_t queueFamilyIndex) {
        uint32_t imageCount = 0;
        vkGetSwapchainImagesKHR(device, pipeline.swapchain, &imageCount, nullptr);
        images.resize(imageCount);
        vkGetSwapchainImagesKHR(device, pipeline.swapchain, &imageCount, images.data());

        views.resize(imageCount);
        framebuffers.resize(imageCount);
        for (uint32_t i = 0; i < imageCount; ++i) {
            VkImageViewCreateInfo viewInfo{};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = images[i];
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = pipelineFormat;
            viewInfo.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                   VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
            viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            if (vkCreateImageView(device, &viewInfo, nullptr, &views[i]) != VK_SUCCESS)
                throw std::runtime_error("vkCreateImageView failed");

            VkFramebufferCreateInfo fbInfo{};
            fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            fbInfo.renderPass = pipeline.renderPass;
            fbInfo.attachmentCount = 1;
            fbInfo.pAttachments = &views[i];
            fbInfo.width = extent.width;
            fbInfo.height = extent.height;
            fbInfo.layers = 1;
            if (vkCreateFramebuffer(device, &fbInfo, nullptr, &framebuffers[i]) != VK_SUCCESS)
                throw std::runtime_error("vkCreateFramebuffer failed");
        }

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueFamilyIndex;
        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
            throw std::runtime_error("vkCreateCommandPool failed");

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
    }

    void destroy(VkDevice device) {
        for (auto fb : framebuffers) vkDestroyFramebuffer(device, fb, nullptr);
        for (auto view : views) vkDestroyImageView(device, view, nullptr);
        if (commandPool != VK_NULL_HANDLE) vkDestroyCommandPool(device, commandPool, nullptr);
    }
};

int main() {
    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(800, 600, "crater triangle", nullptr, nullptr);

    // --- crater takes over from here -------------------------------------

    // GLFW knows which instance extensions the window system needs.
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    // The InstanceLayers capability class (generated) checks whether the
    // validation layer is installed; the instance consumes its list.
    // The layer is declared 'optional' in glfw_triangle.yaml; declaring it
    // 'required' would make this constructor throw a clear error instead.
    InstanceLayers layers;
    if (!layers.hasKHRONOSValidation) {
        std::fprintf(
            stderr,
            "note: VK_LAYER_KHRONOS_validation is not installed; validation "
            "is disabled. Install your distribution's vulkan-validation "
            "layers (e.g. 'sudo apt install vulkan-validationlayers') to "
            "get validation output.\n");
    }

    VulkanInstance vulkanInstance(layers, glfwExtensionCount, glfwExtensions);

    VkPhysicalDevice physicalDevice = pickPhysicalDevice(vulkanInstance.instance);

    // Selects a queue family (graphics required, compute preferred) and
    // requires samplerAnisotropy. All per glfw_triangle.yaml.
    DeviceCapabilities deviceCapabilities(physicalDevice);

    // Device creation consumes the selected queue family index and the
    // queried features.
    VulkanDevice device(deviceCapabilities, physicalDevice);

    // The surface itself is platform specific; GLFW creates it.
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (glfwCreateWindowSurface(vulkanInstance.instance, window, nullptr, &surface) != VK_SUCCESS)
        throw std::runtime_error("glfwCreateWindowSurface failed");

    // Queries formats/present modes/capabilities and selects per the
    // preference lists in glfw_triangle.yaml.
    SurfaceCapabilities capabilities(physicalDevice, surface);

    auto vertexShader = readFile("examples/generated/triangle.vert.spv");
    auto fragmentShader = readFile("examples/generated/triangle.frag.spv");

    // The swapchain extent is an application choice: the queried
    // currentExtent is 0xFFFFFFFF ("unknown") on Wayland, so clamp the
    // window size into [minImageExtent, maxImageExtent] instead.
    VkExtent2D swapchainExtent;
    {
        VkSurfaceCapabilitiesKHR caps = capabilities.surfaceCapabilities;
        if (caps.currentExtent.width != 0xFFFFFFFFu) {
            swapchainExtent = caps.currentExtent;
        } else {
            int fbWidth = 0, fbHeight = 0;
            glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
            swapchainExtent.width = std::clamp(
                static_cast<uint32_t>(fbWidth), caps.minImageExtent.width, caps.maxImageExtent.width);
            swapchainExtent.height = std::clamp(
                static_cast<uint32_t>(fbHeight), caps.minImageExtent.height, caps.maxImageExtent.height);
        }
    }

    // Creates shader modules, pipeline layout, render pass, graphics
    // pipeline, swapchain and synchronization objects; consumes the
    // selected surface format/present mode and the chosen extent.
    TrianglePipeline triangle(
        capabilities,
        device,
        vertexShader.size(), reinterpret_cast<const uint32_t*>(vertexShader.data()),
        fragmentShader.size(), reinterpret_cast<const uint32_t*>(fragmentShader.data()),
        swapchainExtent);

    // Queues are retrieved, not created; the family index came from the
    // capability selection above.
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device.device, deviceCapabilities.graphicsQueueFamilyIndex, 0, &graphicsQueue);

    FrameResources frames;
    frames.create(device.device, triangle, capabilities.surfaceFormat.format, swapchainExtent, deviceCapabilities.graphicsQueueFamilyIndex);

    // --- render loop: the imperative part stays imperative ----------------
    VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        vkWaitForFences(device.device, 1, &triangle.inFlight, VK_TRUE, UINT64_MAX);
        vkResetFences(device.device, 1, &triangle.inFlight);

        uint32_t imageIndex = 0;
        vkAcquireNextImageKHR(device.device, triangle.swapchain, UINT64_MAX,
                              triangle.imageAvailable, VK_NULL_HANDLE, &imageIndex);

        vkResetCommandBuffer(frames.commandBuffer, 0);
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(frames.commandBuffer, &beginInfo);

        // viewport/scissor are dynamic pipeline states (see the yaml)
        uint32_t width = swapchainExtent.width;
        uint32_t height = swapchainExtent.height;
        VkViewport viewport{0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f};
        VkRect2D scissor{{0, 0}, {width, height}};
        vkCmdSetViewport(frames.commandBuffer, 0, 1, &viewport);
        vkCmdSetScissor(frames.commandBuffer, 0, 1, &scissor);

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = triangle.renderPass;
        renderPassInfo.framebuffer = frames.framebuffers[imageIndex];
        renderPassInfo.renderArea.extent = swapchainExtent;
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        vkCmdBeginRenderPass(frames.commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(frames.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, triangle.pipeline);
        vkCmdDraw(frames.commandBuffer, 3, 1, 0, 0);
        vkCmdEndRenderPass(frames.commandBuffer);
        if (vkEndCommandBuffer(frames.commandBuffer) != VK_SUCCESS)
            throw std::runtime_error("vkEndCommandBuffer failed");

        VkSemaphore waitSemaphores[] = {triangle.imageAvailable};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        VkSemaphore signalSemaphores[] = {triangle.renderFinished};
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &frames.commandBuffer;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;
        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, triangle.inFlight) != VK_SUCCESS)
            throw std::runtime_error("vkQueueSubmit failed");

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &triangle.swapchain;
        presentInfo.pImageIndices = &imageIndex;
        vkQueuePresentKHR(graphicsQueue, &presentInfo);
    }

    vkDeviceWaitIdle(device.device);
    frames.destroy(device.device);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
