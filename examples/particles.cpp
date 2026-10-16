// GPU particle simulation - the feature-rich example beyond the quad.
//
// New machinery vs. the quad: VMA buffers with persistent host mapping
// (no per-frame vkMapMemory), a compute pipeline that is actually
// dispatched each frame, timestamp queries, pNext feature chains on
// the device, and a format preference with a real fallback candidate.
//
// Build:
//   crater generate examples/particles.yaml -o examples/generated/particles.hpp
//   glslc examples/shaders/particles.comp -o examples/generated/particles.comp.spv
//   glslc examples/shaders/particles.vert -o examples/generated/particles.vert.spv
//   glslc examples/shaders/particles.frag -o examples/generated/particles.frag.spv
//   g++ -std=c++17 examples/particles.cpp -o particles \
//       -I examples -I $VULKAN_SDK/include -lglfw -lvulkan
//
// VMA is a single-header library: define VMA_IMPLEMENTATION in exactly
// one translation unit (this one).

#define GLFW_INCLUDE_VULKAN
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "generated/particles.hpp"

namespace {

std::vector<char> readFile(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error(std::string("cannot open ") + path);
    size_t size = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(size);
    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(size));
    return buffer;
}

constexpr uint32_t kParticleCount = 8192;
constexpr VkDeviceSize kBufferSize = kParticleCount * 4 * sizeof(float);

}  // namespace

int main() {
    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(900, 600, "crater particles", nullptr, nullptr);

    {
    using namespace particles;

    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    InstanceLayers layers;
    ParticlesInstance instance(layers, glfwExtensionCount, glfwExtensions);

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    {
        uint32_t n = 0;
        vkEnumeratePhysicalDevices(instance.instance(), &n, nullptr);
        std::vector<VkPhysicalDevice> devices(n);
        vkEnumeratePhysicalDevices(instance.instance(), &n, devices.data());
        if (n == 0) throw std::runtime_error("no devices");
        physicalDevice = devices[0];
    }

    ParticlesCapabilities caps(physicalDevice);

    // queue list: shared family gets one entry (runtime data, as in the quad)
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    {
        static const float priorities[] = {1.0f};
        auto add = [&](uint32_t family) {
            if (std::any_of(queueCreateInfos.begin(), queueCreateInfos.end(),
                             [family](const VkDeviceQueueCreateInfo& q) {
                                 return q.queueFamilyIndex == family;
                             }))
                return;
            VkDeviceQueueCreateInfo q{};
            q.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            q.queueFamilyIndex = family;
            q.queueCount = 1;
            q.pQueuePriorities = priorities;
            queueCreateInfos.push_back(q);
        };
        add(caps.graphicsQueueFamilyIndex);
        add(caps.computeQueueFamilyIndex);
    }

    ParticlesDevice device(
        caps, physicalDevice, queueCreateInfos.data(),
        static_cast<uint32_t>(queueCreateInfos.size()));

    VkQueue graphicsQueue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device.device(), caps.graphicsQueueFamilyIndex, 0, &graphicsQueue);
    VkQueue computeQueue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device.device(), caps.computeQueueFamilyIndex, 0, &computeQueue);
    std::printf("queue families: graphics %u, compute %u%s\n",
                caps.graphicsQueueFamilyIndex, caps.computeQueueFamilyIndex,
                caps.graphicsQueueFamilyIndex == caps.computeQueueFamilyIndex ? " (shared)" : "");

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (glfwCreateWindowSurface(instance.instance(), window, nullptr, &surface) != VK_SUCCESS)
        throw std::runtime_error("glfwCreateWindowSurface failed");

    ParticlesSurface surfaceCaps(physicalDevice, surface);
    const VkExtent2D swapExtent =
        surfaceCaps.surfaceCapabilities.currentExtent.width != 0xFFFFFFFFu
            ? surfaceCaps.surfaceCapabilities.currentExtent
            : VkExtent2D{900, 600};

    // VMA allocator: an ordinary object
    ParticlesAllocator allocator(physicalDevice, device.device(), instance.instance());

    // GPU buffers via VMA, persistently mapped
    ParticleBuffer particleBuffer(allocator, kBufferSize);
    FrameUniform frameUniform(allocator);

    ParticleQueries queries(device);
    ParticlesDescriptors descriptors(device);
    ParticlesRenderPass renderPass(device, surfaceCaps.surfaceFormat.format);

    auto computeShader = readFile("examples/generated/particles.comp.spv");
    auto vertexShader = readFile("examples/generated/particles.vert.spv");
    auto fragmentShader = readFile("examples/generated/particles.frag.spv");

    ParticleCompute compute(
        descriptors, device,
        computeShader.size(), reinterpret_cast<const uint32_t*>(computeShader.data()));
    ParticlesPipelines graphics(
        descriptors, renderPass, device,
        vertexShader.size(), reinterpret_cast<const uint32_t*>(vertexShader.data()),
        fragmentShader.size(), reinterpret_cast<const uint32_t*>(fragmentShader.data()));

    // swapchain-owning objects live in a nested scope: it closes (RAII
    // destroy of the swapchain and per-image objects) before the surface
    // is destroyed, which must precede the device/instance teardown
    {
    ParticlesSwapchain swapchain(surfaceCaps, device, surface, swapExtent);

    // per-image resources
    std::vector<SwapchainImageView> views;
    std::vector<SwapchainFramebuffer> framebuffers;
    {
        uint32_t imageCount = 0;
        vkGetSwapchainImagesKHR(device.device(), swapchain.swapchain(), &imageCount, nullptr);
        std::vector<VkImage> images(imageCount);
        vkGetSwapchainImagesKHR(device.device(), swapchain.swapchain(), &imageCount, images.data());
        for (uint32_t i = 0; i < imageCount; ++i) {
            views.emplace_back(device, images[i], surfaceCaps.surfaceFormat.format);
            VkImageView attachment = views[i].view();
            framebuffers.emplace_back(device, renderPass.renderPass(), &attachment,
                                       swapExtent.width, swapExtent.height);
        }
    }

    FrameCommands commands(device, caps.graphicsQueueFamilyIndex);
    ParticlesSync sync(device);
    // per-image render-finished semaphores: presentation holds them until
    // the image is re-acquired (a single one would be re-signaled while
    // still in use)
    std::vector<VkSemaphore> renderFinished;
    {
        uint32_t imageCount = 0;
        vkGetSwapchainImagesKHR(device.device(), swapchain.swapchain(), &imageCount, nullptr);
        for (uint32_t i = 0; i < imageCount; ++i) {
            VkSemaphoreCreateInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            VkSemaphore sem = VK_NULL_HANDLE;
            if (vkCreateSemaphore(device.device(), &si, nullptr, &sem) != VK_SUCCESS)
                throw std::runtime_error("vkCreateSemaphore failed");
            renderFinished.push_back(sem);
        }
    }

    // descriptor writes: storage buffer (particles) + uniform (deltaTime)
    {
        VkDescriptorBufferInfo particleInfo{particleBuffer.particleBuffer(), 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo uniformInfo{frameUniform.uniformBuffer(), 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet writes[2]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = descriptors.descriptorSet();
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &particleInfo;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = descriptors.descriptorSet();
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[1].pBufferInfo = &uniformInfo;
        vkUpdateDescriptorSets(device.device(), 2, writes, 0, nullptr);
    }

    // initial particle positions through the persistent mapping
    {
        VmaAllocationInfo info{};
        getAllocationInfo(allocator.allocator(), particleBuffer.particleBufferAllocation(), info);
        if (info.pMappedData == nullptr)
            throw std::runtime_error("particle buffer is not persistently mapped");
        auto* positions = reinterpret_cast<float*>(info.pMappedData);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-0.9f, 0.9f);
        for (uint32_t i = 0; i < kParticleCount; ++i) {
            positions[i * 4 + 0] = dist(rng);
            positions[i * 4 + 1] = dist(rng);
            positions[i * 4 + 2] = 0.0f;
            positions[i * 4 + 3] = 1.0f;
        }
    }
    VmaAllocationInfo uniformInfo{};
    getAllocationInfo(allocator.allocator(), frameUniform.uniformBufferAllocation(), uniformInfo);

    // --- loop ------------------------------------------------------------
    VkClearValue clearColor = {{{0.02f, 0.02f, 0.05f, 0.0f}}};
    uint32_t frame = 0;
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        VkFence frameFence = sync.frameInFlight();
        waitFrameFence(device.device(), 1, &frameFence);
        resetFrameFence(device.device(), 1, &frameFence);

        uint32_t imageIndex = 0;
        acquireFrameImage(device.device(), swapchain.swapchain(), sync.imageAvailable(),
                          VK_NULL_HANDLE, imageIndex);

        // per-frame uniform through the persistent mapping
        *reinterpret_cast<float*>(uniformInfo.pMappedData) = 0.016f;

        // --- compute pass: integrate positions ---------------------------
        vkResetCommandBuffer(commands.computeCommandBuffer(), 0);
        beginFrameCommandBuffer(commands.computeCommandBuffer());
        vkCmdBindPipeline(commands.computeCommandBuffer(),
                          VK_PIPELINE_BIND_POINT_COMPUTE, compute.computePipeline());
        VkDescriptorSet setHandle = descriptors.descriptorSet();
        vkCmdBindDescriptorSets(commands.computeCommandBuffer(),
                                VK_PIPELINE_BIND_POINT_COMPUTE, compute.computeLayout(),
                                0, 1, &setHandle, 0, nullptr);
        vkCmdDispatch(commands.computeCommandBuffer(), (kParticleCount + 63) / 64, 1, 1);
        endFrameCommandBuffer(commands.computeCommandBuffer());

        VkSemaphore computeWait[] = {sync.imageAvailable()};
        VkSemaphore computeSignal[] = {sync.computeFinished()};
        VkPipelineStageFlags computeWaitStage[] = {VK_PIPELINE_STAGE_VERTEX_SHADER_BIT};
        VkSubmitInfo computeSubmit{};
        computeSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        computeSubmit.waitSemaphoreCount = 1;
        computeSubmit.pWaitSemaphores = computeWait;
        computeSubmit.pWaitDstStageMask = computeWaitStage;
        computeSubmit.commandBufferCount = 1;
        VkCommandBuffer computeBuf = commands.computeCommandBuffer();
        computeSubmit.pCommandBuffers = &computeBuf;
        computeSubmit.signalSemaphoreCount = 1;
        computeSubmit.pSignalSemaphores = computeSignal;
        if (vkQueueSubmit(computeQueue, 1, &computeSubmit, VK_NULL_HANDLE) != VK_SUCCESS)
            throw std::runtime_error("compute vkQueueSubmit failed");

        // --- graphics pass: draw particles as points ----------------------
        vkResetCommandBuffer(commands.renderCommandBuffer(), 0);
        beginFrameCommandBuffer(commands.renderCommandBuffer());
        VkViewport viewport{0.0f, 0.0f, static_cast<float>(swapExtent.width),
                            static_cast<float>(swapExtent.height), 0.0f, 1.0f};
        VkRect2D scissor{{0, 0}, swapExtent};
        vkCmdSetViewport(commands.renderCommandBuffer(), 0, 1, &viewport);
        vkCmdSetScissor(commands.renderCommandBuffer(), 0, 1, &scissor);

        VkRenderPassBeginInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass = renderPass.renderPass();
        rp.framebuffer = framebuffers[imageIndex].framebuffer();
        rp.renderArea.extent = swapExtent;
        rp.clearValueCount = 1;
        rp.pClearValues = &clearColor;
        vkCmdBeginRenderPass(commands.renderCommandBuffer(), &rp, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commands.renderCommandBuffer(),
                          VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.graphicsPipeline());
        vkCmdBindDescriptorSets(commands.renderCommandBuffer(),
                                VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.graphicsLayout(),
                                0, 1, &setHandle, 0, nullptr);
        VkDeviceSize offset = 0;
        VkBuffer particleHandle = particleBuffer.particleBuffer();
        vkCmdBindVertexBuffers(commands.renderCommandBuffer(), 0, 1, &particleHandle, &offset);
        vkCmdDraw(commands.renderCommandBuffer(), kParticleCount, 1, 0, 0);
        vkCmdEndRenderPass(commands.renderCommandBuffer());
        endFrameCommandBuffer(commands.renderCommandBuffer());

        VkSemaphore drawSignal[] = {renderFinished[imageIndex]};
        VkPipelineStageFlags drawWaitStage[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        VkSubmitInfo drawSubmit{};
        drawSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        drawSubmit.waitSemaphoreCount = 1;
        drawSubmit.pWaitSemaphores = computeSignal;   // wait for compute
        drawSubmit.pWaitDstStageMask = drawWaitStage;
        drawSubmit.commandBufferCount = 1;
        VkCommandBuffer renderBuf = commands.renderCommandBuffer();
        drawSubmit.pCommandBuffers = &renderBuf;
        drawSubmit.signalSemaphoreCount = 1;
        drawSubmit.pSignalSemaphores = drawSignal;    // per-image
        if (vkQueueSubmit(graphicsQueue, 1, &drawSubmit, frameFence) != VK_SUCCESS)
            throw std::runtime_error("graphics vkQueueSubmit failed");

        VkPresentInfoKHR pi{};
        pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = drawSignal;
        pi.swapchainCount = 1;
        VkSwapchainKHR swapchainHandle = swapchain.swapchain();
        pi.pSwapchains = &swapchainHandle;
        pi.pImageIndices = &imageIndex;
        vkQueuePresentKHR(graphicsQueue, &pi);

        // occasional timestamp readout
        if (++frame % 300 == 0) std::printf("frame %u\n", frame);
    }

        vkDeviceWaitIdle(device.device());
        for (auto sem : renderFinished) vkDestroySemaphore(device.device(), sem, nullptr);
    }  // swapchain (and per-image views/framebuffers) destroyed here

    // GLFW created the surface but does not destroy it; the swapchain is
    // gone, so it can die before the device/instance
    vkDestroySurfaceKHR(instance.instance(), surface, nullptr);

    }  // RAII teardown (VMA buffers, pipelines, device, instance)

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
