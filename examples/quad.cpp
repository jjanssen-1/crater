// Feature-rich crater example: an animated textured quad.
//
// Compared to the triangle this exercises: multi-selection queue
// families (graphics + transfer), pNext feature queries, descriptors
// (pool + layout + set), a uniform buffer with bound memory, vertex +
// index buffers uploaded through a staging buffer, a texture (image +
// view + sampler), a render pass, graphics AND compute pipelines,
// cross-class references (the pipeline layout consumes the descriptor
// set layout), and the per-frame call functions.
//
// Build:
//   crater generate examples/quad.yaml -o examples/generated/quad.hpp
//   g++ -std=c++17 examples/quad.cpp -o quad \
//       -I examples -I $VULKAN_SDK/include -lglfw -lvulkan

#define GLFW_INCLUDE_VULKAN
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "generated/quad.hpp"

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

uint32_t findMemoryType(VkPhysicalDevice pd, uint32_t typeBits,
                        VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mem;
    vkGetPhysicalDeviceMemoryProperties(pd, &mem);
    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) &&
            (mem.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    throw std::runtime_error("no matching memory type");
}

VkDeviceMemory allocate(VkDevice dev, VkPhysicalDevice pd, VkMemoryRequirements reqs,
                        VkMemoryPropertyFlags props) {
    VkMemoryAllocateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    info.allocationSize = reqs.size;
    info.memoryTypeIndex = findMemoryType(pd, reqs.memoryTypeBits, props);
    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (vkAllocateMemory(dev, &info, nullptr, &memory) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateMemory failed");
    return memory;
}

// per-image resources (same pattern as the triangle)
class FrameResources {
public:
    FrameResources(const quadvk::QuadDevice& device,
                   const quadvk::QuadDeviceCapabilities& caps,
                   VkSwapchainKHR swapchain, VkRenderPass renderPass,
                   VkFormat format, VkExtent2D extent)
        : commands(device, caps.graphicsQueueFamilyIndex), device_(device.device) {
        uint32_t imageCount = 0;
        vkGetSwapchainImagesKHR(device_, swapchain, &imageCount, nullptr);
        std::vector<VkImage> images(imageCount);
        vkGetSwapchainImagesKHR(device_, swapchain, &imageCount, images.data());
        for (uint32_t i = 0; i < imageCount; ++i) {
            views.emplace_back(device, images[i], format);
            VkImageView attachment = views[i].view;
            framebuffers.emplace_back(device, renderPass, &attachment,
                                       extent.width, extent.height);
        }
    }
    FrameResources(const FrameResources&) = delete;
    FrameResources& operator=(const FrameResources&) = delete;

    VkCommandBuffer commandBuffer() const { return commands.commandBuffer; }
    VkFramebuffer framebuffer(uint32_t i) const { return framebuffers[i].framebuffer; }
    uint32_t imageCount() const { return static_cast<uint32_t>(framebuffers.size()); }

private:
    quadvk::FrameCommands commands;
    std::vector<quadvk::SwapchainImageView> views;
    std::vector<quadvk::SwapchainFramebuffer> framebuffers;
    VkDevice device_;
};

}  // namespace

int main() {
    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(800, 600, "crater quad", nullptr, nullptr);

    {
    using namespace quadvk;

    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    InstanceLayers layers;
    QuadInstance instance(layers, glfwExtensionCount, glfwExtensions);

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    {
        uint32_t n = 0;
        vkEnumeratePhysicalDevices(instance.instance, &n, nullptr);
        std::vector<VkPhysicalDevice> devices(n);
        vkEnumeratePhysicalDevices(instance.instance, &n, devices.data());
        if (n == 0) throw std::runtime_error("no devices");
        physicalDevice = devices[0];
    }

    // capabilities: extensions, graphics + transfer families (one
    // shared query), core + 1.1/1.2 features via pNext
    QuadDeviceCapabilities caps(physicalDevice);
    // the queue list is runtime data: duplicate family entries are
    // invalid, so a shared graphics/transfer family gets ONE entry
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    {
        static const float priorities[] = {1.0f};
        VkDeviceQueueCreateInfo graphics{};
        graphics.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        graphics.queueFamilyIndex = caps.graphicsQueueFamilyIndex;
        graphics.queueCount = 1;
        graphics.pQueuePriorities = priorities;
        queueCreateInfos.push_back(graphics);
        if (caps.transferQueueFamilyIndex != caps.graphicsQueueFamilyIndex) {
            VkDeviceQueueCreateInfo transfer{};
            transfer.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            transfer.queueFamilyIndex = caps.transferQueueFamilyIndex;
            transfer.queueCount = 1;
            transfer.pQueuePriorities = priorities;
            queueCreateInfos.push_back(transfer);
        }
    }

    QuadDevice device(caps, physicalDevice, queueCreateInfos.data(),
                       static_cast<uint32_t>(queueCreateInfos.size()));

    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue transferQueue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device.device, caps.graphicsQueueFamilyIndex, 0, &graphicsQueue);
    vkGetDeviceQueue(device.device, caps.transferQueueFamilyIndex, 0, &transferQueue);
    std::printf("queue families: graphics %u, transfer %u%s\n",
                caps.graphicsQueueFamilyIndex, caps.transferQueueFamilyIndex,
                caps.graphicsQueueFamilyIndex == caps.transferQueueFamilyIndex
                    ? " (shared)" : " (dedicated transfer)");

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (glfwCreateWindowSurface(instance.instance, window, nullptr, &surface) != VK_SUCCESS)
        throw std::runtime_error("glfwCreateWindowSurface failed");

    QuadSurfaceCapabilities surfaceCaps(physicalDevice, surface);

    // --- geometry + uniform + texture (memory is runtime logic) -----
    QuadUniform uniform(device);
    QuadGeometry geometry(device);
    QuadTexture texture(device, VkExtent3D{64, 64, 1});
    QuadSampler sampler(device);

    // bind memory: uniform host-visible, geometry + texture device-local
    VkMemoryRequirements reqs{};
    vkGetBufferMemoryRequirements(device.device, uniform.uniformBuffer, &reqs);
    VkDeviceMemory uniformMemory = allocate(device.device, physicalDevice, reqs,
                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkBindBufferMemory(device.device, uniform.uniformBuffer, uniformMemory, 0);

    vkGetBufferMemoryRequirements(device.device, geometry.stagingBuffer, &reqs);
    VkDeviceMemory stagingMemory = allocate(device.device, physicalDevice, reqs,
                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkBindBufferMemory(device.device, geometry.stagingBuffer, stagingMemory, 0);

    std::vector<VkDeviceMemory> geometryMemories;
    for (auto buf : {geometry.vertexBuffer, geometry.indexBuffer}) {
        vkGetBufferMemoryRequirements(device.device, buf, &reqs);
        VkDeviceMemory m = allocate(device.device, physicalDevice, reqs,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vkBindBufferMemory(device.device, buf, m, 0);
        geometryMemories.push_back(m);
    }
    vkGetImageMemoryRequirements(device.device, texture.textureImage, &reqs);
    VkDeviceMemory textureMemory = allocate(device.device, physicalDevice, reqs,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkBindImageMemory(device.device, texture.textureImage, textureMemory, 0);
    // views require bound memory: create the view after the bind
    QuadTextureView textureView(device, texture.textureImage);

    // checkerboard pixel data (8x8 checkerboard scaled to 64x64)
    std::vector<uint8_t> texelData(64 * 64 * 4);
    for (uint32_t y = 0; y < 64; ++y)
        for (uint32_t x = 0; x < 64; ++x) {
            uint8_t c = ((x / 8) + (y / 8)) % 2 ? 230 : 40;
            size_t o = (y * 64 + x) * 4;
            texelData[o] = c; texelData[o + 1] = c; texelData[o + 2] = c; texelData[o + 3] = 255;
        }

    // upload geometry (pos2 + uv2 per vertex, 6 indices) and the texture
    // via the staging buffer with one-time copy commands on the graphics
    // queue; the transfer queue would be the production choice for this,
    // kept simple here (single submission before the loop)
    struct Vertex { float x, y, u, v; };
    const Vertex vertices[4] = {
        {-0.5f, -0.5f, 0.0f, 0.0f},
        { 0.5f, -0.5f, 1.0f, 0.0f},
        { 0.5f,  0.5f, 1.0f, 1.0f},
        {-0.5f,  0.5f, 0.0f, 1.0f},
    };
    const uint16_t indices[6] = {0, 1, 2, 0, 2, 3};
    {
        // stage: vertices then indices then texels (staging is 64 bytes;
        // uploads happen one at a time, waiting for the GPU between them)
        struct Upload { const void* data; VkDeviceSize size; VkBuffer dst; };
        const Upload uploads[] = {
            {vertices, sizeof(vertices), geometry.vertexBuffer},
            {indices, sizeof(indices), geometry.indexBuffer},
        };
        VkCommandPool uploadPool = VK_NULL_HANDLE;
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = caps.graphicsQueueFamilyIndex;
        vkCreateCommandPool(device.device, &poolInfo, nullptr, &uploadPool);
        for (const auto& up : uploads) {
            if (up.size > 64) throw std::runtime_error("staging buffer too small");
            void* mapped = nullptr;
            vkMapMemory(device.device, stagingMemory, 0, up.size, 0, &mapped);
            std::memcpy(mapped, up.data, static_cast<size_t>(up.size));
            vkUnmapMemory(device.device, stagingMemory);

            VkCommandBufferAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            ai.commandPool = uploadPool;
            ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = 1;
            VkCommandBuffer cmd = VK_NULL_HANDLE;
            vkAllocateCommandBuffers(device.device, &ai, &cmd);
            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &bi);
            VkBufferCopy region{0, 0, up.size};
            vkCmdCopyBuffer(cmd, geometry.stagingBuffer, up.dst, 1, &region);
            vkEndCommandBuffer(cmd);
            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cmd;
            vkQueueSubmit(graphicsQueue, 1, &si, VK_NULL_HANDLE);
            vkQueueWaitIdle(graphicsQueue);
            vkFreeCommandBuffers(device.device, uploadPool, 1, &cmd);
        }
        vkDestroyCommandPool(device.device, uploadPool, nullptr);
    }

    // upload the checkerboard texture via a dedicated staging buffer
    {
        VkBufferCreateInfo sbInfo{};
        sbInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        sbInfo.size = texelData.size();
        sbInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VkBuffer texStaging = VK_NULL_HANDLE;
        vkCreateBuffer(device.device, &sbInfo, nullptr, &texStaging);
        VkMemoryRequirements tr{};
        vkGetBufferMemoryRequirements(device.device, texStaging, &tr);
        VkDeviceMemory texStagingMem = allocate(device.device, physicalDevice, tr,
                                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkBindBufferMemory(device.device, texStaging, texStagingMem, 0);
        void* mapped = nullptr;
        vkMapMemory(device.device, texStagingMem, 0, texelData.size(), 0, &mapped);
        std::memcpy(mapped, texelData.data(), texelData.size());
        vkUnmapMemory(device.device, texStagingMem);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = caps.graphicsQueueFamilyIndex;
        VkCommandPool pool = VK_NULL_HANDLE;
        vkCreateCommandPool(device.device, &poolInfo, nullptr, &pool);
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(device.device, &ai, &cmd);
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bi);

        // image layout transition then copy
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = texture.textureImage;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {64, 64, 1};
        vkCmdCopyBufferToImage(cmd, texStaging, texture.textureImage,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        vkEndCommandBuffer(cmd);
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        vkQueueSubmit(graphicsQueue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);
        vkDestroyCommandPool(device.device, pool, nullptr);
        vkDestroyBuffer(device.device, texStaging, nullptr);
        vkFreeMemory(device.device, texStagingMem, nullptr);
    }

    // descriptors
    QuadDescriptors descriptors(device);

    // render pass, pipelines (needs the descriptor set layout)
    QuadRenderPass renderPass(device, surfaceCaps.surfaceFormat.format);

    auto vertexShader = readFile("examples/generated/quad.vert.spv");
    auto fragmentShader = readFile("examples/generated/quad.frag.spv");
    auto computeShader = readFile("examples/generated/quad.comp.spv");

    QuadPipelines pipelines(descriptors, device,
                             vertexShader.size(), reinterpret_cast<const uint32_t*>(vertexShader.data()),
                             fragmentShader.size(), reinterpret_cast<const uint32_t*>(fragmentShader.data()),
                             computeShader.size(), reinterpret_cast<const uint32_t*>(computeShader.data()),
                             renderPass.renderPass);

    // swapchain-owning objects live in a nested scope: it closes (RAII
    // destroy of the swapchain) before the surface is destroyed, which
    // must precede the device/instance teardown
    {
    QuadSwapchain swapchain(surfaceCaps, device, surface,
                            surfaceCaps.surfaceCapabilities.currentExtent.width != 0xFFFFFFFFu
                                ? surfaceCaps.surfaceCapabilities.currentExtent
                                : VkExtent2D{800, 600},
                            surfaceCaps.surfaceFormat.format);

    QuadSync sync(device);
    const VkExtent2D frameExtent =
        surfaceCaps.surfaceCapabilities.currentExtent.width != 0xFFFFFFFFu
            ? surfaceCaps.surfaceCapabilities.currentExtent
            : VkExtent2D{800, 600};
    FrameResources frames(device, caps, swapchain.swapchain, renderPass.renderPass,
                          surfaceCaps.surfaceFormat.format, frameExtent);
    // per-image render-finished semaphores (presentation holds them until
    // the image is re-acquired; one acquire semaphore in QuadSync)
    std::vector<VkSemaphore> renderFinished;
    for (uint32_t i = 0; i < frames.imageCount(); ++i) {
        VkSemaphoreCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkSemaphore sem = VK_NULL_HANDLE;
        if (vkCreateSemaphore(device.device, &si, nullptr, &sem) != VK_SUCCESS)
            throw std::runtime_error("vkCreateSemaphore failed");
        renderFinished.push_back(sem);
    }

    // descriptor set write: UBO + combined image sampler
    {
        VkDescriptorBufferInfo bufferInfo{uniform.uniformBuffer, 0, VK_WHOLE_SIZE};
        VkDescriptorImageInfo imageInfo{sampler.sampler, textureView.textureView,
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet writes[2]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = descriptors.descriptorSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].pBufferInfo = &bufferInfo;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = descriptors.descriptorSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].pImageInfo = &imageInfo;
        vkUpdateDescriptorSets(device.device, 2, writes, 0, nullptr);
    }

    // --- render loop ---------------------------------------------------
    const VkExtent2D renderExtent =
        surfaceCaps.surfaceCapabilities.currentExtent.width != 0xFFFFFFFFu
            ? surfaceCaps.surfaceCapabilities.currentExtent
            : VkExtent2D{800, 600};
    VkClearValue clearColor = {{{0.02f, 0.02f, 0.05f, 0.0f}}};
    uint32_t frame = 0;
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        waitFrameFence(device.device, 1, &sync.frameInFlight);
        resetFrameFence(device.device, 1, &sync.frameInFlight);

        uint32_t imageIndex = 0;
        acquireFrameImage(device.device, swapchain.swapchain, sync.imageAvailable,
                          VK_NULL_HANDLE, imageIndex);

        // per-frame UBO: the shader reads THREE mat4s (model, view,
        // proj - 192 bytes); writing fewer leaves the rest zero and
        // collapses every vertex to the origin. Column-major.
        float t = static_cast<float>(frame++) * 0.02f;
        const float c = std::cos(t), s = std::sin(t);
        const float mvp[48] = {
            // model: rotation around Z
            c,  s, 0, 0,
           -s,  c, 0, 0,
            0,  0, 1, 0,
            0,  0, 0, 1,
            // view: identity
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
            // proj: identity (clip space)
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        };
        void* mapped = nullptr;
        vkMapMemory(device.device, uniformMemory, 0, sizeof(mvp), 0, &mapped);
        std::memcpy(mapped, mvp, sizeof(mvp));
        vkUnmapMemory(device.device, uniformMemory);

        vkResetCommandBuffer(frames.commandBuffer(), 0);
        beginFrameCommandBuffer(frames.commandBuffer());

        // dynamic viewport/scissor (the pipeline declares them dynamic)
        VkViewport viewport{0.0f, 0.0f, static_cast<float>(renderExtent.width),
                            static_cast<float>(renderExtent.height), 0.0f, 1.0f};
        VkRect2D scissor{{0, 0}, renderExtent};
        vkCmdSetViewport(frames.commandBuffer(), 0, 1, &viewport);
        vkCmdSetScissor(frames.commandBuffer(), 0, 1, &scissor);

        VkRenderPassBeginInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass = renderPass.renderPass;
        rp.framebuffer = frames.framebuffer(imageIndex);
        rp.renderArea.extent = renderExtent;
        rp.clearValueCount = 1;
        rp.pClearValues = &clearColor;
        vkCmdBeginRenderPass(frames.commandBuffer(), &rp, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(frames.commandBuffer(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipelines.graphicsPipeline);
        vkCmdBindDescriptorSets(frames.commandBuffer(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelines.pipelineLayout, 0, 1, &descriptors.descriptorSet, 0, nullptr);
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(frames.commandBuffer(), 0, 1, &geometry.vertexBuffer, &offset);
        vkCmdBindIndexBuffer(frames.commandBuffer(), geometry.indexBuffer, 0, VK_INDEX_TYPE_UINT16);
        // vertex/index data upload happens once, before the loop (see below)
        vkCmdDrawIndexed(frames.commandBuffer(), 6, 1, 0, 0, 0);
        vkCmdEndRenderPass(frames.commandBuffer());
        endFrameCommandBuffer(frames.commandBuffer());

        VkSemaphore waitSem[] = {sync.imageAvailable};
        VkSemaphore signalSem[] = {renderFinished[imageIndex]};
        VkPipelineStageFlags waitStage[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = waitSem;
        si.pWaitDstStageMask = waitStage;
        si.commandBufferCount = 1;
        VkCommandBuffer cmdBuf = frames.commandBuffer();
        si.pCommandBuffers = &cmdBuf;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = signalSem;
        if (vkQueueSubmit(graphicsQueue, 1, &si, sync.frameInFlight) != VK_SUCCESS)
            throw std::runtime_error("vkQueueSubmit failed");

        VkPresentInfoKHR pi{};
        pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = signalSem;
        pi.swapchainCount = 1;
        pi.pSwapchains = &swapchain.swapchain;
        pi.pImageIndices = &imageIndex;
        vkQueuePresentKHR(graphicsQueue, &pi);
    }

        vkDeviceWaitIdle(device.device);

        // raw allocations (not covered by generated classes): free them
        // before the device dies
        for (auto sem : renderFinished) vkDestroySemaphore(device.device, sem, nullptr);
        for (auto m : geometryMemories) vkFreeMemory(device.device, m, nullptr);
        vkFreeMemory(device.device, textureMemory, nullptr);
        vkFreeMemory(device.device, stagingMemory, nullptr);
        vkFreeMemory(device.device, uniformMemory, nullptr);
    }  // swapchain (and per-image objects) destroyed here

    // GLFW created the surface but does not destroy it; the swapchain is
    // gone, so the surface can die before the instance
    vkDestroySurfaceKHR(instance.instance, surface, nullptr);

    }  // namespace-scoped RAII objects destroyed here, before GLFW

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
