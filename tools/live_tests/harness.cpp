// Live-device test harness for crater-generated classes.
//
// Build:
//   crater generate tools/live_tests/specs/common.yaml -o <gen>/common.hpp
//   crater generate tools/live_tests/specs/device_objects.yaml -o <gen>/device_objects.hpp
//   crater generate tools/live_tests/specs/render_objects.yaml -o <gen>/render_objects.hpp
//   g++ -std=c++17 tools/live_tests/harness.cpp -o live_tests \
//       -I <gen> -I <vulkan-headers>/include -lvulkan
//
// Safety: only creates and destroys Vulkan objects with sensible
// parameters. No rendering, no presentation, no file or network access.
// Every test is wrapped so one failure does not stop the sweep; results
// are printed as a table and written to the results file given below.

#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

#include "common.hpp"
#include "device_objects.hpp"
#include "render_objects.hpp"

namespace {

int g_passed = 0;
int g_failed = 0;
std::vector<std::string> g_results;

void report(const char* name, bool ok, const std::string& detail = "") {
    if (ok) {
        ++g_passed;
        std::printf("[PASS] %s\n", name);
        g_results.push_back(std::string("| ") + name + " | PASS | " + detail + " |");
    } else {
        ++g_failed;
        std::printf("[FAIL] %s (%s)\n", name, detail.c_str());
        g_results.push_back(std::string("| ") + name + " | FAIL | " + detail + " |");
    }
    std::fflush(stdout);
}

#define RUN_TEST(name, stmt)                             \
    do {                                                 \
        try {                                            \
            { stmt; }                                    \
            report(#name, true);                         \
        } catch (const std::exception& e) {              \
            report(#name, false, e.what());              \
        }                                                \
    } while (0)

uint32_t pickGraphicsFamily(VkPhysicalDevice pd) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, families.data());
    for (uint32_t i = 0; i < count; ++i) {
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) return i;
    }
    throw std::runtime_error("no graphics queue family");
}

// Minimal valid SPIR-V 1.0 modules, hand-encoded: a "void main(){}"
// entry point per stage. These are structurally valid (validated
// instruction encodings; the earlier bogus padding killed drivers).
//
// Layout: OpCapability Shader / OpMemoryModel Logical GLSL450 /
// OpEntryPoint <stage> %main "main" / OpSource GLSL 450 / OpName /
// %void / %fn = OpTypeFunction %void / %main = OpFunction / OpLabel /
// OpReturn / OpFunctionEnd.
const uint32_t g_spvVertex[] = {
    0x07230203, 0x00010000, 0x00080001, 0x00000005, 0x00000000,
    0x00020011, 0x00000001,                                    // OpCapability Shader
    0x0003000E, 0x00000000, 0x00000001,                        // OpMemoryModel Logical GLSL450
    0x0005000F, 0x00000000, 0x00000003, 0x6E69616D, 0x00000000, // OpEntryPoint Vertex %3 "main"
    0x00030003, 0x00000002, 0x000001C2,                        // OpSource GLSL 450
    0x00040005, 0x00000003, 0x6E69616D, 0x00000000,            // OpName %3 "main"
    0x00020013, 0x00000001,                                    // %1 = OpTypeVoid
    0x00030021, 0x00000002, 0x00000001,                        // %2 = OpTypeFunction %1
    0x00050036, 0x00000001, 0x00000003, 0x00000000, 0x00000002, // %3 = OpFunction %1 None %2
    0x000200F8, 0x00000004,                                    // %4 = OpLabel
    0x000100FD,                                                // OpReturn
    0x00010100,                                                // OpFunctionEnd
};
const uint32_t g_spvFragment[] = {
    0x07230203, 0x00010000, 0x00080001, 0x00000005, 0x00000000,
    0x00020011, 0x00000001,
    0x0003000E, 0x00000000, 0x00000001,
    0x0005000F, 0x00000004, 0x00000003, 0x6E69616D, 0x00000000, // OpEntryPoint Fragment %3 "main"
    0x00030003, 0x00000002, 0x000001C2,
    0x00040005, 0x00000003, 0x6E69616D, 0x00000000,
    0x00020013, 0x00000001,
    0x00030021, 0x00000002, 0x00000001,
    0x00050036, 0x00000001, 0x00000003, 0x00000000, 0x00000002,
    0x000200F8, 0x00000004,
    0x000100FD,
    0x00010100,
};
const uint32_t g_spvCompute[] = {
    0x07230203, 0x00010000, 0x00080001, 0x00000005, 0x00000000,
    0x00020011, 0x00000001,                                    // OpCapability Shader
    0x0003000E, 0x00000000, 0x00000001,                        // OpMemoryModel Logical GLSL450
    0x0005000F, 0x00000005, 0x00000003, 0x6E69616D, 0x00000000, // OpEntryPoint GLCompute %3 "main"
    0x00060010, 0x00000003, 0x00000011, 0x00000001, 0x00000001, 0x00000001, // OpExecutionMode %3 LocalSize 1 1 1
    0x00030003, 0x00000002, 0x000001C2,                        // OpSource GLSL 450
    0x00040005, 0x00000003, 0x6E69616D, 0x00000000,            // OpName %3 "main"
    0x00020013, 0x00000001,                                    // %1 = OpTypeVoid
    0x00030021, 0x00000002, 0x00000001,                        // %2 = OpTypeFunction %1
    0x00050036, 0x00000001, 0x00000003, 0x00000000, 0x00000002, // %3 = OpFunction %1 None %2
    0x000200F8, 0x00000004,                                    // %4 = OpLabel
    0x000100FD,                                                // OpReturn
    0x00010100,                                                // OpFunctionEnd
};

}  // namespace

int main(int argc, char** argv) {
    const char* resultsPath = argc > 1 ? argv[1] : "live_results.md";
    std::printf("crater live tests: starting\n");

    // --- instance + device --------------------------------------------------
    // enumerate instance extensions and enable everything available? No:
    // enabling unknown extensions can fail device creation. Enable none
    // beyond what the loader requires; the generated class takes the list.
    uint32_t extensionCount = 0;
    const char* const* extensions = nullptr;  // none required for these tests

    try {
        LiveInstance liveInstance(extensionCount, extensions);

        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(liveInstance.instance, &deviceCount, nullptr);
        if (deviceCount == 0) throw std::runtime_error("no physical devices");
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(liveInstance.instance, &deviceCount, devices.data());

        VkPhysicalDevice physicalDevice = devices[0];
        uint32_t graphicsFamily = pickGraphicsFamily(physicalDevice);
        std::printf("physical device: %p, graphics family %u\n",
                    (void*)physicalDevice, graphicsFamily);

        LiveDevice liveDevice(physicalDevice, graphicsFamily);
        report("LiveDevice", true);

        // --- device objects --------------------------------------------------
        RUN_TEST(SyncObjects, SyncObjects t(liveDevice.device));
        RUN_TEST(MemoryObjects, MemoryObjects t(liveDevice.device, VkExtent3D{64, 64, 1}));
        RUN_TEST(SamplerObjects, SamplerObjects t(liveDevice.device));
        RUN_TEST(QueryObjects, QueryObjects t(liveDevice.device));
        RUN_TEST(CommandObjects, CommandObjects t(liveDevice.device, graphicsFamily));
        RUN_TEST(DescriptorObjects, DescriptorObjects t(liveDevice.device));

        // views need the image; construct with the class that owns it
        RUN_TEST(ViewObjects, {
            MemoryObjects memory(liveDevice.device, VkExtent3D{64, 64, 1});
            // views require bound memory: allocate and bind (host-visible
            // type, small allocation, freed with the memory object)
            VkMemoryRequirements reqs{};
            vkGetImageMemoryRequirements(liveDevice.device, memory.image, &reqs);
            VkPhysicalDeviceMemoryProperties props{};
            vkGetPhysicalDeviceMemoryProperties(physicalDevice, &props);
            uint32_t typeIndex = 0;
            for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
                if ((reqs.memoryTypeBits & (1u << i)) &&
                    (props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                    typeIndex = i;
                    break;
                }
            }
            VkMemoryAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.allocationSize = reqs.size;
            allocInfo.memoryTypeIndex = typeIndex;
            VkDeviceMemory imageMemory = VK_NULL_HANDLE;
            if (vkAllocateMemory(liveDevice.device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS)
                throw std::runtime_error("vkAllocateMemory failed");
            if (vkBindImageMemory(liveDevice.device, memory.image, imageMemory, 0) != VK_SUCCESS)
                throw std::runtime_error("vkBindImageMemory failed");
            ViewObjects views(liveDevice.device, memory.image);
            vkFreeMemory(liveDevice.device, imageMemory, nullptr);
        });

        // pipeline objects need a shader module
        RUN_TEST(PipelineObjects, PipelineObjects t(
            liveDevice.device, sizeof(g_spvVertex), g_spvVertex));

        // render pass + framebuffer + graphics pipeline
        RUN_TEST(RenderObjects, RenderObjects t(liveDevice.device));
        RUN_TEST(Framebuffers, {
            RenderObjects render(liveDevice.device);
            MemoryObjects memory(liveDevice.device, VkExtent3D{64, 64, 1});
            ViewObjects views(liveDevice.device, memory.image);
            std::vector<VkImageView> attachments{views.view};
            Framebuffers fb(liveDevice.device, render.renderPass,
                            attachments.data(), 64, 64);
        });
        RUN_TEST(Pipeline, {
            PipelineObjects pipelineObjects(liveDevice.device, sizeof(g_spvVertex), g_spvVertex);
            RenderObjects render(liveDevice.device);
            // a second module with a Fragment entry point
            VkShaderModuleCreateInfo fragInfo{};
            fragInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            fragInfo.codeSize = sizeof(g_spvFragment);
            fragInfo.pCode = g_spvFragment;
            VkShaderModule fragModule = VK_NULL_HANDLE;
            if (vkCreateShaderModule(liveDevice.device, &fragInfo, nullptr, &fragModule) != VK_SUCCESS)
                throw std::runtime_error("vkCreateShaderModule (fragment) failed");
            Pipeline p(liveDevice.device,
                       pipelineObjects.pipelineLayout, render.renderPass,
                       pipelineObjects.shaderModule, fragModule);
            vkDestroyShaderModule(liveDevice.device, fragModule, nullptr);
        });
        RUN_TEST(Compute, {
            PipelineObjects pipelineObjects(liveDevice.device, sizeof(g_spvCompute), g_spvCompute);
            Compute c(liveDevice.device, pipelineObjects.pipelineLayout,
                      pipelineObjects.shaderModule);
        });

    } catch (const std::exception& e) {
        std::printf("fatal: instance/device setup failed: %s\n", e.what());
        g_results.push_back(std::string("| setup | FATAL | ") + e.what() + " |");
    }

    // --- results ------------------------------------------------------------
    std::FILE* f = std::fopen(resultsPath, "w");
    if (f) {
        std::fprintf(f, "# crater live test results\n\n");
        std::fprintf(f, "**%d** passed, **%d** failed\n\n", g_passed, g_failed);
        std::fprintf(f, "| test | status | detail |\n|---|---|---|\n");
        for (const auto& line : g_results) std::fprintf(f, "%s\n", line.c_str());
        std::fclose(f);
        std::printf("results written to %s\n", resultsPath);
    }
    std::printf("crater live tests: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
