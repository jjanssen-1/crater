import pytest

from crater.generator import GenerationError
from crater.registry import Registry

from conftest import compile_generated, generate_from_spec


def test_instance_generation(registry: Registry):
    out = generate_from_spec(
        """
VulkanApp:
  VkInstance:
    name: mainInstance
    applicationInfo:
      applicationName: "Test"
      engineName: "TestEngine"
      apiVersion: [1, 3, 0]
""",
        registry,
    )
    assert "class VulkanApp" in out
    assert "vkCreateInstance(&mainInstanceCreateInfo, nullptr, &mainInstance_)" in out
    assert "VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO" in out
    assert 'applicationName_str("Test")' in out
    assert "apiVersion = VK_MAKE_API_VERSION(0, 1, 3, 0)" in out
    assert "vkDestroyInstance(mainInstance_, nullptr);" in out


def test_sibling_ref_orders_creation(registry: Registry):
    out = generate_from_spec(
        """
P:
  VkGraphicsPipeline:
    name: pipeline
    device: "{{ VkDevice device }}"
    layout: layout
  VkPipelineLayout:
    name: layout
    device: "{{ VkDevice device }}"
""",
        registry,
    )
    layout_pos = out.index("vkCreatePipelineLayout")
    pipeline_pos = out.index("vkCreateGraphicsPipelines")
    assert layout_pos < pipeline_pos
    assert ".layout = layout" in out


def test_external_ref_becomes_ctor_param(registry: Registry):
    out = generate_from_spec(
        """
A:
  VkRenderPass:
    name: mainRenderPass
    device: "{{ VkDevice device }}"
B:
  VkGraphicsPipeline:
    device: "{{ VkDevice device }}"
    renderPass: A.mainRenderPass
""",
        registry,
    )
    assert "explicit B(const A& a," in out
    assert ".renderPass = a.mainRenderPass" in out


def test_placeholder_infers_type_from_registry(registry: Registry):
    out = generate_from_spec(
        """
S:
  VkShaderModule:
    device: "{{ VkDevice device }}"
    pCode: {{ code }}
""",
        registry,
    )
    assert "explicit S(VkDevice device, const uint32_t* code)" in out
    assert ".pCode = code;" in out


def test_stage_array_generates_count(registry: Registry):
    out = generate_from_spec(
        """
P:
  VkGraphicsPipeline:
    device: "{{ VkDevice device }}"
    stages:
      - stage: VK_SHADER_STAGE_VERTEX_BIT
        module: {{ vertexModule }}
""",
        registry,
    )
    assert ".stageCount = static_cast<uint32_t>(graphicsPipelineCreateInfo_stages.size());" in out
    assert ".pStages = graphicsPipelineCreateInfo_stages.data();" in out
    assert ".stage = VK_SHADER_STAGE_VERTEX_BIT" in out


def test_plural_command_count_is_one(registry: Registry):
    out = generate_from_spec(
        """
P:
  VkGraphicsPipeline:
    device: "{{ VkDevice device }}"
""",
        registry,
    )
    assert "vkCreateGraphicsPipelines(" in out
    assert "VK_NULL_HANDLE, 1, &graphicsPipelineCreateInfo," in out


def test_pipeline_cache_defaults_to_null(registry: Registry):
    out = generate_from_spec(
        """
P:
  VkGraphicsPipeline:
    device: "{{ VkDevice device }}"
""",
        registry,
    )
    assert "VK_NULL_HANDLE, 1" in out


def test_external_ref_to_missing_object_rejected(registry: Registry):
    with pytest.raises(GenerationError):
        generate_from_spec(
            """
A:
  VkRenderPass:
    name: r
B:
  VkGraphicsPipeline:
    renderPass: A.missing
""",
            registry,
        )


def test_destruction_is_reverse_order(registry: Registry):
    out = generate_from_spec(
        """
P:
  VkGraphicsPipeline:
    name: pipeline
    device: "{{ VkDevice device }}"
    layout: layout
  VkPipelineLayout:
    name: layout
    device: "{{ VkDevice device }}"
""",
        registry,
    )
    dtor = out[out.index("~P()") :]
    assert dtor.index("vkDestroyPipeline") < dtor.index("vkDestroyPipelineLayout")


def test_circular_class_refs_rejected(registry: Registry):
    with pytest.raises(GenerationError):
        generate_from_spec(
            """
A:
  VkRenderPass:
    name: r
    device: "{{ VkDevice device }}"
  VkGraphicsPipeline:
    device: "{{ VkDevice device }}"
    renderPass: B.r
B:
  VkRenderPass:
    name: r
  VkGraphicsPipeline:
    renderPass: A.r
""",
            registry,
        )


TRIANGLE_SPEC = """
VulkanApp:
  VkInstance:
    name: instance
    applicationInfo:
      applicationName: "Triangle"
      engineName: "Crater"
      apiVersion: [1, 3, 0]

VulkanDevice:
  VkDevice:
    name: device
    physicalDevice: {{ VkPhysicalDevice physicalDevice }}
    enabledExtensionNames:
      - VK_KHR_swapchain

TrianglePipeline:
  VkPipelineLayout:
    name: layout
    device: VulkanDevice.device
  VkGraphicsPipeline:
    name: pipeline
    device: VulkanDevice.device
    layout: layout
    renderPass: TriangleFrame.renderPass
    stages:
      - stage: VK_SHADER_STAGE_VERTEX_BIT
        module: {{ vertexModule }}
        name: vert
      - stage: VK_SHADER_STAGE_FRAGMENT_BIT
        module: {{ fragmentModule }}
        name: frag

TriangleFrame:
  VkRenderPass:
    name: renderPass
    device: VulkanDevice.device
  VkSwapchainKHR:
    name: swapchain
    device: VulkanDevice.device
    surface: {{ VkSurfaceKHR surface }}
    imageExtent: {{ VkExtent2D extent }}
  VkSemaphore:
    - name: imageAvailable
      device: VulkanDevice.device
    - name: renderFinished
      device: VulkanDevice.device
  VkFence:
    name: inFlight
    device: VulkanDevice.device
"""


def test_triangle_spec_compiles(registry: Registry, tmp_path):
    out = generate_from_spec(TRIANGLE_SPEC, registry)
    assert compile_generated(out, tmp_path)


def test_generated_header_compiles_simple_case(registry: Registry, tmp_path):
    out = generate_from_spec(
        """
VulkanApp:
  VkInstance:
    name: mainInstance
    applicationInfo:
      applicationName: "Test"
      apiVersion: "1.3"
""",
        registry,
    )
    assert compile_generated(out, tmp_path)


def test_nested_struct_in_array_item(registry: Registry):
    out = generate_from_spec(
        """
P:
  VkGraphicsPipeline:
    device: "{{ VkDevice device }}"
    viewportState:
      scissors:
        - offset:
            x: 5
            y: 6
          extent:
            width: 800
            height: 600
""",
        registry,
    )
    assert "VkOffset2D graphicsPipelineCreateInfo_viewportState_scissors_item0_offset{};" in out
    assert "graphicsPipelineCreateInfo_viewportState_scissors_item0_offset.x = 5;" in out
    assert (
        "graphicsPipelineCreateInfo_viewportState_scissors_item0.extent = "
        "graphicsPipelineCreateInfo_viewportState_scissors_item0_extent;" in out
    )


def test_vulkan_error_exception_type(registry: Registry, tmp_path):
    """All generated throws use crater::VulkanError; VkResult failures
    carry the result code, capability failures use VK_ERROR_UNKNOWN."""
    from conftest import compile_generated, generate_from_spec

    out = generate_from_spec(
        """
T:
  VkSemaphore:
    name: sem
    device: "{{ VkDevice device }}"
""",
        registry,
    )
    assert "class VulkanError : public std::runtime_error" in out
    assert 'VulkanError(std::string("vkCreateSemaphore failed (")' in out
    assert ", result__);" in out
    assert compile_generated(out, tmp_path)


def test_vulkan_error_catchable(registry: Registry, tmp_path):
    """A translation unit catching crater::VulkanError specifically."""
    from conftest import compile_generated, generate_from_spec

    out = generate_from_spec(
        """
T:
  VkSemaphore:
    name: sem
    device: "{{ VkDevice device }}"
""",
        registry,
    )
    header = tmp_path / "generated.hpp"
    header.write_text(out)
    main = tmp_path / "main.cpp"
    main.write_text(
        '#include "generated.hpp"\n'
        "int main() {\n"
        "    try {\n"
        "        T t(VK_NULL_HANDLE);\n"
        "    } catch (const crater::VulkanError& e) {\n"
        "        return e.result() == VK_ERROR_INITIALIZATION_FAILED ? 1 : 0;\n"
        "    }\n"
        "    return 0;\n"
        "}\n"
    )
    import subprocess

    from conftest import find_vulkan_headers

    headers = find_vulkan_headers()
    if headers is None:
        pytest.skip("Vulkan headers not available")
    inc = headers / "include" if (headers / "include/vulkan/vulkan.h").exists() else headers
    proc = subprocess.run(
        ["g++", "-std=c++17", f"-I{tmp_path}", f"-I{inc}", "-fsyntax-only", str(main)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_device_with_children_in_one_class_compiles(registry: Registry, tmp_path):
    """A VkDevice grouped with objects created from it in the SAME class
    must emit valid parent-handle references (the private members, not
    the accessor names). Regression test: both the create calls and the
    destroy calls used the bare instance name, which does not compile."""
    out = generate_from_spec(
        """
App:
  VkDevice:
    name: logicalDevice
    physicalDevice: "{{ VkPhysicalDevice physicalDevice }}"
  VkSwapchainKHR:
    name: swapchain
    device: logicalDevice
    surface: "{{ VkSurfaceKHR surface }}"
""",
        registry,
    )
    assert "vkCreateSwapchainKHR(logicalDevice_, &swapchainCreateInfo" in out
    # the destroy call passes the parent stored from the declared sibling
    assert "vkDestroySwapchainKHR(device_, swapchain_" in out
    assert "device_(logicalDevice_)" in out
    # parent handles are never inferred: the spec must declare them
    with pytest.raises(GenerationError):
        generate_from_spec(
            """
App:
  VkDevice:
    name: logicalDevice
    physicalDevice: "{{ VkPhysicalDevice physicalDevice }}"
  VkSwapchainKHR:
    name: swapchain
    surface: "{{ VkSurfaceKHR surface }}"
""",
            registry,
        )
    assert compile_generated(out, tmp_path)
