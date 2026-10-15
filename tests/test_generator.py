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
    assert "vkCreateInstance(&mainInstanceCreateInfo, nullptr, &mainInstance)" in out
    assert "VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO" in out
    assert 'applicationName_str("Test")' in out
    assert "apiVersion = VK_MAKE_API_VERSION(0, 1, 3, 0)" in out
    assert "vkDestroyInstance(mainInstance, nullptr);" in out


def test_sibling_ref_orders_creation(registry: Registry):
    out = generate_from_spec(
        """
P:
  VkGraphicsPipeline:
    name: pipeline
    layout: layout
  VkPipelineLayout:
    name: layout
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
B:
  VkGraphicsPipeline:
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
    pCode: {{ code }}
""",
        registry,
    )
    assert "explicit S(VkDevice vkDevice, const uint32_t* code)" in out
    assert ".pCode = code;" in out


def test_stage_array_generates_count(registry: Registry):
    out = generate_from_spec(
        """
P:
  VkGraphicsPipeline:
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
  VkGraphicsPipeline: {}
""",
        registry,
    )
    assert "vkCreateGraphicsPipelines(" in out
    assert "VK_NULL_HANDLE, 1, &graphicsPipelineCreateInfo," in out


def test_pipeline_cache_defaults_to_null(registry: Registry):
    out = generate_from_spec(
        """
P:
  VkGraphicsPipeline: {}
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
    layout: layout
  VkPipelineLayout:
    name: layout
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
  VkGraphicsPipeline:
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
    enabledExtensionNames:
      - VK_KHR_swapchain

TrianglePipeline:
  VkPipelineLayout:
    name: layout
  VkGraphicsPipeline:
    name: pipeline
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
  VkSwapchainKHR:
    name: swapchain
    imageExtent: {{ VkExtent2D extent }}
  VkSemaphore:
    - name: imageAvailable
    - name: renderFinished
  VkFence:
    name: inFlight
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
