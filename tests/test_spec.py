import pytest

from crater.registry import Registry

from crater.spec import Placeholder, Ref, StructValue, parse_spec


def test_simple_instance(registry: Registry):
    cs = parse_spec(
        """
VulkanApp:
  VkInstance:
    name: mainInstance
""",
        registry,
    )
    assert len(cs) == 1
    obj = cs[0].objects[0]
    assert obj.instance_name == "mainInstance"
    assert obj.binding.create_command == "vkCreateInstance"


def test_default_instance_name(registry: Registry):
    cs = parse_spec(
        """
VulkanApp:
  VkPipelineLayout: {}
""",
        registry,
    )
    assert cs[0].objects[0].instance_name == "pipelineLayout"


def test_placeholder_with_explicit_type(registry: Registry):
    cs = parse_spec(
        """
S:
  VkShaderModule:
    codeSize: {{ uint32_t shaderSize }}
""",
        registry,
    )
    v = cs[0].objects[0].params["codeSize"]
    assert isinstance(v, Placeholder)
    assert v.name == "shaderSize"
    assert v.explicit_type == "uint32_t"


def test_placeholder_without_type(registry: Registry):
    cs = parse_spec(
        """
S:
  VkShaderModule:
    pCode: {{ code }}
""",
        registry,
    )
    v = cs[0].objects[0].params["pCode"]
    assert isinstance(v, Placeholder) and v.name == "code" and v.explicit_type is None


def test_sibling_and_external_refs(registry: Registry):
    cs = parse_spec(
        """
A:
  VkRenderPass:
    name: mainRenderPass
B:
  VkPipelineLayout:
    name: layout
  VkGraphicsPipeline:
    layout: layout
    renderPass: A.mainRenderPass
""",
        registry,
    )
    pipeline = cs[1].objects[1]
    assert pipeline.params["layout"] == Ref(kind="sibling", object_name="layout")
    assert pipeline.params["renderPass"] == Ref(kind="external", object_name="mainRenderPass", class_name="A")


def test_nested_struct(registry: Registry):
    cs = parse_spec(
        """
VulkanApp:
  VkInstance:
    applicationInfo:
      applicationName: "Test"
      apiVersion: [1, 3, 0]
""",
        registry,
    )
    app = cs[0].objects[0].params["applicationInfo"]
    assert isinstance(app, StructValue)
    assert app.fields["applicationName"] == "Test"


def test_duplicate_names_rejected(registry: Registry):
    with pytest.raises(Exception):
        parse_spec(
            """
C:
  VkRenderPass:
    name: same
  VkPipelineLayout:
    name: same
""",
            registry,
        )


def test_unknown_object_rejected(registry: Registry):
    with pytest.raises(Exception):
        parse_spec(
            """
C:
  VkNotAThing: {}
""",
            registry,
        )
