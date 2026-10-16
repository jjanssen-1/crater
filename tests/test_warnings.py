import pytest

from crater.registry import Registry
from crater.spec import parse_spec
from crater.warnings import collect_warnings


def _warnings_for(spec, registry):
    return [w.message for w in collect_warnings(parse_spec(spec, registry), registry)]


def test_warns_missing_multisample_state(registry: Registry):
    ws = _warnings_for(
        """
P:
  VkGraphicsPipeline:
    name: pipeline
""",
        registry,
    )
    assert any("multisampleState" in w for w in ws)


def test_warns_viewport_counts_without_arrays_or_dynamic(registry: Registry):
    ws = _warnings_for(
        """
P:
  VkGraphicsPipeline:
    name: pipeline
    viewportState:
      viewportCount: 1
      scissorCount: 1
""",
        registry,
    )
    assert any("counts without viewport/scissor arrays" in w for w in ws)


def test_no_warning_for_dynamic_pipeline(registry: Registry):
    ws = _warnings_for(
        """
P:
  VkGraphicsPipeline:
    name: pipeline
    multisampleState:
      rasterizationSamples: VK_SAMPLE_COUNT_1_BIT
    dynamicState:
      dynamicStates: [VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR]
    viewportState:
      viewportCount: 1
      scissorCount: 1
    rasterizationState:
      polygonMode: VK_POLYGON_MODE_FILL
      lineWidth: 1.0
    inputAssemblyState:
      topology: VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
    colorBlendState:
      attachments:
        - colorWriteMask: VK_COLOR_COMPONENT_R_BIT
""",
        registry,
    )
    assert not any("multisampleState" in w or "viewport" in w for w in ws)


def test_warns_zero_default_enum_without_zero_value(registry: Registry):
    # VkQueueGlobalPriority has no 0-value; a priority struct member
    # left unset defaults to an invalid priority
    ws = _warnings_for(
        """
D:
  VkDevice:
    name: device
    pNext:
      - VkDeviceQueueGlobalPriorityCreateInfo: {}
""",
        registry,
    )
    assert any("zero default" in w for w in ws)


def test_examples_are_warning_free(registry: Registry):
    """The shipped examples should produce no warnings."""
    from pathlib import Path

    for name in ("triangle", "quad", "glfw_triangle"):
        spec = Path(__file__).parent.parent / "examples" / f"{name}.yaml"
        if not spec.exists():
            continue
        ws = _warnings_for(spec.read_text(), registry)
        assert ws == [], f"{name}.yaml produced warnings: {ws}"
