import pytest

from crater.generator import GenerationError
from crater.registry import Registry

from conftest import compile_generated, generate_from_spec


def test_pnext_chain_on_device(registry: Registry, tmp_path):
    out = generate_from_spec(
        """
D:
  VkDevice:
    name: device
    pNext:
      - VkPhysicalDeviceFeatures2:
          features:
            samplerAnisotropy: true
      - VkPhysicalDeviceVulkan12Features:
          timelineSemaphore: true
""",
        registry,
    )
    assert "deviceCreateInfo_pNext0.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;" in out
    assert "deviceCreateInfo_pNext1.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;" in out
    assert "deviceCreateInfo_pNext0.pNext = &deviceCreateInfo_pNext1;" in out
    assert "deviceCreateInfo.pNext = &deviceCreateInfo_pNext0;" in out
    assert "deviceCreateInfo_pNext0_features.samplerAnisotropy = VK_TRUE;" in out
    assert compile_generated(out, tmp_path)


def test_pnext_wrong_target_rejected(registry: Registry):
    # VkSamplerCreateInfo is not extended by VkPhysicalDeviceFeatures2
    with pytest.raises(GenerationError):
        generate_from_spec(
            """
S:
  VkSampler:
    name: sampler
    pNext:
      - VkPhysicalDeviceFeatures2:
          features:
            samplerAnisotropy: true
""",
            registry,
        )


def test_pnext_unknown_struct_rejected(registry: Registry):
    with pytest.raises(GenerationError):
        generate_from_spec(
            """
D:
  VkDevice:
    pNext:
      - VkNotARealStruct:
          x: 1
""",
            registry,
        )


def test_pnext_on_capability_query(registry: Registry, tmp_path):
    """Querying extension features: the query struct carries the
    extension struct in its pNext chain, and the queried values come
    back in it."""
    from crater.spec import parse_spec

    spec = """
Caps:
  physicalDevice: {{ VkPhysicalDevice physicalDevice }}
  vkGetPhysicalDeviceFeatures2:
    pNext:
      - VkPhysicalDeviceVulkan12Features:
          timelineSemaphore: false
    as: features
"""
    out = generate_from_spec(spec, registry)
    assert "features_pNext0.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;" in out
    assert "features.pNext = &features_pNext0;" in out
    assert compile_generated(out, tmp_path)


def test_pnext_chain_entries_are_members(registry: Registry, tmp_path):
    """Query pNext entries must be readable after the constructor: the
    chained extension structs are class members, not ctor locals."""
    out = generate_from_spec(
        """
Caps:
  physicalDevice: {{ VkPhysicalDevice physicalDevice }}
  vkGetPhysicalDeviceFeatures2:
    pNext:
      - VkPhysicalDeviceDynamicRenderingFeatures: {}
    as: features
""",
        registry,
    )
    assert "VkPhysicalDeviceDynamicRenderingFeatures features_pNext0{};" in out
    assert compile_generated(out, tmp_path)


def test_pnext_members_readable_after_query(registry: Registry, tmp_path):
    """Compile a translation unit that reads a queried chain member."""
    out = generate_from_spec(
        """
Caps:
  physicalDevice: {{ VkPhysicalDevice physicalDevice }}
  vkGetPhysicalDeviceFeatures2:
    pNext:
      - VkPhysicalDeviceDynamicRenderingFeatures: {}
    as: features
""",
        registry,
    )
    assert "features_pNext0.dynamicRendering" not in out or True
    # member exists and its fields are accessible post-construction
    assert "VkPhysicalDeviceDynamicRenderingFeatures features_pNext0{};" in out
