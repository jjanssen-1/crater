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
