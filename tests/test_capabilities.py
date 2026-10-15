import pytest

from crater.registry import Registry

from crater.spec import SpecError, parse_spec

from conftest import compile_generated


def test_query_inference_shapes(registry: Registry):
    queries = registry.queries()
    surface_formats = queries["vkGetPhysicalDeviceSurfaceFormatsKHR"]
    assert surface_formats.shape == "array"
    assert surface_formats.query_command == "vkGetPhysicalDeviceSurfaceFormatsKHR"
    assert surface_formats.count_param == "pSurfaceFormatCount"
    assert [(i.name, i.type) for i in surface_formats.inputs] == [
        ("physicalDevice", "VkPhysicalDevice"),
        ("surface", "VkSurfaceKHR"),
    ]

    caps = queries["vkGetPhysicalDeviceSurfaceCapabilitiesKHR"]
    assert caps.shape == "struct"
    assert caps.query_command == "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"

    extensions = queries["vkEnumerateDeviceExtensionProperties"]
    assert extensions.shape == "present"
    assert extensions.name_member == "extensionName"


SURFACE_SPEC = """
SurfaceCapabilities:
  vkGetPhysicalDeviceSurfaceFormatsKHR:
    candidates:
      - [VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR]
    as: surfaceFormat
  vkGetPhysicalDeviceSurfacePresentModesKHR:
    candidates: [VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_FIFO_KHR]
    fallback: any
    as: presentMode
  vkEnumerateDeviceExtensionProperties:
    required: [VK_KHR_swapchain]
    optional: [VK_KHR_get_memory_requirements2]
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR: {}
"""


def test_capabilities_class_parses(registry: Registry):
    classes = parse_spec(SURFACE_SPEC, registry)
    assert len(classes) == 1
    caps = classes[0].capabilities
    assert caps is not None
    assert [q.command_name for q in caps.queries] == [
        "vkGetPhysicalDeviceSurfaceFormatsKHR",
        "vkGetPhysicalDeviceSurfacePresentModesKHR",
        "vkEnumerateDeviceExtensionProperties",
        "vkGetPhysicalDeviceSurfaceCapabilitiesKHR",
    ]
    assert caps.inputs == [
        ("physicalDevice", "VkPhysicalDevice"),
        ("surface", "VkSurfaceKHR"),
    ]
    ext = caps.query("vkEnumerateDeviceExtensionProperties")
    assert ext.required == ["VK_KHR_swapchain"]
    assert ext.optional == ["VK_KHR_get_memory_requirements2"]


def test_capabilities_generation(registry: Registry):
    from conftest import generate_from_spec

    out = generate_from_spec(SURFACE_SPEC, registry)
    assert "class SurfaceCapabilities" in out
    # two-call idiom
    assert "vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &surfaceFormatCount, nullptr)" in out
    assert "std::vector<VkSurfaceFormatKHR> surfaceFormatOptions(surfaceFormatCount);" in out
    # first-match selection
    assert "e.format == VK_FORMAT_B8G8R8A8_SRGB && e.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR" in out
    # fallback: any
    assert "if (!presentModeMatched && !presentModeOptions.empty()) {" in out
    # required extension throws when missing
    assert 'throw std::runtime_error("required VK_KHR_swapchain is not available");' in out
    # struct query result member
    assert "VkSurfaceCapabilitiesKHR surfaceCapabilities{};" in out or "VkSurfaceCapabilitiesKHR surfaceCapability{};" in out


def test_capabilities_compile(registry: Registry, tmp_path):
    from conftest import generate_from_spec

    out = generate_from_spec(SURFACE_SPEC, registry)
    assert compile_generated(out, tmp_path)


def test_consumed_capability_values(registry: Registry):
    from conftest import generate_from_spec

    out = generate_from_spec(
        SURFACE_SPEC
        + """
Swapchain:
  VkSwapchainKHR:
    name: swapchain
    imageExtent: SurfaceCapabilities.surfaceCapabilities.currentExtent
    imageFormat: SurfaceCapabilities.surfaceFormat.format
    presentMode: SurfaceCapabilities.presentMode
""",
        registry,
    )
    assert "explicit Swapchain(const SurfaceCapabilities& surfaceCapabilities," in out
    assert "swapchainCreateInfo.imageFormat = surfaceCapabilities.surfaceFormat.format;" in out
    assert "swapchainCreateInfo.presentMode = surfaceCapabilities.presentMode;" in out


def test_declared_inputs(registry: Registry):
    from conftest import generate_from_spec

    out = generate_from_spec(
        """
Dev:
  VkDevice:
    name: device

Caps:
  physicalDevice: {{ VkPhysicalDevice physicalDevice }}
  surface: {{ VkSurfaceKHR surface }}
  vkGetPhysicalDeviceSurfaceFormatsKHR:
    candidates:
      - [VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR]
""",
        registry,
    )
    assert "explicit Caps(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface)" in out


def test_unknown_input_rejected(registry: Registry):
    with pytest.raises(SpecError):
        parse_spec(
            """
Caps:
  notAnInput: 5
  vkGetPhysicalDeviceSurfaceFormatsKHR:
    candidates:
      - [VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR]
""",
            registry,
        )


def test_fallback_requires_candidates(registry: Registry):
    with pytest.raises(SpecError):
        parse_spec(
            """
Caps:
  vkGetPhysicalDeviceSurfacePresentModesKHR:
    fallback: any
""",
            registry,
        )


def test_required_only_for_availability_queries(registry: Registry):
    with pytest.raises(SpecError):
        parse_spec(
            """
Caps:
  vkGetPhysicalDeviceSurfaceFormatsKHR:
    required: [VK_FORMAT_B8G8R8A8_SRGB]
""",
            registry,
        )
