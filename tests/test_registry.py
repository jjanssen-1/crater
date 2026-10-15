from crater.registry import Registry


def test_registry_loads(registry: Registry):
    assert len(registry.commands) > 500
    assert len(registry.structs) > 1000
    assert "VkInstance" in registry.handles


def test_instance_binding(bindings):
    b = bindings["VkInstance"]
    assert b.create_command == "vkCreateInstance"
    assert b.create_info_struct == "VkInstanceCreateInfo"
    assert b.destroy_command == "vkDestroyInstance"
    assert b.parent_handles == []


def test_graphics_pipeline_named_after_creation_function(bindings):
    b = bindings["VkGraphicsPipeline"]
    assert b.handle == "VkPipeline"
    assert b.create_command == "vkCreateGraphicsPipelines"
    assert b.create_info_struct == "VkGraphicsPipelineCreateInfo"
    assert b.destroy_command == "vkDestroyPipeline"
    assert "device" in b.parent_handles


def test_compute_pipeline_is_separate_entry(bindings):
    assert bindings["VkComputePipeline"].create_command == "vkCreateComputePipelines"
    assert bindings["VkComputePipeline"].handle == "VkPipeline"


def test_shader_module(bindings):
    b = bindings["VkShaderModule"]
    assert b.create_command == "vkCreateShaderModule"
    assert b.destroy_command == "vkDestroyShaderModule"
    assert b.parent_handles == ["device"]


def test_memory_allocation_uses_vkAllocateMemory(bindings):
    b = bindings["VkDeviceMemory"]
    assert b.create_command == "vkAllocateMemory"
    assert b.create_info_struct == "VkMemoryAllocateInfo"


def test_member_details(registry: Registry):
    s = registry.resolve_struct("VkInstanceCreateInfo")
    app = s.member("pApplicationInfo")
    assert app is not None and app.is_pointer and app.optional
    layers = s.member("ppEnabledLayerNames")
    assert layers is not None and layers.array_len == "enabledLayerCount"
    count = s.member("enabledLayerCount")
    assert count is not None and count.is_array_len_member


def test_stype_value(registry: Registry):
    s = registry.resolve_struct("VkGraphicsPipelineCreateInfo")
    stype = next(m for m in s.members if m.value)
    assert stype.value == "VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO"
