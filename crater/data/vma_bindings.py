"""Hand-authored bindings for the Vulkan Memory Allocator (VMA).

VMA is a C library without a registry, so its bindings are described
here in the same shapes crater's vk.xml inference produces: creation
commands with CreateInfo structs, handle outputs, and destroy
commands. Everything downstream - spec syntax, generation, warnings -
treats these identically to registry-inferred bindings.

Signatures transcribed from vk_mem_alloc.h (VMA 3.x):
  vmaCreateAllocator(const VmaAllocatorCreateInfo*, VmaAllocator*)
  vmaDestroyAllocator(VmaAllocator)
  vmaCreateBuffer(VmaAllocator, const VkBufferCreateInfo*,
                  const VmaAllocationCreateInfo*, VkBuffer*,
                  VmaAllocation*, VmaAllocationInfo*)
  vmaDestroyBuffer(VmaAllocator, VkBuffer, VmaAllocation)
  vmaCreateImage / vmaDestroyImage: same shape for VkImage
  vmaMapMemory / vmaUnmapMemory / vmaGetAllocationInfo: call functions
"""

# member tuples: (name, type, is_pointer, is_const, optional)
VMA_STRUCTS = {
    "VmaAllocatorCreateInfo": {
        "members": [
            ("flags", "VmaAllocatorCreateFlags", False, False, True),
            ("physicalDevice", "VkPhysicalDevice", False, False, False),
            ("device", "VkDevice", False, False, False),
            ("preferredLargeHeapBlockSize", "VkDeviceSize", False, False, True),
            ("pAllocationCallbacks", "VkAllocationCallbacks", True, True, True),
            ("pDeviceMemoryCallbacks", "VmaDeviceMemoryCallbacks", True, True, True),
            ("pHeapSizeLimit", "VkDeviceSize", True, False, True),
            ("instance", "VkInstance", False, False, True),
            ("vulkanApiVersion", "uint32_t", False, False, True),
        ],
        "stype": None,
    },
    "VmaAllocationCreateInfo": {
        "members": [
            ("usage", "VmaMemoryUsage", False, False, True),
            ("flags", "VmaAllocationCreateFlags", False, False, True),
            ("requiredFlags", "VkMemoryPropertyFlags", False, False, True),
            ("preferredFlags", "VkMemoryPropertyFlags", False, False, True),
            ("memoryTypeBits", "uint32_t", False, False, True),
            ("pool", "VmaPool", False, False, True),
            ("pUserData", "void", True, False, True),
            ("priority", "float", False, False, True),
        ],
        "stype": None,
    },
    "VmaAllocationInfo": {
        "members": [
            ("memoryType", "uint32_t", False, False, False),
            ("deviceMemory", "VkDeviceMemory", False, False, False),
            ("offset", "VkDeviceSize", False, False, False),
            ("size", "VkDeviceSize", False, False, False),
            ("pMappedData", "void", True, False, True),
            ("pUserData", "void", True, False, True),
        ],
        "stype": None,
    },
}

# binding tuples: key -> (create_command, create_info_struct, handle,
#                         destroy_command, extra_info_struct,
#                         extra_output_handle)
# extra_info_struct: a second const CreateInfo the create call takes
#   (the allocation parameters, unlike anything in core Vulkan)
# extra_output_handle: the VmaAllocation produced alongside the primary
#   handle; destroyed by the same destroy command
VMA_BINDINGS = {
    "VmaAllocator": {
        "create_command": "vmaCreateAllocator",
        "create_info_struct": "VmaAllocatorCreateInfo",
        "handle": "VmaAllocator",
        "destroy_command": "vmaDestroyAllocator",
        "extra_info_struct": None,
        "extra_output_handle": None,
        # the allocator needs the vulkan function loader: the generated
        # class takes a VmaVulkanFunctions table as a constructor
        # parameter unless the user enables VMA_STATIC_VULKAN_FUNCTIONS
    },
    "VmaBuffer": {
        "create_command": "vmaCreateBuffer",
        "create_info_struct": "VkBufferCreateInfo",
        "handle": "VkBuffer",
        "destroy_command": "vmaDestroyBuffer",
        "extra_info_struct": "VmaAllocationCreateInfo",
        "extra_output_handle": "VmaAllocation",
    },
    "VmaImage": {
        "create_command": "vmaCreateImage",
        "create_info_struct": "VkImageCreateInfo",
        "handle": "VkImage",
        "destroy_command": "vmaDestroyImage",
        "extra_info_struct": "VmaAllocationCreateInfo",
        "extra_output_handle": "VmaAllocation",
    },
}

VMA_INCLUDE = "#include <vk_mem_alloc.h>"

# call-function signatures for functions: entries (name, params,
# return type, success codes); params: (name, type, ptr, const)
VMA_CALLS = {
    "vmaMapMemory": {
        "params": [
            ("allocator", "VmaAllocator", False, False),
            ("allocation", "VmaAllocation", False, False),
            ("ppData", "void*", True, False),
        ],
        "return": "VkResult",
        "success": ["VK_SUCCESS"],
    },
    "vmaUnmapMemory": {
        "params": [
            ("allocator", "VmaAllocator", False, False),
            ("allocation", "VmaAllocation", False, False),
        ],
        "return": "void",
        "success": [],
    },
    "vmaGetAllocationInfo": {
        "params": [
            ("allocator", "VmaAllocator", False, False),
            ("allocation", "VmaAllocation", False, False),
            ("pAllocationInfo", "VmaAllocationInfo", True, False),
        ],
        "return": "void",
        "success": [],
    },
}
