import pytest

from crater.registry import Registry
from crater.spec import parse_spec, SpecError

from conftest import generate_from_spec


VMA_SPEC = """
use_vma: true

Allocator:
  VmaAllocator:
    name: allocator
    physicalDevice: {{ VkPhysicalDevice physicalDevice }}
    device: {{ VkDevice device }}

VertexStaging:
  VmaBuffer:
    name: stagingBuffer
    size: 65536
    usage: VK_BUFFER_USAGE_TRANSFER_SRC_BIT
    sharingMode: VK_SHARING_MODE_EXCLUSIVE
    allocation:
      usage: VMA_MEMORY_USAGE_AUTO_PREFER_HOST
      flags: VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT
"""


def test_vma_bindings_load(registry: Registry):
    assert not registry.vma_bindings_active()
    registry.load_vma()
    assert registry.vma_bindings_active()
    b = registry.bindings()
    assert b["VmaAllocator"].create_command == "vmaCreateAllocator"
    assert b["VmaAllocator"].destroy_command == "vmaDestroyAllocator"
    assert b["VmaBuffer"].create_command == "vmaCreateBuffer"
    assert b["VmaBuffer"].destroy_command == "vmaDestroyBuffer"
    assert b["VmaImage"].create_command == "vmaCreateImage"


def test_vma_generation(registry: Registry):
    out = generate_from_spec(VMA_SPEC, registry)
    # the VMA include is emitted
    assert "#include <vk_mem_alloc.h>" in out
    # allocator creation from the class refs
    assert "vmaCreateAllocator(&allocatorCreateInfo, &allocator_)" in out
    # buffer creation: allocator ref via getter, both CreateInfos, both outputs
    assert (
        "vmaCreateBuffer(allocator.allocator(), &stagingBufferCreateInfo, "
        "&stagingBufferAllocationCreateInfo, &stagingBuffer_, &stagingBufferAllocation_, nullptr)" in out
    )
    # dual getters
    assert "VkBuffer stagingBuffer() const { return stagingBuffer_; }" in out
    assert "VmaAllocation stagingBufferAllocation() const { return stagingBufferAllocation_; }" in out
    # destruction passes the allocation
    assert "vmaDestroyBuffer(vmaAllocator_, stagingBuffer_, stagingBufferAllocation_);" in out


def test_vma_requires_use_vma():
    """VmaBuffer without use_vma is an unknown object (a fresh registry
    that has not loaded the VMA bindings)."""
    with pytest.raises(SpecError):
        parse_spec(
            """
X:
  VmaBuffer:
    name: b
""",
            Registry(),
        )


def test_vma_allocation_params_validated(registry: Registry):
    """The allocation: struct validates its members against the VMA
    allocation CreateInfo."""
    from crater.generator import GenerationError

    with pytest.raises(GenerationError):
        generate_from_spec(
            """
use_vma: true

A:
  VmaAllocator:
    name: allocator
    physicalDevice: {{ VkPhysicalDevice pd }}
    device: {{ VkDevice dev }}
B:
  VmaBuffer:
    name: b
    size: 64
    allocation:
      notAMember: 1
""",
            registry,
        )


def test_particles_example_generates(registry: Registry, tmp_path):
    """The VMA particle example generates and compiles."""
    from pathlib import Path

    from conftest import compile_generated, generate_from_spec

    spec = Path(__file__).parent.parent / "examples" / "particles.yaml"
    out = generate_from_spec(spec.read_text(), registry)
    # VMA: allocator + persistently-mapped buffers
    assert "#include <vk_mem_alloc.h>" in out
    assert "VMA_ALLOCATION_CREATE_MAPPED_BIT" in out
    # pNext feature chain on the device
    assert "deviceCreateInfo_pNext0.timelineSemaphore = VK_TRUE;" in out
    # two command buffers in the FrameCommands class
    assert "VkCommandBuffer renderCommandBuffer" in out
    assert "VkCommandBuffer computeCommandBuffer" in out
    assert compile_generated(out, tmp_path)
