import pytest

from crater.registry import Registry
from crater.spec import SpecError, parse_spec

from conftest import compile_generated, generate_from_spec


def test_call_functions_generate(registry: Registry, tmp_path):
    out = generate_from_spec(
        """
Sync:
  vkWaitForFences:
    waitAll: true
    timeout: 1000000000
  vkAcquireNextImageKHR:
    timeout: 1000000000
  vkBeginCommandBuffer:
    flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
""",
        registry,
    )
    # hardcoded values baked in
    assert "vkWaitForFences(device, fenceCount, pFences, VK_TRUE, 1000000000)" in out
    assert "vkAcquireNextImageKHR(device, swapchain, 1000000000, semaphore, fence, &pImageIndex)" in out
    assert "beginCommandBufferInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;" in out
    # success codes from the registry
    assert "result__ == VK_SUCCESS || result__ == VK_TIMEOUT" in out
    assert compile_generated(out, tmp_path)


def test_call_info_struct_stype(registry: Registry):
    out = generate_from_spec(
        """
C:
  vkBeginCommandBuffer: {}
""",
        registry,
    )
    assert "beginCommandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;" in out


def test_call_unknown_command_rejected(registry: Registry):
    with pytest.raises(SpecError):
        parse_spec(
            """
C:
  vkNotARealCommand: {}
""",
            registry,
        )


def test_call_unknown_struct_member_rejected(registry: Registry):
    from crater.generator import GenerationError

    with pytest.raises(GenerationError):
        generate_from_spec(
            """
C:
  vkBeginCommandBuffer:
    notAMember: 5
""",
            registry,
        )
