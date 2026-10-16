import pytest

from crater.generator import GenerationError
from crater.registry import Registry
from crater.spec import SpecError, parse_spec

from conftest import compile_generated, generate_from_spec


def test_functions_section_generates(registry: Registry, tmp_path):
    out = generate_from_spec(
        """
functions:
  waitFrameFence:
    command: vkWaitForFences
    waitAll: true
    timeout: 1000000000
  beginOneTime:
    command: vkBeginCommandBuffer
    flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
  acquireFrameImage:
    command: vkAcquireNextImageKHR
    timeout: 1000000000
""",
        registry,
    )
    # hardcoded values baked in
    assert "vkWaitForFences(device, fenceCount, pFences, VK_TRUE, 1000000000)" in out
    assert "vkAcquireNextImageKHR(device, swapchain, 1000000000, semaphore, fence, &pImageIndex)" in out
    assert "beginOneTimeInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;" in out
    # success codes from the registry
    assert "result__ == VK_SUCCESS || result__ == VK_TIMEOUT" in out
    assert compile_generated(out, tmp_path)


def test_function_info_struct_stype(registry: Registry):
    out = generate_from_spec(
        """
functions:
  begin:
    command: vkBeginCommandBuffer
""",
        registry,
    )
    assert "beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;" in out


def test_function_requires_explicit_name_in_class(registry: Registry):
    # per-class call entries still exist but must be named explicitly
    with pytest.raises(SpecError):
        parse_spec(
            """
C:
  vkBeginCommandBuffer: {}
""",
            registry,
        )


def test_function_may_not_overshadow_command(registry: Registry):
    with pytest.raises(SpecError):
        parse_spec(
            """
functions:
  vkBeginCommandBuffer:
    command: vkBeginCommandBuffer
""",
            registry,
        )


def test_function_unknown_command_rejected(registry: Registry):
    with pytest.raises(SpecError):
        parse_spec(
            """
functions:
  f:
    command: vkNotARealCommand
""",
            registry,
        )


def test_function_unknown_struct_member_rejected(registry: Registry):
    with pytest.raises(GenerationError):
        generate_from_spec(
            """
functions:
  begin:
    command: vkBeginCommandBuffer
    notAMember: 5
""",
            registry,
        )
