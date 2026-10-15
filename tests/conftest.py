import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from crater.generator import generate  # noqa: E402
from crater.registry import Registry  # noqa: E402
from crater.spec import parse_spec  # noqa: E402


@pytest.fixture(scope="session")
def registry() -> Registry:
    return Registry()


@pytest.fixture(scope="session")
def bindings(registry: Registry):
    return registry.bindings()


def generate_from_spec(spec_text: str, registry: Registry) -> str:
    return generate(parse_spec(spec_text, registry), registry)


def find_vulkan_headers() -> Path | None:
    for env in ("VK_HEADERS_DIR",):
        path = os.environ.get(env)
        if path and Path(path, "include/vulkan/vulkan.h").exists():
            return Path(path)
    for candidate in ("/usr/include", "/usr/local/include"):
        if Path(candidate, "vulkan/vulkan.h").exists():
            return Path(candidate).parent
    # pip-provided headers or a local checkout
    for candidate in (
        Path("/tmp/vkheaders"),
        Path.home() / ".cache/crater/Vulkan-Headers",
    ):
        if (candidate / "include/vulkan/vulkan.h").exists():
            return candidate
    return None


def compile_generated(header: str, tmp_path: Path) -> bool:
    """Compile a translation unit that instantiates nothing but includes
    the generated header. Returns True on success."""
    headers_dir = find_vulkan_headers()
    if headers_dir is None:
        pytest.skip("Vulkan headers not available; skipping compile check")
    inc = headers_dir / "include" if (headers_dir / "include/vulkan/vulkan.h").exists() else headers_dir
    header_path = tmp_path / "generated.hpp"
    header_path.write_text(header)
    main = tmp_path / "main.cpp"
    main.write_text('#include "generated.hpp"\nint main() { return 0; }\n')
    cmd = [
        "c++" if subprocess.run(["which", "c++"], capture_output=True).returncode == 0 else "g++",
        "-std=c++17",
        f"-I{tmp_path}",
        f"-I{inc}",
        "-fsyntax-only",
        str(main),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f"generated code failed to compile:\n{proc.stderr}")
    return True
