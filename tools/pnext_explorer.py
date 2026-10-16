#!/usr/bin/env python3
"""pNext explorer: query every extension feature/properties struct your
device supports, through registry-validated pNext chains.

Three phases, all automated:

1. probe:    a tiny binary dumps the supported instance/device extensions
             and the device API version (JSON)
2. generate: supported extensions are mapped to their feature/properties
             structs (via vk.xml's structextends + extension membership);
             crater generates a spec whose vkGetPhysicalDeviceFeatures2 /
             vkGetPhysicalDeviceProperties2 queries carry all of them in
             their pNext chains - plus a runner that prints every VK_TRUE
             feature member
3. run:      the queries execute against your real device; results and
             validation output are collected in a Markdown report

Safety: strictly read-only queries (the standard way to ask a device
what it supports before enabling anything). No device extensions are
enabled, no objects created beyond the queries, nothing persistent.

Usage:
    python tools/pnext_explorer.py [--out pnext_results.md]
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from crater.generator import generate  # noqa: E402
from crater.registry import Registry  # noqa: E402
from crater.spec import parse_spec  # noqa: E402

PROBE_SRC = r"""
// pnext explorer probe: dump supported extensions and API version.
#include <cstdio>
#include <vector>
#include <vulkan/vulkan.h>
int main() {
    uint32_t instanceExtCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &instanceExtCount, nullptr);
    std::vector<VkExtensionProperties> instanceExts(instanceExtCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &instanceExtCount, instanceExts.data());

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "crater pnext probe";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.pApplicationInfo = &app;
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&info, nullptr, &instance) != VK_SUCCESS) {
        std::printf("{\"error\": \"vkCreateInstance failed\"}\n");
        return 1;
    }
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        std::printf("{\"error\": \"no devices\"}\n");
        return 1;
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
    VkPhysicalDevice pd = devices[0];

    uint32_t deviceExtCount = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &deviceExtCount, nullptr);
    std::vector<VkExtensionProperties> deviceExts(deviceExtCount);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &deviceExtCount, deviceExts.data());

    uint32_t apiVersion = 0;
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(pd, &props);
    apiVersion = props.apiVersion;

    std::printf("===PNEXT-JSON-BEGIN===\n");
    std::printf("{\n  \"api_version\": %u,\n", apiVersion);
    std::printf("  \"device_extensions\": [\n");
    for (uint32_t i = 0; i < deviceExtCount; ++i)
        std::printf("    \"%s\"%s\n", deviceExts[i].extensionName,
                    i + 1 < deviceExtCount ? "," : "");
    std::printf("  ]\n}\n");
    std::printf("===PNEXT-JSON-END===\n");
    vkDestroyInstance(instance, nullptr);
    return 0;
}
"""


def build(src: str, out: Path, includes: list[str], extra: list[str] | None = None) -> bool:
    cmd = ["g++", "-std=c++17"] + [f"-I{p}" for p in includes] + (extra or [])
    proc = subprocess.run(cmd + [str(src), "-o", str(out)], capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr)
        return False
    return True


def find_vulkan_include() -> Path | None:
    import os

    for env in ("VK_HEADERS_DIR",):
        p = os.environ.get(env)
        if p and (Path(p) / "vulkan/vulkan.h").exists():
            return Path(p)
    for c in ("/usr/include", "/usr/local/include"):
        if Path(c, "vulkan/vulkan.h").exists():
            return Path(c)
    sdk = os.environ.get("VULKAN_SDK")
    if sdk and (Path(sdk) / "include/vulkan/vulkan.h").exists():
        return Path(sdk) / "include"
    return None


def structs_for(registry: Registry, supported: set[str], api_version: int, base: str) -> list[tuple[str, str]]:
    """(struct, extension) pairs extending `base` whose extension the device
    supports (or that come with the core version). Platform-gated ones are
    skipped; their headers are usually absent."""
    # promoted structs keep their extension membership only on an alias
    # name (e.g. VkPhysicalDeviceDynamicRenderingFeatures is core now,
    # but VkPhysicalDeviceDynamicRenderingFeaturesKHR still carries
    # VK_KHR_dynamic_rendering); map target -> supporting alias names
    aliases_of: dict[str, list[str]] = {}
    for aname, s in registry.structs.items():
        if s.alias:
            aliases_of.setdefault(s.alias, []).append(aname)

    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name, s in registry.structs.items():
        if s.alias or base not in s.extends:
            continue
        # the struct's own extension, or one of its aliases'
        candidates = [name] + aliases_of.get(name, [])
        info = None
        for candidate in candidates:
            info = registry.extension_of(candidate)
            if info is not None:
                break
        if info is None:
            continue  # core-version struct; handled below
        ext, protect = info
        if protect:
            continue
        if ext in supported:
            resolved = registry.resolve_struct(name)
            key = resolved.resolved_name if resolved else name
            if key not in seen:
                seen.add(key)
                out.append((name, ext))
    # core-version structs, gated by the device API version and only
    # for the bases they actually extend
    major = (api_version >> 22) & 0x7F
    minor = (api_version >> 12) & 0x3FF
    core_candidates = []
    if (major, minor) >= (1, 1):
        core_candidates += [
            ("VkPhysicalDeviceVulkan11Features", "core 1.1"),
            ("VkPhysicalDeviceVulkan11Properties", "core 1.1"),
        ]
    if (major, minor) >= (1, 2):
        core_candidates += [
            ("VkPhysicalDeviceVulkan12Features", "core 1.2"),
            ("VkPhysicalDeviceVulkan12Properties", "core 1.2"),
        ]
    if (major, minor) >= (1, 3):
        core_candidates += [("VkPhysicalDeviceVulkan13Features", "core 1.3")]
    for name, ext in core_candidates:
        s = registry.resolve_struct(name)
        if s is not None and base in s.extends and name not in seen:
            seen.add(name)
            out.append((name, ext))
    return sorted(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="pnext_results.md")
    args = parser.parse_args()

    include = find_vulkan_include()
    if include is None:
        print("pnext: no Vulkan headers found", file=sys.stderr)
        return 1

    registry = Registry()

    with tempfile.TemporaryDirectory(prefix="pnext_") as tmp_s:
        tmp = Path(tmp_s)
        probe_src = tmp / "probe.cpp"
        probe_src.write_text(PROBE_SRC)
        probe = tmp / "probe"
        print("pnext: building probe ...")
        if not build(probe_src, probe, [str(include)], ["-lvulkan"]):
            return 1
        print("pnext: probing device ...")
        # validation layers print to stdout too; disable them for the
        # probe (pure enumeration, nothing to validate) and slice the
        # JSON between sentinels regardless
        import os
        env = dict(os.environ)
        env["VK_LOADER_LAYERS_DISABLE"] = "*khronos_validation*"
        proc = subprocess.run([str(probe)], capture_output=True, text=True, timeout=60, env=env)
        try:
            begin = proc.stdout.index("===PNEXT-JSON-BEGIN===") + len("===PNEXT-JSON-BEGIN===")
            end = proc.stdout.index("===PNEXT-JSON-END===")
            device_info = json.loads(proc.stdout[begin:end])
        except (ValueError, json.JSONDecodeError):
            print("pnext: probe failed:", proc.stdout, proc.stderr, file=sys.stderr)
            return 1
        if "error" in device_info:
            print("pnext:", device_info["error"], file=sys.stderr)
            return 1

        supported = set(device_info["device_extensions"])
        api_version = device_info["api_version"]
        feature_structs = structs_for(registry, supported, api_version, "VkPhysicalDeviceFeatures2")
        property_structs = structs_for(registry, supported, api_version, "VkPhysicalDeviceProperties2")
        print(f"pnext: device supports {len(supported)} extensions; "
              f"{len(feature_structs)} feature structs, {len(property_structs)} properties structs")

        # --- the declarative spec: two queries, full pNext chains ---------
        chain_f = "\n".join(f"      - {name}: {{}}" for name, _ in feature_structs)
        chain_p = "\n".join(f"      - {name}: {{}}" for name, _ in property_structs)
        pd_placeholder = "physicalDevice: " + "{{" + " VkPhysicalDevice physicalDevice " + "}}"
        spec = (
            "PnextCapabilities:\n"
            f"  {pd_placeholder}\n"
            "  vkGetPhysicalDeviceFeatures2:\n"
            "    as: features\n"
            "    pNext:\n"
            f"{chain_f}\n"
            "  vkGetPhysicalDeviceProperties2:\n"
            "    as: properties\n"
            "    pNext:\n"
            f"{chain_p}\n"
        )
        (tmp / "spec.yaml").write_text(spec)
        header = generate(parse_spec(spec, registry), registry)
        gen = tmp / "generated"
        gen.mkdir()
        (gen / "pnext_caps.hpp").write_text(header)

        # --- generated runner: print every VK_TRUE feature member ---------
        prints: list[str] = []
        for i, (name, ext) in enumerate(feature_structs):
            s = registry.resolve_struct(name)
            if s is None:
                continue
            bools = [m.name for m in s.members if m.type == "VkBool32"]
            if not bools:
                continue
            prints.append(f'    std::printf("## {ext} ({name})\\n");')
            for m in bools:
                prints.append(
                    f'    if (caps.features_pNext{i}.{m}) '
                    f'std::printf("  {m}\\n");'
                )
        runner_src = f"""
// pnext explorer runner: queries every supported extension feature struct
#include "pnext_caps.hpp"
#include <cstdio>
int main() {{
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(VK_NULL_HANDLE, &n, nullptr); // placeholder
    return 0;
}}
"""
        # the runner needs a real instance; embed the probe's setup
        runner_src = f"""
// pnext explorer runner
#include "pnext_caps.hpp"
#include <cstdio>
#include <vector>
#include <vulkan/vulkan.h>
int main() {{
    VkApplicationInfo app{{}};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "crater pnext explorer";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo info{{}};
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.pApplicationInfo = &app;
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&info, nullptr, &instance) != VK_SUCCESS) {{
        std::printf("vkCreateInstance failed\\n");
        return 1;
    }}
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(instance, &n, nullptr);
    std::vector<VkPhysicalDevice> devices(n);
    vkEnumeratePhysicalDevices(instance, &n, devices.data());
    if (n == 0) {{ std::printf("no devices\\n"); return 1; }}

    std::printf("# pnext explorer: querying %zu feature structs, %zu properties structs\\n\\n",
                {len(feature_structs)}, {len(property_structs)});
    PnextCapabilities caps(devices[0]);
    std::printf("queries complete.\\n\\n");
{chr(10).join(prints)}
    vkDestroyInstance(instance, nullptr);
    return 0;
}}
"""
        runner = tmp / "runner.cpp"
        runner.write_text(runner_src)
        runner_bin = tmp / "runner"
        print("pnext: building runner ...")
        if not build(runner, runner_bin, [str(gen), str(include)], ["-lvulkan"]):
            return 1
        print("pnext: running queries ...")
        run = subprocess.run([str(runner_bin)], capture_output=True, text=True, timeout=120)

        # --- report --------------------------------------------------------
        out = Path(args.out)
        with out.open("w") as f:
            f.write("# crater pNext explorer results\n\n")
            f.write(f"Date: {datetime.datetime.now().isoformat()}\n\n")
            f.write(f"Device API version: {api_version >> 22}.{(api_version >> 12) & 0x3FF}.{api_version & 0xFFF}\n")
            f.write(f"Supported device extensions: {len(supported)}\n")
            f.write(f"Feature structs queried: {len(feature_structs)}; "
                    f"properties structs queried: {len(property_structs)}\n\n")
            f.write("## Supported features (VK_TRUE members per extension struct)\n\n")
            f.write("```\n")
            f.write(run.stdout)
            if run.stderr.strip():
                f.write("\n## stderr (validation output)\n\n```\n")
                f.write(run.stderr)
                f.write("\n```\n")
            f.write(f"\nexit code: {run.returncode}\n")
        print(f"pnext: results written to {out} (runner exit {run.returncode})")
        return 0


if __name__ == "__main__":
    sys.exit(main())
