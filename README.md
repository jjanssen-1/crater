# crater

Declarative Vulkan object generation. Describe your Vulkan objects in
YAML, get RAII C++ classes that create them in dependency order and
destroy them automatically.

Vulkan initialization is repetitive: every object needs a
`Vk*CreateInfo` with an `sType`, correctly sized arrays, pointer
plumbing, parent handles and a matching destroy call. crater removes
that boilerplate without hiding anything: every emitted line maps to
information taken from the official Vulkan registry (`vk.xml`), and
all generated code is plain, readable C++ with zero runtime overhead.

## Install

```bash
pip install .
```

Requires Python ≥ 3.10 and PyYAML.

## Usage

```bash
crater generate spec.yaml -o vulkan_objects.hpp
crater bindings VkGraphicsPipeline   # inspect what the registry inferred
```

## Spec format

A spec file is a mapping of **classes** (the generated C++ classes) to
**Vulkan objects**. Object names are the names inferred from the
creation function in the registry — `VkGraphicsPipeline`, not `VkPipeline`,
because graphics pipelines do not have their own handle type.

```yaml
VulkanInstance:
  VkInstance:
    name: mainInstance
    applicationInfo:
      applicationName: "My App"
      engineName: "MyEngine"
      apiVersion: [1, 3, 0]

OurPipeline:
  VkPipelineLayout:
    name: layout
  VkGraphicsPipeline:
    layout: layout                    # sibling reference -> creation order
    renderPass: VulkanInstance.pass   # cross-class reference
    stages:                           # arrays get their count members filled
      - stage: VK_SHADER_STAGE_VERTEX_BIT
        module: {{ vertexModule }}    # value provided by the caller
```

This generates two classes:

```cpp
VulkanInstance(); // creates and later destroys the VkInstance
OurPipeline(const VulkanInstance& vulkanInstance, VkDevice vkDevice,
            VkShaderModule vertexModule);
```

### Values

* **Literals** are emitted verbatim (`VK_SHADER_STAGE_VERTEX_BIT`, `1.0`,
  `true` → `VK_TRUE`); strings assigned to `char*` members become
  `std::string`s with correct lifetime; `[major, minor, patch]` becomes
  `VK_MAKE_API_VERSION`.
* **`{{ [type] name }}`** declares a constructor parameter. The type is
  inferred from the corresponding registry member when omitted (pointer
  members yield pointer parameters).
* **Sibling references** (`layout: layout`) name another object in the
  same class; crater infers the creation order automatically.
* **Cross-class references** (`renderPass: VulkanInstance.pass`) add a
  constructor parameter of the other generated class.
* Everything else — `sType` values, count members, allocators
  (`nullptr`), optional handles like `pipelineCache`
  (`VK_NULL_HANDLE`) — is derived from `vk.xml`. You never specify it.

### Grouping and lifetimes

Objects that "stand together" are grouped in one class; the class
creates them in dependency order and destroys them in reverse order.
Handles that a destructor needs (e.g. the `VkDevice` of a pipeline) are
stored automatically.

## Registry-driven, no special cases

crater contains **no per-type logic**. Which command creates an object,
which `Vk*CreateInfo` it takes, which parameters are parents, what the
destroy command is and whether creation is plural
(`vkCreateGraphicsPipelines`) is all inferred from command signatures in
`vk.xml` (vendored under `crater/data`, pass `--registry` to use
another copy).

## Benchmark: the triangle

`examples/triangle.yaml` declares a complete Vulkan triangle setup —
instance, device, two shader modules, pipeline layout, render pass,
graphics pipeline, swapchain and semaphores — in ~75 lines of YAML.
The generated header compiles with plain `g++` against the Khronos
Vulkan headers, and `examples/main.cpp` shows the full instantiation.

## Development

```bash
pip install -e . pytest
python -m pytest
```

Tests include compile checks of generated code (skipped automatically if
no Vulkan headers are found; set `VK_HEADERS_DIR` or install the
`vulkan` pip package's headers).

## Status

Phase 1: object creation/destruction with full registry inference.
Planned: window-system integration, physical device selection helpers,
command buffer recording.

## License

MIT. The vendored `crater/data/vk.xml` is Copyright 2015-2026 The
Khronos Group Inc. (SPDX: Apache-2.0 OR MIT), see
`crater/data/VULKAN_REGISTRY_LICENSE`.
