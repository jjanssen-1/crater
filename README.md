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

Requires Python ≥ 3.10, PyYAML and a Vulkan registry (`vk.xml`). crater
finds it via `$VULKAN_SDK` / system locations, or you point it at one
with `--registry` / `$CRATER_VK_XML`.

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

### Capabilities (queries and selection)

Some values can only be known at runtime - surface formats, present
modes, available extensions. A **capabilities class** queries them and
selects values per your rules; other classes then consume the results.

```yaml
SurfaceCapabilities:
  physicalDevice: {{ VkPhysicalDevice physicalDevice }}  # query inputs
  surface: {{ VkSurfaceKHR surface }}
  VkSurfaceFormatKHR:            # queried type (inferred from vk.xml)
    candidates:                  # ordered preference; first match wins
      - [VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR]
    as: surfaceFormat            # name of the result member
  VkPresentModeKHR:
    candidates: [VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_FIFO_KHR]
    fallback: any                # take any available if none match
  VkExtensionProperties:         # availability query (extensions/layers)
    required: [VK_KHR_swapchain]   # throws when missing
    optional: [VK_KHR_display]     # becomes a hasXxx flag
  VkSurfaceCapabilitiesKHR: {}   # plain struct query
```

Which command answers a query, its two-call count/array protocol and
its required inputs are all inferred from the registry. Query inputs
are constructor parameters; declared either as a placeholder, a
`Class.object` reference (e.g. `instance: VulkanInstance.mainInstance`)
or left to the default raw handle.

Consuming classes reference results with member paths, which
automatically adds the capabilities class as a constructor parameter:

```yaml
  VkSwapchainKHR:
    imageFormat: SurfaceCapabilities.surfaceFormat.format
    presentMode: SurfaceCapabilities.presentMode
```

Extensions/layers are *not* auto-enabled: capabilities report and
select, enabling stays explicit in the CreateInfos.

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
`vk.xml`. The registry is read from your Vulkan SDK (`$VULKAN_SDK`),
a system location, `--registry` or `$CRATER_VK_XML`; it is not bundled.

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
Phase 2 (in progress): capability queries with preference-based
selection and availability checks.
Planned: window-system integration, physical device selection helpers,
command buffer recording.

## License

MIT.
