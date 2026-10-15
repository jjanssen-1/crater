# crater tools

## stress_test.py

Generates, compile-checks and (optionally) instantiates every creatable
Vulkan object crater can infer from the registry - a full sweep of what
the generator can do, with a results file at the end.

```bash
# compile-only sweep (safe, no Vulkan calls at all)
python tools/stress_test.py --out results.md

# also try runtime instantiation (creates objects with null parents,
# catches everything; crashes are isolated in child processes)
python tools/stress_test.py --run --out results.md

# restrict to specific objects
python tools/stress_test.py --only VkInstance,VkDevice
```

Requirements: g++, Vulkan headers (`VK_HEADERS_DIR` or system install),
and a vk.xml (from your SDK; see the main README).

Statuses in the results file:

| status | meaning |
|---|---|
| COMPILE-OK | generated header compiles |
| GENERATION-ERROR | the generator refused (spec/registry conflict) - a bug |
| COMPILE-ERROR | generated code does not compile - a bug |
| NEEDS-GUARD(...) | platform/extension-gated object; requires the named define and platform headers (e.g. `VK_USE_PLATFORM_ANDROID_KHR`) |
| RUN-OK | class constructed and destructed at runtime |
| RUN-THREW | construction threw (expected for objects needing real parents) |
| RUN-SKIPPED | class needs constructor arguments; runtime test only meaningful with a live device |
| RUN-CRASH | child process died (driver-level; recorded, run continues) |

Safety: nothing reads external data, writes files (beyond the results
file), or has lasting effects. Runtime attempts run in child processes
so even a driver crash cannot take the sweep down.

The last full sweep: 69/78 compile, 9 platform-gated, 0 generation
errors, 0 compile errors.
