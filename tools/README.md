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

## live_tests/

Manual specs with sensible parameters for every core device-level
object, run against a **real Vulkan device** - as opposed to the
compile-only sweep above.

```bash
tools/live_tests/run.sh my_results.md
```

- `specs/*.yaml`: hand-written object definitions (sync, memory, views,
  samplers, descriptors, queries, command pool, pipeline objects,
  render pass, framebuffer, graphics + compute pipeline)
- `harness.cpp`: constructs instance/device, then each generated class
  with real handles; every test is isolated (one failure does not stop
  the sweep) and the outcome table is written to the results file

Safety: only object creation/destruction. No rendering, no
presentation, no external reads/writes beyond the results file.

The sandbox has no GPU, so this is meant to run on real hardware -
please report the results file contents if anything fails.

## pnext_explorer.py

Queries **every extension feature/properties struct your device
supports** through registry-validated pNext chains - the live runtime
test for the pNext support, as thorough as your hardware allows:

1. **probe**: a tiny binary dumps the supported device extensions and
   the API version
2. **generate**: supported extensions are mapped to their
   feature/properties structs (via vk.xml's structextends, extension
   membership and promotion aliases); crater generates a spec whose
   `vkGetPhysicalDeviceFeatures2`/`Properties2` queries carry all of
   them in their pNext chains
3. **run**: the queries execute against your real device and the
   report lists every `VK_TRUE` feature member, plus validation output

```bash
python tools/pnext_explorer.py --out pnext_results.md
```

Safety: strictly read-only capability queries. No extensions are
enabled, no objects are created beyond the queries, nothing persistent.
Platform-gated structs are skipped automatically (their headers are
usually absent); promoted core structs are included per the device's
API version.

## pnext_showcase.py

Chains **every struct that extends each emittable target** (CreateInfo
or capability-query struct) and compile-checks the result - the static
counterpart to pnext_explorer.py:

- 56 targets, 916 registry-validated extender pairs
- platform-gated extenders (63) are classified, not chained
- Vulkan SC-only types are excluded (absent from normal headers)
- types missing from the installed headers (registry/headers skew)
  are classified and reported
- compiles with VK_ENABLE_BETA_EXTENSIONS so beta structs chain too

```bash
python tools/pnext_showcase.py --out pnext_showcase.md
# inspect generated specs and headers:
python tools/pnext_showcase.py --run-specs /tmp/pnext-specs
```

Current sweep: **916/916 compile, 0 generation errors, 0 compile
errors** across 56 targets.

### --live FILE

Generates and runs a live executable: one spec chaining **every
header-declared extender** of every physical-device query target
(Features2 ~270 structs, Properties2 ~120, MemoryProperties2), queried
against your real device. The driver fills the structs it supports and
ignores unknown sTypes (that is how capability queries are designed);
the runner reports which structs came back non-zero, and results are
appended to FILE.

```bash
python tools/pnext_showcase.py --live results.md --out pnext_showcase.md
```

Strictly read-only: nothing is enabled, created, or written beyond the
report. Platform-gated extenders are excluded automatically.

## Troubleshooting: which crater am I running?

Generated headers carry the generator's source hash in their first
line, e.g. `gen 738ba0625aac`. If your header shows a different hash
than the checkout, an older crater produced it. To find it:

```bash
which -a crater            # every crater on PATH, in order
head -1 $(which crater)    # the shim's shebang: which python runs it
type crater                # hashed path, alias, or shell function
python3 -c "import crater, crater.generator, inspect; print(crater.__file__, inspect.getsourcefile(crater.generator._generator_hash))"
pip list 2>/dev/null | grep -i crater   # which installs exist
```

The `crater.__file__` line is decisive: it prints the exact package the
import resolves to. If it is not your checkout directory, that is the
old copy.

Fixes, in order of preference:

```bash
pip install -e .           # editable install: crater runs the repo code
# or, if an old non-editable install shadows it:
pip uninstall crater && pip install -e .
# if the shell cached a moved/removed path:
hash -r                   # bash; or: rehash (zsh)
```

After installing, verify with the header stamp: regenerate and check
that `gen <hash>` matches the fresh checkout.

## Static warnings

`crater generate` prints static warnings for likely-invalid specs -
problems the session's validation runs kept finding on real hardware:

| check | origin |
|---|---|
| graphics pipeline missing a practically-required state (multisample, viewport, ...) | the null pMultisampleState crash |
| viewport/scissor counts without arrays and without dynamic states | the viewport overflow crash |
| non-optional enum members left at their zero default where 0 is not a valid value | the frontFace/priority class |
| count members left at zero while their array pointer is set | the attachmentCount/setLayoutCount class |
| pNext entries with invalid members | extended to chains |

Warnings never block generation; they print to stderr with the class
and object. The shipped examples are warning-free (tested).
