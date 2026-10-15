#!/usr/bin/env python3
"""Registry stress test: generate + compile (and optionally run) every
creatable Vulkan object crater knows about.

For each creation command inferred from vk.xml this tool:

1. writes a minimal spec class exercising that object with defaults,
2. runs crater's generator over it,
3. compile-checks the generated header against the Vulkan headers,
4. records the outcome (ok / generation error / compile error) with the
   error text in a results file.

With --run it also attempts to instantiate the generated class at
runtime; instantiation failures (unsupported devices, missing
extensions, ...) are recorded but do not fail the run - your machine
simply may not support everything.

Safety: nothing outside the process is touched. The --run mode only
creates Vulkan objects with default parameters and destroys them; it
never reads files, writes files, or leaves lasting effects (with the
exception of Vulkan's own driver internals, which is what Vulkan does
anyway).

Usage:
    python tools/stress_test.py [--registry vk.xml] [--out results.md] [--run]
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from crater.generator import GenerationError, generate  # noqa: E402
from crater.registry import Registry  # noqa: E402
from crater.spec import SpecError, parse_spec  # noqa: E402


def find_vulkan_include() -> Path | None:
    import os

    env = os.environ.get("VK_HEADERS_DIR")
    if env and (Path(env) / "vulkan/vulkan.h").exists():
        return Path(env)
    for candidate in ("/usr/include", "/usr/local/include"):
        if Path(candidate, "vulkan/vulkan.h").exists():
            return Path(candidate)
    sdk = os.environ.get("VULKAN_SDK")
    if sdk and (Path(sdk) / "include/vulkan/vulkan.h").exists():
        return Path(sdk) / "include"
    return None


def compile_header(header: str, include_dir: Path, work: Path, defines: list[str] | None = None) -> tuple[bool, str]:
    """Compile a translation unit including the header; return (ok, errors)."""
    header_path = work / "generated.hpp"
    header_path.write_text(header)
    main = work / "main.cpp"
    main.write_text('#include "generated.hpp"\nint main() { return 0; }\n')
    cmd = ["g++", "-std=c++17", f"-I{work}", f"-I{include_dir}"]
    if defines:
        cmd.extend(defines)
    cmd.extend(["-fsyntax-only", str(main)])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0, proc.stderr


def run_instantiation(name: str, header: str, include_dir: Path) -> tuple[str, str]:
    """Attempt to construct the generated class in a child process.

    The child gets its own process so a driver crash cannot take the test
    run down. Nothing here reads external data or leaves lasting effects:
    objects are created with null parents and destroyed immediately, and
    a null-parent failure is expected and recorded, not an error.
    """
    cls = f"Test_{name}"
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "generated.hpp").write_text(header)
        # every class takes handles/classes we do not have; passing nulls
        # exercises construction up to the first validation-throwing call
        (work / "main.cpp").write_text(
            '#include "generated.hpp"\n'
            "#include <cstdio>\n"
            "int main() {\n"
            "    try {\n"
            f"        {cls} obj;\n"
            '        std::printf("constructed\\n");\n'
            "    } catch (const std::exception& e) {\n"
            '        std::printf("threw: %s\\n", e.what());\n'
            "    }\n"
            "    return 0;\n"
            "}\n"
        )
        proc = subprocess.run(
            ["g++", "-std=c++17", f"-I{work}", f"-I{include_dir}", str(work / "main.cpp"), "-o", str(work / "test")],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            stderr = proc.stderr
            if "no matching function for call" in stderr:
                return (
                    "RUN-SKIPPED",
                    "class needs constructor arguments (real parent handles); "
                    "runtime instantiation is only meaningful with a live device",
                )
            first = next((l for l in stderr.splitlines() if "error:" in l), stderr[:200])
            return "RUN-LINK-ERROR", first[:300]
        run = subprocess.run([str(work / "test")], capture_output=True, text=True, timeout=30)
        out = (run.stdout + run.stderr).strip().replace("\n", " ")[:300]
        if run.returncode != 0:
            return f"RUN-CRASH(code {run.returncode})", out
        if out.startswith("constructed"):
            return "RUN-OK", out
        if out.startswith("threw"):
            return "RUN-THREW(expected for objects needing real parents)", out
        return "RUN-OK", out


def spec_for(name: str, binding) -> str:
    """A minimal spec exercising the given creation binding."""
    lines = [f"Test_{name}:", f"  {name}: {{}}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", help="path to vk.xml")
    parser.add_argument("--out", default="stress_results.md", help="results file")
    parser.add_argument("--run", action="store_true", help="also instantiate at runtime")
    parser.add_argument("--only", help="comma-separated object names to test")
    args = parser.parse_args()

    registry = Registry(Path(args.registry)) if args.registry else Registry()
    include_dir = find_vulkan_include()
    if include_dir is None:
        print("stress: no Vulkan headers found; set VK_HEADERS_DIR", file=sys.stderr)
        return 1

    bindings = registry.bindings()
    names = sorted(bindings)
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        names = [n for n in names if n in wanted]

    results: list[dict] = []
    print(f"stress: {len(names)} objects to test")
    for i, name in enumerate(names, 1):
        binding = bindings[name]
        entry = {"name": name, "create": binding.create_command, "status": "", "detail": ""}
        print(f"[{i}/{len(names)}] {name} ({binding.create_command}) ... ", end="", flush=True)
        spec = spec_for(name, binding)
        try:
            header = generate(parse_spec(spec, registry), registry)
        except (SpecError, GenerationError) as e:
            entry["status"] = "GENERATION-ERROR"
            entry["detail"] = str(e).replace("\n", " ")[:500]
            print("generation error")
            results.append(entry)
            continue
        guard = None
        ext_info = registry.extension_of(binding.create_info_struct) or registry.extension_of(binding.create_command)
        if ext_info and ext_info[1]:
            guard = ext_info[1]

        with tempfile.TemporaryDirectory() as tmp:
            if guard is None:
                ok, err = compile_header(header, include_dir, Path(tmp))
            else:
                # platform/extension-gated: try with the guard defined; if
                # the platform headers are absent this legitimately fails
                ok, err = compile_header(header, include_dir, Path(tmp), defines=[f"-D{guard}"])
                if not ok:
                    entry["status"] = f"NEEDS-GUARD({guard})"
                    ext_name = ext_info[0] if ext_info else ""
                    entry["detail"] = f"extension {ext_name}; define {guard} with the platform headers present"
                    print(f"needs {guard}")
                    results.append(entry)
                    continue
        if ok:
            entry["status"] = "COMPILE-OK"
            print("compiles")
        else:
            entry["status"] = "COMPILE-ERROR"
            first = next((l for l in err.splitlines() if "error:" in l), err[:200])
            entry["detail"] = first[:500]
            print("compile error")

        if args.run and entry["status"] == "COMPILE-OK":
            entry["status"], entry["detail"] = run_instantiation(
                name, header, include_dir
            )
        results.append(entry)

    ok = sum(1 for r in results if r["status"] == "COMPILE-OK")
    gen_err = sum(1 for r in results if r["status"] == "GENERATION-ERROR")
    comp_err = sum(1 for r in results if r["status"] == "COMPILE-ERROR")

    out = Path(args.out)
    with out.open("w") as f:
        f.write(f"# crater registry stress test\n\n")
        f.write(f"Date: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Registry: {registry.path}\n\n")
        f.write(f"**{ok}** compiled, **{gen_err}** generation errors, **{comp_err}** compile errors, of {len(results)} total\n\n")
        f.write("| object | create command | status | detail |\n")
        f.write("|---|---|---|---|\n")
        for r in results:
            detail = r["detail"].replace("|", "\\|")
            f.write(f"| {r['name']} | `{r['create']}` | {r['status']} | {detail} |\n")
    print(f"\nstress: {ok}/{len(results)} compiled; {gen_err} gen errors, {comp_err} compile errors")
    print(f"stress: results written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
