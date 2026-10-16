#!/usr/bin/env python3
"""pNext showcase: chain EVERY struct that extends each emittable target.

For each CreateInfo / capability-query struct crater can emit, this
generates a spec whose pNext chain carries every registered extender of
that struct (validated against vk.xml's structextends), then
compile-checks the result. The report shows per-target coverage:
emitted pairs, platform-gated pairs, compile status and any errors.

This is the static counterpart to pnext_explorer.py (which chains only
what the hardware supports, live). Together they cover the pNext
surface from both ends: everything the registry permits (here) and
everything the device reports (explorer).

Usage:
    python tools/pnext_showcase.py [--out pnext_showcase.md] [--run-specs DIR]
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


_DECL_RE = None  # compiled lazily in main: r"\b(Vk[A-Za-z0-9]+)\b"


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


def live_run(registry, include: Path, results_path: Path, specs_dir, declared: set[str]) -> bool:
    """Generate and execute the live pNext test.

    Chains EVERY extender of every physical-device query target into
    vkGetPhysicalDevice{Features,Properties,MemoryProperties}2 queries.
    The driver recognizes the sTypes it supports and fills those structs;
    unknown ones are ignored by design (that is how every Vulkan app
    queries optional support). Strictly read-only.

    The generated runner reports which chain structs came back non-zero,
    i.e. which of the 400+ queried extensions the device actually
    filled.
    """
    import tempfile

    queries = registry.queries()
    # live-qualifying: struct query taking only a physical device
    live_targets = {}
    for cmd, qb in queries.items():
        if qb.shape != "struct" or not qb.output_struct:
            continue
        c = registry.resolve_command(cmd)
        others = [p for p in c.params if not (p.is_pointer and not p.is_const)]
        if all(p.type == "VkPhysicalDevice" for p in others) and others:
            s = registry.resolve_struct(qb.output_struct)
            n_ext = sum(
                1
                for s2 in registry.structs.values()
                if not s2.alias and not registry.is_sc_only(s2.name)
                and qb.output_struct in s2.extends
            )
            if n_ext:
                live_targets[cmd] = n_ext
    if not live_targets:
        print("showcase: no live-qualifying query targets", file=sys.stderr)
        return False

    pd_ph = "physicalDevice: " + "{{" + " VkPhysicalDevice physicalDevice " + "}}"
    spec_lines = ["LivePnextQueries:", "  " + pd_ph]
    chains = {}
    for cmd in sorted(live_targets):
        out_struct = queries[cmd].output_struct
        extenders = [
            s.name
            for s in registry.structs.values()
            if not s.alias
            and not registry.is_sc_only(s.name)
            and out_struct in s.extends
            and s.name in declared
        ]
        as_name = out_struct.replace("VkPhysicalDevice", "").replace("2", "").lower() or "q"
        chains[cmd] = (out_struct, as_name, extenders)
        spec_lines.append(f"  {cmd}:")
        spec_lines.append(f"    as: {as_name}")
        spec_lines.append("    pNext:")
        for e in extenders:
            spec_lines.append(f"      - {e}: {{}}")
    spec = "\n".join(spec_lines) + "\n"

    from crater.spec import parse_spec
    from crater.generator import generate

    header = generate(parse_spec(spec, registry), registry)
    work = Path(tempfile.mkdtemp(prefix="pnlive_"))
    gen = work / "gen"
    gen.mkdir()
    (gen / "live_pnext.hpp").write_text(header)
    if specs_dir:
        (specs_dir / "LIVE.yaml").write_text(spec)
        (specs_dir / "LIVE.hpp").write_text(header)

    # runner: report every chain struct that came back non-zero
    reports = []
    for cmd, (out_struct, as_name, extenders) in sorted(chains.items()):
        for i, e in enumerate(extenders):
            s = registry.resolve_struct(e)
            if s is None:
                continue
            # report the struct if any scalar member is non-zero; nested
            # structs (VkExtent2D etc.) have no comparison against 0
            scalars = (
                m for m in s.members
                if m.name not in ("sType", "pNext")
                and m.type in ("VkBool32", "uint32_t", "uint64_t", "int32_t", "float", "size_t", "uint8_t")
            )
            conds = " || ".join(
                f"queries_.{as_name}_pNext{i}.{m.name} != 0" for m in scalars
            )
            if not conds:
                continue
            reports.append(
                f'    if ({conds}) std::printf("  {e}' + chr(92) + 'n");'
            )
    nl = chr(92) + "n"  # a C newline escape, built explicitly
    runner_lines = [
        "// live pNext showcase: query every extender of every physical-device target",
        '#include "live_pnext.hpp"',
        "#include <cstdio>",
        "#include <vector>",
        "#include <vulkan/vulkan.h>",
        "int main() {",
        "    VkApplicationInfo app{};",
        "    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;",
        '    app.pApplicationName = "crater pnext showcase live";',
        "    app.apiVersion = VK_API_VERSION_1_1;",
        "    VkInstanceCreateInfo info{};",
        "    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;",
        "    info.pApplicationInfo = &app;",
        "    VkInstance instance = VK_NULL_HANDLE;",
        "    if (vkCreateInstance(&info, nullptr, &instance) != VK_SUCCESS) {",
        '        std::printf("vkCreateInstance failed' + nl + '");',
        "        return 1;",
        "    }",
        "    uint32_t n = 0;",
        "    vkEnumeratePhysicalDevices(instance, &n, nullptr);",
        '    if (n == 0) { std::printf("no devices' + nl + '"); return 1; }',
        "    std::vector<VkPhysicalDevice> devices(n);",
        "    vkEnumeratePhysicalDevices(instance, &n, devices.data());",
        '    std::printf("querying with every registered extender chained;' + nl + '");',
        '    std::printf("the driver fills the structs it supports:' + nl + nl + '");',
        "    try {",
        "        LivePnextQueries queries_(devices[0]);",
        '        std::printf("all queries returned.' + nl + nl + 'filled structs:' + nl + '");',
    ]
    runner_lines.extend(reports)
    runner_lines += [
        "    } catch (const std::exception& e) {",
        '        std::printf("query failed: %s' + nl + '", e.what());',
        "        vkDestroyInstance(instance, nullptr);",
        "        return 1;",
        "    }",
        "    vkDestroyInstance(instance, nullptr);",
        "    return 0;",
        "}",
    ]
    runner = chr(10).join(runner_lines)
    (work / "runner.cpp").write_text(runner)
    print("showcase: building live runner ...")
    proc = subprocess.run(
        ["g++", "-std=c++17", f"-I{gen}", f"-I{include}",
         "-DVK_ENABLE_BETA_EXTENSIONS", str(work / "runner.cpp"), "-o", str(work / "runner"), "-lvulkan"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stderr[:3000], file=sys.stderr)
        return False
    print("showcase: running live queries ...")
    run = subprocess.run([str(work / "runner")], capture_output=True, text=True, timeout=120)
    total_structs = sum(len(v[2]) for v in chains.values())
    with results_path.open("a") as f:
        f.write("\n# live run\n\n")
        f.write(f"chained {total_structs} extender structs into "
                f"{len(chains)} physical-device queries; driver filled:\n\n")
        f.write("```\n")
        f.write(run.stdout)
        if run.stderr.strip():
            f.write("\nstderr:\n")
            f.write(run.stderr)
        f.write("```\n")
        f.write(f"exit code: {run.returncode}\n\n")
    print(run.stdout)
    return run.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="pnext_showcase.md")
    parser.add_argument("--target", help="restrict to one target struct")
    parser.add_argument(
        "--run-specs",
        metavar="DIR",
        help="also write every generated spec (and header) to DIR for inspection",
    )
    parser.add_argument(
        "--live",
        metavar="FILE",
        help="generate and run a live executable: chains every extender of "
        "every physical-device query target and reports which structs the "
        "driver filled (results appended to FILE)",
    )
    args = parser.parse_args()

    registry = Registry()
    include = find_vulkan_include()
    if include is None:
        print("showcase: no Vulkan headers found", file=sys.stderr)
        return 1

    bindings = registry.bindings()
    queries = registry.queries()

    # emittable targets: CreateInfos + struct-query outputs
    targets: dict[str, str] = {}
    for b in bindings.values():
        targets.setdefault(b.create_info_struct, "create")
    for q in queries.values():
        if q.shape == "struct" and q.output_struct:
            targets.setdefault(q.output_struct, "query")

    # every extender per target (excluding Vulkan SC-only structs, which
    # are absent from normal headers)
    pairs: dict[str, list[str]] = {}
    sc_skipped = 0
    for name, s in registry.structs.items():
        if s.alias:
            continue
        for t in s.extends:
            if t in targets:
                if registry.is_sc_only(name):
                    sc_skipped += 1
                else:
                    pairs.setdefault(t, []).append(name)

    if args.target:
        pairs = {t: v for t, v in pairs.items() if t == args.target}

    specs_dir = Path(args.run_specs) if args.run_specs else None
    if specs_dir:
        specs_dir.mkdir(parents=True, exist_ok=True)

    import re
    global _DECL_RE
    _DECL_RE = re.compile(r"\b(Vk[A-Za-z0-9]+)\b")
    # types actually declared in the available headers (union of all)
    declared: set[str] = set()
    # only the always-included headers count: platform headers (xlib, 
    # android, ...) declare types but need their VK_USE_PLATFORM macro
    always = [
        include / "vulkan" / "vulkan_core.h",
        include / "vulkan" / "vulkan_beta.h",
        include / "vulkan" / "vulkan.h",
        include / "vulkan_core.h",
    ]
    for h in always:
        try:
            text = h.read_text(errors="ignore")
        except OSError:
            continue
        declared.update(_DECL_RE.findall(text))
    _DECL_RE = None  # keep linters quiet; regex is module-level

    results = []
    total_pairs = ok_pairs = gated_pairs = 0
    gen_errors = compile_errors = 0
    print(f"showcase: {len(pairs)} targets, {sum(len(v) for v in pairs.values())} pairs "
          f"({sc_skipped} SC-only extenders excluded)")

    work = Path(tempfile.mkdtemp(prefix="pnshow_"))
    for i, (target, extenders) in enumerate(sorted(pairs.items()), 1):
        gated: list[str] = []
        plain: list[str] = []
        missing = []
        for e in extenders:
            info = registry.extension_of(e)
            if info and info[1]:
                gated.append(e)
            elif e not in declared:
                missing.append(e)
            else:
                plain.append(e)
        total_pairs += len(plain)
        gated_pairs += len(gated)
        entry = {
            "target": target,
            "kind": targets[target],
            "pairs": len(plain),
            "gated": len(gated),
            "missing": len(missing),
            "status": "",
            "detail": "",
        }
        print(
            f"[{i}/{len(pairs)}] {target} ({targets[target]}): "
            f"{len(plain)} extenders, {len(gated)} platform-gated ... ",
            end="",
            flush=True,
        )

        if not plain:
            entry["status"] = "ALL-GATED"
            print("all gated")
            results.append(entry)
            continue

        # build a spec chaining every plain extender of this target
        chain = "\n".join(f"      - {e}: {{}}" for e in plain)
        if targets[target] == "create":
            # find the binding that uses this CreateInfo
            binding_name = next(
                (n for n, b in bindings.items() if b.create_info_struct == target), None
            )
            if binding_name is None:
                entry["status"] = "SKIPPED"
                print("no binding")
                results.append(entry)
                continue
            spec = f"T:\n  {binding_name}:\n    pNext:\n{chain}\n"
        else:
            cmd = next(
                (n for n, q in queries.items() if q.shape == "struct" and q.output_struct == target),
                None,
            )
            if cmd is None:
                entry["status"] = "SKIPPED"
                print("no query")
                results.append(entry)
                continue
            qb = queries[cmd]
            inputs = "\n".join(
                f"  {p.name}: {{{{ {p.type} {p.name} }}}}"
                for p in qb.inputs
            )
            spec = (
                "T:\n"
                f"{inputs}\n"
                f"  {cmd}:\n    as: q\n    pNext:\n{chain}\n"
            )

        try:
            header = generate(parse_spec(spec, registry), registry)
        except (SpecError, GenerationError) as e:
            gen_errors += 1
            entry["status"] = "GENERATION-ERROR"
            entry["detail"] = str(e).replace("\n", " ")[:400]
            print("generation error")
            results.append(entry)
            continue

        if specs_dir:
            (specs_dir / f"{target}.yaml").write_text(spec)
            (specs_dir / f"{target}.hpp").write_text(header)

        header_path = work / f"{target}.hpp"
        header_path.write_text(header)
        main_cpp = work / "main.cpp"
        main_cpp.write_text('#include "x.hpp"\nint main() { return 0; }\n')
        # include the header under a fixed name
        (work / "x.hpp").write_text(header)
        proc = subprocess.run(
            [
                "g++", "-std=c++17", f"-I{work}", f"-I{include}",
                "-DVK_ENABLE_BETA_EXTENSIONS",
                "-fsyntax-only", str(main_cpp),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            entry["status"] = "OK"
            ok_pairs += len(plain)
            print("ok")
        else:
            compile_errors += 1
            entry["status"] = "COMPILE-ERROR"
            first = next(
                (l for l in proc.stderr.splitlines() if "error:" in l),
                proc.stderr[:200],
            )
            entry["detail"] = first[:400]
            print("compile error")
        results.append(entry)

    if args.live:
        live_ok = live_run(registry, include, Path(args.live), specs_dir, declared)
        if not live_ok:
            print("showcase: live run failed", file=sys.stderr)
            return 1

    out = Path(args.out)
    with out.open("w") as f:
        f.write("# crater pNext showcase\n\n")
        f.write(f"Date: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Registry: {registry.path}\n\n")
        f.write(
            f"**{ok_pairs}/{total_pairs}** extender pairs compile across "
            f"{len(pairs)} targets ({gated_pairs} platform-gated, not chained); "
            f"**{gen_errors}** generation errors, **{compile_errors}** compile errors\n\n"
        )
        f.write("| target | kind | chained | gated | missing-in-headers | status | detail |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in results:
            detail = r["detail"].replace("|", "\\|")
            f.write(
                f"| {r['target']} | {r['kind']} | {r['pairs']} | {r['gated']} "
                f"| {r.get('missing', 0)} | {r['status']} | {detail} |\n"
            )
    print(f"\nshowcase: results written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
