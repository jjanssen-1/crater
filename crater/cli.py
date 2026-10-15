"""Command line entry point for crater.

Usage::

    crater generate spec.yaml -o generated.hpp
    crater bindings [VkName ...]   # inspect inferred registry bindings
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .generator import GenerationError, generate
from .registry import Registry, RegistryNotFoundError
from .spec import SpecError, parse_spec


def _load_registry(args: argparse.Namespace) -> Registry:
    try:
        return Registry(Path(args.registry)) if args.registry else Registry()
    except RegistryNotFoundError as e:
        print(f"crater: {e}", file=sys.stderr)
        raise SystemExit(1)


def _cmd_generate(args: argparse.Namespace) -> int:
    registry = _load_registry(args)
    try:
        classes = parse_spec(Path(args.spec).read_text(), registry)
        output = generate(classes, registry)
    except (SpecError, GenerationError) as e:
        print(f"crater: {e}", file=sys.stderr)
        return 1
    out_path = Path(args.output) if args.output else None
    if out_path is None:
        sys.stdout.write(output)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output)
        print(f"wrote {out_path}")
    return 0


def _cmd_bindings(args: argparse.Namespace) -> int:
    registry = _load_registry(args)
    bindings = registry.bindings()
    names = args.names or sorted(bindings)
    for name in names:
        b = bindings.get(name)
        if b is None:
            print(f"{name}: no creation command inferred")
            continue
        print(
            f"{name}: {b.create_command}({b.create_info_struct})"
            f" -> {b.handle}; destroy: {b.destroy_command or 'none'}"
            f"; parents: {', '.join(b.parent_handles) or 'none'}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crater", description="Declarative Vulkan object generator")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate C++ from a YAML spec")
    gen.add_argument("spec", help="path to the YAML spec file")
    gen.add_argument("-o", "--output", help="output header path (default: stdout)")
    gen.add_argument("--registry", help="path to vk.xml (default: vendored copy)")
    gen.set_defaults(func=_cmd_generate)

    bind = sub.add_parser("bindings", help="list inferred create/destroy bindings")
    bind.add_argument("names", nargs="*", help="object names to inspect (default: all)")
    bind.add_argument("--registry", help="path to vk.xml (default: vendored copy)")
    bind.set_defaults(func=_cmd_bindings)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
