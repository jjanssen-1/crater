"""C++ code generation from the spec IR.

For every class in the spec a RAII C++ class is generated that

* takes constructor parameters for every `{{ placeholder }}` and every
  external Class.object reference,
* creates the Vulkan objects in dependency order,
* stores handles needed by the destructor,
* destroys all objects in reverse creation order.

Everything else (sType values, count members, pointer members, plural
creation functions) is derived from the registry.
"""

from __future__ import annotations

import re
from typing import Optional

from .registry import Member, Registry, Struct
from .spec import ArrayValue, ClassDef, ObjectDef, Placeholder, Ref, StructValue

_VERSION_RE = re.compile(r"^\s*(\d+)\s*\.\s*(\d+)\s*(?:\.\s*(\d+))?\s*$")


class GenerationError(Exception):
    pass


def _lower_first(s: str) -> str:
    return s[0].lower() + s[1:] if s else s


def _indent(lines: list[str], n: int = 4) -> list[str]:
    pad = " " * n
    return [pad + line if line else line for line in lines]


def _version_expr(value) -> Optional[str]:
    if isinstance(value, list) and len(value) in (2, 3) and all(isinstance(v, int) for v in value):
        nums = list(value) + [0] * (3 - len(value))
        return f"VK_MAKE_API_VERSION(0, {nums[0]}, {nums[1]}, {nums[2]})"
    if isinstance(value, str):
        m = _VERSION_RE.match(value)
        if m:
            major, minor, patch = m.group(1), m.group(2), m.group(3) or "0"
            return f"VK_MAKE_API_VERSION(0, {major}, {minor}, {patch})"
    return None


def _member_var_name(member_name: str) -> str:
    if member_name.startswith("p") and member_name[1:2].isupper():
        return member_name[1:2].lower() + member_name[2:]
    return member_name


class ClassGenerator:
    """Generates one RAII C++ class from a ClassDef."""

    def __init__(self, class_def: ClassDef, registry: Registry, class_lookup: dict[str, ClassDef]):
        self.cd = class_def
        self.registry = registry
        self.class_lookup = class_lookup
        self.objects = {o.instance_name: o for o in class_def.objects}
        self._placeholder_types: dict[str, str] = {}

    # ------------------------------------------------------------- ordering

    def ordered_objects(self) -> list[ObjectDef]:
        name_to_obj = {o.instance_name: o for o in self.cd.objects}
        deps: dict[str, set[str]] = {o.instance_name: set() for o in self.cd.objects}
        for o in self.cd.objects:
            for ref in self._refs(o):
                if ref.kind == "sibling" and ref.object_name in name_to_obj:
                    deps[o.instance_name].add(ref.object_name)
            for handle_type in self._required_handle_types(o):
                for other in self.cd.objects:
                    if other is not o and other.binding.handle == handle_type:
                        deps[o.instance_name].add(other.instance_name)
        order: list[ObjectDef] = []
        remaining = list(self.cd.objects)
        while remaining:
            progressed = False
            for o in list(remaining):
                if deps[o.instance_name] <= {x.instance_name for x in order}:
                    order.append(o)
                    remaining.remove(o)
                    progressed = True
            if not progressed:
                names = [o.instance_name for o in remaining]
                raise GenerationError(
                    f"class '{self.cd.name}': circular or unresolvable object references: {names}"
                )
        return order

    # ------------------------------------------------------------- traversal

    def _refs(self, obj: ObjectDef) -> list[Ref]:
        out: list[Ref] = []

        def walk(v):
            if isinstance(v, Ref):
                out.append(v)
            elif isinstance(v, StructValue):
                for x in v.fields.values():
                    walk(x)
            elif isinstance(v, ArrayValue):
                for x in v.items:
                    walk(x)

        for v in obj.params.values():
            walk(v)
        return out

    def _walk_spec(self, o: ObjectDef):
        """Yield (struct, member, value, base_name) for every spec value of
        the object, descending into nested structs and arrays."""
        struct = self.registry.resolve_struct(o.binding.create_info_struct)
        yield from self._walk_struct(struct, o.params, o, o.instance_name)

    def _walk_struct(self, struct: Optional[Struct], fields: dict, o: ObjectDef, base: str):
        if struct is None:
            return
        for key, value in fields.items():
            member = self._match_member(struct, key)
            if member is None:
                raise GenerationError(
                    f"class '{self.cd.name}', object '{o.instance_name}': '{key}' is not a "
                    f"member of {struct.resolved_name}"
                )
            yield struct, member, value, base
            sub = self.registry.resolve_struct(member.type) if member.type.startswith("Vk") else None
            if isinstance(value, StructValue) and sub is not None:
                yield from self._walk_struct(sub, value.fields, o, f"{base}_{_member_var_name(member.name)}")
            elif isinstance(value, ArrayValue) and sub is not None:
                for i, item in enumerate(value.items):
                    if isinstance(item, StructValue):
                        yield from self._walk_struct(
                            sub, item.fields, o, f"{base}_{_member_var_name(member.name)}_item{i}"
                        )

    def _match_member(self, struct: Struct, key: str) -> Optional[Member]:
        for m in struct.members:
            if m.name == key:
                return m
        for m in struct.members:
            if _member_var_name(m.name) == key:
                return m
        return None

    # ----------------------------------------------------------- handle needs

    def _required_handle_types(self, obj: ObjectDef) -> list[str]:
        """Handle types this object needs for creation and destruction."""
        types: list[str] = []
        cmd = self.registry.resolve_command(obj.binding.create_command)
        if cmd is None:
            raise GenerationError(f"unknown command {obj.binding.create_command}")
        for p in cmd.params:
            if p.name in obj.binding.parent_handles and p.type in self.registry.handles and not p.optional:
                if p.type not in types:
                    types.append(p.type)
        # handles needed by explicit Class.object references in values are
        # covered by the external class parameter
        if obj.binding.destroy_command:
            dcmd = self.registry.resolve_command(obj.binding.destroy_command)
            if dcmd is not None:
                for p in dcmd.params:
                    if p.name in obj.binding.destroy_parent_handles and p.type in self.registry.handles:
                        if p.type not in types:
                            types.append(p.type)
        return types

    def _sibling_for_handle(self, handle_type: str) -> Optional[ObjectDef]:
        for o in self.cd.objects:
            if o.binding.handle == handle_type:
                return o
        return None

    def _external_for_handle(self, handle_type: str) -> Optional[tuple[str, str]]:
        for other_name, other in self.class_lookup.items():
            if other_name == self.cd.name:
                continue
            for o in other.objects:
                if o.binding.handle == handle_type:
                    return _lower_first(other_name), o.instance_name
        return None

    # ------------------------------------------------------ constructor shape

    def ctor_params(self) -> list[tuple[str, str]]:
        """(type, name) of generated constructor parameters, in order."""
        params: list[tuple[str, str]] = []
        seen: set[str] = set()

        # external class references first
        for o in self.ordered_objects():
            for ref in self._refs(o):
                if ref.kind == "external":
                    pname = _lower_first(ref.class_name)
                    if pname not in seen:
                        params.append((f"const {ref.class_name}&", pname))
                        seen.add(pname)

        # handle parameters needed for creation/destruction, either as a
        # reference to another generated class or as a raw handle parameter
        for o in self.ordered_objects():
            for handle_type in self._required_handle_types(o):
                if self._sibling_for_handle(handle_type) is not None:
                    continue
                ext = self._external_for_handle(handle_type)
                if ext is not None:
                    pname = ext[0]
                    if pname not in seen:
                        params.append((f"const {ext[0][0].upper() + ext[0][1:]}&", pname))
                        seen.add(pname)
                    continue
                pname = _lower_first(handle_type)
                if pname not in seen:
                    params.append((handle_type, pname))
                    seen.add(pname)

        # placeholders last
        for o in self.ordered_objects():
            for struct, member, value, base in self._walk_spec(o):
                if isinstance(value, Placeholder):
                    if value.name not in seen:
                        params.append((self._placeholder_type(member, value), value.name))
                        seen.add(value.name)
        return params

    def _placeholder_type(self, member: Member, ph: Placeholder) -> str:
        if ph.explicit_type:
            return ph.explicit_type
        return member.type

    # ------------------------------------------------------------ generation

    def generate(self) -> str:
        ctor = self.ctor_params()
        param_list = ", ".join(f"{t} {n}" for t, n in ctor)
        inits = self._ctor_init_list()

        lines: list[str] = [f"class {self.cd.name} {{", "public:"]
        lines.append(f"    explicit {self.cd.name}({param_list}){(' : ' + ', '.join(inits)) if inits else ''} {{")
        body: list[str] = []
        for o in self.ordered_objects():
            body.append("")
            body.append(f"// {o.vulkan_name} '{o.instance_name}'")
            body.extend(self._gen_object_creation(o))
        lines += _indent(body)
        lines.append("    }")
        lines.append("")
        lines.append(f"    ~{self.cd.name}() {{")
        dtor: list[str] = []
        for o in reversed(self.ordered_objects()):
            dtor.extend(self._gen_object_destruction(o))
        lines += _indent(dtor)
        lines.append("    }")
        lines.append("")
        lines.append(f"    {self.cd.name}(const {self.cd.name}&) = delete;")
        lines.append(f"    {self.cd.name}& operator=(const {self.cd.name}&) = delete;")
        lines.append("")
        for o in self.ordered_objects():
            lines.append(f"    {o.binding.handle} {o.instance_name}{{}};")
        private_members = self._private_members()
        if private_members:
            lines.append("")
            lines.append("private:")
            lines.extend(f"    {decl}" for decl in private_members)
        lines.append("};")
        return "\n".join(lines)

    def _ctor_init_list(self) -> list[str]:
        inits: list[str] = []
        seen: set[str] = set()
        for o in self.ordered_objects():
            for handle_type in self._required_handle_types(o):
                pname = _lower_first(handle_type)
                if pname in seen or self._sibling_for_handle(handle_type) is not None:
                    continue
                ext = self._external_for_handle(handle_type)
                expr = pname if ext is None else f"{ext[0]}.{ext[1]}"
                inits.append(f"{pname}_({expr})")
                seen.add(pname)
        return inits

    def _private_members(self) -> list[str]:
        out: list[str] = []
        for handle_type in self._needed_saved_handles():
            out.append(f"{handle_type} {_lower_first(handle_type)}_{{}};")
        return out

    def _needed_saved_handles(self) -> list[str]:
        types: list[str] = []
        for o in self.ordered_objects():
            for handle_type in self._required_handle_types(o):
                if self._sibling_for_handle(handle_type) is not None:
                    continue
                if handle_type not in types:
                    types.append(handle_type)
        return types

    # object creation ------------------------------------------------------

    def _gen_object_creation(self, o: ObjectDef) -> list[str]:
        cmd = self.registry.resolve_command(o.binding.create_command)
        if cmd is None:
            raise GenerationError(f"unknown command {o.binding.create_command}")
        struct = self.registry.resolve_struct(o.binding.create_info_struct)
        if struct is None:
            raise GenerationError(f"unknown struct {o.binding.create_info_struct}")

        required_handles = set(self._required_handle_types(o))
        locals_: list[str] = []
        info_var = f"{o.instance_name}CreateInfo"
        self._emit_struct(locals_, struct, o.params, info_var, o)

        call_args: list[str] = []
        for p in cmd.params:
            if p.name == o.binding.create_info_param:
                call_args.append(f"&{info_var}")
            elif p.is_pointer and not p.is_const and p.type in self.registry.handles:
                call_args.append(f"&{o.instance_name}")
            elif p.type == "uint32_t" and p.len_value is not None and o.binding.plural:
                call_args.append("1")
            elif o.binding.plural and p.len_value is not None:
                call_args.append("1")
            elif p.type in self.registry.handles and not p.is_pointer:
                if p.type not in self.registry.handles or (
                    p.type not in required_handles and self._sibling_for_handle(p.type) is None and self._external_for_handle(p.type) is None
                ):
                    call_args.append("VK_NULL_HANDLE")
                else:
                    call_args.append(self._handle_arg(p))
            else:
                call_args.append("nullptr")

        lines = list(locals_)
        call = f"{o.binding.create_command}({', '.join(call_args)})"
        if cmd.return_type == "VkResult":
            lines.append(f"if ({call} != VK_SUCCESS) {{")
            lines.append(f'    throw std::runtime_error("{o.binding.create_command} failed");')
            lines.append("}")
        else:
            lines.append(f"{call};")
        return lines

    def _handle_arg(self, p) -> str:
        sibling = self._sibling_for_handle(p.type)
        if sibling is not None:
            return sibling.instance_name
        ext = self._external_for_handle(p.type)
        if ext is not None:
            return f"{ext[0]}.{ext[1]}"
        return f"{_lower_first(p.type)}_"

    # struct emission ------------------------------------------------------

    def _emit_struct(self, locals_: list[str], struct: Struct, fields: dict, var: str, o: ObjectDef) -> None:
        locals_.append(f"{struct.resolved_name} {var}{{}};")
        stype = next((m.value for m in struct.members if m.value), None)
        if stype:
            locals_.append(f"{var}.sType = {stype};")
        for key, value in fields.items():
            member = self._match_member(struct, key)
            if member is None:
                raise GenerationError(
                    f"class '{self.cd.name}', object '{o.instance_name}': '{key}' is not a "
                    f"member of {struct.resolved_name}"
                )
            if member.is_array_len_member:
                continue  # counts are derived automatically
            expr = self._emit_member_value(locals_, struct, member, value, o, var)
            if expr is not None:
                locals_.append(f"{var}.{member.name} = {expr};")

    def _emit_member_value(self, locals_: list[str], struct: Struct, member: Member, value, o: ObjectDef, var: str) -> Optional[str]:
        base = _member_var_name(member.name)

        if isinstance(value, Placeholder):
            return value.name
        if isinstance(value, Ref):
            return self._ref_expr(value, o, member)
        if member.is_pointer and isinstance(value, StructValue):
            sub = self.registry.resolve_struct(member.type)
            if sub is None:
                raise GenerationError(f"unknown struct {member.type}")
            subvar = f"{var}_{base}"
            self._emit_struct(locals_, sub, value.fields, subvar, o)
            return f"&{subvar}"
        if member.is_pointer and isinstance(value, ArrayValue):
            arrvar = f"{var}_{base}"
            sub = self.registry.resolve_struct(member.type) if member.type.startswith("Vk") else None
            if sub is not None:
                locals_.append(f"std::vector<{member.type}> {arrvar};")
                for i, item in enumerate(value.items):
                    ivar = f"{arrvar}_item{i}"
                    if isinstance(item, StructValue):
                        self._emit_struct(locals_, sub, item.fields, ivar, o)
                    else:
                        raise GenerationError(
                            f"class '{self.cd.name}': array items of '{member.name}' in "
                            f"{struct.resolved_name} must be mappings"
                        )
                    locals_.append(f"{arrvar}.push_back({ivar});")
            elif member.type == "char":
                locals_.append(f"std::vector<std::string> {arrvar}_storage;")
                locals_.append(f"std::vector<const char*> {arrvar};")
                for item in value.items:
                    if not isinstance(item, str):
                        raise GenerationError(
                            f"class '{self.cd.name}': string array '{member.name}' must contain strings"
                        )
                    locals_.append(f"{arrvar}_storage.emplace_back({self._string_literal(item)});")
                    locals_.append(f"{arrvar}.emplace_back({arrvar}_storage.back().c_str());")
            else:
                locals_.append(f"std::vector<{member.type}> {arrvar};")
                for item in value.items:
                    locals_.append(f"{arrvar}.emplace_back({self._scalar(item)});")
            count = struct.member(member.array_len) if member.array_len else None
            if count is not None:
                locals_.append(f"{var}.{count.name} = static_cast<{count.type}>({arrvar}.size());")
            return f"{arrvar}.data()"
        if isinstance(value, ArrayValue) and not member.is_pointer:
            ver = _version_expr(value.items if value.items else value)
            if ver:
                return ver
        if isinstance(value, list) and not member.is_pointer:
            ver = _version_expr(value)
            if ver:
                return ver
        if member.is_pointer and member.type == "char" and isinstance(value, str):
            svar = f"{var}_{base}_str"
            locals_.append(f"std::string {svar}({self._string_literal(value)});")
            return f"{svar}.c_str()"
        if "Version" in member.name:
            ver = _version_expr(value)
            if ver:
                return ver
        return self._scalar(value)

    def _count_literal(self, value) -> Optional[str]:
        ver = _version_expr(value)
        if ver:
            return ver
        return None

    def _scalar(self, value):
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return "VK_TRUE" if value else "VK_FALSE"
        if value is None:
            return "nullptr"
        if isinstance(value, float):
            return f"{value}f"
        return str(value)

    @staticmethod
    def _string_literal(s: str) -> str:
        escaped = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    def _ref_expr(self, ref: Ref, o: ObjectDef, member: Member) -> str:
        if ref.kind == "sibling":
            if ref.object_name not in self.objects:
                raise GenerationError(
                    f"class '{self.cd.name}', object '{o.instance_name}': reference to "
                    f"unknown sibling object '{ref.object_name}'"
                )
            return ref.object_name
        other = self.class_lookup.get(ref.class_name)
        if other is None:
            raise GenerationError(
                f"class '{self.cd.name}', object '{o.instance_name}': reference to unknown "
                f"class '{ref.class_name}'"
            )
        target = other.object(ref.object_name)
        if target is None:
            raise GenerationError(
                f"class '{self.cd.name}', object '{o.instance_name}': class '{ref.class_name}' "
                f"has no object '{ref.object_name}'"
            )
        if target.binding.handle != member.type:
            raise GenerationError(
                f"class '{self.cd.name}', object '{o.instance_name}': '{ref.class_name}."
                f"{ref.object_name}' is a {target.binding.handle} but member "
                f"'{member.name}' of {member.type if not member.is_pointer else member.type}' expects {member.type}"
            )
        return f"{_lower_first(ref.class_name)}.{ref.object_name}"

    # object destruction ---------------------------------------------------

    def _gen_object_destruction(self, o: ObjectDef) -> list[str]:
        if o.binding.destroy_command is None:
            return [f"// {o.instance_name}: no destroy command inferred"]
        cmd = self.registry.resolve_command(o.binding.destroy_command)
        if cmd is None:
            return []
        args: list[str] = []
        for p in cmd.params:
            if p.type == o.binding.handle and not p.is_pointer:
                args.append(o.instance_name)
            elif p.type in self.registry.handles and not p.is_pointer:
                sibling = self._sibling_for_handle(p.type)
                if sibling is not None and sibling.instance_name != o.instance_name:
                    args.append(sibling.instance_name)
                else:
                    args.append(f"{_lower_first(p.type)}_")
            elif p.is_const and p.is_pointer:
                args.append("nullptr")
        lines = [f"if ({o.instance_name} != VK_NULL_HANDLE) {{"]
        lines.append(f"    {o.binding.destroy_command}({', '.join(args)});")
        lines.append("}")
        return lines


def generate(spec_classes: list[ClassDef], registry: Registry) -> str:
    lookup = {c.name: c for c in spec_classes}
    ordered = _order_classes(spec_classes, lookup)
    out: list[str] = [
        "// Generated by crater. Do not edit.",
        "#pragma once",
        "",
        "#include <stdexcept>",
        "#include <string>",
        "#include <vector>",
        "#include <vulkan/vulkan.h>",
        "",
    ]
    for c in ordered:
        gen = ClassGenerator(c, registry, lookup)
        out.append(gen.generate())
        out.append("")
    return "\n".join(out)


def _order_classes(classes: list[ClassDef], lookup: dict[str, ClassDef]) -> list[ClassDef]:
    def deps(c: ClassDef) -> set[str]:
        out: set[str] = set()
        for o in c.objects:
            for ref in ClassGenerator(c, None, lookup)._refs(o):  # type: ignore[arg-type]
                if ref.kind == "external" and ref.class_name in lookup:
                    out.add(ref.class_name)
        return out

    result: list[ClassDef] = []
    remaining = list(classes)
    while remaining:
        progressed = False
        for c in list(remaining):
            if deps(c) <= {x.name for x in result}:
                result.append(c)
                remaining.remove(c)
                progressed = True
        if not progressed:
            names = [c.name for c in remaining]
            raise GenerationError(f"circular class references: {names}")
    return result
