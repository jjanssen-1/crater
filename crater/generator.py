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
from .spec import (
    ArrayValue,
    CapabilitiesDef,
    ClassDef,
    ObjectDef,
    Placeholder,
    QueryDef,
    Ref,
    StructValue,
)

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
    for prefix in ("pp", "p"):
        if member_name.startswith(prefix) and member_name[len(prefix) : len(prefix) + 1].isupper():
            return member_name[len(prefix)].lower() + member_name[len(prefix) + 1 :]
    return member_name


class CapabilitiesGenerator:
    """Generates a class that queries capabilities at runtime and selects
    values per the spec's rules.

    Selection semantics: the first candidate that the queried set contains
    wins; 'fallback: any' selects the first available item when no candidate
    matches. Required availability entries throw when missing.
    """

    def __init__(self, class_def: ClassDef, registry: Registry, class_lookup: dict[str, ClassDef]):
        self.cd = class_def
        self.registry = registry
        self.class_lookup = class_lookup
        assert class_def.capabilities is not None
        self.caps = class_def.capabilities

    # --------------------------------------------------------------- inputs

    def _input_param(self, name: str, ptype) -> tuple[str, str]:
        """(C++ type, C++ name) for a query input parameter."""
        if isinstance(ptype, Ref) and ptype.kind == "external":
            return f"const {ptype.class_name}&", _lower_first(ptype.class_name)
        if isinstance(ptype, Placeholder):
            if not ptype.explicit_type:
                raise GenerationError(
                    f"class '{self.cd.name}': placeholder for input '{name}' needs an "
                    f"explicit type"
                )
            return ptype.explicit_type, ptype.name
        # raw handle type from the registry
        return str(ptype), name

    def _input_expr(self, name: str, ptype, query_param_name: str) -> str:
        """Expression to pass for a query's command parameter."""
        if isinstance(ptype, Ref) and ptype.kind == "external":
            other = self.class_lookup[ptype.class_name]
            target = other.object(ptype.object_name)
            if target is None:
                raise GenerationError(
                    f"class '{self.cd.name}': class '{ptype.class_name}' has no object "
                    f"'{ptype.object_name}'"
                )
            return f"{_lower_first(ptype.class_name)}.{ptype.object_name}"
        if isinstance(ptype, Placeholder):
            return ptype.name
        return query_param_name

    def ctor_params(self) -> list[tuple[str, str]]:
        params: list[tuple[str, str]] = []
        seen: set[str] = set()
        for pname, ptype in self.caps.inputs:
            cpp_type, cpp_name = self._input_param(pname, ptype)
            if cpp_name in seen:
                continue
            params.append((cpp_type, cpp_name))
            seen.add(cpp_name)
        return params

    # ------------------------------------------------------------ generation

    def generate(self) -> str:
        params = self.ctor_params()
        param_list = ", ".join(f"{t} {n}" for t, n in params)
        lines: list[str] = [f"class {self.cd.name} {{", "public:"]
        lines.append(f"    explicit {self.cd.name}({param_list}) {{")
        body: list[str] = []
        for qd in self.caps.queries:
            body.append("")
            body.append(f"        // {qd.queried_type} -> {qd.as_name}")
            body.extend(_indent(self._gen_query(qd, params), 8))
        lines += body
        lines.append("    }")
        lines.append("")
        lines.append(f"    {self.cd.name}(const {self.cd.name}&) = delete;")
        lines.append(f"    {self.cd.name}& operator=(const {self.cd.name}&) = delete;")
        lines.append("")
        lines.append("    // query results")
        for qd in self.caps.queries:
            lines.extend(f"    {decl}" for decl in self._result_members(qd))
        lines.append("};")
        return "\n".join(lines)

    def _result_members(self, qd: QueryDef) -> list[str]:
        qb = qd.binding
        if qb.shape in ("array", "struct"):
            members = [f"{qd.queried_type} {qd.as_name}{{}};"]
            if qb.shape == "array" and qd.index:
                members.append(f"uint32_t {qd.as_name}Index{{}};")
            return members
        # present: availability flags per declared name
        return [f"bool {self._flag_name(name)}{{false}};" for name in qd.required + qd.optional]

    @staticmethod
    def _flag_name(extension_name: str) -> str:
        parts = extension_name.split("_")
        if parts[0] in ("VK", "VK_LAYER"):
            parts = parts[1:]
        body = parts[0] + "".join(p.capitalize() for p in parts[1:])
        return "has" + body[0].upper() + body[1:]

    def _gen_query(self, qd: QueryDef, params: list[tuple[str, str]]) -> list[str]:
        qb = qd.binding
        cmd = self.registry.resolve_command(qb.query_command)
        if cmd is None:
            raise GenerationError(f"unknown command {qb.query_command}")
        # map command parameter names to expressions
        param_exprs: dict[str, str] = {}
        for p in cmd.params:
            if p.name == qb.output_param or p.name == qb.count_param:
                continue
            matching_input = next(
                (i for i in qb.inputs if i.name == p.name), None
            )
            if matching_input is None:
                param_exprs[p.name] = "nullptr"
            else:
                pname, ptype = next(
                    (pn, pt) for pn, pt in self.caps.inputs if pn == matching_input.name
                )
                param_exprs[p.name] = self._input_expr(pname, ptype, matching_input.name)

        lines: list[str] = []
        if qb.shape == "array":
            vec = f"{qd.as_name}Options"
            count_var = f"{qd.as_name}Count"
            lines.append(f"uint32_t {count_var} = 0;")
            call1 = f"{qb.query_command}({', '.join(self._args(cmd, param_exprs, count=f'&{count_var}', out='nullptr'))})"
            lines.append(self._vk_call(call1, cmd, f"{qb.query_command} (count) failed"))
            lines.append(f"std::vector<{qd.queried_type}> {vec}({count_var});")
            call2 = f"{qb.query_command}({', '.join(self._args(cmd, param_exprs, count=f'&{count_var}', out=f'{vec}.data()'))})"
            lines.append(self._vk_call(call2, cmd, f"{qb.query_command} failed"))
            lines.append("")
            if qd.require or qd.prefer:
                lines.extend(self._gen_predicate_selection(qd, vec))
            elif qd.candidates:
                lines.extend(self._gen_selection(qd, vec))
            else:
                lines.append(f"if ({vec}.empty()) {{")
                lines.append(f'    throw std::runtime_error("{qd.queried_type}: no options available");')
                lines.append("}")
                lines.append(f"{qd.as_name} = {vec}[0];")
                if qd.index:
                    lines.append(f"{qd.as_name}Index = 0;")
            return lines
        if qb.shape == "struct":
            out_addr = f"&{qd.as_name}"
            call = f"{qb.query_command}({', '.join(self._args(cmd, param_exprs, out=out_addr))})"
            lines.append(self._vk_call(call, cmd, f"{qb.query_command} failed"))
            for path in qd.required:
                lines.append(f"if ({qd.as_name}.{path} != VK_TRUE) {{")
                error = f"{qb.query_command}: required capability {path} is not supported"
                lines.append(f'    throw std::runtime_error("{error}");')
                lines.append("}")
            return lines
        # present
        return lines + self._gen_present(qd, cmd, param_exprs)

    def _gen_present(self, qd: QueryDef, cmd, param_exprs: dict[str, str]) -> list[str]:
        qb = qd.binding
        vec = f"{qd.as_name}Available"
        count_var = f"{qd.as_name}Count"
        lines = [
            f"uint32_t {count_var} = 0;",
        ]
        call1 = f"{qb.query_command}({', '.join(self._args(cmd, param_exprs, count=f'&{count_var}', out='nullptr'))})"
        lines.append(self._vk_call(call1, cmd, f"{qb.query_command} (count) failed"))
        lines.append(f"std::vector<{qb.enumerated_type}> {vec}({count_var});")
        call2 = f"{qb.query_command}({', '.join(self._args(cmd, param_exprs, count=f'&{count_var}', out=f'{vec}.data()'))})"
        lines.append(self._vk_call(call2, cmd, f"{qb.query_command} failed"))
        lines.append("")
        for name in qd.required:
            flag = self._flag_name(name)
            lines.append(f"if (std::find_if({vec}.begin(), {vec}.end(), [&](const {qb.enumerated_type}& e) {{ return std::string_view(e.{qb.name_member}) == {self._string_literal(name)}; }}) == {vec}.end()) {{")
            lines.append(f'    throw std::runtime_error("required {name} is not available");')
            lines.append("}")
            lines.append(f"{flag} = true;")
        for name in qd.optional:
            flag = self._flag_name(name)
            lines.append(f"{flag} = std::any_of({vec}.begin(), {vec}.end(), [&](const {qb.enumerated_type}& e) {{ return std::string_view(e.{qb.name_member}) == {self._string_literal(name)}; }});")
        return lines

    def _gen_predicate_selection(self, qd: QueryDef, vec: str) -> list[str]:
        """Selection driven by require/prefer member predicates.

        require: an element qualifies when every listed member has all the
        required bits set. prefer: among qualifying elements, the one with
        the most preferred bits wins (first such element on ties). Single
        pass, no allocation.
        """
        it = f"const {qd.queried_type}&"
        best = f"{qd.as_name}Best"
        best_rank = f"{qd.as_name}BestRank"
        best_index = f"{qd.as_name}BestIndex"
        rank = f"{qd.as_name}Rank"

        def bits_expr(member: str, bits: list[str]) -> str:
            mask = " | ".join(bits)
            return f"((e.{member} & ({mask})) == ({mask}))"

        require_conds = [bits_expr(m, b) for m, b in qd.require.items()]

        lines = [f"uint32_t {best_rank} = 0; // 0 = no match yet"]
        lines.append(f"uint32_t {best_index} = 0;")
        lines.append(f"{qd.queried_type} {best};")
        lines.append(f"for (uint32_t i = 0; i < {qd.as_name}Count; ++i) {{")
        lines.append(f"    {it} e = {vec}[i];")
        if require_conds:
            lines.append(f"    if (!({' && '.join(require_conds)})) {{ continue; }}")
        lines.append(f"    uint32_t {rank} = 1;")
        for member, bits in qd.prefer.items():
            for b in bits:
                lines.append(f"    if ({bits_expr(member, [b])}) {{ ++{rank}; }}")
        lines.append(f"    if ({rank} > {best_rank}) {{")
        lines.append(f"        {best} = e;")
        lines.append(f"        {best_rank} = {rank};")
        lines.append(f"        {best_index} = i;")
        lines.append("    }")
        lines.append("}")
        lines.append(f"if ({best_rank} == 0) {{")
        lines.append(
            f'    throw std::runtime_error("no {qd.queried_type} satisfying the required capabilities");'
        )
        lines.append("}")
        lines.append(f"{qd.as_name} = {best};")
        if qd.index:
            lines.append(f"{qd.as_name}Index = {best_index};")
        return lines

    def _gen_selection(self, qd: QueryDef, vec: str) -> list[str]:
        """First-match selection over candidates."""
        qb = qd.binding
        s = self.registry.resolve_struct(qd.queried_type)
        lines: list[str] = []
        matched = f"{qd.as_name}Matched"
        lines.append(f"bool {matched} = false;")
        if qb.item_type == qd.queried_type and s is not None and len(qd.candidates[0].items if isinstance(qd.candidates[0], ArrayValue) else []) > 1:
            # struct candidates given as rows of member values
            for cand in qd.candidates:
                if not isinstance(cand, ArrayValue):
                    raise GenerationError(
                        f"class '{self.cd.name}': {qd.queried_type} candidates must be lists"
                    )
                conds = []
                for i, item in enumerate(cand.items):
                    if s is None or i >= len(s.members):
                        break
                    member = s.members[i + 1] if s.members and s.members[0].type == "VkStructureType" else s.members[i]
                    conds.append(f"e.{member.name} == {item}")
                cond = " && ".join(conds)
                lines.append(f"if (!{matched}) {{")
                lines.append(f"    for (const {qd.queried_type}& e : {vec}) {{")
                lines.append(f"        if ({cond}) {{")
                lines.append(f"            {qd.as_name} = e;")
                lines.append(f"            {matched} = true;")
                lines.append("        }")
                lines.append("    }")
                lines.append("}")
        else:
            for cand in qd.candidates:
                value = cand.items[0] if isinstance(cand, ArrayValue) else cand
                lines.append(f"if (!{matched} && std::find({vec}.begin(), {vec}.end(), {value}) != {vec}.end()) {{")
                lines.append(f"    {qd.as_name} = {value};")
                lines.append(f"    {matched} = true;")
                lines.append("}")
        if qd.fallback_any:
            lines.append(f"if (!{matched} && !{vec}.empty()) {{")
            lines.append(f"    {qd.as_name} = {vec}[0];")
            lines.append("}")
        else:
            lines.append(f"if (!{matched}) {{")
            lines.append(f'    throw std::runtime_error("no acceptable {qd.queried_type} found");')
            lines.append("}")
        return lines

    def _args(self, cmd, param_exprs: dict[str, str], count: str | None = None, out: str | None = None) -> list[str]:
        qb_out = None
        args: list[str] = []
        for p in cmd.params:
            if count is not None and p.type == "uint32_t" and p.is_pointer:
                args.append(count)
            elif out is not None and p.is_pointer and not p.is_const and p.type != "uint32_t":
                args.append(out)
            elif p.name in param_exprs:
                args.append(param_exprs[p.name])
            else:
                args.append("nullptr")
        return args

    @staticmethod
    def _vk_call(call: str, cmd, error: str) -> str:
        if cmd.return_type == "VkResult":
            return f"if ({call} != VK_SUCCESS) {{ throw std::runtime_error(\"{error}\"); }}"
        return f"{call};"

    @staticmethod
    def _string_literal(s: str) -> str:
        return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'


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
        if member.is_pointer:
            return f"{'const ' if member.is_const else ''}{member.type}*"
        return member.type

    # ------------------------------------------------------------ generation

    def generate(self) -> str:
        ctor = self.ctor_params()
        param_list = ", ".join(f"{t} {n}" for t, n in ctor)
        inits = self._ctor_init_list()

        lines: list[str] = [f"class {self.cd.name} {{", "public:"]
        lines.append(f"    explicit {self.cd.name}({param_list}){(' : ' + ', '.join(inits)) if inits else ''} {{")
        for o in self.ordered_objects():
            lines.append("")
            lines.append(f"        // {o.vulkan_name} '{o.instance_name}'")
            lines.extend(_indent(self._gen_object_creation(o), 8))
        lines.append("    }")
        lines.append("")
        lines.append(f"    ~{self.cd.name}() {{")
        for o in reversed(self.ordered_objects()):
            lines.append("")
            lines.append(f"        // {o.vulkan_name} '{o.instance_name}'")
            lines.extend(_indent(self._gen_object_destruction(o), 8))
        lines.append("    }")
        lines.append("")
        lines.append(f"    {self.cd.name}(const {self.cd.name}&) = delete;")
        lines.append(f"    {self.cd.name}& operator=(const {self.cd.name}&) = delete;")
        lines.append("")
        lines.append("    // Vulkan handles created by this class")
        for o in self.ordered_objects():
            lines.append(f"    {o.binding.handle} {o.instance_name}{{}};")
        private_members = self._private_members()
        if private_members:
            lines.append("")
            lines.append("private:")
            for decl in private_members:
                lines.append(f"    // saved for destruction")
                lines.append(f"    {decl}")
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
            if member.is_array_len_member and not isinstance(value, Placeholder):
                continue  # counts are derived automatically unless given
            expr = self._emit_member_value(locals_, struct, member, value, o, var)
            if expr is not None:
                locals_.append(f"{var}.{member.name} = {expr};")

    def _emit_member_value(self, locals_: list[str], struct: Struct, member: Member, value, o: ObjectDef, var: str) -> Optional[str]:
        base = _member_var_name(member.name)

        if isinstance(value, Placeholder):
            return value.name
        if isinstance(value, Ref):
            expr = self._ref_expr(value, o, member)
            if member.is_pointer and member.type in self.registry.structs:
                return f"&{expr}"
            return expr
        if isinstance(value, StructValue) and (
            member.is_pointer or (member.type.startswith("Vk") and self.registry.resolve_struct(member.type) is not None)
        ):
            sub = self.registry.resolve_struct(member.type)
            if sub is None:
                raise GenerationError(f"unknown struct {member.type}")
            subvar = f"{var}_{base}"
            self._emit_struct(locals_, sub, value.fields, subvar, o)
            return f"&{subvar}" if member.is_pointer else subvar
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
        path = ref.object_name.split(".")
        target = other.object(path[0])
        if target is None:
            if other.capabilities is None or not self._is_capability_member(other, path):
                raise GenerationError(
                    f"class '{self.cd.name}', object '{o.instance_name}': class "
                    f"'{ref.class_name}' has no object '{path[0]}'"
                )
        elif target.binding.handle != member.type and len(path) == 1:
            raise GenerationError(
                f"class '{self.cd.name}', object '{o.instance_name}': '{ref.class_name}."
                f"{path[0]}' is a {target.binding.handle} but member "
                f"'{member.name}' expects {member.type}"
            )
        return f"{_lower_first(ref.class_name)}.{ref.object_name}"

    def _is_capability_member(self, other: ClassDef, path: list[str]) -> bool:
        """Check that a member path resolves inside a capabilities class."""
        assert other.capabilities is not None
        head = path[0]
        for qd in other.capabilities.queries:
            if qd.as_name == head:
                return True
            if qd.index and f"{qd.as_name}Index" == head:
                return True
            for name in qd.required + qd.optional:
                if self._caps_flag_name(name) == head:
                    return True
        return False

    @staticmethod
    def _caps_flag_name(extension_name: str) -> str:
        parts = extension_name.split("_")
        if parts[0] in ("VK", "VK_LAYER"):
            parts = parts[1:]
        body = parts[0] + "".join(p.capitalize() for p in parts[1:])
        return "has" + body[0].upper() + body[1:]

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
        "#include <algorithm>",
        "#include <stdexcept>",
        "#include <string>",
        "#include <string_view>",
        "#include <vector>",
        "#include <vulkan/vulkan.h>",
    ]
    for c in ordered:
        out.append("")
        if c.capabilities is not None:
            out.append(CapabilitiesGenerator(c, registry, lookup).generate())
        else:
            gen = ClassGenerator(c, registry, lookup)
            out.append(gen.generate())
    return "\n".join(out)


def _order_classes(classes: list[ClassDef], lookup: dict[str, ClassDef]) -> list[ClassDef]:
    def deps(c: ClassDef) -> set[str]:
        out: set[str] = set()
        if c.capabilities is not None:
            for pname, ptype in c.capabilities.inputs:
                if isinstance(ptype, Ref) and ptype.kind == "external":
                    out.add(ptype.class_name)
            return out
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
