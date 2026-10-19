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
    CallDef,
    CapabilitiesDef,
    ClassDef,
    ObjectDef,
    PNextChain,
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


_EXPR_RE = re.compile(r"^[A-Za-z_][\w.]*()$|^.*\(\)$")


def _looks_like_expression(value: str) -> bool:
    """Heuristic: contains a call suffix like .size() / .data() ."""
    return "()" in value


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


def emit_pnext_chain(registry, class_name, locals_, chain, target, o, var, emit_struct, members=None) -> str:
    """Emit a pNext chain of extension structs, registry-validated.

    Each entry struct must appear in the target's structextends list
    from vk.xml; the entries are emitted with their sType set, their
    members filled like any struct, and the chain wired via pNext
    pointers. Returns the expression for the head pointer.

    When `members` is given (a list), the chain entries are declared as
    members of that list instead of locals - for capability queries,
    where the queried extension values must be readable after the
    constructor ran.
    """
    head = "nullptr"
    prev = None
    for i, (name, sv) in enumerate(chain.entries):
        ext = registry.resolve_struct(name)
        if ext is None:
            raise GenerationError(
                f"class '{class_name}': unknown pNext struct '{name}'"
            )
        if target.resolved_name not in ext.extends:
            raise GenerationError(
                f"class '{class_name}': '{name}' does not extend "
                f"{target.resolved_name} (per the registry's structextends)"
            )
        ext_var = f"{var}_pNext{i}"
        if members is not None:
            members.append(f"{ext.resolved_name} {ext_var}{{}};")
            body = []
            emit_struct(body, ext, sv.fields, ext_var, o)
            # drop the local re-declaration; keep the assignments
            decl = f"{ext.resolved_name} {ext_var}{{}};"
            body = [l for l in body if l != decl]
            locals_.extend(body)
        else:
            emit_struct(locals_, ext, sv.fields, ext_var, o)
        if prev is None:
            head = f"&{ext_var}"
        else:
            locals_.append(f"{prev}.pNext = &{ext_var};")
        prev = ext_var
    return head


class CallGenerator:
    """Generates a class of plain command invocations (call functions).

    Each call becomes a member function that builds the command's info
    struct from the YAML values, takes the remaining inputs as function
    parameters, invokes the command and checks its result against the
    registry's success codes. Output parameters (non-const pointers)
    become reference parameters.
    """

    def __init__(self, class_def: ClassDef, registry: Registry, class_lookup: dict[str, ClassDef]):
        self.cd = class_def
        self.registry = registry
        self.class_lookup = class_lookup

    def _info_param(self, cmd):
        for p in cmd.params:
            if p.is_const and p.is_pointer and p.type in self.registry.structs:
                return p
        return None

    def _member_expr(self, value, member):
        if isinstance(value, bool):
            return "VK_TRUE" if value else "VK_FALSE"
        if isinstance(value, float):
            return f"{value}f"
        if isinstance(value, str):
            if member.is_pointer and member.type == "char":
                # caller-supplied string literal; keep lifetime in the struct's scope
                return f"std::string({self._string_literal(value)}).c_str()" if False else value
            return value
        return str(value)

    @staticmethod
    def _string_literal(s: str) -> str:
        return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'

    def generate(self) -> str:
        lines: list[str] = [f"// call functions ({self.cd.name})"]
        for call in self.cd.calls:
            lines.append("")
            lines.extend(self._gen_call(call))
        return chr(10).join(lines)

    def _gen_call(self, call: CallDef) -> list[str]:
        cmd = self.registry.resolve_command(call.command_name)
        if cmd is None:
            raise GenerationError(f"unknown command {call.command_name}")
        info_param = self._info_param(cmd)
        info_struct = None
        if info_param is not None:
            info_struct = self.registry.resolve_struct(info_param.type)
            if info_struct is None:
                raise GenerationError(f"unknown struct {info_param.type}")

        info_var = f"{call.as_name}Info"
        fparams: list[tuple[str, str]] = []  # (type, name) of the generated function
        seen: set[str] = set()
        args: list[str] = []
        locals_: list[str] = []
        cmd_param_names = {p.name for p in cmd.params} | {_member_var_name(p.name) for p in cmd.params}

        def add_param(ptype: str, pname: str) -> None:
            if pname not in seen:
                fparams.append((ptype, pname))
                seen.add(pname)

        for p in cmd.params:
            if p is info_param:
                continue
            if p.is_pointer and not p.is_const:
                # output parameter: reference parameter, passed by address
                add_param(f"{p.type}&", p.name)
                args.append(f"&{p.name}")
                continue
            # value declared in the YAML? (by exact or stripped name)
            value = call.params.get(p.name)
            if value is None:
                value = call.params.get(_member_var_name(p.name))
            if value is not None:
                if isinstance(value, Placeholder):
                    add_param(self._param_type(p, value), value.name)
                    args.append(value.name)
                else:
                    args.append(self._member_expr(value, p))
                continue
            # undeclared: function parameter
            add_param(self._param_type(p, None), p.name)
            args.append(p.name)

        # build the info struct from the remaining YAML keys
        if info_param is not None:
            struct_fields = {
                k: v for k, v in call.params.items() if k not in cmd_param_names
            }
            self._emit_struct_for_call(locals_, info_struct, struct_fields, call, info_var, add_param, seen)
            args.insert(cmd.params.index(info_param), f"&{info_var}")

        param_list = ", ".join(f"{t} {n}" for t, n in fparams)
        ret = cmd.return_type if cmd.return_type != "void" else "void"
        lines = [f"inline {ret} {call.as_name}({param_list}) {{"]
        lines.extend(f"    {l}" for l in locals_)

        call_expr = f"{call.command_name}({', '.join(args)})"
        if cmd.return_type == "void":
            lines.append(f"    {call_expr};")
        else:
            success = cmd.successcodes or ["VK_SUCCESS"]
            checks = " || ".join(f"result__ == {s}" for s in success)
            error = f"{call.command_name} failed ("
            lines.append(f"    VkResult result__ = {call_expr};")
            lines.append(f"    if (!({checks})) {{")
            lines.append(
                f'        throw crater::VulkanError(std::string("{error}") + crater::result_name(result__) + ")", result__);'
            )
            lines.append("    }")
            lines.append("    return result__;")
        lines.append("}")
        return lines

    def _param_type(self, p, placeholder):
        if placeholder is not None and placeholder.explicit_type:
            return placeholder.explicit_type
        if p.is_pointer:
            return f"{'const ' if p.is_const else ''}{p.type}*"
        return p.type

    def _emit_struct_for_call(self, locals_, struct, fields, call, var, add_param, seen) -> None:
        locals_.append(f"{struct.resolved_name} {var}{{}};")
        stype = next((m.value for m in struct.members if m.value), None)
        if stype:
            locals_.append(f"{var}.sType = {stype};")
        for key, value in fields.items():
            member = next(
                (m for m in struct.members if m.name == key or _member_var_name(m.name) == key),
                None,
            )
            if member is None:
                raise GenerationError(
                    f"class '{self.cd.name}', call '{call.command_name}': '{key}' is not a "
                    f"member of {struct.resolved_name}"
                )
            if isinstance(value, Placeholder):
                add_param(value.explicit_type or member.type, value.name)
                locals_.append(f"{var}.{member.name} = {value.name};")
            elif isinstance(value, (list, dict)):
                raise GenerationError(
                    f"class '{self.cd.name}', call '{call.command_name}': arrays/nested structs "
                    f"in call info structs are not supported yet ('{key}')"
                )
            else:
                locals_.append(f"{var}.{member.name} = {self._member_expr(value, member)};")


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
        inits = [f"{n}({n})" for _, n in params]
        self._pnext_members: list[str] = []
        lines: list[str] = [f"class {self.cd.name} {{", "public:"]
        lines.append(f"    explicit {self.cd.name}({param_list}){(' : ' + ', '.join(inits)) if inits else ''} {{")
        body: list[str] = []
        i = 0
        queries = self.caps.queries
        while i < len(queries):
            # group consecutive queries against the same command: one
            # two-call fetch, each selection evaluated on the shared results
            j = i
            while j < len(queries) and queries[j].command_name == queries[i].command_name:
                j += 1
            group = queries[i:j]
            if len(group) == 1:
                body.append("")
                body.append(f"        // {group[0].queried_type} -> {group[0].as_name}")
                body.extend(_indent(self._gen_query(group[0], params), 8))
            else:
                body.append("")
                body.append(f"        // {group[0].command_name}: {len(group)} selections")
                body.extend(_indent(self._gen_query_group(group, params), 8))
            i = j
        lines += body
        lines.append("}")
        lines.append("")
        lines.append(f"    {self.cd.name}(const {self.cd.name}&) = delete;")
        lines.append(f"    {self.cd.name}& operator=(const {self.cd.name}&) = delete;")
        lines.append("")
        lines.append("    // constructor inputs, stored for consumers")
        for pname, ptype in self.caps.inputs:
            cpp_type, cpp_name = self._input_param(pname, ptype)
            if cpp_type.startswith("const "):
                continue  # class references are not stored by value
            lines.append(f"    {cpp_type} {cpp_name}{{}};")
        lines.append("")
        lines.append("    // query results")
        for qd in self.caps.queries:
            lines.extend(f"    {decl}" for decl in self._result_members(qd))
        if self._pnext_members:
            lines.append("")
            lines.append("    // pNext chain entries (queried extension values)")
            lines.extend(f"    {decl}" for decl in self._pnext_members)
        lines.append("};")
        return "\n".join(lines)

    def _result_members(self, qd: QueryDef) -> list[str]:
        qb = qd.binding
        if qb.shape in ("array", "struct"):
            members = [f"{qd.queried_type} {qd.as_name}{{}};"]
            if qb.shape == "array" and qd.index:
                members.append(f"uint32_t {qd.as_name}Index{{}};")
            return members
        # present: availability flags plus the list of available declared
        # names, ready to be passed to count/char** members
        members = [f"std::vector<const char*> {qd.as_name};"]
        members.extend(
            f"bool {self._flag_name(name)}{{false}};" for name in qd.required + qd.optional
        )
        return members

    @staticmethod
    def _flag_name(extension_name: str) -> str:
        parts = extension_name.split("_")
        if parts[0] == "VK":
            parts = parts[1:]
        if parts and parts[0] == "LAYER":
            parts = parts[1:]
        body = parts[0] + "".join(p.capitalize() for p in parts[1:])
        return "has" + body[0].upper() + body[1:]

    def _emit_struct_for_query(self, locals_: list[str], struct: Struct, fields: dict, var: str, o) -> None:
        """Emit a struct with the given fields (sType from the registry)."""
        locals_.append(f"{struct.resolved_name} {var}{{}};")
        stype = next((m.value for m in struct.members if m.value), None)
        if stype:
            locals_.append(f"{var}.sType = {stype};")
        for key, value in fields.items():
            member = next(
                (m for m in struct.members if m.name == key or _member_var_name(m.name) == key),
                None,
            )
            if member is None:
                raise GenerationError(
                    f"class '{self.cd.name}': '{key}' is not a member of "
                    f"{struct.resolved_name}"
                )
            if isinstance(value, PNextChain):
                head = emit_pnext_chain(self.registry, self.cd.name, locals_, value, struct, o, var, self._emit_struct_for_query)
                locals_.append(f"{var}.{member.name} = {head};")
                continue
            if isinstance(value, bool):
                expr = "VK_TRUE" if value else "VK_FALSE"
            elif isinstance(value, float):
                expr = f"{value}f"
            elif isinstance(value, str):
                expr = value
            else:
                expr = str(value)
            if member.is_pointer and member.type != "void" and not member.is_const:
                locals_.append(f"{var}.{member.name} = &{expr};")
            else:
                locals_.append(f"{var}.{member.name} = {expr};")

    def _gen_query_group(self, group: list[QueryDef], params: list[tuple[str, str]]) -> list[str]:
        """One two-call fetch for a shared command, then each selection's
        require/prefer/candidate logic evaluated against the shared
        results vector."""
        first = group[0]
        qb = first.binding
        cmd = self.registry.resolve_command(qb.query_command)
        if cmd is None:
            raise GenerationError(f"unknown command {qb.query_command}")
        if qb.shape != "array":
            raise GenerationError(
                f"class '{self.cd.name}': multiple selections need an array "
                f"query, not '{qb.query_command}'"
            )
        # map command parameter names to expressions (shared inputs)
        param_exprs: dict[str, str] = {}
        for p in cmd.params:
            if p.name == qb.output_param or p.name == qb.count_param:
                continue
            matching_input = next((i for i in qb.inputs if i.name == p.name), None)
            if matching_input is None:
                param_exprs[p.name] = "nullptr"
            else:
                pname, ptype = next(
                    (pn, pt) for pn, pt in self.caps.inputs if pn == matching_input.name
                )
                param_exprs[p.name] = self._input_expr(pname, ptype, matching_input.name)

        vec = f"{qb.query_command}_options"
        count_var = f"{qb.query_command}_count"
        lines: list[str] = []
        lines.append(f"uint32_t {count_var} = 0;")
        call1 = f"{qb.query_command}({', '.join(self._args(cmd, param_exprs, count=f'&{count_var}', out='nullptr'))})"
        lines.append(self._vk_call(call1, cmd, f"{qb.query_command} (count) failed"))
        lines.append(f"std::vector<{first.queried_type}> {vec}({count_var});")
        call2 = f"{qb.query_command}({', '.join(self._args(cmd, param_exprs, count=f'&{count_var}', out=f'{vec}.data()'))})"
        lines.append(self._vk_call(call2, cmd, f"{qb.query_command} failed"))

        # each selection evaluates on the shared vector; per-selection
        # locals are prefixed to avoid collisions
        for qd in group:
            lines.append("")
            lines.append(f"// selection: {qd.as_name}")
            self._current_count_var = count_var
            try:
                sel_lines = self._gen_selection_logic(qd, vec, count_var, prefix=qd.as_name)
            finally:
                self._current_count_var = None
            lines.extend(sel_lines)
        return lines

    def _gen_selection_logic(self, qd: QueryDef, vec: str, count_var: str, prefix: str) -> list[str]:
        """The require/prefer/candidate selection for one QueryDef against
        an existing shared results vector (locals prefixed per selection)."""
        lines: list[str] = []
        if qd.require or qd.prefer:
            body = self._gen_predicate_selection(qd, vec)
            # prefix the locals to keep selections independent
            body = [self._prefix_local(l, qd, vec, count_var) for l in body]
            lines.extend(body)
            return lines
        if qd.candidates:
            body = self._gen_selection(qd, vec)
            body = [self._prefix_local(l, qd, vec, count_var) for l in body]
            lines.extend(body)
            return lines
        # first-element default selection
        lines.append(f"if ({vec}.empty()) {{")
        lines.append(f'    throw crater::VulkanError("{qd.queried_type}: no options available");')
        lines.append("}")
        lines.append(f"{qd.as_name} = {vec}[0];")
        if qd.index:
            lines.append(f"{qd.as_name}Index = 0;")
        return lines

    @staticmethod
    def _prefix_local(line: str, qd: QueryDef, vec: str, count_var: str) -> str:
        """Prefix selection-local variable names with the selection's
        as-name so parallel selections do not collide."""
        for name in (f"{qd.as_name}Matched", f"{qd.as_name}BestRank", f"{qd.as_name}BestIndex",
                     f"{qd.as_name}Best", f"{qd.as_name}Rank", f"{qd.as_name}Options",
                     f"{qd.as_name}Count"):
            if name in line and qd.as_name not in name:
                pass
        # selection helpers already use as_name prefixes; only the shared
        # vec/count names must stay untouched. Predicate selection names
        # its locals from as_name already, so nothing to do here.
        return line

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
                lines.append(f'    throw crater::VulkanError("{qd.queried_type}: no options available");')
                lines.append("}")
                lines.append(f"{qd.as_name} = {vec}[0];")
                if qd.index:
                    lines.append(f"{qd.as_name}Index = 0;")
            return lines
        if qb.shape == "struct":
            out_struct = self.registry.resolve_struct(qb.output_struct)
            stype = next((m.value for m in out_struct.members if m.value), None) if out_struct else None
            if stype:
                lines.append(f"{qd.as_name}.sType = {stype};")
            if qd.pnext is not None and out_struct is not None:
                fake_obj = ObjectDef(
                    vulkan_name=qd.command_name,
                    instance_name=qd.as_name,
                    binding=None,
                    params={},
                )
                head = emit_pnext_chain(self.registry, self.cd.name, lines, qd.pnext, out_struct, fake_obj, qd.as_name, self._emit_struct_for_query, members=self._pnext_members)
                lines.append(f"{qd.as_name}.pNext = {head};")
            out_addr = f"&{qd.as_name}"
            call = f"{qb.query_command}({', '.join(self._args(cmd, param_exprs, out=out_addr))})"
            lines.append(self._vk_call(call, cmd, f"{qb.query_command} failed"))
            for path in qd.required:
                lines.append(f"if ({qd.as_name}.{path} != VK_TRUE) {{")
                error = f"{qb.query_command}: required capability {path} is not supported"
                lines.append(f'    throw crater::VulkanError("{error}");')
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
            lines.append(f'    throw crater::VulkanError("required {name} is not available");')
            lines.append("}")
            lines.append(f"{flag} = true;")
            lines.append(f"{qd.as_name}.push_back({self._string_literal(name)});")
        for name in qd.optional:
            flag = self._flag_name(name)
            lines.append(f"{flag} = std::any_of({vec}.begin(), {vec}.end(), [&](const {qb.enumerated_type}& e) {{ return std::string_view(e.{qb.name_member}) == {self._string_literal(name)}; }});")
            lines.append(f"if ({flag}) {{")
            lines.append(f"    {qd.as_name}.push_back({self._string_literal(name)});")
            lines.append("}")
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
        count_ref = getattr(self, "_current_count_var", None) or f"{qd.as_name}Count"
        lines.append(f"for (uint32_t i = 0; i < {count_ref}; ++i) {{")
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
        lines.append("}")
        lines.append("}")
        lines.append(f"if ({best_rank} == 0) {{")
        lines.append(
            f'    throw crater::VulkanError("no {qd.queried_type} satisfying the required capabilities");'
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
                if qd.index:
                    lines.append(f"            {qd.as_name}Index = static_cast<uint32_t>(&e - {vec}.data());")
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
            lines.append(f'    throw crater::VulkanError("no acceptable {qd.queried_type} found");')
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
            return (
                f"{{ VkResult result__ = {call}; "
                f'if (result__ != VK_SUCCESS) {{ throw crater::VulkanError(std::string("{error} (") + crater::result_name(result__) + ")", result__); }} }}'
            )
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

    def _parent_param_names(self, o: ObjectDef) -> set[str]:
        """Spec keys that declare parent handle parameters (command params,
        not create-info members): matched by exact or stripped name."""
        names: set[str] = set()
        cmd = self.registry.resolve_command(o.binding.create_command)
        if cmd is not None:
            for p in cmd.params:
                if p.name in o.binding.parent_handles and p.type in self.registry.handles:
                    names.add(p.name)
                    names.add(_member_var_name(p.name))
        if o.binding.destroy_command:
            dcmd = self.registry.resolve_command(o.binding.destroy_command)
            if dcmd is not None:
                for p in dcmd.params:
                    if p.name in o.binding.destroy_parent_handles and p.type in self.registry.handles:
                        names.add(p.name)
                        names.add(_member_var_name(p.name))
        return names

    def _walk_spec(self, o: ObjectDef):
        """Yield (struct, member, value, base_name) for every spec value of
        the object, descending into nested structs and arrays. Parent handle
        declarations are command parameters, not create-info members: they
        are consumed by the create/destroy call generation, not the struct."""
        struct = self.registry.resolve_struct(o.binding.create_info_struct)
        skip = self._parent_param_names(o) | {'allocation'}
        params = {k: v for k, v in o.params.items() if k not in skip}
        yield from self._walk_struct(struct, params, o, o.instance_name)

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

    def _explicit_parent_param(self, o: ObjectDef, pname: str, ptype: str) -> Any:
        """The spec-declared value for a create/destroy parent handle
        parameter, looked up by exact or stripped parameter name. Same
        sibling/cross-class/placeholder forms as create-info members."""
        v = o.params.get(pname)
        if v is None:
            v = o.params.get(_member_var_name(pname))
        if v is None:
            return None
        if isinstance(v, Ref):
            if v.kind == "sibling" and v.object_name not in self.objects:
                raise GenerationError(
                    f"class '{self.cd.name}', object '{o.instance_name}': parent handle "
                    f"'{pname}' references unknown sibling object '{v.object_name}'"
                )
            if v.kind == "external" and v.class_name not in self.class_lookup:
                raise GenerationError(
                    f"class '{self.cd.name}', object '{o.instance_name}': parent handle "
                    f"'{pname}' references unknown class '{v.class_name}'"
                )
        if isinstance(v, str):
            m = re.match(r"\{\{\s*(.*?)\s*\}\}", v)
            if m:
                v = Placeholder(name=m.group(1))
        return v

    def _require_explicit_parents(self, o: ObjectDef) -> None:
        """Explicitness contract: every parent handle of a create or destroy
        command (device, instance, physicalDevice, allocator, ...) must be
        declared in the spec. No implicit inference from siblings."""
        missing: list[str] = []
        cmd = self.registry.resolve_command(o.binding.create_command)
        if cmd is not None:
            for p in cmd.params:
                if p.name in o.binding.parent_handles and p.type in self.registry.handles and not p.optional:
                    if self._explicit_parent_param(o, p.name, p.type) is None:
                        missing.append(p.name)
        if o.binding.destroy_command:
            dcmd = self.registry.resolve_command(o.binding.destroy_command)
            if dcmd is not None:
                for p in dcmd.params:
                    if (
                        p.name in o.binding.destroy_parent_handles
                        and p.type in self.registry.handles
                        and p.type != o.binding.handle
                        and p.type != "VmaAllocation"
                    ):
                        if self._explicit_parent_param(o, p.name, p.type) is None:
                            if p.name not in missing:
                                missing.append(p.name)
        if missing:
            raise GenerationError(
                f"class '{self.cd.name}', object '{o.instance_name}' ({o.vulkan_name}): "
                f"parent handle parameter(s) {missing} must be declared explicitly "
                f"(a sibling object name, a 'Class.object' reference, or a "
                f"'{{{{ Type name }}}}' placeholder); parent handles are never inferred"
            )

    def _declared_parent_placeholders(self, o: ObjectDef) -> list[tuple[str, str]]:
        """(param_name, type) for parent handles declared as placeholders:
        they become constructor parameters."""
        out: list[tuple[str, str]] = []
        cmd = self.registry.resolve_command(o.binding.create_command)
        if cmd is None:
            return out
        for p in cmd.params:
            if p.name in o.binding.parent_handles and p.type in self.registry.handles:
                v = self._explicit_parent_param(o, p.name, p.type)
                if isinstance(v, Placeholder):
                    ptype = v.explicit_type or p.type
                    out.append((v.name, ptype))
        return out

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

        # parent handles declared in the spec as placeholders are
        # constructor parameters (sibling/cross-class refs need none)
        for o in self.ordered_objects():
            for hname, htype in self._declared_parent_placeholders(o):
                if hname not in seen:
                    params.append((htype, hname))
                    seen.add(hname)

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
        for o in self.ordered_objects():
            self._require_explicit_parents(o)
        ctor = self.ctor_params()
        param_list = ", ".join(f"{t} {n}" for t, n in ctor)
        inits = self._ctor_init_list()

        lines: list[str] = [f"class {self.cd.name} {{", "public:"]
        lines.append(f"    explicit {self.cd.name}({param_list}){(' : ' + ', '.join(inits)) if inits else ''} {{")
        for o in self.ordered_objects():
            lines.append("")
            lines.append(f"        // {o.vulkan_name} '{o.instance_name}'")
            lines.extend(_indent(self._gen_object_creation(o), 8))
        lines.append("}")
        lines.append("")
        lines.append(f"    ~{self.cd.name}() {{ destroyHandles(); }}")
        lines.append("")
        lines.append(f"    {self.cd.name}(const {self.cd.name}&) = delete;")
        lines.append(f"    {self.cd.name}& operator=(const {self.cd.name}&) = delete;")
        # movable: transfers handle ownership (and the saved parent
        # handles); moved-from objects reset their handles so the
        # destructor is a no-op (vector-compatible)
        member_list = [f"{o.instance_name}_" for o in self.ordered_objects()]
        saved_list = [m for m, _ in self._needed_saved_handles()]
        lines.append(f"    {self.cd.name}({self.cd.name}&& other) noexcept {{")
        for m in member_list + saved_list:
            lines.append(f"        {m} = other.{m};")
        for m in member_list:
            lines.append(f"        other.{m} = VK_NULL_HANDLE;")
        lines.append("    }")
        lines.append(f"    {self.cd.name}& operator=({self.cd.name}&& other) noexcept {{")
        lines.append("        if (this != &other) {")
        lines.append("            destroyHandles();")
        for m in member_list + saved_list:
            lines.append(f"            {m} = other.{m};")
        for m in member_list:
            lines.append(f"            other.{m} = VK_NULL_HANDLE;")
        lines.append("        }")
        lines.append("        return *this;")
        lines.append("    }")
        lines.append("")
        # public read-only accessors; the handles themselves are private
        # so they cannot be reassigned (which would break RAII)
        lines.append("    // read-only access to the created handles")
        for o in self.ordered_objects():
            lines.append(
                f"    {o.binding.handle} {o.instance_name}() const {{ return {o.instance_name}_; }}"
            )
            if getattr(o.binding, "extra_info_struct", None):
                lines.append(
                    f"    VmaAllocation {o.instance_name}Allocation() const {{ return {o.instance_name}Allocation_; }}"
                )
        lines.append("")
        lines.append("    void destroyHandles() noexcept {")
        for o in reversed(self.ordered_objects()):
            lines.append("")
            lines.append(f"        // {o.vulkan_name} '{o.instance_name}'")
            lines.extend(_indent(self._gen_object_destruction(o), 4))
        lines.append("    }")
        lines.append("")
        lines.append("private:")
        for o in self.ordered_objects():
            lines.append(f"    {o.binding.handle} {o.instance_name}_{{}};")
            extra_struct_name = getattr(o.binding, "extra_info_struct", None)
            if extra_struct_name:
                lines.append(f"    VmaAllocation {o.instance_name}Allocation_{{}};")
        private_members = self._private_members()
        if private_members:
            for decl in private_members:
                lines.append(f"    // saved for destruction")
                lines.append(f"    {decl}")
        lines.append("};")
        return "\n".join(lines)

    def _ctor_init_list(self) -> list[str]:
        inits: list[str] = []
        for member, _ in self._needed_saved_handles():
            inits.append(f"{member}({self._saved_handle_init(member)})")
        return inits

    def _saved_handle_init(self, member: str) -> str:
        """Expression that initializes a stored parent-handle member from
        its declared value (constructor parameter, sibling member, or
        cross-class accessor call on a constructor parameter)."""
        pname = member[:-1]
        for o in self.ordered_objects():
            if not o.binding.destroy_command:
                continue
            dcmd = self.registry.resolve_command(o.binding.destroy_command)
            if dcmd is None:
                continue
            for p in dcmd.params:
                if _member_var_name(p.name) != pname:
                    continue
                v = self._explicit_parent_param(o, p.name, p.type)
                if isinstance(v, Placeholder):
                    return v.name
                if isinstance(v, Ref):
                    if v.kind == "sibling":
                        return f"{v.object_name}_"
                    return f"{_lower_first(v.class_name)}.{v.object_name}()"
        return pname

    def _private_members(self) -> list[str]:
        out: list[str] = []
        for member, ptype in self._needed_saved_handles():
            out.append(f"{ptype} {member}{{}};")
        return out

    def _placeholder_store(self, o: ObjectDef, p, ph: Placeholder) -> str:
        """Member name that stores a placeholder-declared parent handle so
        the destructor can pass it again. The member is declared by
        _private_members and initialized from the constructor parameter."""
        return f"{_member_var_name(p.name)}_"

    def _needed_saved_handles(self) -> list[tuple[str, str]]:
        """(member_name, type) for parent handles declared as placeholders
        that the destroy command needs: they must outlive the constructor."""
        out: list[tuple[str, str]] = []
        for o in self.ordered_objects():
            if not o.binding.destroy_command:
                continue
            dcmd = self.registry.resolve_command(o.binding.destroy_command)
            if dcmd is None:
                continue
            for p in dcmd.params:
                if (
                    p.name in o.binding.destroy_parent_handles
                    and p.type in self.registry.handles
                    and p.type != o.binding.handle
                    and p.type != "VmaAllocation"
                ):
                    v = self._explicit_parent_param(o, p.name, p.type)
                    if v is not None:
                        member = f"{_member_var_name(p.name)}_"
                        if member not in [m for m, _ in out]:
                            out.append((member, p.type))
        return out

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
        # VMA bindings carry a second CreateInfo (the allocation params)
        # declared under the reserved 'allocation:' key
        allocation_fields = o.params.get("allocation")
        extra_struct_name = getattr(o.binding, "extra_info_struct", None)
        if isinstance(allocation_fields, StructValue) and extra_struct_name:
            extra_struct = self.registry.resolve_struct(extra_struct_name)
            if extra_struct is None:
                raise GenerationError(f"unknown struct {extra_struct_name}")
            # 'allocation' and parent handle declarations are not members
            # of the primary CreateInfo
            skip = self._parent_param_names(o) | {"allocation"}
            primary_params = {k: v for k, v in o.params.items() if k not in skip}
            self._emit_struct(locals_, struct, primary_params, info_var, o)
            extra_var = f"{o.instance_name}AllocationCreateInfo"
            self._emit_struct(locals_, extra_struct, allocation_fields.fields, extra_var, o)
        else:
            skip = self._parent_param_names(o) | {"allocation"}
            create_params = {k: v for k, v in o.params.items() if k not in skip}
            self._emit_struct(locals_, struct, create_params, info_var, o)

        extra_var = f"{o.instance_name}AllocationCreateInfo" if isinstance(allocation_fields, StructValue) else None
        call_args: list[str] = []
        for p in cmd.params:
            if p.name == o.binding.create_info_param:
                call_args.append(f"&{info_var}")
            elif extra_var is not None and p.name == "pAllocationCreateInfo":
                call_args.append(f"&{extra_var}")
            elif extra_var is not None and p.name == "pAllocation":
                call_args.append(f"&{o.instance_name}Allocation_")
            elif extra_var is not None and p.name == "pAllocationInfo":
                call_args.append("nullptr")
            elif p.is_pointer and not p.is_const and p.type in self.registry.handles:
                call_args.append(f"&{o.instance_name}_")
            elif p.type == "uint32_t" and p.len_value is not None and o.binding.plural:
                call_args.append("1")
            elif o.binding.plural and p.len_value is not None:
                call_args.append("1")
            elif p.type in self.registry.handles and not p.is_pointer:
                v = self._explicit_parent_param(o, p.name, p.type)
                if v is None:
                    if p.name in o.binding.parent_handles and not p.optional:
                        raise GenerationError(
                            f"class '{self.cd.name}', object '{o.instance_name}': parent handle "
                            f"'{p.name}' must be declared explicitly; parent handles are never inferred"
                        )
                    call_args.append("VK_NULL_HANDLE")
                elif isinstance(v, Ref) and v.kind == "sibling":
                    # a sibling is created earlier in the same constructor;
                    # use its member directly
                    call_args.append(f"{v.object_name}_")
                elif isinstance(v, Ref):
                    # cross-class: call the accessor on the constructor's
                    # class reference (valid during construction)
                    call_args.append(f"{_lower_first(v.class_name)}.{v.object_name}()")
                elif isinstance(v, Placeholder):
                    call_args.append(v.name)
                else:
                    call_args.append(self._scalar(v))
            else:
                call_args.append("nullptr")

        lines = list(locals_)
        call = f"{o.binding.create_command}({', '.join(call_args)})"
        if cmd.return_type == "VkResult":
            lines.append(f"{{ VkResult result__ = {call};")
            lines.append(f"    if (result__ != VK_SUCCESS) {{")
            lines.append(
                f'        throw crater::VulkanError(std::string("{o.binding.create_command} failed (") + crater::result_name(result__) + ")", result__);'
            )
            lines.append("    }")
            lines.append("}")
        else:
            lines.append(f"{call};")
        return lines

    def _handle_arg(self, p) -> str:
        sibling = self._sibling_for_handle(p.type)
        if sibling is not None:
            return f"{sibling.instance_name}_"
        ext = self._external_for_handle(p.type)
        if ext is not None:
            return f"{ext[0]}.{ext[1]}()"
        return f"{_lower_first(p.type)}_"

    # struct emission ------------------------------------------------------

    def _emit_struct(self, locals_: list[str], struct: Struct, fields: dict, var: str, o: ObjectDef) -> None:
        locals_.append(f"{struct.resolved_name} {var}{{}};")
        stype = next((m.value for m in struct.members if m.value), None)
        if stype:
            locals_.append(f"{var}.sType = {stype};")
        # pointer-array members fed by a single Ref/Placeholder (or a
        # plain expression) imply their count member = 1 unless the
        # count is declared explicitly; declared arrays derive their
        # own counts
        derived_counts: dict[str, str] = {}
        for key, value in fields.items():
            member = self._match_member(struct, key)
            if member is None or not member.is_pointer or not member.array_len:
                continue
            if not isinstance(value, (Placeholder, Ref, str)):
                continue
            # only derive a count when array_len names a real count
            # member (null-terminated arrays have no count to set)
            if member.array_len not in {m.name for m in struct.members}:
                continue
            if member.array_len in fields:
                continue
            derived_counts[member.array_len] = "1"
        for count_member, expr in derived_counts.items():
            locals_.append(f"{var}.{count_member} = {expr};")
        for key, value in fields.items():
            member = self._match_member(struct, key)
            if member is None:
                raise GenerationError(
                    f"class '{self.cd.name}', object '{o.instance_name}': '{key}' is not a "
                    f"member of {struct.resolved_name}"
                )
            if member.is_array_len_member and isinstance(value, dict):
                continue  # count given as a struct makes no sense; derive it
            if member.is_array_len_member and not isinstance(value, (Placeholder, Ref, int, float)):
                continue  # counts are derived automatically unless given
            expr = self._emit_member_value(locals_, struct, member, value, o, var)
            if expr is not None:
                locals_.append(f"{var}.{member.name} = {expr};")

    def _emit_pnext_chain(self, locals_: list[str], chain: PNextChain, target: Struct, o: ObjectDef, var: str) -> str:
        return emit_pnext_chain(self.registry, self.cd.name, locals_, chain, target, o, var, self._emit_struct)

    def _emit_member_value(self, locals_: list[str], struct: Struct, member: Member, value, o: ObjectDef, var: str) -> Optional[str]:
        base = _member_var_name(member.name)

        if isinstance(value, PNextChain):
            if member.name != "pNext":
                raise GenerationError(
                    f"class '{self.cd.name}', object '{o.instance_name}': pNext "
                    f"chains are only valid on the pNext member, not '{member.name}'"
                )
            return self._emit_pnext_chain(locals_, value, struct, o, var)
        if isinstance(value, Placeholder):
            return value.name
        if isinstance(value, Ref):
            expr = self._ref_expr(value, o, member)
            if member.is_pointer and (
                member.type in self.registry.structs or member.type in self.registry.handles
            ):
                if expr.endswith("()"):
                    # getter call returns by value; materialize a local
                    # so its address can be taken
                    local = f"{var}_{base}_ref"
                    locals_.append(f"{member.type} {local} = {expr};")
                    return f"&{local}"
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
            # a raw identifier/expression (e.g. capability .data() call) is
            # used as-is; only actual text becomes a std::string
            if _looks_like_expression(value):
                return value
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

    def _ref_expr(self, ref: Ref, o: ObjectDef, member) -> str:
        if ref.kind == "sibling":
            if ref.object_name not in self.objects:
                raise GenerationError(
                    f"class '{self.cd.name}', object '{o.instance_name}': reference to "
                    f"unknown sibling object '{ref.object_name}'"
                )
            # inside the class the sibling handle is the private member;
            # the public accessor is only callable after construction
            return f"{ref.object_name}_"
        other = self.class_lookup.get(ref.class_name)
        if other is None:
            raise GenerationError(
                f"class '{self.cd.name}', object '{o.instance_name}': reference to unknown "
                f"class '{ref.class_name}'"
            )
        path = ref.object_name.split(".")
        head = path[0]
        # capability results may be suffixed with .size()/.data() style calls
        is_call_suffix = any(seg.endswith("()") for seg in path[1:])
        target = other.object(head)
        if target is None:
            if (
                other.capabilities is None
                or not self._is_capability_member(other, path)
            ):
                raise GenerationError(
                    f"class '{self.cd.name}', object '{o.instance_name}': class "
                    f"'{ref.class_name}' has no object '{head}'"
                )
        elif target.binding.handle != member.type and len(path) == 1:
            raise GenerationError(
                f"class '{self.cd.name}', object '{o.instance_name}': '{ref.class_name}."
                f"{path[0]}' is a {target.binding.handle} but member "
                f"'{member.name}' expects {member.type}"
            )
        # object handles are accessed through getters; capability
        # results (vectors, structs, indices) are public data members
        if target is not None:
            return f"{_lower_first(ref.class_name)}.{ref.object_name}()"
        return f"{_lower_first(ref.class_name)}.{ref.object_name}"

    def _is_capability_member(self, other: ClassDef, path: list[str]) -> bool:
        """Check that a member path resolves inside a capabilities class."""
        assert other.capabilities is not None
        head = path[0]
        # stored constructor inputs are exposed members
        for pname, ptype in other.capabilities.inputs:
            _, cpp_name = self._caps_input_param(pname, ptype)
            if cpp_name == head:
                return True
        for qd in other.capabilities.queries:
            if qd.as_name == head:
                return True
            if qd.index and f"{qd.as_name}Index" == head:
                return True
            for name in qd.required + qd.optional:
                if self._caps_flag_name(name) == head:
                    return True
        return False

    def _caps_input_param(self, name: str, ptype) -> tuple[str, str]:
        if isinstance(ptype, Ref) and ptype.kind == "external":
            return f"const {ptype.class_name}&", _lower_first(ptype.class_name)
        if isinstance(ptype, Placeholder):
            return ptype.explicit_type or "", ptype.name
        return str(ptype), name

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
                args.append(f"{o.instance_name}_")
            elif p.name == "allocation" and p.type == "VmaAllocation":
                args.append(f"{o.instance_name}Allocation_")
            elif p.type in self.registry.handles and not p.is_pointer:
                v = self._explicit_parent_param(o, p.name, p.type)
                # declared parents are stored as members so both the
                # constructor and the destructor can pass them
                args.append(f"{_member_var_name(p.name)}_")
            elif p.is_const and p.is_pointer:
                args.append("nullptr")
        lines = [f"if ({o.instance_name}_ != VK_NULL_HANDLE) {{"]
        lines.append(f"    {o.binding.destroy_command}({', '.join(args)});")
        lines.append("}")
        return lines


def _generator_hash() -> str:
    """Short hash of the generator sources: identifies the exact code
    that produced a header (installed package vs. checkout drift is
    visible at a glance)."""
    import hashlib
    from pathlib import Path

    h = hashlib.sha256()
    here = Path(__file__).parent
    for name in ("registry.py", "spec.py", "generator.py", "cli.py"):
        p = here / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _stamp_lines(spec_classes) -> list[str]:
    import datetime

    date = datetime.datetime.now().isoformat(timespec="seconds")
    return [
        f"// Generated by crater (gen {_generator_hash()}) on {date}. Do not edit;",
        "// regenerate with: crater generate <spec> -o <header>",
    ]


_CRATER_HELPER_LINES = [
    "namespace crater {",
    "",
    "// thrown by all generated code when a Vulkan call fails or a",
    "// required capability is not available",
    "class VulkanError : public std::runtime_error {",
    "public:",
    "    explicit VulkanError(const std::string& message)",
    "        : std::runtime_error(message), result_(VK_ERROR_UNKNOWN) {}",
    "    VulkanError(const std::string& message, VkResult result)",
    "        : std::runtime_error(message), result_(result) {}",
    "    VkResult result() const noexcept { return result_; }",
    "private:",
    "    VkResult result_;",
    "};",
    "",
    "}",
    "",
]


_RESULT_NAME_LINES = [
    "// Generated by crater. Do not edit.",
    "#pragma once",
    "",
    "#ifndef CRATER_RESULT_NAME_HPP",
    "#define CRATER_RESULT_NAME_HPP",
    "",
    "#include <algorithm>",
    "#include <stdexcept>",
    "#include <string>",
    "#include <string_view>",
    "#include <vector>",
    "#include <vulkan/vulkan.h>",
    "",
    "namespace crater {",
    "inline const char* result_name(VkResult result) {",
    "    switch (result) {",
    '        case VK_SUCCESS: return "VK_SUCCESS";',
    '        case VK_NOT_READY: return "VK_NOT_READY";',
    '        case VK_TIMEOUT: return "VK_TIMEOUT";',
    '        case VK_EVENT_SET: return "VK_EVENT_SET";',
    '        case VK_EVENT_RESET: return "VK_EVENT_RESET";',
    '        case VK_INCOMPLETE: return "VK_INCOMPLETE";',
    '        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";',
    '        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";',
    '        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";',
    '        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";',
    '        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";',
    '        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";',
    '        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";',
    '        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";',
    '        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";',
    '        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";',
    '        case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";',
    '        case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";',
    '        case VK_ERROR_INVALID_EXTERNAL_HANDLE: return "VK_ERROR_INVALID_EXTERNAL_HANDLE";',
    '        case VK_ERROR_SURFACE_LOST_KHR: return "VK_ERROR_SURFACE_LOST_KHR";',
    '        case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";',
    '        case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";',
    '        case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";',
    '        case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR: return "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR";',
    '        case VK_ERROR_VALIDATION_FAILED_EXT: return "VK_ERROR_VALIDATION_FAILED_EXT";',
    '        default: return "VK_ERROR_<unknown>";',
    "    }",
    "}",
    "}  // namespace crater",
    "#endif  // CRATER_RESULT_NAME_HPP",
    "",
]


def generate(spec_classes: list[ClassDef], registry: Registry) -> str:
    from .spec import spec_namespace

    lookup = {c.name: c for c in spec_classes}
    ordered = _order_classes(spec_classes, lookup)
    ns = spec_namespace()
    # after the includes (up to and including <vulkan/vulkan.h> at index
    # 12) and inside the guard + namespace: split at the namespace line
    _NS_LINE = "namespace crater {"
    _cut = _RESULT_NAME_LINES.index(_NS_LINE) + 1
    out: list[str] = (
        list(_RESULT_NAME_LINES[:_cut]) + _CRATER_HELPER_LINES[1:-3] + list(_RESULT_NAME_LINES[_cut:])
    )
    if registry.vma_bindings_active():
        vma_inc = _RESULT_NAME_LINES.index('#include <vulkan/vulkan.h>')
        out.insert(vma_inc + 1, "#include <vk_mem_alloc.h>")
    out[0:1] = _stamp_lines(spec_classes)
    if ns:
        out.append(f"namespace {ns} {{")
    for c in ordered:
        out.append("")
        if c.capabilities is not None:
            out.append(CapabilitiesGenerator(c, registry, lookup).generate())
        elif c.calls:
            out.append(CallGenerator(c, registry, lookup).generate())
        else:
            gen = ClassGenerator(c, registry, lookup)
            out.append(gen.generate())
    if ns:
        out.append("")
        out.append(f"}}  // namespace {ns}")
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
