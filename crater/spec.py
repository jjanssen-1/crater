"""YAML spec parsing into an intermediate representation.

A spec file looks like::

    VulkanInstance:
      VkInstance:
        name: mainInstance
        applicationInfo:
          applicationName: "Test"
          apiVersion: [1, 3, 0]

    OurPipeline:
      VkPipelineLayout:
        name: layout
      VkGraphicsPipeline:
        layout: layout            # reference to the sibling object above
        renderPass: VulkanInstance.mainRenderPass  # cross-class reference
        stages:
          - stage: VK_SHADER_STAGE_VERTEX_BIT
            module: {{ vertexModule }}   # constructor parameter
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

from .registry import CreateInfoBinding, Registry

PLACEHOLDER_RE = re.compile(r"^\{\{\s*(?:(?P<type>[A-Za-z_][\w:<> *,&]*)\s+)?(?P<name>[A-Za-z_]\w*)\s*\}\}$")
_PLACEHOLDER_TOKEN_RE = re.compile(r"\{\{[^{}]*\}\}")


class SpecError(Exception):
    pass


@dataclass
class Placeholder:
    """A `{{ [type] name }}` value, provided to the generated constructor."""

    name: str
    explicit_type: Optional[str] = None

    def render(self) -> str:
        return f"{{{{{self.name}}}}}"


@dataclass
class Ref:
    """A reference to another object.

    kind is 'sibling' (same class) or 'external' (Class.object in another
    generated class, passed to the constructor as a handle).
    """

    kind: str
    object_name: str
    class_name: Optional[str] = None


@dataclass
class StructValue:
    """A nested struct (dict in YAML)."""

    fields: dict[str, "SpecValue"] = field(default_factory=dict)


@dataclass
class ArrayValue:
    """An array of values (list in YAML)."""

    items: list["SpecValue"] = field(default_factory=list)


SpecValue = Any  # Placeholder | Ref | StructValue | ArrayValue | scalar


@dataclass
class ObjectDef:
    vulkan_name: str  # name inferred from the creation function, e.g. VkGraphicsPipeline
    instance_name: str
    binding: CreateInfoBinding
    params: dict[str, SpecValue] = field(default_factory=dict)
    # ctor params this object needs (filled during resolution)
    placeholders: dict[str, Placeholder] = field(default_factory=dict)


@dataclass
class ClassDef:
    name: str
    objects: list[ObjectDef] = field(default_factory=list)
    # capabilities classes only: queries and their selection rules
    capabilities: Optional["CapabilitiesDef"] = None
    # call functions: plain command invocations with declarable info structs
    calls: list["CallDef"] = field(default_factory=list)

    def object(self, name: str) -> Optional[ObjectDef]:
        for o in self.objects:
            if o.instance_name == name:
                return o
        return None


@dataclass
class CallDef:
    """A plain command invocation generated as a member function.

    Values in the YAML fill the command's info struct (and scalar
    parameters); everything else becomes a function parameter. The result
    is checked against the command's success codes from the registry.
    """

    command_name: str
    params: dict[str, Any] = field(default_factory=dict)
    as_name: str = ""  # generated function name; default derived


@dataclass
class QueryDef:
    """One capability query inside a capabilities class.

    For array queries 'candidates' is an ordered preference list (first
    available candidate wins); 'fallback: any' selects the first available
    item when no candidate matches. 'required'/'optional' are availability
    lists for name-carrying queries (extensions, layers).
    """

    command_name: str
    queried_type: str
    binding: Any  # QueryBinding
    as_name: str  # member name exposing the result
    candidates: list = field(default_factory=list)
    fallback_any: bool = False
    required: list = field(default_factory=list)
    optional: list = field(default_factory=list)
    # member -> [constants]; element must satisfy ALL of these
    require: dict[str, list[str]] = field(default_factory=dict)
    # member -> [constants]; ordering preference among require-matches
    prefer: dict[str, list[str]] = field(default_factory=dict)
    index: bool = False  # expose as_name + "Index" (uint32_t) as well


@dataclass
class CapabilitiesDef:
    """A class that queries capabilities at runtime and selects values.

    'inputs' holds constructor parameters needed by the queries, in the
    form (param_name, Placeholder|Ref|type) — e.g. a physical device from
    another class or a surface handle from the caller.
    """

    queries: list[QueryDef] = field(default_factory=list)
    inputs: list[tuple[str, Any]] = field(default_factory=list)

    def query(self, command_name: str) -> Optional[QueryDef]:
        for q in self.queries:
            if q.command_name == command_name:
                return q
        return None


def _parse_value(value: Any) -> SpecValue:
    if isinstance(value, str):
        m = PLACEHOLDER_RE.match(value.strip())
        if m:
            return Placeholder(name=m.group("name"), explicit_type=m.group("type"))
        return value
    if isinstance(value, dict):
        return StructValue(fields={k: _parse_value(v) for k, v in value.items()})
    if isinstance(value, list):
        return ArrayValue(items=[_parse_value(v) for v in value])
    return value


def _default_instance_name(vulkan_name: str) -> str:
    base = vulkan_name[2:] if vulkan_name.startswith("Vk") else vulkan_name
    return base[0].lower() + base[1:]


def _object_from_dict(vulkan_name: str, d: dict, registry: Registry, bindings) -> ObjectDef:
    binding = bindings.get(vulkan_name)
    if binding is None:
        raise SpecError(
            f"unknown Vulkan object '{vulkan_name}': no creation command was "
            f"inferred from the registry for it"
        )
    params = {k: _parse_value(v) for k, v in d.items() if k != "name"}
    instance_name = d.get("name") or _default_instance_name(vulkan_name)
    return ObjectDef(
        vulkan_name=vulkan_name,
        instance_name=instance_name,
        binding=binding,
        params=params,
    )


def _objects_from_entry(vulkan_name: str, entry: Any, registry: Registry, bindings) -> list[ObjectDef]:
    if isinstance(entry, list):
        out = []
        for item in entry:
            if not isinstance(item, dict):
                raise SpecError(f"'{vulkan_name}' entries must be mappings, got {item!r}")
            out.append(_object_from_dict(vulkan_name, item, registry, bindings))
        return out
    if isinstance(entry, dict):
        return [_object_from_dict(vulkan_name, entry, registry, bindings)]
    raise SpecError(f"entry for '{vulkan_name}' must be a mapping or list of mappings")


def _query_def_from_entry(command_name: str, entry: Any, queries: dict, registry: Registry, class_name: str) -> QueryDef:
    qb = queries[command_name]
    if entry is None:
        entry = {}
    if not isinstance(entry, dict):
        raise SpecError(f"class '{class_name}': query '{command_name}' must be a mapping")
    as_name = entry.get("as") or _default_result_name(qb)
    qd = QueryDef(
        command_name=command_name,
        queried_type=qb.item_type or qb.output_struct or "",
        binding=qb,
        as_name=as_name,
    )
    candidates = entry.get("candidates")
    if candidates is not None:
        if qb.shape not in ("array", "struct"):
            raise SpecError(
                f"class '{class_name}': query '{command_name}' does not support candidates"
            )
        if isinstance(candidates, list) and candidates and isinstance(candidates[0], list):
            qd.candidates = [_parse_value(c) for c in candidates]  # list of rows (structs)
        else:
            qd.candidates = [_parse_value(c) for c in candidates]
    fallback = entry.get("fallback")
    if fallback is not None:
        if fallback != "any":
            raise SpecError(
                f"class '{class_name}': unsupported fallback '{fallback}' "
                f"(only 'any' is supported)"
            )
        if not qd.candidates:
            raise SpecError(
                f"class '{class_name}': fallback 'any' requires a candidates list"
            )
        qd.fallback_any = True
    for key in ("required", "optional"):
        if key not in entry:
            continue
        if qb.shape not in ("present", "struct"):
            raise SpecError(
                f"class '{class_name}': '{key}' is only valid for availability "
                f"queries (extensions/layers) and struct queries, not '{command_name}'"
            )
        items = entry[key]
        if not isinstance(items, list) or not all(isinstance(i, str) for i in items):
            raise SpecError(f"class '{class_name}': '{key}' must be a list of names")
        if qb.shape == "struct" and key == "required":
            qd.required = items  # VkBool32 member paths, checked post-query
            continue
        setattr(qd, key, items)
    for key in ("require", "prefer"):
        if key not in entry:
            continue
        if qb.shape != "array":
            raise SpecError(
                f"class '{class_name}': '{key}' needs an array query, not '{command_name}'"
            )
        if not isinstance(entry[key], dict):
            raise SpecError(f"class '{class_name}': '{key}' must be a mapping")
        predicates = {}
        for member, bits in entry[key].items():
            if isinstance(bits, str):
                bits = [bits]
            if not isinstance(bits, list) or not all(isinstance(b, str) for b in bits):
                raise SpecError(
                    f"class '{class_name}': '{key}.{member}' must be a bit name or list of bit names"
                )
            predicates[member] = bits
        setattr(qd, key, predicates)
    unknown = set(entry) - {
        "as",
        "candidates",
        "fallback",
        "required",
        "optional",
        "require",
        "prefer",
        "index",
    }
    if unknown:
        raise SpecError(
            f"class '{class_name}': unknown keys {sorted(unknown)} in query '{command_name}'"
        )
    if entry.get("index"):
        if qb.shape != "array":
            raise SpecError(f"class '{class_name}': 'index' needs an array query")
        qd.index = True
    if qb.shape == "present" and not qd.required and not qd.optional:
        raise SpecError(
            f"class '{class_name}': availability query '{command_name}' needs "
            f"'required' and/or 'optional'"
        )
    if qb.shape == "struct" and qd.optional:
        raise SpecError(
            f"class '{class_name}': 'optional' is not supported for struct queries"
        )
    return qd


def _validate_predicates(qd: QueryDef, registry: Registry, class_name: str) -> None:
    """require/prefer members must exist on the item struct, and bit names
    must be valid constants of the member's FlagBits enum."""
    s = registry.resolve_struct(qd.queried_type) if qd.queried_type else None
    if s is None:
        return
    for key in ("require", "prefer"):
        for member, bits in getattr(qd, key).items():
            m = s.member(member)
            if m is None:
                raise SpecError(
                    f"class '{class_name}': '{qd.command_name}' items have no member "
                    f"'{member}'"
                )
            flag_bits = registry.flag_bits_type(m.type)
            valid = registry.flag_values(flag_bits) if flag_bits else set()
            for b in bits:
                if flag_bits and b not in valid:
                    raise SpecError(
                        f"class '{class_name}': '{b}' is not a valid value of "
                        f"{flag_bits} (member '{member}' of {qd.command_name})"
                    )
                if not flag_bits and not b.replace("_", "").isalnum():
                    raise SpecError(
                        f"class '{class_name}': member '{member}' is not a bitmask; "
                        f"only flag names are supported"
                    )


def _default_call_name(command_name: str) -> str:
    stem = command_name[2:] if command_name.startswith("vk") else command_name
    for prefix in ("GetPhysicalDevice", "EnumerateDevice", "EnumerateInstance", "Get", "Enumerate"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    return stem[0].lower() + stem[1:]


def _validate_struct_required(qd: QueryDef, registry: Registry, class_name: str) -> None:
    """required paths on struct queries must resolve to VkBool32 members."""
    if not qd.required or qd.binding.shape != "struct":
        return
    struct = registry.resolve_struct(qd.binding.output_struct) if qd.binding.output_struct else None
    if struct is None:
        raise SpecError(f"class '{class_name}': cannot resolve {qd.binding.output_struct}")
    for path in qd.required:
        segments = path.split(".")
        current = struct
        target = None
        for seg in segments:
            target = current.member(seg)
            if target is None:
                raise SpecError(
                    f"class '{class_name}': '{qd.command_name}' result has no member "
                    f"'{seg}' (path '{path}')"
                )
            current = registry.resolve_struct(target.type) or current
        if target is None or target.type != "VkBool32":
            raise SpecError(
                f"class '{class_name}': required '{path}' is not a VkBool32 "
                f"member of {qd.binding.output_struct}"
            )


def _default_result_name(qb) -> str:
    stem = qb.query_command
    for prefix in ("vkEnumerateDevice", "vkEnumerateInstance", "vkEnumerate", "vkGetPhysicalDevice", "vkGet"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    if stem.endswith("KHR"):
        stem = stem[:-3]
    return stem[0].lower() + stem[1:]



def _capability_inputs(
    caps: CapabilitiesDef, class_name: str, declared: dict[str, Any]
) -> list[tuple[str, Any]]:
    """Union of the inputs required by all queries, in first-seen order.

    An input the spec declares explicitly overrides the registry-derived
    default (a raw handle constructor parameter). Declarations can be a
    `{{ [type] name }}` placeholder, a Class.object reference, or any value
    expression.
    """
    inputs: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for qd in caps.queries:
        for p in qd.binding.inputs:
            if p.name in seen:
                continue
            seen.add(p.name)
            if p.name in declared:
                inputs.append((p.name, declared[p.name]))
            else:
                inputs.append((p.name, p.type))
    unknown = set(declared) - seen
    if unknown:
        raise SpecError(
            f"class '{class_name}': input(s) {sorted(unknown)} are not required by "
            f"any query in this class"
        )
    return inputs


def _quote_placeholders(text: str) -> str:
    """Wrap bare {{ ... }} tokens in double quotes so YAML treats them as scalars."""

    def repl(m: re.Match) -> str:
        start = m.start()
        if start > 0 and text[start - 1] in ('"', "'"):
            return m.group(0)
        return '"' + m.group(0) + '"'

    return _PLACEHOLDER_TOKEN_RE.sub(repl, text)


CLASS_REF_RE = re.compile(r"^(?P<class>[A-Z][A-Za-z0-9_]*)\.(?P<path>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*(?:\(\))?)*)$")
IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")


def _resolve_refs(classes: list[ClassDef]) -> None:
    """Turn plain string values into Refs where they name another object.

    A value is a reference if it matches 'Class.object' for a defined class
    (external reference) or if it is a bare identifier naming an object in
    the same class (sibling reference).
    """
    class_names = {c.name for c in classes}
    for cd in classes:
        sibling_names = {o.instance_name for o in cd.objects}
        for o in cd.objects:
            for key, value in list(o.params.items()):
                o.params[key] = _resolve_value_refs(value, cd.name, sibling_names, class_names)


def _resolve_value_refs(value: Any, class_name: str, sibling_names: set[str], class_names: set[str]) -> Any:
    if isinstance(value, str):
        m = CLASS_REF_RE.match(value)
        if m and m.group("class") in class_names:
            return Ref(
                kind="external",
                object_name=m.group("path"),
                class_name=m.group("class"),
            )
        if value in sibling_names:
            return Ref(kind="sibling", object_name=value)
        return value
    if isinstance(value, StructValue):
        value.fields = {
            k: _resolve_value_refs(v, class_name, sibling_names, class_names)
            for k, v in value.fields.items()
        }
        return value
    if isinstance(value, ArrayValue):
        value.items = [_resolve_value_refs(v, class_name, sibling_names, class_names) for v in value.items]
        return value
    return value


NAMESPACE_KEY = "namespace"
_SPEC_NAMESPACE = None


def spec_namespace() -> str | None:
    """The namespace requested by the last parsed spec, if any."""
    return _SPEC_NAMESPACE


def parse_spec(text: str, registry: Registry) -> list[ClassDef]:
    data = yaml.safe_load(_quote_placeholders(text))
    if data is None:
        return []
    if not isinstance(data, dict):
        raise SpecError("top level of the spec must be a mapping of class names")
    bindings = registry.bindings()
    queries = registry.queries()
    # an optional top-level 'namespace' key wraps all generated code
    global _SPEC_NAMESPACE
    _SPEC_NAMESPACE = data.pop(NAMESPACE_KEY, None)
    if _SPEC_NAMESPACE is not None and not isinstance(_SPEC_NAMESPACE, str):
        raise SpecError("'namespace' must be a string")

    classes: list[ClassDef] = []
    for class_name, class_body in data.items():
        if not isinstance(class_body, dict):
            raise SpecError(f"class '{class_name}' must be a mapping of Vulkan objects")
        cd = ClassDef(name=class_name)
        is_capabilities = any(k in queries or k.startswith("vk") for k in class_body)
        if is_capabilities and not any(k in queries for k in class_body):
            has_calls = any(
                k.startswith("vk")
                and k in registry.commands
                and k not in queries
                and k not in bindings
                for k in class_body
            )
            if not has_calls:
                raise SpecError(
                    f"class '{class_name}': vk-prefixed entries must be capability "
                    f"queries or call functions; found neither"
                )
        declared_inputs: dict[str, Any] = {}
        for command_name, entry in class_body.items():
            if command_name in queries:
                cd.capabilities = cd.capabilities or CapabilitiesDef()
                qd = _query_def_from_entry(command_name, entry, queries, registry, cd.name)
                _validate_predicates(qd, registry, cd.name)
                _validate_struct_required(qd, registry, cd.name)
                if cd.capabilities.query(command_name) is not None:
                    raise SpecError(
                        f"class '{class_name}': duplicate query for '{command_name}'"
                    )
                cd.capabilities.queries.append(qd)
            elif (
                command_name.startswith("vk")
                and command_name in registry.commands
                and command_name not in queries
                and command_name not in bindings
            ):
                # a call function: plain command invocation
                if entry is None:
                    entry = {}
                if not isinstance(entry, dict):
                    raise SpecError(
                        f"class '{class_name}': call '{command_name}' must be a mapping"
                    )
                entry = dict(entry)
                as_name = entry.pop("as", None) or _default_call_name(command_name)
                cd.calls.append(
                    CallDef(
                        command_name=command_name,
                        params={k: _parse_value(v) for k, v in entry.items()},
                        as_name=as_name,
                    )
                )
            elif is_capabilities:
                # input declaration for the queries: a placeholder, a raw
                # value, or a Class.object reference
                declared_inputs[command_name] = _parse_value(entry)
            else:
                cd.objects.extend(_objects_from_entry(command_name, entry, registry, bindings))
        names = [o.instance_name for o in cd.objects]
        if len(names) != len(set(names)):
            raise SpecError(f"class '{class_name}' contains duplicate object names: {names}")
        if cd.capabilities is not None:
            cd.capabilities.inputs = _capability_inputs(cd.capabilities, class_name, declared_inputs)
        elif is_capabilities and not cd.capabilities and not cd.calls:
            raise SpecError(f"class '{class_name}': no queries declared")
        classes.append(cd)
    _resolve_refs(classes)
    return classes
