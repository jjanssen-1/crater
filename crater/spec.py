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

    def object(self, name: str) -> Optional[ObjectDef]:
        for o in self.objects:
            if o.instance_name == name:
                return o
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


def _quote_placeholders(text: str) -> str:
    """Wrap bare {{ ... }} tokens in double quotes so YAML treats them as scalars."""

    def repl(m: re.Match) -> str:
        start = m.start()
        if start > 0 and text[start - 1] in ('"', "'"):
            return m.group(0)
        return '"' + m.group(0) + '"'

    return _PLACEHOLDER_TOKEN_RE.sub(repl, text)


CLASS_REF_RE = re.compile(r"^(?P<class>[A-Z][A-Za-z0-9_]*)\.(?P<object>[A-Za-z_]\w*)$")
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
            return Ref(kind="external", object_name=m.group("object"), class_name=m.group("class"))
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


def parse_spec(text: str, registry: Registry) -> list[ClassDef]:
    data = yaml.safe_load(_quote_placeholders(text))
    if data is None:
        return []
    if not isinstance(data, dict):
        raise SpecError("top level of the spec must be a mapping of class names")
    bindings = registry.bindings()

    classes: list[ClassDef] = []
    for class_name, class_body in data.items():
        if not isinstance(class_body, dict):
            raise SpecError(f"class '{class_name}' must be a mapping of Vulkan objects")
        cd = ClassDef(name=class_name)
        for vulkan_name, entry in class_body.items():
            cd.objects.extend(_objects_from_entry(vulkan_name, entry, registry, bindings))
        names = [o.instance_name for o in cd.objects]
        if len(names) != len(set(names)):
            raise SpecError(f"class '{class_name}' contains duplicate object names: {names}")
        classes.append(cd)
    _resolve_refs(classes)
    return classes
