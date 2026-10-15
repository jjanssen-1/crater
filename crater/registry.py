"""Parsing of the Vulkan API registry (vk.xml).

Everything the generator knows about Vulkan comes from this module:
struct members, command signatures, which command creates which handle,
and which command destroys it. There are deliberately no special cases
for specific Vulkan types.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_DEV_REGISTRY = Path(__file__).parent / "data" / "vk.xml"


class RegistryNotFoundError(Exception):
    """Raised when no Vulkan registry (vk.xml) could be located."""


def find_registry() -> Path:
    """Locate vk.xml.

    Search order:

    1. $CRATER_VK_XML
    2. a gitignored dev copy at crater/data/vk.xml
    3. the Vulkan SDK pointed to by $VULKAN_SDK (share/vulkan/registry,
       registry, Config)
    4. system locations (/usr/share/vulkan/registry, ...)
    """
    env = os.environ.get("CRATER_VK_XML")
    if env and Path(env).is_file():
        return Path(env)
    if _DEV_REGISTRY.is_file():
        return _DEV_REGISTRY
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        for rel in ("share/vulkan/registry/vk.xml", "registry/vk.xml", "Config/vk.xml"):
            candidate = Path(sdk) / rel
            if candidate.is_file():
                return candidate
    for candidate in (
        Path("/usr/share/vulkan/registry/vk.xml"),
        Path("/usr/local/share/vulkan/registry/vk.xml"),
        Path("/usr/share/vkregistry/vk.xml"),
    ):
        if candidate.is_file():
            return candidate
    raise RegistryNotFoundError(
        "could not find the Vulkan registry (vk.xml). Install the Vulkan "
        "SDK (VULKAN_SDK), or pass --registry / set CRATER_VK_XML to a "
        "vk.xml file."
    )


@dataclass
class Member:
    """A single member of a Vulkan struct."""

    name: str
    type: str
    is_pointer: bool
    is_const: bool
    is_array: bool = False
    array_len: Optional[str] = None
    optional: bool = False
    value: Optional[str] = None  # e.g. sType value like VK_STRUCTURE_TYPE_...
    len_reference: Optional[str] = None  # name of the member holding our count, if any
    is_array_len_member: bool = False  # this member is a count for another member


@dataclass
class Struct:
    name: str
    members: list[Member] = field(default_factory=list)
    alias: Optional[str] = None

    @property
    def resolved_name(self) -> str:
        return self.alias or self.name

    def member(self, name: str) -> Optional[Member]:
        for m in self.members:
            if m.name == name:
                return m
        return None


@dataclass
class Param:
    """A parameter of a Vulkan command."""

    name: str
    type: str
    is_pointer: bool
    is_const: bool
    is_output: bool = False  # written by the command
    optional: bool = False
    len_reference: Optional[str] = None  # parameter name whose value gives our length
    len_value: Optional[str] = None  # we are the count for the parameter named here
    array_len: Optional[str] = None


@dataclass
class Command:
    name: str
    return_type: str
    params: list[Param] = field(default_factory=list)
    successcodes: list[str] = field(default_factory=list)
    alias: Optional[str] = None


@dataclass
class QueryBinding:
    """The inference result: how a capability can be queried.

    Three generic shapes are recognized from command signatures alone:

    * 'array'  - two-call count+array query
                 (vkGetPhysicalDeviceSurfaceFormatsKHR, ...);
    * 'struct' - single output struct pointer
                 (vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
    * 'present'- array query whose item struct carries a name member
                 (extensions, layers); doubles as an availability query.
    """

    query_command: str
    shape: str
    item_type: Optional[str] = None
    output_param: str = ""
    count_param: Optional[str] = None
    output_struct: Optional[str] = None
    inputs: list[Param] = field(default_factory=list)
    enumerated_type: Optional[str] = None
    name_member: Optional[str] = None


@dataclass
class CreateInfoBinding:
    """The inference result: how a handle type is created and destroyed."""

    handle: str
    create_command: str
    create_info_struct: str
    create_info_param: str  # name of the Vk*CreateInfo* parameter
    destroy_command: Optional[str] = None
    # handle parameters of the create command that are not the create info,
    # not the allocator and not the output (e.g. the VkDevice of a vkCreate*).
    parent_handles: list[str] = field(default_factory=list)
    # same, for the destroy command
    destroy_parent_handles: list[str] = field(default_factory=list)
    # parameters of the create command the user may override (e.g. pAllocator)
    extra_params: list[Param] = field(default_factory=list)
    plural: bool = False  # vkCreateGraphicsPipelines creates an array of handles


class Registry:
    def __init__(self, path: Optional[Path] = None):
        if path is None:
            path = find_registry()
        self.path = Path(path)
        self._root = ET.parse(str(self.path)).getroot()
        self.structs: dict[str, Struct] = {}
        self.commands: dict[str, Command] = {}
        self.handles: set[str] = set()
        self.enums: dict[str, dict[str, str]] = {}  # enum type -> {name: value}
        self.bitmasks: dict[str, dict[str, str]] = {}
        self.bitmask_bits: dict[str, str] = {}  # VkQueueFlags -> VkQueueFlagBits
        self._queries: Optional[dict[str, QueryBinding]] = None
        self._extension_of: dict[str, tuple[str, str]] = {}  # name -> (ext, platform)
        self._parse()
        self._parse_extension_membership()

    # ------------------------------------------------------------------ parse

    def _parse(self) -> None:
        for node in self._root.findall("types/type"):
            api = node.get("api")
            if api and "vulkan" not in api.split(",") and node.get("category") == "struct":
                name = node.get("name")
                if name in self.structs:
                    continue
            category = node.get("category")
            name = node.get("name")
            if category == "struct":
                if name:
                    self.structs[name] = self._parse_struct(node)
            elif category == "handle":
                name_node = node.find("name")
                if name_node is not None and name_node.text:
                    self.handles.add(name_node.text)
            elif category == "bitmask":
                nm = name or (node.find("name").text if node.find("name") is not None else None)
                if nm:
                    self.bitmask_bits[nm] = node.get("requires")

        for node in self._root.findall("commands/command"):
            cmd = self._parse_command(node)
            if cmd is None:
                continue
            api = node.get("api")
            if cmd.name in self.commands and api and "vulkan" not in api.split(","):
                continue  # prefer core (vulkan) over restricted variants
            if cmd.name not in self.commands or not api or "vulkan" in api.split(","):
                self.commands[cmd.name] = cmd

        for node in self._root.findall("enums"):
            name = node.get("name")
            if not name:
                continue
            if node.get("type") == "bitmask":
                target = self.bitmasks.setdefault(name, {})
            else:
                target = self.enums.setdefault(name, {})
            for e in node.findall("enum"):
                if e.get("name"):
                    target[e.get("name")] = e.get("value") or e.get("bitpos") or ""

    def _parse_struct(self, node: ET.Element) -> Struct:
        s = Struct(name=node.get("name"), alias=node.get("alias"))
        if s.alias:
            return s
        for m in node.findall("member"):
            type_node = m.find("type")
            name_node = m.find("name")
            if type_node is None or name_node is None:
                continue
            text = "".join(m.itertext())
            member = Member(
                name=name_node.text,
                type=type_node.text,
                is_pointer="*" in text,
                is_const="const" in text,
                optional=m.get("optional") is not None,
                value=m.get("values"),
            )
            lenattr = m.get("len") or ""
            if lenattr and not lenattr.startswith("latexmath"):
                member.array_len = lenattr.split(",")[0]
                member.is_array = True
            s.members.append(member)
        self._mark_count_members(s)
        return s

    @staticmethod
    def _mark_count_members(s: Struct) -> None:
        for m in s.members:
            if m.array_len and m.array_len in {x.name for x in s.members}:
                count = s.member(m.array_len)
                if count is not None:
                    count.is_array_len_member = True

    def _parse_command(self, node: ET.Element) -> Optional[Command]:
        proto = node.find("proto")
        if proto is None:  # alias-only entry
            return None
        proto_type = proto.find("type")
        proto_name = proto.find("name")
        if proto_type is None or proto_name is None:
            return None
        cmd = Command(
            name=proto_name.text,
            return_type=proto_type.text,
            successcodes=(node.get("successcodes") or "").split(","),
        )
        for p in node.findall("param"):
            type_node = p.find("type")
            name_node = p.find("name")
            if type_node is None or name_node is None:
                continue
            api = p.get("api")
            if api and "vulkan" not in api.split(","):
                continue  # restricted variant (e.g. vulkansc) of an existing param
            if any(x.name == name_node.text for x in cmd.params):
                continue  # duplicate variant of the same parameter
            text = "".join(p.itertext())
            param = Param(
                name=name_node.text,
                type=type_node.text,
                is_pointer="*" in text,
                is_const="const" in text,
                optional=p.get("optional") is not None,
            )
            lenattr = p.get("len") or ""
            if lenattr:
                param.array_len = lenattr.split(",")[0]
            cmd.params.append(param)
        self._mark_command_counts(cmd)
        return cmd

    @staticmethod
    def _mark_command_counts(cmd: Command) -> None:
        names = {p.name for p in cmd.params}
        for p in cmd.params:
            if p.array_len and p.array_len in names:
                count = next(x for x in cmd.params if x.name == p.array_len)
                count.len_value = p.name
                p.len_reference = p.array_len

    # ----------------------------------------------------------------- lookup

    def resolve_struct(self, name: str) -> Optional[Struct]:
        s = self.structs.get(name)
        if s is None:
            return None
        if s.alias:
            return self.structs.get(s.alias)
        return s

    def resolve_command(self, name: str) -> Optional[Command]:
        c = self.commands.get(name)
        if c is None:
            return None
        if c.alias:
            return self.commands.get(c.alias)
        return c

    def _alias_commands(self) -> dict[str, Command]:
        out: dict[str, Command] = {}
        for node in self._root.findall("commands/command"):
            alias = node.get("alias")
            name = node.get("name")
            if alias and name and alias in self.commands:
                out[name] = self.commands[alias]
        return out

    def _parse_extension_membership(self) -> None:
        """Map every extension-provided type/command to (extension, platform)."""
        platforms = {}
        for node in self._root.findall("platforms/platform"):
            platforms[node.get("name")] = node.get("protect")
        for ext in self._root.findall("extensions/extension"):
            ext_name = ext.get("name")
            protect = platforms.get(ext.get("platform"))
            for req in ext.findall("require"):
                for item in list(req.findall("type")) + list(req.findall("command")) + list(req.findall("enum")):
                    name = item.get("name") or (item.find("name").text if item.find("name") is not None else None)
                    if name and name not in self._extension_of:
                        self._extension_of[name] = (ext_name, protect or "")

    def extension_of(self, name: str) -> Optional[tuple[str, str]]:
        """(extension name, platform protect) providing the given type or
        command, if it comes from an extension."""
        return self._extension_of.get(name)

    def flag_bits_type(self, flags_type: str) -> Optional[str]:
        """The *FlagBits enum backing a *Flags bitmask type, if any."""
        return self.bitmask_bits.get(flags_type)

    def flag_values(self, flag_bits_type: str) -> set[str]:
        """Valid constant names of a *FlagBits enum."""
        return set(self.bitmasks.get(flag_bits_type, {})) | set(self.enums.get(flag_bits_type, {}))

    # -------------------------------------------------------------- inference

    def bindings(self) -> dict[str, CreateInfoBinding]:
        """Infer creation/destruction info for every creatable handle.

        The rules are generic, driven purely by command signatures:

        * a "create command" is named vkCreate* or vkAllocate*, takes exactly
          one const Vk*CreateInfo* / Vk*Info* parameter and writes exactly one
          handle (or one handle array) via its last output parameter;
        * a "destroy command" is named vkDestroy* and returns void.

        Bindings are keyed by the *name inferred from the creation function*
        (e.g. vkCreateGraphicsPipelines -> "VkGraphicsPipeline"), because the
        resulting handle type may be shared between several creation
        functions (VkPipeline is produced by graphics, compute and ray
        tracing pipeline creation alike).
        """
        commands = dict(self.commands)
        commands.update(self._alias_commands())

        bindings: dict[str, CreateInfoBinding] = {}

        for cmd in commands.values():
            if not (cmd.name.startswith("vkCreate") or cmd.name.startswith("vkAllocate")):
                continue
            info_param = None
            for p in cmd.params:
                if p.is_const and p.is_pointer and ("CreateInfo" in p.type or p.type.endswith("Info")):
                    info_param = p
            if info_param is None:
                continue
            outputs = [p for p in cmd.params if p.is_pointer and not p.is_const and p.type in self.handles]
            if len(outputs) != 1:
                continue
            handle_param = outputs[0]
            parents = [
                p
                for p in cmd.params
                if p is not info_param and p is not handle_param and p.type in self.handles and not p.is_pointer
            ]
            extras = [
                p
                for p in cmd.params
                if p is not info_param
                and p is not handle_param
                and p.type not in self.handles
                and p.type != cmd.return_type
                and p.is_const
                and p.is_pointer
            ]
            plural = handle_param.len_reference is not None
            key = self._creation_name(cmd.name, plural) if cmd.name.startswith("vkCreate") else handle_param.type
            binding = CreateInfoBinding(
                handle=handle_param.type,
                create_command=cmd.name,
                create_info_struct=info_param.type,
                create_info_param=info_param.name,
                parent_handles=[p.name for p in parents],
                plural=plural,
            )
            binding.extra_params = extras
            bindings[key] = binding

        # destroy inference
        for binding in bindings.values():
            destroy = self._find_destroy(commands, binding)
            if destroy is not None:
                name, parents = destroy
                binding.destroy_command = name
                binding.destroy_parent_handles = parents

        return bindings

    @staticmethod
    def _creation_name(command: str, plural: bool) -> str:
        stem = command[len("vkCreate") :]
        if plural and stem.endswith("s"):
            stem = stem[:-1]
        return f"Vk{stem}"

    def _find_destroy(self, commands: dict[str, Command], binding: CreateInfoBinding):
        candidates = []
        for cmd in commands.values():
            if cmd.name.startswith("vkDestroy") and cmd.return_type == "void":
                if any(p.type == binding.handle and not p.is_pointer for p in cmd.params):
                    candidates.append(cmd)
        if not candidates:
            return None
        # prefer the destroy command whose name matches the create command
        stem_create = binding.create_command[len("vkCreate") :]
        preferred = f"vkDestroy{stem_create}"
        exact = [c for c in candidates if c.name == preferred]
        if exact:
            cmd = exact[0]
        else:
            # fall back: the destroy command matching the create info stem
            stem_info = binding.create_info_struct[len("Vk") :].replace("CreateInfo", "")
            match = [c for c in candidates if c.name == f"vkDestroy{stem_info}"]
            cmd = match[0] if match else candidates[0]
        parents = [
            p.name
            for p in cmd.params
            if p.type in self.handles and p.type != binding.handle and not p.is_pointer
        ]
        return cmd.name, parents

    def _parse_extension_membership(self) -> None:
        """Map every extension-provided type/command to (extension, platform)."""
        platforms = {}
        for node in self._root.findall("platforms/platform"):
            platforms[node.get("name")] = node.get("protect")
        for ext in self._root.findall("extensions/extension"):
            ext_name = ext.get("name")
            protect = platforms.get(ext.get("platform"))
            for req in ext.findall("require"):
                for item in list(req.findall("type")) + list(req.findall("command")) + list(req.findall("enum")):
                    name = item.get("name") or (item.find("name").text if item.find("name") is not None else None)
                    if name and name not in self._extension_of:
                        self._extension_of[name] = (ext_name, protect or "")

    def extension_of(self, name: str) -> Optional[tuple[str, str]]:
        """(extension name, platform protect) providing the given type or
        command, if it comes from an extension."""
        return self._extension_of.get(name)

    def flag_bits_type(self, flags_type: str) -> Optional[str]:
        """The *FlagBits enum backing a *Flags bitmask type, if any."""
        return self.bitmask_bits.get(flags_type)

    def flag_values(self, flag_bits_type: str) -> set[str]:
        """Valid constant names of a *FlagBits enum."""
        return set(self.bitmasks.get(flag_bits_type, {})) | set(self.enums.get(flag_bits_type, {}))

    # -------------------------------------------------------------- inference

    def queries(self) -> dict[str, QueryBinding]:
        """Infer capability queries, keyed by command name.

        A command is a query if its name starts with vkGet/vkEnumerate and it
        has output parameters but creates nothing. Recognition rules are
        purely structural:

        * array shape: a non-const pointer to a struct/enum type with a len
          attribute pointing at a non-const count pointer parameter (the
          classic two-call idiom);
        * struct shape: exactly one non-const struct pointer output and no
          count involved;
        * present shape: an array query whose item struct carries a name
          member (extensionName/layerName) - the same enumerate command
          doubles as an availability query.
        """
        if self._queries is not None:
            return self._queries
        commands = dict(self.commands)
        commands.update(self._alias_commands())
        queries: dict[str, QueryBinding] = {}

        for cmd in commands.values():
            if cmd.name.startswith(("vkCreate", "vkAllocate")):
                continue
            if not cmd.name.startswith(("vkGet", "vkEnumerate")):
                continue
            array_param = None
            count_param = None
            for p in cmd.params:
                if (
                    not p.is_const
                    and p.is_pointer
                    and p.array_len
                    and p.len_reference
                    and (p.type in self.structs or p.type in self.enums or p.type in self.handles)
                ):
                    count = next((x for x in cmd.params if x.name == p.len_reference), None)
                    if count is not None and count.is_pointer and not count.is_const:
                        array_param = p
                        count_param = count
                        break
            if array_param is not None and count_param is not None:
                inputs = [
                    x
                    for x in cmd.params
                    if x is not array_param and x is not count_param and not x.is_const
                ]
                queries[cmd.name] = QueryBinding(
                    query_command=cmd.name,
                    shape="array",
                    item_type=array_param.type,
                    output_param=array_param.name,
                    count_param=count_param.name,
                    inputs=inputs,
                )
                continue
            # struct shape
            outputs = [p for p in cmd.params if not p.is_const and p.is_pointer and p.type in self.structs]
            if len(outputs) == 1 and outputs[0].array_len is None:
                p = outputs[0]
                others = [x for x in cmd.params if x is not p]
                # every other parameter must be an input: by value, or an
                # optional const pointer (allocator-style), never a second output
                if all((not x.is_pointer and not x.is_const) or x.is_const or x.type in self.handles for x in others):
                    inputs = [x for x in others if not x.is_const]
                    queries[cmd.name] = QueryBinding(
                        query_command=cmd.name,
                        shape="struct",
                        output_param=p.name,
                        output_struct=p.type,
                        inputs=inputs,
                    )

        # present shape: array queries whose item struct carries a name member
        for cmd_name, qb in list(queries.items()):
            if qb.shape != "array":
                continue
            s = self.resolve_struct(qb.item_type) if qb.item_type else None
            if s is None:
                continue
            name_member = next(
                (
                    m.name
                    for m in s.members
                    if m.type == "char" and m.is_array and "name" in m.name.lower()
                ),
                None,
            )
            if name_member is None:
                continue
            queries[cmd_name] = QueryBinding(
                query_command=qb.query_command,
                shape="present",
                item_type=qb.item_type,
                output_param=qb.output_param,
                count_param=qb.count_param,
                enumerated_type=qb.item_type,
                name_member=name_member,
                inputs=qb.inputs,
            )

        self._queries = queries
        return queries
