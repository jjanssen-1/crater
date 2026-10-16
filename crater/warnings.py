"""Static spec warnings: flag likely-invalid generated code.

Every check derives from the registry plus a few semantic rules learned
from real validation failures - see the table in the README. Warnings
never block generation; they are printed to stderr and returned for
tooling.
"""

from __future__ import annotations

from dataclasses import dataclass

from .generator import ClassGenerator
from .spec import ArrayValue, Placeholder, PNextChain, Ref, StructValue


@dataclass
class Warning:
    message: str


# enum members whose zero value is not a valid enum constant: the
# registry lists valid values; if none has value 0 and the member is
# not optional, the zero-initialized default is invalid
def _check_zero_enums(gen: ClassGenerator, warnings: list[Warning]) -> None:
    for o in gen.ordered_objects():
        struct = gen.registry.resolve_struct(o.binding.create_info_struct)
        if struct is None:
            continue
        for struct_, members_path in _walk_members(gen, struct, o.params, o):
            for m in struct_.members:
                if m.type not in gen.registry.enums or m.optional or m.value:
                    continue
                values = gen.registry.enums[m.type]
                # an enum's 0-value (e.g. the _NONE/_MAX_ENUM sentinel) is
                # fine; flag only when 0 is not among the valid values at
                # all AND the spec does not set the member
                yield_path = members_path.get(m.name)
                if yield_path is not None:
                    continue
                if not any(v == "0" for v in values.values()):
                    warnings.append(
                        Warning(
                            f"class '{gen.cd.name}', object '{o.instance_name}': member "
                            f"'{m.name}' of {struct_.resolved_name} is a non-optional enum "
                            f"({m.type}) left at its zero default, which is not a valid "
                            f"value; set it explicitly"
                        )
                    )


def _walk_members(gen, struct, fields, o, prefix=""):
    """Yield (struct, {member.name: value}) for a struct tree."""
    members = {m.name: None for m in struct.members}
    for key, value in fields.items():
        member = gen._match_member(struct, key)
        if member is not None:
            members[member.name] = value
    yield struct, members
    for key, value in fields.items():
        member = gen._match_member(struct, key)
        if member is None:
            continue
        sub = gen.registry.resolve_struct(member.type) if member.type.startswith("Vk") else None
        if sub is not None and isinstance(value, StructValue):
            yield from _walk_members(gen, sub, value.fields, o, prefix + key + ".")
        elif sub is not None and isinstance(value, ArrayValue):
            for i, item in enumerate(value.items):
                if isinstance(item, StructValue):
                    yield from _walk_members(gen, sub, item.fields, o, f"{prefix}{key}[{i}].")

    # pNext chain entries are structs too
    for key, value in fields.items():
        if isinstance(value, PNextChain):
            for entry_name, sv in value.entries:
                ext = gen.registry.resolve_struct(entry_name)
                if ext is not None:
                    yield from _walk_members(gen, ext, sv.fields, o, prefix + key + ".")


def _check_count_pointer_pairs(gen: ClassGenerator, warnings: list[Warning]) -> None:
    """Count members left at zero while their array pointer is set, or
    arrays whose count is derived but the pointer stays null."""
    for o in gen.ordered_objects():
        struct = gen.registry.resolve_struct(o.binding.create_info_struct)
        if struct is None:
            continue
        for struct_, members in _walk_members(gen, struct, o.params, o):
            declared = {k: v for k, v in members.items() if v is not None}
            for m in struct_.members:
                if m.is_array_len_member:
                    ptr = next(
                        (p for p in struct_.members if p.array_len == m.name), None
                    )
                    if ptr is None:
                        continue
                    count_set = m.name in declared
                    ptr_set = ptr.name in declared or any(
                        k == ptr.name for k in declared
                    )
                    # a declared array/list is handled elsewhere (count derived)
                    ptr_value = declared.get(ptr.name)
                    is_list = isinstance(ptr_value, ArrayValue)
                    if count_set or ptr_set is False:
                        continue
                    if ptr_set and not count_set and not is_list:
                        # pointer set but count left at zero: the derived
                        # count only fires for ref/placeholder strings; a
                        # list derives its own; other kinds are suspicious
                        if isinstance(ptr_value, (Placeholder, Ref, str)):
                            continue  # derived = 1 by the generator
                        warnings.append(
                            Warning(
                                f"class '{gen.cd.name}', object '{o.instance_name}': "
                                f"'{m.name}' stays 0 while '{ptr.name}' is set; the "
                                f"count will not match"
                            )
                        )


def _check_required_pointer_states(gen: ClassGenerator, warnings: list[Warning]) -> None:
    """Pipeline-type states the registry marks optional but Vulkan
    requires for the pipeline kind (pMultisampleState for graphics).
    The registry's own 'noautovalidity' handles most; this covers the
    known practically-required set, inferred structurally."""
    # graphics pipeline create infos
    if not any(o.binding.create_info_struct == "VkGraphicsPipelineCreateInfo" for o in gen.ordered_objects()):
        return
    for o in gen.ordered_objects():
        if o.binding.create_info_struct != "VkGraphicsPipelineCreateInfo":
            continue
        declared = set(o.params.keys())
        for member_name, hint in (
            ("multisampleState", "pMultisampleState"),
            ("rasterizationState", "pRasterizationState"),
            ("viewportState", "pViewportState"),
            ("inputAssemblyState", "pInputAssemblyState"),
            ("colorBlendState", "pColorBlendState"),
        ):
            if member_name not in declared:
                warnings.append(
                    Warning(
                        f"class '{gen.cd.name}', object '{o.instance_name}': graphics "
                        f"pipeline does not declare '{member_name}' ({hint}); it is "
                        f"required for graphics pipelines even though the registry "
                        f"marks it optional"
                    )
                )
        # dynamic viewport/scissor without arrays requires the dynamicState
        vp = o.params.get("viewportState")
        if isinstance(vp, StructValue):
            fields = vp.fields
            counts = {k: v for k, v in fields.items() if k.endswith("Count")}
            has_arrays = any(isinstance(v, ArrayValue) for v in fields.values())
            dyn = o.params.get("dynamicState")
            has_dynamic = isinstance(dyn, StructValue) and any(
                isinstance(v, ArrayValue) for v in dyn.fields.values()
            )
            if counts and not has_arrays and not has_dynamic:
                warnings.append(
                    Warning(
                        f"class '{gen.cd.name}', object '{o.instance_name}': viewport "
                        f"state declares counts without viewport/scissor arrays and no "
                        f"dynamic states; declare them dynamic or provide the arrays"
                    )
                )


def collect_warnings(spec_classes, registry):
    """All static warnings for the parsed spec."""
    warnings: list[Warning] = []
    lookup = {c.name: c for c in spec_classes}
    for c in spec_classes:
        if c.capabilities is not None or c.calls:
            continue  # capability/call classes: separate checks later
        gen = ClassGenerator(c, registry, lookup)
        _check_zero_enums(gen, warnings)
        _check_count_pointer_pairs(gen, warnings)
        _check_required_pointer_states(gen, warnings)
    return warnings
