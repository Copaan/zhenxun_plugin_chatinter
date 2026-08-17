"""Provider-specific plugin tool assembly for the unified chat agent."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from typing import TYPE_CHECKING, Any

from zhenxun.services.cache import CacheDict

from .command_index import CommandCandidate
from .meta_tools import (
    _candidate_from_snapshot,
    render_command_candidate_context,
)
from .models.pydantic_models import CommandToolSnapshot, PluginKnowledgeBase
from .native_command_tools import build_native_command_tools
from .native_executor import NativeCommandExecutionContext
from .plugin_skill_index import PluginSkill, PluginSkillIndex
from .route_text import normalize_message_text
from .skill_dispatch_tools import build_plugin_skill_dispatch_tools
from .turn_runtime import estimate_text_tokens
from .web_access import (
    candidate_web_search_kind,
    tools_for_web_candidate,
)

if TYPE_CHECKING:
    from .host_llm import HostModelCandidate
    from .provider_capability import ProviderCapabilityAdapter

_DETAIL_PROTOCOL_MARGIN_TOKENS = 4_096
_AMBIGUITY_RESULT_TOKEN_LIMIT = 16_000
_CANDIDATE_CONTEXT_TOKEN_LIMIT = 4_096
_DISPATCH_SELECTIONS: CacheDict[tuple[str, ...]] = CacheDict(
    "CHATINTER_MIXED_TOOL_SELECTIONS",
    expire=6 * 60 * 60,
    max_items=512,
)


@dataclass(frozen=True, slots=True)
class MixedToolCatalog:
    skill_index: PluginSkillIndex
    known_commands: tuple[CommandToolSnapshot, ...]
    available_commands: tuple[CommandToolSnapshot, ...]
    initial_candidates: tuple[CommandCandidate, ...]
    knowledge_base: PluginKnowledgeBase
    session_id: str | None
    command_context: NativeCommandExecutionContext
    task_modes: dict[str, str] = field(default_factory=dict, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class MixedToolView:
    tools: dict[str, Any]
    command_candidate_text: str
    native_command_ids: tuple[str, ...]
    indexed_command_ids: tuple[str, ...]
    skill_tool_names: tuple[str, ...] = ()
    tool_priority_names: tuple[str, ...] = ()
    required_tool_names: tuple[str, ...] = ()
    native_tool_bindings: tuple[tuple[str, str], ...] = ()
    indexed_tool_bindings: tuple[tuple[str, str], ...] = ()
    initial_candidates: tuple[CommandCandidate, ...] = ()
    base_candidate_contexts: tuple[tuple[str, str], ...] = ()
    candidate_token_budget: int = 0
    schema_tokens: int = 0
    schema_omitted_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolSchemaSelection:
    tools: dict[str, Any]
    schema_tokens: int
    omitted_names: tuple[str, ...]


def build_mixed_tool_catalog(
    *,
    skill_index: PluginSkillIndex,
    known_commands: list[CommandToolSnapshot],
    available_commands: list[CommandToolSnapshot],
    initial_candidates: list[CommandCandidate],
    knowledge_base: PluginKnowledgeBase,
    session_id: str | None,
    command_context: NativeCommandExecutionContext,
) -> MixedToolCatalog:
    return MixedToolCatalog(
        skill_index=skill_index,
        known_commands=tuple(_stable_snapshots(known_commands)),
        available_commands=tuple(_stable_snapshots(available_commands)),
        initial_candidates=tuple(initial_candidates),
        knowledge_base=knowledge_base,
        session_id=session_id,
        command_context=command_context,
    )


def assemble_candidate_tool_view(
    catalog: MixedToolCatalog,
    *,
    adapter: ProviderCapabilityAdapter,
    candidate: HostModelCandidate,
    context_window_tokens: int,
    output_reserve_tokens: int,
    base_prompt_tokens: int,
    base_tools: dict[str, Any] | None = None,
) -> MixedToolView:
    profile = adapter.profile
    if not profile.supports_tools or adapter.max_tools <= 0:
        return MixedToolView(
            tools={},
            command_candidate_text="",
            native_command_ids=(),
            indexed_command_ids=(),
            skill_tool_names=(),
        )

    reserve_web = int(
        candidate_web_search_kind(
            candidate,
            scope="chat",
            has_client_tools=True,
        )
        is not None
    )
    client_capacity = max(adapter.max_tools - reserve_web, 0)
    base_capacity = min(client_capacity, len(base_tools or {}))
    base_items = sorted((base_tools or {}).items())[:base_capacity]
    plugin_capacity = max(client_capacity - len(base_items), 0)
    semantic = sorted(
        (
            snapshot
            for snapshot in catalog.available_commands
            if _is_semantic_snapshot(snapshot)
        ),
        key=_native_snapshot_sort_key,
    )
    semantic_ids = {
        normalize_message_text(snapshot.command_id) for snapshot in semantic
    }
    indexed_skill_modules = {
        _module_key(skill.plugin_module) for skill in catalog.skill_index.skills
    }
    nonsemantic_skill_count = len(
        {
            _module_key(snapshot.plugin_module)
            for snapshot in catalog.available_commands
            if _module_key(snapshot.plugin_module) in indexed_skill_modules
            and normalize_message_text(snapshot.command_id) not in semantic_ids
        }
    )
    selected_native = (
        semantic if nonsemantic_skill_count + len(semantic) <= plugin_capacity else []
    )
    native_ids = {
        normalize_message_text(snapshot.command_id) for snapshot in selected_native
    }
    native_semantic_names: dict[str, set[str]] = {}
    for snapshot in selected_native:
        semantic_name = normalize_message_text(
            str(snapshot.meta.get("semantic_tool_name") or "")
        ).casefold()
        if semantic_name:
            native_semantic_names.setdefault(
                _module_key(snapshot.plugin_module),
                set(),
            ).add(semantic_name)
    indexed_available = [
        snapshot
        for snapshot in catalog.available_commands
        if normalize_message_text(snapshot.command_id) not in native_ids
    ]
    dispatch_skills = _available_dispatch_skills(
        catalog.skill_index,
        indexed_available,
        known_commands=list(catalog.known_commands),
        native_semantic_names=native_semantic_names,
    )
    detail_token_budget = max(
        int(context_window_tokens)
        - max(int(output_reserve_tokens), 0)
        - max(int(base_prompt_tokens), 0)
        - _DETAIL_PROTOCOL_MARGIN_TOKENS,
        1,
    )
    ambiguity_token_budget = min(
        max(detail_token_budget // 2, 1),
        _AMBIGUITY_RESULT_TOKEN_LIMIT,
    )
    dispatch_tools = build_plugin_skill_dispatch_tools(
        skills=dispatch_skills,
        known_commands=list(catalog.known_commands),
        available_commands=indexed_available,
        knowledge_base=catalog.knowledge_base,
        session_id=catalog.session_id,
        command_context=catalog.command_context,
        task_modes=catalog.task_modes,
        ambiguity_token_budget=ambiguity_token_budget,
        result_char_budget=_adapter_tool_result_char_budget(adapter),
    )
    dispatch_capacity = max(plugin_capacity - len(selected_native), 0)
    dispatch_items = _select_dispatch_items(
        dispatch_tools,
        capacity=dispatch_capacity,
        initial_candidates=catalog.initial_candidates,
        session_id=catalog.session_id,
        selection_fingerprint=catalog.skill_index.fingerprint,
    )
    exposed_skill_ids = {
        normalize_message_text(command_id)
        for _name, tool in dispatch_items
        for command_id in tool.skill.command_ids
    }
    skill_tools_by_command_id = {
        normalize_message_text(command_id): name
        for name, tool in dispatch_items
        for command_id in tool.skill.command_ids
    }
    native_candidates = [_candidate_from_snapshot(item) for item in selected_native]
    native_tools = build_native_command_tools(
        native_candidates,
        execution_context=catalog.command_context,
    )
    combined = [
        *base_items,
        *((tool.binding.tool_name, tool) for tool in native_tools),
        *dispatch_items,
    ]
    tools = dict(sorted(combined, key=lambda item: item[0]))

    with_web = tools_for_web_candidate(
        tools,
        candidate=candidate,
        scope="chat",
    )
    final_tools = dict(with_web or {})
    base_names = tuple(name for name, _tool in base_items)
    native_names = tuple(tool.binding.tool_name for tool in native_tools)
    dispatch_names = tuple(name for name, _tool in dispatch_items)
    local_names = {*base_names, *native_names, *dispatch_names}
    web_names = tuple(name for name in final_tools if name not in local_names)
    tool_priority_names = _stable_unique(
        (*base_names, *native_names, *dispatch_names, *web_names)
    )
    candidate_token_budget = min(
        max(detail_token_budget - ambiguity_token_budget, 1),
        _CANDIDATE_CONTEXT_TOKEN_LIMIT,
    )
    initial_candidates = [
        item
        for item in catalog.initial_candidates
        if normalize_message_text(item.schema.command_id) in exposed_skill_ids
    ]
    base_candidate_contexts = tuple(
        (name, str(getattr(tool, "candidate_context", "") or "").strip())
        for name, tool in base_items
        if str(getattr(tool, "candidate_context", "") or "").strip()
    )
    command_candidate_text = _render_bounded_candidate_context(
        base_candidate_contexts=base_candidate_contexts,
        initial_candidates=tuple(initial_candidates),
        skill_tools_by_command_id=skill_tools_by_command_id,
        token_budget=candidate_token_budget,
    )
    return MixedToolView(
        tools=final_tools,
        command_candidate_text=command_candidate_text,
        native_command_ids=tuple(snapshot.command_id for snapshot in selected_native),
        indexed_command_ids=tuple(
            snapshot.command_id
            for snapshot in indexed_available
            if normalize_message_text(snapshot.command_id) in exposed_skill_ids
        ),
        skill_tool_names=tuple(name for name, _tool in dispatch_items),
        tool_priority_names=tool_priority_names,
        required_tool_names=base_names,
        native_tool_bindings=tuple(
            (snapshot.command_id, tool.binding.tool_name)
            for snapshot, tool in zip(selected_native, native_tools, strict=True)
        ),
        indexed_tool_bindings=tuple(
            (
                snapshot.command_id,
                skill_tools_by_command_id[normalize_message_text(snapshot.command_id)],
            )
            for snapshot in indexed_available
            if normalize_message_text(snapshot.command_id) in skill_tools_by_command_id
        ),
        initial_candidates=tuple(initial_candidates),
        base_candidate_contexts=base_candidate_contexts,
        candidate_token_budget=candidate_token_budget,
    )


async def bound_candidate_tool_view_schema(
    view: MixedToolView,
    *,
    token_budget: int,
) -> MixedToolView:
    try:
        selection = await select_tools_within_schema_budget(
            view.tools,
            token_budget=token_budget,
            priority_names=view.tool_priority_names,
            required_names=view.required_tool_names,
        )
    except ValueError:
        selection = await select_tools_within_schema_budget(
            view.tools,
            token_budget=token_budget,
            priority_names=view.tool_priority_names,
        )
    selected_names = set(selection.tools)
    skill_tool_names = tuple(
        name for name in view.skill_tool_names if name in selected_names
    )
    native_command_ids = tuple(
        command_id
        for command_id, tool_name in view.native_tool_bindings
        if tool_name in selected_names
    )
    indexed_command_ids = tuple(
        command_id
        for command_id, tool_name in view.indexed_tool_bindings
        if tool_name in selected_names
    )
    selected_skill_by_command = {
        command_id: tool_name
        for command_id, tool_name in view.indexed_tool_bindings
        if tool_name in selected_names
    }
    initial_candidates = tuple(
        item
        for item in view.initial_candidates
        if normalize_message_text(item.schema.command_id) in selected_skill_by_command
    )
    command_candidate_text = _render_bounded_candidate_context(
        base_candidate_contexts=tuple(
            (tool_name, text)
            for tool_name, text in view.base_candidate_contexts
            if tool_name in selected_names
        ),
        initial_candidates=initial_candidates,
        skill_tools_by_command_id=selected_skill_by_command,
        token_budget=min(
            max(view.candidate_token_budget, 0),
            max(int(token_budget) - selection.schema_tokens, 0),
        ),
    )
    if selection.omitted_names:
        _n = len(selection.omitted_names)
        _preview = ", ".join(selection.omitted_names[:8])
        if _n > 8:
            _preview += f" (+{_n - 8} more)"
        _omit_note = (
            f"\n[schema_budget_exceeded: {_n} tool(s) not exposed"
            f" — {_preview}]"
        )
        command_candidate_text = (command_candidate_text or "") + _omit_note
    return replace(
        view,
        tools=selection.tools,
        command_candidate_text=command_candidate_text,
        native_command_ids=native_command_ids,
        indexed_command_ids=indexed_command_ids,
        skill_tool_names=skill_tool_names,
        initial_candidates=initial_candidates,
        schema_tokens=selection.schema_tokens,
        schema_omitted_names=selection.omitted_names,
    )


async def select_tools_within_schema_budget(
    tools: dict[str, Any] | None,
    *,
    token_budget: int,
    priority_names: tuple[str, ...] = (),
    required_names: tuple[str, ...] = (),
) -> ToolSchemaSelection:
    available = dict(tools or {})
    if not available:
        return ToolSchemaSelection({}, 0, ())

    required = tuple(name for name in required_names if name in available)
    ordered = _stable_unique(
        (
            *required,
            *(name for name in priority_names if name in available),
            *sorted(available),
        )
    )
    selected_names: list[str] = []
    schema_payloads = {
        name: await _tool_schema_payload(name, available[name]) for name in ordered
    }
    schema_tokens = 0
    budget = max(int(token_budget), 0)
    for index, name in enumerate(ordered):
        trial_names = (*selected_names, name)
        trial_tokens = _schema_payload_tokens(
            [schema_payloads[item] for item in trial_names]
        )
        if trial_tokens <= budget:
            selected_names.append(name)
            schema_tokens = trial_tokens
            continue
        if name in required:
            raise ValueError(
                "required tool schema exceeds available prompt budget: "
                f"tool={name} required={trial_tokens} available={budget}"
            )
        omitted = tuple(ordered[index:])
        return ToolSchemaSelection(
            tools={name: available[name] for name in selected_names},
            schema_tokens=schema_tokens,
            omitted_names=omitted,
        )

    return ToolSchemaSelection(
        tools={name: available[name] for name in selected_names},
        schema_tokens=schema_tokens,
        omitted_names=(),
    )


async def tool_schema_tokens(tools: dict[str, Any] | None) -> int:
    schemas = [
        await _tool_schema_payload(name, (tools or {})[name])
        for name in sorted(tools or {})
    ]
    return _schema_payload_tokens(schemas)


async def _tool_schema_payload(name: str, tool: Any) -> dict[str, Any]:
    definition = await tool.get_definition()
    payload = (
        definition.model_dump(mode="json")
        if hasattr(definition, "model_dump")
        else {
            "name": str(getattr(definition, "name", name) or name),
            "description": str(getattr(definition, "description", "") or ""),
            "parameters": getattr(definition, "parameters", {}) or {},
        }
    )
    return {"name": name, "schema": payload}


def _schema_payload_tokens(schemas: list[dict[str, Any]]) -> int:
    if not schemas:
        return 0
    return estimate_text_tokens(
        json.dumps(
            schemas,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        )
    )


def _render_bounded_candidate_context(
    *,
    base_candidate_contexts: tuple[tuple[str, str], ...],
    initial_candidates: tuple[CommandCandidate, ...] = (),
    skill_tools_by_command_id: dict[str, str] | None = None,
    token_budget: int,
) -> str:
    budget = max(int(token_budget), 0)
    if budget <= 0:
        return ""
    sections: list[str] = []
    for _tool_name, text in base_candidate_contexts:
        candidate = "\n".join((*sections, text))
        if estimate_text_tokens(candidate) > budget:
            break
        sections.append(text)
    used_tokens = estimate_text_tokens("\n".join(sections)) if sections else 0
    separator_tokens = estimate_text_tokens("\n") if sections else 0
    command_budget = max(budget - used_tokens - separator_tokens, 0)
    command_context = render_command_candidate_context(
        list(initial_candidates),
        token_budget=command_budget,
        skill_tools_by_command_id=skill_tools_by_command_id,
    )
    if command_context:
        candidate = "\n".join((*sections, command_context))
        if estimate_text_tokens(candidate) <= budget:
            sections.append(command_context)
    return "\n".join(sections)


def _adapter_tool_result_char_budget(adapter: ProviderCapabilityAdapter) -> int:
    direct = getattr(adapter, "max_tool_result_chars", None)
    if direct is not None:
        return max(int(direct or 0), 1)
    profile = getattr(adapter, "profile", None)
    protocol = getattr(profile, "protocol", None)
    mcp = getattr(protocol, "mcp", None) or getattr(profile, "mcp", None)
    return max(int(getattr(mcp, "max_result_chars", 12_000) or 12_000), 1)


def _stable_snapshots(
    snapshots: list[CommandToolSnapshot],
) -> list[CommandToolSnapshot]:
    by_id: dict[str, CommandToolSnapshot] = {}
    for snapshot in snapshots:
        command_id = normalize_message_text(snapshot.command_id)
        if command_id and command_id not in by_id:
            by_id[command_id] = snapshot
    return sorted(by_id.values(), key=_native_snapshot_sort_key)


def _available_dispatch_skills(
    skill_index: PluginSkillIndex,
    available_commands: list[CommandToolSnapshot],
    *,
    known_commands: list[CommandToolSnapshot],
    native_semantic_names: dict[str, set[str]],
) -> tuple[PluginSkill, ...]:
    available_by_module: dict[str, dict[str, CommandToolSnapshot]] = {}
    for snapshot in available_commands:
        module_key = _module_key(snapshot.plugin_module)
        command_id = normalize_message_text(snapshot.command_id)
        if module_key and command_id:
            available_by_module.setdefault(module_key, {})[command_id] = snapshot

    known_by_module: dict[str, dict[str, CommandToolSnapshot]] = {}
    for snapshot in known_commands:
        module_key = _module_key(snapshot.plugin_module)
        command_id = normalize_message_text(snapshot.command_id)
        if module_key and command_id:
            known_by_module.setdefault(module_key, {})[command_id] = snapshot

    projected: list[PluginSkill] = []
    for skill in skill_index.skills:
        module_key = _module_key(skill.plugin_module)
        snapshots_by_id = available_by_module.get(module_key, {})
        known_by_id = known_by_module.get(module_key, {})
        # Expose only currently-available commands on the skill; known-but-
        # unavailable ids stay reachable inside the dispatch tool via its
        # module-scoped known lookup (unavailable_in_context branch).
        command_ids = tuple(
            command_id
            for command_id in skill.command_ids
            if normalize_message_text(command_id) in snapshots_by_id
            and normalize_message_text(command_id) in known_by_id
        )
        if not command_ids:
            continue
        # Metadata (input_types, output_modes, etc.) is derived from available
        # snapshots only — these describe what the model can execute right now.
        snapshots = [
            snapshots_by_id[normalize_message_text(command_id)]
            for command_id in command_ids
        ]
        semantic_names = {
            normalize_message_text(
                str(snapshot.meta.get("semantic_tool_name") or "")
            ).casefold()
            for snapshot in snapshots
            if isinstance(snapshot.meta, dict)
            and normalize_message_text(
                str(snapshot.meta.get("semantic_tool_name") or "")
            )
        }
        semantic_names.difference_update(native_semantic_names.get(module_key, set()))
        projected.append(
            replace(
                skill,
                command_ids=command_ids,
                command_count=len(command_ids),
                semantic_tools=tuple(
                    contract
                    for contract in skill.semantic_tools
                    if normalize_message_text(contract.name).casefold()
                    in semantic_names
                ),
                input_types=_skill_input_types(snapshots),
                output_modes=_stable_values(
                    snapshot.output_mode for snapshot in snapshots
                ),
                side_effects=_stable_values(
                    snapshot.side_effect for snapshot in snapshots
                ),
            )
        )
    return tuple(projected)


def _skill_input_types(
    snapshots: list[CommandToolSnapshot],
) -> tuple[str, ...]:
    values: list[str] = []
    for snapshot in snapshots:
        values.extend(snapshot.input_requirements)
        values.extend(slot.type for slot in snapshot.slots)
        values.extend(key for key, required in snapshot.requires.items() if required)
    return _stable_values(values)


def _stable_values(values: Any) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                normalized
                for value in values
                if (normalized := normalize_message_text(str(value or "")))
            },
            key=lambda value: (value.casefold(), value),
        )
    )


def _select_dispatch_items(
    dispatch_tools: dict[str, Any],
    *,
    capacity: int,
    initial_candidates: tuple[CommandCandidate, ...],
    session_id: str | None,
    selection_fingerprint: str,
) -> list[tuple[str, Any]]:
    stable_items = list(dispatch_tools.items())
    if capacity <= 0:
        return []
    if len(stable_items) <= capacity:
        return stable_items

    tool_by_command_id = {
        normalize_message_text(command_id): name
        for name, tool in stable_items
        for command_id in tool.skill.command_ids
    }
    related_names: list[str] = []
    for candidate in initial_candidates:
        name = tool_by_command_id.get(
            normalize_message_text(candidate.schema.command_id)
        )
        if name and name not in related_names:
            related_names.append(name)

    cache_key = _dispatch_selection_key(
        session_id=session_id,
        selection_fingerprint=selection_fingerprint,
        tool_names=tuple(name for name, _tool in stable_items),
    )
    all_names = {name for name, _tool in stable_items}
    # Retrieve the full priority order from cache (not capacity-trimmed).
    # The cache stores ALL tools in priority order so capacity changes between
    # turns don't invalidate the selection and cause instability.
    cached_order: list[str] = [
        name
        for name in (_DISPATCH_SELECTIONS.get(cache_key, ()) if cache_key else ())
        if name in all_names
    ]
    related_set = set(related_names)
    cached_order = [
        *related_names,
        *(name for name in cached_order if name not in related_set),
    ]
    # Fill any remaining names in stable alphabetical order.
    for name, _tool in stable_items:
        if name not in cached_order:
            cached_order.append(name)
    # Persist the full ordering so capacity changes in subsequent turns still
    # get a stable prefix.
    if cache_key:
        _DISPATCH_SELECTIONS[cache_key] = tuple(cached_order)
    selected_names = set(cached_order[:capacity])
    return [(name, tool) for name, tool in stable_items if name in selected_names]


def _dispatch_selection_key(
    *,
    session_id: str | None,
    selection_fingerprint: str,
    tool_names: tuple[str, ...],
) -> str:
    normalized_session = normalize_message_text(str(session_id or ""))
    if not normalized_session:
        return ""
    digest = hashlib.blake2s(digest_size=16)
    for value in (
        normalized_session,
        selection_fingerprint,
        *tool_names,
    ):
        digest.update(value.encode("utf-8", "ignore"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _native_snapshot_sort_key(
    snapshot: CommandToolSnapshot,
) -> tuple[str, str, str]:
    semantic_name = ""
    if isinstance(snapshot.meta, dict):
        semantic_name = normalize_message_text(
            str(snapshot.meta.get("semantic_tool_name") or "")
        )
    return (
        semantic_name.casefold(),
        _module_key(snapshot.plugin_module),
        normalize_message_text(snapshot.command_id).casefold(),
    )


def _is_semantic_snapshot(snapshot: CommandToolSnapshot) -> bool:
    return bool(
        isinstance(snapshot.meta, dict)
        and snapshot.meta.get("semantic_tool_name")
        and isinstance(snapshot.meta.get("semantic_contract"), dict)
    )


def _module_key(value: str) -> str:
    return normalize_message_text(value).casefold()


def _stable_unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


__all__ = [
    "MixedToolCatalog",
    "MixedToolView",
    "ToolSchemaSelection",
    "assemble_candidate_tool_view",
    "bound_candidate_tool_view_schema",
    "build_mixed_tool_catalog",
    "select_tools_within_schema_budget",
    "tool_schema_tokens",
]
