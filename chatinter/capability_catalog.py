"""Compact capability catalog for the unified ChatInter agent.

The catalog renders every permission-visible plugin command into one
deterministic text block placed in the system prompt.  Providers cache prompt
prefixes byte-for-byte, so the same command inventory must always render to
the identical string: plugins and commands are sorted deterministically and no
wall-clock or per-turn data may appear here.

Catalog lines are derived exclusively from command snapshot metadata.  Plugin
or command specific wording must never be added in code.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib

from .models.pydantic_models import CommandToolSnapshot
from .route_text import normalize_message_text

_DESCRIPTION_CLIP = 64
_ALIAS_LIMIT = 3
_FAMILY_COLLAPSE_THRESHOLD = 12
_COLLAPSED_SAMPLE_HEADS = 3
_CATALOG_CACHE_LIMIT = 8

_CATALOG_CACHE: OrderedDict[str, "CapabilityCatalog"] = OrderedDict()


@dataclass(frozen=True)
class CapabilityCatalog:
    """Byte-stable catalog text plus bookkeeping for metrics."""

    text: str
    fingerprint: str
    command_count: int
    plugin_count: int
    collapsed_modules: tuple[str, ...]


def build_capability_catalog(
    command_tools: list[CommandToolSnapshot],
) -> CapabilityCatalog:
    fingerprint = _catalog_fingerprint(command_tools)
    cached = _CATALOG_CACHE.get(fingerprint)
    if cached is not None:
        _CATALOG_CACHE.move_to_end(fingerprint)
        return cached

    grouped: dict[str, list[CommandToolSnapshot]] = {}
    plugin_names: dict[str, str] = {}
    for tool in command_tools:
        module = normalize_message_text(tool.plugin_module) or "unknown"
        grouped.setdefault(module, []).append(tool)
        if module not in plugin_names:
            plugin_names[module] = normalize_message_text(tool.plugin_name) or module

    sections: list[str] = []
    collapsed: list[str] = []
    command_count = 0
    for module in sorted(grouped):
        tools = sorted(grouped[module], key=lambda item: item.command_id)
        command_count += len(tools)
        if len(tools) > _FAMILY_COLLAPSE_THRESHOLD:
            collapsed.append(module)
            sections.append(_collapsed_section(module, plugin_names[module], tools))
        else:
            sections.append(_plugin_section(module, plugin_names[module], tools))

    catalog = CapabilityCatalog(
        text="\n".join(sections),
        fingerprint=fingerprint,
        command_count=command_count,
        plugin_count=len(grouped),
        collapsed_modules=tuple(collapsed),
    )
    _CATALOG_CACHE[fingerprint] = catalog
    while len(_CATALOG_CACHE) > _CATALOG_CACHE_LIMIT:
        _CATALOG_CACHE.popitem(last=False)
    return catalog


def _plugin_section(
    module: str,
    plugin_name: str,
    tools: list[CommandToolSnapshot],
) -> str:
    lines = [f"## {plugin_name}"]
    lines.extend(_command_line(tool) for tool in tools)
    return "\n".join(lines)


def _collapsed_section(
    module: str,
    plugin_name: str,
    tools: list[CommandToolSnapshot],
) -> str:
    heads: list[str] = []
    for tool in tools:
        head = normalize_message_text(tool.head)
        if head and head not in heads:
            heads.append(head)
        if len(heads) >= _COLLAPSED_SAMPLE_HEADS:
            break
    sample = "、".join(heads)
    return "\n".join(
        (
            f"## {plugin_name}（目录已折叠）",
            f"- 共 {len(tools)} 条命令（如 {sample} 等），"
            "用 search_plugins 检索本插件的具体命令",
        )
    )


def _command_line(tool: CommandToolSnapshot) -> str:
    head = normalize_message_text(tool.head) or tool.command_id
    aliases = _catalog_aliases(tool, head)
    head_part = f"{head}({'/'.join(aliases)})" if aliases else head
    description = _clip(
        normalize_message_text(tool.description),
        _DESCRIPTION_CLIP,
    )
    marks = _requirement_marks(tool)
    parts = [f"- {tool.command_id} | {head_part}"]
    if description and description != head:
        parts.append(description)
    line = " | ".join(parts)
    return f"{line} {marks}" if marks else line


def _catalog_aliases(tool: CommandToolSnapshot, head: str) -> list[str]:
    aliases: list[str] = []
    for alias in tool.aliases:
        text = normalize_message_text(alias)
        if text and text != head and text not in aliases:
            aliases.append(text)
        if len(aliases) >= _ALIAS_LIMIT:
            break
    return aliases


def _requirement_marks(tool: CommandToolSnapshot) -> str:
    requires = dict(tool.requires or {})
    marks: list[str] = []
    if requires.get("image"):
        marks.append("需图片")
    if requires.get("reply"):
        marks.append("需回复")
    if requires.get("at") or tool.target_requirement == "required":
        marks.append("需指定目标")
    if tool.actor_scope == "self_only":
        marks.append("仅对自己")
    if requires.get("text"):
        marks.append("需文本参数")
    return f"[{'/'.join(marks)}]" if marks else ""


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _catalog_fingerprint(command_tools: list[CommandToolSnapshot]) -> str:
    digest = hashlib.blake2s(digest_size=16)
    for tool in sorted(command_tools, key=lambda item: item.command_id):
        for value in (
            tool.command_id,
            tool.plugin_module,
            tool.plugin_name,
            tool.head,
            "/".join(tool.aliases),
            tool.description,
            tool.actor_scope,
            tool.target_requirement,
            str(sorted((tool.requires or {}).items())),
        ):
            digest.update(str(value).encode("utf-8", "ignore"))
            digest.update(b"\x00")
        digest.update(b"\x01")
    return digest.hexdigest()


__all__ = [
    "CapabilityCatalog",
    "build_capability_catalog",
]
