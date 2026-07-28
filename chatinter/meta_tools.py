"""Fixed meta tools exposed to the unified chat agent.

Only these three definitions ever reach the model, so the request tools array
stays byte-stable across turns — a prompt-cache prefix requirement.  Dynamic
data (candidate commands, slot schemas, execution results) flows through tool
RESULTS, never through tool definitions.

No plugin-specific wording may appear here: cards are rendered purely from
command snapshot metadata.
"""

from __future__ import annotations

from typing import Any

from .command_index import CommandCandidate, _schema_from_tool_snapshot
from .llm_compat import ToolDefinition, ToolExecutable, ToolResult
from .models.pydantic_models import CommandToolSnapshot
from .native_command_tools import NativeCommandToolBinding
from .native_executor import NativeCommandExecutionContext
from .route_text import normalize_message_text
from .task_frame import PAYLOAD_HINT_FIELD, TARGET_HINT_FIELD, TASK_TEXT_FIELD
from .tool_retriever import CommandToolRetriever

SEARCH_PLUGINS_TOOL_NAME = "search_plugins"
GET_COMMAND_DETAILS_TOOL_NAME = "get_command_details"
CALL_PLUGIN_TOOL_NAME = "call_plugin"

_SEARCH_LIMIT_DEFAULT = 8
_SEARCH_LIMIT_MAX = 16
_DETAIL_LIMIT = 6
_SUGGESTION_LIMIT = 5
_USAGE_CLIP = 300


class SearchPluginsTool:
    """Fallback lookup over the full command inventory."""

    def __init__(self, retriever: CommandToolRetriever):
        self._retriever = retriever

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=SEARCH_PLUGINS_TOOL_NAME,
            description=(
                "按关键词检索可用插件命令。当 <plugin_catalog> 里没有合适条目、"
                "或目标插件被标注为已折叠时使用。返回候选命令的简要卡片。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索关键词（功能、命令名或用户的说法）",
                    },
                    "limit": {
                        "type": "integer",
                        "description": f"返回数量上限，默认 {_SEARCH_LIMIT_DEFAULT}",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        del context
        query = normalize_message_text(str(kwargs.get("query", "") or ""))
        limit = _coerce_limit(kwargs.get("limit"))
        if not query:
            return ToolResult(
                output={"status": "error", "error": "query 不能为空"},
                display_content="search_plugins 缺少 query",
                is_error=True,
            )
        result = self._retriever.retrieve(query, limit=limit)
        cards = [
            _summary_card(candidate.tool)
            for candidate in result.candidates[:limit]
            if candidate.tool is not None
        ]
        return ToolResult(
            output={
                "status": "ok",
                "query": query,
                "total_commands": result.total_commands,
                "candidates": cards,
                "note": (
                    "检索结果按词面匹配排序，仅供参考；"
                    "确定目标后用 get_command_details 获取参数定义。"
                ),
            },
            display_content=f"search_plugins: {len(cards)} 个候选",
        )


class GetCommandDetailsTool:
    """Full parameter schema lookup for chosen commands."""

    def __init__(self, snapshots_by_id: dict[str, CommandToolSnapshot]):
        self._by_id = snapshots_by_id

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=GET_COMMAND_DETAILS_TOOL_NAME,
            description=(
                "获取命令的完整参数定义（槽位、类型、示例、前置条件）。"
                "调用 call_plugin 之前先用它确认参数。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "目录或检索结果里的 command_id 列表",
                    },
                },
                "required": ["command_ids"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        del context
        raw_ids = kwargs.get("command_ids")
        if isinstance(raw_ids, str):
            raw_ids = [raw_ids]
        if not isinstance(raw_ids, list) or not raw_ids:
            return ToolResult(
                output={"status": "error", "error": "command_ids 不能为空"},
                display_content="get_command_details 缺少 command_ids",
                is_error=True,
            )
        details: list[dict[str, Any]] = []
        unknown: list[dict[str, Any]] = []
        for raw_id in raw_ids[:_DETAIL_LIMIT]:
            command_id = normalize_message_text(str(raw_id or ""))
            snapshot = self._by_id.get(command_id)
            if snapshot is None:
                unknown.append(
                    {
                        "command_id": command_id,
                        "suggestions": _suggest_command_ids(self._by_id, command_id),
                    }
                )
            else:
                details.append(_detail_card(snapshot))
        return ToolResult(
            output={
                "status": "ok" if details else "not_found",
                "commands": details,
                "unknown": unknown,
            },
            display_content=(
                f"get_command_details: {len(details)} 条，未知 {len(unknown)} 条"
            ),
        )


class CallPluginTool:
    """Validated execution of one plugin command through the native route."""

    def __init__(
        self,
        snapshots_by_id: dict[str, CommandToolSnapshot],
        command_context: NativeCommandExecutionContext,
    ):
        self._by_id = snapshots_by_id
        self._command_context = command_context

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=CALL_PLUGIN_TOOL_NAME,
            description=(
                "执行一条插件命令。每次调用执行一个任务；"
                "用户提出多个任务时分多次调用。"
                "执行结果里 messages_sent 非空表示插件输出已直接发给用户。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command_id": {
                        "type": "string",
                        "description": "目录或详情里的 command_id",
                    },
                    "task_text": {
                        "type": "string",
                        "description": "该任务对应的用户原话片段（必填）",
                    },
                    "slots": {
                        "type": "object",
                        "description": "命令参数槽位，键为槽位名",
                        "additionalProperties": True,
                    },
                    "target_hint": {
                        "type": "string",
                        "description": "目标用户提示（昵称或@），仅当命令需要目标时填",
                    },
                    "payload_hint": {
                        "type": "string",
                        "description": "补充内容提示，仅当命令需要额外内容时填",
                    },
                },
                "required": ["command_id", "task_text"],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        del context
        command_id = normalize_message_text(str(kwargs.get("command_id", "") or ""))
        snapshot = self._by_id.get(command_id)
        if snapshot is None:
            return ToolResult(
                output={
                    "status": "unknown_command",
                    "error": f"command_id 不存在：{command_id or '<空>'}",
                    "suggestions": _suggest_command_ids(self._by_id, command_id),
                },
                display_content=f"call_plugin: 未知命令 {command_id}",
            )
        candidate = _candidate_from_snapshot(snapshot)
        binding = NativeCommandToolBinding(
            tool_name=CALL_PLUGIN_TOOL_NAME,
            candidate=candidate,
        )
        raw_slots = _build_raw_slots(kwargs)
        if all(
            item.schema.command_id != command_id
            for item in self._command_context.candidates
        ):
            self._command_context.candidates.append(candidate)
        result = await self._command_context.execute_tool(
            binding=binding,
            raw_slots=raw_slots,
        )
        output = (
            dict(result.output)
            if isinstance(result.output, dict)
            else {"value": result.output}
        )
        if not output.get("ok", False):
            output.setdefault("command_schema", _detail_card(snapshot))
            return ToolResult(
                output=output,
                display_content=result.display_content,
            )
        return result


def build_meta_tools(
    *,
    command_tools: list[CommandToolSnapshot],
    retriever: CommandToolRetriever,
    command_context: NativeCommandExecutionContext,
) -> dict[str, ToolExecutable]:
    snapshots_by_id = snapshots_by_command_id(command_tools)
    return {
        SEARCH_PLUGINS_TOOL_NAME: SearchPluginsTool(retriever),
        GET_COMMAND_DETAILS_TOOL_NAME: GetCommandDetailsTool(snapshots_by_id),
        CALL_PLUGIN_TOOL_NAME: CallPluginTool(snapshots_by_id, command_context),
    }


def snapshots_by_command_id(
    command_tools: list[CommandToolSnapshot],
) -> dict[str, CommandToolSnapshot]:
    result: dict[str, CommandToolSnapshot] = {}
    for tool in command_tools:
        command_id = normalize_message_text(tool.command_id)
        if command_id and command_id not in result:
            result[command_id] = tool
    return result


def _build_raw_slots(kwargs: dict[str, Any]) -> dict[str, Any]:
    slots = kwargs.get("slots")
    raw_slots: dict[str, Any] = dict(slots) if isinstance(slots, dict) else {}
    for key, value in kwargs.items():
        if key in {"command_id", "slots", TASK_TEXT_FIELD, TARGET_HINT_FIELD,
                   PAYLOAD_HINT_FIELD}:
            continue
        raw_slots.setdefault(key, value)
    raw_slots[TASK_TEXT_FIELD] = str(kwargs.get(TASK_TEXT_FIELD, "") or "")
    target_hint = str(kwargs.get(TARGET_HINT_FIELD, "") or "")
    payload_hint = str(kwargs.get(PAYLOAD_HINT_FIELD, "") or "")
    if target_hint:
        raw_slots[TARGET_HINT_FIELD] = target_hint
    if payload_hint:
        raw_slots[PAYLOAD_HINT_FIELD] = payload_hint
    return raw_slots


def _candidate_from_snapshot(snapshot: CommandToolSnapshot) -> CommandCandidate:
    return CommandCandidate(
        plugin_module=snapshot.plugin_module,
        plugin_name=snapshot.plugin_name,
        schema=_schema_from_tool_snapshot(snapshot),
        score=0.0,
        reason="unified_call_plugin",
        tool=snapshot,
    )


def _summary_card(snapshot: CommandToolSnapshot) -> dict[str, Any]:
    required_slots = [
        f"{slot.name}:{slot.type}"
        for slot in snapshot.slots
        if slot.required
    ]
    card: dict[str, Any] = {
        "command_id": snapshot.command_id,
        "plugin": snapshot.plugin_name,
        "head": snapshot.head,
        "aliases": list(snapshot.aliases[:4]),
        "description": normalize_message_text(snapshot.description)[:120],
    }
    if required_slots:
        card["required_slots"] = required_slots
    examples = [text for text in snapshot.examples[:2] if normalize_message_text(text)]
    if examples:
        card["examples"] = examples
    marks = _requirement_summary(snapshot)
    if marks:
        card["requires"] = marks
    return card


def _detail_card(snapshot: CommandToolSnapshot) -> dict[str, Any]:
    card: dict[str, Any] = {
        "command_id": snapshot.command_id,
        "plugin": snapshot.plugin_name,
        "head": snapshot.head,
        "aliases": list(snapshot.aliases[:6]),
        "description": normalize_message_text(snapshot.description)[:200],
        "slots": [
            {
                "name": slot.name,
                "type": slot.type,
                "required": slot.required,
                **({"aliases": list(slot.aliases[:3])} if slot.aliases else {}),
                **(
                    {"description": normalize_message_text(slot.description)[:80]}
                    if slot.description
                    else {}
                ),
                **({"choices": list(slot.choices[:12])} if slot.choices else {}),
                **({"default": slot.default} if slot.default not in (None, "") else {}),
            }
            for slot in snapshot.slots
        ],
        "render": snapshot.render or snapshot.head,
    }
    usage = normalize_message_text(str(snapshot.usage or ""))
    if usage:
        card["usage"] = usage[:_USAGE_CLIP]
    examples = [text for text in snapshot.examples[:3] if normalize_message_text(text)]
    if examples:
        card["examples"] = examples
    marks = _requirement_summary(snapshot)
    if marks:
        card["requires"] = marks
    return card


def _requirement_summary(snapshot: CommandToolSnapshot) -> list[str]:
    requires = dict(snapshot.requires or {})
    marks: list[str] = []
    if requires.get("image"):
        marks.append("需图片")
    if requires.get("reply"):
        marks.append("需回复")
    if requires.get("at") or snapshot.target_requirement == "required":
        marks.append("需指定目标")
    if snapshot.actor_scope == "self_only":
        marks.append("仅对自己")
    if requires.get("text"):
        marks.append("需文本参数")
    return marks


def _suggest_command_ids(
    snapshots_by_id: dict[str, CommandToolSnapshot],
    query: str,
) -> list[str]:
    text = normalize_message_text(query).casefold()
    if not text:
        return []
    scored: list[tuple[float, str]] = []
    for command_id, snapshot in snapshots_by_id.items():
        haystacks = [
            command_id.casefold(),
            normalize_message_text(snapshot.head).casefold(),
            *(normalize_message_text(alias).casefold() for alias in snapshot.aliases),
        ]
        best = max(
            (_text_similarity(text, haystack) for haystack in haystacks if haystack),
            default=0.0,
        )
        if best >= 0.3:
            scored.append((best, command_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [command_id for _, command_id in scored[:_SUGGESTION_LIMIT]]


def _text_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if left in right or right in left:
        return 0.8
    left_grams = _bigrams(left)
    right_grams = _bigrams(right)
    if not left_grams or not right_grams:
        return 0.0
    return 2 * len(left_grams & right_grams) / (len(left_grams) + len(right_grams))


def _bigrams(text: str) -> set[str]:
    return {text[i : i + 2] for i in range(len(text) - 1)}


__all__ = [
    "CALL_PLUGIN_TOOL_NAME",
    "GET_COMMAND_DETAILS_TOOL_NAME",
    "SEARCH_PLUGINS_TOOL_NAME",
    "CallPluginTool",
    "GetCommandDetailsTool",
    "SearchPluginsTool",
    "build_meta_tools",
    "snapshots_by_command_id",
]
