"""Stable plugin-scoped dispatch tools for the ChatInter mixed-chat agent."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any
import unicodedata

from .command_index import CommandCandidate, _schema_from_tool_snapshot
from .llm_compat import ToolDefinition, ToolExecutable, ToolResult
from .models.pydantic_models import CommandToolSnapshot, PluginKnowledgeBase
from .native_command_tools import NativeCommandToolBinding
from .native_executor import NativeCommandExecutionContext
from .plugin_skill_index import PluginSkill
from .route_text import normalize_message_text
from .task_frame import (
    PAYLOAD_HINT_FIELD,
    TARGET_HINT_FIELD,
    TARGET_REF_FIELD,
    TASK_TEXT_FIELD,
)
from .token_compat import estimate_text_tokens
from .tool_cards import project_command_card
from .tool_retriever import CommandToolRetriever

_TOOL_NAME_PREFIX = "ci_skill_"
_TOOL_NAME_SLUG_LIMIT = 40
_TOOL_NAME_PART_PATTERN = re.compile(r"[^a-z0-9_]+")
_ALIASES_TEXT_LIMIT = 480
_PRECAUTIONS_TEXT_LIMIT = 720
_CAPABILITY_VALUES_TEXT_LIMIT = 360
_SEMANTIC_CONTRACTS_TEXT_LIMIT = 1600
_TASK_MODE_FIELD = "task_mode"
_TASK_MODES = {"action", "query", "help"}
# 词法零召回后由 embedding 通道收窄的候选上限；再由既有 token/字符预算裁剪。
_VECTOR_RECALL_TOP_N = 8


@dataclass(frozen=True)
class _SelectionPlan:
    """`_selection_result` 的中间结果：候选集合与降级标记。

    拆出这一层是为了让同步渲染路径与 async 向量通道路径共享同一份候选计算，
    从而不必把 `CommandToolRetriever.retrieve` 改成 async。
    """

    snapshots: list[CommandToolSnapshot]
    reason: str
    full_listing_fallback: bool
    recall: str = ""


class PluginSkillDispatchTool:
    chatinter_plugin_tool_kind = "skill_dispatch"

    def __init__(
        self,
        skill: PluginSkill,
        *,
        known_commands: Sequence[CommandToolSnapshot],
        available_commands: Sequence[CommandToolSnapshot],
        knowledge_base: PluginKnowledgeBase,
        session_id: str | None,
        command_context: NativeCommandExecutionContext,
        task_modes: dict[str, str] | None = None,
        ambiguity_token_budget: int | None = None,
        result_char_budget: int | None = None,
    ) -> None:
        self.skill = skill
        self.name = skill_dispatch_tool_name(skill.plugin_module)
        self._skill_command_keys = frozenset(
            _command_key(command_id) for command_id in skill.command_ids
        )
        # Module-scoped known lookup: keeps known-but-unavailable commands
        # reachable for the unavailable_in_context diagnostic branch even
        # though skill.command_ids only exposes available commands.
        self._known_by_key = _skill_snapshots_by_key(
            skill,
            known_commands,
            restrict_to_skill=False,
        )
        self._available_by_key = {
            key: snapshot
            for key, snapshot in _skill_snapshots_by_key(
                skill,
                available_commands,
            ).items()
            if key in self._known_by_key
        }
        self._command_context = command_context
        self._ambiguity_token_budget = (
            max(int(ambiguity_token_budget), 1)
            if ambiguity_token_budget is not None
            else None
        )
        self._result_char_budget = (
            max(int(result_char_budget), 1) if result_char_budget is not None else None
        )
        self._task_modes = task_modes if task_modes is not None else {}
        self._retriever = CommandToolRetriever(
            _plugin_knowledge_base(skill, knowledge_base),
            session_id=session_id,
            tools=list(self._available_by_key.values()),
        )

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=_tool_description(self.skill),
            parameters={
                "type": "object",
                "properties": {
                    TASK_TEXT_FIELD: {
                        "type": "string",
                        "minLength": 1,
                        "description": "当前工具调用对应的用户原话或任务片段",
                    },
                    _TASK_MODE_FIELD: {
                        "type": "string",
                        "enum": ["action", "query", "help"],
                        "description": (
                            "用户原始目标：action=执行、生成、发送或变换；"
                            "query=查询状态、记录或搜索内容；help=查看能力、列表或用法。"
                            "不得根据候选工具改变原始目标"
                        ),
                    },
                    "command_id": {
                        "type": ["string", "null"],
                        "description": (
                            "仅在候选中已知具体命令时填写该 Skill 内的 command_id；"
                            "尚未选择时填写 null，由 Skill 内部检索"
                        ),
                    },
                    TARGET_REF_FIELD: {
                        "type": ["string", "null"],
                        "description": (
                            "可选的受限操作目标。仅在当前请求明确承接对话关系时，"
                            "填写 <relevant_people> 中已有的 target_ref；"
                            "不得填写昵称或用户 ID，否则填写 null"
                        ),
                    },
                    "slots": {
                        "type": "object",
                        "description": "命令参数槽位，键为槽位名",
                        "additionalProperties": True,
                    },
                },
                "required": [TASK_TEXT_FIELD, _TASK_MODE_FIELD],
                "additionalProperties": False,
            },
        )

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        del context
        task_text = normalize_message_text(str(kwargs.get(TASK_TEXT_FIELD, "") or ""))
        if not task_text:
            return self._not_executed(
                "invalid",
                error="task_text 不能为空",
                reason="missing_task_text",
            )
        task_mode_requested = normalize_message_text(
            str(kwargs.get(_TASK_MODE_FIELD, "") or "")
        ).casefold()
        task_mode = self._task_mode(task_text, task_mode_requested)
        if task_mode is None:
            return self._not_executed(
                "invalid_tool_arguments",
                error="task_mode 必须是 action、query 或 help",
                reason="invalid_task_mode",
            )
        if (
            task_mode_requested in _TASK_MODES
            and task_mode
            and task_mode_requested != task_mode
        ):
            # 同一 task_text 的 task_mode 以首次声明为准（用户原始目标不变）；
            # 换 mode 属于迁就候选的改写，按原 mode 返回候选并转普通聊天。
            return self._selection_result(
                task_text=task_text,
                task_mode=task_mode,
                reason="task_mode_mismatch",
                response_policy="chat_without_clarification",
            )

        command_id = normalize_message_text(str(kwargs.get("command_id", "") or ""))
        # 防御模型把 JSON null 序列化成字符串字面量（如 "null"/"None"）而不是真正
        # 的 JSON null：这类字面量应视为“未提供 command_id”，走 selection 分支
        # 由 Skill 内部检索，而不是被当成真实 command_id 去查找并误判为越权命令。
        if command_id.casefold() in {"null", "none", "nil", "undefined", ""}:
            command_id = ""
        if command_id:
            return await self._dispatch_command(
                command_id=command_id,
                task_text=task_text,
                task_mode=task_mode,
                kwargs=kwargs,
            )

        return await self._selection_result_with_vector(
            task_text=task_text,
            task_mode=task_mode,
        )

    def _task_mode(self, task_text: str, requested: str) -> str | None:
        key = normalize_message_text(task_text).casefold()
        existing = self._task_modes.get(key)
        mode = requested.casefold()
        if mode in _TASK_MODES:
            # 首次声明的 mode 是该任务的用户原始目标，之后保持粘性。
            if existing and existing != mode:
                return existing
            self._task_modes[key] = mode
            return mode
        # 无合法显式值时复用缓存
        if existing:
            return existing
        if not mode:
            return ""
        return None

    def _selection_result(
        self,
        *,
        task_text: str,
        task_mode: str = "",
        requested_command_id: str = "",
        reason: str = "",
        response_policy: str = "",
        _max_candidates: int | None = None,
    ) -> ToolResult:
        prepared = self._prepare_selection(
            task_text=task_text,
            task_mode=task_mode,
            requested_command_id=requested_command_id,
            reason=reason,
        )
        if isinstance(prepared, ToolResult):
            return prepared
        return self._render_selection(
            prepared,
            requested_command_id=requested_command_id,
            response_policy=response_policy,
            _max_candidates=_max_candidates,
        )

    async def _selection_result_with_vector(
        self,
        *,
        task_text: str,
        task_mode: str = "",
        requested_command_id: str = "",
        reason: str = "",
        response_policy: str = "",
        _max_candidates: int | None = None,
    ) -> ToolResult:
        """零词法召回时先走 embedding 第二通道，再渲染候选卡。

        向量通道不可用/未命中阈值时，行为与同步 ``_selection_result`` 完全一致
        （全量降级列表 + ``recall=skill_full_listing``）。
        """

        prepared = self._prepare_selection(
            task_text=task_text,
            task_mode=task_mode,
            requested_command_id=requested_command_id,
            reason=reason,
        )
        if isinstance(prepared, ToolResult):
            return prepared
        if prepared.full_listing_fallback:
            ranked = await self._vector_ranked_snapshots(task_text, prepared.snapshots)
            if ranked:
                prepared = _SelectionPlan(
                    snapshots=ranked,
                    reason=prepared.reason,
                    full_listing_fallback=True,
                    recall="skill_vector",
                )
        return self._render_selection(
            prepared,
            requested_command_id=requested_command_id,
            response_policy=response_policy,
            _max_candidates=_max_candidates,
        )

    async def _vector_ranked_snapshots(
        self,
        task_text: str,
        snapshots: Sequence[CommandToolSnapshot],
    ) -> list[CommandToolSnapshot]:
        try:
            from .command_vector_recall import CommandVectorRecall

            ranked = await CommandVectorRecall.rank(
                self.skill.plugin_module,
                snapshots,
                task_text,
                limit=_VECTOR_RECALL_TOP_N,
            )
        except Exception:
            return []
        if not ranked:
            return []
        by_id = {
            str(snapshot.command_id): snapshot
            for snapshot in snapshots
            if snapshot.command_id
        }
        ordered: list[CommandToolSnapshot] = []
        for command_id, _score in ranked:
            snapshot = by_id.get(command_id)
            if snapshot is not None:
                ordered.append(snapshot)
        return ordered

    def _prepare_selection(
        self,
        *,
        task_text: str,
        task_mode: str,
        requested_command_id: str,
        reason: str,
    ) -> _SelectionPlan | ToolResult:
        requested = self._requested_identity_snapshots(requested_command_id)
        retrieval = self._retriever.retrieve(
            task_text,
            limit=None,
            context=self._skill_retrieval_context(),
        )
        direct = self._retrieved_snapshots(retrieval.candidates)
        family = self._action_family_snapshots(task_text)
        contextual = self._contextual_snapshots()
        all_snapshots = _merge_snapshot_sources(
            requested,
            family,
            direct,
            contextual,
        )
        if task_mode:
            snapshots = [
                snapshot
                for snapshot in all_snapshots
                if _snapshot_task_mode(snapshot) == task_mode
            ]
        else:
            snapshots = all_snapshots
        full_listing_fallback = False
        if not snapshots:
            if task_mode:
                fallback = all_snapshots[:5]
                fallback_cards = [
                    {
                        "command_id": s.command_id,
                        "head": s.head,
                        "task_mode": _snapshot_task_mode(s),
                    }
                    for s in fallback
                    if s.command_id
                ]
                return self._not_executed(
                    "selection_required",
                    candidates=[],
                    candidate_count=0,
                    displayed_candidate_count=0,
                    omitted_candidate_count=0,
                    truncated=False,
                    task_mode=task_mode,
                    response_policy="chat_without_clarification",
                    reason=reason or "no_task_mode_compatible_command",
                    **({"fallback_commands": fallback_cards} if fallback_cards else {}),
                )
            # 零召回（检索/家族/上下文/请求命中均为空）。若该 skill 仍有可用命令，
            # 降级返回全部可用命令作为候选，而不是直接让模型放弃——检索是静态
            # BM25，召回失败不代表 skill 内真的没有匹配的命令（例如“啃他”查询
            # 未命中，但“纳西妲啃”这个命令确实在该 skill 里）。真正没有可用命令
            # 时才回落到 not_found，让模型自然聊天。
            fallback_snapshots = sorted(
                self._available_by_key.values(),
                key=_snapshot_stable_key,
            )
            if not fallback_snapshots:
                payload: dict[str, Any] = {
                    "error": "该插件内没有匹配当前任务的命令",
                    "query": _clip_text(task_text, 1_000),
                }
                if requested_command_id:
                    payload["requested_command_id"] = _clip_text(
                        requested_command_id,
                        256,
                    )
                if reason:
                    payload["reason"] = _clip_text(reason, 128)
                return self._not_executed(
                    "not_found",
                    **payload,
                )
            snapshots = fallback_snapshots
            reason = reason or "fallback_full_listing"
            full_listing_fallback = True
        return _SelectionPlan(
            snapshots=list(snapshots),
            reason=reason,
            full_listing_fallback=full_listing_fallback,
            recall="skill_full_listing" if full_listing_fallback else "",
        )

    def _render_selection(
        self,
        plan: _SelectionPlan,
        *,
        requested_command_id: str,
        response_policy: str,
        _max_candidates: int | None,
    ) -> ToolResult:
        snapshots = plan.snapshots
        requested_command_id = _clip_text(requested_command_id, 256)
        reason = _clip_text(plan.reason, 128)
        if _max_candidates is not None:
            snapshots = snapshots[:_max_candidates]
        cards = _project_ambiguous_cards(
            snapshots,
            token_budget=self._ambiguity_token_budget,
            char_budget=self._result_char_budget,
            requested_command_id=requested_command_id,
            reason=reason,
        )
        payload: dict[str, Any] = {
            "candidates": cards,
            "candidate_count": len(snapshots),
            "displayed_candidate_count": len(cards),
            "omitted_candidate_count": max(len(snapshots) - len(cards), 0),
            "truncated": len(cards) < len(snapshots),
        }
        if requested_command_id:
            payload["requested_command_id"] = requested_command_id
        if reason:
            payload["reason"] = reason
        if response_policy:
            payload["response_policy"] = response_policy
        if plan.recall == "skill_vector":
            # 词法零召回后由 embedding 第二通道按语义相似度排序并收窄的候选。
            payload["recall"] = "skill_vector"
            payload["note"] = (
                "本次词法检索在该插件内未命中候选，以下是按语义相似度排序的候选"
                "（可能因长度预算被截断）。请从中判断是否有与用户请求语义匹配的"
                "命令；如果都不匹配，请不要调用命令，直接按普通聊天回复。"
            )
        elif plan.full_listing_fallback:
            # 明确标记这是零召回后的全量降级列表，并提示模型：候选是该插件
            # 全部可用命令（可能被预算截断），需自行判断语义是否匹配，不匹配
            # 就直接按普通聊天回复，不要强行选一个命令执行。
            payload["recall"] = "skill_full_listing"
            payload["note"] = (
                "本次检索在该插件内未命中任何候选，以下是该插件全部可用命令"
                "（可能因长度预算被截断）。请从中判断是否有与用户请求语义匹配的"
                "命令；如果都不匹配，请不要调用命令，直接按普通聊天回复。"
            )
        while cards and not self._selection_output_fits(payload):
            cards.pop()
            payload["displayed_candidate_count"] = len(cards)
            payload["omitted_candidate_count"] = len(snapshots) - len(cards)
            payload["truncated"] = True
        return self._not_executed("selection_required", **payload)

    def _skill_retrieval_context(self) -> dict[str, Any]:
        raw_context = getattr(self._command_context, "retrieval_context", None)
        context = dict(raw_context) if isinstance(raw_context, dict) else {}
        context["skill_scoped"] = True
        return context

    def _selection_output_fits(self, payload: dict[str, Any]) -> bool:
        output = _selection_payload(
            payload["candidates"],
            requested_command_id=str(payload.get("requested_command_id") or ""),
            reason=str(payload.get("reason") or ""),
            truncated=bool(payload.get("truncated")),
        )
        serialized = _compact_json(output)
        return (
            self._ambiguity_token_budget is None
            or estimate_text_tokens(serialized) <= self._ambiguity_token_budget
        ) and (
            self._result_char_budget is None
            or len(serialized) <= self._result_char_budget
        )

    def _requested_identity_snapshots(
        self,
        requested_command_id: str,
    ) -> list[CommandToolSnapshot]:
        identity = _local_command_identity(requested_command_id)
        if not identity:
            return []
        return sorted(
            [
                snapshot
                for snapshot in self._available_by_key.values()
                if any(
                    identity in candidate_identity
                    for value in (snapshot.head, *snapshot.aliases)
                    if (candidate_identity := _command_identity(value))
                )
            ],
            key=_snapshot_stable_key,
        )

    def _action_family_snapshots(
        self,
        task_text: str,
    ) -> list[CommandToolSnapshot]:
        task_identity = _command_identity(task_text)
        if not task_identity:
            return []
        grouped: dict[
            tuple[str, str, tuple[str, bool, str]],
            dict[str, CommandToolSnapshot],
        ] = {}
        for snapshot in self._available_by_key.values():
            if not _is_action_family_member(snapshot):
                continue
            if not _snapshot_context_compatible(
                snapshot,
                self._skill_retrieval_context(),
            ):
                continue
            key = _command_key(snapshot.command_id)
            for value in (snapshot.head, *snapshot.aliases):
                identity = _command_identity(value)
                for position, fragment in _stable_identity_fragments(identity):
                    if fragment in task_identity:
                        group_key = (
                            position,
                            fragment,
                            _action_family_contract(snapshot),
                        )
                        grouped.setdefault(group_key, {})[key] = snapshot
        families = [
            (position, fragment, snapshots)
            for (position, fragment, _contract), snapshots in grouped.items()
            if len(snapshots) >= 2
        ]
        if not families:
            return []
        families.sort(
            key=lambda item: (
                -len(item[1]),
                task_identity.index(item[1]),
                item[0],
                item[1],
            )
        )
        _position, _fragment, family = families[0]
        return sorted(family.values(), key=_snapshot_stable_key)

    async def _dispatch_command(
        self,
        *,
        command_id: str,
        task_text: str,
        task_mode: str,
        kwargs: dict[str, Any],
    ) -> ToolResult:
        key = _command_key(command_id)
        if key not in self._skill_command_keys and key not in self._known_by_key:
            return self._selection_result(
                task_text=task_text,
                task_mode=task_mode,
                requested_command_id=command_id,
                reason="command_out_of_skill",
                _max_candidates=3,
            )
        known = self._known_by_key.get(key)
        if known is None:
            return self._not_executed(
                "not_found",
                command_id=command_id,
                error="当前 Skill 缺少该命令的可用定义",
                reason="command_snapshot_missing",
            )
        snapshot = self._available_by_key.get(key)
        if snapshot is None:
            return self._not_executed(
                "unavailable_in_context",
                command_id=known.command_id,
                error="当前会话条件不满足，或该命令在当前场景不可用",
                command_schema=project_command_card(known),
            )
        if task_mode and _snapshot_task_mode(snapshot) != task_mode:
            correct_mode = _snapshot_task_mode(snapshot)
            # 降级为警告：用正确的 mode 过滤候选，返回候选卡让模型自行修正
            # 不再强制 chat_without_clarification，避免命令填对了也被终局拦截
            return self._selection_result(
                task_text=task_text,
                task_mode=correct_mode,
                requested_command_id=command_id,
                reason=(
                    f"task_mode_mismatch: 命令 {command_id} 属于 {correct_mode}，"
                    f"请将 task_mode 改为 {correct_mode} 后重新调用"
                ),
                _max_candidates=3,
            )
        return await self._execute_snapshot(snapshot, kwargs=kwargs)

    async def _execute_snapshot(
        self,
        snapshot: CommandToolSnapshot,
        *,
        kwargs: dict[str, Any],
    ) -> ToolResult:
        candidate = _candidate_from_snapshot(snapshot, skill=self.skill)
        binding = NativeCommandToolBinding(
            tool_name=self.name,
            candidate=candidate,
        )
        if all(
            _command_key(item.schema.command_id) != _command_key(snapshot.command_id)
            for item in self._command_context.candidates
        ):
            self._command_context.candidates.append(candidate)
        execution_count = len(self._command_context.executions)
        result = await self._command_context.execute_tool(
            binding=binding,
            raw_slots=_build_raw_slots(kwargs),
        )
        execution = (
            self._command_context.executions[-1]
            if len(self._command_context.executions) > execution_count
            else None
        )
        execution_started = bool(
            execution is not None and (execution.execution_started or execution.success)
        )
        executed = bool(execution is not None and execution.success)
        output = (
            dict(result.output)
            if isinstance(result.output, dict)
            else {"value": result.output}
        )
        output.setdefault("status", "executed" if executed else "not_executed")
        output["plugin_execution"] = execution_started
        output["executed"] = executed
        output.setdefault("skill_id", self.skill.skill_id)
        output.setdefault("plugin_module", self.skill.plugin_module)
        output.setdefault("command_id", snapshot.command_id)
        return ToolResult(
            output=output,
            display_content=result.display_content,
            is_error=result.is_error,
            is_retryable=result.is_retryable,
        )

    def _retrieved_snapshots(
        self,
        candidates: Iterable[CommandCandidate],
    ) -> list[CommandToolSnapshot]:
        snapshots: list[CommandToolSnapshot] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = _command_key(candidate.schema.command_id)
            snapshot = self._known_by_key.get(key)
            if snapshot is None or key in seen:
                continue
            seen.add(key)
            snapshots.append(snapshot)
        return snapshots

    def _contextual_snapshots(self) -> list[CommandToolSnapshot]:
        return self._retrieved_snapshots(self._command_context.candidates)

    def _not_executed(self, status: str, **payload: Any) -> ToolResult:
        output = {
            "status": status,
            "plugin_execution": False,
            "executed": False,
            "skill_id": self.skill.skill_id,
            "plugin_module": self.skill.plugin_module,
            **payload,
        }
        return ToolResult(
            output=output,
            display_content=f"{self.name}: {status}",
            is_retryable=status in {"ambiguous", "selection_required"},
        )


def _merge_snapshot_sources(
    *sources: Iterable[CommandToolSnapshot],
) -> list[CommandToolSnapshot]:
    merged: list[CommandToolSnapshot] = []
    seen: set[str] = set()
    for source in sources:
        for snapshot in source:
            key = _command_key(snapshot.command_id)
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(snapshot)
    return merged


def _snapshot_stable_key(snapshot: CommandToolSnapshot) -> tuple[str, str]:
    command_id = normalize_message_text(snapshot.command_id)
    return command_id.casefold(), command_id


def _stable_identity_fragments(identity: str) -> Iterable[tuple[str, str]]:
    if len(identity) < 2:
        return ()
    fragments: list[tuple[str, str]] = []
    for length in range(1, len(identity)):
        for position, fragment in (
            ("prefix", identity[:length]),
            ("suffix", identity[-length:]),
        ):
            if len(fragment) == 1 and not _is_single_cjk(fragment):
                continue
            fragments.append((position, fragment))
    return tuple(dict.fromkeys(fragments))


def _is_single_cjk(value: str) -> bool:
    return len(value) == 1 and "\u4e00" <= value <= "\u9fff"


def _is_action_family_member(snapshot: CommandToolSnapshot) -> bool:
    return snapshot.command_role in {"execute", "template", "random"}


def _snapshot_task_mode(snapshot: CommandToolSnapshot) -> str:
    if snapshot.command_role in {"helper", "usage", "catalog"}:
        return "help"
    intents = set(snapshot.intent_types or [])
    if intents & {"generate", "random", "transform", "mutate", "send", "play"}:
        return "action"
    if snapshot.side_effect == "query" or intents & {"query", "status"}:
        return "query"
    return "action"


def _action_family_contract(snapshot: CommandToolSnapshot) -> tuple[str, bool, str]:
    return snapshot.output_mode, bool(snapshot.generative), snapshot.side_effect


def _snapshot_context_compatible(
    snapshot: CommandToolSnapshot,
    context: dict[str, Any],
) -> bool:
    requires = snapshot.requires or {}
    has_image = bool(context.get("has_image")) or _reply_image_count(context) > 0
    has_reply = bool(context.get("has_reply"))
    has_at = bool(context.get("has_at"))
    has_verified_target = bool(context.get("has_verified_target"))
    accepts_target = _snapshot_accepts_target(snapshot)
    image_required = (
        bool(requires.get("image"))
        or snapshot.payload_policy == "image_only"
        or any(slot.type == "image" and slot.required for slot in snapshot.slots)
    )
    if image_required and not (
        has_image or ((has_at or has_verified_target) and accepts_target)
    ):
        return False
    if requires.get("reply") and not has_reply:
        return False
    if snapshot.target_requirement != "required":
        return True
    sources = set(snapshot.target_sources or [])
    accepts_at = accepts_target
    accepts_reply = "reply" in sources
    return bool(
        (has_at and accepts_at)
        or (has_verified_target and accepts_at)
        or (has_reply and accepts_reply)
    )


def _snapshot_accepts_target(snapshot: CommandToolSnapshot) -> bool:
    sources = set(snapshot.target_sources or [])
    return bool(
        sources & {"at", "nickname", "reply"}
        or snapshot.allow_at
        or (snapshot.requires or {}).get("at")
    )


def _reply_image_count(context: dict[str, Any]) -> int:
    try:
        return max(int(context.get("reply_image_count", 0) or 0), 0)
    except (TypeError, ValueError):
        return 0


def skill_dispatch_tool_name(plugin_module: str) -> str:
    module = normalize_message_text(plugin_module).casefold()
    if not module:
        raise ValueError("plugin_module cannot be empty")
    digest = hashlib.blake2s(module.encode("utf-8"), digest_size=4).hexdigest()
    tail = module.rsplit(".", 1)[-1]
    ascii_tail = (
        unicodedata.normalize("NFKD", tail).encode("ascii", "ignore").decode("ascii")
    )
    slug = _TOOL_NAME_PART_PATTERN.sub("_", ascii_tail).strip("_") or "plugin"
    slug = slug[:_TOOL_NAME_SLUG_LIMIT].rstrip("_") or "plugin"
    return f"{_TOOL_NAME_PREFIX}{slug}_{digest}"


def build_plugin_skill_dispatch_tools(
    *,
    skills: Iterable[PluginSkill],
    known_commands: Sequence[CommandToolSnapshot],
    available_commands: Sequence[CommandToolSnapshot],
    knowledge_base: PluginKnowledgeBase,
    session_id: str | None,
    command_context: NativeCommandExecutionContext,
    task_modes: dict[str, str] | None = None,
    ambiguity_token_budget: int | None = None,
    result_char_budget: int | None = None,
) -> dict[str, ToolExecutable]:
    result: dict[str, ToolExecutable] = {}
    ordered_skills = sorted(
        skills,
        key=lambda item: (
            normalize_message_text(item.plugin_module).casefold(),
            normalize_message_text(item.skill_id).casefold(),
        ),
    )
    for skill in ordered_skills:
        tool = PluginSkillDispatchTool(
            skill,
            known_commands=known_commands,
            available_commands=available_commands,
            knowledge_base=knowledge_base,
            session_id=session_id,
            command_context=command_context,
            task_modes=task_modes,
            ambiguity_token_budget=ambiguity_token_budget,
            result_char_budget=result_char_budget,
        )
        if tool.name in result:
            raise ValueError(f"duplicate plugin Skill tool name: {tool.name}")
        result[tool.name] = tool
    return result


def _skill_snapshots_by_key(
    skill: PluginSkill,
    snapshots: Sequence[CommandToolSnapshot],
    *,
    restrict_to_skill: bool = True,
) -> dict[str, CommandToolSnapshot]:
    module_key = normalize_message_text(skill.plugin_module).casefold()
    allowed = (
        {_command_key(command_id) for command_id in skill.command_ids}
        if restrict_to_skill
        else None
    )
    result: dict[str, CommandToolSnapshot] = {}
    for snapshot in sorted(
        snapshots,
        key=lambda item: (
            _command_key(item.command_id),
            normalize_message_text(item.command_id),
        ),
    ):
        key = _command_key(snapshot.command_id)
        if (
            not key
            or (allowed is not None and key not in allowed)
            or normalize_message_text(snapshot.plugin_module).casefold() != module_key
        ):
            continue
        result.setdefault(key, snapshot)
    return result


def _plugin_knowledge_base(
    skill: PluginSkill,
    knowledge_base: PluginKnowledgeBase,
) -> PluginKnowledgeBase:
    module_key = normalize_message_text(skill.plugin_module).casefold()
    return PluginKnowledgeBase(
        plugins=[
            plugin
            for plugin in knowledge_base.plugins
            if normalize_message_text(plugin.module).casefold() == module_key
        ],
        user_role=knowledge_base.user_role,
    )


def _candidate_from_snapshot(
    snapshot: CommandToolSnapshot,
    *,
    skill: PluginSkill,
) -> CommandCandidate:
    return CommandCandidate(
        plugin_module=snapshot.plugin_module,
        plugin_name=snapshot.plugin_name,
        schema=_schema_from_tool_snapshot(snapshot),
        score=0.0,
        reason=f"skill_dispatch:{skill.skill_id}",
        family=snapshot.family,
        tool=snapshot,
    )


def _build_raw_slots(kwargs: dict[str, Any]) -> dict[str, Any]:
    slots = kwargs.get("slots")
    raw_slots = dict(slots) if isinstance(slots, dict) else {}
    raw_slots[TASK_TEXT_FIELD] = str(kwargs.get(TASK_TEXT_FIELD, "") or "")
    for field in (TARGET_HINT_FIELD, TARGET_REF_FIELD, PAYLOAD_HINT_FIELD):
        value = str(kwargs.get(field, "") or "")
        if value:
            raw_slots[field] = value
    return raw_slots


def _project_ambiguous_cards(
    snapshots: Sequence[CommandToolSnapshot],
    *,
    token_budget: int | None,
    char_budget: int | None,
    requested_command_id: str,
    reason: str,
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for snapshot in snapshots:
        card = project_command_card(snapshot)
        remaining_tokens = (
            token_budget
            - _selection_payload_tokens(
                cards,
                requested_command_id=requested_command_id,
                reason=reason,
                truncated=True,
            )
            if token_budget is not None
            else None
        )
        remaining_chars = (
            char_budget
            - _selection_payload_chars(
                cards,
                requested_command_id=requested_command_id,
                reason=reason,
                truncated=True,
            )
            if char_budget is not None
            else None
        )
        fitted = (
            _fit_ambiguous_card(
                card,
                token_budget=remaining_tokens,
                char_budget=remaining_chars,
            )
            if token_budget is not None or char_budget is not None
            else card
        )
        if fitted is None:
            continue
        trial_cards = [*cards, fitted]
        trial_tokens = _selection_payload_tokens(
            trial_cards,
            requested_command_id=requested_command_id,
            reason=reason,
            truncated=len(trial_cards) < len(snapshots),
        )
        trial_chars = _selection_payload_chars(
            trial_cards,
            requested_command_id=requested_command_id,
            reason=reason,
            truncated=len(trial_cards) < len(snapshots),
        )
        if token_budget is not None and trial_tokens > token_budget:
            break
        if char_budget is not None and trial_chars > char_budget:
            break
        cards = trial_cards
    return cards


def _fit_ambiguous_card(
    card: dict[str, Any],
    *,
    token_budget: int | None,
    char_budget: int | None,
) -> dict[str, Any] | None:
    if token_budget is not None and token_budget <= 0:
        return None
    if char_budget is not None and char_budget <= 0:
        return None
    if _card_fits(card, token_budget=token_budget, char_budget=char_budget):
        return card

    command_id = str(card.get("command_id", "") or "")
    if not command_id:
        return None
    compact: dict[str, Any] = {"command_id": command_id}
    if not _card_fits(
        compact,
        token_budget=token_budget,
        char_budget=char_budget,
    ):
        return None

    priority = (
        "head",
        "usage",
        "plugin",
        "description",
        "render",
        "slots",
        "accepted_inputs",
        "required_context",
        "aliases",
        "examples",
        "use_cases",
        "anti_use_cases",
        "output_mode",
        "side_effect",
        "execution_policy",
        "source_of_truth",
        "requires_real_result",
    )
    for key in priority:
        if key not in card:
            continue
        _add_card_value(
            compact,
            key,
            card[key],
            token_budget=token_budget,
            char_budget=char_budget,
        )
    return compact


def _add_card_value(
    target: dict[str, Any],
    key: str,
    value: Any,
    *,
    token_budget: int | None,
    char_budget: int | None,
) -> None:
    if isinstance(value, list):
        accepted: list[Any] = []
        for item in value:
            candidate_item = _compact_card_list_item(item)
            trial = {**target, key: [*accepted, candidate_item]}
            if not _card_fits(
                trial,
                token_budget=token_budget,
                char_budget=char_budget,
            ):
                break
            accepted.append(candidate_item)
        if accepted:
            target[key] = accepted
        return

    trial = {**target, key: value}
    if _card_fits(
        trial,
        token_budget=token_budget,
        char_budget=char_budget,
    ):
        target[key] = value
        return
    if not isinstance(value, str):
        return

    low = 0
    high = len(value)
    best = ""
    while low <= high:
        middle = (low + high) // 2
        clipped = value[:middle].rstrip()
        if middle < len(value) and clipped:
            clipped += "…"
        if clipped and _card_fits(
            {**target, key: clipped},
            token_budget=token_budget,
            char_budget=char_budget,
        ):
            best = clipped
            low = middle + 1
        else:
            high = middle - 1
    if best:
        target[key] = best


def _compact_card_list_item(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    keys = (
        "name",
        "type",
        "required",
        "description",
        "for",
        "any_of",
    )
    return {key: value[key] for key in keys if key in value}


def _card_fits(
    card: dict[str, Any],
    *,
    token_budget: int | None,
    char_budget: int | None,
) -> bool:
    serialized = _compact_json(card)
    return (
        token_budget is None or estimate_text_tokens(serialized) <= token_budget
    ) and (char_budget is None or len(serialized) <= char_budget)


def _selection_payload(
    cards: list[dict[str, Any]],
    *,
    requested_command_id: str,
    reason: str,
    truncated: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "selection_required",
        "plugin_execution": False,
        "executed": False,
        "candidates": cards,
        "truncated": truncated,
    }
    if requested_command_id:
        payload["requested_command_id"] = requested_command_id
    if reason:
        payload["reason"] = reason
    return payload


def _selection_payload_tokens(
    cards: list[dict[str, Any]],
    *,
    requested_command_id: str,
    reason: str,
    truncated: bool,
) -> int:
    return estimate_text_tokens(
        _compact_json(
            _selection_payload(
                cards,
                requested_command_id=requested_command_id,
                reason=reason,
                truncated=truncated,
            )
        )
    )


def _selection_payload_chars(
    cards: list[dict[str, Any]],
    *,
    requested_command_id: str,
    reason: str,
    truncated: bool,
) -> int:
    return len(
        _compact_json(
            _selection_payload(
                cards,
                requested_command_id=requested_command_id,
                reason=reason,
                truncated=truncated,
            )
        )
    )


def _compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _tool_description(skill: PluginSkill) -> str:
    metadata = {
        "plugin": _clip_text(skill.plugin_name, 160),
        "description": _clip_text(skill.description, 360),
        "aliases": _bounded_values(
            skill.aliases,
            item_limit=120,
            total_limit=_ALIASES_TEXT_LIMIT,
        ),
        "usage": _clip_text(skill.usage, 600),
        "introduction": _clip_text(skill.introduction, 480),
        "precautions": _bounded_values(
            skill.precautions,
            item_limit=240,
            total_limit=_PRECAUTIONS_TEXT_LIMIT,
        ),
        "input_types": _bounded_values(
            skill.input_types,
            item_limit=80,
            total_limit=_CAPABILITY_VALUES_TEXT_LIMIT,
        ),
        "output_modes": _bounded_values(
            skill.output_modes,
            item_limit=80,
            total_limit=_CAPABILITY_VALUES_TEXT_LIMIT,
        ),
        "side_effects": _bounded_values(
            skill.side_effects,
            item_limit=80,
            total_limit=_CAPABILITY_VALUES_TEXT_LIMIT,
        ),
        "semantic_tools": _semantic_contracts(skill),
    }
    compact = {key: value for key, value in metadata.items() if value}
    payload = json.dumps(
        compact,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (
        f"插件级能力契约：{payload}\n"
        "已有明确 command_id 时可直接指定；否则仅在本插件内部检索具体命令。"
    )


def _semantic_contracts(skill: PluginSkill) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    used = 0
    contracts = sorted(
        skill.semantic_tools,
        key=lambda item: (
            _single_line(item.name).casefold(),
            _single_line(item.name),
        ),
    )
    for contract in contracts:
        item = {
            "name": _clip_text(contract.name, 120),
            "description": _clip_text(contract.description, 320),
            "use_cases": _bounded_values(
                contract.use_cases,
                item_limit=180,
                total_limit=360,
            ),
            "anti_use_cases": _bounded_values(
                contract.anti_use_cases,
                item_limit=180,
                total_limit=360,
            ),
            "output_mode": contract.output_mode,
            "side_effect": contract.side_effect,
            "execution_policy": contract.execution_policy,
            "requires_real_result": contract.requires_real_result,
        }
        compact = {
            key: value
            for key, value in item.items()
            if value is not None and value != []
        }
        size = len(
            json.dumps(
                compact,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        if result and used + size > _SEMANTIC_CONTRACTS_TEXT_LIMIT:
            break
        result.append(compact)
        used += size
    return result


def _bounded_values(
    values: Iterable[object],
    *,
    item_limit: int,
    total_limit: int,
) -> list[str]:
    normalized = sorted(
        {_single_line(value) for value in values if _single_line(value)},
        key=lambda value: (value.casefold(), value),
    )
    result: list[str] = []
    used = 0
    for value in normalized:
        clipped = _clip_text(value, item_limit)
        if result and used + len(clipped) > total_limit:
            break
        if not result and len(clipped) > total_limit:
            clipped = _clip_text(clipped, total_limit)
        result.append(clipped)
        used += len(clipped)
    return result


def _clip_text(value: object, limit: int) -> str:
    text = _single_line(value)
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


def _single_line(value: object) -> str:
    return " ".join(normalize_message_text(str(value or "")).split())


def _command_key(command_id: object) -> str:
    return normalize_message_text(str(command_id or "")).casefold()


def _local_command_identity(command_id: object) -> str:
    command_key = _command_key(command_id)
    return _command_identity(command_key.rsplit(".", 1)[-1])


def _command_identity(value: object) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", _command_key(value))


__all__ = [
    "PluginSkillDispatchTool",
    "build_plugin_skill_dispatch_tools",
    "skill_dispatch_tool_name",
]
