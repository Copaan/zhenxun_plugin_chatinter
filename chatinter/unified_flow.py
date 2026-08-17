"""Unified pipeline stage: one agent turn for chat and plugin invocation.

Replaces the old two-path shape (structured plugin router + separate chat
degrade call).  Both group and private turns run the same UnifiedChatAgent;
plugin tools are attached when the scenario allows them and the command
snapshot is non-empty.
"""

from __future__ import annotations

import asyncio

from nonebot.adapters import Bot, Event

from zhenxun.utils.utils import get_entity_ids

from .agents.core import AgentResult, UnifiedChatRequest
from .agents.unified_chat_agent import UnifiedChatAgent
from .chat_handler import (
    RerouteExecutionResult,
    consume_reroute_cancellation_receipt,
)
from .group_plugin_flow import _execute_native_tool_route
from .gscore_adapter import get_gscore_adapter
from .meta_tools import render_command_candidate_context
from .mixed_tool_catalog import build_mixed_tool_catalog
from .native_executor import (
    NativeCommandExecutionContext,
    NativeToolExecutionResult,
    NativeValidatedRoute,
)
from .native_route import NativeRouteReport
from .pipeline_stages import (
    _build_agent_stage_hooks,
    _run_direct_agent_turn,
    _send_delayed_reply_status,
    _set_agent_stage_result,
)
from .plugin_registry import PluginRegistry
from .plugin_skill_index import build_plugin_skill_index, log_skill_debug_once
from .route_text import is_usage_question, normalize_message_text
from .tool_retriever import CommandToolRetriever
from .turn_frame import PipelineStage, TurnFrame


async def stage_unified_run(
    *,
    frame: TurnFrame,
    bot: Bot,
    event: Event,
) -> None:
    route_completed_hook, reply_hook = _build_agent_stage_hooks(
        frame=frame,
    )
    message_text = normalize_message_text(frame.route_message or frame.current_message)
    report = NativeRouteReport(helper_mode=is_usage_question(message_text))
    command_context: NativeCommandExecutionContext | None = None
    tools = None
    tool_catalog = None
    command_candidate_text = ""
    available_snapshots = list(frame.command_tools or [])
    gscore_tool = await get_gscore_adapter().build_tool(frame)
    if gscore_tool is not None:
        tools = {gscore_tool.name: gscore_tool}
        command_candidate_text = gscore_tool.candidate_context
        frame.update_tags(gscore_capabilities=float(gscore_tool.capability_count))
    knowledge_base = frame.knowledge_base
    if (
        frame.allow_plugin_tools
        and knowledge_base is not None
        and knowledge_base.plugins
    ):
        skill_snapshots = PluginRegistry.build_command_tool_snapshots(
            knowledge_base,
            selection_context=None,
        )
        skill_index = build_plugin_skill_index(knowledge_base, skill_snapshots)
    else:
        skill_snapshots = []
        skill_index = None
    if skill_index is not None and skill_index.skills:
        submitted_action_keys: set[str] = set()
        retrieval = CommandToolRetriever(
            knowledge_base,
            session_id=frame.session_key,
            tools=available_snapshots,
        ).retrieve(
            _command_retrieval_text(frame, message_text),
            limit=None,
            context=_retrieval_context(frame),
        )
        candidates = list(retrieval.candidates)
        report.note_candidate_policy(
            reason="local_command_retrieval",
            limit=0,
        )
        report.candidate_total = max(
            report.candidate_total,
            retrieval.total_commands,
        )
        report.lexical_candidates = len(candidates)
        report.note_tool_pool(len(candidates))
        report.note_prompt_exposure(candidates)

        async def execute_native_route(
            validated: NativeValidatedRoute,
            route_report: NativeRouteReport,
        ) -> NativeToolExecutionResult:
            try:
                return await _execute_native_tool_route(
                    bot=bot,
                    event=event,
                    trace=frame.trace,
                    validated=validated,
                    knowledge_plugins=knowledge_base.plugins,
                    current_message=message_text,
                    user_id=frame.user_id,
                    group_id=frame.group_id,
                    session_id=frame.session_key,
                    has_reply=frame.has_reply,
                    extra_image_segments=frame.reply_image_segments_for_reroute,
                    reply_image_count=frame.reply_image_count,
                    route_report=route_report,
                    mention_profiles=frame.mention_profiles,
                    submitted_action_keys=submitted_action_keys,
                )
            except asyncio.CancelledError:
                receipt = consume_reroute_cancellation_receipt()
                if receipt is not None and receipt.execution_uncertain:
                    _project_cancelled_reroute_receipt(
                        frame=frame,
                        receipt=receipt,
                        validated=validated,
                        message_text=message_text,
                    )
                raise

        dialogue_context_pack = getattr(frame, "dialogue_context_pack", None)
        command_context = NativeCommandExecutionContext(
            candidates=candidates,
            has_reply=frame.has_reply,
            report=report,
            route_executor=execute_native_route,
            message_text=message_text,
            event_target_hint=_event_target_hint(frame=frame, bot=bot),
            target_refs=(
                dialogue_context_pack.action_target_refs()
                if dialogue_context_pack is not None
                else {}
            ),
            retrieval_context=_retrieval_context(frame),
        )
        command_candidate_text = render_command_candidate_context(candidates)
        tool_catalog = build_mixed_tool_catalog(
            skill_index=skill_index,
            known_commands=skill_snapshots,
            available_commands=available_snapshots,
            initial_candidates=candidates,
            knowledge_base=knowledge_base,
            session_id=frame.session_key,
            command_context=command_context,
        )
        frame.chat_tool_exposure_state = "plugin_tools_exposed"
        log_skill_debug_once(skill_index)
        frame.update_tags(
            skill_commands=float(skill_index.command_count),
            skill_count=float(len(skill_index.skills)),
            retrieved_commands=float(len(candidates)),
        )

    history_scope = _bound_history_scope(frame)
    request = UnifiedChatRequest(
        message_text=message_text,
        session_key=frame.session_key,
        budget_controller=frame.budget_controller,
        messages=list(frame.agent_messages or []),
        report=report,
        scenario=frame.scenario,
        user_id=history_scope["user_id"],
        group_id=history_scope["group_id"],
        bot_id=history_scope["bot_id"],
        platform=history_scope["platform"],
        channel_id=history_scope["channel_id"],
        command_candidate_text=command_candidate_text,
        tools=tools,
        tool_catalog=tool_catalog,
        command_context=command_context,
        context_bundle=frame.context_bundle,
        context_xml=frame.context_xml,
    )

    agent_result_holder: list[AgentResult] = []

    async def run_agent():
        agent_result = await UnifiedChatAgent().run(request)
        agent_result_holder.append(agent_result)
        return agent_result.to_main_result()

    progress_task = (
        asyncio.create_task(_send_delayed_reply_status(frame))
        if frame.scenario == "private_chat"
        else None
    )
    try:
        main_result = await _run_direct_agent_turn(
            message_text=message_text,
            report=report,
            budget_controller=frame.budget_controller,
            route_completed_hook=route_completed_hook,
            reply_hook=reply_hook,
            run_agent=run_agent,
        )
    finally:
        if progress_task is not None:
            progress_task.cancel()
            await asyncio.gather(progress_task, return_exceptions=True)
    _set_agent_stage_result(frame=frame, main_result=main_result)
    if agent_result_holder:
        frame.agent_observations = list(
            getattr(agent_result_holder[-1], "observations", ())
        )
        _tag_agent_observations(frame)
    frame.stage(PipelineStage.AGENT_RUN)


def _project_cancelled_reroute_receipt(
    *,
    frame: TurnFrame,
    receipt: RerouteExecutionResult,
    validated: NativeValidatedRoute,
    message_text: str,
) -> None:
    route_result = validated.route_result
    frame.cancelled_reroute_receipt = {
        "status": "uncertain",
        "execution_uncertain": True,
        "execution_started": bool(receipt.execution_started),
        "task_stopped": bool(receipt.task_stopped),
        "trace_id": receipt.trace_id,
        "command": receipt.command,
        "command_id": route_result.command_id if route_result is not None else "",
        "plugin_name": (
            route_result.decision.plugin_name if route_result is not None else ""
        ),
        "plugin_module": (
            route_result.decision.plugin_module if route_result is not None else ""
        ),
        "task_text": (
            validated.task_frame.effective_text
            if validated.task_frame is not None
            else message_text
        ),
    }


def _tag_agent_observations(frame: TurnFrame) -> None:
    if not frame.agent_observations:
        return
    metadata = getattr(frame.agent_observations[-1], "metadata", None)
    if not isinstance(metadata, dict):
        return
    executions = tuple(metadata.get("tool_executions") or ())
    model_requests = tuple(metadata.get("model_requests") or ())
    selected_commands = tuple(
        dict.fromkeys(
            str(item.get("command_id", "") or "")
            for item in executions
            if isinstance(item, dict) and str(item.get("command_id", "") or "")
        )
    )
    retrieved_commands = tuple(
        str(item or "")
        for item in metadata.get("retrieved_command_ids", ())
        if str(item or "")
    )
    exposed_commands: tuple[str, ...] = ()
    if model_requests and isinstance(model_requests[-1], dict):
        exposed_commands = tuple(
            str(item or "")
            for key in ("native_command_ids", "indexed_command_ids")
            for item in model_requests[-1].get(key, ())
            if str(item or "")
        )
    frame.update_tags(
        agent_model_requests=float(len(model_requests)),
        agent_tool_executions=float(len(executions)),
        plugin_outcome=str(metadata.get("plugin_outcome", "") or ""),
        failure_layer=str(metadata.get("failure_layer", "") or ""),
        retrieved_command_ids="|".join(retrieved_commands),
        exposed_command_ids="|".join(dict.fromkeys(exposed_commands)),
        selected_command_ids="|".join(selected_commands),
    )


def _retrieval_context(frame: TurnFrame) -> dict[str, bool | int | str]:
    selection = frame.selection_context
    target = getattr(frame, "verified_action_target", None)
    has_verified_target = bool(getattr(target, "is_resolved", False))
    return {
        "has_reply": bool(frame.has_reply),
        "has_at": bool(getattr(selection, "has_at", False)),
        "has_image": bool(getattr(selection, "has_image", False)),
        "reply_image_count": len(frame.reply_image_segments_for_reroute or []),
        "has_verified_target": has_verified_target,
        "verified_target_source": (
            str(getattr(target, "source", "") or "")
            if has_verified_target
            else ""
        ),
    }


def _event_target_hint(*, frame: TurnFrame, bot: Bot) -> str:
    target = getattr(frame, "verified_action_target", None)
    if target is None or not bool(getattr(target, "is_resolved", False)):
        return ""
    if str(getattr(target, "source", "") or "") not in {
        "at",
        "reply",
        "alias",
        "self_nickname",
    }:
        return ""
    target_user_id = normalize_message_text(
        str(getattr(target, "user_id", "") or "")
    )
    if not target_user_id or target_user_id == str(getattr(bot, "self_id", "") or ""):
        return ""
    return f"[@{target_user_id}]"


def _command_retrieval_text(frame: TurnFrame, message_text: str) -> str:
    current = normalize_message_text(message_text)
    bundle = getattr(frame, "context_bundle", None)
    if not bool(getattr(frame, "has_reply", False)) or bundle is None:
        return current
    parts = [current] if current else []
    for section in getattr(bundle, "sections", ()):
        if getattr(section, "name", "") != "reply_layers":
            continue
        quoted = normalize_message_text("\n".join(getattr(section, "lines", ())))
        if quoted and quoted not in parts:
            parts.append(quoted)
    return "\n".join(parts)


def _bound_history_scope(frame: TurnFrame) -> dict[str, str | None]:
    session = getattr(frame, "session", None)
    user_id = str(getattr(frame, "user_id", "") or "")
    group_id = str(getattr(frame, "group_id", "") or "") or None
    bot_id = str(getattr(frame, "bot_id", "") or "") or None
    platform = None
    channel_id = None
    if session is not None:
        try:
            entity = get_entity_ids(session)
            user_id = str(entity.user_id or user_id)
            group_id = str(entity.group_id or "") or None
            channel_id = str(entity.channel_id or "") or None
        except Exception:
            pass
        platform = str(getattr(session, "platform", "") or "") or None
    return {
        "user_id": user_id,
        "group_id": group_id,
        "bot_id": bot_id,
        "platform": platform,
        "channel_id": channel_id,
    }


__all__ = ["stage_unified_run"]
