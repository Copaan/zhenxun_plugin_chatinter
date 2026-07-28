"""Unified pipeline stage: one agent turn for chat and plugin invocation.

Replaces the old two-path shape (structured plugin router + separate chat
degrade call).  Both group and private turns run the same UnifiedChatAgent;
plugin tools are attached when the scenario allows them and the command
snapshot is non-empty.
"""

from __future__ import annotations

import asyncio

from nonebot.adapters import Bot, Event

from .agents.core import UnifiedChatRequest
from .agents.unified_chat_agent import UnifiedChatAgent
from .capability_catalog import build_capability_catalog
from .group_plugin_flow import _execute_native_tool_route
from .meta_tools import build_meta_tools
from .middleware import TurnMiddlewareState
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
from .route_text import is_usage_question, normalize_message_text
from .tool_retriever import CommandToolRetriever
from .turn_frame import PipelineStage, TurnFrame


async def stage_unified_run(
    *,
    frame: TurnFrame,
    bot: Bot,
    event: Event,
    middleware_state: TurnMiddlewareState,
    middleware,
) -> None:
    frame.stage(PipelineStage.AGENT_RUN)
    route_completed_hook, reply_hook = _build_agent_stage_hooks(
        frame=frame,
        middleware_state=middleware_state,
        middleware=middleware,
    )
    message_text = normalize_message_text(
        frame.route_message or frame.current_message
    )
    report = NativeRouteReport(helper_mode=is_usage_question(message_text))
    command_context: NativeCommandExecutionContext | None = None
    tools = None
    catalog_text = ""
    snapshots = list(frame.command_tools or [])
    knowledge_base = frame.knowledge_base
    if frame.allow_plugin_tools and snapshots and knowledge_base is not None:
        report.note_candidate_policy(
            reason="unified_catalog",
            limit=len(snapshots),
        )
        report.candidate_total = max(report.candidate_total, len(snapshots))

        async def execute_native_route(
            validated: NativeValidatedRoute,
            route_report: NativeRouteReport,
        ) -> NativeToolExecutionResult:
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
                route_report=route_report,
                mention_profiles=frame.mention_profiles,
            )

        command_context = NativeCommandExecutionContext(
            candidates=[],
            has_reply=frame.has_reply,
            report=report,
            route_executor=execute_native_route,
            message_text=message_text,
        )
        catalog = build_capability_catalog(snapshots)
        catalog_text = catalog.text
        tools = build_meta_tools(
            command_tools=snapshots,
            retriever=CommandToolRetriever(
                knowledge_base,
                session_id=frame.session_key,
                tools=snapshots,
            ),
            command_context=command_context,
        )
        frame.update_tags(
            catalog_commands=float(catalog.command_count),
            catalog_plugins=float(catalog.plugin_count),
            catalog_collapsed=float(len(catalog.collapsed_modules)),
        )

    request = UnifiedChatRequest(
        message_text=message_text,
        session_key=frame.session_key,
        budget_controller=frame.budget_controller,
        messages=list(frame.agent_messages or []),
        report=report,
        scenario=frame.scenario,
        catalog_text=catalog_text,
        tools=tools,
        command_context=command_context,
    )

    async def run_agent():
        return (await UnifiedChatAgent().run(request)).to_main_result()

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


__all__ = ["stage_unified_run"]
