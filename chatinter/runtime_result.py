"""Result construction helpers for ChatInter main requests."""

from __future__ import annotations

from dataclasses import replace
from inspect import isawaitable
from typing import Any

from .llm_compat import ToolResult
from .main_request_models import (
    MainRequestOutput,
    MainRequestReplyHook,
    MainRequestResult,
    MainRequestRouteHook,
    MainRequestTimelineItem,
)
from .native_executor import NativeToolExecutionResult
from .native_route import NativeRouteDecision, NativeRouteReport, NativeRouteResult
from .route_text import normalize_message_text, normalize_reply_text


_MAIN_STAGE = "main_request"

def _fallback_result(
    *,
    report: NativeRouteReport,
    reason: str,
    reply: str,
    timeline: list[MainRequestTimelineItem] | None = None,
) -> MainRequestResult:
    decision = NativeRouteDecision(action="chat", confidence=0.0, reason=reason)
    report.finalize(reason=reason, stage=_MAIN_STAGE)
    return MainRequestResult(
        decision=decision,
        route_result=None,
        report=report,
        timeline=(
            *(timeline or []),
            MainRequestTimelineItem(
                role="system",
                kind="fallback",
                content=reason,
            ),
        ),
        output=MainRequestOutput(final_text=reply, memory_text=reply),
    )


def _is_catalog_tool_result(result: ToolResult) -> bool:
    output = result.output if isinstance(result.output, dict) else {}
    return output.get("status") in {
        "retrieved",
        "capability_candidates_retrieved",
    }

async def _finalize_result(
    result: MainRequestResult,
    *,
    route_completed_hook: MainRequestRouteHook | None,
    reply_hook: MainRequestReplyHook | None,
) -> MainRequestResult:
    if route_completed_hook is not None:
        maybe_awaitable = route_completed_hook(result)
        if maybe_awaitable is not None:
            await maybe_awaitable

    output = result.output
    if not output.should_send:
        return result

    final_text = normalize_reply_text(output.final_text)
    if not final_text:
        final_text = (
            _fallback_final_reply(list(result.executions)) or "我暂时没想好怎么回答你。"
        )
    if reply_hook is not None:
        maybe_reply = reply_hook(final_text)
        final_text = (
            await maybe_reply if isawaitable(maybe_reply) else str(maybe_reply or "")
        )
    final_text = normalize_reply_text(final_text)
    if not final_text:
        final_text = "我暂时没想好怎么回答你。"
    final_timeline = _with_final_timeline(
        result.timeline,
        final_text=final_text,
        should_send=True,
    )
    memory_text = normalize_message_text(output.memory_text) or _timeline_memory_text(
        list(final_timeline),
        fallback=final_text,
    )
    return replace(
        result,
        timeline=final_timeline,
        output=replace(
            output,
            final_text=final_text,
            memory_text=memory_text,
            should_send=True,
        ),
    )

def _first_route(
    executions: list[NativeToolExecutionResult],
) -> NativeRouteResult | None:
    for execution in executions:
        if execution.route_result is not None:
            return execution.route_result
    return None

def _fallback_final_reply(executions: list[NativeToolExecutionResult]) -> str:
    if not executions:
        return ""
    success_count = sum(1 for item in executions if item.success)
    latest = executions[-1]
    if latest.display_text:
        return latest.display_text
    if success_count:
        return "处理好了。"
    message = str(latest.output.get("error", "") or latest.reason or "").strip()
    return message or "这个暂时没处理成功。"

def _timeline_memory_text(
    timeline: list[MainRequestTimelineItem] | tuple[MainRequestTimelineItem, ...],
    *,
    fallback: str = "",
) -> str:
    lines: list[str] = []
    for item in timeline:
        text = _timeline_item_summary(item)
        if text:
            lines.append(text)
    if fallback:
        lines.append(normalize_message_text(f"assistant: {fallback}"))
    return "\n".join(dict.fromkeys(line for line in lines if line))[:4000]

def _timeline_item_summary(item: MainRequestTimelineItem) -> str:
    role = normalize_message_text(item.role)
    kind = normalize_message_text(item.kind)
    prefix = f"{role}/{kind}".strip("/")
    if item.tool_name:
        prefix = f"{prefix}:{normalize_message_text(item.tool_name)}"
    content = normalize_message_text(item.content)
    if not content:
        output = (
            item.metadata.get("output") if isinstance(item.metadata, dict) else None
        )
        content = _compact_output_summary(output)
    if not content:
        arguments = (
            item.metadata.get("arguments") if isinstance(item.metadata, dict) else None
        )
        content = _compact_output_summary(arguments)
    if not content:
        return ""
    return f"{prefix}: {content}"[:800]

def _compact_output_summary(value: Any) -> str:
    if not isinstance(value, dict):
        return normalize_message_text(str(value or ""))[:500]
    parts: list[str] = []
    for key in (
        "status",
        "ok",
        "command_id",
        "rendered_command",
        "matched_plugin",
        "task_text",
        "error",
        "remaining_task_hint",
    ):
        item = value.get(key)
        if item not in ("", [], {}, None):
            parts.append(f"{key}={normalize_message_text(str(item))}")
    messages = value.get("messages_sent")
    if isinstance(messages, list) and messages:
        parts.append(
            "messages_sent="
            + " | ".join(
                normalize_message_text(str(message or ""))
                for message in messages[:3]
                if normalize_message_text(str(message or ""))
            )
        )
    artifacts = value.get("artifacts")
    if isinstance(artifacts, list) and artifacts:
        summaries = [
            normalize_message_text(str(item.get("summary", "") or ""))
            for item in artifacts[:3]
            if isinstance(item, dict)
            and normalize_message_text(str(item.get("summary", "") or ""))
        ]
        if summaries:
            parts.append("artifacts=" + " | ".join(summaries))
    return "；".join(parts)[:500]

def _user_timeline_item(message_text: str) -> MainRequestTimelineItem:
    return MainRequestTimelineItem(
        role="user",
        kind="current_user",
        content=message_text,
    )


def _with_final_timeline(
    timeline: tuple[MainRequestTimelineItem, ...],
    *,
    final_text: str,
    should_send: bool,
) -> tuple[MainRequestTimelineItem, ...]:
    if not final_text and not should_send:
        return timeline
    return (
        *timeline,
        MainRequestTimelineItem(
            role="assistant",
            kind="final_output",
            content=final_text,
        ),
    )

__all__ = [
    "_fallback_final_reply",
    "_fallback_result",
    "_finalize_result",
    "_timeline_memory_text",
    "_user_timeline_item",
]
