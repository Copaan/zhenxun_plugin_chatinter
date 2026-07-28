"""Unified chat agent: one tool-loop turn for chat and plugin invocation.

Intent recognition, plugin execution and the conversational reply all happen
in a single model context.  The model sees the full capability catalog in the
system prompt and three fixed meta tools; whether to invoke a plugin or just
chat is its own decision inside one loop — there is no separate router call
and no chat-degrade second call.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, cast

from zhenxun.services.ai.core.engine.token_counter import parse_usage_info

from ..config import (
    CHAT_RESPONSE_TIMEOUT_SECONDS,
    build_agent_generation_config,
    get_agent_max_output_tokens,
    get_agent_model,
    get_fallback_models,
    get_unified_max_tool_steps,
    resolve_agent_context_window_tokens,
)
from ..llm_compat import AI, LLMMessage, ToolInvoker, ToolResult
from ..main_request_models import (
    MainRequestOutput,
    MainRequestResult,
    MainRequestTimelineItem,
)
from ..meta_tools import CALL_PLUGIN_TOOL_NAME
from ..native_route import NativeRouteDecision
from ..provider_failover import request_with_failover
from ..route_text import normalize_message_text, normalize_reply_text
from ..runtime_result import _first_route, _timeline_memory_text
from ..turn_runtime import estimate_text_tokens
from .core import (
    UNIFIED_CHAT_TOOL_SCOPE,
    AgentObservation,
    AgentResult,
    UnifiedChatRequest,
    estimate_prompt_tokens,
    fallback_text,
    provider_adapter_for,
)

if TYPE_CHECKING:
    from ..llm_compat import LLMResponse
    from ..provider_capability import ProviderCapabilityAdapter

_UNIFIED_STAGE = "unified_chat_agent"
_TOOL_ARGS_CLIP = 500
_CHAT_PROTOCOL_MARGIN_TOKENS = 2_048

_TOOL_POLICY_PROMPT = """<plugin_tooling>
你可以调用本机器人的插件功能来完成用户请求。可用功能目录见 <plugin_catalog>，\
条目格式：command_id | 命令头(别名) | 描述 [前置条件]。
- 用户明确要求执行、查询、生成某项功能时：从目录选定命令，先用 get_command_details \
获取参数定义，再用 call_plugin 执行。目录里没有合适条目或插件标注"目录已折叠"时，\
用 search_plugins 检索。
- call_plugin 的 task_text 必须是用户对该任务的原话片段；一次调用只执行一个任务，\
多个任务分多次调用。
- 纯聊天、观点交流、常识问答直接回复，不要调用工具；谈论某个功能本身不等于要执行它。
- 执行结果 ok=true 且 messages_sent 非空表示插件输出已直接发给用户：此时最终回复输出\
空内容，除非确有必要补一句简短说明。
- 执行失败时，根据错误信息和返回的参数定义修正后最多重试一次；仍失败就用一句话告诉\
用户原因。
- 目录里没有的功能不要编造，直接说明做不到。
</plugin_tooling>"""


class UnifiedChatAgent:
    """Boundary for the merged chat + plugin-invocation turn."""

    async def run(self, request: UnifiedChatRequest) -> AgentResult:
        started = time.perf_counter()
        trace_id = f"unified-{int(time.time() * 1000):x}"
        ai = AI(session_id=f"chatinter-unified:{request.session_key or 'global'}")
        model_name = get_agent_model("chat")
        generation_config = build_agent_generation_config("chat")
        tools = dict(request.tools or {})
        messages = _augment_system_message(
            list(request.messages),
            catalog_text=request.catalog_text,
            has_tools=bool(tools),
        )
        messages = _fit_chat_messages(
            messages,
            max_input_tokens=resolve_agent_context_window_tokens("chat", model_name),
            output_reserve_tokens=get_agent_max_output_tokens("chat"),
        )
        timeline: list[MainRequestTimelineItem] = [
            MainRequestTimelineItem(
                role="user",
                kind="current_user",
                content=request.message_text,
            ),
        ]
        invoker = ToolInvoker()
        adapter_holder: dict[str, ProviderCapabilityAdapter] = {
            "adapter": provider_adapter_for(model_name),
        }
        command_tool_results: list[ToolResult] = []
        final_text = ""
        max_steps = get_unified_max_tool_steps()
        tool_steps = 0
        loop_budget = max_steps + 1
        while loop_budget > 0:
            loop_budget -= 1
            allow_tools = bool(tools) and tool_steps < max_steps
            response = await self._request(
                ai=ai,
                model_name=model_name,
                generation_config=generation_config,
                messages=messages,
                tools=tools if allow_tools else None,
                adapter_holder=adapter_holder,
                request=request,
                trace_id=trace_id,
            )
            adapter = adapter_holder["adapter"]
            tool_calls = (
                adapter.tool_calls_for_execution(list(response.tool_calls or []))
                if allow_tools
                else []
            )
            if not tool_calls:
                final_text = normalize_reply_text(str(response.text or ""))
                break
            tool_steps += 1
            messages.append(
                LLMMessage.assistant_tool_calls(
                    list(tool_calls),
                    content=str(response.text or ""),
                )
            )
            for call in tool_calls:
                function_name = str(call.function.name or "")
                timeline.append(
                    MainRequestTimelineItem(
                        role="assistant",
                        kind="tool_call",
                        tool_name=function_name,
                        metadata={"arguments": _safe_arguments(call)},
                    )
                )
                _call, result = await invoker.execute_tool_call(call, tools)
                if function_name == CALL_PLUGIN_TOOL_NAME:
                    command_tool_results.append(result)
                messages.append(
                    adapter.tool_result_message(
                        tool_call=call,
                        function_name=function_name,
                        result=result.output,
                    )
                )
                timeline.append(
                    MainRequestTimelineItem(
                        role="tool",
                        kind="tool_result",
                        tool_name=function_name,
                        metadata={"output": _compact_output(result)},
                    )
                )

        executions = (
            list(request.command_context.executions)
            if request.command_context is not None
            else []
        )
        handled_by_tools = bool(executions)
        success_any = any(item.success for item in executions)
        if request.report.final_reason == "init":
            request.report.finalize(reason="unified_chat", stage=_UNIFIED_STAGE)
        if not handled_by_tools and not final_text:
            final_text = fallback_text("")
        should_send = bool(final_text)
        result = MainRequestResult(
            decision=NativeRouteDecision(
                action="execute" if handled_by_tools else "chat",
                confidence=0.9 if handled_by_tools else 0.85,
                reason="unified_chat_agent",
            ),
            route_result=_first_route(executions),
            report=request.report,
            executions=tuple(executions),
            tool_results=tuple(command_tool_results),
            timeline=tuple(timeline),
            output=MainRequestOutput(
                final_text=final_text,
                memory_text=_timeline_memory_text(timeline, fallback=final_text),
                should_send=should_send,
                outcome=(
                    "chat_completed"
                    if not handled_by_tools
                    else "tool_completed"
                    if success_any
                    else "tool_failed"
                ),
                feedback_kind=(
                    "chat_completed"
                    if not handled_by_tools
                    else "tool_completed"
                    if success_any
                    else "tool_failed"
                ),
                record_chat_feedback=not handled_by_tools,
                observation_reason=(
                    "route_success"
                    if success_any
                    else "reroute_failed"
                    if handled_by_tools
                    else "chat_completed"
                ),
            ),
        )
        return AgentResult(
            agent_kind="unified_chat",
            main_result=result,
            observations=(
                AgentObservation(
                    kind="unified_tool_loop" if tool_steps else "unified_chat_only",
                    status="ok",
                    metadata={
                        "tool_steps": tool_steps,
                        "executions": len(executions),
                    },
                ),
            ),
            tool_scope=UNIFIED_CHAT_TOOL_SCOPE,
            elapsed_ms=max(int((time.perf_counter() - started) * 1000), 0),
        )

    async def _request(
        self,
        *,
        ai: AI,
        model_name: str,
        generation_config: Any,
        messages: list[LLMMessage],
        tools: dict[str, Any] | None,
        adapter_holder: dict[str, "ProviderCapabilityAdapter"],
        request: UnifiedChatRequest,
        trace_id: str,
    ) -> "LLMResponse":
        estimated_prompt_tokens = estimate_prompt_tokens(messages)

        async def _do_request(model: str | None) -> "LLMResponse":
            candidate_adapter = provider_adapter_for(model or model_name)
            adapter_holder["adapter"] = candidate_adapter
            prepared = candidate_adapter.prepare_model_request(
                messages=messages,
                tools=tools,
                tool_choice="auto" if tools else None,
                generation_config=generation_config,
            )
            return await ai.generate_internal(
                prepared.messages,
                model=model,
                config=prepared.generation_config,
                tools=cast(Any, prepared.tools),
                tool_choice=prepared.tool_choice,
                timeout=float(CHAT_RESPONSE_TIMEOUT_SECONDS),
            )

        outcome = await request_with_failover(
            primary_model=model_name,
            fallback_models=get_fallback_models(model_name),
            request_fn=_do_request,
            trace_id=trace_id,
        )
        if request.budget_controller is not None:
            usage_info = getattr(outcome.response, "usage_info", None)
            cached_prompt_tokens = 0
            if isinstance(usage_info, dict) and usage_info:
                usage = parse_usage_info(usage_info)
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
                cached_prompt_tokens = int(
                    getattr(usage, "prompt_cache_hit_tokens", 0) or 0
                )
            else:
                prompt_tokens = estimated_prompt_tokens
                completion_tokens = estimate_text_tokens(
                    str(outcome.response.text or "")
                )
            request.budget_controller.record_model_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_prompt_tokens=cached_prompt_tokens,
            )
        return outcome.response


def _fit_chat_messages(
    messages: list[LLMMessage],
    *,
    max_input_tokens: int,
    output_reserve_tokens: int,
) -> list[LLMMessage]:
    """Drop only complete old dialogue groups when a chat prompt is oversized."""

    fitted = list(messages)
    limit = max(
        int(max_input_tokens)
        - max(int(output_reserve_tokens), 0)
        - _CHAT_PROTOCOL_MARGIN_TOKENS,
        1,
    )
    if estimate_prompt_tokens(fitted) <= limit or len(fitted) <= 2:
        return fitted

    stable = [fitted[0]]
    current = [fitted[-1]]
    groups: list[list[LLMMessage]] = []
    for message in fitted[1:-1]:
        role = str(getattr(message, "role", "") or "")
        if role in {"system", "user"} or not groups:
            groups.append([message])
        else:
            groups[-1].append(message)

    kept: list[list[LLMMessage]] = []
    for group in reversed(groups):
        recent = [item for part in reversed(kept) for item in part]
        candidate = stable + group + recent + current
        if estimate_prompt_tokens(candidate) > limit:
            break
        kept.append(group)

    return stable + [item for group in reversed(kept) for item in group] + current


def _augment_system_message(
    messages: list[LLMMessage],
    *,
    catalog_text: str,
    has_tools: bool,
) -> list[LLMMessage]:
    if not has_tools or not catalog_text:
        return messages
    sections = [
        _TOOL_POLICY_PROMPT,
        f"<plugin_catalog>\n{catalog_text}\n</plugin_catalog>",
    ]
    addition = "\n\n".join(sections)
    if messages and messages[0].role == "system":
        base = str(messages[0].content or "")
        return [
            LLMMessage.system(f"{base}\n\n{addition}" if base else addition),
            *messages[1:],
        ]
    return [LLMMessage.system(addition), *messages]


def _safe_arguments(call: Any) -> Any:
    raw = getattr(getattr(call, "function", None), "arguments", "") or ""
    if isinstance(raw, dict):
        parsed: Any = raw
    else:
        try:
            parsed = json.loads(str(raw) or "{}")
        except Exception:
            parsed = str(raw)
    text = json.dumps(parsed, ensure_ascii=False, default=str)
    if len(text) > _TOOL_ARGS_CLIP:
        return {"_clipped": text[:_TOOL_ARGS_CLIP]}
    return parsed


def _compact_output(result: ToolResult) -> dict[str, Any]:
    output = result.output
    if isinstance(output, dict):
        return output
    return {"value": normalize_message_text(str(output or ""))[:_TOOL_ARGS_CLIP]}


__all__ = ["UnifiedChatAgent"]
