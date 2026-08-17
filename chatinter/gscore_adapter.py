"""GScore bridge client for ChatInter mixed-chat routing and execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import hmac
import json
import time
from typing import Any, Literal

import aiohttp

from zhenxun.services.log import logger

from .config import get_gscore_bridge_config
from .llm_compat import RunContext, ToolDefinition, ToolResult
from .route_text import normalize_message_text
from .token_compat import estimate_text_tokens

_API_PREFIX = "/api/chatinter-bridge/v1"
_ROUTE_TIMEOUT_SECONDS = 2.0
_CAPABILITY_TIMEOUT_SECONDS = 3.0
_EXECUTE_TIMEOUT_SECONDS = 4.0
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_CAPABILITY_SCHEMA_TOKEN_BUDGET = 24_000
_LEGAL_TRIGGER_TYPES = frozenset(
    {"command", "fullmatch", "keyword", "prefix", "regex", "suffix"}
)

GScoreRouteDisposition = Literal[
    "claimed",
    "unmatched",
    "interactive",
    "blocked",
    "unknown",
    "disabled",
]


@dataclass(frozen=True, slots=True)
class GScoreTriggerPattern:
    trigger_type: str
    keyword: str
    prefix: str = ""
    to_me: bool = False

    @property
    def command(self) -> str:
        return f"{self.prefix}{self.keyword}"


@dataclass(frozen=True, slots=True)
class GScoreCapability:
    capability_id: str
    name: str
    description: str = ""
    plugin: str = ""
    metadata_sources: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    parameters: dict[str, Any] | None = None
    context_tags: tuple[str, ...] = ()
    capability_domain: str = ""
    trigger_patterns: tuple[GScoreTriggerPattern, ...] = ()
    trigger_type: str = ""
    trigger_keyword: str = ""
    trigger_prefix: str = ""
    trigger_to_me: bool = False
    command_starts: tuple[str, ...] = ()

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        command_starts: tuple[str, ...] = (),
    ) -> GScoreCapability | None:
        if not isinstance(payload, dict):
            return None
        capability_id = normalize_message_text(
            str(payload.get("capability_id") or payload.get("id") or "")
        )
        if not capability_id:
            return None
        plugin_payload = payload.get("plugin")
        plugin_mapping = plugin_payload if isinstance(plugin_payload, dict) else {}
        service_payload = payload.get("service")
        service_mapping = service_payload if isinstance(service_payload, dict) else {}
        trigger_payload = payload.get("trigger")
        trigger_payload = trigger_payload if isinstance(trigger_payload, dict) else {}
        trigger_patterns = _trigger_patterns_from_payload(trigger_payload)
        primary_trigger = (
            trigger_patterns[0] if trigger_patterns else GScoreTriggerPattern("", "")
        )
        plugin_name = normalize_message_text(
            str(
                plugin_mapping.get("name")
                or (plugin_payload if isinstance(plugin_payload, str) else "")
            )
        )
        service_name = normalize_message_text(
            str(
                service_mapping.get("name")
                or (service_payload if isinstance(service_payload, str) else "")
            )
        )
        parameters = payload.get("input_schema") or payload.get("parameters")
        metadata_sources = _accepted_metadata_sources(
            payload,
            has_trigger=bool(trigger_patterns),
        )
        return cls(
            capability_id=capability_id,
            name=normalize_message_text(
                str(
                    payload.get("name")
                    or service_name
                    or primary_trigger.keyword
                    or capability_id
                )
            ),
            description=normalize_message_text(
                str(payload.get("description") or service_name or "")
            ),
            plugin=plugin_name,
            metadata_sources=metadata_sources,
            aliases=_string_tuple(
                payload.get("aliases") or plugin_mapping.get("aliases")
            ),
            examples=_string_tuple(payload.get("examples")),
            parameters=dict(parameters) if isinstance(parameters, dict) else None,
            context_tags=_string_tuple(
                payload.get("context_tags") or payload.get("tags")
            ),
            capability_domain=normalize_message_text(
                str(payload.get("capability_domain") or payload.get("domain") or "")
            ),
            trigger_patterns=trigger_patterns,
            trigger_type=primary_trigger.trigger_type,
            trigger_keyword=primary_trigger.keyword,
            trigger_prefix=primary_trigger.prefix,
            trigger_to_me=primary_trigger.to_me,
            command_starts=command_starts,
        )


@dataclass(frozen=True, slots=True)
class GScoreRouteResult:
    disposition: GScoreRouteDisposition
    revision: str = ""
    matches: tuple[str, ...] = ()
    reason: str = ""

    @property
    def suppress_chatinter(self) -> bool:
        return self.disposition in {
            "claimed",
            "interactive",
            "blocked",
        }


class GScoreBridgeError(RuntimeError):
    def __init__(self, message: str, *, uncertain: bool = False) -> None:
        super().__init__(message)
        self.uncertain = uncertain


class GScoreExecutionTool:
    name = "gscore_execute"
    chatinter_plugin_tool_kind = "gscore"

    def __init__(
        self,
        adapter: GScoreAdapter,
        capabilities: tuple[GScoreCapability, ...],
        message_payload: dict[str, Any],
        ws_bot_id: str,
        revision: str,
        source_request_id: str,
    ) -> None:
        self._adapter = adapter
        self._capabilities = _merge_capabilities(capabilities)
        self._capability_ids = frozenset(
            item.capability_id for item in self._capabilities
        )
        self._message_payload = message_payload
        self._ws_bot_id = ws_bot_id
        self._revision = revision
        self._source_request_id = source_request_id

    @property
    def capability_count(self) -> int:
        return len(self._capabilities)

    @property
    def candidate_context(self) -> str:
        return "GScore 外部插件候选能力：\n" + "\n".join(
            _capability_card(item) for item in self._capabilities
        )

    async def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "执行本轮候选列表中的 GScore 外部插件能力。"
                "仅在用户明确需要候选能力时调用，不要把普通聊天改写成插件操作。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "capability_id": {
                        "type": "string",
                        "description": "本轮 GScore 候选能力卡中的稳定 capability_id",
                    },
                    "command_text": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            "按能力卡触发器构造的完整 GScore 命令文本；"
                            "必须能由所选 trigger 重新匹配"
                        ),
                    },
                },
                "required": ["capability_id", "command_text"],
                "additionalProperties": False,
            },
        )

    async def execute(
        self,
        context: RunContext | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        del context
        capability_id = normalize_message_text(str(kwargs.get("capability_id") or ""))
        command_text = normalize_message_text(str(kwargs.get("command_text") or ""))
        if capability_id not in self._capability_ids or not command_text:
            return ToolResult(
                output={
                    "status": "invalid_arguments",
                    "plugin_execution": False,
                    "executed": False,
                },
                is_error=True,
                is_retryable=False,
            )
        execute_payload = {
            "request_id": _execution_request_id(
                self._source_request_id,
                capability_id,
                command_text,
            ),
            "ws_bot_id": self._ws_bot_id,
            "message": self._message_payload,
            "capability_id": capability_id,
            "revision": self._revision,
            "command_text": command_text,
        }
        try:
            response = await self._adapter.execute(execute_payload)
        except asyncio.TimeoutError as exc:
            logger.warning(f"ChatInter GScore execute result unknown: {exc}")
            return _external_delivery_result("unknown", uncertain=True)
        except GScoreBridgeError as exc:
            logger.warning(f"ChatInter GScore execute failed: {exc}")
            if exc.uncertain:
                return _external_delivery_result("unknown", uncertain=True)
            return ToolResult(
                output={
                    "status": "unavailable",
                    "plugin_execution": False,
                    "executed": False,
                },
                is_error=True,
                is_retryable=False,
            )

        disposition = normalize_message_text(
            str(response.get("disposition") or response.get("status") or "unknown")
        ).casefold()
        if disposition in {"accepted", "duplicate"}:
            delivery_state, delivery_observed = _delivery_observation(response)
            return _external_delivery_result(
                disposition,
                submitted=True,
                uncertain=not delivery_observed,
                delivery_state=delivery_state,
                delivery_observed=delivery_observed,
            )
        if disposition == "unknown":
            return _external_delivery_result("unknown", uncertain=True)
        return ToolResult(
            output={
                "status": disposition or "rejected",
                "plugin_execution": False,
                "executed": False,
            },
            is_error=disposition not in {"rejected", "blocked", "unavailable"},
            is_retryable=False,
        )


class GScoreAdapter:
    def __init__(self) -> None:
        self._capabilities: tuple[GScoreCapability, ...] = ()
        self._capabilities_loaded = False
        self._revision = ""
        self._capability_lock = asyncio.Lock()
        self._session_lock = asyncio.Lock()
        self._http_session: aiohttp.ClientSession | None = None
        self._revision_epoch = 0
        self._config_fingerprint = ""

    @property
    def enabled(self) -> bool:
        config = get_gscore_bridge_config()
        return bool(config["enabled"] and config["url"] and config["secret"])

    def _sync_configuration(self) -> tuple[bool, bool]:
        config = get_gscore_bridge_config()
        enabled = bool(config["enabled"] and config["url"] and config["secret"])
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "enabled": enabled,
                    "url": config["url"],
                    "secret": config["secret"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        changed = fingerprint != self._config_fingerprint
        if changed:
            self._config_fingerprint = fingerprint
            self._capabilities = ()
            self._capabilities_loaded = False
            self._revision = ""
            self._revision_epoch += 1
        return enabled, changed

    async def _prepare(self) -> bool:
        enabled, changed = self._sync_configuration()
        if changed:
            await self.close()
        return enabled

    async def route_turn(self, frame: Any) -> GScoreRouteResult:
        if not await self._prepare() or not _mixed_tools_allowed(frame):
            return GScoreRouteResult("disabled")
        message_payload = build_gscore_event_payload(frame)
        if not message_payload:
            return GScoreRouteResult("disabled")
        payload = {
            "request_id": _route_request_id(
                frame,
                str(message_payload.get("msg_id", "") or ""),
            ),
            "ws_bot_id": _gscore_ws_bot_id(),
            "message": message_payload,
        }
        try:
            response = await self._request_json(
                "POST",
                "/route",
                payload,
                timeout_seconds=_ROUTE_TIMEOUT_SECONDS,
            )
        except (GScoreBridgeError, asyncio.TimeoutError) as exc:
            logger.warning(f"ChatInter GScore route result unknown: {exc}")
            return GScoreRouteResult("unknown", reason=type(exc).__name__)

        disposition = normalize_message_text(
            str(response.get("disposition") or "unknown")
        ).casefold()
        if disposition not in {
            "claimed",
            "unmatched",
            "interactive",
            "blocked",
            "unknown",
        }:
            disposition = "unknown"
        revision = normalize_message_text(str(response.get("revision") or ""))
        matches = _string_tuple(response.get("matches"))
        self._observe_revision(revision)
        return GScoreRouteResult(
            disposition=disposition,
            revision=revision,
            matches=matches,
            reason=normalize_message_text(str(response.get("reason") or "")),
        )

    async def build_tool(self, frame: Any) -> GScoreExecutionTool | None:
        if not await self._prepare() or not _mixed_tools_allowed(frame):
            return None
        message_payload = build_gscore_event_payload(frame)
        if not message_payload:
            return None
        capabilities = await self.get_capabilities()
        if not capabilities:
            return None
        capabilities = _select_capabilities(
            capabilities,
            _message_text(message_payload),
            token_budget=_CAPABILITY_SCHEMA_TOKEN_BUDGET,
        )
        if not capabilities:
            return None
        return GScoreExecutionTool(
            self,
            capabilities,
            message_payload,
            _gscore_ws_bot_id(),
            self._revision,
            _route_request_id(
                frame,
                str(message_payload.get("msg_id", "") or ""),
            ),
        )

    async def get_capabilities(self) -> tuple[GScoreCapability, ...]:
        if not await self._prepare():
            return ()
        if self._capabilities_loaded:
            return self._capabilities
        async with self._capability_lock:
            if self._capabilities_loaded:
                return self._capabilities
            for _attempt in range(2):
                observed_epoch = self._revision_epoch
                try:
                    response = await self._request_json(
                        "GET",
                        "/capabilities",
                        None,
                        timeout_seconds=_CAPABILITY_TIMEOUT_SECONDS,
                    )
                except (GScoreBridgeError, asyncio.TimeoutError) as exc:
                    logger.warning(
                        f"ChatInter GScore capability discovery failed: {exc}"
                    )
                    return ()
                revision = normalize_message_text(str(response.get("revision") or ""))
                if (
                    self._revision_epoch != observed_epoch
                    and self._revision
                    and revision != self._revision
                ):
                    continue
                items = response.get("capabilities")
                parsed = tuple(
                    item
                    for payload in items or ()
                    if (item := GScoreCapability.from_payload(payload)) is not None
                )
                capabilities = _merge_capabilities(parsed)
                self._observe_revision(revision)
                self._capabilities = capabilities
                self._capabilities_loaded = True
                return self._capabilities
            return ()

    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request_json(
            "POST",
            "/execute",
            payload,
            timeout_seconds=_EXECUTE_TIMEOUT_SECONDS,
        )

    def _observe_revision(self, revision: str) -> None:
        if revision and self._revision and revision != self._revision:
            self._capabilities = ()
            self._capabilities_loaded = False
        if revision and revision != self._revision:
            self._revision_epoch += 1
        if revision:
            self._revision = revision

    async def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        *,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        config = get_gscore_bridge_config()
        if not config["enabled"] or not config["url"] or not config["secret"]:
            raise GScoreBridgeError("bridge is not configured")
        body = (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            if payload is not None
            else b""
        )
        timestamp = str(int(time.time()))
        signature = hmac.new(
            str(config["secret"]).encode(),
            timestamp.encode() + b"." + body,
            hashlib.sha256,
        ).hexdigest()
        headers = {
            "Accept": "application/json",
            "X-ChatInter-Timestamp": timestamp,
            "X-ChatInter-Signature": signature,
        }
        if body:
            headers["Content-Type"] = "application/json"
        url = f"{config['url']}{_API_PREFIX}{path}"
        side_effecting = method == "POST" and path == "/execute"
        try:
            session = await self._get_http_session()
            async with session.request(
                method,
                url,
                data=body if body else None,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout_seconds),
            ) as response:
                raw = bytearray()
                async for chunk in response.content.iter_chunked(64 * 1024):
                    if len(raw) + len(chunk) > _MAX_RESPONSE_BYTES:
                        raise GScoreBridgeError(
                            "bridge response is too large",
                            uncertain=side_effecting,
                        )
                    raw.extend(chunk)
                if response.status >= 400:
                    raise GScoreBridgeError(
                        f"bridge returned HTTP {response.status}",
                        uncertain=side_effecting and response.status >= 500,
                    )
        except asyncio.TimeoutError:
            raise
        except aiohttp.ClientConnectorError as exc:
            raise GScoreBridgeError(str(exc)) from exc
        except aiohttp.ServerDisconnectedError as exc:
            raise GScoreBridgeError(
                str(exc),
                uncertain=side_effecting,
            ) from exc
        except aiohttp.ClientError as exc:
            raise GScoreBridgeError(
                str(exc),
                uncertain=side_effecting,
            ) from exc
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GScoreBridgeError(
                "bridge returned invalid JSON",
                uncertain=side_effecting,
            ) from exc
        if not isinstance(decoded, dict):
            raise GScoreBridgeError(
                "bridge response must be an object",
                uncertain=side_effecting,
            )
        data = decoded.get("data") if "data" in decoded else decoded
        if not isinstance(data, dict):
            raise GScoreBridgeError(
                "bridge response data must be an object",
                uncertain=side_effecting,
            )
        return data

    async def _get_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is not None and not self._http_session.closed:
            return self._http_session
        async with self._session_lock:
            if self._http_session is None or self._http_session.closed:
                self._http_session = aiohttp.ClientSession(
                    connector=aiohttp.TCPConnector(limit=8, ttl_dns_cache=300),
                )
            return self._http_session

    async def close(self) -> None:
        session = self._http_session
        self._http_session = None
        if session is not None and not session.closed:
            await session.close()


def build_gscore_event_payload(frame: Any) -> dict[str, Any]:
    event_context = getattr(frame, "event_context", None)
    if event_context is None:
        return {}
    event = getattr(frame, "event", None)
    event_id = normalize_message_text(str(getattr(event_context, "event_id", "") or ""))
    payload = {
        "bot_id": _event_adapter_id(event),
        "bot_self_id": str(getattr(event_context, "bot_id", "") or ""),
        "msg_id": event_id,
        "user_type": "group" if getattr(event_context, "group_id", None) else "direct",
        "group_id": getattr(event_context, "group_id", None),
        "user_id": str(getattr(event_context, "user_id", "") or ""),
        "sender": _event_sender(event),
        "user_pm": _event_user_pm(frame, event),
        "content": _event_content(frame),
    }
    return (
        payload
        if payload["bot_id"] and payload["user_id"] and payload["content"]
        else {}
    )


def get_gscore_adapter() -> GScoreAdapter:
    return _GSCORE_ADAPTER


def _external_delivery_result(
    status: str,
    *,
    submitted: bool = False,
    uncertain: bool = True,
    delivery_state: str = "unknown",
    delivery_observed: bool = False,
) -> ToolResult:
    output: dict[str, Any] = {
        "status": status,
        "plugin_execution": True,
        "submitted": submitted,
        "executed": delivery_observed,
        "execution_uncertain": uncertain,
        "external_delivery": delivery_observed,
        "delivery_observed": delivery_observed,
        "delivery_state": delivery_state,
    }
    if delivery_observed:
        output["delivery_owner"] = "gscore"
    return ToolResult(
        output=output,
        is_error=not submitted,
        is_retryable=False,
    )


def _delivery_observation(response: dict[str, Any]) -> tuple[str, bool]:
    state = normalize_message_text(str(response.get("delivery_state") or "unknown"))
    state = state.casefold()
    observed = response.get("delivery_observed") is True or state in {
        "complete",
        "completed",
        "delivered",
        "observed",
        "sent",
    }
    return (state if state else "unknown"), observed


def _capability_card(capability: GScoreCapability) -> str:
    fields = [f"[{capability.capability_id}]", capability.name]
    if capability.description:
        fields.append(capability.description)
    if capability.aliases:
        fields.append("aliases=" + ", ".join(capability.aliases))
    if capability.examples:
        fields.append("examples=" + " | ".join(capability.examples))
    fields.append(
        "triggers="
        + " | ".join(
            _trigger_pattern_projection(pattern)
            for pattern in capability.trigger_patterns
        )
    )
    fields.append(
        "command_text=填写能匹配上述任一 trigger 的完整实际命令；"
        "regex 需填写实际匹配文本，不要填写正则本身"
    )
    return " ; ".join(fields)


def _select_capabilities(
    capabilities: tuple[GScoreCapability, ...],
    message_text: str,
    *,
    token_budget: int,
) -> tuple[GScoreCapability, ...]:
    query = _search_normalize(message_text)
    if not query or token_budget <= 0:
        return ()
    ranked = sorted(
        (
            (_capability_score(capability, query), capability)
            for capability in capabilities
        ),
        key=lambda item: (-item[0], item[1].capability_id),
    )
    selected: list[GScoreCapability] = []
    used_tokens = 0
    for score, capability in ranked:
        if score <= 0:
            break
        card_tokens = estimate_text_tokens(_capability_card(capability)) + 16
        if used_tokens + card_tokens > token_budget:
            continue
        selected.append(capability)
        used_tokens += card_tokens
    return tuple(sorted(selected, key=lambda item: item.capability_id))


def _capability_score(capability: GScoreCapability, query: str) -> float:
    weighted_fields = (
        *((pattern.keyword, 5.0) for pattern in capability.trigger_patterns),
        *((pattern.command, 5.0) for pattern in capability.trigger_patterns),
        *((value, 4.0) for value in capability.aliases),
        (capability.name, 3.5),
        *((value, 3.0) for value in capability.examples),
        *((value, 2.5) for value in capability.context_tags),
        (capability.description, 2.0),
        (capability.capability_domain, 1.75),
        (capability.plugin, 1.5),
        *((value, 1.0) for value in _schema_search_fields(capability.parameters)),
    )
    fields: dict[str, float] = {}
    for value, weight in weighted_fields:
        normalized = _search_normalize(value)
        if normalized:
            fields[normalized] = max(fields.get(normalized, 0.0), weight)
    query_units = _search_units(query)
    score = 0.0
    for value, weight in fields.items():
        if value in query:
            score += weight * 2.0
        value_units = _search_units(value)
        if not value_units or not query_units:
            continue
        overlap = len(value_units & query_units)
        if overlap:
            precision = overlap / len(query_units)
            recall = overlap / len(value_units)
            score += weight * (precision * 0.65 + recall * 0.35)
    return score


def _search_normalize(value: object) -> str:
    return "".join(
        char.casefold()
        for char in normalize_message_text(str(value or ""))
        if char.isalnum()
    )


def _search_units(value: str) -> set[str]:
    if len(value) < 2:
        return {value} if value else set()
    return {value[index : index + 2] for index in range(len(value) - 1)}


def _message_text(message_payload: dict[str, Any]) -> str:
    content = message_payload.get("content")
    if not isinstance(content, list):
        return ""
    return " ".join(
        normalize_message_text(str(item.get("data") or ""))
        for item in content
        if isinstance(item, dict) and str(item.get("type") or "") == "text"
    )


def _string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, list | tuple | set | frozenset):
        values = value
    else:
        values = ()
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = normalize_message_text(str(item or ""))
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return tuple(result)


def _exact_string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, list | tuple | set | frozenset):
        values = value
    else:
        values = ()
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = normalize_message_text(str(item or ""))
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return tuple(result)


def _accepted_metadata_sources(
    payload: dict[str, Any],
    *,
    has_trigger: bool,
) -> tuple[str, ...]:
    raw_sources = _merge_string_tuples(
        (
            _string_tuple(payload.get("source") or payload.get("metadata_source")),
            _string_tuple(payload.get("metadata_sources")),
        )
    )
    accepted = _string_tuple(
        tuple(
            normalized
            for source in raw_sources
            if (normalized := _canonical_metadata_source(source))
        )
    )
    if not raw_sources and has_trigger:
        return ("trigger",)
    return accepted


def _canonical_metadata_source(source: str) -> str:
    normalized = source.casefold().replace("-", "_")
    if normalized in {"to_ai", "toai"}:
        return "to_ai"
    if "trigger" in normalized:
        return "trigger"
    return ""


def _trigger_patterns_from_payload(
    trigger: dict[str, Any],
) -> tuple[GScoreTriggerPattern, ...]:
    patterns: list[GScoreTriggerPattern] = []

    def add(mapping: dict[str, Any]) -> None:
        trigger_type = normalize_message_text(
            str(mapping.get("type") or trigger.get("type") or "")
        ).casefold()
        keyword = normalize_message_text(str(mapping.get("keyword") or ""))
        prefix = normalize_message_text(str(mapping.get("prefix") or ""))
        if trigger_type in _LEGAL_TRIGGER_TYPES and keyword:
            patterns.append(
                GScoreTriggerPattern(
                    trigger_type=trigger_type,
                    keyword=keyword,
                    prefix=prefix,
                    to_me=bool(mapping.get("to_me", trigger.get("to_me", False))),
                )
            )

    add(trigger)
    for key in ("patterns", "routes"):
        nested = trigger.get(key)
        if isinstance(nested, list | tuple):
            for item in nested:
                if isinstance(item, dict):
                    add(item)

    base_type = normalize_message_text(str(trigger.get("type") or "")).casefold()
    base_keyword = normalize_message_text(str(trigger.get("keyword") or ""))
    for prefix in _exact_string_tuple(trigger.get("prefixes")):
        add({"type": base_type, "keyword": base_keyword, "prefix": prefix})
    for command in _exact_string_tuple(trigger.get("commands")):
        if base_type == "regex":
            add({"type": base_type, "keyword": command})
            continue
        if base_keyword and command.endswith(base_keyword):
            add(
                {
                    "type": base_type,
                    "keyword": base_keyword,
                    "prefix": command[: -len(base_keyword)],
                }
            )
        else:
            add({"type": base_type, "keyword": command})
    return _merge_trigger_patterns(patterns)


def _merge_trigger_patterns(
    patterns: Any,
) -> tuple[GScoreTriggerPattern, ...]:
    unique = {
        (
            pattern.trigger_type,
            pattern.prefix,
            pattern.keyword,
            pattern.to_me,
        ): pattern
        for pattern in patterns
        if pattern.trigger_type in _LEGAL_TRIGGER_TYPES and pattern.keyword
    }
    return tuple(unique[key] for key in sorted(unique))


def _trigger_pattern_projection(pattern: GScoreTriggerPattern) -> str:
    if pattern.trigger_type == "regex":
        return (
            f"regex(prefix={pattern.prefix},pattern={pattern.keyword},"
            f"to_me={str(pattern.to_me).lower()})"
        )
    return (
        f"{pattern.trigger_type}(pattern={pattern.command},"
        f"to_me={str(pattern.to_me).lower()})"
    )


def _schema_search_fields(schema: dict[str, Any] | None) -> tuple[str, ...]:
    if not schema:
        return ()
    values: list[str] = []

    def add(value: object) -> None:
        if isinstance(value, str):
            text = normalize_message_text(value)
            if text:
                values.append(text)

    def walk(value: object) -> None:
        if isinstance(value, dict):
            add(value.get("title"))
            add(value.get("description"))
            properties = value.get("properties")
            if isinstance(properties, dict):
                for name, definition in properties.items():
                    add(name)
                    walk(definition)
            required = value.get("required")
            if isinstance(required, list | tuple):
                for item in required:
                    add(item)
            enum = value.get("enum")
            if isinstance(enum, list | tuple):
                for item in enum:
                    add(item)
            for key in ("items", "anyOf", "oneOf", "allOf"):
                nested = value.get(key)
                if isinstance(nested, list | tuple):
                    for item in nested:
                        walk(item)
                elif isinstance(nested, dict):
                    walk(nested)

    walk(schema)
    return _string_tuple(values)


def _merge_capabilities(
    capabilities: tuple[GScoreCapability, ...],
) -> tuple[GScoreCapability, ...]:
    grouped: dict[str, list[GScoreCapability]] = {}
    for capability in capabilities:
        grouped.setdefault(capability.capability_id, []).append(capability)
    merged = (
        _merge_capability_group(grouped[capability_id])
        for capability_id in sorted(grouped)
    )
    return tuple(
        capability
        for capability in merged
        if capability.metadata_sources and capability.trigger_patterns
    )


def _merge_capability_group(
    capabilities: list[GScoreCapability],
) -> GScoreCapability:
    ranked = sorted(
        capabilities,
        key=lambda item: -_capability_source_priority(item),
    )

    def first_text(field: str) -> str:
        return next(
            (value for item in ranked if (value := str(getattr(item, field) or ""))),
            "",
        )

    trigger_sources = tuple(
        item
        for item in ranked
        if "trigger" in item.metadata_sources and item.trigger_patterns
    )
    if not trigger_sources:
        trigger_sources = tuple(
            item
            for item in ranked
            if "to_ai" in item.metadata_sources and item.trigger_patterns
        )
    trigger_patterns = _merge_trigger_patterns(
        pattern for item in trigger_sources for pattern in item.trigger_patterns
    )
    primary_trigger = (
        trigger_patterns[0] if trigger_patterns else GScoreTriggerPattern("", "")
    )
    parameters = next(
        (item.parameters for item in ranked if item.parameters is not None),
        None,
    )
    return GScoreCapability(
        capability_id=ranked[0].capability_id,
        name=first_text("name"),
        description=first_text("description"),
        plugin=first_text("plugin"),
        metadata_sources=_merge_string_tuples(item.metadata_sources for item in ranked),
        aliases=_merge_string_tuples(item.aliases for item in ranked),
        examples=_merge_string_tuples(item.examples for item in ranked),
        parameters=parameters,
        context_tags=_merge_string_tuples(item.context_tags for item in ranked),
        capability_domain=first_text("capability_domain"),
        trigger_patterns=trigger_patterns,
        trigger_type=primary_trigger.trigger_type,
        trigger_keyword=primary_trigger.keyword,
        trigger_prefix=primary_trigger.prefix,
        trigger_to_me=primary_trigger.to_me,
        command_starts=_merge_string_tuples(item.command_starts for item in ranked),
    )


def _merge_string_tuples(values: Any) -> tuple[str, ...]:
    return _string_tuple(tuple(item for group in values for item in group))


def _metadata_source_priority(source: str) -> int:
    normalized = source.casefold().replace("-", "_")
    if normalized in {"to_ai", "toai"}:
        return 20
    if "trigger" in normalized:
        return 10
    return 0


def _capability_source_priority(capability: GScoreCapability) -> int:
    return max(
        (_metadata_source_priority(source) for source in capability.metadata_sources),
        default=0,
    )


def _route_request_id(frame: Any, event_id: str) -> str:
    parts = [
        str(getattr(frame, "bot_id", "") or ""),
        str(getattr(frame, "group_id", "") or ""),
        str(getattr(frame, "user_id", "") or ""),
        event_id,
    ]
    if not event_id:
        parts.extend(
            (
                str(getattr(frame, "turn_generation", 0) or 0),
                f"{float(getattr(frame, 'started_at', 0.0) or 0.0):.9f}",
            )
        )
    stable = "|".join(parts)
    return hashlib.sha256(stable.encode()).hexdigest()[:32]


def _execution_request_id(
    source_request_id: str,
    capability_id: str,
    command_text: str,
) -> str:
    stable = json.dumps(
        {
            "source_request_id": source_request_id,
            "capability_id": capability_id,
            "command_text": command_text,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(stable.encode()).hexdigest()[:32]


def _gscore_ws_bot_id() -> str:
    try:
        from nonebot import get_driver

        return str(getattr(get_driver().config, "gsuid_core_botid", "NoneBot2") or "")
    except Exception:
        return "NoneBot2"


def _mixed_tools_allowed(frame: Any) -> bool:
    return bool(
        getattr(frame, "allow_plugin_tools", False)
        and str(getattr(frame, "scenario", "") or "") != "superuser_agent"
    )


def _event_adapter_id(event: Any) -> str:
    module = str(getattr(getattr(event, "__class__", None), "__module__", "") or "")
    class_name = str(getattr(getattr(event, "__class__", None), "__name__", "") or "")
    if ".onebot.v12." in module:
        return "onebot_v12"
    if ".onebot.v11." in module:
        return "onebot"
    if ".qq." in module:
        if class_name in {
            "C2CMessageCreateEvent",
            "GroupAtMessageCreateEvent",
            "GroupMessageCreateEvent",
        }:
            return "qqgroup"
        return "qqguild"
    parts = module.split(".")
    if len(parts) > 3 and parts[:2] == ["nonebot", "adapters"]:
        return parts[2]
    return ""


def _event_sender(event: Any) -> dict[str, Any]:
    sender = getattr(event, "sender", None)
    if sender is None:
        return {}
    if isinstance(sender, dict):
        source = sender
    elif hasattr(sender, "model_dump"):
        source = sender.model_dump()
    elif hasattr(sender, "dict"):
        source = sender.dict()
    else:
        return {}
    allowed = {"user_id", "nickname", "card", "role", "sex", "age"}
    return {key: value for key, value in source.items() if key in allowed}


def _event_user_pm(frame: Any, event: Any) -> int:
    if bool(getattr(frame, "is_superuser", False)):
        return 1
    sender = _event_sender(event)
    role = str(sender.get("role", "") or "").casefold()
    if role == "owner":
        return 2
    if role in {"admin", "administrator"}:
        return 3
    return 6


def _host_command_starts() -> tuple[str, ...]:
    try:
        from nonebot import get_driver

        raw = getattr(get_driver().config, "command_start", ())
    except Exception:
        return ()
    values = raw if isinstance(raw, str | list | tuple | set | frozenset) else ()
    if isinstance(values, str):
        values = (values,)
    return tuple(str(item) for item in values if str(item))


def _strip_host_command_start(text: Any) -> Any:
    value = str(text or "")
    stripped = value.strip()
    for start in _host_command_starts():
        if stripped.startswith(start):
            return stripped[len(start) :]
    return text


def _event_content(frame: Any) -> list[dict[str, Any]]:
    event = getattr(frame, "event", None)
    try:
        message = event.get_message() if event is not None else None
    except Exception:
        message = None
    content: list[dict[str, Any]] = []
    if message is not None:
        for index, segment in enumerate(message):
            segment_type = str(getattr(segment, "type", "") or "")
            data = getattr(segment, "data", {})
            data = data if isinstance(data, dict) else {}
            value: Any = None
            if segment_type == "text":
                value = data.get("text")
                if index in {0, 1}:
                    value = _strip_host_command_start(value)
            elif segment_type == "at":
                value = data.get("qq") or data.get("user_id") or data.get("target")
            elif segment_type in {"image", "record", "video"}:
                value = data.get("url") or data.get("file")
            elif segment_type == "reply":
                value = data.get("id") or data.get("message_id")
            elif segment_type == "file":
                file_name = data.get("name") or data.get("file_name") or "file"
                file_value = data.get("url") or data.get("file") or data.get("id")
                value = f"{file_name}|{file_value}" if file_value else None
            if value is not None and str(value).strip():
                content.append({"type": segment_type, "data": value})
    for item in _reply_image_content(event):
        if item not in content:
            content.append(item)
    if not content:
        text = normalize_message_text(
            str(
                getattr(frame, "route_message", "")
                or getattr(frame, "current_message", "")
            )
        )
        if text:
            content.append({"type": "text", "data": text})
    return content


def _reply_image_content(event: Any) -> list[dict[str, Any]]:
    reply = getattr(event, "reply", None)
    message = getattr(reply, "message", None)
    if message is None:
        return []
    result: list[dict[str, Any]] = []
    for segment in message:
        if str(getattr(segment, "type", "") or "") != "image":
            continue
        data = getattr(segment, "data", {})
        data = data if isinstance(data, dict) else {}
        value = data.get("url") or data.get("file")
        if value is not None and str(value).strip():
            result.append({"type": "image", "data": value})
    return result


def _cached_native_match(
    capabilities: tuple[GScoreCapability, ...],
    message: dict[str, Any],
) -> bool:
    text = "".join(
        str(item.get("data") or "").strip()
        for item in message.get("content", ())
        if isinstance(item, dict) and item.get("type") == "text"
    )
    if not text:
        return False
    is_tome = message.get("user_type") == "direct" or any(
        isinstance(item, dict)
        and item.get("type") == "at"
        and str(item.get("data") or "") == str(message.get("bot_self_id") or "")
        for item in message.get("content", ())
    )
    for capability in capabilities:
        for pattern in capability.trigger_patterns:
            if pattern.to_me and not is_tome:
                continue
            prefix = pattern.prefix
            keyword = pattern.keyword
            head = pattern.command
            if pattern.trigger_type == "fullmatch" and text == head:
                return True
            if pattern.trigger_type == "command" and text.startswith(head):
                return True
            if (
                pattern.trigger_type == "prefix"
                and text.startswith(head)
                and text != head
            ):
                return True
            if (
                pattern.trigger_type == "keyword"
                and text.startswith(prefix)
                and keyword in text
            ):
                return True
            if (
                pattern.trigger_type == "suffix"
                and text.startswith(prefix)
                and text.endswith(keyword)
                and text != head
            ):
                return True
    return False


_GSCORE_ADAPTER = GScoreAdapter()


__all__ = [
    "GScoreAdapter",
    "GScoreBridgeError",
    "GScoreCapability",
    "GScoreExecutionTool",
    "GScoreRouteResult",
    "build_gscore_event_payload",
    "get_gscore_adapter",
]
