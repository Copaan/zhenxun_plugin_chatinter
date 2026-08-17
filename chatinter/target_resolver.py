"""Command-aware target resolution and missing-context gate.

The pre-route pass can only guess target policy from the raw message.  Once a
native tool has selected a concrete command, this module re-checks the command
schema and resolves nickname targets again before rerouting to NoneBot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Literal

from nonebot.adapters import Bot

from .member_similarity import ALIAS_AMBIGUOUS_TOP, score_alias_in_message
from .native_route import NativeRouteResult
from .person_registry import normalize_alias_key
from .route_execution import (
    extract_at_tokens,
    extract_image_tokens,
    find_route_command_schema,
)
from .route_text import contains_any, normalize_message_text
from .schema_policy import CommandTargetPolicy, resolve_command_target_policy
from .target_context import (
    _TARGET_REQUIRED_ACTION_HINTS,
    build_mention_profiles,
    enrich_route_message_with_fuzzy_target,
    extract_fuzzy_target_hint,
    needs_target_for_route,
)
from .target_policy import TargetPolicy

TargetResolveStatus = Literal[
    "not_needed",
    "present",
    "resolved",
    "ambiguous",
    "missing",
    "invalid",
]

VerifiedActionTargetSource = Literal[
    "at",
    "reply",
    "alias",
    "self_nickname",
    "unknown",
]

_EXPLICIT_AT_PATTERN = re.compile(
    r"\[@(?P<bracket>[^\]\s]+)\]" r"|(?<![0-9A-Za-z_])@(?P<plain>\d{5,20})(?!\d)"
)

# Nicknames that double as everyday vocabulary.  A literal hit on one of these
# is *not* on its own proof that the speaker is naming a person: "番茄炒蛋怎么做"
# must not bind 番茄 as an execution target.  Whatever their length, these keys
# additionally require a directive/second-person signal in the message.
_AMBIENT_WORD_ALIAS_KEYS: frozenset[str] = frozenset(
    normalize_alias_key(value)
    for value in (
        # 食物
        "番茄",
        "西红柿",
        "番茄炒蛋",
        "西红柿炒蛋",
        "苹果",
        "香蕉",
        "馒头",
        "包子",
        "饺子",
        "蛋糕",
        "牛奶",
        "土豆",
        "茄子",
        "辣椒",
        "橙子",
        "柠檬",
        "西瓜",
        "草莓",
        "樱桃",
        "面条",
        "米饭",
        "火锅",
        "奶茶",
        "咖啡",
        "巧克力",
        "土豆丝",
        "红烧肉",
        "糖醋排骨",
        # 颜色
        "红色",
        "蓝色",
        "绿色",
        "黑色",
        "白色",
        "粉色",
        "紫色",
        "黄色",
        "橙色",
        # 通用词
        "可爱",
        "宝宝",
        "大佬",
        "老板",
        "机器人",
        "猫猫",
        "狗狗",
        "兔子",
        "月亮",
        "星星",
        "太阳",
        "天气",
        "游戏",
        "音乐",
        "视频",
        "图片",
        "表情",
        "头像",
    )
    if normalize_alias_key(value)
)
# Second-person / addressing signals; complements the imported causative verbs
# (_TARGET_REQUIRED_ACTION_HINTS = 给/帮/替/让/叫 from target_context).
_SECOND_PERSON_HINTS = ("你", "您", "妳")
# Characters that, immediately before an alias, mark it as a directive object.
_ALIAS_DIRECTIVE_PREFIX_CHARS = frozenset("给帮替让叫喊请@")


@dataclass(frozen=True)
class TargetResolveResult:
    status: TargetResolveStatus
    message_text: str
    mention_profiles: dict[str, dict[str, str]] = field(default_factory=dict)
    prompt: str = ""
    target_hint: str = ""
    resolved_target_ids: tuple[str, ...] = ()

    @property
    def blocked(self) -> bool:
        return self.status in {"ambiguous", "invalid"} or (
            self.status == "missing" and bool(self.prompt)
        )

    @property
    def resolved(self) -> bool:
        return self.status in {"present", "resolved"}


@dataclass(frozen=True)
class VerifiedActionTarget:
    """A single group member that may be supplied to a target-aware command."""

    user_id: str | None = None
    source: VerifiedActionTargetSource = "unknown"
    confidence: float = 0.0
    ambiguous: bool = False

    @property
    def is_resolved(self) -> bool:
        return bool(self.user_id) and not self.ambiguous


def resolve_verified_action_target(
    *,
    event_context,
    addressee,
    alias_candidates,
    speaker_profile=None,
    reply_has_image: bool = False,
) -> VerifiedActionTarget:
    """Bind only current-turn, unambiguous evidence to an execution target."""

    if bool(getattr(event_context, "is_private", False)):
        return VerifiedActionTarget()

    current_user_id = str(getattr(event_context, "user_id", "") or "").strip()
    bot_id = str(getattr(event_context, "bot_id", "") or "").strip()
    member_mentions = tuple(
        dict.fromkeys(
            str(getattr(item, "user_id", "") or "").strip()
            for item in tuple(getattr(event_context, "mentions", ()) or ())
            if str(getattr(item, "user_id", "") or "").strip()
            not in {current_user_id, bot_id}
        )
    )
    if len(member_mentions) == 1:
        mention_id = member_mentions[0]
        if _has_competing_alias_evidence(
            event_context=event_context,
            alias_candidates=alias_candidates,
            excluded_user_ids={mention_id, current_user_id, bot_id},
        ):
            # "@张三 帮小明签到" — the @ and the named member disagree; refuse
            # rather than silently acting on the wrong person.
            return VerifiedActionTarget(source="at", ambiguous=True)
        return VerifiedActionTarget(
            user_id=mention_id,
            source="at",
            confidence=0.95,
        )
    if len(member_mentions) > 1:
        return VerifiedActionTarget(source="at", ambiguous=True)

    alias_target = _resolve_verified_alias_target(
        event_context=event_context,
        alias_candidates=alias_candidates,
        current_user_id=current_user_id,
        bot_id=bot_id,
    )
    if alias_target.is_resolved or alias_target.ambiguous:
        return alias_target

    speaker_target = _resolve_verified_speaker_alias_target(
        event_context=event_context,
        speaker_profile=speaker_profile,
        current_user_id=current_user_id,
        bot_id=bot_id,
    )
    if speaker_target.is_resolved:
        return speaker_target

    source = str(getattr(addressee, "source", "") or "")
    target_user_id = str(getattr(addressee, "target_user_id", "") or "").strip()
    if (
        source == "reply"
        and not reply_has_image
        and not bool(getattr(addressee, "ambiguous", False))
        and target_user_id
        and target_user_id not in {current_user_id, bot_id}
    ):
        return VerifiedActionTarget(
            user_id=target_user_id,
            source="reply",
            confidence=float(getattr(addressee, "confidence", 0.84) or 0.84),
        )

    return VerifiedActionTarget()


def _has_competing_alias_evidence(
    *,
    event_context,
    alias_candidates,
    excluded_user_ids: set[str],
) -> bool:
    """True when the message literally names a member other than the @ target."""

    message_text = normalize_message_text(
        str(getattr(event_context, "message_text_with_tags", "") or "")
    )
    if not message_text:
        return False
    hits, _ = _literal_alias_hits(
        message_text=message_text,
        alias_candidates=alias_candidates,
        excluded_user_ids={item for item in excluded_user_ids if item},
    )
    return any(alias_key not in _AMBIENT_WORD_ALIAS_KEYS for _, _, _, alias_key in hits)


def _resolve_verified_speaker_alias_target(
    *,
    event_context,
    speaker_profile,
    current_user_id: str,
    bot_id: str,
) -> VerifiedActionTarget:
    if speaker_profile is None or not current_user_id or current_user_id == bot_id:
        return VerifiedActionTarget()
    profile_user_id = str(getattr(speaker_profile, "user_id", "") or "").strip()
    if profile_user_id != current_user_id:
        return VerifiedActionTarget()
    if str(getattr(speaker_profile, "conflict_state", "") or "").strip():
        return VerifiedActionTarget()

    message_text = normalize_message_text(
        str(getattr(event_context, "message_text_with_tags", "") or "")
    ).rstrip(" 　\t\r\n，,。.!！？?：:；;")
    nickname = normalize_message_text(
        str(getattr(event_context, "nickname", "") or "")
    )
    if not message_text or not nickname:
        return VerifiedActionTarget()
    nickname_key = normalize_alias_key(nickname)
    generic_self_keys = {
        normalize_alias_key(value)
        for value in ("我", "自己", "本人", "我的", "我自己", "自己的")
    }
    if not 2 <= len(nickname_key) <= 16 or nickname_key in generic_self_keys:
        return VerifiedActionTarget()
    if not message_text.casefold().endswith(nickname.casefold()):
        return VerifiedActionTarget()
    prefix = message_text[: -len(nickname)]
    if (
        prefix
        and nickname[0].isascii()
        and nickname[0].isalnum()
        and prefix[-1].isascii()
        and prefix[-1].isalnum()
    ):
        return VerifiedActionTarget()
    return VerifiedActionTarget(
        user_id=current_user_id,
        source="self_nickname",
        confidence=0.96,
    )


def _alias_occurrence_neighbors(
    message_text: str,
    alias_key: str,
) -> tuple[tuple[str, str], ...]:
    """(previous raw char, next normalized char) for each alias occurrence."""

    compact: list[str] = []
    origins: list[int] = []
    for index, char in enumerate(message_text):
        normalized = normalize_alias_key(char)
        if not normalized:
            continue
        compact.append(normalized)
        origins.append(index)
    compact_key = "".join(compact)
    if not alias_key or alias_key not in compact_key:
        return ()
    neighbors: list[tuple[str, str]] = []
    start = compact_key.find(alias_key)
    while start >= 0:
        end = start + len(alias_key)
        origin = origins[start]
        previous = message_text[origin - 1] if origin > 0 else ""
        following = compact_key[end] if end < len(compact_key) else ""
        neighbors.append((previous, following))
        start = compact_key.find(alias_key, start + 1)
    return tuple(neighbors)


def _is_cjk_char(char: str) -> bool:
    return bool(char) and "一" <= char <= "鿿"


def _alias_evidence_passes_gate(message_text: str, alias_key: str) -> bool:
    """Reject alias hits that are ambient chatter rather than a real callout."""

    if not alias_key:
        return False
    # Bare name: the whole message is just the nickname -> a 称呼/greeting,
    # never an execution instruction.
    if normalize_alias_key(message_text) == alias_key:
        return False
    if alias_key not in _AMBIENT_WORD_ALIAS_KEYS:
        return True
    # Directive / second-person signal anywhere in the message.
    if contains_any(message_text, _TARGET_REQUIRED_ACTION_HINTS) or contains_any(
        message_text, _SECOND_PERSON_HINTS
    ):
        return True
    for previous, following in _alias_occurrence_neighbors(message_text, alias_key):
        # 给番茄 / 帮番茄 / @番茄 -> the alias sits in an explicit target slot.
        if previous in _ALIAS_DIRECTIVE_PREFIX_CHARS:
            return True
        # 番茄的头像 -> possessive use, i.e. a person who owns something.
        if following == "的":
            return True
        # The ambient word is the head of a longer CJK compound
        # (番茄炒蛋怎么做 / 苹果好吃吗) -> vocabulary, not a callout.
        if not _is_cjk_char(following):
            return True
    return False


def _literal_alias_hits(
    *,
    message_text: str,
    alias_candidates,
    excluded_user_ids: set[str],
) -> tuple[list[tuple[int, float, str, str]], bool]:
    """Re-score alias candidates on literal containment in the raw message.

    Returns ``(hits, has_conflicted)`` where each hit is
    ``(alias_key_length, score, user_id, alias_key)``.
    """

    hits: list[tuple[int, float, str, str]] = []
    has_conflicted = False
    for candidate in alias_candidates or ():
        profile = getattr(candidate, "profile", None)
        if profile is None:
            continue
        user_id = str(getattr(profile, "user_id", "") or "").strip()
        if not user_id or user_id in excluded_user_ids:
            continue
        if str(getattr(profile, "conflict_state", "") or "").strip():
            has_conflicted = True
            continue
        alias_key = normalize_alias_key(
            str(getattr(candidate, "matched_alias", "") or "")
        )
        score = score_alias_in_message(message_text, alias_key)
        if score <= 0.0:
            continue
        hits.append((len(alias_key), score, user_id, alias_key))
    # Longest literal match wins: a message containing 小番茄 binds 小番茄, not
    # the shorter 番茄 it happens to contain, and that is not an ambiguity.
    hits.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return hits, has_conflicted


def _resolve_verified_alias_target(
    *,
    event_context,
    alias_candidates,
    current_user_id: str,
    bot_id: str,
) -> VerifiedActionTarget:
    message_text = normalize_message_text(
        str(getattr(event_context, "message_text_with_tags", "") or "")
    )
    if not message_text:
        return VerifiedActionTarget()

    hits, has_conflicted = _literal_alias_hits(
        message_text=message_text,
        alias_candidates=alias_candidates,
        excluded_user_ids={current_user_id, bot_id},
    )
    if not hits:
        if has_conflicted:
            return VerifiedActionTarget(source="alias", ambiguous=True)
        return VerifiedActionTarget()

    top_length, top_score, top_user_id, top_alias_key = hits[0]
    if len(hits) > 1:
        second_length, _, second_user_id, _ = hits[1]
        # Only equally strong literal evidence for two different people is a
        # genuine ambiguity.
        if second_length == top_length and second_user_id != top_user_id:
            return VerifiedActionTarget(source="alias", ambiguous=True)
    if top_score < ALIAS_AMBIGUOUS_TOP:
        return VerifiedActionTarget()
    if not _alias_evidence_passes_gate(message_text, top_alias_key):
        return VerifiedActionTarget()
    return VerifiedActionTarget(
        user_id=top_user_id,
        source="alias",
        confidence=min(top_score, 0.96),
    )


def _route_command_policy(
    route_result: NativeRouteResult,
    knowledge_plugins,
) -> tuple[object | None, CommandTargetPolicy | None]:
    schema = find_route_command_schema(route_result, knowledge_plugins)
    command_policy = (
        resolve_command_target_policy(schema) if schema is not None else None
    )
    return schema, command_policy


def _schema_image_min(schema: object | None) -> int:
    if schema is None:
        return 0
    try:
        return max(int(getattr(schema, "image_min", 0) or 0), 0)
    except Exception:
        return 0


def _schema_accepts_image_context(schema: object | None) -> bool:
    if schema is None:
        return False
    try:
        policy = resolve_command_target_policy(schema)
        if policy.allow_image_as_target or policy.allow_reply_image_as_target:
            return True
    except Exception:
        pass
    if _schema_image_min(schema) > 0:
        return True
    if not hasattr(schema, "image_max"):
        return False
    image_max = getattr(schema, "image_max", 0)
    if image_max is None:
        return True
    try:
        return int(image_max) > 0
    except Exception:
        return False


def _schema_heads(schema: object | None, route_result: NativeRouteResult) -> set[str]:
    values: list[object] = [
        getattr(schema, "command", "") if schema is not None else "",
        getattr(schema, "head", "") if schema is not None else "",
        route_result.decision.command,
    ]
    if schema is not None:
        values.extend(getattr(schema, "aliases", None) or [])
    heads: set[str] = set()
    for value in values:
        text = normalize_message_text(str(value or ""))
        if text:
            heads.add(text.split(" ", 1)[0])
    return heads


def _has_target_or_image_context(*messages: str) -> bool:
    for message in messages:
        if extract_at_tokens(message) or extract_image_tokens(message):
            return True
    return False


def _append_unique_context_tokens(message_text: str, *sources: str) -> str:
    message = normalize_message_text(message_text)
    existing = set(extract_at_tokens(message)) | set(extract_image_tokens(message))
    tokens: list[str] = []
    for source in sources:
        for token in [*extract_at_tokens(source), *extract_image_tokens(source)]:
            if token in existing:
                continue
            existing.add(token)
            tokens.append(token)
    if not tokens:
        return message
    return normalize_message_text(f"{message} {' '.join(tokens)}")


def _explicit_at_ids(message_text: str) -> tuple[str, ...]:
    values: list[str] = []
    for match in _EXPLICIT_AT_PATTERN.finditer(message_text or ""):
        user_id = (match.group("bracket") or match.group("plain") or "").strip()
        if user_id and user_id not in values:
            values.append(user_id)
    return tuple(values)


def _canonicalize_explicit_at_tokens(message_text: str) -> str:
    def replace_match(match: re.Match[str]) -> str:
        user_id = (match.group("bracket") or match.group("plain") or "").strip()
        return f"[@{user_id}]" if user_id else match.group(0)

    return normalize_message_text(_EXPLICIT_AT_PATTERN.sub(replace_match, message_text))


def _remove_explicit_at_ids(
    message_text: str,
    excluded_ids: set[str],
) -> str:
    if not excluded_ids:
        return normalize_message_text(message_text)

    def replace_match(match: re.Match[str]) -> str:
        user_id = (match.group("bracket") or match.group("plain") or "").strip()
        return "" if user_id in excluded_ids else match.group(0)

    return normalize_message_text(_EXPLICIT_AT_PATTERN.sub(replace_match, message_text))


def _remove_context_placeholders(message_text: str) -> str:
    text = _remove_explicit_at_ids(
        message_text,
        set(_explicit_at_ids(message_text)),
    )
    for token in extract_image_tokens(text):
        text = text.replace(token, "")
    return normalize_message_text(text)


def _identity_key(value: object) -> str:
    return "".join(
        character.casefold() for character in str(value or "") if character.isalnum()
    )


def _target_label_key(target_hint: str) -> str:
    without_target = _EXPLICIT_AT_PATTERN.sub("", target_hint or "")
    for token in extract_image_tokens(without_target):
        without_target = without_target.replace(token, "")
    return _identity_key(without_target)


def _profile_identity_keys(
    user_id: str,
    profile: dict[str, str],
) -> set[str]:
    return {
        key
        for key in (
            _identity_key(user_id),
            _identity_key(profile.get("display_name")),
            _identity_key(profile.get("nickname")),
            _identity_key(profile.get("user_name")),
            _identity_key(profile.get("alias_key")),
            _identity_key(profile.get("uid")),
        )
        if key
    }


def _conflicting_target_sources(*sources: tuple[str, ...]) -> bool:
    distinct = [frozenset(source) for source in sources if source]
    return bool(distinct) and any(source != distinct[0] for source in distinct[1:])


async def _resolve_verified_explicit_targets(
    *,
    group_id: str | None,
    bot_id: str | None,
    task_message: str,
    ambient_message: str,
    target_hint: str,
    mention_profiles: dict[str, dict[str, str]],
) -> TargetResolveResult | None:
    hint_ids = _explicit_at_ids(target_hint)
    task_ids = _explicit_at_ids(task_message)
    ambient_ids = _explicit_at_ids(ambient_message)
    normalized_bot_id = str(bot_id or "").strip()
    excluded_ids: set[str] = set()
    if normalized_bot_id and normalized_bot_id not in hint_ids:
        excluded_ids.add(normalized_bot_id)
        task_ids = tuple(
            user_id for user_id in task_ids if user_id != normalized_bot_id
        )
        ambient_ids = tuple(
            user_id for user_id in ambient_ids if user_id != normalized_bot_id
        )
    if not hint_ids and not task_ids and not ambient_ids:
        return None

    if _conflicting_target_sources(hint_ids, task_ids, ambient_ids):
        return TargetResolveResult(
            status="ambiguous",
            message_text=task_message,
            mention_profiles=mention_profiles,
            target_hint=target_hint,
        )

    target_ids = tuple(dict.fromkeys((*hint_ids, *task_ids, *ambient_ids)))
    verified_profiles = {
        str(user_id).strip(): dict(profile)
        for user_id, profile in mention_profiles.items()
        if str(user_id).strip() and isinstance(profile, dict)
    }
    unresolved_ids = [
        user_id for user_id in target_ids if user_id not in verified_profiles
    ]
    if unresolved_ids and group_id:
        local_profiles = await build_mention_profiles(
            group_id,
            " ".join(f"[@{user_id}]" for user_id in unresolved_ids),
            bot_id=bot_id,
            bot=None,
        )
        verified_profiles.update(local_profiles)
    if any(user_id not in verified_profiles for user_id in target_ids):
        return TargetResolveResult(
            status="invalid",
            message_text=task_message,
            mention_profiles=verified_profiles,
            target_hint=target_hint,
        )

    label_key = _target_label_key(target_hint)
    if label_key:
        if len(hint_ids) != 1:
            return TargetResolveResult(
                status="ambiguous",
                message_text=task_message,
                mention_profiles=verified_profiles,
                target_hint=target_hint,
            )
        hinted_id = hint_ids[0]
        if label_key not in _profile_identity_keys(
            hinted_id,
            verified_profiles[hinted_id],
        ):
            return TargetResolveResult(
                status="invalid",
                message_text=task_message,
                mention_profiles=verified_profiles,
                target_hint=target_hint,
            )

    canonical_hint = _canonicalize_explicit_at_tokens(target_hint)
    canonical_task = _canonicalize_explicit_at_tokens(
        _remove_explicit_at_ids(task_message, excluded_ids)
    )
    canonical_ambient = _canonicalize_explicit_at_tokens(
        _remove_explicit_at_ids(ambient_message, excluded_ids)
    )
    enriched_message = _append_unique_context_tokens(
        canonical_task,
        canonical_hint,
        canonical_ambient,
    )
    return TargetResolveResult(
        status=(
            "resolved"
            if enriched_message != normalize_message_text(task_message)
            else "present"
        ),
        message_text=enriched_message,
        mention_profiles=verified_profiles,
        target_hint=canonical_hint,
        resolved_target_ids=target_ids,
    )


def _command_can_use_target(
    *,
    schema: object | None,
    command_policy: CommandTargetPolicy | None,
) -> bool:
    if command_policy is not None:
        if (
            command_policy.allow_at_as_target
            or command_policy.allow_image_as_target
            or command_policy.allow_reply_image_as_target
            or command_policy.media_related
            or command_policy.target_requirement != "none"
        ):
            return True
    return _schema_accepts_image_context(schema)


async def resolve_pre_route_target(
    *,
    group_id: str | None,
    bot: Bot | None = None,
    original_message: str,
    route_message: str,
    mention_profiles: dict[str, dict[str, str]],
    target_policy: TargetPolicy,
    command_heads: set[str] | None = None,
) -> TargetResolveResult:
    (
        enriched_message,
        enriched_profiles,
        prompt,
    ) = await enrich_route_message_with_fuzzy_target(
        group_id=group_id,
        bot=bot,
        original_message=original_message,
        route_message=route_message,
        mention_profiles=mention_profiles,
        target_policy=target_policy,
        command_heads=command_heads,
    )
    if prompt:
        return TargetResolveResult(
            status="not_needed",
            message_text=route_message,
            mention_profiles=enriched_profiles,
            target_hint=extract_fuzzy_target_hint(route_message, command_heads),
        )
    if enriched_message != route_message:
        return TargetResolveResult(
            status="resolved",
            message_text=enriched_message,
            mention_profiles=enriched_profiles,
            target_hint=extract_fuzzy_target_hint(route_message, command_heads),
            resolved_target_ids=_explicit_at_ids(enriched_message),
        )
    if _has_target_or_image_context(route_message):
        return TargetResolveResult(
            status="present",
            message_text=route_message,
            mention_profiles=enriched_profiles,
            resolved_target_ids=_explicit_at_ids(route_message),
        )
    return TargetResolveResult(
        status="not_needed",
        message_text=route_message,
        mention_profiles=enriched_profiles,
    )


async def resolve_execution_target(
    *,
    group_id: str | None,
    bot: Bot | None = None,
    bot_id: str | None = None,
    route_result: NativeRouteResult,
    knowledge_plugins,
    task_message: str,
    ambient_message: str,
    target_hint: str = "",
    mention_profiles: dict[str, dict[str, str]] | None = None,
    use_ambient_target_context: bool = False,
) -> TargetResolveResult:
    schema, command_policy = _route_command_policy(
        route_result,
        knowledge_plugins,
    )
    target_policy = command_policy or TargetPolicy()
    mention_profiles = dict(mention_profiles or {})
    if not _command_can_use_target(
        schema=schema,
        command_policy=command_policy,
    ):
        return TargetResolveResult(
            status="not_needed",
            message_text=task_message,
            mention_profiles=mention_profiles,
        )

    explicit_target_hint = normalize_message_text(target_hint)
    allow_at_target = bool(
        command_policy is not None and command_policy.allow_at_as_target
    )
    image_target_allowed = bool(
        _schema_accepts_image_context(schema)
        or (
            command_policy is not None
            and (
                command_policy.allow_image_as_target
                or command_policy.allow_reply_image_as_target
            )
        )
    )
    excluded_bot_ids = (
        {str(bot_id).strip()}
        if bot_id
        and str(bot_id).strip()
        and str(bot_id).strip() not in _explicit_at_ids(explicit_target_hint)
        else set()
    )
    target_task_message = _remove_explicit_at_ids(
        task_message,
        excluded_bot_ids,
    )
    if use_ambient_target_context:
        target_task_message = _remove_context_placeholders(target_task_message)
    target_ambient_message = _remove_explicit_at_ids(
        ambient_message,
        excluded_bot_ids,
    )
    if allow_at_target:
        explicit_resolution = await _resolve_verified_explicit_targets(
            group_id=group_id,
            bot_id=bot_id,
            task_message=target_task_message,
            ambient_message=target_ambient_message,
            target_hint=explicit_target_hint,
            mention_profiles=mention_profiles,
        )
        if explicit_resolution is not None:
            return explicit_resolution
    if image_target_allowed and extract_image_tokens(explicit_target_hint):
        image_context = " ".join(extract_image_tokens(explicit_target_hint))
        enriched_message = _append_unique_context_tokens(
            target_task_message,
            image_context,
        )
        return TargetResolveResult(
            status=(
                "resolved" if enriched_message != target_task_message else "present"
            ),
            message_text=enriched_message,
            mention_profiles=mention_profiles,
            target_hint=explicit_target_hint,
        )

    if image_target_allowed and any(
        extract_image_tokens(message)
        for message in (target_task_message, target_ambient_message)
    ):
        image_context = " ".join(
            dict.fromkeys(
                [
                    *extract_image_tokens(target_task_message),
                    *extract_image_tokens(target_ambient_message),
                ]
            )
        )
        return TargetResolveResult(
            status="present",
            message_text=_append_unique_context_tokens(
                target_task_message,
                image_context,
            ),
            mention_profiles=mention_profiles,
            target_hint=explicit_target_hint,
        )

    command_heads = _schema_heads(schema, route_result)
    if command_policy is not None and command_policy.allow_at_as_target and bot_id:
        self_target_message = (
            target_ambient_message
            if use_ambient_target_context
            else target_task_message
        )
        self_target = await resolve_pre_route_target(
            group_id=group_id,
            bot=bot,
            original_message=self_target_message,
            route_message=self_target_message,
            mention_profiles=mention_profiles,
            target_policy=target_policy,
            command_heads=command_heads,
        )
        if self_target.status in {"resolved", "present"}:
            return self_target

    target_lookup_parts = [
        target_ambient_message if use_ambient_target_context else target_task_message,
        explicit_target_hint,
    ]
    target_lookup_message = normalize_message_text(
        " ".join(item for item in target_lookup_parts if item)
    )
    extracted_target_hint = extract_fuzzy_target_hint(
        target_lookup_message or target_task_message,
        command_heads,
    )
    resolved_target_hint = explicit_target_hint or extracted_target_hint
    resolved = await resolve_pre_route_target(
        group_id=group_id,
        bot=bot,
        original_message=target_lookup_message or target_task_message,
        route_message=target_lookup_message or target_task_message,
        mention_profiles=mention_profiles,
        target_policy=target_policy,
        command_heads=command_heads,
    )
    if resolved.status in {"resolved", "ambiguous", "missing"}:
        if resolved.status == "resolved":
            resolved_message = _append_unique_context_tokens(
                target_task_message,
                resolved.message_text,
            )
            return TargetResolveResult(
                status="resolved",
                message_text=resolved_message,
                mention_profiles=resolved.mention_profiles,
                prompt=resolved.prompt,
                target_hint=resolved.target_hint or resolved_target_hint,
                resolved_target_ids=_explicit_at_ids(resolved.message_text),
            )
        return resolved

    target_required = (
        command_policy is not None and command_policy.target_requirement == "required"
    )
    target_required = target_required or needs_target_for_route(
        target_ambient_message if use_ambient_target_context else target_task_message,
        target_ambient_message if use_ambient_target_context else target_task_message,
        target_policy=target_policy,
    )
    target_required = target_required or bool(
        resolved_target_hint
        and (
            _schema_accepts_image_context(schema)
            or (
                command_policy is not None
                and (
                    command_policy.allow_at_as_target
                    or command_policy.allow_image_as_target
                )
            )
        )
    )
    if target_required:
        return TargetResolveResult(
            status="missing",
            message_text=target_task_message,
            mention_profiles=mention_profiles,
            target_hint=resolved_target_hint,
        )

    return TargetResolveResult(
        status="not_needed",
        message_text=target_task_message,
        mention_profiles=mention_profiles,
        target_hint=resolved_target_hint,
    )


__all__ = [
    "TargetResolveResult",
    "VerifiedActionTarget",
    "resolve_execution_target",
    "resolve_pre_route_target",
    "resolve_verified_action_target",
]
