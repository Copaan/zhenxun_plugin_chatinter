from __future__ import annotations

from .addressee_resolver import AddresseeResult, format_addressee_xml
from .event_context import ChatInterEventContext
from .person_registry import PersonProfile, RelevantPerson
from .thread_resolver import ThreadContext, format_thread_xml


class DialogueContextPack:
    def __init__(
        self,
        *,
        event_context: ChatInterEventContext,
        speaker_profile: PersonProfile | None,
        addressee: AddresseeResult | None,
        thread: ThreadContext | None,
        relevant_people: tuple[RelevantPerson, ...] = (),
        action_target_user_ids: tuple[str, ...] = (),
        alias_target_user_ids: tuple[str, ...] = (),
    ) -> None:
        self.event_context = event_context
        self.speaker_profile = speaker_profile
        self.addressee = addressee
        self.thread = thread
        self.relevant_people = relevant_people
        self.action_target_user_ids = action_target_user_ids
        self.alias_target_user_ids = alias_target_user_ids

    def to_context_xml(self) -> str:
        return build_group_dialogue_context(
            event_context=self.event_context,
            speaker_profile=self.speaker_profile,
            addressee=self.addressee,
            thread=self.thread,
            relevant_people=self.relevant_people,
            action_target_user_ids=self.action_target_user_ids,
            alias_target_user_ids=self.alias_target_user_ids,
        )

    def action_target_refs(self) -> dict[str, str]:
        allowed_user_ids = _action_target_user_ids(
            self.thread,
            self.action_target_user_ids,
            self.alias_target_user_ids,
        )
        if not allowed_user_ids:
            return {}
        refs: dict[str, str] = {}
        for index, person in enumerate(
            _visible_relevant_people(self.speaker_profile, self.relevant_people)[:8],
            start=1,
        ):
            user_id = str(person.profile.user_id or "").strip()
            if user_id in allowed_user_ids:
                refs[f"person:{index}"] = user_id
        return refs


def append_group_dialogue_context(
    context_xml: str,
    *,
    event_context: ChatInterEventContext,
    speaker_profile: PersonProfile | None,
    addressee: AddresseeResult | None,
    thread: ThreadContext | None,
    relevant_people: tuple[RelevantPerson, ...] = (),
    action_target_user_ids: tuple[str, ...] = (),
    alias_target_user_ids: tuple[str, ...] = (),
) -> str:
    packed = build_group_dialogue_context(
        event_context=event_context,
        speaker_profile=speaker_profile,
        addressee=addressee,
        thread=thread,
        relevant_people=relevant_people,
        action_target_user_ids=action_target_user_ids,
        alias_target_user_ids=alias_target_user_ids,
    )
    if not packed:
        return context_xml
    return f"{context_xml}\n{packed}"


def build_group_dialogue_context(
    *,
    event_context: ChatInterEventContext,
    speaker_profile: PersonProfile | None,
    addressee: AddresseeResult | None,
    thread: ThreadContext | None,
    relevant_people: tuple[RelevantPerson, ...] = (),
    action_target_user_ids: tuple[str, ...] = (),
    alias_target_user_ids: tuple[str, ...] = (),
) -> str:
    lines: list[str] = []
    lines.extend(_event_lines(event_context))
    if speaker_profile is not None:
        lines.extend(_turn_identity_lines(event_context, speaker_profile))
    visible_people = _visible_relevant_people(speaker_profile, relevant_people)
    if visible_people:
        lines.extend(
            _relevant_people_lines(
                visible_people,
                target_user_ids=_action_target_user_ids(
                    thread,
                    action_target_user_ids,
                    alias_target_user_ids,
                ),
            )
        )
    if addressee is not None:
        lines.extend(format_addressee_xml(addressee))
    if thread is not None:
        lines.extend(format_thread_xml(thread))
    if not lines:
        return ""
    return "\n".join(lines)


def _visible_relevant_people(
    speaker_profile: PersonProfile | None,
    relevant_people: tuple[RelevantPerson, ...],
) -> tuple[RelevantPerson, ...]:
    speaker_id = speaker_profile.user_id if speaker_profile is not None else ""
    return tuple(
        person
        for person in relevant_people
        if not person.is_current_speaker
        and (not speaker_id or person.profile.user_id != speaker_id)
    )


def _thread_user_ids(thread: ThreadContext | None) -> set[str]:
    if thread is None:
        return set()
    return {
        str(user_id).strip()
        for user_id in thread.related_user_ids
        if str(user_id).strip()
    }


def _action_target_user_ids(
    thread: ThreadContext | None,
    recent_user_ids: tuple[str, ...],
    alias_target_user_ids: tuple[str, ...] = (),
) -> set[str]:
    """执行侧可用的 target_ref 白名单。

    三个来源:
    - thread 参与者(结构证据)
    - 最近发言者(弱证据: 只是说过话)
    - 本轮消息里被别名字面点名的人(强证据: 用户当前就在叫这个人)
    """
    return (
        _thread_user_ids(thread)
        | {
            str(user_id).strip()
            for user_id in recent_user_ids
            if str(user_id).strip()
        }
        | {
            str(user_id).strip()
            for user_id in alias_target_user_ids
            if str(user_id).strip()
        }
    )


def _event_lines(event_context: ChatInterEventContext) -> list[str]:
    lines = ["<event_context>"]
    lines.append(f"adapter={_xml_escape(event_context.adapter)}")
    lines.append(f"chat_type={'private' if event_context.is_private else 'group'}")
    lines.append(f"user_id={_xml_escape(event_context.user_id)}")
    if event_context.group_id:
        lines.append(f"group_id={_xml_escape(event_context.group_id)}")
    if event_context.bot_id:
        lines.append(f"bot_id={_xml_escape(event_context.bot_id)}")
    lines.append(f"is_to_me={int(event_context.is_to_me)}")
    if event_context.mentions:
        lines.append(
            "mentions="
            + ",".join(_xml_escape(item.user_id) for item in event_context.mentions)
        )
    if event_context.reply:
        if event_context.reply.sender_id:
            lines.append(
                f"reply_sender_id={_xml_escape(event_context.reply.sender_id)}"
            )
    if event_context.images:
        lines.append(f"image_count={len(event_context.images)}")
    lines.append("</event_context>")
    return lines


def _turn_identity_lines(
    event_context: ChatInterEventContext,
    speaker_profile: PersonProfile,
) -> list[str]:
    lines = ["<turn_identity>"]
    lines.append(f"current_speaker_user_id={_xml_escape(speaker_profile.user_id)}")
    lines.append(
        f"current_speaker_display_name={_xml_escape(speaker_profile.display_name)}"
    )
    if speaker_profile.group_card:
        lines.append(
            f"current_speaker_group_card={_xml_escape(speaker_profile.group_card)}"
        )
    if speaker_profile.nickname:
        lines.append(
            f"current_speaker_nickname={_xml_escape(speaker_profile.nickname)}"
        )
    if speaker_profile.aliases:
        lines.append(
            "current_speaker_aliases="
            + _xml_escape("、".join(speaker_profile.aliases[:6]))
        )
    if event_context.group_id:
        lines.append(f"current_group_id={_xml_escape(event_context.group_id)}")
    lines.append("</turn_identity>")
    return lines


def _relevant_people_lines(
    people: tuple[RelevantPerson, ...],
    *,
    target_user_ids: set[str] | None = None,
) -> list[str]:
    lines = ["<relevant_people>"]
    allowed_targets = target_user_ids or set()
    for index, person in enumerate(people[:8], start=1):
        profile = person.profile
        fields = [
            f"index={index}",
            f"user_id={_xml_escape(profile.user_id)}",
            f"display_name={_xml_escape(profile.display_name)}",
            f"is_current_speaker={int(person.is_current_speaker)}",
        ]
        if profile.user_id in allowed_targets:
            fields.insert(1, f"target_ref=person:{index}")
        if profile.group_card:
            fields.append(f"group_card={_xml_escape(profile.group_card)}")
        if profile.nickname:
            fields.append(f"nickname={_xml_escape(profile.nickname)}")
        if profile.aliases:
            fields.append(f"aliases={_xml_escape('、'.join(profile.aliases[:6]))}")
        if person.matched_alias:
            fields.append(f"matched_alias={_xml_escape(person.matched_alias)}")
        if profile.conflict_state:
            fields.append(f"conflict_state={_xml_escape(profile.conflict_state)}")
        lines.append("; ".join(fields))
    lines.append("</relevant_people>")
    return lines


def _xml_escape(value: str) -> str:
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .strip()
    )


__all__ = [
    "DialogueContextPack",
    "append_group_dialogue_context",
    "build_group_dialogue_context",
]
