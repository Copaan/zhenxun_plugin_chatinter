"""Embedding second channel for skill-scoped command recall.

Static BM25 lexical recall (see ``command_index``) misses synonym verbs
("咬他" vs. the "纳西妲啃" command) and intent paraphrases ("反击回去" vs. a
meme family).  When the lexical channel returns zero candidates the skill
dispatch tool degrades to listing every available command of that plugin;
this module upgrades that degradation from "the model reads the whole table"
into "the model reads the semantically ranked top N".

Design constraints (deliberate, do not widen without a reason):

* Skill-scoped only.  There is no global pre-retrieval layer — the channel is
  invoked with the snapshots of one plugin skill.
* Triggered only on zero lexical recall, so the steady state cost is zero.
* Document text is ``head + aliases + retrieval_phrases + capability_text``.
  ``description``/``usage`` are excluded on purpose: meme-style families share
  hundreds of near-identical descriptions and would drown the signal.
* No hash-vector fallback.  A hash vector carries no semantic information, so
  ranking by it would be worse than the existing full listing.  When embedding
  is unavailable the channel returns ``None`` and the caller keeps the plain
  full-listing degradation.
* Nothing here may raise into dispatch.  Every failure path returns ``None``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, ClassVar

from .llm_compat import embed_documents, embed_query, list_embedding_models
from .log_compat import logger
from .models.pydantic_models import CommandToolSnapshot

# Same data root as knowledge_rag_retrieval._INDEX_PATH
# ("data/cache/chatinter/plugin_vector_index.json"), one directory per concern.
_VECTOR_DIR = Path("data/cache/chatinter/command_vectors")
_EMBEDDING_COOLDOWN = 600.0
_MIN_SIMILARITY = 0.30
_QUERY_CACHE_MAX_SIZE = 64
_DOC_TEXT_LIMIT = 512
_PERSIST_VERSION = 1


@dataclass
class _SkillIndex:
    fingerprint: str
    vectors: dict[str, list[float]] = field(default_factory=dict)


def build_document_text(snapshot: CommandToolSnapshot) -> str:
    """head + aliases + retrieval_phrases + capability_text（去空去重）."""

    parts: list[str] = []
    seen: set[str] = set()

    def _push(value: Any) -> None:
        text = str(value or "").strip()
        if not text or text in seen:
            return
        seen.add(text)
        parts.append(text)

    _push(getattr(snapshot, "head", ""))
    for alias in getattr(snapshot, "aliases", None) or ():
        _push(alias)
    for phrase in getattr(snapshot, "retrieval_phrases", None) or ():
        _push(phrase)
    _push(getattr(snapshot, "capability_text", ""))
    return " / ".join(parts)[:_DOC_TEXT_LIMIT]


def _normalize_vector(values: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in values))
    if norm <= 0:
        return [0.0] * len(values)
    return [float(value) / norm for value in values]


def _cosine(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    return float(sum(float(a) * float(b) for a, b in zip(vec_a, vec_b)))


def _fingerprint(pairs: Sequence[tuple[str, str]], model_tag: str) -> str:
    digest = hashlib.sha256()
    digest.update(model_tag.encode("utf-8"))
    for command_id, text in pairs:
        digest.update(b"\x00")
        digest.update(command_id.encode("utf-8"))
        digest.update(b"\x01")
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def _cache_path(skill_key: str) -> Path:
    slug = hashlib.sha1(skill_key.encode("utf-8")).hexdigest()[:20]
    return _VECTOR_DIR / f"{slug}.json"


class CommandVectorRecall:
    """Lazy, per-skill, disk-persisted embedding index for command recall."""

    _indexes: ClassVar[dict[str, _SkillIndex]] = {}
    _locks: ClassVar[dict[str, asyncio.Lock]] = {}
    _query_cache: ClassVar[dict[tuple[str, str], list[float]]] = {}
    _embedding_supported: ClassVar[bool | None] = None
    _embedding_disabled_until: ClassVar[float] = 0.0

    @classmethod
    def reset(cls) -> None:
        """Test helper: drop in-memory state (disk cache untouched)."""

        cls._indexes = {}
        cls._locks = {}
        cls._query_cache = {}
        cls._embedding_supported = None
        cls._embedding_disabled_until = 0.0

    @classmethod
    def available(cls) -> bool:
        if cls._embedding_disabled_until and (
            time.monotonic() < cls._embedding_disabled_until
        ):
            return False
        if cls._embedding_supported is None:
            try:
                cls._embedding_supported = bool(list_embedding_models())
            except Exception:
                cls._embedding_supported = False
        return bool(cls._embedding_supported)

    @classmethod
    def _disable(cls, exc: BaseException | None = None) -> None:
        cls._embedding_disabled_until = time.monotonic() + _EMBEDDING_COOLDOWN
        if exc is not None:
            logger.debug(f"chatinter command vector recall disabled temporarily: {exc}")

    @classmethod
    async def rank(
        cls,
        skill_key: str,
        snapshots: Sequence[CommandToolSnapshot],
        query: str,
        *,
        limit: int | None = None,
        min_score: float = _MIN_SIMILARITY,
    ) -> list[tuple[str, float]] | None:
        """Rank ``snapshots`` by cosine similarity to ``query``.

        Returns ``None`` when the channel is unavailable (disabled, no
        embedding model, or any failure) — the caller must then keep its own
        degradation path.  Returns ``[]`` when every candidate scored below
        ``min_score``.
        """

        try:
            return await cls._rank(
                skill_key,
                snapshots,
                query,
                limit=limit,
                min_score=min_score,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # never let dispatch fail on recall
            cls._disable(exc)
            return None

    @classmethod
    async def _rank(
        cls,
        skill_key: str,
        snapshots: Sequence[CommandToolSnapshot],
        query: str,
        *,
        limit: int | None,
        min_score: float,
    ) -> list[tuple[str, float]] | None:
        from .config import command_vector_recall_enabled

        query_text = str(query or "").strip()
        if not query_text or not snapshots:
            return None
        if not command_vector_recall_enabled():
            return None
        if not cls.available():
            return None

        pairs: list[tuple[str, str]] = []
        for snapshot in snapshots:
            command_id = str(getattr(snapshot, "command_id", "") or "").strip()
            text = build_document_text(snapshot)
            if command_id and text:
                pairs.append((command_id, text))
        if not pairs:
            return None

        index = await cls._ensure_index(skill_key, pairs)
        if index is None:
            return None
        query_vector = await cls._embed_query_cached(query_text)
        if not query_vector:
            return None

        scored: list[tuple[str, float]] = []
        for command_id, _text in pairs:
            vector = index.vectors.get(command_id)
            if not vector:
                continue
            score = _cosine(query_vector, vector)
            if score >= min_score:
                scored.append((command_id, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        if limit is not None and limit > 0:
            scored = scored[:limit]
        return scored

    @classmethod
    def _lock(cls, skill_key: str) -> asyncio.Lock:
        lock = cls._locks.get(skill_key)
        if lock is None:
            lock = asyncio.Lock()
            cls._locks[skill_key] = lock
        return lock

    @classmethod
    async def _ensure_index(
        cls,
        skill_key: str,
        pairs: Sequence[tuple[str, str]],
    ) -> _SkillIndex | None:
        model_tag = cls._model_tag()
        fingerprint = _fingerprint(pairs, model_tag)

        cached = cls._indexes.get(skill_key)
        if cached is not None and cached.fingerprint == fingerprint:
            return cached

        async with cls._lock(skill_key):
            cached = cls._indexes.get(skill_key)
            if cached is not None and cached.fingerprint == fingerprint:
                return cached

            persisted = await cls._load_persisted(skill_key, fingerprint)
            if persisted is not None:
                cls._indexes[skill_key] = persisted
                return persisted

            if not cls.available():
                return None
            texts = [text for _command_id, text in pairs]
            try:
                vectors = await cls._embed_documents(texts)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                cls._disable(exc)
                return None
            if not vectors or len(vectors) != len(texts):
                cls._disable(RuntimeError("embedding result invalid"))
                return None

            index = _SkillIndex(fingerprint=fingerprint)
            for (command_id, _text), vector in zip(pairs, vectors):
                normalized = _normalize_vector([float(v) for v in vector])
                if any(normalized):
                    index.vectors[command_id] = normalized
            if not index.vectors:
                return None
            cls._indexes[skill_key] = index
            await cls._save_persisted(skill_key, index)
            return index

    @classmethod
    def _model_tag(cls) -> str:
        try:
            from .config import command_vector_recall_model

            return command_vector_recall_model() or "default"
        except Exception:
            return "default"

    @classmethod
    async def _embed_documents(cls, texts: list[str]) -> list[list[float]]:
        model = cls._model_tag()
        if model and model != "default":
            from zhenxun.services.ai.llm.api import embed

            response = await embed(texts, model=model, task="document")
            return list(response.embeddings)
        return await embed_documents(texts)

    @classmethod
    async def _embed_query_cached(cls, text: str) -> list[float]:
        model_tag = cls._model_tag()
        key = (model_tag, text)
        cached = cls._query_cache.get(key)
        if cached is not None:
            return cached
        if not cls.available():
            return []
        try:
            if model_tag and model_tag != "default":
                from zhenxun.services.ai.llm.api import embed

                response = await embed(text, model=model_tag, task="query")
                raw = list(response.vector)
            else:
                raw = list(await embed_query(text))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            cls._disable(exc)
            return []
        vector = _normalize_vector([float(value) for value in raw])
        if not any(vector):
            return []
        if len(cls._query_cache) >= _QUERY_CACHE_MAX_SIZE:
            cls._query_cache.clear()
        cls._query_cache[key] = vector
        return vector

    @classmethod
    async def _load_persisted(
        cls,
        skill_key: str,
        fingerprint: str,
    ) -> _SkillIndex | None:
        path = _cache_path(skill_key)

        def _read() -> dict[str, Any] | None:
            try:
                if not path.exists():
                    return None
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None
            return payload if isinstance(payload, dict) else None

        try:
            payload = await asyncio.to_thread(_read)
        except Exception:
            return None
        if not payload:
            return None
        if int(payload.get("version") or 0) != _PERSIST_VERSION:
            return None
        if str(payload.get("fingerprint") or "") != fingerprint:
            return None
        raw_vectors = payload.get("vectors")
        if not isinstance(raw_vectors, dict):
            return None
        index = _SkillIndex(fingerprint=fingerprint)
        for command_id, values in raw_vectors.items():
            if not isinstance(values, list) or not values:
                continue
            try:
                index.vectors[str(command_id)] = [float(value) for value in values]
            except (TypeError, ValueError):
                continue
        return index if index.vectors else None

    @classmethod
    async def _save_persisted(cls, skill_key: str, index: _SkillIndex) -> None:
        path = _cache_path(skill_key)
        payload = {
            "version": _PERSIST_VERSION,
            "skill_key": skill_key,
            "fingerprint": index.fingerprint,
            "vector_type": "embedding",
            "vectors": index.vectors,
        }

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        try:
            await asyncio.to_thread(_write)
        except Exception as exc:
            logger.debug(f"chatinter command vector cache write failed: {exc}")


__all__ = [
    "CommandVectorRecall",
    "build_document_text",
]
