"""
title: HolmesGPT
author: AI Investigation POC
version: 1.0.2
license: MIT
description: Cancellation-safe Open WebUI 0.11.0 Pipe for HolmesGPT 0.43.0.
requirements: httpx
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import time
import uuid
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

# Open WebUI and Holmes must see these paths on the same PVC.
UPLOADS = Path("/app/backend/data/uploads")
ATTACHMENTS = Path("/app/backend/data/investigation-attachments")

MAX_FILE_CHARS = 100_000
MAX_TOTAL_CHARS = 300_000
MAX_ARRAY_LENGTH = 300
NOISE_KEYS = {"evidence", "timestamp.us", "unattributed_ms_estimate"}

# HolmesGPT 0.43.0 documented SSE event names.
HOLMES_TOOL_START = "start_tool_calling"
HOLMES_TOOL_RESULT = "tool_calling_result"
HOLMES_AI_MESSAGE = "ai_message"
HOLMES_FINAL = "ai_answer_end"
HOLMES_APPROVAL = "approval_required"
HOLMES_TOKEN_COUNT = "token_count"
HOLMES_COMPACTION_START = "conversation_history_compaction_start"
HOLMES_COMPACTED = "conversation_history_compacted"

EventEmitter = Callable[[dict[str, Any]], Awaitable[Any]]


def _parse_sse(lines: Iterator[str | bytes]) -> Iterator[tuple[str, str]]:
    """Synchronous parser retained for local/unit testing."""
    event_type = "message"
    data_parts: list[str] = []
    for raw in lines:
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        line = line.rstrip("\r\n")
        if line == "":
            if data_parts:
                yield event_type, "\n".join(data_parts)
            event_type, data_parts = "message", []
        elif line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            value = line[5:]
            data_parts.append(value[1:] if value.startswith(" ") else value)
        # Comments (including upstream heartbeats), id, and retry are ignored.
    if data_parts:
        yield event_type, "\n".join(data_parts)


async def _parse_sse_async(lines: Any) -> AsyncIterator[tuple[str, str]]:
    """Parse named SSE events and flush a final unterminated event."""
    event_type = "message"
    data_parts: list[str] = []
    async for raw in lines:
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        line = line.rstrip("\r\n")
        if line == "":
            if data_parts:
                yield event_type, "\n".join(data_parts)
            event_type, data_parts = "message", []
        elif line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            value = line[5:]
            data_parts.append(value[1:] if value.startswith(" ") else value)
    if data_parts:
        yield event_type, "\n".join(data_parts)


def _json_payload(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


def _sanitize_answer(text: str) -> str:
    """Remove model reasoning and Holmes-only markers from user-visible text."""
    # Prefer an explicit final block when a reasoning model supplies one.
    final_match = re.search(r"<final\b[^>]*>(.*?)</final>", text, flags=re.IGNORECASE | re.DOTALL)
    if final_match:
        text = final_match.group(1)

    # Some OpenAI-compatible reasoning models may place reasoning tags in the
    # content string. This transport stays provider-agnostic and removes those
    # tags only at the final Holmes-to-OpenWebUI boundary.
    text = re.sub(
        r"<think\b[^>]*>.*?</think>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"<reasoning\b[^>]*>.*?</reasoning>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # Some providers omit the opening tag while retaining the closing tag.
    for closing_tag in ("</think>", "</reasoning>"):
        closing_index = text.lower().rfind(closing_tag)
        if closing_index >= 0:
            text = text[closing_index + len(closing_tag) :]
    # If a reasoning block was truncated before its closing tag, preserve only
    # content that appeared before it; never expose the unterminated reasoning.
    opening_match = re.search(r"<(?:think|reasoning)\b[^>]*>", text, flags=re.IGNORECASE)
    if opening_match:
        text = text[: opening_match.start()]

    text = re.sub(r"!\[[^\]]*\]\(tool-image://[^)]*\)", "", text)
    text = re.sub(r"<img\b[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<<\[.*?\]>>", "", text, flags=re.DOTALL)
    marker = "### Executive Summary"
    marker_index = text.find(marker)
    if 0 <= marker_index < 2_000:
        text = text[marker_index:]
    return text.strip()


def _message_text(content: Any) -> str:
    """Convert OpenAI/Open WebUI message content to plain text for Holmes."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") in ("text", "input_text") and isinstance(item.get("text"), str):
            parts.append(item["text"])
        elif isinstance(item.get("content"), str):
            parts.append(item["content"])
    return "\n".join(part for part in parts if part)


def _latest_user_index(messages: list[Any]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, dict) and message.get("role") == "user":
            return index
    return None


def _conversation_history(body: dict[str, Any], latest_user_index: int) -> list[dict[str, str]]:
    """Translate prior OpenWebUI turns to Holmes history.

    Holmes 0.43.0 requires the first history message to be a system message.
    Tool messages are intentionally omitted because OpenWebUI tool-call IDs do not
    necessarily form a valid Holmes-native tool history.
    """
    translated: list[dict[str, str]] = []
    for message in list(body.get("messages") or [])[:latest_user_index]:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            continue
        content = _message_text(message.get("content"))
        if content.strip():
            translated.append({"role": role, "content": content})

    if not translated:
        return []
    if translated[0]["role"] != "system":
        translated.insert(
            0,
            {
                "role": "system",
                "content": "Use the prior conversation only as context for this SRE investigation.",
            },
        )
    return translated


def _simplify_json(data: Any) -> Any:
    if isinstance(data, dict):
        return {
            key: _simplify_json(value)
            for key, value in data.items()
            if key not in NOISE_KEYS and not key.startswith("kubernetes.")
        }
    if isinstance(data, list):
        items = [_simplify_json(value) for value in data[:MAX_ARRAY_LENGTH]]
        if len(data) > MAX_ARRAY_LENGTH:
            items.append({"omitted_items": len(data) - MAX_ARRAY_LENGTH})
        return items
    return data


def _safe_path(path: Path) -> bool:
    try:
        resolved = path.resolve(strict=True)
        return resolved.is_file() and any(
            resolved.is_relative_to(base.resolve()) for base in (UPLOADS, ATTACHMENTS)
        )
    except (OSError, ValueError):
        return False


def _collect_files(body: dict[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    candidates = list(body.get("files") or [])
    for message in reversed(body.get("messages") or []):
        if isinstance(message, dict) and message.get("role") == "user":
            candidates += list(message.get("files") or [])
            break
    for item in candidates:
        if not isinstance(item, dict):
            continue
        info = item.get("file") or item
        if not isinstance(info, dict):
            continue
        key = str(info.get("id") or item.get("id") or info.get("path") or uuid.uuid4())
        if key not in seen:
            seen.add(key)
            found.append(item)
    return found


def _resolve_file(item: dict[str, Any]) -> Path | None:
    info = item.get("file") or item
    if not isinstance(info, dict):
        return None

    direct = info.get("path")
    if isinstance(direct, str) and direct:
        candidate = Path(direct)
        if _safe_path(candidate):
            return candidate

    file_id = info.get("id") or item.get("id")
    filename = Path(str(info.get("filename") or info.get("name") or "")).name
    if not isinstance(file_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", file_id):
        return None

    candidates = [UPLOADS / f"{file_id}_{filename}"] if filename else []
    candidates += list(UPLOADS.glob(f"{file_id}_*"))[:20]
    for candidate in candidates:
        if _safe_path(candidate):
            return candidate
    return None


def _inline_content(path: Path, limit: int) -> str:
    if limit <= 0:
        return ""
    try:
        if path.suffix.lower() not in (".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".log"):
            return f"[Binary attachment {path.name}: requires Holmes read_attachment tool]"
        with path.open("r", encoding="utf-8", errors="replace") as file_handle:
            content = file_handle.read(min(limit * 3, 1_000_000))
        if path.suffix.lower() == ".json":
            try:
                content = json.dumps(_simplify_json(json.loads(content)), ensure_ascii=False)
            except ValueError:
                pass
        if len(content) > limit:
            content = content[:limit] + "\n[Truncated]"
        return content
    except OSError as exc:
        logger.warning("Cannot read attachment name=%s error=%s", path.name, type(exc).__name__)
        return ""


def _prepare_attachments(body: dict[str, Any]) -> tuple[list[Path], str]:
    """Copy UI uploads to the Holmes-readable shared PVC directory.

    This deliberately preserves the existing durable attachment behavior. The
    copied paths are not removed at request completion because Holmes and other
    investigation flows may refer to them after the HTTP stream closes.
    """
    files = _collect_files(body)
    if not files:
        return [], ""

    destination_directory = ATTACHMENTS / str(uuid.uuid4())
    copied: list[Path] = []
    inline: list[str] = []
    remaining = MAX_TOTAL_CHARS

    for item in files:
        source = _resolve_file(item)
        if source is None:
            logger.warning("Could not resolve an OpenWebUI attachment descriptor")
            continue
        try:
            destination_directory.mkdir(parents=True, exist_ok=True)
            destination = destination_directory / f"{uuid.uuid4().hex[:8]}_{source.name}"
            shutil.copyfile(source, destination)
            destination.chmod(0o644)
            copied.append(destination)
        except OSError as exc:
            logger.warning(
                "PVC attachment copy failed name=%s error=%s",
                source.name,
                type(exc).__name__,
            )
            content = _inline_content(source, min(MAX_FILE_CHARS, remaining))
            if content:
                inline.append(f"### {source.name}\n```\n{content}\n```")
                remaining -= len(content)

    return copied, "\n\n".join(inline)


def _wrap_prompt(prompt: str, has_files: bool) -> str:
    lower = prompt.lower()
    apm = any(word in lower for word in ("apm", "trace", "span", "transaction", "dependency"))
    sre = any(
        word in lower
        for word in ("cluster", "kafka", "mongodb", "postgres", "prometheus", "kubernetes", "pod")
    )
    if apm:
        instruction = "Use the apm-investigation skill when relevant."
    elif sre:
        instruction = "Use the sre-orchestrator skill when relevant."
    else:
        instruction = "Answer directly; choose an investigation skill only if needed."
    if has_files:
        instruction += (
            " Investigate the attached evidence first. Do not query live APM, "
            "Prometheus, or Grafana unless necessary."
        )
    return f"{instruction}\n\nUser request: {prompt}\n\nProvide a final answer with evidence and uncertainty."


def _build_holmes_request(
    body: dict[str, Any],
    extra_files: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any] | None, bool]:
    messages = list(body.get("messages") or [])
    latest_index = _latest_user_index(messages)
    if latest_index is None:
        return None, False

    prompt = _message_text(messages[latest_index].get("content"))
    if not prompt.strip():
        return None, False

    attachment_body = dict(body)
    attachment_body["files"] = list(body.get("files") or []) + list(extra_files or [])
    paths, inline_fallback = _prepare_attachments(attachment_body)
    has_files = bool(
        paths
        or inline_fallback
        or "### metrics/" in prompt
        or "### dashboards/" in prompt
    )

    if paths:
        prompt += (
            "\n\nEvidence files accessible inside the Holmes pod:\n"
            + "\n".join(f"- {path}" for path in paths)
            + "\nUse the read_attachment tool. Do not use bash/shell to read files."
        )
    if inline_fallback:
        prompt += "\n\nInline evidence:\n" + inline_fallback

    request_body: dict[str, Any] = {
        "ask": _wrap_prompt(prompt, has_files),
        "stream": True,
        "enable_tool_approval": False,
    }
    history = _conversation_history(body, latest_index)
    if history:
        request_body["conversation_history"] = history
    return request_body, has_files


def _tool_name(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("tool_name", "name", "tool"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:100]
    return "tool"


def _progress_message(tool_name: str) -> str:
    lower = tool_name.lower()
    if "grafana" in lower or "dashboard" in lower:
        return "Checking Grafana dashboards…"
    if any(name in lower for name in ("prometheus", "metric", "thanos")):
        return "Checking Prometheus metrics…"
    if any(name in lower for name in ("apm", "trace", "span", "transaction")):
        return "Checking APM traces…"
    return f"Running {tool_name}…"


def _final_analysis(payload: Any) -> str | None:
    """Extract only the terminal assistant answer from Holmes 0.43.0."""
    if not isinstance(payload, dict):
        return None
    analysis = payload.get("analysis")
    if isinstance(analysis, str) and analysis.strip():
        sanitized = _sanitize_answer(analysis)
        if sanitized:
            return sanitized

    # Holmes' terminal SSE also contains its complete conversation history.
    # This is the safest fallback when the top-level analysis field is empty or
    # has been polluted with provider-specific reasoning tags.
    history = payload.get("conversation_history") or payload.get("messages")
    if isinstance(history, list):
        for message in reversed(history):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = _message_text(message.get("content"))
            if content.strip():
                sanitized = _sanitize_answer(content)
                if sanitized:
                    return sanitized
    return None


class Pipe:
    class Valves(BaseModel):
        HOLMES_URL: str = Field(default="http://holmes:8000", description="HolmesGPT 0.43.0 base URL")
        HOLMES_API_KEY: str = Field(default="", description="Optional Holmes X-API-Key")
        HOLMES_CONNECT_TIMEOUT: float = Field(default=10.0)
        HOLMES_READ_TIMEOUT: float = Field(default=300.0)
        HOLMES_HEALTH_TIMEOUT: float = Field(default=5.0)
        HOLMES_HEALTH_CACHE_SECONDS: float = Field(default=10.0)
        HOLMES_HEALTHCHECK_ENABLED: bool = Field(default=True)
        HOLMES_HEARTBEAT_SECONDS: float = Field(
            default=15.0,
            description="Emit a non-content OpenWebUI status heartbeat while Holmes is silent",
        )
        HOLMES_CONNECT_RETRIES: int = Field(
            default=1,
            description="Safe default is one attempt; retries are connect-only and pre-headers",
        )
        HOLMES_CONNECT_BACKOFF: float = Field(default=1.0)
        HOLMES_EVENT_QUEUE_SIZE: int = Field(default=64)
        HOLMES_TRUST_ENV: bool = Field(
            default=False,
            description="Keep false for closed-cluster traffic unless an internal proxy is required",
        )

    def __init__(self) -> None:
        self.id = "holmes"
        self.name = "HolmesGPT"
        self.description = "Investigate operational issues via HolmesGPT"
        self.valves = self.Valves()
        self._health_lock = asyncio.Lock()
        self._health_ok_until = 0.0

    def _setting(self, key: str) -> Any:
        current = getattr(self.valves, key)
        raw = os.getenv(key)
        if raw is None:
            return current
        if isinstance(current, bool):
            return raw.lower() in ("true", "1", "yes", "on")
        try:
            return type(current)(raw)
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid setting key=%s", key)
            return current

    def _headers(self, request_id: str) -> dict[str, str]:
        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "X-Request-ID": request_id,
        }
        api_key = str(self._setting("HOLMES_API_KEY")).strip()
        if api_key:
            headers["X-API-Key"] = api_key
        return headers

    async def _check_health(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        request_id: str,
    ) -> None:
        if not bool(self._setting("HOLMES_HEALTHCHECK_ENABLED")):
            return
        now = time.monotonic()
        if now < self._health_ok_until:
            return

        async with self._health_lock:
            now = time.monotonic()
            if now < self._health_ok_until:
                return

            # Holmes 0.43.0 exposes both paths. Readiness is preferred; health is
            # only a compatibility fallback when readiness is not routed.
            for path in ("/readyz", "/healthz"):
                response = await client.get(
                    base_url + path,
                    headers=self._headers(request_id),
                    timeout=float(self._setting("HOLMES_HEALTH_TIMEOUT")),
                )
                if response.status_code == 404 and path == "/readyz":
                    continue
                response.raise_for_status()
                self._health_ok_until = time.monotonic() + max(
                    0.0, float(self._setting("HOLMES_HEALTH_CACHE_SECONDS"))
                )
                return
            raise RuntimeError("Holmes health endpoints are unavailable")

    @staticmethod
    async def _emit_status(
        emitter: EventEmitter | None,
        description: str,
        *,
        done: bool,
        hidden: bool = False,
    ) -> None:
        if emitter is None:
            return
        try:
            await emitter(
                {
                    "type": "status",
                    "data": {
                        "description": description,
                        "done": done,
                        "hidden": hidden,
                    },
                }
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # A status delivery failure must not destroy the model response.
            logger.debug("OpenWebUI status emission failed error=%s", type(exc).__name__)

    @staticmethod
    async def _emit_error(emitter: EventEmitter | None, message: str) -> None:
        if emitter is None:
            return
        try:
            await emitter(
                {
                    "type": "chat:message:error",
                    "data": {"error": {"content": message}},
                }
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("OpenWebUI error emission failed error=%s", type(exc).__name__)

    async def pipe(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any] | None = None,
        __event_emitter__: EventEmitter | None = None,
        __metadata__: dict[str, Any] | None = None,
        __request__: Any = None,
        __files__: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        del __user__  # Reserved for future authorization/audit enrichment.
        del __request__  # Cancellation is delivered by OpenWebUI through generator close.

        request_id = uuid.uuid4().hex[:12]
        metadata = __metadata__ or {}
        chat_id = str(metadata.get("chat_id") or "-")[:64]
        message_id = str(metadata.get("message_id") or "-")[:64]
        started = time.monotonic()
        outcome = "started"
        first_event_at: float | None = None
        final_event_at: float | None = None
        headers_at: float | None = None
        heartbeat_count = 0
        queue_high_water = 0
        event_counts: Counter[str] = Counter()
        token_usage: dict[str, Any] = {}
        finish_reason = "-"
        final_chars = 0
        reader_task: asyncio.Task[None] | None = None

        request_body, has_files = _build_holmes_request(body, __files__)
        if request_body is None:
            outcome = "invalid_request"
            yield "No user message found."
            return

        base_url = str(self._setting("HOLMES_URL")).rstrip("/")
        chat_url = base_url + "/api/chat"
        timeout = httpx.Timeout(
            connect=float(self._setting("HOLMES_CONNECT_TIMEOUT")),
            read=float(self._setting("HOLMES_READ_TIMEOUT")),
            write=30.0,
            pool=30.0,
        )
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(
            maxsize=max(1, int(self._setting("HOLMES_EVENT_QUEUE_SIZE")))
        )

        logger.info(
            "HOLMES_PIPE start request_id=%s chat_id=%s message_id=%s files=%s history_messages=%d prompt_chars=%d",
            request_id,
            chat_id,
            message_id,
            has_files,
            len(request_body.get("conversation_history") or []),
            len(str(request_body.get("ask") or "")),
        )

        async with httpx.AsyncClient(
            timeout=timeout,
            trust_env=bool(self._setting("HOLMES_TRUST_ENV")),
        ) as client:
            try:
                await self._emit_status(
                    __event_emitter__,
                    "Checking HolmesGPT readiness…",
                    done=False,
                )
                await self._check_health(client, base_url, request_id)
                await self._emit_status(
                    __event_emitter__,
                    "Starting SRE investigation…",
                    done=False,
                )

                async def read_holmes_stream() -> None:
                    nonlocal first_event_at, headers_at, queue_high_water
                    attempts = max(1, int(self._setting("HOLMES_CONNECT_RETRIES")))
                    for attempt in range(attempts):
                        response_headers_received = False
                        try:
                            async with client.stream(
                                "POST",
                                chat_url,
                                json=request_body,
                                headers=self._headers(request_id),
                            ) as response:
                                response_headers_received = True
                                headers_at = time.monotonic()
                                response.raise_for_status()
                                content_type = response.headers.get("content-type", "")
                                if "text/event-stream" not in content_type.lower():
                                    raise RuntimeError(
                                        f"Holmes returned unexpected content type: {content_type or 'missing'}"
                                    )

                                async for event_type, raw in _parse_sse_async(response.aiter_lines()):
                                    if raw == "[DONE]":
                                        return
                                    if first_event_at is None:
                                        first_event_at = time.monotonic()
                                    event_counts[event_type] += 1
                                    await queue.put((event_type, _json_payload(raw)))
                                    queue_high_water = max(queue_high_water, queue.qsize())
                                return
                        except (httpx.ConnectError, httpx.ConnectTimeout):
                            # Never retry once Holmes accepted the request headers. A POST
                            # may execute tools, so replay after that point is unsafe.
                            if response_headers_received or attempt + 1 >= attempts:
                                raise
                            await asyncio.sleep(
                                max(0.0, float(self._setting("HOLMES_CONNECT_BACKOFF")))
                                * (2**attempt)
                            )

                reader_task = asyncio.create_task(
                    read_holmes_stream(),
                    name=f"holmes-reader-{request_id}",
                )

                final_answer: str | None = None
                terminal_error: str | None = None
                terminal_event = False
                heartbeat_seconds = max(
                    5.0, float(self._setting("HOLMES_HEARTBEAT_SECONDS"))
                )

                while not terminal_event:
                    if reader_task.done() and queue.empty():
                        break
                    try:
                        event_type, payload = await asyncio.wait_for(
                            queue.get(), timeout=heartbeat_seconds
                        )
                    except asyncio.TimeoutError:
                        if reader_task.done() and queue.empty():
                            break
                        heartbeat_count += 1
                        elapsed = int(time.monotonic() - started)
                        await self._emit_status(
                            __event_emitter__,
                            f"HolmesGPT is still investigating… ({elapsed}s)",
                            done=False,
                        )
                        continue

                    if event_type == HOLMES_TOOL_START:
                        await self._emit_status(
                            __event_emitter__,
                            _progress_message(_tool_name(payload)),
                            done=False,
                        )
                    elif event_type == HOLMES_TOOL_RESULT:
                        await self._emit_status(
                            __event_emitter__,
                            f"Finished {_tool_name(payload)}; analyzing results…",
                            done=False,
                        )
                    elif event_type == HOLMES_AI_MESSAGE:
                        # This event is an intermediate tool-loop message and may
                        # contain a provider-neutral Holmes reasoning field. Do not
                        # emit its text, reasoning, or an event-per-iteration status.
                        pass
                    elif event_type == HOLMES_TOKEN_COUNT:
                        if isinstance(payload, dict):
                            metadata_payload = payload.get("metadata")
                            if isinstance(metadata_payload, dict):
                                usage = metadata_payload.get("usage")
                                if isinstance(usage, dict):
                                    token_usage = usage
                    elif event_type == HOLMES_COMPACTION_START:
                        await self._emit_status(
                            __event_emitter__,
                            "Compacting investigation history…",
                            done=False,
                        )
                    elif event_type == HOLMES_COMPACTED:
                        await self._emit_status(
                            __event_emitter__,
                            "Investigation history compacted; continuing…",
                            done=False,
                        )
                    elif event_type == HOLMES_FINAL:
                        final_event_at = time.monotonic()
                        if isinstance(payload, dict):
                            final_metadata = payload.get("metadata")
                            if isinstance(final_metadata, dict):
                                raw_finish_reason = final_metadata.get("finish_reason")
                                if isinstance(raw_finish_reason, str):
                                    finish_reason = raw_finish_reason
                        final_answer = _final_analysis(payload)
                        final_chars = len(final_answer) if final_answer else 0
                        if not final_answer:
                            if finish_reason == "length":
                                terminal_error = (
                                    "HolmesGPT used its output budget before producing a final answer. "
                                    "Increase the model output-token limit or reduce conversation history."
                                )
                            else:
                                terminal_error = (
                                    "HolmesGPT finished without a user-visible final answer. "
                                    "The reasoning output was intentionally not exposed."
                                )
                        terminal_event = True
                    elif event_type == HOLMES_APPROVAL:
                        terminal_error = (
                            "HolmesGPT requested tool approval, but this Pipe is configured "
                            "for non-interactive investigations."
                        )
                        terminal_event = True
                    elif event_type == "error":
                        terminal_error = "HolmesGPT reported an investigation error."
                        terminal_event = True
                    # Unknown events are ignored and counted for the completion log.

                if terminal_event and reader_task is not None and not reader_task.done():
                    # ai_answer_end is the protocol completion boundary. Do not wait
                    # for a misbehaving upstream socket to close after the final event.
                    reader_task.cancel()

                if reader_task is not None:
                    try:
                        await reader_task
                    except asyncio.CancelledError:
                        if not terminal_event:
                            raise

                if terminal_error:
                    outcome = "upstream_terminal_error"
                    await self._emit_error(__event_emitter__, terminal_error)
                    await self._emit_status(
                        __event_emitter__,
                        "Investigation ended with an error",
                        done=True,
                    )
                    yield terminal_error
                    return

                if not final_answer:
                    outcome = "upstream_ended_without_final"
                    message = "HolmesGPT ended the stream without a final answer."
                    await self._emit_error(__event_emitter__, message)
                    await self._emit_status(
                        __event_emitter__,
                        "Investigation ended without a final answer",
                        done=True,
                    )
                    yield message
                    return

                outcome = "success"
                await self._emit_status(
                    __event_emitter__,
                    "Investigation complete",
                    done=True,
                )
                # This is the only successful assistant-content yield. OpenWebUI
                # 0.11.0 wraps it as SSE and emits finish_reason + [DONE].
                yield final_answer

            except asyncio.CancelledError:
                outcome = "cancelled"
                logger.info("HOLMES_PIPE cancelled request_id=%s", request_id)
                raise
            except httpx.HTTPStatusError as exc:
                outcome = f"http_{exc.response.status_code}"
                message = f"HolmesGPT request failed with HTTP {exc.response.status_code}."
                logger.exception("HOLMES_PIPE upstream_status_error request_id=%s", request_id)
                await self._emit_error(__event_emitter__, message)
                await self._emit_status(__event_emitter__, "Investigation failed", done=True)
                yield message
            except (httpx.HTTPError, RuntimeError) as exc:
                outcome = type(exc).__name__
                message = f"HolmesGPT is unavailable: {type(exc).__name__}."
                logger.exception("HOLMES_PIPE upstream_error request_id=%s", request_id)
                await self._emit_error(__event_emitter__, message)
                await self._emit_status(__event_emitter__, "Investigation failed", done=True)
                yield message
            except Exception as exc:
                outcome = type(exc).__name__
                message = "The HolmesGPT investigation failed unexpectedly."
                logger.exception("HOLMES_PIPE unexpected_error request_id=%s", request_id)
                await self._emit_error(__event_emitter__, message)
                await self._emit_status(__event_emitter__, "Investigation failed", done=True)
                yield message
            finally:
                # Cleanup only. Yielding here would break async-generator close and
                # is the primary failure mode fixed from the original Pipe.
                if reader_task is not None and not reader_task.done():
                    reader_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await reader_task

                ended = time.monotonic()
                total_ms = int((ended - started) * 1000)
                headers_ms = int((headers_at - started) * 1000) if headers_at else -1
                first_event_ms = (
                    int((first_event_at - started) * 1000) if first_event_at else -1
                )
                final_event_ms = (
                    int((final_event_at - started) * 1000) if final_event_at else -1
                )
                logger.info(
                    "HOLMES_PIPE end request_id=%s outcome=%s total_ms=%d headers_ms=%d first_event_ms=%d final_event_ms=%d finish_reason=%s final_chars=%d heartbeats=%d queue_high_water=%d events=%s prompt_tokens=%s completion_tokens=%s",
                    request_id,
                    outcome,
                    total_ms,
                    headers_ms,
                    first_event_ms,
                    final_event_ms,
                    finish_reason,
                    final_chars,
                    heartbeat_count,
                    queue_high_water,
                    dict(event_counts),
                    token_usage.get("prompt_tokens", "-"),
                    token_usage.get("completion_tokens", "-"),
                )
