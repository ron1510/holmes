"""
title: HolmesGPT
author: AI Investigation POC
version: 0.5.0
license: MIT
description: Open WebUI pipe for HolmesGPT investigations with SSE progress and answers.
requirements: httpx
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import re
import shutil
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
UPLOADS = Path('/app/backend/data/uploads')
ATTACHMENTS = Path('/app/backend/data/investigation-attachments')
MAX_FILE_CHARS = 100_000
MAX_TOTAL_CHARS = 300_000
MAX_ARRAY_LENGTH = 300
NOISE_KEYS = {'evidence', 'timestamp.us', 'unattributed_ms_estimate'}
ANSWER_EVENTS = {'ai_answer_end', 'ai_answer', 'answer', 'final_answer'}
DELTA_EVENTS = {'ai_answer_delta', 'answer_delta', 'text_delta', 'token', 'ai_answer_chunk'}
TOOL_START = {'tool_call', 'start_tool_calling'}
TOOL_END = {'tool_result', 'tool_calling_result'}


def _parse_sse(lines: Iterator[str | bytes]) -> Iterator[tuple[str, str]]:
    """Parse SSE; preserve data spacing and flush the final unterminated event."""
    kind = 'message'
    parts: list[str] = []
    for raw in lines:
        line = raw.decode('utf-8', 'replace') if isinstance(raw, bytes) else raw
        line = line.rstrip('\r\n')
        if line == '':
            if parts:
                yield kind, '\n'.join(parts)
            kind, parts = 'message', []
        elif line.startswith('event:'):
            kind = line[6:].strip()
        elif line.startswith('data:'):
            value = line[5:]
            parts.append(value[1:] if value.startswith(' ') else value)
        # SSE comments and other fields intentionally ignored.
    if parts:
        yield kind, '\n'.join(parts)


async def _parse_sse_async(lines) :
    """Parse SSE; preserve data spacing and flush the final unterminated event."""
    kind = 'message'
    parts: list[str] = []
    async for raw in lines:
        line = raw.decode('utf-8', 'replace') if isinstance(raw, bytes) else raw
        line = line.rstrip('\r\n')
        if line == '':
            if parts:
                yield kind, '\n'.join(parts)
            kind, parts = 'message', []
        elif line.startswith('event:'):
            kind = line[6:].strip()
        elif line.startswith('data:'):
            value = line[5:]
            parts.append(value[1:] if value.startswith(' ') else value)
        # SSE comments and other fields intentionally ignored.
    if parts:
        yield kind, '\n'.join(parts)


def _payload(data: str) -> Any:
    try:
        return json.loads(data)
    except (ValueError, TypeError):
        return data


def _extract_text(value: Any, *, allow_nested: bool = True) -> str | None:
    """Conservative text extraction; never stringify arbitrary JSON as the answer."""
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, list):
        texts = [_extract_text(x) for x in value]
        result = ''.join(x for x in texts if x)
        return result or None
    if not isinstance(value, dict):
        return None
    for key in ('analysis', 'answer', 'content', 'text', 'message', 'response', 'output', 'delta'):
        if key in value:
            text = _extract_text(value[key])
            if text:
                return text
    if allow_nested:
        for key in ('data', 'result', 'final', 'choices'):
            if key in value:
                text = _extract_text(value[key])
                if text:
                    return text
    return None


def _extract_tool_name(data: str) -> str:
    obj = _payload(data)
    if isinstance(obj, dict):
        for candidate in (obj, obj.get('tool_call'), obj.get('function')):
            if isinstance(candidate, dict):
                for key in ('tool_name', 'name', 'tool', 'toolset_name'):
                    name = candidate.get(key)
                    if isinstance(name, str) and name:
                        return name
    return 'tool'


def _progress(name: str) -> str:
    lower = name.lower()
    if 'grafana' in lower or 'dashboard' in lower:
        return '🔍 Checking Grafana dashboards...'
    if any(x in lower for x in ('prometheus', 'metric', 'thanos')):
        return '🔍 Checking Prometheus metrics...'
    if any(x in lower for x in ('apm', 'trace', 'span', 'transaction')):
        return '🔍 Checking APM traces...'
    return f'🔍 Checking {name[:80]}...'


def _sanitize(text: str) -> str:
    text = re.sub(r'!\[[^\]]*\]\(tool-image://[^)]*\)', '', text)
    text = re.sub(r'<img\b[^>]*>', '', text, flags=re.I)
    text = re.sub(r'<<\[.*?\]>>', '', text, flags=re.S)
    return text


def _strip_preamble(text: str) -> str:
    """Avoid discarding valid answers that do not have an Executive Summary."""
    marker = '### Executive Summary'
    i = text.find(marker)
    if i >= 0 and i < 2000:
        return _sanitize(text[i:])
    return _sanitize(text)


def _safe_path(path: Path) -> bool:
    try:
        resolved = path.resolve(strict=True)
        return resolved.is_file() and any(resolved.is_relative_to(base.resolve()) for base in (UPLOADS, ATTACHMENTS))
    except (OSError, ValueError):
        return False


def _collect_files(body: dict[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    candidates = list(body.get('files') or [])
    for msg in reversed(body.get('messages') or []):
        if isinstance(msg, dict) and msg.get('role') == 'user':
            candidates += list(msg.get('files') or [])
            break
    for item in candidates:
        if not isinstance(item, dict):
            continue
        info = item.get('file') or item
        if not isinstance(info, dict):
            continue
        key = str(info.get('id') or item.get('id') or info.get('path') or uuid.uuid4())
        if key not in seen:
            seen.add(key)
            found.append(item)
    return found


def _resolve_file(item: dict[str, Any]) -> Path | None:
    info = item.get('file') or item
    if not isinstance(info, dict):
        return None
    direct = info.get('path')
    if isinstance(direct, str) and direct:
        candidate = Path(direct)
        if _safe_path(candidate):
            return candidate
    fid = info.get('id') or item.get('id')
    filename = Path(str(info.get('filename') or info.get('name') or '')).name
    if not isinstance(fid, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', fid):
        return None
    candidates = [UPLOADS / f'{fid}_{filename}'] if filename else []
    candidates += list(UPLOADS.glob(f'{fid}_*'))[:20]
    for candidate in candidates:
        if _safe_path(candidate):
            return candidate
    return None


def _simplify_json(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _simplify_json(v) for k, v in data.items()
                if k not in NOISE_KEYS and not k.startswith('kubernetes.')}
    if isinstance(data, list):
        items = [_simplify_json(v) for v in data[:MAX_ARRAY_LENGTH]]
        if len(data) > MAX_ARRAY_LENGTH:
            items.append({'omitted_items': len(data) - MAX_ARRAY_LENGTH})
        return items
    return data


def _inline_content(path: Path, limit: int) -> str:
    if limit <= 0:
        return ''
    try:
        if path.suffix.lower() not in ('.txt', '.md', '.csv', '.json', '.yaml', '.yml', '.log'):
            return f'[Binary attachment {path.name}: requires Holmes read_attachment tool]'
        # Bounded read; do not load unlimited attachments into memory.
        with path.open('r', encoding='utf-8', errors='replace') as fh:
            content = fh.read(min(limit * 3, 1_000_000))
        if path.suffix.lower() == '.json':
            try:
                content = json.dumps(_simplify_json(json.loads(content)), ensure_ascii=False)
            except ValueError:
                pass
        if len(content) > limit:
            content = content[:limit] + '\n[Truncated]'
        return content
    except OSError as exc:
        logger.warning('Cannot read attachment %s: %s', path.name, exc)
        return ''


def _prepare_attachments(body: dict[str, Any]) -> tuple[list[Path], str]:
    files = _collect_files(body)
    if not files:
        return [], ''
    dest_dir = ATTACHMENTS / str(uuid.uuid4())
    copied: list[Path] = []
    inline: list[str] = []
    remaining = MAX_TOTAL_CHARS
    for item in files:
        source = _resolve_file(item)
        if source is None:
            logger.warning('Could not resolve attachment descriptor')
            continue
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / (str(uuid.uuid4())[:8] + '_' + source.name)
            shutil.copyfile(source, dest)
            dest.chmod(0o644)
            copied.append(dest)
        except OSError as exc:
            logger.warning('PVC copy failed for %s: %s', source.name, exc)
            content = _inline_content(source, min(MAX_FILE_CHARS, remaining))
            if content:
                inline.append(f'### {source.name}\n```\n{content}\n```')
                remaining -= len(content)
    return copied, '\n\n'.join(inline)


def _latest_prompt(body: dict[str, Any]) -> str:
    for msg in reversed(body.get('messages') or []):
        if isinstance(msg, dict) and msg.get('role') == 'user':
            content = msg.get('content', '')
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return '\n'.join(x.get('text', '') for x in content
                                 if isinstance(x, dict) and x.get('type') == 'text')
    return ''


def _build_holmes_prompt(body: dict[str, Any]) -> tuple[str, bool]:
    prompt = _latest_prompt(body)
    if not prompt.strip():
        return '', False
    paths, fallback = _prepare_attachments(body)
    has_files = bool(paths or fallback or '### metrics/' in prompt or '### dashboards/' in prompt)
    if paths:
        prompt += ('\n\nEvidence files accessible inside the Holmes pod:\n'
                   + '\n'.join(f'- {path}' for path in paths)
                   + '\nUse the read_attachment tool. Do not use bash/shell to read files.')
    if fallback:
        prompt += '\n\nInline evidence:\n' + fallback
    return prompt, has_files


def _wrap_prompt(prompt: str, has_files: bool) -> str:
    lower = prompt.lower()
    apm = any(word in lower for word in ('apm', 'trace', 'span', 'transaction', 'dependency'))
    sre = any(word in lower for word in ('cluster', 'kafka', 'mongodb', 'postgres', 'prometheus', 'kubernetes', 'pod'))
    if apm:
        instruction = 'Use the apm-investigation skill when relevant.'
    elif sre:
        instruction = 'Use the sre-orchestrator skill when relevant.'
    else:
        instruction = 'Answer directly; choose an investigation skill only if needed.'
    if has_files:
        instruction += (' Investigate the attached evidence first. Do not query live APM, '
                        'Prometheus or Grafana unless necessary.')
    return f'{instruction}\n\nUser request: {prompt}\n\nProvide a final answer with evidence and uncertainty.'


class Pipe:
    class Valves(BaseModel):
        HOLMES_URL: str = Field(default='http://holmes:8000', description='HolmesGPT base URL')
        HOLMES_TIMEOUT: int = Field(default=300)
        HOLMES_CONNECT_TIMEOUT: float = Field(default=20.0)
        HOLMES_RETRIES: int = Field(default=3)
        HOLMES_BACKOFF: float = Field(default=1.0)
        HOLMES_WORD_DELAY: float = Field(default=0.0, description='Optional simulated delay for full-answer-only Holmes')
        HOLMES_HEARTBEAT_SECONDS: float = Field(default=5.0, description='Reserved; synchronous SSE does not create heartbeat threads')
        REASONING_TAGS: bool = Field(default=True, description='Use <think> tags; requires Open WebUI reasoning parser support')
        SHOW_TOOL_PROGRESS: bool = Field(default=True)
        LOG_SSE_TYPES: bool = Field(default=False)

    def __init__(self) -> None:
        self.id = 'holmes'
        self.name = 'HolmesGPT'
        self.description = 'Investigate operational issues via HolmesGPT'
        self.valves = self.Valves()

    def _setting(self, key: str) -> Any:
        current = getattr(self.valves, key)
        raw = os.getenv(key)
        if raw is None:
            return current
        if isinstance(current, bool):
            return raw.lower() in ('true', '1', 'yes', 'on')
        try:
            return type(current)(raw)
        except ValueError:
            return current

    async def pipe(self, body: dict[str, Any], __user__: dict[str, Any] | None = None):
        prompt, has_files = _build_holmes_prompt(body)
        if not prompt:
            yield 'No user message found.'
            return
        url = str(self._setting('HOLMES_URL')).rstrip('/') + '/api/chat'
        wrapped = _wrap_prompt(prompt, has_files)
        request_id = uuid.uuid4().hex[:10]
        started = time.monotonic()
        logger.warning('HOLMES_DIAG pipe_start id=%s files=%s prompt_chars=%d',
                       request_id, has_files, len(wrapped))
        chunks = 0
        try:
            async for chunk in self._stream_response(url, {'ask': wrapped, 'stream': True}, request_id=request_id):
                chunks += 1
                yield chunk
        finally:
            logger.warning('HOLMES_DIAG pipe_end id=%s chunks=%d duration_s=%.3f',
                           request_id, chunks, time.monotonic() - started)

    async def _stream_response(self, url: str, json_body: dict[str, Any], request_id: str = "-") -> Iterator[str]:
        thinking = False
        answer_started = False
        streamed_parts: list[str] = []
        last_progress = ''
        tags = bool(self._setting('REASONING_TAGS'))

        def close_thinking() -> Iterator[str]:
            nonlocal thinking
            if thinking:
                thinking = False
                if tags:
                    yield '\n</think>\n\n'
                else:
                    yield '\n\n'

        try:
            try:
                timeout = httpx.Timeout(connect=float(self._setting('HOLMES_CONNECT_TIMEOUT')),
                                        read=float(self._setting('HOLMES_TIMEOUT')),
                                        write=30.0, pool=30.0)
                async with httpx.AsyncClient(timeout=timeout, trust_env=True) as client:
                    retries = max(1, int(self._setting('HOLMES_RETRIES')))
                    for attempt in range(retries):
                        try:
                            logger.warning('HOLMES_DIAG http_start id=%s attempt=%d', request_id, attempt + 1)
                            async with client.stream('POST', url, json=json_body,
                                                     headers={'Accept': 'text/event-stream'}) as response:
                                response.raise_for_status()
                                logger.warning('HOLMES_DIAG http_headers id=%s status=%d', request_id, response.status_code)
                                last_event = time.monotonic()
                                event_count = 0
                                async for event_type, raw in _parse_sse_async(response.aiter_lines()):
                                    now = time.monotonic()
                                    event_count += 1
                                    logger.warning('HOLMES_DIAG sse_event id=%s index=%d type=%s gap_s=%.3f bytes=%d',
                                                   request_id, event_count, event_type, now - last_event, len(raw))
                                    last_event = now
                                    if self._setting('LOG_SSE_TYPES'):
                                        logger.info('Holmes SSE event=%s bytes=%d', event_type, len(raw))
                                    if raw == '[DONE]':
                                        break
                                    if event_type in TOOL_START:
                                        if answer_started or not self._setting('SHOW_TOOL_PROGRESS'):
                                            continue
                                        message = _progress(_extract_tool_name(raw))
                                        if message == last_progress:
                                            continue
                                        last_progress = message
                                        if not thinking:
                                            thinking = True
                                            if tags:
                                                yield '<think>\n'
                                        yield message + '\n'
                                        continue
                                    if event_type in TOOL_END or event_type in (
                                        'conversation_history_compaction_start', 'conversation_history_compacted'
                                    ):
                                        continue
                                    if event_type == 'ai_answer_start':
                                        for _chunk in close_thinking():
                                            yield _chunk
                                        answer_started = True
                                        continue
                                    if event_type in DELTA_EVENTS:
                                        for _chunk in close_thinking():
                                            yield _chunk
                                        answer_started = True
                                        fragment = _extract_text(_payload(raw))
                                        if fragment:
                                            streamed_parts.append(fragment)
                                            yield fragment  # true incremental streaming when upstream provides deltas
                                        continue
                                    if event_type in ANSWER_EVENTS:
                                        for _chunk in close_thinking():
                                            yield _chunk
                                        answer_started = True
                                        final = _extract_text(_payload(raw))
                                        if not final:
                                            if not streamed_parts:
                                                yield 'HolmesGPT finished without an extractable final answer. Check SSE event payload format.'
                                            continue
                                        if streamed_parts:
                                            prior = ''.join(streamed_parts)
                                            # End event may contain a full cumulative answer. Avoid duplicating streamed text.
                                            if final.startswith(prior):
                                                remainder = final[len(prior):]
                                                if remainder:
                                                    yield remainder
                                            elif prior.startswith(final) or final == streamed_parts[-1]:
                                                pass
                                            else:
                                                logger.warning('Final answer differs from streamed deltas; keeping streamed answer')
                                        else:
                                            final = _strip_preamble(final)
                                            delay = float(self._setting('HOLMES_WORD_DELAY'))
                                            if delay > 0:
                                                for line in final.splitlines(keepends=True):
                                                    yield line
                                                    await asyncio.sleep(delay)
                                            else:
                                                yield final
                                        continue
                                    if event_type == 'error':
                                        for _chunk in close_thinking():
                                            yield _chunk
                                        yield '\nHolmesGPT reported an error during investigation.'
                                        logger.error('HolmesGPT error event (details omitted)')
                                        continue
                                    if event_type == 'approval_required':
                                        for _chunk in close_thinking():
                                            yield _chunk
                                        yield '\nHolmesGPT requested tool approval, not configured in this interface.'
                                        continue
                                    if event_type == 'message':
                                        # Some deployments send JSON in unnamed SSE events.
                                        payload = _payload(raw)
                                        if isinstance(payload, dict) and isinstance(payload.get('type'), str):
                                            nested_type = payload['type']
                                            if nested_type in DELTA_EVENTS:
                                                for _chunk in close_thinking():
                                                    yield _chunk
                                                answer_started = True
                                                fragment = _extract_text(payload)
                                                if fragment:
                                                    streamed_parts.append(fragment)
                                                    yield fragment
                                            elif nested_type in ANSWER_EVENTS:
                                                for _chunk in close_thinking():
                                                    yield _chunk
                                                answer_started = True
                                                final = _extract_text(payload)
                                                if final and not streamed_parts:
                                                    yield _strip_preamble(final)
                                        elif isinstance(payload, str) and payload.strip() and not answer_started:
                                            # Plain text non-SSE fallback, not a tool result.
                                            for _chunk in close_thinking():
                                                yield _chunk
                                            answer_started = True
                                            yield payload
                                        continue
                                    logger.debug('Unhandled Holmes SSE type %s', event_type)
                            break
                        except httpx.ConnectError:
                            if attempt + 1 == retries:
                                raise
                            await asyncio.sleep(float(self._setting('HOLMES_BACKOFF')) * 2 ** attempt)
            except httpx.HTTPError as exc:
                for _chunk in close_thinking():
                    yield _chunk
                logger.exception('HolmesGPT streaming request failed')
                yield f'HolmesGPT request failed: {type(exc).__name__}'
        finally:
            for _chunk in close_thinking():
                yield _chunk
