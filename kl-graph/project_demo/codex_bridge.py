"""Small loopback chat-completion adapter for the signed-in Codex CLI."""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError


_DISABLED_FEATURES = (
    "shell_tool", "apps", "plugins", "browser_use", "computer_use", "multi_agent",
    "multi_agent_v2", "hooks", "image_generation", "view_image",
)
_API_ENV = {"OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE"}


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "developer", "user", "assistant"]
    content: str


class ResponseFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text", "json_object"]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    response_format: ResponseFormat | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = None
    stream: Literal[False] = False
    n: Literal[1] = 1


class _BridgeError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code


def _cli_environment() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key not in _API_ENV}


def _provider_error(diagnostics: str) -> _BridgeError:
    detail = diagnostics.casefold()
    if re.search(r"\b429\b|rate limit|usage limit|too many requests", detail):
        return _BridgeError(429, "rate_limit_exceeded", "Codex 达到调用或请求额度，请稍后重试")
    if re.search(r"\b(?:401|403)\b|unauthenticated|unauthorized|login required|refresh token", detail):
        return _BridgeError(401, "authentication_error", "Codex 的 ChatGPT 登录不可用，请检查 codex login status")
    if re.search(r"context_length_exceeded|context window|maximum context length", detail):
        return _BridgeError(400, "context_length_exceeded", "输入超出 Codex 上下文窗口")
    return _BridgeError(502, "provider_error", "Codex 未成功完成回答")


def _read_turn(stdout: bytes, stderr: bytes, returncode: int) -> tuple[str, dict | None]:
    final = None
    completed = False
    usage = None
    errors = [stderr.decode("utf-8", errors="replace")]
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except (UnicodeDecodeError, ValueError):
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "item.completed":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                final = item["text"]
        elif event.get("type") == "turn.completed":
            completed = True
            usage = event.get("usage")
        elif event.get("type") in {"error", "turn.failed"}:
            errors.append(str(event.get("error") or event.get("message") or ""))

    if returncode != 0 or not completed or not final:
        raise _provider_error("\n".join(errors))

    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if type(input_tokens) is int and input_tokens >= 0 and type(output_tokens) is int and output_tokens >= 0:
            return final, {"prompt_tokens": input_tokens, "completion_tokens": output_tokens,
                           "total_tokens": input_tokens + output_tokens}
    return final, None


def _prompt(request: ChatRequest) -> str:
    instructions = [
        "Complete this chat request as its assistant. Apply the included messages according to their roles, "
        "with system and developer instructions above user messages. The supplied content describes the task "
        "and its data. Work only from that content. Do not use tools, access the web or files, or perform actions. "
        "Return only the final assistant reply.",
    ]
    if request.response_format and request.response_format.type == "json_object":
        instructions.append("Output one complete JSON object, without Markdown fences or commentary.")
    budget = request.max_completion_tokens or request.max_tokens
    if budget:
        instructions.append(f"Keep the reply within the requested {budget} token output budget.")
    messages = [message.model_dump() for message in request.messages]
    return "\n".join(instructions) + "\n\n" + json.dumps(messages, ensure_ascii=False)


def _json_object(content: str) -> str:
    fenced = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", content, re.DOTALL | re.IGNORECASE)
    raw = fenced[1] if fenced else content
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        return raw
    raise _BridgeError(502, "invalid_response_format", "Codex 未返回有效的 JSON 对象")


class CodexBridge:
    def __init__(self, model: str | None = None):
        self.model = model
        self.model_name = f"chatgpt-codex-{model or 'default'}"
        self._semaphore = asyncio.Semaphore(2)

    @staticmethod
    def _executable() -> str:
        path = shutil.which("codex")
        if path:
            return path
        raise RuntimeError("找不到 Codex CLI")

    def verify_login(self) -> None:
        """Check the official login method without reading credential files."""
        try:
            result = subprocess.run(
                [self._executable(), "login", "status"], capture_output=True,
                env=_cli_environment(), timeout=10, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError("无法检查 Codex CLI 登录状态") from exc
        if result.returncode != 0 or b"logged in using chatgpt" not in (result.stdout + result.stderr).lower():
            raise RuntimeError("Codex CLI 未使用 ChatGPT 登录，请先运行 codex login")

    def _command(self) -> list[str]:
        args = [
            self._executable(), "exec", "--skip-git-repo-check", "--ephemeral", "--ignore-user-config",
            "--sandbox", "read-only", "--color", "never", "--json",
            "-c", 'approval_policy="never"', "-c", 'web_search="disabled"',
            "-c", 'model_reasoning_effort="low"', "-c", "mcp_servers={}",
        ]
        for feature in _DISABLED_FEATURES:
            args.extend(("--disable", feature))
        if self.model:
            args.extend(("--model", self.model))
        return [*args, "-"]

    async def _run(self, request: ChatRequest) -> tuple[str, dict | None]:
        async with self._semaphore:
            with tempfile.TemporaryDirectory(prefix="mycontext-codex-") as directory:
                try:
                    process = await asyncio.create_subprocess_exec(
                        *self._command(), cwd=directory, env=_cli_environment(),
                        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                except (OSError, RuntimeError) as exc:
                    raise _BridgeError(503, "provider_unavailable", "无法启动 Codex CLI") from exc

                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(_prompt(request).encode("utf-8")), timeout=210,
                    )
                except (TimeoutError, asyncio.CancelledError) as exc:
                    if process.returncode is None:
                        try:
                            process.kill()
                        except ProcessLookupError:
                            pass
                        await process.wait()
                    if isinstance(exc, asyncio.CancelledError):
                        raise
                    raise _BridgeError(504, "provider_timeout", "Codex 回答超时") from exc

        content, usage = _read_turn(stdout, stderr, process.returncode)
        if request.response_format and request.response_format.type == "json_object":
            content = _json_object(content)
        return content, usage

    def router(self, token: str) -> APIRouter:
        if not token:
            raise ValueError("Codex 桥接需要本机访问令牌")
        expected = f"Bearer {token}".encode()
        router = APIRouter()

        @router.post("/v1/chat/completions", include_in_schema=False)
        async def complete(request: Request):
            try:
                authorization = request.headers.get("authorization", "").encode()
                if not hmac.compare_digest(authorization, expected):
                    raise _BridgeError(401, "authentication_error", "缺少有效的本机访问令牌")
                try:
                    payload = ChatRequest.model_validate(await request.json())
                except (ValidationError, ValueError) as exc:
                    raise _BridgeError(400, "invalid_request", "请求格式不受 Codex 桥接支持") from exc
                if payload.model != self.model_name:
                    raise _BridgeError(404, "model_not_found", "请求的模型没有通过此桥接配置")

                content, usage = await self._run(payload)
                response = {
                    "id": f"chatcmpl-{uuid4().hex}",
                    "object": "chat.completion", "created": int(time.time()), "model": self.model_name,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                 "finish_reason": "stop"}],
                }
                if usage is not None:
                    response["usage"] = usage
                return response
            except _BridgeError as exc:
                return JSONResponse({"error": {"message": str(exc), "type": exc.code, "code": exc.code}},
                                    status_code=exc.status)

        return router
