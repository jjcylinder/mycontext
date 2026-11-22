import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from project_demo import codex_bridge


def _stdout(*events):
    return "\n".join(json.dumps(event) for event in events).encode()


def _app(bridge):
    app = FastAPI()
    app.include_router(bridge.router("test-token"))
    return app


def _request(model):
    return {"model": model, "messages": [{"role": "system", "content": "系统格式"},
                                         {"role": "user", "content": "用户材料"}],
            "response_format": {"type": "json_object"}, "max_tokens": 500, "temperature": 0.1}


def test_use_chatgpt_login_status_without_api_overrides(monkeypatch):
    monkeypatch.setattr(codex_bridge.shutil, "which", lambda _name: "/bin/codex")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("CODEX_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8765/v1")

    def fake_run(args, *, env, **kwargs):
        assert args == ["/bin/codex", "login", "status"]
        assert not set(env).intersection(codex_bridge._API_ENV)
        return SimpleNamespace(returncode=0, stdout=b"Logged in using ChatGPT", stderr=b"")

    monkeypatch.setattr(codex_bridge.subprocess, "run", fake_run)
    codex_bridge.CodexBridge().verify_login()


@pytest.mark.asyncio
async def test_chat_completion_preserves_roles_and_only_returns_final_response(monkeypatch):
    monkeypatch.setattr(codex_bridge.shutil, "which", lambda _name: "/bin/codex")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    call = {}

    class Process:
        returncode = 0

        async def communicate(self, prompt):
            guidance, messages = prompt.decode().rsplit("\n\n", 1)
            assert json.loads(messages) == _request("unused")["messages"]
            assert "one complete JSON object" in guidance and "500 token" in guidance
            return _stdout(
                {"type": "item.completed", "item": {"type": "reasoning", "text": "internal"}},
                {"type": "item.completed", "item": {"type": "agent_message", "text": "draft"}},
                {"type": "item.completed", "item": {"type": "agent_message", "text": '```json\n{"ok":true}\n```'}},
                {"type": "turn.completed", "usage": {"input_tokens": 12, "output_tokens": 8}},
            ), b"internal stderr"

    async def fake_exec(*args, **kwargs):
        call.update(kwargs)
        assert args[:2] == ("/bin/codex", "exec") and args[-1] == "-"
        assert all(part in args for part in ("--ephemeral", "--ignore-user-config", "read-only",
                                             'approval_policy="never"', 'web_search="disabled"'))
        assert all(feature in args for feature in codex_bridge._DISABLED_FEATURES)
        assert Path(kwargs["cwd"]).is_dir()
        assert not set(kwargs["env"]).intersection(codex_bridge._API_ENV)
        return Process()

    monkeypatch.setattr(codex_bridge.asyncio, "create_subprocess_exec", fake_exec)
    bridge = codex_bridge.CodexBridge()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app(bridge)), base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json=_request(bridge.model_name),
                                     headers={"Authorization": "Bearer test-token"})

    assert response.status_code == 200
    result = response.json()
    assert result["choices"][0]["message"] == {"role": "assistant", "content": '{"ok":true}'}
    assert result["usage"] == {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}
    assert "internal" not in response.text and "draft" not in response.text
    assert not Path(call["cwd"]).exists()


@pytest.mark.asyncio
async def test_invalid_requests_do_not_invoke_codex_and_cli_failures_are_sanitized(monkeypatch):
    bridge = codex_bridge.CodexBridge("example")
    calls = []

    async def failed(payload):
        calls.append(payload)
        codex_bridge._read_turn(b"", b"429 rate limited secret backend detail", 1)

    monkeypatch.setattr(bridge, "_run", failed)
    request = _request(bridge.model_name)
    auth = {"Authorization": "Bearer test-token"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app(bridge)), base_url="http://test") as client:
        denied = await client.post("/v1/chat/completions", json=request)
        wrong_model = await client.post("/v1/chat/completions", json={**request, "model": "other"}, headers=auth)
        invalid = await client.post("/v1/chat/completions", json={**request, "stream": True}, headers=auth)
        failure = await client.post("/v1/chat/completions", json=request, headers=auth)

    assert [response.status_code for response in (denied, wrong_model, invalid, failure)] == [401, 404, 400, 429]
    assert len(calls) == 1
    assert failure.json()["error"]["code"] == "rate_limit_exceeded"
    assert "secret backend detail" not in failure.text
