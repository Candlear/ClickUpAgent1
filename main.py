"""
FastAPI-Proxy: Langdock -> Gemini (mehrstufiges Function Calling) -> ClickUp MCP.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Any

import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from google.generativeai.types import FunctionDeclaration, Tool
from pydantic import BaseModel, ConfigDict, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from mcp_helper import mcp_tools_to_gemini_declarations
from mcp_session import MCPManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
REQUEST_TIMEOUT_S = float(os.getenv("AGENT_REQUEST_TIMEOUT_S", "120"))
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "8"))

_secret_header_re = re.compile(r"^Bearer\s+(.+)$", re.I)


def _prod_like() -> bool:
    return (
        os.getenv("ENVIRONMENT", "").lower() == "production"
        or os.getenv("RAILWAY_ENVIRONMENT", "").lower() == "production"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    mgr = MCPManager()
    app.state.mcp = mgr
    try:
        await mgr.startup()
    except Exception:
        logger.exception("MCP startup failed")
    yield
    await mgr.shutdown()


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Langdock ClickUp MCP Proxy", version="0.2.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: str = Field(..., min_length=1, max_length=32000)


def _configure_gemini() -> None:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    genai.configure(api_key=key)


def _parse_function_call_response(response: Any) -> tuple[str, dict[str, Any]] | None:
    for cand in getattr(response, "candidates", None) or []:
        parts = getattr(cand.content, "parts", None) or []
        for part in parts:
            fc = getattr(part, "function_call", None)
            if fc is None:
                continue
            name = getattr(fc, "name", None) or fc.get("name")
            args = getattr(fc, "args", None) or fc.get("args") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if name:
                return str(name), dict(args) if isinstance(args, dict) else {}
    return None


def _response_text(response: Any) -> str:
    try:
        t = (response.text or "").strip()
        if t:
            return t
    except Exception:
        pass
    for cand in getattr(response, "candidates", None) or []:
        parts = getattr(cand.content, "parts", None) or []
        for part in parts:
            t = getattr(part, "text", None)
            if t:
                return str(t).strip()
    return ""


def _function_response_part(name: str, payload: dict[str, Any]) -> Any:
    protos = getattr(genai, "protos", None)
    if protos is not None:
        return protos.Part(
            function_response=protos.FunctionResponse(name=name, response=payload),
        )
    return {
        "function_response": {"name": name, "response": payload},
    }


def _gemini_tool_result_payload(mcp_result: dict[str, Any]) -> dict[str, Any]:
    try:
        blob = json.dumps(mcp_result, default=str)[:12000]
    except (TypeError, ValueError):
        blob = str(mcp_result)[:12000]
    return {"result": blob}


async def _run_agent_pipeline(manager: MCPManager, user_message: str) -> dict[str, Any]:
    if not manager.is_ready():
        if manager.is_hosted():
            detail = "Hosted MCP not connected; complete OAuth via GET /oauth/start"
        else:
            detail = "Community MCP not ready; set CLICKUP_API_KEY and check logs"
        raise RuntimeError(detail)

    _configure_gemini()

    mcp_tools = await manager.list_tools()
    declarations = mcp_tools_to_gemini_declarations(mcp_tools)
    if not declarations:
        return {
            "ok": False,
            "error": "no_mcp_tools",
            "stopped_reason": "error",
            "steps": [],
        }

    try:
        fds = [
            FunctionDeclaration(
                name=d["name"],
                description=d.get("description") or "",
                parameters=d.get("parameters") or {"type": "object", "properties": {}},
            )
            for d in declarations
        ]
        gemini_tool = Tool(function_declarations=fds)
    except Exception as e:
        logger.exception("Building Gemini tools failed")
        return {
            "ok": False,
            "error": "gemini_tool_build_failed",
            "detail": str(e),
            "stopped_reason": "error",
            "steps": [],
        }

    allowed_names = {d["name"] for d in declarations}
    model = genai.GenerativeModel(GEMINI_MODEL, tools=[gemini_tool])

    system_hint = (
        "You help the user by calling ClickUp MCP tools. "
        "Use tools when needed; you may call multiple tools in sequence. "
        "When the task is done, reply with a short natural-language summary (no more tool calls)."
    )
    initial = f"{system_hint}\n\nUser request:\n{user_message}"

    tool_config_auto = {
        "function_calling_config": {"mode": "AUTO"},
    }

    def _send_chat(chat_session: Any, msg: Any, tool_cfg: dict[str, Any]) -> Any:
        try:
            return chat_session.send_message(msg, tool_config=tool_cfg)
        except TypeError:
            return chat_session.send_message(msg)

    try:
        chat = model.start_chat(enable_automatic_function_calling=False)
        response = await asyncio.to_thread(_send_chat, chat, initial, tool_config_auto)
    except Exception as e:
        logger.exception("Gemini initial request failed")
        return {
            "ok": False,
            "error": "gemini_request_failed",
            "detail": str(e),
            "stopped_reason": "error",
            "steps": [],
        }

    steps: list[dict[str, Any]] = []

    for step_idx in range(MAX_AGENT_STEPS):
        parsed = _parse_function_call_response(response)
        if not parsed:
            text = _response_text(response)
            return {
                "ok": True,
                "stopped_reason": "completed",
                "steps": steps,
                "final_text": text,
            }

        tool_name, tool_args = parsed
        if tool_name not in allowed_names:
            return {
                "ok": False,
                "error": "unknown_tool_from_gemini",
                "tool_name": tool_name,
                "stopped_reason": "error",
                "steps": steps,
            }

        try:
            tool_result = await manager.call_tool(tool_name, tool_args)
        except Exception as e:
            logger.exception("MCP call_tool failed")
            return {
                "ok": False,
                "error": "mcp_call_failed",
                "tool_name": tool_name,
                "tool_args": tool_args,
                "detail": str(e),
                "stopped_reason": "error",
                "steps": steps,
            }

        steps.append(
            {
                "step": step_idx + 1,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "tool_result": tool_result,
            }
        )

        fr_part = _function_response_part(
            tool_name, _gemini_tool_result_payload(tool_result)
        )
        try:
            response = await asyncio.to_thread(
                _send_chat, chat, fr_part, tool_config_auto
            )
        except Exception as e:
            logger.exception("Gemini follow-up failed")
            return {
                "ok": False,
                "error": "gemini_request_failed",
                "detail": str(e),
                "stopped_reason": "error",
                "steps": steps,
            }

    return {
        "ok": False,
        "error": "max_steps_exceeded",
        "stopped_reason": "max_steps",
        "steps": steps,
        "final_text": _response_text(response),
    }


@app.middleware("http")
async def webhook_and_prod_checks(request: Request, call_next):
    path = request.url.path
    if _prod_like() and not os.getenv("LANGDOCK_WEBHOOK_SECRET"):
        if path == "/agent":
            return JSONResponse(
                {"ok": False, "error": "server_misconfigured"},
                status_code=503,
            )
    secret = os.getenv("LANGDOCK_WEBHOOK_SECRET")
    if secret and path in {"/agent"}:
        auth = (request.headers.get("authorization") or "").strip()
        m = _secret_header_re.match(auth)
        token = m.group(1) if m else (request.headers.get("x-webhook-secret") or "")
        if token != secret:
            return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    return await call_next(request)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready(request: Request) -> dict[str, Any]:
    mgr: MCPManager = request.app.state.mcp
    return {
        "ready": mgr.is_ready(),
        "mode": mgr.mode,
        "oauth_in_progress": mgr.oauth_flow_active,
    }


@app.get("/oauth/start")
async def oauth_start(request: Request) -> RedirectResponse:
    mgr: MCPManager = request.app.state.mcp
    if not mgr.is_hosted():
        raise HTTPException(
            status_code=400,
            detail="OAuth only for CLICKUP_MCP_MODE=hosted",
        )

    async def run_oauth() -> None:
        try:
            await mgr.begin_interactive_oauth_connection()
        except Exception:
            logger.exception("Interactive OAuth / MCP connect failed")

    asyncio.create_task(run_oauth())
    try:
        url = await mgr.wait_for_redirect_url(timeout=120.0)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="OAuth redirect URL was not produced in time",
        ) from None
    return RedirectResponse(url, status_code=302)


@app.get("/oauth/callback")
async def oauth_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
) -> HTMLResponse:
    mgr: MCPManager = request.app.state.mcp
    if not code:
        return HTMLResponse(
            "<p>Missing <code>code</code> query parameter.</p>",
            status_code=400,
        )
    if mgr.complete_oauth_callback(code, state):
        return HTMLResponse(
            "<p>Authorization successful. You can close this tab.</p>",
            status_code=200,
        )
    return HTMLResponse(
        "<p>No pending OAuth flow, or session already completed.</p>",
        status_code=400,
    )


@app.post("/agent")
@limiter.limit(os.getenv("AGENT_RATE_LIMIT", "30/minute"))
async def agent(request: Request, body: AgentRequest) -> JSONResponse:
    mgr: MCPManager = request.app.state.mcp
    try:
        result = await asyncio.wait_for(
            _run_agent_pipeline(mgr, body.message),
            timeout=REQUEST_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out") from None
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception:
        logger.exception("Unhandled agent error")
        raise HTTPException(status_code=500, detail="internal_error") from None

    status = 200 if result.get("ok") else 422
    return JSONResponse(content=result, status_code=status)
