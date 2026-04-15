"""
Hilfsfunktionen für MCP-stdio-Client: Subprocess (npx) starten, Session, Tools.
"""

from __future__ import annotations

import copy
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

# Öffentliches Paket; @modelcontextprotocol/server-clickup existiert auf npm nicht.
_DEFAULT_NPX_PACKAGE = "@taazkareem/clickup-mcp-server"


def _subprocess_env() -> dict[str, str]:
    """Umgebung für den MCP-Kindprozess: aktuelle env + explizite ClickUp-Keys."""
    env = dict(os.environ)
    clickup_key = os.getenv("CLICKUP_API_KEY")
    if clickup_key:
        env["CLICKUP_API_KEY"] = clickup_key
    team = os.getenv("CLICKUP_TEAM_ID")
    if team:
        env["CLICKUP_TEAM_ID"] = team
    return env


def _pick_json_schema_type(types_list: list[Any]) -> str:
    """JSON-Schema-Unions (type als Liste); Gemini braucht einen einzelnen Typ-String."""
    non_null = [x for x in types_list if x != "null" and x is not None]
    if not non_null:
        return "string"
    # Bevorzugt gängige Typen, sonst erstes Element
    for preferred in ("object", "array", "string", "number", "integer", "boolean"):
        if preferred in non_null:
            return preferred
    first = non_null[0]
    return str(first) if first is not None else "string"


def normalize_json_schema_for_gemini(schema: Any) -> Any:
    """
    Rekursiv: `type`-Felder, die Listen sind (Union), in einen String verwandeln.
    Sonst schlägt google.generativeai mit "'list' object has no attribute 'upper'" fehl.
    """
    if isinstance(schema, dict):
        out: dict[str, Any] = {}
        for k, v in schema.items():
            if k == "type" and isinstance(v, list):
                out[k] = _pick_json_schema_type(v)
            else:
                out[k] = normalize_json_schema_for_gemini(v)
        return out
    if isinstance(schema, list):
        return [normalize_json_schema_for_gemini(x) for x in schema]
    return schema


def build_stdio_params() -> StdioServerParameters:
    package = os.getenv("CLICKUP_MCP_NPX_PACKAGE", _DEFAULT_NPX_PACKAGE).strip()
    extra = os.getenv("CLICKUP_MCP_NPX_ARGS", "")
    args = ["-y", package]
    if extra:
        args.extend(extra.split())
    return StdioServerParameters(
        command="npx",
        args=args,
        env=_subprocess_env(),
    )


def mcp_tools_to_gemini_declarations(
    tools: list[types.Tool],
) -> list[dict[str, Any]]:
    """Konvertiert MCP-Tools in ein Format, das google-generativeai als tools=[] akzeptiert."""
    declarations: list[dict[str, Any]] = []
    for t in tools:
        schema: dict[str, Any]
        if t.inputSchema is not None:
            if isinstance(t.inputSchema, dict):
                schema = copy.deepcopy(t.inputSchema)
            else:
                schema = copy.deepcopy(dict(t.inputSchema))
        else:
            schema = {"type": "object", "properties": {}}
        if schema.get("type") != "object":
            schema = {"type": "object", "properties": {"value": schema}}
        schema = normalize_json_schema_for_gemini(schema)
        declarations.append(
            {
                "name": t.name,
                "description": (t.description or "")[:8000],
                "parameters": schema,
            }
        )
    return declarations


def _serialize_tool_result(result: types.CallToolResult) -> dict[str, Any]:
    out: dict[str, Any] = {
        "isError": bool(result.isError),
        "content": [],
    }
    if result.structuredContent is not None:
        out["structuredContent"] = result.structuredContent
    for block in result.content:
        if isinstance(block, types.TextContent):
            out["content"].append({"type": "text", "text": block.text})
        elif type(block).__name__ == "ImageContent":
            out["content"].append(
                {
                    "type": "image",
                    "mimeType": getattr(block, "mimeType", None),
                    "has_data": bool(getattr(block, "data", None)),
                }
            )
        else:
            out["content"].append({"type": type(block).__name__})
    return out


@asynccontextmanager
async def clickup_mcp_session() -> AsyncIterator[ClientSession]:
    """
    Kontextmanager: startet den ClickUp-MCP-Server als Subprocess und liefert eine ClientSession.
    """
    params = build_stdio_params()
    logger.info("Starting MCP stdio: command=%s args=%s", params.command, params.args)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def list_clickup_tools(session: ClientSession) -> list[types.Tool]:
    r = await session.list_tools()
    return list(r.tools)


async def call_clickup_tool(
    session: ClientSession,
    name: str,
    arguments: dict[str, Any] | None,
) -> dict[str, Any]:
    result = await session.call_tool(name, arguments=arguments or {})
    return _serialize_tool_result(result)
