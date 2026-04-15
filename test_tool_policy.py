"""Tests for tool_policy (no MCP server required)."""

from __future__ import annotations

from mcp import types

from tool_policy import filter_mcp_tools, is_tool_allowed


def _tool(name: str) -> types.Tool:
    return types.Tool(name=name, inputSchema={"type": "object", "properties": {}})


def test_delete_task_allowed():
    assert is_tool_allowed("delete_task") is True


def test_delete_space_and_get_audit_logs_blocked():
    assert is_tool_allowed("delete_space") is False
    assert is_tool_allowed("get_audit_logs") is False


def test_unknown_tool_allowed():
    assert is_tool_allowed("custom_or_future_tool") is True


def test_filter_mcp_tools_keeps_delete_task_drops_blocked():
    tools = [
        _tool("delete_task"),
        _tool("delete_space"),
        _tool("get_task"),
    ]
    out = filter_mcp_tools(tools)
    assert [t.name for t in out] == ["delete_task", "get_task"]


def test_filter_mcp_tools_empty_input():
    assert filter_mcp_tools([]) == []
