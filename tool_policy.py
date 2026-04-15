"""
Blockiert riskante ClickUp-MCP-Tools im Proxy (vor Gemini und vor call_tool).
"""

from __future__ import annotations

from mcp import types

# Sehr hohes Risiko (delete_task explizit ausgenommen) + hohes Risiko — siehe Projekt-Doku.
BLOCKED_CLICKUP_TOOLS: frozenset[str] = frozenset(
    {
        "delete_space",
        "delete_folder",
        "delete_list",
        "delete_bulk_tasks",
        "delete_webhook",
        "delete_user_group",
        "delete_goal",
        "delete_key_result",
        "delete_comment",
        "delete_chat_message",
        "delete_chat_channel",
        "remove_guest",
        "delete_time_entry",
        "delete_space_tag",
        "set_space_permissions",
        "set_folder_permissions",
        "set_list_permissions",
        "invite_guest",
        "add_guest_to_task",
        "add_guest_to_list",
        "add_guest_to_folder",
        "edit_guest",
        "create_webhook",
        "update_webhook",
        "get_audit_logs",
    }
)


def is_tool_allowed(name: str) -> bool:
    return name not in BLOCKED_CLICKUP_TOOLS


def filter_mcp_tools(tools: list[types.Tool]) -> list[types.Tool]:
    return [t for t in tools if is_tool_allowed(t.name)]
