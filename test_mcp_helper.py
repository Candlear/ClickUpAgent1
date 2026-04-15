"""Tests for MCP -> Gemini schema normalization."""

from __future__ import annotations

import pytest

from mcp_helper import normalize_json_schema_for_gemini


def test_normalize_type_union_list_to_string():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": ["string", "null"], "description": "x"},
            "n": {"type": ["number", "integer"]},
        },
    }
    out = normalize_json_schema_for_gemini(schema)
    assert out["properties"]["name"]["type"] == "string"
    assert out["properties"]["n"]["type"] == "number"


def test_nested_list_types():
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": ["string", "null"]},
            }
        },
    }
    out = normalize_json_schema_for_gemini(schema)
    assert out["properties"]["items"]["items"]["type"] == "string"


def test_gemini_function_declaration_accepts_normalized_schema():
    google = pytest.importorskip("google.generativeai")
    from google.generativeai.types import FunctionDeclaration

    schema = normalize_json_schema_for_gemini(
        {
            "type": "object",
            "properties": {
                "opt": {"type": ["string", "null"]},
            },
        }
    )
    fd = FunctionDeclaration(
        name="t",
        description="d",
        parameters=schema,
    )
    assert fd.name == "t"
