"""
Persistente OAuth-Token-Speicherung für den MCP Hosted-Modus (Datei, optional verschlüsselt).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

logger = logging.getLogger(__name__)


def _fernet():
    key = (os.getenv("CLICKUP_OAUTH_ENCRYPTION_KEY") or "").strip()
    if not key:
        return None
    try:
        from cryptography.fernet import Fernet
    except ImportError as e:
        raise RuntimeError("cryptography is required when CLICKUP_OAUTH_ENCRYPTION_KEY is set") from e
    return Fernet(key.encode() if isinstance(key, str) else key)


class FileTokenStorage:
    """TokenStorage-Implementierung: JSON-Datei, optional Fernet-Verschlüsselung."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = asyncio.Lock()
        self._fernet = _fernet()

    def _read_raw(self) -> dict[str, Any] | None:
        if not self.path.is_file():
            return None
        raw = self.path.read_bytes()
        if self._fernet:
            raw = self._fernet.decrypt(raw)
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else None

    def _write_raw(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        blob = json.dumps(data, separators=(",", ":"), default=str).encode("utf-8")
        if self._fernet:
            blob = self._fernet.encrypt(blob)
        self.path.write_bytes(blob)
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    async def get_tokens(self) -> OAuthToken | None:
        async with self._lock:
            try:
                data = await asyncio.to_thread(self._read_raw)
            except Exception:
                logger.exception("Failed to read OAuth token file")
                return None
            if not data or "tokens" not in data:
                return None
            try:
                return OAuthToken.model_validate(data["tokens"])
            except Exception:
                logger.exception("Invalid OAuth token data")
                return None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        async with self._lock:
            data = await asyncio.to_thread(self._read_raw) or {}
            data["tokens"] = tokens.model_dump(mode="json", exclude_none=True)
            await asyncio.to_thread(self._write_raw, data)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        async with self._lock:
            try:
                data = await asyncio.to_thread(self._read_raw)
            except Exception:
                return None
            if not data or "client_info" not in data:
                return None
            try:
                return OAuthClientInformationFull.model_validate(data["client_info"])
            except Exception:
                return None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        async with self._lock:
            data = await asyncio.to_thread(self._read_raw) or {}
            data["client_info"] = client_info.model_dump(mode="json", exclude_none=True)
            await asyncio.to_thread(self._write_raw, data)
