"""
Langfristige MCP-Verbindung: community (stdio/npx) oder hosted (Streamable HTTP + OAuth).
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import AsyncExitStack
from typing import Any, Callable, Coroutine

import httpx
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.auth import OAuthClientProvider
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from pydantic import AnyUrl

from mcp_helper import build_stdio_params, call_clickup_tool, list_clickup_tools
from oauth_storage import FileTokenStorage

logger = logging.getLogger(__name__)

CLICKUP_MCP_MODE_HOSTED = "hosted"
CLICKUP_MCP_MODE_COMMUNITY = "community_stdio"

DEFAULT_MCP_URL = "https://mcp.clickup.com/mcp"
DEFAULT_TOKEN_PATH = "/data/clickup_oauth.json"


def _unpack_streamable_streams(streams: Any) -> tuple[Any, Any]:
    if isinstance(streams, tuple) and len(streams) >= 2:
        return streams[0], streams[1]
    raise TypeError(f"Unexpected streamable_http_client yield: {type(streams)}")


class MCPManager:
    """Eine ClientSession, serialisiert mit Lock; Reconnect bei Bedarf."""

    def __init__(self) -> None:
        self.mode = (os.getenv("CLICKUP_MCP_MODE") or CLICKUP_MCP_MODE_COMMUNITY).strip().lower()
        self.mcp_url = (os.getenv("CLICKUP_MCP_URL") or DEFAULT_MCP_URL).strip()
        token_path = (os.getenv("CLICKUP_OAUTH_TOKEN_PATH") or DEFAULT_TOKEN_PATH).strip()
        self._storage = FileTokenStorage(token_path)
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None
        self._lock = asyncio.Lock()
        self._http_client: httpx.AsyncClient | None = None
        self._oauth_code_fut: asyncio.Future[tuple[str, str | None]] | None = None
        self._pending_redirect_url: str | None = None
        self._redirect_ready = asyncio.Event()
        self._reconnect_backoff_s = float(os.getenv("MCP_RECONNECT_BACKOFF_S", "2"))
        self._reconnect_max = int(os.getenv("MCP_RECONNECT_MAX_ATTEMPTS", "3"))
        self._oauth_interactive_lock = asyncio.Lock()
        self._oauth_flow_active = False

    def is_hosted(self) -> bool:
        return self.mode == CLICKUP_MCP_MODE_HOSTED

    def is_ready(self) -> bool:
        return self._session is not None

    @property
    def oauth_flow_active(self) -> bool:
        return self._oauth_flow_active

    async def startup(self) -> None:
        if self.mode == CLICKUP_MCP_MODE_COMMUNITY:
            if not os.getenv("CLICKUP_API_KEY"):
                logger.warning("CLICKUP_API_KEY missing; community MCP will not start until set")
                return
            await self._connect_community()
        elif self.mode == CLICKUP_MCP_MODE_HOSTED:
            tok = await self._storage.get_tokens()
            if not tok:
                logger.info("Hosted MCP: no stored OAuth tokens; use GET /oauth/start")
                return
            await self._connect_hosted_silent()
        else:
            raise ValueError(f"Unknown CLICKUP_MCP_MODE={self.mode!r}")

    async def shutdown(self) -> None:
        await self._stack.aclose()
        self._stack = AsyncExitStack()
        self._session = None
        self._http_client = None

    def _oauth_metadata(self) -> Any:
        from mcp.shared.auth import OAuthClientMetadata

        redirect = (os.getenv("OAUTH_REDIRECT_URI") or "").strip()
        if not redirect:
            raise RuntimeError("OAUTH_REDIRECT_URI is required for hosted OAuth")
        scope = (os.getenv("CLICKUP_OAUTH_SCOPE") or "").strip() or None
        return OAuthClientMetadata(
            client_name=os.getenv("OAUTH_CLIENT_NAME", "Langdock ClickUp Proxy"),
            redirect_uris=[AnyUrl(redirect)],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            scope=scope,
        )

    async def _on_oauth_redirect(self, url: str) -> None:
        self._pending_redirect_url = url
        self._redirect_ready.set()

    async def _on_oauth_callback(self) -> tuple[str, str | None]:
        loop = asyncio.get_running_loop()
        self._oauth_code_fut = loop.create_future()
        return await self._oauth_code_fut

    def complete_oauth_callback(self, code: str, state: str | None) -> bool:
        fut = self._oauth_code_fut
        if fut is None or fut.done():
            return False
        fut.set_result((code, state))
        return True

    async def wait_for_redirect_url(self, timeout: float = 120.0) -> str:
        self._redirect_ready.clear()
        self._pending_redirect_url = None
        await asyncio.wait_for(self._redirect_ready.wait(), timeout=timeout)
        if not self._pending_redirect_url:
            raise RuntimeError("OAuth redirect URL not set")
        return self._pending_redirect_url

    async def begin_interactive_oauth_connection(self) -> None:
        """Nach Redirect: läuft weiter bis Token gespeichert und Session steht."""
        async with self._oauth_interactive_lock:
            await self._begin_interactive_oauth_connection_impl()

    async def _begin_interactive_oauth_connection_impl(self) -> None:
        async with self._lock:
            self._oauth_flow_active = True
            try:
                await self._disconnect()
                oauth = OAuthClientProvider(
                    server_url=self.mcp_url,
                    client_metadata=self._oauth_metadata(),
                    storage=self._storage,
                    redirect_handler=self._on_oauth_redirect,
                    callback_handler=self._on_oauth_callback,
                    timeout=float(os.getenv("OAUTH_FLOW_TIMEOUT_S", "300")),
                )
                self._http_client = httpx.AsyncClient(
                    auth=oauth,
                    follow_redirects=True,
                    timeout=httpx.Timeout(120.0),
                )
                await self._stack.enter_async_context(self._http_client)
                streams = await self._stack.enter_async_context(
                    streamable_http_client(self.mcp_url, http_client=self._http_client)
                )
                read, write = _unpack_streamable_streams(streams)
                self._session = await self._stack.enter_async_context(
                    ClientSession(read, write)
                )
                await self._session.initialize()
                logger.info("Hosted MCP session initialized after OAuth")
            except Exception:
                try:
                    await self._disconnect()
                except Exception:
                    logger.exception("Cleanup after failed OAuth connect")
                raise
            finally:
                self._oauth_flow_active = False

    async def _connect_hosted_silent(self) -> None:
        oauth = OAuthClientProvider(
            server_url=self.mcp_url,
            client_metadata=self._oauth_metadata(),
            storage=self._storage,
            redirect_handler=None,
            callback_handler=None,
            timeout=float(os.getenv("OAUTH_FLOW_TIMEOUT_S", "300")),
        )
        self._http_client = httpx.AsyncClient(
            auth=oauth,
            follow_redirects=True,
            timeout=httpx.Timeout(120.0),
        )
        await self._stack.enter_async_context(self._http_client)
        streams = await self._stack.enter_async_context(
            streamable_http_client(self.mcp_url, http_client=self._http_client)
        )
        read, write = _unpack_streamable_streams(streams)
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        logger.info("Hosted MCP connected with stored tokens")

    async def _connect_community(self) -> None:
        params: StdioServerParameters = build_stdio_params()
        logger.info("Starting community MCP stdio: %s %s", params.command, params.args)
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        logger.info("Community MCP session initialized")

    async def _disconnect(self) -> None:
        await self._stack.aclose()
        self._stack = AsyncExitStack()
        self._session = None
        self._http_client = None

    async def reconnect(self) -> None:
        async with self._lock:
            await self._disconnect()
            if self.mode == CLICKUP_MCP_MODE_COMMUNITY:
                await self._connect_community()
            elif self.mode == CLICKUP_MCP_MODE_HOSTED:
                await self._connect_hosted_silent()

    async def list_tools(self) -> list[types.Tool]:
        return await self._with_retry(lambda s: list_clickup_tools(s))

    async def call_tool(self, name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        return await self._with_retry(lambda s: call_clickup_tool(s, name, arguments))

    async def _with_retry(
        self,
        fn: Callable[[ClientSession], Coroutine[Any, Any, Any]],
    ) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self._reconnect_max):
            try:
                async with self._lock:
                    if not self._session:
                        raise RuntimeError("MCP session not ready")
                    return await fn(self._session)
            except Exception as e:
                last_exc = e
                logger.warning("MCP operation failed (attempt %s): %s", attempt + 1, e)
                if attempt + 1 >= self._reconnect_max:
                    break
                try:
                    await self.reconnect()
                except Exception as re:
                    logger.warning("Reconnect failed: %s", re)
                await asyncio.sleep(self._reconnect_backoff_s * (attempt + 1))
        raise last_exc if last_exc else RuntimeError("MCP retry exhausted")
